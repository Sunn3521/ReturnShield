"""ReturnShield launcher - double-click supervisor for the full stack.

Stdlib only. Responsibilities, in order:

1. Locate the application folder (the directory the exe/script lives in).
2. Find a Python interpreter and check the required packages; optionally
   install them from requirements.txt when ``--install`` is passed.
3. Free the ports if a *ReturnShield* process already owns them (reuse it
   rather than starting a second copy); refuse to touch foreign listeners.
4. Start the FastAPI backend, wait until /api/v1/meta answers and identifies
   itself as ReturnShield.
5. Start the Streamlit dashboard, wait until it serves HTTP.
6. Open the browser and stay alive, cleaning up both children on exit.

Every knob has an environment override so QA can run two stacks side by side:

    RS_API_PORT=8010 RS_UI_PORT=8510 RS_NO_BROWSER=1 ReturnShield.exe

If the delivery carries a bundled runtime (``runtime/``, see build_runtime.py)
it is used automatically and no system Python is required. When an earlier run
left something behind:

    ReturnShield.exe --stop
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser

API_PORT = int(os.getenv("RS_API_PORT", "8000"))
UI_PORT = int(os.getenv("RS_UI_PORT", "8501"))
API_HOST = os.getenv("RS_API_HOST", "127.0.0.1")
UI_HOST = os.getenv("RS_UI_HOST", "127.0.0.1")
OPEN_BROWSER = os.getenv("RS_NO_BROWSER", "") not in {"1", "true", "yes"}
#: Generous defaults. The first start loads the model bundle and compiles the
#: Streamlit script cache, and the dashboard is routinely the slower of the two.
API_WAIT_SECONDS = int(os.getenv("RS_API_WAIT", "300"))
UI_WAIT_SECONDS = int(os.getenv("RS_UI_WAIT", "420"))
#: How many consecutive ports to try when the configured one is taken, so a
#: stray listener is never a fatal error.
PORT_SCAN_LIMIT = int(os.getenv("RS_PORT_SCAN", "20"))

REQUIRED_MODULES = (
    "streamlit", "fastapi", "uvicorn", "pandas", "numpy",
    "sklearn", "xgboost", "shap", "joblib", "plotly", "pyarrow", "httpx",
)

CHILDREN: list[subprocess.Popen] = []


def app_dir() -> str:
    """Install root: the folder the exe/script lives in."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def source_root(install_root: str) -> str:
    """The directory that actually contains src/api.py.

    The delivery keeps source under app/ so the top level stays readable;
    running from a source checkout has src/ beside the launcher. Both layouts
    must work from the same binary.
    """
    for candidate in (install_root, os.path.join(install_root, "app")):
        if os.path.exists(os.path.join(candidate, "src", "api.py")):
            return candidate
    return install_root


def log(message: str) -> None:
    print(f"[ReturnShield] {message}", flush=True)


def bundled_python() -> str | None:
    """The relocated interpreter shipped inside the delivery, when present.

    ``build_runtime.py`` drops an embeddable CPython into ``runtime/``. When it
    is there the delivery needs no system Python at all, so it must win over
    whatever happens to be on PATH.
    """
    candidate = os.path.join(app_dir(), "runtime", "python.exe")
    return candidate if os.path.exists(candidate) else None


def find_python() -> str:
    """Interpreter used for the two child processes.

    Order: explicit RS_PYTHON, the bundled runtime, the interpreter running this
    script (covers the dev case), ``python`` on PATH, then the Windows ``py``
    launcher.
    """
    explicit = os.getenv("RS_PYTHON", "").strip()
    if explicit:
        return explicit
    bundled = bundled_python()
    if bundled:
        return bundled
    if not getattr(sys, "frozen", False):
        return sys.executable
    for candidate in ("python", "py"):
        try:
            out = subprocess.run(
                [candidate, "--version"], capture_output=True, text=True, timeout=30,
            )
            if out.returncode == 0:
                return candidate
        except (OSError, subprocess.TimeoutExpired):
            continue
    return "python"


def port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.5)
        return sock.connect_ex((host, port)) == 0


def identify_returnshield(host: str, port: int) -> bool:
    """True when the listener on this port answers as ReturnShield AI."""
    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/api/v1/meta", timeout=5,
        ) as response:
            import json
            return json.load(response).get("service") == "ReturnShield AI"
    except Exception:
        return False


def endpoint_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return 200 <= response.status < 400
    except Exception:
        return False


def wait_for(urls, deadline_seconds: int, label: str, procs=()) -> bool:
    """Poll every candidate URL until any of them answers. True as soon as one is.

    A list is accepted so a component that serves a health route before its root
    page (Streamlit's ``/_stcore/health`` is exactly that) still counts as ready.
    The heartbeat matters as much as the polling: a slow first start should read
    as progress rather than a hang.

    ``procs`` short-circuits the wait when one of the children has already died.
    Without it a process that exited on its first line (a bad argument, a missing
    file) looked exactly like a slow start, and the launcher sat on the full
    timeout before saying anything - which is how a dashboard that never started
    produced a 7-minute silence and a "still starting" banner.
    """
    if isinstance(urls, str):
        urls = [urls]
    log(f"Waiting for {label} (up to {deadline_seconds}s; first start is slow)...")
    for url in urls:
        log(f"  probe: {url}")
    started = time.time()
    deadline = started + deadline_seconds
    next_beat = started + 15
    while time.time() < deadline:
        for proc in procs:
            code = proc.poll()
            if code is not None:
                log(f"ERROR: {label} exited immediately (code {code}).")
                return False
        for url in urls:
            if endpoint_ready(url):
                log(f"{label} is ready after {time.time() - started:.0f}s.")
                return True
        if time.time() >= next_beat:
            log(f"  ... still waiting for {label} ({time.time() - started:.0f}s elapsed)")
            next_beat = time.time() + 15
        time.sleep(3)
    log(f"ERROR: {label} did not become ready within {deadline_seconds}s.")
    return False


def stop_windows_tree(pid: int) -> None:
    subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        capture_output=True,
    )


def setup_job_object() -> None:
    """Windows: put the whole stack in a Job Object that dies with the launcher.

    ``taskkill /F`` on the parent bypasses Python's signal handling, so the
    finally-block never runs and the API/UI would be orphaned. Kill-on-close
    is an OS contract instead: when the launcher process exits - Ctrl+C,
    window close, taskkill, crash - the kernel terminates every process in the
    job. Requires Win8+ (nested jobs); failure is non-fatal because the
    graceful cleanup path still exists.
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
            )]

        class BASIC_LIMITS(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class EXTENDED_LIMITS(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BASIC_LIMITS),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        JobObjectExtendedLimitInformation = 9
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        kernel32 = ctypes.windll.kernel32
        # Handles are 64-bit; without explicit argtypes ctypes truncates them
        # to 32-bit c_int and every call afterwards fails with an invalid handle.
        kernel32.CreateJobObjectW.restype = ctypes.c_void_p
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.SetInformationJobObject.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
        ]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            log(f"Job object unavailable (err {kernel32.GetLastError()}); Ctrl+C cleanup still applies.")
            return
        limits = EXTENDED_LIMITS()
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            job, JobObjectExtendedLimitInformation,
            ctypes.byref(limits), ctypes.sizeof(limits),
        ):
            log(f"Job object config failed (err {kernel32.GetLastError()}); Ctrl+C cleanup still applies.")
            return
        if not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
            log(f"Job assignment failed (err {kernel32.GetLastError()}); Ctrl+C cleanup still applies.")
            return
        # Hold the handle open for the process lifetime - closing it is the trigger.
        globals()["_JOB_HANDLE"] = job
        log("Job object active: children terminate with the launcher (taskkill-proof).")
    except Exception as exc:  # noqa: BLE001 - best effort hardening
        log(f"Job object setup skipped: {exc}")


def cleanup() -> None:
    for child in CHILDREN:
        if child.poll() is not None:
            continue
        if os.name == "nt":
            stop_windows_tree(child.pid)
        else:
            child.terminate()
    for child in CHILDREN:
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                child.kill()
            except OSError:
                pass
    CHILDREN.clear()


def child_env() -> dict:
    """Environment for every child: the launcher's env plus the explicit home.

    Passed explicitly rather than relied on inheriting ``os.environ``: a frozen
    onefile bootloader re-execs itself, and an explicit dict removes any doubt
    about which environment block the child receives.
    """
    env = dict(os.environ)
    env["RETURNSHIELD_HOME"] = ASSET_ROOT
    env["PYTHONPATH"] = APP_ROOT
    return env


def run_checked(command: list[str], label: str) -> bool:
    log(f"{label}...")
    try:
        result = subprocess.run(command, cwd=APP_ROOT, env=child_env())
    except OSError as exc:
        log(f"ERROR: could not run {label}: {exc}")
        return False
    if result.returncode != 0:
        log(f"ERROR: {label} failed with exit code {result.returncode}.")
        return False
    return True


APP_ROOT = source_root(app_dir())
os.chdir(APP_ROOT)


def assets_root() -> str:
    """The directory holding models/, data/, reports/.

    Packaged layout: beside the exe (source lives one level down in app/).
    Source layout: the same folder that holds src/. Returning APP_ROOT when
    neither matches keeps a bare source run working.
    """
    install = app_dir()
    for candidate in (install, os.path.join(install, "..")):
        if os.path.exists(os.path.join(candidate, "models", "model_bundle.joblib")):
            return os.path.abspath(candidate)
    return APP_ROOT


ASSET_ROOT = assets_root()
os.environ.setdefault("PYTHONPATH", APP_ROOT)
#: Every module resolves assets through src/paths.py. PYTHONPATH/cwd point at
#: the source (so `import src.api` works); RETURNSHIELD_HOME points at the
#: assets (models/, data/, reports/) - in the packaged layout these differ.
os.environ["RETURNSHIELD_HOME"] = ASSET_ROOT
os.environ.setdefault("RETURNSHIELD_API_URL", f"http://{API_HOST}:{API_PORT}")
LOG_DIR = os.path.join(app_dir(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def deps_ok(python: str) -> bool:
    probe = "import " + ", ".join(REQUIRED_MODULES)
    try:
        result = subprocess.run(
            [python, "-c", probe], cwd=APP_ROOT, env=child_env(),
            capture_output=True, timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def ensure_deps(python: str, install: bool) -> bool:
    if deps_ok(python):
        return True
    if not install:
        log("Required Python packages are missing on this machine.")
        log("Fix it with:   ReturnShield.exe --install")
        log("(or: pip install -r requirements.txt)")
        return False
    return run_checked(
        [python, "-m", "pip", "install", "-r", "requirements.txt"],
        "Installing required packages (one-time, several minutes)...",
    ) and deps_ok(python)


def ensure_assets() -> bool:
    """Model assets ship at the assets root; only rebuild when truly absent.

    A retrain takes minutes and produces a different model than the one QA
    validated, so it must only ever happen as an explicit last resort.
    """
    missing = [
        name for name in ("models/model_bundle.joblib", "models/policy.json")
        if not os.path.exists(os.path.join(ASSET_ROOT, name.replace("/", os.sep)))
    ]
    if not missing:
        log(f"Model assets present ({ASSET_ROOT}).")
        return True
    log(f"Missing model assets: {', '.join(missing)}")
    return run_checked(
        [find_python(), os.path.join(APP_ROOT, "run_pipeline.py")],
        "Building models with run_pipeline.py (one-time, several minutes)...",
    )


def claim_port(host: str, port: int, label: str, reuse_check=None) -> int | None:
    """Return a usable port: reuse our own listener, else step past a foreign one.

    A foreign listener no longer aborts the start. The scan walks forward and
    reports the substitution, so an unrelated program squatting on 8000 or 8501
    cannot stop the stack from coming up.
    """
    for candidate in range(port, port + PORT_SCAN_LIMIT):
        if not port_open(host, candidate):
            if candidate != port:
                log(f"{label} port {port} is busy; using {candidate} instead.")
            return candidate
        if reuse_check is not None and reuse_check(host, candidate):
            log(f"{label} already running on port {candidate}; reusing it.")
            return candidate
    log(f"ERROR: no free {label} port in {port}-{port + PORT_SCAN_LIMIT - 1}.")
    return None


def stop_by_port(port: int) -> None:
    """Force-stop whatever listens on this port (Windows only).

    The Job Object covers every ordinary exit. This is the backstop for the one
    case it cannot cover: the launcher itself killed outright, leaving children
    with no parent left alive to run cleanup.
    """
    if os.name != "nt":
        return
    script = (
        f"$c=Get-NetTCPConnection -LocalPort {port} -State Listen "
        f"-ErrorAction SilentlyContinue; "
        f"foreach($x in $c){{Stop-Process -Id $x.OwningProcess -Force "
        f"-ErrorAction SilentlyContinue}}"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True,
    )


def run_stop() -> int:
    """``ReturnShield.exe --stop``: kill any stack running on the configured ports."""
    log("Stopping any ReturnShield stack on the configured ports...")
    for label, host, port in (
        ("api", API_HOST, API_PORT), ("dashboard", UI_HOST, UI_PORT),
    ):
        if port_open(host, port):
            log(f"  stopping {label} on port {port}")
            stop_by_port(port)
        else:
            log(f"  no {label} listening on port {port}")
    log("Done.")
    return 0


def main() -> int:
    global OPEN_BROWSER
    if "--stop" in sys.argv:
        return run_stop()
    install = "--install" in sys.argv
    no_browser = "--no-browser" in sys.argv
    if no_browser:
        OPEN_BROWSER = False
    setup_job_object()

    log(f"Install folder: {app_dir()}")
    log(f"Application source: {APP_ROOT}")
    log(f"Assets root: {ASSET_ROOT}")
    python = find_python()
    log(f"Using interpreter: {python}")

    if not os.path.exists(os.path.join(APP_ROOT, "src", "api.py")):
        log("ERROR: src/api.py not found - the delivery folder is incomplete.")
        log("Expected it in either the install folder or its app/ subfolder.")
        return 1
    if not ensure_deps(python, install):
        return 1
    if not ensure_assets():
        return 1

    api_reused = False
    api_port = claim_port(API_HOST, API_PORT, "api", reuse_check=identify_returnshield)
    if api_port is None:
        return 1
    if port_open(API_HOST, api_port):
        api_reused = True  # identify_returnshield already confirmed it
    os.environ["RETURNSHIELD_API_URL"] = f"http://{API_HOST}:{api_port}"

    ui_port = claim_port(UI_HOST, UI_PORT, "ui")
    if ui_port is None:
        return 1

    # Everything from the first spawn onward lives inside one try/finally. An
    # early ``return 1`` used to skip cleanup entirely, orphaning a half-started
    # API the moment its readiness probe timed out.
    api_proc: subprocess.Popen | None = None
    try:
        if not api_reused:
            api_log = open(os.path.join(LOG_DIR, "api.log"), "ab")
            log(f"Starting API on http://{API_HOST}:{api_port} ...")
            api_proc = subprocess.Popen(
                [python, "-m", "uvicorn", "src.api:app",
                 "--host", API_HOST, "--port", str(api_port), "--log-level", "info"],
                cwd=APP_ROOT, env=child_env(), stdout=api_log, stderr=subprocess.STDOUT,
            )
            CHILDREN.append(api_proc)
            if not wait_for(
                [f"http://{API_HOST}:{api_port}/api/v1/meta",
                 f"http://{API_HOST}:{api_port}/api/v1/health"],
                API_WAIT_SECONDS, "API", procs=[api_proc],
            ):
                log(f"       See {os.path.join(LOG_DIR, 'api.log')} for the reason.")
                return 1
            if not identify_returnshield(API_HOST, api_port):
                log("ERROR: something answered on the API port but it is not ReturnShield.")
                return 1
            log("API verified as ReturnShield AI.")
        else:
            log("API health assumed OK (existing ReturnShield instance).")

        ui_log = open(os.path.join(LOG_DIR, "streamlit.log"), "ab")
        log(f"Starting dashboard on http://{UI_HOST}:{ui_port} ...")
        # Absolute script path on purpose: the packaged delivery keeps the source
        # in app/ while the assets sit beside the exe, so a bare "app.py" only
        # resolves when the child's cwd happens to be app/. "File does not exist:
        # app.py" from Streamlit is exactly that mistake, and it made the whole
        # dashboard unreachable.
        ui_proc = subprocess.Popen(
            [python, "-m", "streamlit", "run", os.path.join(APP_ROOT, "app.py"),
             "--server.address", UI_HOST, "--server.port", str(ui_port),
             "--server.headless", "true", "--server.maxUploadSize", "1000"],
            cwd=APP_ROOT, env=child_env(), stdout=ui_log, stderr=subprocess.STDOUT,
        )
        CHILDREN.append(ui_proc)

        # Streamlit answers /_stcore/health well before its root page finishes
        # the first render, so either endpoint counts as ready. A dashboard that
        # is merely slow must not tear down a healthy API, so a miss is a warning
        # and the stack keeps running - but a dashboard process that has ALREADY
        # exited never will, and saying "READY" over it is worse than failing.
        dashboard_ready = wait_for(
            [f"http://{UI_HOST}:{ui_port}/",
             f"http://{UI_HOST}:{ui_port}/_stcore/health"],
            UI_WAIT_SECONDS, "Dashboard", procs=[ui_proc],
        )
        if ui_proc.poll() is not None:
            log("ERROR: the dashboard process exited before it served the UI.")
            log(f"       See {os.path.join(LOG_DIR, 'streamlit.log')} for the reason.")
            return 1
        if not dashboard_ready:
            log("WARNING: the dashboard is taking longer than expected.")
            log("         The API is healthy and the dashboard normally appears")
            log("         shortly. Keep this window open, or raise RS_UI_WAIT.")

        log("=" * 56)
        log("READY" if dashboard_ready else "READY (dashboard still starting)")
        log(f"  Dashboard:  http://{UI_HOST}:{ui_port}")
        log(f"  API:        http://{API_HOST}:{api_port}")
        log(f"  API docs:   http://{API_HOST}:{api_port}/docs")
        log(f"  Logs:       {LOG_DIR}")
        log("  Stop:       close this window, press Ctrl+C, or run --stop")
        log("=" * 56)

        if OPEN_BROWSER:
            webbrowser.open(f"http://{UI_HOST}:{ui_port}")

        while True:
            for child in CHILDREN:
                code = child.poll()
                if code is not None:
                    log(f"A component exited (code {code}); shutting down the stack.")
                    return code or 0
            time.sleep(2)
    except KeyboardInterrupt:
        log("Shutting down...")
        return 0
    finally:
        cleanup()
        log("Stopped.")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # never die with a raw traceback the user can't read
        log(f"Unexpected error: {exc}")
        cleanup()
        raise SystemExit(1)
