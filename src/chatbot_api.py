from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
import requests


PROVIDERS = {
    "OpenRouter (Free)": {"base_url": "https://openrouter.ai/api/v1", "model": "openrouter/free", "key_env": "OPENROUTER_API_KEY", "kind": "openai"},
    "Groq (Free)": {"base_url": "https://api.groq.com/openai/v1", "model": "openai/gpt-oss-120b", "key_env": "GROQ_API_KEY", "kind": "openai"},
    "Google Gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta", "model": "gemini-2.5-flash", "key_env": "GEMINI_API_KEY", "kind": "gemini"},
    "Hugging Face": {"base_url": "https://router.huggingface.co/v1", "model": "openai/gpt-oss-120b:groq", "key_env": "HF_TOKEN", "kind": "openai"},
    "Ollama (Local)": {"base_url": "http://localhost:11434/v1", "model": "gemma3", "key_env": "", "kind": "openai"},
    "OpenAI": {"base_url": "https://api.openai.com/v1", "model": "gpt-5.6-luna", "key_env": "OPENAI_API_KEY", "kind": "openai"},
}


@dataclass
class ChatbotConfig:
    provider: str = "OpenRouter (Free)"
    api_key: str = ""
    model: str = ""
    base_url: str = ""
    timeout: int = 60

    @classmethod
    def from_env(cls, provider: str | None = None, api_key: str | None = None, model: str | None = None, base_url: str | None = None):
        provider = provider or os.getenv("AI_PROVIDER", "OpenRouter (Free)")
        meta = PROVIDERS.get(provider, PROVIDERS["OpenRouter (Free)"])
        return cls(provider=provider, api_key=(api_key if api_key is not None else os.getenv(meta["key_env"], "")).strip(), model=(model or os.getenv("AI_MODEL", meta["model"])).strip(), base_url=(base_url or os.getenv("AI_BASE_URL", meta["base_url"])).rstrip("/"), timeout=int(os.getenv("AI_TIMEOUT", "60")))

    @property
    def enabled(self) -> bool:
        return self.provider == "Ollama (Local)" or bool(self.api_key)


def build_tools() -> list[dict[str, Any]]:
    return [
        {"type":"function","function":{"name":"get_live_status","description":"Get current ReturnShield live server status.","parameters":{"type":"object","properties":{},"additionalProperties":False}}},
        {"type":"function","function":{"name":"get_operations_summary","description":"Summarize current active returns, decisions, risk, and return value.","parameters":{"type":"object","properties":{},"additionalProperties":False}}},
        {"type":"function","function":{"name":"search_returns","description":"Search current returns by return/customer ID, decision, or highest risk.","parameters":{"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer","minimum":1,"maximum":50}},"required":["query","limit"],"additionalProperties":False}}},
        {"type":"function","function":{"name":"inspect_return","description":"Inspect one return request.","parameters":{"type":"object","properties":{"return_id":{"type":"string"}},"required":["return_id"],"additionalProperties":False}}},
        {"type":"function","function":{"name":"search_customer","description":"Find customer return history.","parameters":{"type":"object","properties":{"customer_id":{"type":"string"},"limit":{"type":"integer","minimum":1,"maximum":100}},"required":["customer_id","limit"],"additionalProperties":False}}},
        {"type":"function","function":{"name":"get_coordinated_accounts","description":"Find coordinated abuse infrastructure and high-risk accounts.","parameters":{"type":"object","properties":{"limit":{"type":"integer","minimum":1,"maximum":100}},"required":["limit"],"additionalProperties":False}}},
        {"type":"function","function":{"name":"get_model_metrics","description":"Get held-out model and business metrics.","parameters":{"type":"object","properties":{},"additionalProperties":False}}},
        {"type":"function","function":{"name":"control_live_server","description":"Start, stop, or generate transactions on the built-in live demo server.","parameters":{"type":"object","properties":{"operation":{"type":"string","enum":["start","stop","generate"]},"count":{"type":"integer","minimum":1,"maximum":1000}},"required":["operation"],"additionalProperties":False}}},
    ]


class ReturnShieldChatbot:
    """Multi-provider general chatbot with ReturnShield tool calling."""
    def __init__(self, config: ChatbotConfig): self.config = config

    def _system(self, active_df, report, live_status):
        total = len(active_df) if isinstance(active_df, pd.DataFrame) else 0
        return f"""You are ReturnShield AI, a general-purpose assistant embedded in a merchant loss-prevention product. Answer ANY normal question naturally. For ReturnShield questions, use tools when current data is needed and never invent values. Distinguish predicted risk from observed/simulated abuse outcomes. State that demo data is synthetic when relevant. Actions (start/stop/generate) require explicit user intent. Be concise, useful, and conversational.\n\nCurrent context: active records={total}; live_status={json.dumps(live_status or {}, default=str)}; report_available={bool(report)}."""

    def _openai_chat(self, messages, tools, handlers):
        """OpenAI-compatible chat with real ReturnShield tool calling."""
        url = self.config.base_url.rstrip('/') + '/chat/completions'
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        body = {"model": self.config.model, "messages": messages, "tools": tools, "tool_choice": "auto", "temperature": 0.3}
        last_data, action = [], None
        for _ in range(8):
            r = requests.post(url, headers=headers, json=body, timeout=self.config.timeout)
            r.raise_for_status()
            data = r.json()
            msg = data.get("choices", [{}])[0].get("message", {})
            calls = msg.get("tool_calls") or []
            if not calls:
                return msg.get("content") or "I couldn't produce a response.", last_data, action
            messages.append(msg)
            for call in calls:
                fn = call.get("function", {})
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    args = {}
                try:
                    result = handlers[name](**args) if name in handlers else {"error": f"Unknown tool: {name}"}
                except Exception as exc:
                    result = {"error": str(exc)}
                if isinstance(result, dict):
                    if "_table" in result:
                        tbl = result.pop("_table")
                        if isinstance(tbl, list):
                            last_data = tbl
                    if result.get("_action"):
                        action = result.get("_action")
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": json.dumps(result, default=str),
                })
            body["messages"] = messages
        return "I couldn't complete that request.", last_data, action

    def _gemini(self, messages, handlers):
        """Gemini native REST with real function calling into ReturnShield tools."""
        url = f"{self.config.base_url}/models/{self.config.model}:generateContent?key={self.config.api_key}"
        system = messages[0]["content"] if messages and messages[0].get("role") == "system" else ""

        declarations = []
        for tool in build_tools():
            fn = tool["function"]
            declarations.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            })

        contents = []
        for m in messages:
            role = m.get("role")
            if role == "system":
                continue
            if role == "tool":
                # Tool responses are represented as functionResponse parts by the loop below.
                continue
            contents.append({"role": "model" if role == "assistant" else "user", "parts": [{"text": str(m.get("content", ""))}]})

        last_data, action = [], None
        for _ in range(8):
            body = {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": contents,
                "tools": [{"functionDeclarations": declarations}],
                "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
                "generationConfig": {"temperature": 0.3},
            }
            r = requests.post(url, json=body, timeout=self.config.timeout)
            r.raise_for_status()
            payload = r.json()
            candidate = payload.get("candidates", [{}])[0]
            parts = candidate.get("content", {}).get("parts", [])
            function_calls = [p.get("functionCall") for p in parts if p.get("functionCall")]
            text = "".join(p.get("text", "") for p in parts if p.get("text"))
            if not function_calls:
                return text or "I couldn't produce a response.", last_data, action

            # Preserve the model turn before sending function responses.
            contents.append({"role": "model", "parts": parts})
            response_parts = []
            for fc in function_calls:
                name = fc.get("name", "")
                args = fc.get("args") or {}
                try:
                    result = handlers[name](**args) if name in handlers else {"error": f"Unknown tool: {name}"}
                except Exception as exc:
                    result = {"error": str(exc)}
                if isinstance(result, dict):
                    if "_table" in result:
                        tbl = result.pop("_table")
                        if isinstance(tbl, list):
                            last_data = tbl
                    if result.get("_action"):
                        action = result.get("_action")
                response_parts.append({"functionResponse": {"name": name, "response": result}})
            contents.append({"role": "user", "parts": response_parts})
        return "I couldn't complete that request.", last_data, action

    def chat(self,message,history,active_df,report,live_status,tool_handlers):
        if not self.config.enabled:
            return {"enabled":False,"answer":f"{self.config.provider} is not configured. Add its key in AI settings, or select Ollama (Local) if you have Ollama installed.","data":[],"action":None,"provider":self.config.provider}
        msgs=[{"role":"system","content":self._system(active_df,report,live_status)}]
        for x in (history or [])[-20:]:
            if x.get("role") in {"user","assistant"} and x.get("content"): msgs.append({"role":x["role"],"content":str(x["content"])})
        msgs.append({"role":"user","content":message})
        try:
            if PROVIDERS.get(self.config.provider,{}).get("kind")=="gemini":
                answer,data,action=self._gemini(msgs,tool_handlers)
            else:
                answer,data,action=self._openai_chat(msgs,build_tools(),tool_handlers)
            return {"enabled":True,"answer":answer,"data":data,"action":action,"provider":self.config.provider,"model":self.config.model}
        except Exception as exc:
            return {"enabled":False,"answer":f"{self.config.provider} request failed: {exc}","data":[],"action":None,"provider":self.config.provider,"model":self.config.model}
