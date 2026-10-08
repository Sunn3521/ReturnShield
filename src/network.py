from __future__ import annotations

from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

INFRASTRUCTURE_COLUMNS = ("device_id", "address_id", "payment_fingerprint")

_infra_cache: dict[str, pd.DataFrame] = {}


def infrastructure_lookup(data_dir: str = "data/raw") -> pd.DataFrame:
    """customer_id -> shared-infrastructure identifiers, cached per data dir.

    Only the three identifiers cluster analysis needs; keeping the cache narrow
    means a regenerated ``customers.csv`` stays cheap to reload.
    """
    cached = _infra_cache.get(data_dir)
    if cached is not None:
        return cached
    path = Path(data_dir) / "customers.csv"
    if not path.exists():
        _infra_cache[data_dir] = pd.DataFrame(columns=["customer_id", *INFRASTRUCTURE_COLUMNS])
        return _infra_cache[data_dir]
    cols = ["customer_id", *[c for c in INFRASTRUCTURE_COLUMNS if c in _read_header(path)]]
    lookup = pd.read_csv(path, usecols=cols)
    lookup["customer_id"] = lookup["customer_id"].astype(str).str.strip().str.upper()
    for col in INFRASTRUCTURE_COLUMNS:
        if col in lookup.columns:
            lookup[col] = lookup[col].astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA})
    _infra_cache[data_dir] = lookup
    return lookup


def _read_header(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as fh:
        return fh.readline().strip().split(",")


def attach_infrastructure_ids(df: pd.DataFrame, data_dir: str = "data/raw") -> pd.DataFrame:
    """Restore infrastructure identifiers on a frame that only kept aggregates.

    Scored exports (``reports/test_predictions.csv``) carry
    ``device_linked_accounts`` but not the raw ids, so coordinated-account
    analysis silently found nothing. The ids live in ``customers.csv`` keyed by
    ``customer_id``, so join them back on. Frames that already carry an id are
    left untouched, and nothing is invented when the lookup misses.
    """
    if df.empty or "customer_id" not in df.columns:
        return df
    missing = [c for c in INFRASTRUCTURE_COLUMNS if c not in df.columns]
    if not missing:
        return df
    lookup = infrastructure_lookup(data_dir)
    if lookup.empty:
        return df
    want = [c for c in missing if c in lookup.columns]
    if not want:
        return df
    out = df.copy()
    out["_rs_cust_key"] = out["customer_id"].astype(str).str.strip().str.upper()
    out = out.merge(
        lookup.rename(columns={"customer_id": "_rs_cust_key"})[["_rs_cust_key", *want]],
        on="_rs_cust_key", how="left",
    ).drop(columns=["_rs_cust_key"])
    return out


def build_abuse_graph(data_dir: str = "data/raw", min_cluster_size: int = 2) -> tuple[nx.Graph, pd.DataFrame]:
    p = Path(data_dir)
    customers = pd.read_csv(p / "customers.csv")
    returns = pd.read_csv(p / "returns.csv")
    outcomes = pd.read_csv(p / "return_outcomes.csv")
    
    # Aggregate customer level return statistics
    cust_returns = returns.merge(outcomes, on="return_id", how="left")
    cust_stats = cust_returns.groupby("customer_id").agg(
        total_returns=("return_id", "count"),
        abusive_returns=("abusive_return", "sum"),
        total_refund_value=("return_value", "sum")
    ).reset_index()
    
    customers = customers.merge(cust_stats, on="customer_id", how="left").fillna(0)
    
    G = nx.Graph()
    
    # Add nodes and edges
    for _, row in customers.iterrows():
        cid = str(row["customer_id"])
        did = str(row["device_id"])
        aid = str(row["address_id"])
        pid = str(row["payment_fingerprint"])
        
        c_type = str(row.get("latent_type", "normal"))
        returns_cnt = int(row["total_returns"])
        abusive_cnt = int(row["abusive_returns"])
        
        G.add_node(cid, type="customer", latent_type=c_type, returns=returns_cnt, abusive=abusive_cnt, refund=float(row["total_refund_value"]))
        G.add_node(did, type="device")
        G.add_node(aid, type="address")
        G.add_node(pid, type="payment")
        
        G.add_edge(cid, did, relation="uses_device")
        G.add_edge(cid, aid, relation="ships_to")
        G.add_edge(cid, pid, relation="pays_with")
        
    # Extract clusters / connected components
    clusters = []
    for cluster_id, comp in enumerate(nx.connected_components(G)):
        subG = G.subgraph(comp)
        cust_nodes = [n for n, d in subG.nodes(data=True) if d.get("type") == "customer"]
        if len(cust_nodes) >= min_cluster_size:
            tot_ret = sum(subG.nodes[n]["returns"] for n in cust_nodes)
            tot_abu = sum(subG.nodes[n]["abusive"] for n in cust_nodes)
            tot_ref = sum(subG.nodes[n]["refund"] for n in cust_nodes)
            clusters.append({
                "cluster_id": f"CLUSTER_{cluster_id:03d}",
                "node_count": len(comp),
                "customer_count": len(cust_nodes),
                "total_returns": tot_ret,
                "abusive_returns": tot_abu,
                "cluster_abuse_rate": tot_abu / max(tot_ret, 1),
                "total_refund_value": tot_ref,
                "customers": cust_nodes,
                "nodes": list(comp)
            })
            
    cluster_df = pd.DataFrame(clusters).sort_values("cluster_abuse_rate", ascending=False).reset_index(drop=True) if clusters else pd.DataFrame()
    return G, cluster_df


def plot_cluster_graph(G: nx.Graph, cluster_nodes: list[str], title: str = "Suspicious Abuse Ring Cluster") -> go.Figure:
    subG = G.subgraph(cluster_nodes)
    pos = nx.spring_layout(subG, seed=42, k=0.5)
    
    edge_x = []
    edge_y = []
    for edge in subG.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.2, color='#D6DBE1'),
        hoverinfo='none',
        mode='lines'
    )
    
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    node_symbol = []
    
    for node in subG.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        data = subG.nodes[node]
        ntype = data.get("type", "unknown")
        
        if ntype == "customer":
            node_symbol.append("circle")
            node_size.append(24)
            ret = data.get("returns", 0)
            abu = data.get("abusive", 0)
            ref = data.get("refund", 0.0)
            if abu > 0 or data.get("latent_type") in ("abusive", "coordinated") or float(data.get("risk", 0.0)) >= 0.68:
                node_color.append("#D13434") # Crimson Red for high risk
            else:
                node_color.append("#049FD9") # Blue for normal customer
            node_text.append(f"<b>Customer {node}</b><br>Returns: {ret}<br>Abusive: {abu}<br>Refunds: ₹{ref:,.0f}")
        else:
            node_symbol.append("diamond")
            node_size.append(18)
            node_color.append("#EAA200") # Orange for infrastructure node
            node_text.append(f"<b>{ntype.capitalize()}: {node}</b>")
            
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[n for n in subG.nodes()],
        textposition="top center",
        hovertext=node_text,
        textfont=dict(color="#1A2432", size=10),
        hovertemplate="%{hovertext}<extra></extra>",
        hoverlabel=dict(bgcolor='#1A2432', bordercolor='#2C3849', font=dict(color='white', size=12)),
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            symbol=node_symbol,
            line_width=2,
            line=dict(color='#FFFFFF')
        )
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title=dict(text=title, font=dict(size=16)),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=20,r=20,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    template="plotly_white",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                 ))
    return fig


def build_active_abuse_graph(data: pd.DataFrame, risk_threshold: float = 0.70, min_cluster_size: int = 2) -> tuple[nx.Graph, pd.DataFrame]:
    """Build a graph from the currently active/live scored records.

    Only high-risk records are included. Infrastructure IDs are intentionally non-sensitive
    synthetic identifiers or merchant-provided pseudonymous identifiers.
    """
    required = {"customer_id", "device_id", "address_id", "payment_fingerprint"}
    if not required.issubset(data.columns):
        return nx.Graph(), pd.DataFrame()

    df = data.copy()
    # Keep all current records in the graph so shared infrastructure remains visible;
    # risk is used to classify/highlight customers, not to delete their relationships.
    G = nx.Graph()

    for _, row in df.iterrows():
        cid = str(row["customer_id"])
        did = str(row["device_id"])
        aid = str(row["address_id"])
        pid = str(row["payment_fingerprint"])
        risk = float(row.get("risk_probability", 0.0))
        decision = str(row.get("decision", ""))
        G.add_node(cid, type="customer", risk=risk, decision=decision, return_id=str(row.get("return_id", "")))
        G.add_node(did, type="device")
        G.add_node(aid, type="address")
        G.add_node(pid, type="payment")
        G.add_edge(cid, did, relation="uses_device")
        G.add_edge(cid, aid, relation="ships_to")
        G.add_edge(cid, pid, relation="pays_with")

    clusters = []
    for cluster_id, comp in enumerate(nx.connected_components(G)):
        sg = G.subgraph(comp)
        customers = [n for n, d in sg.nodes(data=True) if d.get("type") == "customer"]
        if len(customers) >= min_cluster_size:
            risks = [float(sg.nodes[n].get("risk", 0.0)) for n in customers]
            clusters.append({
                "cluster_id": f"LIVE_CLUSTER_{cluster_id:04d}",
                "customer_count": len(customers),
                "node_count": len(comp),
                "avg_risk": float(np.mean(risks)) if risks else 0.0,
                "max_risk": float(np.max(risks)) if risks else 0.0,
                "high_risk_customers": int(sum(r >= risk_threshold for r in risks)),
                "customers": customers,
                "nodes": list(comp),
            })

    cdf = pd.DataFrame(clusters)
    if not cdf.empty:
        cdf = cdf.sort_values(["high_risk_customers", "avg_risk", "customer_count"], ascending=[False, False, False]).reset_index(drop=True)
    return G, cdf
