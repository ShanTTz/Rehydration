from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

import numpy as np

from .schema import CommentNode, ThreadState
from .simulator import StepTrace


def compute_metrics(state: ThreadState, traces: Iterable[StepTrace] = ()) -> Dict[str, float]:
    trace_items = list(traces)
    intent_requests = sum(item.intent_requests for item in trace_items)
    pool_exhausted = sum(item.pool_exhausted for item in trace_items)
    supportive_requests = sum(
        item.supportive_intent_requests for item in trace_items
    )
    antagonistic_requests = sum(
        item.antagonistic_intent_requests for item in trace_items
    )
    supportive_exhausted = sum(
        item.supportive_pool_exhausted for item in trace_items
    )
    antagonistic_exhausted = sum(
        item.antagonistic_pool_exhausted for item in trace_items
    )
    pool_metrics = {
        "intent_requests": float(intent_requests),
        "pool_exhausted": float(pool_exhausted),
        "pool_exhaustion_rate": float(pool_exhausted / intent_requests)
        if intent_requests
        else 0.0,
        "supportive_pool_exhausted": float(supportive_exhausted),
        "antagonistic_pool_exhausted": float(antagonistic_exhausted),
        "supportive_intent_requests": float(supportive_requests),
        "antagonistic_intent_requests": float(antagonistic_requests),
        "supportive_pool_exhaustion_rate": float(
            supportive_exhausted / supportive_requests
        )
        if supportive_requests
        else 0.0,
        "antagonistic_pool_exhaustion_rate": float(
            antagonistic_exhausted / antagonistic_requests
        )
        if antagonistic_requests
        else 0.0,
    }
    all_nodes = list(state.comments.values())
    nodes = [node for node in all_nodes if not _is_root_post(node)]
    if not nodes:
        return {
            "comment_volume": 0,
            "engagement_volume": float(sum(node.likes + node.dislikes for node in all_nodes)),
            "max_depth": 0,
            "mean_leaf_depth": 0.0,
            "depth_variance": 0.0,
            "toxic_density": 0.0,
            "conflict_spiral_rate": 0.0,
            "external_reactions": float(state.external_reactions),
            **pool_metrics,
        }

    depths = np.array([node.depth for node in nodes], dtype=float)
    leaves = [node for node in nodes if node.reply_count == 0]
    leaf_depths = np.array([node.depth for node in leaves], dtype=float)
    toxic_nodes = [node for node in nodes if node.toxicity >= 0.30 or node.metadata.get("polarity") == "antagonistic"]
    reactions = sum(node.likes + node.dislikes for node in all_nodes)
    trace_replies = sum(item.replies for item in trace_items)

    return {
        "comment_volume": float(len(nodes)),
        "engagement_volume": float(len(nodes) + reactions),
        "max_depth": float(depths.max()),
        "mean_leaf_depth": float(leaf_depths.mean()) if len(leaf_depths) else 0.0,
        "depth_variance": float(depths.var()),
        "toxic_density": float(len(toxic_nodes) / len(nodes)),
        "conflict_spiral_rate": _conflict_spiral_rate(nodes),
        "external_reactions": float(state.external_reactions),
        "reply_actions": float(trace_replies),
        **pool_metrics,
    }


def _conflict_spiral_rate(nodes: List[CommentNode]) -> float:
    by_parent = {node.node_id: node.parent_id for node in nodes}
    node_by_id = {node.node_id: node for node in nodes}
    toxic_chain_nodes = 0
    for node in nodes:
        if node.toxicity < 0.30 and node.metadata.get("polarity") != "antagonistic":
            continue
        parent_id = by_parent.get(node.node_id)
        while parent_id:
            parent = node_by_id.get(parent_id)
            if parent is None:
                break
            if parent.toxicity >= 0.30 or parent.metadata.get("polarity") == "antagonistic":
                toxic_chain_nodes += 1
                break
            parent_id = by_parent.get(parent_id)
    return toxic_chain_nodes / max(1, len(nodes))


def _is_root_post(node: CommentNode) -> bool:
    return bool(node.metadata.get("root_post"))


def summarize_records(records: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for record in records:
        grouped[str(record["condition"])].append(record)
    summary: Dict[str, Dict[str, float]] = {}
    metrics = [
        "comment_volume",
        "engagement_volume",
        "max_depth",
        "mean_leaf_depth",
        "depth_variance",
        "toxic_density",
        "conflict_spiral_rate",
        "external_reactions",
    ]
    for condition, rows in grouped.items():
        out: Dict[str, float] = {"n": float(len(rows))}
        for metric in metrics:
            values = np.array([float(row.get(metric, 0.0)) for row in rows], dtype=float)
            out[f"{metric}_mean"] = float(values.mean()) if len(values) else 0.0
            out[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary[condition] = out
    return summary


def cohen_d(a: Iterable[float], b: Iterable[float]) -> float:
    av = np.array(list(a), dtype=float)
    bv = np.array(list(b), dtype=float)
    if len(av) < 2 or len(bv) < 2:
        return 0.0
    pooled = np.sqrt(((len(av) - 1) * av.var(ddof=1) + (len(bv) - 1) * bv.var(ddof=1)) / (len(av) + len(bv) - 2))
    return 0.0 if pooled == 0 else float((av.mean() - bv.mean()) / pooled)
