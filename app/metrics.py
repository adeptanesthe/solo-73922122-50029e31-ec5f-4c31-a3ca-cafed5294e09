"""反事实评估核心指标：截断 IPS 与自归一化（SNIPS）点击率。

纯函数实现，不依赖 FastAPI / SQLite，便于单元测试。

定义（按会话）：
- 权重 w_i = min(1 / propensity_i, cap)，cap 为用户输入的截断上限
- 截断 IPS 点击率  = sum(w_i * click_i) / n_logged_events
- 自归一化点击率    = sum(w_i * click_i) / sum(w_i)
- 有效样本量 ESS    = (sum w_i)^2 / sum(w_i^2)
仅对"新策略候选集中、且有日志曝光"的事件计入权重。

实现说明：权重及求和用 :class:`fractions.Fraction` 精确计算，最后才转换为
双精度浮点数。这样即使截断上限很小（权重下溢到 1e-200 量级）或很大
（权重达到 1e200 量级），ESS 等比值型指标仍按原定义得到正确结果，而不会
因为中间量平方/求和溢出或下溢而误判为 0 或 NaN。只有最终结果本身超出
有限浮点数范围时，才抛出 :class:`MetricRangeError`。
"""

import math
from dataclasses import dataclass
from fractions import Fraction

FIELD_LABELS = {
    "weight": "逆倾向权重 w",
    "sum_w": "权重和 Σw",
    "ess": "有效样本量 ESS",
    "ips_ctr": "截断 IPS 点击率",
    "snips_ctr": "SNIPS 点击率",
    "baseline_ctr": "旧策略实测点击率",
}


class MetricRangeError(ValueError):
    """指标或中间结果超出双精度浮点的有限可表示范围。"""

    def __init__(self, message: str, session_id: str | None = None, field: str | None = None):
        super().__init__(message)
        self.session_id = session_id
        self.field = field


@dataclass
class Exposure:
    item_id: str
    clicked: int
    propensity: float


def _to_finite_float(value, field: str, session_id: str | None = None) -> float:
    """把精确分数转换为 float；溢出或非有限值时抛出 MetricRangeError。"""
    label = FIELD_LABELS.get(field, field)
    where = f"会话 {session_id} 的" if session_id is not None else "总体"
    message = f"{where}{label} 超出有限浮点数（双精度）范围"
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise MetricRangeError(message, session_id=session_id, field=field) from exc
    if not math.isfinite(result):
        raise MetricRangeError(message, session_id=session_id, field=field)
    return result


def _weight_fraction(propensity: float, cap: float) -> Fraction:
    if not 0 < propensity <= 1:
        raise ValueError(f"propensity 必须在 (0,1] 区间，得到 {propensity}")
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError(f"截断上限必须是有限正数，得到 {cap}")
    return min(Fraction(1) / Fraction(propensity), Fraction(cap))


def truncated_weight(propensity: float, cap: float) -> float:
    """截断逆倾向权重：min(1/p, cap)。"""
    return _to_finite_float(_weight_fraction(propensity, cap), "weight")


def evaluate_session(exposures, candidate_ranks: dict, cap: float, session_id: str | None = None) -> dict:
    """评估单个会话。

    exposures: list[Exposure] 该会话全部日志曝光
    candidate_ranks: dict item_id -> 新策略排序位次
    session_id: 可选，仅用于超出浮点范围时的错误信息。
    """
    logged_items = {e.item_id for e in exposures}
    uncovered = [
        {"item_id": item, "new_rank": candidate_ranks[item]}
        for item in candidate_ranks
        if item not in logged_items
    ]
    uncovered.sort(key=lambda x: x["new_rank"])

    matched = [e for e in exposures if e.item_id in candidate_ranks]
    n_events = len(exposures)
    weights = [_weight_fraction(e.propensity, cap) for e in matched]
    sum_w = sum(weights, Fraction(0))
    sum_wc = sum((w for w, e in zip(weights, matched) if e.clicked), Fraction(0))
    sum_w2 = sum((w * w for w in weights), Fraction(0))
    ess = (sum_w * sum_w / sum_w2) if sum_w2 > 0 else Fraction(0)

    n_clicks = sum(1 for e in exposures if e.clicked)
    return {
        "n_events": n_events,
        "n_candidates": len(candidate_ranks),
        "matched": len(matched),
        "sum_w": _to_finite_float(sum_w, "sum_w", session_id),
        "ess": _to_finite_float(ess, "ess", session_id),
        "ips_ctr": _to_finite_float(
            sum_wc / n_events if n_events else Fraction(0), "ips_ctr", session_id
        ),
        "snips_ctr": _to_finite_float(sum_wc / sum_w, "snips_ctr", session_id)
        if sum_w > 0
        else None,
        "baseline_ctr": _to_finite_float(Fraction(n_clicks, n_events), "baseline_ctr", session_id)
        if n_events
        else None,
        "uncovered": uncovered,
    }


def _mean(values, field: str) -> float:
    """算术平均；用精确分数求和，避免大数相加时中间溢出。"""
    total = sum((Fraction(v) for v in values), Fraction(0))
    return _to_finite_float(total / len(values), field)


def _round_metric(value):
    """展示用四舍五入到 6 位小数；极小的非零值保留原值，避免变成 0。"""
    if value is None:
        return None
    rounded = round(value, 6)
    if rounded == 0.0 and value != 0.0:
        return value
    return rounded


def evaluate_dataset(sessions: dict, cap: float, min_ess: float) -> dict:
    """评估整个数据集。

    sessions: dict session_id -> {"exposures": [Exposure], "candidates": {item_id: rank}}

    排除规则（被排除的会话不计入通过结论）：
    1. 存在新策略候选未被日志覆盖（无对应曝光、拿不到倾向分数）→ excluded_uncovered
    2. 无任何匹配候选曝光，或有效样本量 ESS < min_ess → excluded_insufficient
    """
    per_session = []
    uncovered_list = []
    insufficient_list = []
    included = []

    for session_id in sorted(sessions):
        data = sessions[session_id]
        res = evaluate_session(data["exposures"], data["candidates"], cap, session_id)

        status = "included"
        reason = None
        if res["uncovered"]:
            status = "excluded_uncovered"
            reason = "存在未被日志覆盖的新策略候选"
            for u in res["uncovered"]:
                uncovered_list.append({"session_id": session_id, **u})
        elif res["matched"] == 0:
            status = "excluded_insufficient"
            reason = "无匹配的新策略候选曝光"
            insufficient_list.append(
                {"session_id": session_id, "reason": reason, "matched": 0, "ess": res["ess"]}
            )
        elif res["ess"] < min_ess:
            status = "excluded_insufficient"
            reason = f"有效样本量 ESS={res['ess']:.3f} 低于阈值 {min_ess}"
            insufficient_list.append(
                {"session_id": session_id, "reason": reason, "matched": res["matched"], "ess": res["ess"]}
            )

        row = {
            "session_id": session_id,
            "n_events": res["n_events"],
            "n_candidates": res["n_candidates"],
            "matched": res["matched"],
            "sum_w": _round_metric(res["sum_w"]),
            "ess": _round_metric(res["ess"]),
            "ips_ctr": _round_metric(res["ips_ctr"]),
            "snips_ctr": _round_metric(res["snips_ctr"]),
            "baseline_ctr": _round_metric(res["baseline_ctr"]),
            "status": status,
            "reason": reason,
        }
        per_session.append(row)
        if status == "included":
            included.append((row, res))

    if included:
        included_sessions = len(included)
        ips_mean = _mean([res["ips_ctr"] for _, res in included], "ips_ctr")
        snips_mean = _mean([res["snips_ctr"] for _, res in included], "snips_ctr")
        baseline_mean = _mean([res["baseline_ctr"] for _, res in included], "baseline_ctr")
        overall = {
            "included_sessions": included_sessions,
            "excluded_sessions": len(per_session) - included_sessions,
            "total_events": sum(row["n_events"] for row, _ in included),
            "ips_ctr": _round_metric(ips_mean),
            "snips_ctr": _round_metric(snips_mean),
            "baseline_ctr": _round_metric(baseline_mean),
        }
        # 判定使用未四舍五入的精确平均值，避免极小/极大指标被舍入改变结论
        verdict = "pass" if ips_mean >= baseline_mean else "fail"
    else:
        overall = {
            "included_sessions": 0,
            "excluded_sessions": len(per_session),
            "total_events": 0,
            "ips_ctr": None,
            "snips_ctr": None,
            "baseline_ctr": None,
        }
        verdict = "no_data"

    return {
        "cap": cap,
        "min_ess": min_ess,
        "overall": overall,
        "verdict": verdict,
        "per_session": per_session,
        "uncovered_candidates": uncovered_list,
        "insufficient_sessions": insufficient_list,
    }
