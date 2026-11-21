"""指标核心逻辑的单元测试。"""

import math

import pytest

from app.metrics import (
    Exposure,
    MetricRangeError,
    evaluate_dataset,
    evaluate_session,
    truncated_weight,
)


def test_truncated_weight_caps_large_values():
    # 倾向 0.01 → 1/p = 100，截断上限 10 → 权重应为 10
    assert truncated_weight(0.01, 10.0) == 10.0
    # 倾向 0.5 → 1/p = 2，未触及上限
    assert truncated_weight(0.5, 10.0) == 2.0
    # 倾向 1.0 → 权重恒为 1
    assert truncated_weight(1.0, 5.0) == 1.0


def test_truncated_weight_rejects_invalid():
    with pytest.raises(ValueError):
        truncated_weight(0.0, 10.0)
    with pytest.raises(ValueError):
        truncated_weight(1.5, 10.0)
    with pytest.raises(ValueError):
        truncated_weight(0.5, 0.0)


def test_evaluate_session_math():
    exposures = [
        Exposure("A", clicked=1, propensity=0.5),   # 候选，w=2
        Exposure("B", clicked=0, propensity=0.25),  # 候选，w=4
        Exposure("C", clicked=1, propensity=0.5),   # 非候选，不计权重
    ]
    res = evaluate_session(exposures, {"A": 1, "B": 2}, cap=10.0)
    assert res["matched"] == 2
    assert res["sum_w"] == pytest.approx(6.0)
    # 截断 IPS = (2*1 + 4*0) / 3 条曝光 = 2/3
    assert res["ips_ctr"] == pytest.approx(2.0 / 3.0)
    # SNIPS = 2 / 6 = 1/3
    assert res["snips_ctr"] == pytest.approx(1.0 / 3.0)
    # ESS = 36 / (4+16) = 1.8
    assert res["ess"] == pytest.approx(1.8)
    assert res["uncovered"] == []
    assert res["baseline_ctr"] == pytest.approx(2.0 / 3.0)


def test_evaluate_session_truncation_applied():
    exposures = [Exposure("A", clicked=1, propensity=0.01)]
    res = evaluate_session(exposures, {"A": 1}, cap=10.0)
    # 权重被截断为 10 而非 100
    assert res["sum_w"] == pytest.approx(10.0)
    assert res["ips_ctr"] == pytest.approx(10.0)
    assert res["snips_ctr"] == pytest.approx(1.0)


def test_tiny_cap_keeps_ess_and_flags_insufficient_correctly():
    # 截断上限 1e-200：权重下溢到 1e-200 量级，原先 ESS 会因 w^2 下溢成 0
    exposures = [
        Exposure("A", clicked=1, propensity=0.5),
        Exposure("B", clicked=0, propensity=0.5),
    ]
    res = evaluate_session(exposures, {"A": 1, "B": 2}, cap=1e-200)
    assert res["sum_w"] == pytest.approx(2e-200, rel=1e-9)
    # ESS = (2e-200)^2 / (2 * 1e-400) = 2.0，按原定义仍达标
    assert res["ess"] == pytest.approx(2.0, rel=1e-12)
    assert res["ips_ctr"] == pytest.approx(5e-201, rel=1e-9)
    assert res["snips_ctr"] == pytest.approx(0.5, rel=1e-12)

    out = evaluate_dataset({"s": {"exposures": exposures, "candidates": {"A": 1, "B": 2}}}, 1e-200, 2.0)
    assert out["overall"]["included_sessions"] == 1
    assert out["per_session"][0]["status"] == "included"
    assert out["verdict"] in ("pass", "fail")


def test_huge_cap_ess_is_exact():
    # 截断上限 1e200：权重上溢到 1e200 量级，原先 w^2 上溢为 inf，ESS 变成 NaN
    exposures = [
        Exposure("A", clicked=1, propensity=1e-200),
        Exposure("B", clicked=0, propensity=1e-200),
    ]
    res = evaluate_session(exposures, {"A": 1, "B": 2}, cap=1e200)
    assert res["sum_w"] == pytest.approx(2e200, rel=1e-9)
    assert res["ess"] == pytest.approx(2.0, rel=1e-12)
    assert res["ips_ctr"] == pytest.approx(5e199, rel=1e-9)
    assert res["snips_ctr"] == pytest.approx(0.5, rel=1e-12)
    for key in ("sum_w", "ess", "ips_ctr", "snips_ctr", "baseline_ctr"):
        assert math.isfinite(res[key])

    out = evaluate_dataset({"s": {"exposures": exposures, "candidates": {"A": 1, "B": 2}}}, 1e200, 2.0)
    assert out["overall"]["included_sessions"] == 1
    assert math.isfinite(out["overall"]["ips_ctr"])
    assert out["verdict"] == "pass"


def test_huge_cap_low_ess_still_excluded():
    # 权重差异大时 ESS 依然按定义衡量样本量：1e200 与 1e100 各一个 → ESS≈1
    exposures = [
        Exposure("A", clicked=1, propensity=1e-200),
        Exposure("B", clicked=0, propensity=1e-100),
    ]
    res = evaluate_session(exposures, {"A": 1, "B": 2}, cap=1e200)
    assert res["ess"] == pytest.approx(1.0, rel=1e-12)
    out = evaluate_dataset({"s": {"exposures": exposures, "candidates": {"A": 1, "B": 2}}}, 1e200, 2.0)
    assert out["per_session"][0]["status"] == "excluded_insufficient"
    assert out["verdict"] == "no_data"


def test_metric_out_of_float_range_raises():
    # 会话指标本身超出双精度范围（权重和 2e308）→ 明确抛出 MetricRangeError
    exposures = [
        Exposure("A", clicked=1, propensity=1e-308),
        Exposure("B", clicked=0, propensity=1e-308),
    ]
    with pytest.raises(MetricRangeError) as exc:
        evaluate_session(exposures, {"A": 1, "B": 2}, cap=1e308)
    assert exc.value.field == "sum_w"

    # 逐会话评估时由 evaluate_dataset 补上会话号
    sessions = {"s": {"exposures": exposures, "candidates": {"A": 1, "B": 2}}}
    with pytest.raises(MetricRangeError) as exc2:
        evaluate_dataset(sessions, cap=1e308, min_ess=0.0)
    assert exc2.value.session_id == "s"


def test_truncated_weight_rejects_non_finite_cap():
    with pytest.raises(ValueError):
        truncated_weight(0.5, float("inf"))
    with pytest.raises(ValueError):
        truncated_weight(0.5, float("nan"))


def test_dataset_average_does_not_overflow_intermediately():
    # 两个会话各自 IPS≈1e308 量级，两数相加会溢出成 inf，但平均值仍在范围内
    sessions = {
        "a": {"exposures": [Exposure("A", 1, 1e-308)], "candidates": {"A": 1}},
        "b": {"exposures": [Exposure("B", 1, 0.5e-308)], "candidates": {"B": 1}},
    }
    res = evaluate_dataset(sessions, cap=1.5e308, min_ess=0.0)
    assert res["overall"]["included_sessions"] == 2
    assert math.isfinite(res["overall"]["ips_ctr"])
    assert res["overall"]["ips_ctr"] == pytest.approx(1.25e308, rel=1e-9)


def test_evaluate_dataset_exclusions():
    sessions = {
        "ok": {
            "exposures": [Exposure("A", 1, 0.5), Exposure("B", 0, 0.5)],
            "candidates": {"A": 1, "B": 2},
        },
        "has_uncovered": {
            "exposures": [Exposure("A", 1, 0.5)],
            "candidates": {"A": 1, "Z": 2},  # Z 无曝光 → 未覆盖
        },
        "no_match": {
            "exposures": [Exposure("Q", 1, 0.5)],
            "candidates": {},  # 无候选 → 样本不足
        },
    }
    res = evaluate_dataset(sessions, cap=10.0, min_ess=1.0)
    assert res["overall"]["included_sessions"] == 1
    assert res["overall"]["excluded_sessions"] == 2
    assert [u["item_id"] for u in res["uncovered_candidates"]] == ["Z"]
    assert res["uncovered_candidates"][0]["session_id"] == "has_uncovered"
    assert [s["session_id"] for s in res["insufficient_sessions"]] == ["no_match"]
    # 结论只来自 ok 会话：IPS = (2*1+2*0)/2 = 1.0，SNIPS = 2/4 = 0.5
    assert res["overall"]["ips_ctr"] == pytest.approx(1.0)
    assert res["overall"]["snips_ctr"] == pytest.approx(0.5)
    assert res["verdict"] == "pass"  # 1.0 >= baseline 0.5


def test_evaluate_dataset_ess_threshold():
    sessions = {
        "s": {
            "exposures": [Exposure("A", 1, 0.5), Exposure("B", 0, 0.25)],
            "candidates": {"A": 1, "B": 2},
        }
    }
    # ESS = 36/20 = 1.8 < 2.0 → 样本不足，结论为 no_data
    res = evaluate_dataset(sessions, cap=10.0, min_ess=2.0)
    assert res["overall"]["included_sessions"] == 0
    assert res["verdict"] == "no_data"
    assert res["insufficient_sessions"][0]["session_id"] == "s"
