"""API 端到端测试：导入去重、评估、排除规则、修改后重新评估。"""

import json

import pytest

from app import db

from .conftest import SAMPLE_CSV, import_csv


def test_import_and_dedup_same_event_no(client):
    r = import_csv(client, SAMPLE_CSV)
    assert r.status_code == 200
    body = r.json()
    assert body["exposure_rows"] == 7
    assert body["inserted"] == 7
    assert body["duplicates_skipped"] == 0

    # 追加同一文件：相同事件号只处理一次，全部跳过
    r2 = import_csv(client, SAMPLE_CSV)
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["inserted"] == 0
    assert body2["duplicates_skipped"] == 7

    status = client.get("/api/status").json()
    assert status["events"] == 7
    assert status["sessions"] == 4


def test_import_validation_error(client):
    bad = "session_id,event_no,item_id,position,clicked,propensity,new_rank\ns1,e1,A,1,1,0,1\n"
    r = import_csv(client, bad)
    assert r.status_code == 400
    assert "propensity" in r.json()["detail"]


def test_import_missing_column(client):
    bad = "session_id,event_no,item_id\ns1,e1,A\n"
    r = import_csv(client, bad)
    assert r.status_code == 400
    assert "缺少必需列" in r.json()["detail"]


def test_evaluate_flow_and_exclusions(client):
    assert import_csv(client, SAMPLE_CSV).status_code == 200
    r = client.post("/api/evaluate", json={"cap": 10.0, "min_ess": 1.0})
    assert r.status_code == 200
    res = r.json()

    # s1: IPS=(2*1+4*0)/3=2/3, SNIPS=2/6=1/3, baseline=2/3
    # s2: IPS=(2*1+2*0)/2=1.0, SNIPS=2/4=0.5, baseline=0.5
    # s3: 候选 F 未覆盖 → 排除；s4: 无候选 → 样本不足排除
    overall = res["overall"]
    assert overall["included_sessions"] == 2
    assert overall["excluded_sessions"] == 2
    assert overall["ips_ctr"] == pytest.approx((2 / 3 + 1.0) / 2, abs=1e-6)
    assert overall["snips_ctr"] == pytest.approx((1 / 3 + 0.5) / 2, abs=1e-6)
    assert overall["baseline_ctr"] == pytest.approx((2 / 3 + 0.5) / 2, abs=1e-6)
    assert res["verdict"] == "pass"

    # 未覆盖候选单列，且不计入结论
    assert res["uncovered_candidates"] == [{"session_id": "s3", "item_id": "F", "new_rank": 1}]
    # 样本不足单列
    assert len(res["insufficient_sessions"]) == 1
    assert res["insufficient_sessions"][0]["session_id"] == "s4"

    status_by_session = {s["session_id"]: s["status"] for s in res["per_session"]}
    assert status_by_session == {
        "s1": "included",
        "s2": "included",
        "s3": "excluded_uncovered",
        "s4": "excluded_insufficient",
    }


def test_uncovered_session_not_counted_in_conclusion(client):
    # 只有一个会话且存在未覆盖候选 → 无有效会话，无法得出结论
    csv_text = (
        "session_id,event_no,item_id,position,clicked,propensity,new_rank\n"
        "s1,e1,A,1,1,0.5,1\n"
        "s1,,B,,,,2\n"
    )
    assert import_csv(client, csv_text).status_code == 200
    res = client.post("/api/evaluate", json={"cap": 10.0, "min_ess": 1.0}).json()
    assert res["overall"]["included_sessions"] == 0
    assert res["overall"]["ips_ctr"] is None
    assert res["verdict"] == "no_data"
    assert res["uncovered_candidates"][0]["item_id"] == "B"


def test_reevaluate_after_log_modification(client):
    assert import_csv(client, SAMPLE_CSV).status_code == 200
    first = client.post("/api/evaluate", json={"cap": 10.0, "min_ess": 1.0}).json()

    # 修改日志：替换为一份新日志后重新评估
    # 候选 A、B 均无点击（IPS=0），非候选 C 有点击（旧策略实测 CTR=1/3）→ 应判未通过
    new_csv = (
        "session_id,event_no,item_id,position,clicked,propensity,new_rank\n"
        "n1,x1,A,1,0,0.5,1\n"
        "n1,x2,B,2,0,0.5,2\n"
        "n1,x3,C,3,1,0.5,\n"
    )
    r = import_csv(client, new_csv, mode="replace")
    assert r.status_code == 200
    assert client.get("/api/status").json()["events"] == 3

    second = client.post("/api/evaluate", json={"cap": 10.0, "min_ess": 1.0}).json()
    assert second["id"] != first["id"]
    assert second["overall"]["included_sessions"] == 1
    assert second["overall"]["ips_ctr"] == pytest.approx(0.0)
    assert second["overall"]["baseline_ctr"] == pytest.approx(1 / 3, abs=1e-6)
    assert second["verdict"] == "fail"  # 0 < baseline 1/3

    # 历史保留两次评估
    history = client.get("/api/evaluations").json()
    assert len(history) == 2
    assert {h["id"] for h in history} == {first["id"], second["id"]}
    # 可按 ID 回看历史评估
    detail = client.get(f"/api/evaluations/{first['id']}").json()
    assert detail["overall"]["included_sessions"] == 2


def test_clear_logs(client):
    assert import_csv(client, SAMPLE_CSV).status_code == 200
    r = client.delete("/api/logs")
    assert r.status_code == 200
    assert r.json()["deleted_events"] == 7
    assert client.get("/api/status").json()["events"] == 0
    # 清空后无数据，评估返回 400
    assert client.post("/api/evaluate", json={"cap": 10.0}).status_code == 400


def test_index_page_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "推荐策略反事实评估工具" in r.text


EXTREME_CSV = """session_id,event_no,item_id,position,clicked,propensity,new_rank
s1,e1,A,1,1,0.5,1
s1,e2,B,2,0,0.5,2
"""


def test_tiny_cap_session_still_meets_ess_threshold(client):
    # 截断上限 1e-200：权重极小，但 ESS 按定义仍为 2.0，不应被判为样本不足
    assert import_csv(client, EXTREME_CSV).status_code == 200
    r = client.post("/api/evaluate", json={"cap": 1e-200, "min_ess": 2.0})
    assert r.status_code == 200
    res = r.json()
    s = res["per_session"][0]
    assert s["status"] == "included"
    assert s["ess"] == pytest.approx(2.0, rel=1e-12)
    assert s["sum_w"] == pytest.approx(2e-200, rel=1e-9)
    assert s["ips_ctr"] == pytest.approx(5e-201, rel=1e-9)
    assert res["overall"]["included_sessions"] == 1
    # 极小截断下 IPS 与权重同量级，低于旧策略 0.5 → 按原定义判未通过（不是“样本不足”）
    assert res["verdict"] == "fail"
    assert res["insufficient_sessions"] == []


def test_huge_cap_evaluation_succeeds_and_history_opens(client):
    # 倾向 1e-200 → 未截断时权重 1e200，ESS 的中间平方会溢出
    csv_text = (
        "session_id,event_no,item_id,position,clicked,propensity,new_rank\n"
        "s1,e1,A,1,1,1e-200,1\n"
        "s1,e2,B,2,0,1e-200,2\n"
    )
    assert import_csv(client, csv_text).status_code == 200
    r = client.post("/api/evaluate", json={"cap": 1e200, "min_ess": 2.0})
    assert r.status_code == 200
    res = r.json()
    assert res["per_session"][0]["status"] == "included"
    assert res["per_session"][0]["ess"] == pytest.approx(2.0, rel=1e-12)
    assert res["overall"]["included_sessions"] == 1

    # 新记录可以正常打开，且不含非有限数
    detail = client.get(f"/api/evaluations/{res['id']}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["needs_reevaluation"] is False
    assert body["overall"]["ips_ctr"] == pytest.approx(5e199, rel=1e-9)
    history = client.get("/api/evaluations").json()
    assert history[0]["needs_reevaluation"] is False
    assert history[0]["included_sessions"] == 1


def test_out_of_range_evaluation_reports_reason_and_keeps_history(client):
    assert import_csv(client, SAMPLE_CSV).status_code == 200
    ok = client.post("/api/evaluate", json={"cap": 10.0, "min_ess": 1.0})
    assert ok.status_code == 200
    before = client.get("/api/status").json()["evaluations"]

    # 权重和 2e308 超出双精度范围 → 评估失败，页面可显示原因
    csv_text = (
        "session_id,event_no,item_id,position,clicked,propensity,new_rank\n"
        "x1,y1,A,1,1,1e-308,1\n"
        "x1,y2,B,2,0,1e-308,2\n"
    )
    assert import_csv(client, csv_text, mode="replace").status_code == 200
    r = client.post("/api/evaluate", json={"cap": 1e308, "min_ess": 1.0})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["reason"] == "metric_out_of_range"
    assert "超出有限浮点数" in detail["message"]
    assert detail["session_id"] == "x1"

    # 失败的评估不写历史，历史条数保持不变
    assert client.get("/api/status").json()["evaluations"] == before
    assert len(client.get("/api/evaluations").json()) == before
    # 原有历史仍可打开
    assert client.get(f"/api/evaluations/{ok.json()['id']}").status_code == 200


def test_non_finite_evaluation_params_rejected(client):
    assert import_csv(client, SAMPLE_CSV).status_code == 200
    raw = b'{"cap": 1e999, "min_ess": 1.0}'
    r = client.post("/api/evaluate", content=raw, headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_legacy_non_finite_history_flagged_for_reevaluation(client):
    assert import_csv(client, SAMPLE_CSV).status_code == 200
    good = client.post("/api/evaluate", json={"cap": 10.0, "min_ess": 1.0}).json()

    # 模拟旧版本写入的脏记录（历史版本允许 NaN/Infinity 落库）
    legacy = {
        "cap": 1e300,
        "min_ess": 2.0,
        "overall": {"included_sessions": 1, "excluded_sessions": 0, "total_events": 2,
                    "ips_ctr": float("nan"), "snips_ctr": float("inf"), "baseline_ctr": 0.5},
        "verdict": "fail",
        "per_session": [],
        "uncovered_candidates": [],
        "insufficient_sessions": [],
    }
    conn = db.get_conn()
    try:
        conn.execute(
            "INSERT INTO evaluations (dataset_id, cap, min_ess, created_at, result_json) VALUES (?,?,?,?,?)",
            (db.DEFAULT_DATASET_ID, 1e300, 2.0, "2026-01-01T00:00:00+00:00",
             json.dumps(legacy)),
        )
        conn.commit()
        legacy_id = conn.execute("SELECT MAX(id) id FROM evaluations").fetchone()["id"]
    finally:
        conn.close()

    history = {h["id"]: h for h in client.get("/api/evaluations").json()}
    assert history[legacy_id]["needs_reevaluation"] is True
    assert history[legacy_id]["verdict"] == "invalid"
    assert history[legacy_id]["ips_ctr"] is None
    # 正常记录不受影响
    assert history[good["id"]]["needs_reevaluation"] is False
    assert history[good["id"]]["verdict"] == "pass"

    detail = client.get(f"/api/evaluations/{legacy_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["needs_reevaluation"] is True
    assert "非有限数" in body["reason"]
    # 响应必须是严格 JSON（浏览器 JSON.parse 不接受裸 NaN/Infinity 字面量）
    assert '"NaN"' not in detail.text and '": NaN' not in detail.text
    assert '": Infinity' not in detail.text and '": -Infinity' not in detail.text
    assert client.get(f"/api/evaluations/{good['id']}").json()["needs_reevaluation"] is False


def test_import_rejects_non_finite_propensity(client):
    bad = "session_id,event_no,item_id,position,clicked,propensity,new_rank\ns1,e1,A,1,1,nan,1\n"
    r = import_csv(client, bad)
    assert r.status_code == 400
    assert "有限数字" in r.json()["detail"]
