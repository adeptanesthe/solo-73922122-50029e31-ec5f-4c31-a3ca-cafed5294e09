"""推荐策略反事实评估工具 — FastAPI 入口。"""

import json
import math
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from . import db
from .importer import CsvFormatError, parse_log_csv
from .metrics import Exposure, MetricRangeError, evaluate_dataset

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
SAMPLE_CSV = BASE_DIR.parent / "sample_data" / "sample_log.csv"


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="推荐策略反事实评估工具", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """请求校验失败时返回严格 JSON。

    FastAPI 默认会把非法输入（如超出双精度范围的 1e999 被解析成 inf）原样
    放回错误响应，导致 json.dumps 抛错、页面拿不到失败原因。
    """
    try:
        errors = _safe_json(exc.errors())
        json.dumps(errors, allow_nan=False)
    except (TypeError, ValueError):
        errors = [{"msg": "请求体无法解析"}]
    return JSONResponse(
        status_code=422,
        content={
            "detail": "请求参数无效：截断上限与 ESS 阈值必须是有限的数字"
            "（cap > 0，min_ess ≥ 0）",
            "errors": errors,
        },
    )


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/sample-csv")
def sample_csv():
    if not SAMPLE_CSV.exists():
        raise HTTPException(status_code=404, detail="示例文件不存在")
    return FileResponse(SAMPLE_CSV, media_type="text/csv", filename="sample_log.csv")


@app.get("/api/status")
def status():
    conn = db.get_conn()
    try:
        ds = db.DEFAULT_DATASET_ID
        events = conn.execute("SELECT COUNT(*) c FROM events WHERE dataset_id=?", (ds,)).fetchone()["c"]
        sessions = conn.execute(
            "SELECT COUNT(DISTINCT session_id) c FROM events WHERE dataset_id=?", (ds,)
        ).fetchone()["c"]
        candidates = conn.execute(
            "SELECT COUNT(*) c FROM candidates WHERE dataset_id=?", (ds,)
        ).fetchone()["c"]
        evaluations = conn.execute(
            "SELECT COUNT(*) c FROM evaluations WHERE dataset_id=?", (ds,)
        ).fetchone()["c"]
        return {"events": events, "sessions": sessions, "candidates": candidates, "evaluations": evaluations}
    finally:
        conn.close()


@app.post("/api/import")
async def import_logs(file: UploadFile = File(...), mode: str = Form("append")):
    if mode not in ("append", "replace"):
        raise HTTPException(status_code=400, detail="mode 只能是 append 或 replace")
    raw = (await file.read()).decode("utf-8-sig")
    try:
        exposures, candidates = parse_log_csv(raw)
    except CsvFormatError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="文件必须是 UTF-8 编码的 CSV")

    ds = db.DEFAULT_DATASET_ID
    conn = db.get_conn()
    try:
        if mode == "replace":
            conn.execute("DELETE FROM events WHERE dataset_id=?", (ds,))
            conn.execute("DELETE FROM candidates WHERE dataset_id=?", (ds,))

        inserted = 0
        for r in exposures:
            cur = conn.execute(
                """INSERT OR IGNORE INTO events
                   (dataset_id, session_id, event_no, item_id, position, clicked, propensity, new_rank)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (ds, r["session_id"], r["event_no"], r["item_id"], r["position"],
                 r["clicked"], r["propensity"], r["new_rank"]),
            )
            inserted += cur.rowcount

        # 候选集 = 仅候选行 ∪ 曝光行中带 new_rank 的物品
        cand_rows = list(candidates)
        cand_rows += [
            {"session_id": r["session_id"], "item_id": r["item_id"], "new_rank": r["new_rank"]}
            for r in exposures if r["new_rank"] is not None
        ]
        cand_inserted = 0
        for c in cand_rows:
            cur = conn.execute(
                "INSERT OR IGNORE INTO candidates (dataset_id, session_id, item_id, new_rank) VALUES (?,?,?,?)",
                (ds, c["session_id"], c["item_id"], c["new_rank"]),
            )
            cand_inserted += cur.rowcount
        conn.commit()
    finally:
        conn.close()

    return {
        "mode": mode,
        "exposure_rows": len(exposures),
        "inserted": inserted,
        "duplicates_skipped": len(exposures) - inserted,
        "candidate_only_rows": len(candidates),
        "candidate_items_inserted": cand_inserted,
    }


@app.delete("/api/logs")
def clear_logs():
    ds = db.DEFAULT_DATASET_ID
    conn = db.get_conn()
    try:
        e = conn.execute("DELETE FROM events WHERE dataset_id=?", (ds,)).rowcount
        c = conn.execute("DELETE FROM candidates WHERE dataset_id=?", (ds,)).rowcount
        conn.commit()
        return {"deleted_events": e, "deleted_candidates": c}
    finally:
        conn.close()


class EvalRequest(BaseModel):
    cap: float = Field(
        10.0, gt=0, allow_inf_nan=False, description="截断上限 M：权重 w=min(1/p, M)"
    )
    min_ess: float = Field(
        2.0, ge=0, allow_inf_nan=False, description="有效样本量阈值，低于则判为样本不足"
    )

def _load_sessions(conn, ds: int) -> dict:
    sessions: dict = {}
    for e in conn.execute(
        "SELECT * FROM events WHERE dataset_id=? ORDER BY session_id, position", (ds,)
    ).fetchall():
        s = sessions.setdefault(e["session_id"], {"exposures": [], "candidates": {}})
        s["exposures"].append(
            Exposure(item_id=e["item_id"], clicked=e["clicked"], propensity=e["propensity"])
        )
        if e["new_rank"] is not None:
            s["candidates"][e["item_id"]] = e["new_rank"]
    for c in conn.execute("SELECT * FROM candidates WHERE dataset_id=?", (ds,)).fetchall():
        s = sessions.setdefault(c["session_id"], {"exposures": [], "candidates": {}})
        s["candidates"][c["item_id"]] = c["new_rank"]
    return sessions


@app.post("/api/evaluate")
def evaluate(req: EvalRequest):
    ds = db.DEFAULT_DATASET_ID
    conn = db.get_conn()
    try:
        sessions = _load_sessions(conn, ds)
        if not sessions:
            raise HTTPException(status_code=400, detail="当前没有日志数据，请先导入")
        try:
            result = evaluate_dataset(sessions, req.cap, req.min_ess)
        except MetricRangeError as exc:
            # 不写入历史：失败的评估不产生记录，历史条数保持不变
            raise HTTPException(
                status_code=422,
                detail={
                    "message": f"{exc}。请调整截断上限或日志中的倾向分数后重试。",
                    "reason": "metric_out_of_range",
                    "session_id": exc.session_id,
                    "field": exc.field,
                },
            )
        try:
            payload = json.dumps(result, ensure_ascii=False, allow_nan=False)
        except ValueError:
            # 理论上 evaluate_dataset 已保证有限值；作为最后一道防线，拒绝落库
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "结果超出有限浮点数范围，未写入历史记录。",
                    "reason": "metric_out_of_range",
                },
            )
        created_at = db.utcnow()
        cur = conn.execute(
            "INSERT INTO evaluations (dataset_id, cap, min_ess, created_at, result_json) VALUES (?,?,?,?,?)",
            (ds, req.cap, req.min_ess, created_at, payload),
        )
        conn.commit()
        eval_id = cur.lastrowid
    finally:
        conn.close()
    return {"id": eval_id, "created_at": created_at, **result}


def _safe_json(value):
    """把 NaN/Infinity 替换为 None，保证浏览器 JSON.parse 能解析。"""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    return value


def _history_summary(eval_id, cap, min_ess, created_at, result_json) -> dict:
    """历史列表行。旧记录若含 NaN/Infinity 则标为需重评，其余字段照常返回。"""
    item = {
        "id": eval_id,
        "cap": cap if isinstance(cap, (int, float)) and math.isfinite(cap) else None,
        "min_ess": min_ess if isinstance(min_ess, (int, float)) and math.isfinite(min_ess) else None,
        "created_at": created_at,
    }
    invalid = {
        **item,
        "verdict": "invalid",
        "included_sessions": None,
        "ips_ctr": None,
        "snips_ctr": None,
        "needs_reevaluation": True,
    }
    if not math.isfinite(cap) or not math.isfinite(min_ess):
        return invalid
    try:
        res = json.loads(result_json)
    except (TypeError, ValueError):
        return invalid
    if db.has_non_finite(res) or "verdict" not in res or "overall" not in res:
        return invalid
    return {
        **item,
        "verdict": res["verdict"],
        "included_sessions": res["overall"]["included_sessions"],
        "ips_ctr": res["overall"]["ips_ctr"],
        "snips_ctr": res["overall"]["snips_ctr"],
        "needs_reevaluation": False,
    }


@app.get("/api/evaluations")
def list_evaluations():
    ds = db.DEFAULT_DATASET_ID
    conn = db.get_conn()
    try:
        rows = conn.execute(
            "SELECT id, cap, min_ess, created_at, result_json FROM evaluations WHERE dataset_id=? ORDER BY id DESC",
            (ds,),
        ).fetchall()
        return [
            _history_summary(r["id"], r["cap"], r["min_ess"], r["created_at"], r["result_json"])
            for r in rows
        ]
    finally:
        conn.close()


@app.get("/api/evaluations/{eval_id}")
def get_evaluation(eval_id: int):
    conn = db.get_conn()
    try:
        r = conn.execute(
            "SELECT id, created_at, result_json FROM evaluations WHERE id=?", (eval_id,)
        ).fetchone()
        if r is None:
            raise HTTPException(status_code=404, detail="评估记录不存在")
        result = json.loads(r["result_json"])
        resp = {"id": r["id"], "created_at": r["created_at"], **_safe_json(result)}
        if db.has_non_finite(result):
            # 旧记录中的 NaN/Infinity 无法在浏览器 JSON.parse 中安全展示
            resp["needs_reevaluation"] = True
            resp["reason"] = "历史记录含有 NaN/Infinity 等非有限数，指标不可展示，请点击“运行评估”重新计算"
        else:
            resp["needs_reevaluation"] = False
            resp["reason"] = None
        return resp
    finally:
        conn.close()
