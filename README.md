# 推荐策略反事实评估工具

基于 **FastAPI + SQLite** 的 Web 工具：编辑在浏览器导入旧策略曝光日志（含新策略候选排序），
按会话计算**截断逆倾向加权点击率（Truncated IPS CTR）**与**自归一化点击率（SNIPS CTR）**，
用于上线前反事实评估新推荐策略。

## 快速启动

### Docker Compose（推荐）

```bash
docker compose up --build
```

打开 http://localhost:8000 。SQLite 数据持久化在命名卷 `recs_eval_data` 中。

### 本地运行

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 运行自测

```bash
# 本地
python -m pytest tests -v
# 或在容器内
docker compose run --rm web python -m pytest tests -v
```

## 日志 CSV 格式

列（表头必须齐全，顺序不限）：

| 列 | 含义 |
|---|---|
| `session_id` | 用户会话标识 |
| `event_no` | 事件号（**相同事件号只处理一次**，重复自动跳过） |
| `item_id` | 物品 ID |
| `position` | 曝光位置（≥1 的整数） |
| `clicked` | 点击标记（0/1） |
| `propensity` | 旧策略倾向分数，(0,1] |
| `new_rank` | 该物品在新策略候选排序中的位次（≥1）；留空表示未被新策略推荐 |

两种行：

- **曝光行**：`event_no` 非空，字段按上表校验。
- **仅候选行**：`event_no` 留空，只填 `session_id,item_id,new_rank`，表示新策略推荐了但日志中
  没有曝光的物品，用于识别"新策略未覆盖的候选"。

示例见 `sample_data/sample_log.csv`（页面内也可直接下载）。

## 指标定义（按会话）

设会话内与候选集匹配的曝光事件权重 `w_i = min(1/propensity_i, M)`，`M` 为用户输入的截断上限：

- **截断 IPS 点击率** = `Σ w_i·click_i / 会话曝光数`
- **自归一化 SNIPS 点击率** = `Σ w_i·click_i / Σ w_i`
- **有效样本量 ESS** = `(Σ w_i)² / Σ w_i²`

指标权重与求和用精确分数（`fractions.Fraction`）计算，最后才转成双精度浮点。
因此截断上限取极小值（如 `1e-200`，权重下溢到 1e-200 量级）或极大值
（如 `1e200`，权重上溢到 1e200 量级）时，ESS 等比值指标仍按原定义计算，
不会因为 `w²` 下溢/上溢而误判"样本不足"或得到 NaN。

- 结果（各指标及总体平均）在有限浮点数范围内：照常给出逐会话判定与总体结论；
- 结果本身超出有限浮点数范围：评估返回 422，页面在结论区说明失败字段与原因
  （如"总体权重和 Σw 超出有限浮点数（双精度）范围"），**不写入历史记录**，
  历史条数保持不变；
- 历史记录中旧版本遗留的 NaN/Infinity：在历史列表标为**需重评**，仍可点开查看说明；
  其余记录照常查看。

总体结论 = 纳入会话指标的算术平均。判定规则（供编辑参考）：纳入会话的平均截断 IPS 点击率
≥ 旧策略实测点击率 → **通过**，否则 **未通过**。

## 排除规则（不计入通过结论）

以下记录在页面**单列展示**，且**不计入**通过结论：

1. **新策略未覆盖的候选**：候选物品在日志中无对应曝光（拿不到倾向分数）→ 其所在会话整体排除；
2. **样本不足的会话**：无匹配候选曝光，或 ESS 低于阈值（默认 2.0，可在页面调整）→ 会话排除。

## 修改日志后重新评估

- 导入时选 **替换** 模式（或先点"清空日志"）上传新日志，再点 **运行评估**；
- 每次评估的参数与结果都存入 `evaluations` 表，页面底部"历史评估"可回看任意一次。

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 操作页面 |
| POST | `/api/import` | 导入 CSV（`file` + `mode=append|replace`） |
| DELETE | `/api/logs` | 清空曝光与候选 |
| POST | `/api/evaluate` | 运行评估，JSON `{cap, min_ess}` |
| GET | `/api/evaluations` | 历史评估列表 |
| GET | `/api/evaluations/{id}` | 单次评估详情 |
| GET | `/api/status` | 当前数据量 |
| GET | `/api/sample-csv` | 下载示例 CSV |

## 目录结构

```
app/
  main.py       # FastAPI 路由
  metrics.py    # 截断 IPS / SNIPS / ESS 纯函数
  importer.py   # CSV 解析校验
  db.py         # SQLite 连接与建表
  static/index.html  # 操作页面
tests/          # pytest 自测（指标 + API 端到端）
sample_data/    # 示例日志
Dockerfile / docker-compose.yml
```
