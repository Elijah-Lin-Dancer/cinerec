# CineRec 验收标准

> **补录说明**：本文件补录于 2026-11-16（存量项目 0-R 补录），非目标冻结时的原始记录。
> 基线数字为 2026-11-16 在 Linux 沙箱（Python 3.10 / CPU / 无神经栈）实测，非抄自 README。

## 基线实测（2026-11-16）

| 项 | 结果 |
|---|---|
| pytest（非 e2e） | 110 passed, 8 skipped |
| 覆盖率 | 51.30%（门槛 45%） |
| lite `/api/health` | `{"status":"ok"}`（冷启动约 30 s） |
| full（Streamlit） | HTTP 303（正常重定向） |
| 两域产物 | 7 个 multiseed JSON + lite `recs_cache.json` 均在库 |

## M 条（交付判据）

| 编号 | 判据 | 验证方式 | 不通过的样子 |
|---|---|---|---|
| M1 | 两域评估产物齐全：MovieLens 与 Amazon 各含 `eval_multiseed.json`、`ablation_multiseed.json`、`coldstart_multiseed.json`，且 `n_seeds=5` | 解析两域 JSON 断言 `n_seeds==5` 且键齐全 | 任一文件缺失或 `n_seeds≠5` |
| M2 | 8 模型（6 梯子 + BPR-MF/VBPR）在两域同协议下有完整 HR/NDCG/Recall@5/10/20（每模型 9 指标） | 解析两域 `eval_multiseed.json` 逐模型断言 9 个指标键 | 任一模型缺任一指标 |
| M3 | pytest 非 e2e 全绿且覆盖率 ≥45% | `python -m pytest -p no:warnings --cov` | 任一失败或覆盖率 <45% |
| M4 | CI 三 job（test/e2e/docker）全绿 | `gh run list --limit 3` | 任一 job 红 |
| M5 | lite `/api/health` 返回 `{"status":"ok"}`（冷启动 ≤60 s） | `curl -sSL --max-time 60 https://cinerec-rfew.onrender.com/api/health` | 超时或非 ok |
| M6 | 双模式 Docker 构建通过 | `docker build --build-arg APP_MODE=lite -t cinerec:lite .` 及 full 同理 | 任一构建失败 |
| M7 | Streamlit 全页中英切换生效：默认中文，切换后侧栏/标签页/图表标题/指标卡全为中文 | `tests/test_e2e_a11y.py` + 人工抽查截图 | 任一界面元素仍是英文 |

## S 条（次要判据）

| 编号 | 判据 | 验证方式 | 不通过的样子 |
|---|---|---|---|
| S1 | 预印本 DOI 一致：`10.20944/preprints202610.2504.v1` | `grep -n "preprints202610.2504" streamlit_app.py docs/paper.tex` | 两处不一致 |
| S2 | Locust 压测报告存在且 0 失败 | `reports/loadtest.md` 含 0 失败记录 | 缺失或失败 >0 |

## N 条（延后备忘，不作交付判据）

| 编号 | 内容 | 状态 |
|---|---|---|
| N1 | 预印本 v2 补正式显著性检验（统计检验 + epoch 预算对齐） | 未开工，属方向重估范围（2026-11-16） |
