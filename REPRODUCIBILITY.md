# Reproducibility

## 本机 Python

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements-lock.txt
.venv/Scripts/pip install -e .
bdmtf audit-data
bdmtf evaluate-fidelity
bdmtf build-final-paper
```

Linux 使用 `.venv/bin/pip`。Windows/Linux 脚本位于 `scripts/`。

## 论文工具链

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_paper_toolchain.ps1
bdmtf build-original-manuscript
bdmtf build-final-paper
bdmtf build-paper-redline
```

## Docker

```bash
docker compose run --rm bdmtf
docker compose run --rm causal
docker compose run --rm paper
```

当前仓库分别锁定 Python 3.12、R 因果环境和 Tectonic/latexdiff 论文环境。

## 外部公开数据

```bash
bdmtf fetch-tbbt
bdmtf fetch-tbbt --execute
bdmtf fetch-tbbt --import-data
bdmtf collect-lemmy
bdmtf collect-lemmy-outcomes
bdmtf collect-hackernews --stories 5000 --max-comments-per-story 300 --workers 48 --discovery algolia
bdmtf build-story-matches
bdmtf build-story-matches --resolve-urls
bdmtf run-lemmy-agent-replay
bdmtf run-lemmy-content-matched-validation
```

TBBT 支持断点续传、发布者 MD5 校验、按档案检查点和内存有界聚合。

## API 与前瞻实验

密钥只从环境变量读取；此前暴露的密钥必须更换。无 API 时可重放缓存。
`make-prospective-split --freeze-at ...` 必须在未来数据采集前执行。
人类实验必须先有伦理审批、预注册、知情同意和招募预算。
