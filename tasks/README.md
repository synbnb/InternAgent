# sci_tasks: 科研论文复现任务

用 InternAgent 自动复现论文成果的任务集，评分对齐 [ResearchClawBench](https://github.com/ResearchClawBench)。

---

## 快速开始

```bash
# 默认任务 (Astronomy_000)
bash scripts/run_sci_debug.sh

# 指定任务
bash scripts/run_sci_debug.sh Life_000
bash scripts/run_sci_debug.sh Chemistry_001

# 用完整路径
bash scripts/run_sci_debug.sh tasks/sci_tasks/Energy_002
```

调试用 `config/debug_config.yaml`（2 rounds × 2 ideas × 2 runs），生产用 `config/default_config.yaml`。

---

## 环境依赖

在项目根目录下 `.env` 文件中配置（必需）：

```ini
OPENAI_API_KEY=sk-...
OPENAI_API_BASE_URL=http://your-api-host/v1   # 或 OPENAI_BASE_URL

# 可选：走代理
http_proxy=http://127.0.0.1:7890
https_proxy=http://127.0.0.1:7890
```

Python 依赖：`pip install structai python-dotenv`（已安装可跳过）。

---

## 任务目录结构

每个 sci 任务位于 `tasks/sci_tasks/{Domain}_{NNN}/`，由 ResearchClawBench 提供：

```
Life_000/
├── task_info.json          # 任务描述 + 数据清单
├── data/                   # 只读数据文件 (xlsx/csv/fasta/...)
├── related_work/           # 相关论文 PDF (只读)
└── target_study/           # 评分用 ground truth
    ├── paper.pdf           # 目标论文
    ├── checklist.json      # 评分清单 (8 项 × weight)
    └── images/             # 论文原图 (评分时作为对照)
```

`checklist.json` 每项包含：

| 字段 | 说明 |
|-----|------|
| `type` | `"text"` 或 `"image"` |
| `weight` | 权重 (所有 item 加权平均) |
| `content` | 该 item 期望内容的描述 |
| `keywords` | 关键技术点 (判官会核对) |
| `path` | 仅 image 类型：原图在 `target_study/` 下的相对路径 |

---

## 自定义运行

```bash
python launch_discovery.py \
    --task tasks/sci_tasks/Life_000 \
    --config config/default_config.yaml \
    --exp_backend claudecode
```

关键 CLI 参数：

| 参数 | 说明 |
|------|------|
| `--task` | 任务名 (`Life_000`) 或完整路径 |
| `--config` | YAML 配置文件 |
| `--exp_backend` | `claudecode` / `openhands` / `iflow` / `aider` |
| `--output_dir` | 结果目录 (默认 `results/{task}/`) |
| `--resume` | 从已有 launch 文件夹断点续跑 |
| `--skip_idea_generation` | 跳过 idea 生成，用 `--idea_path` 指定现有 ideas |

配置文件中 sci_task 相关项：

```yaml
sci_task:
  scorer_model: "gpt-5.1"              # LLM 评分模型 (对齐官方默认)
  evaluation_mode: "llm_judge"
  default_launcher: "python code/experiment.py"

experiment:
  model: "claude-sonnet-4-5-20250929"  # coder agent 模型
  max_runs: 2                           # 每个 idea 的实验轮数
  max_parallel_experiments: 4
```

---

## 结果目录结构

运行后在 `results/{task}/{timestamp}_launch/` 下生成：

```
results/Life_000/20260415_130021_launch/
├── session_1776258021/                      # Round 1
│   └── 20260415_130502_{IdeaName}/          # 每个 idea 一个 experiment folder
│       ├── INSTRUCTIONS.md                  # agent 收到的任务说明 (评分时作为 judge 上下文)
│       ├── launcher.sh
│       ├── data/ → symlink                  # 数据/相关文献/target_study 都是软链
│       ├── related_work/ → symlink
│       ├── target_study/ → symlink
│       ├── code/                            # agent 写的代码
│       ├── outputs/                         # 中间输出
│       ├── report/
│       │   ├── report.md                    # 最终报告 (必需，评分对象)
│       │   └── images/                      # 生成的图
│       ├── run_0/final_info.json            # baseline (score=0)
│       ├── run_1/
│       │   ├── code/                        # 本轮代码快照
│       │   ├── report/                      # 本轮报告
│       │   ├── INSTRUCTIONS.md              # 拷贝自 experiment folder
│       │   ├── target_study/
│       │   └── final_info.json              # 本轮评分
│       └── run_2/...
└── session_1776266473/                      # Round 2 (incremental mode 会基于 Round 1 最优继续)
```

---

## 评分方法

`final_info.json` 格式：

```json
{
  "sci_task": {
    "means": {
      "total_score": 40.6,
      "item_0_score": 48,
      "item_1_score": 48,
      ...
    }
  }
}
```

评分完全复用 [ResearchClawBench](https://github.com/ResearchClawBench) `evaluation/score.py`（项目里 symlink 成 `internagent/rcb_evaluation/`），判官会看：

1. **报告原文** (`report/report.md`)
2. **生成图** (`report/images/`, `outputs/*.png`)
3. **原论文图** (`target_study/images/`) —— 仅 image 类型 item
4. **INSTRUCTIONS.md** —— 告诉判官 agent 收到什么任务
5. **checklist 的 keywords** —— 核对关键技术点是否被覆盖

rubric 分两种模式：
- **Mode A (Objective)**: 指标/定量结果，**50 分 = 与论文持平**，0-100
- **Mode B (Subjective)**: 机理/定性分析，同上

所以 30-40 分属正常 AI 复现水平，50+ 已经很难得。

---

## 手动重新打分

如果评分代码变更后想给已跑完的任务重新打分：

```python
from dotenv import load_dotenv
load_dotenv()
from internagent.sci_eval import score_run, write_final_info

run_dir = "results/Life_000/.../run_1"
checklist = f"{run_dir}/target_study/checklist.json"
scores = score_run(run_dir, checklist, model="gpt-5.1")
write_final_info(run_dir, scores)
print(f"Total: {scores['total_score']}/100")
```

---

## 常见问题

**Q: 评分时 API 超时？**
A: 确认 `.env` 里 `OPENAI_API_BASE_URL` 可达；如果用外网模型，确认 `http_proxy` / `https_proxy` 已设置。

**Q: 判官打分很低是不是有 bug？**
A: 不是。官方 rubric 里 50 分对标论文水平，30-40 分是正常 AI 复现分数。对比旧版本请看 `run_N/final_info_old.json`（rescore 时保留的备份）。

**Q: 新加一个 sci 任务？**
A: 在 `tasks/sci_tasks/{Domain}_{NNN}/` 下准备好 `task_info.json`、`data/`、`related_work/`、`target_study/` 即可，格式参考现有任务。
