# sci_tasks 流水线人机交互改进方案

> **面向用户**: 生物医药科研人员  
> **目标**: 在保持现有系统架构的前提下，在关键环节引入人工干预和交互能力，让科研人员能引导、审查、调整实验流程，避免全自动迭代走向错误方向。

---

## 一、现有流程全景

```
论文PDF → Paper-to-Task (自动) → task_info.json + checklist.json
                                    ↓
                          normalize_sci_task() → prompt.json
                                    ↓
                    ┌───── discovery loop (loop_rounds) ─────┐
                    │  IdeaGenerator (MAS 完全自动)           │
                    │    ↓                                    │
                    │  ExperimentRunner (Claude 自动编/执)     │
                    │    ↓                                    │
                    │  评分 (LLM Judge 自动)                   │
                    └────────────────────────────────────────┘
                                    ↓
                          discovery_summary.json
```

**关键问题**：整个流程启动后在 **3 个层面**完全无人工干预：

| 层面 | 问题 | 后果 |
|------|------|------|
| **想法生成** | 生成→推演→排序全自动，用户无法参与选题 | 可能生成科研上无意义的方向 |
| **代码实现** | Claude 自动写代码、自迭代、自己改 bug | 代码可能有逻辑错误但表面能跑通 |
| **结果评价** | LLM Judge 自动打分，用户只看最终分数 | 忽视错误结论，错过正确方向 |

---

## 二、七个人机交互介入点

### 介入点 1：Task Review — 审查和修改任务定义

**时机**：Paper-to-Task 生成 `task_info.json` 和 `checklist.json` 后，流水线启动前

**现状**：生成后直接可供流水线读取，但修改需要通过前端编辑器手动编辑 JSON

**改进方案**：

```
现有前端编辑器（已实现）→ 增强为结构化审查页面
```

**前端交互设计**：

```
┌─────────────────────────────────────────────────────────────┐
│  📋 任务审查与调整                              [确认 开始流水线] │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  📝 研究任务描述                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ 复现论文《ProtTrans...》的核心发现...                │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  📊 数据集列表                    [+ 添加数据集]              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ ● UniRef — 蛋白质序列数据库          [编辑] [删除]   │    │
│  │ ● BFD — 宏基因组蛋白质数据库         [编辑] [删除]   │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  ✅ 评分项清单（权重可调）                                     │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ ◉ 模型架构与训练规模     权重: ████░░░░ 0.25  [编辑] │    │
│  │ ◉ 数据来源与预处理       权重: ████░░░░ 0.20  [编辑] │    │
│  │ ◉ 自监督学习目标         权重: ████░░░░ 0.20  [编辑] │    │
│  │ ◉ 下游任务性能           权重: ████░░░░ 0.20  [编辑] │    │
│  │ ◉ 计算资源与可复现性     权重: ███░░░░░ 0.15  [编辑] │    │
│  │                                     [+ 添 加]         │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  📄 论文原文审查               [查看原文] [AI 润色版]       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**功能要点**：
- 结构化表单替代纯 JSON 编辑器（数据集的增删改、权重滑动条）
- checklist 权重实时预览（总和必须 = 1.0）
- 提供"太严格/太宽松"一键调整建议
- 确认后锁定，生成最终版 `task_info.json` + `checklist.json`

---

### 介入点 2：Idea Review — 审查和选择研究方向

**时机**：MAS 想法生成完成后，进入实验执行之前

**现状**：`IdeaGenerator` 输出 `top_ideas` 后直接传入 `ExperimentRunner`，用户不可见

**改进方案**：在 `launch_discovery.py` 的想法生成和实验执行之间插入**人工审批环节**

**设计**：

```
IdeaGenerator.generate_ideas()
        ↓
  top_ideas + session_json  ← 原本直接输出
        ↓
  **NEW: 写入 ideas.json 后暂停，等待前端确认**
        ↓
  用户在前端浏览 → 选择/修改/否决
        ↓
  **确认后才继续执行实验**
```

**数据流**：
```
launch_discovery.py:
  top_ideas, session_json = idea_generator.generate_ideas()
  
  # ★ 新增：保存待审批状态
  save_pending_ideas(args.output_dir, session_id, top_ideas)
  
  # ★ 新增：等待前端确认（轮询或 WebSocket）
  approved_ideas = wait_for_human_approval(
      ideas_dir=args.output_dir,
      session_id=session_id,
      timeout=3600,  # 最长等待 1 小时
      polling_interval=10  # 每 10 秒检查一次
  )
  
  # 如果超时或用户拒绝，退出或使用默认
  if not approved_ideas:
      logger.warning("Human approval timeout or rejected, exiting")
      sys.exit(0)
  
  # 继续使用经过审批的想法
  top_ideas = approved_ideas
```

**前端交互**：

```
┌─────────────────────────────────────────────────────────────┐
│  💡 研究方向评估                              [确认选择 开始实验] │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  共生成 8 个候选方向，请选择 1-3 个进行实验                    │
│                                                              │
│  ┌──── 候选 1 ────────────────────────────────────────────┐  │
│  │ ⭐ 新颖度 8/10 • 可行性 7/10 • 相关性 9/10             │  │
│  │                                                        │  │
│  │ **标题**: 使用 ProtT5 嵌入进行蛋白质二级结构预测         │  │
│  │                                                        │  │
│  │ **方法**: 利用预训练的 ProtT5 模型的嵌入表示，           │  │
│  │ 结合线性分类器，在无需 MSA 的情况下进行二级结构预测...   │  │
│  │                                                        │  │
│  │ [编辑方法描述] [查看相关论文]  □ 选择此方向              │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──── 候选 2 ────────────────────────────────────────────┐  │
│  │ ⭐ 新颖度 6/10 • 可行性 9/10 • 相关性 7/10             │  │
│  │                                                        │  │
│  │ **标题**: 基于 XLNet 的蛋白质家族分类                    │  │
│  │ ...                                                    │  │
│  │ [编辑方法描述] [查看相关论文]  □ 选择此方向              │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  [刷新候选] [让 AI 再生成一批]     已选: 0/3                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**后端 API**（新增 Web 端点）：
- `GET /pipeline/pending_ideas?launch_dir=...` — 获取待审批的想法列表
- `POST /pipeline/approve_ideas` — 提交批准的想法选择和修改
- `GET /pipeline/status?launch_dir=...` — 查询状态（waiting_approval/running/completed）

---

### 介入点 3：Code Review — 审查生成的实验代码

**时机**：Claude Code 完成代码编写后，执行实验之前

**现状**：代码生成 → 立即执行 `bash launcher.sh`，用户从未看到代码

**改进方案**：在 `experiments_utils_claude.py` 的每次代码生成后插入**代码审查暂停点**

**设计**：

```
# experiments_utils_claude.py 中的内循环
for run_num in range(1, max_runs + 1):
    # Claude 生成/修改代码
    claude_code_runner.run(prompt, cwd=folder_name)
    
    # ★ 新增：代码审查暂停点
    if config.get('human_review', {}).get('code_review', True):
        notify_frontend_code_ready(folder_name, run_num)
        approval = wait_for_code_approval(folder_name, timeout=1800)
        if approval == 'reject':
            # 用户提供了修改意见
            revise_code_with_feedback(folder_name, approval.feedback)
            continue  # 重新审查
        elif approval == 'skip':
            pass  # 跳过此实验
    
    # 执行实验
    run_experiment(folder_name)
```

**前端交互**（代码 diff 视图）：

```
┌─────────────────────────────────────────────────────────────┐
│  📝 代码审查 — run_1                      [批准] [修改建议] [跳过] │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌── experiment.py ───────────────────────────────────────┐  │
│  │  from transformers import T5Model                       │  │
│  │  import torch                                          │  │
│  │                                                         │  │
│  │  # 加载 ProtT5 模型                                     │  │
│  │  model = T5Model.from_pretrained("Rostlab/prot_t5_xl")  │  │
│  │                                                         │  │
│  │  # 提取嵌入                                             │  │
│  │  def extract_embeddings(sequences):                     │  │
│  │      ...                                                │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                              │
│  📋 代码修改意见                                             │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ [输入修改建议...]                                     │    │
│  │                                                      │    │
│  │ 常见操作：                                            │    │
│  │ [🔧 修复路径] [🔧 添加数据校验] [🔧 补充注释]         │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

### 介入点 4：Result Review — 审查实验结果，人工评分

**时机**：每个实验运行完成后，进入下一轮迭代之前

**现状**：LLM Judge 自动评分后，直接决定下一步方向

**改进方案**：展示实验结果，允许用户**补充评分**或**否决**自动评分

**设计**：

```
实验完成 → LLM Judge 自动评分
                ↓
        ★ 显示评分结果 + 实验报告
                ↓
        用户确认/修正评分
                ↓
        用户的评分影响下一步决策
```

**前端交互**：

```
┌─────────────────────────────────────────────────────────────┐
│  📊 实验结果 — 候选1: ProtT5 二级结构预测                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌── 自动评分 ──────────────────────────────────────────┐   │
│  │                                                     │   │
│  │  总分: 72.5/100                    [编辑评分]        │   │
│  │                                                     │   │
│  │  ├ 模型架构与训练规模     80/100  ████████░░ 权重0.25│   │
│  │  ├ 数据来源与预处理       65/100  ██████░░░░ 权重0.20│   │
│  │  ├ 自监督学习目标         70/100  ███████░░░ 权重0.20│   │
│  │  ├ 下游任务性能           60/100  ██████░░░░ 权重0.20│   │
│  │  └ 计算资源与可复现性     90/100  █████████░ 权重0.15│   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  📄 实验报告 (report.md)          [查看完整报告]              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ # ProtTrans 复现报告                                  │    │
│  │                                                     │    │
│  │ ## 二级结构预测结果                                  │    │
│  │ 使用 ProtT5 嵌入 + SVM 分类器达到 Q3=78.5%...       │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  🖼 生成图表                        [查看全部图片]           │
│  ┌────┐ ┌────┐ ┌────┐                                      │
│  │ 📊 │ │ 📈 │ │ 📉 │                                      │
│  └────┘ └────┘ └────┘                                      │
│                                                              │
│  📝 科研人员反馈                                             │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ ● 二级结构预测的 Q3 分数偏低，论文报告的是 82%        │    │
│  │ ● 建议检查嵌入提取是否正确，可能未使用最优层          │    │
│  │ ○ 其他问题...                                         │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  [重新运行] [接受并继续] [修改代码后继续]                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**关键价值**：科研人员可以看到**实际数值**（论文声称的 82% vs 复现的 78.5%），判断是代码问题还是模型问题，而不是只看一个自动评分。

---

### 介入点 5：Mid-Flight Correction — 运行中动态调整

**时机**：流水线运行过程中，用户可随时查看状态并调整方向

**现状**：启动后零干预，要么等完成要么手动 kill

**改进方案**：实现**运行中干预**能力

**设计**：

```
启动流水线 → 实时日志流 → 随时可暂停/修改参数/继续
```

**前端"控制面板"**：

```
┌─────────────────────────────────────────────────────────────┐
│  🎮 流水线控制面板                    [⏸暂停] [🛑停止] [✏调整] │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  当前阶段: 实验执行 — run_1/5  (候选 2/3)                    │
│  后端: claudecode  模型: deepseek-chat                       │
│                                                              │
│  ┌── 实时日志 ──────────────────────────────────────────┐   │
│  │ [14:23:01] ✅ 代码生成完成                            │   │
│  │ [14:23:02] ▶ 正在执行 bash launcher.sh...            │   │
│  │ [14:23:45] ⏳ 运行中 - ETA 5分钟                      │   │
│  │ [14:28:12] ! 训练损失: 0.342, 验证精度: 0.781        │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  可调整参数：                                                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ max_runs: [5]  |  模型: [deepseek-chat ▼]           │    │
│  │ temperature: [0.7]  |  继续执行当前方向               │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

### 介入点 6：Prompt Customization — 动态编辑关键 Prompt

**时机**：启动流水线前，或在运行中暂停时

**现状**：`CODER_PROMPT_SCI_TASK` 和 `NEXT_EXPERIMENT_PROMPT_SCI` 硬编码在 `prompts.py` 中

**改进方案**：将关键 prompt 模板化，暴露给前端编辑

**实现方式**：

```yaml
# config/default_config.yaml 新增
prompts:
  coder_initial: |
    Your goal is to reproduce the findings from a scientific paper by implementing the following approach.
    
    ## Reproduction Approach
    {idea_description}
    ...
    ## Custom Instructions
    {custom_instructions}
    
  next_experiment: |
    Run {RUN_NUM} completed. Here are the results:
    {RESULTS}
    ...
    ## Custom Instructions
    {custom_instructions}
```

```yaml
# 附加：用户可以在前端编辑的自定义指令
custom_instructions: |
  特别注意：
  1. 所有代码必须包含完整的错误处理
  2. 实验结果需要与论文中的 Table 2 进行直接对比
  3. 图表需要包含置信区间
```

**前端交互**：

```
┌─────────────────────────────────────────────────────────────┐
│  ✏ Prompt 定制                            [恢复默认] [保存] │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  📋 初始代码生成指令                                          │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ 你的任务是复现论文中的核心发现...                     │    │
│  │ ...                                                  │    │
│  │                                                      │    │
│  │  额外要求（将追加到实验指令末尾）：                    │    │
│  │ ┌──────────────────────────────────────────────────┐ │    │
│  │ │ 1. 重点关注论文中 Table 2 的实验结果              │ │    │
│  │ │ 2. 使用相同的评估指标（MSE, MAE, R²）             │ │    │
│  │ │ 3. 代码必须包含详细的中文注释                      │ │    │
│  │ └──────────────────────────────────────────────────┘ │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  📋 实验迭代指令                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ ...                                                  │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

### 介入点 7：Comparison Dashboard — 多候选横向对比

**时机**：多轮/多候选实验完成后，选择最优结果

**现状**：系统自动选 `overall_improvement_rate` 最高的

**改进方案**：提供可视化对比面板，让用户做最终决策

**前端交互**：

```
┌─────────────────────────────────────────────────────────────┐
│  📊 实验结果对比                            [选择最佳 进行下一轮] │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  候选方向                  │ 总分 │ 架构 │ 数据 │ 目标 │ 性能 │
│  ──────────────────────────┼──────┼──────┼──────┼──────┼──────┤
│  ▸ ProtT5 + SVM 二级结构   │ 72.5 │ 80   │ 65   │ 70   │ 60   │
│  ▸ XLNet 蛋白质家族分类    │ 68.0 │ 75   │ 70   │ 65   │ 55   │
│  ▸ BERT 嵌入 + 定位预测    │ 81.0 │ 85   │ 75   │ 80   │ 78   │ ← 最佳
│                                                              │
│  ┌── BERT 嵌入 + 定位预测 — 各维度趋势 ─────────────────┐   │
│  │                                                      │   │
│  │  分数                                                    │   │
│  │  100│  ┌──┐                                             │   │
│  │   80│  │  │  ┌──┐  ┌──┐                                │   │
│  │   60│  │  │  │  │  │  │  ┌──┐                          │   │
│  │   40│  │  │  │  │  │  │  │  │                          │   │
│  │   20│  │  │  │  │  │  │  │  │                          │   │
│  │    0└──┴──┴──┴──┴──┴──┴──┴──                         │   │
│  │      run_1 run_2 run_3 run_4 run_5                     │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  [查看详情] [导出报告] [用此结果进入下一轮]                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、实现路线图

### 阶段 1：基础设施（不修改原项目核心代码）

| 模块 | 改动 | 文件 |
|------|------|------|
| **状态文件约定** | 定义标准化的状态文件格式 `pipeline_state.json`，包含 `status: waiting_approval \| approved \| rejected`、`pending_ideas`、`feedback` 等字段 | 新文件 |
| **Web 端点** | 新增 `/pipeline/*` API 端点用于查询和更新状态 | `web_app.py` |
| **前端标签页** | 新增"流水线智能控制"标签页，包含状态面板、想法审批、代码审查、结果浏览 | `index.html` |

### 阶段 2：审批介入（最关键）

| 介入点 | 改动量 | 复杂度 |
|--------|--------|--------|
| ① Task Review | 前端增强，后端状态管理 | 低 |
| ② Idea Review | 修改 `launch_discovery.py` 主循环，插入等待点 | **中** |
| ③ Code Review | 修改 `experiments_utils_claude.py` 内循环 | **高** |
| ④ Result Review | 新增前端结果展示，修改评分流程 | 中 |

### 阶段 3：增强功能

| 功能 | 说明 |
|------|------|
| ⑤ Mid-Flight Correction | 运行时参数调整（需要重构进程通信） |
| ⑥ Prompt Customization | 将 prompts.py 的模板迁移到配置文件 |
| ⑦ Comparison Dashboard | 前端可视化面板 |

---

## 四、推荐优先级

```
优先级 P0（核心价值高，改动量小）：
  └─ ② Idea Review — 想法审批
  └─ ④ Result Review — 结果审查与人工评分
  
优先级 P1（价值高，改动量适中）：
  └─ ① Task Review — 任务定义审查增强
  └─ ⑦ Comparison Dashboard — 对比看板

优先级 P2（价值高，改动量大）：
  └─ ③ Code Review — 代码审查
  └─ ⑥ Prompt Customization — Prompt 编辑

优先级 P3（锦上添花）：
  └─ ⑤ Mid-Flight Correction — 运行中调整
```

---

## 五、关键设计原则

### 1. 非侵入式集成

不改动原项目的核心 MAS 代码（`internagent/` 目录），只在以下位置做少量扩展：

```
修改:
  launch_discovery.py    — 在 main() 中插入审批等待点
  experiments_utils_claude.py — 在代码生成后插入审查暂停点（可选）
  
新增:
  paper_to_task/interaction/pipeline_hooks.py — 所有等待/通知逻辑
  paper_to_task/interaction/static/pipeline_control.js — 前端控制逻辑
```

### 2. 状态驱动

使用文件系统作为状态通信机制，避免进程间复杂通信：

```
等待审批状态:
  results/{task_name}/{launch_id}/.human_review/
    ideas_pending.json       — 待审批的想法
    ideas_approved.json      — 已审批的想法  
    code_review_{run}.json   — 待审查的代码
    code_feedback_{run}.json — 审查反馈
    result_review.json       — 待确认的结果
    result_feedback.json     — 用户评分修正
    pipeline_state.json      — 当前状态标记
```

### 3. 超时处理

所有人工等待点必须设置超时：

| 等待点 | 默认超时 | 超时行为 |
|--------|----------|----------|
| Task Review | 30 分钟 | 使用当前配置继续 |
| Idea Review | 60 分钟 | 使用系统排名前 N 自动继续 |
| Code Review | 30 分钟 | 自动批准（跳过审查）|
| Result Review | 30 分钟 | 以 LLM Judge 评分为准 |

---

## 六、快速启动 P0 实现示例

### 后端 — `launch_discovery.py` 的关键修改

```python
# 在 main() 中，IdeaGenerator 完成之后：
top_ideas, session_json = asyncio.run(idea_generator.generate_ideas())

# ===== 新增：人工审批环节 =====
if config.get('human_review', {}).get('idea_review', True):
    logger.info("⏸ Waiting for human idea approval...")
    
    # 保存待审批状态
    pending_file = osp.join(args.output_dir, ".human_review", "ideas_pending.json")
    os.makedirs(osp.dirname(pending_file), exist_ok=True)
    with open(pending_file, 'w') as f:
        json.dump({
            'status': 'waiting',
            'session_id': session_id,
            'ideas': [
                {
                    'id': idea.get('id', f'idea_{i}'),
                    'title': idea.get('refined_method_details', {}).get('title', ''),
                    'description': idea.get('refined_method_details', {}).get('description', ''),
                    'method': idea.get('refined_method_details', {}).get('method', ''),
                    'scores': idea.get('scores', {}),
                }
                for i, idea in enumerate(top_ideas)
            ]
        }, f, indent=2)
    
    # 轮询等待审批（最长等待时间）
    from paper_to_task.interaction.pipeline_hooks import wait_for_human_approval
    approved = wait_for_human_approval(
        pending_file,
        timeout=config.get('human_review', {}).get('idea_timeout', 3600),
        poll_interval=10
    )
    
    if approved:
        # 使用用户批准的 ideas
        selected_ids = [idea['id'] for idea in approved]
        top_ideas = [idea for idea in top_ideas if idea.get('id') in selected_ids]
        logger.info(f"✓ Human approved {len(top_ideas)} ideas")
        
        # 如果有修改，更新想法
        for feedback in approved:
            if feedback.get('modified_method'):
                for idea in top_ideas:
                    if idea.get('id') == feedback['id']:
                        idea['refined_method_details']['method'] = feedback['modified_method']
    else:
        logger.warning("⚠ Human approval timeout or rejected, using system defaults")
# ===== 结束：人工审批环节 =====
```

### 前端 — 新增控制面板

在 `index.html` 的"流水线"标签页中增加二级子标签：

```
流水线主标签页
  ├── ▶ 启动配置（现有）
  ├── ⏳ 运行监控（现有增强）
  ├── 💡 想法审批（新增）
  ├── 📝 代码审查（新增）
  └── 📊 结果评估（新增）
```

### 新增文件 `paper_to_task/interaction/pipeline_hooks.py`

```python
"""
流水线人工介入钩子
提供文件系统状态轮询机制，实现非侵入式的人机交互
"""
import os
import json
import time
import logging

logger = logging.getLogger(__name__)


def write_state(state_file: str, state: dict):
    """写入流水线状态"""
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)


def read_state(state_file: str) -> dict:
    """读取流水线状态"""
    if not os.path.exists(state_file):
        return {'status': 'unknown'}
    with open(state_file, 'r') as f:
        return json.load(f)


def wait_for_state(
    state_file: str,
    target_status: str,
    timeout: int = 3600,
    poll_interval: int = 10
) -> dict:
    """
    轮询等待状态文件达到目标状态
    
    Args:
        state_file: 状态文件路径
        target_status: 期望状态值（如 'approved', 'rejected'）
        timeout: 超时秒数
        poll_interval: 轮询间隔秒数
    
    Returns:
        最终的状态字典，超时返回空 dict
    """
    start = time.time()
    while time.time() - start < timeout:
        state = read_state(state_file)
        if state.get('status') == target_status:
            return state
        time.sleep(poll_interval)
    return {}
```

---

## 七、总结

| 维度 | 现状 | 改进后 |
|------|------|--------|
| **用户参与度** | 零干预，全自动 | 7 个关键介入点可选参与 |
| **方向可控性** | 不可控，可能偏离 | 想法审批 + 结果审查可纠偏 |
| **透明度** | 黑盒，事后看报告 | 实时查看代码、结果、评分 |
| **灵活性** | 固定 prompt，固定流程 | 可编辑 prompt，动态调整参数 |
| **对原系统修改** | — | 极小（<=3 个文件） |
| **新增代码** | — | 前端 + 钩子模块，可独立部署 |

**核心哲学**：科研人员不是流水线的旁观者，而是**实验的指导者**。每个关键决策点都应该提供"系统建议 + 人工确认"的双重验证，让 AI 处理重复劳动，让人把握科研方向。
