# InternAgent 科学论文复现功能技术分析

## 功能概述

InternAgent 的科学论文复现（sci_tasks）功能能够自动读取已发表的科学论文和相关数据，自主编写分析代码、运行实验、迭代优化，并生成研究报告和评分。

## 核心工作流程

### 完整流程图

```
用户输入 sci_tasks 目录
    ↓
detect_task_type() 检测任务类型
    ↓
normalize_sci_task() 转换任务格式
    ├─ 读取 task_info.json
    ├─ 读取 checklist.json  
    ├─ 生成 synthetic prompt.json
    └─ 保存到 launch 目录
    ↓
IdeaGenerator 生成复现策略
    ├─ DRAgent 生成背景知识
    ├─ GenerationAgent 生成复现方案
    ├─ ReflectionAgent 评估可行性
    ├─ EvolutionAgent 优化策略
    └─ RankingAgent 选择最佳方案
    ↓
ExperimentRunner 执行实验
    ├─ MethodDevelopmentAgent 生成代码
    ├─ RefinementAgent 优化代码
    ├─ ClaudeCodeRunner 执行代码
    ├─ 迭代调试错误
    └─ 生成实验结果
    ↓
ReportGenerator 生成报告
    ├─ 分析实验结果
    ├─ 生成可视化图表
    └─ 撰写研究报告
    ↓
SciEvaluator 评分
    ├─ 读取 report.md
    ├─ 对比 checklist.json
    ├─ LLM 评判员逐项评分
    └─ 生成 final_info.json
```

## 详细技术流程

### 阶段1：任务检测与格式转换

#### 1.1 任务类型检测

**代码位置**：`launch_discovery.py: detect_task_type()`

```python
def detect_task_type(task_dir: str) -> str:
    """检测任务类型：sci 或 auto"""
    if osp.exists(osp.join(task_dir, "task_info.json")):
        return "sci"
    return "auto"
```

**检测逻辑**：
- 检查是否存在 `task_info.json` 文件
- 如果存在 → sci 任务（科学论文复现）
- 如果不存在 → auto 任务（算法优化）

#### 1.2 任务格式转换

**代码位置**：`launch_discovery.py: normalize_sci_task()`

**输入**：
```json
// task_info.json
{
  "task": "复现 ProtTrans 论文的蛋白质二级结构预测",
  "data": [
    {
      "name": "protein_sequences.csv",
      "path": "data/protein_sequences.csv",
      "description": "蛋白质序列数据"
    }
  ]
}
```

**处理流程**：
```python
def normalize_sci_task(task_dir: str, output_path: str) -> dict:
    # 1. 读取 task_info.json
    task_info = json.load(open("task_info.json"))
    
    # 2. 读取 checklist.json
    checklist = json.load(open("target_study/checklist.json"))
    
    # 3. 构建数据清单
    data_manifest = "\n".join([
        f"- {d['name']}: {d['description']}" 
        for d in task_info['data']
    ])
    
    # 4. 构建 checklist 约束
    constraints = [
        f"Item {i} (type={t['type']}, weight={t['weight']}): {t['content'][:200]}"
        for i, t in enumerate(checklist)
    ]
    
    # 5. 生成合成任务描述
    task_description = (
        f"Reproduce findings from a scientific paper.\n\n"
        f"## Research Task\n{task_info['task']}\n\n"
        f"## Available Data\n{data_manifest}\n\n"
        f"## Evaluation Criteria\n{constraints}\n\n"
        f"## Workspace Layout\n"
        "- Write analysis code in `code/`\n"
        "- Save outputs in `outputs/`\n"
        "- Write report as `report/report.md`\n"
    )
    
    # 6. 保存为 prompt.json
    prompt_data = {
        "task_description": task_description,
        "domain": domain,
        "task_type": "sci"
    }
    json.dump(prompt_data, open(output_path, 'w'))
```

**输出**：兼容 MAS 管道的 `prompt.json`

### 阶段2：创意生成阶段

#### 2.1 会话创建

**代码位置**：`orchestration_agent.py: create_session()`

```python
async def create_session(self, goal_description, domain, background=""):
    # 1. 生成背景知识（如果启用 DR Agent）
    if dr_enabled:
        background = await dr_agent.execute({
            "task": f"Generate background for: {goal_description}"
        })
    
    # 2. 创建 Task 对象
    task = Task(
        id=f"task_{timestamp}",
        description=goal_description,
        domain=domain,
        background=background
    )
    
    # 3. 创建 WorkflowSession
    session = WorkflowSession(
        id=f"session_{timestamp}",
        task=task,
        max_iterations=4
    )
    
    # 4. 持久化会话
    await memory_manager.store_session(session)
    self.active_sessions[session_id] = session
    
    return session_id
```

#### 2.2 创意生成状态机

**状态转换**：
```
INITIAL → GENERATING → REFLECTING → EVOLVING → RANKING
```

**GENERATING 状态**：
```python
async def _run_generation_phase(self, session):
    # 1. 调用 GenerationAgent
    response = await generation_agent.execute({
        "task": session.task.description,
        "domain": session.task.domain,
        "background": session.task.background
    })
    
    # 2. 创建 Idea 对象
    ideas = []
    for item in response['ideas']:
        idea = Idea(
            id=item['id'],
            text=item['content'],
            rationale=item.get('rationale', ''),
            iteration=session.iterations_completed + 1
        )
        ideas.append(idea)
    
    # 3. 添加到会话
    session.ideas.extend(ideas)
    
    # 4. 转换状态
    await self._update_session_state(session, WorkflowState.REFLECTING)
```

**REFLECTING 状态**：
```python
async def _run_reflection_phase(self, session):
    # 1. 调用 ReflectionAgent
    critiques = await reflection_agent.execute({
        "ideas": session.ideas,
        "task": session.task.description
    })
    
    # 2. 更新 Idea 对象
    for idea, critique in zip(session.ideas, critiques):
        idea.critiques = critique['points']
        idea.scores = critique['scores']
    
    # 3. 转换状态
    await self._update_session_state(session, WorkflowState.EVOLVING)
```

**EVOLVING 状态**：
```python
async def _run_evolution_phase(self, session):
    # 1. 选择 top ideas
    top_ideas = self._select_top_ideas(session.ideas)
    
    # 2. 调用 EvolutionAgent
    evolved = await evolution_agent.execute({
        "ideas": top_ideas,
        "failed_ideas": task_memory.get_failed_ideas(),
        "successful_patterns": task_memory.get_successful_patterns()
    })
    
    # 3. 创建新 Idea
    for new_idea in evolved['ideas']:
        idea = Idea(
            id=new_idea['id'],
            text=new_idea['content'],
            parent_id=new_idea['parent_id'],
            iteration=session.iterations_completed + 1
        )
        session.ideas.append(idea)
    
    # 4. 转换状态
    await self._update_session_state(session, WorkflowState.RANKING)
```

**RANKING 状态**：
```python
async def _run_ranking_phase(self, session):
    # 1. 调用 RankingAgent
    rankings = await ranking_agent.execute({
        "ideas": session.ideas,
        "criteria": config['ranking_criteria']
    })
    
    # 2. 更新评分
    for idea, ranking in zip(session.ideas, rankings):
        idea.score = ranking['final_score']
        idea.scores.update(ranking['dimension_scores'])
    
    # 3. 选择 Top-N
    top_ids = [idea.id for idea in 
               sorted(session.ideas, key=lambda x: x.score, reverse=True)[:top_k]]
    session.top_ideas = top_ids
    
    # 4. 转换状态
    await self._update_session_state(session, WorkflowState.COMPLETED)
```

### 阶段3：实验执行阶段

#### 3.1 实验环境设置

**代码位置**：`stage.py: setup_experiment_folder()`

```python
def setup_experiment_folder(self, base_dir, results_dir, idea):
    # 1. 提取创意信息
    idea_info = self._extract_idea_info(idea)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    idea_name = f"{timestamp}_{idea_info['name']}"
    
    # 2. 创建实验文件夹
    folder_name = osp.join(results_dir, idea_name)
    shutil.copytree(base_dir, folder_name)
    
    # 3. 创建 run_0 基线
    run0_dir = osp.join(folder_name, "run_0")
    os.makedirs(run0_dir)
    shutil.copy2("experiment.py", f"{run0_dir}/experiment.py")
    
    # 4. 创建说明文件
    with open(f"{folder_name}/notes.txt", 'w') as f:
        f.write(f"# {idea_info['name']}\\n")
        f.write(f"# {idea_info['description']}\\n")
    
    return folder_name, idea_name
```

**目录结构**：
```
{timestamp}_{idea_name}/
├── run_0/                    # 基线代码
│   ├── experiment.py
│   └── final_info.json       # 基线结果
├── data/ → symlink           # 软链接到数据
├── target_study/ → symlink   # 软链接到目标论文
├── related_work/ → symlink    # 软链接到相关论文
└── notes.txt                 # 创意说明
```

#### 3.2 代码生成

**代码位置**：`experiments_utils_claude.py: perform_experiments()`

**第一轮代码生成**：
```python
# Run 1: 从零开始生成代码
if run_num == 1:
    # 使用 sci_task 专用提示词
    prompt = CODER_PROMPT_SCI_TASK.format(
        task=task_info['task'],
        data_manifest=data_manifest,
        checklist_summary=checklist_summary
    )
else:
    # 后续轮：基于前一轮结果优化
    prompt = NEXT_EXPERIMENT_PROMPT_SCI.format(
        previous_result=previous_result,
        errors=errors,
        improvements_needed=improvements
    )
```

**sci_task 专用提示词模板**：
```python
CODER_PROMPT_SCI_TASK = """
You are a computational biologist reproducing a published paper.

## Task
{task}

## Available Data
{data_manifest}

## Evaluation Criteria
{checklist_summary}

## Instructions
1. Write analysis code in `code/experiment.py`
2. Load data from `data/` directory
3. Implement the methods described in the paper
4. Save results to `outputs/` directory
5. Generate figures for key findings
6. Write a detailed report in `report/report.md`

Please start by writing the initial experiment.py file.
"""
```

#### 3.3 代码执行

**代码位置**：`experiments_utils_claude.py: run_experiment()`

```python
def run_experiment(folder_name, run_num, timeout=None):
    # 1. 创建运行目录
    run_dir = osp.join(folder_name, f"run_{run_num}")
    os.makedirs(run_dir)
    
    # 2. 复制文件到运行目录
    for item in os.listdir(folder_name):
        if item.startswith("run_"):
            continue
        src = osp.join(folder_name, item)
        dst = osp.join(run_dir, item)
        
        if osp.isdir(src):
            # sci 任务特殊处理：软链接数据目录
            if task_type == 'sci' and item in SCI_SYMLINK_DIRS:
                os.symlink(osp.abspath(src), dst)
            else:
                shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    
    # 3. 执行实验代码
    result = subprocess.run(
        ['python', 'code/experiment.py'],
        cwd=run_dir,
        capture_output=True,
        text=True,
        timeout=timeout
    )
    
    # 4. 处理输出
    if result.returncode == 0:
        logger.info("Experiment completed successfully")
    else:
        logger.error(f"Experiment failed: {result.stderr}")
        # 提取错误信息用于下一轮优化
        traceback, message = info_traceback(result.stderr)
    
    return result.returncode, traceback, message
```

#### 3.4 迭代优化

**多轮实验循环**：
```python
for run_num in range(1, max_runs + 1):
    # 1. 运行实验
    return_code, traceback, message = run_experiment(
        folder_name, run_num, timeout=run_timeout
    )
    
    # 2. 如果成功，保存结果
    if return_code == 0:
        logger.info(f"Run {run_num} succeeded")
        break
    
    # 3. 如果失败且还有轮次，生成优化代码
    if run_num < max_runs:
        # 构建错误修复提示词
        error_prompt = NEXT_EXPERIMENT_PROMPT.format(
            previous_output=last_output,
            error_message=message,
            traceback=traceback,
            improvements_needed="Fix the errors and improve performance"
        )
        
        # 调用 Claude 生成修复代码
        claude_runner = ClaudeCodeRunner(proxy_settings, model)
        next_prompt = claude_runner.run(error_prompt, cwd=folder_name)
        
        # 更新代码
        last_output = next_prompt
```

### 阶段4：报告生成阶段

#### 4.1 实验结果收集

```python
def collect_experiment_results(run_dir):
    """收集实验结果"""
    results = {
        'outputs': [],
        'figures': [],
        'metrics': {}
    }
    
    # 1. 收集输出文件
    outputs_dir = osp.join(run_dir, 'outputs')
    if osp.exists(outputs_dir):
        for file in os.listdir(outputs_dir):
            results['outputs'].append(file)
    
    # 2. 收集生成的图表
    for ext in ['*.png', '*.jpg', '*.pdf']:
        results['figures'].extend(glob.glob(osp.join(run_dir, 'report', ext)))
    
    # 3. 提取性能指标
    final_info_path = osp.join(run_dir, 'final_info.json')
    if osp.exists(final_info_path):
        with open(final_info_path) as f:
            final_info = json.load(f)
            results['metrics'] = final_info.get('sci_task', {}).get('means', {})
    
    return results
```

#### 4.2 报告生成

**由 MethodDevelopmentAgent 或 ExpAnalyzeAgent 生成**

```python
# Agent 生成的报告模板
REPORT_TEMPLATE = """
# Research Report: {task_title}

## Methods

{methods_description}

## Results

{results_summary}

### Key Findings

{key_findings}

### Performance Metrics

{performance_metrics}

## Discussion

{discussion}

## Conclusion

{conclusion}
"""
```

**报告位置**：`run_N/report/report.md`

### 阶段5：自动评分阶段

#### 5.1 评分流程

**代码位置**：`sci_eval.py: score_run()`

```python
def score_run(workspace_dir, checklist_path, model="gpt-5.1"):
    # 1. 读取 checklist
    with open(checklist_path) as f:
        checklist = json.load(f)
    
    # 2. 读取生成的报告
    workspace = Path(workspace_dir)
    report_text = _read_report(workspace)
    
    # 3. 读取实验指令
    instructions_path = workspace / "INSTRUCTIONS.md"
    instructions = instructions_path.read_text() if instructions_path.exists() else ""
    
    # 4. 查找生成的图片
    generated_images = _find_generated_images(workspace)
    
    # 5. 初始化 LLM 评判员
    agent = LLMAgent(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_version=model,
        system_prompt="You are a strict scientific peer reviewer..."
    )
    
    # 6. 逐项评分
    scores = {}
    total_weight = 0.0
    weighted_sum = 0.0
    
    for i, item in enumerate(checklist):
        # 获取目标图片（如果是 image 类型）
        target_path = None
        if item.get('type') == 'image':
            target_rel = item.get('path', '')
            target_path = safe_resolve(workspace / "target_study", target_rel)
        
        # 单项评分
        result = _score_single_item(
            agent, report_text, item, target_path, 
            generated_images, instructions
        )
        
        # 计算加权分数
        weight = float(item.get('weight', 1.0))
        score = result['score']
        scores[f'item_{i}_score'] = score
        scores[f'item_{i}_reasoning'] = result['reasoning']
        
        weighted_sum += score * weight
        total_weight += weight
    
    # 7. 计算总分
    total_score = weighted_sum / total_weight if total_weight > 0 else 0
    scores['total_score'] = round(total_score, 2)
    
    return scores
```

#### 5.2 单项评分机制

**代码位置**：`rcb_evaluation/score.py: _score_single_item()`

```python
def _score_single_item(agent, report_text, item, target_path, generated_images, instructions):
    # 1. 构建评分提示词
    if item['type'] == 'text':
        prompt = f"""
You are evaluating a research report against a specific criterion.

## Criterion
{item['content']}

## Keywords to check
{item.get('keywords', [])}

## Report Excerpt
{report_text}

## Instructions given to the author
{instructions}

## Scoring Rubric
- 50: Meets the criterion at the level reported in the original paper
- >50: Exceeds the original paper's results
- <50: Falls short of the original paper's results

Please provide:
1. A score from 0-100
2. Brief reasoning for the score
"""
    
    elif item['type'] == 'image':
        # 图片类型需要视觉比较
        prompt = f"""
Compare the generated figure against the target figure from the paper.

## Criterion
{item['content']}

## Generated Figure
{generated_images[0]}

## Target Figure
{target_path}

## Scoring Rubric
(Same as text type)

Please provide:
1. A score from 0-100
2. Brief reasoning
"""
    
    # 2. 调用 LLM 评判员
    response = agent.run(prompt)
    
    # 3. 解析评分
    score = extract_score(response)
    reasoning = extract_reasoning(response)
    
    return {'score': score, 'reasoning': reasoning}
```

#### 5.3 评分标准

**ResearchClawBench 标准评分**：

| 分数范围 | 含义 |
|---------|------|
| **50** | 与原论文结果相当 |
| **>50** | 优于原论文结果 |
| **<50** | 劣于原论文结果 |

**评分维度**：
- **定量结果**：具体数值、性能指标
- **定性分析**：机理解释、方法讨论
- **图表质量**：可视化效果、信息完整性

#### 5.4 评分结果保存

```python
def write_final_info(run_dir, scores):
    """保存评分结果"""
    means = {k: v for k, v in scores.items() if not k.endswith('_reasoning')}
    final_info = {"sci_task": {"means": means}}
    
    final_info_path = osp.join(run_dir, 'final_info.json')
    with open(final_info_path, 'w') as f:
        json.dump(final_info, f, indent=2)
    
    return final_info_path
```

**输出格式**：
```json
{
  "sci_task": {
    "means": {
      "total_score": 42.5,
      "item_0_score": 48,
      "item_0_reasoning": "报告正确描述了方法...",
      "item_1_score": 35,
      "item_1_reasoning": "定量结果略低于论文报告...",
      "item_2_score": 45,
      "item_2_reasoning": "图表清晰展示了主要发现..."
    }
  }
}
```

## 数据流详解

### 主数据流

```
sci_tasks/ProteinBio_001/
├── task_info.json           # 输入：任务描述
├── checklist.json           # 输入：评分标准
├── data/*.csv               # 输入：实验数据
└── target_study/
    ├── paper.pdf            # 输入：目标论文
    └── images/              # 输入：参考图表
    ↓
normalize_sci_task()
    ↓
prompt.json                 # 中间格式：合成任务描述
    ↓
WorkflowSession             # 运行时状态
    ├── task: Task对象
    ├── ideas: List[Idea]
    ├── state: WorkflowState
    └── iterations_completed: int
    ↓
perform_experiments()
    ├── code/experiment.py   # Agent 生成代码
    ├── outputs/             # 实验输出
    └── report/report.md    # Agent 生成报告
    ↓
score_run()
    ↓
final_info.json             # 输出：评分结果
```

### Idea 对象演进

```
GenerationAgent 创建
    ↓
{
    "id": "idea_001",
    "text": "使用 ProtBERT 嵌入进行二级结构预测",
    "score": 0.0,
    "iteration": 1
}
    ↓
ReflectionAgent 评估
    ↓
{
    "id": "idea_001",
    "text": "...",
    "score": 7.5,
    "critiques": ["需要更详细的数据预处理", "模型选择合理"],
    "iteration": 1
}
    ↓
EvolutionAgent 优化
    ↓
{
    "id": "idea_002",
    "text": "改进的 ProtBERT 方法...",
    "score": 0.0,
    "parent_id": "idea_001",
    "iteration": 1
}
    ↓
RankingAgent 排名
    ↓
{
    "id": "idea_002",
    "score": 8.2,
    "scores": {
        "novelty": 7.5,
        "feasibility": 8.5,
        "impact": 8.0
    }
}
    ↓
MethodDevelopmentAgent 开发
    ↓
{
    "id": "idea_002",
    "method_details": {
        "name": "protbert_ss_prediction",
        "description": "...",
        "code_structure": "...",
        "dependencies": ["torch", "transformers"]
    }
}
    ↓
RefinementAgent 精炼
    ↓
{
    "refined_method_details": {
        "optimizations": ["批量处理", "缓存嵌入"],
        "error_handling": "..."
    }
}
```

## Agent 协作机制

### sci_task 专用 Agent 流程

```
启动阶段
    ↓
┌──────────────────────────────────────┐
│ DRAgent (可选)                        │
│ 输入：task_description               │
│ 输出：background 知识                │
└──────────────────────────────────────┘
    ↓
创意生成循环
    ↓
┌──────────────────────────────────────┐
│ GenerationAgent                      │
│ 输入：task, background, domain       │
│ 输出：raw_ideas (List[Idea])         │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ ReflectionAgent                      │
│ 输入：session.ideas                  │
│ 输出：critiques, scores              │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ EvolutionAgent                       │
│ 输入：top_ideas, failed_ideas        │
│ 输出：evolved_ideas                  │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ RankingAgent                         │
│ 输入：session.ideas                  │
│ 输出：top_ideas (List[str])         │
└──────────────────────────────────────┘
    ↓
实验执行循环
    ↓
┌──────────────────────────────────────┐
│ MethodDevelopmentAgent               │
│ 输入：top_ideas                      │
│ 输出：method_details, code           │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ RefinementAgent                      │
│ 输入：method_details                 │
│ 输出：refined_method_details         │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ ClaudeCodeRunner                     │
│ 输入：code, folder                   │
│ 输出：execution results              │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ ExpAnalyzeAgent (可选)               │
│ 输入：experiment results             │
│ 输出：analysis, improvements          │
└──────────────────────────────────────┘
    ↓
评分阶段
    ↓
┌──────────────────────────────────────┐
│ SciEvaluator (LLM评判员)              │
│ 输入：report.md, checklist.json      │
│ 输出：scores (Dict)                  │
└──────────────────────────────────────┘
```

### 信息传递机制

#### 机制1：WorkflowSession 共享状态

```python
# 所有 Agent 通过 WorkflowSession 共享信息
session = WorkflowSession(
    task=task,
    ideas=[],
    state=WorkflowState.GENERATING
)

# Agent 1 添加信息
session.ideas.append(new_idea)

# Agent 2 读取信息
for idea in session.ideas:
    process(idea)

# Agent 3 更新信息
idea.score = new_score
```

#### 机制2：Idea 对象演进

```python
# Idea 对象在 Agent 间传递和演进
idea = Idea(id="idea_1", text="原始创意")

# ReflectionAgent 添加评估
idea.critiques = ["批评1", "批评2"]
idea.scores = {"novelty": 7.5}

# EvolutionAgent 基于评估生成新创意
new_idea = Idea(
    id="idea_2",
    text="改进创意",
    parent_id="idea_1"  # 指向父创意
)

# RankingAgent 排名
idea.score = 8.5

# MethodDevelopmentAgent 添加方法细节
idea.method_details = {"code": "...", "params": {...}}

# RefinementAgent 精炼方法
idea.refined_method_details = {"optimized_code": "..."}
```

#### 机制3：文件系统通信

```python
# 通过文件系统传递大文件
# 输入
sci_tasks/ProteinBio_001/data/*.csv
    ↓ (软链接)
experiment_folder/data/*.csv
    ↓ (读取)
code/experiment.py 处理数据
    ↓ (写入)
experiment_folder/outputs/*.csv
    ↓ (读取)
Agent 分析结果
    ↓ (生成)
experiment_folder/report/report.md
    ↓ (读取)
SciEvaluator 评分
```

## 规划与决策机制

### 规划层次

#### 层次1：工作流规划（状态机）

**决策点**：状态转换

```python
# 状态转换规则
if session.state == INITIAL:
    next_state = GENERATING
elif session.state == GENERATING:
    if generation_success:
        next_state = REFLECTING
    elif need_literature:
        next_state = EXTERNAL_DATA
    else:
        next_state = ERROR
elif session.state == REFLECTING:
    next_state = EVOLVING
# ... 更多状态转换
```

#### 层次2：Agent 规划

**GenerationAgent 规划**：
```python
def plan_generation(task, background):
    # 1. 分析任务需求
    requirements = analyze_task(task)
    
    # 2. 搜索相关文献
    if do_survey:
        literature = scholar_agent.search(task)
    
    # 3. 生成创意策略
    strategy = {
        "approach": "基于文献的方法",
        "key_factors": extract_factors(literature),
        "novelty_angle": identify_gaps(literature)
    }
    
    # 4. 执行创意生成
    ideas = generate_ideas(requirements, strategy, background)
    
    return ideas
```

**MethodDevelopmentAgent 规划**：
```python
def plan_method_development(top_ideas):
    methods = []
    for idea in top_ideas:
        # 1. 分析创意可行性
        feasibility = assess_feasibility(idea)
        
        # 2. 设计实现方案
        implementation = {
            "language": "Python",
            "framework": select_framework(idea),
            "steps": break_down_into_steps(idea)
        }
        
        # 3. 生成代码结构
        code_structure = design_code(idea, implementation)
        
        methods.append({
            "idea_id": idea.id,
            "implementation": implementation,
            "code": code_structure
        })
    
    return methods
```

#### 层次3：实验执行规划

**实验循环规划**：
```python
def plan_experiment_iteration(folder, max_runs):
    iteration_plan = []
    
    for run_num in range(1, max_runs + 1):
        if run_num == 1:
            # 第一轮：从零开始
            plan = {
                "goal": "实现基础功能",
                "focus": "正确性",
                "tolerance": "宽松"
            }
        else:
            # 后续轮：优化改进
            previous_result = load_previous_result(run_num - 1)
            
            if previous_result['errors']:
                plan = {
                    "goal": "修复错误",
                    "focus": "debugging",
                    "target_errors": previous_result['errors']
                }
            elif previous_result['performance'] < target:
                plan = {
                    "goal": "提升性能",
                    "focus": "optimization",
                    "target_metrics": previous_result['weak_metrics']
                }
            else:
                # 已达标，提前结束
                break
        
        iteration_plan.append(plan)
    
    return iteration_plan
```

### 决策机制

#### 决策1：创意筛选决策

```python
def decide_top_ideas(ideas, top_k, criteria):
    # 多维度评分
    for idea in ideas:
        idea.scores = {
            "novelty": evaluate_novelty(idea, criteria),
            "feasibility": evaluate_feasibility(idea, resources),
            "impact": evaluate_impact(idea, task)
        }
    
    # 加权综合评分
    for idea in ideas:
        idea.score = (
            idea.scores['novelty'] * criteria['novelty_weight'] +
            idea.scores['feasibility'] * criteria['feasibility_weight'] +
            idea.scores['impact'] * criteria['impact_weight']
        )
    
    # 选择 Top-K
    top_ideas = sorted(ideas, key=lambda x: x.score, reverse=True)[:top_k]
    
    return top_ideas
```

#### 决策2：迭代优化决策

```python
def decide_next_action(run_result, run_num, max_runs):
    # 分析当前结果
    if run_result['success']:
        if run_result['performance'] >= target_performance:
            # 已达标，停止迭代
            return "STOP"
        elif run_num >= max_runs:
            # 达到最大轮次，停止
            return "STOP"
        else:
            # 继续优化
            if run_result['performance'] < baseline * 0.8:
                # 性能下降严重，回滚
                return "ROLLBACK"
            else:
                # 性能有提升，继续优化
                return "CONTINUE_OPTIMIZE"
    else:
        # 执行失败，修复错误
        return "DEBUG"
```

#### 决策3：评分决策

```python
def decide_score(report, criterion):
    # LLM 评判员评分
    evaluation = llm_judge.evaluate(report, criterion)
    
    # 评分校准
    if criterion['type'] == 'text':
        # 文本类型：侧重内容完整性
        score = calibrate_text_score(evaluation)
    elif criterion['type'] == 'image':
        # 图片类型：侧重视觉相似度
        score = calibrate_image_score(evaluation, reference_image)
    
    # 加权调整
    final_score = score * criterion['weight']
    
    return final_score
```

## 关键技术特点

### 特点1：全自动流程

**从论文到报告的自动化**：
- 输入：论文PDF + 数据文件
- 输出：分析代码 + 实验结果 + 研究报告 + 评分

**无人工干预**：
- Agent 自动分析论文
- 自动生成代码
- 自动调试错误
- 自动生成报告

### 特点2：智能迭代优化

**多轮实验机制**：
- Run 0：基线（原始代码）
- Run 1：首次实现
- Run 2：错误修复
- Run N：性能优化

**自适应调整**：
- 根据错误信息调整代码
- 根据性能结果优化策略
- 智能选择优化方向

### 特点3：标准化评分

**ResearchClawBench 对齐**：
- 使用官方评分 rubric
- LLM 评判员一致性
- 可重复的评分标准

**多维度评估**：
- 定量结果评估
- 定性分析评估
- 图表质量评估

### 特点4：可扩展性

**简单添加新任务**：
```bash
# 1. 创建目录
mkdir sci_tasks/tasks/MyDomain_001

# 2. 准备文件
# - task_info.json
# - data/*.csv
# - target_study/paper.pdf
# - target_study/checklist.json

# 3. 运行
bash scripts/run_sci.sh MyDomain_001
```

## 运行示例

### 命令行运行

```bash
# 基础运行
python launch_discovery.py \
    --task sci_tasks/tasks/ProteinBio_001 \
    --config config/default_config.yaml \
    --exp_backend claudecode

# 快速测试（优化配置）
python launch_discovery.py \
    --task sci_tasks/tasks/ProteinBio_001 \
    --config config/default_config.yaml \
    --exp_backend claudecode
    # 配置中设置：
    # workflow.loop_rounds = 1
    # workflow.top_ideas_count = 2
    # experiment.max_runs = 1
```

### 结果查看

```bash
# 查看实验报告
cat results/ProteinBio_001/*_launch/*/run_1/report/report.md

# 查看评分结果
cat results/ProteinBio_001/*_launch/*/run_1/final_info.json

# 查看生成代码
ls results/ProteinBio_001/*_launch/*/run_1/code/

# 查看实验输出
ls results/ProteinBio_001/*_launch/*/run_1/outputs/
```

## 总结

InternAgent 的科学论文复现功能是一个高度自动化的系统，核心特点：

1. **全自动流程**：从论文到报告的端到端自动化
2. **智能Agent协作**：专业Agent团队分工协作
3. **状态机驱动**：清晰的工作流状态管理
4. **迭代优化**：多轮实验自动调试和优化
5. **标准化评分**：与ResearchClawBench对齐的评分系统
6. **高度可扩展**：简单的任务添加流程

该功能使得科研人员可以快速验证论文结果，无需编写代码，极大提升了科研效率。

---

*基于 InternAgent-1.5 sci_tasks 功能分析*
*分析时间：2024-06-03*
