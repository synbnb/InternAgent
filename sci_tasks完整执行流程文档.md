# InternAgent sci_tasks 完整执行流程文档

## 📋 目录

1. [执行流程概述](#执行流程概述)
2. [启动阶段](#启动阶段-launch_discoverypy)
3. [创意生成阶段](#创意生成阶段-ideagenerator)
4. [实验执行阶段](#实验执行阶段-experimentrunner)
5. [代码生成与执行阶段](#代码生成与执行阶段-claudecoderunner)
6. [评分阶段](#评分阶段-sci_evalpy)
7. [完整调用链](#完整调用链)
8. [数据流向](#数据流向)
9. [关键文件清单](#关键文件清单)

---

## 执行流程概述

### 整体架构

```
用户命令
    ↓
[启动阶段] launch_discovery.py (main)
    ↓
[创意生成] IdeaGenerator.generate_ideas()
    ↓ InternAgentInterface
    ↓ MAS Workflow (OrchestrationAgent)
    ↓ 多个Agent (Generation, Reflection, Evolution, Ranking, MethodDevelopment, Refinement)
    ↓
[实验执行] ExperimentRunner.run_experiments()
    ↓ 为每个idea创建实验文件夹
    ↓ 调用具体backend (claudecode/ifth/openhands)
    ↓
[代码生成] perform_experiments_claudecode()
    ↓ ClaudeCodeRunner.run() (调用Claude CLI)
    ↓ 生成code/experiment.py
    ↓
[运行实验] run_experiment()
    ↓ bash launcher.sh
    ↓ 执行code/experiment.py
    ↓ 生成final_info.json + report/report.md
    ↓
[评分阶段] sci_eval.score_run()
    ↓ LLM评判员评分
    ↓ 生成final_info.json (含分数)
    ↓
[结果汇总] 返回结果到main()
    ↓ 保存discovery_summary.json
    ↓ 完成
```

---

## 启动阶段 (launch_discovery.py)

### 文件: `launch_discovery.py`

#### 函数调用顺序

```
main()
    ↓
parse_arguments() [解析命令行参数]
    ↓
detect_task_type() [检测任务类型: sci vs auto]
    ↓
normalize_sci_task() [如果是sci任务，生成prompt.json]
    ↓
main() 主循环 [多轮迭代 loop_rounds]
    ↓
IdeaGenerator(...).generate_ideas()
    ↓
ExperimentRunner(...).run_experiments()
    ↓
[结果汇总与保存]
```

#### 关键代码段

**1. 任务类型检测**
```python
# launch_discovery.py:37
def detect_task_type(task_dir: str) -> str:
    """检测是'sci'任务还是'auto'任务"""
    if osp.exists(osp.join(task_dir, "task_info.json")):
        return "sci"
    return "auto"
```

**2. sci任务规范化**
```python
# launch_discovery.py:49
def normalize_sci_task(task_dir: str, output_path: str) -> dict:
    """读取task_info.json + checklist.json，生成prompt.json"""
    
    # 读取task_info.json
    task_info = json.load(open("task_info.json"))
    
    # 读取checklist.json
    checklist = json.load(open("target_study/checklist.json"))
    
    # 生成任务描述
    task_description = (
        f"Reproduce the findings from a scientific paper in the {domain} domain.\n\n"
        f"## Research Task\n{task_info.get('task', '')}\n\n"
        f"## Available Data\n{data_manifest}\n\n"
        f"## Evaluation Criteria ({len(checklist)} checklist items)\n"
        + "\n".join(constraints)
    )
    
    # 保存为prompt.json
    prompt_data = {
        "system": f"You are a scientific researcher...",
        "task_description": task_description,
        "domain": domain,
        "task_type": "sci"
    }
```

**3. 主循环**
```python
# launch_discovery.py:850
for round_num in range(start_round, loop_rounds + 1):
    # Step 1: 创意生成
    idea_generator = IdeaGenerator(args, logger, round_num, config)
    top_ideas, session_json = asyncio.run(idea_generator.generate_ideas())
    
    # Step 2: 实验执行
    experiment_runner = ExperimentRunner(args, logger, config, session_id=session_id)
    results = experiment_runner.run_experiments(
        base_dir=base_code_dir,
        results_dir=experiment_results_dir,
        ideas=top_ideas
    )
    
    # Step 3: 结果汇总
    all_round_results.append({
        'round': round_num,
        'session_id': session_id,
        'results': results
    })
```

---

## 创意生成阶段 (IdeaGenerator)

### 文件: `internagent/stage.py`

#### 类: `IdeaGenerator`

#### 函数调用顺序

```
IdeaGenerator.__init__()
    ↓ 初始化 InternAgentInterface
    ↓ 初始化 IdeaGraph (如果启用)
    ↓
IdeaGenerator.generate_ideas() [async]
    ↓
await self.load_task() [创建session]
    ↓
while self.status != "completed":
    ↓
await self.interface.run_session()
    ↓ MAS工作流执行
    ↓
[检查状态: awaiting_feedback → completed → error]
    ↓
await self.interface.get_top_ideas()
    ↓ 返回top_ideas
```

#### 关键代码

**1. 初始化**
```python
# internagent/stage.py:68
class IdeaGenerator:
    def __init__(self, args, logger, round_num=1, config=None):
        self.interface = InternAgentInterface(
            args.config,
            work_dir=args.task_dir,
            task_name=args.task_name,
            exp_backend=args.exp_backend
        )
        self.idea_graph = IdeaGraph(...)  # 如果启用
```

**2. 生成创意**
```python
# internagent/stage.py:283
async def generate_ideas(self):
    """运行MAS生成创意"""
    await self.load_task()  # 创建session
    
    while self.status != "completed":
        full_status = await self.interface.get_session_status(self.session_id)
        self.status = full_status['state']
        
        if self.status == "awaiting_feedback":
            # 添加反馈
            await self.interface.add_feedback(self.session_id, feedback)
        elif self.status == "completed":
            break
        
        # 运行session
        await self.interface.run_session(self.session_id, status_callback)
    
    # 获取top ideas
    top_ideas = await self.interface.get_top_ideas(self.session_id)
    return top_ideas
```

---

## MAS工作流阶段

### 文件: `internagent/mas/interface.py` + `internagent/mas/workflow/orchestration_agent.py`

#### 函数调用顺序

```
InternAgentInterface.create_session()
    ↓
发送 HTTP POST 到 MAS服务
    ↓ 返回 session_id
    
InternAgentInterface.run_session(session_id)
    ↓
OrchestrationAgent.execute_session()
    ↓
[Phase 1: Paper Gathering] (如果do_survey=true)
    ↓
SurveyAgent.execute(context)
    ↓
搜索外部数据库 (arXiv, Semantic Scholar等)
    ↓ 返回 paper_lst
    
    ↓
[Phase 2: Idea Generation]
    ↓
GenerationAgent.execute(context, params)
    ↓
接收: goal, paper_lst, feedback
    ↓
生成多个hypotheses (创意)
    ↓ 返回 hypotheses
    
    ↓
[Phase 3: Reflection] (如果启用)
    ↓
ReflectionAgent.execute()
    ↓
评估每个hypothesis
    ↓ 生成 critiques
    
    ↓
[Phase 4: Evolution] (如果启用)
    ↓
EvolutionAgent.execute()
    ↓
基于反馈进化hypotheses
    ↓ 返回 evolved_hypotheses
    
    ↓
[Phase 5: Method Development]
    ↓
MethodDevelopmentAgent.execute()
    ↓
将hypothesis转换为具体方法
    ↓ 返回 method_details
    
    ↓
[Phase 6: Ranking]
    ↓
RankingAgent.execute()
    ↓
对所有创意评分排序
    ↓ 返回 ranking_scores
    
    ↓
[Phase 7: Refinement] (如果启用)
    ↓
RefinementAgent.execute()
    ↓
精炼method_details
    ↓ 返回 refined_method_details
    
    ↓
[完成] session状态 → "completed"
```

---

## 实验执行阶段 (ExperimentRunner)

### 文件: `internagent/stage.py`

#### 类: `ExperimentRunner`

#### 函数调用顺序

```
ExperimentRunner.__init__()
    ↓ 初始化 GPU分配器
    ↓ 初始化 OnlineMemorySaver (如果启用)
    ↓
ExperimentRunner.run_experiments(base_dir, results_dir, ideas)
    ↓
[遍历每个idea]
    ↓
_run_single_experiment(idx, idea, base_dir, results_dir)
    ↓
[获取GPU资源]
    ↓ gpu_allocator.get_gpu_env()
    ↓
[根据backend调用]
    ↓
run_claude_experiment() [如果backend=claudecode]
    ↓ run_iflow_experiment() [如果backend=ifth]
    ↓ run_openhands_experiment() [如果backend=openhands]
    ↓
返回: {success, folder_name, performance}
```

#### 关键代码

**1. 初始化**
```python
# internagent/stage.py:468
class ExperimentRunner:
    def __init__(self, args, logger, config=None, session_id=None, base_code_dir=None):
        # 初始化GPU分配器
        self._init_gpu_allocator()
        
        # 初始化在线记忆
        if config.get("memory", {}).get("online_memory", {}).get("enabled", False):
            self.memory_saver = OnlineMemorySaver(config, task_name)
```

**2. 运行实验**
```python
# internagent/stage.py:1154
def run_experiments(self, base_dir, results_dir, ideas):
    """为所有ideas运行实验（支持并行）"""
    
    if max_parallel == 1:
        # 顺序执行
        for idx, idea in enumerate(ideas, 1):
            result = self._run_single_experiment(idx, idea, base_dir, results_dir)
            results.append(result)
    else:
        # 并行执行
        with ThreadPoolExecutor(max_workers=self.max_parallel_experiments) as executor:
            futures = {executor.submit(self._run_single_experiment, ...): idea}
            for future in as_completed(futures):
                results.append(future.result())
    
    return results
```

**3. 单个实验执行**
```python
# internagent/stage.py:1043
def _run_single_experiment(self, idx, idea, base_dir, results_dir):
    """运行单个实验"""
    
    # 获取GPU
    gpu_ids = self.gpu_allocator.get_gpu_env()
    
    # 根据backend执行
    if self.backend == "claudecode":
        success, folder_name = self.run_claude_experiment(...)
    elif self.backend == "ifth":
        success, folder_name = self.run_iflow_experiment(...)
    
    # 计算性能提升
    if success:
        performance = self._calculate_experiment_performance(folder_name, base_dir)
    
    # 保存到在线记忆
    if success and self.memory_saver:
        self.memory_saver.save_idea_result(idea, folder_name, session_id)
    
    return {success, folder_name, performance}
```

---

## 代码生成与执行阶段 (ClaudeCodeRunner)

### 文件: `internagent/experiments_utils_claude.py`

#### 主要函数

```
perform_experiments_claudecode()
    ↓
初始化 ClaudeCodeRunner
    ↓
_build_sci_initial_prompt() [构建初始prompt]
    ↓
while run < max_runs + 1:
    ↓
    # Claude生成代码
    claude_runner.run(prompt, cwd=folder_name)
        ↓ 调用Claude CLI
        ↓ 生成/修改 code/experiment.py
        ↓ 返回输出
        ↓
    # 运行实验
    run_experiment(folder_name, run_num)
        ↓ 创建 run_N 目录
        ↓ 复制文件到run_N/
        ↓ bash launcher.sh
        ↓ 执行 code/experiment.py
        ↓ 生成 final_info.json + report/report.md
        ↓ 返回 (return_code, next_prompt)
        ↓
    # sci任务评分
    if task_type == 'sci':
        _handle_sci_run_scoring()
        ↓ sci_eval.score_run()
        ↓ LLM评判员评分
        ↓ 写入 final_info.json
        ↓
    # 判断是否继续
    if return_code == 0:
        run += 1  # 成功，下一轮
    else:
        current_iter += 1  # 失败，重试
    ↓
返回 success (True/False)
```

#### 关键代码

**1. 初始prompt构建**
```python
# internagent/experiments_utils_claude.py:436
def _build_sci_initial_prompt(idea_info, task_info, checklist, max_runs, folder_name):
    """构建sci任务的初始prompt"""
    
    # 从task_info提取任务描述
    task_description = task_info.get('task', '')
    data_manifest = format_data_items(task_info.get('data', []))
    
    # 从checklist提取评分标准
    checklist_summary = format_checklist(checklist)
    
    # 使用预定义模板
    return CODER_PROMPT_SCI_TASK.format(
        idea_description=idea_info["description"],
        method=idea_info["method"],
        task_description=task_description,
        data_manifest=data_manifest,
        checklist_count=len(checklist),
        checklist_summary=checklist_summary,
        max_runs=max_runs
    )
```

**2. 主执行循环**
```python
# internagent/experiments_utils_claude.py:307
def perform_experiments(idea, folder_name, ...):
    """执行多轮实验"""
    
    # 构建初始prompt
    if task_type == 'sci':
        next_prompt = _build_sci_initial_prompt(...)
    else:
        next_prompt = CODER_PROMPT_OPENHANDS.format(...)
    
    # 主循环
    run = 1
    while run < max_runs + 1:
        # Claude生成代码
        claude_output = claude_runner.run(next_prompt, cwd=folder_name)
        
        if "ALL_COMPLETED" in claude_output:
            break
        
        # 运行实验
        return_code, next_prompt, traceback, message = run_experiment(
            folder_name, run, timeout, gpu_ids
        )
        
        # sci任务评分
        if task_type == 'sci':
            return_code, next_prompt = _handle_sci_run_scoring(...)
        
        # 判断下一步
        if return_code == 0:
            run += 1  # 成功，进入下一轮
        else:
            current_iter += 1  # 失败，重试
    
    return True/False
```

**3. 运行单次实验**
```python
# internagent/experiments_utils_claude.py:147
def run_experiment(folder_name, run_num, timeout, gpu_ids):
    """运行单次实验"""
    
    # 创建run目录
    run_dir = osp.join(folder_name, f"run_{run_num}")
    
    # 复制文件（sci任务特殊处理软链接）
    for item in os.listdir(folder_name):
        if task_type == 'sci' and item in SCI_SYMLINK_DIRS:
            os.symlink(src, dst)  # 软链接data/, related_work/, target_study/
        else:
            shutil.copytree(src, dst)  # 复制code/
    
    # 执行命令
    command = ["bash", "launcher.sh"]
    process = subprocess.Popen(command, cwd=run_dir, env=env)
    process.wait(timeout=timeout)
    
    # 读取结果
    if returncode == 0:
        results = json.load(open("run_N/final_info.json"))
        next_prompt = NEXT_EXPERIMENT_PROMPT.format(RUN_NUM=run_num, RESULTS=results)
    else:
        next_prompt = f"Error: {traceback}... Please fix this error"
    
    return returncode, next_prompt, traceback, message
```

---

## 评分阶段 (sci_eval.py)

### 文件: `internagent/sci_eval.py`

#### 函数调用顺序

```
_handle_sci_run_scoring()
    ↓
检查: report.md是否存在？final_info.json是否存在？
    ↓
sci_eval.score_run(workspace_dir, checklist_path, model)
    ↓
读取checklist.json
    ↓
读取report/report.md
    ↓
读取INSTRUCTIONS.md (可选)
    ↓
初始化LLM评判员 (structai.LLMAgent)
    ↓
遍历每个checklist项
    ↓
_score_single_item()
    ↓
判断类型 (text vs image)
    ↓
LLM评分
    ↓ 返回单项分数
    ↓
汇总所有分数
    ↓
write_final_info()
    ↓ 写入 final_info.json
```

#### 关键代码

**1. 评分入口**
```python
# internagent/sci_eval.py:38
def score_run(workspace_dir, checklist_path, model="gpt-5.1"):
    """使用LLM评判员评分"""
    
    # 读取checklist
    with open(checklist_path) as f:
        checklist = json.load(f)
    
    # 读取report
    report_text = _read_report(workspace)
    
    # 读取instructions
    instructions = (workspace / "INSTRUCTIONS.md").read_text()
    
    # 初始化LLM评判员
    agent = LLMAgent(
        api_key=OPENAI_API_KEY,
        model_version=model,
        system_prompt="You are a strict scientific peer reviewer..."
    )
    
    # 并行评分每个checklist项
    for i, item in enumerate(checklist):
        score = _score_single_item(
            agent, report_text, item, 
            target_path=None,  # 如果是image类型，会设置
            generated_images=generated_images,
            instructions=instructions
        )
        scores[i] = score
    
    # 汇总分数
    total_score = sum(score['score'] * score['weight'] for score in scores)
    
    return {'total_score': total_score, 'item_scores': scores}
```

**2. 单项评分**
```python
# internagent/rcb_evaluation/score.py
def _score_single_item(agent, report_text, item, target_path, generated_images, instructions):
    """评分单个checklist项"""
    
    # 构建评分prompt
    prompt = f"""
    You are evaluating a research report.
    
    Instructions:
    {instructions}
    
    Criterion:
    {item['content']}
    
    Report:
    {report_text}
    
    Score 0-10 based on how well the report satisfies the criterion.
    """
    
    # 调用LLM
    response = agent.run(prompt)
    
    # 解析分数
    score = extract_score(response)
    
    return {'score': score, 'rationale': response}
```

---

## 完整调用链

### 从命令行到结果

```
$ python launch_discovery.py --task sci_tasks/tasks/ProteinBio_001 --exp_backend claudecode

[1] launch_discovery.py:main()
    |
    ├─→ parse_arguments()
    ├─→ detect_task_type() → "sci"
    ├─→ normalize_sci_task()
    |   ├─→ 读取 task_info.json
    |   ├─→ 读取 target_study/checklist.json
    |   └─→ 写入 prompt.json
    |
    └─→ [循环开始]
        |
        ├─→ IdeaGenerator(args, logger, round_num, config)
        |   └─→ generate_ideas()
        |       |
        |       ├─→ load_task() → InternAgentInterface.create_session()
        |       |   └─→ HTTP POST to MAS server
        |       |       └─→ 返回 session_id
        |       |
        |       └─→ [while status != "completed"]
        |           |
        |           └─→ interface.run_session(session_id)
        |               |
        |               └─→ OrchestrationAgent.execute_session()
        |                   |
        |                   ├─→ [Phase 1: Paper Gathering] (do_survey=true)
        |                   |   └─→ SurveyAgent.execute(context)
        |                   |       ├─→ 基于task_description生成搜索查询
        |                   |       ├─→ 搜索外部数据库 (arXiv, Semantic Scholar)
        |                   |       └─→ 返回 paper_lst
        |                   |
        |                   ├─→ [Phase 2: Idea Generation]
        |                   |   └─→ GenerationAgent.execute(context)
        |                   |       ├─→ 输入: goal, paper_lst, feedback
        |                   |       ├─→ LLM生成多个hypotheses
        |                   |       └─→ 返回 hypotheses
        |                   |
        |                   ├─→ [Phase 3: Reflection]
        |                   |   └─→ ReflectionAgent.execute()
        |                   |       ├─→ 评估每个hypothesis
        |                   |       └─→ 返回 critiques
        |                   |
        |                   ├─→ [Phase 4: Evolution]
        |                   |   └─→ EvolutionAgent.execute()
        |                   |       ├─→ 基于critiques进化hypotheses
        |                   |       └─→ 返回 evolved_hypotheses
        |                   |
        |                   ├─→ [Phase 5: Method Development]
        |                   |   └─→ MethodDevelopmentAgent.execute()
        |                   |       ├─→ 转换hypothesis为具体方法
        |                   |       └─→ 返回 method_details
        |                   |
        |                   ├─→ [Phase 6: Ranking]
        |                   |   └─→ RankingAgent.execute()
        |                   |       ├─→ 评分排序所有hypotheses
        |                   |       └─→ 返回 ranking_scores
        |                   |
        |                   ├─→ [Phase 7: Refinement]
        |                   |   └─→ RefinementAgent.execute()
        |                   |       ├─→ 精炼method_details
        |                   |       └─→ 返回 refined_method_details
        |                   |
        |                   └─→ 状态 → "completed"
        |
        ├─→ interface.get_top_ideas(session_id)
        |   └─→ 返回 top_ideas (例如: 2个top ideas)
        |
        └─→ ExperimentRunner(args, logger, config, session_id)
            └─→ run_experiments(base_dir, results_dir, ideas)
                |
                └─→ [遍历每个idea]
                    |
                    └─→ _run_single_experiment(idx, idea, ...)
                        |
                        ├─→ gpu_allocator.get_gpu_env() → "0" (GPU ID)
                        |
                        └─→ run_claude_experiment(base_dir, results_dir, idea, gpu_ids)
                            |
                            ├─→ setup_sci_experiment_folder()
                            |   ├─→ 创建实验目录
                            |   ├─→ 软链接 data/, related_work/, target_study/
                            |   ├─→ 创建 code/, outputs/, report/
                            |   ├─→ 创建 launcher.sh
                            |   └─→ 创建 run_0/final_info.json (baseline)
                            |
                            ├─→ perform_experiments_claudecode(idea, cwd, ...)
                            |   |
                            |   └─→ [主循环: run 1 to max_runs]
                            |       |
                            |       ├─→ _build_sci_initial_prompt()
                            |       |   ├─→ 读取 task_info (从内存)
                            |       |   ├─→ 读取 checklist (从内存)
                            |       |   └─→ 生成 prompt (CODER_PROMPT_SCI_TASK)
                            |       |
                            |       ├─→ ClaudeCodeRunner.run(prompt, cwd)
                            |       |   ├─→ 调用Claude CLI (claude命令)
                            |       |   ├─→ 传递prompt给Claude
                            |       |   ├─→ Claude读取代码
                            |       |   ├─→ Claude生成/修改 code/experiment.py
                            |       |   └─→ 返回输出
                            |       |
                            |       ├─→ run_experiment(folder_name, run_num)
                            |       |   ├─→ 创建 run_N/ 目录
                            |       |   ├─→ 复制文件到run_N/
                            |       |   |   ├─→ code/ → 复制
                            |       |   |   ├─→ data/ → 软链接 (sci)
                            |       |   |   ├─→ related_work/ → 软链接 (sci)
                            |       |   |   └─→ target_study/ → 软链接 (sci)
                            |       |   |
                            |       |   ├─→ subprocess.Popen(["bash", "launcher.sh"])
                            |       |   |   ├─→ 执行 launcher.sh
                            |       |   |   ├─→ launcher.sh → python code/experiment.py
                            |       |   |   ├─→ 生成 outputs/ 结果
                            |       |   |   ├─→ 生成 report/report.md
                            |       |   |   └─→ 生成 final_info.json (可选)
                            |       |   |
                            |       |   └─→ 返回 (return_code, next_prompt)
                            |       |
                            |       ├─→ _handle_sci_run_scoring() (如果return_code==0)
                            |       |   ├─→ 检查 report.md 和 final_info.json
                            |       |   ├─→ sci_eval.score_run()
                            |       |   |   ├─→ 读取 checklist.json
                            |       |   |   ├─→ 读取 report/report.md
                            |       |   |   ├─→ 初始化 LLM评判员
                            |       |   |   ├─→ 遍历checklist项，LLM评分
                            |       |   |   └─→ 返回 scores dict
                            |       |   └─→ write_final_info(scores)
                            |       |       └─→ 写入 run_N/final_info.json
                            |       |
                            |       └─→ 判断: return_code==0? → run++ : current_iter++
                            |
                            └─→ 返回 success (True/False)
                    |
                    └─→ 返回 {success, folder_name, performance}
        |
        └─→ [循环结束]
            |
            ├─→ 汇总所有round结果
            ├─→ 保存 discovery_summary.json
            └─→ 退出
```

---

## 数据流向

### 输入数据流

```
用户文件系统
    ↓
sci_tasks/tasks/ProteinBio_001/
    ├─→ task_info.json
    │   └─→ normalize_sci_task() → prompt.json
    │       └─→ InternAgentInterface.create_session()
    │           └─→ MAS Workflow (所有Agent)
    │
    └─→ target_study/
        ├─→ checklist.json
        │   └─→ normalize_sci_task() → prompt.json
        │       └─→ [最终传递到评分]
        │           └─→ sci_eval.score_run()
        │
        └─→ paper.pdf
            └─→ [不被读取，仅作为存档]
```

### 创意生成数据流

```
prompt.json (task_description)
    ↓
SurveyAgent (如果do_survey=true)
    ├─→ 输入: task_description
    ├─→ 搜索: 外部数据库
    └─→ 输出: paper_lst
        ↓
GenerationAgent
    ├─→ 输入: goal, paper_lst, feedback
    ├─→ LLM生成: hypotheses
    └─→ 输出: hypotheses[{text, rationale, ...}]
        ↓
ReflectionAgent (如果启用)
    ├─→ 输入: hypotheses
    ├─→ LLM评估: critiques
    └─→ 输出: hypotheses[{..., critiques}]
        ↓
EvolutionAgent (如果启用)
    ├─→ 输入: hypotheses, critiques
    ├─→ LLM进化: evolved_hypotheses
    └─→ 输出: evolved_hypotheses
        ↓
MethodDevelopmentAgent
    ├─→ 输入: hypotheses
    ├─→ LLM转换: method_details
    └─→ 输出: ideas[{method_details: {name, method, description}}]
        ↓
RankingAgent
    ├─→ 输入: ideas
    ├─→ LLM评分: ranking_scores
    └─→ 输出: ideas[{..., score}]
        ↓
RefinementAgent (如果启用)
    ├─→ 输入: ideas
    ├─→ LLM精炼: refined_method_details
    └─→ 输出: top_ideas[{refined_method_details}]
```

### 实验执行数据流

```
top_ideas (创意列表)
    ↓
ExperimentRunner.run_experiments()
    ↓
[遍历每个idea]
    ↓
setup_sci_experiment_folder()
    ├─→ 创建实验目录
    ├─→ 软链接 data/, related_work/, target_study/
    └─→ 创建基础文件结构
        ↓
perform_experiments_claudecode()
    ↓
[循环: run 1 to max_runs]
    ↓
_build_sci_initial_prompt()
    ├─→ 输入: idea_info, task_info, checklist
    └─→ 输出: prompt (给Claude)
        ↓
ClaudeCodeRunner.run()
    ├─→ 调用Claude CLI
    ├─→ 读取prompt
    ├─→ 读取现有代码
    ├─→ LLM生成新代码
    └─→ 写入 code/experiment.py
        ↓
run_experiment()
    ├─→ bash launcher.sh
    ├─→ python code/experiment.py
    ├─→ 生成 outputs/ (数据)
    ├─→ 生成 report/report.md
    └─→ 生成 final_info.json (或失败)
        ↓
_handle_sci_run_scoring() (如果成功)
    ├─→ sci_eval.score_run()
    │   ├─→ 读取report.md
    │   ├─→ 读取checklist.json
    │   ├─→ LLM评判员评分
    │   └─→ 写入final_info.json (带分数)
    └─→ 判断是否继续
        ↓
返回: success, folder_name, performance
```

---

## 关键文件清单

### 核心执行文件

| 文件 | 作用 | 关键函数/类 |
|------|------|------------|
| `launch_discovery.py` | 启动入口 | `main()`, `normalize_sci_task()` |
| `internagent/stage.py` | 创意生成+实验执行 | `IdeaGenerator`, `ExperimentRunner` |
| `internagent/mas/interface.py` | MAS接口 | `InternAgentInterface` |
| `internagent/mas/workflow/orchestration_agent.py` | MAS工作流编排 | `OrchestrationAgent` |
| `internagent/experiments_utils_claude.py` | Claude代码生成 | `perform_experiments()`, `ClaudeCodeRunner` |
| `internagent/sci_eval.py` | sci任务评分 | `score_run()` |

### Agent文件

| 文件 | 作用 |
|------|------|
| `internagent/mas/agents/generation_agent.py` | 生成创意 |
| `internagent/mas/agents/reflection_agent.py` | 评估创意 |
| `internagent/mas/agents/evolution_agent.py` | 进化创意 |
| `internagent/mas/agents/method_development_agent.py` | 方法开发 |
| `internagent/mas/agents/ranking_agent.py` | 创意排序 |
| `internagent/mas/agents/refinement_agent.py` | 精炼方法 |
| `internagent/mas/agents/survey_agent.py` | 文献调研 |

### 配置和提示词文件

| 文件 | 作用 |
|------|------|
| `config/default_config.yaml` | 全局配置 |
| `internagent/prompts.py` | 提示词模板 (CODER_PROMPT_SCI_TASK等) |
| `sci_tasks/tasks/*/task_info.json` | 任务描述 |
| `sci_tasks/tasks/*/target_study/checklist.json` | 评分标准 |

---

## 总结

### sci_tasks执行流程的5个主要阶段

1. **启动准备阶段**
   - 检测任务类型
   - 读取并规范化sci任务 (normalize_sci_task)
   - 生成prompt.json

2. **创意生成阶段**
   - 初始化MAS session
   - 文献调研 (可选, do_survey=true)
   - 多Agent协作生成创意
   - 返回top ideas

3. **实验准备阶段**
   - 为每个idea创建实验文件夹
   - 设置软链接 (data/, related_work/, target_study/)
   - 初始化baseline (run_0/)

4. **代码生成与执行阶段**
   - Claude生成代码 (多轮迭代)
   - 执行实验 (bash launcher.sh)
   - 生成report和结果
   - LLM评判员评分 (sci任务)

5. **结果汇总阶段**
   - 收集所有实验结果
   - 计算性能提升
   - 保存summary和报告

### 关键设计特点

1. **分离关注点**: 目标论文 vs 相关论文
2. **多Agent协作**: Generation → Reflection → Evolution → MethodDev → Ranking → Refinement
3. **软链接优化**: data/, related_work/, target_study/ 使用软链接避免复制
4. **LLM评判员**: 使用LLM对sci任务进行客观评分
5. **迭代优化**: 支持多轮迭代和多轮循环 (loop_rounds)

---

*文档生成时间: 2024-06-08*
*版本: 1.0*
