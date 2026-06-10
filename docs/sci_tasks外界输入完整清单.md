# sci_tasks外界输入完整清单

## 📋 概述

本文档详细列出InternAgent的sci_tasks流程中所有外界输入来源，帮助您理解启动一个科学论文复现任务需要准备什么。

---

## 🗂️ 一、必需文件输入（用户准备）

### 1.1 核心配置文件

#### `task_info.json`（必需）
**位置**：`tasks/{domain}_XXX/task_info.json`

**作用**：定义复现任务的元数据

**必需字段**：
```json
{
  "task": "复现ProtTrans论文的核心发现：使用蛋白质语言模型进行二级结构预测",
  "data": [
    {
      "name": "protein_sequences_sample.csv",
      "description": "包含1000条蛋白质序列样本及其二级结构标签"
    },
    {
      "name": "pretrained_embeddings.json", 
      "description": "预训练蛋白质语言模型生成的嵌入向量"
    }
  ]
}
```

**为什么必需**：
- `normalize_sci_task()`首先读取此文件
- 生成`task_description`字段
- 生成`data_manifest`字段

**代码位置**：`launch_discovery.py:67-90`

#### `target_study/checklist.json`（强烈推荐）
**位置**：`tasks/{domain}_XXX/target_study/checklist.json`

**作用**：定义评分标准和验证指标

**格式**：
```json
[
  {
    "type": "quantitative",
    "weight": 0.3,
    "content": "报告二级结构预测的准确率（Q3 score），与论文报告值对比",
    "rubric": "准确率在论文报告值±5%范围内：3分；±10%范围：2分；超过±10%：1分"
  },
  {
    "type": "image",
    "weight": 0.2,
    "content": "生成混淆矩阵热图，展示H/E/C三类预测性能",
    "path": "images/confusion_matrix.png"
  }
]
```

**为什么推荐**：
- 评分系统`sci_eval.py`读取此文件
- 如果不存在，checklist为空，无法评分
- 定义了LLM-as-judge的评分依据

**代码位置**：`launch_discovery.py:91-95`

#### `target_study/paper.pdf`（可选，仅存档）
**位置**：`tasks/{domain}_XXX/target_study/paper.pdf`

**作用**：
- ❌ **不被代码读取**
- ✅ 作为"任务来源证明"存档
- ✅ 供用户手动参考

**为什么不被读取**：
- `normalize_sci_task()`不读取PDF
- 整个流程中没有代码调用`extract_text_from_pdf()`处理此文件
- 系统依赖用户手动提供的`task_info.json`

### 1.2 数据文件（根据task_info.json中定义）

#### `data/` 目录
**内容**：`task_info.json`中`data`字段列出的所有文件

**示例结构**：
```
tasks/ProteinBio_001/
├── data/
│   ├── protein_sequences_sample.csv
│   ├── pretrained_embeddings.json
│   └── protein_features.csv
```

**访问方式**：
- 在实验代码中通过相对路径访问：`data/filename.csv`
- 通过软链接链接到每个实验workspace

**代码位置**：`internagent/experiments_utils_claude.py:494-500`

#### `related_work/` 目录
**内容**：相关论文PDF（可选，仅人工参考）

**作用**：
- ❌ 不被代码自动读取
- ✅ 供用户或代码生成Agent人工参考

**访问方式**：通过相对路径访问：`related_work/paper.pdf`

---

## ⚙️ 二、配置文件输入

### 2.1 主配置文件

#### `config/default_config.yaml`（必需）
**位置**：项目根目录

**作用**：全局配置所有Agent和工具的行为

**关键配置项**：

**1. 模型提供商配置**：
```yaml
models:
  default_provider: "openai"
  openai:
    model_name: "deepseek-v4-pro"
    api_key: ""  # 通过环境变量OPENAI_API_KEY设置
    temperature: 0.7
```

**2. Agent配置**：
```yaml
agents:
  generation:
    model_provider: "default"
    generation_count: 15
    creativity: 0.7
    do_survey: true  # ← 启用文献调研
  
  survey:
    model_provider: "default"
    max_papers: 50
    sources: ["arxiv", "crossref", "web_search"]
```

**3. 科学任务配置**：
```yaml
sci_task:
  scorer_model: "deepseek-v4-pro"  # LLM-as-judge模型
  evaluation_mode: "llm_judge"
  default_launcher: "python code/experiment.py"
```

**4. 实验配置**：
```yaml
experiment:
  model: "glm-5"
  max_runs: 1  # 最大运行次数
  max_parallel_experiments: 1
  gpu_per_experiment: 1.0
```

**代码位置**：`launch_discovery.py:47-50`，加载配置文件

### 2.2 可选配置文件

#### `config/feedback_global.json`
**作用**：离线反馈，用于指导Idea Generation

**命令行参数**：`--offline_feedback`

**格式**：
```json
{
  "failed_ideas": [
    "避免使用过于简单的baseline方法"
  ],
  "successful_patterns": [
    "结合多任务学习通常效果好"
  ]
}
```

#### `config/mem_store/` 目录
**作用**：Task Memory存储历史实验结果

**配置**：
```yaml
memory:
  task_memory:
    enabled: true
    memory_dir: "./config/mem_store"
    top_k: 5
```

---

## 🔑 三、环境变量输入

### 3.1 LLM API密钥

#### 必需密钥（根据配置选择）

**OpenAI**：
```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_API_BASE_URL="https://api.openai.com/v1"  # 可选
```

**DeepSeek**：
```bash
export DEEPSEEK_API_KEY="sk-..."
export DS_API_BASE_URL="https://api.deepseek.com"  # 可选
```

**Anthropic**：
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

**代码位置**：
- `internagent/mas/models/openai_model.py:__init__()`
- `internagent/mas/models/r1_model.py:__init__()`

### 3.2 外部服务API密钥（可选）

#### 文献搜索服务

**Semantic Scholar**：
```bash
export S2_API_KEY="..."  # 可选，提高API限额
```

**配置位置**：`config/default_config.yaml`
```yaml
tools:
  literature_search:
    api_keys:
      semantic_scholar: ""  # 或通过S2_API_KEY环境变量
```

**Web搜索（如果启用）**：
```bash
export SERPER_API_KEY="..."  # Google搜索API
export GOOGLE_API_KEY="..."
export SEARCH_ENGINE_ID="..."
```

**代码位置**：`internagent/mas/tools/literature_search.py:__init__()`

### 3.3 系统配置

**GPU配置**：
```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"  # 指定可用GPU
```

**代码位置**：`internagent/stage.py:_check_gpu_availability()`

---

## 🖥️ 四、命令行参数输入

### 4.1 基本参数

#### `--task`
**默认值**：`"AutoSeg"`

**说明**：任务名称或路径

**用法**：
```bash
# 使用任务名称（从tasks/目录查找）
python launch_discovery.py --task ProteinBio_001

# 使用绝对路径
python launch_discovery.py --task /path/to/ProteinBio_001
```

#### `--config`
**默认值**：`'config/default_config.yaml'`

**说明**：配置文件路径

**用法**：
```bash
python launch_discovery.py --config config/custom_config.yaml
```

#### `--output_dir`
**默认值**：`None`（使用`results/{task_name}`）

**说明**：结果输出目录

**用法**：
```bash
python launch_discovery.py --output_dir custom_results/ProteinBio
```

### 4.2 实验执行参数

#### `--exp_backend`（必需）
**选项**：`"openhands" | "claudecode" | "iflow"`

**说明**：实验执行后端

**用法**：
```bash
python launch_discovery.py --exp_backend claudecode
```

#### `--mode`
**默认值**：`"experiment"`

**选项**：`"experiment" | "report"`

**说明**：
- `experiment`：执行完整实验流程
- `report`：仅生成报告

#### `--ref_code_path`
**默认值**：`None`（使用`{task_dir}/experiment.py`）

**说明**：baseline代码路径

### 4.3 创意生成参数

#### `--skip_idea_generation`
**类型**：布尔标志

**说明**：跳过创意生成，使用已有ideas

**配合参数**：`--idea_path`

**用法**：
```bash
python launch_discovery.py --skip_idea_generation --idea_path results/ProteinBio_001/20260603_095102_launch/ideas.json
```

#### `--offline_feedback`
**默认值**：`'config/feedback_global.json'`

**说明**：离线反馈文件路径

### 4.4 恢复参数

#### `--resume`
**默认值**：`None`

**说明**：从已有launch文件夹恢复

**用法**：
```bash
python launch_discovery.py --resume results/ProteinBio_001/20260603_095102_launch
```

**代码位置**：`launch_discovery.py:140-180`

---

## 🌐 五、外部API输入（运行时）

### 5.1 LLM API调用

#### 调用的Agent

| Agent | 用途 | 配置项 |
|-------|------|--------|
| **GenerationAgent** | 生成创意 | `agents.generation.model_provider` |
| **SurveyAgent** | 文献调研 | `agents.survey.model_provider` |
| **MethodDevelopmentAgent** | 方法开发 | `agents.method_development.model_provider` |
| **代码生成Agent** | 生成实验代码 | `experiment.model` |
| **评分Agent** | LLM-as-judge | `sci_task.scorer_model` |

#### API调用流程
```
launch_discovery.py启动
    ↓
加载config/default_config.yaml
    ↓
根据model_provider初始化模型
    ↓
各Agent调用LLM API生成内容
    ↓
API调用记录在日志中
```

**代码位置**：
- `internagent/mas/models/`（模型实现）
- `internagent/mas/agents/`（Agent实现）

### 5.2 文献搜索API（可选）

#### 调用条件
- `do_survey: true`（默认启用）
- GenerationAgent执行时触发

#### 数据源

**1. arXiv API**：
- URL：`https://export.arxiv.org/api/query`
- 无需API密钥
- 返回论文摘要和元数据

**2. CrossRef API**：
- URL：`https://api.crossref.org/works`
- 无需API密钥（有限额）
- 返回论文元数据

**3. Semantic Scholar API**：
- URL：`https://api.semanticscholar.org/graph/v1/paper/search`
- 可选API密钥（提高限额）
- 返回论文全文/摘要

**4. Web Search（如果配置）**：
- 需要`SERPER_API_KEY`或`GOOGLE_API_KEY`

#### 返回数据格式
```json
{
  "title": "Protein Language Models...",
  "abstract": "We present...",
  "authors": ["..."],
  "year": 2022,
  "content": "Full text content...",  // 某些数据源
  "source": "arxiv"
}
```

**代码位置**：
- `internagent/mas/tools/literature_search.py`
- `internagent/mas/agents/survey_agent.py`

### 5.3 评分API（可选）

#### 调用条件
- `evaluation_mode: "llm_judge"`（默认）
- 实验完成后自动调用

#### 评分流程
```
实验完成
    ↓
读取report.md
    ↓
读取checklist.json
    ↓
调用sci_task.scorer_model
    ↓
LLM根据checklist评分
    ↓
生成评分报告
```

**代码位置**：`internagent/sci_eval.py:score_run()`

---

## 📊 六、完整输入依赖图

```
用户准备文件
├── tasks/{domain}_XXX/
│   ├── task_info.json              ← 必需
│   ├── target_study/
│   │   ├── checklist.json          ← 强烈推荐
│   │   ├── paper.pdf               ← 可选（存档）
│   ├── data/                       ← 根据task_info.json定义
│   │   ├── *.csv
│   │   ├── *.json
│   │   └── ...
│   └── related_work/               ← 可选（人工参考）
│       └── *.pdf

配置文件
├── config/
│   ├── default_config.yaml         ← 必需
│   ├── feedback_global.json        ← 可选
│   └── mem_store/                  ← 可选（自动管理）

环境变量
├── OPENAI_API_KEY                  ← 必需（如使用OpenAI）
├── DEEPSEEK_API_KEY                ← 必需（如使用DeepSeek）
├── S2_API_KEY                      ← 可选（文献搜索）
└── CUDA_VISIBLE_DEVICES            ← 可选（GPU配置）

命令行参数
├── --task ProteinBio_001           ← 必需
├── --exp_backend claudecode        ← 必需
├── --config config/...yaml         ← 可选
├── --output_dir results/...        ← 可选
└── --resume results/.../..._launch ← 可选

外部API（运行时）
├── LLM API                          ← 必需
├── arXiv API                        ← 可选（文献调研）
├── CrossRef API                     ← 可选（文献调研）
├── Semantic Scholar API             ← 可选（文献调研）
└── Web Search API                   ← 可选（文献调研）
```

---

## 🎯 七、最小化输入清单

### 如果您想快速启动，这是**最小必需输入**：

### 文件准备
1. ✅ 创建`tasks/YourTask_001/task_info.json`
2. ✅ 创建`tasks/YourTask_001/target_study/checklist.json`
3. ✅ 准备`tasks/YourTask_001/data/`中的数据文件

### 配置设置
4. ✅ 确认`config/default_config.yaml`存在
5. ✅ 设置API密钥环境变量（例如`export OPENAI_API_KEY="..."`）

### 启动命令
6. ✅ 运行：
```bash
export OPENAI_API_KEY="your-key-here"
python launch_discovery.py --task YourTask_001 --exp_backend claudecode
```

### 可选但推荐
- 📝 `target_study/paper.pdf`（存档）
- 📝 `related_work/`目录（参考材料）
- 📝 `config/feedback_global.json`（指导创意生成）

---

## 🔍 八、输入验证检查点

### 启动前检查

| 检查项 | 命令/方法 | 预期结果 |
|--------|----------|---------|
| **task_info.json存在** | `ls tasks/YourTask_001/task_info.json` | 文件存在 |
| **checklist.json存在** | `ls tasks/YourTask_001/target_study/checklist.json` | 文件存在 |
| **数据文件存在** | `ls tasks/YourTask_001/data/` | 包含定义的文件 |
| **API密钥设置** | `echo $OPENAI_API_KEY` | 显示密钥 |
| **配置文件存在** | `ls config/default_config.yaml` | 文件存在 |
| **GPU可用** | `nvidia-smi` | 显示GPU信息 |
| **Python环境** | `python --version` | Python 3.8+ |

### 启动时检查

启动后，日志会显示：
```
[INFO] Loading config from config/default_config.yaml
[INFO] Task directory: tasks/YourTask_001
[INFO] Loading task_info.json... ✓
[INFO] Loading checklist.json... ✓
[INFO] Found 3 data files
[INFO] Model provider: openai
[INFO] GPU check: 4 GPUs available
```

如果任何输入缺失，系统会报错。

---

## 📝 九、常见问题

### Q1：没有paper.pdf能运行吗？
**A**：✅ 可以。`paper.pdf`只是存档，不被代码读取。

### Q2：checklist.json是必需的吗？
**A**：❌ 不是必需的，但强烈推荐。没有checklist无法评分。

### Q3：可以不提供data/目录吗？
**A**：❌ 不能。`task_info.json`中列出的数据文件必须存在，否则实验代码无法运行。

### Q4：可以使用其他LLM吗？
**A**：✅ 可以。在`config/default_config.yaml`中配置不同的`model_provider`。

### Q5：API密钥必须通过环境变量设置吗？
**A**：❌ 不是。也可以直接写在`config/default_config.yaml`中（但不推荐，存在安全风险）。

---

*文档生成时间：2024-06-09*  
*适用版本：InternAgent 1.5*
