# SurveyAgent论文数据存储分析

## 🎯 核心答案

**SurveyAgent搜索到的相关论文主要存储在：**

### 主要存储位置

**文件路径**：`results/{task_name}/traj_{session_id}.json`

**字段位置**：`session.task.background` → References部分

---

## 📊 完整数据流程

### 1️⃣ SurveyAgent执行阶段

#### 调用入口
**文件**：`internagent/mas/workflow/orchestration_agent.py`

```python
# 第1步：检查是否启用文献调研
if generation_agent.config.get("do_survey", False):
    logger.info(f"Survey Agent: Conduct in-depth literature research on task {session.id}")
    survey_agent = self._get_agent("survey")
    
    # 第2步：SurveyAgent搜索外部数据库
    survey_results = await survey_agent.execute(session.task.to_dict(), {})
    
    # 第3步：提取论文列表
    paper_lst = survey_results.get("papers", [])
    web_results = survey_results.get("web_results", [])
```

#### SurveyAgent返回的数据结构
**文件**：`internagent/mas/agents/survey_agent.py`

```python
async def execute(self, context, params) -> Dict[str, Any]:
    results = {
        "papers": [],      # ← 学术论文列表
        "web_results": []  # ← 网页搜索结果
    }
    
    # 搜索外部数据库
    papers, _ = await self.advanced_query_paper(context=context)
    results["papers"] = papers
    
    return results
```

**papers数组格式**：
```json
[
  {
    "title": "ProtTrans: Toward Understanding the Language of Life",
    "abstract": "We present a large-scale study...",
    "authors": ["..."],
    "year": 2021,
    "content": "Full text content...",
    "source": "arxiv",
    "url": "https://...",
    "citationCount": 150
  }
]
```

### 2️⃣ GenerationAgent使用阶段

#### paper_lst被传递给GenerationAgent
**文件**：`internagent/mas/workflow/orchestration_agent.py`

```python
# 第4步：传递给GenerationAgent
context = {
    "goal": session.task.to_dict(),
    "iteration": session.iterations_completed,
    "feedback": session.feedback_history,
    "paper_lst": paper_lst,  # ← 传递论文列表
    "task_name": getattr(self, 'task_name', None) or session.task.domain
}

# 第5步：GenerationAgent生成创意
response = await generation_agent.execute(context, {})
```

#### GenerationAgent使用paper_lst
**文件**：`internagent/mas/agents/generation_agent.py`

```python
# 第6步：构建包含论文信息的prompt
if self.do_survey and paper_lst:
    logger.info("Add literature information to prompt")
    literature_prompt = "# Literature Information\n"
    for paper in paper_lst:
        literature_prompt += f"- {paper['title']} ({paper['year']})\n {paper['abstract']} \n\n"
    prompt += literature_prompt

# 第7步：LLM基于论文信息生成创意
response = await self._call_model(prompt=prompt)
```

**注意**：此时paper_lst只在内存中临时存在。

### 3️⃣ 持久化存储阶段

#### Session保存机制
**文件**：`internagent/mas/memory/memory_manager.py:FileSystemMemoryManager`

```python
async def save_session(self, session_id: str, session_data: Dict[str, Any]) -> None:
    """保存session数据到文件系统"""
    
    # 添加更新时间戳
    session_data["update_time"] = time.time()
    
    # 保存到文件：results/{task_name}/traj_{session_id}.json
    task_dir = self._get_task_dir(session_data)
    file_path = os.path.join(task_dir, f"traj_{session_id}.json")
    await self._write_json_file(file_path, session_data)
```

#### 实际存储位置

**文件路径示例**：
```
results/ProteinBio_001/traj_session_1780481098.json
```

**数据结构**：
```json
{
  "id": "session_1780481098",
  "task": {
    "id": "task_proteinbio_001",
    "description": "复现ProtTrans论文的核心发现...",
    "domain": "ProteinBio",
    "background": "# Background for Reproducing ProtTrans...\n\n## Introduction\n\n...\n\n## References\n\n[[1]] 📄 Benchmarking secondary structure prediction - https://doi.org/...\n[[2]] 📄 AttSec: protein secondary structure prediction - https://doi.org/...\n...",
    "constraints": [],
    "ref_code_path": ""
  },
  "ideas": [...],
  "iterations_completed": 1,
  "state": "completed"
}
```

---

## 🔍 论文数据的三种存在形态

### 形态1：原始搜索结果（运行时临时）
**位置**：内存中的`paper_lst`变量

**生命周期**：
- 创建：SurveyAgent.execute()返回
- 使用：GenerationAgent.execute()中使用
- 销毁：GenerationAgent执行完成后

**特点**：
- ❌ 不持久化
- ❌ 不保存到文件
- ✅ 包含完整的论文元数据（title, abstract, authors, year, content）

### 形态2：Prompt中的论文信息（LLM输入）
**位置**：GenerationAgent的prompt字符串

**格式**：
```python
"# Literature Information\n"
"- ProtTrans: Toward Understanding the Language of Life (2021)\n"
" We present a large-scale study...\n\n"
"- Protein Secondary Structure Prediction (2022)\n"
" This paper proposes...\n\n"
```

**特点**：
- ❌ 临时构建，不保存
- ✅ 只包含title + year + abstract
- ✅ 用于指导LLM生成创意

### 形态3：Background中的References（持久化）
**位置**：`traj_{session_id}.json`中的`task.background`字段

**格式**：
```
## References

[[1]] 📄 Benchmarking secondary structure prediction for fold recognition - https://doi.org/10.1002/prot.10408

[[2]] 📄 AttSec: protein secondary structure prediction by capturing local patterns from attention map - https://doi.org/10.21203/rs.3.rs-2433490/v1

[[3]] 📄 ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning - https://doi.org/10.1109/tpami.2021.3095381
```

**特点**：
- ✅ **持久化保存**
- ✅ 存储在`traj_{session_id}.json`文件中
- ❌ 只包含title + url，没有abstract
- ✅ 可人工查看和引用

---

## 📁 实际文件示例

### traj_session_1780481098.json中的References部分

**文件**：`results/ProteinBio_001/traj_session_1780481098.json`

**Background字段包含**：
```json
{
  "task": {
    "background": "# Background for Reproducing ProtTrans...\n\n... (63880 字符的背景信息) ...\n\n## References\n\n[[1]] 📄 Benchmarking secondary structure prediction for fold recognition - https://doi.org/10.1002/prot.10408\n\n[[2]] 📄 AttSec: protein secondary structure prediction by capturing local patterns from attention map - https://doi.org/10.21203/rs.3.rs-2433490/v1\n\n[[3]] 📄 ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning - https://doi.org/10.1109/tpami.2021.3095381\n\n[[4]] 📄 ProtTrans: Towards Cracking the Language of Life's Code Through Self-Supervised Learning - https://www.biorxiv.org/content/biorxiv/early/2021/\n\n... (共约10-15篇论文引用)"
  }
}
```

---

## 🔍 为什么有两种存储格式？

### 原因分析

**1. 临时格式 vs. 持久格式**

| 格式 | 用途 | 位置 | 内容 |
|------|------|------|------|
| **paper_lst（临时）** | LLM生成创意时使用 | 内存中 | 完整元数据 + abstract |
| **References（持久）** | 记录文献来源 | 文件中 | Title + URL |

**2. 为什么References不包含abstract？**

- 💾 **节省存储空间**：Abstract可能很长（数千字符），10-15篇论文就会占据大量空间
- 📝 **人工可读性**：References格式更像学术引用，方便人类查看
- 🔗 **追溯性**：保留URL和DOI，允许用户追溯原始论文

**3. 论文数据的实际用途**

```
SurveyAgent搜索（获取完整数据）
    ↓
GenerationAgent使用（临时，包含abstract）
    ↓
生成创意
    ↓
保存到Background（持久，仅title+url）
```

---

## 🎯 检查论文数据存储的方法

### 方法1：查看traj文件
```bash
# 查看最新的traj文件
ls -lt results/ProteinBio_001/traj_*.json | head -1

# 提取References部分
cat results/ProteinBio_001/traj_session_*.json | jq '.task.background' | grep -A 20 "## References"
```

### 方法2：查看Background字段
```python
import json

with open('results/ProteinBio_001/traj_session_1780481098.json') as f:
    data = json.load(f)
    
background = data['task']['background']

# 查找References部分
lines = background.split('\n')
ref_idx = next((i for i, line in enumerate(lines) if '## References' in line), None)

if ref_idx:
    for line in lines[ref_idx:ref_idx+20]:
        print(line)
```

### 方法3：查看Session复制文件
```bash
# Session目录中也保存了traj.json
ls results/ProteinBio_001/20260603_095102_launch/session_*/traj.json

# 内容与原始traj文件相同
```

---

## 📊 数据流程图

```
外部数据库（arXiv, Semantic Scholar等）
    ↓
SurveyAgent搜索
    ↓
paper_lst = [
  {title, abstract, authors, year, content, ...}
]
    ↓
GenerationAgent使用（临时）
    ├─→ Prompt构建：包含title + abstract
    └─→ LLM生成创意
    ↓
创意生成完成
    ↓
Session保存到traj_{session_id}.json
    ↓
Background字段包含：
    ## References
    [[1]] 📄 title - url
    [[2]] 📄 title - url
    ...
```

---

## 🔑 关键要点

1. **SurveyAgent搜索的论文数据不会完整持久化**
   - ❌ 原始`paper_lst`（包含abstract）不会被保存
   - ✅ 只有title + URL被保存到background的References部分

2. **论文数据的三种生命周期**
   - 运行时：`paper_lst`（完整数据，临时）
   - Prompt中：Literature Information（title + abstract，临时）
   - 文件中：References（title + URL，持久）

3. **主要存储位置**
   - **文件**：`results/{task_name}/traj_{session_id}.json`
   - **字段**：`session.task.background`
   - **部分**：References章节

4. **如何查看存储的论文**
   - 直接打开`traj_{session_id}.json`文件
   - 查看`task.background`字段
   - 搜索"## References"部分

---

*分析完成时间：2024-06-09*  
*适用版本：InternAgent 1.5*
