# 修正分析：sci_tasks中的文献调研功能

## 🎯 核心发现

### 您是对的！sci_tasks确实有文献调研功能

**配置证据**：
```yaml
# config/default_config.yaml
agents:
  generation:
    do_survey: true  # ← 文献调研是启用的！

  survey:
    model_provider: "default"
    max_papers: 50
    sources: ["arxiv", "crossref", "web_search"]
```

## 📊 sci_tasks中的两种"论文"

### 区别：目标论文 vs. 相关论文

| 类型 | 位置 | 是否被读取 | 用途 |
|------|------|-----------|------|
| **目标论文** | `target_study/paper.pdf` | ❌ **不被读取** | 存档和人工参考 |
| **相关论文** | 外部数据库搜索 | ✅ **被读取和分析** | 提供背景和方法参考 |

### 具体流程

```
sci_tasks启动
    ↓
normalize_sci_task() 生成task_description
    "Reproduce the findings from a scientific paper in the ProteinBio domain.
     ## Research Task
     复现ProtTrans论文的核心发现..."
    ↓
SurveyAgent接收task_description
    ↓
基于task_description生成搜索查询
    "ProteinBERT", "protein language model", "secondary structure prediction"
    ↓
搜索外部数据库（arXiv, Semantic Scholar, CrossRef）
    ↓
获取相关论文列表（title, abstract, content）
    ↓
GenerationAgent使用这些相关论文作为背景
```

## 🔍 关键代码分析

### 1. SurveyAgent的工作流程

**文件**：`internagent/mas/workflow/orchestration_agent.py`

```python
# GenerationAgent的配置
paper_lst = None
if generation_agent.config.get("do_survey", False):
    logger.info(f"Survey Agent: Conduct in-depth literature research on task {session.id}")
    survey_agent = self._get_agent("survey")
    if survey_agent:
        # ← 传递session.task.to_dict()，包含task_description
        survey_results = await survey_agent.execute(session.task.to_dict(), {})
        paper_lst = survey_results.get("papers", [])  # ← 获取相关论文列表
```

### 2. SurveyAgent如何生成查询

**文件**：`internagent/mas/agents/survey_agent.py`

```python
async def advanced_query_paper(self, context) -> Tuple[List[Dict[str, Any]], List[str]]:
    # 从context中提取任务描述
    goal_description = context.get("description", {})
    domain = context.get("domain", "")

    # 基于goal_description生成搜索查询
    init_keyword_query_prompt = (
        f"You are a researcher doing literature review on {goal_description}. "
        f"Propose keywords for Semantic Scholar. "
        f"Return KeywordQuery('...') only."
    )
    
    # 生成初始查询（例如："protein language model"）
    response = await self._call_model(prompt=init_keyword_query_prompt)
    init_query = response
    
    # 搜索外部数据库
    init_paper_lst = await self.literature_search_query(init_query, 10)
```

### 3. LiteratureSearch搜索外部数据库

**文件**：`internagent/mas/tools/literature_search.py`

```python
async def multi_source_search(self, query: str, sources: List[str], max_results: int):
    """从多个外部源搜索学术论文"""
    all_results = {}
    
    for source in sources:
        if source == "arxiv":
            papers = await self._search_arxiv(query, max_results)
        elif source == "crossref":
            papers = await self._search_crossref(query, max_results)
        elif source == "semantic_scholar":
            papers = await self._search_semantic_scholar(query, max_results)
        
        all_results[source] = papers
    
    return all_results
```

**关键点**：
- ❌ **不读取**target_study/paper.pdf
- ✅ **搜索**外部数据库（arXiv, Semantic Scholar等）
- ✅ **基于**task_description生成搜索关键词

## 📝 实际运行示例

### ProtTrans论文复现的文献调研

**输入到SurveyAgent**：
```json
{
  "description": "Reproduce the findings from a scientific paper in the ProteinBio domain.\n\n## Research Task\n复现ProtTrans论文的核心发现：使用蛋白质语言模型进行二级结构预测...",
  "domain": "ProteinBio"
}
```

**SurveyAgent生成的查询**：
```
1. "protein language model"
2. "ProtBERT"
3. "secondary structure prediction"
4. "protein sequence embedding"
```

**从外部数据库获取的相关论文**：
```json
[
  {
    "title": "ProtBERT: Pretrained Protein Language Model",
    "abstract": "We present ProtBERT...",
    "content": "Full text content...",
    "source": "arxiv"
  },
  {
    "title": "Protein Secondary Structure Prediction using Deep Learning",
    "abstract": "This paper proposes...",
    "content": "Full text content...",
    "source": "semantic_scholar"
  }
]
```

**传递给GenerationAgent**：
```python
context = {
    "goal": session.task.to_dict(),
    "paper_lst": paper_lst,  # ← 相关论文列表（来自外部搜索）
    "task_name": "ProteinBio"
}

# GenerationAgent基于这些相关论文生成创意
```

## 🎯 修正后的结论

### sci_tasks中的论文处理

| 功能 | 是否读取PDF | 具体说明 |
|------|------------|---------|
| **目标论文处理** | ❌ **不读取** | target_study/paper.pdf不被任何代码读取 |
| **相关论文搜索** | ✅ **读取** | SurveyAgent搜索并读取外部相关论文的全文/摘要 |
| **文献调研** | ✅ **执行** | 基于task_description搜索外部数据库 |

### 为什么有文献调研但不读目标论文？

这是一个**设计选择**：

**目标论文（target_study/paper.pdf）**：
- 作为"复现目标"的存档
- 用户手动提供task_info.json作为结构化描述
- 不需要代码"理解"论文内容

**相关论文（外部搜索）**：
- 提供**背景信息**（这个领域有哪些相关工作）
- 提供**方法参考**（别人用了什么方法）
- 帮助GenerationAgent生成**更合理的实施策略**

### 类比理解

```
目标论文：
- 就像"考试题目"
- 存在试卷上，但不需要系统"读懂"它
- 用户提供task_info.json告诉系统"要考什么"

相关论文：
- 就像"参考资料"
- 系统搜索相关书籍、论文
- 帮助系统理解背景和生成答案
```

## 📊 完整的论文处理流程

```
用户准备：
1. 下载论文PDF → target_study/paper.pdf
2. 手动总结 → task_info.json
3. 定义评分标准 → checklist.json

系统运行：
1. normalize_sci_task() 读取 task_info.json + checklist.json
   ↓
2. SurveyAgent 基于 task_description 搜索外部相关论文
   ↓
3. GenerationAgent 使用相关论文生成实施策略
   ↓
4. MethodDevelopmentAgent 转换为具体方法
   ↓
5. 代码生成Agent 实现代码
   ↓
6. 评分系统读取 report.md + checklist.json
```

**关键点**：
- ✅ 第2步：相关论文被搜索和读取
- ❌ target_study/paper.pdf始终不被读取

## 💡 为什么这样设计？

### 优点

1. **分离关注点**：
   - 目标论文：定义"要做什么"
   - 相关论文：提供"怎么做的参考"

2. **提高效率**：
   - 不需要解析目标论文PDF
   - 直接使用用户提供的结构化描述

3. **扩大背景**：
   - 搜索外部论文获得更广的视野
   - 不局限于目标论文的方法

### 可能的改进

如果要真正利用目标论文：

```python
def normalize_sci_task_with_pdf_extraction(task_dir, output_path):
    # 1. 读取task_info.json
    task_info = json.load(open("task_info.json"))
    
    # 2. 读取并提取paper.pdf
    paper_path = osp.join(task_dir, "target_study", "paper.pdf")
    paper_text = extract_text_from_pdf(paper_path)
    
    # 3. LLM提取论文方法
    extracted_method = llm_extract_method(paper_text)
    
    # 4. 结合提取的方法
    prompt_data = {
        "task_description": task_info['task'],
        "paper_method": extracted_method,  # ← 从PDF提取
        "constraints": constraints
    }
```

**当前InternAgent没有这样做**。

---

## 总结

### 回答您的问题

**"sci_tasks流程中也有文献调研啊，跟论文没有关系吗？"**

**答案**：
1. ✅ **有关系，但不是您想的那种关系**
2. ✅ **文献调研确实存在**（do_survey: true）
3. ❌ **但不读取target_study/paper.pdf**
4. ✅ **而是搜索外部相关论文**

### 两种论文的清晰区别

```
target_study/paper.pdf
    ↓
❌ 不被读取
    ↓
作用：存档 + 人工参考


task_description（来自task_info.json）
    ↓
✅ 用来生成搜索查询
    ↓
外部数据库（arXiv, Semantic Scholar等）
    ↓
✅ 搜索到的相关论文被读取和分析
    ↓
作用：提供背景和方法参考
```

感谢您的质疑！这让我发现了之前分析中的不完整之处。

---

*修正分析完成时间：2024-06-08*
*关键修正：sci_tasks确实有文献调研，但搜索的是外部相关论文，不读取target_study/paper.pdf*
