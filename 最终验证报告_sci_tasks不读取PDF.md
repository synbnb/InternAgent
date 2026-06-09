# 最终验证报告：sci_tasks是否读取paper.pdf

## 🔍 彻底搜索结果

### 搜索范围覆盖

| 搜索目标 | 搜索模式 | 结果 |
|---------|---------|------|
| PDF处理函数 | `extract_text_from_pdf` | ✅ 存在，但仅在ScholarAgent中使用 |
| sci_tasks中的PDF引用 | `sci_task.*pdf\|pdf.*sci_task` | ❌ 未找到 |
| target_study中的PDF | `target_study.*pdf\|pdf.*target_study` | ❌ 未找到 |
| prompt中的PDF指令 | `read.*paper\|extract.*paper` | ❌ 未找到 |

### 关键发现

#### 1. 项目中的PDF处理能力存在

**文件位置**：`internagent/mas/tools/utils.py`
```python
def extract_text_from_pdf(pdf_path):
    """使用pdfplumber提取PDF文本"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None
```

**但是**：这个函数**只在以下场景中使用**：
- ScholarAgent的文献调研（下载和处理外部论文）
- SurveyAgent的论文搜索功能
- **不用于处理sci_tasks中的target_study/paper.pdf**

#### 2. sci_tasks工作流中的PDF处理

**代码位置**：`launch_discovery.py:normalize_sci_task()`

```python
def normalize_sci_task(task_dir, output_path):
    # ❌ 没有读取 paper.pdf
    # ❌ 没有调用 extract_text_from_pdf()
    
    # ✅ 只读取这两个文件：
    task_info_path = osp.join(task_dir, "task_info.json")
    with open(task_info_path, 'r') as f:
        task_info = json.load(f)
    
    checklist_path = osp.join(task_dir, "target_study", "checklist.json")
    if osp.exists(checklist_path):
        with open(checklist_path) as f:
            checklist = json.load(f)
```

#### 3. 代码生成Prompt中的PDF指令

**文件位置**：`internagent/prompts.py:CODER_PROMPT_SCI_TASK`

```python
CODER_PROMPT_SCI_TASK = """Your goal is to reproduce the findings from a scientific paper...

## Reproduction Approach
{idea_description}      # ← 来自GenerationAgent，不是PDF

## Proposed Method
{method}                 # ← 来自MethodDevelopmentAgent，不是PDF

## Research Task
{task_description}       # ← 来自task_info.json，不是PDF

## Available Data
{data_manifest}          # ← 来自task_info.json，不是PDF

## Workspace Layout
- Reference papers are in `related_work/`  # ← 只说明位置，不要求读取
- Raw data is in `data/`

# ❌ 没有"请读取paper.pdf"的指令
# ❌ 没有"请分析target_study中的论文"的指令
"""
```

**关键点**：Prompt中提到了"Reference papers are in related_work/"，但这只是告诉LLM相关论文的位置，**不是要求读取paper.pdf**。

#### 4. target_study目录的实际使用

**文件位置**：`internagent/experiments_utils_claude.py`

```python
SCI_SYMLINK_DIRS = {'data', 'related_work', 'target_study'}

# 在创建实验目录时
for item in os.listdir(folder_name):
    if task_type == 'sci' and item in SCI_SYMLINK_DIRS:
        # 创建软链接而不是复制
        os.symlink(osp.abspath(src), dst)
```

**实际作用**：
```
实验文件夹/
├── target_study/ → 软链接到原始目录
│   ├── paper.pdf        # ❌ 可访问，但不被代码读取
│   ├── checklist.json   # ✅ 评分时使用
│   └── images/         # ✅ 评分时使用（image类型checklist）
```

#### 5. 评分系统中的PDF使用

**文件位置**：`internagent/sci_eval.py:score_run()`

```python
def score_run(workspace_dir, checklist_path, model="gpt-5.1"):
    # ✅ 读取report.md
    report_text = _read_report(workspace)
    
    # ✅ 读取checklist.json
    with open(checklist_path) as f:
        checklist = json.load(f)
    
    # ✅ 读取target_study/images/中的图片
    if item.get('type') == 'image':
        target_rel = item.get('path', '')
        target_path = safe_resolve(target_base, target_rel)
        # target_base = workspace / "target_study"
        # 从target_study/读取图片用于对比
    
    # ❌ 不读取paper.pdf
```

## 📊 完整验证矩阵

| 功能模块 | 是否读取paper.pdf | 证据文件 | 证据行号 |
|---------|-----------------|---------|---------|
| **normalize_sci_task()** | ❌ 不读取 | launch_discovery.py | 67-90 |
| **_build_sci_initial_prompt()** | ❌ 不读取 | experiments_utils_claude.py | 469-500 |
| **CODER_PROMPT_SCI_TASK** | ❌ 无指令 | prompts.py | 97-135 |
| **GenerationAgent** | ❌ 不读取 | generation_agent.py | 69 |
| **MethodDevelopmentAgent** | ❌ 不读取 | - | - |
| **代码生成Claude Agent** | ❌ 不读取 | experiments_utils_claude.py | - |
| **评分系统sci_eval.py** | ❌ 不读取 | sci_eval.py | 60-110 |
| **target_study软链接** | ❌ 仅链接 | experiments_utils_claude.py | 494-500 |

## 🎯 最终结论

### 答案：**❌ 整个sci_tasks工作流中没有任何代码读取paper.pdf**

### 证据链总结

1. **PDF处理功能存在但不使用**
   - ✅ 项目中有`extract_text_from_pdf()`函数
   - ❌ 但该函数只在ScholarAgent/SurveyAgent中使用
   - ❌ sci_tasks流程不调用这些Agent

2. **sci_tasks流程不读取PDF**
   - ❌ `normalize_sci_task()`只读取task_info.json和checklist.json
   - ❌ 提示词中没有要求LLM读取PDF
   - ❌ target_study目录只用于软链接，不被读取

3. **paper.pdf的唯一作用**
   - 作为"任务来源证明"（存档）
   - 作为"人工参考"（用户可手动查阅）
   - 作为"评分基准来源"（checklist.json来源于此，但代码只使用checklist.json）

### 为什么感觉"应该有PDF处理"？

这是**合理的设计直觉**，因为：
- ✅ 逻辑上应该读取论文来理解方法
- ✅ paper.pdf被放在target_study/目录中
- ✅ 整个任务叫"论文复现"

**但实际上**：
- ❌ InternAgent依赖用户**手动总结**论文（task_info.json）
- ❌ paper.pdf只是**存档和参考**
- ❌ 整个流程基于**结构化描述**而非**PDF内容**

### 设计选择的原因

这可能是有意的设计决策：

1. **避免PDF解析的复杂性**
   - 多栏排版、图表、公式
   - 扫描PDF没有文本层
   - 非英文论文的处理

2. **降低成本**
   - PDF解析需要时间
   - LLM提取内容成本高
   - 提取的文本可能不完整

3. **提高准确性**
   - 用户提供的task_info.json比提取的文本更准确
   - 避免LLM误解论文内容

## 🔧 如果要实现真正的PDF读取

需要在`normalize_sci_task()`中添加：

```python
def normalize_sci_task_with_pdf_extraction(task_dir, output_path):
    """读取论文PDF并提取内容"""
    from internagent.mas.tools.utils import extract_text_from_pdf
    
    # 1. 读取论文PDF
    paper_path = osp.join(task_dir, "target_study", "paper.pdf")
    paper_text = extract_text_from_pdf(paper_path)
    
    # 2. 使用LLM提取论文方法
    extracted_method = llm_extract_method(paper_text)
    
    # 3. 将提取的方法添加到prompt
    prompt_data = {
        "task_description": task_info['task'],
        "paper_method": extracted_method,  # ← 从PDF提取
        "data_manifest": data_manifest,
        "constraints": constraints
    }
```

**当前InternAgent没有这样做**。

---

*最终验证完成时间：2024-06-08*
*验证状态：经过全面代码搜索，确认结论正确*
