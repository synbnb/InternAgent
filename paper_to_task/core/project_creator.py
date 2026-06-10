"""
项目创建模块 - 创建sci_tasks项目结构
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..utils.file_utils import (
    ensure_directory, write_json_file, read_json_file,
    copy_file_safe, create_readme, find_available_task_name
)


class ProjectCreator:
    """项目创建器"""

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化项目创建器

        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.base_dir = Path(self.config.get('sci_tasks_base', 'sci_tasks/tasks'))

    def create_project(self,
                       task_name: str,
                       task_info: Dict,
                       checklist: List[Dict],
                       pdf_path: Optional[str] = None,
                       domain: str = "Science") -> Dict[str, Any]:
        """
        创建完整的项目结构

        Args:
            task_name: 任务名称
            task_info: 任务信息
            checklist: 检查清单
            pdf_path: PDF文件路径（可选）
            domain: 领域名称

        Returns:
            创建结果
        """
        # 查找可用的任务ID
        task_id = find_available_task_name(self.base_dir, domain)
        task_dir = self.base_dir / task_id

        try:
            # 创建目录结构
            self._create_directory_structure(task_dir)

            # 创建文件
            self._create_files(task_dir, task_info, checklist, task_name)

            # 复制PDF文件
            if pdf_path:
                self._copy_reference_paper(task_dir, pdf_path)

            # 生成说明文档
            self._generate_documentation(task_dir, task_info, task_name)

            return {
                'success': True,
                'task_id': task_id,
                'task_path': str(task_dir),
                'message': f'成功创建任务: {task_id}',
                'next_steps': self._generate_next_steps(task_dir, task_id)
            }

        except Exception as e:
            # 清理已创建的内容
            if task_dir.exists():
                shutil.rmtree(task_dir)

            return {
                'success': False,
                'error': str(e),
                'message': f'创建任务失败: {e}'
            }

    def _create_directory_structure(self, task_dir: Path) -> None:
        """创建目录结构"""
        directories = [
            task_dir,
            task_dir / "data",
            task_dir / "target_study",
            task_dir / "target_study" / "images",
            task_dir / "target_study" / "paper"
        ]

        for directory in directories:
            ensure_directory(directory)

    def _create_files(self, task_dir: Path,
                     task_info: Dict,
                     checklist: List[Dict],
                     task_name: str) -> None:
        """创建项目文件"""
        # task_info.json
        task_info_path = task_dir / "task_info.json"
        write_json_file(task_info_path, task_info, indent=2)

        # checklist.json
        checklist_path = task_dir / "target_study" / "checklist.json"
        write_json_file(checklist_path, checklist, indent=2)

        # DATA_README.md
        data_readme = self._generate_data_readme(task_info)
        create_readme(task_dir / "DATA_README.md", data_readme)

        # TASK_STATUS.md
        status_content = self._generate_status_content(task_name)
        create_readme(task_dir / "TASK_STATUS.md", status_content)

    def _copy_reference_paper(self, task_dir: Path, pdf_path: str) -> bool:
        """复制参考论文"""
        try:
            pdf_source = Path(pdf_path)
            if not pdf_source.exists():
                return False

            pdf_dest = task_dir / "target_study" / "paper" / pdf_source.name
            return copy_file_safe(pdf_source, pdf_dest)

        except Exception as e:
            print(f"复制PDF文件失败: {e}")
            return False

    def _generate_documentation(self, task_dir: Path,
                               task_info: Dict,
                               task_name: str) -> None:
        """生成文档"""
        # 生成README.md
        readme_content = self._generate_readme(task_dir.name, task_info, task_name)
        create_readme(task_dir / "README.md", readme_content)

        # 生成QUICK_START.md
        quick_start_content = self._generate_quick_start(task_dir.name)
        create_readme(task_dir / "QUICK_START.md", quick_start_content)

    def _generate_data_readme(self, task_info: Dict) -> str:
        """生成数据说明文件"""
        readme = "# 数据文件说明\n\n"

        data_items = task_info.get("data", [])
        if data_items:
            readme += "## 可用数据文件\n\n"
            for item in data_items:
                readme += f"### {item['name']}\n"
                readme += f"- **路径**: `{item['path']}`\n"
                readme += f"- **描述**: {item['description']}\n\n"
        else:
            readme += "## 数据文件\n\n"
            readme += "请根据论文描述准备相应的数据文件。\n\n"

        readme += "## 数据准备指南\n\n"
        readme += "1. 从论文补充材料或相关网站下载数据\n"
        readme += "2. 将数据文件放置在 `data/` 目录下\n"
        readme += "3. 更新 `task_info.json` 中的文件路径\n\n"

        return readme

    def _generate_status_content(self, task_name: str) -> str:
        """生成状态文件内容"""
        return f"""# Task Status

**任务名称**: {task_name}
**创建时间**: {self._get_current_time()}
**状态**: 初始化完成

## 下一步操作

1. 准备数据文件
2. 运行实验: `python launch_discovery.py --task sci_tasks/tasks/<this_task_id>`
3. 查看结果和分析

## 任务进度

- [ ] 数据准备
- [ ] 环境配置
- [ ] 代码实现
- [ ] 实验运行
- [ ] 结果分析
- [ ] 报告撰写
"""

    def _generate_readme(self, task_id: str,
                        task_info: Dict,
                        task_name: str) -> str:
        """生成README文件"""
        return f"""# {task_id}

## 任务描述

{task_info.get('task', '复现论文核心发现')}

## 研究背景

{task_info.get('background', '研究背景待补充')}

## 研究目标

{task_info.get('research_goal', '验证论文核心发现')}

## 数据文件

请查看 `DATA_README.md` 获取详细的数据文件说明。

## 评分标准

评分标准定义在 `target_study/checklist.json` 中。

## 快速开始

```bash
# 启动实验
python launch_discovery.py --task sci_tasks/tasks/{task_id}

# 查看结果
# 结果将保存在 results/{task_id}/ 目录下
```

## 项目结构

```
{task_id}/
├── task_info.json              # 任务信息
├── DATA_README.md             # 数据说明
├── TASK_STATUS.md             # 任务状态
├── data/                      # 数据文件目录
└── target_study/              # 参考研究
    ├── checklist.json         # 评分标准
    ├── images/                # 参考图表
    └── paper/                 # 参考论文
```

## 注意事项

1. 确保数据文件已正确放置
2. 检查环境配置
3. 按照评分标准完成任务

---

*由 Paper-to-Task 系统自动生成*
"""

    def _generate_quick_start(self, task_id: str) -> str:
        """生成快速开始指南"""
        return f"""# 快速开始指南

## 1. 数据准备

```bash
# 创建数据目录（如果不存在）
mkdir -p sci_tasks/tasks/{task_id}/data

# 下载数据文件并放置到data目录
# 根据DATA_README.md的说明准备数据
```

## 2. 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
# 根据需要修改配置文件
```

## 3. 运行实验

```bash
# 基本运行（使用完整路径）
python launch_discovery.py --task sci_tasks/tasks/{task_id}

# 指定输出目录
python launch_discovery.py --task sci_tasks/tasks/{task_id} --output results/

# 查看详细日志
python launch_discovery.py --task sci_tasks/tasks/{task_id} --verbose
```

## 4. 查看结果

```bash
# 结果目录
ls results/{task_id}/

# 查看报告
cat results/{task_id}/*_launch/session_*/report.md
```

## 5. 评分和评估

系统将自动根据 `checklist.json` 中的标准对结果进行评分。

## 常见问题

**Q: 数据文件在哪里下载？**
A: 请查看论文的补充材料或官方网站。

**Q: 如何修改任务配置？**
A: 编辑 `task_info.json` 文件。

**Q: 如何查看详细日志？**
A: 使用 `--verbose` 参数或查看 `results/` 目录下的日志文件。

---

如有其他问题，请参考主项目README。
"""

    def _generate_next_steps(self, task_dir: Path, task_id: str) -> List[str]:
        """生成下一步操作指导"""
        return [
            f"1. 准备数据文件到 {task_dir}/data/ 目录",
            f"2. 查看并理解 {task_dir}/task_info.json 中的任务描述",
            f"3. 运行: python launch_discovery.py --task sci_tasks/tasks/{task_id}",
            f"4. 查看结果: results/{task_id}/ 目录",
            "5. 根据checklist.json标准进行自评"
        ]

    def _get_current_time(self) -> str:
        """获取当前时间"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def validate_project(self, task_dir: Path) -> Dict[str, Any]:
        """
        验证项目结构

        Args:
            task_dir: 任务目录

        Returns:
            验证结果
        """
        required_files = [
            task_dir / "task_info.json",
            task_dir / "target_study" / "checklist.json",
            task_dir / "DATA_README.md"
        ]

        required_dirs = [
            task_dir / "data",
            task_dir / "target_study",
            task_dir / "target_study" / "images"
        ]

        missing_files = [f for f in required_files if not f.exists()]
        missing_dirs = [d for d in required_dirs if not d.exists()]

        return {
            'valid': len(missing_files) == 0 and len(missing_dirs) == 0,
            'missing_files': [str(f) for f in missing_files],
            'missing_dirs': [str(d) for d in missing_dirs]
        }

    def update_task_info(self, task_dir: Path, updates: Dict) -> bool:
        """
        更新任务信息

        Args:
            task_dir: 任务目录
            updates: 更新内容

        Returns:
            是否成功
        """
        try:
            task_info_path = task_dir / "task_info.json"

            # 读取现有内容
            task_info = read_json_file(task_info_path)
            if not task_info:
                return False

            # 更新内容
            task_info.update(updates)

            # 写回文件
            return write_json_file(task_info_path, task_info, indent=2)

        except Exception as e:
            print(f"更新task_info失败: {e}")
            return False

    def get_project_summary(self, task_dir: Path) -> Optional[Dict[str, Any]]:
        """
        获取项目摘要

        Args:
            task_dir: 任务目录

        Returns:
            项目摘要
        """
        try:
            task_info_path = task_dir / "task_info.json"
            checklist_path = task_dir / "target_study" / "checklist.json"

            task_info = read_json_file(task_info_path)
            checklist = read_json_file(checklist_path)

            if not task_info or not checklist:
                return None

            return {
                'task_id': task_dir.name,
                'task': task_info.get('task', ''),
                'research_goal': task_info.get('research_goal', ''),
                'data_count': len(task_info.get('data', [])),
                'checklist_items': len(checklist),
                'status': self._get_project_status(task_dir)
            }

        except Exception as e:
            print(f"获取项目摘要失败: {e}")
            return None

    def _get_project_status(self, task_dir: Path) -> str:
        """获取项目状态"""
        status_file = task_dir / "TASK_STATUS.md"
        if status_file.exists():
            content = status_file.read_text()
            if "已完成" in content:
                return "completed"
            elif "进行中" in content:
                return "in_progress"
        return "initialized"
