"""
文件工具 - 文件路径处理和目录操作
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any


def safe_resolve(base: Path, target: str) -> Path:
    """
    安全解析路径，防止路径遍历攻击

    Args:
        base: 基础路径
        target: 目标相对路径

    Returns:
        解析后的绝对路径
    """
    # 解析路径
    target_path = Path(target).expanduser()

    # 如果是绝对路径，检查是否在base下
    if target_path.is_absolute():
        try:
            target_path.relative_to(base)
        except ValueError:
            raise ValueError(f"路径 {target} 不在基础路径 {base} 下")

    # 如果是相对路径，相对于base解析
    else:
        target_path = (base / target_path).resolve()

        # 检查是否在base下
        try:
            target_path.relative_to(base)
        except ValueError:
            raise ValueError(f"解析后的路径 {target_path} 不在基础路径 {base} 下")

    return target_path


def ensure_directory(path: Path) -> Path:
    """
    确保目录存在，不存在则创建

    Args:
        path: 目录路径

    Returns:
        创建的目录路径
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_file_safe(src: Path, dst: Path) -> bool:
    """
    安全地复制文件

    Args:
        src: 源文件路径
        dst: 目标文件路径

    Returns:
        是否成功
    """
    try:
        ensure_directory(dst.parent)
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        print(f"复制文件失败 {src} -> {dst}: {e}")
        return False


def write_json_file(path: Path, data: Any, indent: int = 2) -> bool:
    """
    写入JSON文件

    Args:
        path: 文件路径
        data: 要写入的数据
        indent: 缩进空格数

    Returns:
        是否成功
    """
    try:
        import json
        ensure_directory(path.parent)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"写入JSON文件失败 {path}: {e}")
        return False


def read_json_file(path: Path) -> Optional[Any]:
    """
    读取JSON文件

    Args:
        path: 文件路径

    Returns:
        读取的数据，失败返回None
    """
    try:
        import json
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"读取JSON文件失败 {path}: {e}")
        return None


def find_available_task_name(base_dir: Path, domain: str = "Science") -> str:
    """
    查找可用的任务名称

    Args:
        base_dir: 基础目录
        domain: 领域名称

    Returns:
        可用的任务ID，如 "ProteinBio_001"
    """
    base_dir = Path(base_dir)
    prefix = domain

    # 查找现有的任务
    existing_tasks = []
    if base_dir.exists():
        for item in base_dir.iterdir():
            if item.is_dir() and item.name.startswith(prefix):
                # 提取数字部分
                suffix = item.name[len(prefix):]
                if suffix.startswith('_'):
                    try:
                        num = int(suffix[1:])
                        existing_tasks.append(num)
                    except ValueError:
                        pass

    # 生成新的任务ID
    if existing_tasks:
        new_num = max(existing_tasks) + 1
    else:
        new_num = 1

    return f"{prefix}_{new_num:03d}"


def create_readme(path: Path, content: str) -> bool:
    """
    创建README文件

    Args:
        path: 文件路径
        content: 文件内容

    Returns:
        是否成功
    """
    try:
        ensure_directory(path.parent)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"创建README失败 {path}: {e}")
        return False


def get_file_size(path: Path) -> int:
    """
    获取文件大小

    Args:
        path: 文件路径

    Returns:
        文件大小（字节）
    """
    try:
        return path.stat().st_size
    except Exception:
        return 0


def format_size(size_bytes: int) -> str:
    """
    格式化文件大小

    Args:
        size_bytes: 字节数

    Returns:
        格式化的大小字符串
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def cleanup_temp_files(temp_dir: Path, max_age_hours: int = 24) -> int:
    """
    清理临时文件

    Args:
        temp_dir: 临时目录
        max_age_hours: 最大保留时间（小时）

    Returns:
        清理的文件数量
    """
    import time

    if not temp_dir.exists():
        return 0

    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    cleaned_count = 0

    try:
        for file_path in temp_dir.iterdir():
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                    except Exception:
                        pass
    except Exception as e:
        print(f"清理临时文件失败: {e}")

    return cleaned_count
