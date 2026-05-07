import re
import threading
from typing import Dict, List, Optional


class ReferenceManager:
    """
    参考文献管理器，用于收集和管理URL引用
    支持区分网页和论文类型，并记录标题
    线程安全：支持多线程并发访问
    """
    
    def __init__(self):
        self.references = []  # 存储所有引用信息
        self.url_to_tag = {}  # URL到标签的映射
        self._lock = threading.Lock()  # 线程锁，保证并发安全
        
    def add_url(self, url: str, title: Optional[str] = None, ref_type: str = "webpage") -> str:
        """
        添加URL并返回对应的引用标签（线程安全）
        
        Args:
            url: 要添加的URL
            title: 论文或网页标题（可选）
            ref_type: 引用类型，'paper'或'webpage'（默认为'webpage'）
            
        Returns:
            引用标签，如 "1", "2" 等
        """
        with self._lock:
            # 如果URL已存在，返回现有标签
            if url in self.url_to_tag:
                return self.url_to_tag[url]
            
            # 添加新URL
            reference = {
                "url": url,
                "title": title or self._extract_title_from_url(url),
                "type": ref_type
            }
            self.references.append(reference)
            tag = f"{len(self.references)}"
            self.url_to_tag[url] = tag
            return tag
    
    def _extract_title_from_url(self, url: str) -> str:
        """
        从URL中提取简单的标题（作为默认值）
        
        Args:
            url: URL字符串
            
        Returns:
            提取的标题
        """
        try:
            # 提取域名作为默认标题
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc or parsed.path
            # 移除www.前缀
            domain = re.sub(r'^www\.', '', domain)
            return domain
        except Exception:
            return url[:50] + "..." if len(url) > 50 else url
    
    def update_reference(self, url: str, title: Optional[str] = None, ref_type: Optional[str] = None) -> bool:
        """
        更新已存在的引用信息（线程安全）
        
        Args:
            url: 要更新的URL
            title: 新的标题（可选）
            ref_type: 新的类型（可选）
            
        Returns:
            是否更新成功
        """
        with self._lock:
            if url not in self.url_to_tag:
                return False
            
            tag = self.url_to_tag[url]
            index = int(tag[1:]) - 1  # 从标签获取索引
            
            if 0 <= index < len(self.references):
                if title is not None:
                    self.references[index]["title"] = title
                if ref_type is not None:
                    self.references[index]["type"] = ref_type
                return True
            
            return False
    
    def get_reference_list(self) -> List[Dict[str, str]]:
        """
        获取所有参考文献列表（线程安全）
        
        Returns:
            包含标签、URL、标题和类型的字典列表
        """
        with self._lock:
            result = []
            for i, ref in enumerate(self.references):
                result.append({
                    "tag": f"{i+1}",
                    "url": ref["url"],
                    "title": ref["title"],
                    "type": ref["type"]
                })
            return result
    
    def get_reference_text(self, include_type: bool = True) -> str:
        """
        获取格式化的参考文献文本
        
        Args:
            include_type: 是否在输出中包含类型标记
            
        Returns:
            格式化的参考文献字符串
        """
        if not self.references:
            return ""
        
        ref_lines = ["\n\n## References"]
        for i, ref in enumerate(self.references):
            tag = f"{i+1}"
            title = ref["title"]
            url = ref["url"]
            ref_type = ref["type"]
            
            if include_type:
                type_label = "📄" if ref_type == "paper" else "🌐"
                ref_lines.append(f"[{tag}] {type_label} {title} - {url}")
            else:
                ref_lines.append(f"[{tag}] {title} - {url}")
        
        return "\n".join(ref_lines)
    
    def get_papers(self) -> List[Dict[str, str]]:
        """
        获取所有论文类型的引用
        
        Returns:
            论文引用列表
        """
        return [ref for ref in self.get_reference_list() if ref["type"] == "paper"]
    
    def get_webpages(self) -> List[Dict[str, str]]:
        """
        获取所有网页类型的引用
        
        Returns:
            网页引用列表
        """
        return [ref for ref in self.get_reference_list() if ref["type"] == "webpage"]
    
    def clear(self):
        """清空所有参考文献（线程安全）"""
        with self._lock:
            self.references.clear()
            self.url_to_tag.clear()
    
    def __len__(self) -> int:
        """返回引用数量（线程安全）"""
        with self._lock:
            return len(self.references)
    
    def __contains__(self, url: str) -> bool:
        """检查URL是否已存在（线程安全）"""
        with self._lock:
            return url in self.url_to_tag

