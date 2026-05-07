import json
import re
from typing import Any, Tuple, Optional

_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.DOTALL)

def _strip_code_fences(s: str) -> str:
    # 去掉围栏 ```json ... ```
    return _JSON_FENCE_RE.sub("", s)

def _extract_json_segment(s: str) -> str:
    """
    从包含说明文字的 response 中抽出最大可能的 JSON 片段：
    取第一个 '{' 或 '[' 到最后一个 '}' 或 ']'
    改进：更智能地匹配对应的括号
    """
    start_obj = s.find("{")
    start_arr = s.find("[")
    starts = [x for x in [start_obj, start_arr] if x != -1]
    if not starts:
        return s.strip()
    
    start = min(starts)
    start_char = s[start]
    
    # 尝试找到匹配的结束括号
    if start_char == '{':
        # 对于对象，尝试匹配括号
        bracket_count = 0
        for i in range(start, len(s)):
            if s[i] == '{':
                bracket_count += 1
            elif s[i] == '}':
                bracket_count -= 1
                if bracket_count == 0:
                    return s[start:i+1].strip()
    elif start_char == '[':
        # 对于数组，尝试匹配方括号
        bracket_count = 0
        for i in range(start, len(s)):
            if s[i] == '[':
                bracket_count += 1
            elif s[i] == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    return s[start:i+1].strip()
    
    # 如果匹配失败，回退到原来的逻辑
    last_curly = s.rfind("}")
    last_square = s.rfind("]")
    end = max(last_curly, last_square)
    if end == -1 or end <= start:
        return s[start:].strip()
    return s[start:end+1].strip()

def _normalize_quotes(s: str) -> str:
    # 智能引号 => 普通引号
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    # 键名使用单引号 -> 双引号
    s = re.sub(r"(?<!\\)'([A-Za-z0-9_\-\.]+)'\s*:", r'"\1":', s)
    # 字符串字面量的单引号 -> 双引号（尽量保守，不碰到 URL 中的单引号）
    s = re.sub(r":\s*'([^']*)'", lambda m: ': "' + m.group(1).replace('"', '\\"') + '"', s)
    return s

def _fix_multiline_strings(s: str) -> str:
    """
    修复多行字符串中的换行符问题
    """
    # 将未转义的换行符转换为转义的换行符
    s = re.sub(r'(?<!\\)\n', '\\n', s)
    s = re.sub(r'(?<!\\)\r', '\\r', s)
    s = re.sub(r'(?<!\\)\t', '\\t', s)
    return s

def _python_literals_to_json(s: str) -> str:
    # Python -> JSON
    s = re.sub(r"\bNone\b", "null", s)
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    # NaN/Infinity -> null（更安全）
    s = re.sub(r"\bNaN\b", "null", s, flags=re.IGNORECASE)
    s = re.sub(r"\b-?Infinity\b", "null", s, flags=re.IGNORECASE)
    return s

def _remove_trailing_commas(s: str) -> str:
    # 去掉结尾多余逗号： {...,} 或 [...,]
    return re.sub(r",\s*(?=[}\]])", "", s)

def _insert_missing_commas_between_blocks(s: str) -> str:
    # 把相邻的 }{ 或 ][ 或 }[ 或 ]{ 之间补逗号
    return re.sub(r"([}\]])\s*([{\[])", r"\1,\2", s)

def _unquote_object_strings(s: str) -> str:
    # 修复数组里对象被当成字符串的情况： ,"{...}" 或 "[{...}]" 中的 "{...}"
    s = re.sub(r',\s*"\s*\{', r', {', s)
    s = re.sub(r'\}\s*"\s*(?=\s*[,}\]])', r'}', s)
    # 修复单个对象被整体加了引号的情况："{"a":1}" -> {"a":1}
    s = re.sub(r'"\s*\{', '{', s)
    s = re.sub(r'\}\s*"', '}', s)
    return s

def _fix_embedded_quotes_before_object(s: str) -> str:
    # 你这次的错误：数组元素前多了一个引号 ,"{...}"，或对象后紧跟 ,"{"n1"...}
    s = re.sub(r',\s*"\s*\{', r', {', s)
    return s

def _balance_braces(s: str) -> str:
    # 简单配平：只在缺少少量右括号/右方括号时补上
    open_curly = s.count("{")
    close_curly = s.count("}")
    if open_curly > close_curly:
        s += "}" * (open_curly - close_curly)
    open_sq = s.count("[")
    close_sq = s.count("]")
    if open_sq > close_sq:
        s += "]" * (open_sq - close_sq)
    return s

def _collapse_control_chars(s: str) -> str:
    # 去掉裸的控制字符（避免破坏 JSON 解析）
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", s)

def _last_resort_wrap(s: str) -> str:
    # 最后手段：如果看起来像 key:value 且不是以 { 开头，尝试包个花括号
    t = s.strip()
    if not t.startswith(("{", "[")) and re.search(r'"\w+"\s*:', t):
        return "{" + t + "}"
    return s

def try_loads(s: str) -> Tuple[Optional[Any], Optional[Exception]]:
    try:
        return json.loads(s), None
    except Exception as e:
        return None, e

def repair_json_string(text: str) -> str:
    """
    尝试将含噪声/不规范的 JSON 文本修复为有效的 JSON 字符串。
    返回修复后的 JSON 字符串，不进行解析。
    失败则抛出 ValueError，并带上最后一次解析异常。
    """
    transforms = []

    # 0) 预处理：去围栏，抽 JSON 体
    s = _strip_code_fences(text)
    s = _extract_json_segment(s)

    # 1) 直接尝试验证是否为有效 JSON
    obj, err = try_loads(s)
    if obj is not None:
        return s  # 返回原始字符串，因为它已经是有效的 JSON

    # 2) 渐进式修复流水线（顺序有意义）
    transforms.append(_collapse_control_chars)
    transforms.append(_normalize_quotes)
    transforms.append(_fix_multiline_strings)
    transforms.append(_python_literals_to_json)
    transforms.append(_remove_trailing_commas)
    transforms.append(_fix_embedded_quotes_before_object)
    transforms.append(_unquote_object_strings)
    transforms.append(_insert_missing_commas_between_blocks)
    transforms.append(_balance_braces)
    transforms.append(_last_resort_wrap)

    for tf in transforms:
        s = tf(s)
        obj, err = try_loads(s)
        if obj is not None:
            return s  # 返回修复后的有效 JSON 字符串

    # 3) 再尝试一次去尾逗号 + 配平（有时多步后才起效）
    s = _remove_trailing_commas(_balance_braces(s))
    obj, err2 = try_loads(s)
    if obj is not None:
        return s

    raise ValueError(f"Failed to repair JSON. Last error: {err2 or err}")

def repair_and_load_json(text: str) -> Any:
    """
    尝试将含噪声/不规范的 JSON 文本修复并解析为 Python 对象。
    失败则抛出 ValueError，并带上最后一次解析异常。
    """
    transforms = []

    # 0) 预处理：去围栏，抽 JSON 体
    s = _strip_code_fences(text)
    s = _extract_json_segment(s)

    # 1) 直接尝试
    obj, err = try_loads(s)
    if obj is not None:
        return obj

    # 2) 渐进式修复流水线（顺序有意义）
    transforms.append(_collapse_control_chars)
    transforms.append(_normalize_quotes)
    transforms.append(_fix_multiline_strings)
    transforms.append(_python_literals_to_json)
    transforms.append(_remove_trailing_commas)
    transforms.append(_fix_embedded_quotes_before_object)
    transforms.append(_unquote_object_strings)
    transforms.append(_insert_missing_commas_between_blocks)
    transforms.append(_balance_braces)
    transforms.append(_last_resort_wrap)

    for tf in transforms:
        s = tf(s)
        obj, err = try_loads(s)
        if obj is not None:
            return obj

    # 3) 再尝试一次去尾逗号 + 配平（有时多步后才起效）
    s = _remove_trailing_commas(_balance_braces(s))
    obj, err2 = try_loads(s)
    if obj is not None:
        return obj

    raise ValueError(f"Failed to repair/parse JSON. Last error: {err2 or err}")
