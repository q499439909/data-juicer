"""Deterministic zh-CN presentation metadata for the operator library.

The generated JSON is the runtime source of truth.  These helpers keep names
and fallback summaries consistent when the live registry adds an operator.
They never alter executable operator identifiers or parameter keys.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

LOCALE_PATH = Path(__file__).with_name("operator_i18n") / "zh-CN.json"

_SUFFIXES = {
    "aggregator": "聚合器",
    "deduplicator": "去重器",
    "filter": "过滤器",
    "grouper": "分组器",
    "mapper": "映射器",
    "pipeline": "流水线",
    "selector": "选择器",
}

_PHRASES = {
    "active_speaker_detect": "活跃说话人检测",
    "add_gaussian_noise": "添加高斯噪声",
    "agent_bad_case_signal": "智能体不良样本信号",
    "agent_dialog_normalize": "智能体对话规范化",
    "agent_insight_llm": "智能体大模型洞察",
    "agent_skill_insight": "智能体技能洞察",
    "agent_tool_relevance": "智能体工具相关性",
    "agent_tool_type": "智能体工具类型",
    "agent_trace_coherence": "智能体轨迹连贯性",
    "aspect_ratio": "宽高比",
    "atomic_action_segment": "原子动作分段",
    "bad_case": "不良样本",
    "camera_calibration": "相机标定",
    "camera_pose": "相机位姿",
    "character_attributes": "人物属性",
    "character_locations": "人物位置",
    "clean_copyright": "版权信息清理",
    "coreference": "共指消解",
    "difference_area_generator": "差异区域生成",
    "difference_caption_generator": "差异描述生成",
    "document_line": "文档行级",
    "error_recovery": "错误恢复",
    "face_attribute_emotion": "人脸属性与情绪",
    "field_selector": "字段选择",
    "following_difficulty": "遵循难度",
    "from_examples": "从示例生成",
    "from_human_tracks": "基于人物轨迹",
    "from_summarizer": "基于摘要器",
    "from_text": "从文本生成",
    "human_preference_annotation": "人工偏好标注",
    "human_tracks": "人物轨迹",
    "in_context_influence": "上下文影响",
    "incorrect_substrings": "错误子串",
    "instruction_following": "指令遵循",
    "key_frame": "关键帧",
    "language_id_score": "语言识别得分",
    "line_length": "行长度",
    "main_character": "主要人物",
    "memory_consistency": "记忆一致性",
    "motion_score": "运动得分",
    "non_chinese_character": "非中文字符",
    "non_repetition": "非重复性",
    "object_segmenting": "目标分割",
    "pair_similarity": "配对相似度",
    "phrase_grounding_recall": "短语定位召回率",
    "prompt2prompt": "提示词到提示词",
    "quality_score": "质量得分",
    "remove_background": "背景移除",
    "remove_watermark": "水印移除",
    "repeat_sentences": "重复句子",
    "resize_aspect_ratio": "按宽高比缩放",
    "resize_resolution": "按分辨率缩放",
    "sentiment_intensity": "情感强度",
    "specified_field": "指定字段",
    "specified_numeric_field": "指定数值字段",
    "speech_emotion": "语音情绪",
    "split_by_duration": "按时长切分",
    "split_by_key_frame": "按关键帧切分",
    "split_by_scene": "按场景切分",
    "support_text": "支撑文本",
    "task_relevance": "任务相关性",
    "text_matching": "图文匹配",
    "text_similarity": "图文相似度",
    "topic_shift": "话题转移",
    "whole_body_pose_estimation": "全身姿态估计",
    "with_uid": "带唯一标识",
    "words_num": "词数",
}

_WORDS = {
    "agent": "智能体", "alphanumeric": "字母数字", "analysis": "分析", "annotation": "标注",
    "asr": "语音识别", "audio": "音频", "aesthetics": "美学", "attribute": "属性",
    "attributes": "属性", "augmentation": "增强", "average": "平均", "background": "背景",
    "batch": "批处理", "bibliography": "参考文献", "blur": "模糊", "body": "人体",
    "bts": "BTS", "calibrate": "校准", "caption": "描述", "captioning": "描述生成",
    "character": "字符", "characters": "字符", "chinese": "中文", "chunk": "分块",
    "clarification": "澄清", "clean": "清理", "clip": "片段", "coherence": "连贯性",
    "comments": "注释", "condition": "条件", "consistency": "一致性", "content": "内容",
    "convert": "转换", "copyright": "版权", "count": "数量", "counter": "计数",
    "cpp": "C++", "deepcalib": "DeepCalib", "dependency": "依存关系", "depth": "深度",
    "detect": "检测", "detection": "检测", "demographic": "人口统计属性", "dialog": "对话",
    "difference": "差异", "difficulty": "难度", "diffusion": "扩散生成", "document": "文档",
    "download": "下载", "droidcalib": "DroidCalib", "duration": "时长", "email": "电子邮箱",
    "embd": "嵌入", "emotion": "情绪", "engine": "引擎", "entities": "实体",
    "entity": "实体", "event": "事件", "examples": "示例", "expand": "展开",
    "export": "导出", "extract": "提取", "extraction": "提取", "face": "人脸",
    "ffmpeg": "FFmpeg", "field": "字段", "file": "文件", "fix": "修复",
    "flagged": "标记", "frames": "帧", "frequency": "频率", "from": "基于",
    "fused": "融合", "general": "通用", "generate": "生成", "generator": "生成",
    "gpt": "GPT", "grounding": "定位", "hand": "手部", "header": "页眉",
    "html": "HTML", "human": "人物", "identity": "身份", "image": "图像",
    "influence": "影响", "insight": "洞察", "instruction": "指令", "intent": "意图",
    "ip": "IP 地址", "keyword": "关键词", "lambda": "Lambda", "language": "语言",
    "latex": "LaTeX", "length": "长度", "lerobot": "LeRobot", "links": "链接",
    "llm": "大语言模型", "long": "长词", "macro": "宏", "matching": "匹配",
    "maximum": "最大", "megasam": "MegaSAM", "merge": "合并", "meta": "元数据",
    "minhash": "MinHash", "mllm": "多模态大模型", "mmpose": "MMPose", "moge": "MoGe",
    "motion": "运动", "most": "最", "naive": "基础", "nested": "嵌套",
    "nickname": "昵称", "nlpaug": "NLPaug 英文增强", "nlpcda": "NLPCDA 中文增强",
    "nmf": "NMF", "normalization": "规范化", "normalize": "规范化", "nsfw": "不良内容",
    "num": "数量", "numeric": "数值", "ocr": "文字识别", "optimize": "优化",
    "overlay": "叠加", "pair": "配对", "perplexity": "困惑度", "pii": "个人敏感信息",
    "pose": "姿态", "preference": "偏好", "prompt": "提示词", "proactivity": "主动性",
    "punctuation": "标点", "python": "Python", "qa": "问答", "query": "问题",
    "raft": "RAFT", "random": "随机", "range": "范围", "ray": "Ray",
    "recall": "召回率", "reassembly": "重组", "reconstruction": "重建", "redaction": "脱敏",
    "relation": "关系", "relevance": "相关性", "relevant": "相关", "remove": "移除",
    "repartition": "重新分区", "repeat": "重复", "repetition": "重复度", "replace": "替换",
    "resolution": "分辨率", "response": "回答", "reverse": "反向", "s3": "S3",
    "sam": "SAM", "scene": "场景", "score": "得分", "segment": "分段",
    "segmenting": "分割", "sentence": "句子", "sequential": "顺序", "shape": "形状",
    "signal": "信号", "similarity": "相似度", "simhash": "SimHash", "size": "大小",
    "skill": "技能", "smooth": "平滑", "snr": "信噪比", "special": "特殊",
    "specified": "指定", "split": "切分", "subplot": "子图", "success": "成功",
    "suffix": "后缀", "summarizer": "摘要器", "table": "表格", "tables": "表格",
    "tagger": "标记", "tagging": "标注", "tags": "标签", "tex": "TeX",
    "text": "文本", "token": "词元", "tool": "工具", "topk": "Top-K",
    "topic": "话题", "trace": "轨迹", "trajectory": "轨迹", "type": "类型",
    "uid": "唯一标识", "undistort": "畸变校正", "unicode": "Unicode", "upload": "上传",
    "usage": "用量", "vggt": "VGGT", "video": "视频", "vlm": "视觉语言模型",
    "vllm": "vLLM", "watermark": "水印", "whitespace": "空白字符", "word": "词语",
    "words": "词语", "wrapped": "封装", "yolo": "YOLO", "zh": "中文", "en": "英文",
    "3d": "三维", "area": "区域", "ratio": "占比", "action": "动作", "age": "年龄",
    "gender": "性别", "atomic": "原子", "camera": "相机", "main": "主要", "line": "行",
    "all": "全部", "alone": "单独", "api": "API", "assistant": "助手", "auto": "自动",
    "axes": "维度", "calibration": "校准", "chars": "字符", "check": "检查", "choices": "候选项",
    "compute": "计算", "compress": "压缩", "config": "配置", "copy": "复制", "discard": "丢弃",
    "dominant": "主要", "empty": "空值", "enable": "启用", "estimation": "估计", "fail": "失败",
    "fields": "字段", "first": "首个", "for": "所需", "hard": "困难", "hawor": "HaWoR",
    "head": "头部", "high": "高", "history": "历史", "hint": "提示", "id": "标识",
    "include": "包含", "in": "在", "json": "JSON", "key": "字段", "keys": "字段列表",
    "label": "标签", "lang": "语言", "latency": "延迟", "len": "长度", "lineage": "血缘信息",
    "low": "低", "manual": "手动", "max": "最大", "medium": "中等", "messages": "消息",
    "min": "最小", "model": "模型", "ms": "毫秒", "must": "必须", "negative": "负向",
    "on": "针对", "output": "输出", "overrides": "覆盖", "overwrite": "覆盖写入", "params": "参数",
    "path": "路径", "poor": "较差", "precision": "精确", "preferred": "首选", "preview": "预览",
    "primary": "主要", "quality": "质量", "reply": "回复", "request": "请求", "result": "结果",
    "round": "轮次", "rounds": "轮次", "run": "运行", "sampling": "采样", "sentiment": "情感",
    "signals": "信号", "strict": "严格", "substrings": "子串", "suspect": "疑似", "system": "系统",
    "threshold": "阈值", "thresholds": "阈值", "tiers": "等级", "tokens": "词元", "total": "总计",
    "types": "类型", "user": "用户", "value": "值", "watchlist": "观察列表", "with": "带有",
    "to": "到", "specific": "特定", "chars": "字符", "stopwords": "停用词", "figure": "图表",
    "context": "上下文", "extractor": "提取", "suspect": "疑似", "ptlflow": "PTLFlow",
    "imgdiff": "图像差异", "value": "值",
}

_PARAM_PHRASES = {
    "min_len": "最小长度", "max_len": "最大长度", "text_key": "文本字段",
    "image_key": "图像字段", "audio_key": "音频字段", "video_key": "视频字段",
    "batch_size": "批次大小", "num_proc": "进程数", "model_path": "模型路径",
    "model_name": "模型名称", "api_key": "API 密钥", "api_endpoint": "API 端点",
    "save_dir": "保存目录", "save_path": "保存路径", "output_dir": "输出目录",
    "threshold": "阈值", "min_score": "最低得分", "max_score": "最高得分",
    "device": "计算设备", "seed": "随机种子", "language": "语言",
}


def _translate_identifier(value: str) -> str:
    value = str(value or "").strip().lower()
    if value in _PARAM_PHRASES:
        return _PARAM_PHRASES[value]
    remaining = value
    parts: list[str] = []
    phrases = {**_PHRASES, **_PARAM_PHRASES}
    while remaining:
        match = next((key for key in sorted(phrases, key=len, reverse=True) if remaining == key or remaining.startswith(key + "_")), None)
        if match:
            parts.append(phrases[match])
            remaining = remaining[len(match):].lstrip("_")
            continue
        token, _, remaining = remaining.partition("_")
        parts.append(_WORDS.get(token, token.upper() if len(token) <= 4 else "配置"))
    return "".join(parts) or value


def translated_operator_name(name: str, category: str) -> str:
    stem = name
    suffix = _SUFFIXES.get(category, "算子")
    marker = f"_{category}"
    if stem.endswith(marker):
        stem = stem[: -len(marker)]
    elif marker in stem:
        stem = stem.replace(marker, "", 1).strip("_")
    return f"{_translate_identifier(stem)}{suffix}"


def translated_parameter_name(name: str) -> str:
    translated = _translate_identifier(name)
    return translated if re.search(r"[\u3400-\u9fff]", translated) else f"{translated} 参数"


def translated_summary(display_name: str, category: str, modalities: list[str]) -> str:
    subject = {"text": "文本", "image": "图像", "audio": "音频", "video": "视频", "multimodal": "多模态"}.get(
        next((item for item in modalities if item != "general"), ""), "数据"
    )
    if category == "filter":
        return f"{display_name}用于按相应质量或属性条件筛选{subject}样本，仅保留满足配置要求的数据。筛选阈值、目标字段及处理行为可通过初始化参数调整。"
    if category == "deduplicator":
        return f"{display_name}用于识别并移除{subject}数据中的重复或高度相似样本。匹配范围、相似度策略和输出方式可通过初始化参数调整。"
    if category == "aggregator":
        return f"{display_name}用于汇总{subject}样本中的相关字段或统计信息，并将聚合结果写入配置的输出字段。"
    if category == "selector":
        return f"{display_name}用于按照配置规则从{subject}数据中选择目标样本或字段。选择范围和排序方式可通过初始化参数调整。"
    if category == "grouper":
        return f"{display_name}用于按照指定键或规则对{subject}样本进行分组，并为后续批量处理或聚合提供结构化输入。"
    if category == "pipeline":
        return f"{display_name}用于组织并执行一组连续的{subject}数据处理步骤。执行引擎、并行度和输入输出行为可通过初始化参数调整。"
    return f"{display_name}用于对{subject}样本执行对应的数据转换、分析或增强处理。输入字段、模型设置和输出行为可通过下列初始化参数调整。"


_OPERATOR_TEXT_OVERRIDES = {
    "image_aesthetics_filter": {
        "summary": "基于 Hugging Face 图像美学预测模型的过滤器，用美学得分筛选图像，只保留得分落在指定范围内的样本。",
        "description": (
            "该算子使用 Hugging Face 模型预测每张图像的美学得分，并保留预测得分位于最低分与最高分之间的样本。"
            "对于包含多张图像的样本，可通过 any_or_all 选择判定策略：any 表示任意一张图像达标即可保留，all 表示所有图像都达标才保留。"
            "预测结果会缓存到 image_aesthetics_scores 字段；样本中没有图像时仍会保留。"
            "当模型名称包含 shunk031/aesthetics-predictor 时，原始得分会除以 10 进行归一化。"
        ),
    },
    "image_aspect_ratio_filter": {
        "summary": "基于图像宽度除以高度（W/H）的宽高比过滤器，只保留宽高比落在指定范围内的图像样本。",
        "description": (
            "该算子以图像宽度除以高度（W/H）计算每张图像的宽高比，并将结果缓存到 aspect_ratios 字段。"
            "只有宽高比位于设定的最小值与最大值之间时，图像才算满足条件。"
            "对于包含多张图像的样本，any 表示至少一张达标即可保留，all 表示全部达标才保留；样本中没有图像时不会被过滤掉。"
        ),
    },
    "image_face_count_filter": {
        "summary": "基于 OpenCV 人脸分类器的数量过滤器，按检测到的人脸数量筛选图像样本。",
        "description": (
            "该算子使用 OpenCV 分类器检测图像中的人脸，并仅保留人脸数量位于指定范围内的样本。"
            "对于包含多张图像的样本，any 表示任意一张图像的人脸数量达标即可保留，all 表示所有图像都达标才保留。"
            "检测到的人脸数量会缓存到 face_counts 字段；样本中没有图像时，该字段会记录为空数组。"
        ),
    },
    "image_face_ratio_filter": {
        "summary": "基于 OpenCV 人脸检测的面积占比过滤器，按最大人脸面积占整张图像面积的比例筛选样本。",
        "description": (
            "该算子使用 OpenCV 分类器检测人脸，并计算每张图像中最大人脸面积占图像总面积的比例，结果记录为 face_ratios。"
            "只有该比例位于设定的最小值与最大值之间时，图像才算满足条件。"
            "对于包含多张图像的样本，any 表示任意一张达标即可保留，all 表示全部达标才保留；样本中没有图像时仍会保留。"
        ),
    },
}

_METHOD_RULES = (
    (r"\byolo\b", "YOLO 模型"),
    (r"\bclip\b", "CLIP 模型"),
    (r"\bffmpeg\b", "FFmpeg"),
    (r"\bminhash\b", "MinHash 算法"),
    (r"\bsimhash\b", "SimHash 算法"),
    (r"\bnmf\b", "NMF 方法"),
    (r"\bmmpose\b", "MMPose 模型"),
    (r"\bsam\b", "SAM 模型"),
    (r"large language model|\bllm\b", "大语言模型"),
    (r"vision.language model|\bvlm\b", "视觉语言模型"),
    (r"optical flow|\braft\b", "光流估计"),
    (r"regular expression|\bregex\b", "正则表达式"),
    (r"(?:uses?|using|based on)[^.]{0,40}hugging face|hugging face model", "Hugging Face 模型"),
    (r"(?:uses?|using|based on)[^.]{0,40}opencv|opencv (?:classifier|model)", "OpenCV"),
)


def _operator_subject(modalities: list[str]) -> str:
    labels = {"text": "文本", "image": "图像", "audio": "音频", "video": "视频", "multimodal": "多模态"}
    return labels.get(next((item for item in modalities if item != "general"), ""), "数据")


def _operator_concept(display_name: str, category: str) -> str:
    suffix = _SUFFIXES.get(category, "算子")
    return display_name[: -len(suffix)] if suffix and display_name.endswith(suffix) else display_name


def _detected_method(name: str, original: str) -> str | None:
    # An algorithm encoded in the registered name is more authoritative than
    # incidental library or dataset names mentioned later in the docstring.
    intro = original.split("\n\n", 1)[0]
    for haystack in (name.lower(), intro.lower()):
        method = next((label for pattern, label in _METHOD_RULES if re.search(pattern, haystack)), None)
        if method:
            return method
    return None


def translated_operator_text(
    name: str,
    display_name: str,
    category: str,
    modalities: list[str],
    original: str,
) -> dict[str, str]:
    """Build concise and detailed Chinese copy from operator-specific metadata.

    Carefully translated overrides cover semantics whose edge behaviour matters. The
    fallback still names the operator's own method/concept instead of collapsing all
    operators in a category into one generic paragraph.
    """
    override = _OPERATOR_TEXT_OVERRIDES.get(name)
    if override:
        return dict(override)

    subject = _operator_subject(modalities)
    concept = _operator_concept(display_name, category)
    method = _detected_method(name, original)
    basis = (f"基于 {method}" if method and method[0].isascii() else f"基于{method}") if method else None
    attribute_basis = f"{basis}{' 的' if method and method[-1].isascii() else '的'}" if basis else None

    if category == "filter":
        summary = (
            f"{attribute_basis}{display_name}，按{concept}条件筛选{subject}样本，只保留满足配置规则的数据。"
            if basis else
            f"按{concept}的计算或判定结果筛选{subject}样本，只保留满足配置规则的数据。"
        )
        detail = (
            f"该算子{basis}评估每个{subject}样本，并依据{concept}条件决定是否保留。"
            if basis else
            f"该算子计算或检查每个{subject}样本的{concept}，并依据配置条件决定是否保留。"
        )
    elif category == "deduplicator":
        basis = basis or f"基于{concept}特征"
        attribute_basis = attribute_basis or f"{basis}的"
        summary = f"{attribute_basis}{display_name}，用于发现并移除{subject}数据中的重复或高度相似样本。"
        detail = f"该算子{basis}比较{subject}样本，识别重复或高度相似的数据并执行去重。"
    elif category == "mapper":
        summary = (
            f"{attribute_basis}{display_name}，用于对{subject}样本执行{concept}处理并写回结果。"
            if basis else
            f"用于对{subject}样本执行{concept}处理，并将生成或转换后的结果写回数据。"
        )
        detail = (
            f"该算子{basis}对输入的{subject}样本执行{concept}处理，并把处理结果写入配置字段。"
            if basis else
            f"该算子对输入的{subject}样本执行{concept}处理，并把生成或转换后的结果写入配置字段。"
        )
    else:
        summary = f"{attribute_basis}{display_name}，用于对{subject}样本执行{concept}处理。" if basis else f"用于对{subject}样本执行{concept}处理。"
        detail = f"该算子{basis}处理{subject}样本，以完成{concept}任务。" if basis else f"该算子处理{subject}样本，以完成{concept}任务。"

    original_lower = original.lower()
    additions = []
    if "minimum" in original_lower and "maximum" in original_lower:
        additions.append("最小值与最大值参数共同限定有效范围，只有落在该范围内的结果才满足条件。")
    if "'any'" in original_lower and "'all'" in original_lower:
        additions.append("多媒体样本可选择 any 或 all 策略：any 表示任一媒体达标即可，all 表示全部媒体都要达标。")
    cached = re.search(r"cached in (?:the )?['\"]([^'\"]+)['\"] field", original, re.IGNORECASE)
    if cached:
        additions.append(f"计算结果会缓存到 {cached.group(1)} 字段，供后续算子复用。")
    if "if no images are present" in original_lower and ("sample is kept" in original_lower or "retained" in original_lower or "not filtered out" in original_lower):
        additions.append("样本中没有图像时，该样本不会被过滤掉。")
    additions.append("具体输入字段、处理强度、执行方式和输出行为可通过下方初始化参数调整。")
    return {"summary": summary, "description": detail + "".join(additions)}


def translated_parameter_summary(display_name: str, required: bool) -> str:
    qualifier = "必须提供" if required else "可按需设置"
    return f"用于配置“{display_name}”；该参数{qualifier}，具体取值范围请参考英文原始说明。"


@lru_cache(maxsize=1)
def load_zh_cn() -> dict[str, Any]:
    try:
        payload = json.loads(LOCALE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": 1, "operators": {}}
    if not isinstance(payload, dict) or not isinstance(payload.get("operators"), dict):
        return {"schema_version": 1, "operators": {}}
    return payload


def localize_catalog_item(item: dict[str, Any]) -> dict[str, Any]:
    entry = load_zh_cn()["operators"].get(item["name"])
    if not isinstance(entry, dict):
        return {**item, "display_name_zh": item["name"], "description_zh": item["description"], "translation_status": "pending"}
    return {
        **item,
        "display_name_zh": str(entry.get("display_name") or item["name"]),
        "description_zh": str(entry.get("summary") or entry.get("description") or item["description"]),
        "translation_status": "translated",
    }


def localize_detail(operator: dict[str, Any]) -> dict[str, Any]:
    entry = load_zh_cn()["operators"].get(operator["name"])
    if not isinstance(entry, dict):
        return {
            **operator,
            "display_name_zh": operator["name"],
            "summary_zh": operator["description"],
            "description_zh": operator["description"],
            "translation_status": "pending",
        }
    parameter_entries = entry.get("parameters") if isinstance(entry.get("parameters"), dict) else {}
    parameters = []
    for parameter in operator["parameters"]:
        localized = parameter_entries.get(parameter["name"], {})
        parameters.append(
            {
                **parameter,
                "display_name_zh": str(localized.get("display_name") or parameter["name"]),
                "description_zh": str(localized.get("description") or parameter["description"]),
            }
        )
    return {
        **operator,
        "display_name_zh": str(entry.get("display_name") or operator["name"]),
        "summary_zh": str(entry.get("summary") or entry.get("description") or operator["description"]),
        "description_zh": str(entry.get("description") or operator["description"]),
        "translation_status": "translated",
        "parameters": parameters,
    }


def contains_han(value: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", value or ""))
