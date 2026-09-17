"""Maintainer-only builder for the reviewed DJ built-in model catalog.

The builder resolves movable Hugging Face names once, downloads only small
metadata/code/tokenizer files, and uses Hub LFS SHA256 identities for weights.
It never runs as part of DSH startup or Plan execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from .common import write_json_atomic


def consumer(operator, parameter, default, *, when=None, subpath=None):
    value = {"operator": operator, "parameter": parameter, "defaults": [default]}
    if when:
        value["when"] = when
    if subpath:
        value["subpath"] = subpath
    return value


HF_MODELS = {
    "CompVis/stable-diffusion-v1-4": [
        consumer("image_diffusion_mapper", "hf_diffusion", "CompVis/stable-diffusion-v1-4")
    ],
    "DAMO-NLP-SG/VideoLLaMA3-7B": [
        consumer(
            "video_captioning_face_attribute_emotion_mapper", "video_describe_model_path", "DAMO-NLP-SG/VideoLLaMA3-7B"
        ),
        consumer(
            "video_captioning_from_human_tracks_mapper", "video_describe_model_path", "DAMO-NLP-SG/VideoLLaMA3-7B"
        ),
    ],
    "EleutherAI/pythia-6.9b-deduped": [
        consumer("alphanumeric_filter", "hf_tokenizer", "EleutherAI/pythia-6.9b-deduped", when={"tokenization": True}),
        consumer("token_num_filter", "hf_tokenizer", "EleutherAI/pythia-6.9b-deduped"),
    ],
    "Falconsai/nsfw_image_detection": [
        consumer("image_nsfw_filter", "hf_nsfw_model", "Falconsai/nsfw_image_detection"),
        consumer("video_nsfw_filter", "hf_nsfw_model", "Falconsai/nsfw_image_detection"),
    ],
    "FunAudioLLM/SenseVoiceSmall": [
        consumer("video_audio_ASR_mapper", "model_dir_ASR", "FunAudioLLM/SenseVoiceSmall"),
        consumer("video_audio_speech_emotion_mapper", "model_dir_emo", "FunAudioLLM/SenseVoiceSmall"),
    ],
    "Helsinki-NLP/opus-mt-zh-en": [
        consumer("query_intent_detection_mapper", "zh_to_en_hf_model", "Helsinki-NLP/opus-mt-zh-en"),
        consumer("query_sentiment_detection_mapper", "zh_to_en_hf_model", "Helsinki-NLP/opus-mt-zh-en"),
        consumer("query_topic_detection_mapper", "zh_to_en_hf_model", "Helsinki-NLP/opus-mt-zh-en"),
    ],
    "MIT/ast-finetuned-audioset-10-10-0.4593": [
        consumer("video_tagging_from_audio_mapper", "hf_ast", "MIT/ast-finetuned-audioset-10-10-0.4593")
    ],
    "Qwen/Qwen-Audio": [consumer("video_captioning_from_audio_mapper", "hf_qwen_audio", "Qwen/Qwen-Audio")],
    "Qwen/Qwen2-7B-Instruct": [consumer("sentence_augmentation_mapper", "hf_model", "Qwen/Qwen2-7B-Instruct")],
    "Qwen/Qwen2.5-0.5B": [
        consumer("instruction_following_difficulty_filter", "hf_model", "Qwen/Qwen2.5-0.5B"),
        consumer("llm_perplexity_filter", "hf_model", "Qwen/Qwen2.5-0.5B"),
    ],
    "Qwen/Qwen2.5-7B-Instruct": [
        consumer("generate_qa_from_examples_mapper", "hf_model", "Qwen/Qwen2.5-7B-Instruct"),
        consumer(
            "llm_ray_vllm_engine_pipeline", "api_or_hf_model", "Qwen/Qwen2.5-7B-Instruct", when={"is_hf_model": True}
        ),
        consumer("optimize_prompt_mapper", "api_or_hf_model", "Qwen/Qwen2.5-7B-Instruct", when={"is_hf_model": True}),
        consumer("optimize_qa_mapper", "api_or_hf_model", "Qwen/Qwen2.5-7B-Instruct", when={"is_hf_model": True}),
        consumer("optimize_query_mapper", "api_or_hf_model", "Qwen/Qwen2.5-7B-Instruct", when={"is_hf_model": True}),
        consumer("optimize_response_mapper", "api_or_hf_model", "Qwen/Qwen2.5-7B-Instruct", when={"is_hf_model": True}),
        consumer("text_tagging_by_prompt_mapper", "hf_model", "Qwen/Qwen2.5-7B-Instruct"),
        consumer(
            "vlm_ray_vllm_engine_pipeline", "api_or_hf_model", "Qwen/Qwen2.5-7B-Instruct", when={"is_hf_model": True}
        ),
    ],
    "Qwen/Qwen2.5-VL-7B-Instruct": [
        consumer(
            "image_tagging_vlm_mapper", "api_or_hf_model", "Qwen/Qwen2.5-VL-7B-Instruct", when={"is_api_model": False}
        )
    ],
    "Qwen/Qwen3-VL-8B-Instruct": [
        consumer("video_captioning_from_vlm_mapper", "hf_model", "Qwen/Qwen3-VL-8B-Instruct")
    ],
    "Ruicheng/moge-2-vitl": [consumer("video_camera_calibration_moge_mapper", "model_path", "Ruicheng/moge-2-vitl")],
    "Salesforce/blip-itm-base-coco": [
        consumer("image_text_matching_filter", "hf_blip", "Salesforce/blip-itm-base-coco")
    ],
    "Salesforce/blip2-opt-2.7b": [
        consumer("image_captioning_mapper", "hf_img2seq", "Salesforce/blip2-opt-2.7b"),
        consumer("image_diffusion_mapper", "hf_img2seq", "Salesforce/blip2-opt-2.7b", when={"caption_key": None}),
        consumer("video_captioning_from_frames_mapper", "hf_img2seq", "Salesforce/blip2-opt-2.7b"),
    ],
    "alibaba-pai/pai-qwen1_5-7b-doc2qa": [
        consumer("generate_qa_from_text_mapper", "hf_model", "alibaba-pai/pai-qwen1_5-7b-doc2qa")
    ],
    "amrul-hzz/watermark_detector": [
        consumer("image_watermark_filter", "hf_watermark_model", "amrul-hzz/watermark_detector"),
        consumer("video_watermark_filter", "hf_watermark_model", "amrul-hzz/watermark_detector"),
    ],
    "audeering/wav2vec2-large-robust-24-ft-age-gender": [
        consumer("video_audio_detect_age_gender_mapper", "hf_audio_mapper", None)
    ],
    "bespin-global/klue-roberta-small-3i4k-intent-classification": [
        consumer(
            "query_intent_detection_mapper", "hf_model", "bespin-global/klue-roberta-small-3i4k-intent-classification"
        )
    ],
    "dstefa/roberta-base_topic_classification_nyt_news": [
        consumer("query_topic_detection_mapper", "hf_model", "dstefa/roberta-base_topic_classification_nyt_news")
    ],
    "facebook/VGGT-1B": [consumer("vggt_mapper", "vggt_model_path", "facebook/VGGT-1B")],
    "facebook/sam-3d-body-dinov3": [],
    "facebook/sam2.1-hiera-tiny": [
        consumer("video_object_segmenting_mapper", "sam2_hf_model", "facebook/sam2.1-hiera-tiny")
    ],
    "google/owlvit-base-patch32": [
        consumer("phrase_grounding_recall_filter", "hf_owlvit", "google/owlvit-base-patch32")
    ],
    "kpyu/video-blip-opt-2.7b-ego4d": [
        consumer("video_captioning_from_video_mapper", "hf_video_blip", "kpyu/video-blip-opt-2.7b-ego4d")
    ],
    "llava-hf/llava-v1.6-vicuna-7b-hf": [consumer("mllm_mapper", "hf_model", "llava-hf/llava-v1.6-vicuna-7b-hf")],
    "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis": [
        consumer(
            "query_sentiment_detection_mapper",
            "hf_model",
            "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
        )
    ],
    "mrm8488/flan-t5-large-finetuned-openai-summarize_from_feedback": [
        consumer("video_captioning_from_summarizer_mapper", "hf_summarizer", None)
    ],
    "openai/clip-vit-base-patch32": [
        consumer("image_pair_similarity_filter", "hf_clip", "openai/clip-vit-base-patch32"),
        consumer("image_text_similarity_filter", "hf_clip", "openai/clip-vit-base-patch32"),
        consumer("text_pair_similarity_filter", "hf_clip", "openai/clip-vit-base-patch32"),
        consumer("video_frames_text_similarity_filter", "hf_clip", "openai/clip-vit-base-patch32"),
    ],
    "shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE": [
        consumer("image_aesthetics_filter", "hf_scorer_model", ""),
        consumer("video_aesthetics_filter", "hf_scorer_model", ""),
    ],
    "stabilityai/stable-diffusion-xl-base-1.0": [
        consumer("sdxl_prompt2prompt_mapper", "hf_diffusion", "stabilityai/stable-diffusion-xl-base-1.0")
    ],
}

# Metadata for this gated model is visible, but its files cannot be hashed
# without accepting the upstream license. It remains explicitly blocked.
HF_MODELS.pop("facebook/sam-3d-body-dinov3", None)

BLOCKED_REQUIREMENTS = [
    {
        "operator": "image_sam_3d_body_mapper",
        "model_type": "sam_3d_body",
        "backend": "modelscope-or-huggingface",
        "model_id": "facebook/sam-3d-body-dinov3",
        "reason": "gated_model_requires_accepted_license_and_authenticated_maintainer_validation",
    },
    {
        "operator": "image_tagging_mapper",
        "model_type": "recognizeAnything",
        "reason": "upstream_3gb_ram_weight_has_no_published_sha256_and_runtime_source_is_an_unpinned_git_dependency",
    },
    {
        "operator": "video_tagging_from_frames_mapper",
        "model_type": "recognizeAnything",
        "reason": "upstream_3gb_ram_weight_has_no_published_sha256_and_runtime_source_is_an_unpinned_git_dependency",
    },
    {
        "operator": "video_camera_calibration_deepcalib_mapper",
        "model_type": "deepcalib",
        "parameter": "model_path",
        "defaults": ["weights_10_0.02.h5"],
        "reason": "google_drive_archive_has_no_maintainer_validated_immutable_identity",
    },
    {
        "operator": "video_hand_reconstruction_mapper",
        "model_type": "wilor",
        "reason": "runtime_requires_a_licensed_mano_file_and_an_unpinned_git_source_checkout",
    },
    {
        "operator": "video_hand_reconstruction_hawor_mapper",
        "model_type": "hawor",
        "reason": "runtime_requires_licensed_mano_files_and_an_unpinned_git_source_checkout",
    },
    {
        "operator": "video_depth_estimation_mapper",
        "model_type": "video_depth_anything",
        "reason": "runtime_source_repository_clone_is_not_yet_materialized_from_an_immutable_code_lock",
    },
    {
        "operator": "vggt_mapper",
        "model_type": "vggt",
        "reason": "runtime_source_repository_clone_is_not_yet_materialized_from_an_immutable_code_lock",
    },
    {
        "operator": "video_human_tracks_extraction_mapper",
        "model_type": "YOLOv8_human",
        "reason": "third_party_repository_and_bundled_weight_are_not_present_in_the_release_checkout",
    },
    {
        "operator": "video_human_tracks_extraction_mapper",
        "model_type": "face_detect_S3FD",
        "reason": "third_party_repository_is_cloned_from_an_unpinned_branch_at_runtime",
    },
    {
        "operator": "video_active_speaker_detect_mapper",
        "model_type": "Light_ASD",
        "reason": "third_party_repository_and_bundled_weight_are_not_present_in_the_release_checkout",
    },
]


def blocked_language_resource(operator, model_type, *, when=None):
    item = {
        "operator": operator,
        "model_type": model_type,
        "reason": "language_selected_resource_set_is_dynamic_and_has_no_reviewed_per_language_hash_manifest",
    }
    if when:
        item["when"] = when
    return item


BLOCKED_REQUIREMENTS.extend(
    [
        blocked_language_resource("flagged_words_filter", "sentencepiece", when={"tokenization": True}),
        blocked_language_resource("perplexity_filter", "sentencepiece"),
        blocked_language_resource("perplexity_filter", "kenlm"),
        blocked_language_resource("stopwords_filter", "sentencepiece", when={"tokenization": True}),
        blocked_language_resource("word_repetition_filter", "sentencepiece", when={"tokenization": True}),
        blocked_language_resource("words_num_filter", "sentencepiece", when={"tokenization": True}),
        blocked_language_resource(
            "remove_words_with_incorrect_substrings_mapper", "sentencepiece", when={"tokenization": True}
        ),
        blocked_language_resource("sentence_split_mapper", "nltk"),
        blocked_language_resource("phrase_grounding_recall_filter", "nltk_pos_tagger"),
        blocked_language_resource("text_action_filter", "spacy"),
        blocked_language_resource("text_entity_dependency_filter", "spacy"),
    ]
)


EXEMPT_REQUIREMENTS = [
    {
        "model_type": "api",
        "reason": "remote API identifiers do not download local model artifacts",
    },
    {
        "operator": "image_mmpose_mapper",
        "model_type": None,
        "reason": "operator requires caller-supplied local deployment configs and model files",
    },
    {
        "operator": "llm_analysis_filter",
        "model_types": ["huggingface", "vllm"],
        "when": {"is_hf_model": False},
        "reason": "built-in default selects the API branch; explicit HF selection is rejected without a curated lock",
    },
    {
        "operator": "llm_condition_filter",
        "model_types": ["huggingface", "vllm"],
        "when": {"is_hf_model": False},
        "reason": "built-in default selects the API branch; explicit HF selection is rejected without a curated lock",
    },
    {
        "operator": "llm_extract_mapper",
        "model_types": ["huggingface", "vllm"],
        "when": {"is_hf_model": False},
        "reason": "built-in default selects the API branch; explicit HF selection is rejected without a curated lock",
    },
    {
        "operator": "text_embd_similarity_filter",
        "model_type": "embedding",
        "when": {"is_hf_model": False},
        "reason": "built-in default selects the API branch; explicit local-model selection requires a separate curated lock",
    },
]


HTTP_MODELS = [
    {
        "lock_id": "http-fasttext-lid-176-7e69ec5451bc",
        "backend": "http-file",
        "url": "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
        "filename": "lid.176.bin",
        "size": 131266198,
        "sha256": "7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e",
        "runtime_packages": ["fasttext-wheel"],
        "consumers": [consumer("language_id_score_filter", "model_path", "lid.176.bin")],
    },
    {
        "lock_id": "http-fastsam-x-v8.2.0-752cadc2828e",
        "backend": "http-file",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/FastSAM-x.pt",
        "filename": "FastSAM-x.pt",
        "size": 144972346,
        "sha256": "752cadc2828edb1cd4bc4f9eb587100631af06ea2108f4c9ed56df4755701e76",
        "runtime_packages": ["ultralytics"],
        "consumers": [consumer("image_segment_mapper", "model_path", "FastSAM-x.pt")],
    },
    {
        "lock_id": "http-fastsam-s-v8.2.0-c9f78716a81c",
        "backend": "http-file",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/FastSAM-s.pt",
        "filename": "FastSAM-s.pt",
        "size": 23851578,
        "sha256": "c9f78716a81c7aff0d608ccc73e1b82ab3aaad86005049f6a92106a0be6d0844",
        "runtime_packages": ["ultralytics"],
        "consumers": [consumer("image_segment_mapper", "model_path", "FastSAM-s.pt")],
    },
    {
        "lock_id": "http-yolo11n-v8.3.0-0ebbc80d4a76",
        "backend": "http-file",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        "filename": "yolo11n.pt",
        "size": 5613764,
        "sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
        "runtime_packages": ["ultralytics"],
        "consumers": [consumer("image_detection_yolo_mapper", "model_path", "yolo11n.pt")],
    },
    {
        "lock_id": "http-yoloe-11l-seg-v8.3.0-a993fb0fc7c8",
        "backend": "http-file",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11l-seg.pt",
        "filename": "yoloe-11l-seg.pt",
        "size": 70982416,
        "sha256": "a993fb0fc7c8830939ae14e6434a925dd1179428158c2761482eb8a8d8a3699f",
        "runtime_packages": ["ultralytics"],
        "consumers": [consumer("video_object_segmenting_mapper", "yoloe_path", "yoloe-11l-seg.pt")],
    },
]


def hf_file(lock_id, repo_id, revision, path, size, sha256, operator, parameter, default, *, repo_type="models"):
    return {
        "lock_id": lock_id,
        "backend": "http-file",
        "url": f"https://huggingface.co/{repo_type + '/' if repo_type != 'models' else ''}{repo_id}/resolve/{revision}/{path}",
        "filename": Path(path).name,
        "size": size,
        "sha256": sha256,
        "runtime_packages": ["torch"],
        "upstream": {"repository": repo_id, "revision": revision, "path": path},
        "consumers": [consumer(operator, parameter, default)],
    }


HTTP_MODELS.extend(
    [
        hf_file(
            "hf-file-dwpose-yolox-l-7860ae79de6c",
            "yzd-v/DWPose",
            "1a7144101628d69ee7a3768d1ee3a094070dc388",
            "yolox_l.onnx",
            216746733,
            "7860ae79de6c89a3c1eb72ae9a2756c0ccfbe04b7791bb5880afabd97855a411",
            "video_whole_body_pose_estimation_mapper",
            "onnx_det_model",
            "yolox_l.onnx",
        ),
        hf_file(
            "hf-file-dwpose-pose-724f4ff2439e",
            "yzd-v/DWPose",
            "1a7144101628d69ee7a3768d1ee3a094070dc388",
            "dw-ll_ucoco_384.onnx",
            134399116,
            "724f4ff2439ed61afb86fb8a1951ec39c6220682803b4a8bd4f598cd913b1843",
            "video_whole_body_pose_estimation_mapper",
            "onnx_pose_model",
            "dw-ll_ucoco_384.onnx",
        ),
        hf_file(
            "hf-file-vda-small-13379300b739",
            "depth-anything/Video-Depth-Anything-Small",
            "256875362cff76724b920335dfb4b29dd611f66e",
            "video_depth_anything_vits.pth",
            116440756,
            "13379300b739e659f076a59d52e9801bd8d38c541a7e71f73bbca4dcfb013609",
            "video_depth_estimation_mapper",
            "video_depth_model_path",
            "video_depth_anything_vits.pth",
        ),
        hf_file(
            "hf-file-vda-base-775e578e8f94",
            "depth-anything/Video-Depth-Anything-Base",
            "7231d0c6e260f54f103eba5005d01f6efa4f43db",
            "video_depth_anything_vitb.pth",
            458247082,
            "775e578e8f9431ec0496514aa466bd0a1f67c28d0f518267809f35a43c04329b",
            "video_depth_estimation_mapper",
            "video_depth_model_path",
            "video_depth_anything_vitb.pth",
        ),
        hf_file(
            "hf-file-vda-large-43df27c6b396",
            "depth-anything/Video-Depth-Anything-Large",
            "7aafbcb5c6af0bac741aad2b6471894fb4761afa",
            "video_depth_anything_vitl.pth",
            1538392012,
            "43df27c6b396042ba34ff7b798ab279f64d204d2e86d7a373968f8fa36d0e6fa",
            "video_depth_estimation_mapper",
            "video_depth_model_path",
            "video_depth_anything_vitl.pth",
        ),
        hf_file(
            "hf-file-metric-vda-small-3c28432b4e1f",
            "depth-anything/Metric-Video-Depth-Anything-Small",
            "273d090f2ce17df50c2872d82c8322c45da5b4dd",
            "metric_video_depth_anything_vits.pth",
            116444063,
            "3c28432b4e1f0d7bb31cad5151b6313b49457db5aa58d82e85bfb0f8b1311b33",
            "video_depth_estimation_mapper",
            "video_depth_model_path",
            "metric_video_depth_anything_vits.pth",
        ),
        hf_file(
            "hf-file-metric-vda-base-f6f58576b968",
            "depth-anything/Metric-Video-Depth-Anything-Base",
            "f6a245abad4b5a5b0d26722c8e1767ef310c547d",
            "metric_video_depth_anything_vitb.pth",
            458249567,
            "f6f58576b9680a112f2428d4f39aff92656d3ae85745b0164675ace8d5b1fade",
            "video_depth_estimation_mapper",
            "video_depth_model_path",
            "metric_video_depth_anything_vitb.pth",
        ),
        hf_file(
            "hf-file-metric-vda-large-24eba25342e7",
            "depth-anything/Metric-Video-Depth-Anything-Large",
            "607fcdbd454b95c3bd39abbd3054142869a527d3",
            "metric_video_depth_anything_vitl.pth",
            1538348216,
            "24eba25342e7ee0f054be25da6a852ebd7cfa40cd6bf41a353ecb6abfe24e620",
            "video_depth_estimation_mapper",
            "video_depth_model_path",
            "metric_video_depth_anything_vitl.pth",
        ),
        hf_file(
            "hf-file-wilor-model-3e97aafc7dd0",
            "rolpotamias/WiLoR",
            "99fe3d7acff8104ecca1055df7467709506c2fa6",
            "pretrained_models/wilor_final.ckpt",
            2564989533,
            "3e97aafc7dd08d883a4cc5a027df61fdb6fda6136dbd1319405413862ada6bb2",
            "video_hand_reconstruction_mapper",
            "wilor_model_path",
            "wilor_final.ckpt",
            repo_type="spaces",
        ),
        hf_file(
            "hf-file-wilor-config-f69cb52704df",
            "rolpotamias/WiLoR",
            "99fe3d7acff8104ecca1055df7467709506c2fa6",
            "pretrained_models/model_config.yaml",
            2233,
            "f69cb52704df88ef29a7cfe03f35a677a92c1ce08169b729ca7f5862f05d4297",
            "video_hand_reconstruction_mapper",
            "wilor_model_config",
            "model_config.yaml",
            repo_type="spaces",
        ),
        hf_file(
            "hf-file-wilor-detector-5ef3df44e42d",
            "rolpotamias/WiLoR",
            "99fe3d7acff8104ecca1055df7467709506c2fa6",
            "pretrained_models/detector.pt",
            53582271,
            "5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033",
            "video_hand_reconstruction_mapper",
            "detector_model_path",
            "detector.pt",
            repo_type="spaces",
        ),
        hf_file(
            "hf-file-hawor-model-4d1cc43853c1",
            "ThunderVVV/HaWoR",
            "da6335f47f9806308992d5ae1002a4cc5f7252c2",
            "hawor/checkpoints/hawor.ckpt",
            3267481572,
            "4d1cc43853c190d6f2c10d9b6295c73109f0faf9ef41ac817a2b31d94b4823f2",
            "video_hand_reconstruction_hawor_mapper",
            "hawor_model_path",
            "hawor.ckpt",
        ),
        hf_file(
            "hf-file-hawor-config-edfe12dc14ce",
            "ThunderVVV/HaWoR",
            "da6335f47f9806308992d5ae1002a4cc5f7252c2",
            "hawor/model_config.yaml",
            2743,
            "edfe12dc14ce371d698da722b59acfed5b4a38a7f8f5116cbc1fce459a07dd2d",
            "video_hand_reconstruction_hawor_mapper",
            "hawor_config_path",
            "model_config.yaml",
        ),
        hf_file(
            "hf-file-hawor-detector-5ef3df44e42d",
            "ThunderVVV/HaWoR",
            "da6335f47f9806308992d5ae1002a4cc5f7252c2",
            "external/detector.pt",
            53582271,
            "5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033",
            "video_hand_reconstruction_hawor_mapper",
            "hawor_detector_path",
            "detector.pt",
        ),
    ]
)


def _selected_paths(siblings):
    names = [item.rfilename for item in siblings]
    has_safetensors = any(name.endswith(".safetensors") for name in names)
    is_diffusers = "model_index.json" in names
    selected = []
    for name in names:
        lower = name.casefold()
        basename = Path(name).name.casefold()
        if basename in {"readme.md", "license", "license.md", ".gitattributes"}:
            continue
        if "/.ipynb_checkpoints/" in "/" + lower or lower.endswith((".png", ".jpg", ".jpeg", ".gif")):
            continue
        if any(token in lower for token in ("flax", "tf_model", "openvino", ".onnx", ".gguf")):
            continue
        if lower.endswith(".safetensors"):
            if ".fp16." in lower or (is_diffusers and "/" not in name):
                continue
            selected.append(name)
            continue
        if lower.endswith((".bin", ".pt", ".pth", ".ckpt")):
            if has_safetensors:
                continue
            selected.append(name)
            continue
        if lower.endswith((".json", ".txt", ".model", ".spm", ".tiktoken", ".py", ".yaml", ".yml")):
            selected.append(name)
    return sorted(set(selected))


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(output: Path, *, refresh: bool = False):
    api = HfApi()
    models = []
    previous = {}
    if output.exists() and not refresh:
        existing = json.loads(output.read_text(encoding="utf-8"))
        previous = {
            model.get("model_id"): model
            for model in existing.get("models", [])
            if model.get("backend") == "huggingface" and model.get("model_id")
        }
    for model_id, consumers in sorted(HF_MODELS.items(), key=lambda item: item[0].casefold()):
        if model_id in previous:
            model = previous[model_id]
            model["consumers"] = consumers
            models.append(model)
            continue
        info = api.model_info(model_id, files_metadata=True)
        by_name = {item.rfilename: item for item in info.siblings or []}
        files = []
        for name in _selected_paths(info.siblings or []):
            sibling = by_name[name]
            lfs_hash = getattr(sibling.lfs, "sha256", None) if sibling.lfs else None
            if lfs_hash:
                digest = lfs_hash
            else:
                local = Path(hf_hub_download(repo_id=model_id, filename=name, revision=info.sha))
                digest = _sha256(local)
            files.append({"path": name, "size": int(sibling.size), "sha256": digest})
        if not files:
            raise RuntimeError(f"No runtime files selected for {model_id}")
        slug = "-".join(part for part in model_id.casefold().replace("_", "-").replace("/", "-").split("-") if part)
        models.append(
            {
                "lock_id": f"hf-{slug}-{info.sha[:12]}",
                "backend": "huggingface",
                "model_id": model_id,
                "revision": info.sha,
                "runtime_packages": ["huggingface-hub", "torch", "transformers"],
                "files": files,
                "consumers": consumers,
            }
        )
    # OpenCV's classifier is shipped by the uv-locked wheel, not a model registry.
    models.append(
        {
            "lock_id": "py-opencv-haarcascade-frontalface-alt-4.11.0.86",
            "backend": "python-distribution",
            "distribution": "opencv-contrib-python",
            "version": "4.11.0.86",
            "module": "cv2",
            "resource": "data/haarcascade_frontalface_alt.xml",
            "size": 676709,
            "sha256": "6281df13459cc218ff047d02b2ae3859b12ff14a93ffe8952f7b33fad7b9697b",
            "consumers": [
                consumer("image_face_count_filter", "cv_classifier", ""),
                consumer("image_face_ratio_filter", "cv_classifier", ""),
                consumer("image_face_blur_mapper", "cv_classifier", ""),
                consumer("video_face_blur_mapper", "cv_classifier", ""),
            ],
        }
    )
    models.extend(HTTP_MODELS)
    value = {
        "schema_version": 1,
        "dj_version": "1.5.4",
        "dj_source_revision": "a2e21c98532f8767cb590cd5b35c0752a47f56d1",
        "audit": {
            "hf_repositories": len(HF_MODELS),
            "http_artifacts": len(HTTP_MODELS),
            "method": "reviewed prepare_model call sites and operator defaults; immutable revisions and SHA256 identities",
        },
        "blocked_requirements": BLOCKED_REQUIREMENTS,
        "exempt_requirements": EXEMPT_REQUIREMENTS,
        "models": models,
    }
    write_json_atomic(output, value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(Path(__file__).with_name("builtin_model_catalog.json")))
    parser.add_argument("--refresh", action="store_true", help="re-resolve every movable Hub name")
    args = parser.parse_args(argv)
    result = build(Path(args.output).resolve(), refresh=args.refresh)
    print(json.dumps({"output": args.output, "models": len(result["models"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
