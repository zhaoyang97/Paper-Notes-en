---
title: >-
  [Paper Note] PixFoundation 2.0: Do Video Multi-Modal LLMs Use Motion in Visual Grounding?
description: >-
  [NeurIPS 2025][Video Understanding][Motion-centric evaluation] By introducing four motion-centric probing techniques and the MoCentric-Bench benchmark, this paper demonstrates that current video multimodal LLMs fail to genuinely exploit motion information in pixel-level visual grounding tasks and can be deceived by static keyframes.
tags:
  - NeurIPS 2025
  - Video Understanding
  - Motion-centric evaluation
  - visual grounding
  - video MLLM
  - referring segmentation
  - motion understanding
date: 2026-05-08
content_hash: c5866b49e0411cf0
---

# PixFoundation 2.0: Do Video Multi-Modal LLMs Use Motion in Visual Grounding?

**Conference**: NeurIPS 2025
**arXiv**: [2509.02807](https://arxiv.org/abs/2509.02807)
**Code**: [https://github.com/MSiam/PixFoundation-2.0](https://github.com/MSiam/PixFoundation-2.0)
**Area**: Video Understanding / Visual Grounding
**Keywords**: Motion-centric evaluation, visual grounding, video MLLM, referring segmentation, motion understanding

## TL;DR

By introducing four motion-centric probing techniques and the MoCentric-Bench benchmark, this paper demonstrates that current video multimodal LLMs fail to genuinely exploit motion information in pixel-level visual grounding tasks and can be deceived by static keyframes.

## Background & Motivation

**Background**: Multimodal large language models perform well on high-level video tasks (QA, captioning), but their pixel-level visual grounding capabilities have not been thoroughly investigated.

**Limitations of Prior Work**: Existing benchmarks assume motion is necessary, yet in practice a single static frame is often sufficient to resolve motion-referring expressions. Video MLLMs claim to leverage temporal information but may rely solely on powerful visual encoders and LLMs.

**Key Challenge**: There is no means to distinguish "true motion" (temporal dynamics) from "pseudo-motion" (simulatable by static keyframes), and no benchmark exists for evaluating motion-order understanding.

**Key Insight**: The paper designs two categories of probing techniques—motion existence and motion order—and constructs MoCentric-Bench to compel models to utilize genuine motion.

## Method

### Overall Architecture

A three-tier structure: (1) problem diagnosis—analyzing deficiencies in existing benchmarks such as RefDAVIS and MeVIS; (2) motion-centric probe design—automatically synthesizing fake motion and reversed motion; (3) baselines and adaptation—providing a single-frame strong baseline and a LoRA fine-tuned variant of Sa2VA.

### Key Designs

1. **Motion Existence Probe**

    - **Function**: Determines whether a model genuinely relies on motion information.
    - **Mechanism**: Qwen2.5-VL is used for coarse-grained temporal localization to identify keyframes; fake videos (repeated keyframes) are generated and combined with original videos into four layout configurations to test whether models are deceived.
    - **Design Motivation**: Motion-referring expressions (e.g., "jumping to the left") may be resolvable from a single frame combined with spatial layout alone.

2. **Motion Order Probe**

    - **Function**: Determines whether a model understands temporal direction.
    - **Mechanism**: GPT-4o is used to convert expressions such as "pull" into their reverse counterparts (e.g., "push"); videos are played in reverse to test whether models can distinguish forward from backward motion.
    - **Design Motivation**: A model that truly understands motion should be able to differentiate forward and reversed video sequences.

3. **MLLM + SAM 2.0 Strong Baseline**

    - **Function**: Validates the competitiveness of single-frame approaches.
    - **Mechanism**: Qwen2.5-VL performs bounding-box localization on keyframes; SAM 2.0 generates full segmentation masks—no temporal reasoning is involved.
    - **Design Motivation**: If this single-frame baseline approaches state-of-the-art performance, it indicates that existing datasets are insufficiently motion-centric.

### Loss & Training

The Sa2VA★ fine-tuned variant applies LoRA to the visual encoder and is trained with supervised learning on MoCentric-Bench synthetic data.

## Key Experimental Results

### Main Results (Existing Benchmarks)

| Method | RefDAVIS-17 J&F | MeVIS val J&F |
|--------|-----------------|----------------|
| LISA | 64.8 | 37.2 |
| VideoGLAMM | 69.5 | 45.2 |
| Sa2VA | 75.2 | 51.5 |
| MLLM+SAM2 (single-frame baseline) | 70.5 | 44.5 |
| MLLM+SAM2† (keyframe) | **71.7** | **46.9** |

### Ablation Study (MoCentric-Bench)

| Model | Orig. val | Single-frame mix | Drop% | Reversed mix | Drop% |
|-------|-----------|-----------------|-------|--------------|-------|
| VidGLAMM | 48.2 | 21.6 | -55.2% | 34.0 | -29.4% |
| Sa2VA | 58.9 | 28.5 | -51.6% | 61.1 | +3.7% |
| MLLM+SAM2† | 57.4 | 28.1 | -51.0% | 53.1 | -7.5% |
| Sa2VA★ (fine-tuned) | 63.1 | 38.2 | -39.5% | 56.4 | -10.6% |

### Key Findings

- The single-frame baseline matches or surpasses multiple state-of-the-art methods, exposing a heavy reliance on static information in existing datasets.
- All models suffer substantial performance drops under single-frame mixing (39–55%), indicating that they are deceived by fake motion.
- Sa2VA even shows a marginal gain (+3.7%) under reversed mixing, demonstrating that it has no understanding of motion direction whatsoever.

## Highlights & Insights

- **First motion-centric evaluation**: Systematically exposes methodological issues in existing benchmarks, with far-reaching implications for evaluation protocol design in video understanding.
- **Automated data synthesis**: VLM + LLM pipelines automatically generate fake and reversed motion, resulting in a low-cost and scalable approach.
- **Clear identification of weaknesses**: The state-of-the-art method (Sa2VA) suffers a dramatic performance drop on motion-centric tasks (51.5 → 28.5), a finding that will drive video MLLMs toward genuine motion understanding.

## Limitations & Future Work

- MoCentric-Bench is relatively small in scale (32–152 objects); larger-scale validation is needed.
- Fine-tuning processes only the first five frames, which may be insufficient to fully exploit motion.
- Reversed expressions may be semantically incompatible with certain videos, requiring manual filtering.

## Related Work & Insights

- **vs. ATP (Buch et al., 2022)**: ATP also contrasts single-frame and full-video performance, but is limited to video-level understanding tasks; this paper is the first to systematically study pixel-level grounding tasks in this manner.
- **vs. Kowal et al. (2022)**: Prior work analyzes the ratio of static to dynamic content; this paper proposes a complete probing framework.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First motion-centric visual grounding evaluation
- Experimental Thoroughness: ⭐⭐⭐⭐ Four probing techniques are comprehensive; the benchmark could be larger
- Writing Quality: ⭐⭐⭐⭐⭐ Problem formulation is clear; methods are readily reproducible
- Value: ⭐⭐⭐⭐⭐ Advances video MLLMs toward genuine motion understanding

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] MUVR: A Multi-Modal Untrimmed Video Retrieval Benchmark with Multi-Level Visual Correspondence](muvr_a_multi-modal_untrimmed_video_retrieval_benchmark_with_multi-level_visual_c.md)
- [\[ICCV 2025\] AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning](../../ICCV2025/video_understanding/aim_adaptive_inference_multimodal_llms_token_merging_pruning.md)
- [\[ICCV 2025\] DynImg: Key Frames with Visual Prompts are Good Representation for Multi-Modal Video Understanding](../../ICCV2025/video_understanding/dynimg_key_frames_with_visual_prompts_are_good_representation_for_multi-modal_vi.md)
- [\[ICCV 2025\] OVG-HQ: Online Video Grounding with Hybrid-modal Queries](../../ICCV2025/video_understanding/ovg-hq_online_video_grounding_with_hybrid-modal_queries.md)
- [\[NeurIPS 2025\] When Thinking Drifts: Evidential Grounding for Robust Video Reasoning](when_thinking_drifts_evidential_grounding_for_robust_video_reasoning.md)

<!-- RELATED:END -->
