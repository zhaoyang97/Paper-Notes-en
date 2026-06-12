---
title: >-
  [Paper Note] Jailbreaking Multimodal Large Language Models using Multi-Clip Video
description: >-
  [ACL2026][Multimodal VLM][Multimodal Safety] This paper constructs MCV SafetyBench to evaluate the safety of video MLLMs. It finds that multi-clip…
tags:
  - "ACL2026"
  - "Multimodal VLM"
  - "Multimodal Safety"
  - "Video Input"
  - "MLLM Evaluation"
  - "Attack Success Rate"
  - "Image Filtering Defense"
date: 2026-05-08
content_hash: e31bdf7e05580013
---

# Jailbreaking Multimodal Large Language Models using Multi-Clip Video

**Conference**: ACL2026  
**arXiv**: [2606.02111](https://arxiv.org/abs/2606.02111)  
**Code**: https://github.com/ChoongwonKang/MCV_Jailbreak.git  
**Area**: Multimodal VLM / Video Safety Evaluation  
**Keywords**: Multimodal Safety, Video Input, MLLM Evaluation, Attack Success Rate, Image Filtering Defense  

## TL;DR
This paper constructs MCV SafetyBench to evaluate the safety of video MLLMs. It finds that multi-clip, multi-context video inputs systematically increase attack success rates (ASR), whereas simple frame-based image filtering can significantly mitigate these risks.

## Background & Motivation
**Background**: MLLMs have expanded from image-text understanding to video understanding, capable of processing dynamic scenes, temporal information, and complex visual contexts. Meanwhile, multimodal safety research indicates that visual inputs are often more likely to weaken a model's safety alignment than pure text.

**Limitations of Prior Work**: Existing multimodal safety research focuses primarily on image attacks, such as embedding unsafe contexts or text within images. Although the video modality is longer, more dynamic, and possesses more complex contexts, systematic analysis regarding which specific video attributes lead to safety misalignment is still lacking.

**Key Challenge**: Video models need to integrate information across multiple temporal segments. While richer information aids task understanding, diverse contexts may also dilute or confuse the model's identification of harmful intent, making safety boundaries more fragile.

**Goal**: The authors aim to isolate several factors in video input: whether video is more vulnerable than images, whether dynamic video is more vulnerable than static video, and whether diverse clips are more vulnerable than repeated clips. Based on these findings, they propose a simple defense.

**Key Insight**: The paper constructs MCV SafetyBench, which contains multiple short clips per sample, and observes changes in model risk by incrementally increasing the number of clips. The notes discuss high-level mechanisms for evaluation and defense without reproducing specific harmful prompt content.

**Core Idea**: Treat "video context diversity" as a controllable variable to systematically evaluate how it affects MLLM safety alignment, and utilize the relative robustness of the image modality to implement frame-based filtering defense.

## Method

### Overall Architecture
The paper first constructs MCV SafetyBench and then evaluates the ASR of eight video MLLMs under different input settings. The dataset includes 1,460 queries across 13 risk categories related to OpenAI usage policy; each query corresponds to four 2-second clips combined into an 8-second multi-clip video, creating both standard video versions and videos with integrated visual text, totaling 2,920 videos.

Two settings are compared: the Explicit setting provides harmful intent as text along with the video, while the Implicit setting embeds text intent as visual text within the video. The paper uses GPT-4o-mini to score model outputs from 1 to 5 following CLAS-style rules. A score of 5 is counted as a success, and correlation validation was performed by 10 human annotators on 200 samples.

### Key Designs
1. **MCV SafetyBench construction**:
	- **Function**: Provides a video safety evaluation set with controllable clip counts and context diversity.
	- **Mechanism**: Extracts semantic components (subject, object, action, atmosphere) from existing risk queries and reconstructs them into video generation prompts using GPT-4o. Multiple 2-second clips are generated using Wan2.2-T2V-A14B and concatenated into multi-clip videos. 220 samples with insufficient expression or diversity were manually removed.
	- **Design Motivation**: Directly collecting real-world videos makes it difficult to control clip counts, semantic diversity, and risk categories. Synthetic pipelines allow for clearer experimental variables.

2. **Controlled attack settings**:
	- **Function**: Distinguishes contributions from factors such as video length, dynamism, visual text, and context diversity.
	- **Mechanism**: Besides the original multi-clip video, the paper compares settings including sampled image frames, static videos, videos repeating the same clip, and videos combining different clips. All videos are input at a uniform frame rate to avoid frame rate becoming a confounding variable.
	- **Design Motivation**: If repeating the same clip also increases risk, the primary cause is video length; if only combinations of different clips significantly increase risk, the key factor is context diversity.

3. **Representation analysis and image filtering defense**:
	- **Function**: Explains why multi-clip videos weaken safety alignment and proposes a low-cost mitigation strategy.
	- **Mechanism**: The authors extract hidden states from the last layer for the final input tokens and use PCA to observe harmful/benign video representations. Results show that as the number of clips increases, sample representations shift from harmful anchors toward benign directions. The defense randomly samples frames and uses the same target model to judge safety via image input before deciding whether to process the video.
	- **Design Motivation**: If the safety of the image modality is stronger than that of the video modality, image filtering can serve as a front-line defense. While not a fundamental cure for video safety weaknesses, it is easy to implement.

### Loss & Training
This study focuses on evaluation and defense experiments and does not train new MLLMs. The core metric is Attack Success Rate (ASR), defined as the number of harmful responses divided by the total number of harmful inputs. The judge model is GPT-4o-mini, where a score of 1 indicates explicit rejection and 5 indicates full compliance with the violating intent; only a score of 5 is counted as a success. In human validation, the correlation between model scores and human ratings was 0.6229 (std=0.069), and the average correlation among human annotators was 0.766 (std=0.144).

## Key Experimental Results

### Main Results
| Model | Explicit 1-Clip | Explicit 4-Clip | Implicit 1-Clip | Implicit 4-Clip | Main Phenomenon |
|------|-----------------|-----------------|-----------------|-----------------|----------|
| Qwen2.5-VL-7B | 50.75 | 68.70 | 69.04 | 80.27 | Most significant increase with clip count |
| Qwen2.5-VL-32B | 71.71 | 81.10 | 79.79 | 82.33 | Larger models are not necessarily safer |
| Qwen2.5-VL-72B | 43.70 | 57.60 | 74.52 | 76.10 | Stable in Explicit, but still fragile in Implicit |
| Qwen3-VL-8B | 55.48 | 57.40 | 72.40 | 73.15 | Implicit is higher overall and less sensitive to clip count |
| InternVL3.5-8B | 46.16 | 58.08 | 64.04 | 65.27 | Dynamic video significantly higher than image frames |
| LLaVA-Video-7B | 66.58 | 66.85 | 49.86 | 50.68 | Lower Implicit ASR, likely due to weaker OCR |

### Ablation Study
| Setting / Defense | Qwen2.5-VL-7B | Qwen3-VL-8B | InternVL3.5-8B | LLaVA-Video-7B | Average ASR |
|-------------|---------------|-------------|-----------------|---------------|---------|
| Image Frame Attack | 50.93 | 58.89 | 46.47 | 33.39 | 47.42 |
| Static Video Attack | 68.11 | 72.26 | 64.66 | 40.82 | 61.46 |
| Clip-Rep (Repeated) | 63.68 | 55.02 | 44.91 | 28.27 | 47.97 |
| Original Multi-clip | 77.23 | 72.57 | 64.78 | 49.86 | 66.11 |
| Undefended 4-Clip | 80.27 | 73.15 | 65.27 | 50.68 | 67.34 |
| Safe system | 70.48 | 57.05 | 33.01 | 50.62 | 52.79 |
| AdaShield | 73.49 | 15.68 | 23.01 | 5.62 | 29.45 |
| Image filtering | 33.63 | 0.62 | 29.66 | 5.55 | 17.37 |

### Key Findings
- On most models, a higher number of clips leads to a higher ASR; Qwen2.5-VL-7B's Explicit ASR rose from 50.75 to 68.70, and Implicit from 69.04 to 80.27.
- The video modality is more fragile than sampled image frames, dynamic video is more fragile than static video, and combinations of different clips are more fragile than repeated clips, indicating that the key is "diverse context" rather than just "more frames."
- Illegal Activity and Hate Speech categories showed significant growth in the Explicit setting, with average ASRs increasing from 43.19 to 63.19 and 22.90 to 40.88, respectively.
- Image filtering reduced the average ASR of four models from 67.34 to 17.37, outperforming Safe system and AdaShield.

## Highlights & Insights
- The most significant contribution of the paper is not proposing a stronger attack, but decomposing video safety weaknesses into controllable variables: image vs. video, static vs. dynamic, and repetition vs. diversity.
- Representation analysis provides an interesting explanation: multi-clip inputs cause internal model representations to shift toward "benign" regions, suggesting safety recognition may be diluted by rich context.
- The defense approach is pragmatic. Since the image modality is more stable, using sampled image filtering as a fallback is simple yet experimentally effective.
- Implications for video MLLM deployment: Safety evaluation results for images cannot be directly extrapolated to video; video inputs require independent safety gating and benchmarks.

## Limitations & Future Work
- The authors note that experiments were limited to a maximum of 5 clips and 10 seconds total duration, not covering longer videos or more complex temporal narratives. As long-video understanding improves, risk patterns may change.
- Image filtering is only an indirect utilization of the image modality's relative stability and does not solve the inherent problems of video representation and video safety alignment.
- The dataset relies on text-to-video generation. Although the authors reproduced experimental trends with HunyuanVideo-1.5, there remains a gap between synthetic and real complex videos.
- ASR relies on a GPT-4o-mini judge and rule templates; human correlation is moderately strong but not perfect. Stricter multi-judge setups, human auditing, and severity grading are still necessary.

## Related Work & Insights
- **vs. Image Jailbreak**: Previous studies focused on image text, complex layouts, or image contexts; this work pushes the risk into video clip diversity and temporal context.
- **vs. Video Safety Observations by Hu et al. / Liu et al.**: These works noted that video might be more fragile; this paper further deconstructs the input attributes causing this fragility.
- **vs. Prompt-based Defense**: Safe system and AdaShield have limited effectiveness under video input, indicating that text system prompts alone are insufficient and modality-level filtering is required.
- **Inspiration for Future Research**: Video safety models should explicitly model risk aggregation across temporal segments rather than judging safety only on single frames or final text responses.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Systematically controlling multi-clip video variables is valuable; the attack format builds upon existing visual text and video safety research.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across 8 open-source models, closed-source model supplements, generator replication, clip count scaling, representation analysis, and defense comparisons.
- Writing Quality: ⭐⭐⭐⭐☆ Clear storyline with sufficient tables.
- Value: ⭐⭐⭐⭐⭐ Direct reference significance for video MLLM safety evaluation and deployment defense.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] DMN: A Compositional Framework for Jailbreaking Multimodal LLMs with Multi-Image Inputs](dmn_a_compositional_framework_for_jailbreaking_multimodal_llms_with_multi-image_.md)
- [\[ICCV 2025\] Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency](../../ICCV2025/multimodal_vlm/jailbreaking_multimodal_large_language_models_via_shuffle_inconsistency.md)
- [\[ACL 2026\] LaMI: Augmenting Large Language Models via Late Multi-Image Fusion](lami_augmenting_large_language_models_via_late_multi-image_fusion.md)
- [\[CVPR 2026\] Video-Only ToM: Enhancing Theory of Mind in Multimodal Large Language Models](../../CVPR2026/multimodal_vlm/video-only_tom_enhancing_theory_of_mind_in_multimodal_large_language_models.md)
- [\[CVPR 2026\] GroundVTS: Visual Token Sampling in Multimodal Large Language Models for Video Temporal Grounding](../../CVPR2026/multimodal_vlm/groundvts_visual_token_sampling_in_multimodal_large_language_models_for_video_te.md)

</div>

<!-- RELATED:END -->
