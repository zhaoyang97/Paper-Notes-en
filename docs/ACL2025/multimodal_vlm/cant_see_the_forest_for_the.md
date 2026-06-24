---
title: >-
  [Paper Note] MMSafeAware: Can't See the Forest for the Trees: Benchmarking Multimodal Safety Awareness for Multimodal LLMs
description: >-
  [ACL 2025][Multimodal VLM][multimodal safety] This work proposes MMSafeAware, the first multimodal safety awareness benchmark that simultaneously evaluates "unsafe content identification" and "over-sensitivity." It contains 1,500 image-text pairs across 29 safety scenarios. Evaluating 9 MLLMs reveals that all models suffer from a severe trade-off between safety and helpfulness—GPT-4V misclassifies `$36.1\%$` of unsafe inputs as safe while misclassifying `$59.9\%$$` of safe in…
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "multimodal safety"
  - "safety awareness"
  - "over-sensitivity"
  - "benchmark"
  - "MLLM"
  - "helpfulness-harmlessness trade-off"
date: 2026-05-08
content_hash: 84efdb5614c96a45
---

# MMSafeAware: Can't See the Forest for the Trees: Benchmarking Multimodal Safety Awareness for Multimodal LLMs

**Conference**: ACL 2025  
**arXiv**: [2502.11184](https://arxiv.org/abs/2502.11184)  
**Code**: [https://github.com/Jarviswang94/MMSafetyAwareness](https://github.com/Jarviswang94/MMSafetyAwareness)  
**Area**: Multimodal Safety  
**Keywords**: multimodal safety, safety awareness, over-sensitivity, benchmark, MLLM, helpfulness-harmlessness trade-off

## TL;DR

This work proposes MMSafeAware, the first multimodal safety awareness benchmark that simultaneously evaluates "unsafe content identification" and "over-sensitivity." It contains 1,500 image-text pairs across 29 safety scenarios. Evaluating 9 MLLMs reveals that all models suffer from a severe trade-off between safety and helpfulness—GPT-4V misclassifies `$36.1\%$` of unsafe inputs as safe while misclassifying `$59.9\%$$` of safe inputs as unsafe. None of the three mitigation methods can fundamentally resolve this issue.

## Background & Motivation

**Definition and Importance of Multimodal Safety Awareness**: MLLMs should correctly identify the safety of multimodal content, which is the first step to prevent generating unsafe responses and a prerequisite for MLLMs to serve as safety evaluators. However, a systematic evaluation framework is currently lacking.

**New Challenges from Cross-modal Semantic Fusion**: Individually harmless images and texts may convey unsafe information when combined (e.g., memes), and vice versa. This requires MLLMs to not only understand each modality but also effectively fuse cross-modal information to judge safety.

**Over-sensitivity Neglected**: Existing safety benchmarks (e.g., MM-Safety, HateMemes) only evaluate the ability to detect unsafe content, neglecting the over-sensitivity of models. Over-sensitivity leads to refusing a large number of safe requests, severely undermining helpfulness.

**Incomplete Coverage of Existing Benchmarks**: Most safety benchmarks are text-only or image-only. Even multimodal ones cover few scenarios (e.g., MM-Safety only has 13 categories; MossBench has only 3 over-safety scenarios) and fail to concurrently cover three major dimensions: typical unsafe content, instruction attacks, and over-safety.

**Double-Edged Sword Effect of Safety System Prompts**: Safety-oriented system prompts (such as "please be safe") are commonly used in deployment, but their impact on over-sensitivity has not been systematically studied.

**Unknown Effectiveness of Mitigation Methods**: Whether methods like prompt engineering, Visual Contrastive Decoding (VCD), and vision-centric reasoning fine-tuning can simultaneously improve safety identification and reduce over-sensitivity needs to be validated on a comprehensive benchmark.

## Method

### Overview

MMSafeAware consists of two subsets: (1) An unsafe subset (1,000 image-text pairs) that tests whether MLLMs can identify content that becomes unsafe when combined (measuring harmlessness); (2) An over-safety subset (500 image-text pairs) that tests whether MLLMs raise false alarms on actually safe content (measuring helpfulness). It covers 29 safety scenarios, and all data have been manually audited by three annotators.

### Key Designs

1. **Construction Principles of the Unsafe Subset**

    - **Function**: Each test case is composed of an image and text that are individually harmless but express unsafe information when combined. It covers 17 unsafe scenarios, including 14 typical scenarios (such as self-harm, pornography, financial crimes, and hate speech under four major dimensions: physical, psychological, property, and social safety) and 3 instruction attack scenarios (role-play, inquiry with unsafe views, and goal hijacking).
    - **Mechanism**: The safety of the same image or text varies drastically depending on how it is paired. This forces the MLLM to fuse both modalities to make a correct judgment instead of relying on a single modality.
    - **Design Motivation**: Real-world multimodal unsafe content (e.g., memes, illustrated news) often conveys dangerous information through cross-modal combinations rather than a single modality. This construction is more aligned with real-world threats.

2. **Construction Principles of the Over-safety Subset**

    - **Function**: Each test case contains an image or text that individually appears unsafe but is actually safe when combined. It covers 12 over-safety scenarios, including 8 adapted from XSTest (definitions, homophones, safe targets, safe contexts, real discrimination against non-existent groups, etc.) and 4 newly designed scenarios (tautology, public domain copyright, pseudo-goal hijacking, and pseudo-role-play).
    - **Mechanism**: An ideal MLLM needs to balance helpfulness and harmlessness—over-sensitivity is equivalent to degrading helpfulness.
    - **Design Motivation**: XSTest introduced over-sensitivity at the text level; this work extends it to multimodal scenarios for the first time.

3. **Three Mitigation Methods**

    - **Prompting** (applicable to closed-source models): Explicitly instructs the model to "consider the meaning of the text within the context of the image," encouraging cross-modal fusion.
    - **Visual Contrastive Decoding (VCD)** (applicable to open-source models): Contrasts the output distributions from clean and noisy visual inputs to strengthen the model's focus on visual information.
    - **Vision-Centric Reasoning Fine-tuning (VRTuning)** (applicable to open-source models): Fine-tunes the model on long-thought multimodal reasoning datasets, introducing structured intermediate reasoning steps to jointly analyze images and text.
    - **Design Motivation**: Case studies suggest that the core reason for MLLM failures is over-reliance on a single modality (typically text). These three methods encourage cross-modal information fusion from the prompt, decoding, and training levels, respectively.

## Key Experimental Results

### Table 1: Main Results — Safety Awareness Accuracy of 9 MLLMs

| Model | Typical Unsafe ↑ | Attack ↑ | Over-safety ↑ | Overall ↑ |
|------|:---:|:---:|:---:|:---:|
| GPT-4V | 63.9 | 68.4 | 41.1 | 57.8 |
| GPT-4o | 81.3 | 88.7 | 25.0 | 65.0 |
| Gemini 1.5 | 86.6 | 81.5 | 18.5 | 62.2 |
| Gemini 1.5 Pro | 81.2 | 74.2 | 40.8 | 65.4 |
| Bard | 73.8 | 61.4 | 28.6 | 54.6 |
| Claude-3 | **100.0** | **99.1** | **1.1** | 66.7 |
| LLaVA-1.5-7B | 95.9 | 97.7 | 6.0 | 66.5 |
| Qwen-VL-7B | 86.5 | 95.2 | 13.7 | 65.1 |
| InstructBLIP | 66.1 | 43.9 | 20.5 | 43.5 |
| **Human** | **90.7** | **92.8** | **95.2** | **92.9**|

### Table 2: Impact of Safety System Prompts

| Model | Unsafe ↑ | Over-safety ↑ | Overall ↑ |
|------|:---:|:---:|:---:|
| GPT-4V | 68.2 → 70.4 (+2.2) | 36.0 → 32.1 (-3.9) | 57.7 → 57.6 |
| GPT-4o | 86.6 → 88.2 (+1.6) | 22.7 → 21.9 (-0.8) | 65.7 → 66.1 |
| Gemini 1.5 | 82.8 → 84.5 (+1.7) | 29.9 → 25.6 (-4.3) | 65.5 → 64.9 |
| Gemini 1.5 Pro | 75.0 → 80.9 (+5.9) | 39.3 → 31.2 (-8.1) | 63.3 → 64.3 |

### Table 3: Effects of Three Mitigation Methods

| Model + Method | Unsafe ↑ | Over-safety ↑ | Overall ↑ |
|------|:---:|:---:|:---:|
| GPT-4V + Prompt | 68.6 | 42.1 | 59.9 |
| GPT-4o + Prompt | 87.9 | 28.4 | 68.5 |
| Gemini 1.5 + Prompt | 89.8 | 39.4 | 73.3 |
| LLaVA + VCD | 88.2 | 15.3 | 63.9 |
| LLaVA + VRTuning | 81.5 | 17.3 | 60.1 |
| Qwen-VL + VCD | 82.5 | 20.1 | 61.7 |
| Qwen-VL + VRTuning | 58.1 | 35.6 | 50.6 |
| InstructBLIP + VRTuning | 70.6 | 29.6 | 56.9 |

## Key Findings

- **All MLLMs Are Insufficiently Safe**: GPT-4V misclassifies `$36.1\%$` of unsafe inputs as safe, and Bard misclassifies `$26.2\%$`—making them clearly unreliable as safety evaluators.
- **Over-sensitivity Is More Severe Than Safety Deficiencies**: Claude-3 is near-perfect on the unsafe subsets (`$100\%$` / `$99.1\%$`), but achieves only `$1.1\%$` on the over-safety subset, meaning it rejects almost everything. LLaVA achieves only `$6.0\%$`.
- **Fundamental Trade-off Between Safety and Helpfulness**: No single model performs well on both subsets concurrently. Claude-3 is the safest but most over-sensitive, while GPT-4V is relatively moderate but lacks safety.
- **Safety System Prompts Are a Double-Edged Sword**: Adding "please be safe" increases the accuracy on unsafe subsets by `$1.6 \sim 5.9$` percentage points, but decreases it on over-safety subsets by `$0.8 \sim 8.1$` percentage points—making the prompt counterproductive for helpfulness.
- **Attention Analysis Reveals the Root Cause of Failures**: In failure cases, LLaVA allocates much higher attention to text tokens (such as "kill", "I", "you") than to image tokens, indicating a tendency of the model to ignore visual information.
- **All Three Mitigation Methods Are Insufficient**: Prompting helps closed-source models to some extent (an overall improvement of `$7.8\%$` for Gemini 1.5), whereas VCD and VRTuning show limited or even negative effects on open-source models—the issue essentially remains unresolved.
- **Humans Far Outperform All Models**: Humans achieve `$90\%+$` across all three dimensions (overall `$92.9\%$`), outperforming the best model by more than 25 percentage points.

## Highlights & Insights

- **The Dual-Subset Design ("Unsafe + Over-safe")** precisely defines a complete evaluation framework for multimodal safety awareness—translating the helpfulness-harmlessness trade-off from theoretical discussions into a quantifiable benchmark for the first time.
- **Comprehensive Coverage of 29 Safety Scenarios**: Far exceeding prior works such as MM-Safety (13) and MossBench (3), encompassing three major dimensions: typical unsafe content, instruction attacks, and over-safety.
- **The Finding That "Safety System Prompts Exacerbate Over-sensitivity"**: This offers an important warning for deployment—blindly adding safety prompts can backfire.
- **Five Failure Modes from GPT-4V Case Analysis**: Correct/partially correct, factual error, misunderstanding images, over-alignment, and goal-hijacked—providing a diagnostic framework for future improvements.

## Limitations & Future Work

- The safety judgment criteria for image-text combinations carry some subjectivity, despite quality control via three-annotator labeling and majority voting (where `$4.7\%$` of the data was discarded).
- None of the three mitigation methods fundamentally solve the problem—this work focuses more on "defining the problem" rather than "solving the problem".
- Only 9 MLLMs were evaluated; the performance of newer-generation models (e.g., GPT-4o-mini, Claude-3.5) remains unknown.
- The over-safety subset (500 samples) is relatively small compared to the unsafe subset (1,000 samples), resulting in a limited number of samples for some scenarios.
- Image sources are from Google Images (Creative Commons), which may introduce distribution bias.

## Related Work & Insights

- **Text-only Safety Benchmarks**: Such as SafetyBench (6 scenarios), XSTest (first to focus on over-sensitivity), and SafetyAssessBench (including attack scenarios), but they do not involve multimodal safety.
- **Image-only Safety Benchmarks**: Such as ChemiSafety and ViolenceBench, which only focus on image content itself and fail to capture safety concerns arising from cross-modal combinations.
- **Multimodal Safety Benchmarks**: Such as HateMemes (hate memes, 1 category), MM-Safety (13 categories but lacks over-safety), HADES (5 categories), and MossBench (3 over-safety categories but lacks an unsafe subset). MMSafeAware is the first to cover both unsafe and over-safe aspects, with a scenario count (29) that far exceeds all prior works.
- **Multimodal Content Understanding**: Discussions on feature-level vs. decision-level fusion. The advantage of decision-level fusion lies in using the most suitable method for each modality, whereas feature-level fusion can better capture cross-modal correlations.
- **Safety Improvements for MLLMs**: Wang et al. (2024) utilized system prompts to enhance safety, while VCD (Leng et al., 2024) reinforced visual attention through contrastive decoding, though these remain localized improvements.

## Rating

- Novelty: ⭐⭐⭐⭐ First multimodal benchmark to evaluate both safety and over-sensitivity simultaneously, with comprehensive coverage of 29 scenarios.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation of 9 models + 3 mitigation methods + case analysis + attention analysis + safety prompt experiments.
- Writing Quality: ⭐⭐⭐⭐ Precise problem definition, clear logic in the dual-subset design, and insightful case analysis.
- Value: ⭐⭐⭐⭐⭐ Quantitatively revealing the "safety vs. helpfulness" trade-off carries important practical significance for MLLM deployment and defines a new research direction of multimodal safety awareness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Don't Miss the Forest for the Trees: Attentional Vision Calibration for Large Vision Language Models](dont_miss_the_forest_for_the_trees_attentional_vision_calibration_for_large_visi.md)
- [\[ACL 2025\] Can LLMs Deceive CLIP? Benchmarking Adversarial Compositionality of Pre-trained Multimodal Representation via Text Updates](adversarial_compositionality_clip.md)
- [\[ACL 2025\] VLSBench: Unveiling Visual Leakage in Multimodal Safety](vlsbench_unveiling_visual_leakage_in_multimodal_safety.md)
- [\[ACL 2025\] Finding Needles in Images: Can Multi-modal LLMs Locate Fine Details?](finding_needles_in_images_can_multi-modal_llms_locate_fine_details.md)
- [\[ICCV 2025\] Vision-Language Models Can't See the Obvious](../../ICCV2025/multimodal_vlm/vision-language_models_cant_see_the_obvious.md)

</div>

<!-- RELATED:END -->
