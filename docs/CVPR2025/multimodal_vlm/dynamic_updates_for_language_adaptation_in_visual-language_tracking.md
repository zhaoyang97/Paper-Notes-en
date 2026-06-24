---
title: >-
  [Paper Note] Dynamic Updates for Language Adaptation in Visual-Language Tracking
description: >-
  [CVPR 2025][Multimodal VLM][Visual-Language Tracking] DUTrack is proposed to resolve the semantic inconsistency between static references and dynamic targets in visual-language tracking by dynamically updating multi-modal reference information (template frames + language descriptions), outperforming the best vision-only trackers on LaSOT for the first time.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Visual-Language Tracking"
  - "Dynamic Update"
  - "Multi-modality Reference"
  - "Large Language Model"
  - "Object Tracking"
date: 2026-05-08
content_hash: c0ad32f6e6a7331b
---

# Dynamic Updates for Language Adaptation in Visual-Language Tracking

**Conference**: CVPR 2025  
**arXiv**: [2503.06621](https://arxiv.org/abs/2503.06621)  
**Code**: [https://github.com/GXNU-ZhongLab/DUTrack](https://github.com/GXNU-ZhongLab/DUTrack)  
**Area**: Video Understanding  
**Keywords**: Visual-Language Tracking, Dynamic Update, Multi-modality Reference, Large Language Model, Object Tracking

## TL;DR

DUTrack is proposed to resolve the semantic inconsistency between static references and dynamic targets in visual-language tracking by dynamically updating multi-modal reference information (template frames + language descriptions), outperforming the best vision-only trackers on LaSOT for the first time.

## Background & Motivation

Visual-language (VL) tracking relies on natural language descriptions and template frames to localize the target. However, existing methods suffer from a fundamental flaw: **the multi-modal reference information is static**. Specifically:

1. **Fixed language descriptions** — The initial language annotations can only describe the state of the target at a single moment, failing to reflect target appearance variations throughout the video (e.g., color, pose, and scale changes).
2. **Fixed template frames** — The initial template frame only captures the target's starting appearance, gradually deviating from the actual state during long-term tracking.

Consequently, the performance of VL trackers has lagged behind the state-of-the-art vision-only trackers (e.g., ODTrack, AQATrack), which wastes the language information during long-term sequences. The authors argue that the core reason lies in the **semantic gap between static references and dynamic targets**.

## Method

### Overall Architecture

DUTrack consists of four major components: (1) multi-modal interaction module — utilizing a one-stream architecture to process visual and language features in a unified manner; (2) Dynamic Template Capture Module (DTCM) — extracting regions from search frames that highly match the language as dynamic templates; (3) Dynamic Language Update Module (DLUM) — utilizing an LLM to generate dynamic language descriptions of the current target; and (4) tracking head — outputting bounding box predictions.

### Key Designs

1. **One-Stream Multi-modal Interaction**:
    - Function: Unified extraction and fusion of visual and language features.
    - Mechanism: Utilizing HiViT as the backbone, search and template frames are transformed into tokens $S_t \in \mathbb{R}^{N_S \times D}$ through a 3-stage downsampling ($4\times4$ embedding + two $2\times2$ merging). The language input is transformed into $L_t \in \mathbb{R}^{N_L \times D}$ ($N_L=16, D=512$) via a BERT tokenizer. These tokens are then concatenated and fed into unified multi-head self-attention for interaction.
    - Design Motivation: The one-stream architecture is more efficient than two-stream. Under the same ViT-base backbone, DUTrack achieves 43.5fps with 69.9M parameters, whereas JointNLT and MMTrack require 153M and 176.9M parameters, respectively.

2. **Dynamic Template Capture Module (DTCM)**:
    - Function: Capturing image regions from the search frame that highly match the language description to serve as the dynamic template update.
    - Mechanism: Utilizing the attention map of the [CLS] token on the search region in multi-head self-attention, $A_{l2s} = \text{Softmax}(\frac{Q_{CLS} \cdot K_S^T}{\sqrt{d}})$, the top-k patches with the highest attention scores are selected. The image regions corresponding to their index-aligned patches are used as the dynamic template. These patches represent the latest appearance of the target that best matches the current language description.
    - Design Motivation: Attention weights naturally encode the matching degree between language and vision, enabling high-quality dynamic templates to be obtained with almost zero extra computation; top-k=3 is the optimal choice.

3. **Dynamic Language Update Module (DLUM)**:
    - Function: Dynamically generating language annotations that describe the current state of the target during tracking.
    - Mechanism: A target change-based update strategy is designed. It compares the current tracking prediction $r_i: [x_2, y_2, w_2, h_2]$ with the record snapshot from the last update $r_{stamp}: [x_1, y_1, w_1, h_1]$ across three types of changes: scale change $\Delta S = \frac{w_1 h_1}{w_2 h_2}$, displacement change $\Delta D = \sqrt{(x_1-x_2)^2 + (y_1-y_2)^2}$, and color change $\Delta C = \sqrt{(R_1-R_2)^2 + (G_1-G_2)^2 + (B_1-B_2)^2}$. When the changes exceed predefined thresholds, BLIP is used to generate a new language description.
    - Design Motivation: Instead of updating the language description every frame (which would introduce excessive overhead), updates are triggered only when the target appearance changes significantly. This ensures information timeliness while controlling computational costs.

### Loss & Training

**Two-stage training**:
- **Stage 1** (150 epochs): Trained for vision-only tracking capability on LaSOT, GOT-10K, COCO, TrackingNet, and TNL2K without using language information. Uses the AdamW optimizer with a learning rate and weight decay of $1 \times 10^{-4}$, and 60K samples per epoch.
- **Stage 2** (50 epochs): Introduced the dynamic multi-modal reference update mechanism on LaSOT, GOT-10K, and TNL2K, using the language annotations generated by DTLLM-VLT as inputs.

During inference, top-k is set to 3, and BLIP is used as the LLM.

## Key Experimental Results

### Main Results

| Dataset | Metric | DUTrack-384 | Prev. Best VL | Best Vision-only | Gain |
|--------|------|------|----------|------|------|
| LaSOT | AUC | 74.1% | UVLTrack-L 71.3% | ODTrack 73.2% | +0.9% vs Vision-only |
| LaSOT | P | 82.9% | UVLTrack-L 78.3% | ODTrack 80.6% | +2.3% vs Vision-only |
| LaSOText | AUC | 52.5% | UVLTrack-L 51.2% | AQATrack 52.7% | Comparable |
| TNL2K | AUC | 65.6% | UVLTrack-L 64.8% | ODTrack 60.9% | +4.7% vs Vision-only |
| OTB99-Lang | AUC | 71.3% | MMTrack 70.5% | - | +0.8% |
| GOT-10K | AO | 77.8% | - | ODTrack 77.0% | +0.8% |

### Ablation Study

| Configuration | LaSOT AUC | LaSOT P | Description |
|------|---------|------|------|
| Baseline (No update) | 71.0% | 75.9% | Static reference |
| +DTCM (top-k=3) | 71.7% | 78.1% | Dynamic template +1.8% P |
| +DLUM (Static language) | 72.4% | 80.3% | Language information is effective |
| +DLUM (Dynamic, highest frequency) | 73.0% | 81.6% | Frequent updates perform best |
| BLIP as LLM | 73.0% | 81.6% | Concise generation is best |
| BLIP-2 | 73.2% | 81.7% | Slightly better |
| DTLLM-Detailed | 72.5% | 80.6% | Detailed descriptions inject noise instead |

### Key Findings

- **Historic Breakthrough**: DUTrack is the first to enable VL trackers to outperform the best vision-only trackers on LaSOT (74.1% vs ODTrack 73.2%), demonstrating that the dynamic update mechanism can truly unlock the potential of language information.
- Complementary effects of DTCM and DLUM: Using DTCM alone yields a +0.7% AUC improvement, and adding DLUM contributes an additional +1.3% AUC.
- Language description style: Concise styles perform better than detailed ones, as overly detailed descriptions tend to introduce redundant noise.
- Attention visualization indicates that static language annotations suffer from obvious attention misalignment, while dynamic language can rectify this mismatch.

## Highlights & Insights

- **Accurate Core Insight**: The fundamental reason why VL trackers were underperforming compared to vision-only trackers is not the lack of interaction designs, but rather the mismatch of static references. This is a very insightful problem discovery.
- **Practically Viable**: Features high practicality with an inference speed of 43.5fps and only 69.9M parameters.
- **Simple Module Design**: DTCM directly leverages existing attention maps with almost zero additional computation; DLUM's update strategy is based on simple displacement/scale/color changes.
- **Effective on Vision-only Benchmarks**: Language descriptions can still be generated and improve performance even on GOT-10K (which has no manual language annotations).

## Limitations & Future Work

- The thresholds for update frequency require manual tuning, lacking an adaptive mechanism.
- The quality of language descriptions generated by the LLM is constrained by the performance of BLIP.
- More complex update strategies (e.g., adaptive updates based on tracking confidence) have not been explored.
- The improvement on LaSOText is not pronounced, potentially due to fluctuations from the small size of the test set (only 150 sequences).

## Related Work & Insights

- Difference from dynamic reference trackers like STARK: STARK only updates visual templates, whereas DUTrack updates both visual and language references.
- Relationship with DTLLM-VLT: DTLLM also uses an LLM to generate language descriptions, but DUTrack integrates this into a complete dynamic update framework.
- Inspiration: In any task requiring long-term references (such as video object segmentation or re-identification), dynamically updating reference information is worth considering.

## Rating
- Novelty: ⭐⭐⭐⭐ Dynamically updating multi-modal references is a clear and effective paradigm, enabling VL tracking to outperform vision-only tracking for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐ Very comprehensive evaluation, featuring 6 benchmarks and multi-dimensional ablation analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and well-organized explanation of modules.
- Value: ⭐⭐⭐⭐ Points out the key direction of "dynamic reference" for the VL tracking community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Realistic Test-Time Adaptation of Vision-Language Models](realistic_test-time_adaptation_of_vision-language_models.md)
- [\[CVPR 2025\] Improving Personalized Search with Regularized Low-Rank Parameter Updates](improving_personalized_search_with_regularized_low-rank_parameter_updates.md)
- [\[CVPR 2025\] Rethinking Few-Shot Adaptation of Vision-Language Models in Two Stages](rethinking_few-shot_adaptation_of_vision-language_models_in_two_stages.md)
- [\[CVPR 2026\] MVLM: Template-Free Tracking via Vision-Language Margin Confidence and Memory-Gated Tracking](../../CVPR2026/multimodal_vlm/mvlm_template-free_tracking_via_vision-language_margin_confidence_and_memory-gat.md)
- [\[CVPR 2026\] Dynamic Logits Adjustment and Exploration for Test-Time Adaptation in Vision Language Models](../../CVPR2026/multimodal_vlm/dynamic_logits_adjustment_and_exploration_for_test-time_adaptation_in_vision_lan.md)

</div>

<!-- RELATED:END -->
