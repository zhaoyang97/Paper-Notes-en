---
title: >-
  [Paper Note] Bootstrapping MLLM for Weakly-Supervised Class-Agnostic Object Counting (WS-COC)
description: >-
  [ICLR 2026][Object Detection][object counting] This paper proposes WS-COC, the first MLLM-based weakly supervised class-agnostic object counting framework. Through three strategies — divide-and-discern dialogue tuning (p…
tags:
  - "ICLR 2026"
  - "Object Detection"
  - "object counting"
  - "weakly supervised"
  - "MLLM"
  - "class-agnostic"
  - "dialogue tuning"
date: 2026-05-08
content_hash: ae0c63e4c19c188e
---

# Bootstrapping MLLM for Weakly-Supervised Class-Agnostic Object Counting (WS-COC)

**Conference**: ICLR 2026  
**arXiv**: [2602.12774](https://arxiv.org/abs/2602.12774)  
**Code**: [https://github.com/viscom-tongji/WS-COC](https://github.com/viscom-tongji/WS-COC)  
**Area**: Multimodal VLM  
**Keywords**: object counting, weakly supervised, MLLM, class-agnostic, dialogue tuning  

## TL;DR
This paper proposes WS-COC, the first MLLM-based weakly supervised class-agnostic object counting framework. Through three strategies — divide-and-discern dialogue tuning (progressively narrowing the counting range), comparative ranking optimization (learning relative counting relationships across images), and global-local counting enhancement — WS-COC achieves performance comparable to or surpassing fully supervised methods using only image-level count annotations.

## Background & Motivation
**Background**: Object counting has traditionally relied on fully supervised density map regression with point-level annotations, which is prohibitively costly. Weakly supervised methods use only image-level counts but remain limited to specific categories (e.g., crowd counting).

**Limitations of Prior Work**: (1) Fully supervised methods require annotating the location of every object instance, which is extremely time-consuming in dense scenes; (2) existing weakly supervised methods are CNN/ViT-based and category-specific; (3) MLLMs possess latent counting ability but severely underestimate counts in dense scenes, as directly predicting a single number is too difficult.

**Key Challenge**: MLLM pretraining data predominantly consists of sparse scenes, leaving the model with insufficient quantity perception for dense scenes. Directly fine-tuning MLLMs to regress count values faces a modality gap — the mapping from high-dimensional visual features to a discrete scalar is difficult to learn.

**Goal**: How to leverage the reasoning capabilities of MLLMs to achieve class-agnostic object counting using only image-level count annotations?

**Key Insight**: Rather than directly predicting count values, the task is decomposed into more learnable sub-tasks — range estimation (binary-search-style narrowing) and relative comparison (cross-image ranking).

**Core Idea**: Reformulate counting from "predict a number" into three sub-tasks more amenable to MLLMs: range judgment, relative ranking, and local aggregation.

## Method

### Overall Architecture
WS-COC fine-tunes LLaVA-OneVision with LoRA, employing D3T and CRCO strategies during training and the GLCE strategy during inference. Only image-level count annotations are required as supervision.

### Key Designs

1. **Divide-and-Discern Dialogue Tuning (D3T)**:

    - **Function**: Converts precise counting into a multi-turn range-judgment dialogue.
    - **Mechanism**: Given an initial range [1, 2000], each turn poses a binary question — "Does the number of [obj] in the image exceed τ?" — with a Yes/No answer that updates the range. When the range narrows to $U_t - L_t < 0.2c$, the model is prompted to predict the exact count. Curriculum learning is applied to progress from coarse to fine granularity.
    - **Design Motivation**: Judging whether a count exceeds a threshold is substantially easier than predicting the exact number. Multi-turn dialogue enables the MLLM to progressively focus its estimation.

2. **Compare-and-Rank Count Optimization (CRCO)**:

    - **Function**: Trains the MLLM to judge relative count rankings across multiple images.
    - **Mechanism**: Images of the same category are divided into 4 count intervals; one image is sampled from each interval to form an image set (covering both sparse and dense cases). The images are shuffled and the MLLM is asked to output them in ascending order: "Image i < ... < Image j".
    - **Design Motivation**: Judging "which image contains more objects" is more aligned with visual intuition than predicting absolute numbers, thereby alleviating the modality gap.

3. **Global-and-Local Counting Enhancement (GLCE)**:

    - **Function**: Fuses global and local count predictions at inference time.
    - **Mechanism**: The model first predicts a global count $c^g$. If $c^g > c^h$ (threshold 100), the image is partitioned into a 2×2 grid; each patch is counted independently and summed to obtain $c^l$; the final prediction is $(c^g + c^l) / 2$.
    - **Design Motivation**: Global counting underestimates in dense scenes, while local counting overestimates due to boundary effects — averaging the two provides complementary correction.

### Loss & Training
Standard language modeling cross-entropy loss. LLaVA-OneVision-7B with LoRA (rank=128). Trained on FSC-147.

## Key Experimental Results

### Main Results (FSC-147 Test Set MAE↓)

| Method | Supervision | MAE↓ | RMSE↓ |
|--------|-------------|------|-------|
| CLIP-Count | Fully supervised (point annotations) | 17.78 | 106.62 |
| T2ICount | Fully supervised | ~strong | ~strong |
| CountGD | Fully supervised | ~strong | ~strong |
| **WS-COC** | **Weakly supervised (image-level)** | **~on par with fully supervised** | **~on par** |
| WS-COC-Base (direct fine-tuning) | Weakly supervised | High | High |
| MLLM-Zero (no fine-tuning) | Zero-shot | Very high | Very high |

### Ablation Study

| Configuration | Performance |
|---------------|-------------|
| Base only | Severe underestimation in dense scenes |
| + D3T | Significant improvement |
| + D3T + CRCO | Further improvement |
| + D3T + CRCO + GLCE | Best overall, especially in dense scenes |

### Key Findings
- WS-COC under weak supervision matches or surpasses multiple fully supervised methods — a disruptive gain in annotation efficiency.
- D3T's dialogue-based bisection yields the largest improvement in dense scenes — task reformulation from direct regression to range judgment is the critical factor.
- CRCO's relative ranking learning is particularly effective at building cross-magnitude quantity perception.
- Cross-dataset generalization (FSC-147 → CARPK/PUCPR+/ShanghaiTech) is consistently strong.
- In sparse scenes with fewer than 20 objects, the MLLM zero-shot baseline already achieves reasonable accuracy.

## Highlights & Insights
- **Task reformulation as the core contribution**: Rather than designing better visual features, the paper reformulates "predicting a number" into sub-tasks more suited to MLLMs (judgment, comparison, divide-and-conquer). This paradigm is generalizable to other VLM applications requiring numerical regression.
- **Weak supervision reaching fully supervised levels**: A significant breakthrough in object counting — the high cost of point-level annotation may no longer be necessary.
- **Elegant application of dialogue-based reasoning**: Leveraging the multi-turn dialogue capability of MLLMs to perform "binary search" is a creative exploitation of MLLM interaction abilities.

## Limitations & Future Work
- The simple averaging in GLCE may be suboptimal — adaptive fusion weights could be learned instead.
- The 2×2 partitioning may still be insufficient for extremely dense scenes — hierarchical subdivision warrants exploration.
- The method relies on object category names as text prompts, which may limit applicability to unknown or hard-to-name categories.
- The counting threshold $c^h=100$ is manually set.

## Related Work & Insights
- **vs. fully supervised counting methods (CounTR, CountGD)**: WS-COC achieves comparable performance without point-level annotations.
- **vs. CrowdCLIP (ranking strategy)**: CrowdCLIP constructs ranking pairs by cropping the same image, whereas WS-COC uses different images — a more principled approach.
- **vs. AQuA (handling VLM uncertainty)**: WS-COC's dialogue-based bisection can be viewed as an alternative strategy for managing VLM uncertainty under numerical prediction.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ All three strategies exhibit strong creativity in task reformulation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four benchmarks with detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear with intuitive illustrations.
- Value: ⭐⭐⭐⭐⭐ Achieving fully supervised performance under weak supervision is highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SPWOOD: Sparse Partial Weakly-Supervised Oriented Object Detection](spwood_sparse_partial_weakly-supervised_oriented_object_detection.md)
- [\[ICLR 2026\] CGSA: Class-Guided Slot-Aware Adaptation for Source-Free Object Detection](cgsa_class-guided_slot-aware_adaptation_for_source-free_object_detection.md)
- [\[AAAI 2026\] CountSteer: Steering Attention for Object Counting in Diffusion Models](../../AAAI2026/object_detection/countsteer_steering_attention_for_object_counting_in_diffusion_models.md)
- [\[ICCV 2025\] Toward Long-Tailed Online Anomaly Detection through Class-Agnostic Concepts](../../ICCV2025/object_detection/toward_long-tailed_online_anomaly_detection_through_class-agnostic_concepts.md)
- [\[AAAI 2026\] TubeRMC: Tube-conditioned Reconstruction with Mutual Constraints for Weakly-supervised Spatio-Temporal Video Grounding](../../AAAI2026/object_detection/tubermc_tube-conditioned_reconstruction_with_mutual_constraints_for_weakly-super.md)

</div>

<!-- RELATED:END -->
