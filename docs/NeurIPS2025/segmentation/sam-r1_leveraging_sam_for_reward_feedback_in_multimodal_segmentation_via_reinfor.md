---
title: >-
  [Paper Note] SAM-R1: Leveraging SAM for Reward Feedback in Multimodal Segmentation via Reinforcement Learning
description: >-
  [NeurIPS 2025][Segmentation][Reasoning Segmentation] SAM-R1 proposes an end-to-end reasoning segmentation framework that, for the first time…
tags:
  - "NeurIPS 2025"
  - "Segmentation"
  - "Reasoning Segmentation"
  - "Reinforcement Learning"
  - "SAM"
  - "Multimodal Large Language Models"
  - "GRPO"
date: 2026-05-08
content_hash: 08d7b9b863e92b14
---

# SAM-R1: Leveraging SAM for Reward Feedback in Multimodal Segmentation via Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2505.22596](https://arxiv.org/abs/2505.22596)  
**Code**: None  
**Area**: Segmentation
**Keywords**: Reasoning Segmentation, Reinforcement Learning, SAM, Multimodal Large Language Models, GRPO

## TL;DR

SAM-R1 proposes an end-to-end reasoning segmentation framework that, for the first time, incorporates SAM as a reward provider within the reinforcement learning training loop. Combined with a tiered IoU accuracy reward, asymmetric clipping, and token-level loss normalization in an improved GRPO algorithm, the method achieves a gIoU of 60.2% on the ReasonSeg zero-shot benchmark—surpassing Seg-Zero and other approaches—using only 3K training samples.

## Background & Motivation

Multimodal large language models (MLLMs) have achieved substantial progress in visual understanding, with capabilities extending to reasoning segmentation—a task requiring models to interpret implicit user queries and perform logical reasoning to produce pixel-level segmentation masks. LISA first demonstrated the feasibility of connecting MLLMs with segmentation models via special tokens. However, existing approaches suffer from several critical issues:

**High data cost**: These methods rely on large-scale annotated datasets for joint fine-tuning, and acquiring annotations that include explicit reasoning chains is particularly expensive.

**Catastrophic forgetting**: Models trained via SFT perform well in-domain but generalize poorly out-of-domain.

**Insufficient reasoning ability**: Models struggle to accurately interpret intent when faced with ambiguous or complex textual queries.

Recent research demonstrates that reinforcement learning can endow large models with reasoning capabilities without requiring annotated reasoning data (e.g., DeepSeek-R1 using rule-based rewards). Seg-Zero has introduced GRPO into reasoning segmentation, but its reasoning model and segmentation decoder are fully decoupled, preventing pixel-level feedback and increasing the risk of reward hacking.

**Core insight**: Directly integrating the segmentation model (SAM) into the RL training loop as a reward provider, enabling the MLLM to receive fine-grained, task-relevant segmentation feedback within an end-to-end framework.

## Method

### Overall Architecture

The SAM-R1 pipeline proceeds as follows:
1. **Input**: User query + image → MLLM (Qwen2.5VL-7B)
2. **Reasoning**: The MLLM generates a chain-of-thought within `<think>...</think>` tags and a structured answer within `<answer>...</answer>` tags (containing bounding boxes, reference points, and textual descriptions)
3. **Segmentation**: MLLM outputs are passed to SAM2-Large to generate segmentation masks
4. **Reward computation**: Predicted masks are compared against ground-truth masks via IoU to compute tiered accuracy rewards
5. **Policy update**: The MLLM's reasoning policy is optimized based on the received rewards

### Key Designs

1. **Tiered Segmentation Accuracy Reward**: Rather than a simple binary reward or continuous IoU value, a piecewise reward scheme is adopted: IoU > 0.8 yields 4 points, 0.7–0.8 yields 3 points, 0.5–0.7 yields 2 points, and otherwise 0 points. This staircase reward provides incremental improvement signals at low IoU levels while reserving strong positive feedback for predictions that closely match the ground truth.

2. **Improved GRPO Objective**:

    - **Asymmetric clipping**: The single threshold $\varepsilon$ in PPO is replaced with asymmetric bounds $\varepsilon_{\text{low}}$ and $\varepsilon_{\text{high}}$ ($\varepsilon_{\text{high}} = 0.3$, more permissive), while $\varepsilon_{\text{low}}$ is kept unchanged. This allows more aggressive updates when high-advantage actions are identified, while maintaining KL divergence constraints to ensure training stability.
    - **Token-level loss normalization**: In vanilla GRPO, long and short responses incur the same total loss, meaning each token in a longer response is penalized less, incentivizing verbose and low-information outputs. SAM-R1 changes the normalization factor from $\frac{1}{G} \frac{1}{|o_i|}$ to $\frac{1}{\sum_{i=1}^{G}|o_i|}$, ensuring uniform per-token loss and suppressing redundant repetitive outputs.

3. **Three-component Reward Function**:

    - Tiered segmentation accuracy reward (IoU-based, computed from SAM outputs)
    - Reasoning format reward (verifying the think/answer tag structure)
    - Segmentation format reward (verifying JSON compliance for bounding boxes, reference points, and textual markers)

### Loss & Training

Qwen2.5VL-7B serves as the base model and SAM2-Large as the segmentation model, trained on 8×A100 GPUs. Eight responses are sampled per question, with a learning rate of $1.0 \times 10^{-6}$. All images are uniformly resized to $840 \times 840$. Training uses only 3,000 randomly sampled examples from the RefCOCOg training set.

## Key Experimental Results

### Main Results — ReasonSeg Zero-Shot

| Method | val gIoU | val cIoU | test gIoU | test cIoU |
|--------|----------|----------|-----------|-----------|
| LISA-7B | 53.6 | 52.3 | 48.7 | 48.8 |
| LISA-13B | 57.7 | 60.3 | 53.8 | 50.8 |
| Seg-Zero-7B* | 62.0 | 52.0 | 58.3 | 53.4 |
| **SAM-R1** | **64.0** | **55.8** | **60.2** | **54.3** |

### Referring Expression Segmentation (cIoU)

| Method | RefCOCO | RefCOCO+ | RefCOCOg |
|--------|---------|----------|----------|
| LISA-7B | 76.5 | 67.4 | 68.5 |
| PixelLM-7B | 76.5 | 71.7 | 70.5 |
| PerceptionGPT-7B | 78.6 | 73.9 | 71.7 |
| Seg-Zero-7B* | 79.2 | 73.9 | 73.3 |
| **SAM-R1** | **79.2** | **74.7** | 73.1 |

### Ablation Study — Threshold Strategy

| Strategy | ReasonSeg gIoU | ReasonSeg cIoU | RefCOCOg gIoU | RefCOCOg cIoU |
|----------|---------------|---------------|---------------|---------------|
| Fixed 0.5 | 56.5 | 51.9 | 74.7 | 72.8 |
| Fixed 0.7 | 56.2 | 51.6 | 74.9 | 72.6 |
| Fixed 0.8 | 58.6 | 50.8 | 74.6 | 71.9 |
| **Tiered** | **60.2** | **54.3** | **75.4** | **73.1** |

### Ablation Study — Algorithm Components

| Method | Token-level | High Clip | ReasonSeg gIoU | ReasonSeg cIoU | RefCOCOg gIoU | RefCOCOg cIoU |
|--------|-------------|-----------|---------------|---------------|---------------|---------------|
| GRPO Baseline | - | - | 57.8 | 51.2 | 74.1 | 71.8 |
| +Token-level | ✓ | - | 58.0 | 51.7 | 74.5 | 72.4 |
| +High Clip | - | ✓ | 59.1 | 52.8 | 74.9 | 72.5 |
| **Full** | ✓ | ✓ | **60.2** | **54.3** | **75.4** | **73.1** |

### Key Findings

- The tiered threshold strategy outperforms any fixed threshold, with particularly notable OOD gains over the fixed 0.8 threshold (gIoU +1.6, cIoU +3.5).
- Asymmetric high clipping yields greater benefits on OOD reasoning tasks (ReasonSeg gIoU +1.3 vs. RefCOCOg +0.8).
- The two improvements exhibit strong synergy: combined use improves ReasonSeg cIoU by 3.1%, exceeding the sum of their individual contributions.
- 3K training samples prove sufficient—scaling to 10K yields negligible performance change, demonstrating exceptional data efficiency.
- The model achieves 63.8 on the unseen LISA-Grounding benchmark (previous best: 43.9), confirming strong generalization.
- Removing the KL constraint causes training collapse at approximately 100 steps, underscoring its critical role in stability.

## Highlights & Insights

- Elevating SAM from a "downstream segmentation tool" to a "reward provider in RL training" represents the central conceptual innovation.
- The tiered IoU reward design is better suited to RL training than continuous IoU—it provides a clear hierarchical gradient of objectives.
- Asymmetric clipping combined with token-level normalization constitutes a refined and effective improvement over standard GRPO.
- The data efficiency of achieving strong performance with only 3K samples is particularly noteworthy.
- The paper transparently reports several failed attempts (training collapse upon KL removal, failure to generate meaningful negative reference points), contributing valuable negative results to the community.

## Limitations & Future Work

- SAM parameters are frozen and information flows unidirectionally—the segmentation strategy cannot adapt to the reasoning model's needs.
- The model fails to generate meaningful negative reference points, and the RL framework does not incentivize this capability.
- Incomplete segmentation and over-segmentation persist in the high-IoU regime (correct reasoning but insufficient mask precision).
- Experiments are conducted solely on Qwen2.5VL-7B, leaving applicability to other MLLM backbones unverified.
- More sophisticated reward combination strategies (e.g., incorporating semantic consistency rewards) remain unexplored.

## Related Work & Insights

- SAM-R1 extends the DeepSeek-R1 → VLM-R1 → Seg-Zero technical lineage, representing a natural progression of RL-enhanced multimodal reasoning.
- The key distinction from Seg-Zero lies in end-to-end integration versus decoupled design—the end-to-end approach fundamentally reduces reward hacking.
- Promising future directions include joint optimization of SAM and the reasoning model, more sophisticated tiered reward schemes, and extension of the framework to video segmentation.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FineRS: Fine-grained Reasoning and Segmentation of Small Objects with Reinforcement Learning](finers_fine-grained_reasoning_and_segmentation_of_small_objects_with_reinforceme.md)
- [\[ICCV 2025\] E-SAM: Training-Free Segment Every Entity Model](../../ICCV2025/segmentation/e-sam_training-free_segment_every_entity_model.md)
- [\[AAAI 2026\] SAQ-SAM: Semantically-Aligned Quantization for Segment Anything Model](../../AAAI2026/segmentation/saq-sam_semantically-aligned_quantization_for_segment_anything_model.md)
- [\[ICCV 2025\] SAM2Long: Enhancing SAM 2 for Long Video Segmentation with a Training-Free Memory Tree](../../ICCV2025/segmentation/sam2long_enhancing_sam_2_for_long_video_segmentation_with_a.md)
- [\[ICCV 2025\] HiMTok: Learning Hierarchical Mask Tokens for Image Segmentation with Large Multimodal Model](../../ICCV2025/segmentation/himtok_learning_hierarchical_mask_tokens_for_image_segmentation_with_large_multi.md)

</div>

<!-- RELATED:END -->
