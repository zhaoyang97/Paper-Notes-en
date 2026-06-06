---
title: >-
  [Paper Note] Vanish into Thin Air: Cross-prompt Universal Adversarial Attacks for SAM2
description: >-
  [NeurIPS 2025][Segmentation][Adversarial Attack] This paper proposes UAP-SAM2—the first cross-prompt universal adversarial attack against SAM2—which employs a dual semantic shift framework (intra-frame semantic confusion…
tags:
  - "NeurIPS 2025"
  - "Segmentation"
  - "Adversarial Attack"
  - "SAM2"
  - "Universal Adversarial Perturbation"
  - "Video Segmentation"
  - "Semantic Shift"
date: 2026-05-08
content_hash: 7c1066435d745167
---

# Vanish into Thin Air: Cross-prompt Universal Adversarial Attacks for SAM2

**Conference**: NeurIPS 2025
**arXiv**: [2510.24195](https://arxiv.org/abs/2510.24195)  
**Code**: [GitHub](https://github.com/CGCL-codes/UAP-SAM2)  
**Area**: Image Segmentation
**Keywords**: Adversarial Attack, SAM2, Universal Adversarial Perturbation, Video Segmentation, Semantic Shift

## TL;DR

This paper proposes UAP-SAM2—the first cross-prompt universal adversarial attack against SAM2—which employs a dual semantic shift framework (intra-frame semantic confusion + inter-frame semantic inconsistency) to generate a single universal perturbation that causes segmentation targets to "vanish" across different videos, frames, and prompts.

## Background & Motivation

SAM2 is an upgraded version of SAM that extends image segmentation to video segmentation via a memory mechanism: users provide a prompt only in the first frame, and SAM2 continuously tracks and segments the target in subsequent frames. Although the adversarial robustness of SAM has been extensively studied, the robustness of SAM2 remains unexplored.

The authors first empirically demonstrate that **existing SAM attack methods cannot be directly transferred to SAM2**. For example, DarkSAM degrades SAM performance by 98.25%, yet causes only a 22.26% drop against SAM2. This substantial gap stems from two key architectural differences between SAM and SAM2:

**Directional Guidance from the Prompt**: SAM receives an independent prompt for each frame, whereas SAM2 receives a prompt only in the first frame and persistently stores it for reuse in subsequent frames. Even if the first frame is successfully attacked, the perturbation is difficult to propagate to subsequent frames. Experiments show that even with a perturbation budget of 32/255, attacking only the first frame fails to significantly affect SAM2's segmentation in later frames.

**Semantic Entanglement across Consecutive Frames**: SAM2 maintains a memory bank that caches the semantic features of the past $k$ frames and integrates historical information through a memory attention module to guide current-frame segmentation. Attacking a single frame is insufficient to undermine segmentation, as clean features in the memory bank "repair" the influence of the attacked frame. However, simultaneously corrupting historical features in the memory bank can significantly impair current-frame segmentation performance—this is the principle underlying the "avalanche effect."

## Method

### Overall Architecture

UAP-SAM2 constructs the attack along two dimensions: (1) intra-frame semantic distortion—confusing the semantics of foreground and background within the current frame; and (2) inter-frame semantic inconsistency—disrupting semantic continuity across consecutive frames. The overall optimization objective is the sum of three attack losses: $\mathcal{J}_{\text{total}} = \mathcal{J}_{\text{sa}} + \mathcal{J}_{\text{fa}} + \mathcal{J}_{\text{ma}}$.

A target-scanning strategy is also designed to achieve cross-prompt transferability: each frame is evenly divided into $m$ regions, each randomly assigned a prompt, reducing dependence on specific prompts during optimization. The attack targets the output features of the image encoder rather than prompt-dependent masks, further enhancing cross-prompt generalizability.

### Key Designs

1. **Semantic Confusion Attack ($\mathcal{J}_{\text{sa}}$)**: Binary foreground mask $m_+$ and background mask $m_-$ are used to separate the target from the background. The optimization objective is to cause the model to misclassify the foreground as background while reinforcing the classification confidence of background regions. A BCE loss is used to strengthen the attack on pixels near the decision boundary (logits close to 0): $\mathcal{J}_{\text{sa}} = \frac{1}{N}\sum_{i=1}^{N}[\text{BCE}(f_\theta(\tilde{x}_i, \mathcal{P}) \cdot m_+, y_-) + \text{BCE}((1 - f_\theta(\tilde{x}_i, \mathcal{P})) \cdot m_-, y_-)]$, where $y_-$ is a mask filled with threshold $-1$ in the target region and $0$ elsewhere.

2. **Feature Shift Attack ($\mathcal{J}_{\text{fa}}$)**: The distance between adversarial and clean frames in the image encoder's feature space is maximized. The baseline version minimizes cosine similarity. An enhanced version adopts a contrastive learning framework: augmented prototypes of adversarial and clean frames are treated as negative pairs, while frames from other videos serve as positive pairs, and the InfoNCE loss is used to push adversarial features away: $\mathcal{J}_{\text{fa}} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(\text{cos}(\mathcal{E}_{\text{img}}(\tilde{x}_i), e_i)/\tau)}{\sum_{k=1}^{N}\mathbf{1}_{k \neq i}\exp(\text{cos}(\mathcal{E}_{\text{img}}(\tilde{x}_i), \mathcal{E}_{\text{img}}(x_k))/\tau)}$, where $e_i = \frac{1}{\rho}\sum_{j=1}^{\rho}\mathcal{E}_{\text{img}}(\mathcal{T}(x_i))$ is the feature prototype computed from $\rho$ random augmentations.

3. **Memory Misalignment Attack ($\mathcal{J}_{\text{ma}}$)**: Starting from the second frame, the feature discrepancy between consecutive adversarial frames is maximized to induce the avalanche effect—the similarity between the current frame and both the preceding frame and the first frame progressively decreases: $\mathcal{J}_{\text{ma}} = -\frac{1}{N}\sum_{i=1}^{N}\text{cos}(\mathcal{E}_{\text{img}}(\tilde{x}_{i+1}), \mathcal{E}_{\text{img}}(\tilde{x}_i))$. This accumulating semantic inconsistency causes the features stored in the memory bank to become increasingly misaligned with the current frame.

### Loss & Training

The perturbation budget for the UAP is $\epsilon = 10/255$ (universal setting) and $8/255$ (sample-specific setting), with batch size 1 and 10 training epochs. The number of regions is $m=256$, the number of negative samples is 30, and 15 frames per video are used. A fixed random seed of 30 is used to ensure reproducibility. Experiments are conducted on two NVIDIA A100-SXM4 GPUs.

## Key Experimental Results

### Main Results

**Comparison with existing methods (UAP, mIoU%, lower is stronger attack)**:

| Method | YouTube-Video (pt) | DAVIS-Video (pt) | MOSE-Video (pt) | YouTube-Image (pt) | DAVIS-Image (pt) | MOSE-Image (pt) |
|--------|--------------------|------------------|-----------------|--------------------|------------------|-----------------|
| UAPGD | 42.59 | 53.60 | 50.80 | 54.42 | 50.11 | 61.76 |
| AttackSAM | 64.35 | 62.31 | 63.05 | 64.18 | 55.53 | 63.92 |
| DarkSAM | 67.51 | 57.00 | 51.96 | 64.38 | 52.99 | 64.38 |
| **UAP-SAM2** | **37.03** | **42.47** | **33.67** | **27.54** | **48.45** | **50.13** |

UAP-SAM2 achieves an average mIoU of 37.72% on video segmentation, which is 10.28% lower than the best baseline UAPGD (i.e., a stronger attack).

### Ablation Study

| Component Configuration | DAVIS mIoU (SAM2-T) | DAVIS mIoU (SAM2-S) | Note |
|-------------------------|---------------------|---------------------|------|
| A (Semantic Confusion) | ~52 | ~55 | Intra-frame attack only |
| A+B (Semantic Confusion + Feature Shift) | ~45 | ~48 | Feature-level enhancement |
| A+C (Semantic Confusion + Memory Misalignment) | ~43 | ~47 | Inter-frame attack effective |
| A+B+C (Full UAP-SAM2) | **~38** | **~42** | Three components are complementary |

The region count $m=256$ is optimal; using 15 frames approaches the performance of using all frames.

### Key Findings

- Without attack, SAM2 achieves average mIoU > 76%; UAP-SAM2 reduces this to 37.72% (average drop > 38%)
- Video segmentation is more susceptible to attack than image segmentation, validating the effectiveness of the inter-frame semantic inconsistency attack
- Strong cross-dataset and cross-model transferability: UAPs generated on SAM2-T remain effective when transferred to SAM2-S and SAM2.1-T
- Model pruning and data preprocessing defenses (e.g., blurring, occlusion) offer limited protection: pruning 40% severely degrades clean performance while leaving adversarial performance nearly unaffected
- Even at a very small perturbation budget ($\epsilon = 4/255$), UAP-SAM2 still causes a mIoU drop of > 33%

## Highlights & Insights

- **The discovery and exploitation of the "avalanche effect"** is the paper's most significant contribution: by accumulating semantic discrepancies frame by frame, the segmentation of the entire video collapses
- The target-scanning strategy is elegantly designed: rather than directly attacking prompt-dependent masks, it targets the universal features of the image encoder
- The incorporation of contrastive learning (feature shift attack) drives adversarial features away from the intrinsic semantic space
- This work is the first to expose security vulnerabilities in video segmentation foundation models, serving as an important warning for safety-critical applications

## Limitations & Future Work

- The method targets only the SAM2 model family; generalizability to other video segmentation models (e.g., XMem, Cutie) remains unknown
- The current UAP requires iterative optimization, which may be insufficient in efficiency for real-time attack scenarios
- On the defense side, only simple pruning and preprocessing are tested; stronger defenses such as adversarial training are not explored
- The attack assumes full model access (white-box); while black-box transferability is partially validated, performance degrades in that setting

## Related Work & Insights

- Compared to DarkSAM (state-of-the-art attack against SAM), UAP-SAM2 specifically designs inter-frame attacks targeting SAM2's memory mechanism
- The discovery of the avalanche effect suggests that the memory mechanism in video models is a double-edged sword: it improves performance while simultaneously introducing a new attack surface
- This work carries important implications for the design of video segmentation systems in safety-critical domains such as autonomous driving and medical imaging

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First UAP attack against SAM2; the discovery of the avalanche effect is highly insightful
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6 datasets, 3 models, 72 settings; comprehensive ablation and defense experiments
- Writing Quality: ⭐⭐⭐⭐ Clear observation–design–validation narrative structure
- Value: ⭐⭐⭐⭐ Exposes security vulnerabilities of video segmentation foundation models, providing a valuable warning to the community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SANSA: Unleashing the Hidden Semantics in SAM2 for Few-Shot Segmentation](sansa_unleashing_the_hidden_semantics_in_sam2_for_few-shot_segmentation.md)
- [\[NeurIPS 2025\] Instant Video Models: Universal Adapters for Stabilizing Image-Based Networks](instant_video_models_universal_adapters_for_stabilizing_image-based_networks.md)
- [\[AAAI 2026\] RS2-SAM2: Customized SAM2 for Referring Remote Sensing Image Segmentation](../../AAAI2026/segmentation/rs2-sam2_customized_sam2_for_referring_remote_sensing_image_segmentation.md)
- [\[ICLR 2026\] Efficient-SAM2: Accelerating SAM2 with Object-Aware Visual Encoding and Memory Retrieval](../../ICLR2026/segmentation/efficient-sam2_accelerating_sam2_with_object-aware_visual_encoding_and_memory_re.md)
- [\[NeurIPS 2025\] HAODiff: Human-Aware One-Step Diffusion via Dual-Prompt Guidance](haodiff_human-aware_one-step_diffusion_via_dual-prompt_guidance.md)

</div>

<!-- RELATED:END -->
