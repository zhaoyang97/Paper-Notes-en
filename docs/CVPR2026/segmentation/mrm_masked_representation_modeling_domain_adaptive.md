---
title: >-
  [Paper Note] Masked Representation Modeling for Domain-Adaptive Segmentation
description: >-
  [CVPR 2026][Segmentation][unsupervised domain adaptation] The paper proposes Masked Representation Modeling (MRM), which randomly masks and reconstructs features in the encoder's latent space and supervises the reconstru…
tags:
  - "CVPR 2026"
  - "Segmentation"
  - "unsupervised domain adaptation"
  - "masked representation modeling"
  - "semantic segmentation"
  - "auxiliary task"
  - "Rebuilder"
date: 2026-05-08
content_hash: 0952502a70786763
---

# Masked Representation Modeling for Domain-Adaptive Segmentation

**Conference**: CVPR 2026  
**arXiv**: [2509.13801](https://arxiv.org/abs/2509.13801)  
**Code**: [https://github.com/Wenlve-Zhou/MRM](https://github.com/Wenlve-Zhou/MRM)  
**Area**: Semantic Segmentation / Unsupervised Domain Adaptation  
**Keywords**: unsupervised domain adaptation, masked representation modeling, semantic segmentation, auxiliary task, Rebuilder

## TL;DR

The paper proposes Masked Representation Modeling (MRM), which randomly masks and reconstructs features in the encoder's latent space and supervises the reconstruction with a pixel classification loss. As a plug-in auxiliary task it lifts four UDA baselines by an average of +2.3 / +2.8 mIoU on GTA→Cityscapes / Synthia→Cityscapes, with zero inference-time overhead.

## Background & Motivation

**Background**: Unsupervised domain adaptation (UDA) for semantic segmentation transfers knowledge from a labelled source domain (e.g. synthetic GTA) to an unlabelled target domain (e.g. real-world Cityscapes). Mainstream approaches include adversarial training, self-training, and efficient architecture design. Auxiliary self-supervised tasks (e.g. contrastive learning) have been shown to enhance feature discriminativeness, but another important paradigm — masked image modelling (MIM, e.g. MAE) — has barely been explored for UDA segmentation.

**Limitations of Prior Work**: There are two root reasons MIM has not been adopted. First, an input-structure constraint: MIM masks patches at the input and feeds only visible patches into the encoder, which directly conflicts with segmentation architectures such as DeepLab and DAFormer that need to process the full image. Second, an objective conflict: MIM's goal is to reconstruct pixel values inside masked regions element-wise, which is a low-level objective inconsistent with the high-level semantic classification objective required for segmentation, potentially introducing optimization interference.

**Key Challenge**: Masked modelling can bring global contextual reasoning and feature robustness, but its input-side operation and pixel reconstruction objective both block its use in UDA segmentation.

**Goal**: Retain the benefits of masked modelling (global reasoning, feature regularization) while removing both its architectural incompatibility and its objective mismatch with the segmentation task.

**Key Insight**: Move the masking from input space to latent feature space — the encoder still processes the full image, masking is performed on the encoder output, a lightweight module reconstructs the masked features, and the reconstructed features are sent to the original segmentation decoder for pixel classification. This avoids modifying the input pipeline and lets the auxiliary task share the same optimization objective as the main task.

**Core Idea**: Do masked modelling in feature space, not input space; supervise reconstruction with a classification loss, not regression.

## Method

### Overall Architecture

MRM is embedded into existing UDA pipelines as an auxiliary training task and does not modify the network architecture. The full flow: (1) the target-domain image $x^t$ is passed through the encoder $E(\cdot)$ to obtain features $f^t$; (2) the Rebuilder $R(\cdot)$ masks $f^t$ and reconstructs it, producing $f^r$; (3) $f^r$ goes through the original segmentation decoder $D(\cdot)$ for pixel classification, supervised by pseudo-labels $\tilde{y}$ via cross entropy. After training the Rebuilder is removed entirely, so inference is identical to the base model. The total loss is

$$\mathcal{L}_{overall} = \mathcal{L}_{sup} + \mathcal{L}_{uda} + \lambda \mathcal{L}_{mrm}$$

where $\mathcal{L}_{mrm}$ is the pixel classification loss obtained by passing the reconstructed features through the decoder, with $\lambda = 1.0$.

### Key Designs

1. **Representation-level Masking**:

    - Function: random block masking on the encoder's output feature map, instead of on the input image.
    - Mechanism: the encoder processes the full input and outputs $f^t \in \mathbb{R}^{C \times H \times W}$; a binary mask $M$ is generated on the feature map, masked positions are filled by a learnable mask token, and unmasked positions retain their original features.
    - Design Motivation: this avoids modifying the encoder's input pipeline, making MRM fully compatible with any segmentation architecture (CNN-based DeepLab or Transformer-based DAFormer). It is fundamentally different from MAE, which only feeds visible patches to the encoder.

2. **Task-aligned Classification Objective**:

    - Function: supervise reconstruction quality with a pixel-level classification cross entropy (rather than pixel regression / MSE).
    - Mechanism: feed the reconstructed feature $f^r$ directly into the segmentation decoder $D(\cdot)$ for per-pixel classification, supervised by target-domain pseudo-labels via cross entropy. This makes the auxiliary task share the same optimization direction as the main task.
    - Design Motivation: ablations show that a pixel regression objective (as in MAE) actually *hurts* performance (-0.3 mIoU), while the classification objective brings +3.8 mIoU. There is a fundamental conflict between low-level reconstruction and high-level semantic classification, so the auxiliary task must be aligned with the main task.

3. **Lightweight Rebuilder Module**:

    - Function: mask and reconstruct the encoder features during training; removed at inference for zero overhead.
    - Mechanism: it has four steps — (a) Representation Embedding: a linear layer adjusts the channel dimension and bilinear interpolation adjusts the spatial dimension to $16 \times 16 \times 512$; (b) Masking: 40% uniform random masking with a learnable mask token; (c) Transformer Blocks: only 2 Transformer blocks plus absolute position encoding process the sequence; (d) Projector: transposed convolutions restore the original resolution. The output is fused residually as $f^r = M^s \odot f^o + (1-M^s) \odot f^t$, replacing only the masked region.
    - Design Motivation: keeping it extremely lightweight avoids training instability (experiments show that going beyond 2 Transformer blocks degrades performance). For multi-scale models (e.g. DAFormer), only the last-stage feature is rebuilt and upsampled to multi-scale; there is no need to instantiate a Rebuilder at every stage.

### Loss & Training

The MRM loss $\mathcal{L}_{mrm}$ is a per-pixel cross entropy against target-domain pseudo-labels with weight $\lambda = 1.0$, and performance is stable in the range 0.1–2.0. MRM is applied only on target-domain images — applying it on the source domain is harmful (it pulls features toward the source distribution and weakens domain alignment). MRM must train both the encoder and the decoder for the best gain; freezing either component significantly reduces the improvement. The Rebuilder uses a learning rate of $2 \times 10^{-4}$.

## Key Experimental Results

### Main Results

GTA→Cityscapes benchmark:

| Baseline | w/o MRM | w/ MRM | Gain |
|----------|---------|--------|------|
| DACS | 52.1 | 55.9 | +3.8 |
| DAFormer | 68.3 | 70.3 | +2.0 |
| HRDA | 73.8 | 75.4 | +1.6 |
| MIC | 75.9 | **77.5** | +1.6 |

Synthia→Cityscapes benchmark (13-class mIoU):

| Baseline | w/o MRM | w/ MRM | Gain |
|----------|---------|--------|------|
| DACS | 48.3 | 55.8 | +7.5 |
| DAFormer | 60.9 | 62.6 | +1.7 |
| HRDA | 65.8 | 67.1 | +1.3 |
| MIC | 67.3 | **68.1** | +0.8 |

MIC + MRM reaches 77.5 mIoU on GTA→CS, surpassing the previous best QuadMix (76.1) by +1.4 mIoU.

### Ablation Study

Rebuilder design hyper-parameter ablations (DACS + DeepLabV2-ResNet101, GTA→CS):

| Ablation Axis | Setting | mIoU |
|---------------|---------|------|
| Embedding dim | 128 / 256 / **512** / 768 | 54.7 / 55.1 / **55.9** / 53.6 |
| Transformer blocks | 1 / **2** / 4 / 8 | 55.4 / **55.9** / 54.4 / 52.9 |
| Spatial resolution H' = W' | 8 / **16** / 32 / 64 | 55.6 / **55.9** / 54.1 / OOM |
| Mask ratio | best **40%** | **55.9** |
| Weight λ | 0.1 / 0.5 / **1.0** / 2.0 / 10.0 | 54.7 / 55.8 / **55.9** / 55.4 / 52.1 |

Reconstruction-target comparison:

| Reconstruction Target | mIoU | Gain |
|-----------------------|------|------|
| Baseline (no MRM) | 52.1 | — |
| Pixel regression (MAE-style) | 51.8 | -0.3 |
| Teacher-feature reconstruction | 53.5 | +1.4 |
| Student-feature reconstruction | 53.7 | +1.6 |
| **Pixel classification (ours)** | **55.9** | **+3.8** |

Cross-architecture generalization (GTA→CS):

| Encoder | Decoder | UDA Method | w/o MRM | w/ MRM |
|---------|---------|-----------|---------|--------|
| ResNet-50 | DeepLabV2 | DACS | 52.0 | 55.1 (+3.1) |
| ResNet-101 | DeepLabV3+ | DACS | 54.7 | 59.3 (+4.6) |
| ResNet-101 | DeepLabV2 | MIC | 64.2 | 67.1 (+2.9) |
| MiT-B2 | DAFormer Head | DAFormer | 64.2 | 66.3 (+2.1) |
| MiT-B3 | DAFormer Head | MIC | 73.6 | 75.8 (+2.2) |

### Key Findings

- Masking without reconstruction is harmful (-0.2 mIoU), showing that masking in feature space causes irreversible semantic loss and the reconstruction step is essential.
- MRM only helps on the target domain (+3.8); source-only gives +0.8 and source+target gives +3.1 — MRM is essentially a target-domain adaptation regularizer, not a generic self-supervision.
- Encoder and decoder must be trained jointly: encoder-only +2.8, decoder-only +2.1, joint +3.8, confirming that MRM benefits from full-pipeline optimization.
- The optimal mask ratio (40%) is much lower than MAE's 75% — the Rebuilder's capacity is small, so a higher mask ratio causes irreversible semantic loss.

## Highlights & Insights

- **Extremely simple core design**: moving masking from input space to feature space and replacing pixel regression with classification — two simple modifications that resolve both obstacles to MIM in UDA segmentation (architectural incompatibility + objective mismatch), with zero inference overhead. The "do the right thing in the right space" pattern is broadly transferable.
- **Counter-intuitive empirical findings with broad guidance value**: pixel reconstruction is harmful for segmentation, the auxiliary task only helps on the target domain, and contrastive learning only optimizes the encoder while MRM optimizes both encoder and decoder — these findings are useful for designing UDA auxiliary tasks in general.
- **True plug-and-play generality**: works on both CNN (DeepLabV2/V3+) and Transformer (DAFormer) architectures, with consistent +2.1–+4.6 mIoU gains across 5 encoder–decoder–UDA combinations, demonstrating method generality rather than overfit to a single configuration.

## Limitations & Future Work

- The Rebuilder has limited capacity (only 2 Transformer blocks); ablations show that scaling to 4 / 8 blocks degrades performance (training instability), so a stronger and more stable Rebuilder design is a future direction.
- The masking strategy is simple (uniform random); semantics-guided adaptive masking (e.g. higher mask rate for high-uncertainty regions) might bring larger gains.
- Only the UDA setting is validated; whether MRM extends to domain generalization, source-free UDA, or test-time adaptation remains to be explored.
- Reliance on pseudo-label quality means MRM's effectiveness in the early training phase (when pseudo labels are very noisy) is worth further analysis.

## Related Work & Insights

- **vs MAE / MIM**: MAE masks in input space and reconstructs pixels — incompatible with segmentation architectures and misaligned in objective. MRM masks in feature space and reconstructs with a classification objective — perfectly compatible and significantly more effective (pixel regression -0.3 vs pixel classification +3.8).
- **vs contrastive learning (SePiCo, PiPa)**: contrastive learning only enhances encoder feature discriminativeness without training the decoder; MRM jointly optimizes encoder and decoder for more comprehensive regularization.
- **vs MIC**: MIC performs masked consistency in image space (similar to CutOut); MRM performs masked reconstruction in feature space; the two mechanisms are orthogonal and complementary — MIC + MRM reaches 77.5 mIoU, surpassing each in isolation.
- **An information-bottleneck view**: masking is structured noise injection that reduces $I(Z; X)$ (compresses redundant information) while preserving $I(Z; Y)$ (keeps task-relevant information), improving the domain invariance of features.

## Rating

- Novelty: ⭐⭐⭐⭐ — the core design is simple yet clever; components are not new individually, but the combination addresses a practical problem.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — four baselines × two benchmarks, more than ten ablations, five-architecture generalization study — very complete.
- Writing Quality: ⭐⭐⭐⭐⭐ — clear motivation derivation, coherent logical chain of design choices, systematic ablation analysis.
- Value: ⭐⭐⭐⭐ — plug-and-play, zero inference overhead, cross-architecture generality — directly useful for the UDA segmentation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Heuristic Self-Paced Learning for Domain Adaptive Semantic Segmentation under Adverse Conditions](heuristic_self-paced_learning_for_domain_adaptive_semantic_segmentation_under_ad.md)
- [\[CVPR 2026\] Seeing Beyond: Extrapolative Domain Adaptive Panoramic Segmentation](seeing_beyond_extrapolative_domain_adaptive_panoramic_segmentation.md)
- [\[CVPR 2026\] CA-LoRA: Concept-Aware LoRA for Domain-Aligned Segmentation Dataset Generation](ca-lora_concept-aware_lora_for_domain-aligned_segmentation_dataset_generation.md)
- [\[CVPR 2026\] CrossEarth-SAR: A SAR-Centric and Billion-Scale Geospatial Foundation Model for Domain Generalizable Semantic Segmentation](crossearthsar_a_sarcentric_and_billionscale_geospa.md)
- [\[CVPR 2026\] SARMAE: Masked Autoencoder for SAR Representation Learning](sarmae_masked_autoencoder_for_sar_representation_learning.md)

</div>

<!-- RELATED:END -->
