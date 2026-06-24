---
title: >-
  [Paper Note] MAP: Unleashing Hybrid Mamba-Transformer Vision Backbone's Potential with Masked Autoregressive Pretraining
description: >-
  [CVPR 2025][Self-Supervised Learning][Mamba-Transformer] This work proposes Masked Autoregressive Pretraining (MAP). By utilizing a hierarchical pretraining objective that combines local MAE modeling with row-level autoregressive decoding, this work successfully pretrains hybrid Mamba-Transformer vision backbones for the first time, significantly outperforming individual MAE and AR strategies.
tags:
  - "CVPR 2025"
  - "Self-Supervised Learning"
  - "Mamba-Transformer"
  - "hybrid backbone"
  - "masked autoregressive pretraining"
  - "vision backbone"
date: 2026-05-08
content_hash: 974c4dba5f112cf3
---

# MAP: Unleashing Hybrid Mamba-Transformer Vision Backbone's Potential with Masked Autoregressive Pretraining

**Conference**: CVPR 2025  
**arXiv**: [2410.00871](https://arxiv.org/abs/2410.00871)  
**Code**: [yunzeliu/MAP](https://github.com/yunzeliu/MAP)  
**Area**: Self-Supervised Learning  
**Keywords**: Mamba-Transformer, hybrid backbone, masked autoregressive pretraining, self-supervised learning, vision backbone

## TL;DR

This work proposes Masked Autoregressive Pretraining (MAP). By utilizing a hierarchical pretraining objective that combines local MAE modeling with row-level autoregressive decoding, this work successfully pretrains hybrid Mamba-Transformer vision backbones for the first time, significantly outperforming individual MAE and AR strategies.

## Background & Motivation

**Background**: Hybrid Mamba-Transformer networks have recently received widespread attention, as they combine the scalability of Transformers with the efficiency advantages of Mamba in long-sequence modeling.

**Limitations of Prior Work**:
1. **MAE is unsuitable for Mamba**: MAE pretraining significantly improves ViT (+1.4) but is almost ineffective for Vim (+0.2).
2. **AR is unsuitable for Transformer**: AR pretraining is effective for Vim (+1.4), but provides limited improvement for ViT (+0.2).
3. **Hybrid architectures require pretraining strategies compatible with both computing paradigms**: Existing MAE or AR strategies can only fully unleash the potential of one type of module.

**Key Challenge**: Transformers require bidirectional context modeling (where MAE excels), while Mamba requires sequential continuity modeling (where AR excels). Their optimal pretraining strategies are fundamentally different.

**Key Insight**: Design a hierarchical pretraining objective where local MAE allows the Transformer to learn local bidirectional features, while global autoregression enables Mamba to learn cross-region contextual relationships.

## Method

### Overall Architecture

Given an image, random masking is performed first, followed by autoregressive reconstruction on a row-by-row basis:
1. The image is divided into $M$ rows, with each row containing $N$ tokens.
2. Within each row, 50% of the tokens are randomly masked.
3. The HybridNet encoder processes the unmasked tokens.
4. The Transformer Decoder performs row-level autoregressive decoding: the reconstruction of the $i$-th row depends on all tokens from the previous $i-1$ rows plus the unmasked tokens of the current row.

### Key Designs

**1. Hybrid Network Architecture HybridNet (MMMTMMMT)**
- **Function**: Takes 3 Mamba layers + 1 Transformer layer as a block, repeating it 8 times for a total of 32 layers.
- **Mechanism**: Compares training from scratch across various hybrid arrangements (MMMMMMTT, TTMMMMMM, TMMMTMMM, MMMTMMMT), with MMMTMMMT achieving the best performance (83.12%).
- **Design Motivation**: The starting Mamba layers are responsible for sequence feature extraction, while the interspersed Transformer layers enhance local feature modeling and long-range dependencies, balancing local feature extraction and contextual modeling enhancement.

**2. Masked Autoregressive Decoding Strategy**
- **Function**: For the randomly masked image, a Transformer Decoder is used to reconstruct it in a row-level autoregression manner, predicting all masked tokens within a single row at each step.
- **Mechanism**: The loss function is $\mathcal{L} = -\sum_{i=1}^{M}\sum_{j \in \mathbf{M}_i} \log p(\mathbf{x}_{ij} | \mathbf{x}_{i,j \notin \mathbf{M}_i}, \mathbf{r}_{<i})$. Intra-row prediction follows the MAE style (bidirectional), while inter-row prediction follows the AR style (causal).
- **Design Motivation**: Rows are chosen as sub-regions because the default scanning order in most Mamba implementations is row-first. The AR order must align with the Mamba scanning order to maximize benefits (experimentally verified: +2.9 when aligned, and only +0.2 when misaligned).

**3. Key Findings from Pilot Experiments**
- **Relationship between AR and Scanning Order**: Using AR pretraining that is consistent with the scanning order of Vim yields a +2.9 improvement, whereas an inconsistent one yields only +0.2. This is the first time this conclusion has been systematically verified through experiments.
- **Masking Ratio**: The optimal masking ratio for AR pretraining is 20% (Mamba), that for MAE is 75% (Transformer), and the compromise point for MAP is 50%.
- **Reconstruction Target**: Reconstructing normalized raw pixels using MSE loss performs the best, while diffusion loss yields no significant improvement.

### Loss & Training

- **Pretraining**: AdamW optimizer, 1600 epochs, random cropping used as the sole data augmentation, masking ratio of 50%.
- **Fine-tuning**: Direct fine-tuning for 400 epochs.
- **Reconstruction Target**: MSE loss of normalized raw pixels.

## Key Experimental Results

### Main Results (ImageNet-1K Classification)

| Model | Pretraining | Params | Top-1 Acc |
|---|---|---|---|
| HybridNet-B | None | 128M | 83.1 |
| HybridNet-B | MAE | 128M | 83.9 |
| HybridNet-B | AR | 128M | 83.8 |
| HybridNet-B | CL | 128M | 83.1 |
| **HybridNet-B** | **MAP** | **128M** | **84.9** |
| HybridNet-B (384) | MAP | 128M | 85.5 |
| HybridNet-L (384) | MAP | 443M | 86.2 |
| MambaR-B | AR | 99M | 83.7 |
| **MambaR-B** | **MAP** | **99M** | **84.0** |
| ViT-B | MAE | 86M | 83.6 |
| **ViT-B** | **MAP** | **86M** | **83.6** |
| ViT-L | MAE | 307M | 85.9 |
| **ViT-L** | **MAP** | **307M** | **86.1** |
| MambaVision-B | None | 97M | 84.2 |
| **MambaVision-B** | **MAP** | **97M** | **84.9** |
| MambaVision-L | None | 241M | 85.3 |
| **MambaVision-L** | **MAP** | **241M** | **86.4** |

### Ablation Study

**Masking Strategy**:

| Strategy | Accuracy |
|---|---|
| Scratch | 83.1 |
| Random masking | **84.9** |
| Sequential masking | 84.0 |
| Diagonal masking | 83.8 |

**Masking Ratio**:

| Ratio | Accuracy |
|---|---|
| 0% | 83.3 |
| 25% | 84.5 |
| **50%** | **84.9** |
| 75% | 84.2 |

**Decoder Strategy**:

| Strategy | Accuracy |
|---|---|
| AR decoder | 83.7 |
| MAE decoder | 84.1 |
| Local MAE | 84.2 |
| **MAP (ours)** | **84.9** |

### Downstream Tasks

| Task | Backbone | Metric |
|---|---|---|
| ADE20K Semantic Segmentation | HybridNet-S + MAP | mIoU **46.9** (vs 45.6 without pretraining) |
| COCO Detection | HybridNet-Ti + MAP | AP_box **47.3** (vs 45.9 without pretraining) |

### Key Findings

1. **MAP yields the most significant improvement on hybrid architectures**: On HybridNet-B, MAP (+1.8) >> MAE (+0.8) ≈ AR (+0.7) >> CL (0).
2. **MAP is also effective for pure Mamba**: On MambaR-B, MAP (84.0) > AR (83.7) > MAE (83.1). The local MAE mechanism in MAP enhances Mamba's local feature modeling.
3. **MAP outperforms MAE on large models**: On ViT-L, MAP (86.1) > MAE (85.9). The advantage of autoregressive modeling in larger-scale models becomes evident, aligning with the scaling law observed in LLMs.
4. **Further improvement at 384 resolution**: HybridNet-B at 384 resolution achieves a 0.6 improvement over 224 resolution, proving that Mamba's long-sequence modeling capability indeed yields benefits.

## Highlights & Insights

- **In-depth Pilot Study**: Systematically analyzed the different effects of MAE/AR/CL on Transformer vs Mamba, verifying for the first time that the AR order must align with the Mamba scanning order.
- **Elegant Unified Framework**: Organically integrates MAE (local bidirectional) and AR (global causal) into a row-level decoding paradigm.
- **Broad Applicability**: MAP applies not only to the custom HybridNet but also improves existing hybrid frameworks like MambaVision.
- **Optimal 50% Masking Ratio**: Balances between MAE's 75% and AR's 20%, naturally derived from the requirements of the hybrid architecture.

## Limitations & Future Work

- Hybrid architectures still do not outperform pure Transformer + MAE under the same setting (the focus of MAP is on unleashing the potential of hybrid architectures).
- The current row-level partitioning is simple; a more fine-grained clustering strategy could theoretically yield better results.
- Other modalities such as video and point clouds have not been explored (left as future work by the authors).
- Pretraining requires 1600 epochs, representing a relatively large computational overhead.

## Related Work & Insights

- **MAE**: Highly effective for Transformer pretraining, where a 75% high masking ratio and an asymmetric encoder-decoder are key.
- **ARM**: Performs cluster-based AR pretraining for cross-scanned Mamba, essentially a hybrid of row-first and column-first scanning.
- **VAR**: Proposes the next-scale prediction paradigm, preserving spatial locality.
- **MAR**: Uses AR output as a conditioning signal for diffusion models in generation, which inspired the exploration of diffusion loss in this work.

## Rating ⭐⭐⭐⭐

**Novelty**: ⭐⭐⭐⭐ First systematic study on hybrid Mamba-Transformer pretraining; the MAP paradigm is highly insightful.  
**Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Pilot study + main experiments + multiple downstream tasks + exhaustive ablations.  
**Writing Quality**: ⭐⭐⭐⭐ Clearly structured; the pilot study naturally motivates the method design with coherent logic.  
**Value**: ⭐⭐⭐⭐ Provides a general methodology and best practices for pretraining hybrid architectures.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SMILE: Infusing Spatial and Motion Semantics in Masked Video Learning](smile_infusing_spatial_and_motion_semantics_in_masked_video_learning.md)
- [\[CVPR 2025\] From Prototypes to General Distributions: An Efficient Curriculum for Masked Image Modeling](from_prototypes_to_general_distributions_an_efficient_curriculum_for_masked_imag.md)
- [\[ICML 2025\] PDE-Transformer: Efficient and Versatile Transformers for Physics Simulations](../../ICML2025/self_supervised/pde-transformer_efficient_and_versatile_transformers_for_physics_simulations.md)
- [\[ICML 2025\] A Bayesian Model Selection Criterion for Selecting Pretraining Checkpoints](../../ICML2025/self_supervised/a_bayesian_model_selection_criterion_for_selecting_pretraining_checkpoints.md)
- [\[CVPR 2025\] SATA: Spatial Autocorrelation Token Analysis for Enhancing the Robustness of Vision Transformers](sata_spatial_autocorrelation_token_analysis_for_enhancing_the_robustness_of_visi.md)

</div>

<!-- RELATED:END -->
