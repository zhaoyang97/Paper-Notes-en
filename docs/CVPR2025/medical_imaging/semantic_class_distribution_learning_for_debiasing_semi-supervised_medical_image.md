---
title: >-
  [Paper Note] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation
description: >-
  [CVPR2025][Medical Imaging][semi-supervised segmentation] Proposes the plug-and-play Semantic Class Distribution Learning (SCDL) module, which learns class-conditional proxy distributions and performs Class-conditional Distribution Bi-directional Alignment (CDBA) along with Semantic Anchor Constraint (SAC). This explicitly reshapes the class-conditional feature structure in the embedding space to alleviate supervision bias and representation imbalance in semi-supervised medic…
tags:
  - "CVPR2025"
  - "Medical Imaging"
  - "semi-supervised segmentation"
  - "class imbalance"
  - "class-conditional distribution"
  - "proxy learning"
  - "medical image segmentation"
date: 2026-05-08
content_hash: 33d272714de2bb49
---

# Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation

**Conference**: CVPR2025  
**arXiv**: [2603.05202](https://arxiv.org/abs/2603.05202)  
**Code**: [GitHub](https://github.com/Zyh55555/SCDL)  
**Area**: Medical Image  
**Keywords**: semi-supervised segmentation, class imbalance, class-conditional distribution, proxy learning, medical image segmentation

## TL;DR

Proposes the plug-and-play Semantic Class Distribution Learning (SCDL) module, which learns class-conditional proxy distributions and performs Class-conditional Distribution Bi-directional Alignment (CDBA) along with Semantic Anchor Constraint (SAC). This explicitly reshapes the class-conditional feature structure in the embedding space to alleviate supervision bias and representation imbalance in semi-supervised medical image segmentation.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work: The class imbalance problem in Semi-Supervised Medical Image Segmentation (SSMIS)**: Medical segmentation data naturally exhibits a pixel-level long-tailed distribution, where large organs occupy most pixels and dominate gradient updates, leading to insufficient training for small organs.

### Background

**Background: Dual challenges of class imbalance + semi-supervised learning**:
  1. **Supervision signal bias**: Large organs occupy more pixels, causing gradients to bias towards head classes; self-generated pseudo-labels and consistency constraints further reinforce the learning of head classes.
  2. **Representation-level imbalance**: Existing methods (reweighting, output calibration) only operate on the loss or output layer, lacking direct constraints on the class-conditional feature distribution, which leads to tail-class features drifting into head-class regions.

### Key Challenge

**Key Challenge: Key Insight**: Existing methods utilize unlabeled data primarily for local consistency mapping, rarely using them to explicitly correct the skewed class-conditional feature distribution.

## Method

### SCDL Overall Architecture
- A plug-and-play module that can be integrated into existing segmentation networks.
- Fully exploits labeled data to provide semantic supervision, while guiding unlabeled data to participate in learning at the distribution level.

### 1. Class-conditional Distribution Bi-directional Alignment (CDBA)
**Class distribution modeling**: Each semantic class $c$ is represented by a learnable proxy distribution $\mathcal{N}(\mu_c, \operatorname{diag}(\sigma_c^2))$, where both the mean and variance are trainable parameters.

**Soft assignment**: Computes the soft assignment probability $P(c|z)$ of each token embedding to all class proxies using cosine similarity and softmax, allowing each embedding to be associated with multiple classes.

**Bi-directional alignment**:
- **E2P (Embedding to Proxy)**: Weighted cosine distance loss, encouraging embeddings to be close to their soft-assigned proxy distributions.
- **P2E (Proxy to Embedding)**: Encourages each proxy to have high similarity to its assigned embeddings and low similarity to other embeddings, enhancing proxy discriminability.

**Proxy sampling and feature enrichment**:
- **Distribution-weighted prior**: Samples $S$ samples from the proxy distribution and weights the means of each class using the sampling similarity.
- **Center similarity prior**: Directly weights the means of each class using cosine similarity (ignoring variance).
- **Token sampling prior**: Performs local perturbation sampling for each token.
- The three priors are concatenated, projected, and injected into each decoder layer.

### 2. Semantic Anchor Constraint (SAC)
- **Anchor construction**: Extracts class-specific regions using ground-truth masks, obtains class-aware embeddings through the shared encoder, and averages them to serve as semantic anchors.
- **Anchor alignment loss**: Utilizes cosine similarity to constrain each class proxy $\mu_c$ to align with its corresponding semantic anchor.
- Anchors are detached during backpropagation, updating only the proxies without affecting the encoder.
- Ensures that the proxy distributions capture the true class semantics, preventing them from being biased by class frequency imbalances.

## Key Experimental Results

### Datasets
- **Synapse**: 30 CT scans, 13 organ classes, 20% labeled
- **AMOS**: 360 CT scans, 15 organ classes, 5% labeled

### Synapse (20% labeled)

### Main Results

| Method | DSC↑ | ASD↓ |
|------|:---:|:---:|
| VNet (fully) | 68.49 | 6.08 |
| GA-CPS | 66.29 | 5.44 |
| **SCDL-GA-CPS** | **67.50** | **3.32** |
| GA-MagicNet | 66.00 | 3.42 |
| **SCDL-GA-MagicNet** | **66.75** | **3.65** |

- The ASD of SCDL-GA-CPS decreases by 2.12 ($5.44 \rightarrow 3.32$), significantly improving boundary quality.

### AMOS (5% labeled)

### Ablation Study

| Method | DSC↑ | ASD↓ |
|------|:---:|:---:|
| GenericSSL | 35.73 | 45.82 |
| **SCDL-GenericSSL** | **47.35** (+11.62) | **22.84** |
| DHC | 40.11 | 40.65 |
| **SCDL-DHC** | **49.28** (+9.17) | **17.47** (-23.18) |
| GA-MagicNet | 59.15 | 8.66 |
| **SCDL-GA-MagicNet** | **62.16** (+3.01) | **5.65** |

- The improvement is more significant on the highly label-scarce AMOS dataset: SCDL-GenericSSL improves DSC by +11.62%, and SCDL-DHC dramatically drops ASD from 40.65 to 17.47.

### Class-wise Dice Analysis
- The contribution is particularly significant for tail classes: e.g., on Synapse, SCDL-GA-CPS improves the pancreas (PA) from $45.5 \rightarrow 49.4$, and the right adrenal gland (RAG) from $44.7 \rightarrow 49.2$.
- Classes with a Dice score of 0 on AMOS (such as RAG/LAG in GenericSSL and DHC) obtain non-zero Dice scores after incorporating SCDL.

## Highlights & Insights

1. **Plug-and-play design**: SCDL can be seamlessly integrated into various semi-supervised segmentation baselines (GenericSSL, DHC, GA-MagicNet, GA-CPS), consistently yielding improvements.
2. **Representation-level debiasing**: Unlike loss/output-layer regularizations, it directly learns the class-conditional distribution structure in the embedding space, fundamentally addressing representation bias.
3. **Bi-directional alignment mechanism**: E2P + P2E complement each other, pushing embeddings close to the correct proxies while making the proxies discriminative.
4. **Semantic anchor guidance**: Employs labeled data to provide reliable semantic supervision for proxy learning, preventing proxy divergence.
5. **Significant gain in tail classes**: Under extremely scarce labor annotations (5%), it enables previously completely failed classes to achieve meaningful segmentations.

## Limitations & Future Work

1. The proxy distribution is assumed to be a diagonal Gaussian, which may not capture complex multi-modal class-conditional distributions.
2. The semantic anchors in SAC rely on labeled data, so anchor quality may be unstable when annotations are extremely scarce.
3. The concatenation and injection of the three priors (distribution-weighted, center similarity, and token sampling) is relatively naive and lacks adaptive fusion.
4. Evaluated only on CT datasets; the applicability to other modalities such as MRI has not been explored.
5. Sensitivity analysis on hyperparameters, such as the proxy sampling number $S$ and projection dimension, is not fully presented.

## Rating
- Novelty: 4/5 — Class-conditional proxy distribution modeling + bi-directional alignment is a novel and insightful design in semi-supervised segmentation.
- Experimental Thoroughness: 4/5 — Evaluated on two datasets, integrated with multiple baselines, and class-wise analysis provided, though a detailed breakdown of the ablation studies is partially lacking.
- Writing Quality: 4/5 — Clear motivation, rigorous method description, and intuitive illustrations.
- Value: 4/5 — The plug-and-play module is of direct practical value for real-world medical segmentation scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Divide, Conquer, and Aggregate: Asymmetric Experts for Class-Imbalanced Semi-Supervised Medical Image Segmentation](../../CVPR2026/medical_imaging/divide_conquer_and_aggregate_asymmetric_experts_for_class-imbalanced_semi-superv.md)
- [\[CVPR 2025\] Enhancing SAM with Efficient Prompting and Preference Optimization for Semi-supervised Medical Image Segmentation](sam_dpo_semi_supervised.md)
- [\[NeurIPS 2025\] VQ-Seg: Vector-Quantized Token Perturbation for Semi-Supervised Medical Image Segmentation](../../NeurIPS2025/medical_imaging/vq-seg_vector-quantized_token_perturbation_for_semi-supervised_medical_image_seg.md)
- [\[CVPR 2025\] Adaptation of Weakly Supervised Localization in Histopathology by Debiasing Predictions](adaptation_of_weakly_supervised_localization_in_histopathology_by_debiasing_pred.md)
- [\[AAAI 2026\] Bidirectional Channel-selective Semantic Interaction for Semi-Supervised Medical Segmentation](../../AAAI2026/medical_imaging/bidirectional_channel-selective_semantic_interaction_for_semi-supervised_medical.md)

</div>

<!-- RELATED:END -->
