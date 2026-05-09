---
title: >-
  [Paper Note] IMSE: Intrinsic Mixture of Spectral Experts Fine-tuning for Test-Time Adaptation
description: >-
  [ICLR 2026][test-time adaptation] This paper proposes IMSE, which decomposes the linear layers of a pretrained ViT via SVD into "spectral experts" and adapts only the singular values for extremely parameter-efficient test-time adaptation. Combined with a diversity maximization loss and a domain-aware spectral code retrieval mechanism, IMSE achieves state-of-the-art performance across three settings: TTA, CTTA, and progressive CTTA.
tags:
  - ICLR 2026
  - test-time adaptation
  - singular value decomposition
  - mixture of experts
  - continual adaptation
  - distribution shift
date: 2026-05-08
content_hash: f648f8a5a8d99c16
---

# IMSE: Intrinsic Mixture of Spectral Experts Fine-tuning for Test-Time Adaptation

**Conference**: ICLR 2026
**arXiv**: [2603.07926](https://arxiv.org/abs/2603.07926)
**Code**: [github](https://github.com/baek85/IMSE)
**Area**: Code Intelligence
**Keywords**: test-time adaptation, singular value decomposition, mixture of experts, continual adaptation, distribution shift

## TL;DR

This paper proposes IMSE, which decomposes the linear layers of a pretrained ViT via SVD into "spectral experts" and adapts only the singular values for extremely parameter-efficient test-time adaptation. Combined with a diversity maximization loss and a domain-aware spectral code retrieval mechanism, IMSE achieves state-of-the-art performance across three settings: TTA, CTTA, and progressive CTTA.

## Background & Motivation

Test-time adaptation (TTA) aims to adapt source-domain pretrained models online to unknown target domains without access to source data. Existing methods face three key challenges:

**Underutilization of pretrained features**: Large pretrained models contain rich representational capacity, yet how to fully exploit such representations with minimal parameter updates remains underexplored. Existing approaches either tune only batch normalization parameters (limited adaptability) or introduce additional modules (increased inference overhead).

**Feature collapse from entropy minimization**: In unlabeled TTA settings, entropy minimization tends to drive the model to exploit domain-specific rather than class-discriminative features, which can further degrade performance.

**Forgetting of domain knowledge in continual TTA**: Under the CTTA setting, the model must not only preserve pretrained knowledge but also retain and reuse knowledge from previously encountered domains. Existing methods lack efficient mechanisms for domain knowledge preservation and reuse.

## Method

### Overall Architecture

IMSE comprises three core components organized around the idea of treating linear layers as an intrinsic mixture of spectral experts:

- **Intrinsic Mixture of Spectral Experts**: Each linear layer is decomposed via SVD, with each rank-1 component treated as an independent spectral expert.
- **Diversity Maximization Loss**: Encourages diverse utilization of spectral experts to counteract feature collapse induced by entropy minimization.
- **Domain-Aware Spectral Code Retrieval (IMSE-Retrieval)**: Stores and retrieves adapted singular values in the CTTA setting to enable rapid adaptation upon domain switching.

### Key Designs

#### Spectral Experts and Spectral Codes

The SVD decomposition of the linear transformation at layer $l$ is: $W = U\Sigma V^T = \sum_i \sigma_i u_i v_i^T$. Each rank-1 component is treated as the $i$-th **spectral expert**. Since the singular vectors are mutually orthogonal, the outputs of different experts are also orthogonal. The **spectral code** is defined as the collection of singular values across all layers.

During adaptation, **only the singular values are updated** (the orthogonal bases are frozen), thereby preserving the subspace of the pretrained feature extractor while adjusting the contribution weights of each expert to accommodate the new domain.

#### Expert–Input Alignment Statistics

To quantify feature collapse, the normalized alignment between the $i$-th expert and the input is defined as $a = v_i^T x / \|x\|$, with its mean and standard deviation $\text{Std}_i$ computed accordingly. A low standard deviation indicates that the expert captures domain-specific patterns rather than class-discriminative features.

#### Domain-Aware Spectral Code Retrieval (CTTA-specific)

A **domain bank** is maintained to store [domain descriptor, spectral code] pairs. Domain descriptors are accumulated via EMA of the channel-wise mean and variance of patch tokens. Domain-shift detection employs symmetric KL divergence; when the divergence exceeds a threshold, the current spectral code is stored and the spectral code of the most similar historical domain is retrieved for initialization.

### Loss & Training

**Diversity Maximization Loss**: $\mathcal{L}_{\text{dm}} = -\sum_l \frac{1}{r} \sum_i \text{Std}_i^{(l)}$, which maximizes the alignment standard deviation.

Total loss: $\mathcal{L}_{\text{IMSE}} = \mathcal{L}_{\text{entmin}} + \lambda_{\text{dm}} \cdot \mathcal{L}_{\text{dm}}$

Sharpness-Aware Minimization (SAM) is additionally applied for improved stability. The diversity constraint is applied to the later layers closest to the classification head.

## Key Experimental Results

### Main Results

**ImageNet-C (50k) Single-Domain TTA** (ViT-Base, severity 5):

| Pretraining | Method | Mean Accuracy (%) |
|:---:|:---:|:---:|
| Supervised | DPAL | 67.0 |
| Supervised | **IMSE** | **69.0** |
| MAE | DPAL | 65.9 |
| MAE | **IMSE** | **68.3** |
| CLIP | DPAL | 62.3 |
| CLIP | **IMSE** | **65.5** |

IMSE surpasses the previous SOTA DPAL under all three pretraining strategies, with gains of 2.4 and 3.2 percentage points on MAE and CLIP, respectively.

**ImageNet-R / ImageNet-A**:

| Method | ImageNet-R | ImageNet-A |
|:---:|:---:|:---:|
| DPAL | 64.8 | 49.9 |
| **IMSE** | **69.8** | **54.8** |

Improvements of 5.0 pp and 4.9 pp, respectively.

### Ablation Study

**CTTA Setting** (ImageNet-C, 15 continual domains): IMSE-Retrieval outperforms ViDA by 3.4 pp, with only **1/385** of ViDA's trainable parameters. Under progressive CTTA (135 domains), the improvement is 2.4 pp.

### Key Findings

1. **Extreme parameter efficiency**: Tuning singular values alone achieves SOTA across multiple pretraining strategies.
2. **Diversity loss effectively prevents collapse**: Adding $\mathcal{L}_{\text{dm}}$ leads to more balanced utilization of spectral experts.
3. **Domain bank mechanism is practical**: Spectral codes are compact, incurring minimal storage overhead.
4. **Generalizes across pretraining strategies**: Effective under Supervised, MAE, and CLIP pretraining alike.

## Highlights & Insights

- 🔍 **Novel spectral expert perspective**: SVD rank-1 components are reinterpreted as "experts" without any additional architecture, leveraging the intrinsic structure of pretrained weights.
- 💡 **Quantification and mitigation of feature collapse**: The first work to quantify feature collapse in TTA via spectral expert alignment statistics.
- 🔄 **Compactness of spectral codes naturally suits retrieval**: Storage and retrieval costs are extremely low.
- ⚡ **Minimal parameter count**: 385× fewer trainable parameters than ViDA.

## Limitations & Future Work

1. **One-time SVD decomposition cost**: The initial SVD decomposition incurs non-trivial computational overhead.
2. **Robustness of domain descriptors**: May be insufficiently robust under extreme domain drift or with small batch sizes.
3. **Limited to classification**: Extension to detection and segmentation requires additional design effort.
4. **Backbone compatibility**: Primarily validated on ViT; applicability to CNNs remains to be verified.

## Related Work & Insights

| Method | Strategy | Difference from IMSE |
|:---:|:---:|:---:|
| TENT | BN affine + entropy minimization | Tunes only BN; limited adaptability |
| SAR | Sharpness-aware + sample filtering | IMSE additionally incorporates spectral experts and diversity loss |
| DPAL | Domain prompts + adapters | Introduces extra modules; IMSE requires no additional structure |
| ViDA | Visual domain adapter | IMSE uses 1/385 of ViDA's parameters |
| SVFT/SVDiff | Singular-value-only tuning | Focused on LLMs/Diffusion; IMSE is the first to apply this to TTA |

Core insight: The weight matrices of large pretrained models contain an intrinsic "expert" structure with functional differentiation, which can be revealed and exploited directly through SVD.

## Rating

| Dimension | Score |
|:---:|:---:|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |
| Overall | ⭐⭐⭐⭐ |

> A solid TTA contribution with a novel and insightful spectral expert perspective. The experiments comprehensively cover TTA, CTTA, and progressive CTTA settings across multiple pretraining strategies, with remarkable parameter efficiency. The primary limitation is that the method is restricted to classification tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Program Synthesis via Test-Time Transduction](../../NeurIPS2025/code_intelligence/program_synthesis_via_test-time_transduction.md)
- [\[ICLR 2026\] Inference-Time Safety for Code LLMs via Retrieval-Augmented Revision](inference-time_safety_for_code_llms_via_retrieval-augmented_revision.md)
- [\[NeurIPS 2025\] FlyLoRA: Boosting Task Decoupling and Parameter Efficiency via Implicit Rank-Wise Mixture-of-Experts](../../NeurIPS2025/code_intelligence/flylora_boosting_task_decoupling_and_parameter_efficiency_via_implicit_rank-wise.md)
- [\[NeurIPS 2025\] Principled Fine-tuning of LLMs from User-Edits: A Medley of Preference, Supervision, and Reward](../../NeurIPS2025/code_intelligence/principled_fine-tuning_of_llms_from_user-edits_a_medley_of_preference_supervisio.md)
- [\[AAAI 2026\] TAPA: Training-Free Adaptation of Programmatic Agents via LLM-Guided Program Synthesis in Dynamic Environments](../../AAAI2026/code_intelligence/tapas_are_free_training-free_adaptation_of_programmatic_agen.md)

</div>

<!-- RELATED:END -->
