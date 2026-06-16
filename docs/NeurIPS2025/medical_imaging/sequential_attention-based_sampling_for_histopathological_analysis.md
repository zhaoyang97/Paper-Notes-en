---
title: >-
  [Paper Note] Sequential Attention-based Sampling for Histopathological Analysis
description: >-
  [NeurIPS 2025][Medical Imaging][Whole slide image analysis] This paper proposes SASHA, a framework integrating a Hierarchical Attention-based Feature Distillation (HAFED) module with deep reinforcement learning (RL). By…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "Whole slide image analysis"
  - "deep reinforcement learning"
  - "multiple instance learning"
  - "attention-based sampling"
  - "pathological diagnosis"
date: 2026-05-08
content_hash: 5215aeba051fcdbc
---

# Sequential Attention-based Sampling for Histopathological Analysis

**Conference**: NeurIPS 2025
**arXiv**: [2507.05077](https://arxiv.org/abs/2507.05077)  
**Code**: [GitHub](https://github.com/coglabiisc/SASHA)  
**Area**: Medical Imaging
**Keywords**: Whole slide image analysis, deep reinforcement learning, multiple instance learning, attention-based sampling, pathological diagnosis

## TL;DR

This paper proposes SASHA, a framework integrating a Hierarchical Attention-based Feature Distillation (HAFED) module with deep reinforcement learning (RL). By sampling only 10–20% of high-resolution patches, SASHA achieves classification performance on par with full-resolution SOTA methods, while yielding a 4–8× inference speedup and a WSI compression ratio exceeding 16×.

## Background & Motivation

Whole slide images (WSIs) are the primary data modality in digital pathology, yet their gigapixel-scale resolution renders them extremely large, containing thousands of patches. Processing all patches at full resolution incurs prohibitive computational and storage costs. More critically, diagnostically relevant information is typically concentrated in a **small subset of regions** (e.g., tumor areas), making exhaustive processing of the entire slide highly inefficient.

Multiple instance learning (MIL) is the dominant paradigm, treating a WSI as a bag of patches and aggregating features via attention weighting. However, existing MIL methods (ABMIL, TransMIL, ACMIL) still require computing high-resolution features for all patches. RLogist pioneered the use of deep RL for selective patch sampling, but suffers from three major limitations:

**Large accuracy gap**: Partial sampling yields 10–15% lower accuracy than full-resolution SOTA methods.

**Weak feature representations**: RLogist relies on ImageNet-pretrained ResNet-50, which is suboptimal for pathological diagnosis.

**Training instability**: Jointly training the RL policy network and classification network leads to convergence difficulties.

SASHA addresses these issues through: (1) a **label-aware Hierarchical Attention-based Feature Distillation module (HAFED)** for high-quality diagnostic feature learning; (2) **Targeted State Updates (TSU)** that leverage inter-patch feature correlations for efficient information propagation; and (3) **staged training** to reduce the learning difficulty for RL.

## Method

### Overall Architecture

SASHA formulates WSI analysis as a **Markov Decision Process (MDP)**: an RL agent begins with a low-resolution overview and, at each step, selects a patch to zoom into at high resolution, extracts its features, and updates the WSI state. Classification is performed based on the accumulated state. The framework comprises three key components: HAFED, TSU, and a PPO policy network.

### Key Designs

1. **Hierarchical Attention-based Feature Distillation (HAFED)**: A two-stage attention model. *Stage 1 (Feature Aggregator)*: For each low-resolution patch, attention-weighted aggregation is performed over its $k$ high-resolution sub-patches, compressing $U \in \mathbb{R}^{N \times k \times d}$ into $V \in \mathbb{R}^{N \times d}$ to align the feature dimensionality between high- and low-resolution representations. *Stage 2 (Classifier)*: Attention aggregation across all $N$ patches produces a slide-level embedding $h \in \mathbb{R}^d$ for classification. HAFED employs **multiple attention branches** ($M$ heads), trained with both a similarity loss and a label loss following ACMIL, ensuring that different branches capture complementary diagnostic features.

   A key property of HAFED is that it is **label-aware**, in contrast to the generic ImageNet features used in RLogist. During training, all high-resolution patch features are processed; during inference, only RL-selected patches are processed.

2. **Targeted State Update (TSU)**: The initial state is $S_0 = Z$ (low-resolution features). After the RL agent selects patch $a_t$ and obtains high-resolution features $V(a_t)$, **only patches whose features are similar to $a_t$ are updated**:

    - Cosine similarity is computed: $C = \{i: \cos\angle(S_t(i), S_t(a_t)) \geq \tau\}$
    - Patches in $C$ are updated via an MLP: $S_{t+1}(i) = f_S([S_t(i), S_t(a_t), V(a_t)])$
    - The sampled patch is directly replaced: $S_{t+1}(a_t) = V(a_t)$
    - Previously visited patches are masked to prevent re-sampling

   Compared to RLogist's **global update** (which updates all patches), TSU avoids information contamination of unrelated patches. Ablation experiments show that global updates cause a 12.7% drop in accuracy.

3. **PPO Reinforcement Learning Policy**: The agent samples patch indices from the discrete action space $\{1,2,...,N\}$. The intermediate reward is the negative cross-entropy of the classifier: $r_t = -CE(y, \hat{y_t})$. Advantage estimates are computed via GAE. **Critical training strategy**: HAFED is first trained to convergence, after which its weights are **frozen** before RL policy training begins, avoiding the convergence issues that arise from joint training.

### Preprocessing and Feature Extraction

The CLAM toolkit is used for tissue segmentation and patch extraction (256×256×3). A WSI-pretrained ViT encoder extracts $d$-dimensional patch feature embeddings. Low-resolution features are denoted $Z \in \mathbb{R}^{N \times d}$; high-resolution features are $U \in \mathbb{R}^{N \times k \times d}$.

## Key Experimental Results

### Main Results

| Method | Sampling Rate | CAMELYON16 Acc | CAMELYON16 AUC | TCGA-NSCLC Acc | TCGA-NSCLC AUC |
|--------|--------------|----------------|----------------|----------------|----------------|
| ACMIL | 100% | 0.941±0.015 | 0.970±0.011 | 0.906±0.025 | 0.959±0.006 |
| HAFED (Ours) | 100% | **0.963±0.008** | **0.980±0.003** | **0.923±0.011** | **0.966±0.015** |
| RLogist-0.1 | 10% | 0.824 | 0.829 | 0.828 | 0.892 |
| **SASHA-0.1** | **10%** | **0.901±0.021** | **0.918±0.014** | **0.897±0.023** | **0.956±0.023** |
| RLogist-0.2 | 20% | 0.862 | 0.879 | 0.839 | 0.903 |
| **SASHA-0.2** | **20%** | **0.953±0.017** | **0.979±0.008** | **0.912±0.010** | **0.963±0.014** |

### Ablation Study (CAMELYON16, 20% sampling)

| Variant | Accuracy | AUC | F1 | Notes |
|---------|----------|-----|----|-------|
| SASHA Default | 0.964 | 0.980 | 0.953 | Full model |
| ResNet-50 features | 0.860 | 0.817 | 0.780 | ImageNet-pretrained ResNet |
| CONCH encoder | 0.930 | 0.950 | 0.905 | Medical-pretrained encoder |
| Single attention branch | 0.899 | 0.964 | 0.851 | HAFED reduced to single head |
| Global update | 0.837 | 0.824 | 0.779 | TSU replaced with RLogist-style global update |
| Random policy | 0.516 | 0.550 | 0.553 | RL policy replaced with random selection |

### Key Findings

- SASHA-0.2 achieves 98.9% of full-resolution HAFED accuracy using only 20% of patches (0.953 vs. 0.963).
- Inference speed: SASHA-0.1 requires ~14 s/WSI and SASHA-0.2 ~26 s/WSI, compared to ~117 s/WSI for HAFED (4–8× speedup).
- WSI compression ratio exceeds 16×, as HAFED unifies high- and low-resolution features into a shared $N \times d$ representation.
- Patches selected by the RL agent exhibit significantly higher tumor tissue proportions and attention scores ($p < 0.001$), demonstrating interpretable policy behavior.
- Higher sampling ratios yield lower expected calibration error (ECE); SASHA-0.2's ECE is even lower than that of full-resolution ACMIL and DTFD.

## Highlights & Insights

- Directly modeling the pathologist's "scan-then-zoom" workflow as an MDP yields a natural and principled design.
- TSU's local update strategy is grounded in a simple yet effective intuition: high-resolution information from one patch is primarily relevant to patches with similar features.
- Staged training (HAFED first, then RL) elegantly resolves the instability inherent in joint training.
- Interpretability analysis confirms that the RL agent learns a meaningful and discriminative sampling strategy.

## Limitations & Future Work

- Training still requires processing high-resolution features for all patches; training-time efficiency is not reduced.
- Thorough validation is limited to binary classification tasks; multi-class results are relegated to the appendix.
- The effect of observation budget on calibration warrants more in-depth investigation.
- The TSU threshold $\tau$ is a fixed hyperparameter; adaptive learning of this value could be explored.

## Related Work & Insights

- Compared to ZoomMIL, SASHA employs adaptive RL-based selection rather than fixed top-$k$ attention sampling, offering greater flexibility.
- HAFED's multi-branch attention design draws on ACMIL, with the addition of a hierarchical structure.
- The WSI compression ratio is a meaningful new evaluation dimension with direct relevance to real-world clinical deployment.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of RL and MIL is not entirely new, but the HAFED + TSU design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Two benchmarks, comprehensive ablations, interpretability analysis, and calibration analysis.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with rich figures and tables.
- Value: ⭐⭐⭐⭐⭐ — Directly addresses practical efficiency bottlenecks in WSI analysis; the 4–8× inference speedup has significant clinical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Posterior Sampling by Combining Diffusion Models with Annealed Langevin Dynamics](posterior_sampling_by_combining_diffusion_models_with_annealed_langevin_dynamics.md)
- [\[NeurIPS 2025\] MTBBench: A Multimodal Sequential Clinical Decision-Making Benchmark in Oncology](mtbbench_a_multimodal_sequential_clinical_decision-making_benchmark_in_oncology.md)
- [\[NeurIPS 2025\] EndoBench: A Comprehensive Evaluation of Multi-Modal Large Language Models for Endoscopy Analysis](endobench_a_comprehensive_evaluation_of_multi-modal_large_language_models_for_en.md)
- [\[NeurIPS 2025\] RadZero: Similarity-Based Cross-Attention for Explainable Vision-Language Alignment in Chest X-ray](radzero_similarity-based_cross-attention_for_explainable_vision-language_alignme.md)
- [\[ICCV 2025\] M-Net: MRI Brain Tumor Sequential Segmentation Network via Mesh-Cast](../../ICCV2025/medical_imaging/m-net_mri_brain_tumor_sequential_segmentation_network_via_mesh-cast.md)

</div>

<!-- RELATED:END -->
