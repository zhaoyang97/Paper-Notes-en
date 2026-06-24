---
title: >-
  [Paper Note] Multimodal 3D Genome Pre-training
description: >-
  [NeurIPS 2025][Computational Biology][3D genome] This paper proposes MIX-HIC — the first multimodal foundation model for 3D genomics — which integrates Hi-C contact maps and epigenomic signals via cross-modal interaction blocks and cross-modal mapping blocks. Pre-trained on over 1.27 million paired samples, MIX-HIC achieves state-of-the-art performance across three downstream tasks: Hi-C prediction, chromatin loop detection, and CAGE-seq expression prediction.
tags:
  - "NeurIPS 2025"
  - "Computational Biology"
  - "3D genome"
  - "Hi-C"
  - "epigenomics"
  - "multimodal pre-training"
  - "foundation model"
date: 2026-05-08
content_hash: 87847d646fdc4eee
---

# Multimodal 3D Genome Pre-training

**Conference**: NeurIPS 2025
**arXiv**: [2504.09060](https://arxiv.org/abs/2504.09060)  
**Code**: [github.com/myang998/MIX-HIC](https://github.com/myang998/MIX-HIC)  
**Area**: Medical Imaging
**Keywords**: 3D genome, Hi-C, epigenomics, multimodal pre-training, foundation model

## TL;DR
This paper proposes MIX-HIC — the first multimodal foundation model for 3D genomics — which integrates Hi-C contact maps and epigenomic signals via cross-modal interaction blocks and cross-modal mapping blocks. Pre-trained on over 1.27 million paired samples, MIX-HIC achieves state-of-the-art performance across three downstream tasks: Hi-C prediction, chromatin loop detection, and CAGE-seq expression prediction.

## Background & Motivation

Three-dimensional genome organization — including chromatin loops and topologically associating domains — plays a critical role in gene regulation and cellular function. Hi-C technology quantifies 3D chromatin interactions, while epigenomic signals (e.g., ATAC-seq, DNase-seq) reflect chromatin accessibility. Understanding the relationship between these two modalities is essential for elucidating gene expression regulatory mechanisms.

Existing methods face three core challenges:

**First, heterogeneous data fusion is difficult.** Hi-C contact maps are 2D matrix-form spatial interaction data, whereas epigenomic signals are 1D sequential data — the two modalities are inherently heterogeneous. Simply aligning features from both modalities into a shared space primarily captures modality-invariant knowledge (e.g., gene regulatory mechanisms) while neglecting modality-specific features (e.g., precise chemical modifications and chromatin states revealed by epigenomics), leading to information loss. The paper formally proves via Theorem 1 that perfect feature alignment introduces an information gap of at least $\Gamma_q$, causing prediction error to be worse than using raw data directly.

**Second, generalization capability is limited.** Existing methods are typically optimized for a single task and struggle to adapt to diverse downstream tasks such as generation (Hi-C prediction) and regression (expression prediction).

**Third, data scarcity is a persistent issue.** Hi-C experiments are costly, and missing modalities are common in practice. Models must be capable of inferring the semantics of missing modalities from available ones.

The paper's starting point is to construct the first multimodal 3D genome foundation model by: (1) disentangling modality-invariant and modality-specific representations to address information loss; (2) employing cross-modal mapping blocks for missing modality imputation; and (3) pre-training on over one million paired samples to achieve strong generalization.

## Method

### Overall Architecture
MIX-HIC adopts a dual-encoder architecture. The pre-training stage comprises three core components: feature extraction blocks, cross-modal interaction blocks, and cross-modal mapping blocks. The fine-tuning stage incorporates modality fusion blocks and task-specific decoders to adapt to different downstream tasks. Three input modes are supported: bimodal input (MIX-HIC-Bimodal), non-pre-trained bimodal (MIX-HIC-NonPre), and single-modal inference (MIX-HIC-Infer).

### Key Designs

1. **Feature Extraction Block (Dual Encoder)**:

    - Hi-C Encoder: Based on the ViT architecture, the $50 \times 50$ Hi-C contact map is partitioned into patches (patch size = 2) and progressively downsampled through three Transformer encoder layers to produce Hi-C embeddings $X_M^B \in \mathbb{R}^{\alpha_3 \times C_3}$.
    - Epigenomic Encoder: Processes ATAC-seq and DNase-seq signals of length 5,000. Initial embeddings are extracted via convolution and pooling, followed by three Transformer encoder layers, yielding epigenomic embeddings $X_E^B \in \mathbb{R}^{\beta_3 \times C_3}$.
    - Design Motivation: The Transformer architecture simultaneously models spatial interactions in Hi-C and sequential dependencies in epigenomic signals. Independent encoders for each modality preserve their respective feature spaces.

2. **Cross-Modal Interaction Block**:

    - Function: Disentangles modality-invariant and modality-specific features.
    - Mechanism: Four independent fully connected networks extract modality-invariant representations $X_M^I, X_E^I$ and modality-specific representations $X_M^S, X_E^S$ from $X_M^B$ and $X_E^B$, respectively. A contrastive loss $\mathcal{L}_{\text{con}}$ brings modality-invariant representations closer: $\mathcal{L}_{\text{con}} = \frac{1}{2}(\mathcal{L}_{\text{pair}}(\hat{X_E^I}, \hat{X_M^I}) + \mathcal{L}_{\text{pair}}(\hat{X_M^I}, \hat{X_E^I}))$. An orthogonality loss $\mathcal{L}_{\text{orth}} = \frac{1}{2}(\langle \hat{X_M^S}, \hat{X_M^I} \rangle + \langle \hat{X_E^S}, \hat{X_E^I} \rangle)$ enforces orthogonality between modality-specific and modality-invariant information.
    - Design Motivation: Theorem 1 proves that perfect alignment incurs information loss; thus, both shared knowledge and modality-exclusive information must be preserved simultaneously.

3. **Cross-Modal Mapping Block**:

    - Function: Learns implicit semantic relationships between modalities to support missing modality inference.
    - Mechanism: The invariant and specific representations of each modality are concatenated into $X_M^{\text{Concat}}$ and $X_E^{\text{Concat}}$. After length alignment via 1D adaptive pooling, fully connected layers learn the mappings $X_{\text{M2E}}$ and $X_{\text{E2M}}$. The mapping loss is: $\mathcal{L}_{\text{mapping}} = \frac{1}{2}(\|X_{\text{M2E}} - X_E^{\text{Concat}}\|_2^2 + \|X_{\text{E2M}} - X_M^{\text{Concat}}\|_2^2)$
    - Design Motivation: Given the high cost of Hi-C experiments and frequent modality absence, this module enables the model to infer Hi-C features from epigenomic data alone, addressing the data scarcity problem.

### Loss & Training
Pre-training total loss: $\mathcal{L}_{\text{pretrain}} = \mathcal{L}_{\text{con}} + \mathcal{L}_{\text{orth}} + \mathcal{L}_{\text{mapping}}$

Task-specific losses are used during fine-tuning: BCE loss for chromatin loop detection, and MSE loss for Hi-C prediction and CAGE-seq expression prediction. Data are normalized using RPGC (CAGE-seq) and KR (Hi-C) normalization followed by log transformation.

The pre-training dataset spans four cell lines (HepG2, HCT116, IMR90, WTC11), yielding 1,275,948 paired samples after quality filtering — the largest 3D genome paired dataset to date.

## Key Experimental Results

### Main Results

**Hi-C Contact Map Prediction ($R^2$)**

| Method | GM12878 | K562 | Gain |
|------|---------|------|------|
| Epiphany | 0.7970 | 0.6547 | - |
| EPCOT-LSTM | 0.7993 | 0.7840 | - |
| C.Origami | 0.7958 | 0.7055 | - |
| **MIX-HIC-Infer** | **0.8724** | **0.8001** | **+9.3% / +2.1%** |

**Chromatin Loop Detection**

| Method | GM12878 F1 | K562 F1 | GM12878 AUROC | K562 AUROC |
|------|-----------|---------|---------------|------------|
| Peakachu | 0.8015 | 0.7900 | 0.8766 | 0.8834 |
| DLoopCaller | 0.8250 | 0.7932 | 0.9046 | 0.8924 |
| **MIX-HIC-Bimodal** | **0.8420** | **0.8267** | **0.9209** | **0.9194** |

**CAGE-seq Expression Prediction ($R^2$)**

| Method | GM12878 | K562 |
|------|---------|------|
| EPCOT-Transformer | 0.8578 | 0.8230 |
| EPI-Graph | 0.7965 | 0.8211 |
| **MIX-HIC-Bimodal** | **0.8833** | **0.9077** |

### Ablation Study

**Loss Component Ablation (AUROC, Chromatin Loop Detection)**

| $\mathcal{L}_{\text{con}}$ | $\mathcal{L}_{\text{orth}}$ | $\mathcal{L}_{\text{mapping}}$ | GM12878 | K562 |
|:---:|:---:|:---:|---------|------|
| ✓ | - | - | 0.9136 | 0.9099 |
| ✓ | ✓ | - | 0.9183 | 0.9156 |
| ✓ | ✓ | ✓ | 0.9209 | 0.9194 |

**Modality Combination Ablation (Hi-C Prediction, $R^2$)**

| Configuration | Pre-trained | GM12878 | K562 | Note |
|------|--------|---------|------|------|
| Epi only | No | 0.8481 | 0.7709 | Single-modal baseline |
| Epi + Inferred Hi-C | Yes | 0.8724 | 0.8001 | Cross-modal mapping effective |
| Epi + Hi-C | No | 0.8614 | 0.8755 | Non-pre-trained bimodal |
| Epi + Hi-C | Yes | 0.8833 | 0.9077 | Full pre-trained version (best) |

### Key Findings
- MIX-HIC achieves a 9.3% improvement over the runner-up on Hi-C prediction — the largest margin across all tasks — demonstrating the value of pre-training in capturing cross-modal semantic relationships.
- Few-shot experiments show that with only 10% of training data, MIX-HIC-Bimodal achieves approximately 0.9 AUROC, comparable to other SOTA methods trained on full data.
- MIX-HIC maintains top performance in cross-cell-line evaluation, indicating strong generalization to unseen cell types.
- The non-pre-trained bimodal variant underperforms the Hi-C single-modal baseline on K562 loop detection, empirically confirming Theorem 1's claim that naive feature alignment causes information loss.

## Highlights & Insights
- MIX-HIC is the first multimodal foundation model in the 3D genomics domain, pioneering the introduction of the foundation model paradigm to this field. Its pre-training data scale of over 1.27 million pairs substantially exceeds existing work in the area.
- Theorem 1 formally demonstrates that perfect modality alignment is inferior to disentangled learning of modality-invariant and modality-specific representations — an insight that extends beyond bioinformatics and carries broader implications for multimodal fusion tasks in general.

## Limitations & Future Work
- Hi-C resolution is fixed at 5 kb; modeling at higher resolutions (e.g., 1 kb) would require larger data and model capacity.
- Pre-training covers only four cell lines; extending to more cell types and species could further improve generalization.
- The cross-modal mapping block, while effective, yields relatively modest gains (~0.5% AUROC); stronger modality imputation strategies warrant further exploration.

## Related Work & Insights
- **vs. Epiphany**: Epiphany predicts Hi-C from epigenomic data alone; under the same setting, MIX-HIC-Infer achieves a 9.3% improvement through pre-training-acquired cross-modal semantics.
- **vs. EPCOT**: EPCOT exhibits high performance variance across datasets, whereas MIX-HIC achieves robust performance through large-scale pre-training.
- **vs. RefHiC**: RefHiC's pre-training is limited to small-scale data and a single task; MIX-HIC's million-scale pre-training endows it with multi-task generalization capability.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First multimodal foundation model in the 3D genomics domain; Theorem 1 provides valuable theoretical insight.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three downstream tasks, two cell lines, and comprehensive few-shot, cross-cell-line, and ablation experiments.
- Writing Quality: ⭐⭐⭐⭐ Rigorous structure with tight integration of theory and experiments.
- Value: ⭐⭐⭐⭐⭐ Establishes a new paradigm for 3D genome research; dataset and code are fully open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Synthetic Visual Genome](../../CVPR2025/computational_biology/synthetic_visual_genome.md)
- [\[NeurIPS 2025\] Manipulating 3D Molecules in a Fixed-Dimensional E(3)-Equivariant Latent Space](manipulating_3d_molecules_in_a_fixed-dimensional_e3-equivariant_latent_space.md)
- [\[NeurIPS 2025\] Towards Unified and Lossless Latent Space for 3D Molecular Latent Diffusion Modeling](towards_unified_and_lossless_latent_space_for_3d_molecular_latent_diffusion_mode.md)
- [\[ICML 2025\] Graph Generative Pre-trained Transformer (G2PT)](../../ICML2025/computational_biology/graph_generative_pre-trained_transformer.md)
- [\[NeurIPS 2025\] Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design](uncertainty-aware_multi-objective_reinforcement_learning-guided_diffusion_models.md)

</div>

<!-- RELATED:END -->
