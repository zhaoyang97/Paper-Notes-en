---
title: >-
  [Paper Note] Multimodal Protein Language Models for Enzyme Kinetic Parameters: From Substrate Recognition to Conformational Adaptation
description: >-
  [CVPR2025][Computational Biology][Enzyme kinetic prediction] This paper proposes the ERBA adapter, which models enzyme kinetic prediction as a staged conditioning process of "substrate recognition $\rightarrow$ conformational adaptation". It injects substrate semantics via MRCA, fuses active-site 3D geometry via G-MoE, and preserves PLM priors via ESDA, consistently outperforming existing methods on three kinetic endpoints: kcat, Km, and Ki.
tags:
  - "CVPR2025"
  - "Computational Biology"
  - "Enzyme kinetic prediction"
  - "Protein language models"
  - "Multimodal fusion"
  - "Mixture-of-Experts"
  - "Distribution alignment"
date: 2026-05-08
content_hash: a0ed53f0c3ca989d
---

# Multimodal Protein Language Models for Enzyme Kinetic Parameters: From Substrate Recognition to Conformational Adaptation

**Conference**: CVPR2025  
**arXiv**: [2603.12845](https://arxiv.org/abs/2603.12845)  
**Code**: To be confirmed  
**Area**: Computational Biology  
**Keywords**: Enzyme kinetic prediction, Protein language models, Multimodal fusion, Mixture-of-Experts, Distribution alignment

## TL;DR

This paper proposes the ERBA adapter, which models enzyme kinetic prediction as a staged conditioning process of "substrate recognition $\rightarrow$ conformational adaptation". It injects substrate semantics via MRCA, fuses active-site 3D geometry via G-MoE, and preserves PLM priors via ESDA, consistently outperforming existing methods on three kinetic endpoints: kcat, Km, and Ki.

## Background & Motivation

- Accurate prediction of enzyme kinetic parameters (kcat, Km, Ki) is crucial for high-throughput protein design and synthetic biology, enabling the screening of large candidate libraries prior to wet-lab experiments.
- Existing methods (DLKcat, UniKP, CataPro, CatPred) fuse enzyme and substrate representations through shallow operations (concatenation, single-layer cross-attention) and regress directly, implicitly treating catalysis as a static compatibility problem.
- In contrast, the actual catalytic process comprises two stages: (1) substrate recognition and localization, and (2) conformational adaptation of the active site to stabilize the transition state. Both stages jointly determine the kinetic parameters.
- Although protein language models (PLMs) provide rich evolutionary constraints and biochemical priors, current works only utilize PLM features passively, failing to explicitly condition the PLM on substrate and geometric information. Furthermore, naive injection of 3D information may disrupt the pre-trained biochemical semantics.

## Method

### Overall Architecture: Enzyme-Reaction Bridging Adapter (ERBA)

The kinetic prediction is remodeled as a staged conditioning process: $$ŷ = G^{(2)}(M^{(1)}(S_e, S_m), S_g)$$, which first performs substrate recognition followed by conformational adaptation, aligning with the enzymatic mechanism.

### Stage 1: Molecular Recognition Cross-Attention (MRCA)

- The enzyme sequence is encoded by the shallow layers of the PLM into residue embeddings $H_e \in \mathbb{R}^{L_e \times D}$, and the substrate SMILES is encoded into $H_m \in \mathbb{R}^{L_m \times D}$ by an MPNN.
- Through a single-layer cross-attention mechanism, the enzyme embeddings serve as Queries while the substrate embeddings serve as Keys/Values. The calculated attention matrix injects substrate semantics into the enzyme representation.
- Residual connections + LayerNorm generate the substrate-aware representation $H^{(1)}$, highlighting residues relevant to the substrate.

### Stage 2: Geometry-aware Mixture-of-Experts (G-MoE)

- Active site residues are encoded into geometric descriptors $H_g$ by E-GNN.
- G-Gating & Router: The pocket region tokens output by MRCA and the geometric descriptors are pooled and concatenated. They then go through softmax gating + Top-k sparse activation ($n=4$ experts, $k=2$).
- Each expert performs pocket-local, geometry-modulated low-rank adaptation: local pocket regions are conditionally excited using low-rank matrices $U_n, V_n$ and geometric biases $B_n$, while non-pocket residues remain unchanged.
- After sparse aggregation, a final MLP yields the geometric conditioned representation $H^{(2)}$.
- A balancing regularization term $\mathcal{L}_{G-MoE}$ is used to prevent expert collapse.

### Enzyme-Substrate Distribution Alignment (ESDA)

- Core Idea: Align the distributions of the three-stage representations in RKHS (Reproducing Kernel Hilbert Space) using RBF kernel MMD.
- Align the distribution distances of $H^{(1)}$ with $H^{(0)}$ (sequence embeddings), and $H^{(2)}$ with $H^{(0)}$.
- Ensure that the injection of substrate and geometric information evolves within the semantic manifold of the PLM, rather than drifting to a new manifold.
- The bandwidth $\sigma$ is determined via the median heuristic, and an unbiased diagonal-free normalized mini-batch MMD estimator is used.

### Loss & Training

- Regression is performed in the $\log_{10}$ space, employing a heteroscedastic Gaussian NLL loss to model the positivity and multiplicative noise of the kinetic constants.
- Total Loss: $\mathcal{L} = \mathcal{L}_{task} + \lambda_1 \cdot \mathcal{L}_{G-MoE} + \lambda_2 \cdot \mathcal{L}_{ESDA}$ ($\lambda_1 = 0.01$, $\lambda_2 = 0.1$).
- Parameter-efficient fine-tuning of the top PLM layers is conducted using LoRA (rank = 8, scaling = 16).

## Key Experimental Results

### Datasets

- kcat: 23,197 records; Km: 41,174 records; Ki: 11,929 records (from BRENDA and SABIO-RK).

### Comparison with SOTA (Table 1, Exp I)

| Method | kcat R² | kcat RMSE | Km R² | Km RMSE | Ki R² | Ki PCC |
|------|---------|-----------|-------|---------|-------|--------|
| CatPred | 0.40 | 1.30 | 0.49 | 0.93 | 0.45 | 0.60 |
| CataPro | 0.41 | 1.33 | 0.41 | 1.01 | - | - |
| **ERBA (Ours)** | **0.54** | **1.13** | **0.61** | **0.70** | **0.61** | **0.78** |

### OOD Generalization (Table 3, EITLEM Test Set)

| Method | kcat R² | kcat PCC | Km R² | Km PCC |
|------|---------|----------|-------|--------|
| EITLEM | 0.27 | 0.50 | 0.23 | 0.43 |
| CatPred | 0.25 | 0.46 | 0.30 | 0.45 |
| **Ours** | **0.50** | **0.70** | **0.55** | **0.69** |

### Ablation Study (Table 4, Ki Endpoint)

- PLM Baseline $\rightarrow$ +MRCA: $R^2$ $0.49 \rightarrow 0.51$
- +G-MoE: $R^2 \rightarrow 0.54$
- +ESDA (Full): $R^2 \rightarrow 0.61$, PCC $\rightarrow 0.78$

### Backbone Scaling

- ERBA consistently improves performance across ESM2 (8M $\rightarrow$ 3B), ProtT5-3B, and Ankh3 (1.8B/5.7B), orthogonal to model size and pre-training schemes.

## Highlights & Insights

1. **Mechanism-aligned problem modeling**: Decomposes the catalytic process into substrate recognition and conformational adaptation, mapped respectively to MRCA and G-MoE. This is biologically more grounded than shallow fusion.
2. **Distribution-level alignment (ESDA)**: Uses MMD in RKHS for distribution alignment instead of feature alignment, elegantly resolving the PLM semantic drift problem during multimodal fine-tuning.
3. **Geometry-aware sparse routing**: G-MoE routes based jointly on pocket topology and sequence-substrate features, activating only 2 out of 4 experts. This preserves sparsity while capturing pocket heterogeneity.
4. **Strong OOD generalization**: On the EITLEM test set, the kcat $R^2$ jumps from the second-best 0.27 to 0.50, demonstrating that mechanism alignment effectively enhances out-of-distribution generalization.
5. **Cross-backbone universality**: Achieves consistent gains across PLMs ranging from 8M to 5.7B parameters.

## Limitations & Future Work

1. Dependance of 3D structures on prediction tools (OpenFold/ESMFold), where structure prediction errors can propagate to kinetic prediction.
2. Only handles single-substrate scenarios, without considering cofactors, mutation combinations, and time-resolved structural changes.
3. The Ki dataset is relatively small (11,929 records), and some enzyme classes (such as EC-6 Ligases) have extremely few samples (21 records), limiting prediction performance.
4. The heteroscedastic Gaussian assumption may underfit extreme values (e.g., extremely fast/slow enzymes).
5. The number of experts in G-MoE is fixed to 4, and the impact of different expert numbers is not explored.

## Rating

- Novelty: ⭐⭐⭐⭐ — Merging mechanism-aligned staged conditional modeling with RKHS distribution alignment represents a significant innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Detailed experiments cover three endpoints, multiple backbones, OOD generalization, ablation studies, and fusion order.
- Writing Quality: ⭐⭐⭐⭐ — The motivation is clear and equations are rigorous, though some notations are dense.
- Value: ⭐⭐⭐⭐ — Provides a general multimodal PLM adaptation paradigm for enzyme kinetic prediction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Steering Protein Language Models](../../ICML2025/computational_biology/steering_protein_language_models.md)
- [\[ACL 2025\] Concept Bottleneck Language Models For Protein Design](../../ACL2025/computational_biology/concept_bottleneck_language_models_for_protein_design.md)
- [\[NeurIPS 2025\] g-DPO: Scalable Preference Optimization for Protein Language Models](../../NeurIPS2025/computational_biology/g-dpo_scalable_preference_optimization_for_protein_language_models.md)
- [\[CVPR 2025\] SHREC: A Spectral Embedding-Based Approach for Ab-Initio Reconstruction of Helical Molecules](shrec_a_spectral_embedding-based_approach_for_ab-initio_reconstruction_of_helica.md)
- [\[ICML 2025\] Elucidating the Design Space of Multimodal Protein Language Models](../../ICML2025/computational_biology/elucidating_the_design_space_of_multimodal_protein_language_models.md)

</div>

<!-- RELATED:END -->
