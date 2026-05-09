---
title: >-
  [Paper Note] Multimodal Protein Language Models for Enzyme Kinetic Parameters: From Substrate Recognition to Conformational Adaptation
description: >-
  [CVPR 2026][Medical Imaging][enzyme kinetics prediction] This paper proposes **ERBA (Enzyme-Reaction Bridging Adapter)**, which reformulates enzyme kinetic parameter prediction as a **staged multimodal conditional generation problem** — first injecting substrate information via MRCA to capture substrate recognition specificity, then integrating active-site 3D geometry via G-MoE to capture conformational adaptation, with ESDA distribution alignment to preserve PLM semantic priors.
tags:
  - CVPR 2026
  - Medical Imaging
  - enzyme kinetics prediction
  - protein language model
  - multimodal fusion
  - mixture of experts
  - cross-modal adapter
date: 2026-05-08
content_hash: cc9b007b1a7558be
---

# Multimodal Protein Language Models for Enzyme Kinetic Parameters: From Substrate Recognition to Conformational Adaptation

**Conference**: CVPR 2026
**arXiv**: [2603.12845](https://arxiv.org/abs/2603.12845)
**Code**: None
**Area**: Medical Imaging / Bioinformatics
**Keywords**: enzyme kinetics prediction, protein language model, multimodal fusion, mixture of experts, cross-modal adapter

## TL;DR

This paper proposes **ERBA (Enzyme-Reaction Bridging Adapter)**, which reformulates enzyme kinetic parameter prediction as a **staged multimodal conditional generation problem** — first injecting substrate information via MRCA to capture substrate recognition specificity, then integrating active-site 3D geometry via G-MoE to capture conformational adaptation, with ESDA distribution alignment to preserve PLM semantic priors.

## Background & Motivation

**Background**: High-throughput protein design and synthetic biology increasingly rely on computational prediction of enzyme kinetic parameters ($k_\text{cat}$, $K_m$, $K_i$) to screen candidate molecules prior to wet-lab experiments. Existing methods have evolved from sequence-only approaches to multimodal pipelines incorporating sequence, substrate, and structure.

**Limitations of Prior Work**: Most pipelines encode enzymes and substrates independently and fuse them via **shallow fusion** (concatenation + single-layer cross-attention) for regression, implicitly treating catalysis as a **static compatibility problem**: $\hat{y} = \psi(S_e \oplus S_m \oplus S_g)$.

**Key Challenge**: The true catalytic process is inherently staged — the enzyme first **recognizes and positions the substrate** (substrate recognition), then **adaptively reconfigures the active pocket geometry** (conformational adaptation) to stabilize the transition state. Shallow fusion ignores this staged nature, and naive injection of 3D information risks **corrupting the biochemical semantic priors learned during PLM pretraining**.

**Goal**: To construct a staged conditioning framework aligned with enzymological mechanisms, enabling hierarchical injection of substrate chemistry and pocket geometry while preserving PLM priors.

**Key Insight**: Reformulate kinetic prediction as $\hat{y} = f_\theta^{(2)}(f_\theta^{(1)}(S_e, S_m), S_g)$, where $f^{(1)}$ captures substrate-conditioned molecular recognition and $f^{(2)}$ performs conformational adaptation via geometric awareness.

**Core Idea**: ERBA = MRCA (substrate recognition cross-attention) + G-MoE (geometry-aware mixture of experts routing) + ESDA (distribution alignment regularization).

## Method

### Overall Architecture

ERBA acts as an **adapter** inserted into a pretrained PLM, realizing two-stage conditioning:
$$\hat{y} = \mathcal{G}^{(2)}(\underbrace{\mathcal{M}^{(1)}(S_e, S_m)}_{\text{substrate recognition}}, S_g)$$

The PLM shallow layers output residue embeddings $\mathbf{H}_e \in \mathbb{R}^{L_e \times D}$; the substrate is encoded by an MPNN encoder to yield $\mathbf{H}_m \in \mathbb{R}^{L_m \times D}$; the active site is encoded by an E-GNN to produce geometric descriptors $\mathbf{H}_g \in \mathbb{R}^{L_g \times D}$.

### Key Designs

#### 1. MRCA (Molecular Recognition Cross-Attention)

- **Function**: Injects substrate semantics into the enzyme representation to simulate the substrate recognition stage.
- **Mechanism**: Single-layer cross-attention with enzyme tokens as queries and substrate tokens as keys/values:
  $$\mathbf{A}_{em} = \text{Softmax}\left(\frac{(\mathbf{H}_e \mathbf{W}_Q)(\mathbf{H}_m \mathbf{W}_K)^\top}{\sqrt{d_k}}\right)$$
  $$\mathbf{Z}_{em} = \mathbf{A}_{em}(\mathbf{H}_m \mathbf{W}_V)$$
  Residual connection + LayerNorm yields the substrate-aware representation $\mathbf{H}^{(1)}$.
- **Design Motivation**: The attention matrix $\mathbf{A}_{em}$ naturally aligns enzyme residues with substrate atoms, highlighting substrate-relevant residues.

#### 2. G-MoE (Geometry-Aware Mixture of Experts)

- **Function**: Integrates 3D pocket geometry and captures conformational adaptation via sparse expert routing.
- **Mechanism**:
    - **Routing vector**: Concatenates recognition features from the pocket region and geometric descriptors: $\mathbf{v}_{emg} = [\text{Pool}(\mathbf{H}^{(1)}[\mathcal{P}]) \oplus \text{Pool}(\mathbf{H}_g)]$
    - **Sparse gating**: $\tilde{\boldsymbol{\alpha}} = \text{Top-}k(\text{softmax}(\mathbf{W}_\text{gate} \mathbf{v}_{emg}))$, activating only $k$ geometry-relevant experts
    - **Each expert** performs geometry-modulated low-rank adaptation local to the pocket:
    $$E_n(\mathbf{H}^{(1)}, \mathbf{H}_g) = \mathbf{H}^{(1)} + \mathbf{V}_n \sigma(\mathbf{U}_n \mathbf{H}^{(1)}[\mathcal{P}] + \mathbf{B}_n \Gamma(\mathbf{H}_g))$$
    where $r \ll D$ with GELU activation
    - **Aggregation**: $\mathbf{H}^{(2)} = \text{MLP}(\sum_{n \in \text{Top}k} \tilde{\alpha} E_n)$
- **Design Motivation**: Different pocket topologies and residue arrangements form heterogeneous geometric regimes that a single adapter cannot capture — MoE sparse routing allocates specialized experts for different geometric patterns.

#### 3. ESDA (Enzyme-Substrate Distribution Alignment)

- **Function**: Aligns distributions in the reproducing kernel Hilbert space (RKHS) to prevent multimodal fine-tuning from corrupting PLM priors.
- **Mechanism**: Uses RBF kernel maximum mean discrepancy (MMD) to align the distributions of sequence-only, sequence+substrate, and sequence+substrate+structure representations onto the PLM manifold.
- **Design Motivation**: Direct injection of 3D information tends to dominate training and erode the biochemical semantics — evolutionary constraints and catalytic patterns — learned by the PLM.

### Loss & Training

An **heteroscedastic Gaussian** prediction head models the positivity and multiplicative noise of kinetic constants in $\log_{10}$ space, supplemented by a G-MoE load balancing regularizer:
$$\mathcal{L}_{\text{G-MoE}} = \|\bar{\boldsymbol{\alpha}} - \frac{1}{n}\mathbf{1}\|_2^2 + \|\bar{\mathbf{n}} - \frac{k}{n}\mathbf{1}\|_2^2$$

## Key Experimental Results

### Main Results: Comparison with Existing SOTA (Exp I)

| Method | $k_\text{cat}$ R²↑ | $k_\text{cat}$ RMSE↓ | $K_m$ R²↑ | $K_i$ R²↑ |
|------|-----|------|-----|-----|
| DLKcat (2022) | 0.01 | 1.78 | - | - |
| CatPred (2025) | 0.40 | 1.30 | 0.49 | 0.45 |
| CataPro (2025) | 0.41 | 1.33 | 0.41 | - |
| **ERBA (Ours)** | **0.54** | **1.13** | **0.61** | **0.61** |

### Ablation Study on PLM Backbone (Exp II)

| PLM Backbone | w/o ERBA $k_\text{cat}$ R² | +ERBA $k_\text{cat}$ R² | Gain |
|---------|-----------|-----------|------|
| Ankh3-1.8B | 0.41 | **0.50** | +0.09 |
| Ankh3-5.7B | 0.43 | **0.52** | +0.09 |
| ProtT5-3B | 0.39 | **0.47** | +0.08 |
| ESM2-150M | 0.30 | **0.38** | +0.08 |

### Key Findings

1. **Comprehensive superiority across all endpoints**: ERBA surpasses all existing SOTA on $k_\text{cat}$, $K_m$, and $K_i$, with $k_\text{cat}$ R² improving from 0.41 to 0.54.
2. **Consistent gains across backbones**: ERBA yields stable improvements across all tested PLM backbones (ESM2-8M/35M/150M, ProtT5-3B, Ankh3-1.8B/5.7B).
3. **Value of geometric information**: Compared with CatPred — the only prior method also leveraging 3D structure — ERBA achieves across-the-board improvements, demonstrating that staged injection is more effective than shallow concatenation.
4. Larger PLM backbones yield stronger baseline performance, and the magnitude of ERBA's gain remains relatively consistent across scales.

## Highlights & Insights

1. **Mechanism-aligned modeling paradigm**: The paper reframes the computational biology problem from "static compatibility" to "staged conditioning," perfectly mirroring the enzymatic catalytic mechanism (recognition → adaptation → reaction). This physically/chemically motivated model design is worth generalizing to other scientific ML problems.
2. **MoE for heterogeneous geometric regimes**: Active pocket topology varies enormously across enzymes; using sparse MoE routing to assign different experts to different geometric patterns is a highly natural design choice.
3. **ESDA distribution protection**: Performing distribution alignment in the RKHS more elegantly preserves PLM semantics compared to simple KL divergence or L2 regularization.

## Limitations & Future Work

1. **Dynamic information not considered**: The current framework conditions on static structures without incorporating molecular dynamics trajectories or time-resolved structural cues.
2. **Cofactor and mutation effects**: The model currently accounts only for enzyme sequence, substrate, and pocket structure; cofactors, which are critical to many enzymatic reactions, are not yet incorporated.
3. **Dataset scale limitations**: Experimental enzyme kinetics data remain scarce; although cross-dataset testing indicates improved OOD generalization, larger-scale validation is still needed.
4. Pocket structure depends on the availability of predicted or experimental structures.

## Related Work & Insights

- **Adapter paradigm**: The design philosophy of ERBA parallels Adapter/LoRA in NLP/CV — inserting lightweight modules into frozen large models to inject new modality information.
- **MoE in scientific ML**: The geometry-aware sparse routing mechanism is transferable to materials science, drug discovery, and other domains requiring heterogeneous structural inputs.
- **Distribution alignment regularization**: The RKHS-MMD approach of ESDA is applicable to any scenario where pretraining semantics must be preserved during fine-tuning.
- **Evolution of enzyme kinetics prediction methods**: DLKcat (CNN+GNN) → TurNup (ESM-1b + boosting) → UniKP/CataPro (ProtT5 + SMILES) → CatPred (sequence+substrate+structure shallow fusion) → ERBA (mechanism-aligned staged deep fusion).
- **PLM backbone comparison**: The ESM2 series (8M–3B) demonstrates scaling benefits; ProtT5-3B offers robust encoder-decoder performance; Ankh3's multi-task pretraining yields stronger residue-level priors.

## Implementation Notes

- PLM hidden dimension $D=1024$, maximum sequence length $L_e=1024$
- G-MoE: $n=4$ experts, Top-$k=2$ routing, adapter rank $r \ll D$
- LoRA: rank 8, scaling factor 16, dropout 0.1, applied only to the top layers of the PLM
- Optimizer: AdamW, learning rate $1\times10^{-4}$
- 3D structures sourced from OpenFold/ESMFold predictions

## Rating

⭐⭐⭐⭐ — The mechanism-driven modeling paradigm is elegantly designed, and the consistent cross-backbone gains are convincing; however, the paper title is somewhat misleading (it does not resemble a conventional CV/medical imaging paper).

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] GLEAM: A Multimodal Imaging Dataset and HAMM for Glaucoma Classification](gleam_a_multimodal_imaging_dataset_and_hamm_for_gl.md)
- [\[CVPR 2026\] MultiModalPFN: Extending Prior-Data Fitted Networks for Multimodal Tabular Learning](multimodalpfn_extending_prior-data_fitted_networks_for_multimodal_tabular_learni.md)
- [\[CVPR 2026\] SVC 2026: The Second Multimodal Deception Detection Challenge and the First Domain Generalized Remote Physiological Measurement Challenge](svc_2026_the_second_multimodal_deception_detection_challenge_and_the_first_domai.md)
- [\[CVPR 2026\] EI: Early Intervention for Multimodal Imaging based Disease Recognition](ei_early_intervention_for_multimodal_imaging_based_disease_recognition.md)
- [\[CVPR 2026\] MUST: Modality-Specific Representation-Aware Transformer for Diffusion-Enhanced Survival Prediction with Missing Modality](must_modality-specific_representation-aware_transformer_for_diffusion-enhanced_s.md)

<!-- RELATED:END -->
