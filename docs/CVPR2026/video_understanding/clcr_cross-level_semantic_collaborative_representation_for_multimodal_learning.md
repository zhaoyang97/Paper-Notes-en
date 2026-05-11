---
title: >-
  [Paper Note] CLCR: Cross-Level Semantic Collaborative Representation for Multimodal Learning
description: >-
  [CVPR 2026][Video Understanding][Cross-level semantic alignment] This paper proposes the CLCR framework, which organizes each modality's features into three semantic hierarchy levels (shallow/middle/deep). An intra-level…
tags:
  - "CVPR 2026"
  - "Video Understanding"
  - "Cross-level semantic alignment"
  - "shared-private disentanglement"
  - "multimodal fusion"
  - "sentiment analysis"
  - "event localization"
date: 2026-05-08
content_hash: e0d965dc97f2b282
---

# CLCR: Cross-Level Semantic Collaborative Representation for Multimodal Learning

**Conference**: CVPR 2026
**arXiv**: [2602.19605](https://arxiv.org/abs/2602.19605)
**Code**: N/A
**Area**: Video Understanding
**Keywords**: Cross-level semantic alignment, shared-private disentanglement, multimodal fusion, sentiment analysis, event localization

## TL;DR

This paper proposes the CLCR framework, which organizes each modality's features into three semantic hierarchy levels (shallow/middle/deep). An intra-level Controlled Exchange Domain (IntraCED) restricts cross-modal interaction to the shared subspace only, while an inter-level Collaborative Aggregation Domain (InterCAD) enables adaptive cross-level fusion, addressing the cross-level semantic asynchrony problem in multimodal learning.

## Background & Motivation

Multimodal learning aims to capture shared and private information from multiple modalities (language, vision, acoustics). Two dominant paradigms both suffer from a common limitation:

**Feature disentanglement methods** (MISA, DMD, etc.): Learn modality-invariant/modality-specific subspaces, but assume cross-modal interaction occurs at a single semantic level.

**Dynamic calibration methods** (MLA, ARL, etc.): Adjust contribution weights at the sample/modality level, but equally ignore hierarchical structure.

**Core Problem: Cross-Level Semantic Asynchrony**
- Shallow layers capture lexical/frame-level cues; middle layers encode phrase/prosodic structure; deep layers reflect discourse intent/event context.
- Mixing tokens from different levels leads to semantic confusion, error propagation, and private information leakage.
- From an information bottleneck perspective, unstructured mixing tends to increase $I(Z;N)$ rather than $I(Z;Y)$.

## Method

### Overall Architecture

CLCR consists of three core components:
1. **Semantic-Hierarchy Encoder**: Constructs three semantic hierarchy levels for each modality.
2. **Intra-level Controlled Exchange Domain (IntraCED)**: Performs controlled cross-modal exchange independently at each level.
3. **Inter-level Collaborative Aggregation Domain (InterCAD)**: Synchronizes and aggregates the final task representation across levels.

### Key Designs

#### 1. Semantic-Hierarchy Encoder

For each modality $m \in \{L, V, A\}$, three-level features of uniform width $d$ are constructed:

$$H_\ell^{(m)} = \text{LN}(Z_\ell^{(m)} W_\ell^{(m)} + P_\ell^{(m)})$$

- **Language modality**: Early/middle/late layers of pretrained BERT → lexical-syntactic / phrase-level sentiment / discourse intent.
- **Visual/acoustic modalities**: Three-stage TCN with increasing receptive fields → local appearance / part structure / long-range scene context.

#### 2. IntraCED: Intra-Level Controlled Exchange

**Shared-private orthogonal decomposition**: Orthonormal bases $U_\ell^{sh}$ and $U_{\ell,m}^{pr}$ are learned via Stiefel parameterization:

$$h_{\ell,t,sh}^{(m)} = h_{\ell,t}^{(m)} P_\ell^{sh}, \quad h_{\ell,t,pr}^{(m)} = h_{\ell,t}^{(m)} P_{\ell,m}^{pr}$$

Only shared components participate in cross-modal exchange; private components are fully isolated.

**Controlled Token Budget**: Not all shared tokens are worth exchanging. The shared evidence strength $e_{\ell,t}^{(m)} = \|h_{\ell,t,sh}^{(m)}\|_2$ is measured for each token, mapped to activation weights via learnable scales and level-specific thresholds, and projected onto a truncated simplex to enforce sparsity:

$$\boldsymbol{\alpha}_\ell^{(m)} = \text{Proj}_{\Delta(B_\ell)}(\tilde{\boldsymbol{\alpha}}_\ell^{(m)})$$

where $B_\ell$ is a learnable budget controlling the number of tokens participating in exchange.

**Three-modality shared-space exchange**: Each modality queries the shared token pool of the remaining modalities:

$$\tilde{h}_{\ell,t,sh}^{(m)} = \alpha_{\ell,t}^{(m)} \text{Attn}(Q_{\ell,t}^{(m)}, K_\ell^{(-m)}, V_\ell^{(-m)})$$

The budget $\alpha$ controls how much external evidence each token absorbs.

#### 3. InterCAD: Inter-Level Collaborative Aggregation

**Cross-level semantic synchronization**: Mean pooling + LN are applied to the shared/private streams at each level and modality to obtain summaries $s_\ell^{(m)}$ and $p_\ell^{(m)}$. Level weights $\omega = [\omega_1, \omega_2, \omega_3]$ are computed via MLP + softmax:

$$\bar{s}^{(m)} = \sum_{\ell=1}^3 \omega_\ell s_\ell^{(m)}, \quad \bar{p}^{(m)} = \sum_{\ell=1}^3 \omega_\ell p_\ell^{(m)}$$

**Modality selection and private aggregation**:
- Shared path: A global context $\bar{g}$ serves as query; per-modality $\bar{s}^{(m)}$ serve as keys; scaled dot-product attention selects the most informative modality.
- Private path: Confidence-gated aggregation $\eta_m = \sigma(w_p^\top \text{LN}(W_p \bar{p}^{(m)}))$.

Final task representation: $\hat{y} = f_\theta(z_{sh} \oplus u_{pr})$

### Loss & Training

$$\mathcal{L}_{all} = \mathcal{L}_{task} + \lambda_{inter} \mathcal{L}_{Inter} + \lambda_{intra} \mathcal{L}_{Intra}$$

**Intra-level regularization $\mathcal{L}_{Intra}$**: A whitening cross-correlation identifiability regularizer that penalizes correlation between private streams of different modalities and between private-shared streams of the same modality.

**Inter-level regularization $\mathcal{L}_{Inter}$**: Three constraints —
- $\mathcal{L}_{pr}$: Reduces cross-level private redundancy.
- $\mathcal{L}_{sp}$: Suppresses cross-level shared-private leakage.
- $\mathcal{L}_{mix}$: Penalizes the simultaneous activation of semantically incompatible level pairs.

Training configuration: SGD (momentum 0.9), lr 1e-3, weight decay 1e-4, batch size 64, 100 epochs, A100 GPU.

## Key Experimental Results

### Main Results

**Table 1: Audio-Visual Benchmarks (Acc% / F1%)**

| Method | CREMA-D Acc | KS Acc | AVE Acc | UCF101 Acc |
|--------|-------------|--------|---------|------------|
| ARL | 76.46 | 74.09 | 72.61 | 83.06 |
| D&R | 73.52 | 69.10 | 69.62 | 82.11 |
| **CLCR** | **77.92** | **75.41** | **73.82** | **83.64** |

**Table 2: Multimodal Sentiment Analysis (CMU-MOSI / CMU-MOSEI)**

| Method | MOSI MAE↓ | MOSI Acc-2 | MOSEI MAE↓ | MOSEI Acc-2 |
|--------|-----------|------------|------------|-------------|
| DLF | 0.731 | 85.06 | 0.536 | 85.42 |
| EMOE | 0.710 | 85.4 | 0.536 | 85.3 |
| **CLCR** | **0.678** | **88.05** | **0.511** | **87.96** |

### Ablation Study

**Table 3: Key Component Ablation (MOSI MAE↓ / KS Acc)**

| Variant | MOSI MAE | KS Acc |
|---------|----------|--------|
| w/o Hierarchy | 0.720 | 71.9 |
| w/o IntraCED | 0.703 | 73.0 |
| w/o InterCAD | 0.699 | 73.4 |
| Full Mix (shuffled levels) | 0.743 | 70.3 |
| w/o both regularizations | 0.725 | 71.2 |
| **CLCR (full)** | **0.678** | **75.41** |

### Key Findings

1. **Semantic hierarchy is essential**: Removing the hierarchical structure causes the largest performance drop; Full Mix (completely shuffled) performs worst.
2. **IntraCED is more critical than InterCAD**: Removing IntraCED generally yields a larger performance drop, indicating that intra-level shared/private separation and controlled exchange are the key factors.
3. **Optimal sparsity of token budget**: Performance peaks at participation rate $r \approx 0.68$ ($\gamma \approx 1.0$); fully dense exchange performs worst.
4. **Noise robustness**: Under Gaussian noise injection experiments, CLCR exhibits the smallest performance degradation compared to baseline methods.
5. **Adaptive modality importance**: The language modality dominates on MOSI, while visual modality receives the highest weight on KS; CLCR adapts automatically.

## Highlights & Insights

1. **Problem formulation of cross-level semantic asynchrony**: The paper provides an information bottleneck perspective explaining why mixing features across levels degrades representation quality.
2. **Controlled token budget mechanism**: Differentiable sparse token selection is achieved via truncated simplex projection, avoiding noisy dense fusion.
3. **Dual protection of shared-private structure**: Orthogonal projection (structural constraint) and whitening cross-correlation regularization (statistical constraint) are applied jointly.
4. **Comprehensive validation across six benchmarks**: Covers four task types — emotion recognition, event localization, sentiment analysis, and action recognition.

## Limitations & Future Work

1. The three-level hierarchy is a hard-coded design; different tasks may require a different number of levels.
2. Computational overhead analysis is insufficient — actual training time for whitening operations and Stiefel parameterization is not reported.
3. Validation is limited to classification/regression tasks and has not been extended to generative multimodal tasks.
4. Handling of missing-modality scenarios (addressed only in ablation analysis) has not been developed into a systematic solution.

## Related Work & Insights

- **MISA**: A classical approach for modality-invariant and modality-specific subspace decomposition; CLCR extends this by introducing hierarchical structure.
- **DMD**: Graph-based cross-modal knowledge distillation; CLCR replaces distillation with controlled attention.
- **ARL**: Dual-path calibration strategy; CLCR achieves analogous functionality through the modality selection mechanism in InterCAD.
- The token budget mechanism is transferable to vision-language pretraining for controlling the granularity of cross-modal interaction.

## Rating

- **Novelty**: ★★★★☆ — The problem formulation of cross-level semantic asynchrony and the controlled exchange design are original.
- **Technical Depth**: ★★★★★ — Orthogonal decomposition + truncated simplex projection + whitening regularization; theoretically well-grounded.
- **Experimental Thoroughness**: ★★★★★ — Six benchmarks, detailed ablations, t-SNE visualization, noise robustness, and hyperparameter sensitivity analysis.
- **Writing Quality**: ★★★★☆ — Framework diagrams are clear, but the high density of notation raises the reading barrier.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VideoChat-M1: Collaborative Policy Planning for Video Understanding via Multi-Agent Reinforcement Learning](videochatm1_collaborative_policy_planning_for_vide.md)
- [\[ICLR 2026\] From Vicious to Virtuous Cycles: Synergistic Representation Learning for Unsupervised Video Object-Centric Learning](../../ICLR2026/video_understanding/from_vicious_to_virtuous_cycles_synergistic_representation_learning_for_unsuperv.md)
- [\[AAAI 2026\] Explicit Temporal-Semantic Modeling for Dense Video Captioning via Context-Aware Cross-Modal Interaction](../../AAAI2026/video_understanding/explicit_temporal-semantic_modeling_for_dense_video_captioning_via_context-aware.md)
- [\[AAAI 2026\] SUGAR: Learning Skeleton Representation with Visual-Motion Knowledge for Action Recognition](../../AAAI2026/video_understanding/sugar_learning_skeleton_representation_with_visual-motion_knowledge_for_action_r.md)
- [\[CVPR 2026\] OpenMarcie: Dataset for Multimodal Action Recognition in Industrial Environments](openmarcie_dataset_for_multimodal_action_recognition_in_industrial_environments.md)

</div>

<!-- RELATED:END -->
