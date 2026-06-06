---
title: >-
  [Paper Note] Mitigating Error Accumulation in Co-Speech Motion Generation via Global Rotation Diffusion and Multi-Level Constraints
description: >-
  [AAAI 2026][Time Series][co-speech motion generation] This paper proposes GlobalDiff, a framework that, for the first time, performs diffusion-based generation in the global joint rotation space…
tags:
  - "AAAI 2026"
  - "Time Series"
  - "co-speech motion generation"
  - "global rotation"
  - "diffusion model"
  - "error accumulation"
  - "skeletal constraints"
date: 2026-05-08
content_hash: 3728bdbb66d4c1eb
---

# Mitigating Error Accumulation in Co-Speech Motion Generation via Global Rotation Diffusion and Multi-Level Constraints

**Conference**: AAAI 2026
**arXiv**: [2511.10076](https://arxiv.org/abs/2511.10076)  
**Code**: [https://xiangyue-zhang.github.io/GlobalDiff](https://xiangyue-zhang.github.io/GlobalDiff)  
**Area**: Time Series
**Keywords**: co-speech motion generation, global rotation, diffusion model, error accumulation, skeletal constraints

## TL;DR

This paper proposes GlobalDiff, a framework that, for the first time, performs diffusion-based generation in the global joint rotation space, fundamentally eliminating error accumulation in hierarchical forward kinematics. A three-level joint–bone–motion constraint scheme compensates for the structural priors lost under global representation. GlobalDiff achieves state-of-the-art performance on multi-speaker co-speech motion generation benchmarks, improving FGD by 46% over the previous best method.

## Background & Motivation

Holistic co-speech motion generation aims to synchronize full-body poses, gestures, and facial expressions with speech, serving as a key technology for natural communication in virtual characters, with broad applications in virtual humans, interactive games, and human–robot collaboration.

### Core Problem: Hierarchical Error Accumulation

Existing diffusion methods operate in **local joint rotation space**, where each joint's rotation $R_k^{\text{local}}$ is defined relative to its parent. To obtain global positions, forward kinematics (FK) must recursively compose these rotations:

$$R_k^{\text{global}} = R_1^{\text{local}} R_2^{\text{local}} \cdots R_k^{\text{local}}$$

$$q_k = \left(\prod_{i=1}^{k-1} R_i^{\text{local}}\right)(t_k - t_{k-1}) + \cdots + q_1$$

This entails several consequences:

1. **Small errors at root or intermediate joints propagate and amplify along the kinematic chain**, causing significant deviations at end-effectors (fingers, hands, feet).
2. The deeper a joint lies in the skeletal tree, the more transformations are involved and the larger the accumulated error.
3. Backpropagation through the FK chain involves deep nonlinear matrix multiplications, resulting in **unstable gradients** that hinder effective training.

### Challenges of Global Rotation

Directly predicting global rotations $R_k^{\text{global}}$ eliminates recursive dependencies but introduces new issues:
- Local rotations **implicitly preserve joint relationships** through the hierarchical skeletal structure.
- Global rotations treat each joint **independently**, discarding natural structural constraints.
- Without additional guidance, physically implausible poses or broken kinematic chains may be produced.

## Method

### Overall Architecture

GlobalDiff adopts a conditional flow matching (CFM) framework:
- **Input**: noisy motion sequence $x_t$, audio features $a$, speaker identity, and a seed motion clip.
- **Output**: clean global joint rotations and translations $x_1 \in \mathbb{R}^{T \times (J \times 6 + 3)}$ in 6D rotation format.
- **Facial expressions**: estimated directly from prosodic features and speaker ID via a shallow Transformer encoder, leveraging the approximate one-to-one correspondence between phonemes and lip movements.

**Regional decomposition**: Motion is decomposed into hand joints $\mathbf{H}_t$ and body joints $\mathbf{B}_t$, each concatenated with the corresponding expression or audio features and processed by separate Motion Generation Blocks (MGB).

**Flow matching objective**:
$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0 \sim p_0, x_1 \sim p_1} \|f_\theta(x_t, c) - x_1\|^2$$

### Key Designs

#### 1. **Global Rotation Prediction (Eliminating Error Accumulation)**

Core formulation — position computation in global space:
$$q_k = q_{\text{root}} + \sum_{(i \to j) \in \pi(k)} R_i^{\text{global}}(t_j - t_i)$$

where $\pi(k)$ denotes the unique parent-to-child path from the root to joint $k$.

**Key advantages**:
- Position computation becomes an **additive** operation along the path, avoiding recursive rotation composition.
- Each joint has a **direct and stable gradient** with respect to its global rotation.
- Hierarchical error accumulation is completely eliminated.

**Design Motivation**: In local rotation methods, backpropagating position loss requires passing through deep matrix multiplications along the FK chain, causing gradient instability. Global rotation reduces this to an additive path, yielding more stable training.

#### 2. **Joint Structure Constraint ($\mathcal{L}_j$) — Virtual Anchor Points**

**Problem**: Position loss $\mathcal{L}_{pos}$ alone provides insufficient constraint, since multiple valid rotations can yield the same joint position (rotational ambiguity), especially for terminal joints whose positions do not involve their own rotations.

**Solution**: For each joint $k$, $N$ non-coplanar virtual anchor points $\{v_k^n\}_{n=1}^N$ are defined and aligned between the predicted and ground-truth rotations:
$$\hat{v}_k^n = R_k^{\text{global}} \cdot v_k^n, \quad \tilde{v}_k^n = R_k^{\text{gt}} \cdot v_k^n$$
$$\mathcal{L}_j = \frac{1}{KN} \sum_{k=1}^{K} \sum_{n=1}^{N} \|\hat{v}_k^n - \tilde{v}_k^n\|_2^2$$

Since the anchor points span 3D space, aligning them **uniquely constrains the rotation**, resolving ambiguity.

#### 3. **Bone Structure Constraint ($\mathcal{L}_s$) — Angular Matrix**

**Problem**: Joint-level constraints only ensure local rotation fidelity for individual joints and cannot capture the global skeletal structure, which is governed by interdependent geometric relationships among bones.

**Solution**: A pairwise Angular Matrix (AM) is constructed to capture angular relationships between all bone pairs:
$$b_{k \to j} = \frac{q_j - q_k}{\|q_j - q_k\|_2}$$
$$\mathcal{A}_{kj, k'j'} = b_{k \to j}^\top b_{k' \to j'}$$
$$\mathcal{L}_s = \frac{1}{|\mathcal{B}|} \sum_{(k,j),(k',j') \in \mathcal{B}} \|\mathcal{A}_{kj,k'j'} - \tilde{\mathcal{A}}_{kj,k'j'}\|_2^2$$

Aligning predicted and ground-truth angular matrices constrains global skeletal relationships and maintains anatomically plausible bone configurations.

#### 4. **Temporal Structure Constraint ($\mathcal{L}_m$) — Multi-Scale VAE**

**Problem**: Spatial constraints cannot capture the temporal structure of motion. Co-speech gestures are inherently rhythmic and must be synchronized with speech prosody.

**Solution**: A shared multi-scale variational encoder $g(\cdot)$ extracts temporal embeddings from both predicted and ground-truth motion sequences, aligning their dynamic patterns:
$$z^{\text{gen}} = g(\hat{X}), \quad z^{\text{gt}} = g(X)$$
$$\mathcal{L}_m = \|z^{\text{gen}} - z^{\text{gt}}\|_2^2$$

### Loss & Training

Total loss $= \mathcal{L}_{\text{simple}} + \mathcal{L}_{pos} + \mathcal{L}_j + \mathcal{L}_s + \mathcal{L}_m$

- Training setup: 4 × NVIDIA V100, 1000 epochs, batch size 128, approximately 17 hours.
- Optimizer: ADAM, learning rate 1e-4.
- Number of virtual anchor points: 6.
- Seed pose frames: 8 (the last 8 frames of the previous clip serve as the seed for the next during streaming inference).

## Key Experimental Results

### Main Results

Comparison on the BEAT2 dataset (All Speakers):

| Method | FGD↓ | BeatAlign→ | Diversity→ | MSE↓ |
|--------|------|-----------|-----------|------|
| CaMN | 0.512 | 0.200 | 5.58 | — |
| EMAGE | 0.692 | 0.284 | 6.06 | 6.908 |
| HoloGest | 0.646 | 0.803 | 13.53 | — |
| RAG-GESTURE | 0.487 | 0.514 | 9.94 | — |
| **GlobalDiff (Ours)** | **0.263** | **0.404** | **8.24** | **4.144** |

Single speaker (1 Speaker):

| Method | FGD↓ | BeatAlign→ | Diversity→ | MSE↓ |
|--------|------|-----------|-----------|------|
| HoloGest | 0.534 | 0.795 | 14.15 | — |
| EMAGE | 0.570 | 0.793 | 11.41 | 7.680 |
| **GlobalDiff (Ours)** | **0.478** | **0.705** | **13.73** | **6.330** |

In the all-speaker setting, FGD improves from the second-best 0.487 to 0.263, a **~46% improvement**.

### Ablation Study

Contribution of each component (single speaker, Speaker 2):

| Configuration | FGD↓ | BeatAlign→ | Diversity→ | Note |
|--------------|------|-----------|-----------|------|
| Ours (local) | 0.594 | 0.578 | 9.33 | Local rotation baseline |
| Ours (global) | 0.592 | 0.693 | 13.08 | Switch to global rotation |
| + $\mathcal{L}_j$ | 0.574 | 0.665 | 12.30 | Add joint constraint |
| + $\mathcal{L}_j$ + $\mathcal{L}_s$ | 0.517 | 0.593 | 13.78 | Add bone constraint |
| + $\mathcal{L}_j$ + $\mathcal{L}_s$ + $\mathcal{L}_m$ | **0.478** | **0.705** | **13.73** | Full constraints |

### Key Findings

1. **Global vs. local rotation**: Switching to global rotation substantially improves BeatAlign and Diversity; fingertip trajectories are smoother (Figure 6 compares 300-frame right middle fingertip trajectories, showing high-frequency oscillations in the local method).
2. **Effect of $\mathcal{L}_j$**: Resolves anatomically implausible finger configurations (e.g., thumb flipping, pinky twisting).
3. **Effect of $\mathcal{L}_s$**: FGD improves markedly (0.574→0.517), addressing structural inconsistencies such as body tilting and unbalanced strides.
4. **Effect of $\mathcal{L}_m$**: All metrics reach their best values, ensuring rhythmic consistency and temporal smoothness.
5. **User study**: 28 participants consistently preferred GlobalDiff across three dimensions: naturalness, semantic consistency, and motion–speech synchronization.

## Highlights & Insights

- **Core insight is exceptionally clear**: Hierarchical error accumulation from local rotations is a fundamental yet long-overlooked issue; global rotation is the natural remedy.
- **Three-level constraint design is progressively structured**: joint → bone → motion, compensating for lost structural priors along three orthogonal dimensions — rotation fidelity, spatial topology, and temporal dynamics.
- **Virtual anchor points elegantly resolve rotational ambiguity**: Non-coplanar points convert rotation constraints into position constraints, neatly handling the fact that terminal joint positions do not involve their own rotations.
- **Practical streaming inference design**: The 8-frame seed mechanism enables long-sequence generation requiring only an initial 8-frame context.

## Limitations & Future Work

- Although global rotation eliminates FK error propagation, it sacrifices the **local self-consistency guarantees** inherent to hierarchical structures, which must be recovered through explicit constraints.
- The dimensionality of the angular matrix $\mathcal{A}$ grows quadratically with the number of joints, potentially incurring high computational cost for high-resolution skeletal models.
- BeatAlign in the all-speaker setting (0.404 vs. GT 0.477) reveals remaining room for improvement in rhythmic alignment.
- Validation is limited to the BEAT2 dataset; generalizability requires evaluation on additional datasets.

## Related Work & Insights

- **VQ-VAE methods** (EMAGE, SemTalk): discretization-based approaches yield limited diversity.
- **Diffusion methods** (DiffSHEG, HoloGest, RAG-Gesture): all operate in local rotation space and are subject to error accumulation.
- **MDM (Tevet et al., 2022)**: the position supervision paradigm proposed therein inspired the constraint design in this work.
- **Inspiration**: The idea of global rotation representation combined with structural constraints merits exploration in other tasks involving skeletal hierarchies, such as motion retargeting and pose estimation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First diffusion-based co-speech motion generation in global rotation space; the three-level constraint design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Quantitative + qualitative + ablation + user study, though evaluation on a single dataset is a minor limitation.
- Writing Quality: ⭐⭐⭐⭐⭐ — Problem formulation is clear, methodological derivations are rigorous, and visualizations are rich.
- Value: ⭐⭐⭐⭐⭐ — A 46% FGD improvement is substantial; the approach is generalizable and offers important insights to the motion generation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] iTimER: Reconstruction Error-Guided Irregularly Sampled Time Series Representation Learning](beyond_observations_reconstruction_error-guided_irregularly_sampled_time_series_.md)
- [\[ICLR 2026\] VoT: Event-Driven Reasoning and Multi-Level Alignment Unlock the Value of Text for Time Series Forecasting](../../ICLR2026/time_series/unlocking_the_value_of_text_event-driven_reasoning_and_multi-level_alignment_for.md)
- [\[AAAI 2026\] TSGDiff: Rethinking Synthetic Time Series Generation from a Pure Graph Perspective](tsgdiff_rethinking_synthetic_time_series_generation_from_a_pure_graph_perspectiv.md)
- [\[AAAI 2026\] M2FMoE: Multi-Resolution Multi-View Frequency Mixture-of-Experts for Extreme-Adaptive Time Series Forecasting](m2fmoe_multi-resolution_multi-view_frequency_mixture-of-experts_for_extreme-adap.md)
- [\[AAAI 2026\] Coherent Multi-Agent Trajectory Forecasting in Team Sports with CausalTraj](coherent_multi-agent_trajectory_forecasting_in_team_sports_with_causaltraj.md)

</div>

<!-- RELATED:END -->
