---
title: >-
  [Paper Note] STAR: Learning Diverse Robot Skill Abstractions through Rotation-Augmented Vector Quantization
description: >-
  [ICML 2025][Robotics][Skill Abstraction] The STAR framework is proposed to address the codebook collapse problem of VQ-VAEs through Rotation-Augmented Residual Skill Quantization (RaRSQ), while modeling dependencies between skills via a Causal Skill Transformer (CST). It achieves an overall success rate of 93.6% on the LIBERO benchmark, outperforming the previous SOTA method QueST by approximately 12%.
tags:
  - "ICML 2025"
  - "Robotics"
  - "Skill Abstraction"
  - "Vector Quantization"
  - "Codebook Collapse"
  - "Autoregressive Skill Synthesis"
  - "Robot Manipulation"
date: 2026-05-08
content_hash: 6e2f42f5c14f4071
---

# STAR: Learning Diverse Robot Skill Abstractions through Rotation-Augmented Vector Quantization

**Conference**: ICML 2025 Spotlight  
**arXiv**: [2506.03863](https://arxiv.org/abs/2506.03863)  
**Code**: [Available](https://STAR.github.io)  
**Area**: Robotics  
**Keywords**: Skill Abstraction, Vector Quantization, Codebook Collapse, Residual Quantization, Imitation Learning

## TL;DR

The STAR framework is proposed to address the codebook collapse problem of VQ-VAEs through Rotation-Augmented Residual Skill Quantization (RaRSQ), while modeling dependencies between skills via a Causal Skill Transformer (CST). It achieves an overall success rate of 93.6% on the LIBERO benchmark, outperforming the previous SOTA method QueST by approximately 12%.

## Background & Motivation

Multi-task visuomotor policy learning remains a core challenge in the field of robotic manipulation. While individual manipulation tasks already face difficulties such as multimodal action distributions, action spaces in multi-task scenarios are highly entangled, with features of different tasks crossing and overlapping with each other.

**Intuitive Solution**: Decompose complex actions into reusable skill abstractions to form a hierarchical framework. Recent studies utilize latent variable models such as VQ-VAEs to discretize continuous action spaces into skill representations.

**Two Key Bottlenecks**:

**Codebook Collapse**: Most codebook vectors in VQ-VAE training go unused, with only a few being frequently selected. The root cause is that the Straight-Through Estimator (STE) applies the same gradient to all encoder outputs mapped to the same codebook vector, ignoring the geometric relationships between embeddings.

**Difficulty in Skill Composition**: Existing methods fail to model dependencies between different skills, leading to a lack of temporal consistency in action sequences for long-horizon tasks.

**Core Idea**: Encoding the geometric relationships between action sequences into the residual quantization process is key to learning diverse and reusable skills. This is achieved by replacing the "one-size-fits-all" gradient propagation of STE with rotation matrices.

## Method

### Overall Architecture

STAR adopts a **two-stage training** scheme: Stage 1 trains RaRSQ to learn discrete skill representations from expert demonstrations; Stage 2 freezes RaRSQ and trains CST to predict skill sequences based on observations.

### Key Designs

#### 1. Rotation-Augmented Residual Skill Quantization (RaRSQ)

**Function**: Encodes continuous action sequences into hierarchical discrete skills while preventing codebook collapse.

**Mechanism**: Replace STE gradient propagation with rotation matrices at each layer of residual quantization. Given the encoder output $\mathbf{z} = \phi(\mathbf{a}_{t:t+T})$ and the initial residual $\mathbf{r}_0 = \mathbf{z}$, for each layer $d$:

$$k_d = \arg\min_k \|\mathbf{r}_{d-1} - \mathbf{e}_{(d,k)}\|_2^2$$

$$\tilde{\mathbf{q}}_d = \text{sg}\left[\frac{\|\mathbf{e}_{(d,k_d)}\|}{\|\mathbf{r}_{d-1}\|}\mathbf{R}_d\right]\mathbf{r}_{d-1}, \quad \mathbf{r}_d = \mathbf{r}_{d-1} - \tilde{\mathbf{q}}_d$$

The rotation matrix $\mathbf{R}_d = \mathbf{I} - 2\hat{\mathbf{r}}_d\hat{\mathbf{r}}_d^T + 2\hat{\mathbf{q}}_d\hat{\mathbf{r}}_{d-1}^T$ preserves the angular information between embeddings. During backpropagation, $\frac{\partial\hat{\mathbf{z}}}{\partial\mathbf{r}_{d-1}} = \frac{\|\mathbf{e}_{(d,k_d)}\|}{\|\mathbf{r}_{d-1}\|}\mathbf{R}_d$, enabling embeddings at different positions to obtain differentiated gradients based on their relative angles to the codebook vectors.

**Design Motivation**: Applying the same gradient to all points within the same partition using STE leads to convergent embedding collapse. The rotation mechanism preserves angular information—pushing nearby directions away and pulling different directions closer—thereby maintaining codebook diversity. The residual structure achieves exponential representation capacity with $K^D$ combinations.

#### 2. Causal Skill Transformer (CST)

**Function**: Predicts skill code sequences autoregressively and generates continuous actions given multimodal observations.

**Mechanism**: Models skill selection via hierarchical conditional probability:

$$P(k_1,\ldots,k_D|\mathbf{o}_{t-h:t},\boldsymbol{\tau}) = \prod_{d=1}^D P(k_d|k_{<d},\mathbf{g}_t)$$

**Action Refinement**: Introduces an offset prediction head inspired by BeT to compensate for discretization precision loss: $\hat{\mathbf{a}}_t = \psi(\sum_d \mathbf{R}_d\mathbf{e}_{d,k_d}) + \zeta_{\text{ref}}(\mathbf{g}_t)$

**Design Motivation**: The coarse-to-fine hierarchy of residual quantization naturally forms a skill dependency structure (coarse-grained motion primitives to fine adjustments), which perfectly fits the autoregressive mechanism.

#### 3. Inference Process

Nucleus sampling with temperature $\tau$ and threshold $p$ is used to sample skill codes. Action sequences are then generated via the decoder and offset, with rolling replanning.

### Loss & Training

**Stage 1**: $\mathcal{L} = \|\mathbf{a} - \psi(\hat{\mathbf{z}})\|_2^2 + \beta\sum_d\|\text{sg}[\mathbf{r}_{d-1}] - \tilde{\mathbf{q}}_d\|_2^2$ (reconstruction + commitment)

**Stage 2**: $\mathcal{L} = -\sum_d\log P(k_d^*|k_{<d},\mathbf{g}_t) + \lambda\|\mathbf{a}_t - \hat{\mathbf{a}}_t\|^2$ (skill prediction + action refinement)

## Key Experimental Results

### Main Results (LIBERO Benchmark, Success Rate %)

| Method | Object | Spatial | Goal | Long | 90 | Overall |
|------|--------|---------|------|------|----|---------|
| OpenVLA | 88.4 | 84.7 | 79.2 | 53.7 | - | 76.5 |
| VQ-BeT | 90.3 | 88.7 | 61.3 | 59.7 | 84.2 | 76.8 |
| QueST | 90.0 | 84.5 | 76.7 | 69.1 | 87.4 | 81.5 |
| **STAR** | **98.3** | **95.5** | **95.0** | **88.5** | **90.8** | **93.6** |

MetaWorld MT50: STAR achieves 92.7%, outperforming all baselines by 2.1%-5.4%.

### Ablation Study

| Configuration | Object | Long | Overall | Description |
|------|--------|------|---------|------|
| STAR Full | 98.3 | 88.5 | **93.6** | - |
| w/o AR | 95.3 | 83.3 | 89.5 | Remove autoregressive, -4.1% |
| w/o Rotation | 93.7 | 85.7 | 91.0 | Remove rotation, -2.6% |
| w/o Both | 93.3 | 81.5 | 87.8 | Remove both, -5.8% (synergistic effect) |

### Key Findings

1. **Codebook Utilization**: RaRSQ utilizes all 16/16 code words (100%), whereas VQ-VAE uses only 7/16 (43.8%), with mean frequencies of 6.25% vs. 14.29%.
2. **Most Significant Improvement in Complex Tasks**: LIBERO-Long +19.4%, Goal +18.3%—scenarios where codebook collapse has the most severe impact.
3. **Real-world Robots**: Drawer manipulation success rate of 30% (vs. VQ-BeT 10%, QueST 0%), object placement success rate of 60% (vs. VQ-BeT 30%, QueST 40%).
4. Rotation and autoregression exhibit a synergistic effect: removing both (-5.8%) leads to a larger drop than individual removals (-4.1% and -2.6%).

## Highlights & Insights

- Understanding codebook collapse from the geometric perspective of gradient propagation is a precise entry point—STE's "one-size-fits-all" gradient is identified as the root cause.
- Rotation augmentation is a lightweight yet highly efficient technique that only modifies the gradient propagation path without increasing inference computational overhead.
- The two-stage design clearly separates "skill learning" from "skill composition".
- The combination of residual quantization and autoregression naturally forms a coarse-to-fine skill hierarchy, which aligns well with the structure of manipulation tasks.

## Limitations & Future Work

- The codebook size $K$ and quantization depth $D$ require manual tuning, lacking an adaptive mechanism.
- As an imitation learning method, it heavily relies on the quality and coverage of expert demonstration data.
- The real-world robot experiments are relatively small-scale (only 2 tasks with 10 trials each), and there remains room for improvement in overall success rate.
- Comparison against recent generative methods such as Diffusion Policy in real-world scenarios is missing.
- The action refinement mechanism is inherited from BeT, which may not be the optimal design choice.

## Related Work & Insights

- Transferring the rotation technique from the field of image generation to robot skill learning demonstrates the successful migration of cross-domain methods.
- The hierarchical decomposition of skills (coarse-to-fine) aligns well with the hierarchical structure of human motion planning.
- The solution to codebook collapse provides valuable insights for all VQ-VAE downstream tasks (e.g., speech, image, and video compression).
- Nucleus sampling during the inference stage introduces controllable diversity to action generation, which benefits multimodal action distribution modeling.

## Rating
- Novelty: ⭐⭐⭐⭐ The application of the rotation mechanism in robot skill quantization is novel, although the rotation technique itself has prior precedents.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Full 5 subsets of LIBERO + MetaWorld MT50 + real-world experiments + comprehensive ablation studies + codebook analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, rigorous technical exposition, and intuitive diagrams.
- Value: ⭐⭐⭐⭐ The 93.6% success rate on LIBERO is a substantial improvement, addressing the practical bottleneck: Learning Diverse Robot Skill Abstractions through Rotation-Augmented Vector Quantization

**Conference**: ICML 2025  
**arXiv**: [2506.03863](https://arxiv.org/abs/2506.03863)  
**Code**: [https://STAR.github.io](https://STAR.github.io)  
**Area**: Robotics  
**Keywords**: Skill Abstraction, Vector Quantization, Codebook Collapse, Autoregressive Skill Synthesis, Robot Manipulation

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] CommVQ: Commutative Vector Quantization for KV Cache Compression](commvq_commutative_vector_quantization_for_kv_cache_compression.md)
- [\[ICCV 2025\] iManip: Skill-Incremental Learning for Robotic Manipulation](../../ICCV2025/robotics/imanip_skill-incremental_learning_for_robotic_manipulation.md)
- [\[ICCV 2025\] Certifiably Optimal Anisotropic Rotation Averaging](../../ICCV2025/robotics/certifiably_optimal_anisotropic_rotation_averaging.md)
- [\[NeurIPS 2025\] Policy Compatible Skill Incremental Learning via Lazy Learning Interface](../../NeurIPS2025/robotics/policy_compatible_skill_incremental_learning_via_lazy_learning_interface.md)
- [\[CVPR 2026\] DynBridge: Bridging Imagination and Control through Interaction Dynamics for Robot Manipulation](../../CVPR2026/robotics/dynbridge_bridging_imagination_and_control_through_interaction_dynamics_for_robo.md)

</div>

<!-- RELATED:END -->
