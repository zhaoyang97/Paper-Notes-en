---
title: >-
  [Paper Note] A Practical Guide for Incorporating Symmetry in Diffusion Policy
description: >-
  [NeurIPS 2025][LLM Pretraining][Diffusion Policy] This paper presents a practical guide for incorporating symmetry into diffusion policies. Through three simple and composable methods — invariant representations (relativ…
tags:
  - "NeurIPS 2025"
  - "LLM Pretraining"
  - "Diffusion Policy"
  - "Equivariance"
  - "SE(3) Invariance"
  - "Frame Averaging"
  - "Robot Manipulation"
date: 2026-05-08
content_hash: 8021a2e417c3b3d7
---

# A Practical Guide for Incorporating Symmetry in Diffusion Policy

**Conference**: NeurIPS 2025
**arXiv**: [2505.13431](https://arxiv.org/abs/2505.13431)  
**Code**: [https://sym-in-dp.github.io](https://sym-in-dp.github.io) (project page)  
**Area**: Diffusion Models / Robot Policy Learning
**Keywords**: Diffusion Policy, Equivariance, SE(3) Invariance, Frame Averaging, Robot Manipulation

## TL;DR
This paper presents a practical guide for incorporating symmetry into diffusion policies. Through three simple and composable methods — invariant representations (relative trajectory actions + eye-in-hand perception), equivariant visual encoders, and Frame Averaging — the proposed approach achieves performance on par with or exceeding fully equivariant diffusion policies across 12 MimicGen tasks, while substantially reducing implementation complexity.

## Background & Motivation

**Background**: Equivariant neural networks have demonstrated advantages in sample efficiency and generalization for robot policy learning, yet their practical adoption is limited by implementation complexity. Each policy framework (Q-learning, actor-critic, diffusion) requires its own custom equivariant design.

**Limitations of Prior Work**: Fully equivariant diffusion policies (e.g., EquiDiff) require specialized equivariant layers and complex mathematical derivations, and are incompatible with pretrained encoders. This forces practitioners to choose between leveraging symmetry and maintaining implementation simplicity.

**Key Challenge**: The performance gains from symmetry versus the high barrier of equivariant architecture design appear mutually exclusive.

**Goal**: Can symmetry be incorporated into standard Diffusion Policy in a simple, modular manner, without end-to-end equivariant design?

**Key Insight**: The authors observe that symmetry can be introduced at the representation level (action representations + perception) rather than at the network architecture level, greatly simplifying implementation.

**Core Idea**: Decompose symmetry into two independent modules — invariant representations and equivariant encoders — thereby avoiding the complexity of end-to-end equivariant design.

## Method

### Overall Architecture
The input consists of a single RGB image from an eye-in-hand camera and the robot state; the output is a relative trajectory action over 16 future steps. The framework comprises three composable modules: (1) invariant representations — replacing absolute trajectory actions with relative ones; (2) equivariant visual encoder — substituting the standard CNN with an equivariant CNN or applying Frame Averaging; (3) a standard Diffusion Policy denoising head — requiring no modification.

### Key Designs

1. **Symmetry Analysis of Action Representations**:

    - **Function**: Systematic analysis of the symmetry properties of three action representations (absolute / relative / delta trajectories).
    - **Mechanism**: Absolute trajectories $a = \{A_t, A_{t+1}, ...\}$ are equivariant under global SE(3) transformations ($g \cdot a = \{gA_t, ...\}$), whereas relative trajectories $a^r = \{A_t^r, ...\}$ (each step expressed relative to the current gripper frame) are invariant ($g \cdot a^r = a^r$), since relative poses as right-multiplication factors are unaffected by global transformations.
    - **Design Motivation**: Invariant action representations reduce the learning function space, facilitating training. Simply switching from absolute to relative actions yields a 5–7% performance improvement.

2. **SE(3)-Invariant Policy Learning (Eye-in-Hand Perception + Relative Trajectories)**:

    - **Function**: Demonstrates that the combination of eye-in-hand camera observations and relative trajectory actions is naturally SE(3)-invariant.
    - **Mechanism**: Eye-in-hand images are invariant under global SE(3) transformations ($g \cdot I_t = I_t$), and relative trajectories are also invariant; therefore the learned function $\bar{\pi}: o \mapsto a^r$ is invariant. Reconstructing the absolute trajectory via $A_{t+i} = T_t A_{t+i}^r$ yields an automatically equivariant policy $\pi(g \cdot o) = g \cdot \pi(o)$.
    - **Design Motivation**: SE(3) equivariance is achieved without any equivariant network layers, greatly simplifying implementation. Experiments show a 4.7% improvement over the original Diffusion Policy.

3. **Equivariant Visual Encoder + Frame Averaging**:

    - **Function**: Replaces the CNN encoder in standard diffusion policy with an equivariant encoder, or uses Frame Averaging to equivariantize a pretrained encoder.
    - **Mechanism**: Frame Averaging converts an arbitrary function $\Phi$ into an equivariant function via group averaging: $\Psi(x) = \frac{1}{|G|} \sum_{g \in G} \rho_y(g) \Phi(\rho_x(g)^{-1} x)$. Using the $C_8$ group (8 discrete rotations), the pretrained ResNet-18 processes 8 rotated versions of the input, and the outputs are aggregated via a weighted average using the regular representation.
    - **Design Motivation**: Equivariant encoders extract rotation-aware features (+9.1%), though they are trained from scratch. Frame Averaging combines the expressive power of pretrained weights with equivariance, and represents the recommended best practice.

### Loss & Training
- Standard DDPM training loss: $\mathcal{L} = \|\varepsilon_\theta(o, a + \varepsilon^k, k) - \varepsilon^k\|^2$
- AdamW optimizer (lr=$10^{-4}$, wd=$10^{-6}$) with cosine learning rate schedule and 500-step warmup
- Training for 600 epochs, evaluated every 10 epochs (60 evaluations total); best mean success rate is reported
- Delta trajectories are theoretically invariant but perform poorly empirically — hypothesized to result from insufficient information for the diffusion denoising process

## Key Experimental Results

### Main Results
12 robot manipulation tasks from MimicGen, trained on 100 expert demonstrations, evaluated over 50 rollouts.

| Method | Mean Success Rate | Input | Architectural Complexity |
|--------|------------------|-------|--------------------------|
| Pretrain + FA (Ours, best) | 61.4% | Single RGB eye-in-hand | Low |
| EquiDiff (Vo) | 63.9% | 4×RGBD | High |
| EquiDiff (Im) | 53.7% | In-Hand+Ext | High |
| Equi Enc (Ours) | 55.8% | Single RGB eye-in-hand | Medium |
| CNN Enc (baseline) | 46.7% | Single RGB eye-in-hand | Low |
| Original Diffusion Policy | 42.0% | In-Hand+Ext | Low |

### Ablation Study

| Configuration | Mean Success Rate | Notes |
|---------------|-----------------|-------|
| Relative traj. + eye-in-hand | 46.7% | Invariant representation only, CNN encoder |
| Absolute traj. + eye-in-hand | 40.8% | No invariance |
| Delta traj. + eye-in-hand | 37.9% | Theoretically invariant but poor information content |
| Equi Enc vs. CNN Enc | +9.1% | Contribution of equivariant encoder |
| Pretrain+FA vs. Pretrain | +4.1% | Contribution of Frame Averaging |

### Key Findings
- **Relative trajectories consistently outperform absolute trajectories**: improvements in 10 out of 12 tasks, averaging +5.9% (eye-in-hand) / +7.4% (eye-in-hand + external), achievable by simply changing the coordinate frame.
- **Equivariant encoder contributes the most**: switching from CNN to equivariant encoder yields a 9.1% improvement, indicating that symmetry-aware features are more critical than invariant representations alone.
- **Delta trajectories underperform expectations**: despite being theoretically as invariant as relative trajectories, performance is 8.8% lower, attributed to the lack of temporal structure information in velocity-based representations.
- **Single monocular RGB eye-in-hand approaches 4×RGBD EquiDiff**: Pretrain+FA falls only 2.5% short, with far superior practical utility.
- **Failure modes of eye-in-hand are occlusion and limited field of view**: in long-horizon tasks such as Coffee Preparation, insufficient viewpoint information degrades performance.

## Highlights & Insights
- **Coordinate frame choice is symmetry**: simply switching action representations from world frame to gripper frame yields a 5–7% gain — an insight broadly applicable to all policy learning methods at virtually zero cost.
- **Frame Averaging bridges pretraining and equivariance**: this is the first application of Frame Averaging to robot diffusion policies, enabling a pretrained ResNet to become automatically equivariant at inference time, and directly transferable to any policy framework using pretrained encoders.
- **Modular design outperforms end-to-end equivariance**: introducing symmetry separately at the representation and encoder levels avoids the complexity of equivariant denoising network design. This divide-and-conquer philosophy is transferable to other policy frameworks such as ACT and VLA.

## Limitations & Future Work
- **Limited field of view with eye-in-hand**: single monocular RGB is insufficient for occlusion-heavy and long-horizon tasks; fisheye cameras or temporal memory mechanisms are suggested.
- **Simulation-only validation**: no real-robot experiments are conducted; UMI systems are a potential deployment platform.
- **Computational overhead**: Frame Averaging requires 8 separate forward passes, incurring approximately twice the GPU time of the original DP (though faster than EquiDiff).
- **Delta trajectories remain underutilized**: theoretically invariant but empirically weak; better sequential structure design may be needed.
- **Continuous groups unexplored**: only discrete $C_8$ is used; whether continuous SO(2) equivariance yields further improvements remains an open question.

## Related Work & Insights
- **vs. EquiDiff [wang2024equivariant]**: EquiDiff requires a complex end-to-end equivariant design with 4×RGBD inputs and voxelization. The proposed approach achieves comparable performance using only a single RGB eye-in-hand camera and modular symmetry, with substantially higher practical utility.
- **vs. Diffusion Policy [chi2023diffusion]**: the original DP incorporates no symmetry; modifying only the action representation and encoder yields an average improvement of 19.4%.
- **vs. EquiBot [yang2024equibot]**: similarly an equivariant diffusion policy, but EquiBot requires end-to-end SE(3)-equivariant design, whereas the proposed method is more modular.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Not an entirely new technique, but the systematic representation analysis combined with Frame Averaging applied to DP constitutes a novel contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 12 tasks, extensive ablations, comparison of 3 methods, and 3 action representations × 2 observation types — highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Theoretical derivations are clear, experimental organization is systematic, and figures are intuitive.
- **Value**: ⭐⭐⭐⭐ Highly practical for the robot policy learning community, providing a low-cost best-practice guide for incorporating symmetry.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Composition and Alignment of Diffusion Models using Constrained Learning](composition_and_alignment_of_diffusion_models_using_constrai.md)
- [\[NeurIPS 2025\] Next Semantic Scale Prediction via Hierarchical Diffusion Language Models](next_semantic_scale_prediction_via_hierarchical_diffusion_language_models.md)
- [\[NeurIPS 2025\] Deep Compositional Phase Diffusion for Long Motion Sequence Generation](deep_compositional_phase_diffusion_for_long_motion_sequence_generation.md)
- [\[ICLR 2026\] RECON: Robust symmetry discovery via Explicit Canonical Orientation Normalization](../../ICLR2026/llm_pretraining/recon_robust_symmetry_discovery_via_explicit_canonical_orientation_normalization.md)
- [\[ICCV 2025\] FlowMo: Flow to the Mode — Mode-Seeking Diffusion Autoencoders for State-of-the-Art Image Tokenization](../../ICCV2025/llm_pretraining/flow_to_the_mode_mode-seeking_diffusion_autoencoders_for_state-of-the-art_image_.md)

</div>

<!-- RELATED:END -->
