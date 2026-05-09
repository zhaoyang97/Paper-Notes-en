---
title: >-
  [Paper Note] Self-Consistency for LLM-Based Motion Trajectory Generation and Verification
description: >-
  [CVPR2026][Multimodal VLM][self-consistency] This paper extends the self-consistency paradigm of LLMs from natural language reasoning to the visual domain. It defines shape families for motion trajectories via a Lie transformation group hierarchy, and clusters multiple LLM-sampled trajectories under transformation-invariant distance metrics to achieve unsupervised trajectory generation improvement (+4–6%) and verification (precision +11.8%), without any training.
tags:
  - CVPR2026
  - Multimodal VLM
  - self-consistency
  - motion trajectory
  - Lie transformation groups
  - shape family
  - unsupervised verification
date: 2026-05-08
content_hash: 7b44aec49d853328
---

# Self-Consistency for LLM-Based Motion Trajectory Generation and Verification

**Conference**: CVPR2026
**arXiv**: [2603.29301](https://arxiv.org/abs/2603.29301)
**Code**: [majiaju.io/trajectory-self-consistency](https://majiaju.io/trajectory-self-consistency)
**Area**: Multimodal VLM
**Keywords**: self-consistency, motion trajectory, Lie transformation groups, shape family, unsupervised verification

## TL;DR
This paper extends the self-consistency paradigm of LLMs from natural language reasoning to the visual domain. It defines shape families for motion trajectories via a Lie transformation group hierarchy, and clusters multiple LLM-sampled trajectories under transformation-invariant distance metrics to achieve unsupervised trajectory generation improvement (+4–6%) and verification (precision +11.8%), without any training.

## Background & Motivation
**Self-consistency** is an effective technique in LLM reasoning: sample multiple times → find the most consistent answer. In text domains such as mathematical reasoning, consistency checking is straightforward (direct numerical comparison). However, LLMs are also widely used to generate visual outputs (SVG, 3D scenes, animations, etc.), raising the question of how to extend self-consistency to the visual domain.

**Core Challenge**: In the visual domain, two outputs can almost never match at the pixel level. The deeper reason is the **under-specification** of prompts — "move the circle in a logarithmic spiral path" does not describe a single trajectory but a **shape family** (encompassing all logarithmic spirals at different positions, scales, and orientations). It is therefore necessary to define when two trajectories should be considered "consistent."

**Core Idea**: Model shape families as a prototype trajectory plus a geometric transformation group (rigid, similarity, affine, etc.); two trajectories are considered consistent if one can be transformed into the other under the allowed transformations of the group. The hierarchy of transformation groups is used to automatically recover shape families.

## Method

### Overall Architecture
Given a prompt describing the desired trajectory → (1) sample $N$ diverse trajectories using an LLM → (2) cluster under transformation-invariant distance metrics for each transformation group in the Lie group hierarchy → (3) select the most appropriate transformation group via a decision criterion → (4) use the centroid of the largest cluster as the self-consistent generation, or check whether a new trajectory belongs to this shape family for verification.

### Key Designs

1. **Shape Families and the Lie Transformation Group Hierarchy**: Define a shape family $\mathcal{F}(o, W) = \{w(o) | w \in W\}$ (prototype trajectory $o$ + transformation group $W$). A group hierarchy is constructed: rigid SE(2) ⊂ rigid+reflection E(2) ⊂ similarity Sim⁺(2) ⊂ similarity+reflection Sim(2) ⊂ affine Aff(2), along with anisotropic similarity groups. Each group has a corresponding invariant distance metric $d_W(t_1, t_2) = \min_{w \in W} \frac{1}{n}\sum_i \|w(t_{1,i}) - t_{2,i}\|^2$, solved via a generalized ICP algorithm.

2. **Two Unsupervised Decision Criteria** for selecting transformation group $W$:

    - **Majority-Consensus**: Traverse the hierarchy upward from the most restrictive group, selecting the first group for which the largest cluster exceeds 50%. This is conservative (favoring stricter groups), yielding high precision but low recall.
    - **Hierarchical-Consistency**: Traverse downward from the most permissive group, selecting the strictest group that does not cause the largest cluster to lose members. This better balances precision and recall.

3. **Diversity Sampling Strategy**: Rather than sampling independently and repeatedly, the LLM is prompted to generate $k$ trajectories in one call covering the "tails" of the distribution, sampling in batches until $N$ trajectories are obtained.

4. **Verification**: After recovering shape family $\mathcal{F}(o, W)$, a query trajectory $t$ is deemed consistent with the prompt if its distance to prototype $o$ under $d_W$ is $< \tau$.

### Loss & Training
- **Fully unsupervised and training-free**; only LLM API access is required.
- Hyperparameters: $N=19$ sampled trajectories, $n=100$ resampled points, $\tau$ as clustering threshold (insensitive: F1 varies only 7.2% over a 32× range).
- Average single distance computation: 67 ms (CPU).

## Key Experimental Results

### Trajectory Generation Accuracy

| Method | Decision Criterion | GPT-4.1 | GPT-5 |
|--------|--------------------|---------|-------|
| LLM-Direct | - | 62.1% | 79.1% |
| Ours | Majority-Consensus | **68.0%** | **83.3%** |
| Ours | Hierarchical-Consistency | 66.7% | 82.6% |
| Ours | Oracle (ground-truth $W$) | 68.5% | 83.5% |

### Trajectory Verification

| Method | Precision | Recall | F1 |
|--------|-----------|--------|----|
| GPT-4.1 (VLM) | 62.0 | 96.9 | 75.6 |
| GPT-5 (VLM) | 74.0 | 84.7 | 79.0 |
| Ours (Majority-Consensus) | **85.8** | 66.1 | 74.6 |
| Ours (Hierarchical-Consistency) | 80.5 | 89.0 | **84.6** |
| Ours (Oracle) | 87.9 | 83.3 | 85.6 |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|------------|-------|
| $N=10$ samples | F1 near saturation | 10 samples already provide sufficient signal |
| $\tau$ sweep 0.25–8.0 | F1 varies only 7.2% | Insensitive to threshold |
| Multi-prototype extension | F1: 71.0→88.9 | Handles ambiguous prompts by returning multiple large clusters |

### Key Findings
- Unsupervised Majority-Consensus approaches the Oracle upper bound (68.0 vs. 68.5 on GPT-4.1).
- GPT-4.1 as a VLM verifier exhibits strong positive bias (predicted positive rate 90%, true base rate 50%), yielding only 62% precision.
- Self-consistency verification precision exceeds the VLM baseline by 11.8% (85.8 vs. 74.0).
- When Majority-Consensus errs, 95.6% of errors select an overly strict group; when Hierarchical-Consistency errs, 80.6% select an overly permissive group — the two criteria are complementary.
- Performance stabilizes at $N \geq 10$; large-scale sampling is unnecessary.

## Highlights & Insights
- **Extending self-consistency from discrete to continuous geometric domains**: replacing simple identity matching with transformation-group-based consistency is a conceptually significant generalization.
- **Elegant exploitation of Lie group hierarchy**: different shape families require different transformation groups, and the hierarchy provides a framework for unsupervised automatic selection.
- **Unique finding that verification benefits more than generation**: the advantage of self-consistency is larger on verification because shape-family membership checking is intrinsically a well-defined geometric problem once the family is recovered.
- Provides an alternative to VLM-based approaches for automatic evaluation and verification of LLM visual generation.

## Limitations & Future Work
- Only handles shape families describable by a single prototype and transformation group; inapplicable to ambiguous descriptions such as "curved path."
- Multi-prototype cases (e.g., heptagram variants {7/2} and {7/3}) require special treatment.
- ICP-based distance computation is moderately sensitive to noise and discretization error.
- Only the geometric shape of trajectories is verified; other animation attributes (velocity, timing, etc.) are not addressed.

## Related Work & Insights
- **vs. Original Self-Consistency**: The original method supports only discrete identity matching (numerical equality); this work generalizes consistency to transformation-invariant distances in the continuous geometric domain.
- **vs. MoVer**: MoVer verifies low-level animation properties using a first-order logic DSL but cannot express geometric shape families.
- **Insight**: The approach of extending self-consistency by defining domain-specific equivalence classes is generalizable to other visual and creative domains such as 3D generation and music.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Extending self-consistency from text to the visual domain is a highly innovative conceptual contribution; the Lie group hierarchy and decision criterion designs are elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ — A synthetic benchmark of 224 prompts and 2,240 verification trajectories, with comprehensive comparisons across two LLMs and multiple decision criteria; limited to synthetic data.
- Writing Quality: ⭐⭐⭐⭐⭐ — Mathematical definitions are precise, intuitions are clearly explained, and figures are excellent.
- Value: ⭐⭐⭐⭐ — Opens a new paradigm for automatic verification of LLM visual generation, though the application scope (motion graphics trajectories) is relatively narrow.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] GenDoP: Auto-regressive Camera Trajectory Generation as a Director of Photography](../../ICCV2025/multimodal_vlm/gendop_auto-regressive_camera_trajectory_generation_as_a_director_of_photography.md)
- [\[ICLR 2026\] Let's Think in Two Steps: Mitigating Agreement Bias in MLLMs with Self-Grounded Verification](../../ICLR2026/multimodal_vlm/lets_think_in_two_steps_mitigating_agreement_bias_in_mllms_with_self-grounded_ve.md)
- [\[CVPR 2026\] EvoLMM: Self-Evolving Large Multimodal Models with Continuous Rewards](evolmm_self-evolving_large_multimodal_models_with_continuous_rewards.md)
- [\[ICLR 2026\] Evaluating VLMs' Spatial Reasoning Over Robot Motion: A Step Towards Robot Planning with Motion Preferences](../../ICLR2026/multimodal_vlm/evaluating_vlms_spatial_reasoning_over_robot_motion_a_step_towards_robot_plannin.md)
- [\[CVPR 2026\] Mitigating Multimodal Hallucinations via Gradient-based Self-Reflection](mitigating_multimodal_hallucinations_via_gradient-based_self-reflection.md)

<!-- RELATED:END -->
