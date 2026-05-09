---
title: >-
  [Paper Note] AutoQD: Automatic Discovery of Diverse Behaviors with Quality-Diversity Optimization
description: >-
  [ICLR 2026][Reinforcement Learning][quality-diversity optimization] This paper proposes AutoQD, which automatically generates behavior descriptors by embedding policy occupancy measures via random Fourier features, enabling the discovery of diverse, high-quality policies in continuous control tasks without manual descriptor design. Effectiveness is demonstrated across 6 standard environments.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - quality-diversity optimization
  - behavior descriptor
  - occupancy measure
  - random Fourier features
  - policy embedding
date: 2026-05-08
content_hash: 552fbff284fc85a4
---

# AutoQD: Automatic Discovery of Diverse Behaviors with Quality-Diversity Optimization

**Conference**: ICLR 2026
**arXiv**: [2506.05634](https://arxiv.org/abs/2506.05634)
**Code**: [GitHub](https://github.com/conflictednerd/autoqd-code)
**Area**: Reinforcement Learning
**Keywords**: quality-diversity optimization, behavior descriptor, occupancy measure, random Fourier features, policy embedding

## TL;DR
This paper proposes AutoQD, which automatically generates behavior descriptors by embedding policy occupancy measures via random Fourier features, enabling the discovery of diverse, high-quality policies in continuous control tasks without manual descriptor design. Effectiveness is demonstrated across 6 standard environments.

## Background & Motivation

1. **State of the Field**: Quality-Diversity (QD) optimization aims to find collections of policies that are both high-performing and behaviorally diverse, with demonstrated success in robotic locomotion and game content generation.

2. **Limitations of Prior Work**: QD algorithms rely heavily on manually designed behavior descriptors (e.g., foot contact patterns for bipedal robots), requiring substantial domain expertise, and predefined diversity dimensions may miss interesting behavioral variants.

3. **Root Cause**: Existing unsupervised QD methods (e.g., AURORA) learn behavior spaces via autoencoder-based state reconstruction but lack a theoretical connection to policy behavior. Skill discovery methods in RL (e.g., DIAYN) require a preset number of skills and do not optimize task reward.

4. **Paper Goals**: Provide a theoretically grounded method for automatically generating behavior descriptors without domain knowledge or a predefined number of skills.

5. **Starting Point**: Under standard assumptions, there is a bijection between a policy and its occupancy measure; thus the occupancy measure is a complete characterization of policy behavior.

6. **Core Idea**: Embed occupancy measures using random Fourier features so that embedding distances approximate the MMD distance, then apply PCA to obtain low-dimensional behavior descriptors.

## Method

### Overall Architecture
Given an MDP environment, the system outputs an archive of diverse, high-quality policies. The pipeline proceeds as: collect policy trajectories → embed policies via random Fourier features → reduce to low-dimensional behavior descriptors via weighted PCA → perform QD optimization with CMA-MAE → periodically update descriptors.

### Key Designs

1. **Policy Embedding via RFF**:
   - **Function**: Map policies into Euclidean space such that distances reflect behavioral differences.
   - **Mechanism**: Define a $D$-dimensional random feature map $\phi(s,a) = \sqrt{2/D}[\cos(\mathbf{w}_1^T[s;a]+b_1),\ldots,\cos(\mathbf{w}_D^T[s;a]+b_D)]$, with policy embedding $\psi^\pi = \frac{1}{n}\sum_j(1-\gamma)\sum_t\gamma^t\phi(s_t^j,a_t^j)$. A theorem proves that $\|\psi^{\pi_1}-\psi^{\pi_2}\| \approx MMD(\rho^{\pi_1},\rho^{\pi_2})$ holds with high probability.
   - **Design Motivation**: MMD with a Gaussian kernel is a valid metric on the space of occupancy measures; RFF provides a computationally efficient finite-dimensional approximation.

2. **Behavior Descriptor Extraction (cwPCA)**:
   - **Function**: Project high-dimensional embeddings into $k$-dimensional behavior descriptors.
   - **Mechanism**: Apply fitness-weighted PCA to policy embeddings in the archive (weighted by fitness), so that higher-quality policies exert greater influence on principal component directions. An affine calibration step maps projections into $[-1,1]$.
   - **Design Motivation**: Biases behavioral diversity exploration toward high-quality policies; PCA captures the most salient dimensions of behavioral variation.

3. **Iterative Algorithm (AutoQD)**:
   - **Function**: Alternate between QD optimization and descriptor updates.
   - **Mechanism**: At scheduled update steps, recompute embeddings for all policies in the archive, update the affine transformation parameters $\mathbf{A}, \mathbf{b}$, and resume CMA-MAE optimization.
   - **Design Motivation**: As exploration progresses, the dominant directions of behavioral variation may shift, necessitating dynamic descriptor updates.

### Loss & Training
- Black-box optimization: CMA-MAE (gradient-free).
- Policy parameterization: Toeplitz matrices to reduce parameter count.
- Kernel bandwidth $\sigma$ selected via the median heuristic.
- Embedding dimension $D$ set proportionally to the state-action dimensionality.

## Key Experimental Results

### Main Results

| Environment | Metric | AutoQD | RegularQD (manual) | Best Baseline |
|---|---|---|---|---|
| Ant | GT QD (×10⁴) | **361.43** | 182.58 | 19.24 |
| HalfCheetah | GT QD (×10⁴) | **30.78** | 24.91 | 11.38 |
| Hopper | qVS | **1.94** | 1.35 | 1.81 |
| Swimmer | VS | **16.92** | 4.67 | 7.21 |
| BipedalWalker | GT QD (×10⁴) | **6.09** | 1.81 | 3.36 |

### Ablation Study

| Configuration | Key Metric | Notes |
|---|---|---|
| PCA without fitness weighting | Performance degrades | Low-quality policies corrupt principal component directions |
| Varying $k$ | $k=2$ generally optimal | Higher-dimensional archives are harder to fill |

### Key Findings
- AutoQD outperforms manual descriptor methods on 5 of 6 environments.
- Slight underperformance on HalfCheetah and Walker2d, as the method discovers low-reward but diverse behaviors such as "sliding."
- Adaptation experiments: under changes in friction/mass, AutoQD's policy archive maintains higher robustness.
- The GT QD score of 361.43 (×10⁴) on Ant far exceeds the manual descriptor baseline of 182.58, demonstrating the substantial advantage of automatic descriptors.
- PCA without fitness weighting allows low-quality policies to distort principal component directions, confirming the necessity of the cwPCA design.

## Highlights & Insights
- Theoretical rigor: the mathematical derivation chain from occupancy measures to MMD to RFF is complete and coherent.
- Automatically discovered behavior descriptors may reveal interesting behavioral variants overlooked by manual descriptors.
- Integration with CMA-MAE makes the method scalable to continuous, high-dimensional behavior spaces.
- The policy embedding technique is reusable in other settings such as imitation learning and inverse reinforcement learning.
- The theoretical foundation of occupancy measures as a complete characterization of policy behavior provides stronger guarantees than autoencoder-based methods such as AURORA.

## Limitations & Future Work
- Accurate policy embedding estimation in highly stochastic environments requires a large number of trajectories.
- Low-dimensional behavior descriptors may concentrate exploration on simple, stable behaviors.
- The kernel bandwidth is fixed; dynamic adjustment could better capture behavioral differences across learning stages.
- No integration with gradient-based QD methods (PGA-MAP-Elites, PPGA).
- The theoretical lower bound on trajectory count for embedding quality may be conservative in practice; fewer trajectories may suffice.
- Scalability to high-dimensional state-action spaces (e.g., humanoid control) remains to be validated.
- The choice of descriptor update frequency significantly affects final diversity, yet no automatic scheduling strategy is provided.

## Related Work & Insights
- **vs. AURORA**: AURORA learns state representations via autoencoders as descriptors, lacking a theoretical connection to policy behavior.
- **vs. DIAYN**: DIAYN maximizes skill-state mutual information, requires a preset number of skills, and does not optimize task reward.
- **vs. DvD-ES**: DvD-ES characterizes policies by action differences at random states, without the theoretical guarantees of AutoQD.
- The occupancy measure theoretical foundation provides a more principled basis for QD optimization than heuristic approaches.
- The method has direct value in downstream applications such as robotic motion generation and game level design.
- The relationship between AutoQD and intrinsic motivation exploration methods in RL warrants further investigation.
- Strategies for dynamically updating behavior descriptors in non-stationary environments are worth exploring.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — Using occupancy measure embeddings as behavior descriptors is a theoretically elegant innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 6 environments, 5 baselines, 3 metrics.
- **Writing Quality**: ⭐⭐⭐⭐ — Theory and experiments are tightly integrated.
- **Value**: ⭐⭐⭐⭐ — Significant contribution to QD optimization and open-ended learning.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Post-training Large Language Models for Diverse High-Quality Responses](post-training_large_language_models_for_diverse_high-quality_responses.md)
- [\[ICLR 2026\] AutoTool: Automatic Scaling of Tool-Use Capabilities in RL via Decoupled Entropy Constraints](autotool_scaling_tool_use.md)
- [\[ICLR 2026\] SUSD: Structured Unsupervised Skill Discovery through State Factorization](susd_structured_unsupervised_skill_discovery_through_state_factorization.md)
- [\[ICLR 2026\] Whatever Remains Must Be True: Filtering Drives Reasoning in LLMs, Shaping Diversity](whatever_remains_must_be_true_filtering_drives_reasoning_in_llms_shaping_diversi.md)
- [\[ICLR 2026\] Rethinking Policy Diversity in Ensemble Policy Gradient in Large-Scale Reinforcement Learning](rethinking_policy_diversity_in_ensemble_policy_gradient_in_large-scale_reinforce.md)

<!-- RELATED:END -->
