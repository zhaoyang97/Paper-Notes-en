---
title: >-
  [Paper Note] Neural Force Field: Few-shot Learning of Generalized Physical Reasoning
description: >-
  [ICLR 2026][neural force field] This paper proposes Neural Force Field (NFF), which models object interactions as continuous force fields. A neural operator learns the force field function, and an ODE integrator decodes trajectories from it. NFF achieves few-shot state-of-the-art on three benchmarks—I-PHYRE (100 trajectories), N-body (200 trajectories), and PHYRE (0.012M samples, 267× fewer than prior SOTA)—reducing cross-scenario RMSE by 32–64% and achieving near-human performance on planning tasks.
tags:
  - ICLR 2026
  - neural force field
  - Neural ODE
  - few-shot physical reasoning
  - ODE solver
  - interactive planning
date: 2026-05-08
content_hash: 4514fd932e370047
---

# Neural Force Field: Few-shot Learning of Generalized Physical Reasoning

**Conference**: ICLR 2026
**arXiv**: [2502.08987](https://arxiv.org/abs/2502.08987)
**Code**: [Project Page](https://neuralforcefield.github.io/)
**Area**: Other
**Keywords**: neural force field, Neural ODE, few-shot physical reasoning, ODE solver, interactive planning

## TL;DR

This paper proposes Neural Force Field (NFF), which models object interactions as continuous force fields. A neural operator learns the force field function, and an ODE integrator decodes trajectories from it. NFF achieves few-shot state-of-the-art on three benchmarks—I-PHYRE (100 trajectories), N-body (200 trajectories), and PHYRE (0.012M samples, 267× fewer than prior SOTA)—reducing cross-scenario RMSE by 32–64% and achieving near-human performance on planning tasks.

## Background & Motivation

**Background**: Physical reasoning is a core capability of AI. Humans can rapidly abstract physical principles from a small number of observations and generalize to new environments, yet existing AI models struggle in out-of-distribution (OOD) scenarios even when trained on massive datasets.

**Limitations of Prior Work**:
- Existing GNN/Transformer methods (IN, SlotFormer) represent object interactions via implicit latent vectors, which tends to overfit observed trajectories rather than capturing physical principles, leading to poor OOD generalization.
- Discrete latent space decoding cannot explain how objects traverse obstacles (e.g., a green ball passing through a black wall), resulting in physical inconsistencies.
- The risk of overfitting is amplified in few-shot settings, demanding strong physics-based inductive biases.
- Interactive reasoning requires active experimentation and feedback adaptation, yet existing methods lack backward planning capabilities.

**Key Challenge**: There is a need for a physical representation that can both learn from extremely few samples and generalize in OOD scenarios—one that encodes physical principles rather than statistical patterns.

**Goal**: Develop an agent with human-like few-shot physical learning ability that achieves robust generalization across diverse environments.

**Key Insight**: Force fields are a natural abstraction in physics—force is the causal driver of motion change. Representing interactions as force fields rather than state transitions is inherently compositional and generalizable.

**Core Idea**: A neural operator learns a continuous force field function; ODE integration enforces physical consistency; and the low dimensionality of force fields makes few-shot learning tractable.

## Method

### Overall Architecture

The NFF framework operates in three stages: (1) construct a dynamic interaction graph with objects as nodes and contact/attraction relations as edges; (2) a neural operator predicts the continuous force field $\mathbf{F}(\mathbf{z}^q(t))$; (3) an ODE integrator (Runge-Kutta/Euler) integrates the force field into velocity and displacement trajectories. During training, long trajectories are segmented for autoregressive prediction, minimizing MSE loss.

### Key Designs

1. **Neural Operator Force Field**:
   - **Function**: Predicts a continuous force field from object states and the interaction graph.
   - **Mechanism**: Based on the DeepONet framework, the force field function is defined as $\mathbf{F}(\mathbf{z}^q(t)) = \sum_{i \in \mathcal{G}(q)} \mathbf{W}(f_\theta(\mathbf{z}^i(t)) \odot f_\phi(\mathbf{z}^q(t))) + \mathbf{b}$, where $\mathcal{G}(q)$ is the neighbor set of the query object, $f_\theta$ and $f_\phi$ are neural networks, $\odot$ denotes element-wise multiplication, and $\mathbf{W} \in \mathbb{R}^{d_\text{hidden} \times d_\text{force}}$ maps hidden features to the low-dimensional force space.
   - **Design Motivation**: Force fields are low-dimensional (2D/3D force vectors), making them easier to learn from few samples than high-dimensional latent vectors. The function-space learning capacity of neural operators enables force field patterns to generalize to novel interaction graphs.

2. **ODE Integration for Trajectory Decoding**:
   - **Function**: Converts the learned force field into physically consistent trajectories.
   - **Mechanism**: Object motion is governed by a second-order ODE: $\mathbf{a}^q(t) = \frac{d^2 x^q(t)}{dt^2} = \frac{\mathbf{F}(\mathbf{z}^q(t))}{m^q}$, integrated to yield $\mathbf{x}(t) = \mathbf{x}(0) + \int_0^t \mathbf{v}(t)dt$ and $\mathbf{v}(t) = \mathbf{v}(0) + \int_0^t \frac{\mathbf{F}(\mathbf{z}^q(t))}{m^q}dt$.
   - **Design Motivation**: ODE integration guarantees trajectory continuity and physical consistency, eliminating the object-tunneling artifacts of discrete decoding. High-precision integration (step size $1e\text{-}3$) improves fine-grained collision modeling.

3. **Forward-Backward Interactive Planning**:
   - **Function**: Uses the learned force field for goal-directed planning.
   - **Mechanism**: Forward planning samples 500 action candidates, evaluates them using NFF as a mental simulator, and executes the optimal sequence. Backward planning reverses the ODE time direction to infer initial conditions from a goal state: $\mathbf{x}(0) = \mathbf{x}(t) + \int_t^0 \mathbf{v}(t)dt$.
   - **Design Motivation**: The reversibility of ODEs makes backward computation naturally efficient. A 5-round interactive learning protocol (execute → observe deviation → update model → replan) mimics human trial-and-error learning.

### Loss & Training

Training minimizes MSE loss between predicted and ground-truth trajectories. A key strategy is segmenting long trajectories into short units during training to mitigate accumulated error under teacher forcing. At evaluation time, only the initial state is provided and the model predicts all future dynamics.

## Key Experimental Results

### Main Results: Trajectory Prediction (Primary Metric: RMSE↓)

| Benchmark | Setting | IN | SlotFormer | SEGNO | **NFF** | Gain |
|---|---|---|---|---|---|---|
| I-PHYRE | Within | 0.124 | 0.067 | 0.203 | **0.048** | 28%↓ vs SlotFormer |
| I-PHYRE | Cross | 0.194 | 0.206 | 0.314 | **0.131** | 32%↓ vs IN |
| N-body | Within [0,T] | 0.200 | 0.214 | 0.079 | **0.097** | — |
| N-body | Cross [0,3T] | 6.942 | 2.533 | 2.759 | **1.226** | 52%↓ vs SlotFormer |
| PHYRE | Cross AUCCESS↑ | — | 21.04 | — | **49.22** | +134% vs SlotFormer |

### Ablation Study (N-body Cross RMSE↓)

| Configuration | Cross RMSE | Notes |
|---|---|---|
| NFF (precision 1e-3) | 1.226 | Full model |
| NFF (precision 5e-3) | 1.251 | Reduced precision → performance drop |
| NFF (adaptive) | 1.788 | Adaptive integration underperforms fixed high precision |
| w/o ODE (degrades to IN) | 3.518 | ODE grounding is critical |
| w/o NOL (MLP replacing DeepONet) | 1.347 | Neural operator improves generalization |

### Key Findings

- **Remarkable data efficiency**: I-PHYRE uses only 100 trajectories (10 games × 10 samples), N-body only 200, and PHYRE only 12K (267× fewer than RPIN's 3.2M).
- **Force field visualization validates learning**: The learned gravitational field closely matches the ground-truth field (Figure 5b); collision, sliding, and friction force fields are also correctly captured (Figure 5a).
- **ODE grounding is key to generalization**: Removing the ODE causes Cross RMSE to surge from 1.226 to 3.518 (2.87×).
- **Planning approaches human performance**: In I-PHYRE interactive planning, NFF's cumulative success probability after 5 refinement rounds approaches human level, whereas IN and SlotFormer perform even below random sampling.
- **Object consistency**: In PHYRE visual tasks, RPIN incorrectly deforms the gray cup into a gray ball, SlotFormer exhibits object disappearance, while NFF maintains object consistency throughout.

## Highlights & Insights

- **"Force fields = the right level of abstraction for physics"**: Rather than learning *how* states transition, NFF learns *why* they do—force is the causal driver of motion change, and causal representations generalize naturally.
- **The essential difference between continuous and discrete**: Discrete decoding cannot explain object tunneling (Figure 2); continuous ODE integration naturally avoids physical inconsistencies.
- **Few-shot learning mirrors human physical intuition**: Humans also extract physical laws from limited experience (e.g., infant intuitive physics). NFF's low-dimensional force field representation mimics this cognitive process.
- **Elegance of backward planning**: Reversing the ODE time direction directly recovers initial conditions from a goal state, offering orders-of-magnitude greater efficiency than gradient-based iterative optimization (Table A3).

## Limitations & Future Work

- Evaluation is limited to synthetic/abstract reasoning datasets; real-world physical scenarios have not been validated.
- The framework assumes deterministic rigid-body environments and does not address stochastic environments, soft bodies, or fluids.
- Training a single model under varying friction and elasticity may introduce additional challenges.
- The visual input version relies on object masks and does not achieve end-to-end learning from pixels to force fields.

## Related Work & Insights

- **vs. IN (Battaglia et al., 2016)**: IN uses latent vectors with discrete transitions; its Cross RMSE is 2.87× higher than NFF, and its planning performance falls below random sampling.
- **vs. SlotFormer (Wu et al., 2023)**: SlotFormer uses Transformer with slot attention; its AUCCESS on PHYRE Cross is only 21.04 (NFF: 49.22), and it suffers from object disappearance.
- **vs. SEGNO (Liu et al., 2024b)**: SEGNO also uses ODEs but lacks force field representation; its within-distribution performance occasionally surpasses NFF but cross-distribution generalization is significantly worse (2.759 vs. 1.226).
- **vs. Kofinas et al. (2023)**: This work also employs a "field" concept but learns a latent field rather than an explicit force field; NFF is more physically grounded.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Introducing the physics concept of force fields into a learning system, with ODE integration guaranteeing physical consistency, represents a paradigm innovation in physical reasoning representation learning.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three benchmarks (I-PHYRE/N-body/PHYRE) with multiple settings (prediction/planning), detailed ablations, and force field visualizations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Physical intuition and methodological design are seamlessly integrated; figures are clear (especially Figure 2's continuous vs. discrete comparison); motivation is compelling.
- **Value**: ⭐⭐⭐⭐ Makes foundational contributions to physical reasoning, cognitive AI, and few-shot learning; the force field representation may inspire broader research on physical world models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Is Meta-Learning Out? Rethinking Unsupervised Few-Shot Classification with Limited Entropy](../../ICCV2025/others/is_meta-learning_out_rethinking_unsupervised_few-shot_classification_with_limite.md)
- [\[CVPR 2026\] DirPA: Addressing Prior Shift in Imbalanced Few-shot Crop-type Classification](../../CVPR2026/others/dirpa_addressing_prior_shift_in_imbalanced_few-shot_crop-type_classification.md)
- [\[ICLR 2026\] Learning on a Razor's Edge: Identifiability and Singularity of Polynomial Neural Networks](learning_on_a_razors_edge_identifiability_and_singularity_of_polynomial_neural_n.md)
- [\[ICCV 2025\] Doodle Your Keypoints: Sketch-Based Few-Shot Keypoint Detection](../../ICCV2025/others/doodle_your_keypoints_sketch-based_few-shot_keypoint_detection.md)
- [\[ICLR 2026\] A Federated Generalized Expectation-Maximization Algorithm for Mixture Models with an Unknown Number of Components](a_federated_generalized_expectation-maximization_algorithm_for_mixture_models_wi.md)

</div>

<!-- RELATED:END -->
