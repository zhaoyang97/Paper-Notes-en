---
title: >-
  [Paper Note] Periodic Skill Discovery
description: >-
  [NeurIPS 2025][Robotics][Unsupervised skill discovery] This paper proposes Periodic Skill Discovery (PSD), a framework that maps states onto a circular latent space to naturally encode periodicity…
tags:
  - "NeurIPS 2025"
  - "Robotics"
  - "Unsupervised skill discovery"
  - "periodic behavior"
  - "circular latent space"
  - "locomotion control"
date: 2026-05-08
content_hash: a06934cc9f7894cc
---

# Periodic Skill Discovery

**Conference**: NeurIPS 2025
**arXiv**: [2511.03187](https://arxiv.org/abs/2511.03187)  
**Code**: Available (jonghaepark.github.io/psd)  
**Area**: Reinforcement Learning / Skill Discovery
**Keywords**: Unsupervised skill discovery, periodic behavior, circular latent space, locomotion control, robotics

## TL;DR

This paper proposes Periodic Skill Discovery (PSD), a framework that maps states onto a circular latent space to naturally encode periodicity, enabling unsupervised discovery of diverse locomotion skills with varying periods.

## Background & Motivation

### Root Cause

**Background**: Unsupervised skill discovery is an important direction in reinforcement learning, aiming to learn diverse behaviors without relying on extrinsic rewards. However, existing methods overlook a fundamental issue:

**Ignoring the periodic nature of skills**: Most methods focus on maximizing mutual information between states and skills, or maximizing traversal distance in the latent space.

**Periodicity requirements in locomotion tasks**: Many robotic tasks, particularly locomotion, inherently require periodic behaviors at different temporal scales (e.g., walking, running, jumping).

**Limitations of Prior Work**: Mutual-information-based methods such as DIAYN struggle to naturally discover skills with varying periods.

The core motivation of PSD is to exploit the topological structure of a circular latent space to naturally encode periodicity, thereby discovering locomotion skills with diverse periods.

### Paper Goals

### Overall Architecture

The PSD framework comprises three core components:
1. **Circular latent space encoder**: Maps states onto the unit circle.
2. **Temporal-distance-aware training**: Trains the encoder to capture temporal distance information.
3. **Periodic skill policy**: Generates behaviors based on periodic representations in the latent space.

### Key Designs

**Circular latent space**:
- Maps state $s$ to an angle $\phi(s) \in [0, 2\pi)$ on the unit circle $\mathcal{S}^1$.

## Method

### Overall Architecture

The PSD framework comprises three core components:
1. **Circular latent space encoder**: Maps states onto the unit circle.
2. **Temporal-distance-aware training**: Trains the encoder to capture temporal distance information.
3. **Periodic skill policy**: Generates behaviors based on periodic representations in the latent space.

### Key Designs

**Circular latent space**:
- Maps state $s$ to an angle $\phi(s) \in [0, 2\pi)$ on the unit circle $\mathcal{S}^1$.
- Circular topology naturally supports periodicity: traversing from $0$ to $2\pi$ constitutes one complete cycle.
- Different angular velocities correspond to different skill periods.

**Temporal distance encoding**:
- The encoder is trained such that the arc-length distance between two points on the circle approximates the temporal distance between the corresponding states.
- This ensures that temporally adjacent states are also adjacent on the circle.
- One complete locomotion cycle corresponds to one full rotation around the circle.

**Skill parameterization**:
- Each skill is parameterized by its angular frequency $\omega$ in the circular latent space.
- Small $\omega$ → slow-period motion (e.g., slow walking).
- Large $\omega$ → fast-period motion (e.g., rapid running).
- Different values of $\omega$ naturally yield skills with distinct periods.

**Pixel observation support**:
- The encoder can directly process pixel-level observations.
- No hand-crafted state features are required.

### Loss & Training

- **Temporal distance contrastive loss**: Ensures the encoder correctly captures temporal distances.
- **Mutual information term**: Promotes skill diversity.
- **Policy gradient method**: Updates the skill policy.
- **Staged training**: The encoder is trained first, followed by the skill policy.

## Key Experimental Results

### Main Results

Skill discovery results across multiple MuJoCo robotic environments:

| Environment | Method | Skill Diversity | Periodicity Coverage | Downstream Task Performance |
|-------------|--------|----------------|---------------------|-----------------------------|
| Ant | DIAYN | Moderate | Poor | Baseline |
| Ant | CIC | Good | Poor | Slightly better |
| Ant | **PSD** | **Best** | **Best** | **Best** |
| HalfCheetah | DIAYN | Moderate | Poor | Baseline |
| HalfCheetah | CIC | Good | Poor | Slightly better |
| HalfCheetah | **PSD** | **Best** | **Best** | **Best** |

Downstream task (hurdle crossing) performance:

| Method | Ant Hurdle Success Rate | HalfCheetah Hurdle Success Rate |
|--------|------------------------|--------------------------------|
| DIAYN | Low | Low |
| CIC | Moderate | Moderate |
| PSD | **High** | **High** |
| PSD + DIAYN (combined) | **Highest** | **Highest** |

### Ablation Study

| Ablation | Variant | Effect |
|----------|---------|--------|
| Latent space topology | Euclidean vs. circular | Circular significantly outperforms Euclidean |
| Temporal distance | With vs. without | Temporal distance encoding is critical |
| Pixel vs. state | Observation type | Remains effective under pixel observations |
| Period range | Narrow vs. wide | Wider range yields more diverse skills |

### Key Findings

1. Circular latent space is naturally suited to the periodic requirements of locomotion tasks.
2. Skills discovered by PSD exhibit clearly diverse periods, ranging from slow crawling to fast running.
3. Periodic skills significantly outperform non-periodic baselines on downstream tasks such as hurdle crossing.
4. PSD is complementary to existing skill discovery methods (e.g., DIAYN), and their combination yields further improvement.

## Highlights & Insights

1. **Clear geometric intuition**: Circle = periodicity — a concise yet effective design principle.
2. **Filling a gap**: The first systematic exploitation of periodic structure in skill discovery.
3. **Complementary combination**: PSD extends rather than replaces existing methods, expanding the skill repertoire.
4. **Pixel-level applicability**: Requires no hand-crafted features, making it applicable to more general settings.

## Limitations & Future Work

1. The circular latent space is primarily suited to single-frequency periodic motions; modeling multi-frequency composite motions remains unexplored.
2. Validation is mainly conducted on locomotion tasks; applicability to other periodic tasks (e.g., manipulation, aerial locomotion) is not examined.
3. The staged training procedure could potentially be replaced by more efficient end-to-end training.
4. Skill discovery over a continuous spectrum of periods, rather than a discrete set, is not explored.
5. Sim-to-real transfer for deployment on physical robots is not addressed.

## Related Work & Insights

- **DIAYN**: Unsupervised skill discovery via mutual information maximization.
- **CIC**: Contrastive intrinsic control.
- **DADS**: Dynamics-aware skill discovery.
- **Spectral RL**: Leveraging spectral decomposition to understand state space structure.
- **Insight**: Incorporating topological or geometric structure into latent space design is a promising research direction.

## Rating

- Novelty: ⭐⭐⭐⭐ (Encoding periodicity via a circular latent space is an innovative idea)
- Technical Depth: ⭐⭐⭐⭐ (Theoretical motivation is clear and well-grounded)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Multiple environments + downstream tasks + ablations)
- Value: ⭐⭐⭐⭐ (Direct applicability to locomotion control)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Policy Compatible Skill Incremental Learning via Lazy Learning Interface](policy_compatible_skill_incremental_learning_via_lazy_learning_interface.md)
- [\[ICCV 2025\] iManip: Skill-Incremental Learning for Robotic Manipulation](../../ICCV2025/robotics/imanip_skill-incremental_learning_for_robotic_manipulation.md)
- [\[AAAI 2026\] Human-Centric Open-Future Task Discovery: Formulation, Benchmark, and Scalable Tree-Based Search](../../AAAI2026/robotics/human-centric_open-future_task_discovery_formulation_benchmark_and_scalable_tree.md)
- [\[CVPR 2026\] AtomicVLA: Unlocking the Potential of Atomic Skill Learning in Robots](../../CVPR2026/robotics/atomicvla_unlocking_the_potential_of_atomic_skill_learning_in_robots.md)
- [\[ACL 2026\] Breaking Down and Building Up: Mixture of Skill-Based Vision-and-Language Navigation Agents](../../ACL2026/robotics/breaking_down_and_building_up_mixture_of_skill-based_vision-and-language_navigat.md)

</div>

<!-- RELATED:END -->
