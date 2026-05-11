---
title: >-
  [Paper Note] TeamHOI: Learning a Unified Policy for Cooperative Human-Object Interactions with Any Team Size
description: >-
  [CVPR 2026][multi-agent cooperation] TeamHOI proposes a framework using a Transformer-based decentralized policy network and Masked Adversarial Motion Prior (Masked AMP)…
tags:
  - "CVPR 2026"
  - "multi-agent cooperation"
  - "physics-based human motion control"
  - "human-object interaction"
  - "Transformer policy network"
  - "adversarial motion prior"
date: 2026-05-08
content_hash: 7a493ad8832336c6
---

# TeamHOI: Learning a Unified Policy for Cooperative Human-Object Interactions with Any Team Size

**Conference**: CVPR 2026
**arXiv**: [2603.07988](https://arxiv.org/abs/2603.07988)
**Code**: [Project Page](https://splionar.github.io/TeamHOI)
**Area**: Other
**Keywords**: multi-agent cooperation, physics-based human motion control, human-object interaction, Transformer policy network, adversarial motion prior

## TL;DR

TeamHOI proposes a framework using a Transformer-based decentralized policy network and Masked Adversarial Motion Prior (Masked AMP), enabling a single policy to generalize to cooperative carrying tasks with any number of agents, achieving 97%+ success rate for 2–8 humanoid agents cooperatively carrying a table.

## Background & Motivation

**Background**: Physics-based humanoid control has made significant progress in single-agent behaviors (locomotion, grasping, manipulation), but many real-world tasks (e.g., carrying large heavy objects) require multi-agent coordination of physical actions—a scenario existing frameworks struggle to handle.

**Limitations of Prior Work**: Most existing methods employ fixed-size input MLP policy networks, constraining policies to a fixed team size (e.g., SMPLOlympics only supports fixed small teams) and preventing flexible adaptation to varying numbers of collaborators. Methods such as CooHOI do not model inter-agent perception at all, relying solely on shared object dynamics as an implicit communication channel, and therefore fail to capture the essential nature of human cooperation—continuously perceiving teammates and dynamically adjusting behavior. Moreover, multi-person collaborative motion capture data is nearly nonexistent, and directly using single-person reference motions severely limits the diversity of learnable cooperative behaviors and constrains cooperation patterns to only front-back lifting. CooHOI also requires pre-specified contact points for each agent (oracle assignment), preventing agents from autonomously inferring appropriate positions for stable carrying.

**Key Challenge**: The combination of fixed-capacity network architectures, absence of inter-agent perception, and scarcity of multi-person motion reference data fundamentally limits the scalability and behavioral diversity of existing cooperative HOI systems.

**Goal**: To develop a single unified policy that (1) generalizes across arbitrary team sizes, (2) enables agents to autonomously determine formation without oracle assignment, (3) produces diverse and physically realistic cooperative behaviors from limited single-person reference data.

## Method

### Overall Architecture

TeamHOI extends the AMP framework into a flexible multi-agent reinforcement learning setting. The core design comprises three components:

- **Transformer Policy Network**: Each agent's observation is decomposed into multiple tokens (proprioception, object state, goal position, teammate cues), processed through alternating self-attention and cross-attention layers. Self-attention operates over the observing agent's own tokens; cross-attention allows it to attend to a variable number of teammate tokens, enabling generalization to arbitrary team sizes.
- **Unified Policy Training**: Environments with different team sizes (2–8 agents) are instantiated in parallel, training a single policy across diverse configurations. PPO advantage values are normalized independently per team size to ensure training stability.
- **Masked AMP**: Decouples motion realism from object interaction, enabling limited single-person reference motions to produce diverse cooperative behaviors.

### Key Designs

**1. Transformer Policy Network and Teammate Tokens**

Observation components (all in the agent's local coordinate frame):

- Proprioception $\in \mathbb{R}^{223}$: joint states and root kinematics
- Object center $\in \mathbb{R}^{3}$: 3D coordinates of the table center
- Candidate contact points $\in \mathbb{R}^{64 \times 3}$: 64 points uniformly sampled along the table perimeter
- Nearest hand-object points $\in \mathbb{R}^{2 \times 3}$: candidate contact points nearest to each hand
- Goal position $\in \mathbb{R}^{3}$: target x,y coordinates and height/lift indicator of the table
- Teammate cues $\in \mathbb{R}^{(n-1) \times 9}$: per teammate root position (2D), orientation (6D rotation), and relative angle (1D)

Each observation component is encoded into a 64-dimensional token via an independent three-layer MLP tokenizer. The Transformer consists of 3 layers of alternating self-attention and cross-attention, each with 2 attention heads and 512-dimensional feed-forward layers. The updated learnable embedding $e$ is passed through an MLP `[1024, 512, 28]` to output target joint rotations.

**2. Masked Adversarial Motion Prior (Masked AMP)**

Two discriminator networks are trained:
- $D_{\text{full}}$: evaluates full-body reference motion
- $D_{\text{mask}}$: excludes body parts involved in object interaction (hands and forearms)

Mixed style reward:
$$r_t^{\text{style}} = \sigma(\alpha_t) \, r_t^{\text{mask}} + (1 - \sigma(\alpha_t)) \, r_t^{\text{full}}$$

where $\sigma$ is the sigmoid function and $\alpha_t$ is a continuous interaction indicator (e.g., agent-object distance). When not interacting with the object, the full discriminator ensures whole-body motion realism; during interaction, the masked discriminator frees the hands so that task reward guides behavior.

**3. Formation Reward**

Composed of two complementary components:

- **Angular spread reward** $r_{\text{ang}}$: encourages $m$ agents to distribute evenly around the table with ideal spacing $2\pi/m$:
$$r_{\text{ang}} = \exp\!\left(-k_\theta \frac{1}{2}\left[(\Delta\phi_i^{\text{ccw}} - \frac{2\pi}{m})^2 + (\Delta\phi_i^{\text{cw}} - \frac{2\pi}{m})^2\right]\right)$$

- **Principal axis coverage reward** $r_{\text{cov}}$: measures how well agents' support region covers the object's principal axes, computing per-axis coverage ratio $g_i = \min(d_i^+ / \ell_i^+, d_i^- / \ell_i^-)$ via convex hull projection, with $r_{\text{cov}} = \frac{1}{2}(g_1 + g_2)$.

Combined formation reward: $r_{\text{form}} = 0.25 \, r_{\text{ang}} + 0.75 \, r_{\text{cov}}$

### Loss & Training

Overall reward: $r_t = r_t^{\text{task}} + \lambda_{\text{AMP}} \, r_t^{\text{style}}$

- **Task reward** $r_t^{\text{task}}$: includes components for five phases—approaching, contacting, lifting, carrying, and placing—as well as the formation reward.
- **Style reward** $r_t^{\text{style}}$: output of the Masked AMP mixed discriminator.
- **Discriminator loss**: Standard GAN loss; $D_{\text{full}}$ and $D_{\text{mask}}$ are trained separately to distinguish reference vs. policy-generated state transitions.

The policy is optimized with PPO; advantage values are normalized independently per team size.

## Experiments

### Main Results

Evaluated on a cooperative table-carrying task with three table geometries—square (1.6m×1.6m), rectangular (2.0m×1.2m), and circular (diameter 2.0m)—with weights of 50–70 kg. Each evaluation runs 10,000 simulation episodes.

| Method | Formation | 2-agent SR (%) | 4-agent SR (%) | 8-agent SR (%) | 4-agent CoopRate | 8-agent CoopRate |
|--------|-----------|---------------|---------------|---------------|-----------------|-----------------|
| CooHOI*-2 | Predefined | 97.5 | 73.2 | 10.1 | 54.6% | 1.0% |
| CooHOI*-4 | Predefined | 95.5 | 94.5 | 61.5 | 92.1% | 27.2% |
| CooHOI*-8 | Predefined | 29.4 | 52.4 | 42.2 | 93.6% | 81.6% |
| **TeamHOI** | **Autonomous** | **99.1** | **99.2** | **97.5** | **96.1%** | **90.1%** |

**Heavy-load setting (5× table weight)**: In the 4-agent scenario TeamHOI achieves 3.5% SR (small teams can barely lift); in the 8-agent scenario TeamHOI achieves **81.1%** SR, while all CooHOI* baselines remain below 15%.

### Ablation Study

| Ablation | Effect |
|----------|--------|
| Remove Masked AMP | Success rate in the lifting phase drops significantly; hand-object interaction failures are frequent |
| Angular spread reward only (no principal axis coverage) | Agents do not distribute along the principal axes; unnatural diagonal gaits emerge |
| Full method | Agents align along principal axes, exhibit natural symmetric gaits, and carry stably |

### Key Findings

- **One policy for all configurations**: TeamHOI achieves high success rates across all 2–8 agent configurations with a single policy, whereas each CooHOI* variant performs well only at the team size it was trained on.
- **Zero-shot generalization**: The model generalizes zero-shot to configurations with 16 agents.
- **Autonomous vs. predefined formation**: TeamHOI agents must autonomously infer positions—a harder task—yet substantially outperform baselines that use pre-specified contact points.
- **Critical role of Masked AMP**: The masking strategy allows single-person lateral walking reference motions to be reused as lateral carrying motions, greatly expanding the diversity of viable cooperative behaviors.

## Highlights & Insights

1. **Scalable decentralized architecture**: Cross-attention over variable-length teammate tokens elegantly resolves the fixed-input-size limitation, enabling a single policy to support arbitrary team sizes.
2. **Elegant design of Masked AMP**: Only the body parts involved in object interaction are masked; full-body motion realism is preserved during non-interaction phases, while task reward guides the masked regions—unlocking diverse cooperative behaviors from limited single-person data.
3. **Generality of the formation reward**: The combination of angular spread and principal axis coverage is robust to team size and object geometry; the principal axis coverage reward further generalizes to irregular geometries and non-uniform mass distributions.
4. **No oracle assignment required**: Agents start from random initial positions and autonomously infer appropriate positions to form stable formations.

## Limitations & Future Work

1. **Single task**: Validation is limited to table carrying; the framework has not been extended to other cooperative HOI tasks (e.g., pushing, pulling, tossing and catching).
2. **Simplified hand model**: Finger-less spherical hands are used; fine-grained grasping is not addressed.
3. **Horizontal carrying only**: Table height is fixed slightly below the standing hand position, avoiding more complex interactions such as bending down or lifting overhead.
4. **Limited reference motions**: The method still relies on a small set of walking and pick-up motions from AMASS, which may be insufficient for more complex cooperative behaviors.
5. **Homogeneous agents only**: All agents share the same policy and body morphology; heterogeneous team cooperation is not explored.

## Related Work & Insights

- **AMP / ASE / PMP**: The adversarial motion prior family; Masked AMP draws inspiration from PMP's partial prior concept but adopts masking combined with task reward rather than learning separate partial priors directly.
- **TokenHSI**: Pioneer of Transformer policy networks with task tokenization; TeamHOI adopts and extends this architecture with multi-agent cross-attention.
- **CooHOI**: The most closely related cooperative HOI work; it relies solely on object dynamics for implicit communication, lacks inter-agent perception, and exhibits limited behavioral diversity—serving as the primary baseline.
- **PHC**: A foundational physics-based humanoid control framework adopted by several multi-character interaction works.

## Rating

- Novelty: ⭐⭐⭐⭐ — Masked AMP and principal axis coverage reward are novel; the Transformer teammate token idea is natural but effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 10k-episode evaluation, multiple geometries, 2–8 agent configurations, heavy-load testing, and comprehensive ablations; validation on more task types is lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear structure, complete derivations, intuitive figures, and well-motivated problem formulation.
- Value: ⭐⭐⭐⭐ — Provides a solid foundation for scalable multi-agent physical cooperation; broader task generalization remains to be demonstrated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SyncDiff: Synchronized Motion Diffusion for Multi-Body Human-Object Interaction Synthesis](../../ICCV2025/others/syncdiff_synchronized_motion_diffusion_for_multi-body_human-object_interaction_s.md)
- [\[ICLR 2026\] Agnostics: Learning to Synthesize Code in Any Programming Language with a Universal RL Environment](../../ICLR2026/others/agnostics_learning_to_code_in_any_programming_language_via_reinforcement_with_a_.md)
- [\[AAAI 2026\] HyperSHAP: Shapley Values and Interactions for Explaining Hyperparameter Optimization](../../AAAI2026/others/hypershap_shapley_values_and_interactions_for_explaining_hyperparameter_optimiza.md)
- [\[CVPR 2026\] LoViF 2026 Challenge on Human-oriented Semantic Image Quality Assessment](lovif_2026_semantic_quality_assessment_challenge.md)
- [\[ICLR 2026\] A Single Architecture for Representing Invariance Under Any Space Group](../../ICLR2026/others/a_single_architecture_for_representing_invariance_under_any_space_group.md)

</div>

<!-- RELATED:END -->
