---
title: >-
  [Paper Note] Flow-Equivariant World Models: Memory for Partially Observed Dynamic Environments
description: >-
  [ICML 2026][Reinforcement Learning][World Models] FloWM utilizes **time-parameterized symmetries** (flow equivariance) in latent space to maintain structured dynamic memory—addressing the problem of objects disappearing…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "World Models"
  - "Partial Observability"
  - "Flow Equivariance"
  - "Structured Memory"
  - "Dynamic Environment Prediction"
date: 2026-05-08
content_hash: 824b32bf64de18f7
---

# Flow-Equivariant World Models: Memory for Partially Observed Dynamic Environments

**Conference**: ICML 2026  
**arXiv**: [2601.01075](https://arxiv.org/abs/2601.01075)  
**Code**: TBD  
**Area**: Reinforcement Learning / World Models / Representation Learning  
**Keywords**: World Models, Partial Observability, Flow Equivariance, Structured Memory, Dynamic Environment Prediction

## TL;DR
FloWM utilizes **time-parameterized symmetries** (flow equivariance) in latent space to maintain structured dynamic memory—addressing the problem of objects disappearing after going out-of-bounds in partially observed environments. This results in long-horizon prediction accuracy far exceeding diffusion and recurrent baselines (SSIM 0.9525 vs. DFoT 0.8885 in 3D Block World at 210 steps).

## Background & Motivation

**Background**: World models are core to embodied intelligence, requiring simultaneous prediction of self-motion and external object dynamics. Existing methods primarily employ large-scale Latent Diffusion Transformers (CogVideoX-style), which achieve realistic visual quality but face critical weaknesses in partially observed scenarios (limited agent field-of-view).

**Limitations of Prior Work**:
- **Sliding Window Information Loss**: Self-attention windows must discard historical information beyond their range. When the agent turns back to the original view, out-of-bounds objects vanish from the context, and the model cannot track them.
- **View-Dependent Memory Cannot Handle Dynamics**: Existing memory augmentation methods (e.g., WORLDMEM) store observations from specific viewpoints, failing to maintain consistency when external objects move.
- **Long-Horizon Prediction Failure**: Diffusion-based schemes begin to hallucinate after a certain prediction depth (generating objects out of thin air or forgetting existing ones).

**Key Challenge**: The world possesses temporal structure (self-motion + external object motion), but existing models ignore this structure, instead using general attention mechanisms for brute-force encoding—leading to an inability to distinguish between "objects being temporarily invisible" and "objects not existing."

**Goal**: (1) Establish a theoretical framework to formalize structured dynamic memory using flow equivariance; (2) Design a scalable implementation supporting 2D/3D partially observed environments; (3) Validate advantages in long-horizon prediction and downstream planning tasks.

**Key Insight**: Drawing from equivariance in group theory—if a data generation process respects group symmetries, incorporating these symmetries into the model significantly improves generalization (as seen in molecular dynamics with 1000x improvements). This paper innovates by extending **static group equivariance to time-parameterized flows**, naturally yielding structured memory.

**Core Idea**: Maintain a "velocity channel" stack in latent space—each channel corresponding to a different motion flow (e.g., velocity vectors). During each update, the entire latent state is first transformed based on the agent's action (achieving self-motion equivariance), then fused with new observations (achieving external motion equivariance), allowing the memory to align automatically with the world structure.

## Method

### Overall Architecture
A three-tier progressive design:
1. **General Flow-Equivariant Framework**: Extends from group equivariance to time-parameterized flows, deriving global flow-equivariant constraints.
2. **2D/3D Instantiation**: Constructs specific network architectures for different tasks.
3. **Partial Observability Patches**: Maintains objects beyond the field of view in latent space, automatically translating them via flow transformations.

**Input**: Observation sequence $\{f_t\}$ (images) + Action sequence $\{a_t\}$ (displacement/rotation)  
**Latent State**: Structured memory $h_t: G \to \mathbb{R}^K$, where $G$ is the world coordinate space  
**Output**: Predicted observation $\hat{f}_{t+1}$ (decoded from a subset of the latent space view)

### Key Designs

1. **Flow-Equivariant Recurrence**:
    - **Function**: The core mechanism for maintaining latent state transformations according to world dynamics and self-motion.
    - **Mechanism**: Employs the recurrence formula $h_{t+1}(\nu) = T_{a_t}^{-1} \psi_1(\nu) \cdot U_\theta[h_t(\nu); E_\theta[f_t, h_t](\nu)]$, where $\psi_1(\nu)$ is the single-step flow transformation (translation/rotation) for velocity channel $\nu$, and $T_{a_t}$ is the transformation derived from the action. The latent state contains a stack of "velocity channels," each flowing automatically according to its velocity; after fusing new observations, the overall memory is inversely transformed based on agent actions to maintain alignment in the agent's reference frame.
    - **Design Motivation**: By explicitly modeling temporal symmetry, objects are remembered even when outside the field of view, and their trajectories are updated consistently with visible objects. This drastically reduces learning complexity compared to unstructured memory (self-attention), speeding up convergence by over 100x.

2. **Velocity Channels and Trivial Lifting**:
    - **Function**: Separates different motion modes (self vs. external dynamics) to avoid interference.
    - **Mechanism**: The encoder performs "trivial lifting" $E_\theta[f_t; h_t](\nu) = E_\theta[f_t; h_t](\nu')$ for all velocity channels. During updates, the external object flow $\psi_1(\nu)$ and self-action flow $\psi_1(-a_t)$ combine algebraically into $\psi_1(\nu - a_t)$, automatically sorting the velocity channels. This avoids explicit velocity estimation for external objects, letting the model learn implicit velocity representations.
    - **Design Motivation**: Parameter sharing significantly reduces latent dimensionality; compared to explicit velocity prediction, this design is more robust and scales to unseen dynamics.

3. **Learned Equivariance of the ViT Encoder**:
    - **Function**: Maps first-person views to abstract top-down views while maintaining equivariance.
    - **Mechanism**: In 2D/3D extensions, explicit equivariance constraints are not forced on the encoder (due to high costs of 3D back-projection). Instead, it relies on the recurrence $T_{a_t}^{-1} \psi_1(\nu)$ to "encourage" the encoder to learn equivariance at each step. Probe network experiments verify that after training, the equivariant error drops from 6.96 to 0.22 (96% accuracy in recovering object positions), far outperforming the baseline's 2.36.
    - **Design Motivation**: Utilizing feed-forward approximations instead of exact equivariant constraints provides powerful encoder expressive capacity and scalability. Experiments prove this compromise does not damage performance.

## Key Experimental Results

### Main Results: 2D MNIST World (Partially Observed)

| Model | 20-step MSE | 150-step MSE | 20-step PSNR | 150-step PSNR | 150-step SSIM |
|------|--------|---------|----------|-----------|----------|
| **FloWM (Full)** | 0.0005 | **0.0018** | 32.99 | **27.56** | **0.9813** |
| FloWM (No VC) | 0.0041 | 0.0334 | 23.83 | 14.77 | 0.7729 |
| FloWM (No SME) | 0.1234 | 0.1317 | 9.088 | 8.805 | 0.0127 |
| DFoT Baseline | 0.1448 | 0.2111 | 8.394 | 6.755 | 0.2434 |
| DFoT-SSM | 0.1277 | 0.1688 | 8.940 | 7.726 | 0.3146 |

Full FloWM maintains a PSNR of 27.56 even at 150 steps (7.5x the training length), whereas baselines collapse by step 20; learning curves show FloWM converges 100x faster.

### 3D Block World (Rigid Body Dynamics + Partially Observed)

| Model | 70-step MSE | 210-step MSE | 70-step SSIM | 210-step SSIM | Planning Success |
|------|--------|----------|----------|-----------|----------|
| **FloWM (Full)** | 0.000603 | **0.001539** | 0.9673 | **0.9525** | **0.727** |
| FloWM (No VC) | 0.007615 | 0.009614 | 0.9045 | 0.8935 | — |
| FloWM (No SME) | 0.009579 | 0.012625 | 0.8782 | 0.8631 | — |
| DFoT | 0.011759 | 0.021684 | 0.9377 | 0.8885 | 5.571 |
| DreamerV3 RSSM | 0.016360 | 0.016470 | 0.8799 | 0.8782 | 6.449 |

For 210-step predictions (3x the training length), FloWM achieves an SSIM of 0.9525, while all baselines drop to 0.8-0.89. Baselines exhibit severe hallucinations—generating objects, forgetting existing ones, and blurry ghosting—while FloWM remains clear. In the downstream "Find the Red Block" planning task, FloWM achieves an average distance of 0.727, compared to 5-6 for baselines.

### Key Findings
- By decoding object positions in latent space using probe networks, FloWM recovers locations with 96% accuracy, while DFoT / DFoT-SSM fall below 1%.
- Equivariant error drops from 6.96 → 0.22 (↓96%) after training, validating the effectiveness of learned equivariance.
- Removing velocity channels (No VC) or self-motion equivariance (No SME) causes performance to plummet—both modules are essential.

## Highlights & Insights
- **Theoretical Elegance**: Uses Lie groups/algebras to unify temporal symmetries. The "flow equivariance" concept naturally leads to structured memory recurrence formulas, avoiding ad-hoc mechanisms. It is highly scalable—adding a new symmetry only requires extending the velocity channel set, not redesigning the network.
- **Fundamental Insight into Partial Observability**: Identifies that failure in existing methods stems from "confusing invisibility with non-existence," whereas flow equivariance forces the model to distinguish the two. It explains why the universality of self-attention is a weakness here.
- **Pragmatic Learned Equivariance**: Instead of forcing exact equivariant constraints (costly in 3D back-projection), it relies on recurrence induction. Proving the model automatically learns equivariant representations, this "weak constraint + strong induction" philosophy is transferable to other geometric tasks.
- **Transferable Techniques**: The trivial lifting of velocity channels is applicable to other multi-modal problems; using probe networks to verify if the latent space has learned the intended structure.

## Limitations & Future Work
- Experiments were conducted only in controlled environments (discrete actions, known parameterizations), not real-world autonomous driving or robotics datasets.
- Latent space mapping assumes a fixed size, which might exceed capacity during prolonged interaction.
- Currently, only rigid body motion is supported; deformation and soft bodies are not handled.
- Encoder design fragility: Although learned equivariance is proven effective, the process is implicit and lacks theoretical guarantees, potentially leading to failure cases in large-scale applications.
- Improvements: Extension to continuous velocity families; adaptive latent space sizing; combining explicit depth prediction with visual encoding to strengthen the reliability of equivariant constraints.

## Related Work & Insights
- **vs. DreamerV3**: Uses recurrent latent states for dynamics but lacks spatial structure. In partially observed environments, it confuses viewpoint movement with world changes; FloWM explicitly decouples these, leading to 10x better long-horizon performance.
- **vs. Diffusion Forcing (DFoT)**: Uses fixed-length sliding windows + diffusion without an explicit memory mechanism, failing completely after 150 steps (hallucinations); FloWM remains precise.
- **vs. WORLDMEM / Memory Bank schemes**: Stores historical observation frames, but retrieval still relies on self-attention fusion which cannot handle dynamics; FloWM uses flow transformations to actively update memory locations, naturally adapting to dynamics.
- **vs. Neural Mapping / EgoMap**: Early works had "structured mapping" ideas, but they were not parameterized as flows, lacked formal equivariance theory, and did not support prediction. FloWM can be seen as a modernized and generalized version of this idea.

## Rating
- Novelty: ⭐⭐⭐⭐⭐  Systematically introduces flow equivariance to world models, formalizing symmetry under partial observability with an elegant theoretical framework.
- Experimental Thoroughness: ⭐⭐⭐⭐  2D/3D benchmarks are comprehensive with deep ablation; however, benchmark environments are relatively simple (low noise/occlusion/stochasticity), making real-world generalization hard to assess.
- Writing Quality: ⭐⭐⭐⭐⭐  Clear logical flow, mathematically rigorous but accessible, with intuitive diagrams.
- Value: ⭐⭐⭐⭐⭐  Deep insights into world models and partial observability; the methodology is generalizable to other tasks requiring structured representations (SLAM, navigation).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Partially Equivariant Reinforcement Learning in Symmetry-Breaking Environments](../../ICLR2026/reinforcement_learning/partially_equivariant_reinforcement_learning_in_symmetry-breaking_environments.md)
- [\[ICML 2026\] Parameter-free Dynamic Regret: Time-varying Movement Costs, Delayed Feedback, and Memory](parameter-free_dynamic_regret_time-varying_movement_costs_delayed_feedback_and_m.md)
- [\[ICML 2026\] Break the Block: Dynamic-size Reasoning Blocks for Diffusion Large Language Models via Monotonic Entropy Descent with Reinforcement Learning](break_the_block_dynamic-size_reasoning_blocks_for_diffusion_large_language_model.md)
- [\[ICML 2026\] Perceptual Flow Network for Visually Grounded Reasoning](perceptual_flow_network_for_visually_grounded_reasoning.md)
- [\[ICML 2026\] Learning Query-Aware Budget-Tier Routing for Runtime Agent Memory](learning_query-aware_budget-tier_routing_for_runtime_agent_memory.md)

</div>

<!-- RELATED:END -->
