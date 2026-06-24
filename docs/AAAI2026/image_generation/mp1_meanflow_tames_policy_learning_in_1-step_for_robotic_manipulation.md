---
title: >-
  [Paper Note] MP1: MeanFlow Tames Policy Learning in 1-step for Robotic Manipulation
description: >-
  [AAAI 2026][Image Generation][Robotic Manipulation] This work introduces the MeanFlow paradigm to the robot learning domain for the first time. By incorporating 3D point cloud inputs and a Dispersive Loss, MP1 generates action trajectories in a single network forward pass (1-NFE), achieving state-of-the-art success rates with an inference latency of only 6.8 ms on robotic manipulation tasks.
tags:
  - "AAAI 2026"
  - "Image Generation"
  - "Robotic Manipulation"
  - "MeanFlow"
  - "Single-Step Inference"
  - "Flow Matching"
  - "Dispersive Loss"
date: 2026-05-08
content_hash: aa6c7d1d5586553e
---

# MP1: MeanFlow Tames Policy Learning in 1-step for Robotic Manipulation

**Conference**: AAAI 2026
**arXiv**: [2507.10543](https://arxiv.org/abs/2507.10543)  
**Code**: [github.com/LogSSim/MP1](https://github.com/LogSSim/MP1)  
**Area**: Image Generation / Robotic Manipulation
**Keywords**: Robotic Manipulation, MeanFlow, Single-Step Inference, Flow Matching, Dispersive Loss

## TL;DR

This work introduces the MeanFlow paradigm to the robot learning domain for the first time. By incorporating 3D point cloud inputs and a Dispersive Loss, MP1 generates action trajectories in a single network forward pass (1-NFE), achieving state-of-the-art success rates with an inference latency of only 6.8 ms on robotic manipulation tasks.

## Background & Motivation

### State of the Field

Generative models have become the dominant approach for policy learning in robotic manipulation. Diffusion-based methods (e.g., Diffusion Policy, DP3) effectively handle multimodal action distributions but require multi-step (~10-step) denoising iterations, resulting in inference latencies of approximately 130 ms. Flow matching methods (e.g., FlowPolicy) achieve single-step sampling via consistency constraints, reducing inference time to ~12 ms, yet still rely on additional structural assumptions.

### Limitations of Prior Work

**Slow inference in diffusion models**: Multi-step denoising leads to inference latencies as high as ~130 ms, making real-time control (e.g., force-controlled manipulation, high-frequency feedback loops) infeasible.

**Flow matching methods constrained by consistency assumptions**: Methods such as FlowPolicy enforce consistency losses to enable 1-NFE inference, but this introduces additional structural assumptions that may limit the expressive capacity of the learned policy.

**ODE solver errors**: Conventional flow matching integrates an instantaneous velocity field at inference time; numerical ODE solvers introduce errors that degrade trajectory accuracy.

### Root Cause

How can genuinely single-step action generation be achieved **without any consistency constraints or ODE solvers**, while maintaining or surpassing the task success rates of multi-step methods?

### Core Idea

The MeanFlow paradigm directly learns the interval-averaged velocity field rather than the instantaneous velocity field. By exploiting the "MeanFlow Identity"—a differential identity—training is reduced to a simple regression objective. At inference time, a single forward pass maps noise directly to an action trajectory, entirely eliminating ODE solver errors and consistency constraints. A Dispersive Loss is further introduced as a training-time regularizer to enhance feature-space discriminability and improve few-shot generalization.

## Method

### Overall Architecture

MP1 takes 3D point clouds and robot states as input. A visual encoder and a state encoder extract conditional features $\mathbf{c} = (\mathbf{f}_v, \mathbf{f}_s)$, which are fed into a UNet integrated with MeanFlow to generate action trajectories. Training combines a CFG regression loss $\mathcal{L}_{cfg}$ and a Dispersive Loss $\mathcal{L}_{disp}$; inference requires only a single forward pass.

### Key Designs

#### 1. **MeanFlow Policy Generation (Core Contribution)**

**Function**: Learn the interval-averaged velocity field in place of the instantaneous velocity field, enabling single-step action generation without an ODE solver.

**Mechanism**: Standard flow matching learns the instantaneous velocity field $v(z_t, t)$ and requires ODE integration at inference. MeanFlow instead learns the **averaged velocity field** over the interval $[r, t]$:

$$u(z_t, r, t) \triangleq \frac{1}{t-r} \int_r^t v(z_\tau, \tau) d\tau$$

Direct learning from this integral definition is intractable, but differentiating with respect to $t$ yields the **"MeanFlow Identity"**:

$$u(z_t, r, t) = v(z_t, t) - (t-r) \frac{d}{dt} u(z_t, r, t)$$

Based on this identity, the training objective reduces to a simple regression:

$$\mathcal{L}(\theta) = \mathbb{E}_{t,r,x,\epsilon} \|u_\theta(z_t, r, t) - sg(u_{tgt})\|_2^2$$

where $u_{tgt} = v_t - (t-r)(v_t \partial_z u_\theta + \partial_t u_\theta)$ and $sg(\cdot)$ denotes stop-gradient.

**Inference formula**:
$$\mathbf{A}_0 = \mathbf{A}_1 - u_\theta^{cfg}(\mathbf{A}_1, 0, 1 | \mathbf{c})$$

The action trajectory $\mathbf{A}_0$ is obtained in a single step from noise $\mathbf{A}_1 \sim \mathcal{N}(0, I)$.

**Design Motivation**: MeanFlow eliminates three limitations simultaneously: (1) multi-step denoising (diffusion models), (2) ODE solver errors (flow matching), and (3) consistency constraints (FlowPolicy).

#### 2. **Classifier-Free Guidance (CFG) Integration**

**Function**: Incorporate CFG into the MeanFlow framework to strengthen conditional control.

**Mechanism**: During training, a guided velocity is constructed by blending conditional and unconditional predictions at mixing ratio $\omega$:

$$\tilde{v}_t \triangleq \omega v_t(\mathbf{A}_t | \mathbf{A}_0, \mathbf{c}) + (1-\omega) u_\theta^{cfg}(\mathbf{A}_t, t, t | \emptyset)$$

This guided velocity replaces the ordinary instantaneous velocity in constructing the MeanFlow training objective.

**Design Motivation**: CFG has been shown to substantially improve conditional generation quality in image synthesis. However, conventional CFG requires multi-step inference. The MeanFlow framework allows CFG to be integrated without sacrificing 1-NFE, since MeanFlow learns the averaged rather than instantaneous velocity.

#### 3. **Dispersive Loss**

**Function**: Regularize the intermediate feature space of the policy network during training to prevent feature collapse across different states and improve generalization.

**Mechanism**: Analogous to contrastive learning without positive pairs, a repulsive force is applied to intermediate representations of different samples within a batch:

$$\mathcal{L}_{Disp}(\theta) = \log \mathbb{E}_{i,j \in \mathcal{B}} \left[\exp\left(-\frac{\|\mathbf{z}_{\mathbf{A},i} - \mathbf{z}_{\mathbf{A},j}\|_2^2}{\tau}\right)\right]$$

where $\mathbf{z}_{\mathbf{A},i}$ denotes the output features of the UNet downsampling block and $\tau=1$ is a temperature hyperparameter.

**Design Motivation**: A pure regression objective only matches each state to its corresponding trajectory, without explicitly constraining the structure of the feature space. This leads to "feature collapse"—different scene states are mapped to similar latent points, preventing the policy from distinguishing subtle state differences that require distinct actions. The Dispersive Loss forces features to spread out, implicitly sharpening the policy's sensitivity to fine-grained scene variations, with particularly pronounced effects in few-shot learning settings.

**Key Advantage**: Computed only during training; zero additional overhead at inference, preserving 1-NFE speed.

### Loss & Training

Total loss:
$$\mathcal{L}_{total}(\theta) = \mathcal{L}_{cfg}(\theta) + \lambda \mathcal{L}_{Disp}(\theta)$$

$\lambda = 0.5$ balances the two terms. Training details: 10 expert demonstrations, batch size 128, AdamW optimizer, learning rate 0.0001, 3000 epochs for Adroit, 1000 epochs for Meta-World.

## Key Experimental Results

### Main Results

Success rates and inference speeds across 37 tasks (3 Adroit + 34 Meta-World):

| Method | NFE | Adroit Hammer | Adroit Door | Adroit Pen | MW Easy(21) | MW Medium(4) | MW Hard(4) | MW V.Hard(5) | Avg. |
|--------|-----|-------|------|-----|------|--------|------|---------|------|
| DP3 | 10 | 100±0 | 56±5 | 46±10 | 87.3±2.2 | 44.5±8.7 | 32.7±7.7 | 39.4±9.0 | 68.7±4.7 |
| FlowPolicy | 1 | 98±1 | 61±2 | 54±4 | 84.8±2.2 | 58.2±7.9 | 40.2±4.5 | 52.2±5.0 | 71.6±3.5 |
| **MP1** | **1** | **100±0** | **69±2** | **58±5** | **88.2±1.1** | **68.0±3.1** | **58.1±5.0** | **67.2±2.7** | **78.9±2.1** |

MP1 outperforms DP3 by **10.2%** and FlowPolicy by **7.3%**, with lower standard deviation (2.1% vs. 3.5%).

Inference speed comparison:

| Method | NFE | Avg. Inference / ms |
|--------|-----|-------------|
| DP3 | 10 | 132.2±11.2 |
| Simple DP3 | 10 | 97.0±9.2 |
| FlowPolicy | 1 | 12.6±1.5 |
| **MP1** | **1** | **6.8±0.1** |

MP1 achieves an inference latency of only **6.8 ms**, approximately **2× faster** than FlowPolicy and **~19× faster** than DP3.

Real-world experiments (5 tasks, 20 evaluations each):

| Task | MP1 | FlowPolicy | DP3 |
|------|-----|-----------|-----|
| Hammer | **90% / 18.6s** | 70% / 22.3s | 70% / 31.1s |
| Drawer Close | **100% / 8.8s** | 90% / 15.7s | 80% / 20.2s |
| Heat Water | **90% / 23.4s** | 60% / 31.1s | 70% / 38.8s |
| Stack Block | **80% / 27.2s** | 50% / 29.6s | 60% / 35.1s |
| Spoon | **90% / 22.6s** | 80% / 26.7s | 70% / 28.3s |

### Ablation Study

Dispersive Loss ablation (10 selected tasks):

| Configuration | Adroit Door | Adroit Pen | MW Coffee Pull | MW Disassemble | MW Pick Place Wall | Note |
|------|------------|-----------|---------------|----------------|-------------------|------|
| MP1 (full) | **58±5** | **69±2** | **92.3±3.7** | **74.0±1.4** | **64.3±1.2** | Standard config |
| MP1 − Disp Loss | 55±6 | 68±4 | 90.7±2.1 | 72.7±0.5 | 60.3±2.4 | Dispersive Loss removed |

MeanFlow ratio ablation:

| Flow Ratio | Adroit Pen | MW Dial Turn | MW Coffee Pull | MW Assembly | Avg. |
|-----------|-----------|-------------|---------------|------------|------|
| 0 (= FM) | 53±5 | 81±1 | 62±4 | 97±2 | 72.6±3.0 |
| **0.50** | **58±5** | **90±2** | **92±4** | **98±1** | **82.4±2.6** |
| 1.0 | 0±0 | 0±0 | 12±5 | 0±0 | 2.4±1.0 |

When $r=t$ (ratio=0), the model degenerates to standard flow matching and performance drops substantially; ratio=1.0 leads to complete failure.

### Key Findings

1. **MeanFlow outperforms flow matching**: Increasing the ratio from 0 (standard FM) to 0.5 (MeanFlow) raises the average success rate from 72.6% to 82.4%, validating the advantage of the averaged velocity field.
2. **Dispersive Loss provides greater benefit on harder tasks**: On the Meta-World Push task, the Dispersive Loss yields a 23.3 percentage-point improvement (50.7→74.0).
3. **Strong few-shot learning capability**: High success rates are achieved with only 10 demonstrations; increasing to 20 accelerates convergence with minimal effect on final performance.
4. **Successful sim-to-real transfer**: MP1 consistently outperforms baselines across all 5 real-robot tasks while completing them in shorter time.

## Highlights & Insights

1. **First application of MeanFlow in robotics**: Successful transfer from the image generation domain demonstrates the generality of this paradigm.
2. **Genuinely 1-NFE**: Unlike FlowPolicy, MP1 requires neither consistency constraints nor ODE solvers, yielding a theoretically cleaner formulation.
3. **Elegant design of Dispersive Loss**: As a training-time regularizer, it incurs zero inference overhead while substantially improving few-shot generalization. This can be viewed as a uniformity constraint on the feature space, potentially transferable to other policy learning settings.
4. **6.8 ms inference latency**: This reaches real-time control levels (~150 Hz), making it practically significant for high-frequency control tasks.

## Limitations & Future Work

1. **Evaluation limited to Adroit and Meta-World**: Although 37 tasks are included, the task types are relatively narrow (primarily tabletop manipulation); applicability to locomotion, navigation, and other domains remains unverified.
2. **Limited scale of real-world experiments**: 5 tasks with 10 evaluations each provide limited statistical significance.
3. **Dependency on 10 demonstrations**: Human demonstrations are still required; zero-shot learning is not addressed.
4. **UNet backbone**: The use of a conventional UNet rather than a Transformer-based architecture (e.g., DiT) may constrain expressiveness on more complex tasks.

## Related Work & Insights

- **MeanFlow (image generation)**: The theoretical foundation of MP1, demonstrating that the averaged velocity field enables genuinely single-step generation.
- **DP3**: Provides the foundational framework for 3D point cloud + diffusion-based robot policy learning.
- **FlowPolicy**: The most direct baseline; comparison highlights the advantages of MeanFlow over consistency-constraint-based approaches.
- **Dispersive Loss (Disperse paper)**: Supplies the conceptual basis for feature-space regularization, with demonstrated effectiveness in few-shot learning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First application of MeanFlow in robotics; introduction of Dispersive Loss
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 37 simulation tasks + 5 real-world tasks; multi-dimensional ablation analysis
- **Writing Quality**: ⭐⭐⭐⭐ — Clear methodological derivation; comprehensive experimental comparisons
- **Value**: ⭐⭐⭐⭐⭐ — 6.8 ms inference + SOTA success rates; significant implications for real-time robotic control

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] OMP: One-step Meanflow Policy with Directional Alignment](../../ICML2026/image_generation/omp_one-step_meanflow_policy_with_directional_alignment.md)
- [\[AAAI 2026\] EfficientFlow: Efficient Equivariant Flow Policy Learning for Embodied AI](efficientflow_efficient_equivariant_flow_policy_learning_for_embodied_ai.md)
- [\[CVPR 2026\] Temporal Equilibrium MeanFlow: Bridging the Scale Gap for One-Step Generation](../../CVPR2026/image_generation/temporal_equilibrium_meanflow_bridging_the_scale_gap_for_one-step_generation.md)
- [\[NeurIPS 2025\] Two-Steps Diffusion Policy for Robotic Manipulation via Genetic Denoising](../../NeurIPS2025/image_generation/two-steps_diffusion_policy_for_robotic_manipulation_via_genetic_denoising.md)
- [\[ICML 2026\] Riemannian MeanFlow for One-Step Generation on Manifolds](../../ICML2026/image_generation/riemannian_meanflow_for_one-step_generation_on_manifolds.md)

</div>

<!-- RELATED:END -->
