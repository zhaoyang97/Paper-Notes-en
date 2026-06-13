---
title: >-
  [Paper Note] From Abstraction to Instantiation: Learning Behavioral Representation for Vision-Language-Action Model
description: >-
  [ICML 2026][Robotics][VLA] BehaviorVLA utilizes a causal triple-stream Mamba encoder (VBE) to compress long-horizon demonstrations into time-invariant "behavioral prototypes $z_{\text{proto}}$" and time-varying "phase st…
tags:
  - "ICML 2026"
  - "Robotics"
  - "VLA"
  - "Behavioral Representation"
  - "Mamba"
  - "Flow Matching"
  - "Sim-to-Real"
date: 2026-05-08
content_hash: 035fbec5b77257f3
---

# From Abstraction to Instantiation: Learning Behavioral Representation for Vision-Language-Action Model

**Conference**: ICML 2026  
**arXiv**: [2605.22671](https://arxiv.org/abs/2605.22671)  
**Code**: [BehaviorVLA.github.io](https://BehaviorVLA.github.io)  
**Area**: Robotics / Embodied AI / VLA  
**Keywords**: VLA, Behavioral Representation, Mamba, Flow Matching, Sim-to-Real

## TL;DR
BehaviorVLA utilizes a causal triple-stream Mamba encoder (VBE) to compress long-horizon demonstrations into time-invariant "behavioral prototypes $z_{\text{proto}}$" and time-varying "phase states $z_{\text{phase}}$". A Phase-conditioned Behavior Decoder (PBD) then unfolds the behavioral skeleton into phase-aligned Gaussian priors via a Predictor-Corrector approach to guide a flow matching policy. It achieves new SOTA results across LIBERO, RoboTwin 2.0, and CALVIN benchmarks, matching OpenVLA-OFT performance using only 50% of real-world data.

## Background & Motivation

**Background**: VLA models (such as OpenVLA, $\pi_0$, $\pi_{0.5}$, UniVLA, etc.) map vision-language backbones directly to action sequences, relying on large-scale simulation data to build general manipulation capabilities.

**Limitations of Prior Work**: Performance collapses under distribution drift—changes in lighting, object materials, or camera perspectives lead to failure. Sim-to-Real typically requires extensive real-world fine-tuning as a safety net, which is costly and difficult to scale. Existing "latent action space" approaches (BeT, VQ-BeT, ACT) only alleviate local smoothness but suffer from two fundamental issues: (i) **Short-horizon temporal fragmentation**—cutting trajectories into independent chunks or discrete codebooks loses long-range dependencies; (ii) **Static execution alignment**—decoding actions from a fixed latent variable lacks awareness of the current execution progress, causing misalignment between actions and the actual scene.

**Key Challenge**: While high-dimensional visuomotor trajectories are hypothesized to concentrate near low-dimensional manifolds, standard VLAs learn mappings directly in environmental space without explicit manifold constraints. Introducing such constraints often sacrifices real-time feedback and precision.

**Goal**: To simultaneously achieve specific-to-general **abstraction**—distilling diverse demonstrations into unified behavioral representations—and general-to-specific **instantiation**—projecting abstract behaviors back into precise, state-aligned actions.

**Key Insight**: Explicitly decouple the latent space into a "time-invariant global task topology $z_{\text{proto}}$" and a "time-varying execution phase $z_{\text{phase}}$". The former is locked once via retrieval at the start of an episode to provide a stable skeleton, while the latter updates online at each step to synchronize action generation with physical execution progress.

**Core Idea**: Use a triple-stream Mamba architecture for "abstraction" and a Predictor-Corrector mechanism with phase attention to expand the "skeleton" into an "action prior." This prior is injected into the flow matching velocity field via additive bias, capturing both global topological stability and local reactive control precision.

## Method

### Overall Architecture
The model adds two modules to the $\pi_{0.5}$ backbone: (1) **VBE (Visuomotor Behavior Encoder)**: A causal triple-stream (vision $S_v$ / action $S_a$ / behavior $S_z$) architecture. Each stream uses Mamba for long-horizon temporal filtering followed by cross-stream attention fusion to compress the trajectory into $\{z_{\text{proto}}, z_{\text{phase}}\}$. The $z_{\text{proto}}$ of each offline demonstration is stored in a Behavior Memory Bank. (2) **PBD (Phase-conditioned Behavior Decoder)**: Retrieves Top-K global prototypes to get a weighted $\hat z_{\text{proto}}$, which is unfolded into a sequence of position-encoded latent anchors $\mathbf M$. It uses $z_{\text{phase}}^{(t)}$ as a query for phase attention to obtain a local context $c_t$, projected into a Gaussian prior $\mathcal N(\mu_\psi(c_t), \Sigma)$. Finally, the prior is injected into the flow matching policy's noise embedding via additive bias, and the velocity field $v_\theta$ integrates to produce the final action chunk. During inference, the prototype is retrieved once per episode, while the phase is updated every step for local corrections.

### Key Designs

1. **VBE Causal Triple-Stream Mamba + Progressive Cross-Stream Attention**:

    - **Function**: Compresses heterogeneous, long-horizon vision/action sequences into a "behavioral token" stream, preserving long-range causality while supporting spatial multi-modal fusion.
    - **Mechanism**: Each stream runs an independent Mamba. Discretization via ZOH yields time-varying parameters $\bar{\mathbf A}_t = \exp(\bm \Delta_t \mathbf A)$ and $\bar{\mathbf B}_t = (\bm \Delta_t \mathbf A)^{-1}(\bar{\mathbf A}_t - \mathbf I)\bm \Delta_t \mathbf B$, where $\bm \Delta_t = \text{Softplus}(\text{Linear}(x_t^{(m)}))$ makes the step size input-dependent. This acts as a **selective filter** that suppresses background noise while retaining key events. State recursion follows $h_t^{(m)} = \bar{\mathbf A}_t h_{t-1}^{(m)} + \bar{\mathbf B}_t \text{LN}(x_t^{(m)})$ with gated connections. Spatial fusion utilizes **progressive cross-stream attention**: vision and action streams first align low-level semantics, then the behavior stream extracts global task structure using $[\tilde h^{(v)}_t; \tilde h^{(a)}_t]$ as keys/values. The behavior stream thus becomes an information bottleneck, filtering residual noise and leaving the behavioral topology.
    - **Design Motivation**: Standard frame-level encoders lose long-range causality, and simple multi-modal concatenation loses spatial structure. Mamba provides long-horizon memory with $\mathcal O(L)$ complexity. The triple-stream separation + progressive attention processes time and space independently, preventing the Transformer from struggling with long-sequence and multi-modal complexity simultaneously.

2. **Manifold Coordinate Decoupling: Global Prototype Retrieval + Online Phase State**:

    - **Function**: Separates "what the task is" from "where it is currently" into two latent variables, providing PBD with a stable skeleton and real-time progress.
    - **Mechanism**: Global prototypes are obtained during training by temporal mean-pooling behavior tokens $z_{\text{proto}} = \tfrac{1}{T} \sum_t \tilde h_t^{(z)}$ and stored in a Memory Bank. At $t=0$ during inference, Top-K prototypes are retrieved using $q = \text{MLP}(\Phi(O_0, L))$ and combined via weighted softmax: $\hat z_{\text{proto}} = \sum_{i \in \mathcal N_K} \text{softmax}(\langle q, k_i\rangle/\kappa) \cdot z_{\text{proto}}^{(i)}$, remaining fixed for the episode. The local phase $z_{\text{phase}}^{(t)} = \text{VBE}_{\text{causal}}(z_{\text{phase}}^{(t-1)}, O_t, a_{t-1})$ is updated recursively each step.
    - **Design Motivation**: Embedding "long-range task structure" into a fixed vector stabilizes the semantics of the entire trajectory. Leaving the "step progress" to an online state ensures real-time alignment with physical execution. This orthogonal decomposition prevents a single latent variable from attempting to balance two conflicting requirements (stability vs. sensitivity).

3. **PBD Predictor-Corrector: Phase-aligned Prior + Flow Matching Geometric Bias**:

    - **Function**: Allows the flow matching policy to integrate within an "explicitly biased noise space," maintaining generative capacity for multi-modal distributions while enforcing global topological consistency.
    - **Mechanism**: The Predictor expands $\hat z_{\text{proto}}$ into $H$-step latent anchors $\mathbf M = \mathcal{G}_\phi(\hat z_{\text{proto}}) \oplus \mathbf P_{\text{pos}}$ using a generator $\mathcal{G}_\phi$. Phase state queries $c_t = \text{Progress-Attn}(Q=z_{\text{phase}}^{(t)}, K=\mathbf M, V=\mathbf M)$ perform differential interpolation on anchors, projected into a Gaussian prior $\mu_\psi(c_t)$. The Corrector is a conditional flow matching model: the prior is injected via additive bias $\tilde e(a_\sigma) = e(a_\sigma) + \lambda \cdot \text{Proj}_\phi(\mu_{\text{prior}})$ into the noise embedding. The velocity field $v_\theta$ predicts the OT path velocity $u_\sigma = a_1 - a_0$ on this biased embedding. Bernoulli dropout masks replace fixed $\lambda$ during training to prevent posterior collapse.
    - **Design Motivation**: Standard latent decoders fail to keep up with real-time scene changes. The Predictor-Corrector allows global structure (prior mean) and local precision (flow matching correction) to handle their respective roles. Injecting the prior into the embedding rather than the action mathematically shifts the flow matching manifold toward high-probability regions, acting as a "soft constraint" toward the task topology.

### Loss & Training
**Two-stage training strategy**. Phase 1 (Behavioral Manifold Learning): $\mathcal L_{\text{Stage1}} = \mathcal L_{\text{rec}} + \alpha \mathcal L_{\text{global}} + \beta \mathcal L_{\text{local}}$. The reconstruction loss follows the JEPA concept, regressing the next action and next EMA vision encoding $\Phi_{\text{ema}}(O_{t+1})$ simultaneously. Global loss uses supervised contrastive learning to pull $z_{\text{proto}}$ with the same behavior labels together; local loss uses InfoNCE to differentiate $z_t$ across time-steps and prevent topological collapse. Phase 2 (Prior-guided Policy Tuning): $\mathcal L_{\text{Stage2}} = \mathcal L_{\text{flow}} + \lambda_{\text{prior}} \mathcal L_{\text{prior}}$, where the flow matching loss is MDSE of OT path velocity and the prior loss is the NLL of expert actions under the predicted Gaussian.

## Key Experimental Results

### Main Results
Average success rates in RoboTwin 2.0 Hard setting (domain randomization + noise, 20 tasks / 100 rollouts):

| Method | Bottle | Box Pour | Rolling Pin | Bread | Burger | Container | Average |
|------|------|---------|---------|--------|--------|--------|------|
| DP3 | 3% | 2% | 3% | 1% | 18% | 1% | Low |
| RDT | 75% | 43% | 11% | 2% | 27% | 17% | 20.3% |
| $\pi_0$ | 56% | 80% | 22% | 4% | 4% | 45% | ~25% |
| $\pi_{0.5}$ | 75% | 82% | 32% | 28% | 46% | 55% | ~50% |
| **Ours** | **83%** | **90%** | **41%** | **36%** | **61%** | **62%** | **58%** |

Average success rates across four LIBERO suites:

| Method | Spatial | Object | Goal | Long | Average |
|------|---------|--------|------|------|------|
| Diffusion Policy | 78.5 | 87.5 | 73.5 | 64.8 | 76.1 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| $\pi_{0.5}$ | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| **Ours** | **99.2** | **99.4** | **98.8** | **94.6** | **98.0** |

The improvements in the LIBERO-Long suite are most significant (+2.2 over $\pi_{0.5}$), confirming the value of VBE long-horizon modeling and PBD phase alignment for long manipulation tasks.

### Ablation Study
| Config (VBE / PBD) | LIBERO Long | Real-World Gen. | Real-World Long |
|------------------|-------------|------------------|------------------|
| — / — (Baseline) | 92.4 | 57.0 | 41.0 |
| ✓ / — | 93.8 | 65.0 | 48.0 |
| — / ✓ | 93.4 | 60.0 | 45.0 |
| ✓ / ✓ (Full) | 94.6 | 70.0 | 55.0 |

Real-World: Ours increases average success across 8 real-world tasks by 63% compared to strong baselines and matches full fine-tuned OpenVLA-OFT using only 50% of demonstration data.

### Key Findings
- Removing VBE leads to a 16% drop in real-world performance—without the abstraction end, the model overfits to environmental noise and loses the invariant task structure. Removing PBD leads to a 9.6% drop—without phase alignment, actions misalign with the scene during long-horizon execution.
- Guidance strength $\lambda$ has a clear sweet spot: too small and the prior provides no structural constraint; too large and it suppresses the flow matching's local correction capability.
- Retrieval count $k=5$ is optimal: too few and the prior is biased by a single prototype; too many and irrelevant prototypes disrupt structural guidance.
- t-SNE shows all three streams are essential—removing the vision stream causes tasks with similar actions but different visual semantics (e.g., "moving a pot" vs "wiping a table") to collapse together; removing the action stream leaves only static visual descriptions, failing to distinguish tasks with different manipulation dynamics under the same visual state.

## Highlights & Insights
- **Explicit decoupling into "Task Topology + Execution Progress"**—A one-size-fits-all latent variable cannot be both "stable and sensitive." Ours assigns these conflicting requirements to two variables at different time scales, which is the foundation of the architecture. Decoupling latents by physical dimension is a transferable insight for any long-horizon decision task.
- **Additive bias injection into the Flow Matching velocity field** is an elegantly simple engineering trick for "prior injection." It requires no change to the flow matching objective, only modifying the noise embedding. Mathematically, this shifts the attention manifold toward high-probability regions, acting as a "soft constraint."
- **Matching OpenVLA-OFT with 50% data** is the most impactful practical conclusion—it links data efficiency directly to representation learning (VBE abstraction + prototype retrieval) rather than just larger models or more demos.

## Limitations & Future Work
- Global prototype retrieval depends on the **topological coverage of the offline memory bank**. For novel tasks significantly different from the training distribution, retrieved skeletons might be "geometrically consistent but functionally incorrect," misleading the PBD. Online manifold expansion mechanisms are needed.
- Flow matching Predictor-Corrector requires iterative ODE integration, leading to higher inference latency than pure regression baselines. This is a burden for high-frequency control on compute-constrained hardware; consistency distillation is a potential future direction.
- Evaluations focused on tabletop bimanual manipulation (GALAXEA R1 Lite). More complex scenarios like mobile manipulation or long-range planning (opening doors, crossing rooms) remain unaddressed.
- Behavior labels rely on manual pre-definition. The supervised contrastive loss $\mathcal L_{\text{global}}$ requires paired labels; label quality directly affects prototype space geometry. Weakly/unsupervised versions are the next step.

## Related Work & Insights
- **vs $\pi_{0.5}$ / OpenVLA-OFT**: All use VLA + Flow Matching/Diffusion, but the former rely on implicit vision-action mappings without an explicit "behavioral representation" layer. Ours adds manifold coordinate decoupling to improve generalization and data efficiency.
- **vs BeT / VQ-BeT / ACT**: These share the "latent action space" philosophy, but chop trajectories into chunks or discrete tokens, losing long-range dependencies and using static decoding. Ours uses Mamba for global continuity and phase states for real-time tracking, avoiding temporal fragmentation and static alignment issues.
- **vs MemoryVLA / RPT / ICRT / MTIL**: These works use history as contextual input. Ours **explicitly decomposes** history into retrieved global prototypes and online phase states, distinguishing "task memory" from "execution progress."

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of Triple-stream Mamba, phase-conditioned flow matching, and prototype retrieval is rare in VLAs; the explicit latent splitting is original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers three simulation benchmarks, 8 real-world tasks, data efficiency ablations, and t-SNE visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ The Method section progresses logically from motivation to architecture to parametrization.
- **Value**: ⭐⭐⭐⭐⭐ Matching OpenVLA-OFT with half the data provides a clear path for data-efficient representation learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Dual-Stream Diffusion for World-Model Augmented Vision-Language-Action Model](dual-stream_diffusion_for_world-model_augmented_vision-language-action_model.md)
- [\[ICML 2026\] Contrastive Representation Regularization for Vision-Language-Action Models](contrastive_representation_regularization_for_vision-language-action_models.md)
- [\[ICML 2026\] Seeing Realism from Simulation: Efficient Video Transfer for Vision-Language-Action Data Augmentation](seeing_realism_from_simulation_efficient_video_transfer_for_vision-language-acti.md)
- [\[ICLR 2026\] AutoFly: Vision-Language-Action Model for UAV Autonomous Navigation in the Wild](../../ICLR2026/robotics/autofly_vision-language-action_model_for_uav_autonomous_navigation_in_the_wild.md)
- [\[ICML 2026\] Spatial Memory for Out-of-Vision Manipulation in Vision-Language-Action](spatial_memory_for_out-of-vision_manipulation_in_vision-language-action.md)

</div>

<!-- RELATED:END -->
