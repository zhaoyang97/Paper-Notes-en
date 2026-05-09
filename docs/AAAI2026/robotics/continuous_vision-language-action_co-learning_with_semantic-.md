---
title: >-
  [Paper Note] Continuous Vision-Language-Action Co-Learning with Semantic-Physical Alignment for Behavioral Cloning
description: >-
  [AAAI 2026][Robotics][Behavioral Cloning] This paper proposes the CCoL framework, which addresses both physical discontinuity in action sequences and semantic-physical misalignment in Behavioral Cloning through NeuralODE-driven Multimodal Continuous Co-learning (MCC) and bidirectional cross-attention-based Cross-modal Semantic-Physical Alignment (CSA). CCoL achieves an average relative improvement of 8.0% across three simulation platforms, with up to 19.2% on the bimanual insertion task.
tags:
  - AAAI 2026
  - Robotics
  - Behavioral Cloning
  - Semantic-Physical Alignment
  - NeuralODE
  - Multimodal Continuous Co-Learning
  - Language-Conditioned Manipulation
date: 2026-05-08
content_hash: 6e41ea90336cd108
---

# Continuous Vision-Language-Action Co-Learning with Semantic-Physical Alignment for Behavioral Cloning

**Conference**: AAAI 2026
**arXiv**: [2511.14396](https://arxiv.org/abs/2511.14396)
**Code**: Unavailable (not released)
**Area**: Multimodal VLM / Robot Manipulation / Behavioral Cloning
**Keywords**: Behavioral Cloning, Semantic-Physical Alignment, NeuralODE, Multimodal Continuous Co-Learning, Language-Conditioned Manipulation

## TL;DR
This paper proposes the CCoL framework, which addresses both physical discontinuity in action sequences and semantic-physical misalignment in Behavioral Cloning through NeuralODE-driven Multimodal Continuous Co-learning (MCC) and bidirectional cross-attention-based Cross-modal Semantic-Physical Alignment (CSA). CCoL achieves an average relative improvement of 8.0% across three simulation platforms, with up to 19.2% on the bimanual insertion task.

## Background & Motivation
Language-Conditioned Manipulation (LCM) learns control policies from human demonstrations via behavioral cloning and serves as a cornerstone of embodied AI. The central challenge in BC is **compounding error**—per-step prediction errors accumulate quadratically over time ($O(H^2\epsilon)$), leading to covariate shift.

Three existing mitigation strategies each exhibit fundamental limitations:
1. **Data augmentation** (noise injection, synthetic data): improves diversity but fails to address errors in fine-grained manipulation scenarios.
2. **Expressive representations** (e.g., R3M for semantic feature fusion): aligns language and vision globally but **ignores step-wise semantic adaptation**—e.g., executing "place the cup on the shelf" requires attending to the cup during grasping and the shelf during placement; static fusion cannot achieve such dynamic attention switching.
3. **Temporal abstraction** (ACT's action chunking, AWE's waypoint abstraction): reduces decision steps via segmentation but **introduces physical discontinuities**—abrupt bimanual waypoint transitions produce non-smooth accelerations and jittery trajectories, causing execution failures in long-horizon tasks.

The paper identifies two critical yet underexplored issues:
- **Physical discontinuity**: Discretized action modeling paradigms (e.g., piecewise constant control signals) violate differential continuity constraints.
- **Semantic-physical misalignment**: High-level semantic goals cannot accurately guide physical actions; static fusion methods lack step-wise semantic adaptation.

## Core Problem
How can a BC framework **simultaneously** guarantee: (1) temporal continuity and physical feasibility of action trajectories (smooth velocity and acceleration profiles); and (2) step-wise dynamic alignment between language instructions and visual-proprioceptive representations?

This problem is significant because long-horizon manipulation tasks (e.g., bimanual insertion, multi-stage kitchen tasks) are highly sensitive to both aspects—trajectory discontinuity directly causes execution jitter and task failure, while semantic misalignment causes the robot to attend to incorrect objects or regions at different task stages.

## Method

### Overall Architecture
The CCoL pipeline is as follows:
- **Input**: RGB(-D) visual observations $o_t$, natural language task instruction $l$, robot proprioceptive state $r_t$ (joint position sequences).
- **Encoding**: Three independent encoders—ViT extracts visual features $x_t$, RoBERTa encodes language embeddings $\hat{l}_t$, CVAE+Transformer encodes proprioceptive embeddings $e_t$.
- **MCC module**: Models the continuous-time evolution of proprioceptive embeddings via NeuralODE, then projects all three modalities into a shared space.
- **CSA module**: Bidirectional cross-attention performs step-wise alignment between language semantics and visual-proprioceptive representations.
- **Decoding**: A goal-conditioned decoder predicts the future $k$-step action sequence.
- **Output**: $k$-step joint position / end-effector actions.

### Key Designs

1. **Multimodal Continuous Co-learning (MCC)**: This is the central contribution. Conventional BC predicts actions independently at each step, ignoring the dynamic relationships between adjacent actions. MCC models the temporal evolution of proprioceptive embeddings as the solution to a continuous differential equation via NeuralODE:

    $z(t_\delta) = z_0 + \int_0^{t_\delta} f(z(t), t; \psi) dt$

   A CVAE first maps the proprioceptive [CLS] token to Gaussian distribution parameters $(\mu, \sigma)$; the initial latent state $z_0$ is sampled via reparameterization. A residual MLP $f$ serves as the ODE derivative function, and a Dormand-Prince adaptive step-size solver numerically integrates the equation to yield the continuous latent trajectory $\mathbf{Z}_t$. This trajectory replaces conventional step-wise proprioceptive features and is inherently temporally continuous.

   The three modality features are then projected into a shared $h$-dimensional space via linear layers with ReLU activation; language embeddings are additionally upsampled via bilinear interpolation to match visual resolution for pixel-level synchronization.

2. **Cross-modal Semantic-Physical Alignment (CSA)**: An attention attribution mapper is designed to anchor language tokens to visual-proprioceptive representations at each time step via bidirectional cross-attention:

    - **Language→Visuomotor**: Language embedding $\tilde{l}_t$ as Query; visual-proprioceptive concatenation $X_t = (\tilde{x}_t, \tilde{\mathbf{Z}}_t)$ as Key/Value.
    - **Visuomotor→Language**: Reversed; $X_t$ as Query, $\tilde{l}_t$ as Key/Value.

   Bidirectional attention scores determine correspondences between language tokens (e.g., noun "cube", verb "insert") and physical features (visual regions, joint trajectories). The final fused feature $\tilde{F}_t$ additionally undergoes position-encoded self-attention to maintain temporal consistency.

   **Key effect**: Attention visualizations in experiments demonstrate that CSA dynamically shifts attention in the cube transfer task from the right gripper (grasping stage) → the red cube (transfer stage) → the left gripper (handover stage), achieving step-wise semantic grounding.

3. **CVAE Proprioceptive Encoder**: Joint position sequences are linearly transformed and concatenated with a [CLS] token, augmented with sinusoidal positional encodings, and processed by a Transformer (or TCN as an ablation baseline). The CVAE architecture regularizes the latent space distribution, providing favorable initial conditions for the NeuralODE.

### Loss & Training
The total loss consists of three components:

$$\mathcal{L} = \frac{1}{N}\sum_N \mathcal{L}_{BC} + \mathcal{E}_{disc}$$

- **$\mathcal{L}_{BC} = \mathcal{L}_{recon} + \mathcal{L}_{KL}$**: The standard CVAE ELBO—reconstruction loss ensures decoded trajectories are consistent with expert demonstrations; KL divergence regularizes latent codes toward a standard Gaussian prior.
- **Discontinuity penalty $\mathcal{E}_{disc}$**: Constrains the actual rate of change $dz(t)/dt$ of the latent state to be consistent with the NeuralODE-predicted rate $f(z(t),t;\psi)$, ensuring smooth evolution of the latent trajectory.

Training details: SGD optimizer, learning rate 1e-5, momentum 0.9, action chunking size $k=50$, batch size 8. The ODE solver is evaluated at two discrete time points. Training takes 5.3 hours on an RTX 4090.

## Key Experimental Results

### Aloha MuJoCo (Bimanual Collaboration)

| Task | Metric | CCoL | DIC | AWE | ACT |
|------|--------|------|-----|-----|-----|
| Cube Transfer (scripted) | Success Rate% | **99.0** | 95.9 | 99.0 | 86.0 |
| Bimanual Insertion (scripted) | Success Rate% | **82.0** | 78.1 | 71.0 | 50.0 |
| Cube Transfer (human demo) | Success Rate% | **87.0** | 83.2 | 57.0 | 32.0 |
| Bimanual Insertion (human demo) | Success Rate% | **36.0** | 30.2 | 30.0 | 20.0 |
| Average | Success Rate% | **90.5/61.5** | 87.0/56.7 | 85.0/43.5 | 68.0/26.0 |

CCoL outperforms DIC by +5.8% (relative) on average and AWE by +11.8% (absolute). On the **human demonstration bimanual insertion task**, CCoL achieves a **19.2%** relative improvement over DIC—the most compelling result, as human demonstrations are noisier and better demonstrate the robustness of continuous trajectory modeling.

### RLBench (Multi-Scene)

| Method | LampOn | GrillMeat | Phone | OpenBottle | Avg |
|--------|--------|-----------|-------|------------|-----|
| CCoL (2D) | 93.7 | 82.3 | 44.3 | 51.7 | **68.0** |
| AWE | 85.7 | 74.3 | 34.7 | 46.3 | 60.3 |
| CCoL (3D) | **97.3** | **87.3** | **76.7** | **78.3** | **84.9** |
| 3DDiff | 89.3 | 85.0 | 71.7 | 69.3 | 78.8 |

The 2D setting surpasses AWE by +7.7%; the 3D setting (with RGB-D and 3D tokens) surpasses 3DDiff by +6.1%.

### Franka Kitchen (Long-Horizon)

| Method | Backbone | Single-Task Avg | Long-Horizon Avg |
|--------|----------|-----------------|------------------|
| MPI | ViT-S | 64.4 | 30.9 |
| CCoL | ViT-S | **68.9** (+6.9%) | **36.2** (+17.2%) |
| MPI | ViT-B | 66.9 | 34.2 |
| CCoL | ViT-B | **68.6** | **38.1** (+11.4%) |
| CCoL (frozen) | ViT-B | 65.9 | 34.5 |

On long-horizon tasks (chained ①+②+③ execution), CCoL (ViT-B) achieves 30.1%→38.1%. **With a frozen visual encoder, performance still reaches 34.5%**, indicating that proprioceptive continuous modeling and semantic alignment independently contribute substantial gains.

### Real-World Experiments
Experiments use a 7-DoF Franka Emika Panda with an Intel RealSense D435i; each of three tasks is trained on 50 demonstrations and evaluated over 15 trials:
- Cubes Placement: **86.7%** success rate.
- Pen Lifting / Cube Sliding: Good generalization under unseen object states (varying pen diameters, vase occlusion).

Inference speed: **0.015s (±0.003s)** per action sequence, approximately **67Hz** policy frequency, satisfying real-time requirements.

### Ablation Study
- **Removing MCC**: Bimanual insertion (scripted) drops by **15.0%**—continuous dynamics modeling is the core contributor.
- **Removing CSA**: Cube Transfer (human demo) drops by **9.0%** on average—semantic alignment is more critical under noisy demonstrations.
- **Removing $\mathcal{E}_{disc}$**: Performance degrades (82→78), confirming the discontinuity penalty is effective.
- **Replacing bidirectional attention with average pooling in CSA**: Performs worse than removing CSA entirely (72% < 73%), demonstrating that inappropriate fusion is more harmful than no fusion.
- **Replacing Transformer with TCN for proprioceptive encoding**: Insertion task drops by 13%; further removing MCC causes a 16% drop—the choice of temporal modeling architecture matters significantly.
- **Trajectory smoothness**: Compared to the version without MCC, CCoL reduces velocity fluctuation by **30.8%**, acceleration fluctuation by **32.7%**, and improves minimum acceleration by **20.2%**.
- **ODE time step**: A larger step size (2.0) outperforms a smaller one (0.5), as coarser steps are more robust while finer steps are sensitive to transient noise.

## Highlights & Insights
- **Using NeuralODE for proprioceptive modeling in BC is genuinely novel**—this is not a superficial application of NeuralODE, but rather an organic integration of continuous dynamics modeling with multimodal fusion, where the ODE-generated continuous latent trajectory directly participates in constructing the tri-modal shared space.
- **The semantic grounding effects of bidirectional cross-attention are impressive**—attention visualizations clearly show correspondences between nouns and visual regions, verbs and trajectory patterns, as well as dynamic attention switching across task stages.
- **Quantitative analysis of trajectory smoothness** (>30% reduction in velocity/acceleration fluctuation) provides direct evidence of physical feasibility, which is uncommon in the BC literature.
- **Lightweight design**: No large-scale pretraining or 3D point cloud input is required; the method is already highly competitive under the 2D RGB setting.
- **The competitiveness of the frozen visual encoder** indicates that the core gains stem from proprioceptive modeling and semantic alignment rather than visual encoder fine-tuning.

## Limitations & Future Work
- **Weak language encoder**: RoBERTa is used for text encoding without leveraging the strong semantic understanding of LLMs. The authors acknowledge in the Conclusion that extending to LLM-based methods is a future direction.
- **Limited task complexity**: Simulation tasks are primarily bimanual collaboration and simple multi-stage kitchen tasks; truly complex open-world long-horizon manipulation (e.g., table tidying, complex assembly) is not evaluated.
- **Small-scale real-world experiments**: Only 3 tasks with 50 demonstrations and 15 evaluations each; statistical significance is questionable.
- **No comparison with recent VLA foundation models**: Such as RT-2 and OpenVLA; while the parameter scales differ, readers would benefit from knowing the performance gap.
- **Computational overhead of the ODE solver**: Although inference reaches 67Hz, no comparison of inference speed with baselines is provided; the adaptive step-size solver may become a bottleneck in more complex scenarios.
- **Expressiveness of the CVAE**: The proprioceptive posterior is still restricted to a diagonal Gaussian; although NeuralODE enhances adaptability, this may be insufficient for multimodal action distributions (e.g., multiple feasible trajectories in bimanual tasks). Integration with diffusion models could be explored.

## Related Work & Insights
- **vs ACT (AAAI/RSS)**: ACT pioneered the action chunking paradigm to reduce decision steps, but is fundamentally piecewise prediction with physical discontinuities between adjacent chunks. CCoL directly addresses this via NeuralODE continuous dynamics modeling, significantly outperforming ACT across all benchmarks.
- **vs AWE**: AWE further simplifies trajectories via waypoint abstraction on top of ACT, but waypoint transitions introduce even more severe discontinuities. CCoL matches AWE on Aloha scripted tasks (99% vs 99%) but substantially outperforms it under the noisier human demonstration setting (61.5 vs 43.5).
- **vs DIC (Diffusion)**: DIC models action distributions via conditional denoising diffusion, offering stronger multimodal action modeling capability but lacking explicit temporal continuity constraints. CCoL marginally leads overall, with a clear advantage under human demonstration settings.
- **vs R3M**: R3M performs global visual-language alignment but ignores step-wise semantic adaptation. CCoL's CSA achieves step-wise dynamic alignment via bidirectional attention, substantially outperforming R3M on Franka Kitchen (68.9 vs 54.4 single-task).
- **vs LaDA (CVPR 2026)**: LaDA employs language-anchored action decoupling with soft-label contrastive learning, focusing on semantic interpretability of action representations. CCoL focuses on trajectory continuity and step-wise semantic grounding; the two approaches are complementary—CCoL's temporal continuity modeling could enhance LaDA's action decoding.

**Broader Implications**:
- The **NeuralODE + multimodal fusion** paradigm is generalizable to other multimodal tasks requiring temporal continuity, such as temporal modeling in video understanding and trajectory prediction in autonomous driving.
- The concept of **step-wise semantic grounding** has broad implications for the VLA field—most current VLA models condition on language globally (single input pass); CCoL demonstrates the importance of step-wise dynamic alignment.
- The design philosophy of the discontinuity penalty $\mathcal{E}_{disc}$ can be transferred to diffusion model trajectory generation as an additional regularization term to ensure physical feasibility of generated trajectories.

## Rating
- **Novelty**: ⭐⭐⭐⭐ NeuralODE+CVAE for proprioceptive modeling in BC is a novel combination; the bidirectional attention semantic grounding design is elegant. However, each individual component (NeuralODE, CVAE, cross-attention) is a mature technique.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three simulation platforms plus real-robot experiments, comprehensive ablations, and convincing trajectory smoothness analysis. However, real-world experiments are limited in scale, and comparisons with VLA foundation models are absent.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly derived (compounding error → discontinuity → semantic misalignment); mathematical notation is well-structured. Placing Related Work after the Method section is slightly inconvenient for readers.
- **Value**: ⭐⭐⭐⭐ Lightweight, real-time (67Hz), independent of LLMs or 3D inputs, with a low deployment barrier. However, the absence of open-sourced code limits reproducibility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] 10 Open Challenges Steering the Future of Vision-Language-Action Models](10_open_challenges_steering_the_future_of_vision-language-ac.md)
- [\[CVPR 2026\] CoMo: Learning Continuous Latent Motion from Internet Videos for Scalable Robot Learning](../../CVPR2026/robotics/como_learning_continuous_latent_motion_from_internet_videos_for_scalable_robot_l.md)
- [\[CVPR 2026\] Boosting Vision-Language-Action Finetuning with Feasible Action Neighborhood Prior](../../CVPR2026/robotics/boosting_vision-language-action_finetuning_with_feasible_action_neighborhood_pri.md)
- [\[CVPR 2026\] Adaptive Action Chunking at Inference-time for Vision-Language-Action Models](../../CVPR2026/robotics/adaptive_action_chunking_at_inference-time_for_vision-language-action_models.md)
- [\[AAAI 2026\] SemanticVLA: Semantic-Aligned Sparsification and Enhancement for Efficient Robotic Manipulation](semanticvla_semantic-aligned_sparsification_and_enhancement_for_efficient_roboti.md)

</div>

<!-- RELATED:END -->
