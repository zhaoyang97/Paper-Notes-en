---
title: >-
  [Paper Note] Efficient Robotic Policy Learning via Latent Space Backward Planning
description: >-
  [ICML 2025][Robotics][Robot planning] Proposes Latent Space Backward Planning (LBP), which recursively predicts intermediate subgoals starting from the final goal to sequentially approach the current state. This significantly improves planning efficiency while maintaining task alignment, achieving a new state of the art (SOTA) in both LIBERO-LONG simulation and real-robot long-horizon tasks.
tags:
  - "ICML 2025"
  - "Robotics"
  - "Robot planning"
  - "latent space"
  - "backward planning"
  - "subgoal prediction"
  - "long-horizon manipulation"
date: 2026-05-08
content_hash: 83b54705b5c24a44
---

# Efficient Robotic Policy Learning via Latent Space Backward Planning

**Conference**: ICML 2025  
**arXiv**: [2505.06861](https://arxiv.org/abs/2505.06861)  
**Code**: [Project Page](https://lbp-authors.github.io)  
**Area**: Robotics  
**Keywords**: Robot planning, latent space, backward planning, subgoal prediction, long-horizon manipulation

## TL;DR

Proposes Latent Space Backward Planning (LBP), which recursively predicts intermediate subgoals starting from the final goal to sequentially approach the current state. This significantly improves planning efficiency while maintaining task alignment, achieving a new state of the art (SOTA) in both LIBERO-LONG simulation and real-robot long-horizon tasks.

## Background & Motivation

Robotic planning faces a fundamental "trilemma" in reconciling **efficiency, precision, and sufficient future guidance**.

Existing methods fall into two main categories, both having critical limitations:

**Video planning methods** (e.g., UniPi, HiP, Seer, GR-1): Predict future images frame-by-frame as policy guidance. Although providing rich future information, they are computationally intensive, suffer from accumulating temporal errors, and can mislead downstream policies by generating physically inconsistent frames.

**Coarse-grained subgoal planning methods** (e.g., SuSIE, MimicPlay): Predict sparse intermediate subgoals to improve efficiency. However, they follow a **forward planning** paradigm—predicting subgoals sequentially forward from the current state. This is highly susceptible to cumulative errors that lead to off-task behaviors. While current methods introduce reachability/optimality checks to correct errors, these are post-hoc remedies that increase complexity rather than fundamentally solving the problem.

Core Problem: Can efficient and accurate planning be achieved simultaneously in long-horizon multi-stage tasks?

LBP answers this by mimicking human cognitive planning: imagining the ultimate outcome first, and then backward-decomposing it into progressively actionable smaller goals.

## Method

### Overall Architecture

LBP (Latent Space Backward Planning) consists of three core modules:

1. **Latent Goal Predictor** ($f_g$): Maps the task language description and current observation to the latent space representation $z_g$ of the final goal.
2. **Backward Subgoal Predictor** ($f_w$): Starts from $z_g$ and recursively predicts intermediate subgoals $w_1, w_2, \ldots, w_n$ that are temporally closer and closer to the current state.
3. **Goal-Fusion Policy** ($\pi$): Adaptively aggregates subgoal insights via a Perceiver-style cross-attention mechanism to guide action generation.

The entire planning process is executed in the **latent space** (using DecisionNCE or SigLIP encoders) instead of the pixel space, significantly reducing the computational load.

### Key Designs

#### 1. Grounding Task Objective as Latent Goals

Language instructions typically degrade to simple task identifiers in long-horizon tasks and lack fine-grained guidance capabilities. LBP learns a goal prediction model $f_g$ to map the current state $z_t$ and language feature $\phi_l$ to the final latent goal $z_g$:

$$\max_{f_g} \sum_{\tau \in \mathcal{D}_z} \sum_{1 \leq t \leq H} \mathbb{E}_{p(z_g, \phi_l | \tau)} \log f_g(z_g | z_t, \phi_l)$$

Such goals are not static; instead, they are **dynamically generated based on the current scene**. For instance, given the instruction "place the brown cup in front of the white cup", the final goal state depends on the actual physical location of the white cup.

#### 2. Backward Subgoal Prediction

This is the core novelty of LBP. Unlike traditional forward planning, LBP adopts a backward planning scheme **from the ultimate goal back to the current state**:

- **First step**: Predicts the first subgoal $w_1$ by anchoring on $z_g$ (the final goal), locating it temporally close to the final goal.
- **Recursion**: Each subsequent subgoal $w_i$ is predicted from the preceding level $w_{i-1}$, moving closer and closer to the current state.

All levels of subgoal predictors can **share a single unified model** $f_w$ due to their structural identity. Defining the recursive planning coefficient $\lambda = \frac{\Gamma(w_i) - t}{\Gamma(w_{i-1}) - t}$, the unified objective is formulated as:

$$\max_{f_w} \sum_{\tau \in \mathcal{D}_z} \sum_{1 \leq t < H} \mathbb{E} \left[ \sum_{i=1}^{n} \log f_w(z_{\lambda_i} | z_t, z_{\lambda_{i-1}}, \phi_l) \right]$$

The training objective contains two terms:

- **Ground-truth supervision term**: Evaluates predictions against ground-truth subgoals extracted from trajectories.
- **Self-consistency term**: Uses $f_w$'s own predictions as input for supervision, ensuring the consistency of recursive inference during deployment.

**Three primary advantages of backward planning**:

- The subgoal sequence spans the entire task horizon, offering non-uniform temporal sampling from **coarse to fine**.
- Anchoring on the final goal guarantees task alignment, **mitigating cumulative errors**.
- Recursive predictions reduce the number of planning steps required, enhancing **computational efficiency**.

#### 3. Goal-Fusion Module

The subgoal sequence $c = \{w_n, \ldots, w_1, z_g, \phi_l\} \in \mathbb{R}^{(n+2) \times N_z}$ is high-dimensional; direct concatenation would place a heavy burden on policy learning. LBP introduces a Perceiver-style cross-attention:

- Employs a learnable query vector $z \in \mathbb{R}^{1 \times N_z}$ to apply cross-attention over the subgoal sequence.
- Outputs a compressed context embedding $z_c$.
- Adaptively extracts the most relevant information from subgoals at varying distances.

This allows the policy to **dynamically balance short-term and long-term guidance**: prioritizing far-range subgoals during large displacements to avoid hindering future progress, and prioritizing near-range subgoals during fine manipulation.

### Loss & Training

LBP is trained in **three decoupled phases**:

1. **Goal Predictor $f_g$**: A 2-layer MLP trained via maximum likelihood estimation (Eq. 2).
2. **Subgoal Predictor $f_w$**: A 2-layer MLP trained via maximum likelihood with self-consistency regularization (Eq. 5).
3. **Low-level Policy $\pi$**: ResNet-34 backbone + FiLM language injection + residual MLP, utilizing **diffusion loss** (denoising steps = 25) with an action pooling/chunking size of 6.

Training hyperparameters: High-level planner batch=64, trained for 100k steps; low-level policy batch=64/128, trained for 200k/400k steps (simulation/real-world).

Default setup: 3-step planning (final goal + 2 intermediate subgoals) with $\lambda = 0.5$.

## Key Experimental Results

### Main Results

Results on LIBERO-LONG (10 long-horizon robotic manipulation tasks, 50 expert demonstrations per task):

| Method | Type | Avg. Success (%) | Rel. to LBP |
|------|------|:---:|:---:|
| MTACT | Multi-task Policy | 41.0 | -47.6 |
| OpenVLA | VLM Policy | 54.0 | -34.6 |
| MVP | Pre-trained Rep. | 68.2 | -20.4 |
| MPI | Interactive Rep. | 77.3 | -11.3 |
| Seer | Video Planning | 78.6 | -10.0 |
| SuSIE | Image Edit Subgoal | 76.3 | -12.3 |
| **LBP (SigLIP)** | Latent Backward Planning | 85.0 | -3.6 |
| **LBP (DecisionNCE)** | Latent Backward Planning | **88.6** | — |

Real-robot experiment (AIRBOT 6DoF, 4 long-horizon tasks, 200 demonstrations):

| Task | LCBC | GLCBC | SuSIE | **LBP** |
|------|:---:|:---:|:---:|:---:|
| Stack 3 cups (Avg. Score) | 78.7 | 84.6 | 60.4 | **84.6** |
| Move cups (Avg. Score) | 60.4 | 62.9 | 46.2 | **77.9** |
| Stack 4 cups (Avg. Score) | 55.0 | 45.5 | 42.5 | **72.5** |
| Shift cups (Avg. Score) | 41.8 | 36.1 | 17.7 | **67.1** |

Key Finding: **The longer the task (the more stages involved), the more pronounced LBP's advantage becomes**. In the most challenging task, Shift cups (5 stages), LBP scores 26.6 in the final stage, whereas all baselines score 0.

### Ablation Study

| Configuration | Avg. Success (%) | Description |
|------|:---:|------|
| No Goal No Subgoal (LCBC) | 77.3 | Language-conditioned only |
| Final goal $z_g$ only | 83.3 | +6.0%, visual goal is effective |
| $z_g$ + 1 subgoal ($\lambda=0.5$) | 85.6 | +2.3%, subgoals bring further improvement |
| $z_g$ + 2 subgoals ($\lambda=0.5$) | **88.6** | Optimal configuration |
| $z_g$ + 3 subgoals ($\lambda=0.5$) | 83.0 | Performance drops with excessive subgoals |
| Goal-fusion → average pooling | 79.0 | -9.6%, naive pooling significantly harms performance |
| Forward planning vs Backward planning | — | Forward planning MSE increases exponentially at distant subgoals, whereas backward planning maintains low error throughout. |

### Key Findings

1. **Backward Planning vs. Forward Planning**: Comparing subgoal prediction MSE across 3000 sample points, the error in forward planning scales up dramatically as subgoal distance increases (especially in the hardest Shift Cups task), whereas backward planning maintains low error throughout.
2. **Parallel Planning vs. Backward Planning**: Although parallel prediction avoids cumulative errors, it suffers from overall lower prediction accuracy due to the simultaneous optimization constraint over all subgoals.
3. **Number of Subgoals**: 2 intermediate subgoals are optimal; more subgoals degrade performance, highlighting LBP's efficiency.
4. **Robustness to $\lambda$**: Results for $\lambda = 0.5$ and $\lambda = 0.75$ are close, indicating the framework's insensitivity to this hyperparameter.
5. **Generalization**: LBP still significantly outperforms the LCBC baseline when facing distractor objects and different backgrounds in the Shift Cups task.

## Highlights & Insights

1. **Heuristic Value of the Backward Planning Paradigm**: Planning from the end point back to the start is an elegant and profound approach, similar to human "begin with the end in mind" thinking. Compared to forward planning, backward planning naturally avoids off-task issues induced by cumulative errors.
2. **Lightweight Implementation**: Both the goal predictor and the subgoal predictor are merely 2-layer MLPs, eschewing the need to train heavy image-editing diffusion models (like SuSIE) or perform frame-by-frame video prediction (like Seer).
3. **Unified Subgoal Predictor**: Subgoals at different levels share the same model $f_w$, which is parameter-efficient and simplifies training.
4. **Non-Uniform Temporal Sampling**: The subgoal sequence naturally forms a distribution that is dense in the short term and sparse in the long term. This offers precise operational guidance in the near future while maintaining task alignment for the far future, which aligns better with practical needs than uniform sampling.
5. **Necessity of Goal-Fusion**: Ablations show that naive average pooling results in a 9.6% performance drop, indicating that the adaptive integration of subgoals at different distances is crucial.

## Limitations & Future Work

1. **Subgoal Selection Mechanism**: Currently, LBP relies on a fixed $\lambda$ for uniform recursion. Future work could incorporate **keyframe detection** methods to adaptively select temporal points of subgoals containing the highest information density.
2. **Quality of the Latent Space**: LBP depends heavily on the representation quality of pre-trained encoders (e.g., DecisionNCE / SigLIP). Utilizing better robot-specific encoders may yield further performance gains.
3. **Complexity of the Real World**: Real-world experiments are restricted to cup manipulation (pick-and-place), leaving more complex operations (such as tool-use or deformable objects) unverified.
4. **Training Efficiency**: The framework is trained via three separate phases; the possibility of end-to-end co-training remains unexplored.
5. **Scenario Generalization**: Experiments are only conducted on tabletop manipulation scenes, requiring further validation of generalization performance across diverse tasks and environments.

## Related Work & Insights

- **SuSIE** (Black et al., ICLR 2024): Generates subgoal images via image-editing diffusion models. Operating in the pixel space incurs heavy computational overhead and is prone to hallucinations.
- **Seer** (Tian et al., ICLR 2025): An end-to-end predictive inverse-dynamics model that jointly predicts actions and future video frames.
- **DecisionNCE** (Li et al., ICML 2024): Constructs a latent space reflecting multimodal representations via implicit preference learning.
- **Diffusion Policy** (Chi et al., RSS 2023): Models action distributions using diffusion processes, a formulation adopted by LBP's low-level policy.
- **Perceiver** (Jaegle et al., ICML 2021): A cross-modal attention architecture that inspired the goal-fusion module in LBP.

## Rating

| Dimension | Score (1-5) | Comments |
|------|:---:|------|
| Novelty | 4 | Novel backward planning scheme; elegant design of the unified subgoal predictor. |
| Technical Depth | 4 | Clear theoretical analysis; well-designed recursive self-consistency training. |
| Experimental Thoroughness | 4 | Simulation + real robot; comprehensive ablation study; convincing baseline comparison against forward/parallel baselines. |
| Writing Quality | 4 | Clear motivation, intuitive diagrams, and coherent logic. |
| Value | 4 | Lightweight MLP implementation suitable for real-time deployment. |
| **Total Score** | **4.0** | A solid piece of work addressing practical problems. The backward planning paradigm is worth generalizing to more scenarios. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Closed-loop Long-horizon Robotic Planning via Equilibrium Sequence Modeling](closed-loop_long-horizon_robotic_planning_via_equilibrium_sequence_modeling.md)
- [\[CVPR 2026\] Learning a Unified Latent Action Space from Videos with Action-centric Cycle Consistency](../../CVPR2026/robotics/learning_a_unified_latent_action_space_from_videos_with_action-centric_cycle_con.md)
- [\[CVPR 2026\] Fast-ThinkAct: Efficient Vision-Language-Action Reasoning via Verbalizable Latent Planning](../../CVPR2026/robotics/fast-thinkact_efficient_vision-language-action_reasoning_via_verbalizable_latent.md)
- [\[ICCV 2025\] Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning](../../ICCV2025/robotics/resolving_token-space_gradient_conflicts_token_space_manipulation_for_transforme.md)
- [\[CVPR 2026\] AGiLe: Learning Robust Long-Horizon Manipulation via Affordance-Grounded Bidirectional Latent Planning](../../CVPR2026/robotics/agile_learning_robust_long-horizon_manipulation_via_affordance-grounded_bidirect.md)

</div>

<!-- RELATED:END -->
