---
title: >-
  [Paper Note] DynBridge: Bridging Imagination and Control through Interaction Dynamics for Robot Manipulation
description: >-
  [CVPR 2026][Robotics & Embodied AI][Paper Note] DynBridge proposes "interaction dynamics" as a latent representation to end-to-end couple "imagining the future (trajectory generation)" and "control decision-making (action prediction)". This allows the robot to learn not just "where" the environment changes but also "how" actions cause these changes. It outperforms m
tags:
  - CVPR 2026
  - Robotics & Embodied AI
date: 2026-05-08
content_hash: 66c81f7212dc686c
---
# DynBridge: Bridging Imagination and Control through Interaction Dynamics for Robot Manipulation

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_DynBridge_Bridging_Imagination_and_Control_through_Interaction_Dynamics_for_Robot_CVPR_2026_paper.html)  
**Code**: https://github.com/Wang-Alexz/DynBridge  
**Area**: Robotics / Embodied AI  
**Keywords**: Robot Manipulation, Interaction Dynamics, Imitation Learning, Trajectory Generation, Action Prediction  

## TL;DR
DynBridge proposes "interaction dynamics" as a latent representation to end-to-end couple "imagining the future (trajectory generation)" and "control decision-making (action prediction)". This allows the robot to learn not just "where" the environment changes but also "how" actions cause these changes. It outperforms methods like ATM and GraphMimic on simulation and real-world benchmarks (LIBERO / Meta-World) without requiring pre-training on additional robot data.

## Background & Motivation
**Background**: Recent generative models allow robots to "imagine the future" by predicting subsequent frames using video diffusion or latent video representations, treats these rollouts as intermediate goals for an independent control policy. Another line of research reinforces spatial structural priors, such as generating point trajectories or building object-agent relationship graphs to locate interaction regions.

**Limitations of Prior Work**: Both lines optimize "imagination" and "control" in a decoupled manner. Generators trained to reconstruct future observations tend to prioritize visual realism over physical feasibility—a typical failure is "imagining a drawer opening automatically as the arm approaches" due to common patterns in training videos, even if the policy fails to apply contact force in real execution. Methods strengthening spatial structures are essentially "correlation-driven" within the visual domain and fail to model the causal physical quantities, such as force transfer, that drive interactions.

**Key Challenge**: Environmental evolution and agent actions are **bidirectionally coupled** (actions change the environment, and the environment constrains next actions). Existing methods typically model either "where" (spatial structure, observation-driven) or "how" (latent actions, inverse dynamics pseudo-labels without spatial anchors), but rarely both as an integrated whole. This results in a gap between the "imagined future" and "executable behavior."

**Goal**: Use a shared representation to simultaneously encode "where the environment changes" and "how actions causally induce these changes," supervised end-to-end to bridge the imagination–control gap.

**Core Idea**: Propose **interaction dynamics** latent representations that both foresee "where" spatial changes occur (supervised by trajectory reconstruction) and capture "how" agent actions induce these changes (supervised by action imitation). DynBridge builds an end-to-end framework around this latent representation to form a closed loop between trajectory generation and action prediction.

## Method

### Overall Architecture
Given demonstrations with language instructions and action labels $T=\{(\tau^a_i,\ell_i)\}$, where each trajectory consists of observation-action pairs $\{(o_{i,t},a_{i,t})\}$, the goal is to learn a policy $\pi_\theta$ guided by interaction dynamics. DynBridge consists of three serial modules: first, the **Interaction Dynamics Generator** fuses visual history, language, and learnable "dynamic tokens" to generate latent interaction dynamics $H_t$; then, the **Action-conditioned Dynamics Aggregator** compresses $H_t$ into action-aware compact representations $H^{agg}_t$; finally, the **Dynamics-guided Action Predictor** (Action-Transformer) performs temporal reasoning on $H^{agg}_t$ to autoregressively predict executable actions $\hat a_t$. Crucially, the trajectory decoding branch provides "where" spatial supervision, while the action predictor loss provides "how" causal supervision, jointly optimizing $H_t$.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Visual History + Language Instruction"] --> B["Interaction Dynamics Generator<br/>Dynamic token cross-modal attention<br/>→ Latent Interaction Dynamics Ht"]
    B -->|Trajectory Decoding Branch (Training Only)| W["where: Short-term trajectory L2 supervision"]
    B --> C["Action-conditioned Dynamics Aggregator<br/>Local action query path + Global compression path<br/>→ Hagg"]
    C --> D["Dynamics-guided Action Predictor<br/>Action-Transformer temporal reasoning"]
    D -->|how: Behavior Cloning| E["Executable Action at"]
    E -.End-to-end Joint Optimization.-> B
```

### Key Designs

**1. Interaction Dynamics Generator: Learning "where" and "how" with a single latent representation**

This is the conceptual foundation addressing the gap caused by decoupled imagination and control. Unlike ATM/GraphMimic, it does not generate deterministic explicit trajectories/graphs for downstream use but instead generates a **latent** interaction representation $H_t$.

The **trajectory supervision** uses **per-frame resampling + short-term tracking**: at each frame $o_t$, $N_q$ points $\{p^{(k)}_t=(u^{(k)}_t,v^{(k)}_t)\}$ are sampled uniformly and tracked for $L$ frames using CoTracker to obtain $P_{t:t+L}=\{p^{(k)}_{t:t+L}\}$. This ensures consistency between training and inference while adaptively covering dynamic or partially visible scenes.

The **interaction-attention** uses ResNet-18 for visual features $F_{t-h:t}$ and a frozen 6-layer MiniLM for instructions $G_l$. A set of learnable latent queries $Z_{dyn}=\{z^{(i)}_{dyn}\}$ ("dynamic tokens") queries the multimodal context:

$$H_t=\mathrm{CrossAttn}\big(Q=Z_{dyn},\,K=V=[F_{t-h:t};G_l]\big)$$

$H_t$ retains token-level structure. During training, a lightweight decoder $f_{dyn}$ maps $H_t$ to predicted trajectories $\hat P_{t:t+L}=f_{dyn}(H_t)$, aligned via $\mathcal{L}_{traj}=\lVert\hat P_{t:t+L}-P_{t:t+L}\rVert_2^2$ to inject "where" spatial structures. Simultaneously, $H_t$ serves as a condition for the action predictor, injected with "how" causal dynamics via behavior cloning. **The trajectory decoder is discarded during inference.**

**2. Action-conditioned Dynamics Aggregator: Dual-path compression**

The number of dynamic tokens $N_{tok}$ affects both "representation capacity" and "modal balance." This module uses **adaptive dual-path aggregation** to compress $H_t$ into $M$ ($M<N_{tok}$) action-aware embeddings.

The **local compression path** uses learnable action tokens $A_{act}\in\mathbb{R}^{M\times d}$ to query interaction dynamics. To keep $A_{act}$ stable as a "shared prior," it is **detached (grad-stopped)** within the aggregator, using only a bottleneck adapter: $\hat A_{act}=W_{up}\,\sigma(W_{down}\,\mathrm{sg}(A_{act}))$. The attention score matrix $S_t=\mathrm{Softmax}(Q_A K_{H_t}^\top/\sqrt{d})$ selects relevant interaction features for action patterns: $H^{local}_t=S_t(H_tW_v)$. The **global compression path** provides a stable global reference $H^{global}_t$ via linear projection. The final $H^{agg}_t=H^{local}_t+H^{global}_t$ balances fine-grained action focus with stability.

**3. Dynamics-guided Action Predictor: Temporal reasoning on aggregated dynamics**

A standard Transformer decoder performs decision-making. Input tokens for each timestep $t$ include: action tokens $A_{act}$ (capturing autoregressive action dependencies), aggregated interaction dynamics $H^{agg}_t$, visual embeddings $F_{t-h:t}$, and language embeddings $G_l$. The policy predicts control actions $\hat a_t$ via an MLP head. For continuous control, the MSE loss $\mathcal{L}_{act}=\lVert\hat a_t-a_t^\ast\rVert_2^2$ is used.

### Loss & Training
The framework is trained end-to-end with the total objective:

$$\mathcal{L}_{total}=\mathcal{L}_{act}+\beta\,\mathcal{L}_{traj}$$

$\mathcal{L}_{act}$ is the behavior cloning loss and $\mathcal{L}_{traj}$ is the trajectory reconstruction L2 loss. This joint objective ensures that "where" and "how" mutually shape the interaction dynamics.

## Key Experimental Results

### Main Results
Average success rates across five LIBERO subsets (mean of 3 seeds). DynBridge **beats all baselines without any external pre-training**:

| Method | Ext. Data | Spatial | Object | Goal | Long | 90 |
|------|-----------|---------|--------|------|------|-----|
| BC | ✗ | 0.39 | 0.51 | 0.42 | 0.16 | 0.29 |
| R3M-finetune | ✗ | 0.49 | 0.52 | 0.05 | 0.09 | 0.09 |
| UniPi | ✓ | 0.69 | 0.59 | 0.11 | 0.05 | 0.07 |
| ATM | ✓ | 0.68 | 0.68 | 0.77 | 0.39 | 0.48 |
| GraphMimic | ✓ | 0.88 | 0.89 | 0.87 | 0.56 | 0.67 |
| **Ours** | ✗ | **0.92** | **1.00** | **0.92** | **0.71** | **0.75** |

Gains are most significant in LIBERO-Long (0.71 vs GraphMimic 0.56) and LIBERO-90 (0.75 vs 0.67), indicating that interaction dynamics learn task-agnostic, transferable features.

### Ablation Study

| Configuration | Conclusion | Description |
|------|------|------|
| Full model | Best | Complete DynBridge |
| w/o e2e | Significant drop | Decoupled training of generator and predictor |
| w/o traj | Lower precision | Removing trajectory branch loses "where" supervision |
| ours-coord | Drop | Using explicit trajectory coordinates instead of latent dynamics |
| L=0 | Sharp drop | No future look-ahead |
| w/o Agg | Overall lower | Without aggregator, tokens are either insufficient or redundant |
| w/ actagg (ours) | +17.5% | Action-conditioned aggregation is the most effective |

### Key Findings
- **End-to-end joint optimization is critical**: Decoupled training (\textit{w/o e2e}) causes significant performance drops.
- **Where and how are complementary**: Removing the trajectory branch or decoupling leads to failure; both are needed for causally grounded dynamics.
- **Latent dynamics outperform explicit coordinates**: Latent representations handle multi-modality better than fixed coordinates.
- **Optimal prediction horizon $L$**: $L=0$ fails, but excessively long $L$ introduces uncertainty and error accumulation.
- **Aggregator mitigates token capacity dilemmas**: Using action tokens as queries (actagg) aligns interaction features with decision-making best.
- **Robustness to failure and embodiment transfer**: DynBridge learns causal structures and can even learn from failed demonstrations, outperforming ATM which is easily misled by sub-optimal trajectories.

## Highlights & Insights
- Formalizes the "imagination-control gap" as a learnable **interaction dynamics** latent representation.
- Uses a trajectory decoder as an auxiliary task during training and discards it during inference to minimize deployment overhead while retaining spatial intent.
- **Action-conditioned aggregation (actagg)** is a highly transferable trick: letting the "planned action" actively select relevant interaction features.
- Failure demonstrations are informative: the model learns action-object causal structures rather than just surface trajectories.

## Limitations & Future Work
- Dependency on external video trackers (CoTracker); tracking failures under occlusion/high speed can contaminate supervision.
- Sensitivity to hyperparameters like $L$ and token counts; guidance for selection across different robots is needed.
- Evaluation limited to manipulation; lacks explicit modeling of complex force feedback or tactile signals.

## Related Work & Insights
- **vs ATM (Where-focused)**: ATM relies on absolute coordinates and ignores action semantic alignment, making it brittle to position changes.
- **vs GraphMimic (Where-focused)**: Graph-based spatial relations are still observation-driven correlations; DynBridge emphasizes action-conditioned causal dynamics.
- **vs UniPi (Video Generation)**: Decoupled two-stage generation and inverse dynamics suffer from video-control mismatch.
- **vs VPT / PlaySlot (How-focused)**: Modeling "how" via latent actions lacks explicit spatial grounding; DynBridge adds "where" to compensate.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DAWN: Pixel Motion Diffusion is What We Need for Robot Control](dawn_pixel_motion_diffusion_robot_control.md)
- [\[CVPR 2026\] CLaD: Planning with Grounded Foresight via Cross-Modal Latent Dynamics](clad_planning_with_grounded_foresight_via_cross-modal_latent_dynamics.md)
- [\[ICCV 2025\] Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos](../../ICCV2025/robotics/moto_latent_motion_token_as_the_bridging_language_for_learning_robot_manipulatio.md)
- [\[CVPR 2026\] Contact-Aware Neural Dynamics](contact-aware_neural_dynamics.md)
- [\[CVPR 2026\] BiPreManip: Learning Affordance-Based Bimanual Preparatory Manipulation through Anticipatory Collaboration](bipremanip_learning_affordance-based_bimanual_preparatory_manipulation_through_a.md)

</div>

<!-- RELATED:END -->
