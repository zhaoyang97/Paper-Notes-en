---
title: >-
  [Paper Note] DeepSight: Long-Horizon World Modeling via Latent States Prediction for End-to-End Autonomous Driving
description: >-
  [ICML 2026][Autonomous Driving][End-to-End Driving] DeepSight shifts "future world prediction" from explicit pixel reconstruction (codebook single-frame) to **multi-frame parallel implicit prediction** of DINOv3 semantic…
tags:
  - "ICML 2026"
  - "Autonomous Driving"
  - "End-to-End Driving"
  - "World Model"
  - "Latent Semantic Features"
  - "BEV Long-Horizon Prediction"
  - "Adaptive CoT"
date: 2026-05-08
content_hash: f91c555af2965e4c
---

# DeepSight: Long-Horizon World Modeling via Latent States Prediction for End-to-End Autonomous Driving

**Conference**: ICML 2026  
**arXiv**: [2605.10564](https://arxiv.org/abs/2605.10564)  
**Code**: https://github.com/hotdogcheesewhite/DeepSight  
**Area**: Autonomous Driving / VLM / World Model / End-to-End Planning  
**Keywords**: End-to-End Driving, World Model, Latent Semantic Features, BEV Long-Horizon Prediction, Adaptive CoT

## TL;DR
DeepSight shifts "future world prediction" from explicit pixel reconstruction (codebook single-frame) to **multi-frame parallel implicit prediction** of DINOv3 semantic features in BEV space, with an additional on-demand Adaptive Chain-of-Thought. This enables Qwen2.5-VL-3B to achieve a Driving Score of 86.23 (+7.39) and Success Rate of 71.36% (+13.63) on Bench2Drive closed-loop, with only ~4% extra inference latency.

## Background & Motivation
**Background**: End-to-end autonomous driving has recently integrated VLM/MLLMs extensively—leveraging pretrained world knowledge and language reasoning to enhance decision-making. One line of work (EMMA, SimLingo, ORION) adopts a "textual CoT" approach, explicitly writing scene descriptions and reasoning in natural language; another (FSDrive, HERMES, ReasonPlan) uses "unified generation-understanding," letting VLMs directly predict future frames (pixels or LiDAR) as a world model.

**Limitations of Prior Work**: The authors identify three main issues. (1) Visual representation is suboptimal: codebook autoregressive prediction of future frames emphasizes texture but loses semantics, which is ineffective for driving decisions; (2) Temporal horizon is too short: most existing world models predict only 0.5 seconds ahead, insufficient for safe trajectory planning; (3) Narrow field of view: mainstream approaches focus only on the front view, failing to model surrounding agents, making complex interaction scenarios prone to accidents.

**Key Challenge**: An ideal driving world model must simultaneously provide **precise semantic understanding, accurate spatial localization, long-horizon motion modeling, and fast response**. Existing solutions make trade-offs: explicit pixel reconstruction is slow and texture-biased, textual CoT is slow and spatially imprecise, and front-view input loses surround information.

**Goal**: (1) Use a "semantics-heavy, texture-light" representation as the ground truth for future world states; (2) Predict multiple future frames in a single forward pass; (3) Unify perception in BEV space and invoke CoT as needed.

**Key Insight**: The authors observe that self-supervised features like DINOv3 are inherently rich in semantics and can serve as GT for the "future BEV world." Predicting **latent features** instead of **pixel tokens** saves significant computation, enabling long-horizon, multi-frame parallel prediction. CoT should be a scarce resource—triggered only in long-tail scenarios, not every frame.

**Core Idea**: Use a set of learnable World Queries to predict DINOv3 BEV semantic features for 5 future frames in parallel as the world modeling target, combined with an Adaptive CoT that the model autonomously decides to activate. This decouples but links "world modeling" and "language reasoning."

## Method

### Overall Architecture
DeepSight builds on Qwen2.5-VL-3B, defining a unified generation-understanding model $M_{\text{uni}}$. Input: current $N$-view surround images $\mathbf{I}_t$, 4-frame history $\mathbf{I}_{t-\tau}$, ego state $T_{\text{ego}}$, target point $T_{\text{target}}$, and 5 World Queries $\mathbf{Q}_{\text{world}}=[q_0,\dots,q_4]$. Outputs, all in a **single forward pass**: future 5-frame BEV latent features $\mathbf{F}=[f_0,\dots,f_4]$ (each frame $\Delta t=0.5\text{s}$, covering 2 seconds), adaptive CoT text $T_{\text{cot}}$, and future trajectory point tokens $\mathbf{P}_t$. Inference decodes in the order $p(\mathbf{F}\mid\mathcal{X})\cdot p(T_{\text{cot}}\mid\mathcal{X},\mathbf{F})\cdot p(\mathbf{P}_t\mid\mathcal{X},\mathbf{F},T_{\text{cot}})$.

### Key Designs

1. **Driving World Model with Latent BEV Prediction**:

    - **Function**: Enables the VLM to predict BEV latent semantic features for 5 future frames in parallel, shifting from "short video prediction" to "long-horizon semantic prediction."
    - **Mechanism**: Inject 5 learnable BEV-shaped World Queries $q_k\in\mathbb{R}^{h_{\text{bev}}\times w_{\text{bev}}\times d_{\text{bev}}}$, each corresponding to a future time $t+k\Delta t$. Through transformer self-attention, queries interact with historical/current multi-view features to regress latent features $f_k$ in parallel. GT is extracted using DINOv3 on BEV renderings (or semantic segmentation maps): $f_i=\phi_{\text{dino}}(I_i^{bev})$, with MSE loss. The key is "feature-level distillation"—predicting a semantic-rich latent representation rather than pixels or codebook tokens, making 5-frame prediction much cheaper than autoregressive image reconstruction.
    - **Design Motivation**: Codebook/VAE methods waste model capacity on texture details; DINOv3 self-supervised features naturally encode objects, shapes, and semantic relations, aligning with the true needs of driving tasks. The parallel query mechanism avoids the cumulative delay of autoregressive frame-by-frame decoding, making long-horizon prediction feasible. Table 3 shows VAE's single-frame DS is only 27.75, dropping to 14.66 for five frames; DINOv3 achieves 74.79 for single-frame, rising to 86.57 for five frames, proving that only the "semantic + long-horizon" combination is effective.

2. **Adaptive Chain-of-Thought**:

    - **Function**: The model autonomously decides whether to trigger textual CoT reasoning, avoiding the high cost of invoking the LLM every frame.
    - **Mechanism**: After observing the input and predicting $\mathbf{F}$, the model autoregressively generates CoT text: $T_{\text{cot}}=M_{\text{uni}}(\mathbf{I}_t, \dots, \mathbf{Q}_{\text{world}}\mid \mathbf{F})$. If the scene is simple (e.g., following or going straight), it outputs a placeholder token $T_{\text{cot}}^{\emptyset}$ to skip reasoning; for long-tail cases (complex traffic lights, construction zones, emergency vehicles), it generates a full structured reasoning chain. GT data is synthesized using Qwen3-VL-235B in a three-step auto-labeling pipeline (scene complexity assessment → complexity-driven external knowledge retrieval → behavior decision), producing 1.3M Bench2Drive annotations.
    - **Design Motivation**: CoT is powerful but slow; invoking it every frame undermines real-time performance. Table 5 shows that enabling CoT alone (without world model) yields DS=69.87, far below the world model alone (DS=84.52); combining both reaches 86.23—indicating CoT is a complement, not the main driver. "On-demand triggering" preserves commonsense/logical reasoning in long-tail scenarios while keeping extra latency around 4% (Table 6).

3. **Unified Tokenized Training Pipeline**:

    - **Function**: Unifies trajectory waypoints, CoT text, and future features into the VLM's token training framework.
    - **Mechanism**: Discretize the BEV grid into $K$ cells; each waypoint $p_i=(x_i,y_i)$ is quantized to token index $t_i\in\{1,\dots,K\}$, turning trajectory prediction into token classification with CE loss. The total loss is a weighted sum: $L=\lambda_{\text{traj}}L_{\text{traj}}+\lambda_{\text{cot}}L_{\text{cot}}+\lambda_{\text{world}}L_{\text{world}}$, where $L_{\text{traj}}$ and $L_{\text{cot}}$ are CE, $L_{\text{world}}=\mathrm{MSE}(\mathbf{F},\mathbf{F}^{\text{gt}})$. 
    - **Design Motivation**: Training in a unified token space allows the model to treat "trajectory = a special text," leveraging VLM's pretrained alignment. World Query uses feature regression (MSE) to maintain dense representation accuracy. This hybrid loss design avoids the optimization difficulties of decoupled "text head + trajectory head + feature head."

### Loss & Training
Base model: Qwen2.5-VL-3B; 64 H20-96GB GPUs; lr $2\times10^{-5}$, batch 128, trained for 2 epochs on Bench2Drive. Trajectories cover 2 seconds, with one waypoint every 0.5 seconds. CoT text is auto-synthesized by a 235B teacher.

## Key Experimental Results

### Main Results
Bench2Drive (CARLA V2 closed-loop, 220 routes × 44 scenarios) SOTA comparison (same Think2Drive expert protocol):

| Method | Paradigm | DS↑ | SR(%)↑ | Efficiency↑ |
|--------|----------|-----|--------|-------------|
| VAD | E2E | 42.35 | 15.00 | 157.94 |
| DriveTrans | E2E | 63.46 | 35.01 | 100.64 |
| ReasonPlan | VLM | 64.01 | 34.55 | 180.64 |
| ORION | VLM | 77.74 | 54.62 | 151.48 |
| AutoVLA (Prev. SOTA) | VLM | 78.84 | 57.73 | 146.93 |
| **DeepSight w/o CoT** | VLM | **84.52 (+5.68)** | **65.91 (+8.81)** | **198.80** |
| **DeepSight (full)** | VLM | **86.23 (+7.39)** | **71.36 (+13.63)** | **201.71** |

On multi-capability sub-tasks (Table 2), DeepSight averages 70.20% (+15.48), with overtaking at 91.11%, emergency brake at 78.33%, and merging at 60.00%, far surpassing ORION's 54.72%.

### Ablation Study

| Setting | DS↑ | SR↑ | Notes |
|---------|-----|-----|-------|
| Base (no WM, no CoT) | 58.16 | 28.18 | Trajectory decoding only |
| + Adaptive CoT only | 69.87 | 42.27 | Limited CoT improvement |
| + World Model only | 84.52 | 65.91 | WM is main contributor |
| **+ WM + CoT (full)** | **86.23** | **71.36** | Complementary |

World model representation and horizon (Dev 10 routes):

| Representation | Frames | RC↑ | DS↑ |
|----------------|--------|-----|-----|
| VAE | 1 | 47.56 | 27.75 |
| VAE | 5 | 27.02 | **14.66** (long-horizon degrades) |
| DINOv3 | 1 | 90.49 | 74.79 |
| DINOv3 | 5 | 95.95 | **86.57** (+11.78) |

Viewpoint comparison: front-view DS=77.77 vs BEV DS=86.57, BEV improves by 8.8 DS.

### Key Findings
- **World model is the main contributor**: Removing WM drops DS by over 26, while removing CoT only drops DS by 1.71—showing that semantic long-horizon prediction is fundamental for closed-loop driving, with CoT as a supplement.
- **Latent features + long-horizon are both essential**: VAE codebook representation performs worse with 5 frames than single-frame (14.66 vs 27.75), exposing the inadequacy of pixel-level representations for long-horizon modeling; DINOv3, on the other hand, is strong even for single-frame and improves further with more frames—this is the most convincing contrast in the paper.
- **Minimal latency overhead**: Compared to native VLM, DeepSight adds only +3.57% latency, and with CoT a total of +7.69%; explicit FSDrive adds +60.71%—the parallel latent prediction + on-demand CoT design improves efficiency by an order of magnitude.

## Highlights & Insights
- **Shifting from "predicting pixels" to "predicting semantic features" is a core paradigm shift**: Using DINOv3 as GT changes the world model's training target from "rendering future frames" to "aligning future semantics," providing a supervision signal closer to an oracle for decision tasks. This "self-supervised feature as distillation target" approach can be generalized to any downstream task that requires semantic state rather than pixel details (robotics, video QA, etc.).
- **World Queries enable one-shot long-horizon prediction**: Avoids the cumulative error and delay of autoregressive frame-by-frame rolling; essentially, the "time dimension" is moved from the decoding loop to a "parallel query dimension"—this technique has potential in video generation, trajectory planning, and future state prediction.
- **Adaptive CoT's "on-demand triggering" is a pragmatic engineering design**: The model's own placeholder token mechanism controls the CoT switch, saving most unnecessary reasoning; this "model decides whether to think" meta-capability may become standard for future agent systems.
- **Multi-output joint training + joint distribution factorization**: Factorizing $p(\mathbf{P}_t,T_{\text{cot}},\mathbf{F}\mid\mathcal{X})$ into an interpretable causal sequence (world → reasoning → action) aligns with the physical and human driver decision process, and facilitates ablation of each component.

## Limitations & Future Work
- Despite good real-time performance, training still relies on H20 clusters; inference of the 3B VLM remains a challenge for deployment on vehicle SoCs. The authors plan to develop lighter world models in the future.
- Validation is limited to Bench2Drive closed-loop and nuScenes open-loop; no evaluation on real-world tests or extreme conditions (rain, snow, night). The quality of DINOv3 features in rare weather remains an open question.
- The triggering logic of Adaptive CoT is learned by the model, lacking interpretable/controllable switches—misjudging long-tail scenarios as "no CoT needed" could impact safety; a conservative fallback mechanism could be considered.
- The number of World Queries (5) and 2-second coverage are empirical; longer horizons (4s, 8s) are not explored—uncertainty may explode and diminish returns.
- Compared to the PDM-Lite line (e.g., SimLingo DS=85.94), DeepSight matches or exceeds performance using a weaker Think2Drive expert, but cross-expert fair comparison is not fully explored.

## Related Work & Insights
- **vs FSDrive**: FSDrive also uses VLM as a world model but explicitly predicts the next frame image (codebook autoregressive), with +60.71% latency and only single-step prediction; DeepSight uses latent features + 5-frame parallelism, reducing latency to +3.57% and enabling long-horizon.
- **vs ORION**: ORION integrates VQA into trajectory planning, relying on VLM for semantic reasoning without explicit world modeling; DeepSight outperforms ORION by 15.48% on Bench2Drive multi-capability metrics, indicating that reasoning alone, without spatiotemporal prediction, is insufficient for closed-loop scenarios.
- **vs EMMA / SimLingo**: Pure textual CoT encodes all information into language for LLM reasoning, which is inefficient and spatially imprecise; DeepSight delegates spatial reasoning to BEV feature prediction and commonsense reasoning to adaptive CoT, each serving its role.
- **vs HERMES**: HERMES lets LLMs predict LiDAR point clouds and scene understanding, focusing on point cloud generation quality; DeepSight does not rely on LiDAR, uses only surround images, and shifts the generation target from raw modality to compressed semantic latent.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of "latent semantic features + parallel multi-frame World Query + on-demand CoT" is a novel paradigm in end-to-end driving; while each component is inspired by prior work, the assembly is ingenious.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Bench2Drive 220-route closed-loop + nuScenes open-loop + multi-capability sub-tasks + three types of ablation (representation, viewpoint, CoT) + latency analysis, all covered.
- Writing Quality: ⭐⭐⭐⭐ Motivation → Method → Experiment chain is tight; Fig. 1 clearly illustrates paradigm differences; few formulas but clear structure.
- Value: ⭐⭐⭐⭐⭐ Sets new SOTA on both DS/SR for Bench2Drive, minimal latency overhead, open-source code, and direct reference value for industrial E2E driving stacks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ResWorld: Temporal Residual World Model for End-to-End Autonomous Driving](../../ICLR2026/autonomous_driving/resworld_temporal_residual_world_model_for_end-to-end_autonomous_driving.md)
- [\[NeurIPS 2025\] RAW2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving](../../NeurIPS2025/autonomous_driving/raw2drive_reinforcement_learning_with_aligned_world_models_for_end-to-end_autono.md)
- [\[ICCV 2025\] World4Drive: End-to-End Autonomous Driving via Intention-aware Physical Latent World Model](../../ICCV2025/autonomous_driving/world4drive_end-to-end_autonomous_driving_via_intention-aware_physical_latent_wo.md)
- [\[NeurIPS 2025\] Future-Aware End-to-End Driving: Bidirectional Modeling of Trajectory Planning and Scene Evolution](../../NeurIPS2025/autonomous_driving/future-aware_end-to-end_driving_bidirectional_modeling_of_trajectory_planning_an.md)
- [\[NeurIPS 2025\] Prioritizing Perception-Guided Self-Supervision: A New Paradigm for Causal Modeling in End-to-End Autonomous Driving](../../NeurIPS2025/autonomous_driving/prioritizing_perception-guided_self-supervision_a_new_paradigm_for_causal_modeli.md)

</div>

<!-- RELATED:END -->
