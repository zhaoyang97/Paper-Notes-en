---
title: >-
  [Paper Note] MotionGRPO: Overcoming Low Intra-Group Diversity in GRPO-Based Egocentric Motion Recovery
description: >-
  [ICML 2026][Human Understanding][Full-body Motion Recovery] MotionGRPO formulates head-mounted device (HMD) egocentric full-body motion recovery as an MDP over diffusion sampling. It utilizes GRPO for post-training with…
tags:
  - "ICML 2026"
  - "Human Understanding"
  - "Full-body Motion Recovery"
  - "HMD"
  - "GRPO"
  - "Diffusion Models"
  - "Perlin Noise Injection"
date: 2026-05-08
content_hash: f6dd22674f017d33
---

# MotionGRPO: Overcoming Low Intra-Group Diversity in GRPO-Based Egocentric Motion Recovery

**Conference**: ICML 2026  
**arXiv**: [2605.05680](https://arxiv.org/abs/2605.05680)  
**Code**: https://github.com/3DAgentWorld/MotionGRPO/ (Available)  
**Area**: Human Understanding / Egocentric 3D Motion Recovery / Diffusion Models + Reinforcement Learning  
**Keywords**: Full-body Motion Recovery, HMD, GRPO, Diffusion Models, Perlin Noise Injection

## TL;DR
MotionGRPO formulates head-mounted device (HMD) egocentric full-body motion recovery as an MDP over diffusion sampling. It utilizes GRPO for post-training with a hybrid reward consisting of a "trajectory-conditioned perception model + 4 joint-level sub-rewards." Crucially, it identifies that strong input conditions lead to nearly identical intra-group samples, causing vanishing advantage variance. By injecting Perlin noise into the conditions to restore intra-group diversity, the method reduces MPJPE from EgoAllo's 124.985 mm to 114.207 mm on AMASS/RICH.

## Background & Motivation
**Background**: Mainstream egocentric (HMD) full-body motion recovery methods are diffusion models based on head SLAM signals (e.g., EgoEgo, EgoAllo). These use conditional diffusion to model the "distribution of possible human motions" and perform reverse sampling from pure Gaussian noise.

**Limitations of Prior Work**: (1) The diffusion objective is essentially distribution matching, which lacks strong constraints on individual joint positions, often resulting in joint offsets, foot skating, ground penetration, and jitter. (2) These visual/geometric artifacts cannot be easily addressed by adding simple losses within the diffusion framework; adding joint losses in early timesteps (dominated by noise) leads to instability. (3) Existing RL solutions (e.g., PPO in physical simulators) are unstable and computationally expensive.

**Key Challenge**: A trade-off exists between distribution matching (poor precision) and RL (unstable/expensive). While GRPO appears as an elegant RL variant that eliminates the value-net, the authors found that "strong conditions in motion recovery tasks $\to$ nearly identical group outputs $\to$ advantage $std \approx 0 \to$ vanishing gradients," causing vanilla GRPO to fail.

**Goal**: (1) Introduce GRPO into diffusion-based motion recovery with a meaningful hybrid reward design. (2) Solve the new bottleneck of "low intra-group diversity $\to$ vanishing gradients."

**Key Insight**: Diffusion sampling is viewed as a multi-step MDP where the state is $(c, t, x_t)$ and the action is $x_{t-1}$, with a sparse reward provided only at $t=0$. The authors proposed that strong conditions are the root cause of low output diversity, leading to the counter-intuitive idea of injecting noise into the conditions.

**Core Idea**: Use SDE-based diffusion sampling with shared initial noise to generate a group of samples. Rewards are computed via a learned perception model and 4 joint metrics. Spatially and temporally smooth Perlin noise is injected into the head trajectory conditions to restore the intra-group variance required by GRPO.

## Method

### Overall Architecture
The input consists of HMD CPF head trajectories $\mathbf{H}_{cpf}^{1:T}=\{R^{1:T},\tau^{1:T}\}$, transformed into a condition $\mathbf{c}$ via EgoAllo's invariant function $g(\cdot)$. The diffusion backbone uses a Transformer to learn the SMPL-H representation $\mathbf{M}=\{\Theta,\beta\}$. MotionGRPO performs RL post-training on top of EgoAllo: (1) SDE reverse sampling with shared initial noise generates $G$ samples $\{o_i\}$ for each condition. (2) Two types of rewards are calculated: visual-level (learned perception scoring model $\phi(\cdot)$) and joint-level (rotation, position, aligned-position, and velocity). Sub-reward advantages are calculated using GRPO's group-relative normalization and summed. (3) The diffusion policy is updated using PPO-style importance ratios. (4) Spatially and temporally smooth Perlin noise $\tilde{\mathbf{H}}=\{R,\tau+\lambda\mathcal{P}(t)\}$ is injected into head conditions to increase intra-group variance.

### Key Designs

1.  **Hybrid Reward: Learned Perception Model + 4 Joint Sub-rewards**:
    *   **Function**: Constrains both "global visual naturalness" and "local joint precision," addressing non-differentiable visual artifacts within the diffusion framework.
    *   **Mechanism**: The visual level uses a trajectory-conditioned perception model $\phi$ (spatial-temporal attention Transformer). It takes the SMPL-H skeleton and head trajectory as input and outputs a plausibility score. This model is trained online using InfoNCE contrastive learning, where positive samples are (GT motion, head) and hard negatives are pseudo-samples generated by the current policy during final sampling steps (temperature $\delta=0.07$). Visual reward is $\mathcal{R}_{vis}=\exp(\omega_{vis}\cdot s)$. The joint level includes four terms: $\mathcal{R}_{rot}$ (local rotation L1), $\mathcal{R}_{pos}$ (global position L2), $\mathcal{R}'_{pos}$ (per-frame Procrustes aligned position L2), and $\mathcal{R}_{vel}$ (velocity L2), all mapped to $(0,1]$ via $\exp(-\omega\cdot\text{err})$. Each term undergoes independent group-relative normalization to obtain sub-advantages: $\hat A_i=\sum_k \hat A_{i,k}$.
    *   **Design Motivation**: The perception model captures artifacts like "foot skating, jitter, and penetration" missed by standard losses, while joint metrics align with GT. Both are optimized simultaneously via GRPO’s non-differentiable optimization.

2.  **Online Hard Negatives + Contrastive Reward Training**:
    *   **Function**: Makes the perception score sensitive to motions that are "close to GT but flawed," preventing the model from degrading into a simple "GT vs. random noise" classifier.
    *   **Mechanism**: Negatives are not manually noisy samples but are sampled in real-time from the current policy's last 3 sampling timesteps. These samples are structurally similar to GT but carry typical policy flaws. The InfoNCE loss is: $$\mathcal{L}_{NCE}=-\mathbb{E}\log\frac{\exp(\phi(J^+|H^+)/\delta)}{\exp(\phi(J^+|H^+)/\delta)+\sum_i\exp(\phi(J_i^-|H_i^-)/\delta)}$$.
    *   **Design Motivation**: Static reward models are susceptible to reward hacking. Evolving negatives with the policy provides continuous meaningful gradients, representing an "online preference model" implementation in motion recovery.

3.  **Perlin Noise Injection in Head Conditions**:
    *   **Function**: Injects sufficient variance into GRPO sample groups without destroying the physical smoothness of head signals, ensuring advantage normalization does not involve division by zero.
    *   **Mechanism**: Motion recovery is dominated by the strong condition $\mathbf{c}$, resulting in nearly identical outputs. The advantage formula $\hat A_i=(\mathcal{R}_i-\mu)/\sigma$ becomes numerically unstable as $\sigma\to 0$, causing vanishing gradients. Solution: Perturb head translation with temporally continuous Perlin noise $\mathcal{P}(t)$: $\tilde{\mathbf{H}}=\{R,\tau+\lambda\mathcal{P}(t)\}$, then pass through $g(\cdot)$ to get the perturbed condition $\mathbf{c}$. During sampling, the policy faces slightly out-of-distribution inputs, naturally restoring output variance.
    *   **Design Motivation**: Gaussian white noise would break temporal smoothness and introduce high-frequency jitter, conflicting with head motion priors. Perlin noise is naturally smooth and spectrally controllable, expanding to "neighboring conditions" while maintaining physical plausibility.

### Loss & Training
The GRPO objective is $\mathcal{J}_{GRPO}(\theta)=\mathbb{E}\left[\frac{1}{G}\sum_i\frac{1}{n}\sum_t \frac{\pi_\theta(o_{i,t}|\mathbf{c})}{\pi_{old}(o_{i,t}|\mathbf{c})}\hat A_i\right]$ (clipping and KL omitted). The total reward is $\mathcal{R}_{total}=\mathcal{R}_{vis}+\mathcal{R}_{joint}$. The algorithm follows an outer loop: sample batch $\to$ copy old policy $\to$ inject Perlin noise $\to$ SDE sampling with shared noise for $G$ samples $\to$ compute $\mu/\sigma$ for sub-rewards $\to$ compute advantage $\to$ importance-weighted update across $n$ sampling steps. The perception model and policy are updated alternatingly or in parallel.

## Key Experimental Results

### Main Results

| Dataset | Method | MPJPE↓(mm) | PA-MPJPE↓(mm) | MPJVE↓(mm) | MPJRE↓(°) | Jitter↓ | GP↓(m) | FS↓(m) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| AMASS | EgoEgo | 177.231 | 152.125 | 588.661 | 9.457 | 2.643 | 1.331 | 1.241 |
| AMASS | EgoAllo | 124.985 | 103.958 | 553.221 | 8.777 | 2.394 | 1.143 | 1.290 |
| AMASS | EgoAllo$^\aleph$ (w/ TTO) | 121.651 | 101.034 | 483.471 | 8.728 | 1.455 | 1.099 | 0.479 |
| AMASS | **MotionGRPO** | **114.207** | **95.512** | 531.217 | **8.413** | 2.000 | **0.901** | 1.169 |
| AMASS | **MotionGRPO$^\aleph$** | **111.776** | **93.702** | **461.702** | **8.330** | **1.309** | 0.963 | **0.399** |
| RICH | EgoAllo | 192.686 | 172.724 | 506.992 | 12.734 | 4.135 | 4.145 | 1.094 |
| RICH | **MotionGRPO$^\aleph$** | **184.992** | **167.032** | **378.423** | **11.886** | **1.614** | **3.156** | **0.199** |

### Ablation Study

| Configuration | Key Metrics | Explanation |
| :--- | :--- | :--- |
| Vanilla GRPO (No Perlin) | Intra-group div $\approx$ 0, Adv std $\approx$ 0, Loss plateaued | Validates the "low diversity $\to$ vanishing gradient" hypothesis. |
| Visual Reward Only / Joint Reward Only | Good visual but limited MPJPE gain / vice versa | Necessity of hybrid rewards. |
| Online Negatives vs. Static Noise | Online negatives make model sensitive to typical flaws | Prevents reward hacking. |
| Perlin Noise Scale $\lambda$ | Small $\lambda$: low diversity; Large $\lambda$: breaks prior | Optimal value exists for "just enough" perturbation. |

### Key Findings
*   "Low intra-group diversity $\to$ vanishing advantage variance" is an inherent bottleneck when applying GRPO to reconstruction tasks (vs. generation); this work is among the first to formally address this.
*   The combination of a learned perception model and explicit joint metrics suppresses artifacts like jitter and foot skating better than manual loss terms.
*   Test-time optimization (marked with $\aleph$) further reduces Jitter (2.0 to 1.3) and FS (1.17 to 0.40), showing that MotionGRPO post-training and EgoAllo refinement are complementary.

## Highlights & Insights
*   Provides a clear mathematical justification for GRPO's failure in reconstruction tasks ($\sigma \to 0$), an argument applicable to any RL + strong constraint task.
*   The choice of Perlin noise over Gaussian is domain-aware: preserving temporal smoothness maintains the physical prior of head motion, avoiding self-contradictory training signals.
*   The online contrastive reward model addresses "reward hacking" through self-play, a strategy transferable to other hard-to-evaluate diffusion tasks like video generation or TTS.
*   The overall design follows a paradigm of RL post-training on diffusion pre-training with dual-track (perceptual + geometric) rewards.

## Limitations & Future Work
*   Evaluation is primarily on synthetic device poses (AMASS) and semi-real data (RICH); large-scale verification on real Project Aria data (long-sequence, multi-person, lighting changes) is pending.
*   The Perlin noise scale $\lambda$ is a manually tuned hyperparameter; an adaptive mechanism is lacking.
*   Stability of online contrastive training depends on the policy evolution speed; if the policy evolves too fast, negative sample quality might lag.
*   Only head trajectories are used; the potential for multi-modal fusion with egocentric images or hand observations remains untapped.
*   Visual metrics like FS (reduced to 0.4 m) still have room for improvement to achieve a truly "skate-free" experience.

## Related Work & Insights
*   **vs. EgoAllo (Yi et al., 2025)**: MotionGRPO uses EgoAllo as a base policy and achieves ~8% MPJPE reduction via RL post-training alone, proving untapped potential in EgoAllo's pre-trained priors.
*   **vs. PPO in Physics Simulators**: Traditional RL in physics sims is slow and unstable; GRPO eliminates the value-net and trains directly in the action space via diffusion SDE sampling, offering higher efficiency.
*   **vs. DDPO / DPO for Image Generation**: While both convert diffusion to MDP + RL, image generation is naturally diverse. This work's core contribution is "restoring diversity," highlighting the fundamental difference between reconstruction and generation.

## Rating
*   Novelty: ⭐⭐⭐⭐ One of the first to bring GRPO to diffusion motion recovery and solve the "low diversity" bottleneck; Perlin injection and online rewards are clever.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Main comparisons on AMASS/RICH plus ADT testing and multiple ablations, though direct comparisons with PPO or other RL routes are missing.
*   Writing Quality: ⭐⭐⭐⭐ Clearly explains "why GRPO fails" with the advantage formula; pipeline diagrams and pseudo-code are comprehensive.
*   Value: ⭐⭐⭐⭐ High relevance for VR/AR full-body tracking and the "Diffusion + RLHF" paradigm in structured tasks; open-source code aids reproducibility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] EgoPoseFormer v2: Accurate Egocentric Human Motion Estimation for AR/VR](../../CVPR2026/human_understanding/egoposeformer_v2_accurate_egocentric_human_motion_estimation_for_arvr.md)
- [\[CVPR 2026\] E-3DPSM: A State Machine for Event-Based Egocentric 3D Human Pose Estimation](../../CVPR2026/human_understanding/e-3dpsm_a_state_machine_for_event-based_egocentric_3d_human_pose_estimation.md)
- [\[NeurIPS 2025\] Some Optimizers are More Equal: Understanding the Role of Optimizers in Group Fairness](../../NeurIPS2025/human_understanding/some_optimizers_are_more_equal_understanding_the_role_of_optimizers_in_group_fai.md)
- [\[CVPR 2026\] UniDex: A Robot Foundation Suite for Universal Dexterous Hand Control from Egocentric Human Videos](../../CVPR2026/human_understanding/unidex_a_robot_foundation_suite_for_universal_dexterous_hand_control_from_egocen.md)
- [\[ICCV 2025\] EgoAgent: A Joint Predictive Agent Model in Egocentric Worlds](../../ICCV2025/human_understanding/egoagent_a_joint_predictive_agent_model_in_egocentric_worlds.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[CVPR 2025\] HumanMM: Global Human Motion Recovery from Multi-shot Videos](../../CVPR2025/human_understanding/humanmm_global_human_motion_recovery_from_multi-shot_videos.md)
- [\[CVPR 2026\] EgoPoseFormer v2: Accurate Egocentric Human Motion Estimation for AR/VR](../../CVPR2026/human_understanding/egoposeformer_v2_accurate_egocentric_human_motion_estimation_for_arvr.md)
- [\[CVPR 2025\] FreeUV: Ground-Truth-Free Realistic Facial UV Texture Recovery via Cross-Assembly](../../CVPR2025/human_understanding/freeuv_ground-truth-free_realistic_facial_uv_texture_recovery_via_cross-assembly.md)
- [\[CVPR 2026\] E-3DPSM: A State Machine for Event-Based Egocentric 3D Human Pose Estimation](../../CVPR2026/human_understanding/e-3dpsm_a_state_machine_for_event-based_egocentric_3d_human_pose_estimation.md)
- [\[NeurIPS 2025\] Some Optimizers are More Equal: Understanding the Role of Optimizers in Group Fairness](../../NeurIPS2025/human_understanding/some_optimizers_are_more_equal_understanding_the_role_of_optimizers_in_group_fai.md)

</div>

<!-- RELATED:END -->
