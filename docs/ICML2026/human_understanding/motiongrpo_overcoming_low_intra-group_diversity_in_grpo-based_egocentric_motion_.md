---
title: >-
  [Paper Note] MotionGRPO: Overcoming Low Intra-Group Diversity in GRPO-Based Egocentric Motion Recovery
description: >-
  [ICML 2026][Human Understanding][Full-body motion recovery] MotionGRPO reformulates egocentric full-body motion recovery from head-mounted devices as an MDP over diffusion sampling…
tags:
  - "ICML 2026"
  - "Human Understanding"
  - "Full-body motion recovery"
  - "HMD"
  - "GRPO"
  - "diffusion model"
  - "Perlin noise injection"
date: 2026-05-08
content_hash: c535d2eb2151fd21
---

# MotionGRPO: Overcoming Low Intra-Group Diversity in GRPO-Based Egocentric Motion Recovery

**Conference**: ICML 2026  
**arXiv**: [2605.05680](https://arxiv.org/abs/2605.05680)  
**Code**: https://github.com/3DAgentWorld/MotionGRPO/ (available)  
**Area**: Human Understanding / Egocentric 3D Motion Recovery / Diffusion Models + Reinforcement Learning  
**Keywords**: Full-body motion recovery, HMD, GRPO, diffusion model, Perlin noise injection

## TL;DR
MotionGRPO reformulates egocentric full-body motion recovery from head-mounted devices as an MDP over diffusion sampling, employing GRPO with a "trajectory-conditioned perceptual model + 4 joint-level sub-rewards" hybrid reward for post-training. It identifies the critical bottleneck of "overly strong input conditions leading to nearly identical intra-group samples and vanishing advantage variance," and restores intra-group diversity by injecting Perlin noise into the conditions. On AMASS/RICH, it reduces MPJPE from EgoAllo's 124.985 mm to 114.207 mm.

## Background & Motivation
**Background**: The mainstream for egocentric (HMD) full-body motion recovery is diffusion models based on head SLAM signals (e.g., EgoEgo, EgoAllo), using conditional diffusion to model the "distribution of plausible human motions" and reverse sampling from pure Gaussian noise.

**Limitations of Prior Work**: (1) Diffusion objectives focus on distribution matching, lacking strong constraints on individual joint positions, often resulting in joint misalignment, foot sliding, ground penetration, and jitter; (2) These visual/geometric artifacts cannot be easily penalized within the diffusion framework—early timesteps are still noise, and directly adding joint loss causes instability; (3) Existing RL solutions (e.g., PPO in physics simulators) are unstable and computationally expensive.

**Key Challenge**: One must choose between distribution matching (low accuracy) and RL (unstable/expensive). GRPO appears to be an elegant RL approach without a value-net, but the authors find that "the task's strong conditioning leads to nearly identical group outputs, advantage std≈0, and vanishing gradients," rendering vanilla GRPO ineffective.

**Goal**: (1) Introduce GRPO into diffusion-based motion recovery with meaningful hybrid rewards; (2) Address the new bottleneck of "low intra-group diversity leading to vanishing gradients."

**Key Insight**: The authors treat diffusion sampling as a multi-step MDP, with state $(c,t,x_t)$ and action $x_{t-1}$, and sparse reward only at $t=0$. They propose that "strong conditioning is the root cause of low output diversity, so noise should be injected into the condition"—a reverse perspective.

**Core Idea**: Use SDE-based diffusion sampling with shared initial noise to generate a group of samples, apply a perceptual model and four joint metrics for rewards, and inject temporally smooth Perlin noise into the head condition to restore the intra-group variance required by GRPO.

## Method

### Overall Architecture
The input is the HMD's CPF head trajectory $\mathbf{H}_{cpf}^{1:T}=\{R^{1:T},\tau^{1:T}\}$, which is converted to condition $\mathbf{c}$ via EgoAllo's invariant function $g(\cdot)$. The diffusion backbone uses a transformer to model SMPL-H parameters $\mathbf{M}=\{\Theta,\beta\}$. MotionGRPO performs RL post-training on top of EgoAllo: (1) For each condition, SDE reverse sampling with shared initial noise generates $G$ samples $\{o_i\}$; (2) Two types of rewards are computed—visual-level (learned perceptual scoring model $\phi(\cdot)$) and joint-level (rotation/position/aligned-position/velocity), with group-relative normalization for each sub-reward's advantage, then summed; (3) The diffusion policy is updated using a PPO-style importance ratio; (4) Temporally smooth Perlin noise perturbation $\tilde{\mathbf{H}}=\{R,\tau+\lambda\mathcal{P}(t)\}$ is applied to each condition to increase intra-group variance.

### Key Designs

1. **Hybrid Reward: Learned Perceptual Model + 4 Joint Sub-Rewards**:

    - **Function**: Simultaneously constrains "global visual naturalness" and "local joint accuracy," covering non-differentiable visual artifacts within the diffusion framework.
    - **Mechanism**: The visual-level uses a trajectory-conditioned perceptual model $\phi$ (spatial-attention + temporal-attention transformer), taking SMPL-H skeleton and head trajectory as input and outputting a plausibility score. This model is trained online with InfoNCE contrastive loss: positives are (GT motion, head), hard negatives are pseudo-samples generated by the current policy in the last few sampling steps, with temperature $\delta=0.07$. Visual reward $\mathcal{R}_{vis}=\exp(\omega_{vis}\cdot s)$. The joint-level includes four terms: $\mathcal{R}_{rot}$ (local rotation L1), $\mathcal{R}_{pos}$ (global position L2), $\mathcal{R}'_{pos}$ (per-frame Procrustes-aligned position L2), $\mathcal{R}_{vel}$ (velocity difference L2), all compressed to (0,1] via $\exp(-\omega\cdot\text{err})$. Each is group-relative normalized to obtain sub-advantages, then summed: $\hat A_i=\sum_k \hat A_{i,k}$.
    - **Design Motivation**: The perceptual model captures artifacts such as foot skating, jitter, and penetration that standard losses miss; joint metrics align with GT. Both are optimized together via GRPO's non-differentiable optimization.

2. **Online Hard Negatives + Contrastive Training for the Perceptual Model**:

    - **Function**: Makes the perceptual scorer sensitive to "near-GT but flawed" motions, preventing it from degenerating into a simple "GT vs random noise" classifier.
    - **Mechanism**: Negatives are not manually noised but sampled in real-time from the last 3 sampling timesteps of the current policy—these are structurally close to GT but contain typical policy artifacts; InfoNCE loss $\mathcal{L}_{NCE}=-\mathbb{E}\log\frac{\exp(\phi(J^+|H^+)/\delta)}{\exp(\phi(J^+|H^+)/\delta)+\sum_i\exp(\phi(J_i^-|H_i^-)/\delta)}$.
    - **Design Motivation**: A statically trained reward model is vulnerable to policy exploitation (reward hacking); evolving negatives with the policy ensures meaningful gradients, embodying the "online preference model" approach for motion recovery.

3. **Perlin Noise Injection into Head Condition to Break Low Intra-Group Diversity**:

    - **Function**: Injects sufficient variance into the GRPO sample group without disrupting the physical smoothness of the head signal, preventing division by zero in advantage normalization.
    - **Mechanism**: Motion recovery is dominated by strong condition $\mathbf{c}$, leading to nearly identical outputs; the advantage formula $\hat A_i=(\mathcal{R}_i-\mu)/\sigma$ becomes numerically unstable as $\sigma\to 0$, causing vanishing gradients. The solution: perturb head translation with temporally continuous Perlin noise $\mathcal{P}(t)$: $\tilde{\mathbf{H}}=\{R,\tau+\lambda\mathcal{P}(t)\}$, then pass through $g(\cdot)$ to obtain the perturbed condition $\mathbf{c}$. This exposes the policy to slightly out-of-distribution inputs during sampling, naturally increasing output variance.
    - **Design Motivation**: Directly adding Gaussian white noise would disrupt temporal smoothness and introduce high-frequency jitter, conflicting with head motion priors; Perlin noise is inherently smooth and spectrally controllable, expanding "neighboring conditions" without violating physical plausibility.

### Loss & Training
The GRPO objective is $\mathcal{J}_{GRPO}(\theta)=\mathbb{E}\left[\frac{1}{G}\sum_i\frac{1}{n}\sum_t \frac{\pi_\theta(o_{i,t}|\mathbf{c})}{\pi_{old}(o_{i,t}|\mathbf{c})}\hat A_i\right]$, omitting clip and KL terms. The total reward is $\mathcal{R}_{total}=\mathcal{R}_{vis}+\mathcal{R}_{joint}$ ($\mathcal{R}_{joint}=\mathcal{R}_{rot}+\mathcal{R}_{pos}+\mathcal{R}'_{pos}+\mathcal{R}_{vel}$). Algorithm 1 outlines the outer loop: sample batch → copy old policy → inject Perlin noise perturbation → SDE sample $G$ group members with shared initial noise → compute $\mu/\sigma$ for each sub-reward → calculate advantage → perform importance-weighted updates across $n$ sampling steps. The perceptual model and policy are updated alternately/in parallel.

## Key Experimental Results

### Main Results

| Dataset | Method | MPJPE↓(mm) | PA-MPJPE↓(mm) | MPJVE↓(mm) | MPJRE↓(°) | Jitter↓ | GP↓(m) | FS↓(m) |
|--------|------|-----------|---------------|-------------|------------|---------|--------|--------|
| AMASS | EgoEgo | 177.231 | 152.125 | 588.661 | 9.457 | 2.643 | 1.331 | 1.241 |
| AMASS | EgoAllo | 124.985 | 103.958 | 553.221 | 8.777 | 2.394 | 1.143 | 1.290 |
| AMASS | EgoAllo$^\aleph$ (with test-time optimization) | 121.651 | 101.034 | 483.471 | 8.728 | 1.455 | 1.099 | 0.479 |
| AMASS | **MotionGRPO** | **114.207** | **95.512** | 531.217 | **8.413** | 2.000 | **0.901** | 1.169 |
| AMASS | **MotionGRPO$^\aleph$** | **111.776** | **93.702** | **461.702** | **8.330** | **1.309** | 0.963 | **0.399** |
| RICH | EgoAllo | 192.686 | 172.724 | 506.992 | 12.734 | 4.135 | 4.145 | 1.094 |
| RICH | **MotionGRPO$^\aleph$** | **184.992** | **167.032** | **378.423** | **11.886** | **1.614** | **3.156** | **0.199** |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Vanilla GRPO (no Perlin noise) | Intra-group diversity nearly 0, advantage std≈0, loss does not decrease | Directly validates the "low intra-group diversity → vanishing gradient" hypothesis |
| Visual reward only / Joint reward only | Visual reward improves appearance but limited MPJPE gain / vice versa; combination is optimal | Hybrid reward is necessary |
| Online hard negatives vs static noise negatives | Online hard negatives make the perceptual model more sensitive to policy-typical artifacts, stabilizing long-term reward | Prevents reward hacking |
| Perlin noise scale $\lambda$ | Too small: insufficient diversity; too large: disrupts priors; optimal value exists | Shows Perlin is a "just right" perturbation |

### Key Findings
- "Low intra-group diversity → vanishing advantage variance" is an almost inevitable deadlock when transferring GRPO from generation to reconstruction tasks; this work is among the first to identify and address it.
- The combination of a learned perceptual model and explicit joint metrics suppresses artifacts like jitter, foot skating, and penetration more effectively than hand-crafted losses alone.
- Test-time optimization (marked $\aleph$) further reduces Jitter from 2.0 to 1.3 and FS from 1.17 to 0.40, indicating that MotionGRPO post-training and EgoAllo test-time refinement are complementary.

## Highlights & Insights
- Provides a thorough mathematical explanation for "why GRPO fails on reconstruction tasks"—the denominator $\sigma\to 0$ in the advantage formula; this argument is clear and generalizable to any RL + strong conditioning task.
- Using Perlin noise instead of Gaussian is a highly domain-aware choice: preserving temporal smoothness maintains the physical prior of head motion, avoiding contradictory training signals.
- The online contrastive reward model addresses "reward hacking" via a self-play approach, which can be transferred to other diffusion tasks with challenging reward modeling, such as video generation and TTS.
- The overall design—RL post-training atop diffusion pretraining, with dual perceptual and geometric rewards—serves as a paradigm for "diffusion + RLHF" in structured tasks.

## Limitations & Future Work
- Evaluation is mainly on synthetic device pose (AMASS) and semi-real data (RICH); large-scale validation on real Project Aria long sequences, multi-person, and challenging lighting scenarios is lacking.
- The Perlin noise scale $\lambda$ is a manually tuned hyperparameter without adaptive adjustment; too large may break physical priors, too small may not restore sufficient diversity.
- The stability of online contrastive training for the perceptual model depends on policy evolution speed; if the policy evolves too quickly, negative sample quality may lag.
- Currently only head trajectory is used, without leveraging optional egocentric images or hand observations; the potential for multimodal fusion remains untapped.
- While FS is reduced to 0.4 m, there is still room for improvement before achieving truly imperceptible foot sliding.

## Related Work & Insights
- **vs EgoAllo (Yi et al., 2025)**: MotionGRPO uses EgoAllo as the base policy, achieving about 8% MPJPE reduction with only RL post-training, indicating substantial untapped potential in EgoAllo's pretrained prior.
- **vs PPO in physics simulators**: Traditional RL in physics simulation is slow and unstable; GRPO eliminates the value-net and, combined with diffusion SDE sampling, trains directly in motion space with much higher efficiency.
- **vs DDPO / DPO for image generation**: Both convert diffusion to MDP + RL, but image generation is inherently diverse; this work's core contribution is "restoring diversity," highlighting the fundamental difference between reconstruction and generation.

## Rating
- Novelty: ⭐⭐⭐⭐ Among the first to introduce GRPO to diffusion-based motion recovery, with genuine diagnosis and solution for "low intra-group diversity"; the combination of Perlin injection and online perceptual reward is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Main comparisons on AMASS/RICH, ADT real tests, multiple ablations, but lacks direct comparison with PPO/physics simulators, DPO, and other RL approaches.
- Writing Quality: ⭐⭐⭐⭐ Clearly explains "why GRPO fails" via the advantage formula, with comprehensive pipeline diagrams and algorithm pseudocode; reward formulas are numerous, posing some reading challenge.
- Value: ⭐⭐⭐⭐ Directly valuable for VR/AR full-body tracking and the diffusion + RL post-training paradigm; open-source code further lowers the barrier to reproduction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Fish2Mesh Transformer: 3D Human Mesh Recovery from Egocentric Vision](../../ICCV2025/human_understanding/fish2mesh_transformer_3d_human_mesh_recovery_from_egocentric_vision.md)
- [\[CVPR 2026\] EgoPoseFormer v2: Accurate Egocentric Human Motion Estimation for AR/VR](../../CVPR2026/human_understanding/egoposeformer_v2_accurate_egocentric_human_motion_estimation_for_arvr.md)
- [\[CVPR 2025\] HumanMM: Global Human Motion Recovery from Multi-shot Videos](../../CVPR2025/human_understanding/humanmm_global_human_motion_recovery_from_multi-shot_videos.md)
- [\[NeurIPS 2025\] Some Optimizers are More Equal: Understanding the Role of Optimizers in Group Fairness](../../NeurIPS2025/human_understanding/some_optimizers_are_more_equal_understanding_the_role_of_optimizers_in_group_fai.md)
- [\[CVPR 2026\] E-3DPSM: A State Machine for Event-Based Egocentric 3D Human Pose Estimation](../../CVPR2026/human_understanding/e-3dpsm_a_state_machine_for_event-based_egocentric_3d_human_pose_estimation.md)

</div>

<!-- RELATED:END -->
