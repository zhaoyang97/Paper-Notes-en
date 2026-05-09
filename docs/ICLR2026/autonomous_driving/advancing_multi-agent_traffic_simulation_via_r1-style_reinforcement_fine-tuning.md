---
title: >-
  [Paper Note] SMART-R1: Advancing Multi-agent Traffic Simulation via R1-Style Reinforcement Fine-Tuning
description: >-
  [ICLR 2026][Autonomous Driving][multi-agent traffic simulation] SMART-R1 is the first work to introduce R1-style reinforcement fine-tuning (RFT) into multi-agent traffic simulation. It proposes the Metric-oriented Policy Optimization (MPO) algorithm and an iterative "SFT-RFT-SFT" training strategy, achieving first place on the WOSAC 2025 leaderboard with a Realism Meta score of 0.7858.
tags:
  - ICLR 2026
  - Autonomous Driving
  - multi-agent traffic simulation
  - R1-style
  - reinforcement fine-tuning
  - next-token prediction
  - policy optimization
date: 2026-05-08
content_hash: e42dcaaa0c11ff20
---

# SMART-R1: Advancing Multi-agent Traffic Simulation via R1-Style Reinforcement Fine-Tuning

**Conference**: ICLR 2026
**arXiv**: [2509.23993](https://arxiv.org/abs/2509.23993)
**Code**: N/A
**Area**: Autonomous Driving / Reinforcement Learning
**Keywords**: multi-agent traffic simulation, R1-style, reinforcement fine-tuning, next-token prediction, policy optimization

## TL;DR
SMART-R1 is the first work to introduce R1-style reinforcement fine-tuning (RFT) into multi-agent traffic simulation. It proposes the Metric-oriented Policy Optimization (MPO) algorithm and an iterative "SFT-RFT-SFT" training strategy, achieving first place on the WOSAC 2025 leaderboard with a Realism Meta score of 0.7858.

## Background & Motivation

**Background**: The dominant paradigm in multi-agent traffic simulation is autoregressive modeling based on Next-Token Prediction (NTP) (e.g., SMART), which generates joint agent behaviors by discretizing trajectories into motion tokens. Training typically follows a two-stage pipeline: behavior cloning (BC) pretraining followed by closed-loop SFT (CAT-K rollout).

**Limitations of Prior Work**: (a) The training objectives of BC and SFT (cross-entropy loss) are not directly aligned with the final evaluation metrics (collision rate, off-road rate, and other Realism Meta scores)—these metrics are scalar, sparse, and non-differentiable; (b) covariate shift in autoregressive generation leads to error accumulation in closed-loop simulation; (c) directly applying RL methods such as GRPO or PPO yields poor results, as they rely on comparative sampling or actor-critic architectures.

**Key Challenge**: There exists a gap between the training objective of NTP models (imitating the data distribution) and the evaluation objective (safety and realism metrics), while these evaluation metrics cannot be directly used as differentiable loss functions.

**Goal**: How can non-differentiable evaluation metrics be incorporated into the training of NTP-based traffic simulation models?

**Key Insight**: Drawing inspiration from DeepSeek-R1's multi-stage training strategy, the paper designs an iterative "SFT→RFT→SFT" pipeline with a simplified policy optimization algorithm that directly aligns model training with evaluation metrics.

**Core Idea**: Leverage known reward expectations to simplify advantage estimation, and apply SFT-RFT-SFT iteration to prevent catastrophic forgetting.

## Method

### Overall Architecture
Driving scenarios → tokenization (trajectory → motion tokens; map → map tokens) → Transformer with self-attention and cross-attention → next-token logit prediction. Training consists of four stages: (1) BC pretraining for 64 epochs; (2) closed-loop SFT for 16 epochs (CAT-K rollout); (3) RFT (MPO for metric alignment); (4) a second SFT stage for 16 epochs to recover the data distribution.

### Key Designs

1. **Metric-oriented Policy Optimization (MPO)**:

    - **Function**: Directly uses Realism Meta evaluation metrics as reward signals to optimize the NTP model policy.
    - **Mechanism**: For each scenario, all agent trajectories are generated via full autoregressive rollout, and the official evaluation protocol is applied to compute the Realism Meta score as reward $r$. The advantage function is simplified to $\mathcal{A} = r - \alpha$, where $\alpha = 0.77$ is an empirical threshold approximating the baseline model's average reward. Rollouts exceeding the threshold receive positive reinforcement; those below are penalized. Loss function: $\mathcal{L}_{\text{MPO}} = -(\frac{\pi_\theta}{\bar{\pi}_\theta}\mathcal{A} - \beta D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}])$.
    - **Design Motivation**: GRPO relies on multiple within-group samples to estimate relative advantages, introducing sampling bias; PPO's value model is difficult to optimize; DPO requires preference pairs. In contrast, the expected reward in traffic simulation is relatively predictable (~0.77) and can be used directly as a baseline, eliminating the need for repeated sampling or a value network.
    - **Difference from GRPO**: GRPO normalizes using the within-group mean reward, whereas MPO uses a fixed threshold $\alpha$, yielding a simpler and more stable formulation.

2. **R1-Style "SFT-RFT-SFT" Iterative Training**:

    - **Function**: Performs one round of SFT before and after RFT to prevent catastrophic forgetting.
    - **Mechanism**: The first SFT round (16 epochs) reduces covariate shift; RFT aligns the model with evaluation metrics; the second SFT round (16 epochs) restores adherence to the logged data distribution. The three stages are functionally complementary.
    - **Design Motivation**: SFT followed by RFT alone tends to cause forgetting of the data distribution learned during SFT; two consecutive SFT rounds without RFT underperform the SFT-RFT interleaving scheme. The effectiveness of alternating SFT-RFT has been validated by DeepSeek-R1.

3. **KL Regularization**:

    - **Function**: Incorporates a per-token KL divergence penalty during RFT to prevent the policy from deviating excessively from the reference model.
    - **Mechanism**: An unbiased KL estimator is used: $D_{\text{KL}} = \frac{\pi_{\text{ref}}}{\pi_\theta} - \log\frac{\pi_\theta}{\pi_{\text{ref}}} - 1$, with coefficient $\beta = 0.04$ balancing metric optimization and distribution preservation.
    - **Design Motivation**: A $\beta$ that is too small causes excessive policy drift (losing the BC/SFT prior), while a $\beta$ that is too large suppresses the reward signal.

### Loss & Training
- BC/SFT stages: Standard cross-entropy loss for token distribution alignment
- RFT stage: MPO loss = advantage-weighted policy gradient + KL regularization
- Total epochs match the baseline (64+32); the 32-epoch SFT is restructured as 16 + RFT + 16

## Key Experimental Results

### Main Results
WOSAC 2025 Leaderboard (test set):

| Method | Realism Meta↑ | Kinematics↑ | Interaction↑ | Map↑ | minADE↓ | Params |
|--------|--------------|-------------|-------------|------|---------|--------|
| SMART-base | 0.7725 | 0.472 | 0.804 | 0.912 | 1.393 | 7M |
| SMART-SFT (CAT-K) | 0.7846 | 0.493 | 0.811 | 0.918 | 1.307 | 7M |
| TrajTok | 0.7852 | 0.489 | 0.812 | 0.921 | 1.318 | 10M |
| **SMART-R1** | **0.7858** | **0.494** | 0.811 | 0.920 | **1.289** | 7M |

### Ablation Study

| Training Strategy | Realism Meta↑ | Note |
|------------------|--------------|------|
| BC only | 0.7725 | Baseline |
| SFT | 0.7812 | Improvement from closed-loop SFT |
| SFT → RFT | 0.7848 | Further improvement with RFT |
| SFT → SFT (no RFT) | 0.7809 | Consecutive SFT underperforms SFT+RFT |
| **SFT → RFT → SFT** | **0.7859** | R1-style achieves best performance |

Policy optimization method comparison (after SFT):

| Method | Realism Meta↑ |
|--------|--------------|
| SFT baseline | 0.7812 |
| + PPO | Decrease | Training instability |
| + DPO | Decrease | Sampling bias |
| + GRPO | Decrease | Instability from within-group comparison |
| **+ MPO** | **0.7848** | Benefits from task-specific prior knowledge |

### Key Findings
- RFT yields the most notable improvements on safety-critical metrics (collision rate, off-road rate, traffic light violation rate)—precisely those that BC/SFT cannot directly optimize
- PPO/DPO/GRPO all fail on the traffic simulation task; only MPO is effective—indicating that task-specific characteristics (predictable reward expectation) render general-purpose RL algorithms unsuitable
- $\alpha = 0.77$ is the optimal threshold; higher values yield insufficient positive rewards, while lower values set the bar too low
- $\beta = 0.04$ achieves the best balance for KL regularization

## Highlights & Insights
- The approach of **"using task prior knowledge to simplify RL"** is practically valuable: when the reward distribution is relatively concentrated (unlike the high-variance setting in LLMs), a fixed threshold is more stable than GRPO's within-group comparison. This insight can transfer to other RL settings where rewards are predictable.
- The **SFT-RFT-SFT alternating strategy** validates that the paradigm of "first align with data distribution → then optimize metrics → then restore distribution" is effective beyond the LLM domain, providing a practical template for RLHF in autonomous driving.
- The method is extremely lightweight—it achieves first place on WOSAC using SMART-tiny (7M parameters) without model ensembling or post-processing.

## Limitations & Future Work
- The threshold $\alpha$ in MPO requires manual tuning and depends on the baseline model's average performance, limiting generalizability
- Validation is limited to SMART-tiny (7M); effectiveness on larger models remains unknown
- Whether the Realism Meta metric genuinely reflects driving realism remains debatable (cf. the SPACeR paper)
- The approach can be extended to larger models and multi-round RFT iterations

## Related Work & Insights
- **vs. SMART/CAT-K baseline**: R1-style RFT improves Realism Meta from 0.7846 to 0.7858 without additional parameters
- **vs. RLFTSim (Ahmadi et al., 2025)**: Both apply RL fine-tuning but with different strategies; SMART-R1 achieves better performance (0.7858 vs. 0.7844), likely because MPO is better suited to this task
- **vs. DeepSeek-R1**: SMART-R1 adopts the R1 training paradigm while simplifying advantage estimation, demonstrating that LLM training ideas can transfer across domains to autonomous driving simulation

## Rating
- Novelty: ⭐⭐⭐⭐ First application of R1-style training in traffic simulation; MPO is a concise design tailored to task characteristics
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ First place on WOSAC leaderboard; detailed ablations including optimization method comparisons and hyperparameter sensitivity
- Writing Quality: ⭐⭐⭐⭐ Clear framework, appropriate analogy to LLM training paradigms, thorough experimental analysis
- Value: ⭐⭐⭐⭐ Provides a practical template for RL post-training of traffic simulation models

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] WorldRFT: Latent World Model Planning with Reinforcement Fine-Tuning for Autonomous Driving](../../AAAI2026/autonomous_driving/worldrft_latent_world_model_planning_with_reinforcement_fine-tuning_for_autonomo.md)
- [\[NeurIPS 2025\] AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning](../../NeurIPS2025/autonomous_driving/autovla_a_vision-language-action_model_for_end-to-end_autonomous_driving_with_ad.md)
- [\[NeurIPS 2025\] BayesG: Bayesian Ego-Graph Inference for Networked Multi-Agent Reinforcement Learning](../../NeurIPS2025/autonomous_driving/bayesian_ego-graph_inference_for_networked_multi-agent_reinforcement_learning.md)
- [\[ICCV 2025\] Long-term Traffic Simulation with Interleaved Autoregressive Motion and Scenario Generation](../../ICCV2025/autonomous_driving/long-term_traffic_simulation_with_interleaved_autoregressive_motion_and_scenario.md)
- [\[AAAI 2026\] Backdoor Attacks on Open Vocabulary Object Detectors via Multi-Modal Prompt Tuning](../../AAAI2026/autonomous_driving/backdoor_attacks_on_open_vocabulary_object_detectors_via_multi-modal_prompt_tuni.md)

<!-- RELATED:END -->
