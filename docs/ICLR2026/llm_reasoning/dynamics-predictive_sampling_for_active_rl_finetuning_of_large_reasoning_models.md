---
title: >-
  [Paper Note] Dynamics-Predictive Sampling for Active RL Finetuning of Large Reasoning Models
description: >-
  [ICLR 2026][LLM Reasoning][RL finetuning] This paper models each prompt's solve progress during RL finetuning as a latent Markov dynamical system…
tags:
  - "ICLR 2026"
  - "LLM Reasoning"
  - "RL finetuning"
  - "prompt sampling"
  - "hidden Markov model"
  - "large reasoning models"
  - "online Bayesian inference"
date: 2026-05-08
content_hash: e821e277deb192db
---

# Dynamics-Predictive Sampling for Active RL Finetuning of Large Reasoning Models

**Conference**: ICLR 2026
**arXiv**: [2603.10887](https://arxiv.org/abs/2603.10887)  
**Code**: [github.com/maoyixiu/DPS](https://github.com/maoyixiu/DPS)  
**Area**: LLM Reasoning / RL Finetuning
**Keywords**: RL finetuning, prompt sampling, hidden Markov model, large reasoning models, online Bayesian inference

## TL;DR

This paper models each prompt's solve progress during RL finetuning as a latent Markov dynamical system, and employs lightweight Bayesian inference to online-predict prompt solve states. By prioritizing "partially solved" prompts for sampling, the method achieves comparable or superior reasoning performance to Dynamic Sampling (DS) using fewer than 30% of DS's rollouts.

## Background & Motivation

**Background**: RL finetuning, exemplified by GRPO, has become a core technique for improving LLM reasoning. Large reasoning models (LRMs) such as DeepSeek-R1 and OpenAI o1 have achieved breakthroughs in mathematical competition, code generation, and logical inference through RL finetuning. However, the effectiveness of RL finetuning depends critically on the quality of training data selection—not all prompts contribute equally to policy optimization.

**Limitations of Prior Work**: GRPO computes advantages via within-group normalization: $\hat{A}_i^\tau = \frac{r(\tau, y_i^\tau) - \text{mean}}{\text{std}}$. When all $k$ responses to a prompt are either all correct or all incorrect, $\text{std} = 0$, the advantage function degenerates, and the gradient signal vanishes. Only "partially solved" prompts (some correct, some incorrect) provide effective optimization signals.

**Key Challenge**: The current state-of-the-art online prompt selection method, Dynamic Sampling (DS), expands the candidate batch (typically 3–4× the final batch size) and performs rollouts one by one to filter effective prompts. While this accurately identifies partially solved samples, rollout computation is extremely expensive—generating long chain-of-thought responses often costs more than the finetuning itself. Another approach, History Resampling (HR), filters fully solved prompts at the epoch level, but its "absorbing state" assumption is overly rigid and merely excludes solved prompts rather than actively identifying partially solved ones.

**Goal**: How can one online-predict which prompts are currently "partially solved" without expensive rollouts, thereby achieving training effectiveness comparable to or better than DS at negligible additional cost?

**Key Insight**: The authors treat each prompt's solve progress as a dynamical system evolving over time—as the model is updated, a prompt's solve state (unsolved → partially solved → solved) undergoes transitions. This evolution is modeled as an HMM: latent states represent solve degrees, and observations are intermittent rollout outcomes. Bayesian inference over historical rollout data can then predict future solve states without requiring actual rollouts.

**Core Idea**: Model prompt solve progress as an HMM dynamical system; use online Bayesian inference to predict solve states; select prompts most likely to be "partially solved" before rollout, enabling rollout-free adaptive sampling.

## Method

### Overall Architecture

The core pipeline of DPS consists of two phases at each training step $t$: **(1) Prediction & Selection**—using the prior $\mu_t^{\tau,\text{prior}}$ inferred from the previous step, rank all prompts by their predicted probability of being in the "partially solved" state (State 2), and select the Top-$B$ prompts to form the training batch $\mathcal{B}_t$; **(2) Inference & Update**—execute rollouts on selected prompts to obtain observations, apply Bayes' rule to update state posteriors and transition matrix posteriors, then propagate forward to generate the prior predictions for the next step.

### Key Designs

1. **Three-State Hidden Markov Modeling**:

    - Function: Formally represents each prompt's solve progress as a tractable dynamical system.
    - Mechanism: Define latent states $z_t^\tau \in \{1, 2, 3\}$ corresponding to fully unsolved (all $k$ responses incorrect), partially solved (some correct, some incorrect), and fully solved (all correct), respectively. The initial prior is uniform: $\mu_1^{\text{prior}} = [1/3, 1/3, 1/3]$. State transitions are described by a column-stochastic matrix $\Phi \in \mathbb{R}^{3 \times 3}$. The observation model is degenerate: if a prompt is selected for rollout, its state is exactly observed; otherwise, no observation is made.
    - Design Motivation: The three-state partition aligns with the gradient signal theory of GRPO (only State 2 yields nonzero advantage), while keeping the model parsimonious. Ablation studies confirm that finer or coarser partitions both reduce prediction accuracy.

2. **Online Bayesian Inference Three-Step Pipeline**:

    - Function: Real-time update of state estimates and the transition model at each training step.
    - Mechanism: (i) **Observation update**—if a prompt is selected for rollout, apply Bayes' rule to update the prior $\mu_t^{\text{prior}}$ to the posterior $\mu_t^{\text{post}}$; under the degenerate emission model, the posterior collapses directly to the observed state. (ii) **Transition update**—exploiting Dirichlet–Categorical conjugacy, incrementally update Dirichlet parameters $\alpha_t$ using the posterior transition pseudo-counts $\xi_t(i,j) = \mathbb{P}(z_{t-1}=j, z_t=i \mid y_{1:t})$. (iii) **Next-step prediction**—$\mu_{t+1}^{\text{prior}} = \Phi_t \mu_t^{\text{post}}$, propagating the posterior through the transition matrix to yield the next-step prior.
    - Design Motivation: The classical HMM forward–backward algorithm requires a complete trajectory and cannot be used online. DPS's online inference depends only on the current-step observation and the previous-step posterior; all operations are $3 \times 3$ matrix computations with negligible overhead.

3. **Non-Stationary Exponential Decay Mechanism**:

    - Function: Adapts the transition model to continuously changing solve dynamics as model learning progresses.
    - Mechanism: Introduce a decay factor $\lambda \in (0,1)$; the update rule becomes $\alpha_t = \lambda \cdot \alpha_{t-1} + (1-\lambda) \cdot \alpha_0 + \xi_t$. Smaller $\lambda$ causes the model to forget old statistics more quickly and adapt to new dynamics. This mechanism also **implicitly provides exploration**: the posterior of long-unsampled prompts gradually decays toward the uniform distribution, causing them to be naturally revisited when no clearly high-information prompts are available.
    - Design Motivation: The learning process of LRMs is highly non-stationary—as the policy updates, solve probabilities for prompts continuously change. The stationarity assumption of standard HMMs is violated here; exponential decay is a lightweight yet effective non-stationary extension.

### Prompt Sampling Strategy

All prompts are ranked by $\mu_t^{\tau,\text{prior}}(2)$ (predicted probability of being partially solved), and the Top-$B$ are directly selected to form the training batch. This is a pure exploitation strategy, but the non-stationary decay mechanism provides implicit exploration—unsampled prompts' predictive distributions gradually drift toward uniform, naturally bringing them back into consideration. This design avoids explicit exploration–exploitation trade-off hyperparameter tuning.

### Training Update

DPS is orthogonal to the specific RL algorithm. All experiments in this paper are based on GRPO, implemented within the verl framework. After selecting $B$ prompts per step, $k$ responses are generated per prompt with rewards computed, and the policy is updated using the GRPO objective. The inference overhead of DPS is only $O(|\mathcal{D}| \times 3^2)$ matrix operations, accounting for less than 1% of total training time in practice.

## Key Experimental Results

### Main Results: Mathematical Reasoning (Trained on MATH, Cross-Benchmark Evaluation)

| Method | AIME24 | AMC23 | MATH500 | Minerva | Olympiad | Avg↑ | Rollouts↓ | Runtime↓ |
|--------|--------|-------|---------|---------|----------|------|-----------|----------|
| R1-Distill-1.5B (baseline) | 18.33 | 51.73 | 76.64 | 23.83 | 35.31 | 41.17 | - | - |
| +US | 26.46 | 63.18 | 82.78 | 27.46 | 43.00 | 48.57 | 737k | 27h |
| +HR | 28.13 | 64.61 | 82.88 | 27.37 | 43.15 | 49.23 | 737k | 28h |
| +DS (Oracle) | 31.88 | 67.32 | 84.79 | 29.18 | 46.83 | 52.00 | 2933k | 89h |
| **+DPS (Ours)** | **32.71** | **67.77** | **84.95** | 29.09 | 46.11 | **52.13** | **737k** | **32h** |
| R1-Distill-7B (baseline) | 37.71 | 68.45 | 86.94 | 34.74 | 46.94 | 54.95 | - | - |
| +US | 45.83 | 73.57 | 89.06 | 37.68 | 50.42 | 59.31 | 287k | 30h |
| +HR | 46.46 | 75.98 | 90.01 | 37.94 | 51.50 | 60.38 | 287k | 36h |
| +DS (Oracle) | 49.79 | 78.99 | 90.96 | 37.89 | 54.45 | 62.42 | 1147k | 73h |
| **+DPS (Ours)** | **51.04** | **80.35** | **91.13** | 37.82 | **55.32** | **63.13** | **287k** | **39h** |

**Key Finding**: DPS surpasses DS oracle at both the 1.5B and 7B scales (Avg +0.13/+0.71), while using only 25.1% (737k vs. 2933k) and 25.0% (287k vs. 1147k) of DS's rollouts, respectively. Runtime is approximately 36% (32h vs. 89h) and 53% (39h vs. 73h) of DS.

### Cross-Task Generalization: Countdown Planning & Geometry Visual Reasoning

| Task / Model | Method | Test Accuracy | Rollouts↓ |
|--------------|--------|---------------|-----------|
| Countdown / Qwen2.5-3B | +US | 69.87 / 39.42 | 246k |
| | +HR | 70.19 / 42.10 | 246k |
| | +DS (Oracle) | 74.95 / 47.67 | 1141k |
| | **+DPS** | **74.27 / 47.78** | **246k** |
| Countdown / Qwen2.5-7B | +US | 77.84 / 53.27 | 246k |
| | +HR | 78.15 / 54.54 | 246k |
| | +DS (Oracle) | 81.26 / 60.77 | 1006k |
| | **+DPS** | **81.15 / 59.61** | **246k** |

DPS matches DS performance on both Countdown (numerical planning) and Geometry3k (visual geometric reasoning using Qwen2.5-VL multimodal model), with rollout counts approximately 21–24% of DS. This demonstrates the method's cross-task and cross-modality generalizability.

### Prediction Accuracy and Effective Sample Ratio

- Overall prediction accuracy remains consistently high throughout training; precision, recall, and F1 for Class 2 (partially solved) are robust.
- Confusion matrices show strengthening diagonals and decreasing off-diagonal entries as training progresses, reflecting continuously improving predictive capability.
- **Effective sample ratio** (fraction of partially solved prompts in each batch): DPS achieves ~90%, far exceeding US (~30–50%) and HR (~40–60%), approaching DS oracle levels.

### Ablation Study

**Non-stationary decay $\lambda$**: Setting $\lambda = 1$ (no decay, equal weighting of all history) degrades both performance and prediction accuracy, confirming that solve dynamics are genuinely non-stationary. Setting $\lambda = 0$ (using only the most recent observation) also degrades, as valuable historical information is discarded. Moderate $\lambda$ (e.g., 0.9–0.99) achieves the best balance between responsiveness and history utilization.

**Number of state partitions**: Two states (partially solved vs. other) conflates unsolved and fully solved, masking their distinct dynamics. Four or more states spread limited observations across more categories, leading to sparse estimates. Three states achieves the optimal trade-off between modeling accuracy and data efficiency.

**Response group size $k$**: DPS shows the largest advantage over US at $k = 4$. With smaller $k$, the probability that a prompt simultaneously produces both correct and incorrect responses, $1 - p^k - (1-p)^k$, decreases, causing US's effective sample ratio to drop dramatically. DPS's active prediction compensates for this shortfall.

## Highlights & Insights

- **Elegant fit of HMM modeling**: The three-state HMM precisely maps to the gradient signal structure of GRPO (States 1 and 3 yield zero gradient; State 2 yields nonzero gradient), making state prediction directly equivalent to gradient signal prediction. The degenerate emission model (observation equals true state) drastically simplifies inference; the entire system only needs to maintain a $3 \times 1$ belief vector and a $3 \times 3$ Dirichlet parameter per prompt.
- **Elegant design of implicit exploration**: The non-stationary decay $\lambda$ serves a dual purpose—adapting to non-stationary dynamics while automatically achieving exploration through posterior drift toward the uniform distribution. Experiments (Fig. 7) confirm that smaller $\lambda$ produces more uniform sampling frequency distributions, effectively preventing sampling deadlock.
- **Self-reinforcing accuracy**: DPS preferentially samples State 2 prompts, obtaining more State 2 observations, which in turn improves prediction accuracy for State 2. This positive feedback loop is the intrinsic mechanism by which the method becomes more accurate over time.
- **Implicit connection to curriculum learning**: DPS implicitly implements a data-driven adaptive curriculum—early in training, many prompts are in State 1 (unsolved), so DPS naturally samples those just becoming solvable; later in training, as more prompts transition to State 3 (fully solved), DPS automatically focuses on the remaining challenging samples. This curriculum requires no manually designed difficulty metrics.

## Limitations & Future Work

- **Reward structure assumption**: The current method relies on binary correctness rewards to define the three states. For dense reward (e.g., process reward) or continuous reward settings, the state partition scheme would need to be redesigned. The authors mention extending via partitioning cumulative return intervals, but this has not been validated.
- **Independence assumption across prompts**: Each prompt maintains an independent HMM, without exploiting structural similarities across prompts (e.g., prompts covering the same knowledge point or difficulty level may have similar transition dynamics). Shared transition models or prompt embedding-conditioned priors could potentially improve prediction accuracy.
- **Top-$B$ greedy selection**: The pure exploitation strategy may be suboptimal in extreme scenarios. The authors mention entropy-based prioritization as a future direction—prioritizing prompts with the most uncertain state predictions may be superior in certain settings.
- **Scalability not fully analyzed**: Although inference overhead is $O(|\mathcal{D}| \times 9)$, iterating over the entire prompt set to compute and rank priors at every step may become a bottleneck when the dataset is very large (millions of prompts). Priority-queue-based incremental updates could be considered.
- **Only validated with GRPO**: While the DPS framework is theoretically orthogonal to the RL algorithm, experiments are conducted solely with GRPO. Applicability to other RL algorithms such as PPO and REINFORCE++ remains to be confirmed.

## Related Work & Insights

- **vs. Dynamic Sampling (DS)**: DS is a rollout-intensive oracle approach—it requires full rollouts on a 3–4× candidate set to filter effective prompts. DPS replaces rollouts with prediction, reducing rollout overhead by 75% while maintaining DS-level accuracy. The core distinction is that DS follows a "generate then filter" paradigm, whereas DPS follows "predict then sample."
- **vs. History Resampling (HR)**: HR marks fully solved prompts at the epoch level and excludes them in subsequent epochs. Its limitations are twofold: (1) the epoch-level absorbing state assumption is overly rigid—after model updates, some "solved" prompts may revert to unsolved; (2) it only excludes solved prompts rather than actively identifying partially solved ones, limiting its effectiveness in the early-to-mid stages of training.
- **vs. Offline data filtering**: Static filtering methods based on difficulty estimation or domain balancing cannot adapt to the model's continuously shifting capability frontier, whereas DPS continuously tracks solve dynamics through online inference.
- **Connection to curriculum learning**: DPS can be viewed as an implicit, data-driven form of curriculum learning—unlike traditional curriculum learning, which requires predefined difficulty metrics, DPS directly learns difficulty dynamics from rollout feedback.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Elegantly reformulates prompt sampling as an online HMM state prediction problem; perspective is original and theoretically grounded.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers three task types (math / planning / vision) and multiple model scales (1.5B–7B) with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ — Modeling is clear, derivations are thorough, and experimental presentation is rigorous.
- Value: ⭐⭐⭐⭐⭐ — Achieves state-of-the-art sampling strategy performance with <30% of the rollouts, with direct practical value for large-scale RL finetuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Towards Safe Reasoning in Large Reasoning Models via Corrective Intervention](towards_safe_reasoning_in_large_reasoning_models_via_corrective_intervention.md)
- [\[ACL 2026\] Dissecting Failure Dynamics in Large Language Model Reasoning](../../ACL2026/llm_reasoning/dissecting_failure_dynamics_in_large_language_model_reasoning.md)
- [\[ICLR 2026\] Training Large Reasoning Models Efficiently via Progressive Thought Encoding](training_large_reasoning_models_efficiently_via_progressive_thought_encoding.md)
- [\[ICLR 2026\] RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention in Large Reasoning Models](rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_reasoning_interv.md)
- [\[ICLR 2026\] Reasoning or Retrieval? A Study of Answer Attribution on Large Reasoning Models](reasoning_or_retrieval_a_study_of_answer_attribution_on_large_reasoning_models.md)

</div>

<!-- RELATED:END -->
