---
title: >-
  [Paper Note] BOTS: A Unified Framework for Bayesian Online Task Selection in LLM Reinforcement Finetuning
description: >-
  [ICLR 2026][LLM/NLP][Reinforcement finetuning] This paper proposes BOTS—a unified Bayesian inference framework for online task selection in LLM reinforcement finetuning. BOTS integrates explicit evidence (historical pass rates from direct evaluation) and implicit evidence (difficulty estimates for unevaluated tasks inferred via reference model interpolation), combined with Thompson sampling for exploration–exploitation balance. The framework achieves up to 50% training speedup on math, code, and logic tasks with only 0.2% additional computational overhead.
tags:
  - ICLR 2026
  - LLM/NLP
  - Reinforcement finetuning
  - online task selection
  - Bayesian inference
  - Thompson sampling
  - curriculum learning
date: 2026-05-08
content_hash: 6bb67491d78ec6fc
---

# BOTS: A Unified Framework for Bayesian Online Task Selection in LLM Reinforcement Finetuning

**Conference**: ICLR 2026
**arXiv**: [2510.26374](https://arxiv.org/abs/2510.26374)
**Code**: [GitHub](https://github.com/modelscope/Trinity-RFT/tree/main/examples/bots)
**Area**: LLM/NLP
**Keywords**: Reinforcement finetuning, online task selection, Bayesian inference, Thompson sampling, curriculum learning

## TL;DR

This paper proposes BOTS—a unified Bayesian inference framework for online task selection in LLM reinforcement finetuning. BOTS integrates explicit evidence (historical pass rates from direct evaluation) and implicit evidence (difficulty estimates for unevaluated tasks inferred via reference model interpolation), combined with Thompson sampling for exploration–exploitation balance. The framework achieves up to 50% training speedup on math, code, and logic tasks with only 0.2% additional computational overhead.

## Background & Motivation

**Background**: Reinforcement finetuning (RFT) has become a core technique for aligning LLM capabilities and improving reasoning performance, with models such as DeepSeek-R1 and OpenAI o1 relying on large-scale RFT. However, the efficiency bottleneck of RFT lies in training data selection—specifically, which tasks the model should practice at each step. The prevailing approach uniformly samples all tasks from the training set, resulting in substantial wasted computation on tasks the model has either fully mastered (pass rate = 1) or cannot solve at all (pass rate = 0), leaving effective learning signals extremely sparse.

**Limitations of Prior Work**: Existing task selection solutions fall into four categories, each with notable shortcomings: (1) **Offline curriculum learning** (Parashar et al., Shen et al.) pre-schedules a fixed easy-to-hard ordering that cannot adapt to the model's continuously evolving capabilities—tasks deemed "moderate" earlier may be fully mastered after a capability jump mid-training; (2) **Oversampling-and-filtering** (DAPO, StarPO) generates an oversized rollout batch at each step and filters by pass rate, incurring 2–4× additional inference overhead; (3) **Explicit-evidence-only methods** (LIMRL, TTRL/MoPPS) estimate task difficulty from historical direct evaluations but suffer severe cold-start problems when data is extremely sparse in early training—a task can only obtain a pass-rate estimate after being selected for training; (4) **Implicit-evidence-only methods** (AFRL/DOTS) leverage inter-task similarity to predict pass rates but still require additional rollouts to evaluate anchor tasks and completely discard existing historical evaluation information.

**Key Challenge**: Explicit and implicit evidence offer complementary strengths that are difficult to unify. The authors identify a key complementarity through experiments: explicit evidence (historical direct evaluations) provides accurate and stable difficulty estimates but is unavailable for new tasks or in early training, leading to very slow cold-start; implicit evidence (cross-task inference) quickly supplies difficulty estimates for all tasks with zero history but accumulates increasing error as the model drifts from the reference model over time. Relying on either source alone inevitably fails at some training stage.

**Goal**: (1) How can two heterogeneous evidence sources be fused within a unified framework so that each contributes most at the appropriate training stage? (2) How can implicit evidence be estimated online with negligible computational overhead? (3) How can exploration (evaluating unknown tasks) and exploitation (selecting known moderate tasks) be automatically balanced under evidence uncertainty?

**Key Insight**: The authors observe that task pass rates naturally follow a Bernoulli distribution, whose conjugate prior—the Beta distribution—elegantly encodes "beliefs about task difficulty": the parameters $\alpha$ and $\beta$ serve as accumulated pseudo-counts of successes and failures. By designing appropriate posterior update rules to fuse both evidence types and applying Thompson sampling to draw decisions from the posterior, both the evidence fusion and exploration–exploitation problems can be addressed simultaneously.

**Core Idea**: Reformulate online task selection as non-stationary Bayesian inference; fuse explicit and implicit evidence via a generalized posterior update; and use Thompson sampling to automatically balance exploration and exploitation.

## Method

### Overall Architecture

BOTS executes a three-stage loop at each training step: **task selection → model training and evidence collection → posterior update**. The input is a pool of $N$ training tasks $\{\mathcal{T}^k\}_{k=1}^N$; each task maintains a Beta posterior $\text{Beta}(\alpha_t^k, \beta_t^k)$ representing the current belief about the model's success probability on that task. The per-step procedure is: sample from the posterior to estimate difficulty → select tasks closest to the target difficulty $p^* = 0.5$ to form the training batch $\mathcal{B}_t$ → train the model with an RL algorithm such as GRPO → collect direct evaluation results for selected tasks (explicit evidence) and interpolation-based predictions for unselected tasks (implicit evidence) → update posterior parameters for all tasks by fusing both evidence sources.

### Key Designs

1. **Bayesian Difficulty Modeling and Generalized Posterior Update — Core Fusion Mechanism**

    - **Function**: Model each task's difficulty estimate as a Beta distribution and design an online update rule that simultaneously absorbs explicit and implicit evidence.
    - **Mechanism**: The model's success probability $p_t^k$ for task $\mathcal{T}^k$ is modeled as $\text{Beta}(\alpha_t^k, \beta_t^k)$. The posterior update takes a generalized form: $\alpha_{t+1}^k = (1-\lambda)\alpha_t^k + \lambda\alpha_0^k + (1-\rho)s_t^k + \rho\tilde{s}_t^k$ (with a symmetric update for $\beta$). Here $s_t^k, f_t^k$ are the explicit success/failure counts from direct evaluation (nonzero only for selected tasks $k \in \mathcal{B}_t$), while $\tilde{s}_t^k, \tilde{f}_t^k$ are pseudo-counts combining both evidence types—equal to explicit counts for selected tasks and generated via the interpolation estimator $\tilde{p}(k, \mathcal{B}_t)$ for unselected tasks. Parameter $\lambda \in [0,1]$ controls historical discounting (inducing forgetting by interpolating toward the prior $\alpha_0, \beta_0$), and $\rho \in [0,1]$ controls the fusion weight between explicit and implicit evidence. The authors prove that this update rule preserves closure within the Beta family—the posterior remains a Beta distribution, requiring no approximate inference.
    - **Design Motivation**: Traditional Bayesian updates can only handle direct observations (explicit evidence) and completely ignore unevaluated tasks. By introducing a pseudo-count mechanism, unselected tasks also receive difficulty updates at each step, fundamentally resolving the cold-start problem. The design of $\lambda$ addresses non-stationarity—as model capabilities continuously evolve, older evaluations gradually become stale, necessitating forgetting via regression toward the prior. The authors' theoretical analysis shows that the effective sample size $n_t = \alpha_t + \beta_t$ at steady state satisfies $\rho n/\lambda \leq n_t - n_0 \leq n/\lambda$, meaning $\lambda$ and $\rho$ jointly control the confidence of the estimate.

2. **Ultra-Lightweight Interpolation Plugin — Zero-Overhead Implicit Evidence Generation**

    - **Function**: Use two pre-evaluated reference models (one weak, one strong) to estimate the current model's pass rate on unevaluated tasks without any additional rollouts.
    - **Mechanism**: Given the weak model pass rate $\bar{p}_w^k$ and strong model pass rate $\bar{p}_s^k$ (available in many RL datasets such as GURU), the method computes at each step the current model's relative capability coefficient $\mu_t = (\bar{p}_t^{\text{ref}} - \bar{p}_w^{\text{ref}}) / (\bar{p}_s^{\text{ref}} - \bar{p}_w^{\text{ref}})$—the interpolation position of the current model between the weak and strong references. To reduce noise, $\mu_t$ is smoothed with momentum: $\tilde{\mu}_t = \gamma \tilde{\mu}_{t-1} + (1-\gamma)\mu_t$. The pass rate for any unevaluated task $k$ is then estimated as $\tilde{p}(k, \mathcal{B}_t) = \text{clip}(\tilde{\mu}_t \cdot \bar{p}_s^k + (1-\tilde{\mu}_t) \cdot \bar{p}_w^k, 0, 1)$.
    - **Design Motivation**: In contrast to AFRL's attention-kernel approach, which requires additional rollouts to evaluate anchor tasks, the interpolation method requires only a one-time reference model evaluation and involves only simple vector arithmetic, yielding training overhead $\leq 0.2\%$. Linear interpolation is a strong assumption, but experiments demonstrate sufficient accuracy within the reasonable range between the weak and strong reference models, and the BOTS framework is inherently robust to noise in implicit evidence via the weighting parameter $\rho$.

3. **Thompson Sampling for Task Selection — Automatic Exploration–Exploitation Balance**

    - **Function**: Select the most educationally valuable tasks from the maintained posterior distributions to form each training batch.
    - **Mechanism**: For each task, a sample $\hat{p}_k \sim \text{Beta}(\alpha_t^k, \beta_t^k)$ is drawn from the current posterior. The utility is computed as $\hat{u}_k = -|\hat{p}_k - p^*|$ (with $p^* = 0.5$), and tasks with the highest utility are selected. Because the sampling-induced randomness is proportional to the posterior variance, tasks with high uncertainty (few evaluations) are naturally more likely to be sampled toward extreme values and thus selected for exploration, while tasks with tight posteriors tend to be stably exploited.
    - **Design Motivation**: Greedy selection (using posterior means to pick tasks closest to $p^*$) appears efficient but falls into local optima by ignoring potentially high-value tasks with high posterior uncertainty due to sparse data. Thompson sampling provides theoretical Bayes regret bound guarantees and is trivially simple to implement (a single line: sample from a Beta distribution). Ablation experiments confirm that Thompson sampling produces more stable task selection behavior than greedy selection.

### Loss & Training

BOTS is a task selection framework decoupled from the specific RL algorithm. Experiments use GRPO (Group Relative Policy Optimization) as the underlying training algorithm, generating $n = 16$ rollouts per task with binary rewards (correct/incorrect) driving the policy gradient. Task selection occurs in the outer loop of GRPO—determining which tasks to train on at each step—without modifying GRPO's optimization objective itself.

## Key Experimental Results

**Experimental Setup**: GURU dataset (math, code, and logic subsets); models: Qwen2.5-1.5B-Instruct and Qwen2.5-7B; training algorithm: GRPO; reference models: Qwen2.5-7B-Instruct (weak) and Qwen3-30B-A3B (strong). Evaluation benchmarks include MATH500, AMC23, AIME24 (math), LiveCodeBench (code), and ARC (logic).

**Three Complementary Evaluation Metrics**: ETR (Effective Task Ratio, proportion of tasks with pass rates strictly in $(0,1)$) measures selection quality; TTB (Time-to-Benchmark, steps to reach target performance relative to the random baseline, $<1$ indicates faster) measures training speedup; BSF (Best Score Factor, performance under a fixed budget relative to the random baseline, $>1$ indicates improvement) measures performance gain.

### Main Results: Cross-Domain and Cross-Scale Comparison (Qwen2.5-1.5B-Instruct)

| Method | Math TTB(50%↓) | Math TTB(100%↓) | Math BSF(25%↑) | Code TTB(50%↓) | Code BSF(25%↑) | Logic TTB(50%↓) | Logic BSF(25%↑) |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Random | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Offline (easy→hard) | 0.77 | 0.85 | 1.09 | 0.68 | 1.09 | 1.23 | 0.67 |
| BOTS-MoPPS (explicit only) | 1.02 | 0.89 | 0.98 | 0.78 | 1.09 | 0.87 | 1.13 |
| BOTS-DOTS (implicit only) | 0.70 | 0.73 | 1.08 | 0.67 | 1.17 | 0.66 | 1.28 |
| **BOTS (default)** | **0.85** | **0.72** | **1.06** | **0.58** | **1.17** | **0.85** | **1.19** |

On the 7B model, BOTS achieves TTB(100%) = 0.50 on the logic domain (50% training speedup); on the 1.5B model, it ranks first on 10 and second on 6 of 18 evaluated metrics.

### Ablation Study: Effect of $\rho$ on Evidence Fusion (1.5B, Math Aggregate)

| $\rho$ | Meaning | TTB(50%↓) | TTB(100%↓) | BSF(25%↑) | BSF(100%↑) |
|:---:|:---|:---:|:---:|:---:|:---:|
| 0.0 | Explicit evidence only | 0.98 | — | 0.99 | 0.98 |
| 0.05 | Marginal implicit | 0.78 | 0.64 | 1.06 | 1.06 |
| **0.1** | **Default** | **0.85** | **0.72** | **1.06** | **1.05** |
| 0.2 | Moderate implicit | 0.78 | 0.69 | 1.07 | 1.05 |
| 0.5 | Implicit-leaning | 0.79 | 0.89 | 1.09 | 1.00 |
| 1.0 | Implicit evidence only | 0.78 | 0.99 | 1.05 | 1.01 |

("—" indicates the target performance was not reached within the evaluation window.)

### Key Findings

- **Implicit evidence is critical for cold-start**: With $\rho = 0$ (explicit evidence only), ETR during early training is virtually indistinguishable from random sampling—because most tasks have never been selected for evaluation and thus have no historical data. In contrast, all settings with $\rho > 0$ exhibit a sharp ETR increase in early training, primarily by filtering out completely unsolvable tasks ($p = 0$). This quantifies the role of implicit evidence as a "cold-start accelerator."
- **Over-reliance on implicit evidence is harmful in the long run**: At $\rho = 0.5$ and $\rho = 1.0$, ETR declines in later training—the interpolation estimator's error accumulates as the model's capabilities drift away from the reference models, causing an increasing number of fully mastered tasks ($p = 1$) to be incorrectly identified as "moderate" and selected. This explains why $\rho = 1.0$ achieves TTB(100%) = 0.99 (essentially no speedup).
- **$\lambda$ controls the adaptability–stability tradeoff**: $\lambda = 0$ (no forgetting) causes the posterior to become overconfident and unable to track improvements in model capabilities, degrading task selection quality in later training; $\lambda = 1$ (complete forgetting) inflates posterior variance so severely that Thompson sampling degenerates toward near-random selection; $\lambda = 0.1$ achieves the best balance across all metrics.
- **BOTS-DOTS is the strongest baseline**: Even using only implicit evidence without Thompson sampling, BOTS-DOTS significantly outperforms both Offline and Random baselines, demonstrating that the interpolation plugin itself provides valuable task difficulty signals. Full BOTS further improves long-term performance by correcting for implicit evidence bias through explicit evidence and Thompson sampling exploration.
- **Computational overhead is negligible**: The entire task selection pipeline of BOTS (posterior update + interpolation + sampling) involves only simple vector arithmetic, with a wall-clock overhead of $\leq 0.2\%$ of training time—several times more efficient than oversampling-based methods.

## Highlights & Insights

- **Unifying task selection as Bayesian inference**: The elegance of this modeling choice is that it simultaneously solves three problems—the Beta-Bernoulli conjugacy yields closed-form posterior updates (no MCMC or variational inference required), the posterior variance naturally encodes uncertainty (driving Thompson sampling's exploratory behavior), and the pseudo-count mechanism provides a natural interface for fusing heterogeneous evidence. The entire framework introduces no neural networks or complex optimization; the core implementation amounts to only a few dozen lines of code.
- **Quantitative characterization of explicit/implicit evidence complementarity**: Rather than simply asserting that "the two are complementary" and fusing them, the systematic ablation of $\rho$ from 0 to 1 precisely characterizes the complementarity pattern—the value of explicit evidence increases as training progresses (accumulating richer history), while the value of implicit evidence decreases (as the model drifts further from the reference models). This provides interpretable guidance for choosing the fusion weight.
- **Zero marginal cost design of the interpolation plugin**: The method leverages difficulty labels already present in existing datasets (e.g., reference model evaluations in GURU) as the basis for implicit evidence, avoiding any additional rollouts. This design makes BOTS a genuinely plug-and-play solution—requiring no changes to the training algorithm, no additional inference budget, and no training of auxiliary models.

## Limitations & Future Work

- **Dependence on reference models**: Implicit evidence requires pre-evaluated pass rates from weak and strong reference models. While many datasets (GURU, BigMath) already provide such labels, constructing entirely new datasets still requires a one-time rollout. However, this cost is non-recurring and can be amortized over multiple training runs.
- **Coarseness of linear interpolation**: The assumption that the current model's pass rate is a linear interpolation of the weak and strong model pass rates holds reasonably well for interpolation (when the current model lies between the two references) but degrades significantly under extrapolation (when the model surpasses the strong reference or falls below the weak reference). The authors acknowledge this limitation and propose designing stronger plugins (e.g., embedding-based nonlinear predictors) as a future direction.
- **Binary reward assumption**: The entire Beta-Bernoulli modeling assumes binary (0/1) rewards. For non-binary reward settings (e.g., dialogue quality scoring, code efficiency ranking), the framework would need to be extended to more general conjugate families such as Gaussian-Normal or Dirichlet.
- **Fixed target pass rate**: Using $p^* = 0.5$ as the "most informative difficulty" is theoretically optimal under binary rewards (maximizing the expected gradient), but different training stages may benefit from different targets—earlier stages might favor easier tasks ($p^* = 0.7$) to build foundational skills, with difficulty gradually increasing thereafter. Adaptive $p^*$ scheduling is a direct improvement direction.
- **Non-adaptive fixed hyperparameters**: The optimal values of $\lambda$ and $\rho$ may vary across training stages (larger $\rho$ favoring implicit evidence early on, smaller $\rho$ favoring explicit evidence later), but the current framework uses fixed values throughout training. Designing adaptive schedules $\lambda(t), \rho(t)$ could yield further performance improvements.

## Related Work & Insights

- **vs. MoPPS (Qu et al., TTRL)**: MoPPS can be viewed as a special case of BOTS with $\lambda = 0, \rho = 0$—fully relying on explicit evidence with no historical forgetting. BOTS extends this by adding implicit evidence and a forgetting mechanism, outperforming it across all 18 metrics. This demonstrates that a pure multi-armed bandit formulation is insufficient to address the non-stationarity and cold-start challenges inherent in RFT.
- **vs. DOTS (Sun et al., AFRL)**: DOTS uses an attention kernel in embedding space for cross-task prediction, requiring additional rollouts to evaluate anchor sets and entirely ignoring historical evaluation information. BOTS-DOTS (approximating $\lambda = 1, \rho = 1$) replaces the attention kernel with a simpler interpolation, achieving comparable performance at substantially lower overhead. Full BOTS further improves long-term performance over DOTS by incorporating explicit evidence correction.
- **vs. Oversampling methods (DAPO, StarPO)**: Oversampling generates 4–8× oversized batches to obtain a sufficient number of "moderately difficult" tasks. While effective, this approach is computationally expensive. BOTS replaces brute-force oversampling with intelligent task selection, achieving better training efficiency under the same computational budget.
- **Broader Inspiration**: The design principles of BOTS—Bayesian belief maintenance, heterogeneous evidence fusion, and Thompson sampling—are highly generalizable and transferable to other online selection scenarios: sample selection in active learning, task weight scheduling in multi-task learning, and adaptive scheduling in curriculum learning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The unified perspective reformulating task selection as Bayesian inference is clear and elegant, though the underlying components (Beta-Bernoulli, Thompson sampling, linear interpolation) are all well-established techniques.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive comparisons across multiple domains (math/code/logic) and scales (1.5B/7B), with systematic ablations of $\rho$ and $\lambda$ providing deep understanding; three complementary metrics are thoughtfully designed.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The correspondence between theoretical derivations and experimental validation is exceptionally clear; hyperparameter analysis is supported by both theoretical explanations and empirical verification; practical recommendations (default configurations, reference model selection) are thorough.
- **Value**: ⭐⭐⭐⭐ Addresses a real and widespread task selection bottleneck in RFT; plug-and-play code is open-sourced; default hyperparameters generalize across settings; deployment barrier is extremely low.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] ELLMob: Event-Driven Human Mobility Generation with Self-Aligned LLM Framework](ellmob_event-driven_human_mobility_generation_with_self-aligned_language_models.md)
- [\[ICCV 2025\] Balancing Task-Invariant Interaction and Task-Specific Adaptation for Unified Image Fusion](../../ICCV2025/llm_nlp/balancing_task-invariant_interaction_and_task-specific_adaptation_for_unified_im.md)
- [\[ICLR 2026\] Near-Optimal Online Deployment and Routing for Streaming LLMs](near-optimal_online_deployment_and_routing_for_streaming_llms.md)
- [\[ICLR 2026\] Function Induction and Task Generalization: An Interpretability Study with Off-by-One Addition](function_induction_and_task_generalization_an_interpretability_study_with_off-by.md)
- [\[ACL 2026\] Automatic Combination of Sample Selection Strategies for Few-Shot Learning](../../ACL2026/llm_nlp/automatic_combination_of_sample_selection_strategies_for_few-shot_learning.md)

<!-- RELATED:END -->
