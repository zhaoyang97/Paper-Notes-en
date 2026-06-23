---
title: >-
  [Paper Note] Beyond Markovian: Reflective Exploration via Bayes-Adaptive RL for LLM Reasoning
description: >-
  [ICLR 2026][LLM Reasoning][Bayes-Adaptive RL] This paper reinterprets the "self-reflection" behavior of LLMs through the lens of Bayesian Reinforcement Learning—viewing reflection as information gathering under MDP uncertainty. It proposes the BARL algorithm, which maintains a posterior of MDP hypotheses over candidate answers and switches policies when beliefs co
tags:
  - ICLR 2026
  - LLM Reasoning
  - Bayes-Adaptive RL
date: 2026-05-08
content_hash: 78a64b741f85b731
---
# Beyond Markovian: Reflective Exploration via Bayes-Adaptive RL for LLM Reasoning

**Conference**: ICLR 2026  
**Code**: [https://github.com/shenao-zhang/BARL](https://github.com/shenao-zhang/BARL)  
**Area**: LLM Reasoning / Reinforcement Learning  
**Keywords**: Bayes-Adaptive RL, reflective exploration, MDP posterior, uncertainty-adaptive policy, Token efficiency  

## TL;DR
This paper reinterprets the "self-reflection" behavior of LLMs through the lens of Bayesian Reinforcement Learning—viewing reflection as information gathering under MDP uncertainty. It proposes the BARL algorithm, which maintains a posterior of MDP hypotheses over candidate answers and switches policies when beliefs conflict with reward feedback, simultaneously improving accuracy and token efficiency in mathematical reasoning.

## Background & Motivation
**Background**: Reasoning models trained via RL (such as DeepSeek-R1) exhibit emergent self-reflection behaviors like "rethinking" and "error correction." these are generally regarded as the source of test-time scaling, improving performance by generating longer Chains of Thought (CoT).

**Limitations of Prior Work**: However, it remains unclear **why these reflections are useful and when they should occur**. The optimal policy in conventional RL is Markovian—relying only on the current state $s_t$ and providing the same action distribution regardless of history. This implies that reflective behavior—returning to the same state and changing paths based on new context—adds no value to a Markovian optimal policy, meaning RL lacks the motivation to learn it.

**Key Challenge**: In conventional RL, exploration only occurs during the training phase (trial-and-error). Post-deployment, parameters are frozen and $\epsilon\approx0$, with no mechanism encouraging "exploration during reasoning." Consequently, reflection either emerges inconsistently or lacks theoretical explanation—recent studies have found a weak correlation between reflection frequency and performance.

**Goal**: To provide a **principled theoretical foundation** for reflective exploration and design an algorithm based on this to guide when and how LLMs should explore reflectively.

**Core Idea**: **[Bayesian RL Objective]** Training data $D$ does not uniquely determine the true MDP $M^*$, inducing a posterior $p(M\mid D)$ over possible MDPs. The objective is shifted from "maximizing return on $M^*$" to "maximizing Bayesian expected return over the posterior" $J(\pi)=\mathbb{E}_{M\sim p(M\mid D)}[J_M(\pi)]$. **[Uncertainty-Adaptive Policy]** The optimal policy for this objective must depend on the complete history (belief). Information-gathering actions are naturally incentivized through belief updates, making reflection a rational behavior to "eliminate hypotheses and reduce MDP uncertainty." The paper further proves that the return gap between an adaptive policy and any Markovian policy can be **arbitrarily large**.

## Method

### Overall Architecture
BARL (Bayes-Adaptive RL for LLM Reasoning) maps multiple CoT rollouts of each prompt to individual "MDP hypotheses." It weights the value of each hypothesis using posterior beliefs and attenuates the weight of a hypothesis when predicted rewards are inconsistent with observed rewards, serving as a "policy switch" signal. Overall, it replaces the value under $M^*$ in standard policy gradients with the **posterior-weighted value**.

```mermaid
flowchart TD
    A[Prompt s0] --> B[Sample |M| CoT rollouts]
    B --> C[Extract candidate answers → Construct MDP hypotheses M_i]
    C --> D[Calculate values Q_Mi for each hypothesis]
    D --> E["Posterior Weighting: Model Credibility × Reward Consistency"]
    E --> F[Posterior-weighted value → policy gradient update θ]
    F -->|Decrease hypothesis weight on belief-reward conflict| C
```

### Key Designs

**1. Posterior-Weighted Policy Gradient: Replacing "True MDP" with "Hypothesis Set."** Standard policy gradients $\nabla_\theta J = \mathbb{E}[\sum_t \nabla_\theta\log\pi_\theta(a_t\mid h_t)\,Q^{\pi_\theta}_{M^*}(h_t,a_t)]$ depend on the unknown true MDP. BARL replaces this with the expectation over the posterior $\mathbb{E}_{M\sim p(M\mid D,h_t)}[Q^{\pi_\theta}_M(h_t,a_t)]$. Q-values are used instead of advantages because the latter requires branching Monte-Carlo rollouts at every step, which is computationally expensive; Q-values allow for the reuse of the KV cache from the entire CoT.

**2. Posterior Decomposition: Credibility × Reward Consistency.** Using Bayes' theorem, the posterior of hypothesis $M$ is decomposed into two terms: $p(M\mid D,h_t)\propto p(M\mid D,s_{0:t})\cdot p(r_{0:t-1}\mid s_{0:t},a_{0:t-1},M)$. The first term "considers the CoT without rewards" and is modeled as the probability of the policy generating the answer $y^M_{s_0}$ corresponding to that hypothesis (i.e., **model credibility for that hypothesis**). The second term is the likelihood of observed rewards under that hypothesis, computed as the product of $\exp(-\beta|r_t - r_M(s_t,a_t)|)$. The final posterior-weighted value is the product of three factors:

$$\mathbb{E}_M[Q^{\pi_\theta}_M(h_t,a_t)] = \sum_{i=0}^{|M|} \underbrace{Q^{\pi_\theta}_{M_i}(h_t,a_t)}_{\text{Value under }M_i}\;\underbrace{\pi_\theta(y^{M_i}_{s_0}\mid s_t)}_{\text{Model Credibility}}\;\underbrace{\prod_{t'=0}^{t-1}\exp(-\beta|r_{t'}-r_{M_i}(s_{t'},a_{t'})|)}_{\text{Consistency with observed rewards}}$$

**3. Reflection Signal = Conflict between Belief and Reward.** The third product term is the core of BARL: when a hypothesis has high model credibility (high belief) but its predicted rewards consistently mismatch actual observed rewards, the $\exp(-\beta|\cdot|)$ factor suppresses its weight—this is the explicit signal to "switch policy." In other words, BARL formalizes "when to reflect" as: **when internal beliefs contradict cumulative reward feedback, downweight those highly credible but unlikely optimal hypotheses**. The candidate set $\{M_i\}$ consists of the ground-truth answer $M_0$ plus candidate answers extracted from the model's CoT.

**4. Progress Reward as a Dense Signal.** Beyond sparse outcome verifiers, a progress reward $r(s_t,a_t)=\pi_\phi(y^*_{s_0}\mid s_t+a_t+\texttt{</think>}) - \pi_\phi(y^*_{s_0}\mid s_t+\texttt{</think>})$ is introduced. This measures the increase in the probability of outputting the correct answer after an additional reasoning step. Compared to Monte-Carlo process rewards, it avoids step-wise branching rollouts and reuses the KV cache.

## Key Experimental Results

### Main Results (pass@1, Mean ± SE over three seeds)

| Model | GSM8K | MATH | College | Olympiad | AIME | AMC | Average |
|---|---|---|---|---|---|---|---|
| Qwen-1.5B base | 40.0 | 34.1 | 6.6 | 21.8 | 16.7 | 32.5 | 25.3 |
| GRPO | 83.9 | 71.5 | 45.1 | 31.6 | 17.8 | 55.8 | 51.0 |
| Progress | 84.8 | 72.1 | 45.9 | 35.5 | 14.4 | 55.8 | 51.4 |
| **BARL** | **85.8** | **72.7** | **46.8** | **35.8** | **17.8** | **60.8** | **53.3** |
| Qwen-7B GRPO | 90.3 | 77.6 | 47.0 | 38.5 | 24.5 | 65.0 | 57.1 |
| Qwen-7B **BARL** | **91.7** | **79.2** | **47.5** | **42.0** | **29.0** | **66.7** | **59.4** |
| Llama-8B GRPO | 85.7 | 74.3 | 39.6 | 36.0 | 16.7 | 60.4 | 52.1 |
| Llama-8B **BARL** | 85.4 | 73.9 | **40.4** | **37.2** | **17.8** | **61.7** | **52.7** |

Across three models, BARL leads in average accuracy, with the most significant Gains on difficult benchmarks requiring effective exploration (Olympiad/AMC/AIME).

### Ablation Study (Token Efficiency + Reflection Frequency)

| Dimension | Key Findings |
|---|---|
| Token Efficiency (pass@k vs. Total Tokens) | BARL achieves higher accuracy with fewer tokens: up to **1.63×** less than Progress, **2×** less than GRPO, and **10×+** less than base. |
| Reflection Frequency (Figure 6) | Base models reflect more frequently but have lower accuracy; reflection frequency is **weakly correlated** with performance. |
| Synthetic Tasks (Repeating prompt tokens 3x) | Standard RL memorizes training solutions but fails to generalize; BARL switches policies by eliminating hypotheses to find the true MDP. |
| CoT Effectiveness (Bayesian Q-value) | BARL's CoTs yield higher average Bayesian state-action values, balancing exploration and exploitation. |

### Key Findings
- **Reflection Frequency $\neq$ Performance**: Base models trigger reflection keywords more often but perform worse, suggesting they learn **superficial stylized reflection** during pre-training. BARL learns more effective reflective exploration.
- **Determinants of Performance** are the effectiveness of thinking tokens and exploration efficiency, rather than response length or reflection count.
- **Candidate Set Quality is Crucial**: Synthetic experiments show that a prior (e.g., $|M|=3$) leads to faster convergence than no prior ($|M|=27$). Candidates must be diverse enough to cover deployment uncertainty but restricted to a credible few to narrow the hypothesis space.

## Highlights & Insights
- **Turning Reflection from Heuristics to Mathematics**: Uses Bayes-Adaptive RL to provide a unified explanation of "why, how, and when" for reflective exploration, with Theorems 4.1/4.2/4.3 strictly characterizing the fundamental gap between Markovian and adaptive policies.
- **Elegant Engineering of Reflection Signals**: Implements "belief-reward conflict $\rightarrow$ policy switch" as an $\exp(-\beta|\cdot|)$ product factor. This is analogous to a linearized "best-of-N" but with explicit guidance on when and how to explore.
- **Minimal Computational Overhead**: Candidate answer probabilities are calculated at the end of each step by reusing CoT prefix caches, avoiding the branching rollouts typical of process rewards.
- **Counter-intuitive Empirical Conclusion**: Longer and more frequent reflection is not necessarily better; the key is the "effectiveness" of tokens.

## Limitations & Future Work
- The candidate set $\{M_i\}$ is extracted from the model's own CoT. If the model cannot cover the correct strategy, the hypothesis space may miss the true MDP, limiting BARL's advantages.
- Progress rewards rely on the "increase in the probability of outputting the correct answer" as a proxy, requiring ground-truth answers $y^*_{s_0}$, which is not directly applicable to open-ended tasks.
- Experiments focus on mathematical reasoning with 1.5B/7B/8B scales; effectiveness in larger models or tasks like coding/agents remains to be verified.
- Hyperparameters like $\beta$ and $|M|$ significantly impact hypothesis weights; the paper uses fixed values ($\beta=1, |M|=5$), leaving room for adaptive adjustment.

## Related Work & Insights
- **Connection to Meta-RL**: Uncertainty-adaptive policies can be viewed as using "in-context learning instead of parameter updates." BARL is equivalent to "learning to do in-context learning," emphasizing optimal exploration-exploitation trade-offs under a Bayesian objective.
- **Difference from Progress Rewards/MRT (Qu et al. 2025)**: While MRT rewards strategies that "advance toward the correct answer," BARL additionally encourages exploring multiple credible strategies within a Bayesian framework.
- **Difference from PSRL/Thompson Sampling in LLMs (Arumugam & Griffiths 2025; Dwaracherla et al. 2024)**: Those works place Bayesian exploration in an external algorithmic scaffold or data collection layer; BARL **directly optimizes the Bayesian objective within the RL fine-tuning target**, providing step-level reflection guidance.
- **Insight**: Explaining an "emergent behavior" with a formalized objective (Bayesian return) before deriving an algorithm is a more rigorous research paradigm than heuristic reward engineering.

## Rating
- **Novelty** ⭐⭐⭐⭐⭐: Re-frames reflective exploration with Bayes-Adaptive RL. Originality in both theory (Theorems 4.1–4.3) and algorithm (three-factor posterior weighting) addresses the fundamental question of "why reflection is useful."
- **Experimental Thoroughness** ⭐⭐⭐⭐: Includes synthetic tasks, three models, four benchmarks, and multi-angle ablations on token efficiency/reflection frequency/CoT effectiveness; however, the model scales are relatively small and tasks are limited to mathematics.
- **Writing Quality** ⭐⭐⭐⭐⭐: The "Why/How/When" structure is clear. The logical chain from theoretical foundation to algorithmic implementation is robust, with formulas well-supported by intuitive explanations.
- **Value** ⭐⭐⭐⭐: Provides both a theoretical lens for understanding reflection in RL reasoning models and a plug-and-play, computationally efficient algorithm with direct implications for test-time scaling and efficient inference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MAGO: Beyond Fixed Hyperparameters with Multi-Objective Pareto Optimization for Hybrid LLM Reasoning](mago_beyond_fixed_hyperparameters_with_multi-objective_pareto_optimization_for_h.md)
- [\[ICLR 2026\] Attention as a Compass: Efficient Exploration for Process-Supervised RL in Reasoning Models](attention_as_a_compass_efficient_exploration_for_process-supervised_rl_in_reason.md)
- [\[ICLR 2026\] Tricks or Traps? A Deep Dive into RL for LLM Reasoning](tricks_or_traps_a_deep_dive_into_rl_for_llm_reasoning.md)
- [\[ICLR 2026\] Beyond Magnitude: Leveraging Direction of RLVR Updates for LLM Reasoning](beyond_magnitude_leveraging_direction_of_rlvr_updates_for_llm_reasoning.md)
- [\[ICLR 2026\] RL of Thoughts: Navigating LLM Reasoning with Inference-Time Reinforcement Learning](rl_of_thoughts_navigating_llm_reasoning_with_inference-time_reinforcement_learni.md)

</div>

<!-- RELATED:END -->
