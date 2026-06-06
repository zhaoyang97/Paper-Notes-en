---
title: >-
  [Paper Note] Hista and Numca: Estimate State Value Effectively for LLM Reinforcement Learning
description: >-
  [ICML 2026][Reinforcement Learning][LLM RL] This paper first empirically demonstrates, using a newly constructed State Value Estimation Benchmark (SVEB)…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "LLM RL"
  - "GRPO"
  - "State Value Estimation"
  - "Hindsight"
  - "Hidden State Representation"
date: 2026-05-08
content_hash: d0b90acaa3d9d6f6
---

# Hista and Numca: Estimate State Value Effectively for LLM Reinforcement Learning

**Conference**: ICML 2026  
**arXiv**: [2605.29782](https://arxiv.org/abs/2605.29782)  
**Code**: https://github.com/VOXXXX1874/Hista  
**Area**: Reinforcement Learning / LLM Post-training / Credit Assignment  
**Keywords**: LLM RL, GRPO, State Value Estimation, Hindsight, Hidden State Representation  

## TL;DR
This paper first empirically demonstrates, using a newly constructed State Value Estimation Benchmark (SVEB), that the PPO critic in LLM RL almost entirely degenerates into the group mean reward baseline of GRPO. Subsequently, two state value estimation methods are proposed with the goal of "no additional rollouts and near-zero extra compute": Numca utilizes numerical milestones to refactor mathematical reasoning as goal-conditioned RL for credit assignment; Hista uses LLM final-layer hidden states combined with MinDistance for probability-weighted reward averaging. Both methods reduce MAE below GRPO/PPO across five SVEB subsets and provide consistent improvements for strong algorithms like DAPO/CSIPO on multiple mathematical benchmarks.

## Background & Motivation

**Background**: Following DeepSeek-R1, the RL post-training paradigm represented by GRPO and its successors (DAPO, GSPO, CSIPO) has become the de facto standard for LLM reasoning alignment. Their common framework treats the "entire response" as a single action, using the group mean reward $\bar r$ as the baseline for every token to perform policy gradients. While this design circumvents the difficulty of token-level state value estimation, it sacrifices the fine-grained credit assignment typically provided by a critic in classical RL.

**Limitations of Prior Work**: Upon constructing SVEB, the authors found that the widely used PPO critic does not provide "finer guidance than the group mean" in LLM scenarios. PPO-1 (unseen data) performs nearly identically to GRPO (0.169 vs 0.164), and PPO-N (seen data) is only slightly better than GRPO (0.158 vs 0.164). More directly, the distribution of $\widehat V_{PPO}(s_t)-\widehat V_{GRPO}(s_t)$ was found to be tightly clustered around zero, indicating that the PPO critic output is essentially the group mean reward itself.

**Key Challenge**: Alternative solutions are either non-scalable (PRM and MCTS require expensive annotation or massive additional rollouts; VAPO/VC-PPO still require training a critic of the same scale as the actor) or were not designed to serve as a "baseline estimator" (PRM is mainly for correctness verification). The "computational tax" in LLM RL training loops is extremely high; therefore, a truly usable state value estimator must satisfy: no increase in rollout counts, no introduction of large-scale critics, no dependence on additional human labeling, and "plug-and-play" compatibility with existing GRPO-style training loops.

**Goal**: (i) Quantify "the quality of state value estimation" via a benchmark (SVEB); (ii) Provide a lightweight method immediately applicable to mathematical reasoning (Numca); (iii) Provide a general-purpose method requiring no priors (Hista) and theoretically prove its superiority over mean estimators.

**Key Insight**: The authors reframe LLM reasoning within the classical RL MDP framework: the state is the token prefix, the action is the next token, and rewards are provided only at termination. Thus, the essence of state value $V^\pi(s_t)=\mathbb{E}_\pi[r(s_T)\mid s_t]$ is the "expectation of future terminal rewards starting from this prefix." Any method that can aggregate and average the "multiple rollouts starting from $s_t$" based on some similarity measure is a potential state value estimator; the problem transforms into finding a "state equivalence class" or "state similarity" that is both cheap and meaningful.

**Core Idea**: Utilize a "cheap similarity" that requires no training—equivalence classes of numerical milestones for mathematics (Numca) or MinDistance between LLM final-layer hidden states for general scenarios (Hista)—to directly compute weighted average rewards from existing rollouts under the same prompt as a fine-grained baseline.

## Method

### Overall Architecture
The authors place all methods into a unified MDP: state $s_t=(x_1,\dots,x_t)$, action $a_t\in\mathcal V$, deterministic transitions, and reward $r(s_T)$ given only when $a_t=\langle\mathrm{eos}\rangle$ or at truncation. Given $\mathcal N$ rollouts for a prompt with terminal rewards $r_i$, the goal is to provide $\widehat V(s_t)$ for an intermediate state $s_t$, which is then used as a baseline for policy gradients like in GRPO.

SVEB is constructed by: selecting a set of prompts, running rollouts with a fixed $\pi$, and uniformly sampling intermediate $s_t$ from each; for each $s_t$, $n$ independent completions are sampled to obtain $\widehat V(s_t)=\frac{1}{n}\sum_i r(s_T^{(i)})$ (using MCS@20) as the reference ground truth; finally, performance is scored using MAE. Subsets are divided into five categories: number, math, science, general, and programming, covering DAPO-17K, OpenR1-220K, and multi-domain multiple-choice questions.

Both Numca and Hista are designed as "baseline replacements within the GRPO training pipeline"—without changing rollout counts, introducing new critics, or requiring extra labels, only changing the calculation of $\widehat V(s_t)$.

### Key Designs

1. **SVEB: Evaluating State Value Estimators with Monte Carlo Ground Truth**:
    - Function: Converts "value estimator quality" into a measurable MAE metric and uses it to reveal the degeneration of the PPO critic in LLMs.
    - Mechanism: Reference ground truth $\widehat V(s_t)=\frac{1}{n}\sum_{i=1}^n r(s_T^{(i)})$ is estimated by large-scale MCS; by the law of large numbers, $\widehat V(s_t)\to V^\pi(s_t)$ holds almost surely. The evaluation metric is $\mathrm{MAE}(f,D_s)=\frac{1}{|D_s|}\sum_j |f(s_t^{(j)},\theta)-\widehat V(s_t^{(j)})|$. On SVEB-NUMBER, the MAEs for PPO-1@40 / PPO-N@40 / GRPO@40 are 0.169 / 0.158 / 0.164 ($\lambda=0.95$) and 0.161 / 0.152 when $\lambda=1.0$; simultaneously, the distribution of $\widehat V_{PPO}-\widehat V_{GRPO}$ is centered near zero.
    - Design Motivation: Prior work mostly evaluated baseline quality indirectly via "downstream RL improvement," which is confounded by optimization noise and algorithmic differences. SVEB uses an independent, reproducible offline metric to isolate "estimation accuracy," elevating the observation that "PPO critics are ineffective" from empirical intuition to a quantifiable conclusion, thereby justifying the design decision to abandon training critics.

2. **Numca: Refactoring Math Problems as Goal-Conditioned RL via Numerical Milestones**:
    - Function: Identifies "numbers" in mathematical reasoning tasks as parsable, verifiable intermediate milestones, abstracting rollouts into equivalence classes based on "which numbers were passed," and using this for weighted reward averaging to obtain fine-grained state values.
    - Mechanism: Defines a pattern set $\mathcal P$ (integers, decimals, fractions, etc.), where a milestone $m=(x_i,\dots,x_j)$ is a token subsequence whose decoded string matches a pattern in $\mathcal P$. The abstraction of state $s_t$ is $s_t^M\triangleq \mathbb M(s_t)$, equal to the set of all milestones already appeared in $s_t$. Macro actions $a^M_t=(x_{t+1},\dots,x_{t'-1})$ are the tokens between consecutive abstract states. Given rollouts $\mathcal D_r$, a dictionary $\mathcal T[s^M]=(\mathrm{count}, \mathrm{reward\_sum})$ is maintained. For each rollout, the hit count for each unique abstract state is incremented and terminal rewards are accumulated. Finally, $V(s^M)=\mathcal T[s^M].\mathrm{reward\_sum}/\mathcal T[s^M].\mathrm{count}$, and this value is distributed uniformly to all tokens within the macro action. The process is a dictionary lookup with negligible overhead.
    - Design Motivation: Directly applying HER's "use final state as an alternate goal" fails in text because semantics are not discretely structured. However, "calculating a specific intermediate number" in math is naturally a verifiable sub-goal. The authors used this structural observation to bypass the challenge of unsupervised milestone discovery, resulting in a near-zero cost but significantly effective estimator. On SVEB-NUMBER, Numca@40 MAE drops to 0.132 (vs GRPO 0.175 / PPO-N 0.159), though it reverts to 0.217 on science, indicating it is a math-specific solution.

3. **Hista: Probability-Weighted Reward Averaging via LLM Hidden States and MinDistance**:
    - Function: Provides a state value estimate for any intermediate state $s_t$ under any prompt by "looking across all existing rollouts for the same prompt and weighting by similarity," without requiring domain-specific priors.
    - Mechanism: Uses the final-layer hidden states $\mathbf X_\tau$ and defines MinDistance between two variable-length hidden state sequences: each position in the longer sequence finds its nearest neighbor in the shorter sequence to sum $\ell_2$ distances: $\mathrm{MD}(\mathbf X_1,\mathbf X_2)=\sum_i \min_j \|\mathbf x_{1,i}-\mathbf x_{2,j}\|_2$. Theorem 5.2 proves that the probability of two states yielding the same final reward is inversely proportional to $\mathrm{MD}$: $P(R_1=R_2)\propto 1/\mathrm{MD}(\mathbf X_1^l,\mathbf X_2^l)$. Theorem 5.5 further proves that the bias of the probability-weighted estimator $\widehat V_{PW}(s_t)=\sum_i P_{t,i} r_i/\sum_i P_{t,i}$ is no greater than that of the mean estimator $\widehat V_{avg}(s_t)=\frac{1}{\mathcal N}\sum_i r_i$. For engineering feasibility with long sequences, $\mathbf X_\tau$ is compressed into $\mathbf E_\tau$ using EMA with smoothing factor $\alpha$ and interval $\varphi$, then sliced into a finite state space $\mathcal S^H_\tau$ at sampling interval $\delta$. For each $s_t^H$, the MD to all globally available $s_i^H$ is calculated, and the $k$-nearest neighbors are weighted by $\omega_i=1/\mathrm{MD}(s_t,s_i)$ to get $V(s_t)=\sum_{i=1}^k \omega_i r_i/\sum_{i=1}^k \omega_i$. The space complexity is $O(\lfloor T/d\rfloor^2)$, and the process is GPU-batchable with overhead comparable to GRPO.
    - Design Motivation: The authors view LLM autoregressive training as a form of implicit representation learning; thus, final-layer hidden states are natural state representations—obviating the need for separate representation models or additional sampling like PRM/MCTS. Converting the intuition that "similar states should have similar rewards" into probability weights theoretically reduces bias. Practical implementation reduces estimation costs to the same level as GRPO, allowing it to be directly integrated into any GRPO-style algorithm.

### Loss & Training
All methods retain the original clipping, KL regularization, and importance sampling mechanisms of GRPO/DAPO/CSIPO, only replacing the baseline in the advantage formula: substituting the group mean reward with $\widehat V(s_t)$ provided by Numca or Hista. Numca uses dictionary lookups and numerical regex; Hista uses final-layer hidden states with EMA compression and kNN weighting. Neither requires additional training steps or introduces new model parameters.

## Key Experimental Results

### Main Results
MAE on five SVEB subsets (@40 rollouts, reference MCS@20, lower is better):

| Method | Number ↓ | Math ↓ | Science ↓ | General ↓ | Programming ↓ |
|------|----------|--------|-----------|-----------|---------------|
| GRPO@40 | 0.175 | 0.208 | 0.215 | 0.202 | 0.157 |
| PPO-N@40 | 0.159 | 0.187 | 0.198 | 0.185 | 0.144 |
| Numca@40 | **0.132** ↓0.027 | 0.194 ↑0.007 | 0.217 ↑0.019 | 0.200 ↑0.015 | 0.154 ↑0.010 |
| Hista@40 | 0.142 ↓0.017 | **0.145** ↓0.042 | **0.173** ↓0.025 | **0.157** ↓0.028 | **0.119** ↓0.025 |
| MCS@1 (Ref) | 0.223 | 0.235 | 0.272 | 0.283 | 0.162 |
| MCS@2 (Ref) | 0.133 | 0.160 | 0.188 | 0.139 | 0.113 |

Numca wins significantly only on the number-heavy subset; Hista outperforms GRPO/PPO-N across all five subsets and approaches the accuracy of MCS@2, which uses twice the rollouts.

### Ablation Study
Downstream math evaluation results after replacing the GRPO baseline with Numca using Qwen2.5-Math-1.5B-Instruct:

| Benchmark | Qwen | + GRPO | + Numca |
|-----------|------|--------|---------|
| MATH-500 | 0.740 | 0.746 | **0.760** |
| GSM8K | 0.849 | 0.848 | **0.864** |
| MinervaMath | 0.286 | 0.301 | **0.313** |
| OlympiadBench | 0.425 | 0.413 | **0.426** |
| AMC23 | 0.528 | 0.553 | **0.584** |
| AIME24&25 | 0.104 | 0.113 | 0.104 |
| **AVERAGE** | 0.528 | 0.541 | **0.555** |

Training curves (Fig. 3) show that Numca's validation accuracy rises steadily while GRPO/PPO exhibit instability—the authors attribute this to the fine-grained and more accurate baseline reducing policy gradient variance. Hista applied to DAPO (Fig. 1b) showed significantly better convergence speed and terminal rewards than base DAPO/CSIPO.

### Key Findings
- The observation that "PPO critics degenerate to GRPO baselines in LLM RL" is replicated across multiple data sources, implying that training a full-scale critic is often a waste of compute in practice.
- Higher estimation accuracy $\Rightarrow$ smoother training curves: Numca's training-validation curves are more stable with higher peaks in mathematical scenarios, proving that baseline variance is strongly correlated with RL policy gradient variance—improving the baseline is the cheapest way to reduce variance compared to increasing rollout counts.
- Numca did not beat GRPO on AIME24&25; the authors' analysis suggests that problems with tiny answer spaces but extremely long intermediate steps provide limited utility for number-based milestones, reinforcing the necessity of the general-purpose Hista.

## Highlights & Insights
- **SVEB as a benchmark may be more influential than the methods themselves**: It provides a metric for baseline quality independent of downstream RL, allowing any future baseline to undergo cheap ablation before integration into RL training.
- **Numca is a lightweight application of Hindsight**: Implemented with a dictionary and regex without additional networks, it slashed MAE on SVEB-Number. This logic can transfer to any scenario where intermediate results are quickly verifiable, such as code execution stacks or intermediate lemmas in theorem proving.
- **Hista turns the "LLM = Implicit Representation Learner" belief into a tool**: By using final-layer hidden states, MinDistance, and probability weighting without new models or extra rollouts, it achieved near-optimal estimation accuracy with a theoretical guarantee (Theorem 5.5) that its bias is no worse than GRPO.
- **Alignment of Theory and Engineering**: Theorem 5.2 links MD inversely to "same-reward probability," and Theorem 5.5 provides a bias inequality for weighted vs. mean estimators—providing the "why" rather than just empirical effectiveness.

## Limitations & Future Work
- Numca depends heavily on numerical density and performs similarly to GRPO in non-numerical reasoning (science/general), which is why Hista was developed as a successor.
- Hista's MD measure ($\ell_2$ + MinDistance) relies on theoretical assumptions (A.1, B.2) that have not been extensively verified for real LLM hidden spaces. Probability weighting might also degenerate to mean averaging when rollout diversity is low.
- Evaluations remain centered on offline MAE and math/science/programming benchmarks, lacking coverage of multi-turn dialogues or long-horizon agent tasks (e.g., tool use) where prompt homogeneity and hidden state stability are more complex.
- Space complexity $O((T/d)^2)$ could still strain VRAM for 100k+ token contexts; while EMA and sampling sampling intervals mitigate this, further compression is needed for extreme long-chain reasoning.

## Related Work & Insights
- **vs PPO / VAPO / VC-PPO**: These require training an actor-scale critic; this paper empirically shows such critics converge to group means in LLM RL, justifying the direct substitution with Numca/Hista to save training costs.
- **vs PRM / MCTS credit assignment**: These rely on manual process labels or massive additional rollouts, which are high-cost; Hista reuses hidden states from existing rollouts at near-zero cost and is not limited to "verifying correctness."
- **vs HER / GCRL**: This work is a concrete implementation of HER for LLMs—Numca treats "reaching a number" as a hindsight goal, and Hista uses hidden states to represent semantic goals, removing dependencies on explicit goal definitions.
- **vs GRPO / DAPO / GSPO / CSIPO**: These algorithms focus on modifying clip/IS/KL; Ours is orthogonal—it leaves those mechanisms untouched and only replaces the baseline, allowing it to be stacked with any of them for further gains as shown in Fig. 1b and Fig. 3.

## Rating
- Novelty: ⭐⭐⭐⭐ SVEB + Numca + Hista identifies "baseline quality" as an independent optimization axis; Hista's use of hidden states for weighted baselines is a first.
- Experimental Thoroughness: ⭐⭐⭐⭐ SVEB across five domains, math downstream benchmarks, and application to DAPO over multiple models (Qwen2.5-1.5B/Math), though lacking agent-style tasks.
- Writing Quality: ⭐⭐⭐⭐ The logical chain from "PPO degeneration" to the necessity of Numca/Hista is clear, with a balance of theorems and empiricism.
- Value: ⭐⭐⭐⭐⭐ A near-zero compute baseline replacement that can be directly integrated into existing GRPO/DAPO loops provides immediate engineering benefits for industrial LLM post-training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Unified Value Alignment and Assignment in Cross-Domain Offline Reinforcement Learning](unifying_value_alignment_and_assignment_in_cross-domain_offline_reinforcement_le.md)
- [\[ACL 2026\] Efficient Hyperparameter Optimization for LLM Reinforcement Learning](../../ACL2026/reinforcement_learning/efficient_hyperparameter_optimization_for_llm_reinforcement_learning.md)
- [\[ICML 2026\] DARTS: Distribution-Aware Active Rollout Trajectory Shaping for Accelerating LLM Reinforcement Learning](darts_distribution-aware_active_rollout_trajectory_shaping_for_accelerating_llm_.md)
- [\[ICML 2026\] Multi-Agent Decision-Focused Learning via Value-Aware Sequential Communication](multi-agent_decision-focused_learning_via_value-aware_sequential_communication.md)
- [\[ICLR 2026\] Value Flows](../../ICLR2026/reinforcement_learning/value_flows.md)

</div>

<!-- RELATED:END -->
