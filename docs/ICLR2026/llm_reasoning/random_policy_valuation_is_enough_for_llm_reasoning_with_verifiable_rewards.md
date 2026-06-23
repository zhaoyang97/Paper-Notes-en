---
title: >-
  [Paper Note] Random Policy Valuation is Enough for LLM Reasoning with Verifiable Rewards
description: >-
  [ICLR 2026][LLM Reasoning][RLVR] The authors observe that RLVR for mathematical reasoning corresponds to a simplified MDP with "deterministic transitions + tree structure + binary terminal rewards." In this structure, evaluating the Q-values of a **fixed uniform random policy** followed by softmax sampling bypasses the "evaluation-improvement" cycles
tags:
  - ICLR 2026
  - LLM Reasoning
  - RLVR
date: 2026-05-08
content_hash: 3dfa4da765835cb5
---
# Random Policy Valuation is Enough for LLM Reasoning with Verifiable Rewards

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=ujLgLz6QQa](https://openreview.net/forum?id=ujLgLz6QQa)  
**Code**: https://github.com/tinnerhrhe/ROVER  
**Area**: LLM Reasoning / RLVR  
**Keywords**: RLVR, Random Policy Valuation, Mean Operator, Reasoning Diversity, Mathematical Reasoning

## TL;DR
The authors observe that RLVR for mathematical reasoning corresponds to a simplified MDP with "deterministic transitions + tree structure + binary terminal rewards." In this structure, evaluating the Q-values of a **fixed uniform random policy** followed by softmax sampling bypasses the "evaluation-improvement" cycles and heuristic tricks of PPO/GRPO, achieving a reasoning policy that is both high-quality (pass@1 +8.2, pass@256 +16.8) and highly diverse (+20.5%).

## Background & Motivation
**Background**: LLM post-training with verifiable rewards (RLVR) currently relies almost entirely on PPO and its derivatives (GRPO, REINFORCE++, DAPO). These methods follow the "Generalized Policy Iteration" (GPI) paradigm: alternating between evaluating the value of the current policy and improving the policy accordingly until convergence.

**Limitations of Prior Work**: PPO was originally designed for **general control problems** like video games and robotics—where state transitions may be stochastic, reward structures are complex, and the state space is a graph with cycles. Applying it to LLM reasoning leads to "mismatch" issues: unstable training dynamics and diversity collapse, causing the exploration space to narrow. To mitigate these, engineering efforts pile on heuristics: clipping, KL regularization, data filtering, etc. Each requires careful hyperparameter tuning for specific tasks, significantly increasing complexity.

**Key Challenge**: The fundamental issue with GPI lies in the **non-stationary** evaluation target—as the policy changes, the evaluation object changes, leading to instability and entropy collapse. However, the authors note that the underlying MDP of mathematical reasoning tasks is much simpler than general control. Each action (generating a token) deterministically unfolds into a new branch, every partial sequence has exactly one parent state, the reachable graph is a **tree**, and rewards are binary signals given only at the end (correct=1, incorrect=0). Are we using overly complex tools for a structurally simpler (though large-scale) problem?

**Goal**: Within this specialized tree-structured MDP, find a **minimalist yet effective** RLVR algorithm that maintains both quality and diversity without needing GPI cycles or heuristics.

**Key Insight**: Classical RL generally holds that the "mean operator" (evaluating a uniform policy) is useless for general control problems—it averages across all actions without bias toward the optimal action, providing no useful guidance (Asadi & Littman, 2017). However, the authors prove that in this specialized structure of tree-shaped, deterministic, binary terminal rewards, **the Q-values of a uniform random policy paired with greedy selection is already optimal**.

**Core Idea**: Instead of adding tricks to PPO/GRPO, simply evaluate the "simplest policy" (uniform random) and perform softmax sampling based on its Q-values—one policy evaluation is sufficient, eliminating the need for iterative improvement.

## Method

### Overall Architecture
ROVER (Random Policy Valuation for Diverse Reasoning) formalizes mathematical reasoning as an MDP $M$ with finite steps, deterministic transitions, a tree-structured state space, and binary terminal rewards: the state is the generated token sequence, the action is the next token from vocabulary $V$, transitions are concatenations (deterministic), discount $\gamma=1$, and the reward $r(x,y)\in\{0,1\}$ is provided by a verifier only at the end.

The logic of ROVER is: **Theoretically prove "Evaluating a fixed uniform policy + Greedy" is optimal → Replace greedy with softmax sampling to preserve diversity → Parametrize abstract Q-values using the LLM's own parameters for a scalable algorithm**. It completely abandons the GPI "evaluation $\leftrightarrow$ improvement" loop, retaining only one evaluation of the uniform policy, thereby eliminating value networks, clipping, KL regularization, and non-stationary targets.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Input: prompt x<br/>+ Verifier binary reward"] --> B["Random Policy Evaluation<br/>Q^πu via Mean Operator"]
    B --> C["Softmax Sampling<br/>Swap Greedy for Softmax to keep diversity"]
    C --> D["Intrinsic Q-Parametrization<br/>Relative Q = ρ(logπθ − logπθold)"]
    D --> E["Low-variance Rewards<br/>Group centering + Token-level broadcasting"]
    E -->|Regression vs sg[Q̂] to update θ| B
    E --> F["Output: High quality + High diversity reasoning policy"]
```

### Key Designs

**1. Random Policy Evaluation: Greedy selection using Q-values of the uniform policy yields optimality**

This directly addresses the pain point of "instability + entropy collapse caused by non-stationary GPI targets." Starting from the simplest uniform random policy $\pi_u(a|s)=1/|A|$, the authors estimate its Q-values using the generalized Bellman update with the **mean operator**. Under deterministic transitions and $\gamma=1$, the update simplifies to $\hat{Q}^{\pi_u}(s,a)\leftarrow r(s,a)+\frac{1}{|A|}\sum_{a'\in A}\hat{Q}^{\pi_u}(s',a')$. Theorem 1 proves: In a deterministic, tree-structured MDP with binary terminal rewards ($R(s)\in\{0,R\}, R>0$), the greedy policy $\pi_{\text{greedy}}(s)=\arg\max_a Q^{\pi_u}(s,a)$ **is the optimal policy**.

Why does the "useless" mean operator suddenly work here? Because $Q^{\pi_u}(s,a)$ has a clear interpretation in this structure: **It equals the probability of eventually reaching a correct answer if one takes action $a$ at state $s$ and follows a uniform random policy thereafter**. $Q^{\pi_u}(s,a)=0$ indicates that no subsequent path from $(s,a)$ leads to a correct solution; higher values indicate a "denser concentration of successful paths" in that branch. Thus, being greedy with respect to Q-values is equivalent to pruning all dead ends and prioritizing the most promising branches. The entire process requires evaluating only one **fixed** policy, avoiding off-policy corrections and iterative improvements, thus eliminating non-stationary target issues.

**2. Softmax Sampling: Swapping greedy for soft sampling to recover diversity while maintaining performance**

While pure greedy is optimal, it is deterministic and collapses to a single mode (Figure 5(e): Q-learning and ROVER-greedy only cover one optimal solution), whereas reasoning tasks require diversity to support pass@k and generalization. Leveraging the property that "$Q^{\pi_u}$ is proportional to success probability," the authors replace greedy with softmax sampling:

$$\pi_s(a|s)=\frac{\exp(Q^{\pi_u}(s,a)/\rho)}{\sum_{a'}\exp(Q^{\pi_u}(s,a')/\rho)}$$

This way, actions are selected proportional to their "estimated success probability," allowing for the exploration of multiple valid paths rather than sticking to one. Softmax naturally fits modern LLM decoding frameworks. Theorem 2 provides a performance guarantee: $V^{\pi_s}(s_0)\ge R\left(1-\sum_{s\in P}\Pr_{\pi_s}(s|s_0)\frac{N(s)}{N(s)+\exp(\max_a Q^{\pi_u}(s,a)/\rho)}\right)$, where $N(s)$ is the number of zero-valued actions and $P$ is the set of critical states containing both optimal and sub-optimal actions. The temperature $\rho$ acts as the quality-diversity knob: as $\rho\to0$, it reverts to greedy and the gap to the optimal policy vanishes; larger $\rho$ leads to more diverse sampling. In toy experiments, $\rho=1$ covers all 4 optimal modes while maintaining 100% success rate.

**3. LLM Intrinsic Q Parameterization: Mapping abstract Q directly to policy parameters to eliminate value networks and stabilize training**

The theory is elegant, but the state/action space of an LLM is enormous with long horizons. Training a Q-network from scratch is costly. The authors notice that Q-values and policies are intrinsically linked via $\rho\log\pi_\theta(a|s)$ (which characterizes the relative preference of actions in a state). Thus, **Ours directly represents the Q-function using the LLM's parameters $\theta$**, eliminating the need for a separate value network.

However, using $\rho\log\pi_\theta$ directly as a Q-learning target can cause drift and divergence. To address this, the authors introduce a **relative Q-function** that measures improvement relative to a fixed baseline: $Q(s_t, a_t)=\rho\big(\log\pi_\theta(a_t|s_t)-\log\pi_{\theta_{old}}(a_t|s_t)\big)$, where $\pi_{\theta_{old}}$ is the behavior policy used for sampling in each epoch, acting as a stable anchor. This centers initial Q-values at 0, forcing the model to learn "changes relative to the previous policy" rather than absolute values, suppressing fluctuations in Q-updates.

**4. Low-variance Rewards: Group centering + broadcasting to every token to make sparse rewards dense and stable**

Binary terminal reward signals are sparse and high-variance, making Q-estimation difficult. For each prompt, the authors sample $n$ responses and subtract the mean group reward to get centered rewards $\tilde{r}(x,y_i)=r(x,y_i)-\frac{1}{n}\sum_{i=1}^n r(x,y_i)$ (similar to GRPO's advantage estimation but **omitting the standard deviation normalization**). This reduces estimation variance and enriches sampling of the value landscape. To handle credit assignment for long reasoning chains, this centered reward is **broadcasted to every token in the sequence**, improving training efficiency. The final loss is an MSE regression of the parameterized Q toward the target $\hat{Q}$ (stop-gradient): $L_{\text{ROVER}}=\frac{1}{\sum_i|y_i|}\sum_i\sum_t\|Q(a_t|s_t),\text{sg}[\hat{Q}(a_t|s_t)]\|^2$.

### A Example: Toy Tree-like MDP
The authors designed a tabular environment to verify the theory: starting from an empty state, choosing an action from $\{A,B,C,D\}$ at each step, where only 4 specific sequences (ACD, BDC, CAB, DBA) have a reward of 1. The results (Figure 5) are telling: standard Q-learning ($\epsilon$-greedy) is optimal but converges to only one mode (ACD); ROVER-greedy assigns the highest Q-value to the optimal actions but collapses to one mode (BDC) due to greediness; meanwhile, ROVER ($\rho=1$) **assigns equally high Q-values to all 4 optimal actions**, successfully covering all modes with 100% success rate. This visualizes the trade-off between "greedy for optimality" and "softmax for diversity."

### Loss & Training
Each epoch: fix $\pi_{\theta_{old}}\leftarrow\pi_\theta$ and sample a batch of prompts; rollout $n$ responses per prompt to calculate centered rewards $\tilde{r}$; calculate $Q(a_{t+1}|s_{t+1})=\rho(\log\pi_\theta-\log\pi_{\theta_{old}})$ for each state and backfill the target $\hat{Q}(a_t|s_t)\leftarrow\tilde{r}+\frac{1}{|V|}\sum_{a_{t+1}\in V}Q(a_{t+1}|s_{t+1})$; finally optimize the MSE loss above with AdamW. There is no clipping, KL regularization, or value network throughout the process; temperature $\rho=1$ is kept uniform across all tasks.

## Key Experimental Results

### Main Results
Trained on the DeepScaler dataset using Qwen3-4B/8B-Base, pass@1 comparisons across math and O.O.D benchmarks (selection from Qwen3-8B-Base):

| Method | AIME24 | AIME25 | HMMT25 | OlympiadBench | AMC23 | MATH500 | GPQA-d | Avg. |
|------|--------|--------|--------|---------------|-------|---------|--------|------|
| Base | 11.5 | 8.8 | 0.8 | 34.7 | 48.1 | 68.8 | 29.1 | 28.8 |
| GRPO | 16.8 | 15.1 | 4.8 | 48.6 | 66.9 | 81.9 | 43.8 | 39.7 |
| DAPO | 20.8 | 15.2 | 3.6 | 49.0 | 67.9 | 84.3 | 46.6 | 41.1 |
| REINFORCE++ | 19.4 | 16.7 | 7.1 | 47.6 | 63.5 | 83.6 | 46.3 | 40.6 |
| **ROVER** | **30.6** | **22.7** | **14.6** | **56.4** | **74.8** | **89.6** | **50.2** | **48.4** |

ROVER consistently outperforms the strongest baselines across all model scales (8B average pass@1 is +7.3 higher than the second best); the gap increases with task difficulty—AIME24 +47.1% relative to the best baseline, AIME25 +35.9%, and HMMT25 nearly doubles. It also achieves the best performance on O.O.D GPQA-diamond, indicating strong generalization.

### Ablation Study

| Configuration | Observation | Explanation |
|------|------|------|
| ROVER ($\rho=1$) | Optimal quality+diversity Pareto frontier | Uniform temperature across all experiments |
| $\rho=3$ | Slow convergence, under-utilization | High temperature causes excessive exploration |
| $\rho=0.1$ | Early entropy collapse, limited exploration | Low temperature causes premature exploitation |
| $\rho=0.001$ | Near-deterministic sampling $\rightarrow$ unstable training | Extreme greediness, test score collapse |
| GRPO w/o KL | Entropy collapse | Representative failure mode of the baseline |
| GRPO w/ KL=0.01 | Significantly worse pass@1 | Excessive regularization becomes detrimental |

### Key Findings
- **High entropy is the primary driver of performance**: ROVER's entropy decays gracefully during training but remains significantly higher than baselines (which either collapse or oscillate violently). Sustained exploration is fundamental to achieving both quality and diversity.
- **Non-saturation of pass@k**: Baselines saturate quickly after improving pass@1, sometimes dropping below the base model at large $k$ (e.g., DAPO on AIME25 for $k>4$). ROVER continues to climb as $k$ increases, with pass@256 being +16.8 higher than the strongest baseline.
- **Highest Diversity**: Using the "distinct strategies" metric from NoveltyBench, ROVER is +6.8% over GRPO and +20.5% over the average of the three baselines. In Countdown tasks, it finds 17 different solutions for the same problem, compared to 3 for GRPO.
- **Superior Test-time Scaling**: maj@k rises robustly with $k$, whereas baselines converge confidently to similar incorrect answers due to mode collapse.

## Highlights & Insights
- **Problem structure as a first-class citizen**: Rather than adding more tricks to PPO, the authors take a step back to ask "what does this MDP look like?" and find that the tree+deterministic+binary structure allows the "useless" mean operator to be effective—a paradigm of algorithm simplification driven by structural insight.
- **Elegant probability interpretation of Q-values**: $Q^{\pi_u}(s,a)$ represents the probability of success given random completion. This provides a direct geometric/probabilistic intuition for "pruning dead ends" and justifies softmax sampling.
- **Reusing LLM parameters for intrinsic Q + relative baseline anchors**: Eliminating the value network while using $\pi_{\theta_{old}}$ as an anchor to center Q at 0 is a stable critic technique that could transfer to other RLVR settings.
- **Diversity as a "free byproduct" rather than reward engineering**: Compared to methods that rely on complex task-specific reward engineering or hard-coded post-hoc sampling for diversity, ROVER's diversity emerges naturally from softmax over $Q^{\pi_u}$, supported by performance guarantees.

## Limitations & Future Work
- **Theory depends on specialized structure**: The optimality proof strictly relies on deterministic transitions, tree structures, and binary terminal rewards. If rewards are continuous/intermediate or the state graph has cycles, Theorem 1 no longer holds.
- **"Probabilistic interpretation" of the mean operator depends on uniform random priors**: In reality, token spaces are massive. The $\frac{1}{|V|}\sum_{a'}$ average over vocabulary is an approximation, and its error accumulation over long horizons is not fully explored.
- **Sensitivity to temperature $\rho$**: Ablations show that $\rho$ deviating from 1 hurts performance (especially $\rho=0.001$). Whether $\rho=1$ remains robust across more diverse tasks/models needs verification.
- **Validated primarily on Math/Countdown**: While there are O.O.D results for GPQA-diamond, performance on broader verifiable reward tasks (code, theorem proving) requires further investigation.

## Related Work & Insights
- **vs GRPO/PPO/DAPO**: These follow GPI loops with non-stationary targets and rely on tricks (clip/KL/filtering) that lead to entropy collapse. ROVER evaluates a fixed policy once, avoiding non-stationarity and winning on both quality and diversity. 
- **vs Q-learning ($\epsilon$-greedy)**: Both achieve optimality on toy MDPs, but Q-learning collapses to a single mode due to greediness; ROVER uses softmax to cover all optimal modes while maintaining optimality.
- **vs Classical "Mean Operator Uselessness" (Asadi & Littman, 2017)**: Classical RL considers the mean operator unsuitable for general control. Ours provides the first **theoretical** proof for its optimality when paired with greedy selection in LLM reasoning, grounding previous empirical observations.
- **vs Diverse RL Methods**: Traditional methods use complex reward engineering or post-hoc sampling for diversity without guarantees; ROVER's diversity is a natural consequence of softmax over $Q^{\pi_u}$, with a performance lower bound per Theorem 2.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Uses "problem structure simplification" to reactivate the mean operator with the first optimality proof for LLM reasoning.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple models, benchmarks, pass@k, maj@k, diversity, and O.O.D, though the focus remains on math-like tasks.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear progression from theory to intuition to engineering; toy examples visualize core trade-offs effectively.
- Value: ⭐⭐⭐⭐⭐ Simple yet powerful; challenges the assumption that "RLVR must be complex," providing directional inspiration for post-training methodologies.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Reference-guided Policy Optimization for Molecular Optimization via LLM Reasoning](reference-guided_policy_optimization_for_molecular_optimization_via_llm_reasonin.md)
- [\[ICLR 2026\] Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning](slow-fast_policy_optimization_reposition-before-update_for_llm_reasoning.md)
- [\[ICLR 2026\] Temperature as a Meta-Policy: Adaptive Temperature in LLM Reinforcement Learning](temperature_as_a_meta-policy_adaptive_temperature_in_llm_reinforcement_learning.md)
- [\[ICLR 2026\] Asymmetric Proximal Policy Optimization: Mini-Critics Boost LLM Reasoning](asymmetric_proximal_policy_optimization_mini-critics_boost_llm_reasoning.md)
- [\[ICLR 2026\] On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning](on_the_design_of_kl-regularized_policy_gradient_algorithms_for_llm_reasoning.md)

</div>

<!-- RELATED:END -->
