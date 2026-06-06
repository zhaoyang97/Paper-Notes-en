---
title: >-
  [Paper Note] Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm: Demystifying Some Myths About GRPO and Its Friends
description: >-
  [ICLR 2026][LLM Alignment][GRPO] By constructing a KL-regularized surrogate objective and deriving a pairwise consistency condition from first principles…
tags:
  - "ICLR 2026"
  - "LLM Alignment"
  - "GRPO"
  - "off-policy RL"
  - "importance sampling"
  - "clipping"
  - "REINFORCE"
  - "policy optimization"
date: 2026-05-08
content_hash: b3a29039e65b29c0
---

# Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm: Demystifying Some Myths About GRPO and Its Friends

**Conference**: ICLR 2026
**arXiv**: [2509.24203](https://arxiv.org/abs/2509.24203)  
**Code**: [Trinity-RFT](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/rec_gsm8k)  
**Area**: LLM Alignment / RL
**Keywords**: GRPO, off-policy RL, importance sampling, clipping, REINFORCE, policy optimization

## TL;DR

By constructing a KL-regularized surrogate objective and deriving a pairwise consistency condition from first principles, this paper proves that group-relative REINFORCE (GRPO) is inherently an off-policy algorithm. Component isolation experiments further reveal that clipping is the sole driver of training stability while importance sampling can be entirely removed. Within this unified framework, the paper reinterprets several seemingly independent algorithms—including Kimi OPMD and Meta AsymRE—under a common theoretical lens.

## Background & Motivation

**Background**: GRPO and its variants (DAPO, GiGPO) have become the dominant algorithms for LLM RL training. DeepSeek-R1 achieved breakthrough results in reasoning using GRPO, the Kimi team proposed OPMD, and Meta proposed AsymRE—each offering distinct theoretical justifications, yet the intrinsic relationships among these methods remain unclear.

**Limitations of Prior Work**: The success of GRPO has been attributed to multiple factors—group-relative advantage for variance reduction, importance sampling (IS) for distribution shift correction, and clipping for training stability—yet the true contribution of each component has never been systematically isolated and verified. More critically, GRPO is theoretically regarded as an on-policy algorithm (requiring samples from the current policy for unbiased gradient estimation), but in engineering practice it almost always operates on off-policy data (due to mismatches between rollout generation and model training speed, stale policy data, and delayed reward feedback). This theory–practice gap lacks rigorous explanation.

**Key Challenge**: Classical policy gradient theory requires training data to come from the current policy $\pi_\theta$; off-policy correction relies on importance sampling weights $\pi_\theta(y|x)/\pi_b(y|x)$, which grow exponentially with sequence length in the LLM setting. Existing practice substitutes response-level ratios with token-level ratios, introducing bias without formal guarantees.

**Goal**: (1) Provide a theoretical derivation of GRPO that does not depend on any assumption about the sampling distribution; (2) systematically isolate the contributions of IS and clipping; (3) explain the intrinsic connections among GRPO, OPMD, and AsymRE within a unified framework.

**Key Insight**: The authors observe that starting from a KL-regularized surrogate objective, deriving the pairwise consistency condition satisfied by its optimal solution, constructing a mean-squared surrogate loss that enforces this condition, and then taking a single gradient step at the current parameters yields exactly the GRPO update formula—without ever specifying the distribution from which training data originates.

**Core Idea**: GRPO is an off-policy algorithm; clipping is the sole critical component for stability while IS is largely superfluous; two augmentation principles—regularized policy updates and active shaping of the data distribution—suffice to unify and improve a broad family of RL algorithms.

## Method

### Overall Architecture

The theoretical framework proceeds in three steps. First, a KL-regularized surrogate objective anchored at the previous policy $\pi_{\theta_t}$ is defined as $J(\theta; \pi_{\theta_t}) = \mathbb{E}[r(x,y)] - \tau \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\theta_t})$, and the pairwise consistency condition satisfied by its optimal policy is derived. Second, a mean-squared surrogate loss that enforces this condition using finite samples is constructed. Third, it is shown that taking a single gradient step of this loss at $\theta_t$ is equivalent to the group-relative REINFORCE update—without any assumption on the distribution of training data, thereby naturally supporting off-policy use.

Building on this off-policy interpretation, the authors propose two augmentation principles for handling arbitrary data distributions: (1) regularize the policy update step (e.g., clipping, KL penalty) to prevent catastrophic updates under suboptimal data distributions; (2) actively shape the training data distribution (e.g., sample weighting, discarding low-reward samples) to guide the direction of policy updates. These two principles jointly explain GRPO, OPMD, AsymRE, and various data-weighting heuristics under a single framework.

### Key Designs

1. **Three-Step Derivation: From Surrogate Objective to REINFORCE**

    - **Function**: Prove that group-relative REINFORCE admits a natural off-policy interpretation.
    - **Mechanism**: The optimal solution of the KL-regularized surrogate objective satisfies $\pi^*(y|x) \propto \pi_{\theta_t}(y|x) \exp(r(x,y)/\tau)$, from which the pairwise consistency condition for any two responses $y_1, y_2$ follows: $r_1 - \tau(\log\pi(y_1|x) - \log\pi_{\theta_t}(y_1|x)) = r_2 - \tau(\log\pi(y_2|x) - \log\pi_{\theta_t}(y_2|x))$. A mean-squared loss $\hat{L} = \frac{1}{K^2}\sum_{i<j}(a_i - a_j)^2$ is constructed to enforce this condition; taking its gradient at $\theta = \theta_t$ causes the log-probability difference terms to vanish, yielding $\frac{1}{K}\sum_i (r_i - \bar{r}) \nabla_\theta \log\pi_\theta(y_i|x)$—exactly the GRPO update.
    - **Design Motivation**: Classical policy gradient theory requires on-policy sampling, limiting the theoretical justification of GRPO in asynchronous training pipelines. This derivation bypasses the sampling distribution assumption entirely, grounding off-policy usage in first-principles optimality conditions.

2. **REC Variants: Isolating the Contributions of IS and Clipping**

    - **Function**: Precisely identify the contribution of each GRPO component to training stability.
    - **Mechanism**: A family of REINFORCE-with-Clipping (REC) ablations is designed. REC-OneSide-IS retains IS weights and one-sided clipping but removes advantage normalization; REC-OneSide-NoIS further removes IS weights, retaining only the clipping mask $M_i^t = \mathbb{1}(A_i > 0, \rho_i^t \leq 1+\epsilon_{\text{high}}) + \mathbb{1}(A_i < 0, \rho_i^t \geq 1-\epsilon_{\text{low}})$. Experiments also test expanding the clipping range from the standard $(0.2, 0.2)$ to the aggressive $(0.6, 2.0)$.
    - **Design Motivation**: The community widely assumes IS is central to off-policy correction; experiments show that removing IS leaves performance virtually unchanged (reward curves overlap completely), whereas removing clipping causes immediate training collapse. This demonstrates that clipping functions as an implicit trust-region constraint—bounding the magnitude of each policy update to prevent divergence under limited sample coverage.

3. **Unified Interpretation of OPMD and AsymRE**

    - **Function**: Reveal that three seemingly independent algorithms share the same underlying structure.
    - **Mechanism**: The Kimi OPMD loss decomposes into REINFORCE loss plus a mean-squared regularizer $\frac{\beta}{2K}\sum_i(\log\pi_\theta(y_i|x) - \log\pi_{\text{old}}(y_i|x))^2$ with $\beta = \tau$. The Meta AsymRE baseline shift $\bar{r} - \beta$ is equivalent to REINFORCE loss plus a KL regularizer $\frac{\beta}{K}\sum_i \log\frac{\pi_{\text{old}}(y_i|x)}{\pi_\theta(y_i|x)}$, which converges to $\beta \cdot D_{\text{KL}}(\pi_{\text{old}} \| \pi_\theta)$ in the large-sample limit. Both are instantiations of the "regularized policy update" principle, differing only in the form of regularization.
    - **Design Motivation**: The OPMD paper derives from a pointwise consistency condition of the KL-regularized objective (partially overlapping with this work before diverging at step 2), while the AsymRE paper justifies its baseline shift via a multi-armed bandit analysis. The unified view presented here is cleaner: both reduce to REINFORCE plus some regularization, corresponding to the first augmentation principle.

### Data Weighting Methods (RED Variants)

The uniform weights in the pairwise surrogate loss are generalized to $\sum_{i<j} w_{i,j}(a_i - a_j)^2$, from which a weighted REINFORCE update is derived. Two methods are proposed:

- **RED-Drop**: Discards a subset of low-reward negative samples, training only on $\mathcal{S} \subseteq [K]$. The motivation is that negative gradients increase the risk of entropy collapse (consistent with guidance from the Kimi-Researcher blog), and this approach is theoretically justified within the off-policy framework.
- **RED-Weight**: Applies reward-correlated weights $w_i$ to each sample's gradient term. This decomposes into pairwise-weighted REINFORCE plus a regularization term that imitates high-reward responses, echoing findings in the offline RL literature that regularizing toward high-reward trajectories is more effective than conservatively imitating all trajectories.

### Loss & Training

The core loss is the standard REINFORCE loss $-\frac{1}{K}\sum_i(r_i - \bar{r})\log\pi_\theta(y_i|x)$ plus optional regularization (clipping mask / KL penalty / mean-squared regularizer), with different combinations corresponding to different algorithms. Training uses the Trinity-RFT framework, with `sync_interval` (model synchronization frequency) and `sync_offset` (delay between rollout and training) as explicit controls for the degree of off-policy-ness.

## Key Experimental Results

### Main Results: IS vs. Clipping Ablation (GSM8k, Qwen2.5-1.5B-Instruct)

| Algorithm | Clipping Range | IS | On-Policy Reward | Mixed Reward | Offline Reward |
|---|---|---|---|---|---|
| GRPO | (0.2, 0.2) | ✓ | Converges normally | Converges normally | Converges normally |
| REC-OneSide-IS | (0.2, 0.2) | ✓ | ≈ GRPO | ≈ GRPO | ≈ GRPO |
| REC-OneSide-NoIS | (0.2, 0.2) | ✗ | ≈ GRPO | ≈ GRPO | ≈ GRPO |
| REC-OneSide-NoIS | (0.6, 2.0) | ✗ | **Faster convergence** | **Faster convergence** | Speed↑ but unstable |
| REINFORCE (no clipping) | — | ✗ | **Training collapse** | **Training collapse** | **Training collapse** |

Core conclusion: Removing IS yields completely overlapping reward curves across all three settings, demonstrating that IS is unnecessary. Removing clipping causes immediate collapse, confirming that clipping is the sole indispensable component. Expanding the clipping range accelerates convergence in on-policy and mixed settings, but introduces a speed–stability trade-off in the purely offline setting.

### Ablation Study

| Experimental Setting | Task / Model | Key Findings |
|---|---|---|
| REC variants | ToolACE / Llama-3.2-3B | IS unnecessary; clipping remains the stability key; conclusions consistent across models and tasks |
| RED-Drop | GSM8k / Qwen2.5-1.5B | Discarding low-reward samples is effective in both on- and off-policy settings, performing comparably to REC with expanded range |
| RED-Weight | Guru-Math / Qwen2.5-7B | Weighted method outperforms GRPO on large-scale tasks with comparable KL divergence; positive scaling effect |
| RED-Weight | MATH / Llama-3.1-8B | Cross-model validation; gains persist on harder mathematical tasks |
| OPMD reproduction | GSM8k / Qwen2.5-1.5B | Mean-squared regularization and clipping are complementary, but clipping alone is sufficient |
| AsymRE reproduction | GSM8k / Qwen2.5-1.5B | Baseline shift (KL regularization) is effective but less robust than clipping |
| Offline stress test | GSM8k | Training on offline data sampled solely from the initial policy exposes the stability limits of expanding the clipping range |

### Key Findings

- **IS can be entirely removed**: Across all tested models (1.5B/3B/7B/8B), tasks (GSM8k/MATH/ToolACE/Guru-Math), and degrees of off-policy-ness (on-policy/mixed/offline), removing IS produces no significant performance change. This allows engineering implementations to eliminate the overhead of storing and computing old-policy log probabilities.
- **Clipping is the only indispensable component**: It functions as an implicit trust-region constraint, bounding the ratio $\pi_\theta/\pi_{\text{old}}$. Without it, policy updates become uncontrollable in direction under finite sample coverage.
- **Asymmetric clipping range expansion accelerates training**: Allowing a larger growth of the policy ratio for positive-advantage samples ($\epsilon_{\text{high}} = 2.0$) while moderately relaxing the lower bound for negative-advantage samples ($\epsilon_{\text{low}} = 0.6$) intuitively encourages reinforcing good behaviors while permitting forgetting of bad ones.
- **3-arm bandit counterexample**: Vanilla REINFORCE with behavioral policy $\pi_b = [0.3, 0.6, 0.1]$ and rewards $r = [0, 0.8, 1]$ converges to the suboptimal action $a_2$ rather than the optimal $a_3$, because $\pi_b(a_2)(r(a_2) - \mu_r) > \pi_b(a_3)(r(a_3) - \mu_r)$—demonstrating that off-policy REINFORCE without regularization or data shaping will inevitably fail.

## Highlights & Insights

- **Elegance of the theoretical derivation**: The three-step derivation (surrogate objective → consistency condition → mean-squared loss → single-step gradient = REINFORCE) is logically transparent, with clear physical intuition at each stage. In particular, the fact that log-probability difference terms naturally vanish when the gradient is evaluated at $\theta_t$ is not a coincidental algebraic artifact but a reflection of deep underlying structure.
- **Counterintuitive finding that IS is superfluous**: IS is conventionally regarded as foundational infrastructure for off-policy RL. In the LLM fine-tuning context, however, policy changes are typically small, token-level IS ratios are themselves biased, and the resulting IS weights remain close to 1—making their corrective effect negligible. The true stabilizer is the implicit trust region induced by clipping, a finding that directly simplifies engineering implementations.
- **Explanatory power of the unified framework**: Casting GRPO (clipping regularization), OPMD (mean-squared regularization), and AsymRE (KL regularization) as instances of REINFORCE with different regularization forms connects three independent research threads into a coherent narrative. RED-Drop and RED-Weight, corresponding to the second principle (data distribution shaping), complete the theoretical picture.

## Limitations & Future Work

- **Absence of convergence guarantees**: The off-policy interpretation provides theoretical justification but does not establish formal guarantees of policy improvement or convergence; future work should derive such guarantees under specific assumptions on the data distribution.
- **Unresolved trade-off in the purely offline setting**: Expanding the clipping range can lead to instability in offline settings; the authors identify this as an open problem, potentially requiring adaptive clipping strategies that dynamically adjust the range based on training progress or the degree of off-policy-ness.
- **Single-turn vs. multi-turn RL**: The primary analysis targets one-step RL (single-turn prompt–response); the multi-step generalization is treated in the appendix but lacks experimental validation. The transferability of the conclusions to agentic RL requiring multi-turn environment interaction remains to be verified.
- **Validation limited to mathematical reasoning and tool use**: The reward signals in GSM8k, MATH, and ToolACE are explicit correctness signals; whether the relative contributions of components remain consistent for tasks with more ambiguous rewards (e.g., dialogue quality, creative writing) is unknown.
- **Insufficient analysis of group size $K$**: The pairwise consistency condition is grounded in finite sample pairs; how the choice of $K$ affects the approximation quality of the surrogate loss relative to the true objective, and whether the variance of the group-relative baseline under small $K$ interacts with the degree of off-policy-ness, remain open questions.

## Related Work & Insights

- **vs. PPO**: PPO uses clipping to constrain on-policy update step sizes; this paper demonstrates that the same clipping mechanism serves as a critical stabilizer in the off-policy setting. The key difference is that GRPO's clipping operates on group-relative advantages rather than raw ratios, and GRPO requires no value function.
- **vs. DPO**: DPO is a purely offline preference optimization method (derived from the Bradley–Terry model), whereas off-policy REINFORCE retains online learning capability while tolerating stale data. The two approaches are complementary—DPO for cold-start initialization, GRPO for continual learning.
- **vs. DAPO**: DAPO augments GRPO with token-level entropy bonuses and dynamic sampling for exploration. The analysis presented here provides a theoretical foundation for DAPO's empirical success: DAPO's improvements can all be subsumed under the two principles of regularized policy updates and data distribution shaping.
- **vs. REBEL/CoPG**: These works share the KL-regularized objective and pairwise consistency condition with this paper (steps 1–2 overlap), but they opt for direct minimization of the surrogate loss via multi-step gradient descent. The contribution of this paper lies in discovering that a single gradient step recovers REINFORCE—thereby bridging theory and practice.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Derives an off-policy interpretation of GRPO from first principles; component isolation experiments fundamentally overturn the consensus that IS is central.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers 5 models, 4 tasks, and 3 degrees of off-policy-ness with comprehensive ablations; additional validation in agentic and conversational settings would strengthen the case.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The theoretical derivation is exceptionally clear, with a progressively structured three-step argument; the presentation of the unified framework is textbook quality.
- **Value**: ⭐⭐⭐⭐⭐ — Offers fundamental guidance for the LLM RL community; "remove IS and expand the clipping range" is an immediately actionable engineering optimization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] GVPO: Group Variance Policy Optimization for Large Language Model Post-Training](../../NeurIPS2025/llm_alignment/gvpo_group_variance_policy_optimization_for_large_language_model_post-training.md)
- [\[ICLR 2026\] Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning](slow-fast_policy_optimization_reposition-before-update_for_llm_reasoning.md)
- [\[ICLR 2026\] Learning More with Less: A Dynamic Dual-Level Down-Sampling Framework for Efficient Policy Optimization](learning_more_with_less_a_dynamic_dual-level_down-sampling_framework_for_efficie.md)
- [\[ICLR 2026\] Mitigating the Safety Alignment Tax with Null-Space Constrained Policy Optimization](mitigating_the_safety_alignment_tax_with_null-space_constrained_policy_optimizat.md)
- [\[ICLR 2026\] Hierarchy-of-Groups Policy Optimization for Long-Horizon Agentic Tasks](hierarchy-of-groups_policy_optimization_for_long-horizon_agentic_tasks.md)

</div>

<!-- RELATED:END -->
