---
title: >-
  [Paper Note] $f$-Divergence Regularized RLHF: Two Tales of Sampling and Unified Analyses
description: >-
  [ICML 2026][LLM Alignment][$f$-divergence] This paper establishes, for the first time, $O(\log T)$ regret and $O(1/T)$ suboptimality gap upper bounds for online RLHF under **general $f$-divergence regularization**. Two s…
tags:
  - "ICML 2026"
  - "LLM Alignment"
  - "$f$-divergence"
  - "optimism"
  - "derivative-as-uncertainty"
  - "regret bound"
  - "contextual bandit"
date: 2026-05-08
content_hash: 9c79614589623fb7
---

# $f$-Divergence Regularized RLHF: Two Tales of Sampling and Unified Analyses

**Conference**: ICML 2026  
**arXiv**: [2605.06977](https://arxiv.org/abs/2605.06977)  
**Code**: None (theoretical paper)  
**Area**: RLHF Alignment / Online Learning / Theory  
**Keywords**: $f$-divergence, optimism, derivative-as-uncertainty, regret bound, contextual bandit

## TL;DR
This paper establishes, for the first time, $O(\log T)$ regret and $O(1/T)$ suboptimality gap upper bounds for online RLHF under **general $f$-divergence regularization**. Two sampling strategies are proposed: (1) an optimism-in-face-of-uncertainty approach with a bonus term; (2) a novel **"derivative-as-uncertainty"** perspective—treating $f'$ as an uncertainty signal, enabling derivative-based sampling without explicit confidence bound estimation each round.

## Background & Motivation

**Background**: RLHF has become standard for LLM post-training (e.g., InstructGPT, Llama2, Claude), most commonly as KL-regularized contextual bandit: $J_{\text{KL}}(\pi)=\mathbb{E}[r^*(x,a)-\eta^{-1}D_{\text{KL}}(\pi,\pi_0)]$. Zhao et al. 2025a proved that online KL-RLHF achieves $O(\log T)$ regret, and offline achieves $O(\varepsilon^{-1})$ sample complexity under single-policy coverage.

**Limitations of Prior Work**: KL is not a universal regularizer—Huang et al. 2025 showed that mixed chi-squared regularization better mitigates reward over-optimization; Shan et al. 2024 noted that forward KL is more stable for diffusion model alignment; $\alpha$-divergence offers a more flexible exploration-exploitation trade-off. However, **all existing theoretical analyses are tailored to specific $f$**, lacking a unified framework; Zhao et al. 2025b provided a general $f$-divergence theory but only for offline. Unified online theory remains absent.

**Key Challenge**: Each $f$-divergence yields its own closed-form optimal policy $\pi_f^*(a|x)=\pi_0(a|x)f'^{-1}(\eta(r^*(x,a)-\lambda_f^*(x)))$, where $f'^{-1}$ (denoted $h$) varies greatly—KL is exp, chi-squared is linear, JS is intermediate. **Any online algorithm's regret is dominated by the curvature of $h$**; designing a bonus that works for all $f$ is challenging.

**Goal**: (1) Extend optimism-based RLHF (Xiong 2023, Ye 2024, Zhao 2025a) from KL to general $f$; (2) Provide an alternative algorithm **without explicit confidence ball**, as confidence balls require solving an optimization problem each round, which is impractical for real LLMs; (3) Provide unified regret/suboptimality proofs for both algorithms under general $f$.

**Key Insight**: The authors observe that **the derivative $h'=(f')^{-1}'$ directly indicates "how much reward estimation error is amplified"**. That is, $\pi_\theta-\pi_{\theta'}\approx \pi_0\cdot h'(\eta(r_\theta-\lambda))\cdot\eta\cdot\Delta r$, so where $h'$ is large, the policy is sensitive to reward estimation—thus, more exploration is needed. This translates the "geometric property of $f$-divergence" directly into an "exploration signal".

**Core Idea**: Use the derivative of $f'$ itself as an uncertainty measure, designing $\pi'_\theta(a|x)\propto \pi_0(a|x)\cdot h'(\eta(r_\theta-\lambda))$ as the sampling policy, with two complementary distributions $\pi_\theta^\pm$ as fallback when $h'$ is near zero, achieving $O(\log T)$/$O(1/T)$ guarantees for general $f$.

## Method

### Overall Architecture
Both algorithms are based on the Bradley-Terry preference model and the general objective $J_f(\pi)=\mathbb{E}[r^*(x,a)-\eta^{-1}D_f(\pi,\pi_0|x)]$. Each round $t$:

1. Sample two actions $a_t^1,a_t^2$;
2. Receive preference $y_t$;
3. Estimate reward $r_{\theta_t}$ via MLE (maximize sigmoid likelihood);
4. Construct new policy $\pi_{t+1}$ based on $r_{\theta_t}$.

The two algorithms differ only in step 1 (sampling) and step 4 (policy construction).

### Key Designs

1. **Closed-form Optimal Policy + General Conditions** (Proposition 2.3):

    - **Function**: Expresses the optimal solution for general $f$-divergence objectives in explicit form, forming the basis for both algorithms.
    - **Mechanism**: Under $\pi_0(a|x)>0$ and invertible $f'$ with $0\notin\text{dom}(f')$, $\pi_f^*(a|x)=\pi_0(a|x)\cdot f'^{-1}(\eta(r^*(x,a)-\lambda_f^*(x)))$, where $\lambda_f^*(x)$ is a normalization Lagrange multiplier. For reverse KL, $f'^{-1}(z)=\exp(z-1)$, recovering the familiar softmax.
    - **Design Motivation**: The closed-form allows direct analysis of $\partial J_f/\partial r$ and expressing regret as a quadratic form in reward error; the invertibility condition excludes Total Variation and chi-squared at the boundary, but retains reverse/forward KL, JS, chi-squared-KL, etc.

2. **Optimism Algorithm (Algorithm 1)**:

    - **Function**: Applies the classic "optimism in face of uncertainty" principle to general $f$, achieving $O(\log T)$ regret.
    - **Mechanism**: Each round, perform MLE to obtain $\theta_t$, construct optimistic reward $\hat r_t(\cdot,\cdot)=r_{\theta_t}+\mathbb{E}_{a\sim\pi_t}b_t$, where the bonus $b_t(x,a^1,a^2)=\min\{1,\beta_T U(\xi,x,a^1,a^2;\mathcal{R}_t,\mathcal{D}_t)\}$, and $U$ is an uncertainty measure based on Eluder dimension. Then use $\hat r_t$ and Proposition 2.3 to obtain the new $\pi_{t+1}$.
    - **Design Motivation**: Directly applies the optimism framework, but the regret bound includes an extra $\mathcal{C}(f,\mathcal{R}_\Theta,\eta)=\max h'/h$ term—this is the cost introduced by $f$, quantifying "the flatter $h$ is, the tighter the regret". This is the first such bound for general $f$.

3. **Derivative-as-uncertainty Algorithm (Algorithm 2)**:

    - **Function**: Avoids explicit optimization for confidence balls each round, using the geometry of $h'$ to drive exploration.
    - **Mechanism**: Defines the sampling distribution $\pi'_\theta(a|x)\propto\pi_0(a|x)\cdot h'(\eta(r_\theta(x,a)-\lambda_\theta(x)))$—actions with large $h'$ are sampled more (as the policy is more sensitive to their reward). However, when reward estimation is severely wrong, $h'$ may approach zero, causing exploration to stall; thus, two complementary distributions are added: $\pi_\theta^+\propto\pi'_\theta\exp(r_\theta)$ and $\pi_\theta^-\propto\pi'_\theta\exp(-r_\theta)$, covering cases of reward over- and under-estimation. Each round, with probability $1-p(x)$, sample $(a^1,a^2)$ from $\pi'_\theta$; with $p(x)$, sample one from each of $(\pi^+,\pi^-)$, where $p(x)=\frac{Z^+Z^-}{1+Z^+Z^-}$ is an adaptive mixing weight.
    - **Design Motivation**: The optimism algorithm requires solving $\sup_{R_1,R_2}$ to compute $U$ each round, which is impractical for large-parameter LLMs; the derivative method embeds "exploration intensity" into the known function $h'$, requiring only MLE and weighted sampling, making it engineering-friendly. Theoretically achieves $O(1/T)$ suboptimality gap.

### Loss & Training

Algorithm 1 uses standard BT-MLE:
$\theta_t=\arg\max_\theta\sum_i\big(y_i\log\sigma(r_\theta(x,a_i^1)-r_\theta(x,a_i^2))+(1-y_i)\log\sigma(r_\theta(x,a_i^2)-r_\theta(x,a_i^1))\big)$.

Algorithm 2 uses weighted BT-MLE:
$\mathcal{L}(\theta)=-\frac{1}{t}\sum_i\omega(x_i)\log\sigma(r_\theta(x_i,a_i^\omega)-r_\theta(x_i,a_i^l))$, where $\omega(x)=(\overline T_\theta(x)+Z^+Z^-\overline T_\theta(x))/\overline Z_\theta$ is an importance weight correcting for bias from mixed sampling. $\overline T_\theta(x)=\sum_a\pi_0(a|x)h'(\eta(r_\theta-\lambda_\theta))$.

## Key Experimental Results

This is a **pure theory paper**; the main table presents theoretical bounds:

### Main Results

| Algorithm | Setting | Regret / SubOpt | Applicable $f$ | Notes |
|------|------|-----------------|----------|------|
| Algorithm 1 (optimism) | online RLHF | $O(\eta\,\mathcal{C}(f,\mathcal{R},\eta)\log(N_\mathcal{R}T/\delta)\,d(\mathcal{R},\xi,T))$ | Any invertible $f'$ with $0\notin\text{dom}(f')$ | $d$ is Eluder dim, $O(\log T)$ for linear reward |
| Algorithm 2 (derivative) | online RLHF | $\text{SubOpt}=O(1/T)$ | Same as above | No confidence ball needed |
| Zhao 2025a (KL only) | online KL-RLHF | $O(\log T)$ | Reverse KL only | This paper recovers their bound |
| Zhao 2025b | offline general $f$ | $O(\varepsilon^{-1})$ | General $f$ | Offline only |

### Key Constants Comparison

$\mathcal{C}(f,\mathcal{R},\eta)=\max_{r,x,a}\frac{h'(\eta(r-\lambda))}{h(\eta(r-\lambda))}$:

| $f$ | $h(z)=(f')^{-1}(z)$ | Dominant $\mathcal{C}$ term | Notes |
|------|-----|-------|------|
| reverse KL | $\exp(z-1)$ | $\mathcal{C}=1$ | Simplest, matches Zhao 2025a |
| forward KL | $-1/z$ (restricted domain) | Depends on $r$ range | OOD robust |
| JS | $\log(2x/(1+x))^{-1}$ | Moderate | Softens KL |
| chi-squared-KL | $z+2(x-1)$ | Depends on $\eta$ | Mitigates reward over-opt |

### Key Findings
- **General $f$ does not increase regret order**: All $f$ satisfying the conditions achieve $O(\log T)$; differences are only in the constant $\mathcal{C}(f)$, indicating practitioners can safely choose $f$ based on empirical needs without fear of theoretical regret blowup.
- **Derivative-as-uncertainty is a new perspective**: Previous RLHF theory treated reward estimation error and policy uncertainty separately; this work shows $h'$ alone bridges both. This insight may inspire future RLHF algorithms (even DPO, IPO).
- **The three sampling distributions are cleverly designed**: $\pi'$ follows the derivative signal, $\pi^\pm$ target reward extremes, complementarily covering "high sensitivity but known reward" and "low sensitivity but unknown reward" regions, enabling the MLE-weighted estimation error to close at $O(1/T)$ in the proof.

## Highlights & Insights
- **"$f'$ as an uncertainty signal"** is the most memorable insight—directly linking "divergence curvature" to "exploration need", translating geometric properties into algorithms with surprising simplicity.
- **Algorithm 2's engineering value is significant**: Optimism-type algorithms are nearly infeasible for large models (sup over reward class each round is too costly), while the derivative method only requires computing the known $h'$ function and weighted sampling, making it likely to become a practical RLHF training trick.
- **Clarity of the unified framework**: Through Proposition 2.3 + Lemma C.6 (regret as quadratic reward error) + Eluder dimension, the authors compress the complexity of "general $f$" into a single constant $\mathcal{C}(f,\mathcal{R},\eta)$, resulting in a clean proof structure.

## Limitations & Future Work
- The assumption that $f'$ is invertible and $0\notin\text{dom}(f')$ **excludes Total Variation and pure chi-squared**—precisely those favored in over-optimization literature; these are discussed in Appendix B but without complete bounds.
- Only the contextual bandit framework is considered; **multi-round RL/CoT settings are not addressed**—yet modern RLHF (e.g., o1, DeepSeek-R1) increasingly involve multi-turn/process rewards, requiring theoretical extension.
- No empirical experiments verify whether the derivative algorithm is indeed more efficient than optimism on real LLMs; the purely theoretical results may limit immediate appeal to practitioners.
- The constant $\mathcal{C}(f,\mathcal{R},\eta)$ is not numerically compared across different $f$, so users cannot directly determine "which $f$ is most cost-effective for their task".

## Related Work & Insights
- **vs Zhao 2025a (KL-only online RLHF)**: This work is a strict generalization; KL is the special case with $\mathcal{C}=1$, and the bound form is fully recovered.
- **vs Zhao 2025b (offline general $f$)**: Complementary—Zhao 2025b covers offline, this work covers online, together forming a theoretical closure for $f$-RLHF.
- **vs Huang 2025 (chi-squared regularization)**: Huang empirically showed chi-squared mitigates over-optimization; this work provides the first theoretical guarantee, reassuring the community.
- **vs Wang 2023 / Sun 2024 ($f$-DPO empirical papers)**: They modified DPO's divergence but lacked theory; while this work is on RLHF, not DPO, the analysis framework can be adapted to DPO (DPO's optimal policy also fits Proposition 2.3).

## Rating
- Novelty: ⭐⭐⭐⭐ Derivative-as-uncertainty is a genuinely new perspective; optimism part is an extension from KL
- Experimental Thoroughness: ⭐⭐ No experiments, pure theory; not a flaw but limits immediate impact
- Writing Quality: ⭐⭐⭐⭐ Theorem-proof structure is clear, proof sketches are detailed
- Value: ⭐⭐⭐⭐ Provides the first online theoretical guarantee for $f$-RLHF, and Algorithm 2 has engineering potential

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Greedy Sampling Is Provably Efficient for RLHF](../../NeurIPS2025/llm_alignment/greedy_sampling_is_provably_efficient_for_rlhf.md)
- [\[ICLR 2026\] Uni-DPO: A Unified Paradigm for Dynamic Preference Optimization of LLMs](../../ICLR2026/llm_alignment/uni-dpo_a_unified_paradigm_for_dynamic_preference_optimization_of_llms.md)
- [\[NeurIPS 2025\] LLM Safety Alignment is Divergence Estimation in Disguise](../../NeurIPS2025/llm_alignment/llm_safety_alignment_is_divergence_estimation_in_disguise.md)
- [\[ICLR 2026\] General Exploratory Bonus for Optimistic Exploration in RLHF](../../ICLR2026/llm_alignment/general_exploratory_bonus_for_optimistic_exploration_in_rlhf.md)
- [\[AAAI 2026\] On the Exponential Convergence for Offline RLHF with Pairwise Comparisons](../../AAAI2026/llm_alignment/on_the_exponential_convergence_for_offline_rlhf_with_pairwise_comparisons.md)

</div>

<!-- RELATED:END -->
