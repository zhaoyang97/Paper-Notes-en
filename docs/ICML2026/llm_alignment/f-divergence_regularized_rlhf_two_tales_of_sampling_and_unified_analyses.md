---
title: >-
  [Paper Note] $f$-Divergence Regularized RLHF: Two Tales of Sampling and Unified Analyses
description: >-
  [ICML 2026][LLM Alignment][$f$-divergence] This paper establishes the first $O(\log T)$ regret and $O(1/T)$ suboptimality gap upper bounds for online RLHF under **general $f$-divergence regularization**. It proposes two…
tags:
  - "ICML 2026"
  - "LLM Alignment"
  - "$f$-divergence"
  - "optimism"
  - "derivative-as-uncertainty"
  - "regret bound"
  - "contextual bandit"
date: 2026-05-08
content_hash: 177da40ad1317ddb
---

# $f$-Divergence Regularized RLHF: Two Tales of Sampling and Unified Analyses

**Conference**: ICML 2026  
**arXiv**: [2605.06977](https://arxiv.org/abs/2605.06977)  
**Code**: None (Theoretical Paper)  
**Area**: RLHF Alignment / Online Learning / Theory  
**Keywords**: $f$-divergence, optimism, derivative-as-uncertainty, regret bound, contextual bandit

## TL;DR
This paper establishes the first $O(\log T)$ regret and $O(1/T)$ suboptimality gap upper bounds for online RLHF under **general $f$-divergence regularization**. It proposes two sampling strategies: (1) optimism in the face of uncertainty with an added bonus term; (2) a novel **"derivative-as-uncertainty"** perspective—using $f'$ as an uncertainty signal to design derivative-based sampling without the need for explicit confidence bound estimation in each round.

## Background & Motivation

**Background**: RLHF has become a standard for LLM post-training (e.g., InstructGPT, Llama2, Claude), most commonly formulated as a KL-regularized contextual bandit: $J_{\text{KL}}(\pi)=\mathbb{E}[r^*(x,a)-\eta^{-1}D_{\text{KL}}(\pi,\pi_0)]$. Zhao et al. 2025a proved that online KL-RLHF achieves $O(\log T)$ regret, while offline settings under single-policy coverage achieve $O(\varepsilon^{-1})$ sample complexity.

**Limitations of Prior Work**: KL is not a universal regularizer—Huang et al. 2025 demonstrated that mixed chi-squared better mitigates reward over-optimization; Shan et al. 2024 noted that forward KL is more stable for diffusion model alignment; $\alpha$-divergence provides a more flexible trade-off between exploration and exploitation. However, **current theoretical analyses are performed case-by-case for specific $f$**, lacking a unified framework. While Zhao et al. 2025b provided a general $f$-divergence analysis, it was restricted to the offline setting. Unified theory for the online setting remains a gap.

**Key Challenge**: Each $f$-divergence has its own closed-form optimal policy $\pi_f^*(a|x)=\pi_0(a|x)f'^{-1}(\eta(r^*(x,a)-\lambda_f^*(x)))$, where the shapes of $f'^{-1}$ (denoted as $h$) vary significantly—KL is exponential, chi-squared is linear, and JS lies in between. **The regret of any online algorithm is dominated by the curvature of $h$**, making the design of a universal bonus for all $f$ a major difficulty.

**Goal**: (1) Extend optimism-based RLHF (Xiong 2023, Ye 2024, Zhao 2025a) from KL to general $f$; (2) Provide an alternative algorithm that **does not require explicit confidence balls**, as solving optimization problems for confidence balls in every round is impractical for real-world LLMs; (3) Provide unified regret/suboptimality proofs for both algorithms across general $f$.

**Key Insight**: The authors make a critical observation: **the derivative $h'$ of $h=(f')^{-1}$ intrinsically indicates how much reward estimation error will be amplified**. Specifically, $\pi_\theta-\pi_{\theta'}\approx \pi_0\cdot h'(\eta(r_\theta-\lambda))\cdot\eta\cdot\Delta r$. Thus, areas where $h'$ is large correspond to high sensitivity of $\pi$ to reward estimation, signaling a need for more exploration. This provides a new perspective that translates the "geometrical properties of $f$-divergence" directly into an "exploration signal."

**Core Idea**: Using the derivative of $f'$ itself as an uncertainty measure, a sampling policy $\pi'_\theta(a|x)\propto \pi_0(a|x)\cdot h'(\eta(r_\theta-\lambda))$ is designed. This is complemented by two distributions $\pi_\theta^\pm$ to provide coverage when $h'$ approaches 0, achieving unified $O(\log T)$/$O(1/T)$ guarantees for general $f$.

## Method

### Overall Architecture
Both algorithms are based on the Bradley-Terry preference model and the general objective $J_f(\pi)=\mathbb{E}[r^*(x,a)-\eta^{-1}D_f(\pi,\pi_0|x)]$. In each round $T$:

1. Sample two actions $a_t^1, a_t^2$;
2. Receive preference $y_t$;
3. Estimate reward $r_{\theta_t}$ using MLE (maximizing sigmoid likelihood);
4. Construct a new policy $\pi_{t+1}$ based on $r_{\theta_t}$.

The difference between the two algorithms lies in Step 1 (sampling) and Step 4 (policy construction).

### Key Designs

1. **Closed-form Optimal Policy + General Conditions** (Proposition 2.3):
    - **Function**: Expresses the optimal solution for a general $f$-divergence objective in explicit form, serving as the foundation for both algorithms.
    - **Mechanism**: Under the conditions that $\pi_0(a|x)>0$, $f'$ is invertible, and $0\notin\text{dom}(f')$, the optimal policy is $\pi_f^*(a|x)=\pi_0(a|x)\cdot f'^{-1}(\eta(r^*(x,a)-\lambda_f^*(x)))$, where $\lambda_f^*(x)$ is the normalizing Lagrange multiplier. For reverse KL, $f'^{-1}(z)=\exp(z-1)$, recovering the familiar softmax form.
    - **Design Motivation**: The closed-form solution allows for direct analysis of $\partial J_f/\partial r$ and expresses regret as a quadratic form of reward error. The invertibility condition excludes boundary cases like Total Variation and pure chi-squared but retains mainstream choices like reverse/forward KL, JS, and chi-squared-KL.

2. **Optimism Algorithm (Algorithm 1)**:
    - **Function**: Uses the classic "optimism in the face of uncertainty" to achieve $O(\log T)$ regret for general $f$.
    - **Mechanism**: In each round, MLE yields $\theta_t$, and an optimistic reward is constructed as $\hat r_t(\cdot,\cdot)=r_{\theta_t}+\mathbb{E}_{a\sim\pi_t}b_t$, where the bonus is $b_t(x,a^1,a^2)=\min\{1,\beta_T U(\xi,x,a^1,a^2;\mathcal{R}_t,\mathcal{D}_t)\}$. $U$ is an uncertainty measure based on Eluder dimension. The new $\pi_{t+1}$ is then derived via Proposition 2.3 using $\hat r_t$.
    - **Design Motivation**: It adapts the optimism framework, but the regret bound includes an additional term $\mathcal{C}(f,\mathcal{R}_\Theta,\eta)=\max h'/h$. This represents the cost introduced by $f$, quantifying that "$f$ with flatter $h$ results in tighter regret." This is the first bound provided for general $f$.

3. **Derivative-as-uncertainty Algorithm (Algorithm 2)**:
    - **Function**: Avoids solving for explicit confidence balls in each round by using the geometry of $h'$ to drive exploration.
    - **Mechanism**: Define a sampling distribution $\pi'_\theta(a|x)\propto\pi_0(a|x)\cdot h'(\eta(r_\theta(x,a)-\lambda_\theta(x)))$, where actions with larger $h'$ are sampled more frequently (due to higher policy sensitivity). Since $h'$ might approach zero when reward estimates are severely wrong, two complementary distributions $\pi_\theta^+\propto\pi'_\theta\exp(r_\theta)$ and $\pi_\theta^-\propto\pi'_\theta\exp(-r_\theta)$ are added to cover over-estimated and under-estimated rewards. In each round, $(a^1,a^2)$ are sampled from $\pi'_\theta$ with probability $1-p(x)$, or one each from $(\pi^+,\pi^-)$ with probability $p(x)=\frac{Z^+Z^-}{1+Z^+Z^-}$.
    - **Design Motivation**: Optimism-based algorithms require solving $\sup_{R_1,R_2}$ to calculate $U$, which is impractical for LLMs with massive parameter spaces. The derivative method embeds "exploration intensity" into the known function $h'$, requiring only MLE and weighted sampling, making it implementation-friendly. Theoretically, it achieves an $O(1/T)$ suboptimality gap.

### Loss & Training
Algorithm 1 uses standard BT-MLE:
$\theta_t=\arg\max_\theta\sum_i\big(y_i\log\sigma(r_\theta(x,a_i^1)-r_\theta(x,a_i^2))+(1-y_i)\log\sigma(r_\theta(x,a_i^2)-r_\theta(x,a_i^1))\big)$.

Algorithm 2 uses weighted BT-MLE:
$\mathcal{L}(\theta)=-\frac{1}{t}\sum_i\omega(x_i)\log\sigma(r_\theta(x_i,a_i^\omega)-r_\theta(x_i,a_i^l))$, where $\omega(x)=(\overline T_\theta(x)+Z^+Z^-\overline T_\theta(x))/\overline Z_\theta$ is an importance weight to correct bias from mixed sampling. $\overline T_\theta(x)=\sum_a\pi_0(a|x)h'(\eta(r_\theta-\lambda_\theta))$.

## Key Experimental Results

As this is a **purely theoretical paper**, the main results are theoretical bounds:

### Main Results

| Algorithm | Setting | Regret / SubOpt | Applicable $f$ | Notes |
|------|------|-----------------|----------|------|
| Algorithm 1 (optimism) | online RLHF | $O(\eta\,\mathcal{C}(f,\mathcal{R},\eta)\log(N_\mathcal{R}T/\delta)\,d(\mathcal{R},\xi,T))$ | Any $f$ with invertible $f'$ and $0\notin\text{dom}(f')$ | $d$ is Eluder dim; $O(\log T)$ under linear reward |
| Algorithm 2 (derivative) | online RLHF | $\text{SubOpt}=O(1/T)$ | Same as above | No confidence ball needed |
| Zhao 2025a (KL only) | online KL-RLHF | $O(\log T)$ | Reverse KL only | This paper recovers this bound |
| Zhao 2025b | offline general $f$ | $O(\varepsilon^{-1})$ | General $f$ | Offline only |

### Comparison of Constants

$\mathcal{C}(f,\mathcal{R},\eta)=\max_{r,x,a}\frac{h'(\eta(r-\lambda))}{h(\eta(r-\lambda))}$:

| $f$ | $h(z)=(f')^{-1}(z)$ | Dominant Term in $\mathcal{C}$ | Explanation |
|------|-----|-------|------|
| reverse KL | $\exp(z-1)$ | $\mathcal{C}=1$ | Simplest; matches Zhao 2025a |
| forward KL | $-1/z$ (restricted) | Related to $r$ range | OOD Robust |
| JS | $\log(2x/(1+x))^{-1}$ | Moderate | Moderates KL |
| chi-squared-KL | $z+2(x-1)$ | Related to $\eta$ | Mitigates reward over-opt |

### Key Findings
- **General $f$ does not increase the regret order**: All $f$ satisfying the conditions achieve $O(\log T)$, with differences only in the constant $\mathcal{C}(f)$. This suggests the community can swap $f$ according to empirical needs without fear of theoretical regret explosion.
- **Derivative-as-uncertainty is a new perspective**: Previous RLHF theories handled reward estimation error and policy uncertainty separately. This paper proves $h'$ alone can bridge the two. This insight may inspire future RLHF algorithm designs (including DPO, IPO).
- **The three-distribution sampling design is precise**: $\pi'$ follows the derivative signal while $\pi^\pm$ follow reward extremes. This complementary coverage of "high sensitivity but known reward" and "low sensitivity but unknown reward" ensures the estimation error closes to $O(1/T)$ after weighted MLE.

## Highlights & Insights
- The intuition that **"$f'$ acts as an uncertainty signal"** is the most significant insight of this work. It links the "curvature of divergence" directly to "exploration necessity," elegantly translating geometric properties into algorithms.
- **The engineering value of Algorithm 2 is noteworthy**: Optimism-style algorithms are nearly impossible for LLMs due to the cost of calculating the supremum over reward classes. The derivative method embeds exploration into the known function $h'$, requiring only MLE and weighted sampling, which is developer-friendly.
- **Clarity of the unified framework**: Through Proposition 2.3, Lemma C.6 (writing regret as quadratic reward error), and Eluder dimension, the authors compress the complexity of "general $f$" into a single constant $\mathcal{C}(f,\mathcal{R},\eta)$, resulting in a very clean proof structure.

## Limitations & Future Work
- The assumption that $f'$ is invertible and $0\notin\text{dom}(f')$ **excludes Total Variation and pure chi-squared**—ironically two of the most popular in over-optimization literature. These are discussed in Appendix B but lack full bounds.
- The work is confined to the **contextual bandit framework**, leaving multi-turn RL/CoT settings unaddressed. Modern RLHF (e.g., o1, DeepSeek-R1) increasingly uses multi-turn/process rewards, necessitating theoretical extensions.
- There are no empirical experiments to verify if the derivative algorithm is truly more efficient than optimism on real LLMs; pure theoretical results may have limited immediate impact on practitioners.
- The constant $\mathcal{C}(f,\mathcal{R},\eta)$ is not compared with specific values for different $f$, so it doesn't directly tell users which $f$ is most "cost-effective" for a specific task.

## Related Work & Insights
- **vs Zhao 2025a (KL-only online RLHF)**: This work is a strict generalization; KL is a special case where $\mathcal{C}=1$, and the bound form is perfectly recovered.
- **vs Zhao 2025b (offline general $f$)**: Complementary—they handle the offline case while this paper covers the online case, together closing the theoretical loop for $f$-RLHF.
- **vs Huang 2025 (chi-squared regularization)**: Huang empirically proved chi-squared mitigates over-optimization; this paper provides the first theoretical guarantee.
- **vs Wang 2023 / Sun 2024 ($f$-DPO empirical papers)**: They modified divergences in DPO without theoretical proofs. Although this paper focuses on RLHF rather than DPO, the analysis framework can be adapted (as DPO's optimal policy also follows Proposition 2.3).

## Rating
- Novelty: ⭐⭐⭐⭐ (Derivative-as-uncertainty is a genuine new perspective; optimism is an extension of KL)
- Experimental Thoroughness: ⭐⭐ (Zero experiments, purely theoretical)
- Writing Quality: ⭐⭐⭐⭐ (Theorem structures and proof sketches are clear and detailed)
- Value: ⭐⭐⭐⭐ (Provides the first online theoretical guarantee for $f$-RLHF; Algorithm 2 has engineering potential)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Greedy Sampling Is Provably Efficient for RLHF](../../NeurIPS2025/llm_alignment/greedy_sampling_is_provably_efficient_for_rlhf.md)
- [\[ICML 2026\] UDM-GRPO: Stable and Efficient GRPO for Unified Discrete Diffusion Models](udm-grpo_stable_and_efficient_group_relative_policy_optimization_for_uniform_dis.md)
- [\[ICML 2026\] Efficient Preference Poisoning Attack on Offline RLHF](efficient_preference_poisoning_attack_on_offline_rlhf.md)
- [\[ICML 2026\] Mitigating Reward Hacking in RLHF via Bayesian Non-negative Reward Modeling](mitigating_reward_hacking_in_rlhf_via_bayesian_non-negative_reward_modeling.md)
- [\[ICLR 2026\] Uni-DPO: A Unified Paradigm for Dynamic Preference Optimization of LLMs](../../ICLR2026/llm_alignment/uni-dpo_a_unified_paradigm_for_dynamic_preference_optimization_of_llms.md)

</div>

<!-- RELATED:END -->
