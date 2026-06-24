---
title: >-
  [Paper Note] Can RLHF be More Efficient with Imperfect Reward Models? A Policy Coverage Perspective
description: >-
  [ICML 2025][LLM Alignment][transfer learning] This work identifies a structural property induced by KL regularization in RLHF—the policy coverage over the optimal policy is bounded by its sub-optimality ($\text{Cov}^{\pi^*|\pi} \leq 1 + \kappa \cdot (J(\pi^*) - J(\pi))/\beta$). Based on this, two transfer learning principles are proposed: (1) selecting a transfer policy with high policy value, and (2) self-transfer distilling the policy from online data. The proposed TPO algo…
tags:
  - "ICML 2025"
  - "LLM Alignment"
  - "transfer learning"
  - "KL regularization"
  - "policy coverage"
  - "online RLHF"
  - "sample efficiency"
  - "DPO"
  - "IPO"
  - "XPO"
  - "win rate"
date: 2026-05-08
content_hash: a3e8bd9fde589197
---

# Can RLHF be More Efficient with Imperfect Reward Models? A Policy Coverage Perspective

**Conference**: ICML 2025  
**arXiv**: [2502.19255](https://arxiv.org/abs/2502.19255)  
**Code**: [https://github.com/jiaweihhuang/RLHF_RewardTransfer](https://github.com/jiaweihhuang/RLHF_RewardTransfer)  
**Area**: Alignment RLHF  
**Keywords**: transfer learning, KL regularization, policy coverage, online RLHF, sample efficiency, DPO, IPO, XPO, win rate

## TL;DR

This work identifies a structural property induced by KL regularization in RLHF—the policy coverage over the optimal policy is bounded by its sub-optimality ($\text{Cov}^{\pi^*|\pi} \leq 1 + \kappa \cdot (J(\pi^*) - J(\pi))/\beta$). Based on this, two transfer learning principles are proposed: (1) selecting a transfer policy with high policy value, and (2) self-transfer distilling the policy from online data. The proposed TPO algorithm achieves a regret of $O(W\sqrt{T})$ in the early stage and $O(\sqrt{T})$ in the late stage. It can be modularly integrated with DPO/IPO/XPO, and its effectiveness is validated on the T5 summarization task.

## Background & Motivation

### Sample Efficiency Bottleneck of Online RLHF

Online RLHF requires collecting a large amount of human preference labels to align LLMs, which is highly expensive. Existing methods primarily improve sample efficiency through exploration strategies (such as optimistic exploration in XPO) but overlook a key opportunity: leveraging existing imperfect but relevant reward models to accelerate learning.

### Sources of Imperfect Reward Models

In practical scenarios, multiple types of imperfect rewards are available:

**Cross-lingual reward transfer**: A reward model trained on one language can be applied to another.

**LLM-as-judge**: Evaluations from GPT/LLaMA/Gemini are partially aligned with human preferences.

**Heuristic rewards**: Rule-based metrics such as ROUGE and BERTScore can be obtained at extremely low cost.

### Core Problem

Given $W$ imperfect reward models $\{r_w\}_{w=1}^W$ (with quality unknown a priori), how can they be leveraged to reduce human annotation efforts and learn a near-optimal policy? The key challenges lie in how to select the transfer policy and how to avoid being misled by low-quality sources.

## Method

### Overall Architecture

**Problem Setup**: Contextual bandit (prompt space $\mathcal{S}$, response space $\mathcal{A}$), optimizing the KL-regularized objective:

$$\pi_r^* \leftarrow \arg\max_\pi J_\beta(\pi; r) = \mathbb{E}_{s \sim \rho, a \sim \pi}[r(s,a)] - \beta \cdot \text{KL}(\pi \| \pi_{\text{ref}})$$

Closed-form solution: $\pi_r^*(a|s) \propto \pi_{\text{ref}}(a|s) \cdot e^{r(s,a)/\beta}$

**Transfer Setup**: Given $W$ source rewards $\{r_w\}$ and their corresponding optimal policies $\{\pi_{r_w}^*\}$, the policy value gap is defined as $\Delta(w) := J_\beta(\pi_{r^*}^*) - J_\beta(\pi_{r_w}^*)$, and $\Delta_{\min} := \min_w \Delta(w)$.

### Key Designs

#### Key Finding: Coverage Structure Induced by KL Regularization

**Lemma 3.1 (Core Result of This Work)**:

$$\text{Cov}^{\pi_{r^*}^* | \pi} \leq 1 + \kappa(e^{2R/\beta}) \cdot \frac{J_\beta(\pi_{r^*}^*) - J_\beta(\pi)}{\beta}$$

where $\kappa(x) = \frac{(x-1)^2}{x - 1 - \log x} = O(x)$.

**Core Insight**: Under the regularized setting ($\beta > 0$), the sub-optimality of a policy directly controls its coverage over the optimal policy. This is unique to regularization—in pure reward maximization, the coverage between two deterministic policies with a sub-optimality gap of $2\varepsilon$ can be infinite.

Reason: Regularization rules out near-deterministic policies, leveraging the prior knowledge of $\pi_{\text{ref}}$ to ensure that the remaining policies possess a well-behaved structure.

#### Principle 1: Selecting a Transfer Policy with High Policy Value

According to Lemma 3.1, high policy value $\implies$ small sub-optimality $\implies$ good coverage for $\pi^*$. Therefore, utilizing high-value policies does not conflict with exploration—regularization reconciles the exploration-exploitation dilemma. This is unachievable in pure reward maximization (near-optimality does not imply good coverage).

#### Principle 2: Self-Transfer Learning

**Theorem 3.2**: The sub-optimality of the policy $\pi_{\text{Dstl}}$ distilled using offline learning (RPO) from online collected data satisfies:

$$J_\beta(\pi^*) - J_\beta(\pi_{\text{Dstl}}) \leq \tilde{O}\left(e^{2R}\left(1 + \kappa(e^{2R/\beta}) \cdot \frac{\sum_t (J_\beta(\pi^*) - J_\beta(\pi_t))}{\beta T}\right) \cdot \frac{1}{\sqrt{T}}\right)$$

When the online algorithm is no-regret, $\pi_{\text{Dstl}}$ converges to $\pi^*$ at a rate of $O(T^{-1/2})$—independent of the size of the state/action spaces or the complexity of the policy class. This is faster than the $O(\sqrt{\mathcal{C}(\Pi)/T})$ rate of standard online RLHF, where $\mathcal{C}(\Pi)$ can be very large.

Additional benefit of self-transfer: $\pi_{\text{Dstl}}$ continuously improves to approach $\pi^*$, whereas source policies remain stuck with a fixed non-zero gap, preventing the algorithm from being constrained by sub-optimal sources.

### Loss & Training

**TPO Algorithm (Alg. 1)**:
- Divide $T$ steps into $K = T/N$ blocks, each of size $N$.
- The first $\alpha N$ steps of each block: Run the online algorithm AlgOL (e.g., XPO).
- The remaining $(1-\alpha)N$ steps: Run Transfer Policy Selection (TPS, Alg. 2).
- Finally return the policy distilled using all collected data.

**Transfer Policy Selection (Alg. 2)**:
1. Perform UCB-style optimistic value estimation for each source policy using the MLE reward estimator $\hat{r}_{\text{MLE}}$.
2. Perform pessimistic value estimation (lower bound) for the self-transfer policy using the intrinsic structure of the RPO objective.
3. Select the policy with the highest estimated value.

**Empirical TPO (Alg. 3)**—Practical Version:
- Replace policy value estimation with win rate (significantly reducing computational cost).
- The lower bound of the win rate can control coverage (Lemma 5.1).
- Use a UCB strategy for bandit selection among multiple sources.
- Modularly integrate any policy optimization method (DPO/IPO/XPO) as AlgPO.
- Compare $\hat{\text{WR}}_{\pi_{r_w}^*}$ vs $\hat{\text{WR}}_{\pi_{\text{OL}}} = 0.55$, enabling transfer only when a source is significantly better than the current online policy.

## Key Experimental Results

### Main Results

**XSum Summarization + T5-small (80M)**:

Source reward models: (a) ROUGE-Lsum, (b) BERTScore, (c) T5-base (250M), (d) T5-large (770M)

True reward: sfairXC/FsfairX-LLaMA3-RM-v0.1 (distilled from Llama3-8B)

**Win Rate (%) of TPO compared to three baselines when DPO is used as AlgPO:**

| Iteration | vs No Transfer (Iter-DPO) | vs Pure ROUGE (Worst Source) | vs Pure T5-Large (Best Source) |
|-----------|---------------------------|------------------------------|-------------------------------|
| Iter 1 | >50% (Significant advantage) | >50% (Significant advantage) | ≈50% (Comparable) |
| Iter 2 | >50% | >50% | ≈50% |
| Iter 3 | >50% | >50% | >50% (Outperform) |

Key observation: By Iter 3, TPO even outperforms the baseline that purely uses the best source (T5-Large)—because TPO automatically switches back to online learning, avoiding limitation by the source's sub-optimality.

### Ablation Study

**Source Task Selection Process Analysis (Figure 2)**:

- Iter 1: UCB exploration phase, budget distributed evenly across sources.
- Iter 2: T5-Large is identified as the best source, receiving the most budget.
- Iter 3: $\pi_{\text{OL}}$ surpasses all sources, automatically switching to the no-transfer mode.

This precisely validates the theoretically predicted two-stage behavior.

**Validation of Different AlgPOs**: Similar improvement patterns are observed when IPO and XPO are used as AlgPO, demonstrating that the modular design of TPO is indeed generalizable.

### Key Findings

1. **KL regularization is a "blessing" for transfer learning**: It establishes a structural relationship between coverage and sub-optimality, making simple policy-value-based selection sufficient.
2. **Automatic outperformance mechanism of self-transfer**: The distilled policy continuously improves and eventually surpasses fixed, imperfect sources.
3. **Win rate is an effective proxy for coverage**: Although it acts as a lower bound, it is sufficient to guide transfer decisions.
4. **Practical utility of the 0.55 threshold**: Setting a baseline win rate slightly higher than 0.5 prevents low-quality transfer.

## Highlights & Insights

1. **Deep theoretical insights**: Lemma 3.1 reveals the structural "blessing" of KL regularization—it is not only a tool to prevent overfitting but also creates a geometric structure beneficial for transfer learning.
2. **Elegant transition from theory to practice**: Shifting from precise but computationally costly policy value estimation to efficient, theoretically-backed win rate estimation.
3. **Novel concept of self-transfer**: Using offline distilled policies from online data as transfer candidates—a paradigm where the "future self" assists the "present self."
4. **Collateral improvements to standard online RLHF**: Even when $W=0$ (no sources), relying purely on self-transfer achieves $O(\sqrt{T})$ regret—eliminating the dependence on $\mathcal{C}(\Pi)$ and strictly improving existing results (Corollary 4.5).
5. **Modular design**: The transfer module of TPO can be combined with any policy optimization method, greatly enhancing its practicality.

## Limitations & Future Work

1. **Limited experimental scale**: Validated only on T5-small (80M) without testing on larger LLMs.
2. **Small number of reward models**: Only 4 source rewards were used; the selection efficiency when $W$ is very large remains untested.
3. **Bradley-Terry assumption**: The theory relies on the BT preference model, which real human preferences might not adhere to.
4. **Contextual bandit setting**: Not extended to multi-turn dialogue (MDP), although the authors indicate potential scalability.
5. **Computational overhead**: Best-of-N (N=32) sampling can be costly on large models.
6. **Gap between theoretical TPO and practical TPO**: The theoretical algorithm requires solving a minimax optimization, whereas the practical version directly uses DPO/IPO loss; this gap has not been quantitatively analyzed.
7. **Unexplored state-level transfer**: Current work focuses on policy-level transfer; finer-grained prompt-wise transfer could be more effective.

## Related Work & Insights

- **Exploration in Online RL/RLHF**: XPO (Xie et al. 2024) and Self-Exploring LM (Zhang et al. 2024) focus on exploration strategies, while this work is orthogonal—focusing on utilizing existing reward models.
- **RLAIF vs. This Work**: RLAIF aims to replace human feedback with AI feedback as the ultimate goal, whereas this work uses AI rewards to accelerate learning from human rewards.
- **Iterative DPO/IPO** (Xiong et al. 2024): Serves as the foundation for the online learning component of this paper.
- **Policy Coverage** (Xie et al. 2022): $L^\infty$ coverability was originally used as a complexity measure in online RL, which this paper links with KL regularization to yield new discoveries.
- **RPO** (Liu et al. 2024): A minimax algorithm for offline RLHF, used here for self-transfer distillation.
- **Insights**: The structural properties of KL regularization may also uncover similar beneficial properties in emerging paradigms like Nash Learning from Human Feedback.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (The coverage-suboptimality connection, self-transfer concept, and transition from theory to practice are highly novel)
- Experimental Thoroughness: ⭐⭐⭐ (Tested only on T5-small, with a limited number of sources)
- Writing Quality: ⭐⭐⭐⭐⭐ (The theoretical exposition is highly clear, linking properties, principles, algorithms, and practical implementations systematically)
- Value: ⭐⭐⭐⭐⭐ (Solid theoretical contributions, highly generalizable practical algorithms, and collateral improvements to standard online RLHF results)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Reward Generalization in RLHF: A Topological Perspective](../../ACL2025/llm_alignment/reward_generalization_in_rlhf_a_topological_perspective.md)
- [\[ACL 2025\] Towards Reward Fairness in RLHF: From a Resource Allocation Perspective](../../ACL2025/llm_alignment/reward_fairness_rlhf.md)
- [\[NeurIPS 2025\] Provably Efficient Online RLHF with One-Pass Reward Modeling](../../NeurIPS2025/llm_alignment/provably_efficient_online_rlhf_with_one-pass_reward_modeling.md)
- [\[NeurIPS 2025\] Greedy Sampling Is Provably Efficient for RLHF](../../NeurIPS2025/llm_alignment/greedy_sampling_is_provably_efficient_for_rlhf.md)
- [\[ICLR 2026\] Learning More with Less: A Dynamic Dual-Level Down-Sampling Framework for Efficient Policy Optimization](../../ICLR2026/llm_alignment/learning_more_with_less_a_dynamic_dual-level_down-sampling_framework_for_efficie.md)

</div>

<!-- RELATED:END -->
