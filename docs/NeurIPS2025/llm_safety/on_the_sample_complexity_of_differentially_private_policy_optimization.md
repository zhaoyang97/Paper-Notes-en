---
title: >-
  [Paper Note] On the Sample Complexity of Differentially Private Policy Optimization
description: >-
  [NeurIPS 2025][LLM Safety][Differential Privacy] This paper presents the first systematic study of sample complexity for policy optimization (PO) under differential privacy (DP) constraints. It proposes a unified meta-al…
tags:
  - "NeurIPS 2025"
  - "LLM Safety"
  - "Differential Privacy"
  - "Policy Optimization"
  - "Sample Complexity"
  - "Reinforcement Learning"
  - "Privacy Protection"
date: 2026-05-08
content_hash: a655fb36e1b92612
---

# On the Sample Complexity of Differentially Private Policy Optimization

**Conference**: NeurIPS 2025
**arXiv**: [2510.21060](https://arxiv.org/abs/2510.21060)  
**Authors**: Yi He (Wayne State University), Xingyu Zhou (Wayne State University)
**Code**: Not available  
**Area**: AI Safety
**Keywords**: Differential Privacy, Policy Optimization, Sample Complexity, Reinforcement Learning, Privacy Protection

## TL;DR

This paper presents the first systematic study of sample complexity for policy optimization (PO) under differential privacy (DP) constraints. It proposes a unified meta-algorithm framework and analyzes three private policy optimization algorithms—DP-PG, DP-NPG, and DP-REBEL—proving that the privacy cost typically appears only as a lower-order term in the sample complexity.

## Background & Motivation

### State of the Field
Policy optimization (PO) methods such as REINFORCE, PPO, and GRPO are widely used in robotic control, healthcare, and LLM training. However, privacy concerns become increasingly prominent when PO is deployed in sensitive domains:
- **Healthcare**: Patient interaction data (medical states → prescriptions → outcomes) constitutes sensitive information.
- **LLM Training**: User prompts may contain private information.
- Empirical studies have demonstrated that standard GRPO is susceptible to privacy leakage.

### Limitations of Prior Work
- Existing work on private policy gradients is primarily empirical and lacks theoretical sample complexity guarantees.
- Online regret-based analyses (e.g., private PPO) are limited to tabular or linear settings.
- Standard DP definitions cannot be directly applied to PO for two reasons: (1) PO involves no fixed dataset due to on-policy sampling; (2) changing a single sample alters the policy, which in turn affects all subsequent samples.
- Private policy evaluation only assesses a given policy rather than searching for an optimal one.

### Root Cause
The core question motivating this work is: **How much additional sample complexity does differential privacy impose on policy optimization?** Addressing this requires defining an appropriate DP notion for PO, designing a unified algorithmic framework, and establishing rigorous sample complexity bounds.

## Method

### Problem Formulation
The paper adopts a bandit formulation (with standard extension to MDPs): given initial state $x \sim \rho$, action $y \sim \pi_\theta(\cdot|x)$, and reward $r(x,y) \in [-R_{\max}, R_{\max}]$, the objective is to maximize:
$$J(\theta) = \mathbb{E}_{x \sim \rho, y \sim \pi_\theta(\cdot|x)}[r(x,y)]$$

### DP Definition for PO (Definition 2)
A key contribution is treating the *user* as the unit of privacy. Dataset $D$ consists of $N$ users (e.g., $N$ patients or $N$ prompts), and privacy protection requires that replacing any single user does not significantly alter the output policy:
$$\mathbb{P}[\mathcal{M}(D) \in S] \leq e^\varepsilon \cdot \mathbb{P}[\mathcal{M}(D') \in S] + \delta$$
where $D$ and $D'$ differ in exactly one user.

### Meta-Algorithm (Algorithm 1)
A batched one-pass meta-algorithm is designed as follows:
1. At each round $t$, collect $m$ fresh samples $\bar{D}_t = \{(x_i, y_i, y_i')\}_{i=1}^m$.
2. Compute advantage estimates $\hat{A}_t(x_i, y_i) = r(x_i, y_i) - r(x_i, y_i')$.
3. Invoke the PrivUpdate oracle to update the policy $\theta_{t+1}$.
4. The total sample size is $N = m \cdot T$; the key is balancing per-round accuracy ($m$) against the number of iterations ($T$).

By the one-pass design and DP parallel composition, the overall algorithm satisfies $(\varepsilon, \delta)$-DP for PO as long as PrivUpdate satisfies $(\varepsilon, \delta)$-DP.

### DP-PG: Differentially Private Policy Gradient
- Gaussian noise with variance $\sigma^2 = \frac{16\log(1.25/\delta) R_{\max}^2 G^2}{m^2 \varepsilon^2}$ is added to REINFORCE-style gradient estimates.
- **FOSP convergence** (Theorem 2): $\mathbb{E}[\|\nabla J(\theta_U)\|^2] \leq O_\delta\left(\frac{1}{\sqrt{N}} + \left(\frac{\sqrt{d}}{N\varepsilon}\right)^{2/3}\right)$
- **Global optimality convergence** (Theorem 3): Under Fisher non-degeneracy and compatible function approximation assumptions, the sample complexity is $N \geq O_\delta\left(\frac{1}{\alpha^4 \gamma^4} + \frac{\sqrt{d}}{\alpha^3 \gamma^3 \varepsilon}\right)$.
- **Softmax tabular setting** (Theorem 5): Log-barrier regularization eliminates dependence on $\gamma$, yielding $N \geq O\left(\frac{1}{\alpha^6} + \frac{\sqrt{d}}{\alpha^{9/2} \varepsilon}\right)$.

### DP-NPG: Differentially Private Natural Policy Gradient
- PO is reduced to a series of private regression problems via a private least-squares oracle.
- **Main theorem** (Theorem 4): For any comparator policy $\pi^*$, $J(\pi^*) - \frac{1}{T}\sum_{t=1}^T J(\pi_t) \leq \sqrt{\frac{\beta W^2 \log|\mathcal{Y}|}{2T}} + \frac{\sqrt{C_{\mu \to \pi^*}}}{T}\sum_{t=1}^T \text{err}_t$
- **General function class** (Corollary 1, exponential mechanism): Pure DP ($\delta=0$), $N = \widetilde{O}\left(\left(\frac{1}{\alpha^4} + \frac{1}{\alpha^4\varepsilon}\right) \cdot \log|\mathcal{W}|\right)$.
- **Log-linear low-dimensional** (Corollary 2): $N = \widetilde{O}_\delta\left(\frac{d}{\alpha^4} + \frac{d}{\alpha^3\varepsilon} + \frac{d}{\alpha^2\varepsilon^2}\right)$.
- **Log-linear high-dimensional** (Corollary 3): $N = \widetilde{O}_\delta\left(\frac{1}{\alpha^6} + \frac{1}{\alpha^5\varepsilon}\right)$.

### DP-REBEL: Differentially Private REBEL
The structure mirrors DP-NPG, reducing PO to regression over relative reward differences (Theorem 6), achieving nearly identical sample complexity bounds to DP-NPG.

## Key Experimental Results

### Experiment 1: CartPole-v1 Under Different Privacy Budgets

Evaluated on CartPole-v1 using a two-layer fully connected network (64 hidden units), trained for 100 epochs with batch size 10, $\delta=10^{-5}$, averaged over 3 random seeds.

| Algorithm | $\varepsilon$ | Mean Final Reward | Std Dev | Best Epoch Mean |
|------|:---:|:---:|:---:|:---:|
| PG | N/A | 334.37 | 25.25 | 361.70 |
| DP-PG | 5 | 190.34 | 52.91 | 199.17 |
| DP-PG | 3 | 143.87 | 22.88 | 187.17 |
| NPG | N/A | 492.90 | 10.04 | 500.00 |
| DP-NPG | 5 | 478.73 | 28.05 | 494.70 |
| DP-NPG | 3 | 400.87 | 65.37 | 410.43 |

Key findings: (1) NPG substantially outperforms PG in both private and non-private settings; (2) DP-NPG at $\varepsilon=5$ nearly achieves optimal performance (~500), corroborating the theoretical prediction that privacy cost is a lower-order term; (3) performance degrades with tighter privacy budgets, consistent with theoretical expectations.

### Experiment 2: Summary of Theoretical Sample Complexity Results

| Algorithm | Setting | Non-Private Term | Privacy Cost Term |
|------|------|:---:|:---:|
| DP-PG | FOSP | $O(1/\alpha^4)$ | $O_\delta(\sqrt{d}/(\alpha^3\varepsilon))$ |
| DP-PG | Global (Fisher non-deg.) | $O(1/(\alpha^4\gamma^4))$ | $O_\delta(\sqrt{d}/(\alpha^3\gamma^3\varepsilon))$ |
| DP-PG | Global (Softmax tabular) | $O(1/\alpha^6)$ | $O(\sqrt{d}/(\alpha^{9/2}\varepsilon))$ |
| DP-NPG | General function class (pure DP) | $O(1/\alpha^4)$ | $O(1/(\alpha^4\varepsilon))$ |
| DP-NPG | Log-linear low-dim | $O(d/\alpha^4)$ | $O_\delta(d/(\alpha^3\varepsilon))$ |
| DP-NPG | Log-linear high-dim | $O(1/\alpha^6)$ | $O_\delta(1/(\alpha^5\varepsilon))$ |
| DP-REBEL | All settings | Same as DP-NPG | Same as DP-NPG |

Core conclusion: Across all settings, the privacy cost manifests as an **additive lower-order term** in the sample complexity (when $\varepsilon$ and $d$ are treated as constants).

## Highlights & Insights

- **Principled DP definition for PO**: The paper creatively adapts the "user" concept from the online bandit literature to resolve the subtle question of defining the privacy unit under on-policy learning dynamics, and clarifies the essential distinction between privacy protection in SFT and RL fine-tuning.
- **Unified meta-algorithm framework**: A single batched one-pass framework encompasses DP-PG, DP-NPG, and DP-REBEL, with privacy guarantees obtained naturally via parallel composition.
- **Quantification of privacy cost**: Rigorous proofs across multiple settings establish that the privacy cost is a lower-order term, providing theoretical grounding for practical deployment of privacy-preserving PO.
- **Reduction perspective for NPG**: Reducing DP-NPG/DP-REBEL to private regression problems enables direct reuse of established results from private estimation and supervised learning.
- **Novel private least-squares lemma** (Lemma 2): An exponential-mechanism-based private LS result applicable to sequentially adaptive data, which may hold independent research value.

## Limitations & Future Work

- **One-pass sampling only**: Current results focus on the one-pass setting; sample complexity may be improved when multiple passes over the data are permitted.
- **Limited empirical evaluation**: Experiments are restricted to CartPole-v1, with no evaluation in practically privacy-sensitive scenarios such as LLM-RLHF or healthcare applications.
- **Dimensional dependence of $W$**: In DP-NPG, the weight $W$ may depend on $\sqrt{d}$ due to privatization noise, potentially weakening the final bound.
- **Bandit simplification**: The main analysis focuses on the bandit formulation rather than full MDPs; while the extension is standard, privacy analysis for long-horizon settings is considerably more complex.
- **Strong Fisher non-degeneracy assumption**: Global convergence results rely on positive definiteness of the Fisher matrix ($\gamma > 0$); bounds become loose when $\gamma$ is small.
- **No user-level composition privacy**: Each user interacts only once; the more realistic scenario in which the same user participates multiple times is not addressed.

## Related Work & Insights

- **Rio et al. (2025)**: Empirically studies private policy gradients (PPO) without theoretical sample complexity guarantees; this paper provides the first systematic theoretical analysis.
- **Chowdhury & Zhou (2022)**: Analyzes online regret for private PPO in tabular MDPs; this paper adopts an optimization perspective for general function classes.
- **Zhou & Tan (2024)**: Considers private OPT-PPO for linear MDPs; the present framework is more general, covering NPG and REBEL.
- **Balle et al. (2016)**: Addresses private policy evaluation rather than policy optimization.
- **Non-private PO literature** (Yuan et al. 2023, Agarwal et al. 2021): The non-private terms in this paper exactly recover existing optimal bounds, with the privacy terms appearing as additional additive costs.
- **Private stochastic convex optimization** (Bassily et al. 2019): This paper connects PO to classical private optimization via reduction, though on-policy sampling introduces new challenges.

## Rating

- Novelty: ⭐⭐⭐⭐ — First systematic establishment of sample complexity theory for private PO; the novel treatment of DP definitions in the PO context is a valuable contribution.
- Experimental Thoroughness: ⭐⭐ — Only one environment (CartPole-v1); experiments serve as a proof-of-concept relegated to the appendix.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, complete theoretical framework, and fluent exposition of motivation and technical approach.
- Value: ⭐⭐⭐⭐ — Fills a gap in privacy theory for PO and provides theoretical guidance for algorithm design in privacy-sensitive applications such as RLHF.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Differentially Private Federated Low Rank Adaptation Beyond Fixed-Matrix](differentially_private_federated_low_rank_adaptation_beyond_fixed-matrix.md)
- [\[ICML 2026\] Privacy Amplification in Differentially Private Zeroth-Order Optimization with Hidden States](../../ICML2026/llm_safety/privacy_amplification_in_differentially_private_zeroth-order_optimization_with_h.md)
- [\[ICLR 2026\] wd1: Weighted Policy Optimization for Reasoning in Diffusion Language Models](../../ICLR2026/llm_safety/wd1_weighted_policy_optimization_for_reasoning_in_diffusion_language_models.md)
- [\[NeurIPS 2025\] FedSVD: Adaptive Orthogonalization for Private Federated Learning with LoRA](fedsvd_adaptive_orthogonalization_for_private_federated_learning_with_lora.md)
- [\[NeurIPS 2025\] Position: The Complexity of Perfect AI Alignment -- Formalizing the RLHF Trilemma](position_the_complexity_of_perfect_ai_alignment_--_formalizing_the_rlhf_trilemma.md)

</div>

<!-- RELATED:END -->
