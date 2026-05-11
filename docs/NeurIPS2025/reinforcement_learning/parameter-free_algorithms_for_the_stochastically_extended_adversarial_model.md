---
title: >-
  [Paper Note] Parameter-Free Algorithms for the Stochastically Extended Adversarial Model
description: >-
  [NeurIPS 2025][Reinforcement Learning][online convex optimization] This work presents the first parameter-free algorithms for the Stochastically Extended Adversarial (SEA) model…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "online convex optimization"
  - "parameter-free algorithms"
  - "SEA model"
  - "comparator-adaptive"
  - "Lipschitz-adaptive"
date: 2026-05-08
content_hash: 59cfe3beb13b980d
---

# Parameter-Free Algorithms for the Stochastically Extended Adversarial Model

**Conference**: NeurIPS 2025
**arXiv**: [2510.04685](https://arxiv.org/abs/2510.04685)
**Code**: None
**Area**: Reinforcement Learning / Online Learning
**Keywords**: online convex optimization, parameter-free algorithms, SEA model, comparator-adaptive, Lipschitz-adaptive

## TL;DR

This work presents the first parameter-free algorithms for the Stochastically Extended Adversarial (SEA) model, which bridges adversarial and stochastic online convex optimization. Without prior knowledge of the domain diameter $D$ and/or the Lipschitz constant $G$, the proposed algorithms—built upon Optimistic Online Newton Step (OONS)—achieve regret bounds comparable to those of parameter-aware methods.

## Background & Motivation

- **Background**: The SEA model for online convex optimization (OCO) bridges purely adversarial and purely stochastic settings, with regret bounds depending on the cumulative stochastic variance $\sigma_{1:T}^2$ and the cumulative adversarial variation $\Sigma_{1:T}^2$ rather than on $T$.
- **Limitations of Prior Work**: Existing SEA model algorithms (OFTRL, OMD) require prior knowledge of the domain diameter $D$ and the Lipschitz constant $G$ to set optimal step sizes, limiting their practical applicability. Directly applying existing parameter-free OCO algorithms fails to yield the desirable $\sigma_{1:T}^2$-dependent bounds—only the looser $\tilde{\sigma}_{1:T}^2$ can be obtained—because achieving $\sigma_{1:T}^2$-scaling requires the RVU property, which is considerably more difficult to establish in unbounded domains.
- **Goal**: Develop parameter-free algorithms that adapt to $D$ and $G$ while maintaining optimal dependence on $\sigma_{1:T}^2$ and $\Sigma_{1:T}^2$ in the SEA model.

## Method

### Overall Architecture

The construction proceeds in three progressive steps: (1) establish OONS as the base algorithm with an adaptive step size that matches the state-of-the-art bounds under known parameters; (2) construct CA-OONS (comparator-adaptive) via a multi-scale base learner framework combined with the MsMwC meta-algorithm; (3) construct CLA-OONS (comparator- and Lipschitz-adaptive) by further incorporating gradient clipping and a domain diameter doubling scheme.

### Key Designs

1. **Optimistic Online Newton Step (OONS)**: Builds on the ONS algorithm with an adaptive step size $\eta_t = \min\{\frac{1}{64Dz_T}, \frac{1}{D\sqrt{\sum_{s=1}^{t-1}\|g_s-m_s\|_2^2}}\}$. Two sequences $\{x_t\}$ and $\{x_t'\}$ (optimistic predictions) are maintained, and a second-order information matrix $A_t$ is used to adaptively adjust the update direction. A key structural property is that $D$ appears in the regret decomposition only through the endpoint term $D(z_T - z_1)$—rather than throughout all rounds—which enables subsequent elimination of the dependence on $D$. OONS satisfies the RVU property, including a negative stability term $-z_1^2 \sum_t \|x_t - x_{t-1}\|_2^2$.

2. **CA-OONS (Comparator-Adaptive, Algorithm 2)**: Employs a three-level structure to handle unknown $D$. At the base level, $N = \lceil \log T \rceil$ OONS base learners each operate on a restricted domain of radius $D_j = 2^j$. At the intermediate level, the MsMwC algorithm learns optimal combination weights $w_t^k$ over base learners for each scale $k$. At the top level, a MsMwC-Master algorithm learns the optimal mixture $p_t \in \Delta_\mathcal{S}$ across scales, yielding the final decision $x_t = \sum_k p_{t,k} w_t^k$. The regret is decomposed into MetaRegret and BaseRegret, with each level controlled independently.

3. **CLA-OONS (Comparator- and Lipschitz-Adaptive, Algorithm 4)**: When $G$ is also unknown, gradient clipping is introduced as $\tilde{g}_t = m_t + \frac{B_{t-1}}{B_t}(g_t - m_t)$, where $B_t = \max_{s \leq t} \|g_s - m_s\|_2$. The domain diameter $D_t$ is updated dynamically via a doubling strategy: it is doubled whenever $D_t < \sqrt{\sum_s \|g_s\|_2 / \max_{k \leq s} \|g_k\|_2}$. At most $M = O(\log T)$ resets are triggered, each resetting the matrix $A_{t+1}$ and the optimistic prediction $x'_{t+1}$.

### Loss & Training

The setting is online learning with no explicit training loss. Algorithms rely on the optimistic prediction $m_t = \nabla f_{t-1}(x_{t-1})$. In CA-OONS, the meta-algorithm initializes $p_1' \propto \beta_k^2$, and step sizes at each level are carefully designed to control multi-scale regret. The adaptive step size of OONS takes the form $\eta_t = \min\{\frac{1}{64Dz_T}, \frac{1}{D\sqrt{V_{t-1}}}\}$, ensuring adaptation to gradient variation. The domain diameter doubling in CLA-OONS triggers at most $M = O(\log T)$ resets of the second-order information matrix $A_t$. The expert set $\mathcal{S}$ has size $O(N \log T)$, keeping the total computational overhead manageable.

## Key Experimental Results

### Main Results

| Algorithm | $D$-free | $G$-free | Expected Regret Bound |
|-----------|----------|----------|-----------------------|
| OFTRL/OMD (Sachs et al.) | ✗ | ✗ | $O(\sqrt{\sigma_{1:T}^2} + \sqrt{\Sigma_{1:T}^2})$ |
| OONS (Theorem 3.2) | ✗ | ✗ | $\tilde{O}(\sqrt{\sigma_{1:T}^2} + \sqrt{\Sigma_{1:T}^2})$ |
| CA-OONS (Theorem 4.1) | ✓ | ✗ | $\tilde{O}(\|u\|_2^2 + \|u\|_2(\sqrt{\sigma_{1:T}^2} + \sqrt{\Sigma_{1:T}^2}))$ |
| CLA-OONS (Theorem 4.5) | ✓ | ✓ | $\tilde{O}(\|u\|_2^2(\sqrt{\sigma_{1:T}^2} + \sqrt{\Sigma_{1:T}^2}) + \|u\|_2^4 + \sqrt{\sigma_{1:T} + \mathfrak{G}_{1:T}})$ |

### Ablation Study

| Configuration | Remarks |
|---------------|---------|
| OONS vs. OMD as base algorithm | OMD's regret depends on $D\sqrt{\sum\|g_t\|^2}$, which degrades to $O(T)$ in unbounded domains |
| CA-OONS in bounded domain | Recovers $\tilde{O}(D^2 + D(\sqrt{\sigma_{1:T}^2} + \sqrt{\Sigma_{1:T}^2}))$, matching the state of the art |
| $\|u\|_2^2$ vs. $\|u\|_2$ dependence | The $\|u\|_2^2$ term in gradient-variation regret bounds may be unavoidable under current techniques |

### Key Findings

- The second-order adaptive property of OONS is the critical enabler for eliminating dependence on $D$—OMD/OFTRL cannot achieve this.
- The $\|u\|_2^2$ term in CA-OONS may be inherent under gradient-variation bounds (analogous to the appearance of $d_0^2$ in offline accelerated optimization).
- The additional $\sqrt{\mathfrak{G}_{1:T}}$ term introduced by CLA-OONS is independent of $\|u\|$ and does not grow with the comparator norm.
- $\sigma_{1:T}^2$ is strictly better than $\tilde{\sigma}_{1:T}^2$ (the latter can be arbitrarily large), but achieving $\sigma_{1:T}^2$-scaling requires the RVU property.

## Highlights & Insights

- This is the first work to achieve parameter-free algorithms in the SEA model while simultaneously preserving dependence on $\sigma_{1:T}^2$ and $\Sigma_{1:T}^2$.
- The key observation that $D$ appears only in the endpoint term $D(z_T - z_1)$ within OONS is technically elegant and constitutes the central breakthrough.
- The systematic three-level multi-scale architecture (base learners → intermediate MsMwC → top-level Master) is clearly motivated and well-structured.
- The discussion on adaptivity between gradient-variation bounds and worst-case bounds (Remark 4.3) reveals deep theoretical connections.

## Limitations & Future Work

- The leading term of CLA-OONS is $\|u\|_2^2$ rather than the ideal $\|u\|_2$; improving this dependence remains an important open problem.
- CA-OONS requires $O(\log T)$ gradient queries per round; reducing this to $O(1)$ is an open question.
- The work is purely theoretical with no empirical validation.
- High-probability regret bounds (as opposed to expected bounds) have not been established.
- Whether a single algorithm can simultaneously adapt to both gradient-variation and worst-case bounds remains open.

## Related Work & Insights

- Sachs et al. and Chen et al. establish baseline regret for the SEA model, but with step sizes depending on $D$ and $G$.
- The MsMwC framework from chen2021impossible is adopted for multi-scale step size management.
- The gradient clipping technique from cutkosky2019artificial is adapted for Lipschitz adaptation in CLA-OONS.
- The RVU property bridges online learning with game theory and accelerated optimization, serving as an important theoretical tool.
- Jacobsen et al.'s parameter-free mirror descent can achieve gradient-variation bounds but depends on $\tilde{\sigma}_{1:T}^2$.
- Comparator-adaptive methods from orabona2016coin and cutkosky2018black apply only to the purely adversarial setting.
- The SEA model is well understood in the expert prediction and bandit settings, but the OCO setting is more challenging.
- The online-to-batch conversion maps gradient-variation regret to accelerated convergence rates, revealing a deep connection.
- The doubling trick can remove dependence on $T$, yielding anytime algorithms.
- Related work on tuning-free SGD and DoG step sizes provides relevant but not directly transferable techniques.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First to resolve the parameter-free problem in the SEA model; the endpoint property of OONS is a technically elegant discovery.
- Experimental Thoroughness: ⭐⭐ — Purely theoretical; no experiments.
- Writing Quality: ⭐⭐⭐⭐ — Technically deep with clear hierarchical structure, though notation density is very high.
- Value: ⭐⭐⭐⭐ — Fills a theoretical gap regarding parameter-free algorithms in the SEA model.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] When Can Model-Free Reinforcement Learning be Enough for Thinking?](when_can_model-free_reinforcement_learning_be_enough_for_thinking.md)
- [\[NeurIPS 2025\] Improving Planning and MBRL with Temporally-Extended Actions](improving_planning_and_mbrl_with_temporally-extended_actions.md)
- [\[NeurIPS 2025\] Deep RL Needs Deep Behavior Analysis: Exploring Implicit Planning by Model-Free Agents](deep_rl_needs_deep_behavior_analysis_exploring_implicit_planning_by_model-free_a.md)
- [\[ICLR 2026\] On Discovering Algorithms for Adversarial Imitation Learning](../../ICLR2026/reinforcement_learning/on_discovering_algorithms_for_adversarial_imitation_learning.md)
- [\[NeurIPS 2025\] Parameter Efficient Fine-tuning via Explained Variance Adaptation](parameter_efficient_fine-tuning_via_explained_variance_adaptation.md)

</div>

<!-- RELATED:END -->
