---
title: >-
  [Paper Note] Doubly Robust Alignment for Large Language Models
description: >-
  [NeurIPS 2025][Optimization][RLHF] DRPO draws on doubly robust estimation from causal inference to propose a preference optimization algorithm that maintains consistency whenever either the preference model or the refere…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "RLHF"
  - "doubly robust"
  - "preference optimization"
  - "DPO"
  - "model robustness"
date: 2026-05-08
content_hash: 159822d7c7b56296
---

# Doubly Robust Alignment for Large Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2506.01183](https://arxiv.org/abs/2506.01183)
**Code**: [https://github.com/DRPO4LLM/DRPO4LLM](https://github.com/DRPO4LLM/DRPO4LLM)
**Area**: Optimization
**Keywords**: RLHF, doubly robust, preference optimization, DPO, model robustness

## TL;DR
DRPO draws on doubly robust estimation from causal inference to propose a preference optimization algorithm that maintains consistency whenever either the preference model or the reference policy is correctly specified, outperforming PPO/DPO and their variants both theoretically and empirically.

## Background & Motivation

**Background**: RLHF is the dominant paradigm for LLM alignment. Existing algorithms fall into two broad categories—reward-based methods (e.g., PPO) and preference-based methods (e.g., DPO, IPO)—both of which have achieved notable empirical success.

**Limitations of Prior Work**: Each family of methods is vulnerable to model misspecification. (a) PPO relies on the Bradley–Terry (BT) preference model assumption, is highly sensitive to reward estimation errors, and is prone to reward hacking. (b) DPO bypasses reward estimation but remains sensitive to the specification of the reference policy $\pi_\text{ref}$. (c) Both families exhibit low tolerance for violations of their respective model assumptions.

**Key Challenge**: The BT model assumes that human preferences satisfy transitivity and context-independence, properties that are frequently violated empirically. Moreover, the reference policy may not be known or accurately specified. No existing method is simultaneously robust to misspecification of both the preference model and the reference policy.

**Goal**: To design an RLHF algorithm that guarantees consistency whenever either the preference model or the reference policy is correctly specified.

**Key Insight**: The work draws on doubly robust (DR) estimation from econometrics and causal inference. In treatment effect estimation, DR estimators require only one of the propensity score model or the outcome model to be correctly specified—an analogy that maps naturally onto the roles of the preference model and the reference policy in RLHF.

**Core Idea**: Introduce the DR estimation framework into preference evaluation to construct a preference optimization algorithm that is doubly robust with respect to both the preference model and the reference policy.

## Method

### Overall Architecture
Given a dataset $\mathcal{D} = \{(X, Y^{(1)}, Y^{(2)}, Z)\}$ (prompts, two responses, and preference labels), the goal is to find an optimal policy $\pi$ that maximizes total preference. The approach proceeds in three steps: (1) construct a DR preference evaluation estimator; (2) perform preference optimization based on the DR estimator; and (3) stabilize training via KL regularization.

### Key Designs

1. **Total Preference Definition and Baseline Estimators**

   - **Function**: Define the total preference of the target policy over the reference policy as $p^*(\pi) = \mathbb{P}(\pi \succ \pi_\text{ref})$.
   - **Mechanism**: The Direct Method (DM) estimator plugs in the estimated preference model $\hat{g}$ directly, relying on the correctness of $\hat{g}$. The Importance Sampling (IS) estimator converts the expectation using the importance weight $w(y,x) = \pi(y|x)/\pi_\text{ref}(y|x)$, relying on the correctness of $\hat{\pi}_\text{ref}$.
   - **Design Motivation**: Each baseline estimator depends on the correctness of one model, making neither sufficiently robust on its own.

2. **Doubly Robust (DR) Preference Estimator**

   - **Function**: Combine DM and IS to construct an estimator that is robust to misspecification of either model.
   - **Mechanism**: The estimating function is
$$\psi = \frac{1}{2}\sum_{a=1}^2 \mathbb{E}_{y \sim \pi}[\hat{g}(X,y,Y^{(a)})] + \frac{1}{2}\sum_{a=1}^2 (-1)^{a-1} \frac{\pi(Y^{(a)}|X)}{\hat{\pi}_\text{ref}(Y^{(a)}|X)}[Z - \hat{g}(X,Y^{(1)},Y^{(2)})]$$
     The first term is the DM component; the second is an augmentation term that uses the preference residual $Z - \hat{g}$ to correct the bias of the DM.
   - **Design Motivation**: When $\hat{g} = g^*$, the augmentation term has zero expectation (DM is already correct). When $\hat{\pi}_\text{ref} = \pi_\text{ref}$, the augmentation term automatically recovers the IS estimate. Correctness of either model guarantees consistency.
   - **Distinction from bandit DR**: In pairwise comparisons, each data tuple is used twice (in both directions), effectively reducing variance.

3. **DRPO Preference Optimization**

   - **Function**: Solve for the optimal policy using the DR estimator.
   - **Mechanism**:
$$\hat{\pi} = \arg\max_{\pi \in \Pi} \{\hat{p}_\text{DR}(\pi) - \beta \mathbb{E}_X D_\text{KL}[\pi(\cdot|X) \| \hat{\pi}_\text{ref}(\cdot|X)]\}$$
   - **Implementation Details**: (a) IS ratios are clipped to prevent extreme values; (b) a surrogate objective is designed to support Monte Carlo sampling from the target policy; (c) GRPO-style variance reduction is applied to the KL divergence term.

### Theoretical Guarantees
- **Theorem 2 (MSE)**: The MSE of the DR estimator equals the semiparametric efficiency bound (SEB) plus a product bias term $O(\|\hat{\pi}_\text{ref}/\pi_\text{ref}-1\|^2 \cdot \|\hat{g}-g^*\|^2)$, so the bias is the product of the two individual errors.
- **Corollary 3 (Double Robustness)**: If either model is correctly specified, the MSE converges to zero.
- **Corollary 4 (Semiparametric Efficiency)**: When both models are approximately correct, the MSE asymptotically attains the SEB.
- **Theorem 7 (Suboptimality)**: The suboptimality bound of DRPO is $O(\|\hat{\pi}_\text{ref}/\pi_\text{ref}-1\| \cdot \|\hat{r}-r^*\|)$; the two errors enter multiplicatively, making this bound strictly more robust than PPO's $O(\|\hat{r}-r^*\|)$ or DPO's $O(\|\hat{\pi}_\text{ref}/\pi_\text{ref}-1\|)$.

## Key Experimental Results

### Preference Estimation (IMDb Controlled Experiment)
The doubly robust property is verified in a synthetic environment where both the preference model and the reference policy are under controlled misspecification:
- Both correctly specified: MSE is lowest, approaching zero after 1,500 samples.
- Only preference model correct / only reference policy correct: MSE decreases substantially.
- Both misspecified: MSE is large and does not improve with more samples.
These results align perfectly with the doubly robust theory.

### Main Results (Preference Optimization)

| Dataset | Comparison | DRPO-BT Win Rate |
|---------|------------|-----------------|
| TL;DR | vs Dr. DPO | 72.5% |
| TL;DR | vs rDPO | 65.0% |
| TL;DR | vs cDPO | 63.5% |
| TL;DR | vs CPO | 90.0% |
| TL;DR | vs IPO | 98.5% |
| TL;DR | vs RSO | 69.5% |

| Model | LC Win Rate | Win Rate |
|-------|-------------|----------|
| Dr. DPO | 92.16% | 90.93% |
| rDPO | 86.89% | 85.71% |
| cDPO | 85.05% | 84.28% |
| **DRPO** | **86.38%** | **84.84%** |
| IPO | 78.29% | 78.88% |
| RSO | 80.62% | 79.50% |

### Key Findings
- On TL;DR, DRPO substantially outperforms all DPO variants (98.5% win rate vs. IPO), consistent with the presence of reference policy misspecification in that dataset.
- On HH, DRPO-GPM achieves the best in-domain performance; DRPO-BT achieves a 57% win rate against PPO despite using the same preference model, demonstrating robustness to preference model misspecification.
- Out-of-domain (AlpacaEval 2.0), DRPO performs comparably to DPO variants specifically designed to handle label noise (cDPO, rDPO).

## Highlights & Insights
- **Cross-domain innovation from causal inference to RLHF**: The core idea of DR estimation—bias-corrected augmentation—is elegantly transferred to preference optimization, with strong theoretical foundations.
- **Multiplicative error bound**: In DRPO's suboptimality bound, the two sources of error appear as a product, meaning that even when both models are imperfect but each has moderate accuracy, DRPO can still outperform PPO and DPO.
- **No reliance on the BT assumption**: Theorem 5 provides a performance bound without requiring the BT model assumption, giving the result broad applicability to preference-based RLHF.

## Limitations & Future Work
- IS ratio clipping introduces bias, and no theoretical guidance exists for selecting the clipping threshold.
- When $\pi$ deviates substantially from $\pi_\text{ref}$ (high IS ratios), variance remains a concern.
- Experiments are conducted at limited scale (relatively small open-source models); performance on state-of-the-art large models has not been verified.
- DRPO requires separate training of the preference model $\hat{g}$ and the reference policy $\hat{\pi}_\text{ref}$, increasing pipeline complexity.

## Related Work & Insights
- **vs DPO**: DPO parameterizes the reward as a policy ratio, eliminating the reward model but remaining sensitive to the reference policy. DRPO corrects for reference policy bias via the augmentation term.
- **vs PPO**: PPO follows a two-stage pipeline (reward learning + RL) and is highly sensitive to reward estimation errors. DRPO does not require an accurate reward model.
- **vs IPO**: IPO is also a preference-based method optimizing the same objective but is not robust to reference policy misspecification. DRPO adds an IS augmentation term to address this.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — First complete introduction of the doubly robust framework into RLHF; both the theoretical motivation and technical design are highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Synthetic experiments perfectly validate the theory; real-task comparisons are comprehensive, though model scale is limited.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theoretical derivations are rigorous and clear; the methodological motivation is developed in a well-structured progression; visualizations aid comprehension.
- **Value**: ⭐⭐⭐⭐⭐ — Provides a theoretically superior and practically more robust new paradigm for RLHF, with the potential to become an important reference for the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Constrained Network Slice Assignment via Large Language Models](constrained_network_slice_assignment_via_llms.md)
- [\[NeurIPS 2025\] VERA: Variational Inference Framework for Jailbreaking Large Language Models](vera_variational_inference_framework_for_jailbreaking_large_language_models.md)
- [\[NeurIPS 2025\] Large Language Bayes](large_language_bayes.md)
- [\[NeurIPS 2025\] Training-Free Bayesianization for Low-Rank Adapters of Large Language Models](training-free_bayesianization_for_low-rank_adapters_of_large_language_models.md)
- [\[NeurIPS 2025\] DynaAct: Large Language Model Reasoning with Dynamic Action Spaces](dynaact_large_language_model_reasoning_with_dynamic_action_spaces.md)

</div>

<!-- RELATED:END -->
