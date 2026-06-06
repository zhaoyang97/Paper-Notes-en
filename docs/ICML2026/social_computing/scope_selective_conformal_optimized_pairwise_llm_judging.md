---
title: >-
  [Paper Note] SCOPE: Selective Conformal Optimized Pairwise LLM Judging
description: >-
  [ICML 2026][Social Computing][LLM-as-Judge] SCOPE eliminates positional bias in LLM judging via **Bidirectional Preference Entropy (BPE)** and combines it with **Conformal Risk Control** to achieve finite-sample FDR cont…
tags:
  - "ICML 2026"
  - "Social Computing"
  - "LLM-as-Judge"
  - "Conformal Prediction"
  - "Positional Bias"
  - "False Discovery Rate Control"
  - "Bidirectional Preference Entropy"
date: 2026-05-08
content_hash: 04421815b25582a9
---

# SCOPE: Selective Conformal Optimized Pairwise LLM Judging

**Conference**: ICML 2026  
**arXiv**: [2602.13110](https://arxiv.org/abs/2602.13110)  
**Code**: To be confirmed  
**Area**: LLM Evaluation / Uncertainty Quantification / Conformal Prediction  
**Keywords**: LLM-as-Judge, Conformal Prediction, Positional Bias, False Discovery Rate Control, Bidirectional Preference Entropy  

## TL;DR
SCOPE eliminates positional bias in LLM judging via **Bidirectional Preference Entropy (BPE)** and combines it with **Conformal Risk Control** to achieve finite-sample FDR control—providing statistically valid risk guarantees while maintaining high coverage (e.g., FDR of only 0.099 at 0.583 coverage vs. Vanilla FDR of 0.198 at 1.000 coverage).

## Background & Motivation

**Background**: LLMs are increasingly becoming scalable judging tools for tasks such as pairwise evaluation, reinforcement learning, and leaderboard rankings. Compared to human labeling, LLM judging is low-cost and high-speed.

**Limitations of Prior Work**: Existing LLM judging methods suffer from three main issues: (1) **Systematic Biases**: Positional bias, length bias, and self-preference lead to unreliable judgments; (2) **Uncalibrated Confidence**: Using model probabilities directly as confidence proxies is easily contaminated by bias, where high confidence often corresponds to incorrect judgments; (3) **Lack of Statistical Guarantees**: Even if average calibration appears reasonable, error rate control during actual deployment cannot be guaranteed.

**Key Challenge**: Selective prediction could address the question of "when to trust a judgment," but two obstacles hinder its application: (1) Inability to provide finite-sample statistical guarantees (thresholds tuned on validation sets may be violated on test sets); (2) Uncertainty signals are contaminated by positional bias, meaning high confidence often corresponds to systematic errors.

**Goal**: To design a framework for pairwise LLM judging that provides finite-sample FDR (False Discovery Rate) control guarantees while maximizing coverage under those guarantees.

**Key Insight**: (1) Introduce **Bidirectional Preference Entropy (BPE)**—querying the model in both response orders and aggregating probabilities rather than discrete votes to obtain a cleaner uncertainty signal; (2) Adopt **Conformal Risk Control**—deriving the maximum feasible threshold based on a linearized loss function and finite-sample calibration to ensure marginal FDR $\leq \alpha$.

**Core Idea**: Instead of relying on heuristic thresholds or naive empirical tuning, SCOPE eliminates positional bias through symmetric bidirectional queries and then derives a decision threshold with distribution-free guarantees using conformal theory under the exchangeability assumption.

## Method

### Overall Architecture
Four steps: (1) **Bidirectional Querying**: For a given pair of responses $(r_A, r_B)$, query the LLM judge in both forward and reverse orders to obtain two probability distributions. (2) **Aggregate Uncertainty Signal**: Calculate the average probability of bidirectional preferences and use binary entropy as the uncertainty score. (3) **Conformal Calibration**: On a labeled calibration set, determine the maximum feasible threshold $\hat{\lambda}$ using a linearized loss function and finite-sample necessary/sufficient conditions. (4) **Selective Judging**: During deployment, accept the judgment if the uncertainty score $\leq \hat{\lambda}$; otherwise, abstain.

### Key Designs

1.  **Bidirectional Preference Entropy (BPE)**:
    - **Function**: Eliminates the contamination of uncertainty estimation by positional bias through two forward passes per response pair, resulting in a more neutral uncertainty signal.
    - **Mechanism**: Let the probability of choosing $r_A$ in forward order be $p_{\text{fwd}} = P_\theta(y = A \mid x_{\text{fwd}})$, and the probability of choosing $r_A$ (corresponding to label $B$) in reverse order be $p_{\text{rev}} = P_\theta(y = B \mid x_{\text{rev}})$. If the judge is reliable and unbiased, these should be close. Take the average $\bar{p} = \frac{1}{2}(p_{\text{fwd}} + p_{\text{rev}})$, and calculate the binary entropy $s(x) = -[\bar{p} \log \bar{p} + (1 - \bar{p}) \log(1 - \bar{p})]$.
    - **Design Motivation**: Positional bias causes judges to systematically favor a specific position. Ordinary confidence cannot distinguish between "true certainty" and "bias-induced false certainty." BPE enforces model symmetry via permutation invariance, regarding a judgment as high-confidence only when the same preference is made in both orders.

2.  **Linearized Loss + Finite-Sample Calibration**:
    - **Function**: Transforms the FDR constraint into an easily solvable finite-sample condition to directly calculate the maximum feasible threshold.
    - **Mechanism**: Introduce a linearized loss $L(x, \lambda) = S(x, \lambda) \cdot (E(x) - \alpha)$, where $S(x, \lambda)$ is the selection indicator and $E(x)$ is the error indicator. Key observation: $\mathbb{E}[L(x, \lambda)] \leq 0$ is equivalent to FDR $\leq \alpha$. The finite-sample necessary and sufficient condition is $\sum_{i=1}^n S(x_i, \lambda) \cdot (E(x_i) - \alpha) \leq -1$. The "$-1$" budget absorbs the worst-case scenario for a single test sample.
    - **Design Motivation**: Empirical risk ratios are unstable with small samples. The linearized transformation allows for exact algebraic solving, while the "$-1$" budget provides a distribution-free safety margin.

3.  **Maximum Coverage Threshold Selection**:
    - **Function**: Selects the largest feasible threshold while satisfying FDR constraints to maximize the number of accepted judgments.
    - **Mechanism**: Since a new sample may contribute $-\alpha$ or $1 - \alpha$, constraint feasibility is not necessarily monotonic with $\lambda$. The algorithm directly finds the maximum feasible $\lambda$: $\hat{\lambda} = \sup\{\lambda : \sum_{i=1}^n S(x_i, \lambda) \cdot (E(x_i) - \alpha) \leq -1\}$.
    - **Design Motivation**: Simple threshold searches are inefficient due to unclear coverage-risk trade-offs. Maximizing the feasible threshold ensures mathematical guarantees while fully utilizing the user-specified risk budget.

## Key Experimental Results

### Uncertainty Estimation Quality Comparison

| Model | Method | ECE ↓ | AUROC | AUPRC |
| :--- | :--- | :--- | :--- | :--- |
| Qwen2.5-7B | Predicted Prob. | 0.239 | 0.658 | 0.824 |
| Qwen2.5-7B | Swap-Aggregate | 0.193 | 0.656 | 0.826 |
| Qwen2.5-7B | **BPE** | **0.143** | **0.685** | **0.855** |
| Qwen2.5-32B | **BPE** | **0.172** | **0.729** | **0.908** |
| Llama-3.1-70B | **BPE** | **0.145** | **0.744** | **0.894** |

### Risk Control and Coverage ($\alpha = 0.10$)

| Dataset | Model | Method | Coverage | Actual FDR | Violated? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| MT-Bench | Qwen-7B | Vanilla | 1.000 | 0.269 | ❌ |
| MT-Bench | Qwen-7B | Heuristic | 0.907 | 0.251 | ❌ |
| MT-Bench | Qwen-7B | **SCOPE** | **0.246** | **0.097** | ✅ |
| RewardBench | Qwen-32B | Vanilla | 1.000 | 0.120 | ❌ |
| RewardBench | Qwen-32B | **SCOPE** | **0.983** | **0.098** | ✅ |
| Chatbot Arena | Llama-70B | **SCOPE** | **0.583** | **0.099** | ✅ |

### Key Findings
- **Advantages of BPE**: Averaging two probabilities before calculating entropy yields a richer continuous signal, significantly outperforming discrete voting—AUROC for Qwen-7B increased from 0.656 to 0.685.
- **Statistical Validity**: Vanilla coverage is 100%, but FDR often reaches 0.2–0.27, far exceeding the 0.10 constraint. SCOPE consistently satisfies the constraint across all configurations.
- **Model Scale and Stability**: Stronger models (Llama-70B) maintain higher coverage (0.583 @ $\alpha = 0.10$) under strict constraints, while weaker models (Qwen-7B) see significant coverage drops (0.246). However, even for weak models, SCOPE ensures FDR remains within expectations.

## Highlights & Insights
- **First Pairwise LLM Judging Framework with Finite-Sample FDR Guarantees**: The combination of BPE and linearized loss elegantly solves two problems—the former removes bias sources, and the latter provides mathematical guarantees.
- **Creative Design of Permutation Invariance**: While most bias-removal methods use discrete voting, BPE achieves permutation invariance through probability averaging and entropy—maintaining simplicity (only two forward passes) while obtaining a continuous signal.
- **Deployment Without Retraining**: SCOPE relies only on softmax probabilities or logits, acting as a plug-and-play layer for any judge (including API-only models), reducing deployment costs.
- **Elegant Transformation of Linearized Loss**: Converting a ratio constraint into a linear constraint and using the "$-1$" budget to absorb the worst case is a refined theoretical design.

## Limitations & Future Work
- The exchangeability assumption may fail under real-world distribution shifts (changes in prompts or policy model behavior).
- BPE requires two forward passes, incurring 2x the computational overhead compared to a single query; it is inapplicable to complete black-box APIs where probability access is unavailable.
- The current framework is limited to binary pairwise judgments and does not support multi-response ranking or rubric-based judging.
- In low-coverage scenarios (e.g., Qwen-7B at $\alpha = 0.05$ yielding only 2.4% coverage), the framework's utility is restricted.

## Related Work & Insights
- **vs. Swap-and-Aggregate**: Both use bidirectional queries to eliminate positional bias, but swap-aggregate uses discrete voting while BPE uses probability entropy—BPE signals are continuous and more fine-grained.
- **vs. Simulated Annotators**: Estimates confidence via 5 virtual personas $\times$ 5 few-shot demonstrations; BPE achieves better calibration and discriminative power with only two forward passes.
- **vs. Unexplained Heuristic Selection**: Setting thresholds to $1 - \alpha$ or tuning on validation sets lacks theoretical foundation; SCOPE’s linearized loss and finite-sample calibration are theoretically rigorous.
- **vs. Conservative Confidence Bounds** (Clopper-Pearson): Usually requires rejecting a large volume of queries; SCOPE achieves 2.4x higher coverage at the same risk level through more precise design.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First combination of pairwise LLM judging with finite-sample FDR control; the permutation-invariant design of BPE is unique.
- Experimental Thoroughness: ⭐⭐⭐⭐ Verified across three major benchmarks and four model scales with high statistical robustness (1000 random splits); cross-domain generalization verification is slightly insufficient.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, progressive motivation, and precise symbolic definitions.
- Value: ⭐⭐⭐⭐⭐ Provides the first deployable framework with guarantees for the increasingly important LLM-as-Judge scenario; highly significant for industry applications like RLHF, leaderboards, and auto-eval.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Beyond the Crowd: LLM-Augmented Community Notes for Governing Health Misinformation](../../ACL2026/social_computing/beyond_the_crowd_llm-augmented_community_notes_for_governing_health_misinformati.md)
- [\[AAAI 2026\] Bias Association Discovery Framework for Open-Ended LLM Generations](../../AAAI2026/social_computing/bias_association_discovery_framework_for_open-ended_llm_generations.md)
- [\[ACL 2026\] Why Are We Moral? An LLM-based Agent Simulation Approach to Study Moral Evolution](../../ACL2026/social_computing/why_are_we_moral_an_llm-based_agent_simulation_approach_to_study_moral_evolution.md)
- [\[ACL 2026\] Justice in Judgment: Unveiling (Hidden) Bias in LLM-assisted Peer Reviews](../../ACL2026/social_computing/justice_in_judgment_unveiling_hidden_bias_in_llm-assisted_peer_reviews.md)
- [\[ACL 2026\] Estimating the Black-box LLM Uncertainty with Distribution-Aligned Adversarial Distillation](../../ACL2026/social_computing/estimating_the_black-box_llm_uncertainty_with_distribution-aligned_adversarial_d.md)

</div>

<!-- RELATED:END -->
