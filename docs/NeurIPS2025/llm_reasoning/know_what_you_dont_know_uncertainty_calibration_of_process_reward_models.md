---
title: >-
  [Paper Note] Know What You Don't Know: Uncertainty Calibration of Process Reward Models
description: >-
  [NeurIPS 2025][LLM Reasoning][Process Reward Model] This paper proposes a quantile regression-based calibration method for PRMs, enabling their output scores to more accurately reflect the actual success probability of LLM reasoning. Building on the calibrated PRM, the paper further introduces an Instance-Adaptive Scaling (IAS) strategy for inference-time computation, achieving significant cost reduction while maintaining accuracy.
tags:
  - NeurIPS 2025
  - LLM Reasoning
  - Process Reward Model
  - Calibration
  - Quantile Regression
  - Inference-Time Scaling
  - Adaptive Sampling
date: 2026-05-08
content_hash: 14b001d61112c080
---

# Know What You Don't Know: Uncertainty Calibration of Process Reward Models

**Conference**: NeurIPS 2025
**arXiv**: [2506.09338](https://arxiv.org/abs/2506.09338)
**Code**: [http://young-j-park.github.io/know-what-you-dont-know](http://young-j-park.github.io/know-what-you-dont-know)
**Area**: LLM Reasoning
**Keywords**: Process Reward Model, Calibration, Quantile Regression, Inference-Time Scaling, Adaptive Sampling

## TL;DR
This paper proposes a quantile regression-based calibration method for PRMs, enabling their output scores to more accurately reflect the actual success probability of LLM reasoning. Building on the calibrated PRM, the paper further introduces an Instance-Adaptive Scaling (IAS) strategy for inference-time computation, achieving significant cost reduction while maintaining accuracy.

## Background & Motivation

**Background**: Process Reward Models (PRMs) play a central role in inference-time scaling, guiding methods such as Best-of-N sampling and Beam Search. A PRM scores each intermediate reasoning step to assess the probability that the current reasoning path will ultimately yield a correct answer.

**Limitations of Prior Work**: Even state-of-the-art PRMs (e.g., Qwen-PRM, Shepherd-PRM) suffer from severe miscalibration—they tend to **overestimate success probabilities**, particularly when paired with weaker LLMs or applied to difficult problems. This occurs because PRMs are trained on a specific policy model, and distribution mismatch arises when a different-capacity LLM is used at deployment.

**Key Challenge**: PRM training is coupled to the generation distribution $\pi_\theta$ of a particular policy model, yet deployment may involve models of different capability. A PRM trained on a 72B model will systematically overestimate the success probability of a 1B model.

**Goal**: (1) How can off-the-shelf PRMs be calibrated to produce reliable success probability estimates? (2) How can calibrated probabilities be leveraged to achieve adaptive computational resource allocation?

**Key Insight**: Conventional calibration methods such as temperature scaling are ill-suited for PRMs—since the success probability is itself an intermediate probability rather than a binary label—so the paper proposes quantile regression to predict the distribution of success probabilities, using conservative lower-quantile estimates to guide resource allocation.

**Core Idea**: Fine-tune a PRM with quantile regression to predict lower bounds on success probability, thereby enabling conservative yet reliable instance-adaptive inference budget allocation.

## Method

### Overall Architecture
Given an off-the-shelf PRM and a target LLM, a calibration dataset is constructed via a three-stage data collection pipeline (initial trajectory generation → prefix extraction and Monte Carlo rollout → success probability estimation). The PRM prediction head is then fine-tuned with quantile regression, and the calibrated PRM is used to enable adaptive sampling.

### Key Designs

1. **Three-Stage Calibration Data Collection**:

    - *Function*: Generate multiple reasoning trajectories per problem and estimate the true success probability for each prefix via Monte Carlo rollout.
    - *Mechanism*: Sample 500 problems from the MATH training set, generate $N_{\text{val}}=8$ trajectories per problem, and for each prefix generate $N_{\text{MC}}=8$ continuations; record the empirical accuracy as the ground-truth success probability $\tilde{p}^{(i,t)}$.
    - *Design Motivation*: Monte Carlo sampling provides empirical success probabilities tied to the specific LLM, resolving the mismatch between the PRM and the policy model.

2. **Quantile Regression Calibration**:

    - *Function*: Modify the PRM prediction head to output predictions at multiple quantile levels (e.g., 10%, 50%, 90%).
    - *Mechanism*: Optimize using the weighted quantile loss $\text{wQL}(\hat{r}, \tilde{p}) = \frac{1}{N_q} \sum_{n=1}^{N_q} [\beta_n \cdot \max(0, \tilde{p} - \hat{r}^{(\beta_n)}) + (1-\beta_n) \cdot \max(0, \hat{r}^{(\beta_n)} - \tilde{p})]$.
    - *Design Motivation*: Predicting the conditional mean causes overestimation in approximately 50% of cases. Predicting a low quantile (e.g., the 10th percentile) provides a conservative lower bound, preventing under-allocation of computational resources.

3. **Instance-Adaptive Scaling (IAS)**:

    - *Function*: Dynamically adjust the number of samples per problem based on success probabilities estimated by the calibrated PRM.
    - *Mechanism*: For a problem with success probability $p$, the minimum number of samples required to achieve target accuracy $C$ is $N_{\text{IAS}}(p, C) = \lceil \frac{\log(1-C)}{\log(1-p)} \rceil$.
    - *Design Motivation*: Easy problems (high $p$) require few samples, while difficult problems (low $p$) require more—mirroring the human strategy of spending less time on easy questions and more on hard ones.

### Loss & Training
- Only the PRM prediction head is fine-tuned (with expanded output dimensions to support multi-quantile prediction); the backbone model remains frozen.
- Qwen-PRM-7B and Shepherd-PRM-7B serve as base models for calibration.

## Key Experimental Results

### Main Results — PRM Calibration Quality (Qwen-PRM-7B, Brier Score ↓)

| Dataset | Target LLM | Uncalibrated | Calibrated | Reduction |
|--------|---------|--------|--------|------|
| MATH500 | Llama-3.2-1B | 0.2414 | 0.0692 | −71% |
| MATH500 | Qwen-2.5-7B | 0.1008 | 0.0818 | −19% |
| MATH500 | R1-Qwen-7B | 0.1480 | 0.0828 | −44% |
| AIME24-25 | Llama-3.2-1B | 0.1936 | 0.0029 | −98% |
| AIME24-25 | R1-Qwen-7B | 0.4144 | 0.0694 | −83% |

### Ablation Study — Compute Savings with BoN+IAS

| Dataset | Model | BoN (N=64) | BoN+IAS (Calibrated) | Budget Ratio |
|--------|------|-----------|---------------|-------------|
| MATH500 | Qwen-2.5-7B | 0.854 | 0.837 | 23.4% |
| MATH500 | R1-Qwen-7B | 0.864 | 0.857 | 31.3% |
| MATH500 | Llama-3.2-1B | 0.476 | 0.462 | 63.8% |
| AIME24-25 | R1-Qwen-7B | 0.267 | 0.180 | 96.4% |

### Key Findings
- PRMs systematically overestimate success probabilities, particularly for weaker models and out-of-distribution problems (bias distribution is right-skewed with a peak near 1.0).
- **Calibration is a prerequisite for effective IAS**—uncalibrated PRMs cannot reliably guide resource allocation.
- Low quantiles (e.g., 10%) are more appropriate than the conditional mean for IAS, as they provide conservative estimates.
- IAS saves substantial computation on easy problems while allocating more resources to hard ones, yielding significant overall efficiency gains.

## Highlights & Insights
- **Elegant Application of Quantile Regression**: Unlike conventional calibration methods that predict only the mean, quantile regression yields distributional information about success probabilities (upper bound, median, lower bound). The lower quantile provides a conservative estimate that is critical for resource allocation decisions—it is preferable to over-sample than to miss a correct answer.
- **Root Cause Analysis of PRM Miscalibration**: PRM training is policy-dependent, meaning that the calibration of a given PRM varies entirely across different LLMs. This paper clearly identifies this fundamental cause and offers an elegant remedy.
- **Concise IAS Sampling Complexity Formula**: $N_{\text{IAS}} = \lceil \log(1-C)/\log(1-p) \rceil$ is straightforward and practical, directly transferable to any scenario requiring adaptive sampling.

## Limitations & Future Work
- Calibration data collection requires extensive rollouts from the target LLM, incurring non-trivial cost.
- Validation is limited to mathematical reasoning; performance on code generation, open-domain QA, and other tasks remains untested.
- The Monte Carlo rollout sample size ($N_{\text{MC}}=8$) is relatively small, potentially yielding inaccurate empirical probability estimates.
- **Future Directions**: Cross-model calibration transfer; efficient calibration methods that reduce the number of required rollouts; integration with more complex search strategies such as MCTS.

## Related Work & Insights
- **vs. Snell et al. (2024)**: That work studies a general framework for inference-time scaling but does not address PRM calibration.
- **vs. Shepherd-PRM**: Uses fully automated annotation but with lower precision; the proposed calibration method can compensate for this shortcoming.
- **vs. Temperature Scaling**: Temperature scaling assumes well-calibrated log-odds and is unsuitable for the intermediate probability prediction setting of PRMs.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic study of PRM calibration; the application of quantile regression is highly creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple PRMs, LLMs, and benchmarks, though broader task diversity would strengthen the evaluation.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem formulation is clear; theory and experiments are tightly integrated.
- Value: ⭐⭐⭐⭐⭐ PRM calibration is a critical bottleneck in inference-time scaling; the proposed solution is both concise and practical.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Unlocking Multimodal Mathematical Reasoning via Process Reward Model](unlocking_multimodal_mathematical_reasoning_via_process_reward_model.md)
- [\[NeurIPS 2025\] DreamPRM: Domain-Reweighted Process Reward Model for Multimodal Reasoning](dreamprm_domain-reweighted_process_reward_model_for_multimodal_reasoning.md)
- [\[NeurIPS 2025\] Stop Summation: Min-Form Credit Assignment Is All Process Reward Model Needs for Reasoning](stop_summation_minform_credit_assignment_is_all_process_rewa.md)
- [\[ACL 2026\] Efficient Process Reward Modeling via Contrastive Mutual Information](../../ACL2026/llm_reasoning/efficient_process_reward_modeling_via_contrastive_mutual_information.md)
- [\[NeurIPS 2025\] Large Language Models Can Learn and Generalize Steganographic Chain-of-Thought under Process Supervision](large_language_models_can_learn_and_generalize_steganographic_chain-of-thought_u.md)

<!-- RELATED:END -->
