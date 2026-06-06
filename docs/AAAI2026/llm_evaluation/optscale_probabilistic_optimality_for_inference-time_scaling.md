---
title: >-
  [Paper Note] OptScale: Probabilistic Optimality for Inference-time Scaling
description: >-
  [AAAI 2026][LLM Evaluation][Inference-time scaling] This paper proposes OptScale, a probabilistic optimality framework that models the probability distribution of verifier scores to derive a theoretical lower bound on th…
tags:
  - "AAAI 2026"
  - "LLM Evaluation"
  - "Inference-time scaling"
  - "parallel sampling"
  - "probabilistic optimality"
  - "Best-of-N"
  - "computational efficiency"
date: 2026-05-08
content_hash: 3b1cc4feaeb94a7e
---

# OptScale: Probabilistic Optimality for Inference-time Scaling

**Conference**: AAAI 2026
**arXiv**: [2506.22376](https://arxiv.org/abs/2506.22376)  
**Code**: [GitHub](https://github.com/Albertwyk/OptScale)  
**Area**: LLM Evaluation
**Keywords**: Inference-time scaling, parallel sampling, probabilistic optimality, Best-of-N, computational efficiency

## TL;DR

This paper proposes OptScale, a probabilistic optimality framework that models the probability distribution of verifier scores to derive a theoretical lower bound on the optimal number of samples, dynamically determining the minimum number of samples required per problem and substantially reducing computational overhead while preserving inference accuracy.

## Background & Motivation

**Background**: Inference-time scaling has become a mainstream technique for enhancing LLM reasoning capabilities. By generating multiple candidate answers in parallel and selecting the best, it can significantly improve performance on tasks such as mathematical reasoning.

**Limitations of Prior Work**: Existing methods suffer from a severe efficiency bottleneck—the number of candidate solutions scales linearly with computational cost, and token consumption grows linearly with the sampling count $N$, leading to wasteful use of computational resources. Existing dynamic allocation strategies rely on heuristic rules and lack a rigorous mathematical foundation.

**Key Challenge**: Inference accuracy calls for as many samples as possible, yet computational budgets are limited. Simple problems do not require many samples, while extremely difficult problems may remain unsolvable regardless of how many are drawn. How can the optimal number of samples be determined theoretically?

**Goal**: To establish a probabilistic theoretical foundation for inference-time scaling, derive a closed-form solution for the optimal sampling count, and implement an efficient, practical adaptive sampling algorithm.

**Key Insight**: The Best-of-N selection process is formalized as a probability distribution problem, and extreme value distribution theory is applied to derive the optimal sampling count $N^*$.

**Core Idea**: Verifier scores follow an estimable probability distribution. By leveraging extreme value theory, the minimum number of samples required to meet a target performance level and confidence threshold can be derived analytically.

## Method

### Overall Architecture

Given an input question $q$, the LLM generates $N$ candidate answers, a verifier (e.g., PRM) scores each answer, and the highest-scoring answer is selected. The core of OptScale is dynamically determining $N$: the distribution parameters $(\mu, \sigma)$ of verifier scores are first estimated, then a theoretical formula computes the minimum $N^*$ satisfying a performance threshold and confidence level, and sampling terminates immediately upon reaching $N^*$.

### Key Designs

1. **Probabilistic Optimality Theoretical Framework**: Verifier scores $\{s_i\}$ are assumed to be i.i.d. random variables following a probability density function $f_S(s|\theta, q)$. For $N$ independent samples, the CDF of the maximum $Y = \max\{s_1, \ldots, s_N\}$ is $F_Y(s) = [F_S(s)]^N$. The requirement $P(Y \geq s_{\min}) \geq \alpha$ ensures that with probability at least $\alpha$, at least one answer exceeding the quality threshold $s_{\min}$ is found. Solving this inequality yields a closed-form lower bound on the optimal sampling count:

    - $N^* \geq \lceil \log(1-\alpha) / \log F_S(s_{\min}) \rceil$
    - This formula elegantly transforms "how many samples are needed" into a function of the tail probability of the verifier score distribution.
    - When $F_S(s_{\min})$ is close to 1 (i.e., most samples are of high quality), $N^*$ is small, automatically reducing sampling.
    - When $F_S(s_{\min})$ approaches 0 (i.e., exceeding the threshold is nearly impossible), $N^*$ tends to infinity, signaling that the problem should be abandoned.
    - This is the first closed-form theoretical guidance with mathematical guarantees for parallel inference-time scaling.

2. **OptScale_t (Trainable Variant)**: Verifier scores are modeled as a truncated normal distribution (constrained to $[0,1]$), with core parameters $(\mu, \sigma)$. Two MLPs are trained on offline data to predict prior parameters $\bar{\mu}$ and $\bar{\sigma}$, respectively. At inference time, the MLPs provide prior predictions, which are then dynamically updated via MAP estimation as actual verifier scores $\mathcal{D} = \{s_k\}$ are observed. The MAP objective jointly optimizes a likelihood term (fitting observations) and a prior term (staying close to MLP predictions), balancing global trends with instance-specific signals.

3. **OptScale_0 (Training-free Variant)**: This variant requires no pretrained models whatsoever. For new problems, it initializes $(\mu_0, \sigma_0)$ using a heuristic strategy assuming uniform uncertainty, then directly estimates parameters from observed scores via MLE. As more verifier scores are observed, the parameter estimates are continuously refined. This variant requires no additional models or training data, is plug-and-play, and is suitable for rapid deployment. Experiments show it even outperforms the trainable variant in most scenarios.

4. **Adaptive Termination Mechanism**: Both variants ultimately substitute the updated $(\mu^*, \sigma^*)$ into $F_S(s_{\min})$ to compute $N^*$. Parameters are updated and $N^*$ is recomputed after each new sample is generated. Sampling terminates immediately once the current sample count $N \geq N^*$. This achieves three levels of adaptivity:

    - **Simple problems**: Distribution parameter estimates indicate high probability of quality, yielding a small $N^*$ and enabling rapid early stopping.
    - **Moderately difficult problems**: A moderate number of samples is required until distribution estimates stabilize.
    - **Extremely difficult problems**: Sampling is truncated when $N^*$ reaches a preset upper limit, avoiding meaningless over-sampling.

### Loss & Training

In the training phase, OptScale_t fits an MLP predictor on offline data to learn the mapping from problems to distribution parameters. No additional training is required at inference time. OptScale_0 is entirely training-free and operates purely at inference time.

## Key Experimental Results

### Main Results

Results on Deepseek-R1-Distill-Qwen-7B ($N=8$):

| Method | MATH-500 Acc. | MATH-500 Toks(↓) | GSM8K Acc. | GSM8K Toks(↓) | AIME24 Acc. | AIME24 Toks(↓) |
|---|---|---|---|---|---|---|
| Best-of-N | 94.8 | 22135 | 92.4 | 3582 | 70.0 | 79367 |
| Self-Consistency | 93.4 | 22135 | 90.1 | 3582 | 60.0 | 79367 |
| OptScale_0 | **94.8** | **11354** | **92.4** | **1687** | **70.0** | **49505** |
| OptScale_t | **94.8** | 18236 | **92.4** | 3492 | **70.0** | 53855 |

At $N=64$ on MATH-500: OptScale_0 uses only 110,001 tokens (vs. BoN's 174,693), while accuracy improves from 94.0% to 94.6%.

### Ablation Study

Results on QwQ-32B ($N=60$):

| Method | MATH-500 Acc. | Toks(↓) | AMC23 Acc. | Toks(↓) |
|---|---|---|---|---|
| Best-of-N | 94.8 | 230402 | 97.5 | 420481 |
| Self-Consistency | 95.4 | 230402 | 92.5 | 420481 |
| OptScale_0 | **95.8** | **107720** | **100.0** | **190633** |
| OptScale_t | **95.8** | 106412 | **100.0** | 202603 |

OptScale achieves 100% accuracy on AMC23 while reducing token consumption by approximately 54%.

On Llama-3.1-8B ($N=64$), OptScale_0 reduces GSM8K token consumption from 14,697 to 5,720 (a 61% reduction) while improving accuracy from 89.3 to 89.5.

### Key Findings

- OptScale_0 is more efficient than OptScale_t in most scenarios, as the training-free variant makes early stopping decisions more quickly on simple problems.
- OptScale remains effective on weaker models such as Llama-3.1-8B, with even more pronounced token savings.
- OptScale automatically reduces sampling for simple problems and actively caps computation for difficult ones, avoiding wasteful expenditure.
- MR-Thinking (a self-reflection approach) exhibits accuracy degradation on multiple benchmarks, suggesting that increasing per-sample reasoning depth is less effective than optimizing sampling strategy.

## Highlights & Insights

- This work is the first to establish a rigorous probabilistic theoretical foundation for parallel inference-time scaling, deriving a closed-form lower bound on the optimal sampling count.
- The method is highly practical: OptScale_0 is entirely training-free, plug-and-play, and incurs zero deployment overhead.
- Token consumption can be reduced by 37%–61% while maintaining or even improving accuracy.
- The theoretical framework is general-purpose, independent of specific LLMs or verifiers, and can be combined with any Best-of-N pipeline.

## Limitations & Future Work

- The theory assumes verifier scores are i.i.d.; in practice, different samples for the same problem may be correlated (e.g., at low temperatures).
- The truncated normal distribution assumption may be insufficiently precise for certain tasks; more flexible distribution families (e.g., mixture models) could yield improvements.
- Validation is primarily conducted on mathematical reasoning tasks; generalization to code generation, logical reasoning, and other domains remains to be tested.
- OptScale_t requires offline data to train the predictor; its generalization to entirely novel tasks warrants further evaluation.
- Integration with sequential reasoning methods such as tree search (MCTS) has not been explored.

## Related Work & Insights

- **vs. Self-Consistency**: SC selects answers via majority voting with a fixed $N$ and cannot adapt dynamically; OptScale dynamically adjusts $N$ and selects the best answer using a verifier.
- **vs. SEAL**: SEAL dynamically allocates computation but relies on heuristic uncertainty metrics; OptScale derives a closed-form solution from probabilistic theory.
- **vs. MR-Thinking**: Self-reflection methods increase token overhead per sample, and experiments show accuracy may actually decrease; OptScale achieves greater savings by reducing the number of unnecessary samples.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to introduce extreme value theory into inference-time scaling with a theoretical lower bound, though the technical innovation in probabilistic modeling itself is limited.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers multiple benchmarks and models (7B/8B/32B), but lacks validation on non-mathematical reasoning tasks.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theoretical derivations are clear and complete, with logically consistent presentation from framework to implementation.
- **Value**: ⭐⭐⭐⭐ — Dual contributions in theory and practice; the training-free variant offers high practical value, though the i.i.d. assumption limits the strict applicability of the theoretical framework.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] GuidedSampling: Steering LLMs Towards Diverse Candidate Solutions at Inference-Time](../../ICLR2026/llm_evaluation/guidedsampling_steering_llms_towards_diverse_candidate_solutions_at_inference-ti.md)
- [\[AAAI 2026\] Test-time Diverse Reasoning by Riemannian Activation Steering](test-time_diverse_reasoning_by_riemannian_activation_steering.md)
- [\[AAAI 2026\] Graph Out-of-Distribution Detection via Test-Time Calibration with Dual Dynamic Dictionaries](graph_out-of-distribution_detection_via_test-time_calibration_with_dual_dynamic_.md)
- [\[AAAI 2026\] Think How Your Teammates Think: Active Inference Can Benefit Decentralized Execution](think_how_your_teammates_think_active_inference_can_benefit_decentralized_execut.md)
- [\[ACL 2026\] Pressure-Testing Deception Probes in LLMs: Scaling, Robustness, and the Geometry of Deceptive Representations](../../ACL2026/llm_evaluation/pressure-testing_deception_probes_in_llms_scaling_robustness_and_the_geometry_of.md)

</div>

<!-- RELATED:END -->
