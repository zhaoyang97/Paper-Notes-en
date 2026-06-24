---
title: >-
  [Paper Note] The Limits of Inference Scaling Through Resampling
description: >-
  [ICLR 2026][Reasoning][Inference Scaling] This paper demonstrates both theoretically and empirically that when verifiers are imperfect (e.g., incomplete unit test coverage, non-zero false positive rates), scaling inference compute through "repeated sampling until passing a verifier" hits an **insurmountable accuracy ceiling**. Regardless of the compute budget allocated to a weak model, it cannot match the single-call accuracy of a sufficiently strong model…
tags:
  - "ICLR 2026"
  - "Reasoning"
  - "Inference Scaling"
  - "Resampling"
  - "Imperfect Verifier"
  - "False Positive"
  - "Test-time Compute"
date: 2026-05-08
content_hash: e6cbc43f9a978928
---

# The Limits of Inference Scaling Through Resampling

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=j8H84v6AZ1](https://openreview.net/forum?id=j8H84v6AZ1)  
**Code**: https://github.com/benediktstroebl/inference-scaling-limits  
**Area**: LLM Inference  
**Keywords**: Inference Scaling, Resampling, Imperfect Verifier, False Positive, Test-time Compute

## TL;DR
This paper demonstrates both theoretically and empirically that when verifiers are imperfect (e.g., incomplete unit test coverage, non-zero false positive rates), scaling inference compute through "repeated sampling until passing a verifier" hits an **insurmountable accuracy ceiling**. Regardless of the compute budget allocated to a weak model, it cannot match the single-call accuracy of a sufficiently strong model, and the optimal number of samples is often as low as single digits.

## Background & Motivation

**Background**: Test-time scaling (inference scaling) is widely expected to allow weak models to catch up with strong models by increasing computation. **Verifier-based resampling** is of particular interest: candidates are generated until one passes a verifier (e.g., unit tests). Its appeal stems from an empirical "inference scaling law"—the proportion of tasks where at least one correct solution is found predictably increases with the number of samples over multiple orders of magnitude (as seen in "Large Language Monkeys" by Brown et al. 2024). The same mechanism is used on the training side: filtering data via rejection sampling against a verifier to distill or train reasoning models.

**Limitations of Prior Work**: This optimistic narrative **assumes that verifiers are reliable**. However, in real-world scenarios like coding or reasoning, only **imperfect verifiers** are available—unit tests have incomplete coverage, and LM-as-judge makes errors. These verifiers possess a fatal attribute: **false positives**, where incorrect solutions pass the verifier. Once a solution is accepted, resampling cannot distinguish between a false positive and a true positive.

**Key Challenge**: Resampling only resolves **false negatives** (correct solutions wrongly rejected by the verifier)—eventually, a true positive will be sampled. However, it is **powerless against false positives**: once a false positive is accepted, the process terminates, and resampling cannot reduce the probability of "returning a false positive." Consequently, the false positive rate of the verifier becomes a hard ceiling for resampling accuracy, independent of the compute budget. Worse, the authors observe that weak models have systematically higher false positive rates—they are better at generating "flimsy" solutions that happen to exploit test gaps.

**Goal**: To quantify this intuition into two testable propositions: (1) What is the maximum accuracy a weak model can achieve even with infinite budget? Can it catch up with strong models? (2) When false positives incur real-world costs (e.g., deploying buggy code), what is the optimal number of samples?

**Core Idea**: Reformulate "resampling-based inference scaling" as a **decision problem with false positive costs**. Each additional sample might hit a true positive (benefit) or a false positive (cost). When the expected cost of the latter outweighs the expected benefit of the former, further sampling yields negative utility. This leads to the conclusion that a ceiling exists and the optimal number of samples is finite and low.

## Method

### Overall Architecture

This is a **mechanistic analysis + theoretical modeling** paper. Instead of proposing a new model, it deconstructs the obstacles that resampling-based scaling encounters under imperfect verifiers. The research design is a clear controlled experiment:

The system is split into two parts: a **Generator** and a **Verifier**. Generators consist of various language models (Cohere Command series, GPT-4o, Llama-3.1 series, Code Llama series). The verifiers are **original unit tests from programming benchmarks** (limited coverage, prone to false positives). A key experimental insight is the use of **dual tests**: the original finite unit tests from HumanEval / MBPP serve as the **verifier** (deciding if a solution is accepted), while the more comprehensive hidden tests from HumanEval+ / MBPP+ serve as the **ground truth** (determining if the accepted solution is actually correct). The gap between these two sets directly exposes the false positives.

Based on this framework, the paper advances along three lines: ① Quantifying the "conditional accuracy ceiling" for each model under infinite budget to see if weak models can match strong ones (Section 3); ② Introducing costs for false positives to plot cost-aware inference scaling curves and find the optimal $K$ (Section 4); ③ Using a Bayesian model with easy/hard tasks to generalize these empirical findings into domain-agnostic theoretical conclusions (Appendix C).

### Key Designs

**1. Conditional Accuracy Ceiling: Weak models cannot catch up even with infinite budget**

To address the belief that compute can bridge the gap between weak and strong models, the paper provides a clean counterexample condition. Let $P_{\text{strong}}(\text{Correct})$ be the **single-call** accuracy of a strong model, and $P_{\text{weak}}(\text{Correct} \mid \text{Pass Verifier})$ be the probability that a weak model's solution is truly correct **given it passed the verifier**. If:

$$P_{\text{strong}}(\text{Correct}) > P_{\text{weak}}(\text{Correct} \mid \text{Pass Verifier})$$

Then **no matter how large the compute budget**, the weak model cannot match the strong model's single-call performance. The reason is straightforward: at best, resampling ensures the weak model almost certainly "finds a solution that passes the verifier" (at infinite budget, the hit rate approaches 1), but the proportion of false positives among returned solutions (i.e., $1 - P_{\text{weak}}(\text{Correct}\mid\text{Pass})$) is a **constant unaffected by the number of samples**. Resampling does not reduce this conditional probability. Thus, the "infinite-budget accuracy" of a weak model is pinned to $P_{\text{weak}}(\text{Correct}\mid\text{Pass})$. In the paper's figures, this manifests as a horizontal cutoff: any model below the line can never exceed GPT-4o's single-call accuracy. Empirically, this "false positive rate decreasing linearly with capability" relationship holds consistently across diverse model families (Command, GPT-4o, Llama-3.1).

**2. Pricing False Positives: Optimal sampling $K$ is limited, low, or even zero**

While the previous design addresses the "infinite budget ceiling," a more realistic question for deployment is the risk of each sample. Code that passes unit tests but contains subtle bugs can cause significant losses when deployed. This cost and the benefit of "saving labor" are on different scales and vary by scenario. Thus, the paper introduces a **cost-benefit ratio** (C/B-Ratio): true positive benefit is $V_{TP}=+1$, and false positive cost is $V_{FP}$ (e.g., $0, 1, 2, 4, 8$). If a solution passing the verifier is found within $K$ samples, rewards are settled based on its truth; otherwise, the reward is 0.

By generating 200 samples per task on HumanEval and running 1000 random permutations of the sampling order, the authors calculate the average reward for each $K$ to plot **cost-aware inference scaling curves**. The conclusion is counter-intuitive: **even when compute cost is zero**, the optimal sample count $K_{\text{opt}}$ is **finite and very low**. At C/B-Ratio = 4, $K_{\text{opt}} \le 5$ for all four models. When the ratio is high enough, **the optimal $K$ becomes 0**: the expected cost of a false positive outweighs the expected benefit, resulting in a constant negative reward. This bends the "inference scaling" curve from one that rises indefinitely into one that peaks quickly and then drops, failing to close the gap between weak and strong models.

**3. Bi-modal Difficulty + Bayesian Belief Update: Explaining why false positive rates climb with K**

Design 2 leaves an anomaly: sampling is a memoryless process, so why does the false positive rate **increase with $K$** (Figure 5)? The paper uses a theoretical model of easy/hard tasks to provide a mechanism. Consider two task types: easy $T_1$ (prior $p_1$, accuracy $r_1$) and hard $T_2$ (prior $p_2$, accuracy $r_2$, with $r_1 > r_2$). The verifier has completeness $c$ (probability of accepting a correct solution) and soundness $s$ (probability of rejecting an incorrect solution). The probability of a sample being rejected is:

$$\beta_i = (1-c)\,r_i + s\,(1-r_i), \quad i \in \{1, 2\}$$

The key is **Bayesian belief update**: every time a solution is rejected, the posterior belief that "this is a hard task $T_2$" increases. After $k-1$ rejections, the posterior belief is:

$$p^{(k)}_{T_i} = \frac{\beta_i^{\,k-1}\, p_i}{\beta_1^{\,k-1} p_1 + \beta_2^{\,k-1} p_2}$$

Intuitively, easy tasks are solved in the first few attempts and exit the sampling pool, while **those remaining are increasingly likely to be hard tasks**—which are exactly the tasks prone to producing false positives. By weighting the posterior beliefs into true/false positive probabilities ($P^{(k)}_{TP}, P^{(k)}_{FP}$), the expected value of the $k$-th sample is:

$$EV_k = \left[V_{TP}\, P^{(k)}_{TP} + V_{FP}\, P^{(k)}_{FP}\right] \cdot \left[\beta_1^{\,k-1} p^{(k)}_{T_1} + \beta_2^{\,k-1} p^{(k)}_{T_2}\right]$$

Total reward is $\text{Reward} = \sum_{k=1}^{K} EV_k$, and $K_{\text{opt}}$ is the $K$ that maximizes it. This model, using parameters from Llama-3.1-8B on HumanEval+ ($r_1=0.87, r_2=0.13, c=1, s=0.75, p_1=0.58$), reproduces empirical curves. Since it is not tied to "code," it generalizes the conclusion as a **domain-agnostic** law: any resampling with non-zero false positive rates will hit the same ceiling.

## Key Experimental Results

### Main Results: Generalization Gaps under Infinite Budget (Section 3)

On HumanEval+ / MBPP+, at least 200 samples were generated per task (1000 for Command Light). Original tests served as verifiers, while expanded hidden tests served as ground truth.

| Observed Dimension | Weak Models | Strong Models (e.g., GPT-4o) | Conclusion |
|--------|------|----------|------|
| Single-call Accuracy (x-axis) | Low | High | — |
| Infinite Budget Accuracy (y-axis) | Significantly lower ceiling | Higher ceiling | Weak models have higher FP rates |
| FP Rate vs. Capability | High | Low | Nearly **linearly negatively correlated** across all model families |
| Catching up to GPT-4o via resampling | Models below the cutoff **cannot** | — | Ceiling is independent of compute budget |

This gap is **primarily driven by a small subset of tasks with poor unit tests**: analysis of only these tasks shows a steeper negative correlation between capability and false positive rates.

### Analysis Results: Optimal Sampling Counts with Costs (Section 4)

| Cost-Benefit Ratio (C/B) | Optimal Sampling $K_{\text{opt}}$ | Explanation |
|------|------|------|
| 0 (Standard assumption) | Finite and low | Even at zero cost, the curve peaks early |
| 4 | $K \le 5$ (All four models) | Negative utility of FPs outweighs benefits |
| High (e.g., FP cost ≈ 10x TP benefit) | $K = 0$ (Almost all models) | Optimal strategy is not to sample; the model is effectively "useless" |

### Key Findings
- **FP rate rising with $K$** seems to violate memorylessness but is caused by **bi-modal task difficulty**: easy tasks are solved early, leaving hard tasks that are more likely to yield false positives. This aligns with Chen et al. (2024a) on the "inverse U-shape accuracy curve."
- **Unexplained inter-model differences**: Llama-3.1-70B's FP rate rises sharply with $K$, whereas Code Llama / Command families rise much slower, leading to higher optimal $K$ for the latter (especially at low C/B). The authors admit no intuitive explanation for this as of yet.
- **Worse code quality in false positives (Section 5)**: In terms of naming conventions, line length, and comments, false positive solutions (passing original tests but failing expanded ones) are **categorically worse** than truly correct ones across models. This suggests imperfect verifiers harm not only functional correctness but also maintainability, with weak models suffering more.

## Highlights & Insights
- **Dual-test experimental design is the pivot of the paper**: Using the same samples with original tests as verifiers and expanded tests as ground truth separates "passing the verifier" from "true correctness." This gap is the false positive rate, measurable without manual annotation.
- **Reframing inference scaling as a decision problem**: Previous inference scaling curves assumed zero cost for false positives (C/B=0). By simply reintroducing this neglected cost, the conclusion flips from "infinite rise" to "early peak and drop"—a single modeling assumption shift disrupts the optimistic narrative.
- **Bayesian updates explain a counter-intuitive phenomenon**: Sampling is memoryless, but the fact that "remaining tasks are harder" provides memory. This perspective is transferable to any "sample-then-filter" scenario (e.g., agent planning, math proofs).
- **Warning for the training side**: Many SOTA reasoning models (DeepSeek-R1, Bespoke-Stratos) rely on rejection sampling against verifiers to filter data. If verifiers are imperfect, datasets will contain mislabeled samples. Without stronger base models or more accurate verifiers, the gains from resampling for training will also hit a ceiling.

## Limitations & Future Work
- **Limited to coding tasks**: Code provides a clear example of imperfect verifiers, but behavior in reasoning, web agents, or agent-user interactions may differ. While the theory is domain-agnostic, multi-domain empirical evidence is lacking.
- **Human-written unit tests as verifiers**: Real-world deployments might use LM-generated tests, which introduce inconsistencies and false negative risks, potentially **amplifying** the generalization gap.
- **Unmeasured C/B ratio**: $K_{\text{opt}}$ depends on the cost-benefit ratio. While the paper shows $K$ is low across various ratios, it does not map specific ratios to real deployment scenarios.
- **Uexplored mitigation strategies**: Strategies like refining solutions after they pass the verifier, using diversity-increasing sampling, or fine-tuning on code quality metrics remain open questions.
- **Benchmark contamination / Prompt sensitivity**: Models might be over-optimized for standard tests, and prompt engineering could influence the generation of false positives.

## Related Work & Insights
- **vs. Brown et al. 2024 (Large Language Monkeys)**: They showed the "at least one correct" ratio scales predictably, fueling optimism. This paper points out that their curve assumes an **Oracle verifier**; with imperfect verifiers, "passing" is not "correct," and the scaling law fails.
- **vs. Chen et al. 2024a (Scaling Laws for Compound Inference Systems)**: They observed an inverse U-shape due to task heterogeneity. This paper connects that to the "FP rate rising with $K$" mechanism, further weakening the case for resampling as an effective scaling strategy.
- **vs. Gao et al. 2022 (Reward model over-optimization)**: Both study risks of optimizing against imperfect rewards. While they focus on training-time reward mismatch, this paper focuses on **inference-time resampling**, where the ceiling comes directly from false positives.
- **Call for verifier research**: The authors argue for "verification technology" to be a standalone sub-field with its own metrics and benchmarks—especially since models trained against weak verifiers may learn to exploit them rather than solve the problem, posing safety risks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Reintroduces "false positive cost" into inference scaling models, resulting in a ceiling conclusion contrary to the mainstream optimistic narrative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers four model families and multiple benchmarks/cost ratios, though limited to coding and human tests.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logical chain; empirical and theoretical results validate each other; core thesis explained by a single inequality.
- Value: ⭐⭐⭐⭐⭐ Warns of hidden limits in both inference scaling and rejection-sampling-based training, emphasizing the need for higher verifier quality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] UniScale: Adaptive Unified Inference Scaling through Online Joint Optimization of Model Routing and Test-Time Scaling](../../ICML2026/llm_reasoning/uniscale_adaptive_unified_inference_scaling_via_online_joint_optimization_of_mod.md)
- [\[ICLR 2026\] InftyThink: Breaking the Length Limits of Long-Context Reasoning in Large Language Models](inftythink_breaking_the_length_limits_of_long-context_reasoning_in_large_languag.md)
- [\[ACL 2026\] ToolPRM: Fine-Grained Inference Scaling of Structured Outputs for Function Calling](../../ACL2026/llm_reasoning/toolprm_fine-grained_inference_scaling_of_structured_outputs_for_function_callin.md)
- [\[ICLR 2026\] Strategic Scaling of Test-Time Compute: A Bandit Learning Approach](strategic_scaling_of_test-time_compute_a_bandit_learning_approach.md)
- [\[ICLR 2026\] Latent Veracity Inference for Identifying Errors in Stepwise Reasoning](latent_veracity_inference_for_identifying_errors_in_stepwise_reasoning.md)

</div>

<!-- RELATED:END -->
