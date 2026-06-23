---
title: >-
  [Paper Note] When LLMs Get Significantly Worse: A Statistical Approach to Detect Model Degradations
description: >-
  [ICLR 2026][LLM Evaluation][Quantization] Addressing the question of whether a quantized/sparsified LLM has actually degraded or if the change is merely evaluation noise, this paper formalizes the problem as a statistical hypothesis test. It proposes the **Exact One-sided McNemar Test**, which, instead of examining task-level aggregate accuracy, compares the c
tags:
  - ICLR 2026
  - LLM Evaluation
  - Quantization
date: 2026-05-08
content_hash: f3303dffa600931c
---
# When LLMs Get Significantly Worse: A Statistical Approach to Detect Model Degradations

**Conference**: ICLR 2026  
**Paper**: Published as a conference paper at ICLR 2026 (Amazon)  
**Code**: https://github.com/amazon-science/LLM-Accuracy-Stats (Available)  
**Area**: LLM Evaluation / Model Compression / Statistical Hypothesis Testing  
**Keywords**: Model degradation detection, McNemar test, Quantization, Hypothesis testing, Sample-wise comparison

## TL;DR
Addressing the question of whether a quantized/sparsified LLM has actually degraded or if the change is merely evaluation noise, this paper formalizes the problem as a statistical hypothesis test. It proposes the **Exact One-sided McNemar Test**, which, instead of examining task-level aggregate accuracy, compares the correctness of two models sample-wise. This allows for the detection of even a 0.3% accuracy drop as "true degradation" while maintaining a controlled false positive rate.

## Background & Motivation
**Background**: To reduce LLM inference costs and latency, the industry employs numerous "optimization" techniques—ranging from theoretically lossless efficient kernels and speculative decoding to quantization (INT4/FP8) and sparsification that can slightly alter high-dimensional output distributions. Determining whether an optimization is "safe" typically involves running accuracy benchmarks and observing the percentage drop, then applying an empirical threshold (e.g., "lossless if drop $\le 2\%$").

**Limitations of Prior Work**: Such judgments based on "aggregate accuracy differences" are increasingly untenable for modern LLMs. Even at temperature 0 and with theoretically lossless changes (altering hardware, frameworks, single-GPU vs. tensor parallelism, or even re-running the same command), the non-associativity of floating-point operations $(a+b)+c \neq a+(b+c)$ accumulates errors along the computation graph, causing the same model to generate different answers. Empirical results show these "theoretically lossless" changes produce 1.3%–2.8% sample flips. Consequently, the boundary between "theoretically lossless vs. lossy" is blurred, and fixed thresholds lead to both false negatives for real degradation and false positives for harmless noise.

**Key Challenge**: Evaluation of two models uses the **same set of samples** $x_1,\dots,x_N$, meaning the two accuracy estimates $\hat\gamma$ and $\hat\beta$ are **not independent**. Incorrectly assuming independence when calculating the variance of the difference leads to overestimation (estimated at $\approx \sqrt5 \approx 2.2$ times in typical LLM scenarios), which misclassifies real degradation as statistical fluctuation. $\text{Var}[\hat\gamma-\hat\beta] \neq \tfrac{\gamma(1-\gamma)}{N} + \tfrac{\beta(1-\beta)}{N}$, mirroring the complexity of paired tests used in clinical trials.

**Goal**: To provide a rigorous statistical definition for "model degradation detection" and a framework that: ① controls the type-I error rate (false positive rate), ② maximizes testing power, ③ aggregates decisions across multiple benchmarks into a single conclusion, and ④ reduces evaluation costs.

**Key Insight**: McNemar (1947) proposed a test for "paired proportion comparison" that focuses only on samples where two models **disagree**. The authors adapt this to LLM degradation detection and refine it into a one-sided version with exact p-value calculations to enhance power.

**Core Idea**: **Shift away from aggregate accuracy at the task level; focus on sample-wise correctness comparison.** Estimate the "degradation probability" $q_\downarrow$ on inconsistent samples and use a binomial test to determine if it is significantly greater than $1/2$.

## Method

### Overall Architecture
The method is a statistical testing workflow centered around $2\times2$ contingency tables. The input consists of sample-wise correctness scores ($L\in\{0,1\}$) for a baseline model $M$ and an optimized model $\tilde M$ on the **same set** of $N$ evaluation samples. The output is a p-value (or an aggregated p-value across benchmarks) used to determine if "$\tilde M$ has truly degraded" under a controlled false positive rate.

The pipeline involves: mapping each sample into a $2\times2$ contingency table ($a,b,c,d$ representing both wrong / only baseline correct / only optimized correct / both correct) → defining the degradation probability $q_\downarrow=b/(b+c)$ solely on "disagreement" samples → calculating the p-value via the **Exact One-sided McNemar Test** (effectively a binomial test) → conducting power analysis to determine which samples to prune for cost savings → using three types of aggregated tests (Pooled / Max Drop / Fisher) for multiple benchmarks to reach a final decision.

### Key Designs

**1. Sample-wise Comparison and Degradation Probability: Explicitly Modeling Paired Correlation**

The limitation of previous methods is that using the same samples makes $\hat\gamma$ and $\hat\beta$ dependent, preventing the variance of the accuracy difference from being a simple sum of marginal variances. This paper treats each sample as a "joint correctness experiment," resulting in a $2\times2$ contingency table and defining two core metrics: flip probability $p_\updownarrow := P[L(M(X)) \neq L(\tilde M(X))] = P_b + P_c$, and (conditional) degradation probability:

$$q_\downarrow := P[L(M(X))=1 \mid L(M(X)) \neq L(\tilde M(X))] = \frac{P_b}{p_\updownarrow}.$$

Intuitively, $p_\updownarrow$ quantifies "how frequently the models disagree," while $q_\downarrow$ quantifies "the proportion of disagreements where the baseline was correct and the optimization failed" (i.e., true degradation direction). A critical fact (Fact 1) strictly links this to accuracy: **The optimized model accuracy $\beta < \gamma$ if and only if $q_\downarrow > 1/2$.** This converts the complex problem of comparing two correlated proportions into a standard problem of testing if a binomial parameter $q_\downarrow$ exceeds $1/2$.

**2. Exact One-sided McNemar Test: Controlling False Positives**

With $q_\downarrow$, the test statistic is its empirical estimate $\hat q_\downarrow := \dfrac{b}{b+c}$. For a fixed $b+c$, $b \sim \text{Binomial}(b+c, q_\downarrow)$ under the null hypothesis $q_\downarrow = 1/2$. Since the focus is exclusively on "degradation," a **one-sided** p-value is used for higher sensitivity, calculated exactly via the binomial distribution. The authors relate this to the classic McNemar statistic:

$$\frac{(b-c)^2}{b+c} = (2\hat q_\downarrow - 1)^2(b+c),$$

showing that the classic version is merely a squared scaling of the affine-transformed $\hat q_\downarrow$. The use of an exact one-sided p-value avoids the chi-square approximation, correctly controlling type-I error even for small samples or minor degradations.

**3. Power Analysis and Dataset Compression: Pruning Non-flipping Samples**

To reduce evaluation costs, the authors prove via asymptotic analysis of the accuracy difference $\delta := \gamma - \beta = p_\updownarrow(2q_\downarrow - 1)$ that the test power is determined by the Signal-to-Noise Ratio (SNR):

$$\text{SNR} := \sqrt{N/p_\updownarrow} \, \delta.$$

Observation: Pruning the dataset to keep only inconsistent samples (post-hoc setting $a=d=0$) yields $\delta \to \delta/p_\updownarrow$, $N \to Np_\updownarrow$, and $p_\updownarrow \to 1$. Substituting these shows the **SNR remains unchanged**. Recommendation 1: **For degradation detection, removing samples that are unlikely to flip reduces costs without losing signal.** The authors propose a "noise simulation" where the baseline is run multiple times at finite temperature (e.g., 10 times at Temp=0.3); "never-flip" samples (always correct or always wrong) can be removed. In MMLU-Pro, deleting 5,604 never-flip samples halved the dataset size while retaining nearly all degradation signals.

**4. Three Aggregated Tests for Multi-task Decisions**

To handle multiple benchmarks with varying sample sizes, the authors propose three complementary aggregations: **Pooled** (summing $b,c$ across all tasks; most sensitive when degradation is uniform); **Max Drop** (calculating standardized statistics $z_{\max} = \max_i \hat z_i$ and using Monte Carlo for the null distribution; best when only one benchmark is affected); and **Fisher’s method** (combining p-values via $\chi^2 = -2\sum \ln p_i$). A final **combined decision** uses Bonferroni correction: if any of the three tests reject at level $\alpha$, degradation is detected (controlling type-I error within $3\alpha$).

## Key Experimental Results

### Main Results
Using Leaderboard v2 (6 benchmarks, 25,282 total samples) on Llama-3.1 8B, Llama-3.3 70B, and Mistral-Small-3.1. Selected results for 8B ($\hat\delta$: accuracy drop vs baseline; $p_{\text{pool}}$: Pooled p-value; $\hat p_\updownarrow$: flip probability):

| Variant | Type | $\hat\delta$ | $p_{\text{pool}}$ | $\hat p_\updownarrow$ | Decision |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline (Rerun) | Lossless | -0.02% | 6.29e-01 | 1.31% | No Degradation ✓ |
| A100 Swap | Lossless | -0.14% | 9.21e-01 | 2.73% | No Degradation ✓ |
| FP8 | Lossy | -0.04% | 6.01e-01 | 8.71% | No Degradation ✓ |
| KV-FP8 | Lossy | 0.79% | 1.69e-05 | 9.03% | **Significant Degradation** |
| w4a16 (INT4) | Lossy | 1.73% | 4.80e-15 | 12.63% | **Significant Degradation** |
| 2:4 Sparse | Pruning | 2.59% | 1.09e-19 | 20.99% | **Significant Degradation** |

Crucial Comparison: Theoretically lossless variants show 1.3%–2.8% flipping, but the test correctly identifies "no significant degradation." Meanwhile, KV-FP8 for 8B (0.79% drop) and 70B (0.3% drop) are both detected as significant, whereas fixed threshold rules ($\hat\delta > 2\%$ or $\hat p_\updownarrow \ge 5\%$) fail completely on the 70B model.

### Ablation Study / Efficiency
On MMLU-Pro (12,032 samples) using noise simulation pruning:

| Configuration | Sample Count | w4a16 Flip Rate | KV-FP8 Flip Rate |
| :--- | :--- | :--- | :--- |
| Full | 12,032 | 20.55% | 15.49% |
| Removing 5,604 never-flip | 6,807 | 31.09% | 24.42% |

Halving the dataset size resulted in minimal loss of SNR, consistent with the theory.

### Key Findings
- **Sample-wise comparison is the source of power**: Distinguishing "lossless changes" from "true degradation" relies on sample-wise testing and proper weighting of sample sizes, not on absolute values of $\hat\delta$ or $\hat p_\updownarrow$.
- **Flip probability is a poor metric**: High flip rates can originate from harmless noise; FP8 (8.71% flip) is misreported by fixed flip-rate rules but shows no sign of degradation under statistical testing.
- **Accidental Bug Discovery**: The test revealed that w4a16 for 70B on vLLM had a massive accuracy drop ($\hat\delta=39.46\%$) that was far lower on `transformers`, exposing implementation issues in the inference stack.
- **0.3% is enough for a conclusion**: With proper statistical treatment, even an empirical 0.3% accuracy drop can be confidently attributed to degradation rather than noise.

## Highlights & Insights
- **Turning engineering intuition into controlled statistical decisions**: Replaces arbitrary empirical thresholds with a decision process where the false positive rate is controlled and testing power is analytical. The implementation integrates into `LM Evaluation Harness`, requiring minimal effort.
- **Empirical evidence against fixed thresholds**: Theoretically lossless changes causing up to 2.8% flips and 0.5% single-task deviations prove that no fixed threshold or flip-rate rule can be robust across different models and datasets.
- **Power analysis guides cost savings**: The proof that SNR is conserved when pruning non-flipping samples turns statistical theory into a practical dataset compression strategy.
- **Clear aggregation strategy**: Pooled, Max Drop, and Fisher tests cover various degradation patterns (uniform vs. task-specific), with Bonferroni providing a conservative safety net.

## Limitations & Future Work
- The current focus is on **binary scoring** (correct/incorrect). Extensions for non-binary scores are discussed in the appendix, but processing for continuous or partial-credit tasks requires further validation.
- The combined decision uses Bonferroni ($3\alpha$), which is conservative; since the three aggregated p-values are correlated, there is room to improve power through better merging techniques.
- Noise simulation for sample pruning depends on manual hyperparameters (Temp, iteration count); its transferability across diverse tasks requires more verification.
- The test provides a binary "significant degradation" answer; "significant" does not always mean "practically impactful." Users must still consider the magnitude $\delta$ alongside the p-value.

## Related Work & Insights
- **vs. Dutta et al. (2024)**: They proposed using "score flip probability" but lacked statistical significance quantification and used an arbitrary $0-2\%$ threshold. Ours proves that small deviations are often sufficient for significant degradation detection once correctly modeled.
- **vs. Classic McNemar / Yang et al. (2025b)**: Classic McNemar is for two-sided "which is better" tests. Ours is a one-sided, exact version tailored for optimization-induced degradation.
- **vs. Standard `lm-eval` practice**: Standard methods treat models as independent, overestimating error by $\approx 2.2$ times and causing false negatives. Ours corrects this by accounting for paired correlation.

## Rating
- Novelty: ⭐⭐⭐⭐ Solid application of Exact One-sided McNemar to LLMs with rigorous power analysis and aggregation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple model series, various lossless/lossy variants, 6 benchmarks, and discovered a vLLM bug.
- Writing Quality: ⭐⭐⭐⭐ Clear derivation, closed loop between theory and experiment.
- Value: ⭐⭐⭐⭐⭐ Extremely practical for model compression/deployment quality gates, integrated into industry-standard tools.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LLMs Get Lost In Multi-Turn Conversation](llms_get_lost_in_multi-turn_conversation.md)
- [\[ICLR 2026\] Noisy but Valid: Robust Statistical Evaluation of LLMs with Imperfect Judges](noisy_but_valid_robust_statistical_evaluation_of_llms_with_imperfect_judges.md)
- [\[ICLR 2026\] When to Ensemble: Identifying Token-Level Points for Stable and Fast LLM Ensembling](when_to_ensemble_identifying_token-level_points_for_stable_and_fast_llm_ensembli.md)
- [\[ICLR 2026\] Credit-Budgeted ICPC-Style Coding: When Agents Must Pay for Every Decision](credit-budgeted_icpc-style_coding_when_agents_must_pay_for_every_decision.md)
- [\[ICLR 2026\] Pitfalls in Evaluating Language Model Forecasters](pitfalls_in_evaluating_language_model_forecasters.md)

</div>

<!-- RELATED:END -->
