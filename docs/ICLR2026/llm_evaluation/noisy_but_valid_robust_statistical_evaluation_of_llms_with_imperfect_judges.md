---
title: >-
  [Paper Note] Noisy but Valid: Robust Statistical Evaluation of LLMs with Imperfect Judges
description: >-
  [ICLR 2026][LLM Evaluation][LLM-as-a-Judge] The paper utilizes a small set of human-annotated data to estimate the True Positive Rate and False Positive Rate (TPR/FPR) of an LLM-as-a-Judge. It constructs a "variance-corrected" critical threshold to process massive judge-generated labels, ensuring that the certification test maintains controlled Type-I error (avo
tags:
  - ICLR 2026
  - LLM Evaluation
  - LLM-as-a-Judge
  - Prediction-Powered Inference
date: 2026-05-08
content_hash: 8a027a265c86d3ab
---
# Noisy but Valid: Robust Statistical Evaluation of LLMs with Imperfect Judges

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=hEhxreaLdU](https://openreview.net/forum?id=hEhxreaLdU)  
**Code**: TBD  
**Area**: LLM Evaluation / Statistical Certification  
**Keywords**: LLM-as-a-Judge, Hypothesis Testing, Model Certification, Type-I Error Control, Calibration, Prediction-Powered Inference  

## TL;DR
The paper utilizes a small set of human-annotated data to estimate the True Positive Rate and False Positive Rate (TPR/FPR) of an LLM-as-a-Judge. It constructs a "variance-corrected" critical threshold to process massive judge-generated labels, ensuring that the certification test maintains controlled Type-I error (avoiding misclassifying unsafe models as safe) even when the judge itself is imperfect.

## Background & Motivation

**Background**: To "certify" an LLM (statistically guaranteeing its failure rate is below a safety threshold $\alpha$), researchers currently rely on two main paths: running public benchmarks (GLUE/MMLU, etc.) to measure empirical failure rates, or using human evaluation as the gold standard. The former suffers from data contamination, label noise, and overfitting; the latter is expensive and difficult to scale to the sample sizes required for statistical reliability. Consequently, more work is shifting toward **LLM-as-a-Judge**, using large models to score outputs in batches.

**Limitations of Prior Work**: Current practices almost exclusively treat judge outputs **directly as ground truth**, completely ignoring the judge's own noise—prompt sensitivity, domain dependence, systemic bias, and occasional hallucinations lead to inconsistent or biased annotations. As a result, certification conclusions are built on the **unverified blind faith** that the "judge is accurate," lacking statistical rigor and posing a real risk of certifying unsafe models as safe.

**Key Challenge**: Judge annotations are **abundant (cheap and scalable) but dirty**; human annotations are **accurate but scarce**. How can these two data sources be combined to enjoy the sample size dividends of the judge without contaminating statistical guarantees (especially Type-I error control) with the judge's bias?

**Key Insight (Difference from PPI)**: Prediction-Powered Inference (PPI) also uses "few clean labels + many dirty labels" to improve statistical power, but it treats the judge as a **black-box control variable** purely for variance reduction. This paper's goal is different—it aims for **interpretable certification**: explicitly modeling the judge's error profile (TPR/FPR). While this sacrifices some raw power (experimental results show Noisy HT is weaker than PPI), it gains **diagnostic capability**—telling practitioners whether a judge is fit for use and how accurate it needs to be.

**Goal**: Formalize reliability assessment as a hypothesis test: the null hypothesis $H_0: R_M = \mathbb{E}[S_M] \geq \alpha$ (the model's true failure rate exceeds the tolerance). Rejecting $H_0$ provides a statistical guarantee of "model safety" while strictly controlling Type-I error at $\zeta$ (e.g., 5%).

**Core Idea**: **Rewrite the test for "true failure rate $R_M$" as a proxy test for "noisy failure rate $R_J$."** Use a small dataset to estimate judge parameters and explicitly incorporate the estimation uncertainty into the critical threshold for variance correction. Thus, even if the judge is imperfect and calibration data is limited, the Type-I error remains controlled under finite samples.

## Method

### Overall Architecture

The framework (Noisy HT) ingests two datasets: a large judge-annotated set $D_J$ (large $n_J$) and a small human-annotated set $D_M$ (small $n_M$). First, the judge is run on $D_M$ to obtain an augmented set $\tilde{D}_M$ (containing both human labels $S_M$ and judge labels $S_J$) to estimate the judge's TPR/FPR. Then, the original test for the true failure rate is rewritten as a proxy test for the noisy failure rate. The test statistic $\hat{R}_J$ is computed on $D_J$ and compared against a "variance-corrected critical threshold $c'_J$" that accounts for calibration uncertainty.

```mermaid
flowchart LR
    A[Small set D_M<br/>Human labels S_M] --> B[Run judge for augmented set D̃_M<br/>Contains S_M and S_J]
    B --> C[Estimate TPR̂ / FPR̂<br/>Eq. (5)]
    C --> D[Compute proxy threshold α̂' and<br/>Var-corrected critical value c'_J Eq. (6)]
    E[Large set D_J<br/>Judge labels S_J] --> F[Test Statistic<br/>R̂_J = Mean S_J]
    D --> G{R̂_J < c'_J ?}
    F --> G
    G -->|Yes| H[Reject H_0<br/>Certify Model Safety]
    G -->|No| I[Accept H_0<br/>Do Not Certify]
```

### Key Designs

**1. Proxy Hypothesis Rewriting**: Mapping the "true failure rate" test onto the "noisy failure rate." The key insight is that $R_J = \mathbb{E}[S_J]$ is simply a linear mapping of $R_M$ through the judge's noise: $R_J = \text{TPR}\cdot R_M + \text{FPR}\cdot(1-R_M)$. Therefore, the original test $H_0: R_M \geq \alpha$ can be equivalently rewritten as $H'_0: R_J \geq \alpha'$, where the target threshold is shifted to $\alpha' = \text{FPR} + (\text{TPR}-\text{FPR})\cdot\alpha$, depending only on the judge's TPR/FPR. This step is the pivot of the method—it allows us to legally use the massive judge labels $\hat{R}_J$ for testing without worrying that judge labels differ from true labels, provided the judge is "useful" ($\text{TPR} > \text{FPR}$).

**2. Judge Modeling**: Estimating TPR/FPR and shifting the threshold using the small set. On the augmented set $\tilde{D}_M$, the judge's error profile is estimated using empirical frequencies: $\widehat{\text{TPR}} = \frac{\sum_i \mathbb{1}(S'_{Ji}=1, S_{Mi}=1)}{\sum_i \mathbb{1}(S_{Mi}=1)}$ and $\widehat{\text{FPR}} = \frac{\sum_i \mathbb{1}(S'_{Ji}=1, S_{Mi}=0)}{\sum_i \mathbb{1}(S_{Mi}=0)}$, leading to a plug-in estimate of the threshold $\hat{\alpha}' = \widehat{\text{FPR}} + (\widehat{\text{TPR}}-\widehat{\text{FPR}})\cdot\alpha$. This transforms the judge from a "black box" into a set of explicit error rates, distinguishing this work from PPI and providing diagnostic capabilities.

**3. Variance-Corrected Critical Threshold**: Explicitly incorporating calibration uncertainty into the threshold. The decision threshold is not simply compared to $\hat{\alpha}'$, but rather:

$$c'_J = \hat{\alpha}' + \Phi^{-1}(\zeta)\cdot\sqrt{\frac{\hat{\alpha}'(1-\hat{\alpha}')}{n_J} + \alpha^2\cdot\frac{\widehat{\text{TPR}}(1-\widehat{\text{TPR}})}{n_{M1}} + (1-\alpha)^2\cdot\frac{\widehat{\text{FPR}}(1-\widehat{\text{FPR}})}{n_{M0}}}$$

The three terms inside the square root represent: the variance of the test statistic itself ($\propto 1/n_J$), the variance of the TPR estimate ($\propto 1/n_{M1}$), and the variance of the FPR estimate ($\propto 1/n_{M0}$). **The essence lies in the latter two terms**—they shift the threshold based on the uncertainty of the judge's parameters. Fewer calibration data points ($n_M$) result in larger variance terms and a more conservative threshold. This design ensures Theorem 5.1: $P_e^{(I)} \leq \zeta + O(n_J^{-1/2} + n_{M1}^{-1/2} + n_{M0}^{-1/2})$, keeping Type-I error near $\zeta$ despite estimated judge parameters.

**4. Mechanism (Adoption Criterion) and Oracle Gap**: Theorem 5.4 provides the **necessary and sufficient condition** for Noisy HT to outperform Direct HT (which uses only the small human set) in terms of power (lower Type-II error):

$$(\text{TPR}-\text{FPR})^2 > \frac{\alpha^2\cdot\frac{\text{TPR}(1-\text{TPR})}{R_M} + (1-\alpha)^2\cdot\frac{\text{FPR}(1-\text{FPR})}{1-R_M}}{R_M(1-R_M)}$$

Intuitively, a stronger judge ($\text{TPR}\to 1, \text{FPR}\to 0$) is more likely to satisfy this. Stricter certification (higher $\alpha$ or lower $R_M$) requires a more accurate judge. This inequality defines the "useful/not useful" boundary on the (TPR, FPR) plane (Figure 1-D in the paper). Additionally, Theorem 5.3 identifies the **Oracle Gap**: any valid test estimating judge parameters has lower power than an "Oracle" with known parameters. This gap represents the "statistical cost of validity."

## Key Experimental Results

### Main Results Setting

| Setting | Dataset | Model under test (Classifier/Generator) | Judge |
|------|--------|--------------------------|------|
| Synthetic | Custom | — | Given TPR/FPR |
| Classification | Jigsaw Toxic Comment, Hate Speech Offensive | Qwen2.5-0.5B-Instruct, LLaMA-3.2-1B-Instruct | LLaMA-3.1-8B-Instruct |
| Generation | SafeRLHF | Alpaca-7B | LLaMA-3.1-8B-Instruct, LLaMA-3.3-70B-Instruct |

Baseline methods: Direct HT (human set only), Noisy HT (Ours), Oracle Noisy HT (theoretical upper bound), PPI variants. Typical parameters: $\alpha=0.25, \zeta=0.05, n_M=100, n_J=10000$.

### Key Findings

| Phenomenon | Observation |
|------|------|
| Type-I Control | All methods (Direct/Noisy/PPI) keep Type-I error under the 5% significance level, verifying validity. |
| TPR↑ | Type-II error significantly decreases (more sensitive judges are easier to certify). |
| FPR↓ | Type-II error significantly decreases. |
| Safer Model ($R_M$↓) | Type-II error decreases, making certification easier. |
| Noisy vs Direct | Noisy HT wins only in the high TPR/low FPR region, matching Theorem 5.4; advantage is clear with strong judges or weak classifiers. |
| Noisy vs Oracle | Oracle consistently outperforms Noisy/Direct, quantifying the Oracle Gap. |
| PPI vs Noisy | PPI usually has higher power (especially with poor judges), but **PPI still underperforms compared to Oracle Noisy HT**, suggesting room for PPI improvement via judge modeling. |

### Diagnostic Analysis (Estimator Scale)
As the calibration set $n_M$ increases from 25 to 100 (Jigsaw, 1000 trials), the standard deviation of $\widehat{\text{TPR}}/\widehat{\text{FPR}}$ converges, confirming that increasing $n_M$ is the primary way to reduce the Oracle Gap.

## Highlights & Insights

- **From "Blind Faith" to "Judge Diagnostics"**: A byproduct of the method is an interpretable diagnosis of the judge (TPR/FPR). Practitioners can use this for judge selection and sample size planning rather than just receiving a pass/fail.
- **The Variance-Corrected Threshold is the Key**: Explicitly folding calibration uncertainty into the critical value makes the test "automatically conservative" when labels are scarce, which is fundamental for finite-sample Type-I control.
- **Provable Adoption Criterion**: Theorem 5.4 defines exactly "what kind of judge is worth using" as a boundary on the (TPR, FPR) plane, providing strong engineering guidance.
- **Honest Acknowledgment of Cost**: The paper explicitly quantifies the Oracle Gap and the power trade-off against PPI, explaining the choice of "interpretable validity over raw power."

## Limitations & Future Work

- **Binary Evaluation**: Compressing outputs into pass/fail binary labels cannot handle fine-grained quality levels or multidimensional safety.
- **i.i.d. Assumption**: Requires independent and identically distributed samples; guarantees may fail under distribution shift or adversarial scenarios.
- **Lower Power than PPI**: Sacrifices some statistical power for interpretability, which might not be ideal for all scenarios. The paper suggests future PPI-based improvements.
- **Dependence on Human Label Quality**: TPR/FPR estimation assumes human labels are the gold standard; if human annotations are biased, the certification guarantee is compromised.
- **Inherent Oracle Gap**: As long as parameters are estimated, performance remains below the Oracle; only increasing $n_M$ or introducing prior constraints can reduce this.

## Related Work & Insights

- **LLM-as-a-Judge**: This work acknowledges flaws like bias and prompt sensitivity, providing a paradigm for "treating judges as noisy labels with explicit error estimation" rather than avoiding them.
- **Prediction-Powered Inference (PPI)**: The most direct comparison—both use few clean + many dirty labels, but PPI treats the judge as a black box for power, while this work white-boxes the judge for interpretable certification. They are complementary.
- **Conformal Prediction / Classical Testing**: The method is rooted in distribution-free finite-sample guarantees, bringing LLM certification into a rigorous statistical framework.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The first statistical framework to systematically handle "imperfect judges" for certification; the combination of proxy rewriting, variance correction, and adoption criteria is elegant and original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers synthetic, classification, and generation settings with various pairings; theoretical results match experiments. However, model and dataset scales are relatively small/moderate.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivations, theorem implications, and trade-offs with PPI are clearly explained, with "Implication" sections lowering the barrier for readers.
- **Value**: ⭐⭐⭐⭐ — Provides a both rigorous and interpretable tool for safety-critical LLM certification, offering practical guidance for judge selection and sample size planning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Doubly-Robust LLM-as-a-Judge: Externally Valid Estimation with Imperfect Personas](doubly-robust_llm-as-a-judge_externally_valid_estimation_with_imperfect_personas.md)
- [\[ICLR 2026\] When LLMs Get Significantly Worse: A Statistical Approach to Detect Model Degradations](when_llms_get_significantly_worse_a_statistical_approach_to_detect_model_degrada.md)
- [\[ICLR 2026\] BiasScope: Towards Automated Detection of Bias in LLM-as-a-Judge Evaluation](biasscope_towards_automated_detection_of_bias_in_llm-as-a-judge_evaluation.md)
- [\[ICLR 2026\] FinSearchComp: Towards a Realistic, Expert-Level Evaluation of Financial Search and Reasoning](finsearchcomp_towards_a_realistic_expert-level_evaluation_of_financial_search_an.md)
- [\[ICLR 2026\] Teach2Eval: An Interaction-Driven LLMs Evaluation Method via Teaching Effectiveness](teach2eval_an_interaction-driven_llms_evaluation_method_via_teaching_effectivene.md)

</div>

<!-- RELATED:END -->
