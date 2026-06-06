---
title: >-
  [Paper Note] Position: Stop Chasing the C-index when Evaluating Survival Analysis Models
description: >-
  [ICML 2026][LLM Safety][Survival Analysis] The authors audited 92 survival analysis papers from 2023–2025 and found that approximately 72% of the works used evaluation metrics (especially the misused C-index) that were m…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "Survival Analysis"
  - "C-index"
  - "Censoring Assumptions"
  - "Evaluation Metrics"
  - "Ladder Hypothesis"
date: 2026-05-08
content_hash: c2b4ceefb97a89e7
---

# Position: Stop Chasing the C-index when Evaluating Survival Analysis Models

**Conference**: ICML 2026  
**arXiv**: [2506.02075](https://arxiv.org/abs/2506.02075)  
**Code**: https://github.com/thecml/position-cindex  
**Area**: Medical Imaging / Survival Analysis Evaluation / Clinical Prediction  
**Keywords**: Survival Analysis, C-index, Censoring Assumptions, Evaluation Metrics, Ladder Hypothesis

## TL;DR
The authors audited 92 survival analysis papers from 2023–2025 and found that approximately 72% of the works used evaluation metrics (especially the misused C-index) that were misaligned with their modeling goals and censoring assumptions. They propose the "Double-Helix Ladder Hypothesis": models and metrics must stand on the same level of "censoring assumption," otherwise reported performance and rankings may be biased artifacts.

## Background & Motivation

**Background**: Survival analysis (time-to-event prediction) is widely used in medicine, engineering, and economics. Its defining characteristic is **censoring**—many individuals do not experience the event by the end of the study, providing only a lower bound for the event time. The community almost "by default" uses Harrell's C-index for evaluation: Zhou et al. (2023) found that over 80% of survival papers use the C-index as their primary metric.

**Limitations of Prior Work**: The authors manually reviewed 92 methodology and application papers from 2023–2025 and identified two types of systematic mismatches: (1) **Target-Metric Mismatch**: Works aiming for "time-to-event estimation" or "probability calibration" erroneously report only the C-index, which is a **ranking discrimination** metric; (2) **Missing Censoring Assumptions**: Papers using IPCW, Brier, or KM estimators rarely state or verify whether the "random censoring" assumption holds. Typical counter-examples include MOTOR (a 55M patient foundation model aiming for time prediction but only evaluating C-index and Brier), Zisser-Aran (evaluating ALL survival prediction solely with Harrell’s C-index), and HACSurv (specifically modeling dependent censoring but evaluating with IPCW-IBS, which assumes independent censoring).

**Key Challenge**: The validity of evaluation metrics is embedded with different assumptions regarding $E \perp\!\!\!\perp C$, $E \perp\!\!\!\perp C \mid \boldsymbol{X}$, or dependent censoring. Models themselves are also trained based on specific assumptions. When these are inconsistent, **metrics systematically shift in the wrong direction**, potentially reporting "improvements" even when oracle performance deteriorates.

**Goal**: To decompose the problem into three sub-questions: (a) formalizing "what constitutes a qualified survival metric"; (b) revealing the biased behavior of common metrics under different censoring mechanisms; and (c) providing an actionable decision workflow for metric selection.

**Key Insight**: View the "strength of censoring assumptions" as a ladder (random → conditionally independent → dependent). Align the **model ladder** and the **metric ladder**—if they fall on different rungs, the evaluation becomes meaningless.

**Core Idea**: Reframe the evaluation problem using **5 desiderata + the Ladder Hypothesis**. **Ours** proves through controlled synthetic experiments that when data deviates from the random censoring assumption, the bias of the standard C-index and IBS monotonically amplifies in a direction opposite to the oracle performance.

## Method

This is a position paper; "Method" refers to its analytical framework and controlled experimental design.

### Overall Architecture
The paper constructs a three-layer argumentation structure: (1) **Diagnostic Layer**—quantifying the "target-metric-assumption" mismatch through a meta-analysis of 92 references; (2) **Theoretical Layer**—proposing 5 desiderata and the Ladder Hypothesis, comparing existing metrics against these properties; (3) **Empirical Layer**—fixing a CoxPH model and Weibull event distribution while varying only the censoring mechanism (random / independent / Clayton-copula dependent with $\tau \in \{0.25, 0.5, 0.75\}$) to observe how the gap between oracle and censored metrics evolves with dependency strength. The input is synthetic survival data $\mathcal{D}=\{(\boldsymbol{x}_i, t_i, \delta_i)\}$, and the output consists of bias curves (Fig. 5) and ranking consistency across three censoring types.

### Key Designs

1.  **Five Desiderata for Evaluation Metrics (D1–D5)**:
    *   **Function**: Transitions the definition of a "good metric" from vague preferences to a checkable list, auditing all mainstream metrics (Harrell/Uno/Antolini C-index, IBS, MAE, D-Cal, LL).
    *   **Mechanism**: D1 **proper scoring rule** (optimum achieved when the predicted distribution equals the true distribution); D2 **interpretable** (units are days/months/probability rather than p-values); D3 **model-agnostic** (independent of internal model parameters); D4 **sensitive to miscalibration** (identifies systematic over/underestimation of survival probability); D5 **robust to censoring** (metric handles censoring mechanisms consistently with the actual data mechanism). Table 1 shows that all three C-indices fail D1 and D4; IBS is the only one satisfying D1+D3+D4, but D5 is only "partially satisfied"; MAE is optimal for D3 but fails D4.
    *   **Design Motivation**: Previous discussions on C-index limitations were single-point criticisms (Hartman et al. 2023) lacking a unified framework. Structuring these as desiderata allows researchers to select metrics based on scientific goals.

2.  **Ladder Hypothesis of Model-Metric Consistency**:
    *   **Function**: Projects "model assumptions" and "metric assumptions" onto the same censoring intensity axis, declaring mismatched combinations invalid.
    *   **Mechanism**: Constructs a double-helix ladder (Fig. 4). The left strand represents model evolution (CoxPH/RSF → IWSG/SurvivalBoost → DCSurvival/HACSurv), while the right strand represents metric evolution (KM-based Brier → Uno's CI with IPCW → missing dependent censoring metrics). Each rung corresponds to a censoring assumption: random / conditionally independent / dependent. The proposition is that models and metrics must stand on the same rung. A typical counter-example is HACSurv, which elevates the model to the third rung (modeling dependent censoring) but still uses second-rung Antolini-CI/IBS, leaving its SOTA claims effectively "unverified."
    *   **Design Motivation**: Prior papers either discussed models or metrics in isolation. The Ladder transforms "legitimate SOTA declarations" into a visual judgment of whether both strands have solid nodes on the same rung.

3.  **Controlled Censoring Ablation Experiments**:
    *   **Function**: Isolates the "bias of the metric itself" while keeping actual model performance controllable, demonstrating the consequences of the Ladder Hypothesis.
    *   **Mechanism**: Fixes a CoxPH model and a Weibull distribution, varying only the censoring mechanism: random, $\boldsymbol{X}$-dependent independent, and Clayton copula-injected dependent censoring (Kendall's $\tau \in \{0.25, 0.5, 0.75\}$). Two sets of metrics are calculated: the oracle version (using true $e_i$, reflecting "true model performance") and the censored version (using $t_i, \delta_i$ + KM-based IPCW). **Metric error** = censored − oracle. Results show that while oracle CI drops from 0.634 to 0.609 and oracle IBS rises from 0.090 to 0.245 (model performance is worsening), Harrell’s CI and naive IBS "appear" to improve at certain $\tau$ values.
    *   **Design Motivation**: Pure theoretical criticism rarely convinces researchers who feel the C-index is "generally good enough." Showing cases where the "model degrades while the metric improves" targets this habit directly.

### Loss & Training
As this paper does not propose a new model, there is no training objective. The "experimental loss" is the mean and variance of the metric error (over 100 random seeds).

## Key Experimental Results

### Main Results: Oracle Performance vs. Censored Metric Bias on Synthetic Data

| Censoring Mechanism | #Events (Cens.%) | oracle CI ↑ | oracle IBS ↓ |
| :--- | :--- | :--- | :--- |
| Random | 2641 (73.6%) | 0.634 ± 0.018 | 0.090 ± 0.040 |
| Independent | 3157 (68.4%) | 0.634 ± 0.018 | 0.084 ± 0.037 |
| Dependent ($\tau=0.25$) | 2969 (70.3%) | 0.628 ± 0.021 | 0.132 ± 0.096 |
| Dependent ($\tau=0.50$) | 2758 (72.4%) | 0.618 ± 0.025 | 0.199 ± 1.144 |
| Dependent ($\tau=0.75$) | 2536 (74.6%) | 0.609 ± 0.030 | 0.245 ± 0.157 |

**Key Findings**:
*   **The oracle performance of the model monotonically decreases as dependent censoring strengthens, but Harrell's CI and unweighted IBS error amplifies in inconsistent directions.** At $\tau=0.75$, the measured CI is higher than the oracle, and IBS bias exceeds ±0.1, rendering SOTA comparisons invalid.
*   **The C-index family fails to satisfy proper scoring rules and calibration sensitivity**: Pure ranking information cannot distinguish errors like "systematically predicting all event times as twice their true values," making it unsuitable for clinical decision-making.
*   **Dependent censoring + standard IPCW = Risk of ranking reversal**: Even with moderate dependency (Clayton $\tau = 0.5$), IBS bias reaches $\approx 0.115$ (comparable to the oracle scale), enough to misidentify the "better" model.
*   **IBS is currently the most balanced single metric**, but it still assumes independent censoring. The paper calls for the development of new metrics robust to dependent censoring as the next frontier.

## Highlights & Insights
*   **Shifting from "Criticizing C-index" to "Criticizing Evaluation Culture"**: Rather than just discussing C-index limitations, the paper identifies the triad mismatch of "Target—Metric—Assumption."
*   **The Ladder Hypothesis as a High-Reuse Framework**: This logic applies to any subfield where training assumptions are stronger than evaluation assumptions (e.g., adversarial robustness, causal inference, OOD detection).
*   **Experimental Design (Metric error = censored − oracle)**: Isolating evaluation bias using synthetic data is a paradigm that should be standardized for metric auditing.
*   **Actionable Recommendations 1–3**: (1) Let the target dictate the metric family; (2) Methodology papers must report one metric each from discrimination, error, and calibration; (3) Explicitly state censoring assumptions and perform sensitivity analysis when consistency isn't guaranteed.

## Limitations & Future Work
*   Controlled experiments only used one data generation mechanism (covariate-dependent Weibull + Clayton copula) and one leaner (CoxPH); robustness across other families has not been systematically verified.
*   The "Double-Helix Ladder" is currently a qualitative framework without analytical bounds for metric bias on each rung.
*   The paper does not propose a new metric, giving only principled suggestions rather than a ready-to-use tool for dependent censoring scenarios.
*   The manual review of 92 papers favored high-impact venues; the mismatch rate in "long-tail application papers" might be even higher.

## Related Work & Insights
*   **vs. Hartman et al. (2023)**: While they argue C-index limitations in clinical settings, **ours** upgrades this to a full metric comparison matrix and censoring gradient experiments, targeting even "safe" metrics like IBS.
*   **vs. Qi et al. (2023a, 2024a)**: Those works proposed improvements like MAE-PO for individual survival distribution (ISD). **Ours** places them within the desiderata framework, noting that even new metrics cannot yet bypass the independent censoring assumption.
*   **vs. Liu et al. (2025) HACSurv**: HACSurv advances modeling to dependent censoring, and **ours** uses it as a counter-example where the model advances but the metric does not, invalidating SOTA claims.

## Rating
*   Novelty: ⭐⭐⭐⭐ (Not a new method, but a new meta-level audit and unified framework.)
*   Experimental Thoroughness: ⭐⭐⭐ (Clean synthetic experiments, but limited model/distribution variety and no real-world data replication.)
*   Writing Quality: ⭐⭐⭐⭐⭐ (Structured, clear desiderata tables, and the Ladder diagram is an excellent communication tool.)
*   Value: ⭐⭐⭐⭐⭐ (Directly challenges default community practices; highly significant for the credibility of clinical prediction and biostatistics.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Uncertainty Quantification in LLMs is Just Unsupervised Clustering](position_uncertainty_quantification_in_llms_is_just_unsupervised_clustering.md)
- [\[ICML 2026\] LLM Benchmark Datasets Should Be Contamination-Resistant (Position Paper)](llm_benchmark_datasets_should_be_contamination-resistant.md)
- [\[ACL 2026\] When Models Outthink Their Safety: Unveiling and Mitigating Self-Jailbreak in Large Reasoning Models](../../ACL2026/llm_safety/when_models_outthink_their_safety_unveiling_and_mitigating_self-jailbreak_in_lar.md)
- [\[ICML 2026\] Position: Retire the "Positive Backdoor" Label -- Secret Alignment Requires Strict and Systematic Evaluation](position_retire_the_positive_backdoor_label_--_secret_alignment_requires_strict_.md)
- [\[ACL 2026\] SafetyALFRED: Evaluating Safety-Conscious Planning of Multimodal Large Language Models](../../ACL2026/llm_safety/safetyalfred_evaluating_safety-conscious_planning_of_multimodal_large_language_m.md)

</div>

<!-- RELATED:END -->
