---
title: >-
  [Paper Note] Position: Stop Chasing the C-index when Evaluating Survival Analysis Models
description: >-
  [ICML 2026][LLM Safety][C-index] The authors audited 92 survival analysis papers from 2023–2025 and found that approximately 72% of the works used evaluation metrics (especially the widely misused C-index) that are misaligned with their modeling goals and censoring assumptions. They proposed the "Double-Helix Ladder Hypothesis": models and metrics mus
tags:
  - ICML 2026
  - LLM Safety
  - C-index
  - Ladder Hypothesis
date: 2026-05-08
content_hash: 90ee8c5a0a501c8c
---
# Position: Stop Chasing the C-index when Evaluating Survival Analysis Models

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2506.02075](https://arxiv.org/abs/2506.02075)  
**Code**: https://github.com/thecml/position-cindex  
**Area**: Medical Imaging / Survival Analysis Evaluation / Clinical Prediction  
**Keywords**: Survival Analysis, C-index, Censoring Assumptions, Evaluation Metrics, Ladder Hypothesis

## TL;DR
The authors audited 92 survival analysis papers from 2023–2025 and found that approximately 72% of the works used evaluation metrics (especially the widely misused C-index) that are misaligned with their modeling goals and censoring assumptions. They proposed the "Double-Helix Ladder Hypothesis": models and metrics must stand on the same level of "censoring assumption," otherwise reported performance and rankings may be biased artifacts.

## Background & Motivation

**Background**: Survival analysis (time-to-event prediction) is widely used in medicine, engineering, and economics. Its defining characteristic is **censoring**—many individuals do not experience the event by the end of the study, providing only a lower bound for event times. The community almost "by default" evaluates survival models using Harrell’s C-index; a study by Zhou et al. (2023) found that over 80% of survival papers use the C-index as their primary metric.

**Limitations of Prior Work**: The authors manually reviewed 92 methodology and application papers from 2023–2025, identifying two types of systemic mismatch: (1) **Goal-Metric Mismatch**: Studies aiming for "time-to-event estimation" or "probability calibration" only report the C-index, which is a **discriminative/ranking** metric. (2) **Missing Censoring Assumptions**: Papers using IPCW, Brier, or KM estimators rarely state or verify whether the "random censoring" assumption holds. Typical counter-examples include MOTOR (a 55M patient foundation model aiming for time prediction but evaluated only with C-index and Brier), Zisser-Aran (ALL survival prediction evaluated only with Harrell C-index), and HACSurv (specifically modeling dependent censoring but evaluated using IPCW-IBS which assumes independent censoring).

**Key Challenge**: The validity of evaluation metrics is embedded with different assumptions regarding $E \perp\!\!\!\perp C$, $E \perp\!\!\!\perp C \mid \boldsymbol{X}$, or dependent censoring, while the model itself is also trained based on a specific assumption. When the two are inconsistent, **the metric will systemically shift in the wrong direction**, potentially reporting "gains" even when the oracle ground truth deteriorates.

**Goal**: This work addresses three sub-problems: (a) Formalizing what constitutes a "qualified survival metric"; (b) Revealing the biased behavior of common metrics under different censoring mechanisms; (c) Providing an actionable decision flow for metric selection.

**Key Insight**: The "strength of censoring assumptions" is viewed as a ladder (random → conditionally independent → dependent). By aligning the **model ladder** and the **metric ladder**—if both do not land on the same rung, the evaluation becomes meaningless.

**Core Idea**: The evaluation problem is reframed using **5 desiderata + the Ladder Hypothesis**. **Controlled synthetic experiments prove** that when data deviates from the random censoring assumption, the bias in standard C-index and IBS amplifies monotonically, often in a direction opposite to the oracle performance.

## Method

As a position paper, it does not propose a new model; the "Method" refers to the analytical framework and controlled experimental design used to support its claims.

### Overall Architecture

The paper argues that "the community should stop blindly chasing the C-index, which is misaligned with most modeling goals and censoring assumptions." This claim is substantiated through a three-layer argument: first, a **Diagnosis**—a meta-analysis of 92 papers from 2023–2025 quantifying "Goal-Metric-Assumption" mismatches; second, a **Theory**—proposing 5 desiderata and the Ladder Hypothesis to compare existing metrics; finally, **Empirical Evidence**—fixing a CoxPH model with the same Weibull synthetic survival data $\mathcal{D}=\{(\boldsymbol{x}_i, t_i, \delta_i)\}$ while varying only the censoring mechanism to observe how metric bias evolves relative to censoring dependency (Figure 5).

### Key Designs

**1. Five Desiderata for Evaluation Metrics (D1–D5): Transforming "qualitative preferences" into an auditable checklist**

Previous discussions on "C-index limitations" were isolated criticisms (Hartman et al., 2023), lacking a unified comparison framework. The authors decompose the required properties of evaluation metrics into five auditable standards: D1 **proper scoring rule** (optimal when the predicted distribution equals the true distribution), D2 **interpretable** (units are in days/months/probabilities rather than p-values), D3 **model-agnostic** (independent of model internal parameters), D4 **sensitive to miscalibration** (can identify systemic over- or under-estimation of survival probabilities), and D5 **robust to censoring** (metric handling of censoring aligns with the actual mechanism in the data). Auditing Harrell/Uno/Antolini C-index, IBS, MAE, D-Cal, and LL reveals that all three C-indices fail D1 and D4 (ranking info is neither a proper score nor captures calibration error), IBS is the only one satisfying D1+D3+D4 while D5 is only "partially satisfied," and MAE excels in D3 but fails D4. Researchers can thus select metrics based on scientific goals rather than default to the C-index.

**2. Ladder Hypothesis of Model-Metric Consistency: Projecting model and metric assumptions onto the same axis to identify invalid evaluation pairs**

Previous literature discussed models and metrics separately. The authors construct a double-helix ladder (Figure 4): the left strand represents model evolution (CoxPH/RSF → IWSG/SurvivalBoost → DCSurvival/HACSurv), and the right strand represents metric development (KM-based Brier → Uno's CI with IPCW → missing metrics for dependent censoring). Each rung corresponds to a censoring assumption—random, conditionally independent, and dependent. The Core Idea is that the model and metric must stand on the same rung for the evaluation to be valid. A poignant counter-example is HACSurv, which elevates the model to the third rung (modeling dependent censoring) but uses second-rung Antolini-CI + IBS (assuming independent censoring) for scoring. Its claimed SOTA has "never been verified by a metric on the same rung."

**3. Controlled Censoring Ablation Experiments: Isolating "metric bias" from "model performance"**

To challenge the "C-index is generally sufficient" intuition, the authors fixed a CoxPH model and Weibull event distribution while varying censoring mechanisms: random, $\boldsymbol{X}$-dependent independent censoring, and dependent censoring via Clayton copula (Kendall's $\tau \in \{0.25, 0.5, 0.75\}$). Two sets of metrics were calculated: an oracle version using true event times $e_i$ to reflect "true model performance," and a censored version using observed $t_i, \delta_i$ with KM-based IPCW. The **metric error = censored − oracle** isolates the pure evaluation bias. Results showed that as dependent censoring strengthened, oracle performance declined (CI decreased from 0.634 to 0.609, IBS increased from 0.090 to 0.245), but standard CI and unweighted IBS sometimes "looked better"—this inverse movement confirms the danger of Ladder Hypothesis misalignment.

## Key Experimental Results

### Main Results: Oracle Performance vs. Censoring Metric Bias on Synthetic Data

| Censoring Mechanism | #Events (Cens.%) | oracle CI ↑ | oracle IBS ↓ |
| :--- | :--- | :--- | :--- |
| Random | 2641 (73.6%) | 0.634 ± 0.018 | 0.090 ± 0.040 |
| Independent | 3157 (68.4%) | 0.634 ± 0.018 | 0.084 ± 0.037 |
| Dependent ($\tau=0.25$) | 2969 (70.3%) | 0.628 ± 0.021 | 0.132 ± 0.096 |
| Dependent ($\tau=0.50$) | 2758 (72.4%) | 0.618 ± 0.025 | 0.199 ± 1.144 |
| Dependent ($\tau=0.75$) | 2536 (74.6%) | 0.609 ± 0.030 | 0.245 ± 0.157 |

Key Observation: **The true model performance (oracle) degrades monotonically as dependent censoring increases, but Harrell's CI and unweighted IBS errors amplify and behave inconsistently**, causing SOTA comparisons to fail at high $\tau$.

### Meta-analysis: Evaluation Alignment across 92 Papers

| Metric Dimension | Methodological Papers (% failing) | Application Papers (% failing) |
| :--- | :--- | :--- |
| Goal-Metric Alignment + Censoring Statement | 73% | 68% |
| Reports only Discrimination Metrics | > 80% (Zhou 2023) | Majority |
| Explicitly State/Correct Censoring Assumption | Rare | Almost none |

### Metric-Desiderata Summary (Abridged from Table 1)

| Metric | D1 proper | D3 agnostic | D4 calib | D5 robust |
| :--- | :---: | :---: | :---: | :---: |
| Harrell CI | ✗ | ▲ | ✗ | ✗ |
| Uno CI | ✗ | ▲ | ✗ | ▲ |
| Antolini CI | ✗ | ▲ | ✗ | ✗ |
| IBS | ✓ | ✓ | ✓ | ▲ |
| MAE | ▲ | ✓ | ✗ | ✗ |
| D-Cal | ✗ | ✓ | ✓ | ✗ |

### Key Findings
- **The C-index family fails both proper scoring rule and calibration sensitivity**: Pure ranking information cannot distinguish a model that systemically predicts event times as double their true value, making it insufficient for clinical decision support.
- **Dependent Censoring + Standard IPCW = Risk of Rank Flipping**: Even with moderate dependency (Clayton $\tau = 0.5$), IBS bias reaches $\approx 0.115$, which is large enough to misidentify a worse model as superior.
- **IBS is currently the most balanced single metric** but still assumes independent censoring; the paper calls for the development of metrics robust to dependent censoring as the next frontier.

## Highlights & Insights
- **Shifting the focus from "C-index critique" to "auditing the evaluation culture"**: The work expands the scope to the "Goal-Metric-Assumption" triad and quantifies systemic issues across 92 papers.
- **The Ladder Hypothesis is a highly reusable framework**: This concept of "mismatch between model assumptions and evaluation assumptions" can be applied to other sub-fields like adversarial robustness, causal inference, or OOD detection.
- **Metric error = censored − oracle experimental design**: Isolating evaluation bias using synthetic data is a paradigm that should be standard for auditing metrics to avoid mixing model variance with metric bias.
- **Practical Recommendations (1–3)**: (1) Let the goal determine the metric family; (2) Methodological papers must report metrics across discrimination, error, and calibration; (3) Explicitly declare censoring assumptions and perform sensitivity analysis.

## Limitations & Future Work
- Controlled experiments utilized a single data generation mechanism (covariate-dependent Weibull + Clayton copula) and one learner (CoxPH); the robustness of the Ladder Hypothesis across varied marginal distributions and copula families remains to be systemically verified.
- The "Double-Helix Ladder" is currently a qualitative framework without analytical bounds or consistency proofs for metric bias at specific rungs.
- The paper does not propose a new metric, offering principle-based suggestions rather than ready-to-use tools for dependent censoring.
- The manual review of 92 papers may favor high-impact venues, potentially underestimating mismatch rates in "long-tail" application papers.

## Related Work & Insights
- **vs. Hartman et al. (2023)**: While they argued the clinical limitations of the C-index, this work provides a complete metric comparison matrix and censoring gradient experiments, extending the critique to metrics like IBS/D-Cal.
- **vs. Qi et al. (2023a, 2024a)**: These works proposed improvements like MAE-PO for individual survival distribution (ISD) evaluation; this paper places them within the desiderata framework and highlights that even new metrics struggle with independent censoring assumptions.
- **vs. Liu et al. (2025) HACSurv**: Using HACSurv as a counter-example, this work demonstrates that "model elevation without metric elevation" invalidates SOTA claims.
- **Insight**: The methodology of auditing "metric vs. task requirement" mismatch represents a general ML audit pattern applicable to recommender systems, dialogue evaluation, and RLHF reward modeling.

## Rating
- Novelty: ⭐⭐⭐⭐ (An audit and framework rather than a new method; original synthesis of existing elements).
- Experimental Thoroughness: ⭐⭐⭐ (Clean synthetic experiments, but limited model/distribution variety and no real-world data replication).
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear structure; functional tables and figures; a model for position papers).
- Value: ⭐⭐⭐⭐⭐ (Directly challenges default community practices; significant for clinical prediction and biostatistics trustworthiness).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Uncertainty Quantification in LLMs is Just Unsupervised Clustering](position_uncertainty_quantification_in_llms_is_just_unsupervised_clustering.md)
- [\[ICML 2026\] LLM Benchmark Datasets Should Be Contamination-Resistant (Position Paper)](llm_benchmark_datasets_should_be_contamination-resistant.md)
- [\[ACL 2025\] Answer When Needed, Forget When Not: Language Models Pretend to Forget via In-Context Knowledge Unlearning](../../ACL2025/llm_safety/answer_when_needed_forget_when_not_language_models_pretend_to_forget_via_in-cont.md)
- [\[ACL 2026\] When Models Outthink Their Safety: Unveiling and Mitigating Self-Jailbreak in Large Reasoning Models](../../ACL2026/llm_safety/when_models_outthink_their_safety_unveiling_and_mitigating_self-jailbreak_in_lar.md)
- [\[ACL 2026\] SafetyALFRED: Evaluating Safety-Conscious Planning of Multimodal Large Language Models](../../ACL2026/llm_safety/safetyalfred_evaluating_safety-conscious_planning_of_multimodal_large_language_m.md)

</div>

<!-- RELATED:END -->
