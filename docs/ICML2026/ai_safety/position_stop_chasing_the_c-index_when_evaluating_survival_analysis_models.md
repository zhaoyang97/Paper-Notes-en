---
title: >-
  [Paper Note] Position: Stop Chasing the C-index when Evaluating Survival Analysis Models
description: >-
  [ICML 2026 Spotlight][AI Safety][Survival Analysis] The authors audited 92 survival analysis papers from 2023–2025 and found that approximately 72% of the works used evaluation metrics (especially the overused C-index) that were misaligned with their modeling goals and censoring assumptions. They proposed the "Ladder Hypothesis": models and metrics must stand on the same level of "censoring assumption," otherwise reported performance and rankings may be biased artifacts.
tags:
  - "ICML 2026 Spotlight"
  - "AI Safety"
  - "Survival Analysis"
  - "C-index"
  - "Censoring Assumptions"
  - "Evaluation Metrics"
  - "Ladder Hypothesis"
date: 2026-05-08
content_hash: 0761034def800715
---

# Position: Stop Chasing the C-index when Evaluating Survival Analysis Models

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2506.02075](https://arxiv.org/abs/2506.02075)  
**Code**: https://github.com/thecml/position-cindex  
**Area**: Medical Imaging / Survival Analysis Evaluation / Clinical Prediction  
**Keywords**: Survival Analysis, C-index, Censoring Assumptions, Evaluation Metrics, Ladder Hypothesis

## TL;DR
The authors audited 92 survival analysis papers from 2023–2025 and found that approximately 72% of the works used evaluation metrics (especially the overused C-index) that were misaligned with their modeling goals and censoring assumptions. They proposed the "Ladder Hypothesis": models and metrics must stand on the same level of "censoring assumption," otherwise reported performance and rankings may be biased artifacts.

## Background & Motivation

**Background**: Survival analysis (time-to-event prediction) is widely used in medicine, engineering, and economics. The hallmark of such data is **censoring**—many individuals have not experienced the event by the end of the study, providing only a lower bound for event times. The community almost "defaulted" to using Harrell's C-index for evaluation: Zhou et al. (2023) found that over 80% of survival papers used C-index as the primary metric.

**Limitations of Prior Work**: The authors manually reviewed 92 method and application papers from 2023–2025 and identified two types of systematic mismatches: (1) **Goal-Metric Mismatch**: Using the C-index, a **ranking discrimination** metric, when the goal was "time-to-event estimation" or "probability calibration"; (2) **Missing Censoring Assumptions**: Papers using IPCW, Brier, or KM estimators rarely stated or verified if the "random censoring" assumption held. Typical counterexamples include MOTOR (a foundation model for 55M patients aiming for time prediction but only evaluating C-index and Brier), Zisser-Aran (evaluating ALL survival prediction only with Harrell C-index), and HACSurv (specifically modeling dependent censoring but evaluating with IPCW-IBS, which assumes independent censoring).

**Key Challenge**: The validity of evaluation metrics is embedded with different assumptions such as $E \perp\!\!\!\perp C$, $E \perp\!\!\!\perp C \mid \boldsymbol{X}$, or dependent censoring, and the models themselves are trained based on specific assumptions. When these are inconsistent, **the metrics systematically shift in the wrong direction**, potentially reporting "gains" even when the oracle ground truth deteriorates.

**Goal**: To decompose the issue into three sub-problems: (a) Formally defining "what constitutes a qualified survival metric"; (b) Revealing the biased behavior of common metrics under different censoring mechanisms; (c) Providing an actionable decision workflow for metric selection.

**Key Insight**: Treat the "strength of censoring assumptions" as a ladder (random -> conditionally independent -> dependent), aligning the **model ladder** alongside the **metric ladder**. If the two do not fall on the same rung, the evaluation loses its meaning.

**Core Idea**: Reframing the evaluation problem using **5 desiderata + the Ladder Hypothesis**, and **proving through controlled synthetic experiments** that when data deviates from the random censoring assumption, the bias of standard C-index and IBS amplifies monotonically, often in a direction opposite to oracle performance.

## Method

As a position paper, it does not propose a new model; "Method" refers to the analytical framework and controlled experimental design used to support its claims.

### Overall Architecture

The paper argues that "the C-index, adopted by default in community evaluations, is misaligned with most modeling goals and censoring assumptions and should not be blindly chased." This claim is substantiated through a three-layer argument: first, a **diagnosis** through a meta-analysis of 92 papers (2023–2025) to quantify "goal-metric-assumption" mismatches; then, a **theory** proposing 5 desiderata and the Ladder Hypothesis to compare existing metrics; and finally, **empirical evidence** using a fixed CoxPH model and Weibull synthetic data $\mathcal{D}=\{(\boldsymbol{x}_i, t_i, \delta_i)\}$, varying only the censoring mechanism to observe bias evolution curves (Figure 5).

### Key Designs

**1. Five evaluation metric desiderata (D1–D5): Transforming "what makes a good metric" into a checklist.**

Previous discussions on "C-index limitations" were isolated criticisms (Hartman et al., 2023) lacking a unified comparison framework. The authors split desired properties into five auditable criteria: D1 **proper scoring rule** (optimal only when the predicted distribution equals the true distribution), D2 **interpretable** (units in days/months/probability rather than p-values), D3 **model-agnostic** (not dependent on internal model parameters), D4 **sensitive to miscalibration** (capable of identifying systematic overestimation or underestimation), and D5 **robust to censoring** (metric handling of censoring aligns with the actual mechanism). Auditing Harrell/Uno/Antolini C-index, IBS, MAE, D-Cal, and LL shows: all three C-indices fail D1 and D4, IBS satisfies D1+D3+D4 but only "semi-satisfies" D5, and MAE is best at D3 but fails D4.

**2. Ladder Hypothesis of Model-Metric Consistency: Projecting model and metric assumptions onto the same axis.**

Previous papers discussed models or metrics in isolation. The authors construct a double-helix ladder (Figure 4): the left strand represents model development (CoxPH/RSF → IWSG/SurvivalBoost → DCSurvival/HACSurv), and the right strand represents metric development (KM-based Brier → Uno's CI with IPCW → currently missing dependent censoring metrics). Each rung corresponds to a censoring assumption—random, conditionally independent, or dependent. The core proposition is that the model and metric must stand on the same rung for the evaluation to be valid. HACSurv is a sharp counterexample: it elevates the model to the third rung (dependent censoring) but still uses second-rung Antolini-CI + IBS for scoring.

**3. Controlled Censoring Ablation: Isolating "metric bias" from "model performance."**

To demonstrate the issue empirically, the authors fixed a CoxPH model and a Weibull event distribution while varying the censoring mechanism: random, independent (dependent on $\boldsymbol{X}$), and dependent censoring injected via Clayton copula (Kendall's $\tau \in \{0.25, 0.5, 0.75\}$). Two sets of metrics were calculated: the oracle version using true event times $e_i$ to reflect "true model performance," and the censored version using observed $t_i, \delta_i$ with KM-based IPCW. The **metric error = censored − oracle** isolates pure evaluation bias. Results showed that as dependent censoring increased, oracle performance degraded, but Harrell CI and unweighted IBS sometimes "looked better," demonstrating the inverse movement caused by misaligned rungs.

## Key Experimental Results

### Main Results: Oracle Performance vs. Censoring Metric Bias on Synthetic Data

| Censoring Mechanism | #Events (Cens.%) | oracle CI ↑ | oracle IBS ↓ |
|---|---|---|---|
| Random | 2641 (73.6%) | 0.634 ± 0.018 | 0.090 ± 0.040 |
| Independent | 3157 (68.4%) | 0.634 ± 0.018 | 0.084 ± 0.037 |
| Dependent ($\tau=0.25$) | 2969 (70.3%) | 0.628 ± 0.021 | 0.132 ± 0.096 |
| Dependent ($\tau=0.50$) | 2758 (72.4%) | 0.618 ± 0.025 | 0.199 ± 1.44 |
| Dependent ($\tau=0.75$) | 2536 (74.6%) | 0.609 ± 0.030 | 0.245 ± 0.157 |

Key Observation: **True model performance (oracle) decreases monotonically as dependent censoring strengthens, but Harrell CI and unweighted IBS errors amplify and shift inconsistently**—at $\tau=0.75$, the standard CI measured value is higher than the oracle, rendering SOTA comparisons invalid.

### Meta-analysis: Evaluation Alignment across 92 Papers

| Metric Dimension | Method Papers (Non-compliance %) | Application Papers (Non-compliance %) |
|---|---|---|
| Goal-Metric Alignment + Censoring Assumption Stated | 73% | 68% |
| Proportion reporting only Discrimination metrics | > 80% (Zhou 2023) | Majority |
| Explicitly Stated / Corrected Censoring Assumptions | Very few | Almost none |

### Metric-Desiderata Summary (Table 1 Summary)

| Metric | D1 proper | D3 agnostic | D4 calib | D5 robust |
|---|---|---|---|---|
| Harrell CI | ✗ | ▲ | ✗ | ✗ |
| Uno CI | ✗ | ▲ | ✗ | ▲ |
| Antolini CI | ✗ | ▲ | ✗ | ✗ |
| IBS | ✓ | ✓ | ✓ | ▲ |
| MAE | ▲ | ✓ | ✗ | ✗ |
| D-Cal | ✗ | ✓ | ✓ | ✗ |

### Key Findings
- **The C-index family fails as a proper scoring rule and lacks calibration sensitivity**: Pure ranking information cannot distinguish errors like "predicting all event times as twice the ground truth," making it insufficient for clinical decision-making.
- **Dependent Censoring + Standard IPCW = Risk of Ranking Flips**: Even at moderate dependence ($\tau = 0.5$), the IBS bias reaches $\approx 0.115$ (comparable to oracle magnitude), enough to rank a worse model as better.
- **IBS is currently the most balanced single metric** but still assumes independent censoring; the paper calls for new metrics robust to dependent censoring as the next frontier.

## Highlights & Insights
- **From "Criticizing C-index" to "Criticizing the Evaluation Culture"**: The paper shifts focus from C-index limitations to a structural triplet mismatch of "Goal-Metric-Assumption."
- **The Ladder Hypothesis as a Reusable Conceptual Framework**: Applicable to any subdomain where training assumptions are stronger than evaluation assumptions (e.g., adversarial robustness, causal inference).
- **Metric error = censored − oracle Design**: Using synthetic data to isolate evaluation bias is a paradigm that should be used in all "metric audit" studies to avoid mixing model variance with metric bias.
- **Actionable Recommendations**: (1) Let goals determine the metric family; (2) Method papers must report metrics from discrimination, error, and calibration perspectives; (3) Explicitly declare censoring assumptions and perform sensitivity analysis.

## Limitations & Future Work
- The controlled experiment used only one data generation mechanism (Weibull + Clayton copula) and one learner (CoxPH); the robustness of the Ladder Hypothesis across various marginal distributions and copula families requires further validation.
- The "double-helix ladder" is currently a qualitative framework without analytical bounds or consistency proofs for various rungs.
- No new metric is proposed; recommendations for "what to use under dependent censoring" are principled rather than plug-and-play tools.
- The manual review of 92 papers might be biased toward high-impact venues, potentially underestimating mismatch rates in "long-tail" application papers.

## Related Work & Insights
- **vs. Hartman et al. (2023)**: While they argue C-index limitations in clinical utility, this work builds a complete metric comparison matrix and performs censoring gradient experiments.
- **vs. Qi et al. (2023a, 2024a)**: These works propose improvements like MAE-PO for individual survival distribution (ISD) evaluation; this paper places them in the desiderata framework and notes they still struggle with independent censoring assumptions.
- **vs. Liu et al. (2025) HACSurv**: Using HACSurv as a counterexample, this work argues that "elevating the model without elevating the metric" invalidates SOTA claims.
- **Insight**: Auditing "metric vs. task assumption mismatch" is a general pattern applicable to recommender systems, dialogue evaluation, and RLHF reward modeling.

## Rating
- Novelty: ⭐⭐⭐⭐ Not a new method, but a new meta-level audit and unified framework.
- Experimental Thoroughness: ⭐⭐⭐ Controlled synthetic experiments are clean, but model/distribution variety is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure with intuitive tables and the Ladder diagram; a model position paper.
- Value: ⭐⭐⭐⭐⭐ Directly challenges community defaults, significantly impacting the credibility of clinical prediction and biostatistics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] When Should an AI Scientist Stop? Verifiable Experiment Steering and Refusal for Autonomous Discovery](when_should_an_ai_scientist_stop_verifiable_experiment_steering_and_refusal_for_.md)
- [\[ICLR 2026\] MoReBench: Evaluating Procedural and Pluralistic Moral Reasoning in Language Models, More than Outcomes](../../ICLR2026/ai_safety/morebench_evaluating_procedural_and_pluralistic_moral_reasoning_in_language_mode.md)
- [\[CVPR 2026\] When LoRA Betrays: Backdooring Text-to-Image Models by Masquerading as Benign Adapters](../../CVPR2026/ai_safety/when_lora_betrays_backdooring_text-to-image_models_by_masquerading_as_benign_ada.md)
- [\[ICML 2026\] From Weak Cues to Real Identities: Evaluating Inference-Driven De-Anonymization in LLM Agents](from_weak_cues_to_real_identities_evaluating_inference-driven_de-anonymization_i.md)
- [\[ICML 2026\] Position: 'AI Alignment' Encompasses Competing Technical Priorities](ai_alignment_encompasses_competing_technical_priorities.md)

</div>

<!-- RELATED:END -->
