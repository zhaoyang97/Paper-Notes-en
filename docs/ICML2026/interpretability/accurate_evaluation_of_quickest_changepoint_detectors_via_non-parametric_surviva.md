---
title: >-
  [Paper Note] Accurate Evaluation of Quickest Changepoint Detectors via Non-parametric Survival Analysis
description: >-
  [ICML2026][Interpretability][Online Changepoint Detection] This work reformulates the ARL/ADD evaluation in online quickest changepoint detection (QCD) as a right-censored survival analysis problem. By using Kaplan-Meier curves to estimate detection time and delay under finite and irregular sequence lengths, the proposed method provides more robust and less biased estimators compared to traditional methods that only count triggered samples.
tags:
  - "ICML2026"
  - "Interpretability"
  - "Online Changepoint Detection"
  - "Survival Analysis"
  - "Kaplan-Meier Estimation"
  - "ARL"
  - "ADD"
date: 2026-05-08
content_hash: 9d3ed945d960da96
---

# Accurate Evaluation of Quickest Changepoint Detectors via Non-parametric Survival Analysis

**Conference**: ICML2026  
**arXiv**: [2605.18798](https://arxiv.org/abs/2605.18798)  
**Code**: https://github.com/TaikiMiyagawa/Kaplan-Meier-Average-Run-Length  
**Area**: Time Series / Changepoint Detection / Evaluation Metrics  
**Keywords**: Online Changepoint Detection, Survival Analysis, Kaplan-Meier Estimation, ARL, ADD  

## TL;DR
This work reformulates the ARL/ADD evaluation in online quickest changepoint detection (QCD) as a right-censored survival analysis problem. By using Kaplan-Meier curves to estimate detection time and delay under finite and irregular sequence lengths, the proposed method provides more robust and less biased estimators compared to traditional methods that only count triggered samples.

## Background & Motivation
**Background**: Quickest changepoint detection (QCD) is concerned with identifying when a data stream switches from one distribution to another. In theoretical and simulation studies, Average Run Length (ARL) is commonly used to measure the average waiting time before a false alarm, while Average Detection Delay (ADD) measures how long it takes to alarm after a changepoint occurs. These two form the core trade-off when selecting detection thresholds.

**Limitations of Prior Work**: Real-world datasets rarely provide infinite or regular sequence lengths. Many sequences end before a detector triggers or are truncated before a changepoint occurs. Traditional LB-ARL/LB-ADD metrics only retain samples that "triggered within the sequence length," which is equivalent to discarding right-censored information. This leads to significant negative bias and high variance in scenarios involving short sequences, irregular lengths, high thresholds, or heavy censoring.

**Key Challenge**: The definitions of ARL/ADD assume that the full detection time can be observed. However, real evaluation only observes whether "detection time does not exceed a certain horizon" or "no trigger occurred before the sequence ended." Simply ignoring non-triggered sequences biases metrics toward shorter detection times, while forced extrapolation requires additional distributional assumptions.

**Goal**: The authors aim to estimate the ARL and ADD of any online QCD model on finite, irregular length data without assuming parametric forms (such as exponential distributions) for detection times, while providing a bias analysis.

**Key Insight**: The paper observes that QCD evaluation is highly analogous to medical follow-up studies: a patient's time of death might be censored, just as a detector's alarm time might be censored by sequence length or changepoint location. Since the Kaplan-Meier estimator can estimate survival curves under right-censored data, "remaining un-alarmed" can be treated as "still alive."

**Core Idea**: Treat detection time or delay as the event time and sequence termination or changepoint location as the censoring time. The ARL and ADD are then estimated using the area under the non-parametric Kaplan-Meier survival curve.

## Method
Ours does not propose a new changepoint detector but rather a new evaluator. Given a labeled dataset, a set of sequence lengths, and the detection points provided by a QCD model, it outputs ARL/ADD estimates better suited for finite, irregular data. The key shift is reinterpreting the "sequence ended before triggering" event—previously discarded—as informative right-censored observation in survival analysis.

### Overall Architecture
The evaluation follows two parallel paths for ARL and ADD using a consistent strategy: first, define which samples are valid for the metric; second, define an event time and a censoring time for each sequence; finally, use the area under the Kaplan-Meier curve as the mean metric. For ARL, which focuses on false alarms in the absence of a changepoint, the detection time is $\tau_i$, and the maximum observable non-changepoint duration is $C_i^{ARL}=\min\{\nu_i, T_i\}$. A trigger is an observed event, otherwise it is right-censored. $S_{ARL}(t)=P(\tau>t\mid\nu=\infty)$ is estimated, and KM-ARL is calculated as $\int_0^a \hat S_{ARL}(t)dt$. For ADD, only samples where a changepoint exists and detection occurs no earlier than the changepoint are considered. The event time is the detection delay $\Delta\tau_i = \tau_i - \nu_i$, and the censoring time is the remaining length after the changepoint $C_i^{ADD} = T_i - \nu_i$. KM-ADD is similarly obtained by integrating the estimated $S_{ADD}(t)=P(\Delta\tau>t\mid\Delta\tau\ge 0, \nu<\infty)$. The integration upper limit is unified to the maximum observable time to avoid unfounded extrapolation.

### Key Designs

**1. Mapping QCD to Survival Analysis: Turning "No Alarm" into Informative Censoring**

The most detrimental aspect of traditional LB metrics is that they only average samples that have "already triggered." Higher thresholds leave only a small subset of sequences with shorter detection times, causing the mean to be systematically underestimated. This work fixes this by defining both event and censoring times for each sequence. This way, a sequence that never triggers is no longer discarded but contributes the constraint "no alarm occurred at least until the censoring time," making the average curve much closer to the true ARL/ADD.

**2. KM-ARL and KM-ADD: Non-parametric Metric Estimation from Survival Curve Area**

To estimate mean run length and delay without assuming parametric forms like exponential distributions, the authors directly use the Kaplan-Meier product-limit estimator for the survival function $\hat S(t)=\prod_{j:t_j\le t}(1-d_j/n_j)$, where $d_j$ is the number of detection events at time $t_j$ and $n_j$ is the number of sequences still at risk. ARL/ADD is then the area under this staircase survival curve. Unlike parametric survival methods (e.g., Sahki et al.), the non-parametric KME does not bind to any underlying distribution, fitting the "arbitrary detector, arbitrary data distribution" machine learning evaluation scenario.

**3. Finite-Sample and Truncation Bias Decomposition: Identifying Estimator Reliability**

To ensure interpretability, the total bias of the KM estimator is decomposed into two parts: finite-sample bias, which decays exponentially with sample size, and truncation bias, which arises from an insufficient observation horizon. The authors prove that under a suitable integration limit, the truncation negative bias of KM-ARL/KM-ADD does not exceed that of traditional LB metrics. This decomposition provides clear boundaries: if the true detection time falls outside all observation horizons, no assumption-free method can reliably extrapolate; however, for finite and irregular censoring, KM metrics significantly mitigate LB estimation bias.

This work calculates metrics during the evaluation phase without training new models. Theoretical analysis assumes that the online QCD detector does not look into the future, so the detection points and censoring mechanisms satisfy independent censoring approximately. Experiments cover Window L1, Window Normal, Window AR, NP-FOCuS, CUSUM, EWMA, and simulation-based GSR/CUSUM. Implementation uses Python with `lifelines`, `ruptures`, and `changepoint-online`.

## Key Experimental Results

### Main Results
Main results focus on whether KM metrics are closer to the true ARL/ADD curves under finite and irregular lengths.

| Scenario | Data Setting | Comparison Metrics | Main Result | Explanation |
| :--- | :--- | :--- | :--- | :--- |
| Gaussian ARL (Light Censoring) | Length 1000, 10% sequences w/ changepoint | True ARL / Naive / LB / KM | KM-ARL closely matches True ARL | KM introduces no significant bias when horizon is long enough. |
| Gaussian ARL (Heavy Censoring) | Length 1000, 90% sequences w/ changepoint | True ARL / Naive / LB / KM | LB and Naive bias increases; KM remains stable | KM utilizes sequences censored by changepoints instead of only false alarms. |
| Gaussian ARL (Irregular Length) | Length random in [100,1000] or [30,300] | True ARL / Naive / LB / KM | KM is closest to ground truth in non-extrapolation regions | Irregular lengths cause LB sample sets to fluctuate; KM risk set is more stable. |
| Gaussian ADD | Geometric CP dist, length 100 or [10,100] | True ADD / LB / KM | KM-ADD closer to True ADD for late CPs or short sequences | Undetected delays are treated as right-censored rather than discarded. |
| WISDM Actitracker | 51,326 machine-labeled sequences | LB vs KM curves | KM-ARL/ADD has lower variance and more intuitive model selection | Real data lengths are highly irregular; LB has few samples at high thresholds. |

Statistics for the WISDM Actitracker real dataset:

| Subset | # Sequences | # Frames | Mixed label seq | Avg Length | Min/Max Length | Positive Frame % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| User Labeled | 83 | 5,435 | 29 | 65.5 | 1 / 565 | 0.741 |
| Machine Labeled | 51,326 | 1,369,349 | 51,189 | 26.7 | 1 / 54,401 | 0.684 |

### Ablation Study
Rather than modular ablation, the paper provides a systematic analysis of evaluation conditions.

| Analysis Dimension | Setting | Observation | Insight |
| :--- | :--- | :--- | :--- |
| Sequence Horizon | 1000, 500, 300, 100, etc. | Shorter horizons make all non-extrapolated estimation harder | KM reduces bias in observable regions but cannot predict unobserved tails. |
| Length Irregularity | Fixed vs. Interval Random | Irregularity significantly amplifies LB-ARL/ADD fluctuations | KM risk set correction is most needed for real-world datasets. |
| Changepoint Ratio | 10% vs. 90% with CP | Higher CP ratios increase censoring of ARL false alarm times | KM-ARL leverages "no false alarm before changepoint" info. |
| CP Distribution | Uniform vs. Geometric | Late changepoints reduce observable detection delay | KM-ADD is robust against right-censoring in late changepoint scenarios. |

### Key Findings
- The biggest contribution is "not discarding non-triggered sequences." While the number of samples used by LB metrics drops sharply at high thresholds, KM metrics preserve risk set information, leading to more controllable variance and bias.
- KM metrics are not universal extrapolators. When the true ARL/ADD exceeds the maximum observation horizon, the paper explicitly marks these areas as extrapolation, warning that data length must be increased or parametric tail models introduced.
- In the user-labeled WISDM subset (only 83 sequences), authors suggest using min-max statistics like box plots alongside mean metrics.
- This work elevates "evaluation reliability" to a core methodological contribution. For threshold-sensitive tasks like QCD, metric bias can directly lead to incorrect model selection.

## Highlights & Insights
- The most ingenious aspect is reinterpreting "no alarm" as an informative censored observation in survival analysis. This transition is natural but directly fixes the sample selection bias inherent in traditional empirical estimation.
- Theoretical analysis separates finite-sample bias from truncation bias, allowing practitioners to distinguish whether errors stem from sample size or horizon limitations.
- The method is entirely black-box regarding the detector. As long as detection points, changepoint labels, and sequence lengths are available, KM-ARL/KM-ADD can be applied across sensors, industrial monitoring, and health tracking.

## Limitations & Future Work
- KM-ARL/KM-ADD relies on the independent censoring assumption. While online QCD generally satisfies this, offline detectors look at the whole sequence, potentially creating dependencies between detection points and sequence length.
- The method requires labeled sequences with changepoint annotations and is not directly applicable to entirely unlabeled online streams.
- Under extreme censoring or very small sample sizes, KM still exhibits uncertainty.
- Currently limited to single changepoint types and settings. Multi-changepoint or multi-type scenarios might benefit from competing risks or multi-state survival models.

## Related Work & Insights
- **vs LB-ARL / LB-ADD**: LB only averages samples triggered within the horizon, biasing toward short times; KM includes non-triggered samples as right-censored, reducing bias.
- **vs Parametric Survival ARL**: Unlike Sahki et al., which requires exponential decay assumptions, this non-parametric approach does not require the detection time to follow a specific distribution.
- **vs Standard Time-Series Event Metrics**: Unlike precision/recall or NAB which focus on localization quality, this work focuses on the most common QCD theoretical metrics (ARL/ADD) and fixes their estimation on finite sequences.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Systematically introduces survival analysis into QCD metric estimation with clear problem positioning.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Includes simulation, WISDM, and multiple detectors/censoring conditions.
- Writing Quality: ⭐⭐⭐⭐☆ Clear logic and solid bias analysis; mathematically technical for those without a statistics background.
- Value: ⭐⭐⭐⭐⭐ Highly practical for online changepoint detection and streaming monitoring, especially for finite-length real-world datasets.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Interpretable Image Classification via Non-parametric Part Prototype Learning](../../CVPR2025/interpretability/interpretable_image_classification_via_non-parametric_part_prototype_learning.md)
- [\[AAAI 2026\] Probing Preference Representations: A Multi-Dimensional Evaluation and Analysis Method for Reward Models](../../AAAI2026/interpretability/probing_preference_representations_a_multi-dimensional_evaluation_and_analysis_m.md)
- [\[ACL 2025\] Establishing Trustworthy LLM Evaluation via Shortcut Neuron Analysis](../../ACL2025/interpretability/shortcut_neuron_eval.md)
- [\[ICML 2026\] A Behavioural and Representational Evaluation of Goal-Directedness in Language Model Agents](a_behavioural_and_representational_evaluation_of_goal-directedness_in_language_m.md)
- [\[ICLR 2026\] xRFM: Accurate, scalable, and interpretable feature learning models for tabular data](../../ICLR2026/interpretability/xrfm_accurate_scalable_and_interpretable_feature_learning_models_for_tabular_dat.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[CVPR 2025\] Interpretable Image Classification via Non-parametric Part Prototype Learning](../../CVPR2025/interpretability/interpretable_image_classification_via_non-parametric_part_prototype_learning.md)
- [\[AAAI 2026\] Probing Preference Representations: A Multi-Dimensional Evaluation and Analysis Method for Reward Models](../../AAAI2026/interpretability/probing_preference_representations_a_multi-dimensional_evaluation_and_analysis_m.md)
- [\[ACL 2025\] Establishing Trustworthy LLM Evaluation via Shortcut Neuron Analysis](../../ACL2025/interpretability/shortcut_neuron_eval.md)
- [\[ICML 2026\] A Behavioural and Representational Evaluation of Goal-Directedness in Language Model Agents](a_behavioural_and_representational_evaluation_of_goal-directedness_in_language_m.md)
- [\[ICML 2026\] Disentangling Direction and Magnitude in Transformer Representations: A Double Dissociation Through L2-Matched Perturbation Analysis](disentangling_direction_and_magnitude_in_transformer_representations_a_double_di.md)

</div>

<!-- RELATED:END -->
