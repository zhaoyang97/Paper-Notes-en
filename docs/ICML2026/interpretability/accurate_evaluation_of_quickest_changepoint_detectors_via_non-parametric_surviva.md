---
title: >-
  [Paper Note] Accurate Evaluation of Quickest Changepoint Detectors via Non-parametric Survival Analysis
description: >-
  [ICML2026][Interpretability][Online changepoint detection] This paper reformulates the ARL/ADD evaluation in online quickest changepoint detection (QCD) as a right-censored survival analysis problem. By using Kaplan-Meie…
tags:
  - "ICML2026"
  - "Interpretability"
  - "Online changepoint detection"
  - "survival analysis"
  - "Kaplan-Meier estimation"
  - "ARL"
  - "ADD"
date: 2026-05-08
content_hash: 850bc3d15345ec77
---

# Accurate Evaluation of Quickest Changepoint Detectors via Non-parametric Survival Analysis

**Conference**: ICML2026  
**arXiv**: [2605.18798](https://arxiv.org/abs/2605.18798)  
**Code**: https://github.com/TaikiMiyagawa/Kaplan-Meier-Average-Run-Length  
**Area**: Time Series / Changepoint Detection / Evaluation Metrics  
**Keywords**: Online changepoint detection, survival analysis, Kaplan-Meier estimation, ARL, ADD  

## TL;DR
This paper reformulates the ARL/ADD evaluation in online quickest changepoint detection (QCD) as a right-censored survival analysis problem. By using Kaplan-Meier curves to estimate detection time and delay under finite and irregular sequence lengths, the proposed method is more robust and less biased than traditional estimators that only consider triggered samples.

## Background & Motivation
**Background**: Quickest changepoint detection (QCD) focuses on when a data stream switches from one distribution to another. In theoretical and simulation studies, Average Run Length (ARL) is commonly used to measure the average waiting time before a false alarm, while Average Detection Delay (ADD) measures how long it takes to alarm after a changepoint occurs. Historically, these two metrics form the core trade-off when selecting detection thresholds.

**Limitations of Prior Work**: Real-world datasets rarely provide infinitely or regularly lengthened sequences. Many sequences end before the detector triggers or are truncated before the changepoint. Traditional LB-ARL/LB-ADD metrics only retain samples that "triggered within the sequence length," which is equivalent to discarding right-censored information. This leads to significant negative bias and high variance in scenarios with short sequences, irregular lengths, high thresholds, or heavy censoring.

**Key Challenge**: The definitions of ARL/ADD assume the ability to observe the complete detection time, but real evaluation can only observe "detection time not exceeding a certain horizon" or "not triggered until the end of the sequence." Simply ignoring non-triggered sequences biases indicators toward shorter detection times, while forced extrapolation requires additional distribution assumptions.

**Goal**: The authors aim to estimate the ARL and ADD of any online QCD model on finite, irregular length data without assuming parametric forms (such as exponential distributions) for detection times, while providing bias analysis.

**Key Insight**: The paper observes that QCD evaluation is very similar to medical follow-up: a patient's time of death might be censored, just as a detector's alarm time might be censored by sequence length or changepoint location. Since the Kaplan-Meier estimator can estimate survival curves under right-censored data, "not yet alarmed" can be treated as "still alive."

**Core Idea**: Treat detection time/delay as the event time and sequence termination or changepoint location as the censoring time. Estimate ARL and ADD using the area under the non-parametric Kaplan-Meier survival curve.

## Method
This paper does not propose a new changepoint detector but a new evaluator. Its input includes a dataset with changepoint annotations, a set of sequence lengths, and detection points provided by a QCD model for each sequence. The output consists of ARL/ADD estimates suitable for finite irregular sequences. The core approach defines which samples are valid for ARL or ADD, incorporates undetected samples as right-censored data into the Kaplan-Meier curve, and finally uses the area of the restricted mean survival time as the average metric.

### Overall Architecture
For ARL, the authors focus on how long it takes for the detector to provide a false alarm when no changepoint exists. For each sequence, the detection time is $	au_i$, and the censoring time is defined as $C_i^{ARL}=\min\{\nu_i,T_i\}$, which is the longest observable time before a changepoint or the sequence end. If the detector triggers before this time, it is an observed event; otherwise, it is right-censored. The Kaplan-Meier curve estimates $S_{ARL}(t)=P(\tau>t\mid\nu=\infty)$, and KM-ARL is calculated as $\int_0^a \hat S_{ARL}(t)dt$.

For ADD, the focus is on samples where a changepoint exists and the detection occurs no earlier than the changepoint. The event time is replaced by the detection delay $\Delta\tau_i=\tau_i-\nu_i$, and the censoring time is the remaining length after the changepoint $C_i^{ADD}=T_i-\nu_i$. Similarly, $S_{ADD}(t)=P(\Delta\tau>t\mid\Delta\tau\ge 0,\nu<\infty)$ is estimated first, followed by integration to obtain KM-ADD. In experiments, the upper integration limit is the maximum observable time to avoid unfounded extrapolation where no observations exist.

### Key Designs
1. **Mapping QCD to Survival Analysis**:
    - **Function**: Converts the "no alarm" information in finite sequences into right-censored samples for statistical estimation.
    - **Mechanism**: In ARL, the event time is the detection point $\tau$ and the censoring time is $\min\{\nu,T\}$. In ADD, the event time is the detection delay $\Delta\tau$ and the censoring time is $T-\nu$. This ensures non-triggered sequences are not discarded but inform the estimator that "no alarm was triggered at least until the censoring time."
    - **Design Motivation**: The biggest issue with traditional LB indicators is that they only count triggered samples; higher thresholds lead to fewer samples with shorter detection times. The survival analysis mapping retains partial information from non-triggered samples, making the average curve closer to the true ARL/ADD.

2. **Non-parametric Integral Estimation of KM-ARL and KM-ADD**:
    - **Function**: Estimates average run length and average detection delay without assuming the shape of the detection time distribution.
    - **Mechanism**: Use the Kaplan-Meier product-limit form to estimate the survival function, e.g., $\hat S(t)=\prod_{j:t_j\le t}(1-d_j/n_j)$, where $d_j$ is the number of detection events at time $t_j$ and $n_j$ is the number of sequences still at risk. ARL/ADD are then calculated from the area under the survival curve.
    - **Design Motivation**: Existing parametric survival methods require assumptions like exponential distributions, which may not hold for real sensor data or complex detectors. The non-parametric KME better suits machine learning evaluation needs for "arbitrary detectors and arbitrary underlying distributions."

3. **Finite-sample Bias and Truncation Bias Decomposition**:
    - **Function**: Explains when the KM estimator is reliable and when unbiased estimation is impossible.
    - **Mechanism**: The paper decomposes total bias into finite-sample bias and truncation bias. The former decays exponentially as the sample size increases, while the latter stems from insufficient observation horizons. It proves that under an appropriate upper bound, the truncation negative bias of KM-ARL/KM-ADD is no greater than that of traditional LB indicators.
    - **Design Motivation**: Evaluation metrics themselves require explainability. Bias decomposition informs users: if the true detection time falls outside all observation horizons, no assumption-free method can reliably extrapolate; if censoring is merely finite and irregular, KM indicators can significantly mitigate traditional estimator bias.

### Loss & Training
This work does not train new detection models but calculates metrics during the evaluation phase. Theoretical analysis assumes the online QCD detector does not look into the future; thus, detection points and censoring mechanisms approximately satisfy independent censoring. Experiments evaluate Window L1, Window Normal, Window AR, NP-FOCuS, CUSUM, EWMA, and simulation-based GSR/CUSUM, implemented using tools like Python, lifelines, ruptures, and changepoint-online.

## Key Experimental Results

### Main Results
The main experiments focus on whether KM indicators are closer to true ARL/ADD curves under finite and irregular lengths. Figures 2, 3, and 4 in the paper provide curve results, summarized below by scenario.

| Scenario | Data Setting | Comparison Metrics | Main Results | Explanation |
|------|----------|----------|----------|------|
| Gaussian ARL (Light Censoring) | Length 1000, 10% sequences with changepoints | True ARL / Naive / LB / KM | KM-ARL closely matches True ARL | When horizons are long enough, KM introduces no significant extra bias |
| Gaussian ARL (Heavy Censoring) | Length 1000, 90% sequences with changepoints | True ARL / Naive / LB / KM | LB and Naive bias increases; KM is more stable | KM utilizes sequences censored by changepoints instead of only false alarms |
| Gaussian ARL (Irregular Lengths) | Random lengths in [100,1000] or [30,300] | True ARL / Naive / LB / KM | KM is closest to True curve in non-extrapolated regions | Irregular lengths cause LB sample sets to fluctuate wildly with thresholds; KM risk sets are more stable |
| Gaussian ADD | Geometric changepoint dist, length 100 or [10,100] | True ADD / LB / KM | KM-ADD is closer to True ADD for late changepoints/short sequences | Undetected delay samples are treated as right-censored rather than discarded |
| WISDM Actitracker | 51,326 machine-labeled sequences, length 1 to 54,401 | LB curve vs KM curve | KM-ARL/KM-ADD curves show lower variance and more intuitive model selection | Real data lengths are highly irregular; LB has very few samples at high thresholds |

Real-world experiments utilized a machine-labeled subset of WISDM Actitracker with the following statistics:

| Subset | # of Sequences | # of Frames | Mixed Label Sequences | Avg. Length | Min/Max Length | Positive Frame Ratio |
|------|--------|------|------------------|----------|--------------|------------|
| User-labeled | 83 | 5,435 | 29 | 65.5 | 1 / 565 | 0.741 |
| Machine-labeled | 51,326 | 1,369,349 | 51,189 | 26.7 | 1 / 54,401 | 0.684 |

### Ablation Study
The paper lacks modular model ablation but provides systematic analysis of evaluation conditions, which can be viewed as metric robustness analysis.

| Analysis Dimension | Settings | Observation | Insights |
|----------|------|------|------|
| Max Sequence Length | 1000, 500, 300, 100 | Shorter horizons make non-extrapolated estimation harder; bias is inevitable if ARL exceeds max length | KM only reduces censoring bias within observable regions; it cannot predict unobserved tails |
| Length Irregularity | Fixed length vs random interval length | Irregular lengths significantly amplify LB-ARL/LB-ADD fluctuations | Real datasets benefit most from KM risk set correction |
| Changepoint Ratio | 10% vs 90% with changepoint | Higher ratios make ARL false alarm times more likely to be censored | KM-ARL utilizes the fact that "no false alarm occurred before the changepoint" |
| Changepoint Distribution | Uniform vs Geometric | Late changepoints reduce observable detection delay | KM-ADD is more robust toward right-censoring in late changepoints |
| Detector Type | GSR, CUSUM, Windowed, NP-FOCuS, EWMA | Consistent trends across different detectors | The method is not a trick tied to a specific QCD algorithm |

### Key Findings
- The greatest contribution comes from "not discarding non-triggered sequences." LB indicators suffer from a sharp sample reduction at high thresholds, while KM indicators maintain risk set information based on the fixed dataset size, controlling both variance and bias.
- KM indicators are not universal extrapolators. When the true ARL/ADD exceeds the maximum observation horizon, the paper explicitly labels the region as extrapolation, warning readers that increased data length or parametric tail models are required.
- The user-labeled subset of WISDM contains only 83 sequences; the authors suggest using box plots or min-max statistics rather than relying solely on average metrics for such tiny data.
- The work promotes "evaluation reliability" as a methodological contribution. For threshold-sensitive tasks like QCD, metric bias can directly change conclusions for model selection.

## Highlights & Insights
- The most ingenious aspect is reinterpreting "no alarm" in QCD as informative censored observations in survival analysis. This transition is natural yet directly fixes the sample selection bias that plagues traditional empirical estimation.
- Theoretical sections separate finite-sample bias from truncation bias, allowing readers to distinguish whether error stems from small sample sizes or insufficient horizons, aiding in the design of better evaluation sets.
- The method is completely black-box to the detector. As long as detection points, changepoint annotations, and sequence lengths are available, KM-ARL/KM-ADD can be applied, making it transferable to sensors, industrial monitoring, health monitoring, and event detection.
- This paper also serves as a reminder that "model differences" in machine learning might actually be estimator biases. Especially in online/streaming tasks, handling incomplete, non-triggered, or truncated samples is a core problem.

## Limitations & Future Work
- KM-ARL/KM-ADD relies on the independent censoring assumption. While online QCD generally satisfies this, offline detectors look at the whole sequence, meaning detection points could be correlated with sequence length/changepoint location, necessitating complex dependent censoring methods.
- The method requires multiple sequences with changepoint annotations and cannot be used directly for completely unlabeled online streams.
- Under extreme censoring or very few samples, KM still exhibits large uncertainty. The paper suggests pairing it with box plots, bootstrap, or finite-sample corrections for small datasets.
- Currently, it only handles single changepoint types and settings. Multi-changepoint or multi-type settings could be addressed using competing risks or multi-state survival models.
- Other QCD indicators like PFA are also affected by right-censoring; the paper explores the Aalen-Johansen estimator but has not yet resolved discrete-time event tie issues.

## Related Work & Insights
- **vs LB-ARL / LB-ADD**: LB only averages samples triggered within the horizon, which is simple but biased toward short detection times; the KM method incorporates non-triggered samples into the risk set as right-censored data, reducing bias.
- **vs parametric survival ARL**: Methods like Sahki et al. require exponential decay assumptions; this study uses non-parametric KME, needing no specific distribution for detection times.
- **vs conventional time-series event metrics**: Metrics like Precision/Recall, F-score, and NAB emphasize localization quality or event matching. This paper focuses on ARL/ADD, the most used theoretical metrics in QCD, correcting their estimation on real finite sequences.
- **vs Model Papers like DeepLLR-CUSUM**: Such works might use KME for empirical evaluation, but this paper makes ARL/ADD estimation itself the object of study, providing definitions, bias bounds, and conditions for use.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Systematically introduces survival analysis into QCD metric estimation with clear problem positioning and effective methodology.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Includes simulations, real WISDM data, multiple detectors, and various censoring conditions; would be stronger with more diverse real-world domains.
- Writing Quality: ⭐⭐⭐⭐☆ Clear main narrative and solid bias analysis; contains many formulas, presenting a learning curve for readers without a statistical background.
- Value: ⭐⭐⭐⭐⭐ Extremely practical for evaluating online changepoint detection, anomaly detection, and streaming monitoring, particularly for finite-length real-world datasets.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Probing Preference Representations: A Multi-Dimensional Evaluation and Analysis Method for Reward Models](../../AAAI2026/interpretability/probing_preference_representations_a_multi-dimensional_evaluation_and_analysis_m.md)
- [\[ICML 2026\] A Behavioural and Representational Evaluation of Goal-Directedness in Language Model Agents](a_behavioural_and_representational_evaluation_of_goal-directedness_in_language_m.md)
- [\[ICML 2026\] Understanding LoRA as Knowledge Memory: An Empirical Analysis](understanding_lora_as_knowledge_memory_an_empirical_analysis.md)
- [\[AAAI 2026\] iMAD: Intelligent Multi-Agent Debate for Efficient and Accurate LLM Inference](../../AAAI2026/interpretability/imad_intelligent_multi-agent_debate_for_efficient_and_accura.md)
- [\[ACL 2026\] Interpretable Coreference Resolution Evaluation Using Explicit Semantics](../../ACL2026/interpretability/interpretable_coreference_resolution_evaluation_using_explicit_semantics.md)

</div>

<!-- RELATED:END -->
