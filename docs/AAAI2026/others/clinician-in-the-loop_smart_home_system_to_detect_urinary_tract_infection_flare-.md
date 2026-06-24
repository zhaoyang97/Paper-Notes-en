---
title: >-
  [Paper Note] Clinician-in-the-Loop Smart Home System to Detect Urinary Tract Infection Flare-Ups via Uncertainty-Aware Decision Support
description: >-
  [AAAI 2026][Smart home] This paper proposes a clinician-in-the-loop smart home system that extracts behavioral markers from ambient sensor data and introduces a novel Conformal Calibrated Interval (CCI) method to quantify predictive uncertainty, enabling reliable detection of urinary tract infection (UTI) flare-ups in older adults and supporting an "abstain when uncertain" decision paradigm.
tags:
  - "AAAI 2026"
  - "Smart home"
  - "urinary tract infection detection"
  - "uncertainty quantification"
  - "conformal prediction"
  - "clinical decision support"
date: 2026-05-08
content_hash: 79f22110cc84e3cb
---

# Clinician-in-the-Loop Smart Home System to Detect Urinary Tract Infection Flare-Ups via Uncertainty-Aware Decision Support

**Conference**: AAAI 2026
**arXiv**: [2511.18334](https://arxiv.org/abs/2511.18334)  
**Code**: None  
**Area**: Other (Smart Healthcare / Uncertainty Quantification)
**Keywords**: Smart home, urinary tract infection detection, uncertainty quantification, conformal prediction, clinical decision support

## TL;DR

This paper proposes a clinician-in-the-loop smart home system that extracts behavioral markers from ambient sensor data and introduces a novel Conformal Calibrated Interval (CCI) method to quantify predictive uncertainty, enabling reliable detection of urinary tract infection (UTI) flare-ups in older adults and supporting an "abstain when uncertain" decision paradigm.

## Background & Motivation

### State of the Field
Among older adults with chronic conditions, UTIs are among the most common bacterial infections. Symptoms in this population are frequently atypical—manifesting as delirium, confusion, or falls—leading to delayed diagnosis. Conventional home management relies on inconsistent self-reporting and brief clinical assessments, making effective monitoring difficult.

### Limitations of Prior Work
Smart home systems leverage ambient sensors (passive infrared motion detectors, magnetic door sensors, etc.) to continuously monitor residents' daily behavioral patterns and have been applied to cognitive decline detection and sleep quality assessment. However, existing systems suffer from three key limitations:

**High behavioral variability**: even healthy individuals exhibit highly variable activity patterns.

**Heterogeneous effects of comorbidity combinations** on behavioral signals.

**Analysis-only outputs** without involvement in actionable decision-making.

### Root Cause
Existing ML methods output only binary classifications (UTI / no UTI) without quantifying predictive uncertainty, leaving clinicians unable to assess prediction reliability. The authors argue that:
- When a system provides clear uncertainty information, clinicians can handle detected conditions more confidently and effectively.
- Uncertainty estimates must carry statistical guarantees to earn clinical trust.
- The system should abstain under uncertainty ("I don't know"), flagging ambiguous cases for further nurse evaluation.

## Method

### Overall Architecture

The system consists of three stages:
1. **Sensor data collection**: continuous acquisition of daily activity data from smart home ambient sensors.
2. **Behavioral marker extraction and ML classification**: extraction of clinically relevant behavioral features and training of classification models.
3. **Uncertainty quantification and decision support**: the CCI method constructs calibrated intervals over predicted probabilities to produce three-way decisions.

### Key Designs

#### 1. **Behavioral Feature Extraction**: constructing clinically relevant markers from raw sensor events

Seventeen features are extracted from continuous sensor data, including:
- Nighttime bathroom visit count and nighttime non-bathroom activity level.
- Bathroom percentage (ratio of nighttime to total bathroom visits).
- Health event indicators over the preceding three days.
- Daily movement entropy (Shannon entropy measuring activity dispersion).

SHAP values are used for feature importance evaluation, and the top-5 features are selected as primary inputs. **Design Motivation**: UTIs in older adults commonly manifest as increased nighttime activity and changes in bathroom visit frequency—features that directly correspond to clinical indicators.

#### 2. **Naive Uncertainty Intervals**: simple ensemble baseline using random forests

Each decision tree's probability estimate $p_j(x)$ in the random forest is used to compute the mean $\hat{p}(x)$ and standard deviation $\hat{\sigma}(x)$, yielding the interval:

$$C(x) = [\max\{0, \hat{p}(x) - \hat{\sigma}(x)\}, \min\{1, \hat{p}(x) + \hat{\sigma}(x)\}]$$

While intuitive, this approach lacks statistical coverage guarantees and produces overly wide, unreliable intervals when tree predictions are highly variable or skewed.

#### 3. **Conformal Calibrated Interval (CCI)**: core contribution of this paper

The core idea of CCI is to construct statistically guaranteed adaptive prediction intervals over the probability outputs of a binary classifier.

**Step 1: Label mapping.** Binary labels $y \in \{0,1\}$ are mapped to a continuous space:
$$y' = 0.25 + 0.5 \cdot y$$
This creates distinct interval centers for each class (class 0 → 0.25, class 1 → 0.75), facilitating interval construction in probability space.

**Step 2: Adaptive uncertainty scaling.** A scaling function is defined as:
$$\sigma(p) = 1 + (1 - |p - 0.5|)$$
When the predicted probability is near the decision threshold of 0.5 (maximum uncertainty), the scaling factor increases; when near 0 or 1 (high confidence), it decreases. This encodes the clinical intuition that *boundary predictions warrant greater caution*.

**Step 3: Nonconformity scores and calibration.** Given a calibration set $\{(x_i, y_i)\}_{i=1}^n$, nonconformity scores are computed as:
$$S(p_i, y'_i) = \frac{(y'_i - p_i)^2}{\sigma(p_i)}$$
The $(1-\alpha)$ quantile $\hat{q}$ of these scores is then used as the threshold.

**Step 4: Prediction interval construction.** For a new test sample, the interval is:
$$C(x_{\text{test}}) = \{p \in [0,1] \mid S(p, y'_{\text{test}}) \leq \hat{q}\}$$

**Theoretical guarantee (Theorem 1)**: Under the assumption of exchangeability between calibration and test data, the prediction interval contains the transformed label $y'_{\text{test}}$ with probability at least $1-\alpha$.

#### 4. **Interval-Based Three-Way Decision Rule**

Uncertainty intervals are translated into actionable decisions:
- **"UTI"**: interval lower bound $\geq 0.5$, or right-tail probability $\geq 1-\alpha$.
- **"No UTI"**: interval upper bound $< 0.5$, or left-tail probability $\geq 1-\alpha$.
- **"I don't know"**: interval straddles the decision boundary with no strong certainty on either side → flagged as an ambiguous case for further nurse evaluation.

### Loss & Training

- Data split: 10% test, 40% of the remainder for calibration, and the rest for training.
- Error rate $\alpha = 0.1$.
- All results are averaged over 20 independent runs.
- Baseline ML models: logistic regression (selected as the primary model due to consistent superiority over neural networks) and neural networks, both tuned via GridSearchCV with 3-fold cross-validation.

## Key Experimental Results

### Dataset

Real-world data from the CASAS smart home project: 117 annotated daily samples from 8 households (56 UTI-positive days + 61 UTI-negative days). Participants had a mean age of 83.8 years and all presented with multiple chronic conditions.

### Main Results

| Method | Accuracy | Precision | Recall | F1 | Abstention Rate | Interval Width |
|--------|----------|-----------|--------|-----|----------------|----------------|
| Random Guess | 0.49±0.14 | 0.48±0.29 | 0.24±0.14 | 0.32±0.19 | - | - |
| Base ML Model | 0.69±0.15 | 0.68±0.15 | 0.77±0.15 | 0.72±0.12 | - | - |
| Naive-interval | 0.71±0.26 | 0.60±0.41 | 0.61±0.40 | 0.57±0.36 | 0.73±0.12 | 0.60±0.06 |
| **CCI (Ours)** | **0.72±0.16** | **0.74±0.17** | **0.78±0.17** | **0.75±0.14** | **0.22±0.14** | **0.20±0.05** |

### Ablation Study / Key Comparisons

| Dimension | CCI | Naive |
|-----------|-----|-------|
| F1 Score | 0.75 | 0.57 |
| Abstention Rate | 0.22 (low) | 0.73 (very high) |
| Interval Width | 0.20 (compact) | 0.60 (too wide) |
| Clinical Utility | High (most predictions are actionable) | Low (abstains 73% of the time) |

### Key Findings

1. **CCI outperforms all baselines on every classification metric** (including the base ML model) while abstaining far less frequently than the Naive method (0.22 vs. 0.73).
2. **A survey of 42 nurses** validates the clinical usability of CCI outputs: nurses reported that more compact intervals increased willingness to use the system, and described the CCI visualizations as "clear, trustworthy, and useful."
3. **The Naive method is clinically infeasible**: abstaining 73% of the time means actionable recommendations cannot be provided in the vast majority of cases.
4. **Recall (0.78)** is critical for UTI detection—missed detections can lead to serious complications.

## Highlights & Insights

1. **The "I don't know" design is particularly elegant**: unlike conventional binary outputs, the three-way decision (UTI / No UTI / Abstain) better reflects clinical reality—ambiguous cases are escalated to nurses for further evaluation.
2. **The design motivation of the adaptive scaling function $\sigma(p)$ is well-grounded**: it encodes the clinical intuition that "boundary predictions require greater caution" directly in probability space.
3. **Alignment between theoretical guarantees (finite-sample coverage from conformal prediction) and clinical needs**: nurses explicitly stated that "evidence of success probability would make them more comfortable using the system."
4. **A complete sensor-to-decision closed loop**: the system not only detects events but also provides SHAP-based explanations of model predictions.

## Limitations & Future Work

1. **Extremely small dataset** (117 samples, 8 households); generalizability remains to be validated.
2. **Exchangeability assumption** may be violated by the temporal structure of the data.
3. **Interaction effects among multiple chronic conditions** are not modeled; the current system targets UTI only.
4. **Feature engineering relies on domain knowledge**; end-to-end deep learning approaches are not explored.
5. **The deployment pathway describes an idealized scenario** (EHR integration, real-time pipeline, etc.) without a prototype implementation.

## Related Work & Insights

- This work extends conformal prediction from regression and multi-class classification to binary classification in health monitoring settings.
- Nurse feedback confirms that building trust in AI systems requires "theoretically grounded uncertainty guarantees."
- **Insight**: the value of uncertainty quantification in high-stakes decision-making lies not only in technical correctness but also in strengthening human–AI trust.

## Rating

- Novelty: ⭐⭐⭐⭐ (the CCI design is innovative, though it builds on standard applications of conformal prediction)
- Experimental Thoroughness: ⭐⭐⭐ (dataset is very small at only 117 samples; the nurse survey adds practical validation)
- Writing Quality: ⭐⭐⭐⭐ (clear structure with well-articulated motivation)
- Value: ⭐⭐⭐⭐ (the clinical value of introducing uncertainty quantification into smart health monitoring is clearly demonstrated)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Adaptive Retrieval without Self-Knowledge? Bringing Uncertainty Back Home](../../ACL2025/others/adaptive_retrieval_without_self-knowledge_bringing_uncertainty_back_home.md)
- [\[AAAI 2026\] From Decision Trees to Boolean Logic: A Fast and Unified SHAP Algorithm](from_decision_trees_to_boolean_logic_a_fast_and_unified_shap_algorithm.md)
- [\[AAAI 2026\] From Sequential to Recursive: Enhancing Decision-Focused Learning with Bidirectional Feedback](from_sequential_to_recursive_enhancing_decision-focused_learning_with_bidirectio.md)
- [\[AAAI 2026\] DFDT: Dynamic Fast Decision Tree for IoT Data Stream Mining on Edge Devices](dfdt_dynamic_fast_decision_tree_for_iot_data_stream_mining_on_edge_devices.md)
- [\[CVPR 2026\] Customized Fusion: A Closed-Loop Dynamic Network for Adaptive Multi-Task-Aware Infrared-Visible Image Fusion](../../CVPR2026/others/customized_fusion_a_closed-loop_dynamic_network_for_adaptive_multi-task-aware_in.md)

</div>

<!-- RELATED:END -->
