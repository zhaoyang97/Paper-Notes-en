---
title: >-
  [Paper Note] DeltaSHAP: Explaining Prediction Evolutions in Online Patient Monitoring with Shapley Values
description: >-
  [Interpretability] DeltaSHAP is an explainable AI algorithm designed specifically for online patient monitoring systems. By adapting Shapley values to temporal scenarios, it explains the **evolution (change)** between consecutive predictions rather than absolute prediction values. It provides both the **direction and magnitude** of feature attributions, achieving a 62% improvement in explanation quality and a 33% reduction in computation time on the MIMIC-III benchmark.
tags:
  - "Interpretability"
date: 2026-05-08
content_hash: 3a414f6fe6cbd308
---

# DeltaSHAP: Explaining Prediction Evolutions in Online Patient Monitoring with Shapley Values

- **Conference**: ICML 2025
- **arXiv**: [2507.02342](https://arxiv.org/abs/2507.02342)
- **Code**: [https://github.com/AITRICS/DeltaSHAP](https://github.com/AITRICS/DeltaSHAP)
- **Area**: Medical XAI / Online Patient Monitoring
- **Keywords**: Shapley values, Explainable AI, Online patient monitoring, Time-series attribution, Clinical decision support

## TL;DR

DeltaSHAP is an explainable AI algorithm designed specifically for online patient monitoring systems. By adapting Shapley values to temporal scenarios, it explains the **evolution (change)** between consecutive predictions rather than absolute prediction values. It provides both the **direction and magnitude** of feature attributions, achieving a 62% improvement in explanation quality and a 33% reduction in computation time on the MIMIC-III benchmark.

---

## Background & Motivation

### Clinical Scenario Needs

Online patient monitoring systems (such as early warning systems in the ICU) continuously track patient vital signs and laboratory results to predict the risk of clinical deterioration in real time. However, when clinicians use these systems, the core question they face is not "What is the current predicted risk?" but rather "**Why did the prediction change?**" For example:
- A drop in sepsis probability from 70% to 40% indicates improvement.
- Conversely, a 40% probability rising from 10% indicates severe deterioration.
- The exact same absolute predicted value can have entirely different clinical meanings depending on the context.

Therefore, clinical XAI methods must satisfy three key requirements:

**Explaining consecutive prediction differences**: Rather than isolated absolute predictions at single time points.

**Providing directional attribution**: The contribution of each feature to the prediction must have both a magnitude and a positive/negative direction.

**Real-time computation**: Delivering explanations rapidly in time-sensitive clinical settings.

### Limitations of Prior Work

**General XAI methods** (e.g., LIME, SHAP, IG, DeepLIFT):
- Focus on estimating attributions for absolute predictions and cannot explain prediction changes.
- Attempt to calculate pointwise attributions for all features and all time steps, resulting in a heavy computational burden.
- Ignore the temporal dependencies inherent in time-series data.

**Time-series specific XAI methods** (e.g., FIT, WinIT):
- FIT quantifies feature attribution based on a KL divergence framework, but **only provides attribution magnitude without direction**.
- WinIT extends this to model delayed influences but similarly lacks directional information.
- Both rely on conditional generative models to generate counterfactual samples, making them **computationally expensive** and unsuitable for time-sensitive clinical scenarios.

---

## Method

### Overall Architecture

The core idea of DeltaSHAP is to shift the explanation target from "absolute prediction value" to the "difference in prediction $\Delta$ between consecutive time steps." By employing Shapley value sampling, it efficiently computes the directional contribution of newly observed features to $\Delta$.

The overall workflow consists of three core modules:

1. **Prediction Difference Definition** (Section 4.1): Defines the explanation target as $\Delta$.
2. **Shapley Value Sampling** (Section 4.2): Approximates Shapley values through permutation sampling.
3. **Baseline Selection Strategy** (Section 4.3): Employs forward-filling instead of generative models to handle missing values.

### Key Designs

#### 1. Prediction Difference Modeling

Given an online monitoring model $f: \mathbb{R}^{L \times D} \rightarrow [0,1]$, where $L$ is the maximum sequence length and $D$ is the number of clinical features. At time step $T$, the model performs predictions using a sliding window $\mathbf{X}_{T-W+1:T} \in \mathbb{R}^{W \times D}$.

The explanation target is defined as the prediction difference between two consecutive time steps:

$$\Delta = f(\mathbf{X}_{T-W+1:T}) - f(\mathbf{X}_{T-W+1:T} \setminus \mathbf{X}_T)$$

where $f(\mathbf{X}_{T-W+1:T} \setminus \mathbf{X}_T)$ represents the prediction made using only historical data without the measurement values at the current time step.

The goal is to compute an attribution vector $\phi(f, \mathbf{X}_{T-W+1:T}) \in \mathbb{R}^D$, where $\phi_j$ quantifies the contribution of feature $j$ at time step $T$ to the prediction evolution, satisfying the **efficiency property**:

$$\sum_{j=1}^{D} \phi_j(f, \mathbf{X}_{T-W+1:T}) = \Delta$$

A positive value indicates that the feature drives the prediction higher, while a negative value indicates it pulls the prediction lower.

#### 2. Shapley Value Sampling Approximation

Exactly calculating Shapley values requires evaluating $2^D$ feature coalitions, which is computationally prohibitive for high-dimensional clinical data. DeltaSHAP adopts a permutation sampling approach (Shapley Value Sampling):

By sampling $N$ random permutations $\Omega$, the Shapley value is approximated for each observed feature $j \in \mathcal{F}_{\text{obs}}$ as:

$$\hat{\phi}_j(f, \mathbf{X}_{T-W+1:T}) = \frac{1}{N} \sum_{\pi \in \Omega} [v(S_{\pi,j} \cup \{j\}) - v(S_{\pi,j})]$$

where $S_{\pi,j}$ is the set of features preceding feature $j$ in permutation $\pi$, and $v(S)$ measures the marginal contribution to $\Delta$ when only the subset $S$ of features is observed:

$$v(S) = f(\mathbf{X}_{T-W+1:T-1} \cup \mathbf{X}_T^S) - f(\mathbf{X}_{T-W+1:T} \setminus \mathbf{X}_T)$$

#### 3. Attribution Normalization

To eliminate sampling errors and ensure the efficiency property, the attributions are normalized:

$$\phi_j(f, \mathbf{X}_{T-W+1:T}) = \hat{\phi}_j(f, \mathbf{X}_{T-W+1:T}) \cdot \frac{\Delta}{\sum_{k \in \mathcal{F}_{\text{obs}}} \hat{\phi}_k(f, \mathbf{X}_{T-W+1:T})}$$

This ensures that the sum of all attributions precisely equals the observed prediction difference without altering the relative ranking of feature importance.

#### 4. Baseline Selection: Forward-Filling

Unlike FIT/WinIT, which rely on conditional generative models to handle unobserved features, DeltaSHAP leverages the existing missing-value handling mechanism already present in the preprocessing pipeline. For LSTM models, it directly uses **forward-filling** (i.e., filling missing features with their most recent observed values).

The advantages of this strategy are:
- It aligns with the model's preprocessing pipeline, avoiding out-of-distribution issues.
- It eliminates the need to train additional generative models, significantly reducing computational overhead.
- It naturally handles irregularly sampled clinical data.

### Algorithm Pseudocode

The complete workflow of DeltaSHAP (Algorithm 1):
1. Calculate the prediction difference $\Delta$.
2. Generate $N$ random permutations of the feature observation set $\mathcal{F}_{\text{obs}}$.
3. For each feature in each permutation, calculate the marginal contribution (highly parallelizable).
4. Accumulate marginal contributions and divide by $N$ to obtain the approximated Shapley values.
5. Normalize to satisfy the efficiency property.

Key implementation detail: Both the feature iteration and permutation iteration are **parallelized** in the implementation to further enhance computational efficiency.

---

## Evaluation Metric Innovations

The paper proposes a new suite of evaluation metrics to measure the **faithfulness** of attributions in online time-series forecasting:

### Base Metrics
- **CPD** (Cumulative Prediction Difference): The cumulative change in prediction after progressively removing the most important features.
- **CPP** (Cumulative Prediction Preservation): The cumulative change in prediction after progressively removing the least important features.

### Main Evaluation Metrics
- **AUPD** (Area Under Prediction Difference): The area under the CPD curve ($\uparrow$ higher is better).
- **AUPP** (Area Under Prediction Preservation): The area under the CPP curve ($\downarrow$ lower is better).
- **AUAUCD / AUAUCP**: Dataset-level performance change metrics based on AUC.
- **AUAPRD / AUAPRP**: Dataset-level performance change metrics based on APR.

The advantages of these metrics: (1) They emphasize the influence of top-ranked features; (2) They reduce sensitivity to local anomalies by aggregating removal effects across multiple levels.

---

## Key Experimental Results

### Datasets

| Dataset | Task | ICU Admissions | Prediction Instances | Positive Ratio | Prediction Window |
|--------|------|-----------|------------|------------|----------|
| MIMIC-III | Decompensation Prediction | ~41,000 | ~2.5 Million | 2.5% | 24 Hours |
| PhysioNet 2019 | Sepsis Prediction | ~40,000 | ~1.1 Million | 2.5% | 12 Hours |

### Main Results (Table 1)

**MIMIC-III Decompensation Prediction** (LSTM backbone):

| Method | AUPD ↑ | AUPP ↓ | Wall-Clock Time (s) |
|------|--------|--------|---------------------|
| LIME | 8.20 | 21.58 | 0.22 |
| GradSHAP | 6.20 | 19.68 | 0.03 |
| IG | 13.46 | 14.51 | 0.04 |
| DeepLIFT | **13.95** | 14.35 | 0.03 |
| FO | 13.55 | **14.14** | 1.43 |
| AFO | 13.08 | 15.14 | 39.62 |
| FIT | 12.60 | 16.16 | 0.12 |
| WinIT | 10.06 | 16.56 | 0.30 |
| **DeltaSHAP** | **22.59** | **3.04** | **0.02** |

DeltaSHAP outperforms the second-best method on AUPD by 62% (22.59 vs. 13.95) and reduces AUPP by 78.5% (3.04 vs. 14.14).

**PhysioNet 2019 Sepsis Prediction**:

| Method | AUPD ↑ | AUPP ↓ | Time (s) |
|------|--------|--------|----------|
| AFO | 3.27 | 1.03 | 14.18 |
| FIT | 2.15 | 3.08 | 0.11 |
| **DeltaSHAP** | **3.68** | **0.89** | **0.02** |

### Ablation Study (Table 2, MIMIC-III)

| Configuration | AUPD ↑ | AUPP ↓ | Time (s) |
|------|--------|--------|----------|
| w/o Baseline Selection (Zero-filling) | 8.49 | 18.89 | 0.05 |
| w/o Normalization | 22.58 | 3.05 | 0.05 |
| N=1 | 22.14 | 3.19 | 0.02 |
| N=10 | 22.56 | 3.07 | 0.04 |
| N=100 | 22.61 | 3.04 | 0.09 |
| **DeltaSHAP (N=25)** | **22.58** | **3.05** | **0.05** |

### Key Findings

1. **Baseline selection is crucial**: Removing forward-filling and replacing it with zero-filling causes AUPD to plunge from 22.58 to 8.49—a performance degradation of over 60%, indicating that it is the most critical design choice.
2. **Normalization affects interpretability rather than ranking**: Removing normalization barely affects quantitative metrics. However, normalization guarantees the efficiency property, ensuring that the sum of attributions exactly equals the prediction difference, which enhances clinical interpretability.
3. **N=25 is the optimal trade-off**: While N=1 yields reasonable performance, N=25 achieves the optimal balance between accuracy and efficiency. N=100 offers only marginal gains while doubling computation time.
4. **DeltaSHAP is the only method satisfying AUPD > AUPP**: This indicates that removing important features leads to a larger change in predictions than removing unimportant features, proving that the attributions are highly aligned with model behavior.

---

## Qualitative Case Analysis

The paper validates the clinical consistency of DeltaSHAP through multiple clinical case studies on the MIMIC-III decompensation prediction task:

1. **Oxygen Saturation (SpO2)**: DeltaSHAP correctly identifies a sudden drop in SpO2 from 92% to 60% as a high-risk signal (below 70% indicates acute danger), and a recovery from 60% back to 98% as an improvement signal.
2. **Hyperglycemia**: DeltaSHAP correctly attributes high importance to blood glucose values exceeding 300 mg/dL, aligning with the clinical association between hyperglycemia and cardiac decompensation.
3. **Sudden Drop in Blood Pressure**: Sharp decreases in systolic blood pressure (SBP) and diastolic blood pressure (DBP) are correctly attributed as primary risk factors for decompensation, consistent with clinical knowledge of impaired cardiac output.

---

## Highlights & Insights

1. **Precise Problem Modeling**: Shifting the explanation target from "absolute prediction" to "prediction difference" perfectly matches the actual clinical need of "focusing on changes rather than absolute values."
2. **Simple and Effective Engineering Choices**: Replacing complex generative models with forward-filling to handle unobserved features significantly improves computational efficiency by orders of magnitude while preserving explanation quality (AFO requires 39.62s vs. DeltaSHAP's mere 0.02s).
3. **Comprehensive Evaluation Framework**: The proposed faithfulness evaluation metrics range from instance-level (AUPD/AUPP) to dataset-level (AUAUCD/AUAPRP), filling a major gap in the evaluation of online time-series XAI.
4. **Theoretical Guarantees combined with Practicality**: The efficiency property of Shapley values (where the sum of attributions equals the prediction difference) provides a solid theoretical foundation for clinical explanations, and the normalization step ensures this property holds even under approximation.
5. **Model-Agnostic Nature**: DeltaSHAP does not require access to model gradients or internal states, allowing it to adapt to any black-box model architecture.

---

## Limitations & Future Work

1. **Focus on the Latest Time Step Only**: By only explaining features from the most recent observation to reduce computation, the method may miss the **delayed effects** of observations from earlier time steps.
2. **Insufficient Validation for Single-Point Analysis**: Its performance in non-online, non-monitoring scenarios (such as one-off predictions) has not been fully verified.
3. **Masking Effects in Complex Gating Mechanisms**: Internal gating mechanisms in models like LSTMs might mask the feature interactions that DeltaSHAP attempts to capture.
4. **Validated Only on LSTMs**: Although the method is model-agnostic, the experiments only used LSTM as the backbone and have not been validated on more modern architectures like Transformers.
5. **Extremely Low Positive Ratio** (2.5%): The APR-related indicators in the evaluation results may be skewed by the class imbalance.

---

## Related Work & Insights

- **General XAI Methods**: LIME, SHAP (KernelSHAP, GradSHAP, DeepSHAP), IG, DeepLIFT, FO, AFO — mostly target static predictions (image classification, tabular data) and fail to consider temporal dependencies.
- **Time-Series XAI Methods**: TimeSHAP (Bento et al., 2021), FIT (Tonekaboni et al., 2020), WinIT (Leung et al., 2021), Dynamic Masks (Crabbé & Van Der Schaar, 2021), TimeX++ (Liu et al., 2024a) — FIT and WinIT are the most direct competitors but lack directional attribution and depend on generative models.
- **Shapley Value Sampling**: Mitchell et al. (2022), Strumbelj & Kononenko (2010) — form the technical foundation for DeltaSHAP's sampling approximation.

---

## Rating

⭐⭐⭐⭐ (4/5)

**Pros**: Precise problem definition, simple and highly efficient design, significant performance gains (62% improvement + 33% acceleration), strong clinical consistency, and a comprehensive evaluation framework.

**Cons**: Validated only on LSTMs, does not account for delayed effects, and experimented on only two datasets. Overall, it is a solid piece of work with high clinical utility in the patient monitoring domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Enhancing Automated Interpretability with Output-Centric Feature Descriptions](../../ACL2025/interpretability/output_centric_interpretability.md)
- [\[ACL 2025\] A Dual-Perspective NLG Meta-Evaluation Framework with Automatic Benchmark and Better Interpretability](../../ACL2025/interpretability/a_dual-perspective_nlg_meta-evaluation_framework_with_automatic_benchmark_and_be.md)
- [\[ICLR 2026\] ZeroTuning: Unlocking the Initial Token's Power to Enhance Large Language Models Without Training](../../ICLR2026/interpretability/zerotuning_unlocking_the_initial_tokens_power_to_enhance_large_language_models_w.md)
- [\[ICLR 2026\] SEED-SET: Scalable Evolving Experimental Design for System-level Ethical Testing](../../ICLR2026/interpretability/seed-set_scalable_evolving_experimental_design_for_system-level_ethical_testing.md)
- [\[ICLR 2026\] Joint Distribution–Informed Shapley Values for Sparse Counterfactual Explanations](../../ICLR2026/interpretability/joint_distributioninformed_shapley_values_for_sparse_counterfactual_explanations.md)

</div>

<!-- RELATED:END -->
