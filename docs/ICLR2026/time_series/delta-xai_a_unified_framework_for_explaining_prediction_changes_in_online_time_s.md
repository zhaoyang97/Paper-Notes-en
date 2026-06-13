---
title: >-
  [Paper Note] Delta-XAI: A Unified Framework for Explaining Prediction Changes in Online Time Series Monitoring
description: >-
  [ICLR 2026][Time Series][XAI] This paper proposes Delta-XAI, a unified framework that adapts 14 existing XAI methods to the scenario of explaining prediction changes in online time series monitoring via a wrapper functio…
tags:
  - "ICLR 2026"
  - "Time Series"
  - "XAI"
  - "Online Monitoring"
  - "Feature Attribution"
  - "Integrated Gradients"
date: 2026-05-08
content_hash: 6dc07bdac1036cbe
---

# Delta-XAI: A Unified Framework for Explaining Prediction Changes in Online Time Series Monitoring

**Conference**: ICLR 2026  
**arXiv**: [2511.23036](https://arxiv.org/abs/2511.23036)  
**Code**: [Anonymous GitHub](https://anonymous.4open.science/r/Delta-XAI)  
**Area**: Time Series / Explainable AI  
**Keywords**: XAI, Time Series, Online Monitoring, Feature Attribution, Integrated Gradients

## TL;DR
This paper proposes Delta-XAI, a unified framework that adapts 14 existing XAI methods to the scenario of explaining prediction changes in online time series monitoring via a wrapper function. It further introduces SWING (Shifted Window Integrated Gradients), which constructs integration paths using past observations to capture temporal dependencies, consistently outperforming existing methods across multiple evaluation metrics.

## Background & Motivation
Online time series monitoring models are critical in sensitive domains such as healthcare (e.g., ICU monitoring) and finance, where clinicians and decision-makers need to understand why model predictions change between consecutive time steps. Despite progress in time series XAI, three core problems persist:

**Step-wise Independent Analysis**: Most XAI methods analyze predictions at each time step independently, ignoring temporal dependencies and failing to explain "why the prediction changed from $t-1$ to $t$."

**Underutilization of Online Dynamics**: Existing methods do not fully exploit the nature of online monitoring—data arrives sequentially and predictions are continuously updated.

**Evaluation Difficulty**: There is no systematic evaluation framework tailored to the online setting that can comprehensively assess the faithfulness, sufficiency, and coherence of explanations.

The root cause lies in the absence of an XAI framework that simultaneously explains prediction changes (rather than point predictions), adapts to online dynamics, and supports principled evaluation.

The paper's starting point is not to reinvent XAI methods, but rather to adapt 14 existing methods to the new scenario of "explaining prediction differences" through a unified wrapper function, while also proposing a comprehensive evaluation suite. The core innovation is the SWING method, which captures causal temporal dependencies by incorporating past time-step observations into the integration path.

## Method

### Overall Architecture
Delta-XAI takes as input an online time series monitoring model (e.g., LSTM) and streaming time series data. At each time step $t$, the framework focuses not only on the current prediction but on the prediction change ($\Delta$ prediction) from $t-1$ to $t$. The output is an attribution score for each input feature with respect to the prediction change, helping users understand "which features caused the prediction to shift."

The overall pipeline consists of three stages:
- **Wrapper Adaptation Stage**: 14 XAI methods are uniformly adapted to the prediction-change explanation scenario via a `PredictionDifferenceWrapper`.
- **Attribution Computation Stage**: The adapted methods compute feature contributions to prediction changes.
- **Evaluation Stage**: A multi-dimensional evaluation suite quantifies explanation quality.

### Key Designs

1. **Prediction Difference Wrapper**: This is the core abstraction layer of the framework. For any XAI method $E$, the wrapper transforms it from "explaining a point prediction" to "explaining a prediction change" by redirecting the attribution target from $f(x_t)$ to $f(x_t) - f(x_{t-1})$, i.e., attributing the prediction difference. This allows any existing XAI method (e.g., SHAP, LIME, gradient-based methods) to be seamlessly applied to the online prediction-change explanation setting. The 14 adapted methods span gradient-based (e.g., Saliency, IG, SmoothGrad), perturbation-based (e.g., SHAP, LIME), and attention-based paradigms.

2. **SWING (Shifted Window Integrated Gradients)**: This is the novel method proposed in this paper, motivated by the key insight that standard Integrated Gradients (IG) uses a zero or random baseline as the integration starting point, which in online settings causes two problems: (a) a zero baseline may deviate from the data distribution, inducing out-of-distribution (OOD) effects; and (b) temporal context is ignored. SWING's core mechanism is a "shifted window"—using the observation at the previous time step $x_{t-1}$ as the starting point of the integration path, integrating from $x_{t-1}$ to $x_t$. This offers three advantages: the integration path remains within the data distribution (avoiding OOD), temporal changes are naturally captured, and the integration result directly corresponds to the attribution of the prediction difference.

3. **Multi-dimensional Evaluation Suite**: The framework introduces a principled set of evaluation metrics that assess explanation quality in the online setting from multiple perspectives:

    - **Faithfulness**: Measures whether attribution scores genuinely reflect the model's dependence on features. This is assessed by progressively masking the most/least important features and observing the magnitude of prediction change.
    - **Sufficiency**: Retains only highly attributed features and verifies whether they suffice to reproduce the prediction change.
    - **Coherence**: Assesses whether explanations remain temporally consistent and stable under similar input conditions.
    - Additional dimensions such as compactness are also included.

### Loss & Training
The core contribution of this paper lies in the design and evaluation of explanation methods rather than in training new models. The target model (LSTM) is trained with standard time series prediction loss. SWING requires no additional training and directly computes attributions using model gradients. All metrics in the evaluation suite are quantified through perturbation experiments and statistical tests, without any additional learning process.

## Key Experimental Results

### Main Results
Experiments are conducted primarily on the MIMIC-III clinical dataset using an LSTM as the target prediction model. A systematic comparison is performed across 14 adapted XAI methods and SWING.

| Evaluation Dimension | Best Baseline Method | SWING | Trend |
|---|---|---|---|
| Faithfulness | Integrated Gradients | SWING | Consistently superior to IG |
| Sufficiency | IG / Gradient×Input | SWING | Clear advantage across multiple settings |
| Coherence | IG | SWING | Stronger temporal consistency |

### Ablation Study

| Configuration | Key Finding | Explanation |
|---|---|---|
| Original vs. adapted methods | Adapted methods show significant improvement | Wrapper function is effective |
| Zero-baseline IG vs. SWING | SWING is overall superior | Shifted window avoids OOD effects |
| Classic gradient methods vs. recent methods | Classic methods after adaptation can outperform recent ones | Challenges the "newer is better" intuition |
| Different window sizes | Optimal performance at moderate window size | Overly large windows may introduce noise |

### Key Findings
- Classic gradient-based methods (e.g., IG), after temporal adaptation, can surpass recently proposed specialized time series XAI methods, demonstrating that "classic methods + proper adaptation" may be more effective than purpose-built new methods.
- SWING leverages the natural structure of the online setting (the previous time step as a reference baseline) to simultaneously avoid OOD issues and capture temporal dependencies.
- The evaluation framework reveals trade-offs among different XAI methods across dimensions; no single method dominates all dimensions (with SWING as the exception).
- Evaluating XAI in the online setting is more challenging than in the static setting, requiring additional dimensions such as temporal coherence.

## Highlights & Insights
- **Unified Framework Philosophy**: Rather than reinventing the wheel, the wrapper-based adaptation design allows the framework to immediately benefit from any future XAI advances.
- **Counter-intuitive Finding**: The classic IG method, after adaptation, can outperform methods specifically designed for time series, suggesting that precise problem formulation may matter more than algorithmic novelty.
- **Elegance of SWING**: Simply changing the starting point of the integration path (from a zero baseline to the previous time step) simultaneously resolves the OOD problem and captures temporal dependencies.
- **Completeness of the Evaluation Suite**: Multi-dimensional assessment across faithfulness, sufficiency, and coherence provides a standardized evaluation toolkit for online XAI research.

## Limitations & Future Work
- Experiments are conducted primarily on the MIMIC-III dataset with an LSTM model; generalizability to other datasets (e.g., financial time series) and model architectures (e.g., Transformers) remains to be validated.
- The shifted window mechanism in SWING assumes smooth variation between adjacent time steps, which may limit its effectiveness during extreme abrupt events (e.g., market crashes).
- The wrapper function design for 14 methods may require adaptation for different model architectures; the current implementation primarily targets RNN-based models.
- Despite the comprehensiveness of the evaluation metrics, validation against human expert judgment (human evaluation) is absent.
- Evaluating all 14 methods incurs high computational cost; a method selection strategy is needed for practical deployment.

## Related Work & Insights
- **Integrated Gradients (IG)**: SWING directly builds upon IG, demonstrating that deep understanding of classical methods can inspire efficient new approaches.
- **Time Series XAI (e.g., TimeSHAP, WinIT)**: The unified framework places these methods under a common evaluation standard, providing a valuable comparative perspective.
- **Online Learning and Concept Drift**: The online setting of Delta-XAI has potential connections to concept drift detection, which could be explored in future work.
- The "wrapper adaptation + unified evaluation" paradigm introduced in this paper can be generalized to other XAI application scenarios, such as explaining online recommendation systems.

## Technical Details
- SWING integrates along a straight-line path from $x_{t-1}$ to $x_t$, with the number of integration steps typically set between 50 and 300 to balance accuracy and efficiency.
- The 14 XAI methods supported by the wrapper include: Saliency, InputXGradient, GuidedBackprop, DeepLift, IG (standard and SWING variant), SmoothGrad, GradientSHAP, KernelSHAP, LIME, Occlusion, Feature Ablation, Feature Permutation, and Shapley Value Sampling.
- The LSTM model is trained on MIMIC-III for an online mortality prediction task, with inputs being 48-hour multivariate clinical time series (including vital signs, laboratory tests, and other features).
- The faithfulness metric in the evaluation suite employs a Top-$k$ masking strategy: features with the highest attribution scores are progressively masked, and the decay curve of prediction change magnitude is measured.
- Experiments are conducted on Intel Xeon Silver 4210 CPUs and 8× NVIDIA TITAN RTX GPUs.

## Rating
- Novelty: ⭐⭐⭐⭐ (Unified framework and SWING are novel, though the core contribution is an ingenious combination of existing methods)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Systematic comparison of 14 methods is comprehensive, though the number of datasets is limited)
- Writing Quality: ⭐⭐⭐⭐ (Problem formulation is clear and framework diagrams are well-structured)
- Value: ⭐⭐⭐⭐ (Provides important benchmarks and tools for online time series XAI)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Online Time Series Prediction Using Feature Adjustment](online_time_series_prediction_using_feature_adjustment.md)
- [\[ICLR 2026\] Towards Robust Real-World Multivariate Time Series Forecasting: A Unified Framework](towards_robust_real-world_multivariate_time_series_forecasting_a_unified_framewo.md)
- [\[ICLR 2026\] ResCP: Reservoir Conformal Prediction for Time Series Forecasting](rescp_reservoir_conformal_prediction_for_time_series_forecasting.md)
- [\[ICLR 2026\] SwiftTS: A Swift Selection Framework for Time Series Pre-trained Models via Multi-task Meta-Learning](swiftts_a_swift_selection_framework_for_time_series_pre-trained_models_via_multi.md)
- [\[ICLR 2026\] Uni-NTFM: A Unified Foundation Model for EEG Signal Representation Learning](uni-ntfm_a_unified_foundation_model_for_eeg_signal_representation_learning.md)

</div>

<!-- RELATED:END -->
