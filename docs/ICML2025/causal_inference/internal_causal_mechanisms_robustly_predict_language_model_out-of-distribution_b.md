---
title: >-
  [Paper Note] Internal Causal Mechanisms Robustly Predict Language Model Out-of-Distribution Behaviors
description: >-
  [ICML 2025][Causal Inference][Causal Interpretability] Using identified internal causal mechanisms in LLMs to predict model output correctness on out-of-distribution (OOD) inputs, this work proposes two methods—counterfactual simulation and value probing—achieving an average AUC-ROC improvement of 13.84% over existing baselines in OOD settings.
tags:
  - "ICML 2025"
  - "Causal Inference"
  - "Causal Interpretability"
  - "Out-of-Distribution Generalization"
  - "Correctness Prediction"
  - "Counterfactual Simulation"
  - "Language Models"
date: 2026-05-08
content_hash: d9479be1c7d22c42
---

# Internal Causal Mechanisms Robustly Predict Language Model Out-of-Distribution Behaviors

**Conference**: ICML 2025  
**arXiv**: [2505.11770](https://arxiv.org/abs/2505.11770)  
**Code**: [Yes](https://github.com/explanare/ood-prediction)  
**Area**: Robotics  
**Keywords**: Causal Interpretability, Out-of-Distribution Generalization, Correctness Prediction, Counterfactual Simulation, Value Probing

## TL;DR

Using identified internal causal mechanisms in LLMs to predict model output correctness on out-of-distribution (OOD) inputs, this work proposes two methods—counterfactual simulation and value probing—achieving an average AUC-ROC improvement of 13.84% over existing baselines in OOD settings.

## Background & Motivation

Interpretability research has provided various techniques to identify abstract internal mechanisms in neural networks. A critical question, however, remains insufficiently answered: **Can one conversely predict the model's behavior on unseen data from identified internal mechanisms?** This is the "predictive direction." Most work focuses on the "forward direction"—searching for internal mechanisms that explain success when the model succeeds. But the reverse problem is equally important.

As LLMs are increasingly deployed in high-risk scenarios, preventing erroneous outputs is critical, yet behavioral testing cannot be exhaustive under combinatorily exploding input spaces and distribution shifts. Traditional methods use confidence scores to estimate correctness, but the calibration of deep models is unreliable on out-of-distribution inputs. Existing works train correctness probes using internal representations, but they rely on heuristically selecting feature locations (such as the last layer of the last token), which lacks a causal foundation.

**Key Challenge**: A large number of internal features can predict correctness on in-distribution data, but the vast majority fail in OOD settings. Only the causal features that genuinely participate in the model's problem-solving process can remain robust under distribution shifts.

**Key Insight**: If it is known which internal features causally participate in the model's in-distribution predictions, these features should act as more robust predictors of out-of-distribution behavior. This extends causal interpretability from "post-hoc understanding" to "predicting generalization."

## Method

### Overall Architecture

A two-stage framework: **Stage 1 (Abstraction)**—Identify the abstract causal mechanism (the high-level causal model and its localization in the neural network) used by the model to solve the task on in-distribution data using DAS; **Stage 2 (Prediction)**—Check whether the model still implements the same mechanism on OOD inputs to predict output correctness.

### Key Designs

#### 1. Counterfactual Simulation

**Function**: Check if the model correctly computes key causal variables on new inputs.

**Mechanism**: Given a high-level causal model $\mathcal{H}: \mathcal{X} \to \mathcal{V} \to \mathcal{Y}$, estimate $P(Y|V)$ by marginalizing out background variables:

$$P(Y|V) = \mathbb{E}_B[P(Y|V, B)]$$

Specifically: Given a test input $x_{\text{src}}$ and $x_{\text{base}}$ from the validation set, perform intervention on the subspace $V$ localized by DAS—injecting the causal variable values of $x_{\text{src}}$ into the context of $x_{\text{base}}$, and checking if the output is consistent. The expectation is approximated using the background variables of $k$ validation set samples:

$$f(\mathcal{M}, x_{\text{test}}) = \frac{1}{kn}\sum_{i=1}^k \sum_{t=1}^n -y_{\text{cf},t}\log(y_{\text{inv},t})$$

**Design Motivation**: Essentially, this detects the robustness of causal relationships against background perturbations—if the model's solution remains stable across multiple backgrounds, the model is more likely to predict correctly under OOD. This requires no additional training and directly reuses the localization results from DAS.

#### 2. Value Probing

**Function**: Predict correctness by learning decision boundaries within the representation subspace of causal variables.

**Mechanism**: Train a linear classifier $\tau$ to distinguish different values of the causal variable $\mathcal{V}$, and use the maximum class probability to predict correctness:

$$f(\mathcal{M}, x_{\text{test}}) = \max_{1 \leq i \leq m}\{\tau(x)_i\}$$

The training objective is the standard classification loss: $\ell_{W_\tau} = \mathbb{E}_{x \in \mathbb{V}}[-\mathbb{1}(\bar{v}) \cdot \log(\tau(x))]$

**Design Motivation**: To avoid the multiple forward pass overhead of counterfactual simulation, this requires only a single forward pass to extract the causal subspace features. Low confidence indicates that the representation falls on the boundary between classes or outside the known range of values.

#### 3. Causal Variable Localization (Based on DAS)

Use Distributed Alignment Search to find the orthogonal basis $Q$ via distributed interchange intervention:

$$r_{\text{inv},i} = (I - Q^\top Q)r_{\text{base},i} + Q^\top Q \cdot r_{\text{src},i}$$

The quality of localization is measured by Interchange Intervention Accuracy (IIA); higher IIA indicates more precise localization of the causal variable.

### Loss & Training

- DAS Localization: Minimize counterfactual cross-entropy $\ell_Q = \mathbb{E}[-y_{\text{cf}} \cdot \log y_{\text{inv}}]$
- Value Probing: Standard classification objective
- Counterfactual Simulation: No additional training required

## Key Experimental Results

### Main Results (OOD Setup, AUC-ROC)

| Task/OOD Type | Ours (Counterfactual Simulation) | Confidence Score | Correctness Probe (Last Token) | Gain |
|-------------|-----------|-----------|---------------------|------|
| PriceTag/Currency Format | **0.856** | 0.631 | 0.627 | +22.9% |
| IOI/Language Shift | **0.997** | 0.767 | 0.607 | +23.0% |
| IOI/Add Typos | **0.875** | 0.777 | 0.840 | +3.5% |
| RAVEL/Language Shift | 0.939 | 0.874 | 0.808 | +6.5% |
| MMLU/ICL Exemplar Shift | 0.765 | 0.707 | **0.784** | - |
| UnlearnHP/Template Shift | **0.772** | 0.739 | 0.648 | +3.3% |

### Ablation Study

| Feature Type | Average OOD AUC-ROC | Description |
|---------|----------------|------|
| Causal Variables (Counterfactual Simulation) | Highest | Best in 8/10 OOD tasks |
| Causal Variables (Correctness Probe) | Second Highest | Last Token overlapping with causal variable locations is also effective |
| Background Variables | Low | Non-causal features are not robust |
| Output Probability | Medium | Competitive on constrained output tasks |

### Key Findings

1. **Causal features significantly outperform non-causal features under OOD**: Average AUC-ROC improved by 13.84%.
2. On tasks where causal mechanisms are fully known (e.g., IOI, RAVEL), counterfactual simulation performs nearly perfectly (>0.99).
3. Confidence scores are effective in-distribution but drop the most under OOD.
4. IIA is positively correlated with AUC-ROC, indicating that localization quality directly affects prediction quality.
5. The advantage of causal methods is small on MMLU because multiple subjects share incomplete causal models.

## Highlights & Insights

- Shifts causal interpretability from an "understanding tool" to a "predictive tool," opening up a major new application direction.
- Counterfactual simulation requires no additional training and no OOD labeled data, offering strong practicality.
- Reveals a key insight: among a vast number of internal features, only a very small subset of causally relevant features retain predictive power under OOD.
- Experiments cover a broad spectrum of tasks, from symbolic tasks to knowledge retrieval and instruction following.

## Limitations & Future Work

- Requires pre-identified causal mechanisms (high-level models are manually designed), limiting generality.
- Counterfactual simulation requires $k$ forward passes, incurring heavy computational overhead (about 5-20x that of standard inference).
- Performance is limited on tasks where only partial mechanisms are identified, such as MMLU.
- Experiments were primary conducted on Llama-3-8B-Instruct; applicability to larger models remains to be verified.
- Assumes the same causal mechanism is still applicable under OOD; it may fail if the model switches calculation paths.
- Automating causal variable localization is not explored; how to combine this with automated circuit discovery methods remains an open question.

## Related Work & Insights

- Deeply connected to mechanistic interpretability: demonstrates a powerful way to translate circuit-level understanding into practical applications.
- Implications for AI safety: causal mechanisms can predict when a model will fail, providing new tools for safety monitoring and alignment.
- Complementary to uncertainty quantification (MC Dropout, Deep Ensembles): instead of quantifying uncertainty, it tests whether the model is "thinking in the right way."
- Methodological contribution to the interpretability field: proves the feasibility of the predictive direction "from mechanism to behavior."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ High originality in shifting from causal interpretability to behavioral prediction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Coverage of five tasks across different types, multiple OOD settings, and comprehensive baseline comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem formulation, rigorous methodological derivation, and in-depth experimental analysis.
- Value: ⭐⭐⭐⭐⭐ Opens up a practical direction for causal interpretability, offering significant insights for AI safety and reliability. Mechanisms Robustly Predict Language Model Out-of-Distribution Behaviors

**Conference**: ICML 2025  
**arXiv**: [2505.11770](https://arxiv.org/abs/2505.11770)  
**Code**: None  
**Area**: Robotics (LLM Reliability / Interpretability)  
**Keywords**: Causal Interpretability, Out-of-Distribution Generalization, Correctness Prediction, Counterfactual Simulation, Language Models

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SelfReflect: Can LLMs Communicate Their Internal Answer Distribution?](../../ICLR2026/causal_inference/selfreflect_can_llms_communicate_their_internal_answer_distribution.md)
- [\[ICML 2025\] Isolated Causal Effects of Natural Language](isolated_causal_effects_of_natural_language.md)
- [\[ICML 2025\] Learning Time-Aware Causal Representation for Model Generalization in Evolving Domains](learning_time-aware_causal_representation_for_model_generalization_in_evolving_d.md)
- [\[ACL 2025\] On the Reliability of Large Language Models for Causal Discovery](../../ACL2025/causal_inference/llm_causal_discovery_reliability.md)
- [\[ICML 2025\] MPF: Aligning and Debiasing Language Models post Deployment via Multi Perspective Fusion](mpf_aligning_and_debiasing_language_models_post_deployment_via_multi_perspective.md)

</div>

<!-- RELATED:END -->
