---
title: >-
  [Paper Note] ProbLog4Fairness: A Neurosymbolic Approach to Modeling and Mitigating Bias
description: >-
  [AI Safety] The ProbLog4Fairness framework is proposed, which leverages the probabilistic logic programming language ProbLog to formalize bias mechanisms in data as interpretable logic programs, and integrates bias assumptions into neural network training through distant supervision via DeepProbLog, achieving flexible and principled bias mitigation.
tags:
  - "AI Safety"
date: 2026-05-08
content_hash: 37516902137d35a1
---

# ProbLog4Fairness: A Neurosymbolic Approach to Modeling and Mitigating Bias

- **Conference**: AAAI 2026
- **arXiv**: [2511.09768](https://arxiv.org/abs/2511.09768)
- **Code**: Not provided
- **Area**: AI Safety
- **Keywords**: Algorithmic Fairness, Neurosymbolic AI, Probabilistic Logic Programming, Bias Modeling, DeepProbLog

## TL;DR

The ProbLog4Fairness framework is proposed, which leverages the probabilistic logic programming language ProbLog to formalize bias mechanisms in data as interpretable logic programs, and integrates bias assumptions into neural network training through distant supervision via DeepProbLog, achieving flexible and principled bias mitigation.

## Background & Motivation

- The field of algorithmic fairness contains numerous mutually incompatible definitions of fairness (such as statistical parity, equalized odds, etc.), and choosing which constraint to apply remains normatively controversial.
- Existing methods typically optimize for a fixed bias type or fairness metric, lacking the ability to flexibly model different sources of bias for specific tasks.
- Various bias mechanisms in the data generation process (label bias, measurement bias, historical bias) are significant sources of model unfairness.
- Although causal models can describe bias mechanisms, they are difficult to interpret and integrate into data pipelines in practice.
- **Design Motivation**: Rather than choosing a fixed fairness constraint, it is better to directly model and correct the bias mechanisms in the data, allowing the model to "naturally" generate fairer decisions.

## Method

### 1. ProbLog Bias Modeling Framework

The core idea is to encode bias assumptions as ProbLog probabilistic logic programs. ProbLog is a probabilistic extension of Prolog that supports probabilistic facts $p::f$, representing that fact $f$ is true with probability $p$.

**Bias Represented as Probabilistic Facts**: For binary variables, the bias transition is fully defined by four probabilistic facts, respectively describing:
- $p_1$: Negative bias for the sensitive group (changing 1 to 0)
- $p_2$: Negative bias for the non-sensitive group
- $p_3$: Positive bias for the sensitive group (changing 0 to 1)
- $p_4$: Positive bias for the non-sensitive group

Taking label bias as an example, the ProbLog template is:

$$\tilde{y}(\mathbf{X}) \leftarrow y_h(\mathbf{X}) \wedge \neg \text{label\_neg\_bias}(\mathbf{X})$$
$$\tilde{y}(\mathbf{X}) \leftarrow \neg y_h(\mathbf{X}) \wedge \text{label\_pos\_bias}(\mathbf{X})$$

Where $y_h(\mathbf{X})$ is the unbiased label predicted by the classifier, and $\tilde{y}(\mathbf{X})$ is the observed biased label.

### 2. DeepProbLog Integration and Distant Supervision Training

DeepProbLog allows neural networks to predict parameters of probabilistic facts, enabling logical reasoning on top of neural network predictions. Training workflow:

1. **Define the Logic Program**: Includes ProbLog rules for the classifier $h(\mathbf{X})::y_h(\mathbf{X})$ and the bias mechanisms.
2. **Compile to Circuit**: The logic program is compiled into an arithmetic circuit that computes query probabilities.
3. **Distant Supervision**: Supervised only by the biased labels $\tilde{y}$, gradients are backpropagated through the logic circuit to update the network.
4. **Stripping at Inference Time**: During testing, if unbiased features are available, prediction is made directly using $h(\mathbf{X})$; if only biased features are available, the bias transition mechanism is retained.

The key advantage is that gradient updates account for all unbiased explanations consistent with the observed biased data.

### 3. Modeling Three Types of Bias

**Label Bias**: Unbiased features are available, but the observed label $\tilde{Y}$ is a biased proxy of the true label $Y$. For example, discrimination against specific groups by loan officers in loan approval. This is modeled using Template 1 during training.

**Measurement Bias**: Observed features $\tilde{X}_i$ are noisy proxies of unbiased features $X_i$, while labels rely solely on the unbiased features. For instance, using "number of days worked in the past three years" to measure job stability is unfair to women taking maternity leave. The debiasing process is modeled using Template 2:

$$y(\tilde{\mathbf{X}}) \leftarrow \text{debias}(\tilde{\mathbf{X}}, \mathbf{X}) \wedge y_h(\mathbf{X})$$

**Historical Bias**: Both features and labels are affected by bias, and labels are generated based on biased features. Assuming that the mapping from biased features to biased labels is identical to the mapping from unbiased features to unbiased labels, debiasing is performed at inference time using Template 2.

### 4. Parameter Settings

The bias probability parameters $p_i$ can be set in the following ways:
- Direct assignment based on domain knowledge.
- Estimation from a small-scale data subset containing both biased and unbiased labels.
- Based on Hoeffding's inequality, estimating the parameters within a 10% error margin with 95% confidence requires only 184 samples.

## Experiments

### Synthetic Data Experiments

| Experiment | Bias Type | Key Findings |
|------|----------|----------|
| RQ1: Comparison of different bias types | Label/Measurement/Historical | ProbLog4Fairness is close to the upper bound baseline in both accuracy and statistical disparity, significantly outperforming other mitigation methods. |
| RQ2: $A \not\perp Y$ Scenario | Label Bias | Captures and removes only problematic bias while preserving legitimate correlation; other baselines erroneously force statistical disparity to zero. |
| RQ3: Parameter Sensitivity | Label Bias | Optimal accuracy is achieved at the true parameters and is robust to parameter estimation errors. |

### Real-world Data Experiments

| Dataset | Type | Source of Bias | Sensitive Variable | ProbLog4Fairness Results |
|--------|------|----------|----------|----------------------|
| Student Alcohol | Tabular Data (856 samples) | Annotator bias (subjective bias toward male students) | Gender | F1 score outperforms all mitigation baselines, and statistical disparity is close to the level of unbiased data. |
| CELEB-A | Image Data | Annotation inconsistency (Mouth Slightly Open attribute) | Smiling/Blurry/High Cheekbones | Both F1 and statistical disparity improve significantly when correcting three sensitive attributes simultaneously, outperforming single-attribute correction. |

**Baseline methods** include: Lower (trained directly on biased data), Upper (trained on unbiased data), Unawareness (removing sensitive variables), Massaging (data preprocessing), and Error Parity (post-processing method).

## Key Findings

1. Due to its flexible modeling capability, ProbLog4Fairness simultaneously approaches the ideal upper bound in both accuracy and fairness, which is unachievable by other methods with fixed assumptions.
2. When $A \not\perp Y$ (the sensitive variable is not independent of the label), ours can distinguish problematic bias from legitimate correlation, whereas other methods suffer from over-correction.
3. Parameter estimation has little impact on the results, and a small amount of unbiased data is sufficient to set parameters effectively.
4. In CELEB-A experiments, simultaneously modeling bias from multiple sensitive attributes far outperforms correcting a single attribute, highlighting the importance of flexible modeling.
5. Simplifying assumptions (such as "no positive bias") may unexpectedly improve performance on real data by reducing parameter estimation error.

## Highlights & Insights

- **Unification of Principled and Flexible Approaches**: ProbLog's declarative programming allows users to add and modify bias assumptions flexibly based on specific scenarios without relying on a fixed definition of fairness.
- **High Interpretability**: Bias assumptions are expressed as logical rules, which can be directly understood and validated by domain experts.
- **Unified Framework for Multiple Bias Types**: Label, measurement, and historical biases can all be combined and modeled under a unified framework.
- **Generalization from Tabular to Image Data**: Successfully combining ResNet-50 with ProbLog on CELEB-A demonstrates effectiveness even for high-dimensional, non-tabular data.

## Limitations & Future Work

- Bias parameters need to be predefined or estimated from small-scale unbiased data; if parameters are learned jointly, the classifier becomes unidentifiable.
- Currently, only binary sensitive variables and binary labels are supported; scaling to multi-class requires a large number of probabilistic facts.
- Synthetic experiments are limited to binary/categorical features, and the handling of continuous features has not been discussed in depth.
- The compilation and inference of DeepProbLog are computationally expensive for complex programs (the CELEB-A experiment took approximately 100 hours).
- Lacks direct comparison with recent causal fairness methods (e.g., counterfactual fairness).
- Assumes that the bias mechanism can be correctly modeled, but real-world sources of bias may be more complex than predefined templates.

## Related Work & Insights

- **Fairness Constraint Methods**: Preprocessing (Feldman 2015), in-processing constraints (Kamishima 2011), and post-processing (Hardt 2016), which, however, rely on fixed fairness metrics.
- **Causal Fairness**: Kilbertus 2017 uses causal diagrams to identify discriminatory paths, and Madras 2018 models decisions and confounding variables; ours is more flexible but does not require a complete causal graph.
- **Neurosymbolic Fairness**: Varley 2021 uses SPNs to learn dependencies, Choi 2020 designs probabilistic circuits satisfying independence assumptions, and Wagner 2021 uses LTNs to actively learn constraints.
- **PU Learning and ProbLog**: Verreet 2024 expresses the annotation mechanism of PU learning as a ProbLog template, and ours generalizes this concept to the fairness domain.

## Rating

⭐⭐⭐⭐ (4/5)

Innovatively applies probabilistic logic programming to fairness issues, with an elegantly designed and highly interpretable framework. Experiments comprehensively cover both synthetic and real-world data, with thorough validation across the three bias types. Points are deducted primarily due to scalability issues (high computational overhead, limitation in multi-class scenarios) and the lack of comparison with recent causal fairness methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach](rethinking_target_label_conditioning_in_adversarial_attacks_a_2d_tensor-guided_g.md)
- [\[AAAI 2026\] Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack](privacy_auditing_of_multi-domain_graph_pre-trained_model_under_membership_infere.md)
- [\[AAAI 2026\] Learning to Collaborate: An Orchestrated-Decentralized Framework for Peer-to-Peer Collaborative Learning](learning_to_collaborate_an_orchestrated-decentralized_framework_for_peer-to-peer.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](robust_watermarking_on_gradient_boosting_decision_trees.md)
- [\[ICML 2025\] Retraining with Predicted Hard Labels Provably Increases Model Accuracy](../../ICML2025/ai_safety/retraining_with_predicted_hard_labels_provably_increases_model_accuracy.md)

</div>

<!-- RELATED:END -->
