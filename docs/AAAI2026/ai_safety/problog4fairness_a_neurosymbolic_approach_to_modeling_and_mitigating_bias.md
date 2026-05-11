---
title: >-
  [Paper Note] ProbLog4Fairness: A Neurosymbolic Approach to Modeling and Mitigating Bias
description: >-
  [AI Safety] This paper proposes the ProbLog4Fairness framework, which formalizes bias mechanisms in data as interpretable logic programs using the probabilistic logic programming language ProbLog, and integrates bias assumptions into neural network training via distant supervision in DeepProbLog, enabling flexible and principled bias mitigation.
tags:
  - AI Safety
date: 2026-05-08
content_hash: 243d89fc1c7c9cab
---

# ProbLog4Fairness: A Neurosymbolic Approach to Modeling and Mitigating Bias

- **Conference**: AAAI 2026
- **arXiv**: [2511.09768](https://arxiv.org/abs/2511.09768)
- **Code**: Not provided
- **Area**: AI Safety
- **Keywords**: Algorithmic Fairness, Neurosymbolic AI, Probabilistic Logic Programming, Bias Modeling, DeepProbLog

## TL;DR

This paper proposes the ProbLog4Fairness framework, which formalizes bias mechanisms in data as interpretable logic programs using the probabilistic logic programming language ProbLog, and integrates bias assumptions into neural network training via distant supervision in DeepProbLog, enabling flexible and principled bias mitigation.

## Background & Motivation

- The algorithmic fairness literature contains a large number of mutually incompatible fairness definitions (e.g., statistical parity, equalized odds), and the normative choice among these constraints remains contested.
- Existing methods typically optimize for a fixed bias type or fairness metric, lacking the ability to flexibly model different sources of bias for specific tasks.
- Multiple bias mechanisms in the data-generating process—label bias, measurement bias, and historical bias—are important sources of model unfairness.
- Causal models can describe bias mechanisms but are difficult to interpret and integrate into data pipelines in practice.
- **Core Motivation**: Rather than selecting a fixed fairness constraint, this work directly models and corrects bias mechanisms in the data, allowing the model to produce fairer decisions "naturally."

## Method

### 1. ProbLog Bias Modeling Framework

The core idea is to encode bias assumptions as ProbLog probabilistic logic programs. ProbLog is a probabilistic extension of Prolog that supports probabilistic facts $p::f$, meaning fact $f$ holds with probability $p$.

**Bias Represented as Probabilistic Facts**: For binary variables, a bias transformation is fully defined by four probabilistic facts describing:
- $p_1$: Negative bias for the sensitive group (flipping 1 to 0)
- $p_2$: Negative bias for the non-sensitive group
- $p_3$: Positive bias for the sensitive group (flipping 0 to 1)
- $p_4$: Positive bias for the non-sensitive group

Using label bias as an example, the ProbLog template is:

$$\tilde{y}(\mathbf{X}) \leftarrow y_h(\mathbf{X}) \wedge \neg \text{label\_neg\_bias}(\mathbf{X})$$
$$\tilde{y}(\mathbf{X}) \leftarrow \neg y_h(\mathbf{X}) \wedge \text{label\_pos\_bias}(\mathbf{X})$$

where $y_h(\mathbf{X})$ is the unbiased label predicted by the classifier and $\tilde{y}(\mathbf{X})$ is the observed biased label.

### 2. DeepProbLog Integration and Distant Supervision Training

DeepProbLog allows neural networks to predict parameters of probabilistic facts, enabling logical reasoning on top of neural predictions. The training pipeline proceeds as follows:

1. **Define the Logic Program**: Includes the classifier $h(\mathbf{X})::y_h(\mathbf{X})$ and ProbLog rules encoding bias mechanisms.
2. **Compile to Circuit**: The logic program is compiled into a circuit for computing query probabilities.
3. **Distant Supervision**: Supervision uses only the biased label $\tilde{y}$; gradients are back-propagated through the logic circuit to update the network.
4. **Inference-Time Stripping**: At test time, if unbiased features are available, prediction uses $h(\mathbf{X})$ directly; if only biased features are available, the bias transformation mechanism is retained.

A key advantage is that gradient updates account for all unbiased explanations consistent with the observed biased data.

### 3. Modeling Three Types of Bias

**Label Bias**: Unbiased features are available, but the observed label $\tilde{Y}$ is a biased proxy for the true label $Y$—for example, annotator discrimination against a specific ethnic group in loan approval. Modeled during training using Template 1.

**Measurement Bias**: The observed feature $\tilde{X}_i$ is a noisy proxy for the unbiased feature $X_i$, and the label depends only on the unbiased feature—for example, using "number of working days in the past three years" to measure job stability, which disadvantages women on maternity leave. The debiasing process is modeled using Template 2:

$$y(\tilde{\mathbf{X}}) \leftarrow \text{debias}(\tilde{\mathbf{X}}, \mathbf{X}) \wedge y_h(\mathbf{X})$$

**Historical Bias**: Both features and labels are affected by bias, with labels generated from biased features. The method assumes that the mapping from biased features to biased labels mirrors that from unbiased features to unbiased labels, applying Template 2 for debiasing at test time.

### 4. Parameter Specification

The bias probability parameters $p_i$ can be specified in the following ways:
- Direct assignment based on domain knowledge
- Estimation from a small subset of data with both biased and unbiased labels
- Based on Hoeffding's inequality, only 184 samples are needed to estimate parameters within 10% error at 95% confidence

## Key Experimental Results

### Synthetic Data Experiments

| Experiment | Bias Type | Key Finding |
|---|---|---|
| RQ1: Comparison across bias types | Label / Measurement / Historical | ProbLog4Fairness approaches the upper-bound baseline in both accuracy and statistical disparity, significantly outperforming other mitigation methods |
| RQ2: $A \not\perp Y$ scenario | Label bias | Removes only problematic bias while preserving legitimate correlations; other baselines erroneously force statistical disparity to zero |
| RQ3: Parameter sensitivity | Label bias | Optimal accuracy is achieved at the correct parameter values, and results are robust to parameter estimation errors |

### Real-World Data Experiments

| Dataset | Type | Bias Source | Sensitive Variable | ProbLog4Fairness Results |
|---|---|---|---|---|
| Student Alcohol | Tabular (856 samples) | Annotator bias (subjective bias against male students) | Gender | F1 score outperforms all mitigation baselines; statistical disparity approaches the unbiased data level |
| CELEB-A | Image data | Annotation inconsistency (Mouth Slightly Open attribute) | Smiling / Blurry / High Cheekbones | Simultaneously correcting three sensitive attributes yields significant improvements in both F1 and statistical disparity, outperforming single-attribute correction |

**Baselines** include: Lower (trained directly on biased data), Upper (trained on unbiased data), Unawareness (sensitive variable removed), Massaging (data preprocessing), and Error Parity (post-processing method).

## Key Findings

1. Due to its flexible modeling capability, ProbLog4Fairness simultaneously approaches the ideal upper bound in both accuracy and fairness—an achievement beyond the reach of methods relying on fixed assumptions.
2. When $A \not\perp Y$ (sensitive variable and label are not independent), the proposed method distinguishes problematic bias from legitimate correlation, whereas other methods over-correct.
3. Parameter estimation has limited impact on results; a small amount of unbiased data suffices for effective parameter specification.
4. In the CELEB-A experiments, jointly modeling bias across multiple sensitive attributes substantially outperforms single-attribute correction, highlighting the importance of flexible modeling.
5. Simplified assumptions (e.g., "no positive bias") can sometimes improve performance on real data by reducing parameter estimation error.

## Highlights & Insights

- **Principled Flexibility**: ProbLog's declarative programming paradigm allows users to flexibly add or modify bias assumptions for specific scenarios without relying on fixed fairness definitions.
- **Strong Interpretability**: Bias assumptions are expressed as logical rules that domain experts can directly understand and validate.
- **Unified Framework for Multiple Bias Types**: Label, measurement, and historical bias can all be jointly modeled within the same framework.
- **Generalization from Tabular to Image Data**: The successful integration of ResNet-50 with ProbLog on CELEB-A demonstrates effectiveness on high-dimensional, non-tabular data.

## Limitations & Future Work

- Bias parameters must be specified in advance or estimated from a small amount of unbiased data; jointly learning the parameters renders the classifier non-identifiable.
- The current framework handles only binary sensitive variables and binary labels; extension to multi-class settings requires a substantially larger number of probabilistic facts.
- Synthetic experiments are limited to binary and categorical features; handling of continuous features is not thoroughly discussed.
- Compilation and inference in DeepProbLog are computationally expensive for complex programs (approximately 100 hours for the CELEB-A experiments).
- No direct comparison is made with recent causal fairness methods (e.g., counterfactual fairness).
- The framework assumes that bias mechanisms can be correctly modeled, whereas real-world bias sources may be more complex than the predefined templates.

## Related Work & Insights

- **Fairness Constraint Methods**: Preprocessing (Feldman 2015), in-processing constraints (Kamishima 2011), and post-processing (Hardt 2016) all rely on fixed fairness metrics.
- **Causal Fairness**: Kilbertus 2017 uses causal graphs to identify discriminatory pathways; Madras 2018 models decisions and confounders. The proposed method is more flexible and does not require a complete causal graph.
- **Neurosymbolic Fairness**: Varley 2021 uses SPNs to learn dependencies; Choi 2020 designs probabilistic circuits satisfying independence assumptions; Wagner 2021 uses LTN for active learning of constraints.
- **PU Learning and ProbLog**: Verreet 2024 expresses the labeling mechanism in PU Learning as ProbLog templates; this work extends that idea to the fairness domain.

## Rating

⭐⭐⭐⭐ (4/5)

The paper innovatively applies probabilistic logic programming to fairness problems, with an elegant framework design and strong interpretability. Experiments comprehensively cover both synthetic and real-world data, with thorough validation across three bias types. Points are deducted primarily for scalability limitations (high computational cost, restricted multi-class support) and the absence of comparisons with recent causal fairness methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach](rethinking_target_label_conditioning_in_adversarial_attacks_a_2d_tensor-guided_g.md)
- [\[AAAI 2026\] Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack](privacy_auditing_of_multi-domain_graph_pre-trained_model_under_membership_infere.md)
- [\[AAAI 2026\] Learning to Collaborate: An Orchestrated-Decentralized Framework for Peer-to-Peer Collaborative Learning](learning_to_collaborate_an_orchestrated-decentralized_framework_for_peer-to-peer.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](robust_watermarking_on_gradient_boosting_decision_trees.md)
- [\[ICLR 2026\] Skirting Additive Error Barriers for Private Turnstile Streams](../../ICLR2026/ai_safety/skirting_additive_error_barriers_for_private_turnstile_streams.md)

</div>

<!-- RELATED:END -->
