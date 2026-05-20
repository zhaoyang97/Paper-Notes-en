---
title: >-
  [Paper Note] From Black-box to Causal-box: Towards Building More Interpretable Models
description: >-
  [NeurIPS 2025][Causal Inference][causal interpretability] This paper proposes a formal definition of *causal interpretability*, proves that both black-box models and concept bottleneck models fail to satisfy this propert…
tags:
  - "NeurIPS 2025"
  - "Causal Inference"
  - "causal interpretability"
  - "counterfactual reasoning"
  - "concept-based models"
  - "structural causal models"
  - "interpretability-accuracy tradeoff"
date: 2026-05-08
content_hash: 90190c45124e03a0
---

# From Black-box to Causal-box: Towards Building More Interpretable Models

**Conference**: NeurIPS 2025
**arXiv**: [2510.21998](https://arxiv.org/abs/2510.21998)  
**Code**: To be confirmed  
**Area**: Causal Inference / Explainable AI
**Keywords**: causal interpretability, counterfactual reasoning, concept-based models, structural causal models, interpretability-accuracy tradeoff

## TL;DR
This paper proposes a formal definition of *causal interpretability*, proves that both black-box models and concept bottleneck models fail to satisfy this property, establishes a complete graphical criterion for identifying which model architectures can consistently answer counterfactual queries, and reveals a fundamental tradeoff between causal interpretability and predictive accuracy.

## Background & Motivation
**Background**: Explainable AI (XAI) methods such as LIME, SHAP, and Grad-CAM provide feature attributions, while concept bottleneck models (CBMs) improve interpretability by making predictions through human-understandable concepts (e.g., "smiling," "gender"). Counterfactual reasoning is widely regarded as a key pathway toward truly interpretable models.

**Limitations of Prior Work**: Existing approaches cannot guarantee *counterfactual consistency*—distinct models within the same model class, though perfectly consistent on observational data, may yield contradictory answers to the same counterfactual query. For instance, two CBMs may both predict a face as "unattractive," yet give opposite answers to "Would this person be more attractive if they smiled?"

**Key Challenge**: Users cannot determine which model's counterfactual response is trustworthy, since the two models are observationally indistinguishable. This is fundamentally because the model class imposes insufficient constraints on the counterfactual response space.

**Goal**: (1) Formalize the conditions under which a model can reliably answer counterfactual queries; (2) identify which model architectures satisfy these conditions; (3) maximize predictive capacity subject to satisfying these conditions.

**Key Insight**: The paper builds on structural causal models (SCMs), introducing augmented SCMs (ASCMs) that jointly describe the data-generating process and model predictions, and leverages graphical structure to analyze counterfactual identifiability.

**Core Idea**: Whether a model can consistently answer counterfactual queries depends on whether its prediction feature set $\mathbf{T}$ lies within the intervention target $\mathbf{W}$ and its non-descendants $ND(\mathbf{W})$.

## Method

### Overall Architecture
The paper develops a theoretical framework proceeding as follows: define causal interpretability → prove that black-box models and CBMs do not satisfy it → propose a graphical criterion → identify the maximal admissible feature set → derive a closed-form computation for counterfactuals. The input is the descendant relationships among variables in the causal graph; the output specifies which features a model should use to maintain causal interpretability.

### Key Designs

1. **Augmented Structural Causal Model (ASCM, Definition 1)**

    - *Function*: Jointly models the image-generation process and model predictions.
    - *Mechanism*: Extends a generative-level SCM $\mathcal{M}_0$ (encoding causal relationships among latent concepts $\mathbf{V}$) with an image-generation mechanism $\mathbf{X} \leftarrow f_\mathbf{X}(\mathbf{V}, \mathbf{U_X})$ and a classifier $\hat{Y} \leftarrow f_{\hat{Y}}(\text{subset of } \{\mathbf{V}, \mathbf{X}\})$.
    - *Design Motivation*: Incorporating "how the model predicts" into the causal framework enables counterfactual analysis to simultaneously account for data generation and model behavior.

2. **Definition of Causal Interpretability (Definition 2)**

    - *Function*: Formalizes the condition under which a model class can consistently answer counterfactual queries.
    - *Core Definition*: A model class $\Omega'$ is causally interpretable with respect to query $Q$ if and only if, for all $\mathcal{M}_1, \mathcal{M}_2 \in \Omega'$, agreement on the observational distribution ($P^{\mathcal{M}_1}(\mathbf{V}, \mathbf{X}, \hat{Y}) = P^{\mathcal{M}_2}(\mathbf{V}, \mathbf{X}, \hat{Y})$) implies agreement on the counterfactual quantity ($Q^{\mathcal{M}_1} = Q^{\mathcal{M}_2}$).
    - *Design Motivation*: Directly captures the core requirement of counterfactual reliability—models that are observationally indistinguishable within the same class should also be counterfactually indistinguishable.

3. **Non-Interpretability Results (Proposition 1 + Example 4)**

    - Black-box models never satisfy causal interpretability (Proposition 1): since $\hat{Y}$ depends on $\mathbf{X}$, which is a descendant of all variables.
    - CBMs do not necessarily satisfy causal interpretability (Example 4): when the concepts used for prediction include descendants of the intervention target, different models may still yield divergent counterfactual answers.
    - *Design Motivation*: Dispels the misconception that concept bottleneck models are inherently interpretable.

4. **Graphical Criterion (Theorem 1) and Maximal Admissible Set (Theorem 2)**

    - *Function*: Identifies which feature sets $\mathbf{T}$ render a model causally interpretable.
    - *Core Result*: $\Omega_{GCP(\mathbf{T})}$ is causally interpretable with respect to $Q(\mathbf{W})$ **if and only if** $\mathbf{T} \subseteq \mathbf{W} \cup ND(\mathbf{W})$ (i.e., $\mathbf{T}$ may only contain the intervention target and its non-descendants).
    - The maximal admissible set is unique: $\text{Max-T-Ad}(\mathbf{W}_\star) = \cap_{\mathbf{W}_i \in \mathbf{W}_\star} (\mathbf{W}_i \cup ND(\mathbf{W}_i))$.
    - *Design Motivation*: Maximizing the admissible feature set maximizes predictive accuracy while preserving causal interpretability. Only the descendant relationships of the intervention target need be known; a complete causal graph is not required.

5. **Closed-Form Counterfactual Computation (Theorem 3)**

    - $P(\hat{Y}_{\mathbf{w}'} | \mathbf{x}) = \sum_\mathbf{t} P(\hat{Y} | \mathbf{w}' \cap \mathbf{T}, \mathbf{t} \setminus \mathbf{W}) P(\mathbf{t} | \mathbf{x})$
    - Computation from data requires two steps: a feature extractor $P(\mathbf{T}|\mathbf{X})$ and a classifier $P(\hat{Y}|\mathbf{T})$, with the intervention target substituted accordingly.

6. **Interpretability–Accuracy Tradeoff (Theorem 4)**

    - More features yield higher predictive accuracy but reduce the set of answerable counterfactual queries.
    - Answering a broader range of counterfactual queries requires fewer admissible features, which in turn reduces predictive accuracy.
    - This constitutes a fundamental information-theoretic tradeoff.

### Loss & Training
This is a theoretical contribution; no specific loss function is proposed. Experiments employ standard classification training for GCP models.

## Key Experimental Results

### Synthetic Experiments (BarMNIST)
A custom BarMNIST dataset is constructed with features B (bar), D (digit), and C (color), where $D \rightarrow B$ holds causally.

| Feature Set $\mathbf{T}$ | Predictive Accuracy | Counterfactual Error for $Q(D)$ | Causally Interpretable? |
|---|---|---|---|
| {B,D,C} | **Highest** | **High** | No (B is a descendant of D) |
| {B,D} | Second highest | High | No |
| {D,C} | Medium | **Low** | **Yes** (unique maximal admissible set) |
| {D} | Lowest | Low | Yes |

### Real-World Experiments (CelebA)
Predicting attractiveness under the counterfactual query "Would this person be more attractive if they smiled?"
- Models using the non-descendant feature set {smiling, gender} correctly predict that smiling increases attractiveness, consistent with human intuition.
- Counterfactuals are computed directly from observational data using the closed-form formula in Theorem 3.

### Key Findings
- The graphical criterion of Theorem 1 is perfectly validated empirically: models violating the condition consistently produce unreliable counterfactuals.
- The maximal admissible set {D,C} achieves the optimal balance between accuracy and interpretability.
- The accuracy–interpretability tradeoff is tangible: {B,D,C} achieves the highest accuracy but yields unreliable counterfactuals.

## Highlights & Insights
- The finding that **concept bottleneck models are not inherently interpretable** is highly counterintuitive—it is widely assumed that prediction via concepts constitutes interpretability, yet this paper rigorously demonstrates counterfactual inconsistency at the formal level.
- The **graphical criterion as a necessary and sufficient condition** is remarkably concise: excluding descendants of the intervention target suffices. In practice, only limited causal knowledge (i.e., descendant relationships) is needed, not a complete causal graph.
- The existence of a **unique maximal admissible set** is a strong result—no selection among multiple candidates is required; the optimal solution is uniquely determined.
- **Practical value**: The framework offers principled theoretical guidance for constructing genuinely reliable interpretable AI systems—not all concepts should be used; the feature subset must be chosen according to the counterfactual queries of interest.

## Limitations & Future Work
- The framework assumes that the descendant relationships in the causal graph are known; in practice, these may need to be learned from data.
- The current work addresses only counterfactual predictions in classification tasks; extensions to more complex causal queries (e.g., natural direct and indirect effects) remain to be explored.
- Experimental scale is limited (BarMNIST + CelebA); validation on large-scale, complex datasets is insufficient.
- The accuracy of the feature extractor $P(\mathbf{T}|\mathbf{X})$ directly affects counterfactual estimation quality; error analysis for this component is not thoroughly addressed.

## Related Work & Insights
- **vs. LIME/SHAP**: These methods provide feature attributions but cannot answer counterfactual queries; this paper defines interpretability at a higher level of rigor.
- **vs. Concept Bottleneck Models (CBM)**: CBMs assume that prediction via concepts constitutes interpretability; this paper demonstrates that causal structure must additionally be taken into account.
- **vs. Causal Fairness**: The proposed framework is closely related to the notion of counterfactual fairness in algorithmic fairness, and can be applied to assess model dependence on sensitive features.
- **vs. Pearl's Causal Hierarchy**: This work operates at Level 3 of the hierarchy (counterfactual reasoning), representing a deep connection between XAI and causal inference.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First formal definition of causal interpretability with a complete theoretical framework.
- Experimental Thoroughness: ⭐⭐⭐ Limited experiments for a theory paper, but the synthetic and CelebA results validate the key claims.
- Writing Quality: ⭐⭐⭐⭐⭐ The definition–proposition–theorem logical chain is exceptionally clear, with intuitive examples throughout.
- Value: ⭐⭐⭐⭐⭐ Establishes a causal-theoretic foundation for explainable AI and provides actionable guidance for model design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Hallucinate Less by Thinking More: Aspect-Based Causal Abstention for Large Language Models](../../AAAI2026/causal_inference/hallucinate_less_by_thinking_more_aspect-based_causal_absten.md)
- [\[NeurIPS 2025\] Revealing Multimodal Causality with Large Language Models](revealing_multimodal_causality_with_large_language_models.md)
- [\[NeurIPS 2025\] Practical do-Shapley Explanations with Estimand-Agnostic Causal Inference](practical_do-shapley_explanations_with_estimand-agnostic_causal_inference.md)
- [\[NeurIPS 2025\] Counterfactual Reasoning for Steerable Pluralistic Value Alignment of Large Language Models](counterfactual_reasoning_for_steerable_pluralistic_value_alignment_of_large_lang.md)
- [\[NeurIPS 2025\] Conformal Prediction for Causal Effects of Continuous Treatments](conformal_prediction_for_causal_effects_of_continuous_treatments.md)

</div>

<!-- RELATED:END -->
