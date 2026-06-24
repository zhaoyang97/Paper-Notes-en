---
title: >-
  [Paper Note] Measuring the (Un)Faithfulness of Concept-Based Explanations
description: >-
  [CVPR 2026][Interpretability][Concept explanations] This paper reveals that the faithfulness of existing unsupervised concept-based explanation methods (U-CBEMs) is overestimated due to overly complex surrogate models and flawed deletion-based evaluations. The authors propose SURF (Surrogate Faithfulness), a framework consisting of a simple linear surrogate and dual-space metrics. Validated by a sanity check ("random concepts should be less faithful")…
tags:
  - "CVPR 2026"
  - "Interpretability"
  - "Concept explanations"
  - "faithfulness metrics"
  - "unsupervised concept methods"
  - "surrogate models"
  - "interpretability evaluation"
date: 2026-05-08
content_hash: 93794b7038be9453
---

# Measuring the (Un)Faithfulness of Concept-Based Explanations

**Conference**: CVPR 2026  
**arXiv**: [2504.10833](https://arxiv.org/abs/2504.10833)  
**Code**: Yes (The paper states that the code has been released)  
**Area**: Explainable AI / Model Interpretability  
**Keywords**: Concept explanations, faithfulness metrics, unsupervised concept methods, surrogate models, interpretability evaluation

## TL;DR

This paper reveals that the faithfulness of existing unsupervised concept-based explanation methods (U-CBEMs) is overestimated due to overly complex surrogate models and flawed deletion-based evaluations. The authors propose SURF (Surrogate Faithfulness), a framework consisting of a simple linear surrogate and dual-space metrics. Validated by a sanity check ("random concepts should be less faithful"), this framework demonstrates its correctness and reveals for the first time that several state-of-the-art (SOTA) U-CBEMs are actually unfaithful.

## Background & Motivation

1. **Background**: Deep vision models are difficult to interpret. Concept-based explanation methods (CBEMs) improve interpretability by decomposing intermediate representations into human-understandable semantic concepts (e.g., edges, colors, object parts). Unsupervised CBEMs (U-CBEMs) automatically discover concept activation vectors (CAVs) and their importance scores, avoiding the need for manual concept labeling.

2. **Limitations of Prior Work**: The core evaluation metric for U-CBEMs is "faithfulness"—whether the explanation truly reflects the model's internal computation. However, existing evaluations suffer from two systemic issues: (a) **Overly complex surrogate models**—ICE-Eval reconstructs embeddings via NMF before passing them through the original model, while C-SHAP-Eval trains an additional MLP. These complex surrogates make U-CBEMs "appear" faithful, even if the explanations themselves do not clearly lead to the model output; (b) **Unreliable deletion-based evaluation**—faithfulness is indirectly inferred by removing concepts and observing performance drops. However, inputs after deletion may deviate from the data manifold, leading to unpredictable model behavior.

3. **Key Challenge**: There is an inherent contradiction between interpretability and faithfulness—making an explanation simple and understandable (interpretable) inevitably loses information (reducing faithfulness). Past evaluation methods, by allowing complex surrogates and single-class metrics, artificially allowed U-CBEMs to "show" both high interpretability and high faithfulness, whereas the explanations could not actually derive the model output clearly.

4. **Goal**: (a) Unify the fragmented faithfulness evaluation frameworks; (b) Design a faithfulness metric that satisfies three criteria (simple surrogate, inclusion of concept importance, and full-output measurement); (c) Conduct the first fair faithfulness benchmarking of existing U-CBEMs.

5. **Key Insight**: The authors propose an extremely concise sanity check—if concepts are replaced with random vectors, the faithfulness score should decrease. Shockingly, both existing ICE-Eval and C-SHAP-Eval fail this check.

6. **Core Idea**: Use the model's own final linear layer structure as a surrogate template and propose the zero-parameter linear surrogate SURF. Combined with dual metrics in logit space (MAE) and probability space (EMD), it achieves accurate and reliable evaluation of U-CBEM faithfulness.

## Method

### Overall Architecture

SURF is a metric framework for evaluating the faithfulness of U-CBEM explanations rather than a new concept discovery method. Given an explanation produced by any U-CBEM (a set of CAVs $V_i$ and corresponding importance scores $A_i$), SURF addresses a core question: How accurately can a naive linear combination of these concepts and importance scores reconstruct the model's true output? The workflow uses a zero-parameter linear surrogate to map concept representations directly back to the model's output space, then measures the deviation between the surrogate output and the true model output in both logit and probability spaces. The process introduces no trainable parameters and avoids embedding reconstruction, resulting in extremely low computational cost (200 FLOPs vs. 205M FLOPs for C-SHAP-Eval), ensuring that "surrogate complexity" does not pollute the faithfulness assessment.

### Key Designs

**1. Unified Faithfulness Framework: Enabling Horizontal Comparison**

Previously, each U-CBEM paper utilized its own faithfulness metric, making results incomparable. SURF points out that any faithfulness metric essentially consists of three components: a metric $d$ (how to compare outputs), a surrogate $s$ (how to derive output from the explanation), and a concept projection $\mathcal{P}$ (how to map embeddings into the concept space). In this view, deletion-based methods remove concepts in a "deletion space" (pixels/weights/concepts) and observe performance degradation; surrogate-based methods directly approximate faithfulness integrals (Eq. 3). By fitting all methods into the same $(d, s, \mathcal{P})$ template, systemic differences (e.g., surrogate complexity, use of importance scores) become clear, enabling fair evaluation.

**2. Zero-Parameter Linear Surrogate: Using the Final Linear Layer as a "Perfect Explanation" Template**

The problem with complex surrogates is that they hide the complexity of the explanation—humans still do not know how concepts lead to a prediction after viewing the explanation. SURF's starting point is the observation that the computation of the final linear layer $y_i = \sum_j \mathbf{h}_j^T \mathbf{f}_{i,j}$ can be decomposed as $y_i = \sum_j \mathbf{h}_j^T \mathbf{v}_{i,j}\,\alpha_{i,j}$, where $\mathbf{v}_{i,j}$ is the normalized direction (the CAV) and $\alpha_{i,j}$ is its norm (the importance). Thus, the final linear layer defines an "ideal explanation" form. SURF's surrogate mimics this structure, replacing weights with U-CBEM discovered CAVs and importance:

$$\hat{y}_i = \sum_j \sum_k \alpha_{i,k}\,\mathcal{P}(\mathbf{h}_j; V_i)_k$$

This surrogate is zero-parameter and non-reconstructive, testing the mental step a human interpreter would perform: "Can I get the model output by linearly stacking these concepts by their importance?" If not, the concepts/importance in the explanation are not aligned with the model's actual computation.

**3. Three Desiderata: Hard Constraints for a Good Faithfulness Metric**

SURF uses these criteria to judge legacy metrics: (1) **The surrogate should be as simple as possible**—complex surrogates mask explanation complexity and make faithfulness scores untrustworthy; (2) **All components of the explanation must be used**, especially concept importance $A_i$—if a surrogate ignores $A_i$, shuffling importance would not affect the score, making the metric blind to importance correctness; (3) **Measure error across all output classes**, rather than just the predicted or ground-truth class—single-class metrics miss significant deviations in other classes. SURF satisfies all three, while existing metrics violate at least one: ICE-Eval is reconstruction-based and ignores $A_i$, and C-SHAP-Eval introduces a trainable MLP and ignores $A_i$.

**4. SURF Dual Metrics: Complementary Logit and Probability Spaces**

Instead of a single scalar, SURF provides metrics in two complementary spaces. Logit space uses Mean Absolute Error (MAE) to measure raw output precision:

$$\text{SURF}_{\text{MAE}} = \frac{1}{|\mathcal{V}|\,C} \sum_{\mathbf{x}} \sum_i \left| y_i - \hat{y}_i \right|$$

Probability space uses an EMD-like distance to measure distribution deviation after softmax:

$$\text{SURF}_{\text{EMD}} = \frac{1}{2|\mathcal{V}|} \sum_{\mathbf{x}} \sum_i \left| p_i - \hat{p}_i \right|$$

Logits are unaffected by normalization but have uncontrollable ranges, while probabilities are normalized but have the predicted class amplified by softmax. Used together, low $\text{SURF}_{\text{MAE}}$ ensures logit-level precision, and low $\text{SURF}_{\text{EMD}}$ ensures the entire distribution matches, avoiding the overestimation of faithfulness common in Top-1 Accuracy.

### Loss & Training

SURF is a pure evaluation metric and requires no training. The evaluated U-CBEMs are run according to their original papers, typically discovering 5 concepts per output class.

## Key Experimental Results

### Measure-over-Measure Comparison (Sanity Check)

| Setting | SURF_MAE ↓ | SURF_EMD ↓ | Top-1 ↑ | C-SHAP Top-1 ↑ | ICE Top-1 ↑ |
|------|------------|------------|---------|-----------------|-------------|
| Perfect | 0.00 | 0.000 | 100% | 9.02% | 100% |
| Rand Imp | 2.70 | 0.862 | 97.5% | 9.02% | 100% |
| Full Rand | 3.17 | 0.883 | 1.3% | **97.6%** | 3.3% |

Key Findings: **C-SHAP-Eval unexpectedly reports 97.6% accuracy for a completely random explanation (Full Rand)**, higher than the Perfect setting. ICE-Eval still reports 100% faithfulness under random importance (Rand Imp). Only SURF behaves correctly across all settings.

### U-CBEM Benchmarking (Object Classification, ResNet-50)

| U-CBEM | SURF_MAE ↓ | SURF_EMD ↓ | Top-1 ↑ | Rank Corr ↑ |
|--------|------------|------------|---------|-------------|
| CDISCO | 3.40 | 0.932 | 0.2% | 0.002 |
| ICE | 3.33 | 0.628 | 98.9% | 0.093 |
| CRAFT | 3.19 | 0.878 | 90.6% | 0.068 |
| C-SHAP | 3.28 | 0.882 | 6.3% | 0.005 |
| MCD | 2.60 | 0.426 | 99.4% | 0.145 |
| HU-MCD | 1.97 | 0.384 | 99.7% | 0.149 |
| SAE | **1.04** | **0.195** | 99.2% | **0.366** |

### Key Findings

- **No existing U-CBEM is truly faithful**—Even for the best-performing SAE, the $\text{SURF}_{\text{EMD}}$ reaches 0.195, indicating significant probability distribution deviation. Most methods have $\text{SURF}_{\text{EMD}}$ between 0.4 and 0.93, nearly as poor as random.
- **Top-1 Accuracy is a misleading metric**—While ICE and MCD achieve Top-1 scores of 98-99%, SURF_EMD and Rank Corr reveal massive errors in non-predicted classes. Relying only on Top-1 severely overestimates faithfulness.
- **SAE is the most faithful across all tasks**—Whether in classification, multi-attribute prediction, or age regression, SAE performs best, likely due to its advantages in the completeness criterion.
- **Increasing the number of concepts does not necessarily improve faithfulness**—For CDISCO, CRAFT, C-SHAP, and ICE, faithfulness remains stagnant or decreases as concept counts increase. Only MCD/HU-MCD show monotonic improvement with a natural saturation point.

## Highlights & Insights

- **The sanity check "random concepts should be less faithful" is simple yet powerful**—This intuitive test strikes at the core flaws of previous metrics. Such simple, definitive tests are highly valuable in the XAI field.
- **SURF's zero-parameter design is a key innovation**—By utilizing the structure of the final linear layer, it avoids additional parameters or reconstruction. The 200 FLOPs vs. 205M FLOPs gap is not just about efficiency but addresses the fundamental issue of surrogate complexity polluting faithfulness evaluation.
- **The dual-metric design covers blind spots**—Top-1 only considers the maximum class, Norm L1 focuses on the ground truth, and SURF_MAE ignores relative class importance. SURF_EMD complements these by evaluating the entire probability domain.

## Limitations & Future Work

- **SURF currently only applies to the final linear layer**—Explanations of intermediate layers cannot be evaluated because the mapping from intermediate layers to output is non-linear. Extending SURF to intermediate layers is a critical area for future work.
- **Faithfulness is evaluated without joint interpretability assessment**—A method with low faithfulness but high interpretability may still have practical value. An ideal framework should quantify the faithfulness-interpretability trade-off.
- **U-CBEMs found only 5 concepts per class**—This setting might be disadvantageous for certain methods. While the paper shows trends with varying concept counts, it was only extensively tested on a single task.
- **Predominantly focused on classification tasks**—Only one regression task (age estimation) was included; generalization to other tasks (e.g., segmentation, generation) has not been verified.

## Related Work & Insights

- **vs. ICE / ICE-Eval**: ICE uses NMF to find concepts and reconstructs embeddings for evaluation. SURF reveals that ICE-Eval ignores concept importance, yielding identical scores for random and perfect importance—a fundamental flaw.
- **vs. C-SHAP / C-SHAP-Eval**: C-SHAP uses Shapley values for importance and an MLP surrogate. SURF reveals the MLP sometimes performs better on random concepts, likely because it learns a concept-independent mapping from embeddings to output.
- **vs. CRAFT**: CRAFT uses NMF for recursive decomposition and concept-space deletion for evaluation. SURF's surrogate approach avoids the out-of-distribution issues of deletion.
- **vs. SAE (Sparse Autoencoders)**: SAE performs best under SURF, suggesting it may become the mainstream direction for explaining vision models.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic revelation of flaws in U-CBEM faithfulness evaluation; SURF design is elegant and powerful.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three tasks, seven U-CBEMs, and comprehensive measure-over-measure comparison, though intermediate layer evaluation is missing.
- Writing Quality: ⭐⭐⭐⭐⭐ Highly rigorous logic, from framework unification to desiderata, sanity checks, and benchmarking.
- Value: ⭐⭐⭐⭐⭐ Significant impact on the XAI community; likely to change evaluation standards in the U-CBEM field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] An Analysis of Concept Bottleneck Models: Measuring, Understanding, and Mitigating the Impact of Noisy Annotations](../../NeurIPS2025/interpretability/an_analysis_of_concept_bottleneck_models_measuring_understanding_and_mitigating_.md)
- [\[ICML 2025\] What Makes an Ensemble (Un)interpretable?](../../ICML2025/interpretability/what_makes_an_ensemble_un_interpretable.md)
- [\[ICLR 2026\] Faithfulness Under the Distribution: A New Look at Attribution Evaluation](../../ICLR2026/interpretability/faithfulness_under_the_distribution_a_new_look_at_attribution_evaluation.md)
- [\[CVPR 2026\] Back to the Feature: Explaining Video Classifiers with Video Counterfactual Explanations](back_to_the_feature_explaining_video_classifiers_with_video_counterfactual_expla.md)
- [\[CVPR 2026\] Rethinking Concept Bottleneck Models: From Pitfalls to Solutions](rethinking_concept_bottleneck_models_from_pitfalls_to_solutions.md)

</div>

<!-- RELATED:END -->
