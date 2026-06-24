---
title: >-
  [Paper Note] Operational Open-Set Recognition and PostMax Refinement
description: >-
  [ECCV 2024][AI Safety][Open-Set Recognition] This paper proposes OOSA (Operational Open-Set Accuracy), an evaluation metric for practical deployment scenarios, and PostMax, a post-processing algorithm. By normalizing the maximum class logit with deep feature magnitude and mapping it through a Generalized Pareto Distribution (GPD), logits are converted into reasonable probability estimates, achieving statistically significant SOTA performance in large-scale evaluations.
tags:
  - "ECCV 2024"
  - "AI Safety"
  - "Open-Set Recognition"
  - "PostMax"
  - "OOSA metric"
  - "Extreme Value Distribution"
  - "Logit Normalization"
date: 2026-05-08
content_hash: 13522cf6f7b35405
---

# Operational Open-Set Recognition and PostMax Refinement

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Open-Set Recognition / Classification  
**Keywords**: Open-Set Recognition, PostMax, OOSA metric, Extreme Value Distribution, Logit Normalization

## TL;DR
This paper proposes OOSA (Operational Open-Set Accuracy), an evaluation metric for practical deployment scenarios, and PostMax, a post-processing algorithm. By normalizing the maximum class logit with deep feature magnitude and mapping it through a Generalized Pareto Distribution (GPD), logits are converted into reasonable probability estimates, achieving statistically significant SOTA performance in large-scale evaluations.

## Background & Motivation

**Background**: Open-Set Recognition (OSR) aims to enable classifiers to reject unknown samples that do not belong to any known class during testing. Current mainstream methods typically distinguish between known and unknown samples based on softmax scores, logit values, or feature distances, then make rejection decisions by setting a threshold.

**Limitations of Prior Work**: Existing OSR evaluation schemes face two core issues: (1) most evaluations are conducted on small-scale datasets, which are decoupled from actual deployment scenarios; (2) thresholds are typically tuned on the test set, which is infeasible in practice as the distribution of unknown classes at test time is inaccessible prior to deployment. Consequently, reported performance fails to reflect behavior in real operating scenarios.

**Key Challenge**: Existing evaluation metrics (such as AUROC, FPR95) focus on overall ranking capability across different thresholds rather than the practical requirement of selecting a single fixed threshold before deployment. In real-world scenarios, operators can only use a validation set (containing known samples and surrogate unknowns) to determine a threshold, which is then applied as a fixed threshold during actual testing.

**Goal**: (1) To design an evaluation protocol and metric consistent with operational scenarios; (2) to propose a post-processing method that enhances the rejection capability in open-set recognition, especially under fixed thresholds.

**Key Insight**: The authors observe that logit values output by deep networks vary significantly with feature magnitude; logits of different samples within the same class can differ vastly, making decision-making with a single threshold difficult. Normalizing logits and mapping them to a reasonable probability space would stabilize threshold selection.

**Core Idea**: Stabilize open-set threshold decisions by normalizing logits with deep feature magnitude and mapping them to probabilities using the Generalized Pareto Distribution.

## Method

### Overall Architecture
PostMax is a post-processing method that can be applied to any pre-trained deep classification network without re-training. The pipeline is: input image → pre-trained network extracts features and logits → PostMax normalizes the maximum class logit and performs probability mapping → determine known/unknown using the operational threshold. Evaluation follows the proposed OOSA protocol: determine the threshold on the validation set, then evaluate on an independent test set.

### Key Designs

1. **OOSA Metric (Operational Open-Set Accuracy)**:

    - Function: Provide an evaluation method for open-set recognition that aligns with actual operational scenarios.
    - Mechanism: OOSA requires operators to predict an "operational threshold" from a validation set containing known classes and a set of surrogate unknowns. The predicted threshold is then directly applied to the test set, which features different known and unknown samples. OOSA synthesizes the ability to correctly classify known samples and correctly reject unknown samples into a single metric.
    - Design Motivation: Address the unrealistic assumption of tuning thresholds on the test set in existing evaluations, reflecting the constraint in real deployment where the threshold must be predetermined.

2. **Logit Normalization via Deep Feature Magnitude**:

    - Function: Eliminate instability in logit scores caused by variations in feature magnitude.
    - Mechanism: For the maximum class logit $l_{max}$, compute the L2 norm of the final-layer deep feature $\|f\|$, then use $l_{max} / \|f\|$ as the normalized score. Samples with larger feature magnitudes tend to have larger logits; normalization reduces this bias, making scores across different samples more comparable.
    - Design Motivation: In pre-trained networks, there is a positive correlation between feature magnitude and logit values. Using raw logits directly for threshold decision-making causes samples with large feature magnitudes to be classified overconfidently, leading to unstable thresholds.

3. **Generalized Pareto Distribution Probability Mapping (GPD Mapping)**:

    - Function: Map normalized logit scores to reasonable probability values.
    - Mechanism: Based on Extreme Value Theory, the tail distribution of normalized logit scores is modeled using a Generalized Pareto Distribution (GPD). GPD parameters are fitted on known samples in the validation set, and the scores of all test samples are mapped to the $[0,1]$ probability space via the cumulative distribution function (CDF) of the GPD. A high probability indicates high confidence that a sample belongs to the known classes.
    - Design Motivation: Directly using logit or softmax scores as probability estimates is unreliable, whereas extreme value distributions can better model the tail of the "normal" (known) sample score distribution, providing a statistical foundation for threshold decisions.

### Loss & Training
PostMax is a post-processing method and does not involve any additional training or loss functions. It fits the GPD parameters using the score distribution of known samples on the validation set, and then applies this distribution model to the testing phase. Thus, it can be directly stacked on top of any pre-trained classifier.

## Key Experimental Results

### Main Results
The authors tested multiple pre-trained deep networks, including leading Transformer and CNN architectures, using a large-scale evaluation protocol.

| Method | OOSA (↑) | AUROC (↑) | FPR95 (↓) | Description |
|------|----------|-----------|-----------|------|
| MSP (baseline) | Low baseline | Lower | Higher | Directly uses maximum softmax probability |
| MaxLogit | Medium | Medium | Medium | Uses maximum logit value |
| PostMax (Ours) | **Highest** | **Highest** | **Lowest** | Normalization + GPD mapping |

### Ablation Study

| Configuration | OOSA | Description |
|------|------|------|
| Logit normalization only | Significant gain | Feature magnitude normalization alone is helpful |
| GPD mapping only | Moderate gain | Probability mapping alone has limited effect |
| Full PostMax | Optimal | Combination of both yields the best performance |
| Different surrogate unknown sets | Stable | Performance shows little variation when changing surrogate sets |

### Key Findings
- PostMax achieves consistent improvements across all tested pre-trained networks, demonstrating high generalizability of the method.
- Transformer architectures generally outperform CNN architectures, but PostMax is effective for both.
- PostMax performs stably under different selections of surrogate and test unknown sets, showing robustness to the unknown distribution.
- The experimental results pass statistical significance tests, indicating that the improvements are not random fluctuations.

## Highlights & Insights
- **Zero-training-cost post-processing method**: PostMax operates entirely at inference time without needing to retrain or fine-tune models. It can be directly overlaid on any deployed classification system, offering extremely high practical value.
- **Reversed design from evaluation protocol**: The authors first pinpoint the issues with current evaluations and propose the OOSA metric, then design the PostMax algorithm accordingly. This research pipeline of "defining the problem before solving it" is highly exemplary.
- **Application of Extreme Value Theory in OSR**: Modeling the tail distribution of known sample scores with GPD provides a probabilistic basis for threshold selection, which can be transferred to related tasks such as anomaly detection and OOD detection.

## Limitations & Future Work
- The OOSA evaluation protocol depends on the quality of surrogate unknown sets; if the gap between surrogate and actual unknowns is too large, threshold prediction may become inaccurate.
- PostMax only utilizes the final-layer features and the maximum logit, leaving potential room for improvement by incorporating intermediate-layer features or logit information from all classes.
- The method assumes that the relationship between deep feature magnitude and logits can be linearly normalized, which may not hold for certain architectures.
- Large-scale evaluations are validated only on image classification tasks; its efficacy in tasks like object detection and semantic segmentation has not yet been explored.

## Related Work & Insights
- **vs MSP (Maximum Softmax Probability)**: MSP relies directly on the maximum softmax value and neglects logit magnitude bias and probability calibration issues. PostMax addresses these issues via normalization and GPD mapping.
- **vs OpenMax**: OpenMax also utilizes Extreme Value Theory but requires fitting a Weibull distribution on the training set and modifies the softmax computation. PostMax is simpler, executing sheer post-processing without altering the model output structure.
- **vs ODIN**: ODIN requires temperature scaling and input perturbations, introducing extra computational overhead. PostMax does not require any input modifications.

## Rating
- Novelty: ⭐⭐⭐⭐ Proposes the operational-level evaluation metric OOSA and the post-processing method based on extreme value theory. The approach is clear, though individual components are not entirely brand new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Tested on various architectures, large-scale evaluation, and statistical significance tests, but lacks generalization validation on more diverse tasks.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and a complete logical chain, progressing step-by-step from evaluation limitations to method design.
- Value: ⭐⭐⭐⭐ A plug-and-play method of high practical value, and the OOSA metric will help drive progress in the community.

## Additional Notes
Terrance Boult, one of the authors, is a pioneer in the field of open-set recognition. PostMax inherits the extreme value theory concepts from his prior work on OpenMax, but is simpler and more lightweight, reflecting a mature evolution of the methodology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Open-Insect: Benchmarking Open-Set Recognition of Novel Species in Biodiversity Monitoring](../../NeurIPS2025/ai_safety/open-insect_benchmarking_open-set_recognition_of_novel_species_in_biodiversity_m.md)
- [\[NeurIPS 2025\] Towards Unsupervised Open-Set Graph Domain Adaptation via Dual Reprogramming](../../NeurIPS2025/ai_safety/towards_unsupervised_open-set_graph_domain_adaptation_via_dual_reprogramming.md)
- [\[NeurIPS 2025\] Unlocking Transfer Learning for Open-World Few-Shot Recognition](../../NeurIPS2025/ai_safety/unlocking_transfer_learning_for_open-world_few-shot_recognition.md)
- [\[ECCV 2024\] Unveiling Privacy Risks in Stochastic Neural Networks Training: Effective Image Reconstruction from Gradients](unveiling_privacy_risks_in_stochastic_neural_networks_training_effective_image_r.md)
- [\[ECCV 2024\] One-stage Prompt-based Continual Learning](one-stage_prompt-based_continual_learning.md)

</div>

<!-- RELATED:END -->
