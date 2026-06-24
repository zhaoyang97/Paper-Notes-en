---
title: >-
  [Paper Note] Project-Probe-Aggregate: Efficient Fine-Tuning for Group Robustness
description: >-
  [CVPR 2025][Social Computing][Group Robustness] This paper proposes Project-Probe-Aggregate (PPA), a three-step framework that improves the group robustness of foundation models without group annotations, using less than 0.01% of trainable parameters. PPA projects features to remove class proxies and amplify bias, probes group labels corrected with group priors, and aggregates group weights.
tags:
  - "CVPR 2025"
  - "Social Computing"
  - "Group Robustness"
  - "Spurious Correlation"
  - "Parameter-Efficient Fine-Tuning"
  - "Debiasing"
  - "CLIP"
date: 2026-05-08
content_hash: 4325d7d340b107a9
---

# Project-Probe-Aggregate: Efficient Fine-Tuning for Group Robustness

**Conference**: CVPR 2025  
**arXiv**: [2503.09487](https://arxiv.org/abs/2503.09487)  
**Code**: None  
**Area**: Social Computing  
**Keywords**: Group Robustness, Spurious Correlation, Parameter-Efficient Fine-Tuning, Debiasing, CLIP

## TL;DR

This paper proposes Project-Probe-Aggregate (PPA), a three-step framework that improves the group robustness of foundation models without group annotations, using less than 0.01% of trainable parameters. PPA projects features to remove class proxies and amplify bias, probes group labels corrected with group priors, and aggregates group weights.

## Background & Motivation

Although vision-language foundation models (e.g., CLIP) demonstrate low average error in downstream tasks, they exhibit extremely high error rates on certain subgroups in the presence of spurious correlations. For example, in the Waterbirds dataset, 95% of waterbird images depict a water background. Consequently, models easily rely on the "water background" instead of the "bird's features" for classification, leading to severe failures on the subgroup of "waterbirds with land backgrounds."

Existing approaches to improving group robustness face two primary challenges: (1) requirement of group annotations (high annotation cost); and (2) requirement of retraining the entire model (high computational cost). Strategies based on "failure-based debiasing"—which identify minority groups by first training a biased model, and then train a robust model using the inferred group labels—are popular. However, the identification accuracy of minority groups remains low, and up-weighting strategies lack theoretical guarantees of optimality.

**Core Problem**: How to leverage knowledge from pre-trained foundation models to more efficiently identify minority groups and design theoretically grounded debiasing training algorithms?

## Method

### Overall Architecture

PPA is a three-step linear probing framework that freezes the CLIP backbone and only trains linear classification heads: Step 1 (Project) projects image features onto the null space of class proxies to train a biased classifier; Step 2 (Probe) infers pseudo-group labels using the biased classifier, trains a group classifier, and corrects it using group prior shifts; Step 3 (Aggregate) aggregates within-class group weights to obtain the final debiased classifier. Only two linear layers are trained throughout the entire process.

### Key Designs

#### Key Design 1: Class-Proxy Projection (Project)

**Function**: Amplify the model's reliance on spurious features, thereby improving the accuracy of minority group identification.

**Mechanism**: Utilize the CLIP text encoder to obtain class name embeddings $Z = [\mathbf{z}_1, ..., \mathbf{z}_K]^T$, and compute its null space projection matrix $\Pi = I - Z^T(ZZ^T)^{-1}Z$. After projecting the image features, train a biased classifier $f_\mathsf{b}(\mathbf{x}) = W_\mathsf{b} \Pi \mathbf{x}$ in conjunction with a logit-adjustment loss to handle class imbalance.

**Design Motivation**: Upon removing core class information, the classifier is forced to rely more heavily on spurious features (such as backgrounds) for predictions. Proposition 1 theoretically proves that when the spurious feature is positively correlated with the core feature removed via projection, the spurious feature weight increases ($\gamma' > \gamma$). For example, in Waterbirds, "water background" is positively correlated with "waterbird." Eliminating "waterbird" information increases the weight of "water background", making the biased model more prone to errors on minority groups.

#### Key Design 2: Group Prior Correction Probing (Probe)

**Function**: Train a debiased classifier in a theoretically optimal manner.

**Mechanism**: Define pseudo-group labels as $\hat{g} = (y, \hat{a})$ based on the correct or incorrect predictions of the biased classifier. Train a group classifier $h_\mathsf{d}(\mathbf{x}) = W_\mathsf{d} \mathbf{x}$ using the group logit-adjusted loss: $\ell_\mathsf{gla}(\hat{g}, h_\mathsf{d}(\mathbf{x})) = -\ln \frac{\exp(h_\mathsf{d}(\mathbf{x}) + \tau \cdot \ln \hat{\boldsymbol{\beta}})_{\hat{g}}}{\sum_{g'} \exp(h_\mathsf{d}(\mathbf{x}) + \tau \cdot \ln \hat{\boldsymbol{\beta}})_{g'}}$, where $\hat{\boldsymbol{\beta}}$ represents the group prior.

**Design Motivation**: Traditional methods improve group robustness by upweighting minority samples, but determining the optimal weights requires hyperparameter searching. Proposition 2 proves that group logit adjustment is the Bayes-optimal classifier for minimizing the balanced group error (BGE), providing theoretical guarantees.

#### Key Design 3: Weight Space Aggregation (Aggregate)

**Function**: Convert group classifiers into class classifiers, eliminating the group inference overhead during deployment.

**Mechanism**: Sum the weight vectors of all groups belonging to the same class to obtain the class classifier: $f_\mathsf{d}(\mathbf{x})_y = \mathbf{w}_y^T \mathbf{x}$, where $\mathbf{w}_y^T = \sum_{g \in \hat{\mathcal{G}}(y)} W_{\mathsf{d},g}$. Since summation in the output space of a linear classifier is equivalent to summation in the weight space, only a single forward pass is required during inference.

**Design Motivation**: Directly using a group classifier requires inferring the group label before converting it into a class prediction, which increases inference complexity. Weight aggregation is performed once after training, compressing the $|G|$-dimensional output to a $K$-dimensional space, matching the inference profile of standard classifiers.

### Loss & Training

Two-stage loss: (1) The biased classifier is trained with the logit-adjusted cross-entropy; (2) The group classifier is trained with the group logit-adjusted cross-entropy $\ell_\mathsf{gla}$, where the hyperparameter $\tau$ controls the strength of the prior correction.

## Key Experimental Results

### Main Results (Worst-Group Accuracy, CLIP ResNet-50)

| Method | Group Labels | Waterbirds WGA | CelebA WGA | MetaShift WGA |
|------|--------|---------------|------------|---------------|
| Zero-Shot CLIP | No | 54.2 | 55.0 | 86.2 |
| ERM | No | 7.9 | 11.9 | 75.4 |
| JTT | No | 61.7 | 60.2 | — |
| CA (Contrastive Adapter) | No | — | — | — |
| **PPA (Ours)**| **No** | **Outperforms JTT/CA** | **Outperforms JTT/CA** | **Outperforms JTT** |
| GroupDRO | Yes | 75.1 | 84.1 | 83.2 |

### Minority Group Identification Quality Comparison (Waterbirds)

| Method | Worst-group Recall (%) | Worst-group Precision (%) |
|------|----------------------|-------------------------|
| CA | 46.4 | 15.6 |
| JTT | 78.6 | 24.1 |
| **PPA** | **80.4** | **44.3** |

### Key Findings

1. **Significant improvement in minority group identification**: The worst-group precision of PPA (44.3%) is nearly double that of JTT (24.1%), with a higher recall (80.4% vs 78.6%). This demonstrates that class-proxy projection effectively amplifies the reliance on spurious features.
2. **Extremely low parameter count**: Utilizing less than 0.01% of trainable parameters (only two linear layers), which is far lower than adapter or prompt tuning schemes.
3. **No group annotations required**: Without using any group annotations, the performance matches or even exceeds some methods that require group annotations (such as S-CS and S-CL).
4. **Architecture-agnostic**: The proposed method is effective for both CLIP ResNet-50 and CLIP ViT-L/14, and is compatible with other PEFT paradigms such as prompt tuning and adapters.
5. **Theoretical guarantees**: Proposition 1 proves that the projection increases the weight of spurious features, and Proposition 2 demonstrates that group logit adjustment is BGE-optimal.

## Highlights & Insights

- **Theory-driven design**: Every design step is supported by rigorous mathematical derivation, rather than empirical combination.
- **Leveraging textual priors**: Ingeniously utilizes class-proxy information from the CLIP text encoder as the "core features to be removed", converting pre-trained knowledge into a debiasing tool.
- **Clever weight aggregation**: Translates groups to classes directly in the weight space leveraging the additivity of linear classifiers, yielding zero inference overhead.

## Limitations & Future Work

- **Linear classifier assumption**: The theoretical analysis is based on linear regression or linear classifiers; generalizability to non-linear complex classifiers remains to be verified.
- **Binary attribute constraint**: Validation is currently conducted primarily on binary spurious attributes (e.g., water/land backgrounds); scalability to multi-attribute scenarios remains unknown.
- **Reliance on CLIP quality**: The class-proxy projection relies on the quality of the CLIP text encoder. If the class name embeddings are inaccurate, the projection efficacy may degrade.
- Future work could explore projections in non-linear feature spaces, and debiasing in multi-attribute or continuous attribute settings.

## Related Work & Insights

- **JTT**: Identifies minority groups using misclassified samples from an ERM model, whereas PPA further amplifies this bias through class-proxy projection.
- **Orth-Cali**: Removes bias directions within text embeddings, whereas PPA operates inside the image feature space.
- **Insight**: The strategy of "removing core information to expose spurious dependencies" can be generalized to other fairness and robustness challenges.

## Rating

⭐⭐⭐⭐ — Solid theory, elegant design, and highly practical. The three-step workflow is clean and straightforward, with theoretical guarantees for each step. The extreme efficiency with less than 0.01% parameters is highly commendable. The reliance on the linearity assumption is the only notable limitation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Prompt-Level Distillation: A Non-Parametric Alternative to Model Fine-Tuning for Efficient Reasoning](../../ACL2026/social_computing/prompt-level_distillation_a_non-parametric_alternative_to_model_fine-tuning_for_.md)
- [\[CVPR 2025\] Classifier-guided CLIP Distillation for Unsupervised Multi-label Classification](classifier-guided_clip_distillation_for_unsupervised_multi-label_classification.md)
- [\[NeurIPS 2025\] Noise-Robustness Through Noise: A Framework Combining Asymmetric LoRA with Poisoning MoE](../../NeurIPS2025/social_computing/noise-robustness_through_noise_a_framework_combining_asymmetric_lora_with_poison.md)
- [\[ACL 2026\] BITS Pilani at SemEval-2026 Task 9: Structured Supervised Fine-Tuning with DPO Refinement for Polarization Detection](../../ACL2026/social_computing/bits_pilani_at_semeval-2026_task_9_structured_supervised_fine-tuning_with_dpo_re.md)
- [\[CVPR 2025\] Let Samples Speak: Mitigating Spurious Correlation by Exploiting the Clusterness of Samples](let_samples_speak_mitigating_spurious_correlation_by_exploiting_the_clusterness_.md)

</div>

<!-- RELATED:END -->
