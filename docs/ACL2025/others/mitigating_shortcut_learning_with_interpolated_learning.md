---
title: >-
  [Paper Note] Mitigating Shortcut Learning with InterpoLated Learning
description: >-
  [shortcut learning] This paper proposes InterpoLated Learning (InterpoLL), which mitigates the model's reliance on shortcut features and significantly improves generalization on minority samples by interpolating the representations of majority samples with those of minority samples from the same class.
tags:
  - "shortcut learning"
  - "interpolation"
  - "representation learning"
  - "minority generalization"
  - "ERM"
date: 2026-05-08
content_hash: aa5b28132e6e2b44
---

# Mitigating Shortcut Learning with InterpoLated Learning

| Conference | Year | Paper Link | Code |
|----------|------|---------|------|
| ACL 2025 | 2025 | [arXiv 2507.05527](https://arxiv.org/abs/2507.05527) | - |

**Area**: Natural Language Understanding / Robustness  
**Keywords**: shortcut learning, interpolation, representation learning, minority generalization, ERM

## TL;DR

This paper proposes InterpoLated Learning (InterpoLL), which mitigates the model's reliance on shortcut features and significantly improves generalization on minority samples by interpolating the representations of majority samples with those of minority samples from the same class.

## Background & Motivation

**Problem Definition**: Models trained with Empirical Risk Minimization (ERM) tend to exploit shortcuts (i.e., spurious correlations between input features and labels in the training data); for example, "entailment" samples in MNLI often exhibit high word overlap. Consequently, models perform well on majority samples but perform poorly on minority samples where the shortcut does not hold.

**Limitations of Prior Work**:
- Data augmentation methods (synthesizing minority samples) and sample reweighting methods (upweighting minority samples) **primarily improve the classification layer** but fail to learn representations distinct from ERM, and might even reinforce shortcut features.
- Many approaches rely on **auxiliary models**, introducing substantial computational overhead and hyperparameter tuning complexity.
- Some methods require **prior knowledge of group labels (group annotation)** for minority/majority samples, which are difficult to obtain in real-world scenarios.

**Goal**: To design a model-agnostic shortcut mitigation method that does not require group annotations and truly improves representation learning.

## Method

### Overall Architecture

InterpoLL consists of two phases:

1. **Inferring Minority/Majority Samples**: An underparameterized auxiliary model $f_\phi$ (such as TinyBERT) is used to classify the training set. Incorrectly classified samples are treated as minority samples $g_{\min}$, while correctly classified ones are considered majority samples $g_{\maj}$.
2. **Interpolated Training**: For each majority sample in a mini-batch, its representation is interpolated with that of a minority sample from the same class. The interpolated representation is then used to calculate the loss and update the model.

### Key Designs

- **In-class Minority Interpolation**: For each majority sample $(x_i, y_i) \in g_{\maj}$, a minority sample $(x_j, y_j) \in g_{\min}$ with the same label is randomly selected. Linear interpolation is then performed in the encoder output space: $z_i = (1-\lambda) f_{\text{enc}}(x_i) + \lambda f_{\text{enc}}(x_j)$.
- **Restricted Interpolation Ratio**: $\lambda \sim \text{Uniform}(0, 0.5)$ ensures that the majority representations are only slightly modified. This maintains the capacity to fit majority samples while introducing shortcut-mitigating features from minority samples.
- **Label Invariance**: Since $x_i$ and $x_j$ share the same class, the label remains unchanged after interpolation, thereby avoiding label modifications.

### Loss & Training

A standard cross-entropy loss is used. The original $f_{\text{enc}}(x_i)$ is replaced by the interpolated representation $z_i$ during the forward pass only, while backpropagation proceeds normally:

$$J_{\text{ERM}}(\theta) = \frac{1}{n} \sum_{i=1}^{n} \ell(f_{\text{cls}}(z_i), y_i)$$

## Key Experimental Results

### Main Results

**Natural Language Inference (NLI)**:

| Method | MNLI-ID | MNLI-OOD | FEVER-ID | FEVER-OOD | QQP-ID | QQP-OOD | Avg-OOD |
|------|---------|----------|----------|-----------|--------|---------|---------|
| ERM | 84.9 | 62.4 | 88.4 | 55.9 | 90.2 | 33.8 | 50.7 |
| GroupDRO (Requires Group) | 84.3 | 72.5 | 87.5 | 64.1 | 89.5 | 52.9 | 63.2 |
| InterpoLL (No Group) | 84.6 | **75.6** | 87.8 | **68.7** | 89.8 | **56.9** | **67.1** |

**Text Classification**:

| Method | FDCL18-Avg | FDCL18-Minority | CivilComments-Minority | Avg-Minority |
|------|-----------|-----------------|----------------------|--------------|
| ERM | 81.3 | 35.6 | 63.5 | 49.6 |
| GroupDRO | 76.2 | 57.3 | 69.5 | 63.4 |
| InterpoLL | 78.8 | **61.2** | **73.9** | **67.6** |

### Ablation Study

**Cross-Architecture Generalization** (MNLI → OOD sets such as HANS/PAWS/Sym):

| Model | ERM-Avg | InterpoLL-Avg | Gain |
|------|---------|---------------|------|
| BERT-large | 61.7 | 67.7 | +6.0 |
| RoBERTa-large | 65.4 | 71.9 | +6.5 |
| T5-large | 69.9 | 76.0 | +6.1 |
| T5-3B | 70.8 | 77.3 | +6.5 |

**Domain Generalization (GLUE-X)**: InterpoLL achieves an average improvement of 3.1% across 6 tasks, outperforming the second-best method Minimax by 2.5%.

### Key Findings

1. InterpoLL significantly outperforms group-annotated methods (such as GroupDRO) **without requiring group annotations**.
2. Performance improvements are **consistently effective** across three architectures: encoder, encoder-decoder, and decoder-only.
3. InterpoLL not only improves the classification layer but also **mitigates shortcut features in representations**.
4. The training time is **virtually identical** to ERM, incurring no significant computational overhead.

## Highlights & Insights

- Simple and elegant: Effectively mitigates shortcut learning solely through representation interpolation, without requiring complex adversarial training or multi-stage pipelines.
- High practical value: Outperforms group-dependent methods without requiring group annotations.
- Demonstrates generalization with consistent improvements across different architectures and tasks.
- Provides detailed analyses, including probing experiments of shortcut features within the learned representations.

## Limitations & Future Work

- Inference of minority/majority samples by the auxiliary model contains noise, making the approach dependent on auxiliary model quality.
- The interpolation ratio $\lambda$ range of $[0, 0.5]$ is fixed, which may not be optimal for all tasks.
- Primarily validated on NLU tasks, leaving its applicability to generative tasks unexplored.
- Requires training an auxiliary model to infer sample groups.

## Related Work & Insights

- **Shortcut Mitigation**: GroupDRO (Sagawa et al., 2019), JTT (Liu et al., 2021), DFR (Kirichenko et al., 2023)
- **Mixup family**: Mixup proposed by Zhang et al. (2018) interpolates in the input space, whereas InterpoLL interpolates specific sample pairs in the representation space.
- **Sample Reweighting**: Conf-reg (Utama et al., 2020), Weak-learn (Sanh et al., 2021)

## Rating

| Dimension | Score (1-10) |
|------|-----------|
| Novelty | 7 |
| Utility | 8 |
| Experimental Thoroughness | 9 |
| Writing Quality | 8 |
| Overall Score | 8 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] DISCO: Mitigating Bias in Deep Learning with Conditional Distance Correlation](../../ICML2026/others/disco_mitigating_bias_in_deep_learning_with_conditional_distance_correlation.md)
- [\[CVPR 2026\] Mitigating Instance Entanglement in Instance-Dependent Partial Label Learning](../../CVPR2026/others/mitigating_instance_entanglement_in_instance-dependent_partial_label_learning.md)
- [\[ICLR 2026\] Mitigating Spurious Correlation via Distributionally Robust Learning with Hierarchical Ambiguity Sets](../../ICLR2026/others/mitigating_spurious_correlation_via_distributionally_robust_learning_with_hierar.md)
- [\[ACL 2025\] Value Residual Learning](value_residual_learning.md)
- [\[ACL 2025\] Learning to Reason from Feedback at Test-Time](learning_to_reason_from_feedback_at_test-time.md)

</div>

<!-- RELATED:END -->
