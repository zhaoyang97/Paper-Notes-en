---
title: >-
  [Paper Note] Every Error has Its Magnitude: Asymmetric Mistake Severity Training for Multiclass Multiple Instance Learning
description: >-
  [CVPR2026][Medical Imaging][Multiple Instance Learning] This paper proposes PAMS (Priority-Aware Mistake Severity), a framework that significantly reduces the risk of severe misdiagnosis in multiclass MIL-based WSI diagn…
tags:
  - "CVPR2026"
  - "Medical Imaging"
  - "Multiple Instance Learning"
  - "Mistake Severity"
  - "Whole Slide Image"
  - "Asymmetric Misclassification"
  - "Hierarchical Classification"
  - "Pathological Diagnosis"
date: 2026-05-08
content_hash: 04d0b541c360c740
---

# Every Error has Its Magnitude: Asymmetric Mistake Severity Training for Multiclass Multiple Instance Learning

**Conference**: CVPR2026
**arXiv**: [2603.13682](https://arxiv.org/abs/2603.13682)  
**Code**: To be confirmed  
**Area**: Medical Imaging
**Keywords**: Multiple Instance Learning, Mistake Severity, Whole Slide Image, Asymmetric Misclassification, Hierarchical Classification, Pathological Diagnosis

## TL;DR

This paper proposes PAMS (Priority-Aware Mistake Severity), a framework that significantly reduces the risk of severe misdiagnosis in multiclass MIL-based WSI diagnosis through an asymmetric severity-aware cross-entropy loss (MSCE), semantic feature remix (SFR), and an asymmetric Mikel's Wheel evaluation metric.

## Background & Motivation

1. **Wide adoption of MIL in pathological diagnosis**: Multiple Instance Learning (MIL) models WSIs as patch bags and has become the dominant paradigm in computational pathology; however, existing methods focus primarily on maximizing accuracy while ignoring the varying severity of misclassification errors.
2. **Asymmetric cost of misclassification in clinical settings**: Missing a malignant tumor (false negative) carries far greater consequences than over-diagnosing a normal case as malignant (false positive), yet conventional cross-entropy imposes equal penalties on all errors.
3. **Priority structure in WSI multiclass classification**: Pathologists annotate the most urgent diagnosis when multiple co-existing conditions are observed in a WSI; an implicit priority hierarchy exists among categories, which fundamentally differs from natural image annotation where each object is labeled independently.
4. **Limitations of existing Mistake Severity methods**: Prior approaches define severity weights solely based on inter-class distance (e.g., CDW-CE), disregarding directionality—misclassifications of equal distance in opposite directions carry entirely different clinical risks.
5. **Absence of MS solutions tailored for clinical WSI**: Existing MS research is primarily conducted on natural images and fails to address the annotation constraints inherent to WSIs, including weak labels, complex co-existing conditions, and category priorities.
6. **Limitations of evaluation metrics**: Existing MS metrics (ECC/EMC) rely on symmetric distances and cannot distinguish between misclassifications of different directions, rendering them inadequate for evaluating model safety.

## Method

### Overall Architecture

PAMS organizes the multiclass problem into a hierarchical structure from the finest granularity $\mathcal{H}$ to the root node $\mathcal{R}$, training a dedicated classifier $f_{\theta_h}$ at each level. The training objective is $\mathcal{L} = \lambda_1 \mathcal{L}_{MSCE} + \lambda_2 \mathcal{L}_{HA}$, with SFR applied as a data augmentation strategy.

### Mistake Severity Cross-Entropy (MSCE)

- An asymmetric weight matrix $M^h$ is defined such that when the ground-truth class $c_i^h$ is more urgent than the predicted class $c_j^h$, the penalty is $\alpha^{|i-j|}$ (with $\alpha > 1$); misclassifications in the reverse direction receive a weight of 1.
- The final loss is $\mathcal{L}_{MSCE} = -\sum_h \hat{p}^h M^h (\tilde{Y}^h)^\top \sum_c \tilde{Y}^h[c] \log \hat{p}^h[c]$
- **Core Idea**: A directional regularization weight $\hat{p}^h M^h (\tilde{Y}^h)^\top$ is multiplied before the cross-entropy term, jointly accounting for the severity relationship between the predicted probability distribution and the ground-truth label.
- **Distinction from Weighted CE**: Weighted CE relies on class frequency or fixed weights without modeling the directional asymmetry between predictions and ground-truth labels.

### Hierarchy Alignment (HA)

- Jensen-Shannon divergence is used to align predicted probabilities across adjacent hierarchy levels.
- Predictions from a finer-grained level $\hat{p}^{h+1}$ are aggregated into a coarser representation $\dot{p}^{h+1}$ and aligned with the current level $\hat{p}^h$.
- This ensures consistent predictions across classifiers at different hierarchy levels for the same sample.

### Semantic Feature Remix (SFR)

- Given two WSIs of different priorities ($Y_a \succ Y_b$), all instances from both are clustered into $L$ clusters.
- Clusters are ranked by the proportion of patches from the higher-priority sample $Z_a$, and patches from the top-$k$ clusters in $Z_a$ are selected.
- These semantically representative high-severity patches are mixed into the lower-priority bag $Z_b$ to form a synthetic sample $Z_{a+b}$ with label $Y_a$.
- Efficient GPU-parallel clustering is implemented via the FAISS library.

### Asymmetric Mikel's Wheel Metrics

- Two new metrics are proposed: AsCC (Asymmetric Classification Confidence) and AsMC (Asymmetric Misclassification Confidence).
- The confusion weight is defined as $W_{i,j}^h = 1 + |i-j| + \mathbb{1}(c_i^h \succ c_j^h) \times P$, where $P=2$.
- An additional penalty is applied when a higher-priority class is misclassified as a lower-priority class, reflecting the true clinical risk.

## Key Experimental Results

### Datasets

- **BRACS**: 547 H&E-stained breast cancer WSIs, 7 classes (normal to invasive carcinoma), organized into a three-level hierarchy of benign/atypical/malignant.
- **In-house**: 4,734 colon biopsy WSIs, 7 classes, organized into a three-level hierarchy of benign/serrated/adenoma; includes a test set of 182 complex mixed-condition cases.

### Main Results (Table 1, BRACS + TransMIL)

| Method | ACC | AUC | AsCC | AsMC |
|--------|-----|-----|------|------|
| Cross Entropy | 40.23 | 74.90 | 58.48 | 50.18 |
| Chang et al. | 47.51 | 79.48 | 63.98 | 51.02 |
| Hong et al. (τ=10) | 47.13 | 79.80 | 62.44 | 45.54 |
| CDW-CE | 44.83 | 79.06 | 61.05 | 47.32 |
| **PAMS (Ours)** | **47.59** | **80.61** | **64.92** | **55.65** |

PAMS achieves the best performance across all metrics, with the most pronounced improvements in AsCC and AsMC. It similarly leads across all metrics on the in-house dataset.

### Ablation Study (Table 2, BRACS + TransMIL)

| Ablated Component | ACC Drop | AsMC Drop |
|-------------------|----------|-----------|
| w/o MSCE | -2.46 | -4.84 |
| w/o HA | -2.84 | -0.53 |
| w/o SFR | -0.54 | -4.02 |
| All removed | -7.82 | -1.76 |

- MSCE contributes most to severity-aware metrics (AsMC drop of 4.84).
- SFR also yields a substantial contribution to AsMC (drop of 4.02).
- All three components work synergistically for optimal performance.

### CIFAR-10 Natural Image Experiments (Table 4)

| Method | ACC | AsCC | AsMC |
|--------|-----|------|------|
| CE | 83.24 | 87.23 | 34.84 |
| CDW-CE | 84.11 | 87.87 | 34.63 |
| **MSCE (Ours)** | **85.64** | **89.12** | **35.70** |

These results validate the generalizability of MSCE to the natural image domain.

## Highlights & Insights

- **Asymmetric severity modeling**: This work is the first to introduce directional misclassification penalties in MIL-based WSI diagnosis, accurately reflecting the clinical reality that missed malignancies are more dangerous than over-diagnoses.
- **Semantic data augmentation via SFR**: SFR leverages weak label information to intelligently mix samples in feature space, simulating complex co-existing conditions without requiring pixel-level annotations.
- **Metric innovation**: AsCC/AsMC address the fundamental limitation of existing symmetric metrics that cannot distinguish misclassification direction, making them applicable to all safety-critical classification tasks.
- **Broad generalizability**: The method proves effective across BRACS, the in-house medical dataset, and CIFAR-10 natural images, and is compatible with multiple MIL architectures.

## Limitations & Future Work

- The hierarchical structure must be predefined manually, relying on domain expert knowledge, and different diseases may require different hierarchy designs.
- The hyperparameters $\alpha$ and $P$ in MSCE require tuning; sensitivity analysis is deferred to the supplementary material.
- SFR depends on clustering quality, and the choices of cluster count $L$ and top-$k$ may affect performance.
- Validation is limited to pathology; generalization to other medical imaging modalities such as radiology and dermoscopy remains unexplored.
- The in-house dataset is not publicly released, limiting reproducibility.

## Related Work & Insights

- **vs. Weighted CE**: Uses fixed weights and cannot capture the directional asymmetry between predictions and ground-truth; MSCE computes penalties dynamically.
- **vs. HXE / Soft Labels (Bertinetto et al.)**: Exploits LCA-based hierarchy information but yields limited improvement on severity metrics.
- **vs. HAF (Garg et al.)**: A feature-space regularization approach that generalizes poorly on DTFD-MIL.
- **vs. Hong et al.**: The random remix strategy is effective on the in-house data but unstable on BRACS; SFR is more robust through semantic guidance.
- **vs. CDW-CE**: Distance-based weighting remains symmetric; PAMS's asymmetric design better aligns with clinical requirements.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of an asymmetric severity loss, semantic remix, and asymmetric evaluation metrics forms a coherent framework with a clearly defined problem.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers public and private datasets, multiple MIL architectures, ablation studies, remix strategy comparisons, and natural image generalization experiments.
- Writing Quality: ⭐⭐⭐⭐ — Figures and tables are clearly presented; problem motivation is convincingly established; mathematical derivations are complete.
- Value: ⭐⭐⭐⭐ — Addresses a core safety concern in clinical MIL deployment; the asymmetric metrics have broad applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MIL-PF: Multiple Instance Learning on Precomputed Features for Mammography Classification](milpf_multiple_instance_learning_on_precomputed_fe.md)
- [\[CVPR 2026\] Fair Lung Disease Diagnosis from Chest CT via Gender-Adversarial Attention Multiple Instance Learning](fair_lung_disease_diagnosis_from_chest_ct_via_gend.md)
- [\[ACL 2026\] Multi-View Attention Multiple-Instance Learning Enhanced by LLM Reasoning for Cognitive Distortion Detection](../../ACL2026/medical_imaging/multi-view_attention_multiple-instance_learning_enhanced_by_llm_reasoning_for_co.md)
- [\[CVPR 2026\] Bidirectional Multimodal Prompt Learning with Scale-Aware Training for Few-Shot Multi-Class Anomaly Detection](bidirectional_multimodal_prompt_learning_with_scale-aware_training_for_few-shot_.md)
- [\[AAAI 2026\] Error Correction in Radiology Reports: A Knowledge Distillation-Based Multi-Stage Framework](../../AAAI2026/medical_imaging/error_correction_in_radiology_reports_a_knowledge_distillation-based_multi-stage.md)

</div>

<!-- RELATED:END -->
