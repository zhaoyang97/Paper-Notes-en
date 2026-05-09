---
title: >-
  [Paper Note] Intervening in Black Box: Concept Bottleneck Model for Enhancing Human-Neural Network Mutual Understanding
description: >-
  [ICCV 2025][Object Detection][Interpretability] This paper proposes the CBM-HNMU framework, which approximates the reasoning process of a black-box model via a Concept Bottleneck Model (CBM), automatically identifies and corrects harmful concepts, and distills the corrected knowledge back into the black-box model, enabling systematic model intervention and accuracy improvement beyond the sample level.
tags:
  - ICCV 2025
  - Object Detection
  - Interpretability
  - Concept Bottleneck Model
  - Black-Box Intervention
  - Knowledge Distillation
  - Model Correction
date: 2026-05-08
content_hash: 86975c00a34b1697
---

# Intervening in Black Box: Concept Bottleneck Model for Enhancing Human-Neural Network Mutual Understanding

**Conference**: ICCV 2025
**arXiv**: [2506.22803](https://arxiv.org/abs/2506.22803)
**Code**: [https://github.com/XiGuaBo/CBM-HNMU](https://github.com/XiGuaBo/CBM-HNMU)
**Area**: Object Detection
**Keywords**: Interpretability, Concept Bottleneck Model, Black-Box Intervention, Knowledge Distillation, Model Correction

## TL;DR

This paper proposes the CBM-HNMU framework, which approximates the reasoning process of a black-box model via a Concept Bottleneck Model (CBM), automatically identifies and corrects harmful concepts, and distills the corrected knowledge back into the black-box model, enabling systematic model intervention and accuracy improvement beyond the sample level.

## Background & Motivation

The decision-making processes of deep learning models are increasingly opaque. Existing interpretability methods fall into two main categories:

**Feature attribution methods** (Grad-CAM, Saliency Maps, etc.): capable of explaining *where* a model attends, but unable to answer *what features* are attended to, and lacking intervention capability.

**Concept explanation methods** (ACE, CRAFT, etc.): capable of extracting human-understandable visual concepts, but still confined to passive explanation.

Concept Bottleneck Models (CBMs) introduce an intervnable framework, yet suffer from two major issues: (1) limited representational capacity, with classification accuracy typically lower than black-box models sharing the same backbone; and (2) interventions requiring human prior knowledge and being restricted to the sample level. Post-hoc CBMs partially alleviate the accuracy issue via residual connections, but at the cost of weakened interpretability and limited intervention efficacy.

The core motivation of CBM-HNMU is to ask: can a closed-loop framework be constructed in which humans *understand* the reasoning of neural networks (explanation), while neural networks *understand* human knowledge (intervention and correction)?

## Method

### Overall Architecture

CBM-HNMU consists of three stages: (a) **local approximation** — a CBM approximates the black-box model's reasoning on confusable classes; (b) **concept intervention** — harmful concepts are automatically identified via gradient attribution and corrected; (c) **knowledge transfer** — the corrected CBM knowledge is distilled back into the black-box model.

### Key Designs

1. **Confusable Class Selection**: The black-box classifier is evaluated on a validation set, inter-class misclassification frequencies are recorded, and the class pairs with the highest misclassification rates are extracted to form the confusable class set $\Gamma \subseteq \{1, 2, \ldots, N_{class}\}$. This constrains the scale of the concept bottleneck, enabling the CBM to precisely approximate black-box reasoning on these classes with a limited concept set.

2. **Concept Communication**: Visual concepts $C^{|}$ are extracted from the black-box intermediate layers using CRAFT; natural language concept bottlenecks $C^t$ are generated via ChatGPT-3.5-Turbo; both are then mapped to a shared space $\mathbb{R}^{1 \times d}$ ($d=512$) through CLIP's image/text encoders. Concept scores are computed as:

$$S_i = \frac{1}{n} \sum_{k=1}^{n} E_{img}(C_k^{|}(i)) \times E_{text}(C^t)^T$$

3. **Local Approximation**: A concept weight matrix $W \in \mathbb{R}^{N_\Gamma \times N_c}$ is constructed, with CBM output $P_{cbm}(x_i) = S_i \times W^T$ aligned to the black-box output $P_{org}^\Gamma$ on confusable classes via the L2 norm.

4. **Gradient-Based Concept Intervention**: The core innovation. Two types of harmful concepts are defined:

   - $S_{nT}$: concepts with a negative influence on the correct class
   - $S_{pF}$: concepts with a positive influence on the incorrectly predicted class

   Concept contributions are computed via gradient attribution $G(w_k, P_k(x_i)) = \frac{\partial P_k(x_i)}{\partial w_k} \odot w_k$, globally accumulated and ranked, and the Top-$\bar{q}$ most harmful concepts are selected for removal or replacement.

5. **Knowledge Transfer**: The post-intervention CBM predictions (on confusable classes) and the frozen black-box predictions (on non-confusable classes) are concatenated into a teacher signal $P_t$. A probability residual coefficient $pr = 1 - |softmax(P_{org})^{\Gamma^\complement}|_1$ ensures the prediction distribution of non-intervened classes remains unchanged. Cross-entropy distillation is then used to transfer knowledge back to the black-box model.

### Loss & Training

- Local approximation: L2 loss to align the CBM with the black-box; learning rate 1e-4; 200 epochs.
- Knowledge transfer: cross-entropy distillation loss; teacher temperature $T_1=2.0$; student temperature $T_2=1.5$; learning rate 3e-7; 10 epochs.
- The entire pipeline requires no additional annotation; CLIP and ChatGPT enable full automation.

## Key Experimental Results

### Main Results

| Model | Flower-102 | CIFAR-10 | CIFAR-100 | CUB-200 | FGVC-Aircraft | Avg. Gain |
|-------|-----------|---------|----------|---------|--------------|----------|
| NFResNet50 w/o → w/ | 94.28→95.36 | 80.92→83.56 | 73.58→73.92 | 62.41→62.67 | 64.63→64.94 | ↑0.93% |
| ViT_Small w/o → w/ | 97.24→98.20 | 91.51→92.84 | 81.26→82.43 | 74.75→75.39 | 69.88→70.94 | ↑1.03% |
| GCVit w/o → w/ | 93.58→95.20 | 80.20→80.97 | 72.38→72.55 | 76.99→77.64 | 69.81→71.41 | ↑0.96% |

*All models achieve positive gains across all datasets; the largest single improvement is 2.64% (NFResNet50 on CIFAR-10).*

### Ablation Study

| Method | Flower-102 | CUB-200 | FGVC-Aircraft |
|--------|-----------|---------|--------------|
| CBM-HNMU (Ours) | 95.36 | 62.67 | 64.94 |
| Random Concept Intervention | 94.72 (↓0.64) | 62.51 (↓0.16) | 64.75 (↓0.19) |
| CBM (same backbone) | 92.57 | 54.53 | 56.31 |
| Post-hoc CBM | 93.58 | 58.20 | 64.45 |
| Black-box Baseline | 94.28 | 62.41 | 64.63 |

*CBM-HNMU outperforms both CBM and Post-hoc CBM, and is the only concept bottleneck method capable of surpassing the black-box baseline accuracy.*

### Key Findings

- Performance is most stable when the proportion of intervened classes is below 25%; intervening on all classes leads to a significant increase in approximation error.
- Concept replacement (searching for substitute concepts from an external concept set) can further improve performance.
- A user study (30 participants) confirms that the intervened concepts are consistent with human visual perception, with confidence scores consistently above 0.5.
- The accuracy of non-intervened classes remains largely unchanged before and after intervention, validating the effectiveness of the $pr$ coefficient.

## Highlights & Insights

- The paper innovatively constructs a closed-loop "understand → intervene → correct" framework, breaking the limitations of existing concept explanation methods that are confined to passive interpretation.
- The entire pipeline requires no human annotation (CLIP + ChatGPT), substantially reducing practical deployment costs.
- The intervention mechanism operates beyond the sample level, achieving global correction by modifying the black-box model's parameters.

## Limitations & Future Work

- Concept extraction relies on CRAFT, which is not model-guided; the association between concepts and samples may therefore lack precision.
- The concept bottleneck generated by ChatGPT may contain abstract or hallucinated concepts that are not easily interpretable by humans.
- The effectiveness of intervention depends on the quality of confusable class selection.

## Related Work & Insights

- Compared to Beyond CBM, HNMU not only diagnoses biases but also modifies black-box parameters through distillation.
- The concept replacement strategy (Appendix A) provides a direction for future research on finer-grained control in concept space.
- The framework is extensible to interpretability intervention in detection and segmentation tasks.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Automated Model Evaluation for Object Detection via Prediction Consistency and Reliability](automated_model_evaluation_for_object_detection_via_prediction_consistency_and_r.md)
- [\[CVPR 2026\] Prompt-Free Universal Region Proposal Network](../../CVPR2026/object_detection/prompt-free_universal_region_proposal_network.md)
- [\[ICCV 2025\] VOccl3D: A Video Benchmark Dataset for 3D Human Pose and Shape Estimation under Real Occlusions](voccl3d_a_video_benchmark_dataset_for_3d_human_pose_and_shape_estimation_under_r.md)
- [\[NeurIPS 2025\] DetectiumFire: A Comprehensive Multi-modal Dataset Bridging Vision and Language for Fire Understanding](../../NeurIPS2025/object_detection/detectiumfire_a_comprehensive_multi-modal_dataset_bridging_vision_and_language_f.md)
- [\[NeurIPS 2025\] MSTAR: Box-Free Multi-Query Scene Text Retrieval with Attention Recycling](../../NeurIPS2025/object_detection/mstar_box-free_multi-query_scene_text_retrieval_with_attention_recycling.md)

</div>

<!-- RELATED:END -->
