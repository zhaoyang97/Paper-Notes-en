---
title: >-
  [Paper Note] Beyond Logits: Aligning Feature Dynamics for Effective Knowledge Distillation
description: >-
  [ACL 2025][Model Compression][Knowledge Distillation] This paper proposes a knowledge distillation method that goes beyond logit matching. By aligning the dynamics of feature changes (rather than static feature snapshots) of the teacher and student models during the training process, it achieves more effective knowledge transfer, significantly improving distillation performance on NLP tasks.
tags:
  - "ACL 2025"
  - "Model Compression"
  - "Knowledge Distillation"
  - "Feature Dynamics Alignment"
  - "Logit Distillation"
  - "Intermediate Layer Distillation"
  - "NLP Model Compression"
date: 2026-05-08
content_hash: 0d7ee7e4918b4625
---

# Beyond Logits: Aligning Feature Dynamics for Effective Knowledge Distillation

**Conference**: ACL 2025  
**Code**: None  
**Area**: Model Compression  
**Keywords**: Knowledge Distillation, Feature Dynamics Alignment, Logit Distillation, Intermediate Layer Distillation, NLP Model Compression

## TL;DR
This paper proposes a knowledge distillation method that goes beyond logit matching. By aligning the dynamics of feature changes (rather than static feature snapshots) of the teacher and student models during the training process, it achieves more effective knowledge transfer, significantly improving distillation performance on NLP tasks.

## Background & Motivation

**Background**: Knowledge distillation (KD) is a core technology for model compression, which is mainly divided into two categories: logit distillation (matching teacher-student output distributions) and feature distillation (matching intermediate representations). In the NLP field, methods like DistilBERT and TinyBERT have achieved success by distilling knowledge from BERT into smaller models.

**Limitations of Prior Work**: (1) Logit distillation only utilizes information from the final output layer, losing the rich knowledge in the intermediate layers; (2) Traditional feature distillation matches static snapshots of feature distributions, ignoring the evolution pattern of features during the training process; (3) Due to different layers and dimensions between teacher and student models, directly aligning intermediate features requires complex projection mappings.

**Key Challenge**: Effective knowledge distillation should convey "how to learn" rather than just "what is learned". Existing methods treat knowledge as static distributions or features, ignoring the dynamic nature of the knowledge acquisition process.

**Goal**: To design a distillation method focused on feature dynamics that captures the evolution patterns of the teacher model's features across different training stages and samples, and subsequently transfers this dynamic knowledge to the student model.

**Key Insight**: It is observed that when the teacher model processes different samples, the variation patterns of intermediate layer features contain richer semantic information than static features. For example, when encountering hard samples, the teacher's features change dramatically; this "difficulty-aware" information is crucial for student learning.

**Core Idea**: Aligning the "feature dynamics" of the teacher and student models—specifically, the directions and magnitudes of feature changes with respect to inputs—rather than matching the feature values themselves, thereby transferring the teacher's knowledge more efficiently.

## Method

### Overall Architecture
The inputs are training samples. The teacher and student models perform forward propagation to extract features from various layers. Instead of directly aligning feature values, the method calculates feature changes relative to a reference point and then aligns these changes. The final loss combines the traditional logit distillation loss, feature dynamics alignment loss, and task loss.

### Key Designs

1. **Feature Dynamics Extraction**:

    - **Function**: Extracting feature variation information from intermediate layers of the teacher and student models.
    - **Mechanism**: For each sample, the deviation direction and magnitude of its intermediate layer features relative to a reference point (such as the batch mean or training set mean features) are computed. By calculating the gradient changes of features between consecutive layers or feature differences between adjacent samples, the dynamic evolution patterns of features are captured.
    - **Design Motivation**: Feature dynamics reflect the model's inference process better than static features. Two different feature vectors may correspond to the same decision logic, but their variation patterns can reveal this logic.

2. **Cross-Layer Dynamic Alignment**:

    - **Function**: Achieving effective alignment when the layer structures of the teacher and student are asymmetric.
    - **Mechanism**: Instead of strictly requiring a specific layer of the student model to match a specific layer of the teacher, the student's overall feature dynamics are aligned with the teacher's dynamic patterns. Attention mechanisms or optimal transport are utilized to automatically discover the best cross-layer correspondences, avoiding manual layer mapping.
    - **Design Motivation**: When the teacher has 12 layers and the student has 6 layers, simple mapping schemes like 1-2, 3-4 might not be optimal. Automatic alignment can adapt to different compression ratios.

3. **Dynamics-Aware Distillation Loss**:

    - **Function**: Integrating feature dynamics information into the distillation training objectives.
    - **Mechanism**: The loss function consists of three parts: $L = L_{task} + \alpha L_{logit} + \beta L_{dynamics}$, where $L_{dynamics}$ employs cosine similarity or MSE to measure the discrepancy between the feature dynamics of the teacher and student. A dynamic weight adjustment mechanism may also be introduced, which focuses more on feature alignment in the early stages of training and shifts towards task performance later.
    - **Design Motivation**: A single loss signal is insufficient to comprehensively transfer knowledge; a combination of multiple signals can constrain the student model from different angles.

### Loss & Training
The overall loss is a weighted sum of the task loss, logit distillation loss, and feature dynamics loss. The training strategy may employ staged distillation, first aligning the feature dynamics and then fine-tuning for task performance.

## Key Experimental Results

### Main Results

| Task | Model | Teacher | Student (KD Baseline) | Ours | Gain |
|------|------|------|------------|---------|------|
| SST-2 | BERT→DistilBERT | 93.2 | 91.3 | 92.5 | +1.2 |
| MNLI | BERT→DistilBERT | 84.6 | 82.1 | 83.4 | +1.3 |
| QQP | BERT→DistilBERT | 91.1 | 89.5 | 90.6 | +1.1 |
| QNLI | BERT→DistilBERT | 91.7 | 89.2 | 90.8 | +1.6 |
| SQuAD | BERT→TinyBERT | 88.5 | 85.3 | 87.1 | +1.8 |

### Ablation Study

| Configuration | GLUE Average | Description |
|------|-----------|------|
| Full (logit+dynamics) | Optimal | Full method |
| Logit distillation only | Medium | Degenerates to standard KD |
| Feature dynamics alignment only | Medium-high | Does not use logit signals |
| Static feature alignment | Low | Traditional feature distillation |
| Manual layer mapping | Decreased | vs. Automatic alignment |

### Key Findings
- Feature dynamics alignment outperforms traditional static feature matching on all GLUE benchmark tasks, achieving an average improvement of around 1-2 percentage points.
- When the teacher-student compression ratio is larger (e.g., 12 to 4 layers), the advantage of dynamic alignment becomes more pronounced, indicating that dynamic knowledge is highly valuable under high compression rates.
- The information provided by feature dynamics and logit distillation is complementary, with their combination yielding the best performance.
- Automatic layer alignment is slightly superior to manually specified uniform mapping, though the gap is not as large as expected.

## Highlights & Insights
- **Dynamic vs. Static Perspective Shift**: The cognitive shift from "aligning what features are" to "aligning how features change" is ingenious. This paradigm can be extended to other scenarios requiring knowledge alignment, such as model merging and model aggregation in federated learning.
- **High Practicality**: The proposed method does not require changes to the model architecture, only requiring modifications to the loss function, and can be easily integrated into existing distillation pipelines.

## Limitations & Future Work
- The computation of feature dynamics increases memory and computational overhead during training.
- The effectiveness on LLM distillation (e.g., distilling GPT to smaller models) has not been verified yet, where models have more layers and more complex dynamics.
- The formal definition of "feature dynamics" can be further explored—currently, it is mainly based on feature differences, and whether there are better dynamic representation methods remains to be studied.
- It can be combined with curriculum learning to order training samples based on the complexity of feature dynamics.

## Related Work & Insights
- **vs. TinyBERT**: TinyBERT utilizes static intermediate layer matching; the dynamic alignment proposed in this paper serves as an upgrade to its methodology.
- **vs. DistilBERT**: DistilBERT only employs logit distillation; this work demonstrates that incorporating feature dynamic information yields significant improvements.
- **vs. PKD (Patient Knowledge Distillation)**: PKD extracts knowledge from multiple layers but remains static, whereas this paper introduces the dynamic dimension.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The concept of "feature dynamics" is a novel contribution to the field of distillation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple tasks, multiple compression ratios, and comprehensive ablation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear elaboration of motivation.
- **Value**: ⭐⭐⭐⭐ Holds practical guidance for NLP model distillation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Knowledge Distillation with Refined Logits](../../ICCV2025/model_compression/knowledge_distillation_with_refined_logits.md)
- [\[AAAI 2026\] Distillation Dynamics: Towards Understanding Feature-Based Distillation in Vision Transformers](../../AAAI2026/model_compression/distillation_dynamics_towards_understanding_feature-based_di.md)
- [\[ACL 2025\] Data Laundering: Artificially Boosting Benchmark Results through Knowledge Distillation](data_laundering_artificially_boosting_benchmark_results_through_knowledge_distil.md)
- [\[ACL 2025\] Flipping Knowledge Distillation: Leveraging Small Models' Expertise to Enhance LLMs in Text Matching](flipping_kd_small_to_large.md)
- [\[ECCV 2024\] Improving Knowledge Distillation via Regularizing Feature Direction and Norm](../../ECCV2024/model_compression/improving_knowledge_distillation_via_regularizing_feature_direction_and_norm.md)

</div>

<!-- RELATED:END -->
