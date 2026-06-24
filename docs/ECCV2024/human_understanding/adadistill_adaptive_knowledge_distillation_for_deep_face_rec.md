---
title: >-
  [Paper Note] AdaDistill: Adaptive Knowledge Distillation for Deep Face Recognition
description: >-
  [ECCV 2024][Human Understanding][Knowledge Distillation] This paper proposes AdaDistill, which embeds the knowledge distillation concept into the margin penalty softmax loss. By utilizing EMA-based adaptive class centers (employing simple sample-to-sample knowledge in early stages and complex sample-to-center knowledge in later stages) and a hard sample-aware mechanism, it enhances the discriminative power of lightweight face recognition models without requiring extra hyperpa…
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "Knowledge Distillation"
  - "Face Recognition"
  - "Adaptive Class Centers"
  - "Hard Sample Mining"
  - "margin-based softmax"
date: 2026-05-08
content_hash: d91a95e912a3b998
---

# AdaDistill: Adaptive Knowledge Distillation for Deep Face Recognition

**Conference**: ECCV 2024  
**arXiv**: [2407.01332](https://arxiv.org/abs/2407.01332)  
**Code**: None  
**Area**: Model Compression / Knowledge Distillation / Face Recognition  
**Keywords**: Knowledge Distillation, Face Recognition, Adaptive Class Centers, Hard Sample Mining, margin-based softmax  

## TL;DR
This paper proposes AdaDistill, which embeds the knowledge distillation concept into the margin penalty softmax loss. By utilizing EMA-based adaptive class centers (employing simple sample-to-sample knowledge in early stages and complex sample-to-center knowledge in later stages) and a hard sample-aware mechanism, it enhances the discriminative power of lightweight face recognition models without requiring extra hyperparameters, outperforming SOTA distillation methods on challenging benchmarks such as IJB-B/C and ICCV21-MFR.

## Background & Motivation
SOTA face recognition models (such as ArcFace and CurricularFace) rely on deep networks with millions of parameters (e.g., ResNet50/100), making them difficult to deploy on mobile and edge devices. Knowledge distillation (KD) is an effective approach to improve the performance of lightweight models, but existing KD methods in the face recognition (FR) field present clear **Limitations of Prior Work**:

1. **ReFO (CVPR 2023)** requires the student to completely mimic the teacher's embedding space, which is difficult to achieve due to the limited capacity of shallow student architectures—additionally, it requires a student proxy and multi-stage training, which is complex and computationally expensive.
2. **MarginDistillation** utilizes the teacher's fixed class centers to train the student, but fixed centers are unsuitable across all training stages—the knowledge embedded in class centers is too complex for the student in the early stages of learning.
3. Most KD methods require simultaneous optimization of both the classification loss and distillation loss (such as $\lambda \mathcal{L}_{main} + \beta \mathcal{L}_{KD}$ in Vanilla KD), introducing additional loss weight hyperparameters.

The **Key Challenge** is that the limited capacity of the lightweight student model makes directly learning the teacher's full and complex knowledge (such as class centers or the complete embedding space) ineffective. Thus, a strategy that is tailored to the student's capacity and learning phase is required.

## Core Problem
How to adaptively and efficiently transfer discriminative knowledge (feature representations and class centers) from the teacher model to a lightweight student model considering the student-teacher capacity gap, without requiring multi-stage training or extra hyperparameters?

## Method

### Overall Architecture
Given a batch of face images, feature representations $f^t$ and $f^s$ are extracted concurrently through a frozen teacher model $T$ (ResNet50) and a student model $S$ (MobileFaceNet). Then, based on the teacher's features $f^t$, the adaptive class centers $w^{(k)}$ are computed using an EMA mechanism guided by the student's learning capability metric. Finally, a margin penalty softmax loss (ArcFace/CosFace) is utilized to compute the distance from $f^s$ to the adaptive class centers to train the student, updating only the student's weights.

The key innovation is that the class centers $w$ are no longer static (unlike MarginDistillation) nor does the student directly mimic teacher embeddings (unlike ReFO/MSE); instead, a progressive transition from simple knowledge (individual sample features) to complex knowledge (class centers) is established.

### Key Designs

1. **AMLDistill (Additive Margin Loss Distillation)**: This injects the distillation concept into the margin softmax loss—the student's feature $f^s_i$ is no longer compared with the student's own trained class centers, but with the teacher's class center $w^t_{y_i}$. The loss formulation is $\mathcal{L} = -\log \frac{e^{s(\cos(\theta^t_{y_i}+m_1)-m_2)}}{e^{s(\cos(\theta^t_{y_i}+m_1)-m_2)}+\sum_{j \neq y_i} e^{s\cos(\theta^t_j)}}$. This approach is more rational than MSE distillation, as class centers are more representative of identity than individual samples (the sample-center matching similarity is significantly higher than sample-sample similarity).

2. **Adaptive Class Centers via EMA**: **Core Idea**. In the early stages of training when the student's capability is weak ($\alpha \approx 0$), the class center $w_{y_i}$ is approximately equal to the current sample's teacher feature $f^t_i$, so the student only performs simple sample-to-sample matching. In the later stages as the student's capability increases ($\alpha \to 1$), the class center approaches the teacher's true class center $w^t_{y_i}$, and the student begins to perform complex sample-to-center matching. The update rule is: $w^{(k)}_{y_i} = \alpha \cdot w^{(k-1)}_{y_i} + (1-\alpha) \cdot (f^{t(k)}_i)^T$. The momentum parameter $\alpha$ is automatically determined by the positive cosine similarity between the student $f^s_i$ and the teacher $f^t_i$—the better the student mimics the teacher, the larger $\alpha$ becomes, leading to a more stable class center.

3. **Hard Sample Importance**: On top of $\alpha$, a hard sample weight is introduced: $\alpha' = \lfloor \text{Cos}(f^s_i, f^t_i) \times \text{Cos}(w^{(k-1)}_{y_i}, f^t_i) \rceil^1_0$. When the similarity between a sample's teacher feature $f^t_i$ and its corresponding class center is low (indicating a hard sample), $\alpha'$ decreases. This pulls the class center towards the hard sample direction, forcing the model to focus more on learning from hard samples.

### Loss & Training
- **Distillation loss only**: No additional classification loss or weight balancing ($\beta=0$) is required, avoiding the difficulties of multi-loss optimization.
- Supports both ArcFace ($m_1=0.45$ is optimal) and CosFace ($m_2=0.35$ is optimal) margins.
- Scale parameter $s=64$, consistent with standard FR training.
- SGD optimizer with an initial learning rate lr=0.1, divided by 10 at 80K/140K/210K/280K iterations.
- Batch size=512, trained on 4 RTX 6000 GPUs.
- Training speed: AdaDistill processes each batch in 0.247 seconds (vs 0.207 seconds for Vanilla KD), illustrating an overhead increase of about 20%.

## Key Experimental Results

| Dataset | Metric | AdaArcDistill | ReFO+ (CVPR23) | EKD (CVPR22) | MFN Student |
|--------|------|------|----------|------|------|
| IJB-C | TAR@FAR1e-4 | **93.27** | 92.41 | 90.48 | 89.13 |
| IJB-C | TAR@FAR1e-5 | **89.32** | 87.80 | 84.00 | 81.65 |
| IJB-B | TAR@FAR1e-4 | **91.21** | - | 88.35 | 87.07 |
| ICCV21-MFR Children | TAR@FAR1e-4 | **37.21** | 32.80 | 28.95 | 24.71 |
| ICCV21-MFR Mask | TAR@FAR1e-4 | **35.36** | 32.24 | 32.14 | 27.90 |
| Average of 5 Small Benchmarks | Acc | **95.43** | - | 95.03 | 94.01 |

Teacher ResNet50 (43.59M, 13.64GFLOPs) → Student MobileFaceNet (1.19M, 0.45GFLOPs), with parameters compressed by approximately 37x.

### Ablation Study
- **Adaptive class centers contribute the most**: Moving from ArcDistill (fixed centers) to AdaArcDistill ($\alpha$), the IJB-C TAR@FAR1e-5 increases from 84.57% to 88.23% (+3.66%), demonstrating that adaptive centers are crucial to bridging the capacity gap between teacher and student.
- **Hard sample mining is effective**: Using $\alpha'$ instead of $\alpha$ improves performance from 88.23% to 89.13% (+0.9%), showing that hard sample awareness further enhances discriminative capacity.
- **Choice of teacher architecture**: ResNet50 performs best as the teacher. Excessively strong teachers (ResNet100, TransFace-B) paradoxically lead to a drop in distillation performance—supporting the viewpoint that an overly large capacity gap is detrimental to distillation.
- **Supports Identity-disjoint training**: When training the teacher on MS1MV2 and the student on CASIA-WebFace, there is still a significant improvement (from 92.25 to 94.81 on average), indicating that AdaDistill does not require the teacher and student to share the training set.
- **Low sensitivity to margin value**: The results vary minimally (<0.1% average change) for ArcFace $m=0.40/0.45/0.50$, indicating good robustness.

## Highlights & Insights
- **Simple and elegant adaptive mechanism**: Using the student-teacher cosine similarity as the EMA momentum dynamically implements a "curriculum learning" effect that transitions from simple to complex, introducing no additional hyperparameters—$\alpha$ is completely determined by training dynamics.
- **Single loss design**: It avoids the tedious task of loss weight tuning by eliminating the need to optimize classification loss and distillation loss simultaneously.
- **Insight that class centers are more representative than single samples**: The matching score distribution for sample-center is significantly higher than that for sample-sample, providing solid empirical justification for integrating class centers into distillation.
- **Does not require the teacher and student to share the training set**: Because class centers are dynamically estimated from teacher features via EMA (rather than taking weights directly from the teacher's classification layer), the student can be trained on entirely different datasets—or even synthetic data.

## Limitations & Future Work
- **Significant gap remaining after compression**: Even with the best distillation method, MobileFaceNet (1.19M) achieves 93.27% on IJB-C vs the teacher's 96.05% (a gap of ~3%), which might be insufficient for safety-critical applications.
- **Validated only on a single student architecture (MobileFaceNet)**: The performance on other lightweight architectures (such as EfficientNet or ShuffleNet) remains unexplored.
- **No consideration for feature dimension alignment**: If the embedding dimensions of the teacher and student differ, an extra projection layer would be required, though both teacher and student models output 512 dimensions in this paper.
- **Optimality of teacher selection**: Although ResNet50 outperforms ResNet100 and TransFace-B, a deep theoretical analysis explaining this phenomenon is lacking.

## Related Work & Insights

| Aspect | ReFO (CVPR2023) | MarginDistillation | AdaDistill |
|------|---------|---------|---------|
| Core Idea | Mimicking teacher's embedding space | Training student with teacher's fixed class centers | Adaptive class center distillation |
| Class Center | N/A (Direct feature alignment) | Fixed | Dynamic self-adaptation via EMA |
| Training Stage | Multi-stage (Requires student proxy) | Single-stage | Single-stage |
| Extra Hyperparameters | Required | None | None |
| Shared Training Data | Required | Required | Not Required |
| IJB-C 1e-4 | 92.41% | 85.71% | **93.27%** |

AdaDistill's advantages lie in its simplicity, absence of extra hyperparameters, single-stage training, and its performance comprehensively outperforming ReFO and MarginDistillation on large-scale challenging benchmarks.

## Insights & Connections
- **Scalable directions**: The adaptive EMA class center mechanism of AdaDistill can be extended to open-vocabulary distillation, replacing fixed text embeddings with class prototypes.
- **Generalizability of automatic $\alpha$ adjustment**: Measuring the student's learning progress using the teacher-student feature similarity to automatically govern distillation difficulty is a versatile design that can be transferred to non-FR tasks (e.g., classification, detection, and segmentation distillation).

## Rating
- Novelty: ⭐⭐⭐⭐ The design where EMA momentum is automatically controlled by the student's learning capability is simple and novel, though the overall framework is still based on margin softmax.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive with 10 benchmarks, 4 teacher models, multiple margins, identity-disjoint experiments, and synthetic data experiments.
- Writing Quality: ⭐⭐⭐⭐ Generally clear; Figures 1, 3, and 4 intuitively demonstrate the motivation and design concepts.
- Value: ⭐⭐⭐⭐ Holds direct practical value for the deployment of lightweight models in the FR field, and the adaptive distillation pathway also presents generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] VideoClusterNet: Self-Supervised and Adaptive Face Clustering for Videos](videoclusternet_self-supervised_and_adaptive_face_clustering_for_videos.md)
- [\[ICCV 2025\] CleanPose: Category-Level Object Pose Estimation via Causal Learning and Knowledge Distillation](../../ICCV2025/human_understanding/cleanpose_category-level_object_pose_estimation_via_causal_learning_and_knowledg.md)
- [\[ECCV 2024\] Adaptive High-Frequency Transformer for Diverse Wildlife Re-Identification](adaptive_highfrequency_transformer_for_diverse_wildlife_reid.md)
- [\[ECCV 2024\] ADen: Adaptive Density Representations for Sparse-view Camera Pose Estimation](aden_adaptive_density_representations_for_sparseview_camera.md)
- [\[CVPR 2025\] CryptoFace: End-to-End Encrypted Face Recognition](../../CVPR2025/human_understanding/cryptoface_end-to-end_encrypted_face_recognition.md)

</div>

<!-- RELATED:END -->
