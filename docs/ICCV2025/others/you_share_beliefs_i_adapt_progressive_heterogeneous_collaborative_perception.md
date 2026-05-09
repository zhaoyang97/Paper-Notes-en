---
title: >-
  [Paper Note] You Share Beliefs, I Adapt: Progressive Heterogeneous Collaborative Perception
description: >-
  [ICCV2025][heterogeneous collaborative perception] This paper proposes PHCP, the first framework that addresses the domain gap in heterogeneous collaborative perception at inference time. By leveraging collaborating agents' pseudo labels for few-shot unsupervised domain adaptation, PHCP trains lightweight adapters via self-training to align feature spaces—requiring no joint training—and achieves near-SOTA (HEAL) performance on OPV2V with only a small number of unlabeled samples.
tags:
  - ICCV2025
  - heterogeneous collaborative perception
  - few-shot domain adaptation
  - self-training
  - pseudo labels
  - inference-time adaptation
date: 2026-05-08
content_hash: cbe1d8df82251fea
---

# You Share Beliefs, I Adapt: Progressive Heterogeneous Collaborative Perception

**Conference**: ICCV2025
**arXiv**: [2509.09310](https://arxiv.org/abs/2509.09310)
**Code**: [GitHub](https://github.com/sihaoo1/PHCP)
**Area**: Other
**Keywords**: heterogeneous collaborative perception, few-shot domain adaptation, self-training, pseudo labels, inference-time adaptation

## TL;DR
This paper proposes PHCP, the first framework that addresses the domain gap in heterogeneous collaborative perception at inference time. By leveraging collaborating agents' pseudo labels for few-shot unsupervised domain adaptation, PHCP trains lightweight adapters via self-training to align feature spaces—requiring no joint training—and achieves near-SOTA (HEAL) performance on OPV2V with only a small number of unlabeled samples.

## Background & Motivation

**Background**: Collaborative perception enables vehicles to share information via V2X communication, extending perceptual range and penetrating occlusions. Intermediate feature fusion has become the dominant paradigm, balancing accuracy and bandwidth.

**Heterogeneous Challenge**: In practice, autonomous vehicles from different manufacturers employ different sensor configurations and perception models, causing their encoded intermediate features to reside in distinct semantic spaces (domain gap). Directly fusing heterogeneous features severely degrades performance—experiments show that the direct fusion baseline achieves only ~53% AP.

**Limitations of Prior Work**:
- Methods such as MPDA, PnPDA, and HEAL align features by training adapters or constructing a unified feature space.
- **Key Pain Point**: Whenever a new agent joins a collaboration, joint training on a dataset is required before cooperation can begin.
- Pre-storing models for all potential collaborators is infeasible—these approaches lack scalability.

**Core Problem**: Can model parameters be dynamically adjusted at inference time to accommodate different collaborators, completely bypassing joint training?

## Method

### Problem Formulation: Few-Shot Unsupervised Domain Adaptation
- Heterogeneous collaborative perception is reformulated as a few-shot unsupervised domain adaptation problem.
- Objective: adapt the adapter $\Phi_{i \to ego}$ with a small number of unlabeled samples at inference time.
- Constraints: no labeled data, very few frames, real-time requirements.

### Feature Adapter Design
- Based on CBAM (Convolutional Block Attention Module).
- **Channel Attention Module (CAM)**: aligns channel-wise feature distribution discrepancies across different encoders.
- **Spatial Attention Module (SAM)**: focuses on differences in key spatial regions.
- Design Motivation: visualization analysis reveals systematic misalignment between PointPillars and SECOND encoder feature maps along both channel and spatial dimensions.
- Lightweight design mitigates overfitting under extremely limited training data.

### PHCP Collaboration Pipeline

**Stage I — Adapter Fine-tuning (first $k$ frames)**:
1. Upon establishing a collaboration, the agent transmits both intermediate features $\mathbf{F}_i$ and detection results for the first $k$ frames.
2. High-confidence agent predictions are used as pseudo labels, with confidence scores retained as soft labels.
3. A few-shot training set $\mathcal{D}_i = \{(d_1, p_1), \dots, (d_k, p_k)\}$ is constructed.
4. The fusion network and detection head are frozen; only the adapter $\Phi_{i \to ego}$ is fine-tuned.
5. Training runs for 20 epochs with a warmup + multi-step decay learning rate schedule.

**Stage II — Normal Inference**:
1. The agent transmits only intermediate features (identical to standard intermediate collaboration).
2. The ego vehicle transforms features using the trained adapter: $\mathbf{F}_i' = \Phi_{i \to ego}(\mathbf{F}_i)$.
3. Transformed features are fused and used for final prediction.

### Key Design Choices
- **Adapter-only fine-tuning**: when multiple agents collaborate simultaneously, each adapter is trained independently without interference.
- **Soft pseudo labels**: retaining confidence scores is more robust than one-hot encoding.
- **$k$ framing follows few-shot convention**: 1-shot, 5-shot, and 10-shot settings.

## Key Experimental Results

### Dataset & Setup
- OPV2V and OPV2V-H datasets (CARLA simulation).
- Two heterogeneous agent types: LP (PointPillars encoder) and LS (SECOND encoder).
- 16 scenes, each split into support/query sets.
- Metric: mSAP@IoU 0.3/0.5/0.7.

### vs. Direct Fusion Baseline

| Metric | Direct Fusion | PHCP | Gain |
|--------|--------------|------|------|
| mSAP@0.3 | 59.7 | 92.9 | **+33.2** |
| mSAP@0.5 | 59.5 | 92.4 | **+32.9** |
| mSAP@0.7 | 53.0 | 85.9 | **+32.9** |

### vs. Other Collaborative Perception Methods (mSAP@0.7)

| Method | mSAP@0.7 | Training Data |
|--------|---------|--------------|
| F-Cooper | 63.4 | Full labeled |
| CoBEVT | 72.0 | Full labeled |
| AttFusion | 77.3 | Full labeled |
| V2X-ViT | 82.8 | Full labeled |
| **PHCP (Ours)** | **87.1** | **Few unlabeled** |
| HEAL (SOTA) | 91.7 | Full labeled |

- PHCP outperforms all methods except HEAL while using only a small number of unlabeled samples.
- The gap with HEAL is only 4.6 points, despite HEAL requiring fully labeled data for training.

### Computational Cost

| Stage | Setting | Time | Memory |
|-------|---------|------|--------|
| Training | 1-shot | 1.49s | 1290MB |
| Training | 5-shot | 2.39s | 5604MB |
| Inference | — | 0.07s | 798MB |

- Training requires only 1.5–2.4 seconds (20 iterations) and is executed only once when a new collaboration is established.

### Ablation Study

#### Few-Shot Count
- 1-shot already yields ~50% AP improvement.
- Performance continues to improve with more shots, but with diminishing returns.
- 5-shot achieves a favorable cost-effectiveness trade-off.

#### Pseudo Label Quality

| Confidence Threshold | mSAP@0.7 | wSAP@0.7 |
|---------------------|---------|---------|
| 0.2 | 85.0 | 66.1 |
| **0.5** | **85.9** | **68.0** |
| 0.7 | 85.8 | 67.7 |
| soft | 85.4 | 67.0 |

- Final performance is relatively insensitive to pseudo label quality, attributed to SAM effectively focusing on object regions.

#### Heterogeneous vs. Homogeneous

| Method | mSAP@0.5 | mSAP@0.7 |
|--------|---------|---------|
| Direct Fusion (heterogeneous) | 59.5 | 53.0 |
| **PHCP (heterogeneous)** | **92.4** | **85.9** |
| SECOND homogeneous | 94.2 | 90.5 |
| PointPillars homogeneous | 95.8 | 93.1 |

- PHCP reduces the heterogeneous performance gap from 40+ points to fewer than 8 points.

## Highlights & Insights

- **Reframing the Problem**: redefining heterogeneous collaborative perception from "training-time domain adaptation" to "inference-time few-shot unsupervised domain adaptation" is itself a valuable contribution. Pre-training for every potential collaborator is genuinely impractical in deployment.
- **~50% Improvement from 1-Shot**: the domain gap, while large, is structurally regular—a lightweight attention adapter can substantially alleviate it without complex alignment strategies.
- **Isolation via Adapter-Only Training**: in multi-agent collaboration, independently trained adapters avoid the mutual interference that would arise from full-model fine-tuning, making the approach highly practical for engineering deployment.
- **Low Sensitivity to Pseudo Label Quality**: the spatial attention mechanism provides inherent robustness, reducing dependence on pseudo label precision.

## Limitations & Future Work

1. **Only LiDAR Encoders Validated**: the heterogeneity between PointPillars and SECOND is relatively limited; cross-modal settings (LiDAR vs. camera) remain unverified.
2. **Simulation Data Only**: OPV2V is built on the CARLA simulator; real-world noise, communication latency, and localization errors are not fully addressed.
3. **Fixed $k$ Frames**: the value of $k$ must be preset; no adaptive stopping criterion is provided (e.g., detecting adapter convergence).
4. **Unidirectional Adaptation**: the current design trains adapters only on the ego side; bidirectional adaptation is not considered.
5. **Limited Cross-Scene Generalization**: performance degrades noticeably on cross-scene evaluation in certain extreme scenarios.
6. **Communication Overhead**: Stage I requires transmitting both features and detection results simultaneously, doubling bandwidth demand (albeit only for $k$ frames).

## Related Work & Insights

- **HEAL (Lu et al.)**: establishes a unified feature space so that new agents need only align to the shared space. Achieves the best performance but requires full-data training—PHCP and HEAL are complementary.
- **TFA (Wang et al.)**: introduces a two-stage fine-tuning paradigm; PHCP extends this by restricting fine-tuning to adapters only.
- **CBAM (Woo et al.)**: the channel + spatial attention module is selected as the adapter backbone for its lightweight design and suitability for few-shot training.
- **F-Cooper / V2VNet / AttFusion**: classic collaborative perception methods that assume homogeneous models—PHCP relaxes this assumption.

## Rating
- Novelty: ⭐⭐⭐⭐ First to address heterogeneous collaborative perception at inference time; the problem formulation itself is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ 16 scenes with extensive ablations, but limited to the single OPV2V dataset.
- Writing Quality: ⭐⭐⭐⭐ Clear logical structure with intuitive illustrations.
- Value: ⭐⭐⭐⭐ High practical significance for deployment, but constrained to simulation environments and LiDAR encoders.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] I Am Big, You Are Little; I Am Right, You Are Wrong](i_am_big_you_are_little_i_am_right_you_are_wrong.md)
- [\[NeurIPS 2025\] Contextual Dynamic Pricing with Heterogeneous Buyers](../../NeurIPS2025/others/contextual_dynamic_pricing_with_heterogeneous_buyers.md)
- [\[NeurIPS 2025\] ResNets Are Deeper Than You Think](../../NeurIPS2025/others/resnets_are_deeper_than_you_think.md)
- [\[ICCV 2025\] From Easy to Hard: Progressive Active Learning Framework for Infrared Small Target Detection with Single Point Supervision](from_easy_to_hard_progressive_active_learning_framework_for_infrared_small_targe.md)
- [\[NeurIPS 2025\] Brain-Like Processing Pathways Form in Models With Heterogeneous Experts](../../NeurIPS2025/others/brain-like_processing_pathways_form_in_models_with_heterogeneous_experts.md)

</div>

<!-- RELATED:END -->
