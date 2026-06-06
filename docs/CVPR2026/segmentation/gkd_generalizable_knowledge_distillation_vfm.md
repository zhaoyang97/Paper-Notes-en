---
title: >-
  [Paper Note] GKD: Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation
description: >-
  [CVPR 2026][Segmentation][Knowledge Distillation] This paper proposes the GKD framework, which distills compact student models with cross-domain generalization capability from VFMs via a multi-stage decoupled distillatio…
tags:
  - "CVPR 2026"
  - "Segmentation"
  - "Knowledge Distillation"
  - "Vision Foundation Models"
  - "Domain Generalizable Segmentation"
  - "DINOv2"
  - "Multi-stage Distillation"
date: 2026-05-08
content_hash: 46b73f324fe80ad0
---

# GKD: Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation

**Conference**: CVPR 2026  
**arXiv**: [2603.02554](https://arxiv.org/abs/2603.02554)  
**Code**: [https://github.com/Younger-hua/GKD](https://github.com/Younger-hua/GKD)  
**Area**: Semantic Segmentation / Knowledge Distillation / Domain Generalization  
**Keywords**: Knowledge Distillation, Vision Foundation Models, Domain Generalizable Segmentation, DINOv2, Multi-stage Distillation

## TL;DR

This paper proposes the GKD framework, which distills compact student models with cross-domain generalization capability from VFMs via a multi-stage decoupled distillation strategy (generic feature learning → frozen encoder → task head training) combined with a Query-based Soft Distillation (QSD) mechanism. GKD achieves an average mIoU gain of +10.6% under the F2L setting and +1.9% under the F2F setting.

## Background & Motivation

**Background**: Knowledge distillation (KD) is widely used for semantic segmentation model compression—distilling lightweight student models from large teacher models. Conventional KD methods (CWD/Af-DCD/CIRKD, etc.) focus on preserving source-domain accuracy and perform well in-domain. The paradigm of VFMs (DINOv2/EVA02) as general-purpose feature extractors paired with lightweight decoders has been broadly adopted.

**Limitations of Prior Work**: Conventional KD focuses exclusively on source-domain (in-domain) accuracy, neglecting cross-domain generalization capability. This problem is particularly severe in the VFM era—although VFMs inherently possess strong generalization, student models distilled via conventional KD exhibit degraded generalization. Experiments show that single-stage KD can even **harm** student generalization, with some methods performing worse than the no-distillation baseline.

**Key Challenge**: Single-stage KD suffers from **optimization conflict**—the task loss drives the student to fit source-domain-specific decision boundaries, while the distillation loss encourages the student to approximate the teacher's domain-invariant representations. These two gradient directions are contradictory, leading to training instability (oscillating loss curves) and generalization degradation. This implies that "KD compresses capacity while compromising robustness."

**Goal**: When distilling compact models from VFMs, the goal is to simultaneously compress the model while **preserving or even improving** cross-domain generalization. Two evaluation settings are considered: F2F (VFM→smaller VFM, e.g., DINOv2-L→DINOv2-B) and F2L (VFM→local model, e.g., DINOv2-B→ViT-S).

**Key Insight**: Representation learning and task learning **should not be coupled**. The student should first purely learn the teacher's domain-general representations (without exposure to task labels), after which the encoder is frozen and only the task head is trained.

**Core Idea**: Decouple representation learning from task learning—Stage 1 performs pure feature distillation to acquire domain-general representations; Stage 2 freezes the encoder and trains the task head, complemented by QSD for selective retrieval of the teacher's spatial knowledge.

## Method

### Overall Architecture

The framework follows a two-stage pipeline. **Stage 1 (Domain-General Distillation)** consists of two steps: (i) task-agnostic distillation on a proxy dataset (ImageNet) to close the initial representation gap between the VFM and the student, and (ii) domain-agnostic distillation on the source domain to learn task-relevant but domain-invariant features. Throughout Stage 1, only feature distillation is performed and task labels are never used. **Stage 2 (Task Learning)** freezes the student encoder and trains only the Mask2Former decoder for semantic segmentation, preventing task supervision from corrupting the learned generalizable representations.

### Key Designs

1. **Multi-stage Decoupling Strategy**

    - **Function**: Completely separates feature distillation and task learning, which are typically coupled.
    - **Mechanism**: Stage 1 proceeds in two steps—(i) distillation on ImageNet (proxy dataset): $\min_{\theta_s} \mathbb{E}_{x_P \sim D_P}[\mathcal{L}_{QSD}(\mathcal{F}_{\theta_t}(x_P), \mathcal{F}_{\theta_s}(x_P))]$, learning task-agnostic general visual representations; (ii) distillation on the source domain: $\min_{\theta_s} \mathbb{E}_{x_S \sim D_S}[\mathcal{L}_{QSD}(\mathcal{F}_{\theta_t}(x_S), \mathcal{F}_{\theta_s}(x_S))]$, learning domain-invariant task-relevant features. Stage 2 freezes the encoder $\theta_s$ and trains only the decoder $\theta_h$: $\min_{\theta_h} \mathbb{E}[\mathcal{L}(\mathcal{H}_{\theta_h}(\mathcal{F}_{\theta_s}(x_S)), y_S)]$.
    - **Design Motivation**: Experimental diagnostics reveal that task gradients and distillation gradients interfere with each other—the single-stage loss curve oscillates and is unstable (Fig. 3b), whereas the two-stage approach yields a smooth converging loss curve. Ablations confirm: single-stage MSE 46.4 → two-stage MSE 53.1 (+6.7 mIoU), a substantial improvement.

2. **Query-based Soft Distillation (QSD)**

    - **Function**: Replaces conventional point-wise feature matching with selective spatial knowledge retrieval.
    - **Mechanism**: Student features $v_s \in \mathbb{R}^{B \times N \times C_s}$ serve as queries to retrieve all spatial features $v_t$ from the teacher via attention—attention weights are computed as $W = \varphi(v_s) \cdot v_t^\top$, student features are reconstructed as $v_s' = \sigma(\varphi(v_s) \cdot v_t^\top) \cdot \phi(v_s)$, and MSE alignment is applied: $\mathcal{L}_{feat} = \|v_s' - v_t\|_2^2$, where $\varphi, \phi$ are linear projections. This allows the student to internalize the teacher's **spatial relational structure** rather than merely mimicking local activations—the attention matrix exhibits strong diagonal responses (preserving spatial correspondence) alongside off-diagonal responses (selectively aggregating related semantics).
    - **Design Motivation**: The key advantage of VFMs lies in their domain-invariant spatial structure (confirmed by PCA visualization). Point-wise MSE aligns only local values while ignoring global relationships. QSD enables the student to selectively acquire the teacher's relational structure through attention, rather than mechanically memorizing local activations.

3. **Triple Distillation Objective**

    - **Function**: Comprehensively distills knowledge from the teacher at three levels: features, masks, and global semantics.
    - **Mechanism**: $\mathcal{L}_{QSD} = \alpha \mathcal{L}_{feat} + \beta \mathcal{L}_{mask} + \gamma \mathcal{L}_{cls}$. $\mathcal{L}_{feat}$ performs spatial feature distillation on complete inputs; $\mathcal{L}_{mask}$ reconstructs complete teacher features from randomly masked inputs (revealing VFM's latent knowledge, analogous to DINOv2's MIM approach); $\mathcal{L}_{cls}$ distills the CLS token to transfer global semantic information. All three weights default to 1.0.
    - **Design Motivation**: The three objectives are complementary—masked distillation forces the student to infer global context from partial information, while CLS distillation enforces global semantic consistency.

### Loss & Training

Distillation stage: AdamW, lr=5e-4, weight decay 0.05. F2L setting: 100 epochs on ImageNet (batch 512, 224×224) + 300 epochs on source domain (batch 128, 512×512). F2F setting: 300 epochs directly on source domain. Task stage: Mask2Former, lr=1e-5 (frozen backbone) / 1e-4 (decoder), 40K iterations, batch 4, crop 512×512.

## Key Experimental Results

### Main Results — F2L Setting (DINOv2-B → ViT-S)

| Method | GTAV→Citys | GTAV→BDD | GTAV→Map | Avg | Gain |
|--------|-----------|---------|---------|-----|------|
| Stu baseline (DeiT-S) | 34.9 | 33.8 | 42.8 | 37.2 | - |
| +Vanilla KD | 45.0 | 44.2 | 49.9 | 46.4 | +9.2 |
| +G2SD | 45.2 | 45.9 | 52.3 | 47.8 | +10.6 |
| +Proteus | 47.4 | 44.6 | 50.2 | 47.4 | +10.2 |
| **+GKD** | **54.9** | **49.8** | **57.8** | **54.1** | **+16.9** |

### Ablation Study (GTAV→Citys+BDD+Map Avg, DINOv2-B→ViT-S)

| Configuration | mIoU | Note |
|---------------|------|------|
| Single-stage MSE | 46.4 | Conventional KD baseline |
| Two-stage MSE | 53.1 | +6.7, confirms decoupling is critical |
| Two-stage QSD | 54.1 | +1.0, QSD outperforms MSE |
| Single-stage QSD | 48.8 | Even with QSD, single-stage is far inferior to two-stage |
| w/o $\mathcal{L}_{mask}$ | 53.5 | Masked distillation contributes +0.6 |
| w/o $\mathcal{L}_{cls}$ | 54.0 | CLS distillation contributes marginally +0.1 |

### Key Findings

- **Multi-stage decoupling is the dominant contribution**: Single-stage → two-stage yields +6.7 mIoU, far exceeding any gain from distillation method improvements alone.
- **Remarkable label efficiency under F2L**: GKD with only 1/16 of labels achieves 51.4 mIoU, surpassing Af-DCD trained with full labels (47.1).
- **Effective under F2F as well**: DINOv2-L→DINOv2-B Avg improves from 58.8 to 59.8 (+1.0); DINOv2-B→DINOv2-S from 53.9 to 55.6 (+1.7).
- PCA visualization confirms that after GKD distillation, the student's spatial feature structure is highly consistent with the DINOv2 teacher.

## Highlights & Insights

- **First systematic diagnosis of the generalization bottleneck in KD**: The finding that conventional KD can degrade student generalization is itself of significant value. All prior KD work focused exclusively on source-domain accuracy.
- **Multi-stage decoupling is simple yet effective**: The principle of learning generic features first → freezing the encoder → training the task head is conceptually clean and experimentally well-validated. This paradigm generalizes to any VFM downstream adaptation scenario.
- **Substantial advantage in the F2L setting**: A +10.6% average improvement implies that small ImageNet-pretrained models can nearly match VFM-level generalization capability.
- **Practical significance of label efficiency**: Achieving better performance with 1/16 of the labels compared to conventional KD with full labels has major implications for real-world deployment scenarios with limited annotation resources.

## Limitations & Future Work

- The additional ImageNet pre-distillation stage (100 epochs) increases training time and computational cost.
- Only ViT-based architectures are evaluated; whether CNN-based student models (ResNet/MobileNet) benefit from GKD remains unknown.
- Freezing the encoder during task learning may impose an upper bound on source-domain accuracy—in practice, GKD's source-domain accuracy (GTAV mIoU) is sometimes lower than that of conventional KD.
- Validation is limited to semantic segmentation; more complex tasks such as panoptic segmentation, instance segmentation, and object detection remain to be explored.
- The reasons behind differences in generalization transfer efficiency across different VFM teachers (DINOv2 vs. EVA02) are not analyzed in depth.

## Related Work & Insights

- **vs. Conventional Segmentation KD (CWD/Af-DCD/CIRKD)**: These methods focus solely on source-domain accuracy and comprehensively underperform GKD in cross-domain evaluation, with some even falling below the no-distillation baseline.
- **vs. VFM Distillation (G2SD/Proteus/TinyMIM)**: These methods adopt a "general→specific" paradigm in which the task learning stage remains coupled with distillation. GKD adopts a "general→frozen→task" paradigm that fully isolates distillation from task learning.
- **vs. DGSS Methods (FisherTune/CrossEarth)**: GKD addresses generalization from a distillation perspective, which is complementary to domain generalization methods.
- The principle of "decoupling representation learning and task learning during distillation" is generalizable to all VFM downstream adaptation scenarios—linear probing is essentially the same idea of freezing the encoder.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Multi-stage decoupling is not entirely new, but QSD and the generalization-oriented distillation diagnostic perspective are novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Five benchmarks, dual F2F/F2L settings, multiple VFMs, label efficiency analysis, and multi-source domain extension—exceptionally comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The logical chain from motivation diagnosis → method design → validation is seamless; the loss curve comparison in Fig. 3 is intuitive and compelling.
- **Value**: ⭐⭐⭐⭐⭐ Addresses an overlooked generalization problem in VFM distillation with important practical guidance for real-world deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CrossEarth-SAR: A SAR-Centric and Billion-Scale Geospatial Foundation Model for Domain Generalizable Semantic Segmentation](crossearthsar_a_sarcentric_and_billionscale_geospa.md)
- [\[AAAI 2026\] Causal-Tune: Mining Causal Factors from Vision Foundation Models for Domain Generalized Semantic Segmentation](../../AAAI2026/segmentation/causal-tune_mining_causal_factors_from_vision_foundation_mod.md)
- [\[CVPR 2026\] Seeing Through the Tool: A Controlled Benchmark for Occlusion Robustness in Foundation Segmentation Models](occsam_bench_occlusion_robustness_segmentation.md)
- [\[CVPR 2026\] Brewing Stronger Features: Dual-Teacher Distillation for Multispectral Earth Observation](brewing_stronger_features_dual-teacher_distillation_for_multispectral_earth_obse.md)
- [\[ICML 2026\] Geometry-Preserving Unsupervised Alignment for Heterogeneous Foundation Models](../../ICML2026/segmentation/geometry-preserving_unsupervised_alignment_for_heterogeneous_foundation_models.md)

</div>

<!-- RELATED:END -->
