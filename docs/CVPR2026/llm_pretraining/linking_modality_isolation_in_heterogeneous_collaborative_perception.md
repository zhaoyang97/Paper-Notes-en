---
title: >-
  [Paper Note] Linking Modality Isolation in Heterogeneous Collaborative Perception
description: >-
  [CVPR 2026][LLM Pretraining][Collaborative Perception] CodeAlign constructs a discrete code space via codebooks and cross-modal Feature-Code-Feature (FCF) translation…
tags:
  - "CVPR 2026"
  - "LLM Pretraining"
  - "Collaborative Perception"
  - "Heterogeneous Alignment"
  - "Modality Isolation"
  - "Codebook"
  - "Cross-Modal Translation"
date: 2026-05-08
content_hash: 52ed408ecb00533b
---

# Linking Modality Isolation in Heterogeneous Collaborative Perception

**Conference**: CVPR 2026  
**arXiv**: [2603.00609](https://arxiv.org/abs/2603.00609)  
**Code**: [cxliu0314/CodeAlign](https://github.com/cxliu0314/CodeAlign)  
**Area**: Pretraining  
**Keywords**: Collaborative Perception, Heterogeneous Alignment, Modality Isolation, Codebook, Cross-Modal Translation

## TL;DR

CodeAlign constructs a discrete code space via codebooks and cross-modal Feature-Code-Feature (FCF) translation, becoming the first framework to solve the "modality isolation" problem in heterogeneous collaborative perception—where different modalities never co-occur in training data—using only 8% of HEAL's training parameters with 1024x communication reduction while achieving SOTA perception performance.

## Background & Motivation

1. **Value of collaborative perception**: Multi-agent systems (e.g., connected autonomous vehicles) can build more comprehensive environmental understanding by sharing perception information, compensating for single-vehicle blind spots and occlusions
2. **Heterogeneity problem**: In practice, vehicles from different manufacturers carry different sensor types (LiDAR/Camera), configurations (64-beam/32-beam), and perception models, creating significant domain gaps for feature-level fusion
3. **Ubiquity of modality isolation**: Different institutions collect data at different locations and times, causing many modality pairs to never co-occur in the same scene—e.g., Institution A has only LiDAR data, Institution B has only Camera data, with no spatially overlapping observations
4. **Dependencies and limitations of existing methods**: HEAL requires expensive encoder retraining; STAMP/GT-Space depend on co-occurrence data with spatial correspondence supervision or shared FoV; HMViT and Pyramid Fusion require joint training and suffer severe performance degradation under modality isolation (AP70 drops by 15.21%)
5. **Efficiency bottleneck**: Intermediate fusion methods transmit dense feature maps, incurring massive communication overhead (32MB per exchange), limiting practical deployment
6. **Privacy constraints**: Data from different institutions is subject to privacy regulations, preventing direct sharing of raw data and further complicating cross-modal alignment

## Method

### Overall Architecture: CodeAlign

CodeAlign comprises two training stages and one inference pipeline:

**Stage 1: Code Space Construction**

- A lightweight adapter (4-layer ResNet blocks, 3×3 convolutions) and a learnable codebook (size $D=16$) are inserted between each modality's encoder and backend
- Encoders and backends are frozen; only adapters and codebooks are trained
- Each spatial position in the BEV feature map is mapped to codebook indices via nearest-neighbor quantization: $I_{[h,w]} = \arg\min_\ell \| (\mathcal{P}(F))_{[h,w]} - C[\ell] \|_2^2$
- Communication transmits only codebook index maps ($H \times W \times \log_2(D)$), achieving ~1024x compression compared to raw features ($H \times W \times C$)

**Group Code Space Construction**: Non-isolated modalities share a single codebook, enabling features from different modalities representing the same object to map to the same codebook embedding, achieving natural alignment while reducing the number of cross-modal translators needed.

**Stage 2: FCF Translation (Feature-Code-Feature)**

- **Feature→Code**: Cross-modal translator $T_{m_i \to m_j}$ maps the source modality's dense features to index maps in the target modality's codebook
- **Code→Feature**: Target modality's reconstructor $R_{m_j}$ decodes index maps back into dense features that naturally reside in the target modality's feature space
- The dense-to-code scheme is chosen (rather than dense-to-dense or code-to-code), balancing reconstruction accuracy with communication efficiency

**One-to-Many Code Translator**:

- Shared backbone (stacked ConvNeXt blocks) + modality-specific multi-head outputs
- Training parameters grow linearly with the number of modalities ($(0.5M) \cdot n$), avoiding the quadratic growth of one-to-one approaches
- Data balancing strategy: dynamically adjusts training data ratios based on loss changes across different targets

### Loss & Training

$$L = L_{\text{det}}(\hat{\mathcal{O}}_i, \mathcal{O}_i^0) + L_{\text{pyramid}} + \lambda \sum_{k,j \in \mathcal{G}_s, m_k \neq m_j} L_{\text{sim}}(F_{k \to i}, F_{j \to i})$$

- $L_{\text{det}}$: Detection loss
- $L_{\text{pyramid}}$: Pyramid fusion loss (from HEAL)
- $L_{\text{sim}}$: Smooth L1 feature similarity loss ($\lambda=0.1$), encouraging cross-modal feature consistency

### Local Data Training Protocol

Only source modality local data is used: source modality encoding → translator → target backend detection loss, requiring no cross-institution data transfer and fully complying with data privacy requirements.

## Key Experimental Results

### OPV2V Dataset (Simulation, Multi-Vehicle V2V)

| Method | m1+m7+m2 AP30 | m1+m7+m2 AP50 | m1+m7+m2 AP70 | Params (M) | Comm. |
|--------|:---:|:---:|:---:|:---:|:---:|
| No Collaboration | 81.18 | 79.44 | 68.26 | 0 | 0 |
| Late Fusion | 88.24 | 85.02 | 68.45 | 0 | 0.5KB |
| Pyramid Fusion | 83.95 | 82.93 | 68.91 | 21.4 | 32MB |
| HEAL | 87.80 | 86.98 | 79.89 | 16.0 | 32MB |
| **CodeAlign** | **89.77** | **88.59** | **77.73** | **1.3** | **0.03MB** |

- CodeAlign surpasses HEAL by 1.97/1.61 points in AP30/AP50 in the three-modality scenario
- Training parameters are only 8% of HEAL's (1.3M vs 16.0M)
- Communication is reduced by 1024x (0.03MB vs 32MB)

### DAIR-V2X Dataset (Real World)

| Method | m1+m2 AP30 | m1+m2 AP50 | m1+m2 AP70 |
|--------|:---:|:---:|:---:|
| HEAL | 73.70 | 67.21 | 44.76 |
| **CodeAlign** | **82.03** | **77.37** | **57.84** |

- CodeAlign surpasses HEAL by 13.08 points in AP70 on real-world data, demonstrating stronger generalization

### Ablation Study

- **Modality isolation impact**: Pyramid Fusion's AP70 drops from 80.88% to 65.67% (−15.21%) under modality isolation
- **Group code space vs FCF translation**: For non-isolated modalities, group code space construction outperforms FCF translation by 6.71% AP70
- **Translator structure**: Multi-head translator loses only 0.10% AP50 compared to one-to-one, while parameters drop from $O(n^2)$ to $O(n)$
- **Codebook + frozen encoder + adapter + similarity loss**: Progressively introducing each component recovers AP70 from 77.87% to 79.63%
- **Pose error robustness**: CodeAlign consistently outperforms HEAL under pose perturbations, while Late Fusion rapidly degrades below the no-collaboration baseline

## Highlights & Insights

- **First co-occurrence-free alignment framework**: Solves modality isolation fundamentally through representation consistency instead of spatial correspondence
- **Extreme efficiency**: 8% training parameters + 1024x communication compression, friendly for large-scale deployment
- **Privacy-preserving**: Local data training protocol avoids cross-institution data transfer
- **Strong scalability**: One-to-many translator reduces new modality onboarding cost from $O(n^2)$ to $O(n)$
- **Plug-and-play design**: Freezes original encoders and backends, training only lightweight inserted modules

## Limitations & Future Work

- Information loss from codebook quantization causes AP70 to be slightly lower than HEAL in some scenarios (e.g., m1+m2: 85.56 vs 86.18)
- Codebook size is fixed at 16; smaller codebooks may not sufficiently represent complex scenes
- Evaluation is limited by modality diversity in existing datasets; not validated on large-scale multi-modality (>7 types) scenarios
- Online codebook updating and adaptation mechanisms in dynamic scenes are unexplored
- BEV spatial range is set to ±102.4m; applicability to ultra-long-range scenarios is unverified

## Related Work & Insights

| Method | Supports Modality Isolation | Training | Comm. Efficiency | Core Mechanism |
|--------|:---:|------|------|------|
| HMViT | ✗ | Joint end-to-end | Low (32MB) | Cross-modal attention |
| CodeFilling | ✗ | Shared codebook E2E | High (0.03MB) | Single shared codebook |
| STAMP | ✗ | Contrastive learning | Low (32MB) | Protocol network reference |
| GT-Space | ✗ | GT feature alignment | Low | Ground-truth anchors |
| HEAL | △ (requires encoder retraining) | Backward alignment | Low (32MB) | Encoder retraining |
| **CodeAlign** | **✓** | **Local data training** | **High (0.03MB)** | **FCF translation + codebook** |

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First to define and systematically solve the modality isolation problem; the FCF translation approach is novel
- Experimental Thoroughness: ⭐⭐⭐⭐ — Simulation + real-world datasets with comprehensive multi-scenario ablations; modality variety is limited
- Writing Quality: ⭐⭐⭐⭐ — Clear problem definition and systematic method exposition; some symbols are redundantly defined
- Value: ⭐⭐⭐⭐⭐ — Addresses practical deployment pain points (privacy, efficiency, scalability) with significant engineering impact

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] CHAMMI-75: Pre-training multi-channel models with heterogeneous microscopy images](../../ICLR2026/llm_pretraining/chammi-75_pre-training_multi-channel_models_with_heterogeneous_microscopy_images.md)
- [\[CVPR 2026\] LottieGPT: Tokenizing Vector Animation for Autoregressive Generation](lottiegpt_vector_animation_generation.md)
- [\[CVPR 2026\] FlowMotion: Training-Free Flow Guidance for Video Motion Transfer](flowmotion_training-free_flow_guidance_for_video_motion_transfer.md)
- [\[CVPR 2026\] Evidential Transformation Network: Turning Pretrained Models into Evidential Models for Post-hoc Uncertainty Estimation](evidential_transformation_network_post_hoc_uncertainty_estimation.md)
- [\[CVPR 2026\] Defending Unauthorized Model Merging via Dual-Stage Weight Protection](defending_unauthorized_model_merging_via_dual-stage_weight_protection.md)

</div>

<!-- RELATED:END -->
