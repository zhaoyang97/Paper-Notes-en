---
title: >-
  [Paper Note] UniMMAD: Unified Multi-Modal and Multi-Class Anomaly Detection via MoE-Driven Feature Decompression
description: >-
  [CVPR2026][Multimodal VLM][Anomaly Detection] This paper proposes UniMMAD, the first unified framework that handles multi-modal and multi-class anomaly detection simultaneously with a single parameter set. The core contribution is an MoE-based feature decompression mechanism that adaptively decomposes general multi-modal encoded features into domain-specific unimodal reconstructions, achieving state-of-the-art performance across 9 datasets spanning 3 domains, 12 modalities, and 66 categories.
tags:
  - CVPR2026
  - Multimodal VLM
  - Anomaly Detection
  - Multi-Modal Fusion
  - Mixture-of-Experts
  - Feature Decompression
  - Unified Framework
  - Multi-Class Anomaly Detection
date: 2026-05-08
content_hash: 0d533e83c9757c6b
---

# UniMMAD: Unified Multi-Modal and Multi-Class Anomaly Detection via MoE-Driven Feature Decompression

**Conference**: CVPR2026
**arXiv**: [2509.25934](https://arxiv.org/abs/2509.25934)
**Code**: [yuanzhao-CVLAB/UniMMAD](https://github.com/yuanzhao-CVLAB/UniMMAD)
**Area**: Multi-Modal VLM
**Keywords**: Anomaly Detection, Multi-Modal Fusion, Mixture-of-Experts, Feature Decompression, Unified Framework, Multi-Class Anomaly Detection

## TL;DR

This paper proposes UniMMAD, the first unified framework that handles multi-modal and multi-class anomaly detection simultaneously with a single parameter set. The core contribution is an MoE-based feature decompression mechanism that adaptively decomposes general multi-modal encoded features into domain-specific unimodal reconstructions, achieving state-of-the-art performance across 9 datasets spanning 3 domains, 12 modalities, and 66 categories.

## Background & Motivation

**Severe fragmentation of existing methods**: Current anomaly detection methods treat modalities and categories as independent factors, requiring separately trained specialized models for different modality combinations, leading to difficult deployment and large memory overhead.

**Shared decoder bottleneck in multi-class methods**: Multi-class methods such as UniAD and MambaAD employ shared decoding pathways, but face distorted normality boundaries when dealing with large cross-domain variations (differences in appearance, illumination, scale, background, etc.), resulting in severe domain interference and high false positive rates.

**Multi-sensor collaboration required in industrial scenarios**: In real-world product quality inspection, different products require different sensor combinations (infrared cameras for internal damage, RGB+3D for color and geometric defects), making it impractical to design customized models for each combination.

**Inspiration from unified visual models**: Models such as SegGPT and Spider have demonstrated the feasibility of a single architecture handling multiple tasks, motivating the transfer of this paradigm to the anomaly detection domain.

**Domain heterogeneity challenge**: In multi-modal multi-class scenarios, appearance, illumination, scale, and anomaly semantics vary drastically, making consistent representation learning and anomaly discrimination extremely challenging.

**Efficiency and continual learning requirements**: A practical unified AD model must achieve high accuracy, fast inference, and sparse computation, while adapting to new categories/modalities without catastrophic forgetting.

## Method

### Overall Architecture: "General → Specific" Paradigm

The core idea of UniMMAD is to **decompress** multi-modal features into multiple unimodal features: $f^{\text{gen}} \rightarrow \{u^m\}_{m=1}^M$. The model learns to predict the residual between $f^{\text{gen}}$ and each $u^m$ on normal samples. During inference, decompression fails in anomalous regions, and the resulting deviation serves as the anomaly score. This asymmetric design naturally avoids the shortcut reconstruction problem.

### General Multi-Modal Encoder

- **Input embedding layer**: Pads arbitrary modal inputs to a unified channel dimension $C$, supporting arbitrary modality combinations.
- **Residual blocks**: Three residual blocks progressively extract multi-modal features, refined by inter-modal prior averaging.
- **Feature Compression Module (FCM)**: Adopts a hierarchical bottleneck structure. An internal multi-scale bottleneck uses parallel $1\times1$, $3\times3$, and $5\times5$ convolutions to preserve normal patterns while suppressing scale-sensitive anomalies; an external bottleneck performs finer-grained compression at a higher semantic level, outputting clean general features $f_1^{\text{gen}}, f_2^{\text{gen}}, f_3^{\text{gen}}$.

### Cross Mixture-of-Experts (C-MoE)

**Condition Router**:
- General features are projected as keys/values; domain priors are projected as queries.
- Convolution + global average pooling yields global statistics $g_l^m$, encapsulating domain-specific context and suppressing anomaly leakage.
- The gating function produces top-K expert indices and scores, accompanied by an annealing-style load balancing loss $\mathcal{L}_{\text{MoE}}$ that encourages broad activation early in training and stable routing later.

**Expert Design and Routing**:
- **Fixed experts**: Capture shared knowledge and reduce redundancy.
- **Routed experts**: Selected via top-K gating to provide task-specific capabilities.
- **MoE-in-MoE structure**: Each routed expert (MoE-Leader) is a weighted combination of shared base experts $W \in \mathbb{R}^{N_{\text{exp}} \times O \times I \times K_s \times K_s}$; MoE-Leaders store only combination weights $S \in \mathbb{R}^{N_{\text{exp}} \times O}$, **reducing parameter count by approximately 75%**.
- **Grouped dynamic filtering**: The value tensor is replicated and reshaped with groups $= K_{\text{route}}+1$, enabling all expert filtering operations to be executed in parallel via a single grouped convolution, substantially reducing latency.

### Loss & Training

- **Decompression Consistency Loss** $\mathcal{L}_{\text{DeC}}$: Measures the deviation between decompressed features and original unimodal features using negative cosine similarity, with a focal loss modulation factor $\gamma=2$ to enhance attention to minority classes.
- **Total loss**: $\mathcal{L} = \mathcal{L}_{\text{DeC}} + \mathcal{L}_{\text{MoE}}$, optimized end-to-end.

## Key Experimental Results

### Main Results

Comprehensive evaluation across 9 datasets covering industrial (MVTec-3D, Eyecandies, MulSen-AD), medical (BraTs, UniMed), and conventional industrial (MVTec-AD, VisA) scenarios:

| Dataset | Metric | Best Specialized Model | UniMMAD | Comparison |
|--------|------|------------|---------|------|
| MVTec-3D | AUC_I / AUC_P | 92.4 / 98.9 (CFM) | **92.5 / 99.1** | Surpasses specialized model |
| Eyecandies | AUC_I / AUC_P | 81.8 / 95.8 (CFM) | **85.6 / 96.9** | AUC_I +3.7% |
| MulSen-AD | AUC_I / AUC_P | 78.9 / 97.8 (TripleAD) | **85.5 / 97.9** | AUC_I +6.6% |
| BraTs | AUC_I / AUC_P | 91.8 / 95.7 (PatchCore+MMRD) | **95.8 / 97.5** | AUC_I +4.0% |
| UniMed | AUC_I / AUC_P | 96.1 / 92.7 (INP-Former) | **96.3 / 92.0** | Essentially on par |
| MVTec-AD | AUC_I / AUC_P | 99.2 / 98.2 (INP-Former) | **99.4 / 98.1** | AUC_I +0.2% |
| VisA | MF1_P | 44.4 (INP-Former) | **47.2** | +2.8% (complex multi-instance scenario) |

### Ablation Study

| Component | Mean AUC_I | Mean AUC_P | Mean MF1_P |
|------|-----------|-----------|-----------|
| Baseline | 75.6 | 86.6 | 28.5 |
| + FCM | 77.4 | 86.7 | 28.9 |
| + General→Specific | 84.3 | 96.1 | 37.1 |
| + C-MoE (full) | **91.1** | **96.7** | **42.9** |
| w/o Cross-condition | 85.1 | 95.7 | 37.9 |
| w/o Routed Experts | 85.4 | 96.0 | 37.8 |
| w/o Fixed Expert | 89.4 | 96.5 | 41.5 |
| w/o Multi-scale Exp. | 88.9 | 96.4 | 41.2 |

### Key Findings

- **The General→Specific paradigm contributes most**: Its introduction yields AUC_I +8.9% and AUC_P +10.9%, validating the effectiveness of asymmetric decompression.
- **C-MoE further delivers an average AUC_I improvement of 8.1%**; cross-condition routing and routed experts are the most critical designs.
- **Strong continual learning capability**: Fine-tuning less than 10% of parameters (MoE-leaders, condition router, aggregation convolution) achieves performance close to joint training on new tasks, with degradation on old tasks below 8%.
- **Clear advantages over generalist models (AdaCLIP, MVFA, AA-CLIP)**: UniMMAD outperforms across all datasets, with particularly large margins in multi-modal scenarios.

## Highlights & Insights

- **First unified multi-modal multi-class anomaly detection framework**: A single parameter set covers 3 domains, 12 modalities, and 66 categories, offering strong practical utility.
- **MoE-in-MoE parameter efficiency is elegantly designed**: Routed experts store only $N_{\text{exp}} \times O$ combination weights, reducing parameters by 75% while maintaining sparse activation and fast inference.
- **Grouped dynamic filtering accelerates inference**: Tensor reshaping and grouped convolution merge filtering across multiple experts into a single operation, yielding an efficient engineering implementation.
- **Annealing-style load balancing loss**: The $(1-e/E)^2$ decay coefficient realizes an "explore-then-stabilize" routing strategy, which is more principled than fixed weighting.
- **Exceptionally thorough experiments**: 9 datasets, detailed ablations, continual learning experiments, and qualitative analysis — a breadth rarely seen in the AD literature.

## Limitations & Future Work

- The prior generator relies on a WideResNet50 pretrained model; prior quality may be limited in non-natural image domains (e.g., industrial X-ray, certain medical modalities).
- Continual learning still requires mixing in 1% of old data and is not a fully replay-free scheme.
- Fixed resizing to 256×256 may cause information loss for minute defects requiring high-resolution localization.
- The scalability of 32 MoE-Leaders and 8 base experts in larger-scale scenarios has not been thoroughly validated.
- Pixel-level MF1_P scores remain relatively low overall (40–50%), indicating considerable room for improvement in fine-grained segmentation.

## Related Work & Insights

- **Multi-modal anomaly detection**: M3DM (CVPR2023) employs patch-level contrastive learning to fuse RGB and point clouds; CFM (CVPR2024) proposes lightweight cross-modal mapping; MMRD introduces normal modality for inverse distillation → UniMMAD replaces parameter-independent fusion with a unified encoder.
- **Multi-class anomaly detection**: UniAD (NeurIPS2022) pioneered the shared-model multi-class paradigm; ViTAD/MambaAD improve backbone architectures; INP-Former (CVPR2025) achieves the strongest single-modal multi-class performance → UniMMAD addresses domain interference in shared decoders via MoE.
- **MoE in vision**: V-MoE embeds MoE into ViT; DeepSeekMoE emphasizes parameter efficiency → UniMMAD's cross-condition routing and MoE-in-MoE are novel designs tailored for AD heterogeneity.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ (First unified multi-modal multi-class AD framework; both the General→Specific paradigm and C-MoE are novel designs)
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (9 datasets, 3 domains, 12 modalities, 66 categories, complete ablations + continual learning)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear structure, rich figures and tables, though notation is dense)
- **Value**: ⭐⭐⭐⭐⭐ (The unified framework directly benefits industrial AD deployment; MoE designs are transferable)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] BriMA: Bridged Modality Adaptation for Multi-Modal Continual Action Quality Assessment](brima_bridged_modality_adaptation_for_multi-modal_continual_action_quality_asses.md)
- [\[CVPR 2026\] VideoFusion: A Spatio-Temporal Collaborative Network for Multi-modal Video Fusion](videofusion_a_spatio-temporal_collaborative_network_for_multi-modal_video_fusion.md)
- [\[CVPR 2026\] Multi-Modal Image Fusion via Intervention-Stable Feature Learning](multi-modal_image_fusion_via_intervention-stable_feature_learning.md)
- [\[CVPR 2026\] Multi-Modal Representation Learning via Semi-Supervised Rate Reduction for Generalized Category Discovery](multi-modal_representation_learning_via_semi-supervised_rate_reduction_for_gener.md)
- [\[CVPR 2026\] MMR-AD: A Large-Scale Multimodal Dataset for Benchmarking General Anomaly Detection with MLLMs](mmrad_multimodal_anomaly_detection.md)

</div>

<!-- RELATED:END -->
