---
title: >-
  [Paper Note] Mamba Goes HoME: Hierarchical Soft Mixture-of-Experts for 3D Medical Image Segmentation
description: >-
  [NeurIPS 2025][Medical Imaging][3D Medical Image Segmentation] This paper proposes Mamba-HoME, an architecture that integrates a Hierarchical Soft Mixture-of-Experts (HoME) with the Mamba SSM. Through a two-level token routing mechanism, it achieves local-to-global feature modeling and surpasses existing state-of-the-art methods on 3D medical image segmentation across CT, MRI, and ultrasound modalities, while maintaining linear computational complexity.
tags:
  - NeurIPS 2025
  - Medical Imaging
  - 3D Medical Image Segmentation
  - Mamba
  - Mixture-of-Experts
  - State Space Models
  - Hierarchical Routing
date: 2026-05-08
content_hash: ab60f98b73f6b96f
---

# Mamba Goes HoME: Hierarchical Soft Mixture-of-Experts for 3D Medical Image Segmentation

**Conference**: NeurIPS 2025
**arXiv**: [2507.06363](https://arxiv.org/abs/2507.06363)
**Code**: [github.com/gmum/MambaHoME](https://github.com/gmum/MambaHoME)
**Area**: Medical Imaging
**Keywords**: 3D Medical Image Segmentation, Mamba, Mixture-of-Experts, State Space Models, Hierarchical Routing

## TL;DR
This paper proposes Mamba-HoME, an architecture that integrates a Hierarchical Soft Mixture-of-Experts (HoME) with the Mamba SSM. Through a two-level token routing mechanism, it achieves local-to-global feature modeling and surpasses existing state-of-the-art methods on 3D medical image segmentation across CT, MRI, and ultrasound modalities, while maintaining linear computational complexity.

## Background & Motivation

3D medical image segmentation is a core task in computer-aided diagnosis and interventional treatment, requiring the processing of data from multiple modalities including CT, MRI, and ultrasound. Medical imaging data inherently exhibits a hierarchical structure: local pathologies (e.g., tumors) are nested within larger anatomical structures (e.g., organs), which in turn conform to global anatomical arrangement patterns. This local-to-global spatial hierarchy is critical for segmentation performance.

However, existing methods have notable shortcomings. CNNs offer linear complexity but suffer from limited receptive fields, making it difficult to capture global spatial patterns. Vision Transformers model long-range dependencies via global attention mechanisms, but their quadratic complexity makes them prohibitively expensive on high-resolution 3D data. Mamba (selective state space models) has recently emerged as a promising alternative, capturing long-range dependencies with linear complexity, yet SSMs do not inherently possess the ability to adaptively handle diverse local patterns. Mixture-of-Experts (MoE) enables efficient management of local patterns through dynamic routing, but the combination of SSMs' global efficiency with MoE's local adaptivity has been largely unexplored.

The core starting point of this paper is to design a hierarchical two-level Soft MoE (HoME): the first level routes tokens to experts within local groups to extract local features, while the second level globally aggregates cross-group information. This hierarchical design is seamlessly embedded into the Mamba architecture to enable efficient modeling of local-to-global spatial hierarchies in 3D medical images.

## Method

### Overall Architecture
Mamba-HoME adopts a U-shaped encoder-decoder architecture. The encoder consists of a stem layer followed by cascaded Mamba-HoME Blocks (Mamba-HoMEB), while the decoder restores resolution through upsampling and skip connections. The core innovation lies in the Mamba-HoMEB, which sequentially comprises: Gated Spatial Convolution (GSC) → Mamba layer → HoME layer, with residual connections throughout.

### Key Designs

1. **Hierarchical Soft Mixture-of-Experts (HoME)**:

   - **Function**: A two-level token routing layer that processes features first locally, then globally.
   - **Mechanism**: The input sequence is first partitioned into $G_i$ groups, each containing $K_i$ tokens. During the **grouped slot assignment** phase, tokens within each group are soft-assigned to expert slots via dot products with learnable slot embeddings $E_{\text{slots}}^{(i)} \in \mathbb{R}^{M_i \times d}$ followed by softmax normalization: $A_{b,g,k,m} = \frac{\exp(S_{b,g,k,m})}{\sum_{m'}\exp(S_{b,g,k,m'})}$. The first level of $E_{1,i}$ local experts (FFNs) processes intra-group slot representations via routing-weight-weighted aggregation; the second level of $E_{2,i}$ global experts processes the concatenated slots from all groups to enable cross-group information fusion.
   - **Design Motivation**: Global SMoE incurs a computational cost of $\mathcal{O}(N_i \cdot M_i)$ on long sequences; grouped routing reduces peak memory while preserving locality. The two-level design achieves local feature extraction combined with global context integration without significantly increasing computational overhead.

2. **Mamba-HoMEB (Mamba-HoME Block)**:

   - **Function**: Sequentially integrates the GSC, Mamba, and HoME modules.
   - **Processing Pipeline**: $x_i'^l = f_{\text{GSC}}(x_i^l)$, $\bar{x}_i^l = f_{\text{Mamba}}(f_{\text{Norm}}(x_i'^l)) + x_i'^l$, $x_i^{l+1} = f_{\text{HoME}}(f_{\text{Norm}}(\bar{x}_i^l)) + \bar{x}_i^l$. The GSC module first extracts local spatial priors; the flattened 1D sequence is then passed to the Mamba layer for long-range modeling; finally, the HoME layer performs hierarchical expert refinement.
   - **Design Motivation**: GSC captures local spatial priors, Mamba efficiently processes long sequences, and HoME provides hierarchical expert adaptivity — the three modules are complementary, forming a complete local-to-global modeling chain.

3. **Dynamic Tanh (DyT) Normalization**:

   - **Function**: An efficient normalization method that replaces Layer Normalization.
   - **Core Formula**: $f_{\text{DyT}}(x) = w \cdot \tanh(\alpha \cdot x) + b$, where $w, b \in \mathbb{R}^d$ are learnable parameters and $\alpha$ is a shared scalar.
   - **Design Motivation**: The bounded nature of tanh inherently stabilizes gradients, avoiding the mean/variance computation overhead of LN. This achieves approximately 6% acceleration in both training and inference within the SSM architecture without performance degradation.

### Loss & Training
The $\mathcal{L}_{\text{DiceCE}}$ loss function (Dice + Cross Entropy) is employed with the AdamW optimizer and an initial learning rate of 1e-4 with cosine annealing scheduling. Two configurations are supported: training from scratch and fine-tuning after supervised pre-training on large-scale CT/MRI datasets. Pre-training uses the AbdomenAtlas 1.1 (CT) and TotalSegmentator MRI datasets.

## Key Experimental Results

### Main Results

**PANORAMA + In-house CT (Pancreas Segmentation)**

| Method | PDAC DSC(%) | Pancreas DSC(%) | mDSC(%) | mHD95(mm) | GPU(G) |
|--------|-------------|-----------------|---------|-----------|--------|
| SegMamba | 49.7 | 88.5 | 76.0 | 8.6 | 10.1 |
| uC 3DU-Net | 52.0 | 88.2 | 76.6 | 8.4 | 13.6 |
| SuPreM* | 51.7 | 88.3 | 76.6 | 4.7 | 17.1 |
| **Mamba-HoME** | **54.8** | 88.3 | **77.5** | **4.8** | **11.1** |
| **Mamba-HoME*** | **56.7** | **88.5** | **78.2** | **4.3** | **11.1** |

**FeTA 2022 (Fetal Brain MRI)** — 5-Fold Cross-Validation

| Method | mDSC(%) | mHD95(mm) |
|--------|---------|-----------|
| SegMamba | 85.9 | 3.5 |
| Hermes | 86.5 | 4.0 |
| **Mamba-HoME*** | **87.7** | **2.0** |

**MVSeg (3D Ultrasound)**

| Method | mDSC(%) | mHD95(mm) |
|--------|---------|-----------|
| VSmTrans | 84.4 | 6.2 |
| Swin UNETR | 84.4 | 4.8 |
| **Mamba-HoME*** | **85.0** | **4.1** |

### Ablation Study

| Configuration | PANO mDSC | AMOS mDSC | FeTA mDSC | Notes |
|---------------|-----------|-----------|-----------|-------|
| E1=[4,8,12,16], E2=[8,16,24,32] | 77.5 | 86.3 | 87.5 | Optimal expert count configuration |
| E1=[8,16,24,48], E2=[8,16,24,48] | 76.3 | 86.2 | 87.2 | More experts does not help |
| K=[2048,1024,512,256] | 77.5 | 86.3 | 87.5 | Optimal group size |
| K=[1024,512,256,128] | 77.2 | 86.1 | 87.4 | Smaller groups degrade performance |
| Slots S=4 | 77.5 | 86.3 | 87.5 | Optimal slot count |
| Slots S=1 | 76.2 | 85.9 | 87.3 | Too few slots, insufficient information |
| Layer Norm | 77.4 | 86.2 | 87.5 | Conventional normalization |
| Dynamic Tanh | 77.5 | 86.3 | 87.4 | Comparable performance, ~6% faster |

### Key Findings
- Mamba-HoME achieves state-of-the-art performance across all three modalities (CT/MRI/US), demonstrating strong cross-modal generalizability.
- More experts are not necessarily better; the E1=[4,8,12,16] configuration achieves the best performance with the fewest parameters.
- The pre-trained variant further improves performance on all datasets, with particularly notable gains on the challenging small-target task of PDAC segmentation (+1.9% DSC).
- DyT normalization accelerates both training and inference by approximately 6% while maintaining performance.

## Highlights & Insights
- This work is the first to combine Mamba SSM with hierarchical Soft MoE, establishing a novel paradigm for 3D medical image segmentation. The two-level routing design — first processing via local experts, then fusing via global experts — is highly aligned with the naturally hierarchical anatomical structure of medical images.
- GPU memory consumption is only 11.1 GB (comparable to SegMamba), effectively controlling inference memory even with 170M parameters, owing to grouped routing and the linear-complexity Mamba layers.

## Limitations & Future Work
- The model contains 170.1M parameters, the largest among all compared methods, and inference speed is approximately 30% slower than the baseline.
- The acceleration provided by DyT in SSM architectures is relatively modest (~6%); more aggressive acceleration strategies warrant further exploration.
- Pre-training is limited to CT and MRI modalities, with no ultrasound data included in the pre-training phase.

## Related Work & Insights
- **vs. SegMamba**: Mamba-HoME augments SegMamba with the HoME layer and GSC module, improving mDSC from 76.0 to 77.5 on the PANORAMA dataset, demonstrating the benefit of hierarchical MoE for Mamba.
- **vs. Swin UNETR**: Mamba-HoME consistently outperforms this classic Transformer-based method across multiple datasets while requiring less GPU memory.
- **vs. SuPreM**: Even when trained from scratch, Mamba-HoME achieves performance comparable to or better than SuPreM, which relies on large-scale pre-training.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — First combination of Mamba and hierarchical MoE; the two-level local-global routing design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Five datasets covering CT/MRI/US three modalities; ablation studies are highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematical derivations are clear and the overall structure is well-organized.
- **Value**: ⭐⭐⭐⭐ — Provides a new, efficient, and powerful baseline for 3D medical image segmentation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Dual Mixture-of-Experts Framework for Discrete-Time Survival Analysis](dual_mixture-of-experts_framework_for_discrete-time_survival_analysis.md)
- [\[NeurIPS 2025\] DyG-Mamba: Continuous State Space Modeling on Dynamic Graphs](dyg-mamba_continuous_state_space_modeling_on_dynamic_graphs.md)
- [\[CVPR 2026\] Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation](../../CVPR2026/medical_imaging/decoding_matters_efficient_mambabased_decoder_with.md)
- [\[NeurIPS 2025\] LoMix: Learnable Weighted Multi-Scale Logits Mixing for Medical Image Segmentation](lomix_learnable_weighted_multi-scale_logits_mixing_for_medical_image_segmentatio.md)
- [\[NeurIPS 2025\] UniMRSeg: Unified Modality-Relax Segmentation via Hierarchical Self-Supervised Compensation](unimrseg_unified_modality-relax_segmentation_via_hierarchical_self-supervised_co.md)

</div>

<!-- RELATED:END -->
