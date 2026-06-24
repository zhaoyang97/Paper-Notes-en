---
title: >-
  [Paper Note] Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation
description: >-
  [CVPR 2025][Medical Imaging][medical image segmentation] Deco-Mamba is proposed, a decoder-centric hybrid Transformer-CNN-Mamba architecture. It enhances decoder capabilities via a Co-Attention Gate, Visual State Space Mamba Block, and Deformable Residual Block. By introducing a windowed KL-divergence based distribution-aware deep supervision strategy, it achieves SOTA performance across 7 medical image segmentation benchmarks while maintaining moderate model complexity.
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "medical image segmentation"
  - "Mamba"
  - "decoder-centric"
  - "deep supervision"
  - "KL-divergence"
date: 2026-05-08
content_hash: 0b7aa507fb2e55a1
---

# Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation

**Conference**: CVPR 2025  
**arXiv**: [2603.12547](https://arxiv.org/abs/2603.12547)  
**Code**: To be released (open-sourced upon acceptance)  
**Area**: Medical Images  
**Keywords**: medical image segmentation, Mamba, decoder-centric, deep supervision, KL-divergence

## TL;DR
Deco-Mamba is proposed, a decoder-centric hybrid Transformer-CNN-Mamba architecture. It enhances decoder capabilities via a Co-Attention Gate, Visual State Space Mamba Block, and Deformable Residual Block. By introducing a windowed KL-divergence based distribution-aware deep supervision strategy, it achieves SOTA performance across 7 medical image segmentation benchmarks while maintaining moderate model complexity.

## Background & Motivation
**Background**: The mainstream paradigms for medical image segmentation are U-Net and its variants (Att-UNet, UNet++, etc.). Later, Transformer architectures (TransUNet, Swin-UNet) emerged to capture long-range dependencies, followed by Mamba/SSM architectures (U-Mamba, SliceMamba) to achieve global modeling with linear complexity.

**Limitations of Prior Work**: (a) Most methods are **task-specific**, performing well on a single dataset but generalizing poorly across modalities; (b) Most works focus on **encoder enhancement**, utilizing large pre-trained backbones that increase computational complexity while ignoring decoder design; (c) Although Mamba-based methods reduce computational overhead, they are typically evaluated on only a few modalities, leaving their generalization capabilities unclear.

**Key Challenge**: Strong encoder coupled with a weak decoder—even when the encoder extracts rich semantics, an inadequately designed decoder struggles to accurately restore target boundaries and contextual structures during upsampling, leading to the loss of fine spatial details. Meanwhile, existing cascaded decoders (e.g., MERIT) significantly increase computational complexity.

**Goal**: (a) How to efficiently model multi-scale contexts and long-range dependencies in the decoder? (b) How to improve the decoder's reconstruction accuracy without introducing excessive parameters? (c) How to achieve cross-modality generalization?

**Key Insight**: Innovating design with a decoder-centric focus while keeping the encoder lightweight, embedding the highly efficient global modeling capability of Mamba into the decoder, and proposing a distribution-aware deep supervision to avoid information loss caused by resizing in traditional deep supervision.

**Core Idea**: Lightweight encoder + heavy decoder (CAG + VSSMB + Deformable Convolution), combined with distribution-aware deep supervision based on windowed KL-divergence.

## Method

### Overall Architecture
Deco-Mamba adopts a U-Net architecture with a dual-branch encoder: a CNN branch to extract high-resolution local features and a PVT Transformer branch to extract four-stage global features. The decoder features a six-stage structure, with core modules including the Co-Attention Gate (CAG), Visual State Space Mamba Block (VSSMB), and Double Deformable Residual Block (DDConv). Two variants are provided: V0 (PVT-V2-B0, 9.67M) and V1 (PVT-V2-B2, 46.93M).

### Key Designs

1. **Co-Attention Gate (CAG)**:

    - **Function**: Adaptive fusion of encoder skip-connection features and decoder features.
    - **Mechanism**: Unlike standard AG, which only uses decoder features to gate encoder features, CAG allows encoder and decoder features to **gate each other**. The outputs of two cross-attention gates are concatenated and then refined by Channel Attention: $D_i' = CA[AG(x=X_i, g=D_{i+1}), AG(x=D_{i+1}, g=X_i)]$.
    - **Design Motivation**: Standard AG ignores spatial saliency in decoder features and only considers spatial attention while neglecting channel relationships. Bidirectional gating combined with channel-wise attention addresses these two limitations.

2. **Vision State Space Mamba Block (VSSMB)**:

    - **Function**: Capturing long-range dependencies in the decoder with linear complexity.
    - **Mechanism**: Applying continuous-time SSM to the two spatial dimensions (height and width), propagating contextual information via four-way selective scanning (horizontal, vertical, and their reverse directions). Two VSSMBs are used at the bottleneck, one in each of stages 2–5, and none in the final stage.
    - **Design Motivation**: The complexity of Transformer self-attention is $O(n^2)$, whereas Mamba's SSM is $O(n)$, making it more suitable for handling the progressively increasing spatial resolutions in the decoder.

3. **Deformable Residual Block (DRB)**:

    - **Function**: Restoring local spatial details smoothed out by the global modeling of VSSMB.
    - **Mechanism**: Integrating standard 3×3 convolution and deformable convolution (DCN) within a residual framework. DCN predicts pixel-level offsets and modulation masks, where the offset branch estimates sampling displacements and the modulation branch assigns pixel importance weights in [0, 2].
    - **Design Motivation**: SSM excels at global modeling but may overlook subtle local changes (such as complex tissue boundaries). DRB recovers boundary accuracy through the geometric adaptability of deformable convolution.

### Loss & Training
$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{dice}} + \mathcal{L}_{\text{multi}}$

**Multi-Scale Distribution-Aware (MSDA) Deep Supervision**:
- Operating at the native resolution of each decoder stage, thereby avoiding the information loss caused by resizing intermediate outputs to the GT resolution in traditional deep supervision.
- Averaging GT labels within local windows to obtain the class frequency distribution $\tilde{P}^{(s)}$, converting predictions into $Q^{(s)}$ via a distribution head and softmax, and calculating the KL-divergence.
- Introducing **boundary-aware weighting** $W^{(s)}_{h,w} = (1 - \max_n \tilde{P}^{(s)}_{h,w,n})^\alpha$ to assign higher weights to mixed-class regions (near boundaries).
- During multi-scale aggregation, deeper decoder stages are assigned greater weights: $\lambda_1 < \lambda_2 < \cdots < \lambda_S$.

## Key Experimental Results

### Main Results (Synapse Multi-Organ Segmentation, 8 Classes)

| Method | Dice↑ | HD95↓ | Params (M) | FLOPs (G) |
|------|-------|-------|----------|----------|
| Deco-Mamba-V1 | **85.07** | **14.72** | 46.93 | 17.24 |
| Deco-Mamba-V0 | 83.16 | 15.89 | **9.67** | **9.73** |
| Cascaded-MERIT | 83.59 | 15.99 | 147.86 | 33.31 |
| PAG-TransYnet | 83.43 | 15.82 | 144.22 | 33.65 |
| SliceMamba | 81.95 | 16.04 | - | - |
| Swin-UMamba | 80.34 | 21.51 | 59.88 | 31.35 |

### Ablation Study (Synapse)

| Configuration | Dice↑ | HD95↓ | Description |
|------|-------|-------|------|
| Full Deco-Mamba-V1 | **85.07** | **14.72** | Full model |
| w/o CNN branch | 84.07 | 18.92 | ~1% drop in Dice |
| w/o VSSMB | 83.51 | 15.96 | Removing Mamba results in a 1.56% drop |
| CAG→AG | 82.98 | 15.69 | Standard AG results in a 2.09% drop |
| CAG→CBAM | 84.01 | 16.19 | CBAM is sub-optimal |
| DRB→Standard Conv | 84.53 | 16.18 | Deformable convolution helps |
| Dice only | 83.84 | 14.94 | Without deep supervision |
| Dice + Traditional deep supervision | 84.24 | 15.89 | HD95 deteriorates instead |
| Dice + MSDA (ours) | **85.07** | **14.72** | MSDA is optimal |

### Key Findings
- Deco-Mamba-V0 achieves 83.16% Dice with only 9.67M parameters, close to the 150M-parameter Cascaded-MERIT (83.59%), yielding a **15×** improvement in efficiency.
- Consistent superiority is achieved across 7 datasets (covering CT, MRI, dermoscopy, glands, and nuclei), validating its cross-modality generalization.
- Traditional deep supervision improves Dice but worsens HD95, because forcing low-resolution predictions to upscale destroys boundaries—a flaw that MSDA bypasses.
- On MoNuSeg, Deco-Mamba-V1 outperforms Swin-UMamba by **+4.46%** in Dice and U-Net by **+8.69%**.

## Highlights & Insights
- **Decoder-centric design is counter-intuitive yet effective**: While most works stack stronger encoders, this paper demonstrates that focusing innovation on the decoder while keeping the encoder lightweight achieves comparable or even better results. This provides valuable insights for computationally constrained scenarios (such as edge deployment).
- **MSDA deep supervision resolves the classic resizing issue**: Matching category distributions with KL-divergence at native resolution rather than pixel-level predictions avoids information loss and naturally introduces boundary awareness. This supervision strategy can be utilized independently of the architecture.
- **Windowed distributions are more suitable for multi-scale supervision than pixel-level GT**: Utilizing the GT class frequency within a small window as soft labels provides a smoother representation than hard scaling, naturally reflecting the semantic distribution at lower resolutions.

## Limitations & Future Work
- Currently supports 2D segmentation only, without extension to 3D medical images.
- The code has not been released yet, leaving reproducibility to be verified.
- The PVT encoder relies on pre-trained models, making it difficult to fully decouple the contributions of the encoder from those of the decoder.
- Multiple components (CAG, VSSMB, DRB, MSDA) are combined, and the optimal configuration may vary depending on the dataset.

## Related Work & Insights
- **vs EMCAD**: EMCAD also focuses on the decoder but relies on attention and lightweight convolutions, lacking long-range modeling capabilities. Deco-Mamba addresses this drawback via VSSMB.
- **vs Cascaded-MERIT**: MERIT achieves an 83.59% Synapse Dice using a cascaded encoder-decoder (148M parameters), whereas Deco-Mamba-V1 reaches 85.07% with only 47M parameters, offering higher efficiency and better performance.
- **vs SliceMamba/VM-UNet**: While also Mamba-based, these methods focus primarily on the encoder. Their performance is inferior to Deco-Mamba on complex multi-organ tasks such as BTCV, highlighting the importance of the decoder.

## Rating
- Novelty: ⭐⭐⭐⭐ The decoder-centric approach and distribution-aware deep supervision represent clear and valuable innovations, though the individual components themselves are not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely thorough, with 7 datasets, multiple modalities, detailed ablations, and comparisons across several backbones.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rich illustrations, and detailed methodological descriptions.
- Value: ⭐⭐⭐⭐ Empirical insights regarding decoder design and distribution-aware supervision offer standard references for subsequent research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing_semi-supervised_medical_image.md)
- [\[CVPR 2025\] Enhancing SAM with Efficient Prompting and Preference Optimization for Semi-supervised Medical Image Segmentation](sam_dpo_semi_supervised.md)
- [\[NeurIPS 2025\] Mamba Goes HoME: Hierarchical Soft Mixture-of-Experts for 3D Medical Image Segmentation](../../NeurIPS2025/medical_imaging/mamba_goes_home_hierarchical_soft_mixture-of-experts_for_3d_medical_image_segmen.md)
- [\[CVPR 2025\] BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)
- [\[CVPR 2025\] Show and Segment: Universal Medical Image Segmentation via In-Context Learning](show_and_segment_universal_medical_image_segmentation_via_in-context_learning.md)

</div>

<!-- RELATED:END -->
