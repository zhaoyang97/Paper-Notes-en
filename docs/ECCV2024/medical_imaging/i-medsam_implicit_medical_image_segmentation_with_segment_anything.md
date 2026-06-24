---
title: >-
  [Paper Note] I-MedSAM: Implicit Medical Image Segmentation with Segment Anything
description: >-
  [ECCV 2024][Medical Imaging][Medical Image Segmentation] Proposed is I-MedSAM, which integrates the powerful generalization capability of SAM with the continuous space prediction advantages of Implicit Neural Representations (INR). By enhancing high-frequency boundary information with a frequency adapter and refining segmentation through uncertainty-guided sampling, it outperforms existing discrete and implicit methods with only 1.6M trainable parameters.
tags:
  - "ECCV 2024"
  - "Medical Imaging"
  - "Medical Image Segmentation"
  - "Implicit Neural Representation"
  - "Segment Anything"
  - "Frequency Adapter"
  - "Uncertainty Guided Sampling"
date: 2026-05-08
content_hash: aa7624a868da2d3f
---

# I-MedSAM: Implicit Medical Image Segmentation with Segment Anything

**Conference**: ECCV 2024  
**arXiv**: [2311.17081](https://arxiv.org/abs/2311.17081)  
**Code**: [Yes](https://github.com/ucwxb/I-MedSAM)  
**Area**: Medical Image Segmentation  
**Keywords**: Medical Image Segmentation, Implicit Neural Representation, Segment Anything, Frequency Adapter, Uncertainty Guided Sampling

## TL;DR

Proposed is I-MedSAM, which integrates the powerful generalization capability of SAM with the continuous space prediction advantages of Implicit Neural Representations (INR). By enhancing high-frequency boundary information with a frequency adapter and refining segmentation through uncertainty-guided sampling, it outperforms existing discrete and implicit methods with only 1.6M trainable parameters.

## Background & Motivation

### Limitations of Prior Work

Medical image segmentation is a crucial step in assisting disease diagnosis. Existing approaches face several challenges:

**Inherent Defects of Discrete Representations**: Traditional methods (such as nnUNet, PraNet) and recent SAM adaptation methods (such as MedSAM) are based on pixel-level discrete predictions. This leads to poor spatial flexibility in cross-resolution scenarios and causes discretization artifacts when scaling to higher resolutions. Furthermore, discrete representations suffer from blurriness when capturing fine boundary details, whereas precise boundary delineation (e.g., transition zones between different tissues/anatomical structures) is critical in medical imaging.

**Defects of Implicit Methods**: While Implicit Neural Representations (INR) can convert discrete representations into continuous space to adapt to arbitrary output resolutions, current implicit methods have three main pain points:
   - Limited representation capability of pre-trained encoders, resulting in poor cross-domain transferability.
   - Neglect of high-frequency information in the frequency domain, which is strongly correlated with boundaries.
   - Adoption of random sampling strategies when training INR, underestimating the importance of sampling strategies.

**Parameter Efficiency Issue**: Full fine-tuning of foundation models involves a massive number of parameters (e.g., 126.6M for nnUNet), necessitating more parameter-efficient fine-tuning strategies.

### Mechanism

The design motivation of I-MedSAM is clear: leverage the strong cross-domain generalization capability of SAM to compensate for the deficiencies of implicit method encoders, while acquiring the flexibility of continuous representation through INR. Building on this, a frequency adapter and uncertainty-guided sampling are designed to address boundary quality and sampling efficiency, respectively.

## Method

### Overall Architecture

I-MedSAM consists of two main parts:

**Encoder Part**: Based on SAM's ViT-B image encoder with frozen pre-trained parameters, utilizing LoRA adapters (spatial domain) and frequency adapters (frequency domain) to extract multi-scale features, while using SAM's prompt encoder to process coarse bounding box prompts.

**Decoder Part**: A two-stage implicit segmentation decoder containing a shallow "coarse" INR ($Dec_c$) and a deep "fine" INR ($Dec_f$), which are bridged by uncertainty-guided sampling.

### Key Designs

#### 1. Frequency Adapter (FA)

**Function**: Extracts high-frequency information from the frequency domain to enhance SAM features and improve segmentation boundary quality.

**Mechanism**: Converts features to the frequency domain via Fast Fourier Transform (FFT) to extract the amplitude spectrum:

$$\mathcal{F}_{u,v} = \sum_{h=1}^{H}\sum_{w=1}^{W} f_{h,w} \cdot e^{-j2\pi(\frac{h}{H}u + \frac{w}{W}v)}$$

Each FA consists of a linear down-projection layer → GELU → linear up-projection layer, with $n$ FAs corresponding to the $n$ blocks of the ViT. Experiments demonstrate that the amplitude spectrum provides better representation capability than the phase spectrum.

**Design Motivation**: Boundary information is highly correlated with high-frequency features in the frequency domain. The original SAM encoder operates primarily in the spatial domain. By supplementing frequency domain information through the frequency adapter, subtle variations in tissue boundaries can be captured more precisely.

#### 2. Coarse-to-Fine INR

**Function**: Maps encoder features and coordinates to continuous segmentation outputs.

**Mechanism**: Inspired by NeRF, instead of using a single-stage INR, a two-stage decoding process is adopted:

First, high-frequency positional encoding is applied to the coordinates to avoid learning bias:

$$\gamma(p) = (\sin(2^0\pi p), \cos(2^0\pi p), \cdots, \sin(2^{L-1}\pi p), \cos(2^{L-1}\pi p))$$

The encoded coordinates, image features, and prompt features are concatenated:

$$Z^p = Concat(\gamma(p), Interp(Enc_I(X)), Enc_I(P))$$

Then, decoding is performed in two stages:
- $Dec_c$ (shallow, MLP dimensions [1024, 512]): generates the coarse segmentation map $\hat{o}_i^c$ and coarse features $z_i^c$.
- $Dec_f$ (deep, MLP dimensions [512, 256, 256, 128]): refines the sampled points.

**Design Motivation**: The two-stage design allows the model to first establish a global understanding and then concentrate computational resources on refining difficult areas, which is more efficient than a single-stage INR.

#### 3. Uncertainty Guided Sampling (UGS)

**Function**: Adaptively selects pixel points that require refinement to feed into the fine INR decoder.

**Mechanism**: Employs MC-Dropout to perform $T$ stochastic forward passes, computing the predictive uncertainty (variance) for each pixel:

$$\mu_i = \frac{1}{T}\sum_{t=1}^{T} p_t(o_i^c | z_i^p)$$
$$u_i = \frac{1}{T}\sum_{t=1}^{T} (p_t(o_i^c | z_i^p) - \mu_i)^2$$

Selects the top-$K\%$ (default 12.5%) feature points with the highest variance to feed into $Dec_f$ for refinement, finally combining the coarse and fine predictions as the output.

**Design Motivation**: Different pixels vary in prediction difficulty; uncertainty is higher near boundaries and in challenging regions. Adaptively selecting points with high uncertainty for refinement is more efficient and accurate than random sampling or processing all points.

### Loss & Training

**Loss Function**: A weighted combination of Cross-Entropy loss and Dice loss is adopted:

$$L_{seg}(o_i, \hat{o}_i) = 0.5 \cdot L_{ce}(o_i, \hat{o}_i) + 0.5 \cdot L_{dc}(o_i, \hat{o}_i)$$

**Training Strategy**:
- Freeze the SAM image encoder, training only the adapters, prompt encoder, and INR.
- Simultaneously optimize the coarse and fine stages, gradually reducing the supervision weight of coarse segmentation and increasing that of fine segmentation during training.
- Use the AdamW optimizer with an adapter learning rate of $5 \times 10^{-5}$ and a decoder learning rate of $1 \times 10^{-3}$.
- Set LoRA rank to 4, dropout probability to 0.5, and train for 1000 epochs.

## Key Experimental Results

### Main Results

**Binary Polyp Segmentation (Kvasir-Sessile)**

| Method Type | Method | Dice (%) ↑ | Trainable Params (M) ↓ |
|---------|------|-----------|----------------|
| Discrete | U-Net | 63.89±1.30 | 7.9 |
| Discrete | PraNet | 82.56±1.08 | 30.5 |
| Discrete | nnUNet | 82.97±0.89 | 126.6 |
| Discrete | MedSAM | 82.88±0.55 | 4.1 |
| Implicit | OSSNet | 76.11±1.14 | 5.2 |
| Implicit | SwIPE | 85.05±0.82 | 2.7 |
| **Implicit** | **I-MedSAM** | **91.49±0.52** | **1.6** |

**Multi-class Organ Segmentation (BCV, 13 classes)**

| Method Type | Method | Dice (%) ↑ | Trainable Params (M) ↓ |
|---------|------|-----------|----------------|
| Discrete | nnUNet | 85.15±0.67 | 126.6 |
| Discrete | MedSAM | 85.85±0.81 | 52.7 |
| Implicit | SwIPE | 81.21±0.94 | 4.4 |
| **Implicit** | **I-MedSAM** | **89.91±0.68** | **3.5** |

### Robustness Experiments

**Cross-Resolution (Kvasir-Sessile)**

| Method | 384→128 Dice (%) | 384→896 Dice (%) |
|------|-----------------|-----------------|
| nnUNet | 73.97 | 83.56 |
| MedSAM | 82.39 | 83.19 |
| SwIPE | 81.26 | 84.33 |
| **I-MedSAM** | **91.45** | **91.33** |

**Cross-Domain Generalization**

| Task | Method | Dice (%) |
|------|------|---------|
| Sessile→CVC | nnUNet | 84.91 |
| Sessile→CVC | **I-MedSAM** | **88.83** |
| BCV→AMOS | SwIPE | 82.81 |
| BCV→AMOS | **I-MedSAM** | **86.28** |

### Ablation Study

**Component Ablation (Kvasir-Sessile)**

| LoRA | FA | INR | Sessile Dice (%) | Cross-Domain Dice (%) | 384→128 | 384→896 |
|------|----|-----|-----------------|--------------|---------|---------|
| ✓ | | | 83.61 | 82.57 | 72.73 | 76.46 |
| ✓ | ✓ | | 88.74 | 82.61 | 75.69 | 78.59 |
| ✓ | | ✓ | 88.83 | 83.40 | 88.16 | 88.43 |
| ✓ | ✓ | ✓ | **91.49** | **88.83** | **91.45** | **91.33** |

**Frequency Adapter Ablation**

| Setting | w/o FA | Phase Spectrum | Amplitude Spectrum |
|------|--------|-------|-------|
| Dice (%) | 88.83 | 90.60 | **91.49** |
| HD Distance | 15.44 | 12.67 | **11.59** |

**UGS Sampling Ratio Ablation**

| Setting | w/o UGS | Top-50% | Top-25% | Top-12.5% | Top-6.25% | Top-3.125% |
|------|---------|---------|---------|-----------|-----------|------------|
| Dice (%) | 87.77 | 90.27 | 89.59 | **91.49** | 91.01 | 90.48 |

### Key Findings

1. FA and INR each yield independent improvements, and their combined use produces a $1+1>2$ synergistic effect.
2. The INR decoder shows a more prominent advantage in cross-domain and cross-resolution tasks (raising performance from 72.73/76.46 to 88.16/88.43).
3. The amplitude spectrum is more effective than the phase spectrum and significantly improves boundary quality (reducing HD from 15.44 to 11.59).
4. A 12.5% sampling ratio for UGS is optimal; too much or too little sampling is disadvantageous.
5. I-MedSAM still significantly outperforms all baselines in low-resource (10% training data) scenarios.

## Highlights & Insights

1. **Perfect Combination of Continuous and Discrete**: Instead of simply replacing the decoder, a transformation from discrete to continuous is systematically designed via dual spatial-frequency path encoding + two-stage INR decoding.
2. **Uncertainty-Driven Computation Allocation**: The UGS strategy concentrates more computational resources of the model on "truly difficult pixels", embodying the concept of adaptive computation.
3. **Extreme Parameter Efficiency**: Achieving better performance than nnUNet (126.6M) with only 1.6M trainable parameters results in an efficiency ratio of 79:1.

## Limitations & Future Work

1. Currently only validated on 2D medical images; extending to 3D volumetric segmentation (such as CT/MRI volumes) is a natural direction.
2. Reliance on coarse bounding box prompts as inputs limits the degree of automation.
3. The $T$ forward passes of MC-Dropout increase inference time; more efficient uncertainty estimation methods can be explored.
4. The design of the frequency adapter is relatively simple (only linear layer + GELU); more complex frequency domain processing modules could be introduced.

## Related Work & Insights

- **Implicit Representation for Segmentation**: Works like OSSNet, IOSNet, and SwIPE established the paradigm of using INR for segmentation. I-MedSAM builds on this by introducing a stronger encoder and adaptive sampling.
- **SAM Adaptation**: Works like MedSAM and SAMed focus on adapting SAM to medical images. I-MedSAM innovatively combines SAM with INR.
- **Insight**: The improvement of boundary quality driven by frequency domain information suggests that frequency-domain feature enhancement can also be considered in other dense prediction tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of SAM + INR + Frequency Adapter + UGS is brand new, although each individual component is not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers multi-dimensional evaluations including multi-task, cross-resolution, cross-domain, boundary quality, and low-resource scenarios, with thorough ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation, detailed method descriptions, and well-designed figures/tables.
- **Value**: ⭐⭐⭐⭐ — Offers a highly parameter-efficient and robust solution for medical image segmentation, with 1.6M parameters demonstrating strong practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Show and Segment: Universal Medical Image Segmentation via In-Context Learning](../../CVPR2025/medical_imaging/show_and_segment_universal_medical_image_segmentation_via_in-context_learning.md)
- [\[ECCV 2024\] Alternate Diverse Teaching for Semi-supervised Medical Image Segmentation](alternate_diverse_teaching_for_semi-supervised_medical_image_segmentation.md)
- [\[ECCV 2024\] Adaptive Correspondence Scoring for Unsupervised Medical Image Registration](adaptive_correspondence_scoring_for_unsupervised_medical_ima.md)
- [\[ECCV 2024\] Unsupervised Multi-modal Medical Image Registration via Invertible Translation](unsupervised_multi-modal_medical_image_registration_via_invertible_translation.md)
- [\[CVPR 2025\] NOIR: Neural Operator Mapping for Implicit Representations](../../CVPR2025/medical_imaging/noir_neural_operator_mapping_for_implicit_representations.md)

</div>

<!-- RELATED:END -->
