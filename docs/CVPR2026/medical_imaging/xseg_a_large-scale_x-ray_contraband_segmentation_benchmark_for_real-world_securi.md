---
title: >-
  [Paper Note] XSeg: A Large-scale X-ray Contraband Segmentation Benchmark for Real-World Security Screening
description: >-
  [CVPR 2026][Medical Imaging][X-ray contraband segmentation] This paper introduces XSeg, the largest X-ray contraband segmentation dataset to date (98,644 images, 295,932 instance masks, 30 fine-grained categories), and proposes APSAM, a domain-specialized model that leverages the physical dual-energy properties of X-ray imaging via an Energy-Aware Encoder (EAE) and an Adaptive Point Generator (APG) to intelligently expand user click prompts. APSAM achieves 72.83% mIoU, surpassing SAM fine-tuning by 4.96%.
tags:
  - CVPR 2026
  - Medical Imaging
  - X-ray contraband segmentation
  - security screening dataset
  - SAM adaptation
  - dual-energy encoder
  - adaptive point prompt
date: 2026-05-08
content_hash: 37790f0bcf0fece7
---

# XSeg: A Large-scale X-ray Contraband Segmentation Benchmark for Real-World Security Screening

**Conference**: CVPR 2026
**arXiv**: [2604.03706](https://arxiv.org/abs/2604.03706)
**Code**: N/A
**Area**: Medical Imaging
**Keywords**: X-ray contraband segmentation, security screening dataset, SAM adaptation, dual-energy encoder, adaptive point prompt

## TL;DR

This paper introduces XSeg, the largest X-ray contraband segmentation dataset to date (98,644 images, 295,932 instance masks, 30 fine-grained categories), and proposes APSAM, a domain-specialized model that leverages the physical dual-energy properties of X-ray imaging via an Energy-Aware Encoder (EAE) and an Adaptive Point Generator (APG) to intelligently expand user click prompts. APSAM achieves 72.83% mIoU, surpassing SAM fine-tuning by 4.96%.

## Background & Motivation

1. **Background**: Contraband detection in X-ray security screening images is a core safety requirement in airports, subway systems, and logistics centers. Existing X-ray datasets (SIXray, PIXray, PIDray) are small-scale (<50K images), cover few categories (<15), and primarily provide bounding box annotations rather than segmentation masks.
2. **Limitations of Prior Work**: (1) Data scarcity — the largest existing dataset, PIDray, contains only 47K images and covers merely 12 categories; (2) General-purpose segmentation models such as SAM transfer poorly to the X-ray domain due to substantial differences in color space and texture relative to natural images; (3) Single-point prompts are insufficiently informative in complex, heavily occluded scenes typical of security screening.
3. **Key Challenge**: Accurate segmentation for X-ray security screening requires both large-scale, high-quality annotated data and domain-adapted segmentation methods — both of which are severely lacking.
4. **Goal**: To simultaneously address the data and methodology bottlenecks by constructing a large-scale segmentation benchmark and designing a domain-specialized SAM adaptation framework.
5. **Key Insight**: The physical characteristics of X-ray imaging provide unique signals — dual-energy channels (high-energy/low-energy) can distinguish materials of different compositions, a domain prior unavailable in RGB imagery.
6. **Core Idea**: EAE exploits the max/min channels of X-ray images to extract dual-energy features for decoder initialization; APG expands a single user click into two informative prompt points.

## Method

### Overall Architecture

X-ray image → SAM ViT-L encoder extracts image features + EAE extracts dual-energy features to initialize decoder tokens → user clicks a single point $p_0$ → APG generates an initial mask → K-means identifies two representative points $(p_1, p_2)$ → SAM decoder uses the enriched prompts to predict the final mask.

### Key Designs

1. **Energy-Aware Encoder (EAE)**

   - **Function**: Extracts dual-energy domain priors from X-ray images.
   - **Mechanism**: Computes a high-energy channel $I_H = \max_c I(\cdot,\cdot,c)$ and a low-energy channel $I_L = \min_c I(\cdot,\cdot,c)$, concatenates them, and encodes the result through three Conv+LayerNorm+GELU+MaxPool layers. Channel-wise attention combined with top-k feature selection generates initialization queries that replace SAM's random token initialization.
   - **Design Motivation**: The RGB channels of X-ray images physically encode transmission information at different energy levels — metallic objects exhibit markedly different responses across high- and low-energy channels. Exploiting this physical prior enables the model to better discriminate between material types.

2. **Adaptive Point Generator (APG)**

   - **Function**: Expands a single user click into two more informative prompt points.
   - **Mechanism**: The initial point $p_0$ is used to generate a soft mask $M_0$; a bounding box is extracted and randomly scaled ($s \sim \mathcal{U}(0.9, 1.1)$); K-means clustering ($K=2$) is applied within the box; the farthest point pair $(p_1^*, p_2^*)$ across the two clusters serves as the new prompts (falling back to random sampling if the inter-cluster distance is insufficient).
   - **Design Motivation**: Contraband items in security screening are frequently heavily overlapping, making single-point prompts inadequate to describe target extent. APG adaptively identifies two representative locations, providing richer spatial cues. Ablation results show APG outperforms two randomly sampled points by 1.59 mIoU.

3. **XSeg Dataset Construction**

   - **Function**: Provides a large-scale, high-quality benchmark for X-ray segmentation.
   - **Mechanism**: Images are aggregated from 114Xray, PIXray, PIDray, and real-world screening data; after filtering by resolution, aspect ratio, and sharpness, approximately 150K candidates are refined to 98,644 images. Annotation employs a closed-loop strategy combining MobileSAM-assisted labeling with expert review by security screening professionals over five iterative rounds, yielding 30 fine-grained categories (e.g., scissors are distinguished by metal vs. plastic handles).
   - **Design Motivation**: The scale and annotation granularity of existing datasets are insufficient for training and evaluating deployable segmentation models for security screening.

### Loss & Training

Standard SAM training objectives (Dice + Cross-Entropy). ViT-L/14 backbone, 512×512 input, AdamW optimizer with lr=1e-5, batch size 16, 12 epochs. The majority of parameters are frozen; only EAE, APG, and adapters are trained (11.91M trainable parameters).

## Key Experimental Results

### Main Results

| Method | Backbone | mIoU↑ | Dice↑ | Trainable Params |
|--------|----------|-------|-------|-----------------|
| DeepLabV3+ | ResNet101 | 57.29 | 72.84 | 60.21M |
| Mask2Former | Swin-L | 69.59 | 81.44 | 144.85M |
| SAM (frozen) | ViT-L | 53.82 | 64.99 | 0M |
| SAM (finetune) | ViT-L | 67.87 | 77.45 | 10.06M |
| SAMUS | ViT-L | 68.56 | 78.46 | 43.21M |
| **APSAM** | **ViT-L** | **72.83** | **82.31** | **11.91M** |

### Ablation Study

| Configuration | mIoU↑ | Dice↑ | Note |
|---------------|-------|-------|------|
| w/o EAE & APG (SAM FT) | 67.87 | 77.45 | Baseline |
| w/o APG | 70.89 | 79.50 | EAE contribution: +3.02 |
| w/o EAE | 71.90 | 81.62 | APG contribution: +4.03 |
| **Full (EAE + APG)** | **72.83** | **82.31** | Complementary gains |
| 1 random point | 67.87 | 77.45 | Baseline |
| 2 random points | 70.31 | 80.18 | Multi-point beneficial |
| **APG 2 points** | **71.90** | **81.62** | Intelligent selection superior |

### Key Findings

- APG contributes more (+4.03 mIoU) than EAE (+3.02), indicating that prompt quality has a greater impact on SAM performance than encoder initialization.
- Strong cross-domain generalization: APSAM achieves 71.23% and 83.61% mIoU on PIDray and PIXray respectively, outperforming SAMUS by 4.22% and 3.70%.
- Zero-shot SAM achieves only 53.82% mIoU on X-ray images, confirming a substantial domain gap.
- APSAM with 11.91M trainable parameters outperforms Mask2Former with 144.85M parameters (72.83 vs. 69.59 mIoU).

## Highlights & Insights

- **Principled exploitation of physical priors**: The dual-energy channels of X-ray imaging are not a simple RGB decomposition but carry material-specific physical signals. EAE's use of max/min operations to extract high- and low-energy features is straightforward yet effective.
- **Practical utility of APG**: In operational security screening, operators have time for only a single click — APG automatically expands that click into a more informative two-point prompt, reducing the manual burden at deployment.
- **Long-term dataset value**: With 98K images and 30 fine-grained category annotations, XSeg fills a critical gap in segmentation data for the security screening domain.

## Limitations & Future Work

- Although the data sources are diverse, they are primarily drawn from Chinese security screening systems; differences in color space across X-ray equipment from different countries and manufacturers may limit generalization.
- The 30-category taxonomy may still be insufficient — real-world security screening involves a broader range of contraband types, including liquids and powdered substances.
- Despite five rounds of iterative annotation, labeling quality in heavily occluded scenes remains difficult to guarantee.
- APG's K-means clustering may degrade on extremely elongated objects where both cluster centers fall along the same axis.
- From a security perspective, the false negative rate is more operationally critical than mIoU, yet this metric receives limited analysis in the paper.

## Related Work & Insights

- **vs. SAMUS**: Both are SAM domain adaptation methods, but SAMUS requires 43.21M trainable parameters and does not exploit X-ray physical properties. APSAM achieves superior performance with fewer parameters (11.91M).
- **vs. Mask2Former**: This fully supervised approach requires 145M parameters yet achieves only 69.59% mIoU, suggesting that SAM-based efficient adaptation is a more parameter-effective paradigm.
- **vs. PIDray/PIXray**: XSeg is approximately twice their combined size and provides 2.5× finer category granularity — a quantitative leap that may yield qualitative improvements in model capability.

## Rating

- Novelty: ⭐⭐⭐⭐ EAE and APG are well-motivated designs, though not breakthrough contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive ablations, cross-domain evaluation, multi-framework comparisons, and prompt strategy analysis.
- Writing Quality: ⭐⭐⭐⭐ Dataset construction and methodology are described clearly.
- Value: ⭐⭐⭐⭐⭐ The dataset contribution has long-term impact on the security screening field; the method is directly deployable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Instruction-Guided Lesion Segmentation for Chest X-rays with Automatically Generated Large-Scale Dataset](instruction-guided_lesion_segmentation_for_chest_x-rays_with_automatically_gener.md)
- [\[ICCV 2025\] GEMeX: A Large-Scale, Groundable, and Explainable Medical VQA Benchmark for Chest X-ray Diagnosis](../../ICCV2025/medical_imaging/gemex_a_large-scale_groundable_and_explainable_medical_vqa_benchmark_for_chest_x.md)
- [\[ICLR 2026\] Omni-iEEG: A Large-Scale, Comprehensive iEEG Dataset and Benchmark for Epilepsy Research](../../ICLR2026/medical_imaging/omni-ieeg_a_large-scale_comprehensive_ieeg_dataset_and_benchmark_for_epilepsy_re.md)
- [\[CVPR 2026\] Accelerating Stroke MRI with Diffusion Probabilistic Models through Large-Scale Pre-training and Target-Specific Fine-Tuning](accelerating_stroke_mri_with_diffusion_probabilist.md)
- [\[AAAI 2026\] PulseMind: A Multi-Modal Medical Model for Real-World Clinical Diagnosis](../../AAAI2026/medical_imaging/pulsemind_a_multi-modal_medical_model_for_real-world_clinical_diagnosis.md)

</div>

<!-- RELATED:END -->
