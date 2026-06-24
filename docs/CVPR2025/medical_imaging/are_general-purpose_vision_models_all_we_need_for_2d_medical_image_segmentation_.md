---
title: >-
  [Paper Note] Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation?
description: >-
  [CVPR 2025][Medical Imaging][Medical Image Segmentation] By establishing unified training and evaluation protocols, this study compares 11 specialized and general-purpose vision models across three heterogeneous medical datasets. The findings reveal that General-Purpose Vision Models (GP-VMs) can systematically outperform most Specialized Medical Segmentation Architectures (SMAs) in both segmentation accuracy and interpretability, challenging the traditional assumption that "…
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "Medical Image Segmentation"
  - "General-Purpose Vision Models"
  - "Cross-Dataset Benchmarking"
  - "Interpretability Analysis"
  - "Model Selection"
date: 2026-05-08
content_hash: d6365f85e9b86342
---

# Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation?

**Conference**: CVPR 2025  
**arXiv**: [2603.13044](https://arxiv.org/abs/2603.13044)  
**Code**: [GitHub](https://github.com/VanessaBorst/GPVision4MIS/)  
**Area**: Medical Images  
**Keywords**: Medical Image Segmentation, General-Purpose Vision Models, Cross-Dataset Benchmarking, Interpretability Analysis, Model Selection

## TL;DR

By establishing unified training and evaluation protocols, this study compares 11 specialized and general-purpose vision models across three heterogeneous medical datasets. The findings reveal that General-Purpose Vision Models (GP-VMs) can systematically outperform most Specialized Medical Segmentation Architectures (SMAs) in both segmentation accuracy and interpretability, challenging the traditional assumption that "medical image segmentation necessitates domain-specific architectures."

## Background & Motivation

**Background**: Medical Image Segmentation (MIS) is a fundamental component of computer-aided diagnosis. Since the advent of U-Net, a vast number of specialized architectures tailored to the characteristics of medical images (low contrast, small structures, scarce annotations) have emerged, including Transformer-based architectures like HiFormer and MISSFormer, KAN-based U-KAN, and Mamba-integrated Swin-UMamba. Concurrently, general-purpose vision models (such as SegFormer, SegNeXt, and VWFormer) have demonstrated powerful generalization capabilities on natural images.

**Limitations of Prior Work**: Although general-purpose vision models perform exceptionally well on standard benchmarks like ADE20K, their efficacy in medical imaging lacks systematic validation. Existing comparative studies often rely on self-reported metrics from individual papers. However, variances in data preprocessing, augmentation strategies, and optimization settings are substantial across different studies, suggesting that the claimed "architectural advantages" might merely be a byproduct of discrepancies in experimental design.

**Key Challenge**: There is a lack of rigorous, controlled comparisons under a unified experimental framework—does the domain-specific design of specialized architectures yield genuine advantages, or are the large-scale pretraining and adequate fine-tuning of general-purpose models already sufficient?

**Goal**: To establish a standardized benchmarking framework that systematically compares general-purpose vision models against specialized medical segmentation architectures under unified training and evaluation protocols, complemented by a comprehensive evaluation using interpretability analysis (Grad-CAM).

**Key Insight**: Rather than debating the theoretical value of architectural innovations, this work displays the actual performance of general-purpose models through rigorous controlled experiments. Three datasets covering diverse modalities (dermoscopy, endoscopy, echocardiography) and class structures (binary/multi-class) are selected to eliminate confounding factors.

**Core Idea**: Under standardized conditions, general-purpose vision models can systematically outperform most specialized medical segmentation architectures, achieving the ability to capture clinically relevant structures without requiring domain-specific designs.

## Method

### Overall Architecture

Rather than proposing a new model, this paper designs a controlled benchmarking evaluation framework. Five specialized medical architectures (U-Net, HiFormer-B, MISSFormer, Swin-UMamba, U-KAN-L) and six general-purpose vision models (SegFormer-B3, SegNeXt-L, VWFormer $\times 2$, InternImage-T, TransNeXt-Tiny) are selected and trained under a unified setup across three datasets (ISIC'18 skin lesions, BKAI-IGH polyps, and CAMUS cardiac ultrasound).

### Key Designs

1. **Unified Training Protocol**:

    - **Function**: Eliminate confounding factors introduced by variations in experimental protocols.
    - **Mechanism**: All models share the same ImageNet-pretrained encoder, a $512 \times 512$ input resolution, the AdamW optimizer with a REX learning rate scheduler, a batch size of 8, and a unified data augmentation pipeline. A learning rate grid search ($10^{-4}$ / $5\times10^{-5}$ / $10^{-5}$) is conducted for each model-dataset combination, selecting the optimal setting over 100 epochs, and subsequently executing 150 epochs of 5-fold cross-validation.
    - **Design Motivation**: The claimed "architectural advantages" in existing literature may largely stem from disparate training recipes rather than the architectures themselves. A unified protocol is a prerequisite for fair comparison.

2. **Multi-Dimensional Evaluation Scheme**:

    - **Function**: Transcend simple segmentation accuracy to comprehensively evaluate model behavior.
    - **Mechanism**: Four metrics (mIoU, mDSC, Recall, Precision) are used to quantify segmentation performance, paired with Grad-CAM visualization to analyze the models' attention patterns—investigating whether they genuinely focus on clinically relevant areas. Evaluations employ 5-fold cross-validation alongside global micro-averaging excluding the background class.
    - **Design Motivation**: In medical contexts, models need to not only segment accurately but also "look at the right place"—Grad-CAM analysis reveals whether the models' decision-making processes align with clinical knowledge.

3. **Heterogeneous Dataset Coverage**:

    - **Function**: Verify the cross-modal and cross-task generalization of the conclusions.
    - **Mechanism**: Three highly differentiated datasets are selected: ISIC'18 (RGB dermoscopy, binary classification, 3,565 images), BKAI-IGH NeoPolyp (RGB endoscopy, three-class, 945 images), and CAMUS (grayscale ultrasound, four-class, 1,996 images). Patient-level splitting is applied to CAMUS to prevent information leakage, while deduplication filtering is performed on ISIC'18 and BKAI-IGH.
    - **Design Motivation**: Covering different modalities (RGB/grayscale), diverse class structures (binary/multi-class), varying image qualities, and data scales ensures that the conclusions are not overfitted to any specific scenario.

### Loss & Training

ISIC'18 utilizes binary cross-entropy loss, while BKAI-IGH and CAMUS employ standard cross-entropy loss. All models utilize identical loss functions and dataset-specific augmentation strategies within the same dataset. Early stopping is applied, and models are trained on PyTorch 2.5.1 using $2 \times \text{A100}$ GPUs.

## Key Experimental Results

### Main Results

| Dataset | Model Category | Best Model | mDSC (%) | Best SMA | mDSC (%) | Gain |
|---|---|---|---|---|---|---|
| ISIC'18 | GP-VM | TransNeXt | 91.9±0.7 | Swin-UMamba | 91.3±0.5 | +0.6 |
| BKAI-IGH | GP-VM | VW-MiT | 89.7±0.8 | Swin-UMamba | 88.9±0.6 | +0.8 |
| CAMUS | GP-VM | SegNeXt/VW-MiT | 91.6±0.1 | Swin-UMamba | 91.3±0.3 | +0.3 |
| 3-Dataset Average | GP-VM | VW-MiT | 91.0 | Swin-UMamba | 90.5 | +0.5 |

### Ablation Study (Cross-class performance on datasets)

| Model | BKAI-IGH $C_1$ (Non-neoplastic) | BKAI-IGH $C_2$ (Neoplastic) | CAMUS LV | CAMUS LV Wall | CAMUS LA |
|------|---------------------|---------------------|----------|---------------|----------|
| VW-MiT (GP) | 66.1±4.3 | 92.7±0.9 | 94.6±0.2 | 88.7±0.2 | 91.8±0.2 |
| Swin-UMamba (SMA) | 59.2±3.8 | 92.5±0.6 | 94.4±0.3 | 88.3±0.3 | 91.4±0.4 |
| U-KAN-L (SMA) | 36.9±12.2 | 87.1±0.9 | 94.0±0.2 | 87.4±0.2 | 90.6±0.4 |
| MISSFormer (SMA) | 42.0±6.5 | 87.5±1.8 | 93.8±0.1 | 87.2±0.2 | 90.7±0.2 |

### Key Findings

- **Comprehensive Lead by General-Purpose Models**: Based on the average mDSC across three datasets, the top 6 models are all GP-VMs (VW-MiT 91.0%, VW-Conv/TransNeXt 90.9%, InternImage-T 90.8%, SegNeXt/SegFormer 90.7%), whereas Swin-UMamba (90.5%) is the only specialized model that comes close.
- **Gaps Vary Across Datasets**: The performance gap is most pronounced on BKAI-IGH (where GP-VMs outperform most SMAs by 20+ percentage points in the non-neoplastic polyp category) and narrowest on CAMUS (approximately 1-2%).
- **Grad-CAM Analysis Reveals More Focused Attention in GP-VMs**: General-purpose models focus more precisely on clinically relevant areas in challenging cases. Some GP-VMs generate more accurate attention maps than specialized models, showing that universal feature representations acquired from large-scale pretraining remain highly effective in medical contexts.
- **Swin-UMamba is the Only Competitive SMA**: Serving as the top-performing SMA across all datasets, it minimizes the performance gap to GP-VMs. Other SMAs (specifically U-KAN-L and MISSFormer) exhibit significant performance degradation.

## Highlights & Insights

- **Empirical Evidence Challenging Traditional Hypotheses**: Instead of proposing a novel method, this paper addresses a long-overlooked fundamental question through strictly controlled experiments—do domain-specific architectures provide a genuine advantage? The counterintuitive findings carry profound value for the research community.
- **Grad-CAM Interpretability Analysis**: Beyond numerical metrics, attention visualization confirms that GP-VMs indeed learn clinically relevant representations. This is significantly more convincing than purely numerical comparisons and increases confidence for clinical deployment.
- **Implications for Resource Allocation**: Given that GP-VMs are highly capable, instead of investing vast amounts of resources in engineering novel domain-specific architectures, efforts should pivot toward data curation, training optimization, and out-of-distribution (OOD) generalization evaluation—an insight with far-reaching impacts on the broader medical AI community.

## Limitations & Future Work

- **Limited Dataset Coverage**: The study covers only three 2D datasets and two modalities, without extending to 3D medical images (e.g., CT/MRI volumetric segmentation) or extreme low-data regimes.
- **Slight Imbalance in Model Selection**: Some SMAs (e.g., U-KAN-L, HiFormer-B) possess smaller parameter counts, which might introduce minor bias albeit showing consistent overall trends.
- **Exclusion of Foundation Models like SAM**: The Segment Anything Model and its medical adaptations (MedSAM, SAM-Med2D) represent newly emerged, powerful alternatives that are not included in this comparison.
- **Future Directions**: Extend to additional modalities and 3D scenarios; incorporate OOD evaluations (e.g., training on BKAI and testing on Kvasir-SEG); integrate prompt-based foundation models.

## Related Work & Insights

- **vs Swin-UMamba**: Swin-UMamba combines the long-range modeling of Mamba with the benefits of ImageNet pretraining. As the sole SMA capable of closely approaching GP-VM performance, it underscores that pretraining strategies are more critical than architectural novelty.
- **vs VWFormer**: A general-purpose segmenter based on multi-scale window attention, achieving peak performance in medical contexts without any modifications, demonstrating that general representations are already sufficiently robust.
- **vs SAM Series**: Although SAM is excluded from this evaluation, other works have established its efficacy in medical segmentation. This aligns with the main conclusion herein—general-purpose models are highly viable in the medical domain.

## Rating

- Novelty: ⭐⭐⭐ No method innovation, but the core question is highly valuable and the experimental design is rigorous.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive setup with 11 models $\times$ 3 datasets $\times$ 5-fold cross-validation + Grad-CAM.
- Writing Quality: ⭐⭐⭐⭐ Clear organization, detailed experimental records, and open-source code.
- Value: ⭐⭐⭐⭐ Provides crucial empirical direction for the medical AI community, fostering reasonable resource allocation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)
- [\[CVPR 2025\] Show and Segment: Universal Medical Image Segmentation via In-Context Learning](show_and_segment_universal_medical_image_segmentation_via_in-context_learning.md)
- [\[CVPR 2025\] Latent Drifting in Diffusion Models for Counterfactual Medical Image Synthesis](latent_drifting_in_diffusion_models_for_counterfactual_medical_image_synthesis.md)
- [\[CVPR 2026\] Simple-ViLMedSAM: Simple Text Prompts Meet Vision-Language Models for Medical Image Segmentation](../../CVPR2026/medical_imaging/simple-vilmedsam_simple_text_prompts_meet_vision-language_models_for_medical_ima.md)
- [\[CVPR 2025\] Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation](decoding_matters_efficient_mamba-based_decoder_with_distribution-aware_deep_supe.md)

</div>

<!-- RELATED:END -->
