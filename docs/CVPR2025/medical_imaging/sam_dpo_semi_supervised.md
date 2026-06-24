---
title: >-
  [Paper Note] Enhancing SAM with Efficient Prompting and Preference Optimization for Semi-supervised Medical Image Segmentation
description: >-
  [CVPR 2025][Medical Imaging][SAM] This paper proposes an enhanced SAM framework that generates unsupervised semantic/locational/shape prompts using BiomedCLIP, VQA, and GPT-4, and introduces a DPO-inspired preference alignment loss to simulate human feedback, achieving superior performance in lung, breast tumor, and abdominal organ segmentation under a semi-supervised setup with only 10% labeled data.
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "SAM"
  - "Semi-supervised Segmentation"
  - "Preference Optimization"
  - "Unsupervised Prompting"
date: 2026-05-08
content_hash: 354a819220a48d56
---

# Enhancing SAM with Efficient Prompting and Preference Optimization for Semi-supervised Medical Image Segmentation

**Conference**: CVPR 2025  
**arXiv**: [2503.04639](https://arxiv.org/abs/2503.04639)  
**Code**: None  
**Area**: LLM Alignment  
**Keywords**: SAM, Semi-supervised Segmentation, Preference Optimization, Unsupervised Prompting, Medical Imaging

## TL;DR
This paper proposes an enhanced SAM framework that generates unsupervised semantic/locational/shape prompts using BiomedCLIP, VQA, and GPT-4, and introduces a DPO-inspired preference alignment loss to simulate human feedback, achieving superior performance in lung, breast tumor, and abdominal organ segmentation under a semi-supervised setup with only 10% labeled data.

## Background & Motivation
1. **Background**: Foundation models like SAM have been extended to medical image segmentation (SAM-Med2D, MedSAM), but still rely on geometric prompts (points/boxes) provided by experts, requiring a large amount of annotated data.
2. **Limitations of Prior Work**: (1) Prompt generation relies on manual effort, which is highly inefficient; (2) many medical datasets lack comprehensive annotations, restricting the utilization of data-intensive foundation models; (3) existing human-in-the-loop approaches require complex domain knowledge and independent training of reward functions.
3. **Key Challenge**: Foundation models require a large amount of annotations, but medical annotation is extremely expensive; although human-in-the-loop feedback methods reduce annotation requirements, they require complex reward modeling and cannot be trained end-to-end.
4. **Goal**: (1) Design a human-free prompt generation scheme; (2) bypass reward function training and achieve end-to-end alignment using simple preference scores.
5. **Key Insight**: Utilize BiomedCLIP/VQA/GPT-4 to provide unsupervised prompts with semantic, locational, and general information; replace traditional reward function training in RLHF with DPO loss.
6. **Core Idea**: Unsupervised multi-source prompting + DPO-driven semi-supervised preference alignment.

## Method

### Overall Architecture
Stage 1 (Fine-tuning with 10% labeled data): Input image $\rightarrow$ SAM-Med2D encoder + BiomedCLIP (saliency map $\rightarrow$ box prompt) + MedVInT (shape/position text) + GPT-4 (general disease information) $\rightarrow$ prompt encoder $\rightarrow$ mask decoder $\rightarrow$ segmentation map. Stage 2 (Alignment with the remaining unlabeled data): Generate 4 segmentation candidates using multiple thresholds $\rightarrow$ score by virtual annotator $\rightarrow$ fine-tune decoder via DPO-inspired loss.

### Key Designs

1. **Multi-source Unsupervised Prompt Generation**:

    - **Function**: Automatically generate comprehensive prompts containing semantic, locational, and shape information without human intervention.
    - **Mechanism**: (1) Visual prompts: BiomedCLIP + gScoreCAM are used to generate saliency maps $\rightarrow$ CRF post-processing $\rightarrow$ extraction of bounding boxes and point coordinates; (2) Text prompts: MedVInT answers VQA questions about organ/tumor shape and location (e.g., "What is the shape of the liver?"); (3) General knowledge: GPT-4 provides general descriptions of diseases/organs. The three types of prompts are concatenated and input into the prompt encoder.
    - **Design Motivation**: Prompts in existing SAM methods either require experts (points/boxes) or rely solely on semantic information while lacking locational/shape information. Multi-source information complements each other to provide stronger signals.

2. **DPO-inspired Preference Alignment Loss**:

    - **Function**: Leverage unlabeled data to improve segmentation quality by simulating human preference feedback, eliminating the need to train an independent reward function.
    - **Mechanism**: Generate 4 segmentation candidates for each image at different thresholds (0.3, 0.4, 0.5, 0.6), and score them in bins (grades 1–4) according to their IoU with the GT. The loss function extends the standard DPO to 4 candidates: $\mathcal{L}_{\text{DPO}} = -\mathbb{E}[\log\sigma(\beta_1\log\frac{\pi_\psi(Y_1|I)}{\pi_{\text{fine}}(Y_2|I)} + \beta_2\log\frac{\pi_\psi(Y_2|I)}{\pi_{\text{fine}}(Y_2|I)} - \beta_2\log\frac{\pi_\psi(Y_3|I)}{\pi_{\text{fine}}(Y_3|I)} - \beta_1\log\frac{\pi_\psi(Y_4|I)}{\pi_{\text{fine}}(Y_4|I)})]$, where $\beta_1 > \beta_2$ to assign larger weights to the best and worst candidates.
    - **Design Motivation**: DPO does not require training an independent reward function; the model itself acts as the reward model. Graduated weights for 4 candidates provide richer preference signals than simple pairwise comparisons.

3. **Virtual Annotator Scoring Mechanism**:

    - **Function**: Simulate the quality assessment process of human annotators to provide supervision signals for preference alignment.
    - **Mechanism**: Score candidates using IoU bins (<0.4, 0.4-0.55, 0.55-0.7, >0.7). Although GT is used to calculate IoU, it is only used for scoring rather than direct supervision, satisfying the semi-supervised setup. Ranking can also be used instead of scoring.
    - **Design Motivation**: Simulate simple good/bad judgments of segmentation quality by annotators in real scenarios, without requiring precise pixel-level annotations.

### Loss & Training
Stage 1: Focal Loss + Dice Loss (20:1 weight), trained for 15 epochs on 10% labeled data. Stage 2: DPO loss, trained for 30 epochs on the remaining unlabeled data. $\beta_1=1, \beta_2=0.5$. Adam optimizer, lr=1e-4, halved every 10 epochs.

## Key Experimental Results

### Main Results

| Method | Chest X-ray (20% data) Dice | Breast USD (20%) Dice | AMOS CT (20%) mDice |
|------|---------------------------|---------------------|-------------------|
| U-Net | 58.66 | 57.35 | 59.35 |
| nnU-Net | 60.97 | 59.47 | 65.21 |
| SAM-Med2D | 67.81 | 63.72 | 66.57 |
| **Ours (10% + 10% unlabeled)** | **78.87** | **75.88** | **77.69** |

### Ablation Study

| Configuration | Chest X-ray Dice | Description |
|------|-----------------|------|
| Full (Prompt + Alignment) | 78.87 | Full model |
| - Alignment (Prompt only, 20% labeled) | 79.13 | Fully supervised prompt baseline |
| - Alignment (Prompt only, 10% labeled) | 75.60 | Dropped by 3.5% with halved labels |
| - Alignment - VQA | 73.35 | VQA prompt contribution +2.25% |
| - Alignment - VQA - GPT4 | 72.76 | GPT4 contribution +0.59% |
| - Alignment - VQA - CAM (10%) | 57.02 | GPT4 text-only is extremely poor |

### Key Findings
- Within the 10-50% data range, the proposed method consistently outperforms all fully-supervised SOTA methods, demonstrating the advantages of semi-supervised learning.
- The preference alignment mechanism achieves performance close to the 20% fully-supervised prompt method when using only 10% labeled + 10% unlabeled data.
- The ranking strategy is slightly better than the scoring strategy, and both significantly outperform the baseline that only uses the best candidate.
- The BiomedCLIP saliency map is the most crucial prompt component (contributing +15.74%), with VQA and GPT-4 providing incremental improvements.
- Stage 1 uses Focal Loss + Dice Loss (20:1 weight), trained for 15 epochs on 10% labeled data; Stage 2 uses DPO loss to train on the remaining unlabeled data for 30 epochs. Adam optimizer, lr=1e-4, halved every 10 epochs.
- Four segmentation candidates are generated via multiple thresholds (thresholds 0.3/0.4/0.5/0.6), scored into IoU bins (<0.4/0.4-0.55/0.55-0.7/>0.7 corresponding to grades 1-4), with DPO weights $\beta_1=1, \beta_2=0.5$ giving more weight to the best and worst candidates.

## Highlights & Insights
- **Novel migration of DPO from language models to medical segmentation**: Using thresholding to generate segmentation candidates instead of diverse generation in language models, and using IoU scoring instead of human preference annotations.
- **Practicality of multi-source unsupervised prompts**: The combination of BiomedCLIP + VQA + GPT-4 provides comprehensive and expert-free prompt information, which can be generalized to other medical tasks.
- **Practical significance of the semi-supervised paradigm**: Achieving good performance with only 10% labeled data significantly reduces the cost of medical image annotation.
- **Importance analysis of prompt components**: The BiomedCLIP saliency map (CAM) is the core component—performance plummets from 75.60% to 57.02% (-18.58%) after removal. VQA prompts contribute +2.25%, and GPT-4 general knowledge contributes +0.59%. The three types of prompt information complement each other: visual prompts locate the target region, VQA text prompts describe shape and location, and GPT-4 provides general knowledge about the disease.

## Limitations & Future Work
- Virtual annotators still use GT to calculate IoU, which requires alternatives (such as uncertainty-based scoring) during real deployment.
- 3D segmentation (AMOS-CT) is processed slice-by-slice, failing to fully utilize 3D information.
- Under the 100% data setting, it performs on par with or slightly below fully supervised methods, indicating a limited upper bound of preference alignment. However, its advantages are highly bottleneck-breaking in low-label scenarios of 10-50%, which is the most valuable range in practical deployment.
- Future work can explore truly GT-free preference evaluation methods and more efficient 3D extensions.
- The quality of text prompt information from MedVInT and GPT-4 depends on the coverage of pre-trained models over the medical domain, and the performance may be limited in rare disease scenarios.
- The weight settings of the 4 candidates in the DPO loss ($\beta_1=1, \beta_2=0.5$) are empirical, lacking theoretical optimality analysis.
- Visual prompts are generated using BiomedCLIP + gScoreCAM to produce saliency maps, post-processed by CRF to extract bounding boxes and point coordinates. MedVInT answers VQA questions regarding organ/tumor shape and location (e.g., "What is the shape of the liver?"), and GPT-4 provides general descriptions of diseases/organs. The three types of prompts are concatenated and input into the prompt encoder.
- Evaluated on three datasets of different modalities: Chest X-ray, Breast USD, and AMOS CT, covering 2D X-rays, ultrasound, and 3D CT.

## Related Work & Insights
- **vs SAM-Med2D**: SAM-Med2D requires fully-supervised geometric prompts, whereas the proposed method employs unsupervised prompting + semi-supervised alignment.
- **vs MedCLIP-SAM**: MedCLIP-SAM only uses CLIP semantic prompts to generate pseudo-labels, while the proposed multi-source prompts + preference alignment is more comprehensive.
- **vs Self-Prompt SAM**: Self-Prompt self-generates prompts from output masks, while this method provides stronger external knowledge prompt signals.
- **vs nnU-Net**: nnU-Net achieves a Dice of only 60.97% (Chest X-ray) with 20% data, while ours achieves 78.87% with 10% labeled + 10% unlabeled data, leading by +17.90%.

## Rating

### Implementation Details
Based on SAM-Med2D encoder. Stage 1: Focal Loss + Dice Loss (20:1), 15 epochs.
Stage 2: DPO loss, 30 epochs. Adam, lr=1e-4, halved every 10 epochs.
- **Novelty**: ⭐⭐⭐⭐ Innovative application of DPO in medical segmentation, novel multi-source prompt design
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three datasets across three modalities, multiple data ratios, comprehensive ablations
- **Writing Quality**: ⭐⭐⭐⭐ Clear framework, detailed experiments
- **Value**: ⭐⭐⭐⭐⭐ Extremely high practical value in low-label scenarios

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing_semi-supervised_medical_image.md)
- [\[CVPR 2025\] Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation](decoding_matters_efficient_mamba-based_decoder_with_distribution-aware_deep_supe.md)
- [\[CVPR 2025\] Interactive Medical Image Segmentation: A Benchmark Dataset and Baseline](interactive_medical_image_segmentation_a_benchmark_dataset_and_baseline.md)
- [\[CVPR 2025\] BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)
- [\[CVPR 2025\] Revisiting MAE Pre-Training for 3D Medical Image Segmentation](revisiting_mae_pre-training_for_3d_medical_image_segmentation.md)

</div>

<!-- RELATED:END -->
