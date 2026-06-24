---
title: >-
  [Paper Note] MarineInst: A Foundation Model for Marine Image Analysis with Instance Visual Description
description: >-
  [ECCV 2024][Self-Supervised Learning][Marine Image Analysis] This paper proposes MarineInst, a foundation model for marine image analysis that simultaneously outputs instance masks and semantic descriptions. Additionally, it constructs MarineInst20M—the largest marine image dataset to date (20 million images), supporting multi-level marine visual analysis tasks from image-level scene understanding to region-level instance understanding.
tags:
  - "ECCV 2024"
  - "Self-Supervised Learning"
  - "Marine Image Analysis"
  - "Foundation Models"
  - "Instance Segmentation"
  - "Visual Captioning"
  - "Large-scale Datasets"
date: 2026-05-08
content_hash: 08b3b6c39f09099a
---

# MarineInst: A Foundation Model for Marine Image Analysis with Instance Visual Description

**Conference**: ECCV 2024  
**Code**: [https://marineinst.hkustvgd.com](https://marineinst.hkustvgd.com)  
**Area**: Self-Supervised Learning / Foundation Models  
**Keywords**: Marine Image Analysis, Foundation Models, Instance Segmentation, Visual Captioning, Large-scale Datasets

## TL;DR
This paper proposes MarineInst, a foundation model for marine image analysis that simultaneously outputs instance masks and semantic descriptions. Additionally, it constructs MarineInst20M—the largest marine image dataset to date (20 million images), supporting multi-level marine visual analysis tasks from image-level scene understanding to region-level instance understanding.

## Background & Motivation

**Background**: Large-scale foundation models (e.g., SAM, CLIP, GPT-4V) have made significant progress in general vision tasks, but their application in the marine domain remains extremely limited. Although marine ecosystems cover over 70% of the Earth's surface and contain a vast array of organisms and terrains requiring monitoring and analysis, marine visual analysis still heavily relies on manual annotation and small, domain-specific models.

**Limitations of Prior Work**: (1) **Data Scarcity**: Compared to terrestrial images (e.g., ImageNet and COCO which have millions of annotations), annotated marine image data is extremely scarce, with existing marine datasets being small in scale and covering limited species; (2) **Large Domain Gap**: Marine images differ significantly from general images in terms of lighting (underwater scattering, color shift), perspectives (top-down, underwater, long-range), and content (diverse biological morphologies, complex backgrounds), making direct utilization of existing foundation models ineffective; (3) **Limitations of Existing Models**: Although SAM can generate masks, it lacks semantic information; CLIP provides image-level understanding but cannot perform pixel-level segmentation; multimodal LLMs like LLaVA are restricted to image-level scene descriptions. No single model can simultaneously provide both instance-level masks and semantic descriptions.

**Key Challenge**: Marine visual analysis requires both precise instance segmentation capability and rich semantic understanding. However, existing foundation models either only perform segmentation (lacking semantics) or only perform image-level description (lacking instance-level localization), and they generalize poorly to the marine domain.

**Goal**: (1) How to construct a marine image dataset that is sufficiently large and of high quality to train foundation models? (2) How to design a unified model that can simultaneously output instance masks and semantic descriptions? (3) How to ensure the model exhibits good generalization capabilities across diverse downstream marine analysis tasks?

**Key Insight**: The authors propose a strategy emphasizing both "data" and "model." On the data side, an automated pipeline is designed to construct the large-scale marine dataset MarineInst20M. On the model side, instance segmentation and visual captioning are unified into an "instance visual description" task, training a foundation model capable of simultaneously outputting both masks and textual descriptions.

**Core Idea**: Construct a 20-million-scale marine image dataset and train a unified segmentation-description foundation model to achieve comprehensive marine visual analysis ranging from the image level to the instance level.

## Method

### Overall Architecture
The MarineInst system consists of two major components: (1) **Data Engine MarineInst20M**—which collects, cleans, and annotates 20 million marine images through a semi-automated pipeline, providing instance masks and multi-granularity textual descriptions; (2) **Foundation Model MarineInst**—which adds a semantic description branch to SAM's segmentation architecture, taking marine images as input and outputting masks along with corresponding textual descriptions (e.g., species name, morphological features, behavioral states) for each instance.

### Key Designs

1. **Binary Instance Filtering Pipeline**:

    - **Function**: Automatically generate high-quality marine instance segmentation masks to solve the annotation data scarcity problem.
    - **Mechanism**: A two-stage strategy is adopted. In the first stage, an existing segmentation model (SAM) is used to generate candidate masks for marine images. However, SAM's output in the marine domain contains considerable noise (e.g., mis-segmenting coral backgrounds, merging adjacent fish schools into a single mask). In the second stage, a binary classifier (Binary Instance Filter) is trained to determine whether each candidate mask output by SAM is a valid single instance. The classifier is trained using a small amount of manual annotations (thousands of "good/bad" mask labels) but can process millions of candidate masks. Through this filtering step, approximately 30% of high-quality masks are retained, while noisy, fragmented, and merged masks are discarded.
    - **Design Motivation**: Fully manual annotation of 20 million images is impractical, whereas directly using SAM's outputs yields insufficient quality. The binary filter only needs to judge "good or bad" (such an assessment is much simpler than annotating from scratch) and has minimal training data requirements, making it an efficient quality control method.

2. **Multi-Granularity Instance Captioning**:

    - **Function**: Generate textual descriptions of varying granularities from coarse to fine for each instance mask.
    - **Mechanism**: A vision-language model (VLM, such as InstructBLIP) is utilized to generate descriptions for cropped instance regions. Prompts of three levels of granularity are designed: (1) **Species level**—"What marine organism is this?" $\rightarrow$ outputting e.g., "yellow clownfish"; (2) **Attribute level**—"Describe the appearance features of this instance" $\rightarrow$ outputting e.g., "orange body, white stripes, approximately 10 cm"; (3) **Behavioral level**—"What is this instance doing?" $\rightarrow$ outputting e.g., "swimming and foraging in a coral reef." These three granularities of descriptions are stored as structured annotations, and the appropriate level can be chosen during training according to downstream task requirements.
    - **Design Motivation**: Different downstream tasks require different granularities of semantic information—species identification requires category names, ecological studies require behavioral descriptions, and biological surveys require detailed attributes. Multi-granularity descriptions allow a single dataset to support multiple tasks.

3. **Unified Segmentation-Description Architecture**:

    - **Function**: Simultaneously output instance masks and corresponding textual descriptions within a single forward pass.
    - **Mechanism**: Based on SAM's encoder-decoder architecture, an additional text decoding branch is introduced. The image encoder (ViT-H/L) extracts visual features, the mask decoder retains SAM's original structure to output instance masks, and a newly added description decoder receives instance features aligned with the mask (extracted from the mask area via ROI Align), then yields description texts through an autoregressive text generator. The two decoders share the visual features from the encoder, and the mask decoder's output further guides the description decoder via an attention mechanism to focus on the correct instance region.
    - **Design Motivation**: Decoupling segmentation and description into two independent branches allows the reuse of SAM's powerful segmentation capability while ensuring that the description aligns with the correct instance via mask guidance. This is more controllable and precise than end-to-end multimodal LLM alternatives.

### Loss & Training
The total loss consists of three components: (1) mask segmentation loss (Dice Loss + Focal Loss, following SAM's design); (2) IoU prediction loss (predicting mask quality); (3) text generation loss (cross-entropy loss, generating instance descriptions autoregressively). Training is split into three stages: the first stage pretrains the encoder and mask decoder on MarineInst20M; the second stage jointly trains the description decoder; the third stage jointly fine-tunes the entire model on a high-quality subset. Training employs the AdamW optimizer with a batch size of 64, distributed across 16 A100 GPUs.

## Key Experimental Results

### Main Results

| Task/Dataset | Metric | MarineInst | SAM | Domain SOTA | Gain |
|------------|------|-----------|-----|---------|------|
| Marine Instance Segmentation (MarineDet) | AP | 42.8 | 28.3 | 36.5 | +6.3 vs Domain SOTA |
| Coral Segmentation (CoralSeg) | mIoU | 71.2 | 54.6 | 65.8 | +5.4 vs Domain SOTA |
| Fish Detection (Fish4Knowledge) | mAP | 68.5 | 41.2 | 62.3 | +6.2 vs Domain SOTA |
| Instance Captioning Quality | CIDEr | 89.3 | N/A | 72.1* | +17.2 |

\*Captioning quality of general VLMs on marine data

### Ablation Study

| Configuration | AP | CIDEr | Description |
|------|-----|-------|------|
| Full MarineInst | 42.8 | 89.3 | Full model |
| w/o Binary Filtering | 38.1 | 82.6 | Noisy masks degrade training quality |
| w/o Multi-Granularity Captioning | 42.5 | 71.8 | Using only single-granularity descriptions |
| w/o Mask-Guided Captioning | 41.9 | 75.2 | Descriptions and masks are misaligned |
| SAM Direct Fine-tuning | 39.7 | N/A | No description branch |

### Key Findings
- **Data Quality Over Data Quantity**: Discarding 70% of noisy masks via binary filtering increases AP by 4.7 ($38.1 \rightarrow 42.8$), indicating that raw SAM outputs are highly noisy in the marine domain.
- **Multi-Granularity Captions Significantly Improve Semantic Quality**: Transitioning from single-granularity to three granularities boosts CIDEr from 71.8 to 89.3.
- **Mask Guidance is Crucial for Captioning Accuracy**: Without mask guidance, the captioner might attend to incorrect regions.
- MarineInst demonstrates zero-shot generalization capabilities across over 1,500 marine species.
- It retains reasonable performance on non-marine general datasets (e.g., COCO, with only a 2.3 drop in AP), showing that marine domain adaptation does not incur severe catastrophic forgetting of general abilities.

## Highlights & Insights
- **Engineering Innovation of the Data Engine**: Binary instance filtering serves as a general human-in-the-loop annotation quality control paradigm—training a filter with limited annotations to screen massive automatic annotations. This workflow can be extended to any domain where large-scale annotation is required but human labor is constrained (e.g., agriculture, industrial inspection).
- **Unification of Segmentation and Captioning**: By feeding instance-level visual features into the description generator via mask guidance, the model realizes "saying what it sees," which is more precise than image-level description. This architectural design can be generalized to other scenarios demanding instance-level understanding.
- **Long-term Value of the 20-Million-Scale Marine Dataset**: MarineInst20M not only serves the proposed model but also provides much-needed data infrastructure for the entire marine AI research community.

## Limitations & Future Work
- Multi-granularity descriptions rely heavily on the generation quality of the VLM; the VLM has limited recognition capabilities for rare marine species, potentially generating incorrect species names.
- Issues of color distortion and low contrast in underwater images are not specifically addressed; underwater image enhancement preprocessing could be incorporated.
- The dataset is mainly sourced from public web images, which may exhibit distribution bias toward specific geographic regions and common species.
- The model size is relatively large (ViT-H encoder), posing challenges for deployment on embedded devices such as autonomous underwater vehicles (AUVs).
- Temporal analysis capability is currently lacking—tracking and behavior recognition in underwater videos are critical requirements for marine ecological research, but the current model only processes individual frames.

## Related Work & Insights
- **vs SAM**: SAM provides class-agnostic masks but lacks semantic information and generalizes poorly in underwater domains; MarineInst comprehensively outperforms it in both segmentation accuracy (+14.5 AP) and semantic understanding.
- **vs DeepFish/FathomNet**: Previous marine datasets were scaled at thousands to tens of thousands of images; MarineInst20M is three orders of magnitude larger and covers a broader range of species.
- **vs GLIP/GroundingDINO**: These open-vocabulary detectors perform well in general domains but suffer in the marine domain due to insufficient vision-language data alignment.
- This "domain-specific foundation model" paradigm (first building large-scale domain data, then training a domain foundation model) is also applicable to other data-scarce fields such as agriculture, remote sensing, and medicine.

## Rating
- Novelty: ⭐⭐⭐⭐ Unifying segmentation and captioning into "instance visual description" is innovative, and the design of the data engine is practical.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple downstream marine tasks with extensive ablation and cross-domain experiments.
- Writing Quality: ⭐⭐⭐⭐ Clear and systematic descriptions of dataset construction and model design.
- Value: ⭐⭐⭐⭐⭐ The 20M-scale marine dataset and foundation model hold long-term value for the marine AI research community, filling a critical gap in the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] InfMAE: A Foundation Model in the Infrared Modality](infmae_a_foundation_model_in_the_infrared_modality.md)
- [\[ICCV 2025\] A Token-level Text Image Foundation Model for Document Understanding (TokenFD/TokenVL)](../../ICCV2025/self_supervised/a_tokenlevel_text_image_foundation_model_for_document_unders.md)
- [\[ICLR 2026\] CARL: Camera-Agnostic Representation Learning for Spectral Image Analysis](../../ICLR2026/self_supervised/carl_camera-agnostic_representation_learning_for_spectral_image_analysis.md)
- [\[ECCV 2024\] Efficient Image Pre-Training with Siamese Cropped Masked Autoencoders](efficient_image_pre-training_with_siamese_cropped_masked_autoencoders.md)
- [\[ICML 2025\] Foundation Model Insights and a Multi-Model Approach for Superior Fine-Grained One-shot Subset Selection](../../ICML2025/self_supervised/foundation_model_insights_and_a_multi-model_approach_for_superior_fine-grained_o.md)

</div>

<!-- RELATED:END -->
