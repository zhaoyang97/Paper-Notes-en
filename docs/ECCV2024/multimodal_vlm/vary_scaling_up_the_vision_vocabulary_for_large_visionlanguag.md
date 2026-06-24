---
title: >-
  [Paper Note] Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models
description: >-
  [ECCV 2024][Multimodal VLM][vision vocabulary] This paper proposes the Vary method, which scales up the vision vocabulary of Large Vision-Language Models (LVLMs) by generating and integrating a new vision vocabulary. This empowers the model with new fine-grained visual perception capabilities, such as document-level OCR and chart understanding, while maintaining its original general capabilities.
tags:
  - "ECCV 2024"
  - "Multimodal VLM"
  - "vision vocabulary"
  - "LVLM"
  - "document OCR"
  - "chart understanding"
  - "fine-grained perception"
date: 2026-05-08
content_hash: bb6aec7c3875a774
---

# Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models

**Conference**: ECCV 2024  
**arXiv**: [2312.06109](https://arxiv.org/abs/2312.06109)  
**Code**: [https://github.com/Ucas-HaoranWei/Vary](https://github.com/Ucas-HaoranWei/Vary)  
**Area**: Multimodal/VLM  
**Keywords**: vision vocabulary, LVLM, document OCR, chart understanding, fine-grained perception  

## TL;DR
This paper proposes the Vary method, which scales up the vision vocabulary of Large Vision-Language Models (LVLMs) by generating and integrating a new vision vocabulary. This empowers the model with new fine-grained visual perception capabilities, such as document-level OCR and chart understanding, while maintaining its original general capabilities.

## Background & Motivation
Most popular LVLMs (e.g., BLIP-2, MiniGPT-4, LLaVA, Qwen-VL) rely on the same vision vocabulary—CLIP-ViT. Trained via contrastive learning on over 400M text-image pairs, CLIP covers the majority of natural images and common visual tasks. However, for specialized visual tasks (such as document-level OCR and chart understanding, especially in non-English scenarios), CLIP's vision vocabulary exhibits several limitations of prior work:

1. **Low Encoding Efficiency**: CLIP struggles to efficiently encode dense text information from documents/charts into a fixed number of tokens (typically 256).
2. **Visual OOV Problem**: Images from these specialized scenarios are like "foreign languages" to CLIP, causing out-of-vocabulary (OOV) issues.
3. **Limitations of Prior Work**: mPlug-Owl and Qwen-VL mitigate this by unfreezing CLIP, but this can overwrite existing knowledge, suffers from low training efficiency, and cannot be trained for multiple epochs due to the strong memorization capacity of LLMs.

**Core Idea**: In the NLP field, migrating an English LLM to Chinese requires expanding the text vocabulary to improve encoding efficiency. Similarly, for visual "foreign language" images that CLIP is not adept at handling, the vision vocabulary also needs to be expanded.

## Core Problem
How can one simply and efficiently expand the vision vocabulary for LVLMs to equip them with new capabilities like fine-grained document/chart perception without undermining their original general capabilities?

## Method

### Overall Architecture
The pipeline of Vary is divided into two phases:

1. **Vision Vocabulary Generation (Vary-tiny)**: Train a small pipeline (vocabulary network + OPT-125M) to generate a new vision vocabulary in an autoregressive manner.
2. **Vision Vocabulary Integration (Vary-base)**: Merge the new vocabulary in parallel with the original CLIP vocabulary to equip the LVLM with new capabilities.

### Key Designs

1. **New Vision Vocabulary Network**: A ViTDet (base scale) pre-trained with SAM is used as the backbone. The input resolution is $1024 \times 1024$, stride is 16, and the feature map of the last layer is $64 \times 64 \times 256$. Two convolutional layers are added back-to-back to perform token merging: the first conv (kernel size=3) transforms features to $32 \times 32 \times 512$, and the second conv further transforms them to $16 \times 16 \times 1024$. After flattening, a shape of $256 \times 1024$ is obtained, which aligns with the output shape of CLIP-ViT.

2. **Autoregressive Vocabulary Training (Vary-tiny)**: Vary-tiny is composed of the vocabulary network + OPT-125M small model. The training data consists of:

    - **Positive samples**: Document data (1M Chinese + 1M English pages, extracted from arXiv and CC-MAIN PDFs); chart data ($250\text{k} \times 2$ rendered by matplotlib + $500\text{k} \times 2$ rendered by pyecharts, half Chinese and half English, annotated in python-dict format).
    - **Negative samples**: COCO 120k natural images, with text annotations set as fixed simple sentences (e.g., "It's an image of nature"), ensuring that the new vocabulary does not produce noise over natural images.

3. **Dual Vocabulary Parallel Integration (Vary-base)**: The new and old vision vocabularies are each equipped with an independent linear input embedding layer (input 1024 $\rightarrow$ output 2048). After concatenating the two paths of tokens, the channel size becomes 4096, aligning with the input dimension of the LLM (Qwen-7B or Vicuna-7B). During the phrase of integration, both vision vocabulary networks are **frozen**, and only the input embedding layers and the LLM are trained.

4. **High-Quality Synthetic Data**: Additional data compiled for Vary-base includes:

    - LaTeX rendered documents (0.5M English + 0.4M Chinese, supporting formulas/tables, annotated in Mathpix markdown format).
    - Semantically aligned charts (synthesized using GPT-4 to generate semantically coherent chart contents, with an additional 200k rendered).
    - General data: 4M LAION-COCO image-text pairs (pre-training) + LLaVA-80k/665k + DocVQA/ChartQA training sets (SFT).

### Loss & Training
- **Vary-tiny Training**: Standard autoregressive next-token prediction loss. Batch size 512, trained for 3 epochs with AdamW optimizer and cosine annealing, lr=5e-5.
- **Vary-base Training**: Two phases—pre-training (lr=5e-5, batch 256, 1 epoch) + SFT (lr=1e-5, batch 256, 1 epoch). All weights of the vision vocabulary networks are frozen, and only the input embedding layers and the LLM are trained.
- Input format: `<img>"<image>"</img> "text"`, with autoregressive text output.

## Key Experimental Results

| Dataset/Task | Metric | Vary-base | Compared Method | Remarks |
|------------|------|-----------|----------|------|
| DocVQA (test) | ANLS | **78.2%** | Qwen-VL 65.1%, Pix2Struct 72.1% | Significant Outperformance |
| DocVQA (val) | ANLS | **76.3%** | - | - |
| ChartQA (avg) | Relaxed Acc | **66.1%** (665k) | Qwen-VL 65.7%, Matcha 64.2% | Comparable/Slightly Superior |
| ChartQA (human) | Relaxed Acc | **43.8%** | Matcha 38.2% | +5.6% |
| ChartQA (aug) | Relaxed Acc | **88.3%** | Matcha 90.2% | Slightly Lower |
| MMVet (total) | Score | **36.2%** (Qwen-7B) | LLaVA-13B 32.9%, LLaVA1.5-7B 30.5% | Outperforms larger models |
| English Doc OCR | Edit Distance | **0.106** | Nougat 0.126 | Better |
| English Doc $\rightarrow$ Markdown | Edit Distance | **0.181** | Nougat 0.245 | Substantially Improved |
| English Doc $\rightarrow$ Markdown | F1 | **81.10%** | Nougat 79.97% | +1.13% |
| Chinese Doc OCR | Edit Distance | 0.174 | - | Unique Capability |

### Ablation Study
- **Vary-tiny vs Vary-base**: Vary-tiny (only OPT-125M) already demonstrates Chinese and English OCR capabilities (Chinese Edit Distance 0.266, English 0.197), validating the effectiveness of the new vision vocabulary; upgrading to the 7B LLM yields further improvements.
- **Role of Negative Samples**: Using COCO natural images as negative samples during training ensures that the new vocabulary does not generate noise on natural images, thereby preserving the general capabilities of the model.
- **Vicuna vs Qwen LLM**: Under the same settings, Vary-base (Vicuna-7B, 665k) achieves 32.9% on MMVet, matching LLaVA-13B. Switching to Qwen-7B boosts this to 36.2%, indicating that the Chinese capability of the LLM is crucial for the overall performance.
- **SFT Data Volume**: Upgrading from LLaVA-80k to LLaVA-665k raises ChartQA performance from 65.3% to 66.1%, while DocVQA remains stable (78.2% vs 78.1%).

## Highlights & Insights
- **A New Paradigm for Vision Vocabulary Expansion**: This work is the first to address LVLM capability enhancements from the perspective of "expanding the vision vocabulary", analogous to text vocabulary expansion in NLP. The concept is clean and inspiring.
- **Autoregressive Vocabulary Generation**: Compared to CLIP's contrastive learning, the autoregressive method is better suited for dense perception tasks, enabling higher compression of longer text and supporting more diverse data formats (such as prompt-based VQA data).
- **Decoupled Training Strategy**: The strategy first trains the new vocabulary independently (using a small, low-cost model) and then freezes and integrates it into the large model. This avoids knowledge overwriting and ensures high training efficiency.
- **Powerful Synthetic Data Engine**: A large-scale Chinese-English document and chart synthesis pipeline was constructed (PDF extraction + LaTeX rendering + pyecharts/matplotlib rendering), which holds high reusable value.
- **Chinese Language Capability**: The model features Chinese document OCR and Chinese chart understanding, which was a rare capability at the time.

## Limitations & Future Work
1. **Resolution Constraints**: Although the new vocabulary network accepts $1024 \times 1024$ inputs, the handling of higher resolutions (e.g., 2K/4K documents) remains limited.
2. **Fixed Number of Vocabularies**: The current scheme only expands a single new vision vocabulary. For multiple different types of "visual foreign language" tasks, a more flexible expansion mechanism might be required.
3. **Limited Evaluation Scope**: The evaluations mainly focus on document and chart tasks; other potential areas that could benefit from vocabulary expansion (e.g., remote sensing, medical imaging) are not yet validated.
4. **Fixed Token Count**: Each of the old and new vocabularies outputs 256 tokens, totaling 512 tokens, which might still be insufficient for ultra-long documents.
5. **Bottleneck of the Small Autoregressive Model**: Vary-tiny employs OPT-125M as the decoder, and its capacity might limit the upper-bound quality of the vision vocabulary.

## Related Work & Insights

| Method | Vision Vocabulary | Is Frozen | Special Capability | LLM |
|------|-----------|---------|---------|-----|
| LLaVA | CLIP-L | Frozen | General VQA/Dialogue | Vicuna |
| BLIP-2 | CLIP + Q-Former | Frozen | General VQA/Dialogue | OPT/FlanT5 |
| Qwen-VL | CLIP-G + Cross-Attention | Unfrozen | +OCR/Grounding | Qwen-7B |
| mPlug-Owl | CLIP-L | Unfrozen | +General Enhancement | LLaMA |
| Nougat | Swin Transformer | - | Specialized Doc Parsing | mBART decoder |
| **Ours (Vary)** | **CLIP-L + New Vocabulary** | **Double Frozen** | **+Doc OCR + Chart Understanding** | **Qwen-7B/Vicuna** |

Vary's unique approach lies in not modifying the original CLIP, but instead adding an extra visual pathway to expand capabilities in an "addition" rather than a "rewrite" manner.

## Related Work & Insights
1. **Analogous Thinking of Vision Vocabulary**: Translating the concept of NLP vocabulary expansion to the vision side suggests that the capability bottleneck of LVLMs might lie not only on the LLM side but also heavily on the representation capacity of the vision encoder.
2. **Critical Role of Synthetic Data**: Vary validates the effectiveness of large-scale synthetic data (document rendering, chart rendering) for specific visual tasks, offering ideas for data-scarce scenarios.
3. **Connection to Subsequent Works of Vary**: The Vary series subsequently led to variants like Vary-toy (smaller scale) and Mini-Monkey, continuously advancing document understanding.
4. **Insights on Multi-Vision Encoder Fusion**: The parallel dual-encoder + frozen fusion strategy in Vary has served as a reference for subsequent multi-encoder LVLMs such as InternVL and Cambrian.

## Rating
- Novelty: ⭐⭐⭐⭐ The first to optimize LVLMs from the perspective of vision vocabulary expansion, with a clean and inspiring approach.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple dimensions such as document OCR, DocVQA, ChartQA, and MMVet, with relatively thorough ablations.
- Writing Quality: ⭐⭐⭐⭐ Smooth writing, appropriate analogies, and clear structure.
- Value: ⭐⭐⭐⭐⭐ Pioneered a new research direction in vision vocabulary expansion, with significant subsequent influence (Vary series, multi-encoder LVLMs, etc.).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Attention Prompting on Image for Large Vision-Language Models](attention_prompting_on_image_for_large_visionlanguage_models.md)
- [\[ECCV 2024\] MarvelOVD: Marrying Object Recognition and Vision-Language Models for Robust Open-Vocabulary Object Detection](marvelovd_marrying_object_recognition_and_vision-language_models_for_robust_open.md)
- [\[ECCV 2024\] Robust Calibration of Large Vision-Language Adapters](robust_calibration_of_large_vision-language_adapters.md)
- [\[CVPR 2026\] Vocabulary Scaling Law: Tuning Open-vocabulary Predictors for Their Openness](../../CVPR2026/multimodal_vlm/vocabulary_scaling_law_tuning_open-vocabulary_predictors_for_their_openness.md)
- [\[CVPR 2026\] Do Vision-Language Models Measure Up? Benchmarking Visual Measurement Reading with MeasureBench](../../CVPR2026/multimodal_vlm/do_vision-language_models_measure_up_benchmarking_visual_measurement_reading_wit.md)

</div>

<!-- RELATED:END -->
