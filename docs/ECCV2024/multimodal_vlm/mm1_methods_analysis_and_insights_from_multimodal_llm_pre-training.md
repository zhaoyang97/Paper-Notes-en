---
title: >-
  [Paper Note] MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training
description: >-
  [ECCV 2024][Multimodal VLM][Multimodal Large Language Models] Apple systematically ablates the three primary axes of MLLM construction (architecture, data, and training), deriving key design principles: Image Resolution > Model Size > Training Data; the choice of VL connector type has minimal impact; and the meticulous blending of caption, interleaved, and text-only data is crucial. This systematically constructed MM1 model family (ranging from 3B-30B dense models to up to 64…
tags:
  - "ECCV 2024"
  - "Multimodal VLM"
  - "Multimodal Large Language Models"
  - "Pre-training"
  - "Ablation Studies"
  - "Vision Encoder"
  - "Data Recipe"
  - "Mixture-of-Experts"
date: 2026-05-08
content_hash: 1931e54369292dc8
---

# MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training

**Conference**: ECCV 2024  
**arXiv**: [2403.09611](https://arxiv.org/abs/2403.09611)  
**Code**: None (Internal Apple implementation using the AXLearn framework)  
**Area**: Multimodal VLM  
**Keywords**: Multimodal Large Language Models, Pre-training, Ablation Studies, Vision Encoder, Data Recipe, Mixture-of-Experts

## TL;DR

Apple systematically ablates the three primary axes of MLLM construction (architecture, data, and training), deriving key design principles: Image Resolution > Model Size > Training Data; the choice of VL connector type has minimal impact; and the meticulous blending of caption, interleaved, and text-only data is crucial. This systematically constructed MM1 model family (ranging from 3B-30B dense models to up to 64B MoE models) achieves state-of-the-art performance in few-shot pre-training evaluations.

## Background & Motivation

Existing MLLMs display a severe lack of transparency: closed-source models (such as GPT-4V and Gemini) disclose virtually no architectural or data details, while open-source models release weights but rarely reveal the **design decision process** (i.e., why a particular architecture, data ratio, or training strategy was chosen). It is argued that digesting reproducible design lessons holds more enduring value than any specific component implementation.

The core contribution of MM1 is not to propose a novel architecture, but rather to answer three pivotal design questions through **large-scale systematic ablations**:
1. How to select the vision encoder? Which is more important among resolution, model size, and pre-training objectives?
2. How to connect visual features to the LLM? Is it the connector architecture, token count, or resolution?
3. How to blend pre-training data? What are the optimal proportions for caption, interleaved, and text-only data?

## Method

### Overall Architecture

MM1 adopts a standard decoder-only MLLM architecture: **Vision Encoder** $\rightarrow$ **Vision-Language Connector (VL Connector)** $\rightarrow$ **LLM Decoder**. The input image is first processed by the encoder to extract features, then mapped into a sequence of visual tokens via the connector. These visual tokens are concatenated with text tokens and fed into the autoregressive LLM.

Ablation baseline configuration:
- Vision encoder: ViT-L/14 (CLIP loss, DFN-5B + VeCap-300M, 336×336)
- VL connector: C-Abstractor, 144 image tokens
- Pre-training data: 45% caption + 45% interleaved + 10% text-only
- LLM: 1.2B transformer decoder

### Key Design 1: Vision Encoder Selection

The authors compare two types of visual encoder pre-training objectives (using a 2.9B LLM to ensure sufficient capacity):

**Contrastive Learning (CLIP-style)**: Trained on large-scale image-text data, exhibiting strong semantic understanding but weaker dense prediction capabilities.

**Reconstructive Loss (AIM-style)**: Autoregressive reconstruction loss, which preserves global details of images and theoretically benefits tasks requiring fine-grained understanding such as VQA.

| Encoder | Architecture | Resolution | Pre-training Data | 0-shot | 4-shot | 8-shot |
|--------|------|--------|-----------|--------|--------|--------|
| AIM | ViT/600M | 224 | DFN-2B | 36.6 | 56.6 | 60.7 |
| AIM | ViT/1B | 224 | DFN-2B | 37.9 | 59.5 | 63.3 |
| AIM | ViT/3B | 224 | DFN-2B | 38.9 | 60.9 | 64.9 |
| CLIP | ViT-L | 224 | DFN-5B+VeCap | 36.9 | 58.7 | 62.2 |
| CLIP | ViT-H | 224 | DFN-5B+VeCap | 37.5 | 60.0 | 63.6 |
| CLIP | ViT-L | **336** | DFN-5B+VeCap | **39.9** | **62.4** | **66.0** |
| CLIP | ViT-H | **336** | DFN-5B+VeCap | **40.5** | **62.6** | **66.3** |
| CLIP | ViT-H | 378 | DFN-5B | **40.9** | 62.5 | 66.4 |

**Encoder Rule**: Image resolution has the most significant impact (approx. +3% from 224 to 336), followed by model size (only <1% from ViT-L to ViT-H), and training data (adding VeCap synthetic captions adds 1% to 2% to few-shot). The performance gap between CLIP and AIM is marginal when controlling variables, with CLIP being slightly superior overall.

### Key Design 2: Vision-Language Connector

The authors compare three VL connector architectures:

- **Average Pooling**: $n \times n$ average pooling + linear projection (similar to Emu2)
- **Attention Pooling**: Uses $k$ learnable queries for cross-attention
- **C-Abstractor**: Convolutional mapping based on ResNet blocks (proposed by Honeybee), preserving local information + adaptive pooling

Complete ablations across four settings (combining 64 and 144 tokens, with 224 and 336 resolutions) demonstrate:

**Connector Rule**: The number of visual tokens and image resolution are of primary importance, whereas the type of connector architecture has negligible impact. All three architectures perform almost identically in the 336px/144-token setting. This contradicts the findings of the Honeybee paper, suggesting that connector architecture discrepancies are smoothed out with scaling up training. C-Abstractor is ultimately selected solely because it performs slightly better in specific settings.

### Key Design 3: Pre-training Data Recipe

Utilizing three types of data, the authors summarize four key data principles:

| Data Type | Data Source | Scale |
|---------|--------|------|
| Captioned Images | CC3M, CC12M, HQIPT-204M, COYO, Web Image-Text-1B | 2 billion image-text pairs |
| Synthetic Captions | VeCap | 300 million image-text pairs |
| Interleaved Image-Text | OBELICS + Internal Data | 600 million documents |
| Text-only | Web pages, code, social media, encyclopedias, math | 2T tokens |

**Data Rule 1**: Interleaved data is "indispensable" for few-shot and text-only performance, whereas caption data improves zero-shot performance. Increasing the proportion of captions from 0% to 100% improves zero-shot performance from 25.8% to 39.3%; however, when the interleaved proportion falls below 50%, the 8-shot performance plunges from 61% to 45%.

**Data Rule 2**: Text-only data assists few-shot learning and text understanding. The combination of caption + text-only significantly enhances few-shot performance, while the interleaved + text-only mixture yields smaller gains but preserves pure text capabilities.

**Data Rule 3**: Meticulous blending can balance multimodal and text performance. The optimal ratio is determined to be caption:interleaved:text = 45:45:10 (approximately 5:5:1).

**Data Rule 4**: Synthetic caption data (VeCap), although constituting only 7% of the total dataset, contributes significantly to few-shot performance (+2.4% to 4%).

### Loss & Training

**Pre-training Configuration**:
- All parameters are fully unfrozen (vision encoder + LLM)
- Sequence length of 4096, up to 16 images per sequence ($378 \times 378$), batch size 512
- Trained for 200k steps (approx. 100B tokens)
- Learning rate determined via grid search on smaller models (9M $\rightarrow$ 1.2B) and extrapolated via linear regression in log space
- Extrapolation formula: $\eta = \exp(-0.4214 \ln(N) - 0.5535)$
- Weight decay: $\lambda = 0.1\eta$
- Cosine decay, with a 2,000-step warmup, decaying to 10% of the peak value
- Final actual LR for 3B/7B/30B: 6e-5 / 4e-5 / 2e-5

**MoE Scaling**:
- 3B-MoE: 64 experts, replaced every 2 layers, with total parameters of 64B
- 7B-MoE: 32 experts, replaced every 4 layers, with total parameters of 47B
- Top-2 gating + load balance loss (0.01) + router z-loss (0.001)
- Only FFN layers in the LLM decoder are replaced, leaving the vision encoder and VL connector unchanged

**SFT Configuration**:
- Approximately 1.45 million samples (LLaVA-Conv/Complex, ShareGPT-4V, academic VL datasets, etc.)
- 10k steps, batch size 256, sequence length 2048
- AdaFactor, LR 1e-5, cosine decay
- Both the vision encoder and LLM are unfrozen
- High-resolution: Position embedding interpolation + sub-image decomposition (SPHINX approach)
- Default SFT resolution is $1344 \times 1344$ (five $672 \times 672$ sub-images, totaling 720 tokens/image)

## Key Experimental Results

### Main Pre-training Results (Few-shot SOTA)

| Model | Shot | COCO | NoCaps | TextCaps | VQAv2 | TextVQA | VizWiz | OKVQA |
|------|------|------|--------|----------|-------|---------|--------|-------|
| Flamingo-3B | 0 | 73.0 | – | – | 49.2 | 30.1 | 28.9 | 41.2 |
| Flamingo-3B | 8 | 90.6 | – | – | 55.4 | 32.4 | 38.4 | 44.6 |
| **MM1-3B** | 0 | 73.5 | 55.6 | 63.3 | 46.2 | 29.4 | 15.6 | 26.1 |
| **MM1-3B** | 8 | **114.6** | **104.7** | **88.8** | **63.6** | **44.6** | **46.4** | **48.4** |
| Flamingo-9B | 8 | 99.0 | – | – | 58.0 | 33.6 | 39.4 | 50.0 |
| Emu2-14B | 8 | – | – | – | 59.0 | – | 43.9 | – |
| **MM1-7B** | 8 | **116.3** | **106.6** | **88.2** | **63.6** | **46.3** | **45.3** | **51.4** |
| Flamingo-80B | 8 | 108.8 | – | – | 65.6 | 37.3 | 44.8 | 57.5 |
| IDEFICS-80B | 8 | 114.3 | 105.7 | 77.6 | 64.8 | 35.7 | 46.1 | 55.1 |
| Emu2-37B | 8 | – | – | – | 67.8 | 49.3 | 54.7 | 54.1 |
| **MM1-30B** | 8 | **123.1** | **111.6** | **92.9** | **70.9** | **49.4** | **49.9** | **58.3** |

MM1 leads in few-shot performance across all comparisons. Notably, MM1-30B outperforms Flamingo-80B (which has 2.7 times more parameters) and IDEFICS-80B.

### Main SFT Results (12 Benchmarks)

| Model | VQAv2 | TextVQA | SQA-I | MMMU (v/t) | MathV | MME-P | MME-C | MMB | SEED | POPE | LLaVA-W | MM-Vet |
|------|-------|---------|-------|------------|-------|-------|-------|-----|------|------|---------|--------|
| **MM1-3B-Chat** | 82.0 | 71.9 | 69.4 | 33.9/33.7 | 32.0 | 1482 | 279 | 67.8 | 63.0/68.8 | 87.4 | 72.1 | 43.7 |
| **MM1-3B-MoE** | 82.5 | 72.9 | 76.1 | 38.6/35.7 | 32.6 | 1469 | 303 | 70.8 | 63.9/69.4 | 87.6 | 76.8 | 42.2 |
| LLaVA-1.5-7B | 78.5 | 58.2 | 66.8 | –/– | – | 1511 | 316 | 64.3 | 58.6/66.1 | 85.9 | 63.4 | 31.1 |
| LLaVA-NeXT-7B | 81.8 | 64.9 | 70.1 | 35.8/– | 34.6 | 1519 | 332 | 67.4 | –/70.2 | 86.5 | 81.6 | 43.9 |
| **MM1-7B-Chat** | 82.8 | 72.8 | 72.6 | 37.0/35.6 | 35.9 | 1529 | 329 | 72.3 | 64.0/69.9 | 86.6 | 81.5 | 42.1 |
| **MM1-7B-MoE** | **83.4** | **73.8** | 74.4 | **40.9**/37.9 | **40.9** | **1597** | **395** | 72.7 | **65.5/70.9** | **87.8** | **84.7** | 45.2 |
| **MM1-30B-Chat** | 83.7 | 73.5 | 81.0 | 44.7/40.3 | 39.4 | 1638 | 431 | 75.1 | 65.9/72.1 | 87.6 | 89.3 | 48.7 |

MoE models outperform their corresponding dense models across nearly all benchmarks, demonstrating the substantial potential of MoE in MLLMs.

### Key Findings

1. **Resolution > Model Size > Training Data**: The selection hierarchy for the encoder is clear.
2. **Connector types are largely inconsequential**: Average Pooling, Attention Pooling, and C-Abstractor exhibit nearly identical behavior.
3. **Token count is significant**: Moving from 64 to 144 tokens yields a substantial performance improvement.
4. **Interleaved data acts as the primary driver of few-shot capabilities**: Its structural format is naturally congruent with few-shot inputs.
5. **Pre-training duration directly affects SFT performance**: More pre-training steps lead to superior downstream performance (Figure 7c).
6. **SFT experiences diminishing returns at extremely high resolutions**: $1344\text{px}$ is optimal, while $1792\text{px}$ yields a slight degradation.
7. **Pre-training principles transfer to SFT**: Caption-only pre-training enhances zero-shot capabilities in SFT, while connector-related architectural discrepancies disappear post-SFT as well.
8. **Few-shot capability is preserved post-SFT**: MM1-30B-Chat on MathVista achieves 0-shot 39.4 $\rightarrow$ 4-shot 41.9 $\rightarrow$ 8-shot (mixed resolution) 44.4.

## Highlights & Insights

- **The value of systematic ablations**: The primary contribution of MM1 is not the model itself but rather its reproducible design principles. Such "recipe papers" offer more enduring value to the community than individual state-of-the-art weights.
- **The paradigm-shifting discovery that connectors are inconsequential**: This directly challenges prior research focusing intensively on VL connectors (e.g., Honeybee, BLIP-2), demonstrating that simple projection methods suffice when training scales up.
- **Meticulous data recipe balance**: The 45:45:10 ratio is not an arbitrary choice but an optimal balance validated by systematic ablations. This suggests that the importance of data engineering in multimodal pre-training has been highly underestimated.
- **Efficient scaling of MoE**: 3B-MoE (64B total parameters) achieves comparable or even superior performance to 7B dense models while activating only a fraction of parameters, outlining a promising direction for efficient MLLM deployment.
- **Mixed-resolution few-shot strategy**: Since sub-image decomposition results in prohibitive token counts in few-shot settings, the proposed mixed-resolution strategy (only keeping the final $N$ samples at high resolution) is a highly practical engineering innovation.

## Limitations & Future Work

1. **Bounded ablation capacity**: Component-level ablations use modest 1.2B/2.9B LLMs, and data ablations are restricted to 200k steps; whether these principles still hold strictly at massive scales remains verified.
2. **Imperfect comparison fairness for encoders**: AIM was trained on less than half of CLIP's data volume, suggesting the conclusions should be cautiously interpreted.
3. **Dependency on closed-source datasets**: Web Image-Text-1B and proprietary interleaved data are not reproducible for the broader community.
4. **Absence of video and audio modalities**: The model is restricted to image-text modalities, omitting more general-purpose multimodal scenarios.
5. **Conventional SFT data recipe**: The model mostly follows the SFT recipe established by LLaVA-1.5, leaving systematic SFT data ablations largely unexplored.
6. **Strict token limit per image**: Utilizing only 144 tokens per image (compared to 2880 tokens in LLaVA-NeXT) may constrain fine-grained perception and comprehension.
7. **Lack of RLHF/DPO explorations**: The post-training relies solely on the SFT stage, without incorporating preference alignment techniques.

## Related Work & Insights

- **Flamingo**: The pioneering large-scale interleaved pre-trained MLLM which serves as the direct baseline and inspiration for MM1.
- **IDEFICS/OpenFlamingo**: Open-source replication efforts of Flamingo, but lacking systematic architectural or component ablations.
- **LLaVA-1.5/NeXT**: The primary sources for SFT recipes; the SFT data mixture in MM1 directly adapts elements from these models.
- **VILA**: Also investigates pre-training components, but provides fewer details on optimization and pre-training evaluations.
- **Emu2**: Discloses pre-training details but lacks controlled ablations. MM1 comprehensively outperforms Emu2 in few-shot scenarios.
- **Honeybee**: Introduces C-Abstractor, yet MM1 demonstrates that its advantages fade at larger training scales.

**Takeaway for future investigation**: The ablation methodology of MM1 provides generalized references for scaling up, suggesting that optimizing component choices at smaller scales before extrapolating yields successful results. The learning rate extrapolation formula is a highly practical, directly reusable asset.

## Rating

| Dimension | Score (1-5) | Explanation |
|------|-----------|------|
| Novelty | 3.5 | No architecture novelty, but the systematic ablation methodology and design guidelines are highly valuable |
| Experimental Thoroughness | 5 | Extremely thorough, covering three-axis ablations, multi-scale scale-up validation, and post-SFT transfer analysis |
| Engineering Value | 5 | The data recipe, learning rate extrapolation formula, and mixed-resolution strategies are highly actionable |
| Writing Quality | 4.5 | Well-structured, clear summarization of rules, and highly informative appendix |
| Overall Recommendation | ⭐⭐⭐⭐⭐ | A must-read recipe paper in the MLLM domain; the referential value of the ablation findings far exceeds the weights of the model itself |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Omniview-Tuning: Boosting Viewpoint Invariance of Vision-Language Pre-training Models](omniview-tuning_boosting_viewpoint_invariance_of_vision-language_pre-training_mo.md)
- [\[CVPR 2025\] Multi-Layer Visual Feature Fusion in Multimodal LLMs: Methods, Analysis, and Best Practices](../../CVPR2025/multimodal_vlm/multi-layer_visual_feature_fusion_in_multimodal_llms_methods_analysis_and_best_p.md)
- [\[CVPR 2025\] Multimodal Autoregressive Pre-training of Large Vision Encoders](../../CVPR2025/multimodal_vlm/multimodal_autoregressive_pre-training_of_large_vision_encoders.md)
- [\[ICCV 2025\] SCAN: Bootstrapping Contrastive Pre-training for Data Efficiency](../../ICCV2025/multimodal_vlm/scan_bootstrapping_contrastive_pre-training_for_data_efficiency.md)
- [\[CVPR 2026\] PowerCLIP: Powerset Alignment for Contrastive Pre-Training](../../CVPR2026/multimodal_vlm/powerclip_powerset_alignment_for_contrastive_pre-training.md)

</div>

<!-- RELATED:END -->
