---
title: >-
  [Paper Note] HoloLLM: Multisensory Foundation Model for Language-Grounded Human Sensing and Reasoning
description: >-
  [NEURIPS2025][LLM Safety][Multimodal LLM] This paper proposes HoloLLM, the first framework to integrate rare sensing modalities — including LiDAR, infrared, mmWave radar…
tags:
  - "NEURIPS2025"
  - "LLM Safety"
  - "Multimodal LLM"
  - "Human Sensing"
  - "LiDAR"
  - "mmWave"
  - "WiFi"
  - "Modality Alignment"
date: 2026-05-08
content_hash: 3f87079a299d5e8d
---

# HoloLLM: Multisensory Foundation Model for Language-Grounded Human Sensing and Reasoning

**Conference**: NEURIPS2025  
**arXiv**: [2505.17645](https://arxiv.org/abs/2505.17645)  
**Code**: [NTUMARS/HoloLLM](https://github.com/NTUMARS/HoloLLM)  
**Area**: Autonomous Driving  
**Keywords**: Multimodal LLM, Human Sensing, LiDAR, mmWave, WiFi, Modality Alignment  

## TL;DR
This paper proposes HoloLLM, the first framework to integrate rare sensing modalities — including LiDAR, infrared, mmWave radar, and WiFi — into a multimodal large language model (MLLM). Through a Universal Modality-Injection Projector (UMIP), HoloLLM achieves efficient alignment between sensing modalities and text under data-scarce conditions, improving human action QA and captioning by approximately 30% over existing MLLMs.

## Background & Motivation
Embodied agents in smart home environments (domestic robots, intelligent appliances, etc.) must understand human behavior from diverse sensing inputs and interact via natural language. While Vision-Language Models (VLMs) excel at visual-language perception, relying solely on visual modalities fails under occlusion, low-light, and privacy-sensitive scenarios. For instance, when a person falls behind an obstacle, cameras cannot detect the event, whereas mmWave radar and WiFi signals remain unaffected. Integrating complementary sensing modalities such as LiDAR, infrared, mmWave, and WiFi into MLLMs is therefore a critical direction for robust human sensing.

## Core Problem
Integrating rare sensing modalities into MLLMs faces two principal challenges:

1. **Data scarcity**: RGB/depth images benefit from millions of web-scale paired samples for projector pretraining, whereas mmWave, WiFi, and similar sensing data consist of only thousands of laboratory-collected samples — insufficient for large-scale modality–text alignment pretraining.
2. **Signal heterogeneity**: Substantial differences in physical sensor design (wavelength, frequency) result in signal representations vastly different from common modalities, making it difficult for general-purpose Transformer encoders to learn discriminative features.

## Method

### Overall Architecture
HoloLLM consists of four components: (1) a CLIP visual encoder serving as a universal encoder to generate pre-aligned initial embeddings; (2) a Tailored Encoder designed for each modality to extract fine-grained features; (3) the UMIP, which fuses both feature types into text-aligned multimodal tokens; and (4) LLaMA2-7B as the LLM backbone that receives multimodal tokens and text instructions for reasoning.

### Universal Modality-Injection Projector (UMIP)
UMIP is the core module of this work, adopting a coarse-to-fine progressive feature enhancement strategy:

- **Coarse query generation**: CLIP ViT-L (pretrained on LAION image–text contrastive pairs) serves as a unified encoder, generating initial embeddings $\mathbf{Y}_{CLIP}^m$ for any modality. Owing to CLIP's inherent text-alignment capacity and cross-modal transferability, these embeddings function as "pre-aligned" representations. Adaptive average pooling downsamples them into a fixed number of queries $\mathbf{Q}^m$.
- **Fine-grained key-value generation**: Each modality is equipped with a dedicated encoder (ResNet18 for visual/depth/infrared, PointNet for LiDAR/mmWave, 1D Temporal ResNet18 for RFID, MetaFi for WiFi), extracting heterogeneous features $\mathbf{Y}_T^m$ and converting them into Keys/Values.
- **Iterative coarse-to-fine cross-attention**: UMIP contains $L=8$ blocks, each sequentially performing self-attention, cross-attention (queries adaptively extract text-aligned features from fine-grained Keys/Values), and a feed-forward network (projecting enhanced queries into the LLM text space). The output serves as the enhanced query for the next block.
- The final output $\mathbf{Z}^m = \text{MLP}(\mathbf{Q}_L^m)$ maps the dimensionality from CLIP's 1024 to the LLM's 4096.

### Two-Stage Training Strategy
- **Stage 1**: CLIP is frozen; Tailored Encoders for each modality are pretrained with HAR classification loss.
- **Stage 2**: Tailored Encoders are frozen; the Tokenizer and UMIP are fine-tuned by jointly optimizing classification loss and next-token prediction loss.

### Comparison with Existing Projector Designs
- **Modality-Specific Projector** (PointLLM, ImageBind-LLM, etc.): One encoder plus one projector per modality, requiring large amounts of modality–text paired data for pretraining.
- **Universal Projector** (OneLLM): A unified encoder with a unified projector, eliminating modality-specific pretraining, but lacking the capacity to capture heterogeneous features.
- **UMIP (Ours)**: A unified encoder generates pre-aligned initial embeddings; dedicated encoders extract fine-grained features; progressive cross-attention performs fusion — achieving both data efficiency and feature discriminability.

### Data Construction Pipeline
For the MM-Fi and XRF55 multimodal human sensing datasets, a human–VLM collaborative annotation pipeline is designed:
- **Action QA**: Human experts annotate 5 question templates, which GPT-4o paraphrases and expands to 15; multiple-choice QA pairs are generated by random sampling.
- **Action Caption**: A small number of samples are uniformly sampled for human annotation; the remainder are automatically captioned via LLaVA-Video in-context learning.
- All text annotations are shared across modalities within the same data sample, reducing annotation cost.

## Key Experimental Results

### Datasets and Evaluation Settings
- Datasets: MM-Fi (Video, Depth, LiDAR, mmWave, WiFi) and XRF55 (Video, Depth, Infrared, RFID, WiFi).
- Three evaluation settings: Random Split, Cross-Subject, Cross-Environment.
- Metrics: Accuracy for Action QA/Recognition; METEOR for Action Caption.

### Main Results (MM-Fi, Cross-Environment — Hardest Setting)

| Method | Action QA Avg | Action Caption Avg |
|------|:---:|:---:|
| Tokenpacker | 4.6% | 3.8% |
| Honeybee | 1.7% | 10.4% |
| OneLLM | 5.0% | 9.3% |
| ImageBind | 16.7% | 17.3% |
| **HoloLLM** | **56.4%** | **22.6%** |

HoloLLM surpasses the strongest baseline (ImageBind) by approximately 40 percentage points on the QA task. Under the Random Split setting, it achieves an average QA accuracy of 86.5%, outperforming ImageBind by 40.3%.

### Ablation Study (Cross-Environment)
- Baseline (CLIP + Q-Former): MM-Fi Action QA only 6.2%.
- +Tailored Encoder: improves to 46.6% (+40.4%), demonstrating the critical role of dedicated encoders for heterogeneous sensing features.
- +UMIP: further improves to 56.4% (+9.8%), with notable gains on QA tasks requiring deeper language understanding.
- Similar trends are observed on XRF55: Baseline QA 3.5% → +Tailored Encoder 11.8% → +UMIP 12.4%.
- t-SNE visualizations show that HoloLLM's multimodal tokens exhibit significantly better action-class clustering compared to the Baseline and OneLLM.

## Highlights & Insights
1. **Pioneering contribution**: The first work to integrate rare sensing modalities (mmWave, WiFi, RFID) into an MLLM for language-grounded human sensing and reasoning.
2. **Elegant UMIP design**: Pre-aligned CLIP embeddings serve as "coarse queries" while dedicated encoder features serve as "fine references"; iterative cross-attention progressively fuses them, eliminating the need for large-scale modality–text pretraining data.
3. **Comprehensive benchmark construction**: Establishes the first multi-sensing-modality language-grounded human sensing benchmark, covering Action Recognition, QA, and Captioning across three cross-domain evaluation settings.
4. **t-SNE visualizations** clearly demonstrate that UMIP-generated multimodal tokens exhibit strong intra-class clustering and are well-aligned with text tokens.

## Limitations & Future Work
1. Task coverage is limited to action recognition/QA/captioning; more complex embodied tasks such as task planning and action generation are not addressed.
2. WiFi and RFID modalities exhibit weak generalization in cross-subject and cross-environment settings, with performance approaching chance level.
3. Dataset scale remains small (both MM-Fi and XRF55 are laboratory-scale); scalability to large-scale real-world scenarios has yet to be validated.
4. The LLM backbone is fixed as LLaMA2-7B; larger or more recent base models are not explored.

## Related Work & Insights
- **OneLLM**: Similarly employs a unified encoder with a unified projector but lacks dedicated encoders for heterogeneous sensing features, achieving only 3–5% QA accuracy on sensing modalities.
- **ImageBind**: Aligns multimodal embeddings via contrastive learning and equips dedicated encoders for depth/infrared; performs adequately on some modalities but falls far short of HoloLLM.
- **Tokenpacker / Honeybee**: Use only a shared projector, completely failing to capture modality-specific sensing features, with QA performance near random.
- UMIP occupies a middle ground between modality-specific and universal projectors, balancing generality and modality specificity.

UMIP's coarse-to-fine progressive alignment strategy generalizes to other data-scarce modality–language alignment scenarios (e.g., tactile sensing, EEG). The human–VLM collaborative annotation pipeline constitutes an effective low-cost paradigm for generating rare-modality text pairs. For autonomous driving and embodied intelligence, combining the perceptual capabilities of radar and LiDAR sensors with the reasoning capacity of LLMs is a promising direction worthy of further exploration.

## Rating
- **Novelty**: 8/10 — First to integrate multiple rare sensing modalities into an MLLM; UMIP design is innovative.
- **Experimental Thoroughness**: 7/10 — Two datasets, three evaluation settings, complete ablations and visualizations, but dataset scale is limited.
- **Writing Quality**: 8/10 — Problem formulation is clear, methodology is well-described, and figures are well-crafted.
- **Value**: 7/10 — Opens a new direction of sensing modalities + MLLM, but practical applicability is constrained by data scale and task scope.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Distributive Fairness in Large Language Models: Evaluating Alignment with Human Values](distributive_fairness_in_large_language_models_evaluating_alignment_with_human_v.md)
- [\[NeurIPS 2025\] VMDT: Decoding the Trustworthiness of Video Foundation Models](vmdt_decoding_the_trustworthiness_of_video_foundation_models.md)
- [\[NeurIPS 2025\] Reverse Engineering Human Preferences with Reinforcement Learning](reverse_engineering_human_preferences_with_reinforcement_learning.md)
- [\[NeurIPS 2025\] One Token Embedding Is Enough to Deadlock Your Large Reasoning Model](one_token_embedding_is_enough_to_deadlock_your_large_reasoning_model.md)
- [\[NeurIPS 2025\] Poly-Guard: Massive Multi-Domain Safety Policy-Grounded Guardrail Dataset](poly-guard_massive_multi-domain_safety_policy-grounded_guardrail_dataset.md)

</div>

<!-- RELATED:END -->
