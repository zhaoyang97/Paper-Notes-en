---
title: >-
  [Paper Note] FineMatch: Aspect-based Fine-grained Image and Text Mismatch Detection and Correction
description: >-
  [ECCV 2024][Image Generation][Vision-Language Models] This work proposes FineMatch, a benchmark that defines the task of aspect-based, fine-grained image-text mismatch detection and correction. It contains 49,906 high-quality, human-annotated image-text pairs and demonstrates the limitations of existing VLMs in fine-grained compositional understanding.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Vision-Language Models"
  - "Image-Text Matching"
  - "Compositional Understanding"
  - "Fine-grained Detection"
  - "Benchmark"
date: 2026-05-08
content_hash: 5950ba958ac5f7ba
---

# FineMatch: Aspect-based Fine-grained Image and Text Mismatch Detection and Correction

**Conference**: ECCV 2024  
**arXiv**: [2404.14715](https://arxiv.org/abs/2404.14715)  
**Code**: [Project Page](https://hanghuacs.github.io/finematch/)  
**Area**: Image Generation  
**Keywords**: Vision-Language Models, Image-Text Matching, Compositional Understanding, Fine-grained Detection, Benchmark

## TL;DR

This work proposes FineMatch, a benchmark that defines the task of aspect-based, fine-grained image-text mismatch detection and correction. It contains 49,906 high-quality, human-annotated image-text pairs and demonstrates the limitations of existing VLMs in fine-grained compositional understanding.

## Background & Motivation

Although existing pretrained vision-language models (VLMs) such as GPT-4V and LLaVA exhibit outstanding performance in complex reasoning, they remain limited in **fine-grained compositional understanding**. Specifically, these limitations include:

**Coarse Granularity**: Existing compositional evaluation benchmarks (such as ARO, Winoground, and SUGARCREPE) only require **sentence-level** matching/mismatching judgments, which are overly simplified.

**Lack of Localization Evaluation**: There is no evaluation of whether models can locate which specific phrases mismatch the image.

**Lack of Correction Evaluation**: Models are not required to provide the correct description.

**Limitations of VQA Methods**: Existing fine-grained detection methods rely on a pipeline of generating questions first and then answering via VQA, which lacks flexibility and suffers from error accumulation.

FineMatch proposes a more challenging task: given an image-text pair, the model is required to:
- Identify which phrases in the description mismatch the image
- Determine the mismatch categories (Entity/Relation/Attribute/Number)
- Provide corresponding correction suggestions

## Method

### Overall Architecture

FineMatch consists of two subtasks:

1. **Mismatch Detection (MD)**: Predict the mismatched phrases and their categories $\{(c_j, p_j)\}_{j=1}^M$
2. **Mismatch Detection & Correction (MD&C)**: Predict the mismatched phrases, categories, and corrections $\{(c_j, p_j, o_j)\}_{j=1}^M$

Each image-text pair can contain 0 to 3 mismatch aspects. Mismatches are categorized into four types:
- **Entity**: Objects in the description do not exist or are incorrect in the image.
- **Relation**: Spatial or action relationships between objects are incorrectly described.
- **Attribute**: Attributes of objects (such as color or size) are described incorrectly.
- **Number**: The quantity of objects is described incorrectly.

### Key Designs

#### Data Construction (Three Sources)

**1. GPT-synthesized Textual Data (via aspect graph parsing and node replacement)**

This is the core data generation method:

1. Use GPT-4 with In-Context Learning (ICL) to parse the description into an **Aspect Graph**, where nodes represent atomic aspect entities and edges denote relationships.
2. Randomly replace nodes with counterfactual descriptions while keeping Part-of-Speech (POS) tags unchanged.
3. Convert the aspect graph back to a natural language description.
4. Filter out samples with excessively large semantic gaps using CLIP scores.

**De-biasing**: Previous work (such as SUGARCREPE) has pointed out that rule-generated negative samples may introduce artifact biases. FineMatch employs a three-step de-biasing process:
- Filter out unreasonable content using Vera Scores.
- Filter out grammatical errors using Grammar Scores.
- Filter out samples with excessively large semantic gaps using CLIP Scores.
- Finally revise and validate via human annotation.

**2. Retrieved Image-Text Data**

Natural mismatched image-text pairs are acquired by leveraging semantic differences in text-to-image retrieval systems:
- Select syntactically complex queries from datasets such as NoCaps and VizWiz (filtered by dependency tree depth).
- Retrieve candidate images from LAION-400M, COYO-700M, and Smithsonian Open Access using a ViT-G/14 CLIP model.
- Filter 10K high-quality image-text pairs for human annotation.

**3. Stable Diffusion Generated Image Data**

SD 2.1 is driven to generate images using text prompts from T2I-CompBench, yielding 2.5K image-text pairs.

#### Evaluation Metric: ITM-IoU

A brand-new evaluation metric, ITM-IoU, is proposed with the following core designs:

**Detection Score** (combining character-level and semantic-level matching):
$$Score_{D_j} = \frac{BERTScore(p_j, p_j') + chrF(p_j, p_j')}{2}$$

**Correction Score**:
$$Score_{C_j} = BERTScore(o_j, o_j')$$

**Aspect Total Score** (weighted fusion of the three elements):
$$Score_{Aspect_j} = W_{Ca} \cdot EM(c_j, c_j') + W_{De} \cdot Score_{D_j} + W_{Co} \cdot Score_{C_j}$$

Where the weights are $W_{Ca}=0.2, W_{De}=0.4, W_{Co}=0.4$.

**Final ITM-IoU**:
$$\text{ITM-IoU} = \frac{\sum_{j=0}^M Score_{Aspect_j}}{M} \times \frac{|P_i \cap G_i|}{|P_i \cup G_i|}$$

Experiments demonstrate that ITM-IoU is highly correlated with human evaluation.

#### Application: AutoAlign Self-Correction System

A T2I hallucination detection and correction system is built based on LLaVA-1.6 fine-tuned on FineMatch:
1. The T2I generation module generates an image.
2. The FineMatch fine-tuned VLM detects image-text mismatches.
3. GPT-4 generates image editing prompts.
4. MagicBrush performs image editing.
5. Iterate until image-text alignment is achieved.

### Loss & Training

Training adopts the standard visual instruction tuning paradigm:
$$\mathcal{L} = -\sum_\mathcal{D} \sum_{t=1}^M \log p(P_t | [C_i : I_i], P_{\leq t-1})$$

## Key Experimental Results

### Main Results

**Supervised Learning Results (ITM-IoU)**

| Model | Params | MD↑ | MD&C↑ |
|------|-------|-----|-------|
| OFA-Large | 472M | 19.72 | 21.35 |
| LLaMA-Adapter2 | 7B | 35.84 | 40.76 |
| MiniGPT-4-V2 | 7B | 51.18 | 55.95 |
| InternLM-Xcomposer2-VL | 7B | 58.70 | 61.07 |
| LLaVA-1.5 | 7B | 62.25 | 63.62 |
| LLaVA-1.5 | 13B | 66.02 | 67.13 |
| ShareGPT4V | 13B | 66.06 | 67.21 |
| LLaVA-1.6-Vicuna | 13B | 66.10 | 67.31 |
| **Human Performance** | - | **88.32** | **89.19** |

**In-Context Learning Results**

| Model | Params | MD↑ | MD&C↑ |
|------|-------|-----|-------|
| Otter | 7B | 0.03 | 0.09 |
| OpenFlamingo | 9B | 0.34 | 0.96 |
| Emu2 | 37B | 6.10 | 11.23 |
| Gemini Pro Vision | - | 9.07 | 11.14 |
| GPT-4V | - | 21.92 | 21.58 |

### Ablation Study

**Correlation between Human Evaluation and ITM-IoU**

| Model | ITM-IoU (MD) | Human Rating (1-5) |
|------|-------------|--------------|
| mPLUG-Owl2 | 46.70 | 3.56 |
| MiniGPT-4 | 51.18 | 3.77 |
| LLaVA-1.5-7B | 62.25 | 4.02 |
| LLaVA-1.5-13B | 66.02 | 4.41 |
| GPT-4V (ICL) | 21.92 | 3.35 |

The ranking of ITM-IoU is highly consistent with human evaluation, validating the soundness of the proposed metric.

### Key Findings

1. **ICL is far inferior to supervised learning**: Even GPT-4V's ICL result (21.92) is significantly lower than that of the moderately-sized fine-tuned model LLaVA-1.5-7B (62.25).
2. **Model scale helps**: Upgrading LLaVA-1.5 from 7B to 13B improves ITM-IoU from 62.25 to 66.02.
3. **Large gap with humans**: The best-performing model (67.31) still lags behind human performance (89.19) by more than 22 points.
4. **Both training data and reasoning ability are important**: Improvements in ShareGPT4V and LLaVA-1.6 validate the value of superior pre-training data and stronger reasoning capabilities.
5. **De-biasing of GPT-synthesized data is effective**: The workflow of filtering with Vera + Grammar + CLIP scores, combined with human annotation, successfully eliminates artifact biases.

## Highlights & Insights

1. **Novel task definition**: Upgraded from simple binary matching classification to a structured prediction task involving localization + classification + correction.
2. **Multi-source data construction**: Three data sources (GPT-synthesized, retrieved, and SD-generated) cover various types of image-text mismatch scenarios.
3. **Elegant design of ITM-IoU metric**: It combines character-level and semantic-level evaluations, and introduces the concept of IoU to handle multi-aspect matching.
4. **Revealing VLM capability boundaries**: Even the strongest model, GPT-4V, fails to effectively perform fine-grained image-text matching analysis under a few-shot setting.
5. **Practical application value**: The AutoAlign system demonstrates the utility of FineMatch in T2I hallucination detection and correction.

## Limitations & Future Work

1. **Single correction**: Only one correction is provided for each mismatch, whereas multiple valid corrections may exist in practice.
2. **Large gap with humans**: The best-performing model still lags behind human performance by ~22 points, indicating that the task itself remains highly challenging.
3. **Limited data scale**: Comprising around 50K samples, it is still relatively small compared to large-scale pre-training datasets.
4. **Aspect graph not utilized as input**: Introducing the aspect graph structure into text prompts could potentially boost performance.
5. **Openness of correction evaluation**: BERTScore might not fully capture the semantic correctness of corrections.

## Related Work & Insights

- **ARO / Winoground / SUGARCREPE**: Prior compositionality evaluation benchmarks, which only perform sentence-level judgments.
- **T2I-CompBench**: A compositionality benchmark for text-to-image generation, whose prompts are used by FineMatch to generate data.
- **ControlNet**: Utilized for image editing within the AutoAlign system in combination with FineMatch.
- **Insight**: Fine-grained image-text understanding is an important and unresolved issue. The framework of FineMatch can be extended to video-text matching, multilingual scenarios, etc.

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 4 |
| Theoretical Depth | 3 |
| Experimental Thoroughness | 4.5 |
| Value | 4 |
| Writing Quality | 4 |
| Overall Rating | 3.9 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Fine-Grained Erasure in Text-to-Image Diffusion-based Foundation Models](../../CVPR2025/image_generation/fine-grained_erasure_in_text-to-image_diffusion-based_foundation_models.md)
- [\[ECCV 2024\] Diffusion-based Image-to-Image Translation by Noise Correction via Prompt Interpolation](diffusion-based_image-to-image_translation_by_noise_correction_via_prompt_interp.md)
- [\[ECCV 2024\] Memory-Efficient Fine-Tuning for Quantized Diffusion Model](memory-efficient_fine-tuning_for_quantized_diffusion_model.md)
- [\[CVPR 2025\] FADE: Fine-Grained Erasure in Text-to-Image Diffusion-based Foundation Models](../../CVPR2025/image_generation/fade_fine_grained_erasure_diffusion.md)
- [\[ECCV 2024\] 2S-ODIS: Two-Stage Omni-Directional Image Synthesis by Geometric Distortion Correction](2s-odis_two-stage_omni-directional_image_synthesis_by_geometric_distortion_corre.md)

</div>

<!-- RELATED:END -->
