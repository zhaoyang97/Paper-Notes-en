---
title: >-
  [Paper Note] AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models
description: >-
  [ICCV 2025][Multimodal VLM][AI-generated image detection] This paper proposes AIGI-Holmes, which adapts MLLMs into a "Holmes"-style detector capable of both accurately identifying AI-generated images and providing human-verifiable explanations. This is achieved by constructing the Holmes-Set dataset with explanatory annotations and a carefully designed three-stage training pipeline (visual expert pre-training → SFT → DPO). At inference time, a collaborative decoding strategy…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "AI-generated image detection"
  - "multimodal large language models"
  - "explainable detection"
  - "direct preference optimization"
  - "collaborative decoding"
date: 2026-05-08
content_hash: 836ffa0c9b62550b
---

# AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models

**Conference**: ICCV 2025
**Code**: [https://github.com/wyczzy/AIGI-Holmes](https://github.com/wyczzy/AIGI-Holmes)  
**Area**: Multimodal VLM
**Keywords**: AI-generated image detection, multimodal large language models, explainable detection, direct preference optimization, collaborative decoding

## TL;DR

This paper proposes AIGI-Holmes, which adapts MLLMs into a "Holmes"-style detector capable of both accurately identifying AI-generated images and providing human-verifiable explanations. This is achieved by constructing the Holmes-Set dataset with explanatory annotations and a carefully designed three-stage training pipeline (visual expert pre-training → SFT → DPO). At inference time, a collaborative decoding strategy further enhances generalization.

## Background & Motivation

### Problem Definition
The rapid advancement of AI-generated content (AIGC) technology has enabled highly realistic AI-generated images (AIGI) to be misused for spreading misinformation, threatening public information security. Existing detection methods face two core issues:

**Lack of Explainability**: Current detection models are black boxes, and their outputs are difficult for humans to verify. Without human-verifiable explanations, detection results remain untrustworthy.

**Lack of Generalizability**: AIGC technology evolves rapidly (e.g., FLUX, SD3.5, VAR), and existing methods struggle to generalize to the latest generative techniques.

### Why MLLMs?
MLLMs possess commonsense understanding and natural language generation capabilities, enabling semantic-level analysis of visual content—making them ideal candidates for addressing explainability and generalizability. However, directly applying MLLMs faces two challenges:

- **Scarcity of Training Data**: Existing AIGI detection datasets (e.g., CNNDetection, GenImage, DRCT) contain only visual modality data and lack instruction-tuning datasets suitable for MLLM SFT. FakeBench and LOKI offer preliminary attempts but rely on GPT-4o annotations and are too small in scale.
- **Suboptimal Supervised Fine-Tuning**: Training MLLMs solely on SFT datasets yields limited results, as MLLMs are insufficient in image classification and low-level perception tasks, and SFT models may mechanically replicate explanation templates rather than genuinely understanding the root causes of artifacts or semantic errors.

## Method

### Overall Architecture

AIGI-Holmes consists of two core components: the **Holmes-Set dataset** and the **Holmes Pipeline training framework**.

Architecturally, an NPR (Neighboring Pixel Relationships) visual expert is added on top of LLaVA to capture low-level artifact information. The input processing pipeline is as follows:
- A CLIP visual encoder $F$ extracts high-level semantic features $f_{img}$
- An NPR visual expert $R$ extracts low-level artifact features $f_{npr}$
- Both are injected into the LLM via a projector: $H = \text{LLM}(\text{proj}([f_{img}, f_{npr}]), f_t)$

### Key Design 1: Holmes-Set Dataset

#### Holmes-SFTSet (65K Images)
Data sources consist of two parts:
1. **Existing datasets**: 45K images selected from CNNDetection, GenImage, and DRCT
2. **Expert-filtered images**: 20K images filtered by specialist small models to contain common AI-generated defects (text, human body, face, projective geometry, commonsense, physical laws)

Annotation employs a **Multi-Expert Jury** approach, with cross-annotation and evaluation by four open-source MLLMs (Qwen2VL-72B, InternVL2-76B, InternVL2.5-78B, Pixtral-124B):
- **General Positive Prompt**: Analyzes high-level semantic dimensions (anatomy, physical laws) and low-level dimensions (texture, sharpness)
- **General Negative Prompt**: Generates adversarial annotations that form DPO data pairs $D_1$ with positive annotations
- **Specialist Prompt**: Targeted annotations for specific defects in the 20K expert-filtered images

Quality control adopts an MLLM-as-a-judge paradigm, retaining only annotations with the highest consensus.

#### Holmes-DPOSet (65K + 4K)
To address the "mechanical template copying" problem in SFT models, a human-aligned preference dataset is constructed:
- $D_1$: Natural positive-negative pairs from General Positive/Negative Prompts
- $D_2$: **Manually modified** — 2K human-annotated samples + 2K samples modified using Specialist Prompts. Human experts provide revision suggestions on SFT model outputs (adding correct information, removing erroneous or irrelevant explanations), which are then executed by DeepSeek-V3

### Key Design 2: Holmes Pipeline (Three-Stage Training)

**Stage 1: Visual Expert Pre-training**

The goal is to equip the visual experts with generalization capability in the AIGI detection domain. Binary classification pre-training is performed separately on two visual encoders:
- CLIP-ViT-L/14 is fine-tuned with LoRA ($r=4, \alpha=8$), obtaining classification results via an MLP from the CLS feature $f_{cls}$
- The first two layers of the NPR-based ResNet are fully fine-tuned, with classification also performed via an MLP
- Loss functions: $l_{clip} = l_{bce}(y_{clip}, y), \quad l_{npr} = l_{bce}(y_{npr}, y)$

**Why pre-train?** Neither vanilla CLIP nor ResNet was designed for AIGI detection; they are insufficient in low-level artifact perception and classification when applied directly to downstream tasks. Pre-training endows them with domain-specific feature extraction capabilities.

**Stage 2: Supervised Fine-Tuning (SFT)**

The pre-trained visual experts are integrated into the LLM, and autoregressive text loss training is conducted on Holmes-SFTSet:
- Visual experts are frozen
- The projector and LLM LoRA components (rank=128, α=256) are trained
- Loss: $l_{txt} = l_{ce}(H, H_{txt})$

**Stage 3: Direct Preference Optimization (DPO)**

Human preference alignment is performed on Holmes-DPOSet ($D = D_1 \cup D_2$):

$$L_{DPO}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \left( \log \frac{\pi_\phi(y_w|x)}{\pi_{base}(y_w|x)} - \log \frac{\pi_\phi(y_l|x)}{\pi_{base}(y_l|x)} \right) \right) \right]$$

The key role of DPO is to reshape the MLLM's reasoning pattern, aligning explanations with human judgment standards rather than remaining at the level of suboptimal fine-tuning.

### Key Design 3: Collaborative Decoding

At inference time, the MLLM and pre-trained visual experts make joint decisions by adjusting the logit values for "real" and "fake" tokens:

$$\text{logit}_{new}(y=k) = \alpha \cdot \text{logit}_{raw}(y=k) + \beta \cdot \text{logit}(y_{clip}=k) + \gamma \cdot \text{logit}(y_{npr}=k)$$

where $\alpha=1, \beta=1, \gamma=0.2$.

**Why does this work?** By retaining the MLLM's predictions while incorporating the visual experts' judgments, the approach prevents the MLLM from overfitting to forgery types seen during training, thereby improving generalization to unseen domains.

### Loss & Training

| Stage | Trainable Parameters | Loss Function | Hyperparameters |
|-------|---------------------|---------------|-----------------|
| Visual Expert Pre-training | CLIP LoRA(r=4) + ResNet full params | Binary CE | batch=32, 5 epochs |
| SFT | Projector + LLM LoRA(r=128) | Autoregressive CE | lr=5e-5, batch=16, 3 epochs |
| DPO | Projector + LLM LoRA(r=48) | DPO Loss | lr=5e-7, batch=4, β=0.1, 2 epochs |

## Key Experimental Results

### Main Results

The paper evaluates under three protocols (Protocol-I/II/III), with P3 being the most challenging — training on diffusion model data and testing on entirely new autoregressive generative models and state-of-the-art diffusion models.

**Protocol-III Detection Accuracy (Acc. %)**:

| Method | VAR | FLUX | Janus-Pro-7B | SD3.5-Large | Mean Acc. | Mean A.P. |
|--------|-----|------|--------------|-------------|-----------|-----------|
| CNNSpot | 59.9 | 63.8 | 85.0 | 78.2 | 72.9 | 85.6 |
| NPR | 85.9 | 91.6 | 73.9 | 93.4 | 84.0 | 89.5 |
| UnivFD | 64.3 | 87.8 | 96.4 | 75.7 | 83.6 | 95.9 |
| RINE | 85.0 | 97.8 | 97.2 | 98.9 | 96.2 | 99.5 |
| AIDE | 93.6 | 99.4 | 97.8 | 98.6 | 97.0 | 99.7 |
| **AIGI-Holmes** | **99.6** | **99.4** | **98.0** | **99.9** | **99.2** | **99.9** |

AIGI-Holmes achieves 98%+ accuracy across all generators, with Mean Acc. surpassing AIDE by 2.2% and RINE by 3.0%.

**Explanation Quality Comparison (MLLM vs. AIGI-Holmes)**:

| Model | BLEU-1 | ROUGE-L | CIDEr | ELO Rating |
|-------|--------|---------|-------|------------|
| GPT-4o | 0.433 | 0.308 | 0.005 | 10.271 |
| Pixtral-124B | 0.428 | 0.270 | 0.010 | 10.472 |
| AIGI-Holmes (w/o DPO) | 0.445 | 0.315 | 0.023 | 10.670 |
| **AIGI-Holmes (w/ DPO)** | **0.622** | **0.375** | **0.107** | **11.420** |

### Ablation Study

**Core Component Ablation (Acc. %)**:

| VEP-S | DPO | CD | P1 | P3 |
|-------|-----|----|----|-----|
| ✗ | ✗ | ✗ | 83.3 | 90.1 |
| ✓ | ✗ | ✗ | 84.8 | 92.3 |
| ✓ | ✓ | ✗ | 87.4 | 97.6 |
| ✓ | ✗ | ✓ | 90.8 | 98.9 |
| ✓ | ✓ | ✓ | **93.2** | **99.2** |

- Visual Expert Pre-training yields +2.2% on P3
- DPO contributes +0.4% (but improves ELO Rating by 0.75)
- Collaborative Decoding contributes the most: +1.7%
- The combination of all three improves approximately 10% over the baseline

**Robustness Evaluation (P3 Mean Acc. %)**:

| Method | JPEG QF=75 | Gaussian σ=2 | Resize ×0.5 |
|--------|-----------|-------------|-------------|
| AIDE | 92.8 | 90.7 | 89.2 |
| RINE | 92.4 | 92.8 | 92.3 |
| **AIGI-Holmes** | **99.0** | **97.9** | **95.9** |

### Key Findings

1. **Collaborative Decoding is key to generalization**: By incorporating domain knowledge from visual experts, collaborative decoding effectively prevents the MLLM from overfitting to forgery types seen during training.
2. **DPO is critical for explanation quality**: Although DPO yields limited improvement in detection accuracy (+0.4%), it substantially improves explanation quality (ELO +0.75).
3. **MLLMs focus on high-level semantic features**: Even under perturbations such as JPEG compression and Gaussian blur, explanation quality metrics do not significantly degrade, indicating that the MLLM relies on high-level semantics rather than low-level artifacts.
4. **Multi-expert cross-validation improves data quality**: The Multi-Expert Jury approach is more reliable than single-model annotation.

## Highlights & Insights

1. **Data-driven methodology**: Holmes-Set is the first AIGI detection dataset containing explanatory annotations and human preference data, filling a critical data gap.
2. **Elegant three-stage training design**: The progressive Visual Expert → SFT → DPO design addresses a specific problem at each stage (feature extraction → explanation generation → human alignment).
3. **Inference-stage innovation**: Collaborative Decoding introduces no additional training cost, incorporating visual expert judgments only at inference time — a low-cost yet effective generalization enhancement strategy.
4. **Multi-Expert Jury annotation method**: Using multiple MLLMs for cross-annotation replaces costly GPT-4o or human annotation, reducing cost while maintaining quality.

## Limitations & Future Work

1. **Inference overhead**: Collaborative decoding requires running both the MLLM and visual experts simultaneously, increasing inference time.
2. **Backbone dependency**: The method is built on LLaVA-1.6-mistral-7B; adopting a stronger MLLM backbone may yield further improvements.
3. **DPO data scale**: The manually modified DPO data contains only 4K samples; scaling up may further improve explanation quality.
4. **Real-time detection scenarios**: As an MLLM-based solution, the approach is difficult to deploy for real-time detection requirements.
5. **Video generation detection**: The current work focuses on images; detection of video AIGC (e.g., Sora) remains an unexplored direction.

## Related Work & Insights

- **Inspiration from NPR visual expert**: Low-level neighboring pixel relationships are valuable for AIGI detection but must be combined with high-level semantic features to achieve maximum effectiveness.
- **DPO in specialized domains**: This work demonstrates that DPO is effective not only for general-purpose dialogue but also for improving output quality and human preference alignment in vertical detection tasks.
- **A new paradigm for explainable AI**: Shifting from "post-hoc explanation" to "generative explanation," allowing models to naturally output their reasoning process alongside detection decisions.

## Rating

- Novelty: ⭐⭐⭐⭐ (The three-stage training framework is distinctive in design, though individual components are not entirely novel)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Three protocols, robustness testing, explanation quality evaluation, and comprehensive ablation)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, though some details require consulting the appendix)
- Value: ⭐⭐⭐⭐⭐ (Explainable and generalizable AIGI detection is an important and practically significant direction)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] CompCap: Improving Multimodal Large Language Models with Composite Captions](compcap_improving_multimodal_large_language_models_with_composite_captions.md)
- [\[ICCV 2025\] Bidirectional Likelihood Estimation with Multi-Modal Large Language Models for Text-Video Retrieval](bidirectional_likelihood_estimation_with_multi-modal_large_language_models_for_t.md)
- [\[ICCV 2025\] DocThinker: Explainable Multimodal Large Language Models with Rule-based Reinforcement Learning for Document Understanding](docthinker_explainable_multimodal_large_language_models_with.md)
- [\[ICCV 2025\] Are They the Same? Exploring Visual Correspondence Shortcomings of Multimodal LLMs](are_they_the_same_exploring_visual_correspondence_shortcomings_of_multimodal_llm.md)
- [\[ICCV 2025\] On Large Multimodal Models as Open-World Image Classifiers](on_large_multimodal_models_as_open-world_image_classifiers.md)

</div>

<!-- RELATED:END -->
