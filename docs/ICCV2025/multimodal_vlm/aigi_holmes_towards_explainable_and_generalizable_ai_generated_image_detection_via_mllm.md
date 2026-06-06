---
title: >-
  [Paper Note] AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models
description: >-
  [ICCV 2025][Multimodal VLM][AIGC detection] This paper proposes AIGI-Holmes, which achieves explainable and generalizable AI-generated image detection through the construction of Holmes-Set — an annotated dataset with in…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "AIGC detection"
  - "explainable AI"
  - "MLLM"
  - "DPO"
  - "collaborative decoding"
date: 2026-05-08
content_hash: a40d1fc43181767b
---

# AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models

**Conference**: ICCV 2025
**arXiv**: N/A  
**Code**: [GitHub](https://github.com/wyczzy/AIGI-Holmes)  
**Area**: Multimodal VLM
**Keywords**: AIGC detection, explainable AI, MLLM, DPO, collaborative decoding

## TL;DR

This paper proposes AIGI-Holmes, which achieves explainable and generalizable AI-generated image detection through the construction of Holmes-Set — an annotated dataset with interpretive labels — a three-stage training pipeline (visual expert pre-training → SFT → DPO), and a collaborative decoding strategy. The method attains state-of-the-art detection accuracy on three benchmarks while providing human-verifiable explanations.

## Background & Motivation

AI-generated images (AIGI) are becoming increasingly photorealistic and are being misused to spread misinformation, posing serious threats to public information security. Existing detection methods face two major challenges: (1) **lack of explainability** — black-box models cannot provide human-verifiable evidence; and (2) **lack of generalizability** — they struggle to handle the latest generative techniques (e.g., FLUX, SD3.5, VAR). MLLMs, endowed with commonsense understanding and natural language generation capabilities, are ideal candidates for explainable detection. However, direct SFT yields suboptimal results — MLLMs are insufficient in image classification and low-level perception, and SFT models tend to mechanically replicate explanation templates rather than genuinely understanding the reasons for forgery. Furthermore, there is a lack of interpretive training datasets suitable for the SFT stage.

## Method

### Overall Architecture

The Holmes Pipeline consists of three training stages and one inference strategy. The training stages are: (1) **Visual Expert Pre-training** — rapidly adapting the visual encoder via binary classification on Holmes-SFTSet; (2) **Supervised Fine-Tuning (SFT)** — training the MLLM to generate detection results accompanied by explanations; (3) **Human-Aligned DPO** — performing direct preference optimization on Holmes-DPOSet. During inference, a collaborative decoding strategy is employed to integrate the perceptual signals from the visual expert with the semantic reasoning of the MLLM.

### Key Designs

1. **Holmes Dataset Construction (Multi-Expert Jury Annotation)**: Holmes-SFTSet contains 65K images annotated across high-level semantic dimensions (physical inconsistencies, anatomical errors, text rendering defects) and low-level artifacts (overall tone, texture, edges). Cross-model validation and expert-guided filtering are applied to ensure annotation quality. Holmes-DPOSet constructs contrastive explanation pairs via positive/negative prompting, supplemented by 4K expert-corrected samples, to achieve alignment with human judgment. Images are generated using 18 AIGC methods spanning GANs and diffusion models.

2. **Three-Stage Holmes Pipeline**: The first stage enables the visual encoder to rapidly acquire domain-specific feature extraction capabilities through binary classification (analogous to adapter pre-training). The second stage SFT enables the MLLM to not only detect but also generate explanations — addressing the "black-box" limitation. The third stage DPO learns from preference pairs to fundamentally reshape the MLLM's reasoning patterns, ensuring that explanations align with human judgment criteria rather than remaining at the suboptimal SFT level.

3. **Collaborative Decoding Strategy**: At inference time, the visual expert's model-level perception is integrated with the MLLM's semantic reasoning to create a dual-channel verification process. The visual expert provides low-level artifact detection signals while the MLLM provides high-level semantic analysis; the two are complementary and enhance generalization, particularly when encountering unseen generative methods.

### Loss & Training

Sequential three-stage training: visual expert pre-training (binary cross-entropy) → SFT (standard next-token prediction) → DPO (preference optimization loss). The DPO stage uses contrastive explanation pairs from Holmes-DPOSet as preference data.

## Key Experimental Results

### Main Results

Validated on three benchmark datasets under a generalization setting involving unseen generative methods:

| Method | Unseen GAN Detection | Unseen Diffusion Model Detection | Unseen Autoregressive Model Detection |
|--------|---------------------|----------------------------------|---------------------------------------|
| UnivFD | Baseline | Baseline | Baseline |
| DRCT | Moderate | Moderate | Moderate |
| **AIGI-Holmes** | **SOTA** | **SOTA** | **SOTA** |

Generalization performance on the latest generators (FLUX, SD3.5, VAR, etc.) significantly surpasses existing methods.

### Ablation Study

- SFT only vs. full pipeline: the DPO stage significantly improves both detection accuracy and explanation quality.
- Visual expert pre-training: provides the MLLM with domain-specific feature extraction capability; removing it leads to a clear performance drop.
- Collaborative decoding: fusing visual expert and MLLM outputs outperforms using either in isolation.

### Key Findings

- Directly applying SFT to train MLLMs for AIGC detection is suboptimal; DPO alignment is critical.
- Domain adaptation of the visual encoder (via binary classification pre-training) is key to generalization.
- High-level semantic analysis (physical inconsistencies, anatomical errors) and low-level artifact analysis are complementary.
- Explainability not only enhances trustworthiness but also indirectly improves detection generalizability.

## Highlights & Insights

- The "Holmes-like" design philosophy — not merely identifying real vs. fake, but providing a chain of evidence.
- The three-stage training design is tightly interconnected, with each stage addressing a specific limitation.
- Holmes-DPOSet is the first human-aligned preference dataset for AIGC detection.
- The collaborative decoding strategy elegantly combines the respective strengths of traditional classifiers and MLLMs.

## Limitations & Future Work

- Dataset construction still relies on expert filtering and manual correction, making scaling costly.
- The inference overhead of MLLMs makes real-time large-scale detection impractical.
- Current explanations are primarily grounded in visual artifacts and may fail on high-quality generated images that are free of perceptible artifacts.
- Robustness against adversarial attacks (e.g., adversarial examples targeting the detector) has not been evaluated.

## Related Work & Insights

- UnivFD employs CLIP-ViT features; DRCT introduces reconstruction-contrastive learning; NPR analyzes neighborhood pixel relationships.
- DD-VQA and FFAA pioneer the use of MLLMs for deepfake detection.
- The success of DPO in LLM alignment is effectively transferred to the visual detection domain.
- The Multi-Expert Jury approach used in dataset construction is extensible to other annotation tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The three-stage pipeline and collaborative decoding design are novel.
- **Technical Depth**: ⭐⭐⭐⭐ — Dataset construction and training pipeline design are rigorous.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three benchmarks, generalization tests, and complete ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — The Holmes analogy makes for vivid and engaging presentation.
- **Value**: ⭐⭐⭐⭐ — Explainable detection holds significant value in real-world scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Spatial Preference Rewarding for MLLMs Spatial Understanding](spatial_preference_rewarding_for_mllms_spatial_understanding.md)
- [\[ICCV 2025\] BASIC: Boosting Visual Alignment with Intrinsic Refined Embeddings in Multimodal Large Language Models](basic_boosting_visual_alignment_with_intrinsic_refined_embeddings_in_multimodal_.md)
- [\[ICCV 2025\] CapeLLM: Support-Free Category-Agnostic Pose Estimation with Multimodal Large Language Models](capellm_support-free_category-agnostic_pose_estimation_with_multimodal_large_lan.md)
- [\[ICCV 2025\] Oasis: One Image is All You Need for Multimodal Instruction Data Synthesis](oasis_one_image_is_all_you_need_for_multimodal_instruction_data_synthesis.md)
- [\[ICCV 2025\] DocThinker: Explainable Multimodal Large Language Models with Rule-based Reinforcement Learning for Document Understanding](docthinker_explainable_multimodal_large_language_models_with.md)

</div>

<!-- RELATED:END -->
