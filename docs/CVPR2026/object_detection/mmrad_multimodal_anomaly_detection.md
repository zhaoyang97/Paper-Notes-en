---
title: >-
  [Paper Note] MMR-AD: A Large-Scale Multimodal Dataset for Benchmarking General Anomaly Detection with MLLMs
description: >-
  [CVPR 2026][Object Detection][Anomaly Detection] MMR-AD constructs the largest multimodal reasoning-oriented industrial anomaly detection dataset to date (127K images, 188 product categories…
tags:
  - "CVPR 2026"
  - "Object Detection"
  - "Anomaly Detection"
  - "Multimodal Large Language Models"
  - "Reasoning Dataset"
  - "Reinforcement Learning"
  - "General Anomaly Detection"
date: 2026-05-08
content_hash: cd8d120c5d84bbce
---

# MMR-AD: A Large-Scale Multimodal Dataset for Benchmarking General Anomaly Detection with MLLMs

**Conference**: CVPR 2026  
**arXiv**: [2604.10971](https://arxiv.org/abs/2604.10971)  
**Code**: [https://xcyao00.github.io/MMR-AD](https://xcyao00.github.io/MMR-AD)  
**Area**: Multimodal VLM  
**Keywords**: Anomaly Detection, Multimodal Large Language Models, Reasoning Dataset, Reinforcement Learning, General Anomaly Detection

## TL;DR
MMR-AD constructs the largest multimodal reasoning-oriented industrial anomaly detection dataset to date (127K images, 188 product categories, 395 anomaly types) and proposes Anomaly-R1, a GRPO reinforcement learning-based baseline model that significantly outperforms general-purpose MLLMs.

## Background & Motivation

**Background**: Industrial anomaly detection has progressively evolved from single-class to multi-class to cross-class settings, with General Anomaly Detection (GAD) as the ultimate goal: training a unified model to directly detect anomalies in novel categories without retraining. MLLMs, with their powerful visual understanding and language reasoning capabilities, are regarded as a promising vehicle for achieving GAD.

**Limitations of Prior Work**: (1) MLLM pretraining data exhibits a significant domain gap with industrial AD scenarios; (2) existing AD datasets are image-only and ill-suited for MLLM post-training; (3) existing multimodal AD datasets (MMAD, Anomaly-Instruct-125K) either provide only multiple-choice questions without reasoning, or contain large amounts of non-industrial web data.

**Key Challenge**: General-purpose MLLMs fall far short of practical requirements for industrial AD, particularly in precise anomaly localization, and addressing this gap requires large-scale, high-quality multimodal AD training data.

**Goal**: Construct a large-scale reasoning-oriented multimodal AD dataset suitable for both training and evaluation, and validate a reinforcement learning-based AD baseline model.

**Core Idea**: Manually curate and filter samples from 14 public AD datasets, annotate bounding boxes, automatically generate reasoning-oriented text using a strong MLLM, and train a reasoning-capable AD model via GRPO reinforcement learning.

## Method

### Overall Architecture
Dataset construction: 14 public AD datasets → manual curation to remove low-quality samples → bounding box and text label annotation → Qwen2.5-VL-72B automatic reasoning text generation (reference image + input image + visual/text prompts) → text consistency verification.  
Baseline model: Qwen2.5-VL + LoRA → SFT cold start → GRPO reinforcement learning + contrastive sampling + domain knowledge injection.

### Key Designs

1. **Reasoning-Oriented Text Generation Pipeline**:

    - **Function**: Generate text annotations with detailed reasoning processes for each AD sample.
    - **Mechanism**: A paired normal reference image and a query image are provided to Qwen2.5-VL-72B, supplemented by red bounding box visual prompts and text prompts specifying anomaly type and coordinates. The model is instructed to produce responses in a "reason-then-answer" format. Consistency between predicted and ground-truth regions is used for validation.
    - **Design Motivation**: Anomalies are intrinsically deviations from normality; reference images allow the model to establish a baseline of normalcy. Reasoning-rich text is more conducive to learning step-by-step comparative analysis than simple answer labels.

2. **Contrastive Sampling + Consistency Penalty in GRPO**:

    - **Function**: Enhance reasoning capability and localization precision through reinforcement learning.
    - **Mechanism**: An outcome reward (correct answer: +1) is combined with a consistency penalty (−0.2 per missed bounding box when localization is inaccurate). Contrastive sampling ensures each query has both positive and negative responses: ground-truth texts from MMR-AD serve as guaranteed positive examples, while adversarial prompts are used to generate negative examples from all-positive response sets.
    - **Design Motivation**: Rewarding answer correctness alone reinforces the "blindly predicting Yes" pattern; the consistency penalty compels the model to genuinely learn anomaly localization. Contrastive sampling addresses the zero-gradient issue that arises in GRPO when all sampled responses are identical.

3. **Domain Knowledge Injection**:

    - **Function**: Guide the model to focus on known anomaly types for specific product categories.
    - **Mechanism**: Prompts are augmented with statements such as "This product may exhibit the following anomaly types: broken, deformation, …", directing the model to inspect specific defects rather than treating all visual differences as anomalous.
    - **Design Motivation**: Domain knowledge is necessary to distinguish normal intra-class variation from genuine anomalies in industrial settings.

### Loss & Training
SFT cold start → GRPO reinforcement learning. GRPO employs a PPO clip + KL penalty objective.

## Key Experimental Results

### Main Results

| Model | MVTecAD Detection Acc | MVTecAD Localization Acc | VisA Detection Acc |
|---|---|---|---|
| GPT-4o | ~70% | ~30% | ~65% |
| Gemini-2.5 | ~72% | ~35% | ~68% |
| Anomaly-R1-7B | ~85% | ~60% | ~80% |
| Anomaly-R1-7B† (+ Domain Knowledge) | ~88% | ~65% | ~83% |

### Ablation Study

| Configuration | Detection | Localization | Note |
|---|---|---|---|
| Full (SFT+RL) | Best | Best | Complete model |
| SFT only | Second best | Moderate | RL yields notable localization gains |
| Direct RL (w/o SFT) | Poor | Poor | Cold start is necessary |
| w/o Consistency Penalty | Good | Poor | Model learns to blindly predict Yes |

### Key Findings
- The strongest general-purpose MLLMs (GPT-4o, Gemini-2.5) remain far below practical requirements for industrial AD, with particularly poor precise localization.
- Reasoning-oriented text annotations are more beneficial for learning general AD capabilities than simple answer-only annotations.
- Reinforcement learning yields the most significant improvement over pure SFT in localization precision.
- Domain knowledge injection further boosts performance.

## Highlights & Insights
- **Dataset extensibility**: The raw bounding box annotations are provided, allowing future regeneration of text using stronger MLLMs—a forward-looking design choice worth emulating.
- **Consistency penalty**: Elegantly incorporates localization precision into the reward function, avoiding the reinforcement learning trap of rewarding "correct but imprecise" predictions.

## Limitations & Future Work
- Reasoning texts are generated by Qwen2.5-VL-72B, introducing model-specific biases.
- Despite its scale of 127K images, the dataset remains imbalanced across certain categories.
- Future work may explore additional RL algorithms and larger-scale models.

## Related Work & Insights
- **vs. MMAD**: MMAD provides only a multiple-choice format unsuitable for training; MMR-AD includes reasoning text enabling model fine-tuning.
- **vs. AnomalyGPT**: AnomalyGPT relies on direct SFT without reasoning processes, resulting in poor generalization.

## Rating
- Novelty: ⭐⭐⭐⭐ First large-scale reasoning-oriented AD dataset; the RL baseline has practical value.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive multi-model comparisons, ablations, and RL technique analyses.
- Writing Quality: ⭐⭐⭐⭐ Dataset construction and methodology are described clearly.
- Value: ⭐⭐⭐⭐⭐ The dataset makes a substantial contribution to the AD community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Kaputt: A Large-Scale Dataset for Visual Defect Detection](../../ICCV2025/object_detection/kaputt_a_large-scale_dataset_for_visual_defect_detection.md)
- [\[CVPR 2026\] Bidirectional Multimodal Prompt Learning with Scale-Aware Training for Few-Shot Multi-Class Anomaly Detection](bidirectional_multimodal_prompt_learning_with_scale-aware_training_for_few-shot_.md)
- [\[ICLR 2026\] ForestPersons: A Large-Scale Dataset for Under-Canopy Missing Person Detection](../../ICLR2026/object_detection/forestpersons_a_large-scale_dataset_for_under-canopy_missing_person_detection.md)
- [\[CVPR 2026\] InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models](invad_inversion-based_reconstruction-free_anomaly_detection_with_diffusion_model.md)
- [\[CVPR 2026\] Reasoning-Driven Anomaly Detection and Localization with Image-Level Supervision](reasoning-driven_anomaly_detection_and_localization_with_image-level_supervision.md)

</div>

<!-- RELATED:END -->
