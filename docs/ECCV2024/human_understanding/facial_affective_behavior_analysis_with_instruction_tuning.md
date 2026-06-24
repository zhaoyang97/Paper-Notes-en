---
title: >-
  [Paper Note] Facial Affective Behavior Analysis with Instruction Tuning
description: >-
  [ECCV 2024][Human Understanding][Facial Affective Behavior Analysis] This work proposes the first instruction-tuning dataset for Facial Affective Behavior Analysis (FABA), FABA-Instruct, along with an evaluation benchmark, FABA-Bench, and an efficient MLLM architecture, EmoLA. EmoLA achieves fine-grained description and recognition of emotions and Action Units (AUs) through a facial prior expert module and LoRA adaptation.
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "Facial Affective Behavior Analysis"
  - "Instruction Tuning"
  - "Multimodal Large Language Models"
  - "Action Unit Recognition"
  - "LoRA"
date: 2026-05-08
content_hash: d3f2d4ebc8001cbc
---

# Facial Affective Behavior Analysis with Instruction Tuning

**Conference**: ECCV 2024  
**arXiv**: [2404.05052](https://arxiv.org/abs/2404.05052)  
**Code**: [Yes](https://johnx69.github.io/FABA/)  
**Area**: Human Understanding  
**Keywords**: Facial Affective Behavior Analysis, Instruction Tuning, Multimodal Large Language Models, Action Unit Recognition, LoRA

## TL;DR

This work proposes the first instruction-tuning dataset for Facial Affective Behavior Analysis (FABA), FABA-Instruct, along with an evaluation benchmark, FABA-Bench, and an efficient MLLM architecture, EmoLA. EmoLA achieves fine-grained description and recognition of emotions and Action Units (AUs) through a facial prior expert module and LoRA adaptation.

## Background & Motivation

Facial Affective Behavior Analysis (FABA) is a critical technology for understanding human psychological states, covering two main tasks: Facial Expression Recognition (FER) and Action Unit Recognition (AUR). Traditional methods treat them as discriminative multi-classification or multi-label problems, which present three major limitations:

**Coarse-grained emotion description**: They only classify expressions into seven basic emotions (happiness, sadness, anger, etc.), failing to express subtle nuances such as compound emotions (e.g., "sadness with a forced smile"), exaggerated expressions, or emotional intensity.

**Lack of reasoning capability**: Binary AU annotations only indicate activation, failing to explain the intensity and causal relationships of muscle movements, such as the difference in activation intensity of AU6 (cheek raiser).

**Inability to leverage MLLM advantages**: Multimodal Large Language Models (MLLMs) excel in general visual understanding tasks, but applying them directly to FABA faces three main challenges: **a lack of suitable datasets**, **a lack of evaluation benchmarks**, and **difficulty in capturing facial structural priors**.

The authors argue that natural language description (rather than discrete labels) is a superior way to represent facial affective behavior, as it captures emotional complexity and nuances while being intuitive and quantifiable for humans. This insight drives the design of the entire work.

## Method

### Overall Architecture

EmoLA is built on LLaVA-1.5 and consists of four core components: a visual expert (CLIP-L/14 + MLP), a facial prior expert (pretrained facial landmark detector + MLP), a language expert (tokenizer + word embedding), and a language decoder (Vicuna LLM + LoRA). The input facial image is processed by the two experts to extract visual tokens $H_v$ and prior tokens $H_p$, respectively. These are concatenated with the instruction text tokens $H_q$ and fed into the frozen LLM decoder to generate descriptions in an autoregressive manner.

### Key Designs

1. **FABA-Instruct Instruction Tuning Dataset**: The first instruction tuning dataset designed for FABA. It randomly samples 19,877 in-the-wild facial images from AffectNet and utilizes GPT-4V to annotate fine-grained emotional and AU descriptions via 100 well-designed templates. The average length of emotion descriptions is 50.47 words, and AU descriptions average 207.35 words, containing reasoning information such as muscle movement causes, potential corresponding emotions, and relationships between AUs. These descriptions can express hybrid emotions, exaggerated expressions, emotional intensity, and undefined emotions that traditional labels fail to cover.

2. **Facial Prior Expert Module**: Since the CLIP visual encoder is trained on general image-text pairs, it struggles to capture structural facial details (such as landmark locations). Thus, a pretrained InsightFace landmark detector $f_p$ is introduced to extract the facial prior feature $Z_P = f_p(X_V)$, which is then projected into the token embedding space using an MLP:

$$H_p = g_\theta(Z_P)$$

This prior token provides the LLM with structural details of the face (such as landmark topological relationships) that the visual encoder might ignore. Experiments demonstrate that **using only a single prior token can still maintain a certain level of recognition capability**, indicating that landmark priors are highly representative for FABA tasks.

3. **REGE Evaluation Metric**: Traditional FABA metrics (Accuracy/F1) focus only on recognition capability, while NLG metrics (BLEU/ROUGE) focus only on text quality; neither is complete. REGE evaluates both recognition and generation simultaneously:

$$S_{rege} = S_{re} + S_{ge}$$

where $S_{re}$ is the recognition score (accuracy for FER, F1 for AUR), and $S_{ge}$ is the ROUGE score. For emotion recognition, the emotional category is extracted from the free-form text using a predefined synonym vocabulary, and then accuracy is computed.

### Loss & Training

During training, the visual encoder, prior encoder, word embeddings, and LLM decoder are frozen. Only three sets of parameters $\Theta = \{\theta, \gamma, \phi\}$ representing the prior projector, visual projector, and LoRA are optimized. The optimization objective is the autoregressive language modeling loss:

$$p(X_A|X_V, Z_P, X_Q) = \prod_{i=1}^{L} p_\Theta(x_i | X_V, Z_P, X_Q, X_{A,<i})$$

The model is optimized using AdamW with a learning rate of 1e-4 and a LoRA rank of 128, trained for only 1 epoch on 8 A6000 GPUs. EmoLA outperforms fully fine-tuned baselines while tuning only about 10% of the parameters.

## Key Experimental Results

### Main Results

**Emotion Recognition (RAF-DB)**:

| Method | Accuracy (%) | Remarks |
|------|-------------|------|
| APViT | 91.98 | Prev. SOTA |
| POSTER | 92.05 | Prev. SOTA |
| **EmoLA** | **92.05** | Comparable to SOTA |

**AU Recognition (DISFA, 8 AU Average F1)**:

| Method | Avg. F1 (%) | Remarks |
|------|------------|------|
| PIAP | 63.8 | Prev. SOTA |
| **EmoLA** | **65.1** | +1.3% |

**AU Recognition (GFT, 10 AU Average F1)**:

| Method | Avg. F1 (%) | Remarks |
|------|------------|------|
| EmoCo | 58.6 | Prev. SOTA |
| **EmoLA** | **62.1** | +3.5% |

**FABA-Bench (Comprehensive REGE Score)**:

| Method | Emotion $S_{rege}$ | AU $S_{rege}$ |
|------|-------------------|--------------|
| MiniGPT4-v2 | 77.8 | 37.8 |
| mPLUG-Owl2 | 82.0 | 55.7 |
| Shikra | 94.6 | 86.6 |
| LLaVA-1.5 | 93.9 | 91.4 |
| **EmoLA** | **96.2** | **91.5** |

### Ablation Study

**Effect of Prior Tokens**:

| Configuration | Emotion $S_{re}$ | AU $S_{re}$ | Explanation |
|------|-----------------|------------|------|
| Only $H_p$ | 41.2 | 40.5 | Single token still retains recognition capability |
| Only $H_v$ | 62.5 | 55.3 | Visual token baseline |
| $H_v + H_p$ | **64.5** | **56.3** | Prior provides complementary information |

**Effect of Fine-Tuning Strategies**:

| Configuration | Emotion $S_{re}$ | AU $S_{re}$ | Explanation |
|------|-----------------|------------|------|
| Only $g_\theta$ | 44.9 | 47.7 | Tuning only the prior projector |
| Only $h_\phi + h_\gamma$ | 63.0 | 55.6 | Tuning only LoRA + visual projector |
| Tuning both | **64.5** | **56.3** | Joint optimization is optimal |

### Key Findings

- EmoLA surpasses the fully fine-tuned LLaVA-1.5 using only 10% trainable parameters, proving the efficiency of LoRA combined with the facial prior.
- The facial prior token maintains a certain level of recognition capability even as the sole input (with no visual tokens), indicating that landmark features are highly relevant for FABA.
- EmoLA performs within 0.6% (64.2% vs 64.8%) of ReCoT on BP4D, with the gap attributed to ReCoT's consistency regularization and co-training.

## Highlights & Insights

- **Breakthrough at the Data Level**: This work constructs the first FABA instruction-tuning dataset, upgrading affective analysis from simple classification to "description + reasoning".
- **Lightweight and Efficient**: The facial prior introduces only a single input token but brings a significant performance boost in an elegant, minimal design.
- **Unified Evaluation**: The REGE metric is the first to integrate both recognition capabilities and text generation quality into a single evaluation framework.

## Limitations & Future Work

- Currently, only facial landmarks are utilized as priors; other potential priors like facial recognition features (ArcFace) or facial parsing maps remain unexplored.
- The framework only processes static images and has not been extended to temporal affective analysis in video streams.
- The GPT-4V annotations in FABA-Instruct may contain hallucinations or inconsistencies and have not undergone large-scale manual verification.
- The training data contains only around 19K images, representing a relatively small scale.

## Related Work & Insights

- Consistent with the LLaVA-1.5 architecture, but specialized for this task through the integration of the facial prior expert and LoRA adaptation.
- Insight: Domain-specific prior knowledge can be incorporated into general MLLMs via lightweight token injection, which is a promising approach for other fine-grained visual understanding tasks (e.g., medical imaging, remote sensing).

## Rating

- **Novelty**: ⭐⭐⭐⭐ The first instruction tuning work for FABA, integrating datasets, benchmarks, and models.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 4 traditional datasets + its own built benchmark, with comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation, rich diagrams, and highly persuasive analytical descriptions.
- **Value**: ⭐⭐⭐⭐ Opens the door to the MLLM era for the FABA community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Human Motion Instruction Tuning](../../CVPR2025/human_understanding/human_motion_instruction_tuning.md)
- [\[ECCV 2024\] Generalizable Facial Expression Recognition](generalizable_facial_expression_recognition.md)
- [\[AAAI 2026\] Facial-R1: Aligning Reasoning and Recognition for Facial Emotion Analysis](../../AAAI2026/human_understanding/facial-r1_aligning_reasoning_and_recognition_for_facial_emotion_analysis.md)
- [\[ICLR 2026\] From Pixels to Semantics: Unified Facial Action Representation Learning for Micro-Expression Analysis](../../ICLR2026/human_understanding/from_pixels_to_semantics_unified_facial_action_representation_learning_for_micro.md)
- [\[ICLR 2026\] PersonaX: Multimodal Datasets with LLM-Inferred Behavior Traits](../../ICLR2026/human_understanding/personax_multimodal_datasets_with_llm-inferred_behavior_traits.md)

</div>

<!-- RELATED:END -->
