---
title: >-
  [Paper Note] TextShield-R1: Reinforced Reasoning for Tampered Text Detection
description: >-
  [AAAI 2026][Reinforcement Learning][Tampered Text Detection] This paper proposes TextShield-R1, the first reinforcement learning-based multimodal large language model (MLLM) method for tampered text detection. The approach integrates forensic continual pre-training (a curriculum from natural images to text images), GRPO reinforcement learning (five carefully designed reward functions to reduce annotation dependency), and OCR rectification (leveraging the MLLM's text recognition capability to improve localization accuracy). Together with the newly introduced TFR benchmark (45K+ images, 16 languages, 10 tampering techniques), this work substantially advances the state of the art in interpretable tampered text detection.
tags:
  - AAAI 2026
  - Reinforcement Learning
  - Tampered Text Detection
  - Multimodal Large Language Models
  - GRPO
  - Continual Pre-training
  - Text Forensics
date: 2026-05-08
content_hash: 814a3dc6c5bf31d5
---

# TextShield-R1: Reinforced Reasoning for Tampered Text Detection

**Conference**: AAAI 2026
**arXiv**: [2602.19828](https://arxiv.org/abs/2602.19828)
**Code**: [github.com/qcf-568/TextShield](https://github.com/qcf-568/TextShield)
**Area**: Reinforcement Learning
**Keywords**: Tampered Text Detection, Multimodal Large Language Models, GRPO, Continual Pre-training, Text Forensics

## TL;DR

This paper proposes TextShield-R1, the first reinforcement learning-based multimodal large language model (MLLM) method for tampered text detection. The approach integrates forensic continual pre-training (a curriculum from natural images to text images), GRPO reinforcement learning (five carefully designed reward functions to reduce annotation dependency), and OCR rectification (leveraging the MLLM's text recognition capability to improve localization accuracy). Together with the newly introduced TFR benchmark (45K+ images, 16 languages, 10 tampering techniques), this work substantially advances the state of the art in interpretable tampered text detection.

## Background & Motivation

### State of the Field

The rapid advancement of image editing technologies has greatly lowered the barrier to fabricating tampered text images. Such forgeries are widely exploited for fraud, misinformation propagation, and other malicious purposes, posing serious security threats. Reliable tampered text detection has thus become an urgent need. While MLLMs have demonstrated strong potential for analyzing tampered images and generating textual explanations, they remain significantly limited on this specific task.

### Three Core Challenges Facing MLLMs

**Challenge 1: Insufficient Task Alignment.** Base MLLMs are primarily pre-trained on macro-level perception tasks (image captioning, object recognition) that focus on high-level semantics. Tampered text detection, by contrast, requires **micro-level perception** to identify semantically irrelevant forgery traces (e.g., pixel inconsistencies, texture anomalies). This large gap makes direct fine-tuning prone to confusion and overfitting.

**Challenge 2: Heavy Annotation Dependency.** Existing MLLM-based methods rely heavily on costly forgery explanation annotations, typically obtained via closed-source models such as GPT-4o. However, many credential images (identity cards, contracts) contain sensitive information that prohibits external exposure. Furthermore, forgery traces are often subtle and difficult to annotate automatically, requiring extensive manual cleaning. More critically, supervised fine-tuning in a "spoon-feeding" manner undermines the inherent reasoning and analytical capabilities of MLLMs.

**Challenge 3: Poor Localization Accuracy.** MLLMs are inherently weak at predicting precise text bounding boxes, particularly for dense text. Integrating additional traditional localization models introduces inference latency and may lead to inconsistent predictions or over-reliance on the biases of those models.

### Seven Deficiencies in Existing Benchmarks

The authors also identify seven key deficiencies in existing tampered text detection benchmarks: limited domain coverage (documents or scene text only), narrow scope (no globally generated fake images), imbalanced positive/negative samples, insufficient diversity of tampering techniques, outdated tampering methods, inadequate out-of-distribution (OOD) evaluation, and incomplete annotations.

## Method

### Overall Architecture

TextShield-R1 introduces innovations at three stages—pre-training, fine-tuning, and inference:

- **Pre-training**: Forensic Continual Pre-training (FCP)
- **Fine-tuning**: GRPO with five task-specific reward functions
- **Inference**: OCR Rectification

The design is plug-and-play and does not modify the base MLLM architecture.

### Key Designs

#### 1. **Forensic Continual Pre-training: A Forensic Curriculum from Easy to Hard**

**Core Idea**: Large-scale, high-quality natural image forgery datasets (low-cost, large-scale) are used to warm up the MLLM's tampering detection capability before transferring to text images.

**3D Forensic Learning**: For locally tampered natural images, the MLLM is required to simultaneously output information across three dimensions:
- A **description** of the tampered object (generated using the Describe Anything Model)
- The **bounding box coordinates** of the tampered region (minimum enclosing rectangle computed from the mask)
- A **mask string** of the tampered region (the mask is interpolated to 32×32 and encoded as a 0/1 string)

**Key Trade-off**: Pre-training on natural image forensics erodes the MLLM's OCR capability since text is not involved. The solution is to **interleave OCR-based referring localization tasks**:
- Given a real text image and a randomly selected text instance
- Task (a): Given a bounding box → output the text content
- Task (b): Given the text content → output the bounding box

Pre-training data scale:
- 120K locally tampered natural images (CASIAv1v2, IMD20, NIST16, MIML)
- 120K globally generated fake natural images (Community Forensic)
- 60K COCO + 60K LAION real images
- Real text images from the TFR benchmark training set (for OCR tasks)

#### 2. **GRPO + Five Reward Functions: Reinforcement Learning to Reduce Annotation Dependency**

During fine-tuning, approximately 25% of fully annotated data is used for cold-start SFT; the remaining images are trained under the GRPO framework with weak annotations (no forgery explanation labels).

Five carefully designed reward functions:

| Reward Type | Description | Reward Value |
|---|---|---|
| **Real/Fake Classification Reward** | Three-class: real / globally generated / locally tampered | Correct = 1, otherwise = 0 |
| **Forgery Method Detection Reward** | Determines whether the fake region is copy-pasted or generated | Correct = 1, otherwise = 0 |
| **Tampering Localization Reward** | IoU between predicted box and ground truth | IoU > 0.5: reward = IoU; otherwise = 0 |
| **Tampered Text OCR Reward** | Recognizes the content of tampered text | $1 - \text{normalized Levenshtein distance}$ |
| **Format Reward** | Reasoning within `<think>` tags, answer within `<answer>` tags | Correct format = 1, otherwise = 0 |

**Design Motivation**:
- Classification and method detection rewards guide the model toward correct high-level judgments
- Localization and OCR rewards provide fine-grained pixel-level and character-level feedback
- Format reward ensures structured output
- The forgery method detection reward is particularly insightful: different forgery methods leave distinct traces, encouraging the model to conduct deeper analysis

#### 3. **OCR Rectification: Leveraging Text Recognition to Improve Localization Accuracy**

A post-processing optimization applied at inference time, exploiting the observation that MLLMs excel at text recognition but are weak at precise localization:

1. An OCR engine extracts the content and coordinates of all text in the image
2. The MLLM predicts candidate tampered text and their bounding boxes
3. For each predicted tampered text string, the best match is retrieved from OCR results:
    - Matching criterion: minimum Levenshtein distance
    - Unique match: replace the MLLM's predicted coordinates with OCR coordinates
    - Multiple matches: select the one with the highest DIoU relative to the MLLM prediction
    - No match (normalized Levenshtein distance > 0.2): retain the MLLM's original prediction

### Loss & Training

- Base MLLM: Qwen2.5-VL-7B
- Pre-training: LoRA rank = 64, AdamW, learning rate 1e-4 → 0 cosine decay, 1 epoch
- Fine-tuning: 25% fully annotated cold-start SFT + 75% weakly annotated GRPO

## Key Experimental Results

### Main Results

**Comparison on the TFR Benchmark** (accuracy / OCR accuracy / IoU / reasoning score):

| Method | Test Cls. | Test OCR | Test Loc. | Test Res. | CIS Cls. | CTM Loc. | CL Cls. |
|---|---|---|---|---|---|---|---|
| GPT-4o (zero-shot) | 51.7 | 5.6 | 0.5 | 19.4 | 53.4 | 3.1 | 48.3 |
| Qwen2.5-VL-7B (zero-shot) | 42.6 | 6.4 | 0.1 | 9.5 | 49.9 | 0.6 | 50.1 |
| Qwen2.5-VL-7B (full fine-tune) | 79.1 | 24.3 | 18.2 | 42.9 | 71.1 | 34.2 | 85.1 |
| FakeShield* | 79.1 | 24.3 | 7.6 | 42.8 | 71.1 | 21.8 | 85.1 |
| **TextShield-R1** | **88.1** | **47.6** | **57.8** | **58.8** | **72.9** | **68.3** | **85.5** |

Compared to the full fine-tuning baseline, TextShield-R1 achieves gains of +9.0% (classification), +23.3% (OCR), +39.6% (localization), and +15.9% (reasoning).

### Ablation Study

| ID | Configuration | Test Cls. | Test OCR | Test Loc. | Test Res. |
|---|---|---|---|---|---|
| (1) | Baseline (full fine-tune) | 79.1 | 24.3 | 18.2 | 42.9 |
| (2) | w/o FCP (no forensic pre-training) | 75.8 | 21.9 | 12.7 | 39.0 |
| (3) | w/o GRPO (no reinforcement learning) | 87.6 | 46.8 | 57.7 | 58.6 |
| (4) | w/o OCR Rect. (no OCR rectification) | 88.1 | 47.6 | 42.7 | 58.8 |
| (5) | **TextShield-R1 (full)** | **88.1** | **47.6** | **57.8** | **58.8** |

### Key Findings

1. **Forensic continual pre-training contributes most**: Removing FCP causes across-the-board degradation, with classification dropping from 88.1 to 75.8, demonstrating that the transfer of forensic knowledge from natural images to text images is critical.
2. **GRPO primarily enhances reasoning quality**: The gap between (3) and (5) is small (58.6 vs. 58.8), indicating that GRPO mainly confers "reasoning" ability rather than simply improving classification or localization.
3. **OCR rectification yields large localization gains**: Localization IoU improves from 42.7 to 57.8 (+15.1), validating the effectiveness of leveraging text recognition to enhance localization.
4. **Strong cross-tampering-method (CTM) generalization**: On three tampering methods unseen during training, localization IoU still reaches 68.3.
5. **Zero-shot MLLM capability is highly limited**: GPT-4o achieves only 0.5% IoU on localization, demonstrating that tampered text detection far exceeds the capabilities of general-purpose MLLMs.

## Highlights & Insights

- **First application of RL to tampered text detection**, demonstrating that GRPO with carefully designed rewards can substantially reduce dependency on expensive annotations.
- **The transfer learning paradigm of forensic continual pre-training is elegant**: Natural image forgery data is abundant, high-quality, and cheap to obtain; the curriculum pre-training effectively transfers this knowledge to tampered text detection.
- **OCR rectification is a plug-and-play inference optimization**: It requires no modifications to training and no additional model parameters, yet significantly improves localization by leveraging an existing OCR engine.
- **The TFR benchmark is an independent and substantial contribution**: 45K+ images, 16 languages, 10 tampering techniques, 3 OOD evaluation settings, and reasoning explanation annotations—addressing all seven deficiencies of prior benchmarks in one effort.
- The combined design of five reward functions reflects a deep understanding of the task.

## Limitations & Future Work

- Dependency on an external OCR engine during the OCR rectification stage introduces additional inference latency.
- Experiments are conducted only on Qwen2.5-VL-7B; performance on larger or smaller models remains unknown.
- GRPO still requires approximately 75% of data to have weak annotations (real/fake labels and bounding boxes), and is thus not fully annotation-free.
- The mask encoding in 3D Forensic Learning is limited to 32×32 resolution as a 0/1 string, which is relatively coarse.
- Detection performance may still have room for improvement on high-quality GPT-4o-generated forgeries (the most recent methods in TFR).
- Localization IoU under the cross-lingual (CL) setting (40.6) remains lower than the in-domain setting (57.8), indicating that multilingual generalization warrants further improvement.

## Related Work & Insights

- **FakeShield, ForgeryGPT, and SIDA** use MLLMs to explain forgery traces in natural images, but rely heavily on annotations and are not applicable to text images.
- **DocTamper** is a representative dataset for document tampering detection, but contains no real images and covers only three tampering methods.
- **OSTF** introduces scene text tampering detection but excludes documents and credentials.
- The forensic continual pre-training approach shares methodological similarities with **RLVR** (RL for VLM reasoning): both enhance specific capabilities through reinforcement learning.
- The methodology provides a reference for broader MLLM safety applications, including deepfake detection and AI-generated content detection.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First application of RL to tampered text detection; the three-stage design (pre-training + RL + inference optimization) is complete and highly novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — New benchmark + multi-MLLM comparisons + three OOD evaluation settings + comprehensive ablation study.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem definition is clear and method motivation is well-grounded, though the paper is somewhat lengthy.
- **Value**: ⭐⭐⭐⭐⭐ — Significant practical security value; the TFR benchmark alone constitutes an important contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MMhops-R1: Multimodal Multi-hop Reasoning](mmhops-r1_multimodal_multi-hop_reasoning.md)
- [\[AAAI 2026\] MathSmith: Towards Extremely Hard Mathematical Reasoning by Forging Synthetic Problems with a Reinforced Policy](mathsmith_towards_extremely_hard_mathematical_reasoning_by_forging_synthetic_pro.md)
- [\[ICLR 2026\] RM-R1: Reward Modeling as Reasoning](../../ICLR2026/reinforcement_learning/rm-r1_reward_modeling_as_reasoning.md)
- [\[ICLR 2026\] RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling](../../ICLR2026/reinforcement_learning/rulereasoner_reinforced_rule-based_reasoning_via_domain-aware_dynamic_sampling.md)
- [\[CVPR 2026\] Reasoning-Driven Anomaly Detection and Localization with Image-Level Supervision](../../CVPR2026/reinforcement_learning/reasoning-driven_anomaly_detection_and_localization_with_image-level_supervision.md)

</div>

<!-- RELATED:END -->
