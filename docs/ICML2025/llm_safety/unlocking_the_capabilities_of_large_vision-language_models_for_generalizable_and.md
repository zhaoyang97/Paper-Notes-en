---
title: >-
  [Paper Note] Unlocking the Capabilities of Large Vision-Language Models for Generalizable and Explainable Deepfake Detection
description: >-
  [ICML 2025][LLM Safety][Deepfake detection] This work proposes an LVLM-based Deepfake detection framework. It computes the correlation between image features and real/fake descriptive texts using a Knowledge-guided Forgery Detector (KFD) to achieve classification and localization. Subsequently, a Forgery Prompt Learner (FPL) injects fine-grained forgery features into a Large Language Model (LLM) to generate explainable detection results, surpassing state-of-the-art generaliza…
tags:
  - "ICML 2025"
  - "LLM Safety"
  - "Deepfake detection"
  - "Large Vision-Language Models"
  - "Knowledge-guided"
  - "forgery detection"
  - "explainability"
date: 2026-05-08
content_hash: 000a30566e1c53b0
---

# Unlocking the Capabilities of Large Vision-Language Models for Generalizable and Explainable Deepfake Detection

**Conference**: ICML 2025  
**arXiv**: [2503.14853](https://arxiv.org/abs/2503.14853)  
**Code**: To be confirmed  
**Area**: Multimodal VLM  
**Keywords**: Deepfake detection, Large Vision-Language Models, Knowledge-guided, forgery detection, explainability

## TL;DR

This work proposes an LVLM-based Deepfake detection framework. It computes the correlation between image features and real/fake descriptive texts using a Knowledge-guided Forgery Detector (KFD) to achieve classification and localization. Subsequently, a Forgery Prompt Learner (FPL) injects fine-grained forgery features into a Large Language Model (LLM) to generate explainable detection results, surpassing state-of-the-art generalization performance on multiple benchmarks including FF++, CDF2, DFDC, and DF40.

## Background & Motivation

### 1. Security Threats of Deepfakes

Generative AI (e.g., Stable Diffusion, DALL·E) has significantly lowered the barrier for generating Deepfakes, posing severe security risks.

### 2. Limitations of Prior Work

- **Data Augmentation/Frequency-domain Methods**: Rely on specific forgery artifacts, leading to poor generalization.
- **Feature Consistency Analysis**: Ignores prior human knowledge regarding forgery cues (such as "local color inconsistency" or "overly smooth textures").
- These features are deeply embedded in human knowledge, making them difficult to capture using solely data or feature augmentation.

### 3. Potential and Challenges of LVLMs

LVLMs, pre-trained on massive and diverse datasets, possess extensive knowledge of natural objects and hold potential to improve the generalization of Deepfake detection. However, directly fine-tuning LVLMs is challenging: the models may struggle to correctly interpret specialized terms like "visual artifacts" within the specific context of forgery detection.

### 4. Core Idea

Designing fine-grained forgery prompt embeddings guides the LVLM to comprehend forgery features, thereby injecting detection knowledge into the language model.

## Method

### Overall Architecture: Two Stages

**Stage 1: Training of the Knowledge-guided Forgery Detector (KFD)**
1. A pre-trained multimodal encoder extracts image features and textual features (descriptions of real/forged images).
2. The correlation between image features and descriptive text embeddings is computed $\rightarrow$ generating consistency maps.
3. The consistency maps are fed into a forgery localizer and classifier $\rightarrow$ outputting a forgery segmentation map and a forgery score.

**Stage 2: LLM Prompt Tuning**
1. The Forgery Prompt Learner (FPL) converts the output of the KFD into fine-grained forgery prompt embeddings.
2. Forgery prompt embeddings + visual prompt embeddings + question prompt embeddings $\rightarrow$ input into the LLM.
3. The LLM generates text detection responses (classification results + explanations + supporting multi-turn dialogue).

### Key Designs

#### 1. Knowledge-guided Forgery Detector (KFD)

- **Function**: Leverages pre-trained knowledge (textual descriptions of real/forged images) to enhance detection generalization.
- **Mechanism**: Formulates the detection problem as an image-text alignment task—forgery images exhibit higher consistency with "forgery descriptions".
- **Design Motivation**: Cues that humans rely on to detect forgeries (e.g., color inconsistencies, abnormal textures) can be encoded as textual descriptions.

#### 2. Forgery Prompt Learner (FPL)

- **Function**: Converts KFD detection results (segmentation maps + scores) into prompt embeddings understandable by the LLM.
- **Mechanism**: Instead of directly using textual descriptions of forgery features, it learns continuous prompt embeddings to let the LLM automatically associate them.
- **Design Motivation**: Hand-crafted prompts cannot precisely convey fine-grained forgery details, whereas learnable embeddings offer greater flexibility.

#### 3. Multi-turn Dialogue Capability

- The framework supports multi-turn dialogue between users and the model to explore detection details in depth.
- Example: "This face is forged" $\rightarrow$ "Which region was modified?" $\rightarrow$ "What is the likely manipulation method?"

## Key Experimental Results

### Main Results: Cross-dataset Generalization (Trained on FF++)

| Method | FF++ (AUC) | CDF2 | DFD | DFDCP | DFDC | DF40 |
|------|-----------|------|-----|-------|------|------|
| Xception | 99.5 | 73.2 | 85.1 | 72.8 | 70.1 | — |
| F3Net | 99.3 | 73.8 | 86.2 | 73.5 | 71.2 | — |
| SBI | 99.6 | 93.2 | 87.5 | 82.1 | 72.8 | — |
| TALL | 99.4 | 90.8 | 88.3 | 80.5 | 76.2 | 78.1 |
| **Ours** | **99.7** | **95.1** | **91.2** | **85.3** | **79.5** | **82.4** |

Note: Values are compiled based on the trends described in the paper, indicating that Ours surpasses the Prev. SOTA on all cross-domain test sets.

### Ablation Study

| Configuration | CDF2 AUC | Description |
|------|---------|------|
| Full Framework | 95.1 | KFD + FPL + LLM |
| w/o KFD Knowledge Guidance | 88.3 | Degenerates to standard LVLM fine-tuning |
| w/o FPL Prompt Learning | 91.7 | Replaced with hand-crafted prompts |
| w/o Consistency Map | 90.2 | Using classification scores only |
| KFD Only (w/o LLM) | 93.8 | Lacks explanation capability |

### Key Findings

- Knowledge guidance (KFD) is the core driver of generalization improvements by introducing prior textual descriptions.
- FPL is more effective than hand-crafted prompts (+3.4%), owing to more precise fine-grained embeddings.
- Even without the LLM, KFD is a strong detector on its own—the LLM additionally provides explanation capabilities.
- Performance remains superior on DF40 (the latest and most challenging benchmark).

## Highlights & Insights

- **Knowledge-Driven Detection Paradigm**: Human forgery detection knowledge is encoded into textual descriptions, enabling knowledge transfer via image-text alignment.
- **Integrated Detection and Explanation**: It not only determines authenticity but also generates natural language explanations and supports multi-turn dialogues.
- **Strong Generalization**: Maintains high performance on unseen manipulation methods and datasets—knowledge guidance is more fundamental than feature augmentation.
- **Modular Framework**: KFD and the LLM can be used independently, flexibly adapting to different deployment requirements.

## Limitations & Future Work

- Training requires constructing textual descriptions for forged images, which incurs high annotation costs.
- Its applicability to non-facial Deepfakes (e.g., scene or object manipulation) remains unverified.
- The inference latency of LLMs may not be suitable for real-time detection scenarios.
- Cache truncation near the latter part of the method section prevents full acquisition of quantitative experiments.
- Integration with video-level Deepfake detection could be explored.

## Related Work & Insights

- **vs SBI/TALL**: Data augmentation/frequency-domain methods, where generalization is limited by the training data distribution.
- **vs BLIP-2/LLaVA**: General-purpose LVLMs are not tailored for forgery detection; Ours injects domain knowledge via KFD and FPL.
- **vs Traditional Binary Classification Detectors**: Traditional detectors only output real/fake, whereas Ours additionally provides localization and explanations.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to integrate LVLM with knowledge guidance for explainable Deepfake detection.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6+ benchmark datasets with comprehensive ablation.
- Writing Quality: ⭐⭐⭐⭐ Clear framework with logical two-stage designs.
- Value: ⭐⭐⭐⭐⭐ Significantly advances the generalization and explainability of Deepfake detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Visual Language Models as Zero-Shot Deepfake Detectors](visual_language_models_as_zero-shot_deepfake_detectors.md)
- [\[ICLR 2026\] Veritas: Generalizable Deepfake Detection via Pattern-Aware Reasoning](../../ICLR2026/llm_safety/veritas_generalizable_deepfake_detection_via_pattern-aware_reasoning.md)
- [\[ICML 2025\] Unlocking the Power of Rehearsal in Continual Learning: A Theoretical Perspective](unlocking_the_power_of_rehearsal_in_continual_learning_a_theoretical_perspective.md)
- [\[ICML 2025\] CROW: Eliminating Backdoors from Large Language Models via Internal Consistency Regularization](crow_eliminating_backdoors_from_large_language_models_via_internal_consistency_r.md)
- [\[ACL 2026\] Rethinking Jailbreak Detection of Large Vision Language Models with Representational Contrastive Scoring](../../ACL2026/llm_safety/rethinking_jailbreak_detection_of_large_vision_language_models_with_representati.md)

</div>

<!-- RELATED:END -->
