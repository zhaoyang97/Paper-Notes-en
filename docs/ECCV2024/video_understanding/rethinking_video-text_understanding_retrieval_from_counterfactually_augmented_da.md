---
title: >-
  [Paper Note] Rethinking Video-Text Understanding: Retrieval from Counterfactually Augmented Data
description: >-
  [ECCV 2024][Video Understanding][video-text understanding] Proposal of the Retrieval from Counterfactually Augmented Data (RCAD) task and the Feint6K dataset, revealing that SOTA video-text models lag far behind humans in action semantic understanding (InternVideo 58.2% vs. Human 95.2%), and introduction of the LLM-teacher method to improve action embedding learning via LLM knowledge distillation.
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "video-text understanding"
  - "counterfactual augmentation"
  - "action semantics"
  - "contrastive learning"
  - "LLM-teacher"
date: 2026-05-08
content_hash: 14bc1de9aef70098
---

# Rethinking Video-Text Understanding: Retrieval from Counterfactually Augmented Data

**Conference**: ECCV 2024  
**arXiv**: [2407.13094](https://arxiv.org/abs/2407.13094)  
**Code**: Yes (Project Page)  
**Area**: Video Understanding / Vision-Language  
**Keywords**: video-text understanding, counterfactual augmentation, action semantics, contrastive learning, LLM-teacher

## TL;DR
Proposal of the Retrieval from Counterfactually Augmented Data (RCAD) task and the Feint6K dataset, revealing that SOTA video-text models lag far behind humans in action semantic understanding (InternVideo 58.2% vs. Human 95.2%), and introduction of the LLM-teacher method to improve action embedding learning via LLM knowledge distillation.

## Background & Motivation
**Background**: Video-text foundation models (e.g., InternVideo, LanguageBind) have achieved outstanding results on standard retrieval tasks (up to 87.9% R@1), and are widely considered to possess strong video understanding capabilities.

**Limitations of Prior Work**: Standard evaluation tasks exhibit severe shortcuts and biases. Many questions can be answered solely based on objects and context from a single frame—e.g., seeing "cymbal" suggests "playing cymbals," and seeing an outdoor scene suggests "football."

**Key Challenge**: Existing evaluations cannot distinguish whether a model truly understands cross-frame action semantics or is merely exploiting shortcuts from objects and scenes. The core incremental value of video understanding compared to image understanding—cross-frame reasoning and action semantics—is obscured by existing benchmarks.

**Goal**: (1) To design a legacy-free/shortcut-free evaluation paradigm to expose model weaknesses; (2) To improve model learning of action semantics.

**Key Insight**: By utilizing human-annotated counterfactually modified descriptions—preserving the same objects and scenes while only altering the action—the model is forced to perform cross-frame reasoning. LLM knowledge is leveraged to language-inject more effective contrastive learning of actions.

**Core Idea**: Counterfactual augmentation exposes the model's deficient action understanding after eliminating object shortcuts, and LLM-teacher improves action embeddings through synthetic negative generation and soft-label distillation.

## Method

### Overall Architecture
This work consists of two components: (1) An evaluation framework—the RCAD task and the Feint6K dataset—to evaluate the model's true action understanding ability after removing shortcuts; (2) An improvement method—LLM-teacher—which generates negative descriptions of action variants using LLMs and enhances action representations through soft-labeled contrastive learning.

### Key Designs

1. **RCAD Task Design (Retrieval from Counterfactually Augmented Data)**

    - **Function**: Given a video and a set of candidate descriptions (1 positive and 5 negatives), retrieve the description that semantically matches the video. The negatives contain the same objects as the positive, differing only in the actions.
    - **Mechanism**: Negatives are counterfactually modified—keeping the textual structure and object entities unchanged, while only replacing the action verbs. For example, if the positive is "A man kicks a football," the negative might be "A man catches a football."
    - **Design Motivation**: To eliminate object-based shortcuts, forcing the model to perform cross-frame reasoning to understand action semantics.
    - Supports zero-shot evaluation without requiring downstream fine-tuning.

2. **Feint6K Dataset Construction**

    - **Function**: Construction of a high-quality counterfactually augmented video-text evaluation set.
    - **Mechanism**: Adoption of a human-in-the-loop system where 40 annotators manually modify action descriptions in the MSR-VTT and VATEX validation sets.
        - The new actions must be plausible in context but not happening in the video.
        - Providing annotators with demonstrations and feedback during the training phase.
        - Each annotation is audited; unqualified annotations are sent back for revision.
    - **Scale**: 6,243 videos, derived from the MSR-VTT validation set and VATEX test set.
    - The human baseline R@1 reaches 95.2% (MSR-VTT) and 96.8% (VATEX), demonstrating that the task is solvable and has a unique correct answer.

3. **LLM-teacher Method**

    - **Function**: Improving action embedding learning of video-text models using LLM knowledge.
    - **Mechanism (Three Steps)**:
        - **Synthetic Negative Generation**: Handled by extracting action/object tokens from the original descriptions using an AMR parser, and then generating variant descriptions with two methods:
       - Method I — Mask Filling: Using the MLM capability of XLM-RoBERTa to predict alternative action words.
       - Method II — LLM Chatbot: Leveraging the in-context learning of LLMs to generate more flexible replacements (e.g., modifying prepositions).
        - **Contrastive Learning**: Using synthetic negatives for comparison, where the loss is the temperature-scaled cross-entropy:
    $l = -\log \frac{\exp(\text{sim}(f_v, f_p)/\tau)}{\exp(\text{sim}(f_v, f_p)/\tau) + \sum_{i=1}^{k}\exp(\text{sim}(f_v, f_{n_i})/\tau)}$
        - **LLM Soft-Label Distillation**: Some synthetic negatives may be semantically similar to the original description and should not be treated as strictly negative. Sentence-BERT is used to calculate similarities between descriptions to serve as soft labels from the LLM teacher, aligning the model output using KL divergence:
    $l = \mathcal{L}_{\text{KL}}(z_{\text{video-text}}, z_{\text{LLM}})$
    - By default, 10 action-based synthetic descriptions are generated for each video.
    - **Design Motivation**: In standard contrastive learning, objects serve as a natural shortcut—the model only needs to distinguish between "cymbal" and "football" to minimize the contrastive loss, without ever truly learning action embeddings.

### Loss & Training
- Binary pseudo-label version (LLM-teacher-lbl): Standard cross-entropy contrastive loss.
- Soft-label version (LLM-teacher-lgt): KL divergence alignment with the soft distribution from the LLM teacher.
- Applied to two pre-trained models: SimVTP and InternVideo.

## Key Experimental Results

### Main Results — Standard Retrieval vs. RCAD

| Model | MSR-VTT R@1 | Feint6K R@1 | Gap | Human R@1 |
|------|-------------|-------------|------|----------|
| CLIP (Zero-shot) | 26.3 | 37.3 | — | 95.2 |
| InternVideo (Zero-shot) | 37.5 | 45.8 | -8.3 | 95.2 |
| InternVideo (Fine-tuned) | 49.1 | 58.6 | +9.5 | 95.2 |
| LanguageBind (Zero-shot) | 42.8 | 41.3 | -1.5 | 95.2 |
| SimVTP (Fine-tuned) | 50.2 | 35.7 | -14.5 | 95.2 |
| **+ LLM-teacher-lgt** | 49.5 | **43.5** | +7.8 | — |
| InternVideo (Fine-tuned) | 49.1 | 58.6 | — | 95.2 |
| **+ LLM-teacher-lgt** | 48.9 | **65.8** | +7.2 | — |

### VATEX Subset Results

| Model | VATEX R@1 | Feint6K R@1 | Human R@1 |
|------|-----------|-------------|----------|
| InternVideo (Fine-tuned) | 87.9 | 58.2 | 96.8 |
| + LLM-teacher-lgt | 87.3(-0.6) | **65.6**(+7.4) | — |
| SimVTP (Fine-tuned) | 76.6 | 33.6 | 96.8 |
| + LLM-teacher-lgt | 75.3(-1.3) | **40.1**(+6.5) | — |

### Ablation Study

| Configuration | VATEX R@1 | Feint6K R@1 | Description |
|------|-----------|-------------|------|
| DefaultGP (10 action descriptions, XLM-RoBERTa) | 87.3 | **65.6** | Default configuration |
| 5 action descriptions | 87.6 | 64.7 | -0.9, more negatives are better |
| 5 action + 5 object descriptions | 87.5 | 64.2 | Object negatives do not help |
| LLM Chatbot substitution | 87.0 | 65.9 | Slightly better but slower inference |

### Key Findings
- InternVideo standard retrieval drops from 87.9% to 58.2% on RCAD, a steep decline of 29.7%, lagging far behind humans by 38.6%.
- The cosine similarity change $|\Delta s|$ for object replacement is much larger than that for action replacement, proving that the model's embeddings are far more discriminative for objects than for actions.
- LLM-teacher-lgt (soft label) outperforms LLM-teacher-lbl (hard label) because some synthetic negatives are semantically close to the positive sample.
- LLM-teacher only degrades by 0.2-0.6% on standard retrieval but gains 7.2-7.4% on RCAD.
- Object negatives do not help improve RCAD, verifying that the model already possesses decent object embeddings, and the missing link is the action embedding.

## Highlights & Insights
- **Contribution of Evaluation Paradigm**: Eliminating shortcuts through counterfactual augmentation exposes fundamental deficiencies in action understanding for SOTA video-text models. The plunge from 87.9% to 58.2% is a wake-up call, indicating that high scores on standard benchmarks mostly result from object matching rather than motion/action comprehension.
- **In-depth Analysis of Shortcut Learning**: Objects serve as natural shortcuts in contrastive learning—CLIP pre-training already equips models with excellent object embeddings, meaning the model only needs to distinguish objects to minimize loss during video-text contrast, without needing to learn action semantics. This analysis is profound and backed by experimental evidence (the $\Delta s$ analysis).
- **Elegant Design of LLM-teacher**: It changes training data and objectives without modifying model architectures. Soft-label distillation works better than hard labels because descriptions like "kicking a ball" and "throwing a ball" are semantically close despite being different.

## Limitations & Future Work
- Feint6K is only based on MSR-VTT and VATEX, resulting in limited video diversity.
- RCAD has only 6 candidates per video (1 positive, 5 negatives); increasing the number of candidates could offer higher discriminative power.
- LLM-teacher experiences a slight degradation (0.2-0.6%) on standard retrieval, indicating a trade-off.
- Improvements on the video encoder side (e.g., better temporal modeling) were not explored, with optimization focused solely on training objectives.
- The human baseline is 95.2% instead of 100%, indicating that some counterfactual scenarios might be ambiguous.

## Related Work & Insights
- **vs. InternVideo**: InternVideo is pre-trained on 7 large-scale datasets, reaching 87.9% on standard retrieval but only 58.2% on RCAD. LLM-teacher boosts this to 65.8% without architectural modifications, showing that the issue is the training objective, not the model capacity.
- **vs. CLIP4Clip/ViCLIP**: These models extend CLIP to the video domain, but they all inherit CLIP's object bias, performing similarly poorly on RCAD.
- **vs. Counterfactual VQA**: This work draws inspiration from counterfactual data augmentation in NLP, but represents the first systematic application to video-text understanding evaluation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The counterfactual evaluation paradigm uncovers a blind spot in the field, and the LLM-teacher approach is simple yet effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive multi-model evaluation, human baselines, cosine similarity analysis, and ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Closely-knit motivational flow, with clear logic spanning from evaluation to analysis and method.
- Value: ⭐⭐⭐⭐⭐ Serves as a wake-up call to the field; RCAD could potentially become a new standard evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Text-Guided Video Masked Autoencoder](text-guided_video_masked_autoencoder.md)
- [\[ECCV 2024\] Data Collection-Free Masked Video Modeling](data_collection-free_masked_video_modeling.md)
- [\[NeurIPS 2025\] VGEnt: Graph-Based Retrieval-Reasoning-Augmented Generation for Long Video Understanding](../../NeurIPS2025/video_understanding/vgent_graph-based_retrieval-reasoning-augmented_generation_for_long_video_unders.md)
- [\[CVPR 2025\] Learning Audio-Guided Video Representation with Gated Attention for Video-Text Retrieval](../../CVPR2025/video_understanding/learning_audio-guided_video_representation_with_gated_attention_for_video-text_r.md)
- [\[NeurIPS 2025\] AdaVideoRAG: Omni-Contextual Adaptive Retrieval-Augmented Efficient Long Video Understanding](../../NeurIPS2025/video_understanding/adavideorag_omnicontextual_adaptive_retrievalaugmented_effic.md)

</div>

<!-- RELATED:END -->
