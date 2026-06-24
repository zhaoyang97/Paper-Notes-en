---
title: >-
  [Paper Note] CORDIAL: Can Multimodal Large Language Models Effectively Understand Coherence Relations?
description: >-
  [ACL 2025][Multimodal VLM][Multimodal Discourse Analysis] This paper introduces CORDIAL, the first benchmark designed to evaluate the multimodal discourse analysis capabilities of MLLMs using Coherence Relations. Spanning three discourse domains (disaster management, social media, and online articles), CORDIAL includes coherence relations of varying granularity. Experiments reveal that even Gemini 1.5 Pro and GPT-4o fail to match a simple CLIP-based classifier baseline…
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Multimodal Discourse Analysis"
  - "Coherence Relations"
  - "Image-Text Relations"
  - "Pragmatic Inference"
  - "MLLM Evaluation"
date: 2026-05-08
content_hash: 62c3472ee98ce279
---

# CORDIAL: Can Multimodal Large Language Models Effectively Understand Coherence Relations?

**Conference**: ACL 2025  
**arXiv**: [2502.11300](https://arxiv.org/abs/2502.11300)  
**Code**: [aashish2000/CORDIAL](https://aashish2000.github.io/CORDIAL/)  
**Area**: Multimodal VLM  
**Keywords**: Multimodal Discourse Analysis, Coherence Relations, Image-Text Relations, Pragmatic Inference, MLLM Evaluation

## TL;DR
This paper introduces CORDIAL, the first benchmark designed to evaluate the multimodal discourse analysis capabilities of MLLMs using Coherence Relations. Spanning three discourse domains (disaster management, social media, and online articles), CORDIAL includes coherence relations of varying granularity. Experiments reveal that even Gemini 1.5 Pro and GPT-4o fail to match a simple CLIP-based classifier baseline, highlighting a fundamental deficiency of MLLMs in pragmatic understanding.

## Background & Motivation
Multimodal Large Language Models (MLLMs) have demonstrated exceptional performance across various downstream tasks. However, existing benchmarks primarily focus on evaluating the **factual and logical correctness** of models, overlooking a critical dimension: **the models' capability to comprehend pragmatic cues and cross-modal relations**.

The core problems are:

**Literal vs. Pragmatic Relations**: Existing image-text alignment evaluations (such as CLIPScore) focus solely on literal-level overlap, ignoring non-literal relationships like irony, metaphor, and informational complementarity.

**Limitations of Similarity Scores**: Image-text relations are not simple "similar/dissimilar" binaries but exhibit diverse alignment states at various levels (object-level, scene-level, discourse-level).

**Existence of Multiple Coherence Relations**: In real-world multimodal discourse, image-text connections can take multiple forms, such as describing visible elements (Visible), extending information (Extension), or projecting implicit meanings (Projection).

Key Insight: This work borrows the concept of coherence relations from discourse coherence theory (Hobbs, 1978) to extend text-only discourse analysis to multimodal settings, evaluating whether MLLMs can predict and verify the coherence relations between images and text.

Core Idea: **Use coherence relations instead of similarity scores to evaluate MLLMs' understanding of image-text relationships, establishing a comprehensive evaluation paradigm from literal to pragmatic levels.**

## Method

### Overall Architecture
CORDIAL evaluates the coherence relation prediction and verification capability of MLLMs across three discourse domains:
- Disaster Management (DisRel): Binary classification (Similar/Complementary)
- Social Media (Tweet Subtitles): 5-class single-label classification
- Online Articles (CLUE): Dual settings of 5-class multi-label and single-label classification

### Key Designs

1. **Theoretical Foundations of Coherence Relations**:
    - Based on Hobbs' (1978) discourse coherence theory, four conditions for successful communication are defined: (1) message content is present in the discourse, (2) the message is relevant to the overall context, (3) new/unexpected properties are built upon the listener's prior knowledge, and (4) the speaker provides cues to guide the listener to recognize their intent.
    - Coherence relations represent a finite set of connections satisfying the aforementioned communication functions.

2. **Data Sources of Three Discourse Domains**:

   **DisRel (Disaster Management, Binary Classification)**:
    - 4,600 disaster-related tweet image-text pairs, with 500 in the test set (balanced 50% split).
    - **Similar**: Image and text share the same focus, conveying identical information.
    - **Complementary**: Image and text do not share the same focus, but one helps understand the other.

   **Tweet Subtitles (Social Media, 5-class Single-Label)**:
    - 16,000 open-domain Twitter image-text pairs, with 1,600 in the test set.
    - 3 entity-level relations: **Insertion** (text and image focus on the same visual entity but the text does not mention it explicitly), **Concretization** (both mention the primary entity but details may differ), and **Projection** (entities in text are implicitly related to objects in the image).
    - 2 scene-level relations: **Restatement** (text directly describes image content) and **Extension** (the image extends the story of the text).

   **CLUE (Online Articles, 5-class Multi/Single-Label)**:
    - 4,770 Conceptual Captions image-text pairs, with 1,183 in the test set.
    - 5 relations: **Visible** (text describes visible content in image), **Action** (text describes a dynamic process encompassing the captured moment in image), **Meta** (text guides inference on how the image was produced/presented), **Subjective** (text provides the speaker's reaction/evaluation of the image), and **Story** (text provides an independent description of the image context).
    - Dual evaluation settings of multi-label and single-label.

3. **CLIP Baseline Classifiers**:
    - Extract zero-shot multimodal embeddings using CLIP Text and Image encoders.
    - Train an MLP classifier on the training set of each dataset.
    - Purpose: Provide a simple yet reliable reference point to identify relation types where MLLMs are particularly weak.

4. **Evaluation Setup**:
    - Three prompting strategies: Zero-shot, Few-shot, and Chain-of-Thought (CoT).
    - Two types of tasks: Relation Prediction (RQ1) and Relation Verification (RQ2).
    - Coherence-aware fine-tuning (RQ3) is also explored: Finetuning Llama 3.2-V.

### Loss & Training
- Llama 3.2-V is used as the base model for fine-tuning experiments.
- Coherence-relation-aware fine-tuning is performed on the training sets of each dataset.
- Fine-tuning shows significant performance improvements, though the improvements for certain relation categories are limited.

## Key Experimental Results

### Main Results (Relation Prediction Macro F1)

| Model | DisRel | Tweet Subtitles | Description |
|--------|------|------|------|
| CLIP Classifier Baseline | **0.733** | **0.519** | Strongest simple baseline |
| Gemini 1.5 Pro (best prompt) | 0.699 | 0.271 | Falls behind CLIP baseline |
| GPT-4o (best prompt) | 0.555 | 0.274 | Falls behind CLIP baseline |
| Claude 3.5 Sonnet v2 | 0.669 | **0.323** | Strongest MLLM on Tweet Subtitles |
| InternVL 2.5 26B | 0.658 | 0.190 | Stronger performer among open-source models |

### Tweet Subtitles Class-wise F1

| Model | Insertion | Concretization | Projection | Restatement | Extension |
|------|------|------|------|------|------|
| CLIP Baseline | **0.542** | **0.866** | **0.286** | **0.388** | **0.514** |
| Claude 3.5 Sonnet v2 | 0.180 | 0.725 | 0.138 | 0.316 | 0.256 |
| GPT-4o (Few) | 0.171 | 0.599 | 0.131 | 0.268 | 0.199 |

### Fine-Tuning Outcomes

| Setup | Finetuned Performance | Description |
|------|---------|------|
| DisRel | Significant Improvement | Especially on Zero-shot prompting |
| Tweet Subtitles | Partial Improvement | Limited improvement on pragmatic relations |
| CLUE SL/ML | Mixed Results | Fine-tuning demonstrates improvements across cross-domain dimensions |

### Key Findings
- **Simple CLIP Classifier Generically Defeats MLLMs**: Across all discourse domains, MLP classifiers trained on CLIP embeddings consistently outperform all MLLMs, including GPT-4o and Gemini 1.5 Pro.
- **Pragmatic Relations Pose the Greatest Challenge**: MLLMs perform worst on relations that require understanding pragmatic intent (such as Insertion, Projection, and Extension).
- **Larger Models Generally Outperform Smaller Models**: However, the gap narrows on pragmatic relations.
- **Inconsistent Effects Across Prompting Strategies**: CoT sometimes improves and sometimes degrades performance, and Few-shot is not always effective.
- **Coherence-Aware Fine-Tuning is Effective but Limited**: Fine-tuning can significantly improve the understanding of certain relations, but yields limited improvement for highly pragmatic relations.
- **Models Bias Toward Specific Relation Types**: For instance, multiple models overpredict the Concretization relation (on Tweet Subtitles) because it represents a literal relation.

## Highlights & Insights
- Pioneeringly introduces discourse coherence theory into MLLM evaluation, offering a novel perspective beyond similarity scores.
- The result of the CLIP classifier defeating MLLMs is eye-opening, suggesting that MLLMs' "understanding" might lean more toward knowledge retrieval rather than true relational reasoning.
- The selection of three distinct discourse domains (disaster management, social media, and online articles) covers varying communicative contexts from formal to casual.
- A progressive difficulty design from binary classification to multi-label settings comprehensively exposes MLLM weaknesses.
- Fine-tuning experiments (RQ3) suggest the potential of coherence-aware training, but also highlight fundamental difficulties in pragmatic understanding.

## Limitations & Future Work
- The benchmark scale is relatively limited, with a small number of relation categories in each domain.
- Annotations of coherence relations involve subjectivity and noise, particularly in the multi-label setting.
- More advanced few-shot learning strategies (e.g., retrieval-augmented ICL) are left unexplored.
- Future exploration direction: Constructing coherence-relation-aware pre-training objectives, rather than solely performing downstream fine-tuning.
- The evaluation can be extended to more modalities (e.g., image-text coherence relations in audio or video).
- In MLLM-as-judge scenarios, the deficiency in understanding coherence relations may lead to evaluation bias.

## Related Work & Insights
- Alikhani et al. (2020) extended textual coherence relations to multimodal contexts; CORDIAL establishes its evaluation framework based on this.
- Similarity-based evaluations like CLIPScore: CORDIAL demonstrates that similarity scores are insufficient for capturing comprehensive image-text relations.
- Image-text relation taxonomies such as Marsh & White (2003): Coherence relations in CORDIAL provide a more theoretically-grounded taxonomy perspective.
- Complementary to VLM2-Bench: VLM2-Bench evaluates visual-level alignment, whereas CORDIAL tests semantic/pragmatic-level relation understanding.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Pioneeringly and systematically applies discourse coherence theory to MLLM evaluation, offering a unique perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluates 10+ models, 3 prompting strategies, and 3 domains, though varying test set sizes across domains might affect overall conclusions.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated and coherence relations are precisely defined, though some tables are overly dense.
- Value: ⭐⭐⭐⭐ Pinpoints a critical blind spot in MLLM evaluation, posing sharp questions regarding whether MLLMs truly understand multimodal discourse.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can Multimodal Large Language Models Understand Spatial Relations?](spatialmqa_mllm_spatial_relations.md)
- [\[ACL 2025\] Can MLLMs Understand the Deep Implication Behind Chinese Images?](can_mllms_understand_the_deep_implication_behind_chinese_images.md)
- [\[ACL 2025\] NegVQA: Can Vision Language Models Understand Negation?](negvqa_can_vision_language_models_understand_negation.md)
- [\[ACL 2025\] Can Vision Language Models Understand Mimed Actions?](can_vision_language_models_understand_mimed_actions.md)
- [\[ACL 2025\] Enhance Multimodal Consistency and Coherence for Text-Image Plan Generation](enhance_multimodal_consistency_and_coherence_for_text-image_plan_generation.md)

</div>

<!-- RELATED:END -->
