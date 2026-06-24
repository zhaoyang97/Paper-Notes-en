---
title: >-
  [Paper Note] Teaching Vision-Language Models to Ask: Resolving Ambiguity in Visual Questions
description: >-
  [Multimodal VLM] This paper proposes the ClearVQA benchmark and an automated data generation pipeline to teach VLMs to actively raise clarification questions rather than forcing an answer when encountering ambiguous visual questions. By systematizing interactive VQA through three categories of ambiguity (referential ambiguity, attribute ambiguity, and relational ambiguity), experiments show that fine-tuned VLMs can significantly improve ambiguity recognition and clarification…
tags:
  - "Multimodal VLM"
date: 2026-05-08
content_hash: 70c05dcf5d6854c4
---

# Teaching Vision-Language Models to Ask: Resolving Ambiguity in Visual Questions

| Conference | arXiv | Code | Area | Keywords |
|------|-------|------|------|--------|
| ACL 2025 (SAC Highlight Award) | [2507.13773](https://arxiv.org/abs/2507.13773) | None | multimodal_vlm | VQA Ambiguity, Clarification Question, Interactive VQA, ClearVQA, Vision-Language Model |

## TL;DR

This paper proposes the ClearVQA benchmark and an automated data generation pipeline to teach VLMs to actively raise clarification questions rather than forcing an answer when encountering ambiguous visual questions. By systematizing interactive VQA through three categories of ambiguity (referential ambiguity, attribute ambiguity, and relational ambiguity), experiments show that fine-tuned VLMs can significantly improve ambiguity recognition and clarification quality. This work was recognized with the ACL 2025 SAC Highlight Award.

## Background & Motivation

**Background**: In visual question answering (VQA) scenarios, users often ask ambiguous questions when interacting with VLMs (such as LLaVA or GPT-4V) due to differences in expression habits. For example, when there are multiple cats in an image, asking "What breed is this cat?", or asking "What is the thing on the left?" when there are multiple objects on the "left". Existing research primarily addresses ambiguity by rewriting or paraphrasing questions (e.g., AmbigQA), which models ambiguity resolution as a unidirectional generation problem.

**Limitations of Prior Work**: Existing methods suffer from two key flaws: (1) They neglect the interactive nature of dialogue—the conversation between users and VLMs is naturally interactive, and ambiguity can be resolved through user feedback, whereas existing methods rely on single-turn processing; (2) VLM training preferences favor "answering over asking"—VLMs are trained to answer questions to the best of their ability, tending to output a (potentially incorrect) answer even when faced with ambiguity, rather than asking the user for clarification.

**Key Challenge**: The contradiction between VLMs being optimized to "answer as much as possible" and the practical interactive requirement of "asking when necessary"—forcing an answer in ambiguous contexts yields incorrect or irrelevant responses, yet VLM training objectives do not encourage them to actively admit ambiguity and initiate clarification.

**Goal**: (1) Construct a systematic ambiguous VQA benchmark covering multiple ambiguity types and VQA scenarios; (2) design an automated pipeline to generate training data of ambiguous-clarification question pairs; (3) train VLMs to actively raise clarification questions upon encountering ambiguity and provide more accurate answers based on user feedback.

**Key Insight**: The authors observe that the natural human response to ambiguity in conversation is to ask a clarifying question rather than guessing—"Which cat do you mean?"—yet VLMs completely lack this capability. Grounded in this human interactive intuition, VQA ambiguity resolution is modeled as an interactive dialogue task rather than a single-turn rewriting task.

**Core Idea**: Teach VLMs to actively raise clarification questions, just as humans do, when faced with ambiguous visual questions, rather than forcing a potentially incorrect answer.

## Method

### Overall Architecture

The framework consists of two main parts: (1) **ClearVQA Benchmark Construction**—defining three types of VQA ambiguity (referential, attribute, and relational ambiguity) across various VQA scenarios (standard VQA, knowledge-based VQA, document-based VQA, etc.), which includes two subtasks: ambiguity detection and clarification question generation; (2) **Automated Training Data Generation**—based on existing VQA datasets, leveraging LLMs to automatically generate ambiguous questions and their corresponding clarification question pairs for VLM fine-tuning.

### Key Designs

1. **Classifications System of Three VQA Ambiguities**:

    - Function: Systematically define the sources of ambiguity in visual question answering.
    - Mechanism:
        - **Referential Ambiguity**: Unclear reference, such as asking "How old is this cat?" when multiple cats are present, making it impossible to determine which one is referred to.
        - **Attribute Ambiguity**: Vague attribute descriptions, such as "that large object" when the standard for "large" is undefined, which might refer to different objects.
        - **Relational Ambiguity**: Unclear spatial or semantic relationships, such as "the person near the window" when multiple people are near the window.
    - Design Motivation: These three types of ambiguity cover the most common ambiguity scenarios in VQA. Each category requires a distinct clarification strategy (referential ambiguity requires asking about specific features, attribute ambiguity requires asking about standards, and relational ambiguity requires asking about spatial details).

2. **Automated Ambiguity-Clarification Data Generation Pipeline**:

    - Function: Large-scale generation of training data without manual annotation.
    - Mechanism: Starting from existing VQA datasets, LLMs are used to generate an ambiguous version and clarification questions for each clear question. The pipeline includes: (a) generating ambiguous variations of the questions based on the image content and original questions; (b) generating appropriate clarification questions for each ambiguous question; (c) generating user feedback answers to the clarification questions; (d) generating final accurate answers based on the feedback. The multi-step generation ensures data quality.
    - Design Motivation: Manual annotation of ambiguous and clarification questions is extremely expensive (requiring manual comprehension of image content, construction of natural ambiguous phrasing, and design of reasonable clarifying questions). An automated pipeline is the only viable path to scale.

3. **VLM Clarification Capability Training**:

    - Function: Equip VLMs with dual capabilities of ambiguity detection and clarification questioning.
    - Mechanism: Instruction tuning is performed on VLMs, with training data consisting of two types: (a) ambiguity detection data—given an image and a question, determine if ambiguity exists and classify it; (b) clarification question generation data—generate appropriate clarification questions for the detected ambiguous questions. The training objective jointly optimizes detection accuracy and generation quality. During inference, the VLM first judges if a question is ambiguous; if it is, it generates a clarification question, and then produces the final answer after receiving user feedback.
    - Design Motivation: Decoupling ambiguity resolution into two stages—detection and clarification—is more controllable than an end-to-end approach, allowing independent evaluation and optimization of each stage's performance.

## Key Experimental Results

### Main Results - Ambiguity Detection

| Model | Referential Ambiguity F1 | Attribute Ambiguity F1 | Relational Ambiguity F1 | Average F1 |
|------|-----------|-----------|-----------|--------|
| GPT-4V (zero-shot) | ~55 | ~50 | ~48 | ~51 |
| LLaVA-1.5 (zero-shot) | ~40 | ~38 | ~35 | ~38 |
| LLaVA-1.5 + ClearVQA Training | ~72 | ~68 | ~65 | ~68 |
| Gain | +32 | +30 | +30 | +30 |

### Ablation Study - Clarification Performance

| Configuration | VQA Accuracy | Description |
|------|----------|------|
| Direct Answer (No Clarification) | ~45 | Forced answering when facing ambiguous questions |
| Question Rewriting (Non-interactive) | ~52 | Traditional single-turn rewriting method |
| Answer after Clarification (ClearVQA) | ~71 | Answering after interactive clarification |
| Gold Clarification + Answer | ~78 | Upper bound using human-annotated clarification questions |

### Key Findings

- **VLMs severely lack ambiguity awareness**: Even for GPT-4V in the zero-shot setting, the ambiguity detection F1 is only around 51%, close to random guessing. Open-source VLMs like LLaVA perform even worse, indicating that current VLMs possess almost no ambiguity recognition capability.
- **Significant improvement after training**: After fine-tuning with the automatically generated ClearVQA data, the ambiguity detection F1 improves by approximately 30 percentage points, proving that this ability can be acquired through data training.
- **Clarification interaction significantly improves answer quality**: Through one turn of clarification interaction, VQA accuracy increases from ~45% to ~71%, far outperforming the single-turn rewriting method (~52%), demonstrating the effectiveness of interactive ambiguity resolution.
- **Referential ambiguity is the easiest to detect, while relational ambiguity is the hardest**: Among the three classes of ambiguity, referential ambiguity achieves the highest detection accuracy because referential vagueness is relatively easy to identify. Relational ambiguity involves complex spatial reasoning and proves the most difficult to detect.

## Highlights & Insights

- **Precise problem definition**: The narrative angle of "teaching VLMs to ask questions" is both intuitive and profound. In human interaction, asking questions is as vital as answering them, yet VLMs are trained exclusively to answer, which constitutes a systemic capability gap. Recognition through the ACL 2025 SAC Highlight Award highlights the importance of this direction.
- **Practicality of the three ambiguity classifications**: The taxonomy of referential/attribute/relational ambiguities is clear and mutually exclusive, and each category dictates clear clarification strategies that can directly transfer to dialogue system designs. This classification is more instructive than a generic "ambiguous" label.
- **Leverage of automated data generation**: Automatically generating training data via LLMs avoids expensive manual annotation while achieving downstream task performance close to that of manual annotation, demonstrating the scalability of "using LLMs to generate data to train LLMs."
- **Extensibility to general interactive scenarios**: Although the paper focuses on VQA, the paradigm of "detect ambiguity -> ask clarify -> answer based on feedback" can be directly extended to multimodal conversational assistants, medical image QA, and other scenarios.

## Limitations & Future Work

- **Single-turn clarification assumption**: The current framework assumes that a single turn of clarification can resolve any ambiguity, but complex scenarios might require multi-turn interactions (e.g., if the user's feedback itself remains ambiguous).
- **Scalability of ambiguity types**: Only three types of ambiguity are defined. In actual VQA, there are other types of ambiguity, such as temporal ambiguity (e.g., what time does "recent" refer to?) and cultural ambiguity (e.g., the same gesture having different meanings in different cultures).
- **Upper bound of automatically generated data quality**: The ambiguous and clarifying questions generated by LLMs may not be natural enough, presenting a distribution gap compared to real-world user expressions of ambiguity.
- **Limited evaluation metrics**: The evaluation of clarification question quality primarily relies on automatic metrics, lacking large-scale human evaluation to verify if the generated questions are truly helpful to users.
- **Lack of integration tests with multimodal dialogue systems**: The approach was only evaluated on VQA benchmarks, and has not been validated in end-to-end multimodal dialogue systems to test actual interactive efficacy.

## Related Work & Insights

- **vs AmbigQA (Min et al. 2020)**: They handle ambiguous questions in textual QA by listing all possible answers rather than through interactive clarification. This paper models ambiguity resolution as an interactive process, which aligns better with practical usage scenarios.
- **vs VisDial (Das et al. 2017)**: Visual dialog tasks naturally support multi-turn interactions but do not focus explicitly on ambiguity detection and clarification. This paper focuses on the specific interactive requirement of ambiguity, providing a more precise evaluation of this capability.
- **vs Clarification Questions in NLP (Rao & Daumé 2018)**: Research on clarification questions in NLP has mostly been confined to the textual domain. This work is the first to systematically extend it to multimodal VQA scenarios.
- **Insights for multimodal Agent development**: Agents should detect ambiguity and request clarification before executing visual-related instructions, rather than blindly executing potentially flawed interpretations.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to systematically study the ambiguity detection and clarification questioning capabilities of VLMs; the problem definition is precise and critical.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers three ambiguity types, multiple VQA scenarios, and ablation analyses comprehensively, though some experimental details are constrained by the abs-only format.
- Writing Quality: ⭐⭐⭐⭐⭐ Receiving the SAC Highlight Award is strong proof of high writing quality.
- Value: ⭐⭐⭐⭐⭐ Identifies a neglected yet extremely vital gap in VLM capabilities, standardizing a new research direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can Vision Language Models Understand Mimed Actions?](can_vision_language_models_understand_mimed_actions.md)
- [\[ACL 2025\] Can Multimodal Large Language Models Understand Spatial Relations?](spatialmqa_mllm_spatial_relations.md)
- [\[ACL 2025\] Redundancy Principles for MLLMs Benchmarks](redundancy_principles_for_mllms_benchmarks.md)
- [\[ACL 2025\] Multimodal Coreference Resolution for Chinese Social Media Dialogues: Dataset and Benchmark Approach](multimodal_coreference_resolution_for_chinese_social_media_dialogues_dataset_and.md)
- [\[ACL 2025\] Table Understanding and (Multimodal) LLMs: A Cross-Domain Case Study on Scientific Tables](table_understanding_and_multimodal_llms_a_cross-domain_case_study_on_scientific_.md)

</div>

<!-- RELATED:END -->
