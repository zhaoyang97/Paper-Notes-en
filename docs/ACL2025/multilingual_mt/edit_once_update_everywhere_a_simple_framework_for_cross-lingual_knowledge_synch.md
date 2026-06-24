---
title: >-
  [Paper Note] Edit Once, Update Everywhere: A Simple Framework for Cross-Lingual Knowledge Synchronization in LLMs
description: >-
  [ACL 2025][Multilingual & Machine Translation][Knowledge Editing] This paper proposes the X-KDE framework, which achieves "edit in one language, update in all languages" cross-lingual knowledge synchronization via Cross-lingual Edition Instruction Tuning (XE-IT) and Target-language Preference Optimization (TL-PO). It achieves an average improvement of +8.19% on the Bi-ZsRE and MzsRE benchmarks, significantly outperforming all existing methods in cross-lingual scenarios.
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Knowledge Editing"
  - "Cross-lingual"
  - "Multilingual Synchronization"
  - "Preference Optimization"
  - "Instruction Tuning"
date: 2026-05-08
content_hash: c4884cd513832c25
---

# Edit Once, Update Everywhere: A Simple Framework for Cross-Lingual Knowledge Synchronization in LLMs

**Conference**: ACL 2025  
**arXiv**: [2502.14645](https://arxiv.org/abs/2502.14645)  
**Code**: [https://github.com/YukinoshitaKaren/X_KDE](https://github.com/YukinoshitaKaren/X_KDE)  
**Area**: Multilingual Translation  
**Keywords**: Knowledge Editing, Cross-lingual, Multilingual Synchronization, Preference Optimization, Instruction Tuning

## TL;DR
This paper proposes the X-KDE framework, which achieves "edit in one language, update in all languages" cross-lingual knowledge synchronization via Cross-lingual Edition Instruction Tuning (XE-IT) and Target-language Preference Optimization (TL-PO). It achieves an average improvement of +8.19% on the Bi-ZsRE and MzsRE benchmarks, significantly outperforming all existing methods in cross-lingual scenarios.

## Background & Motivation

**Background**: Knowledge editing enables efficient updates to knowledge within LLMs without comprehensive retraining. Mainstream approaches include parameter-modifying methods (e.g., ROME and MEMIT, which modify FFN parameters), fine-tuning methods (e.g., FT-L and FT-M, which fine-tune specific layers), and in-context methods (e.g., IKE utilizing in-context learning, and LTE utilizing retrieval augmentation).

**Limitations of Prior Work**: (1) Existing methods primarily focus on monolingual editing, neglecting cross-lingual knowledge synchronization—after editing a knowledge item in English, the model still retains the outdated knowledge in other languages such as Chinese, French, or Thai; (2) Parameter-modifying methods (e.g., ROME/MEMIT) suffer from a drastic degradation in performance under cross-lingual scenarios; (3) Even the strongest baseline, LTE, still leaves substantial room for improvement in reliability and portability on target languages.

**Key Challenge**: Multilingual knowledge representations in LLMs are not fully aligned—the same fact is encoded via different parameter pathways across different languages, causing monolingual edits to fail to propagate naturally to other languages.

**Goal**: To achieve genuine cross-lingual knowledge synchronization—ensuring that after editing knowledge in one (dominant) language, the updates are automatically propagated to all other languages.

**Key Insight**: Teach the model the behavioral pattern of "how to update knowledge when encountering an editing instruction" via instruction tuning, and then reinforce cross-lingual consistency using preference optimization.

**Core Idea**: XE-IT teaches the model the behavioral patterns of knowledge editing, and TL-PO propagates the editing effects across languages, jointly achieving cross-lingual knowledge synchronization in a two-stage manner.

## Method

### Overall Architecture
A two-stage training workflow: (1) Stage 1: XE-IT (Cross-lingual Edition Instruction Tuning), which fine-tunes the model on a meticulously constructed parallel dataset to enable it to modify target knowledge based on instructions while retaining irrelevant knowledge; (2) Stage 2: TL-PO (Target-language Preference Optimization), which employs preference optimization techniques such as DPO to ensure the consistency of the editing effects across target languages.

### Key Designs

1. **Cross-lingual Edition Instruction Tuning (XE-IT)**:

    - **Function**: Fine-tunes the model on a constructed parallel dataset where the training data contains editing instructions and corresponding demonstrations of knowledge updates.
    - **Mechanism**: Formulates the knowledge editing task as an instruction-following task—presenting the model with editing instructions of "old fact $\rightarrow$ new fact" along with demonstrations of updates for the same fact in multiple languages. The training objective is to enable the model to learn to: (a) modify the specified knowledge upon receiving editing instructions, (b) keep unrelated knowledge intact (locality), and (c) generalize editing effects to related questions (portability).
    - **Design Motivation**: It is more flexible than traditional knowledge editing methods that directly modify parameters—it eliminates the need to locate and alter specific parameters, and instead allows the model to autonomously learn the behavior of knowledge updating.

2. **Target-language Preference Optimization (TL-PO)**:

    - **Function**: Utilizes techniques like DPO (Direct Preference Optimization) to conduct preference learning, using correct updates in target languages as positive samples and outdated knowledge as negative samples.
    - **Mechanism**: While XE-IT may perform well on the source language, it might struggle with consistency in target languages. TL-PO further reinforces the model's tendency to select new knowledge over outdated knowledge in target languages through preference optimization.
    - **Design Motivation**: Even if the model acquires editing behavioral patterns, cross-lingual propagation still requires additional alignment signals, which preference optimization successfully provides.

3. **High-Quality Cross-lingual Dataset**:

    - **Function**: Constructs a specially designed parallel dataset comprising training samples for cross-lingual knowledge editing.
    - **Mechanism**: Each data instance contains the knowledge state before and after editing, multilingual question-answering pairs, and test samples for locality/portability.
    - **Design Motivation**: Most existing knowledge editing datasets are monolingual or simple bilingual datasets, which are insufficient for training genuine cross-lingual synchronization capabilities.

## Key Experimental Results

### Main Results (Bi-ZsRE, Llama2-Chat-7B, English Edit $\rightarrow$ Cross-lingual Test)

| Method | English Reliability | English Portability | Chinese Reliability | Chinese Portability | Overall Avg. |
|------|-----------------|-----------------|-----------------|-----------------|--------|
| FT-L | 53.51 | 53.31 | 51.81 | 55.14 | 61.90 |
| ROME | 96.09 | 58.87 | 49.94 | 51.81 | 73.43 |
| MEMIT | 95.21 | 57.77 | 52.05 | 52.19 | 74.46 |
| IKE | 99.59 | 71.27 | 67.83 | 58.97 | 73.33 |
| LTE | 99.91 | 77.40 | 76.86 | 67.49 | 84.28 |
| **X-KDE** | **99.93** | **76.41** | **94.81** | **77.43** | **91.04** |

### Ablation Study (MzsRE, 12-Language Average, English Edit)

| Method | English Reliability | Cross-lingual Avg. Reliability | Cross-lingual Avg. Generality |
|------|-----------------|----------------------|---------------------|
| FT-M | 99.96 | 63.65 | 62.04 |
| IKE | 99.65 | 76.78 | 76.65 |
| LTE | 100.00 | 79.38 | 79.16 |
| **X-KDE** | **99.93** | **90.24** | **90.14** |

### Key Findings
- X-KDE achieves an overwhelming advantage in cross-lingual scenarios—Chinese Reliability improves from LTE's 76.86% to 94.81% (+17.95%).
- In the 12-language scenario of MzsRE, the average cross-lingual Reliability increases from 79.38% to 90.24% (+10.86%).
- Parameter-modifying methods (e.g., ROME/MEMIT) suffer severe performance degradation in cross-lingual settings, with Chinese Reliability dropping to merely 49.94% and 52.05%, respectively.
- While maintaining high editing reliability, X-KDE preserves a Locality metric of over 90%+, indicating that irrelevant knowledge remains unaffected.
- X-KDE also maintains consistent advantages in both batch and sequential editing scenarios.

## Highlights & Insights
- The vision of "Edit Once, Update Everywhere" is highly practical—updating knowledge in multilingual LLMs only requires editing in a single language once.
- The two-stage design is elegant: XE-IT addresses "how to edit" while TL-PO ensures "cross-lingual consistency," complementing each other.
- Consistent improvements across 12 languages demonstrate the language-agnostic nature of the method, even for languages like Thai that differ significantly from English.

## Limitations & Future Work
- Evaluated primarily on 7B models; the effectiveness on larger LLMs remains unexplored.
- The preference optimization stage requires additional training data and computational resources.
- Scenarios of knowledge conflicts—such as how to handle pre-existing inconsistencies of the same fact across different languages—are not discussed.

## Related Work & Insights
- **vs. ROME/MEMIT**: Parameter-modifying methods significantly fail in cross-lingual settings, as knowledge for different languages is encoded in different parameter locations.
- **vs. LTE (Jiang et al., 2024)**: LTE achieves knowledge editing via SFT and retrieval augmentation but lacks guarantees for cross-lingual consistency. X-KDE bridges this gap by introducing preference optimization on top of LTE.
- **vs. IKE (Zheng et al., 2023)**: In-context learning approaches are flexible but exhibit poor Locality (56.95% vs. 90.15% for X-KDE).

## Rating
- Novelty: ⭐⭐⭐⭐ The formulation of the cross-lingual knowledge synchronization problem is valuable, and the two-stage scheme is clearly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on two benchmark datasets across 12 languages, covering batch/sequential editing with comparisons against various baselines.
- Writing Quality: ⭐⭐⭐⭐ The method is clearly articulated, and the experimental tables are comprehensive.
- Value: ⭐⭐⭐⭐ Highly practical value for the maintenance of knowledge within multilingual LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Cross-Lingual Transfer of Cultural Knowledge: An Asymmetric Phenomenon](cross-lingual_transfer_of_cultural_knowledge_an_asymmetric_phenomenon.md)
- [\[ACL 2025\] Cross-Lingual Auto Evaluation for Assessing Multilingual LLMs](cross-lingual_auto_evaluation_for_assessing_multilingual_llms.md)
- [\[ACL 2025\] Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs](mid_layer_crosslingual_alignment.md)
- [\[ACL 2025\] A Case Study of Cross-Lingual Zero-Shot Generalization for Classical Languages in LLMs](a_case_study_of_cross-lingual_zero-shot_generalization_for_classical_languages_i.md)
- [\[ACL 2025\] Cross-Lingual Pitfalls: Automatic Probing Cross-Lingual Weakness of Multilingual Large Language Models](crosslingual_pitfalls.md)

</div>

<!-- RELATED:END -->
