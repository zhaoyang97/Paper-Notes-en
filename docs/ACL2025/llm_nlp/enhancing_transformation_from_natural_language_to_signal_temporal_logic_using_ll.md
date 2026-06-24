---
title: >-
  [Paper Note] Enhancing Transformation from Natural Language to Signal Temporal Logic Using LLMs with Diverse External Knowledge
description: >-
  [ACL 2025][LLM (Other)][signal temporal logic] This paper proposes the STL-DivEn dataset (16K samples) and the KGST (Knowledge-Guided STL Translation) framework. By translating natural language to Signal Temporal Logic (STL) via a two-stage "generate-then-refine" pipeline, it achieves an STL Formula Accuracy of 0.5587, significantly outperforming GPT-4 (0.4733) and DeepSeek (0.4790).
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "signal temporal logic"
  - "NL-to-STL"
  - "formal specification"
  - "knowledge-guided refinement"
  - "dataset construction"
date: 2026-05-08
content_hash: 4d6a453dd7dada94
---

# Enhancing Transformation from Natural Language to Signal Temporal Logic Using LLMs with Diverse External Knowledge

**Conference**: ACL 2025  
**arXiv**: [2505.20658](https://arxiv.org/abs/2505.20658)  
**Code**: [https://github.com/YueFang0618/STL-DivEn](https://github.com/YueFang0618/STL-DivEn)  
**Area**: LLM/NLP - Formal Specification  
**Keywords**: signal temporal logic, NL-to-STL, formal specification, knowledge-guided refinement, dataset construction

## TL;DR
This paper proposes the STL-DivEn dataset (16K samples) and the KGST (Knowledge-Guided STL Translation) framework. By translating natural language to Signal Temporal Logic (STL) via a two-stage "generate-then-refine" pipeline, it achieves an STL Formula Accuracy of 0.5587, significantly outperforming GPT-4 (0.4733) and DeepSeek (0.4790).

## Background & Motivation

**Background**: Signal Temporal Logic (STL) is a formal specification language widely used in cyber-physical systems (such as autonomous driving and robotic control) to describe real-time and real-valued constraints. Manually writing STL formulas is time-consuming and error-prone, making the automatic translation from natural language to STL a highly valuable research direction.

**Limitations of Prior Work**: (1) The availability of NL-STL datasets is extremely scarce. DeepSTL randomly samples and generates data via templates, resulting in a severe lack of diversity; (2) Transformer-based models perform poorly when dealing with complex nested temporal constraints; (3) Even advanced models such as GPT-4 and DeepSeek present low accuracy in NL-to-STL translation.

**Key Challenge**: NL-to-STL translation requires a precise understanding of temporal semantics and numerical constraints, but existing models lack both high-quality training data and mechanisms to effectively utilize external knowledge to improve translation accuracy.

**Goal**: (1) Construct a high-quality, diverse NL-STL dataset; (2) Design an effective translation framework that leverages external knowledge to enhance the accuracy of STL generation.

**Key Insight**: Diverse dataset construction guided by clustering, paired with a two-stage translation framework of "fine-tuned generation + retrieval-augmented refinement."

**Core Idea**: Utilizing clustering-guided GPT-4 to generate a diverse NL-STL dataset, and implementing a generate-then-refine workflow where a fine-tuned LLM generates initial STLs, which are then refined by GPT-4 using retrieved similar exemplars to enhance translation accuracy.

## Method

### Overall Architecture
The framework consists of two main parts: (1) STL-DivEn dataset construction—starting from a human-crafted seed set, representative samples are selected via clustering to guide GPT-4 in generating new NL-STL pairs, which are then filtered by rules and manually verified; (2) KGST translation framework—LLaMA 3-8B is first fine-tuned on the dataset to generate initial STLs, and then Top-K similar samples are retrieved from the dataset as external knowledge for GPT-4 to evaluate and refine the initial STLs.

### Key Designs

1. **Diversity-Guided Augmentation**:

    - **Function**: Starting from 120 human-crafted seed pairs, K-means clustering is utilized to select 5 representative seeds to guide GPT-4 in generating new NL-STL pairs.
    - **Mechanism**: Sentence-Transformers is used to map NL-STL pairs into a high-dimensional vector space for clustering, selecting the cluster centers as in-context examples for GPT-4. The newly generated pairs are added to the seed set and dataset after passing syntax checks, Rouge score filtering (< 0.5 is considered sufficiently diverse), and human verification.
    - **Design Motivation**: Direct generation by GPT-4 tends to heavily mimic the provided examples. Selecting representative seeds via clustering maximizes output diversity.

2. **Generate-then-Refine Translation Workflow**:

    - **Function**: In the first step, LLaMA 3-8B is fine-tuned on STL-DivEn to generate initial STL formulas. In the second step, the Top-5 most similar NL-STL pairs are retrieved from the external knowledge base and sent to GPT-4 along with the original natural language and initial STL for refinement.
    - **Mechanism**: Fine-tuned models excel at capturing data distributions but lack precision in complex nested logic, while GPT-4 possesses strong reasoning capabilities but lacks domain knowledge. Combining them leverages their respective strengths.
    - **Design Motivation**: Self-Refine (GPT-4 self-refinement) actually degraded performance, indicating that refinement requires external knowledge reference rather than solely relying on the model's internal capability.

3. **Multi-layer Quality Assurance Mechanism**:

    - **Function**: Executes a two-stage filtering on generated NL-STL pairs—(1) automatic checks based on STL syntax rules; (2) Rouge score computation with the seed set to ensure diversity; (3) manual semantic consistency verification by 7 annotators over 2 months.
    - **Mechanism**: Syntax checks guarantee the correct format of STL formulas, Rouge filtering prevents duplicates, and human verification ensures semantic consistency between the NL and STL.
    - **Design Motivation**: NL-STL pairs generated by LLMs may contain syntax errors, duplicates with the seed set, or semantic inconsistencies.

## Key Experimental Results

### Main Results (STL-DivEn Dataset)

| Model | STL Formula Acc. | Template Acc. | BLEU |
|------|-----------------|---------------|------|
| DeepSTL | 0.1986 | 0.1883 | 0.0293 |
| GPT-3.5 | 0.3018 | 0.3034 | 0.0424 |
| GPT-4 | 0.4733 | 0.4741 | 0.0831 |
| DeepSeek | 0.4790 | 0.4825 | 0.0791 |
| GPT-4 + Self-Refine | 0.4422 | 0.4466 | 0.0521 |
| **KGST** | **0.5587** | **0.5627** | **0.2142** |

### Ablation Study (DeepSTL Dataset — Cross-Dataset Generalization)

| Model | STL Formula Acc. | Template Acc. | BLEU |
|------|-----------------|---------------|------|
| DeepSTL | 0.2002 | 0.2916 | 0.3332 |
| GPT-4 | 0.2262 | 0.3048 | 0.2881 |
| DeepSeek | 0.2537 | 0.3254 | 0.3982 |
| **KGST** | **0.4538** | **0.4939** | **0.5686** |

### Key Findings
- KGST consistently achieves the best performance on both datasets, demonstrating the robustness and cross-dataset generalization ability of the framework.
- Self-Refine actually decreases performance after refinement (0.4733 → 0.4422), showing that STL refinement must rely on external knowledge reference.
- The N-gram diversity of STL-DivEn (2.386) is much higher than that of DeepSTL (1.474), and the number of sub-formulas (14.66 vs 6.98) is also significantly larger.
- In human evaluation, KGST achieves the highest accuracy of 62.4% (STL-DivEn) / 54.6% (DeepSTL).
- The natural language description vocabulary size of STL-DivEn (4,954 unique words) far exceeds that of DeepSTL (265).

## Highlights & Insights
- The dataset construction method can be generalized to data generation for other formal languages—clustering-guided augmentation is a universal strategy to guarantee diversity.
- The failure of Self-Refine reveals an important insight: formal language translation requires precise reference knowledge, as the model's internal "intuition" is not reliable enough.

## Limitations & Future Work
- Relies on GPT-4 for refinement, which is costly and limits deployment independence.
- The seed set only covers three domains (autonomous driving, robotics, and electronics), which may not be fully comprehensive.
- Utilizing the smaller LLaMA 3-8B in the generation phase; larger models might further improve the quality of initial STLs.

## Related Work & Insights
- **vs DeepSTL (He et al., 2022)**: The latter randomly samples and generates data using templates, leaving a severe lack of diversity. This work uses clustering-guided generation and human verification to ensure quality and diversity.
- **vs NL2TL (Chen et al., 2023)**: The latter fine-tunes T5 using an NL-TL dataset created by LLMs but does not utilize external knowledge-augmented refinement.
- **vs DialogueSTL (Mohammadinejad et al., 2024)**: The latter performs STL translation through user interaction and reinforcement learning, but relying on user feedback increases usage complexity.

## Rating
- Novelty: ⭐⭐⭐⭐ The Generate-then-Refine framework and clustering-guided data augmentation are innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two datasets + automatic evaluation + human evaluation are provided, though the ablation studies could be more detailed.
- Writing Quality: ⭐⭐⭐⭐ Complete structure with clear formal definitions of STL.
- Value: ⭐⭐⭐⭐ The dataset and framework practically advance the field of automatic formal specification generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Internal and External Impacts of Natural Language Processing Papers](internal_and_external_impacts_of_natural_language_processing_papers.md)
- [\[ACL 2025\] Factual Knowledge in Language Models: Robustness and Anomalies under Simple Temporal Context Variations](factual_knowledge_in_language_models_robustness_and_anomalies_under_simple_tempo.md)
- [\[ACL 2025\] Cooperating and Competing Through Natural Language](cooperating_and_competing_through_natural_language.md)
- [\[ACL 2025\] Palm: A Culturally Inclusive and Linguistically Diverse Dataset for Arabic LLMs](palm_a_culturally_inclusive_and_linguistically_diverse_dataset_for_arabic_llms.md)
- [\[ACL 2025\] Problem-Solving Logic Guided Curriculum In-Context Learning for LLMs Complex Reasoning](problem-solving_logic_guided_curriculum_in-context_learning_for_llms_complex_rea.md)

</div>

<!-- RELATED:END -->
