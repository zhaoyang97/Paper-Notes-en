---
title: >-
  [Paper Note] A Multilingual Dataset and Empirical Validation for the Mutual Reinforcement Effect in Information Extraction
description: >-
  [ACL 2026][Multilingual & Machine Translation][Mutual Reinforcement Effect] This work constructs the first multilingual MRE Mix dataset (MMM), covering 21 subsets across English, Chinese…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Mutual Reinforcement Effect"
  - "Multilingual Information Extraction"
  - "Word-level & Text-level Joint Modeling"
  - "Dataset Construction"
  - "LLM-assisted Translation"
date: 2026-05-08
content_hash: f5c51d186a5935d7
---

# A Multilingual Dataset and Empirical Validation for the Mutual Reinforcement Effect in Information Extraction

**Conference**: ACL 2026 Findings  
**arXiv**: [2407.10953](https://arxiv.org/abs/2407.10953)  
**Code**: [GitHub/HuggingFace](https://ganchengguang.github.io/MRE/)  
**Area**: Information Extraction / Multilingual NLP  
**Keywords**: Mutual Reinforcement Effect, Multilingual Information Extraction, Word-level & Text-level Joint Modeling, Dataset Construction, LLM-assisted Translation

## TL;DR

This work constructs the first multilingual MRE Mix dataset (MMM), covering 21 subsets across English, Chinese, and Japanese. Through large-scale ablation experiments, it systematically validates that the Mutual Reinforcement Effect (MRE) between word-level and text-level information extraction tasks is a cross-linguistically universal phenomenon.

## Background & Motivation

**Background**: Information Extraction (IE) encompasses multiple subtasks such as Named Entity Recognition, Relation Extraction, and Sentiment Analysis. Traditional approaches typically model these independently. While multi-task learning shares representations, it does not explicitly model the semantic interactions between tasks.

**Limitations of Prior Work**: The Mutual Reinforcement Effect (MRE)—where word-level and text-level IE tasks enhance each other during joint modeling—has previously only been validated in Japanese. The lack of multilingual MRE datasets significantly hinders cross-lingual validation and broader applications.

**Key Challenge**: Is MRE a language-specific phenomenon or a universal cross-lingual mechanism? This fundamental question cannot be answered due to the absence of appropriate data.

**Goal**: To construct a multilingual MRE dataset and systematically verify the universality of MRE across different languages and task combinations.

**Key Insight**: An LLM-assisted dataset translation and alignment framework is proposed to extend the Japanese MRE dataset to English and Chinese, while simultaneously constructing new open-domain datasets.

**Core Idea**: MRE is not a language-specific artifact; instead, it is a universal mechanism of bidirectional dependency between fine-grained word-level semantics and global text-level semantics in IE tasks.

## Method

### Overall Architecture

The work consists of three parts: (1) Construction of the MMM dataset—extending the Japanese MRE dataset into a multilingual version via an LLM-assisted translation framework; (2) Design of OIELLM—an open-domain information extraction model with a unified input/output format; (3) Execution of systematic ablation experiments to verify the cross-lingual effectiveness of MRE.

### Key Designs

1.  **LLM-assisted Dataset Translation Framework**:
    *   **Function**: Efficiently translates the Japanese MRE dataset into English and Chinese while maintaining annotation consistency.
    *   **Mechanism**: Uses rule matching for deterministic translation of fixed label sets, followed by GPT-3.5-Turbo for auxiliary translation of free-text components. Quality is ensured through two-stage rule filtering (removing untranslated characters, aligning entity spans) and manual verification.
    *   **Design Motivation**: Fixed label sets allow deterministic mapping to eliminate ambiguity; LLMs are utilized to reduce repetitive labor rather than replace human involvement, with humans retained for critical quality control.

2.  **Unified Input-Output OIELLM Model**:
    *   **Function**: Jointly generates text-level labels and word-level label-entity pairs in a single decoding process.
    *   **Mechanism**: Inputs consist of the original text plus task instruction words (marked with a "/" prefix). Outputs follow a fixed format where text-level labels are provided first, followed by word-level extraction results. Separation characters ":" and ";" ensure parsing consistency across tasks and languages.
    *   **Design Motivation**: This avoids the length overhead and prompt-induced bias of conversational prompts, allowing the model to focus on learning the structural dependencies between text-level and word-level information.

3.  **Knowledgeable Verbalizer Extension**:
    *   **Function**: Injects word-level supervision signals from MRE data into text-level classifiers.
    *   **Mechanism**: Word-level annotations in MRE Mix data are used to construct knowledge-enhanced verbalizers, strengthening the representation of label tokens in prompt-based text classification.
    *   **Design Motivation**: This validates MRE from an alternative perspective—if word-level information indeed aids text-level tasks, then explicit injection of this data should improve classification performance.

### Loss & Training

OIELLM is fully fine-tuned based on open-source LLMs using standard autoregressive language modeling objectives. The training data covers all 21 subsets of MMM.

## Key Experimental Results

### Main Results

| Model | SCNM TL | SCNM WL | SCNM ALL |
| :--- | :--- | :--- | :--- |
| GPT-4o | 58.30 | 23.42 | 8.57 |
| OIELLM-8B | 84.73 | 88.53 | 61.93 |
| OIELLM-8B* | 87.30 | 89.28 | 64.00 |
| OIELLM-13B | 89.00 | 86.33 | 57.70 |

### Ablation Study

| Configuration | Key Metrics | Description |
| :--- | :--- | :--- |
| MRE Presence Rate | 76% | 16 out of 21 subsets exhibit significant MRE |
| Cross-lingual Consistency | Effective in EN/ZH/JA | MRE is not a language-specific phenomenon |
| Verbalizer Gain | Positive | Word-level supervision injection improves text-level classification |

### Key Findings
*   76% of the MMM sub-datasets show stable mutual reinforcement effects in ablation studies, proving that MRE is a cross-lingual universal mechanism.
*   OIELLM significantly outperforms zero-shot LLMs (GPT-3.5, GPT-4o) in a joint training setting, demonstrating the practical value of MRE.
*   Injecting word-level information into Knowledgeable Verbalizers leads to consistent improvements in text-level classification.

## Highlights & Insights
*   The "Point-Line" abstraction elegantly unifies the relationship between word-level and text-level IE tasks—word-level acts as points while text-level acts as the line, providing mutual constraints.
*   The LLM-assisted translation framework is practical: it combines deterministic mapping, LLM translation, rule filtering, and manual checking, with a clear division of labor for each step.
*   The experimental design is thorough—it not only proves the existence of MRE but also demonstrates its actionable application value through Verbalizer experiments.

## Limitations & Future Work
*   Currently only covers English, Chinese, and Japanese; the effectiveness of MRE in low-resource languages remains unverified.
*   The translation framework still requires 10 multilingual graduate students for manual verification, incurring significant costs for scaling.
*   The theoretical explanation of MRE (why word-level and text-level reinforce each other) is still insufficient.
*   Future work could extend to more languages and a wider variety of IE task combinations.

## Related Work & Insights
*   **vs Traditional Multi-task IE**: Beyond just sharing representations, this work explicitly models and validates bidirectional enhancement between tasks.
*   **vs Unified IE models**: Focuses on empirical validation of the MRE phenomenon rather than model architecture innovation.
*   **vs LLM Zero-shot IE**: Fine-tuned OIELLM significantly outperforms GPT-4o zero-shot, indicating that task-specific training remains essential.

## Rating
*   Novelty: ⭐⭐⭐⭐ First multilingual MRE dataset and systematic cross-lingual validation.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive ablation of 21 subsets and comparisons across multiple models.
*   Writing Quality: ⭐⭐⭐⭐ Clear structure with a vivid "Point-Line" abstraction.
*   Value: ⭐⭐⭐⭐ Provides a critical data resource and empirical foundation for multilingual IE.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] NeoAMT: Neologism-Aware Agentic Machine Translation with Reinforcement Learning](neoamt_neologism-aware_agentic_machine_translation_with_reinforcement_learning.md)
- [\[ACL 2026\] XQ-MEval: A Dataset with Cross-lingual Parallel Quality for Benchmarking Translation Metrics](xq-meval_a_dataset_with_cross-lingual_parallel_quality_for_benchmarking_translat.md)
- [\[ACL 2026\] Alexandria: A Multi-Domain Dialectal Arabic Machine Translation Dataset for Culturally Inclusive and Linguistically Diverse LLMs](alexandria_a_multi-domain_dialectal_arabic_machine_translation_dataset_for_cultu.md)
- [\[ACL 2026\] Reinforcement Learning with Semantic Rewards Enables Low-Resource Language Expansion without Alignment Tax](reinforcement_learning_with_semantic_rewards_enables_low-resource_language_expan.md)
- [\[ACL 2026\] TransLaw: A Large-Scale Dataset and Multi-Agent Benchmark Simulating Professional Translation of Hong Kong Case Law](translaw_a_large-scale_dataset_and_multi-agent_benchmark_simulating_professional.md)

</div>

<!-- RELATED:END -->
