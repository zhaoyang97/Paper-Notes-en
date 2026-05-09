---
title: >-
  [Paper Note] Capabilities and Evaluation Biases of Large Language Models in Classical Chinese Poetry Generation: A Case Study on Tang Poetry
description: >-
  [ACL 2026][LLM Evaluation][Classical poetry generation] This paper proposes a three-step evaluation framework (computational feature extraction + LLM-as-Judge + human expert validation) to systematically assess the Tang poetry generation capabilities of six LLMs. A critical "echo chamber" effect is identified: LLMs systematically overrate machine-generated poems that mimic statistical patterns while violating prosodic rules, diverging significantly from human expert judgments.
tags:
  - ACL 2026
  - LLM Evaluation
  - Classical poetry generation
  - Tang poetry
  - LLM evaluation bias
  - echo chamber effect
  - human-machine evaluation
date: 2026-05-08
content_hash: 07b783fef21b3acf
---

# Capabilities and Evaluation Biases of Large Language Models in Classical Chinese Poetry Generation: A Case Study on Tang Poetry

**Conference**: ACL 2026
**arXiv**: [2510.15313](https://arxiv.org/abs/2510.15313)
**Code**: [https://github.com/boleima/Tang-Poetry](https://github.com/boleima/Tang-Poetry)
**Area**: LLM Evaluation
**Keywords**: Classical poetry generation, Tang poetry, LLM evaluation bias, echo chamber effect, human-machine evaluation

## TL;DR

This paper proposes a three-step evaluation framework (computational feature extraction + LLM-as-Judge + human expert validation) to systematically assess the Tang poetry generation capabilities of six LLMs. A critical "echo chamber" effect is identified: LLMs systematically overrate machine-generated poems that mimic statistical patterns while violating prosodic rules, diverging significantly from human expert judgments.

## Background & Motivation

**Background**: LLMs have demonstrated impressive capabilities in text generation, including creative writing. Classical Chinese poetry—Tang poetry in particular—poses an extreme challenge for AI creativity due to its strict prosodic and tonal constraints and profound cultural depth.

**Limitations of Prior Work**: (1) LLMs still frequently produce incoherence across lines, lack originality in imagery, or reproduce memorized verses; (2) conventional automatic metrics (BLEU, ROUGE) fail to capture prosody, imagery, and aesthetic value; (3) LLM-as-Judge approaches may exhibit systematic biases—models may inflate scores for their own outputs or converge toward peer models.

**Key Challenge**: Poetry generation requires simultaneously satisfying structural correctness and aesthetic quality, yet current automatic evaluation methods cannot reliably measure either dimension, especially in culturally sensitive creative tasks.

**Goal**: To establish a systematic study of LLM-based Tang poetry generation and evaluation, revealing the capability boundaries of LLMs in poetry generation and the biases present in their evaluation.

**Key Insight**: Tang poetry is used as a testbed. Generation tasks are designed across five dimensions (genre, poet style, theme, emotion, and imagery), and a three-step framework is employed to provide multi-level evaluation.

**Core Idea**: LLM-generated poems may approximate human works in surface-level statistical features while exhibiting systematic deficiencies in strict prosodic compliance—deficiencies that LLM evaluators fail to detect, giving rise to an "echo chamber."

## Method

### Overall Architecture

(1) **Large-scale generation**—six LLMs each generate approximately 2,500 poems (15,000 total), covering five poetic dimensions; (2) **Three-step evaluation**—Step 1: automated computational feature extraction (prosodic compliance rate, etc.); Step 2: LLM cross-evaluation (each model evaluates outputs from the other models); Step 3: human expert validation (specialists in classical Chinese literature).

### Key Designs

1. **Multi-dimensional Poetry Generation Design**:

    - Function: Systematically covers all key dimensions of Tang poetry composition.
    - Mechanism: Five dimensions are defined—genre (five- or seven-character jueju/lüshi), poet style (Li Bai / Du Fu / Bai Juyi / Wang Wei / Li Shangyin), theme (landscape / homesickness / historical reflection / pastoral / farewell), emotion (sorrow / serenity / boldness / romance / joy), and imagery (wind / flowers / willows / moon / wild geese). Dimensions are specified via explicit prompts at temperature $T=0.4$.
    - Design Motivation: A controlled experimental design ensures scientifically valid comparisons across models and dimensions.

2. **Computational Feature Extraction (Step 1)**:

    - Function: Objectively quantifies prosodic compliance in generated poems.
    - Mechanism: Automatic detection of adherence to prosodic rules including tonal patterns (平仄), antithesis (对仗), and rhyme schemes, yielding a prosodic compliance rate—the most objectively quantifiable dimension in Tang poetry evaluation.
    - Design Motivation: Prosodic rules constitute hard constraints in Tang poetry; poems that violate them are professionally substandard, yet LLM evaluators may overlook such violations.

3. **LLM Cross-Evaluation and Human Expert Validation (Steps 2 & 3)**:

    - Function: Reveals discrepancies between automatic evaluation and human judgment.
    - Mechanism: In Step 2, each LLM evaluates poems generated by the other models across multiple dimensions including thematic relevance, emotional consistency, imagery/structure, and linguistic authenticity. In Step 3, classical literature experts independently evaluate the same samples. Comparing the two sets of judgments exposes the "echo chamber" effect.
    - Design Motivation: The reliability of LLM-as-Judge is a pressing open question; the poetry domain provides a unique test scenario involving cultural sensitivity and strict formal constraints.

### Loss & Training

No model training is involved. Generation uses temperature $T=0.4$; evaluation is conducted in a zero-shot setting.

## Key Experimental Results

### Main Results

**Performance Tiers Among Six LLMs**

- Tier 1: Qwen2.5-7B-Instruct (highest prosodic compliance rate; best overall quality)
- Tier 2: GLM-4-9B-Chat, DeepSeek-V2-Lite-Chat
- Tier 3: Baichuan2-7B-Chat, Gemma-2-9B-it, Mistral-7B (weaker Chinese poetry capabilities)

### Ablation Study

**"Echo Chamber" Effect**: LLM evaluators systematically assign high scores to machine-generated poems, even when those poems violate strict prosodic rules. Human experts, by contrast, accurately identify prosodic violations and penalize scores accordingly. A tendency toward self-favoring scores is also observed between self-evaluation and cross-evaluation among LLMs.

### Key Findings

- Models with strong Chinese language capabilities (Qwen, GLM, DeepSeek) significantly outperform primarily English-oriented models in Tang poetry generation.
- LLM evaluators tend to overrate poems that mimic statistical patterns while violating prosodic rules—the "echo chamber" effect.
- Prosodic compliance rate is the most discriminative quality indicator, yet is precisely the dimension most readily overlooked by LLM evaluators.
- Generation difficulty varies across dimensions; stylistic imitation proves easier than prosodic compliance.

## Highlights & Insights

- This is the first systematic study of the "echo chamber" effect in LLM-based classical Chinese poetry generation and evaluation.
- The three-step evaluation framework is generalizable to other culturally sensitive creative generation tasks.
- The findings raise a warning regarding the reliability of LLM-as-Judge approaches, particularly for evaluations requiring specialized domain knowledge.
- The dataset and code are publicly available, ensuring strong reproducibility.

## Limitations & Future Work

- Only six open-source models are evaluated; commercial closed-source models are not included.
- Human evaluation is limited in scale due to expert availability constraints.
- The study focuses solely on Tang poetry and does not extend to other poetic forms.
- Future work may explore prosody-aware fine-tuning strategies to improve LLM poetry generation capabilities.

## Related Work & Insights

- The findings provide concrete domain-specific validation of bias phenomena reported in the broader LLM-as-Judge literature (e.g., Clark et al.).
- The work contributes an important creative generation benchmark to the Chinese NLP community.
- The automated prosodic compliance detection method is transferable to other text generation tasks with strict formal constraints.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic study of the echo chamber effect in LLM-based Tang poetry generation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive design spanning 6 models × 5 dimensions × 3-step evaluation.
- Writing Quality: ⭐⭐⭐⭐ Rigorous research design with clear figures and tables.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Attribution, Citation, and Quotation: A Survey of Evidence-based Text Generation with Large Language Models](attribution_citation_and_quotation_a_survey_of_evidence-based_text_generation_wi.md)
- [\[ACL 2026\] Closing the Modality Reasoning Gap for Speech Large Language Models](closing_the_modality_reasoning_gap_for_speech_large_language_models.md)
- [\[ACL 2026\] Exploring the Capability Boundaries of LLMs in Mastering of Chinese Chouxiang Language](exploring_the_capability_boundaries_of_llms_in_mastering_of_chinese_chouxiang_la.md)
- [\[ACL 2026\] E2EDev: Benchmarking Large Language Models in End-to-End Software Development Task](e2edev_benchmarking_large_language_models_in_end-to-end_software_development_tas.md)
- [\[ACL 2026\] AnchorMem: Anchored Facts with Associative Contexts for Building Memory in Large Language Models](anchormem_anchored_facts_with_associative_contexts_for_building_memory_in_large_.md)

<!-- RELATED:END -->
