---
title: >-
  [Paper Note] The GaoYao Benchmark: A Comprehensive Framework for Evaluating Multilingual and Multicultural Abilities of Large Language Models
description: >-
  [ACL 2026][Multilingual & Machine Translation][multilingual benchmark] This paper presents the GaoYao benchmark, comprising 182.3K samples across 26 languages and 51 countries/regions. Through a three-tier cultural evalu…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "multilingual benchmark"
  - "multicultural evaluation"
  - "LLM evaluation"
  - "linguistic fairness"
  - "cultural understanding"
date: 2026-05-08
content_hash: 5ea796a3f84023f3
---

# The GaoYao Benchmark: A Comprehensive Framework for Evaluating Multilingual and Multicultural Abilities of Large Language Models

**Conference**: ACL 2026
**arXiv**: [2604.20225](https://arxiv.org/abs/2604.20225)
**Code**: [github.com/lunyiliu/GaoYao](https://github.com/lunyiliu/GaoYao)
**Area**: Human Understanding / Multilingual Evaluation
**Keywords**: multilingual benchmark, multicultural evaluation, LLM evaluation, linguistic fairness, cultural understanding

## TL;DR
This paper presents the GaoYao benchmark, comprising 182.3K samples across 26 languages and 51 countries/regions. Through a three-tier cultural evaluation framework (general multilingual / cross-cultural / mono-cultural) and nine cognitive sub-layers, combined with a human-localized subjective test set and an expert-validated cross-cultural synthetic dataset SuperBLEnD, GaoYao performs in-depth diagnosis of 20+ flagship and compact LLMs, revealing pronounced geographic digital divides and task-level capability stratification.

## Background & Motivation

**Background**: LLMs are serving users worldwide, and multilingual capability has become a key indicator of their inclusivity. Existing multilingual evaluation benchmarks are numerous, covering tasks such as knowledge QA, reading comprehension, and translation, but each addresses only a single dimension.

**Limitations of Prior Work**: Current benchmarks suffer from three critical limitations: (1) *Fragmented evaluation dimensions* — most benchmarks focus on a single aspect of linguistic competence (e.g., knowledge or reading comprehension), overlooking deep cultural nuances and treating multilingual ability as isolated evaluation points rather than interconnected dimensions rooted in cultural cognition; (2) *Insufficient language coverage for subjective tasks* — key tasks such as instruction following and multi-turn dialogue are evaluated primarily in English, with multilingual extensions relying on low-quality machine translation (e.g., "list words starting with the letter A" becomes meaningless when literally translated into languages without such a letter); (3) *Lack of diagnostic depth* — existing studies stop at superficial leaderboard rankings without revealing the geographic, task-type, or model-architecture factors underlying performance gaps.

**Key Challenge**: Surface-level linguistic fluency does not equate to deep cultural understanding (e.g., the symbol of the "dragon" carries vastly different meanings in Eastern and Western cultures), yet existing benchmarks primarily assess the tip of the iceberg — general linguistic competence — and cannot diagnose a model's true level of cultural sensitivity.

**Goal**: To construct a systematic, high-quality multilingual and multicultural evaluation benchmark with deep diagnostic analytical capability.

**Key Insight**: The design leverages the cultural iceberg model and Bloom's taxonomy of cognitive objectives to build a hierarchical evaluation framework. Expert localization involving 175 person-days ensures native-quality subjective test sets, while a three-stage semi-automated pipeline scales cross-cultural evaluation from 16 cultures to 34.

**Core Idea**: Multilingual evaluation is decomposed into three levels of cultural depth (general multilingual → cross-cultural → mono-cultural), combined with nine cognitive sub-layers to form an evaluation matrix. Human localization — rather than machine translation — is employed to ensure native quality in subjective tasks.

## Method

### Overall Architecture
GaoYao adopts an "integrate + extend + generalize" tri-pronged strategy for benchmark construction: (1) seven cognitive sub-layers are built by integrating existing high-quality open-source datasets (e.g., Include, MMMLU, Belebele, Flores-101, MGSM); (2) two critical subjective task sub-layers (instruction following and multi-turn dialogue) are expert-localized into 19 languages; (3) the cross-cultural evaluation layer is generalized from 16 to 34 cultures via a human–machine collaborative pipeline. Evaluation protocols include objective assessment (rule-based extraction) and subjective assessment (LLM-as-Judge), covering 20+ flagship and compact models.

### Key Designs

1. **Three-Tier Cultural Evaluation Framework + Nine Cognitive Sub-Layers**

    - **Function**: Provides systematic evaluation dimensions for multilingual and multicultural competence.
    - **Mechanism**: Inspired by the cultural iceberg model and Bloom's taxonomy, tasks are organized into three levels of cultural depth: *general multilingual competence* (e.g., reasoning, knowledge QA — universal concepts consistent across languages); *cross-cultural competence* (e.g., the culturally variant meanings of "dragon" — shared concepts with differing cultural instantiations); and *mono-cultural competence* (e.g., China's "Spring Festival travel rush" or India's "Namaste" — concepts unique to a specific culture). The nine cognitive sub-layers span memorization/comprehension (knowledge QA, reading comprehension, translation), application/analysis (reasoning, mathematics), and evaluation/creation (instruction following, multi-turn dialogue, cross-cultural, and mono-cultural assessment).
    - **Design Motivation**: Validation shows that the Spearman rank correlation from general multilingual rankings to mono-cultural rankings drops significantly ($\rho$ from 0.74 to 0.61), demonstrating that the three-tier framework reveals capability decoupling that a single aggregated score would obscure.

2. **Expert-Level Subjective Task Localization (S-AlpacaEval & S-MT-Bench)**

    - **Function**: Extends instruction-following and multi-turn dialogue evaluation from English to 19 languages with native-level quality.
    - **Mechanism**: Twenty native-speaker experts were recruited from language service centers at leading enterprises, contributing 175 person-days of localization effort. The key emphasis is on *localization* rather than translation — for instance, "list words starting with the letter A" is manually reconstructed in target languages based on their phonological and orthographic characteristics, ensuring cognitive task equivalence. A rigorous review-feedback loop was implemented, with third-party reviewers continuously auditing samples and disputes triggering a discussion phase.
    - **Design Motivation**: Machine translation has a relatively minor impact on objective tasks (e.g., multiple-choice questions) but is harmful in subjective evaluation — producing "translationese" that fails to reflect native expression. Experiments (Fig. 7) show that the GaoYao-constructed test sets better differentiate LLM capability tiers.

3. **SuperBLEnD Cross-Cultural Evaluation Set Generalization**

    - **Function**: Expands cross-cultural evaluation from 16 to 34 cultures.
    - **Mechanism**: A three-stage pipeline is employed: (1) *Cultural generalization* — high-quality templates are selected from BLEnD; native-speaker experts are recruited to provide experience-based answers for 18 new cultures, with rigorous human validation leading to the rejection of approximately 41.1% of raw data; (2) *Option synthesis* — Q&A pairs are converted into multiple-choice questions, with answers from other cultures and LLM-generated distractors serving as options; (3) *Linguistic enrichment* — LLMs rewrite question stems and options (grammatical restructuring, voice transformation) to prevent simple pattern matching. Ablation experiments show that enrichment reduces Qwen3-8B accuracy from 78.06% to 57.25% (−20.81%), confirming the removal of shortcut cues.
    - **Design Motivation**: Direct translation retains source-culture concepts, while fully manual creation is prohibitively expensive. The semi-automated pipeline balances coverage breadth and quality.

### Loss & Training
GaoYao is an evaluation benchmark rather than a training methodology. Objective tasks are assessed via rule-based extraction; subjective tasks employ DeepSeek-v3.1 as the LLM-as-Judge, with Qwen3-235B-A22B serving as the reference anchor. All scores are normalized to a 0–100 scale.

## Key Experimental Results

### Main Results (Model Ranking Shifts Across Three Tiers)

| Model | General Multilingual Rank | Cross-Cultural Rank | Mono-Cultural Rank |
|---|---|---|---|
| Gemini-2.5-Pro | #1 | #1 | #8 |
| Doubao-Seed-1.6 | #2 | #14 | #6 |
| Qwen3-235B-A22B | #9 | #11 | #1 |
| DeepSeek-V3.1 | #15 | #16 | #4 |

### Ablation Study (Effect of Linguistic Enrichment in SuperBLEnD)

| Model | Original BLEnD | SuperBLEnD | Δ |
|---|---|---|---|
| Qwen3-235B-A22B | 72.57 | 68.06 | −4.51 |
| Qwen3-8B | 78.06 | 57.25 | −20.81 |
| GPT-5-chat | 78.45 | 70.38 | −8.07 |

### Key Findings
- **Ranking Decoupling**: The Spearman correlation from general multilingual to cross-cultural rankings is 0.74, dropping to 0.61 for mono-cultural rankings. Gemini-2.5-Pro ranks first in general multilingual evaluation but falls to eighth in mono-cultural evaluation, while Qwen3-235B rises from ninth to first — underscoring the necessity of hierarchical evaluation.
- **Digital Divide**: Western European languages consistently achieve the highest scores, while South Asian and African low-resource languages lag significantly behind. Performance correlates strongly with resource level (high > medium > low).
- **Benchmark Saturation**: On established benchmarks such as Belebele, compact models approach flagship-level performance; however, significant gaps emerge on GaoYao's newly constructed subjective test sets, exposing real capability differences.
- **Thinking Mode**: For flagship models, extended reasoning provides selective gains (effective only at higher cognitive layers); for compact models, it yields universal gains (beneficial across all layers).

## Highlights & Insights
- **Cultural Hierarchical Evaluation Framework**: Decomposing "multilingual competence" into three levels of cultural depth reveals capability decoupling that a single aggregated score cannot capture. This framework is transferable to other tasks requiring multidimensional evaluation (e.g., hierarchical assessment of coding or reasoning ability).
- **Localization Over Translation**: The 175 person-days of expert localization may appear costly, but experiments demonstrate that machine translation severely distorts subjective task quality. This sets a quality standard for benchmark construction.
- **Linguistic Enrichment in SuperBLEnD**: Grammatical restructuring and voice transformation upgrade the benchmark from "knowledge retrieval" to "cultural reasoning," effectively eliminating shortcut cues. Qwen3-8B had previously "accidentally" surpassed Qwen3-235B; enrichment restores the correct capability ordering.

## Limitations & Future Work
- Vertical domains (legal, medical, financial) and agentic capabilities (tool use, API calling) are not covered.
- The manual pipeline limits scalability, making it difficult to efficiently extend coverage to hundreds of low-resource languages.
- Task and language distributions are imbalanced (e.g., MGSM covers only 10 languages; SAGE/CultureScope covers only 2 languages/cultures).
- Static benchmarks inevitably lag behind the latest models; a dynamic leaderboard is planned for future release.

## Related Work & Insights
- **vs. Include/MMMLU**: These are objective benchmarks focused on knowledge and reasoning, lacking subjective and cultural dimensions. GaoYao provides comprehensive coverage through its integrate–extend–generalize strategy.
- **vs. WMT/Flores**: These are translation-oriented benchmarks; GaoYao incorporates translation as one of nine sub-layers within a broader framework.
- **vs. BLEnD**: BLEnD covers only 16 cultures and is susceptible to pattern-matching attacks. SuperBLEnD extends coverage to 34 cultures and improves discriminative power through linguistic enrichment.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The three-tier cultural framework and expert-localized subjective test set design represent systematic innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Coverage of 20+ models and 26 languages, with comprehensive ablation and diagnostic analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Framework is clear and experiments are thorough, though slightly verbose.
- **Value**: ⭐⭐⭐⭐⭐ — Fills an important gap in multilingual subjective and cultural evaluation, with lasting community value.
- **Overall**: ⭐⭐⭐⭐⭐ — A top-tier benchmark contribution excelling in framework design, data quality, and analytical depth.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MORPHOGEN: A Multilingual Benchmark for Evaluating Gender-Aware Morphological Generation](morphogen_a_multilingual_benchmark_for_evaluating_gender-aware_morphological_gen.md)
- [\[ACL 2026\] Language Models Entangle Language and Culture](language_models_entangle_language_and_culture.md)
- [\[NeurIPS 2025\] XIFBench: Evaluating Large Language Models on Multilingual Instruction Following](../../NeurIPS2025/multilingual_mt/xifbench_evaluating_large_language_models_on_multilingual_instruction_following.md)
- [\[ACL 2026\] Exploring Two-Phase Continual Instruction Fine-tuning for Multilingual Adaptation in Large Language Models](exploring_continual_fine-tuning_for_enhancing_language_ability_in_large_language.md)
- [\[ACL 2026\] Multilingual Language Models Encode Script Over Linguistic Structure](multilingual_language_models_encode_script_over_linguistic_structure.md)

</div>

<!-- RELATED:END -->
