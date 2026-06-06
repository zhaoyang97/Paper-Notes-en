---
title: >-
  [Paper Note] The GaoYao Benchmark: A Comprehensive Framework for Evaluating Multilingual and Multicultural Abilities of Large Language Models
description: >-
  [ACL 2026][Multilingual & Machine Translation][Multilingual benchmark] This paper proposes the GaoYao benchmark, which contains 182.3K samples across 26 languages and 51 countries/regions. Through a three-tier cultural e…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Multilingual benchmark"
  - "multicultural evaluation"
  - "LLM evaluation"
  - "linguistic fairness"
  - "cultural understanding"
date: 2026-05-08
content_hash: 4fcc13d246ba3c15
---

# The GaoYao Benchmark: A Comprehensive Framework for Evaluating Multilingual and Multicultural Abilities of Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2604.20225](https://arxiv.org/abs/2604.20225)  
**Code**: [github.com/lunyiliu/GaoYao](https://github.com/lunyiliu/GaoYao)  
**Area**: Human Understanding / Multilingual Evaluation  
**Keywords**: Multilingual benchmark, multicultural evaluation, LLM evaluation, linguistic fairness, cultural understanding

## TL;DR
This paper proposes the GaoYao benchmark, which contains 182.3K samples across 26 languages and 51 countries/regions. Through a three-tier cultural evaluation framework (General Multilingual / Cross-cultural / Mono-cultural) and nine cognitive sub-layers, combined with manually localized subjective test sets and the expert-verified cross-cultural synthetic dataset SuperBLEnD, it deeply diagnoses the multilingual capabilities of 20+ flagship and compact LLMs, revealing significant geo-digital divides and task capacity stratification.

## Background & Motivation

**Background**: LLMs are serving global users, and multilingual capability has become a key metric for measuring their inclusiveness. Numerous existing multilingual evaluation benchmarks cover tasks such as knowledge QA, reading comprehension, and translation, but each covers only a single aspect.

**Limitations of Prior Work**: Current benchmarks face three key limitations: (1) Fragmented evaluation dimensions—most focus on a single aspect of linguistic ability (e.g., knowledge or reading comprehension), ignoring deep cultural nuances and treating multilingualism as isolated evaluation points rather than interconnected dimensions rooted in cultural cognition; (2) Insufficient language coverage for subjective tasks—key tasks like instruction following and multi-turn dialogue are primarily evaluated in English, and multilingual extensions rely on low-quality machine translation (e.g., "List words starting with letter A" becomes meaningless when translated literally into languages without the letter A); (3) Lack of diagnostic depth—existing research stops at superficial leaderboard rankings without revealing the correlations with geography, task types, or model architectures behind performance differences.

**Key Challenge**: Superficial linguistic fluency does not equate to deep cultural understanding (e.g., the meaning of "dragon" differs significantly between Eastern and Western cultures). However, existing benchmarks mainly evaluate the "tip of the iceberg" of general linguistic ability and cannot diagnose a model's true level of cultural sensitivity.

**Goal**: Construct a systematic, high-quality multilingual and multicultural evaluation benchmark with deep diagnostic capabilities.

**Key Insight**: Design a hierarchical evaluation framework based on the Cultural Iceberg Model and Bloom’s Taxonomy; ensure the native quality of subjective test sets through 175 person-days of expert localization; and generalize cross-cultural evaluation from 16 cultures to 34 via a three-stage semi-automated process.

**Core Idea**: Divide multilingual evaluation into three levels of cultural depth (General Multilingual → Cross-cultural → Mono-cultural), combined with nine cognitive sub-layers to form an evaluation matrix, ensuring the native quality of subjective tasks through manual localization rather than machine translation.

## Method

### Overall Architecture
GaoYao employs a "Integration + Extension + Generalization" tripartite strategy to construct the benchmark: it integrates existing high-quality open-source datasets (e.g., Include, MMMLU, Belebele, Flores-101, MGSM) for seven cognitive sub-layers; performs expert-level linguistic extension to 19 languages for two key subjective task sub-layers (instruction following and multi-turn dialogue); and generalizes the cross-cultural evaluation layer from 16 to 34 cultures through a human-in-the-loop semi-automated process. The evaluation protocol is divided into objective evaluation (rule extraction) and subjective evaluation (LLM-as-Judge), covering 20+ flagship and compact models.

### Key Designs

1. **Three-tier Cultural Evaluation Framework + Nine Cognitive Sub-layers**:

    - **Function**: Provide systematic evaluation dimensions for multilingual and multicultural capabilities.
    - **Mechanism**: Inspired by the Cultural Iceberg Model and Bloom’s Taxonomy, tasks are divided into three cultural depth layers: General Multilingual Capabilities (e.g., reasoning, knowledge QA—universal concepts consistent across languages), Cross-cultural Capabilities (e.g., cultural differences in the meaning of "dragon"—shared concepts with different cultural variants), and Mono-cultural Capabilities (e.g., "Chunyun" in China, "Namaste" in India—concepts unique to specific cultures). The nine cognitive sub-layers range from memory/understanding (knowledge QA, reading comprehension, translation) to application/analysis (reasoning, math) and evaluation/creation (instruction following, multi-turn dialogue, cross-cultural, and mono-cultural evaluation).
    - **Design Motivation**: Verification shows that the correlation of rankings from general multilingual to mono-cultural tasks drops significantly (Spearman $\rho$ from 0.74 to 0.61), proving the three-tier framework reveals capability decoupling masked by a single score.

2. **Expert-level Subjective Task Localization (S-AlpacaEval & S-MT-Bench)**:

    - **Function**: Extend instruction following and multi-turn dialogue evaluation from English to 19 languages, ensuring native-level quality.
    - **Mechanism**: Twenty native experts were recruited from language service centers of top enterprises, contributing 175 person-days for localization. The key is localization adaptation rather than simple translation—for example, "List words starting with letter A" is manually reconstructed according to the phonetic and writing characteristics of the target language to ensure cognitive task equivalence. A rigorous audit-feedback loop was implemented, where third-party reviewers continuously checked samples, and disputes triggered a discussion phase.
    - **Design Motivation**: Machine translation has a smaller impact on objective tasks (e.g., true/false questions) but is harmful in subjective evaluations—it produces "translationese" and fails to reflect native expressions. Experiments show (Fig. 7) that the test set constructed by GaoYao better distinguishes the capability tiers of LLMs.

3. **SuperBLEnD Cross-Cultural Evaluation Set Generalization**:

    - **Function**: Expand cross-cultural evaluation from 16 to 34 cultures.
    - **Mechanism**: A three-stage process: (1) Cultural Generalization: High-quality templates were filtered from BLEnD, and native experts were recruited to provide answers based on lived experience for 18 new cultures. Approximately 41.1% of raw data was discarded after strict manual verification. (2) Option Synthesis: Q&A pairs were converted into multiple-choice questions, using answers from other cultures and LLM-generated distractors as options. (3) Linguistic Enrichment: LLMs rewrote stems and options (syntactic restructuring, voice transformation) to prevent simple pattern matching. Ablation studies show that after enrichment, the accuracy of Qwen3-8B dropped from 78.06% to 57.25% (-20.81%), proving the removal of shortcuts.
    - **Design Motivation**: Direct translation retains source cultural concepts, and manual creation is costly. This semi-automated process balances coverage and quality.

### Loss & Training
GaoYao is an evaluation benchmark rather than a training method. Objective tasks are evaluated using rule extraction, while subjective tasks use DeepSeek-v3.1 as the LLM-as-Judge, with Qwen3-235B-A22B as the reference anchor. All scores are normalized to 0-100.

## Key Experimental Results

### Main Results (Model Ranking Changes Across Three Tiers)

| Model | General Multilingual Rank | Cross-cultural Rank | Mono-cultural Rank |
|------|---------------|-----------|-----------|
| Gemini-2.5-Pro | #1 | #1 | #8 |
| Doubao-Seed-1.6 | #2 | #14 | #6 |
| Qwen3-235B-A22B | #9 | #11 | #1 |
| DeepSeek-V3.1 | #15 | #16 | #4 |

### Ablation Study (Effect of Linguistic Enrichment in SuperBLEnD)

| Model | Original BLEnD | SuperBLEnD | Δ |
|------|-----------|------------|---|
| Qwen3-235B-A22B | 72.57 | 68.06 | -4.51 |
| Qwen3-8B | 78.06 | 57.25 | -20.81 |
| GPT-5-chat | 78.45 | 70.38 | -8.07 |

### Key Findings
- **Ranking Decoupling**: The Spearman correlation from general multilingual to cross-cultural tasks is 0.74, while to mono-cultural tasks, it is only 0.61. Gemini-2.5-Pro ranks first in general multilingual tasks but drops to eighth in mono-cultural tasks, while Qwen3-235B rises from ninth to first—emphasizing the necessity of tiered evaluation.
- **Digital Divide**: Western European languages consistently score highest, while South Asian and African low-resource languages significantly trail behind. Performance is strongly correlated with resource levels (High > Medium > Low).
- **Benchmark Saturation**: Compact models approach flagship levels on mature benchmarks like Belebele, but the gap is significant on GaoYao's newly constructed subjective test sets, exposing the true capability gap.
- **Thinking Patterns**: For flagship models, there is a selective gain (effective only at high cognitive layers), while for compact models, there is a universal gain (helpful across all levels).

## Highlights & Insights
- **Cultural Tiered Evaluation Framework**: Deconstructing "multilingual capability" into three cultural depth levels reveals capability decoupling that a single score cannot represent. This framework can be migrated to other tasks requiring multi-dimensional evaluation (e.g., tiered evaluation of coding or reasoning capabilities).
- **Localization Over Translation**: 175 person-days of expert localization may seem "expensive," but experiments prove that machine translation severely distorts subjective tasks. This sets a quality benchmark for the construction of evaluation suites.
- **Linguistic Enrichment in SuperBLEnD**: Upgrading the benchmark from "knowledge retrieval" to "cultural reasoning" through syntactic restructuring and voice transformation effectively removed shortcuts. Qwen3-8B, which originally "accidentally" outperformed Qwen3-235B, restored the correct capability hierarchy after enrichment.

## Limitations & Future Work
- Does not cover vertical domains (Law, Medical, Finance) or Agent capabilities (tool use, API calling).
- Manual processes limit scalability, making it difficult to efficiently extend to hundreds of low-resource languages.
- Imbalances exist in task and language distribution (e.g., MGSM only covers 10 languages, SAGE/CultureScope only cover 2 languages/cultures).
- Static benchmarks inevitably lag behind the latest models; a dynamic leaderboard is planned for the future.

## Related Work & Insights
- **vs Include/MMMLU**: These are objective benchmarks focusing on knowledge and reasoning, lacking subjective and cultural dimensions. GaoYao provides comprehensive coverage through integration, extension, and generalization.
- **vs WMT/Flores**: Translation-oriented; GaoYao incorporates translation as one of nine sub-layers into a larger framework.
- **vs BLEnD**: Only covers 16 cultures and is easily exploited by pattern matching. SuperBLEnD extends to 34 cultures and improves discriminative power through linguistic enrichment.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The three-tier cultural framework and the design of expert-localized subjective test sets represent systematic innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 20+ models, 26 languages, comprehensive ablation, and diagnostic analysis provide sufficient evidence.
- **Writing Quality**: ⭐⭐⭐⭐ The framework is clear and experiments are detailed, though slightly lengthy.
- **Value**: ⭐⭐⭐⭐⭐ Fills a critical gap in multilingual subjective and cultural evaluation, offering sustained value to the community.
- **Overall**: ⭐⭐⭐⭐⭐ A top-tier benchmark study with excellent framework design, data quality, and diagnostic depth.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Evaluating Robustness of Large Language Models Against Multilingual Typographical Errors](evaluating_robustness_of_large_language_models_against_multilingual_typographica.md)
- [\[ACL 2026\] LaoBench: A Large-Scale Multidimensional Lao Benchmark for Large Language Models](laobench_a_large-scale_multidimensional_lao_benchmark_for_large_language_models.md)
- [\[ACL 2026\] MORPHOGEN: A Multilingual Benchmark for Evaluating Gender-Aware Morphological Generation](morphogen_a_multilingual_benchmark_for_evaluating_gender-aware_morphological_gen.md)
- [\[ACL 2026\] Language Models Entangle Language and Culture](language_models_entangle_language_and_culture.md)
- [\[ACL 2026\] LLM-XTM: Enhancing Cross-Lingual Topic Models with Large Language Models](llm-xtm_enhancing_cross-lingual_topic_models_with_large_language_models.md)

</div>

<!-- RELATED:END -->
