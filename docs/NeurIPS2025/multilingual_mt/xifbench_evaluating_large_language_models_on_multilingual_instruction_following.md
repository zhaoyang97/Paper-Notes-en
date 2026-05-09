---
title: >-
  [Paper Note] XIFBench: Evaluating Large Language Models on Multilingual Instruction Following
description: >-
  [NeurIPS 2025][multilingual instruction following] This paper proposes XIFBench — the first constraint-driven benchmark systematically evaluating LLMs' multilingual instruction-following capabilities. It comprises 558 instructions (0–5 constraints, 5 categories × 21 dimensions) across 6 languages (high/mid/low resource), and introduces an English-requirement anchoring evaluation protocol that achieves 94.7% cross-lingual evaluation consistency.
tags:
  - NeurIPS 2025
  - multilingual instruction following
  - constraint-based evaluation
  - LLM benchmarking
  - cross-lingual consistency
  - fine-grained evaluation
date: 2026-05-08
content_hash: 95dbd8ed775265cc
---

# XIFBench: Evaluating Large Language Models on Multilingual Instruction Following

**Conference**: NeurIPS 2025
**arXiv**: [2503.07539](https://arxiv.org/abs/2503.07539)
**Authors**: Zhenyu Li, Kehai Chen (HIT Shenzhen), Yunfei Long (QMUL), Xuefeng Bai, Yaoyin Zhang, Xuchen Wei, Juntao Li (Soochow Univ.), Min Zhang
**Code**: [zhenyuli801/XIFBench](https://github.com/zhenyuli801/XIFBench)
**Area**: Multilingual Translation
**Keywords**: multilingual instruction following, constraint-based evaluation, LLM benchmarking, cross-lingual consistency, fine-grained evaluation

## TL;DR

This paper proposes XIFBench — the first constraint-driven benchmark systematically evaluating LLMs' multilingual instruction-following capabilities. It comprises 558 instructions (0–5 constraints, 5 categories × 21 dimensions) across 6 languages (high/mid/low resource), and introduces an English-requirement anchoring evaluation protocol that achieves 94.7% cross-lingual evaluation consistency.

## Background & Motivation

### State of the Field
Instruction following is a core capability for aligning LLMs with human intent, yet significant performance disparities exist across languages of different resource levels in multilingual settings. Existing evaluation methods (e.g., pairwise comparison in AlpacaEval, direct scoring in MT-Bench) are too coarse-grained to reveal how intrinsic instruction factors influence cross-lingual performance.

### Limitations of Prior Work
- **Constraint evaluation limited to English/Chinese**: Benchmarks such as IFEval, FollowBench, and InfoBench primarily target high-resource languages and do not cover mid- or low-resource languages.
- **Insufficient granularity in multilingual evaluation**: M-IFEval (4 languages) and Multi-IF (8 languages) mainly cover high- and mid-resource languages, and inherit IFEval's emphasis on format/numerical constraints while neglecting semantically rich constraint types (e.g., style, situation).
- **Evaluation consistency issues**: Translating evaluation requirements into the target language may introduce translation errors, reducing cross-lingual comparability.

### Root Cause
There is a need for a fine-grained multilingual instruction-following benchmark that covers high/mid/low-resource languages with diverse constraint types, alongside a reliable cross-lingual evaluation protocol.

## Method

### Dataset Construction Pipeline

**Seed instruction preparation**: From the evaluation sets of AlpacaEval, WizardLM, and LIMA, hierarchical clustering yields 131 clusters, from which one representative instruction per cluster is selected. After manual filtering to remove ambiguous, overly difficult, or language-dependent instructions (e.g., "reply in uppercase," "use rhyming"), 106 Easy Set instructions are obtained, each annotated for cultural accessibility (culturally universal/specific).

**Constraint augmentation**: A constraint taxonomy of 5 categories and 21 dimensions is designed:
- **Content**: specifies information the response should include
- **Style**: defines tone and writing style
- **Situation**: describes contextual settings such as role or environment
- **Format**: prescribes the structural requirements of the response
- **Numerical**: involves quantitative constraints on length or count

GPT-4o is used to generate constraints of each type for every instruction; combinations of 1–5 constraints are then sampled and incorporated, forming 465 Hard Set instructions. After human verification, 93 Easy + 465 Hard = 558 instructions are retained.

**Requirement structuring**: Each instruction is decomposed into atomic binary (YES/NO) evaluation requirements (e.g., "Is each section limited to 2 sentences?"), yielding 1,664 requirements in total. Human evaluation shows that over 93.3% of requirements satisfy clarity, completeness, atomicity, and categorical accuracy.

**Multilingual extension**: The 558 English instructions are translated into Chinese, Russian, Arabic, Hindi, and Swahili, producing 3,348 instances in total. Translation quality is validated via GPT-4o plus back-translation, with a cross-lingual inconsistency rate below 1.4%.

### Evaluation Protocol

**Core innovation — English requirement anchoring**: During cross-lingual evaluation, the original English evaluation requirements are retained as semantic anchors rather than being translated into the target language. This avoids semantic drift introduced by translation and ensures cross-lingual comparability.

**Evaluation metrics**:
- **RFR (Requirement Following Rate)**: the proportion of satisfied evaluation requirements across all instructions — a fine-grained perspective.
- **IFR (Instruction Following Rate)**: the proportion of instructions for which all requirements are satisfied — a strict, holistic perspective.

$$\text{RFR}^{(l)}=\frac{\sum_{i}\sum_{r}e^{(l)}_{i,r}}{\sum_{i}|\mathcal{R}_{i}|}, \quad \text{IFR}^{(l)}=\frac{1}{|\mathcal{I}^{(l)}|}\sum_{i}\prod_{r}e^{(l)}_{i,r}$$

## Key Experimental Results

### Main Results

Using GPT-4o as judge, 3 closed-source and 6 open-source models are evaluated across 6 languages.

| Model | En RFR | Zh RFR | Sw RFR | Avg RFR | En IFR | Zh IFR | Sw IFR | Avg IFR |
|-------|--------|--------|--------|---------|--------|--------|--------|---------|
| GPT-4o | 93.6 | 92.5 | 90.8 | 92.2 | 76.9 | 73.3 | 65.6 | 72.2 |
| Gemini-2.0-Flash | 93.3 | 93.0 | 89.5 | 92.1 | 78.1 | 76.7 | 69.2 | 74.7 |
| Claude-3.5-Sonnet | 89.1 | 81.3 | 74.5 | 80.2 | 66.1 | 53.0 | 40.1 | 51.8 |
| Llama-3.1-70B | 91.7 | 83.4 | 73.4 | 82.2 | 70.9 | 48.9 | 34.8 | 49.3 |
| Qwen-2.5-72B | 90.5 | 89.1 | 40.9 | 79.6 | 67.7 | 63.3 | 10.4 | 52.5 |
| Qwen-2.5-7B | 87.8 | 87.4 | 10.0 | 67.6 | 59.9 | 57.3 | 1.1 | 38.8 |
| Llama-3.1-8B | 87.6 | 79.1 | 38.6 | 69.1 | 58.9 | 42.8 | 9.7 | 34.8 |

**Key findings**: (1) Performance correlates strongly with language resource level; IFR for low-resource languages (Swahili) can drop to near zero. (2) The RFR–IFR gap is largest for low-resource languages — models can satisfy individual constraints but struggle to fully follow an entire instruction. (3) Closed-source models exhibit substantially greater cross-lingual robustness than open-source counterparts.

### Ablation Study: Evaluation Protocol Consistency

Three evaluation methods are compared against human annotations for agreement rate.

| Evaluation Method | En | Zh | Ru | Ar | Hi | Sw | Avg | Std |
|-------------------|-----|-----|-----|-----|-----|-----|------|-----|
| Direct Scoring | 71.7 | 56.7 | 53.9 | 55.0 | 58.3 | 69.4 | 60.7 | 6.5 |
| Translated Requirements | 93.7 | 89.1 | 89.1 | 86.7 | 84.6 | 84.9 | 88.5 | 3.1 |
| **English Requirement Anchoring (Ours)** | **95.9** | **96.7** | **95.0** | **93.0** | **95.5** | **92.5** | **94.7** | **1.6** |

The English requirement anchoring protocol achieves the highest agreement rate across all languages (94.7%) with the lowest standard deviation (1.6), validating its reliability and consistency for cross-lingual evaluation.

### Constraint Category Analysis
- **Format and numerical constraints** exhibit high cross-lingual robustness, relying on universal linguistic properties.
- **Style and situation constraints** are most sensitive to language resource level, with the most severe degradation in low-resource languages.
- **Content constraints** fall in between, showing moderate degradation.

### Instruction Complexity Analysis
- IFR decreases approximately linearly as the number of constraints increases.
- The degradation rate shows no clear correlation with language resource level.
- High-capability models (e.g., Gemini-2.0-Flash) exhibit smoother degradation curves.

## Highlights & Insights

- **Systematic coverage**: The first multilingual instruction-following benchmark to simultaneously span high/mid/low-resource languages (6 languages, 3 resource tiers) and diverse constraint types (5 categories, 21 dimensions).
- **English requirement anchoring**: Using shared English requirements as a semantic anchor avoids translation errors, achieving 94.7% cross-lingual evaluation consistency versus 88.5% for translated requirements.
- **Multi-dimensional insights**: The work systematically analyzes the effects of language resource level, constraint category, instruction complexity, and cultural specificity on multilingual instruction following, providing fine-grained understanding previously absent from the literature.
- **High-quality dataset**: With constraint-level translation validation and human quality control, the cross-lingual inconsistency rate is below 1.4% and requirement quality exceeds 93.3%.

## Limitations & Future Work

- **Limited language coverage**: Only 6 languages are included; important languages such as Korean, Japanese, and Portuguese are absent.
- **Reliance on GPT-4o as judge**: The evaluation protocol depends on a specific LLM as an evaluator, whose biases may affect results.
- **Language-dependent constraints excluded**: Constraints such as capitalization, rhyming, and word count are excluded for cross-lingual applicability, yet these remain practically important.
- **Machine-translated source**: Although validation consistency is high, translation quality for extremely low-resource languages (e.g., Swahili) may still carry latent biases.
- **Static benchmark**: Multi-turn dialogue and continuous instruction-following capabilities are not addressed.
- **Small Easy Set**: With only 93 samples, statistical reliability for evaluating instruction following without constraints is limited.

## Related Work & Insights

- **IFEval (Zhou et al. 2023)**: English only, focused on verifiable format/numerical constraints; this paper covers 6 languages plus semantically rich constraints such as style and situation.
- **FollowBench (Jiang et al. 2024)**: Extends to content/situation/style constraints but remains English only; this paper inherits its constraint richness and extends it to a multilingual setting.
- **InfoBench (Qin et al. 2024)**: Proposes requirement-list-based evaluation; this paper adapts the approach to multilingual scenarios and refines it into the English anchoring protocol.
- **M-IFEval / Multi-IF**: Extend IFEval to multiple languages but are limited to high- and mid-resource languages with format-oriented constraints; this paper covers low-resource languages with a more comprehensive constraint taxonomy.
- **Aya Evaluation Suite**: Covers 101 languages but relies on coarse-grained direct scoring; the constraint-level fine-grained evaluation in this paper offers greater diagnostic value.

## Rating

- Novelty: ⭐⭐⭐⭐ — First benchmark to systematically cover resource-level gradients and semantic constraints in multilingual instruction following; the English anchoring protocol is genuinely innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 9 models × 6 languages, multi-dimensional ablation analysis, thorough consistency validation.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, rich figures and tables, detailed method description.
- Value: ⭐⭐⭐⭐ — Fills the gap in fine-grained multilingual instruction-following evaluation; insights are a valuable reference for improving multilingual LLM capabilities.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Exploring the Translation Mechanism of Large Language Models](exploring_the_translation_mechanism_of_large_language_models.md)
- [\[AAAI 2026\] Focusing on Language: Revealing and Exploiting Language Attention Heads in Multilingual Large Language Models](../../AAAI2026/multilingual_mt/focusing_on_language_revealing_and_exploiting_language_attention_heads_in_multil.md)
- [\[NeurIPS 2025\] ParallelPrompt: Extracting Parallelism from Large Language Model Queries](parallelprompt_extracting_parallelism_from_large_language_model_queries.md)
- [\[AAAI 2026\] MIDB: Multilingual Instruction Data Booster for Enhancing Cultural Equality in Multilingual Instruction Synthesis](../../AAAI2026/multilingual_mt/midb_multilingual_instruction_data_booster_for_enhancing_cultural_equality_in_mu.md)
- [\[AAAI 2026\] Consensus-Aligned Neuron Efficient Fine-Tuning Large Language Models for Multi-Domain Machine Translation](../../AAAI2026/multilingual_mt/consensus-aligned_neuron_efficient_fine-tuning_large_language_models_for_multi-d.md)

<!-- RELATED:END -->
