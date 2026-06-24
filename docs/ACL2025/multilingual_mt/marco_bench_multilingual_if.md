---
title: >-
  [Paper Note] Marco-Bench-MIF: On Multilingual Instruction-Following Capability of Large Language Models
description: >-
  [ACL 2025][Multilingual & Machine Translation][multilingual benchmark] Expands the English IFEval benchmark to 30 languages with cultural localization, revealing a 25-35% accuracy gap between high- and low-resource languages in multilingual instruction following, and showing that machine-translated data underestimates model performance by 7-22%.
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "multilingual benchmark"
  - "instruction following"
  - "localization"
  - "cross-lingual evaluation"
  - "IFEval"
date: 2026-05-08
content_hash: 9495dede01009f28
---

# Marco-Bench-MIF: On Multilingual Instruction-Following Capability of Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2507.11882](https://arxiv.org/abs/2507.11882)  
**Code**: [GitHub](https://github.com/AIDC-AI/Marco-Bench-MIF)  
**Area**: Multilingual Translation  
**Keywords**: multilingual benchmark, instruction following, localization, cross-lingual evaluation, IFEval

## TL;DR

Expands the English IFEval benchmark to 30 languages with cultural localization, revealing a 25-35% accuracy gap between high- and low-resource languages in multilingual instruction following, and showing that machine-translated data underestimates model performance by 7-22%.

## Background & Motivation

**Monolingual Limitations of Existing Benchmarks**: Instruction-following evaluation benchmarks like IFEval are predominantly designed for English, falling short of assessing the true capabilities of LLMs in multilingual scenarios.

**Inadequate Quality of Machine Translation Data**: Multilingual datasets such as Multi-IF are generated solely via machine translation, failing to capture linguistic and cultural nuances, which leads to distorted evaluation results.

**Lack of Adaptation to Language-Specific Features**: Different languages possess unique linguistic constraints (e.g., the lack of letter casing in Chinese, different passive voice structures in Japanese), which simple translation cannot handle.

**Need for Cultural Context Localization**: Cultural references in instructions (festivals, locations, company names) must undergo target-language localization to ensure cultural relevance in evaluations.

**Neglect of Low-Resource Languages**: Existing evaluations cover a limited range of languages; the instruction-following capabilities in low-resource languages like Yoruba (yo), Nepali (ne), and Kazakh (kk) remain largely unassessed.

**Compositional Instruction Understanding as a Bottleneck**: LLMs perform reasonably well when satisfying a single instruction, but the prompt-level accuracy when satisfying multiple constraints simultaneously is 10-20% lower than instruction-level accuracy. This performance gap is even more pronounced in multilingual settings.

## Method

### Overall Architecture

Marco-Bench-MIF utilizes a three-stage pipeline (preprocessing $\rightarrow$ translation/localization $\rightarrow$ post-processing) to extend the 541 English instruction-response pairs of IFEval to 30 languages (covering 6 major language families), with 541 instances per language. Automatic translation is combined with two rounds of manual verification to ensure high quality.

### Module 1: Preprocessing—Constraint Classification and Filtering

- **Cardinality Dimension**: Categorizes instructions into single-constraint (SC, 49.9%) and multi-constraint (MC, 50.1%).
- **Type Dimension**: Divides constraints into expression constraints (EC, e.g., formatting/structural requirements) and content constraints (CC, e.g., containing specific information).
- Employs a progressive adaptation strategy: simpler SC+EC are processed first, followed by complex MC+CC to minimize error propagation.
- Performs data filtering to remove ambiguous instructions and balance the distribution of constraint types.

### Module 2: Translation and Localization

- **Translation Pipeline**: Google Translate for initial translation $\rightarrow$ bilingual professional translator proofreading $\rightarrow$ LLM-assisted error correction.
- **Three-Step Localization Method**:
    - Lexical Substitution: Replaces culture-specific terms (names, locations) while keeping the constraint positions unchanged.
    - Topical Transposition: Adapts the scenario background to domains familiar to the target culture.
    - Pragmatic Restructuring: Reorganizes the instruction using the rhetorical conventions of the target language.
- Conducts cultural localization based on ten sociolinguistic dimensions (historical background, social customs, lifestyles, regional features, etc.).
- Creates parallel corpora of MT baselines and localized versions for 5 languages (ar, es, ms, yo, zh) to support comparative experiments.

### Module 3: Post-processing—Multi-layer Quality Assurance

- Combines automatic pattern detection and manual review targeting six common translation failure points: keywords, concluding remarks, echoed content, postscript consistency, case-sensitivity adherence, and Latin character frequencies in non-Latin scripts.
- Double LLM cross-verification: one LLM generates output, and another analyzes failure cases, distinguishing among model capability limits, instruction set defects, and evaluation logic vulnerabilities.
- Systematic localization of the evaluation framework across 30 languages: punctuation alignment, response language verification, multi-paragraph coherence validation, and constrained output checks.

### Evaluation Metrics

- **Strict/Loose**: Strict uses exact rule matching, whereas Loose permits matching after text normalization (e.g., markdown removal, boundary adjustments).
- **Prompt-level/Instruction-level**: Prompt-level requires all instructions within a prompt to be fully satisfied, while Instruction-level evaluates the individual adherence rate of each instruction constraint.

## Experiments

### Table 1: Overall Results (20+ Models, Average across 4 Metrics)

| Model | Prompt(S) | Prompt(L) | Inst.(S) | Inst.(L) | Avg |
|------|-----------|-----------|----------|----------|-----|
| Ministral-8B | 21.74 | 24.49 | 46.45 | 49.72 | 35.60 |
| Qwen2.5-7B | 42.99 | 47.43 | 64.42 | 68.02 | 55.72 |
| Gemma2-27B | 58.86 | 61.35 | 77.21 | 78.78 | 69.05 |
| LLaMA3.3-70B | 67.42 | 70.32 | 80.43 | 82.25 | 75.11 |
| GPT-4o | 71.43 | 75.89 | 84.49 | 87.13 | 79.73 |
| Claude3.5-sonnet | 73.61 | 76.77 | 85.62 | 87.71 | **80.93** |

### Table 2: Linguistic Analysis (30 Languages, Average Instruction-level Loose Accuracy)

| Language Category | Representative Languages | Accuracy Range |
|----------|----------|-----------|
| High-resource (Europe/East Asia) | de, fr, zh, en | 70-90% |
| Mid-resource | ar, ko, tr | 55-70% |
| Low-resource | yo, ne, kk | 29-50% |

### Key Findings

1. **Instruction-level vs. Prompt-level Gap**: The instruction-level accuracy of all models is 10-20% higher than their prompt-level accuracy, with smaller models exhibiting larger gaps (e.g., Ministral-8B exhibits a 24.7 percentage point difference). This indicates that compositional instruction reasoning remains a key bottleneck.
2. **Model Scaling Effect**: Models larger than 70B outperform 8B models by 45-60% in absolute accuracy. However, Qwen2.5-7B already achieves a 64.42% strict instruction-level accuracy, showing that basic instruction comprehension can be realized at smaller scales.
3. **High- vs. Low-Resource Language Divide**: Even Claude 3.5 Sonnet achieves only 62.3% in Yoruba (yo) compared to 90.3% in English, resulting in a gap of approximately 28 percentage points.
4. **Script-Specific Challenges**: Right-To-Left (RTL) scripts (ar, he) are particularly sensitive; for instance, LLaMA-3.3-70B achieves 78.5% in Hebrew vs. 54.7% in Urdu.
5. **Performance Underestimation by MT Data**: Localized data yields 7-22% higher accuracy compared to machine-translated data. This discrepancy is most pronounced in low-resource languages (e.g., Claude's performance in Yoruba is underestimated by 7.1%).
6. **Language-Specific Response Capability**: The multilingual-enhanced model Aya-expanse-32B achieves 95.69% in the "respond in the specified language" task, outperforming most commercial models.

## Highlights & Insights

- Covers a high-quality localized benchmark across 30 languages and 6 major language families, significantly exceeding existing multilingual instruction-following evaluations.
- Systematically resolves cross-lingual and cross-cultural adaptation challenges through a three-step localization method (lexical substitution, topical transposition, and pragmatic restructuring).
- Experiments reveal the systematic underestimation effect of MT-based evaluation data, offering a valuable reference for multilingual evaluation methodologies.
- Conducts fine-grained analyses across multiple dimensions—such as constraint types, script characteristics, and language family transfer—yielding highly practical insights.

## Limitations & Future Work

- The 30 languages covered still do not include non-Latin scripts (e.g., Ge'ez/Ethiopic, Cherokee) or dialectal variations (e.g., Arabic dialects).
- Cultural localization is confined to superficial adaptations (e.g., date formats) and does not delve into deeper pragmatic levels (e.g., Japanese honorific strategies).
- Residual biases persist in automatic localization (e.g., GPT-4's tendency toward formal register), and translation artifacts may still remain in certain languages.
- Evaluates only static prompts without addressing interactive instruction-refinement scenarios.
- Being an extension of IFEval, it inherits the inherent constraints on constraint types and instruction design from the original IFEval.

## Related Work & Insights

- **IFEval** (Zhou et al., 2023): English instruction-following evaluation benchmark, serving as the foundation for the expansion in this work.
- **Multi-IF** (He et al., 2024): Multi-turn multilingual instruction-following benchmark, but primarily relying on MT data.
- **CulturalBench** (Chiu et al., 2024): Cultural knowledge evaluation benchmark.
- **BLEND** (Myung et al., 2024): Multicultural and multilingual everyday knowledge benchmark.
- **CVQA** (Romero et al., 2024): Multicultural and multilingual visual question answering benchmark.

## Rating

- **Novelty**: ⭐⭐⭐ — The methodological framework follows a standard pipeline of translation, localization, and verification; the core contribution lies in its scale and systematic approach.
- **Effectiveness**: ⭐⭐⭐⭐ — Conducts a large-scale evaluation over 20+ models and 30 languages, presenting solid and credible findings.
- **Value**: ⭐⭐⭐⭐ — Fills a gap in multilingual instruction-following evaluation, with direct reference value for the development and assessment of multilingual LLMs.
- **Recommendation Index**: ⭐⭐⭐⭐ — A benchmark paper with outstanding empirical contributions, highly recommended for researchers focusing on multilingual LLM evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] XIFBench: Evaluating Large Language Models on Multilingual Instruction Following](../../NeurIPS2025/multilingual_mt/xifbench_evaluating_large_language_models_on_multilingual_instruction_following.md)
- [\[ACL 2025\] MaXIFE: Multilingual and Cross-lingual Instruction Following Evaluation](maxife_multilingual_and_cross-lingual_instruction_following_evaluation.md)
- [\[ACL 2025\] Disentangling Language and Culture for Evaluating Multilingual Large Language Models](disentangle_language_culture.md)
- [\[ACL 2025\] SIFT-50M: A Large-Scale Multilingual Dataset for Speech Instruction Fine-Tuning](sift-50m_a_large-scale_multilingual_dataset_for_speech_instruction_fine-tuning.md)
- [\[ACL 2025\] Just Go Parallel: Improving the Multilingual Capabilities of Large Language Models](just_go_parallel_improving_the_multilingual_capabilities_of_large_language_model.md)

</div>

<!-- RELATED:END -->
