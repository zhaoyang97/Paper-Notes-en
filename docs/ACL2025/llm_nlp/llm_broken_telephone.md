---
title: >-
  [Paper Note] LLM as a Broken Telephone: Iterative Generation Distorts Information
description: >-
  [ACL 2025][LLM (Other)][Information distortion] Using translation as a testbed to simulate the "Telephone Game" in LLMs, this study finds that information becomes severely distorted after 100 iterations of translation. For example, a news report about a truck driver being fined is transformed into "a car exploded after receiving compensation" after 100 rounds of English-Thai translation. The choice of pivot language, chain complexity, and decoding temperature are identified a…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Information distortion"
  - "iterative generation"
  - "model collapse"
  - "machine translation"
  - "factuality"
date: 2026-05-08
content_hash: 80bd2456549d8e38
---

# LLM as a Broken Telephone: Iterative Generation Distorts Information

**Conference**: ACL 2025  
**arXiv**: [2502.20258](https://arxiv.org/abs/2502.20258)  
**Code**: [https://github.com/amr-mohamedd/LLM-as-a-Broken-Telephone](https://github.com/amr-mohamedd/LLM-as-a-Broken-Telephone)  
**Area**: LLM / NLP  
**Keywords**: Information distortion, iterative generation, model collapse, machine translation, factuality

## TL;DR
Using translation as a testbed to simulate the "Telephone Game" in LLMs, this study finds that information becomes severely distorted after 100 iterations of translation. For example, a news report about a truck driver being fined is transformed into "a car exploded after receiving compensation" after 100 rounds of English-Thai translation. The choice of pivot language, chain complexity, and decoding temperature are identified as key factors regulating the rate of distortion.

## Background & Motivation

**Background**: As LLM-generated content increasingly floods the internet, a loop where models consume their own outputs becomes inevitable. While research on model collapse has shown that iterative training on synthetic data leads to distribution degradation, studies on information distortion during inference-stage iterative generation (non-training) remain virtually non-existent.

**Limitations of Prior Work**:
   - Model collapse literature primarily focuses on the training loop rather than iterative processing during inference.
   - Perez et al. (2024) investigated text attribute evolution (toxicity, positivity, difficulty) in paraphrasing/continuation chains but overlooked translation—one of the most common applications of iterative LLM chains.
   - Existing studies only utilize a single-model, single-language chain, failing to account for multi-model collaboration (e.g., scenarios in multi-agent systems where the output of model A is processed by model B).
   - There is a lack of systematic quantification of how factuality and semantic similarity degrade over iterations.

**Key Challenge**: Multi-agent systems and AI content loops are becoming increasingly common, yet we know very little about how much truth remains after information is repeatedly processed by LLMs.

**Goal**: (1) Quantify the accumulation of information distortion in iterative LLM translation. (2) Analyze the impact of pivot languages, chain complexity, and model combinations on distortion. (3) Explore mitigation strategies.

**Key Insight**: Drawing an analogy to the human Telephone Game, translation chains are designed as controlled iterative generation experiments (each round containing EN $\rightarrow$ pivot language $\rightarrow$ EN) to evaluate the output against the original text after 100 rounds.

**Core Idea**: LLMs act as a "broken telephone"—information progressively distorts through iterative generation, and the rate of distortion is systematically influenced by language selection, chain complexity, and decoding strategies.

## Method

### Overall Architecture
The study evaluates combinations of 3 datasets (BookSum, ScriptBase, News2024) $\times$ 150 documents $\times$ 6 pivot languages (FR/DE/NL/VN/ZH/TH) $\times$ 2 models (Llama-3.1-8B / Mistral-7B) across 100 rounds of iterative translation. At each round, the outputs are compared to the original text using 5 text relevance metrics and the FActScore factuality metric.

### Key Designs

1. **Progressive Complexity in Three Experimental Setups**:
    - **Exp1 Bilingual Self-Loop**: A single model repeatedly translates between EN and a pivot language for 100 rounds across 6 pivot languages $\times$ 2 models $\times$ 3 datasets.
    - **Exp2 Bilingual Duo-Model**: Two different models alternately participate in the same translation chain (simulating multi-agent collaboration) for EN $\leftrightarrow$ FR and EN $\leftrightarrow$ TH.
    - **Exp3 Multi-pivot Multi-model**: 2 to 4 pivot languages and 2 to 3 models are randomly arranged in the same chain to test the upper limit of complexity.
    - **Design Motivation**: Stepping from simple to complex settings to simulate real-world scenarios ranging from simple translation to complex multi-agent, multi-lingual information propagation.

2. **Dual-Axis Evaluation System**:
    - **Textual Relevance**: Evaluated via BLEU (n-gram precision), ROUGE-1 (unigram overlap), CHR-F (character-level), METEOR (paraphrastic variation), and BERTScore (semantic similarity).
    - **Factuality**: Evaluated via FActScore, which decomposes long texts into atomic facts and verifies them using Claude 3.5 Sonnet against the original text as the ground truth.
    - **Design Motivation**: Textual relevance metrics capture surface-level lexical drift, while FActScore assesses deep semantic and factual distortions.

3. **Gradient Quantification Method**:
    - The average gradient of the FActScore curve over iterations is calculated to quantify the speed of distortion.
    - **Design Motivation**: Although different languages or settings might eventually converge to similar low scores, their rates of degradation differ significantly. The gradient serves as a better indicator of risk levels.

### Ablation Study
- **Temperature Ablation**: Evaluated across five settings: 1e-6, 0.25, 0.5, 0.75, and 1.0.
- **Prompt Constraint Ablation**: Evaluated across three prompt designs: simple, base, and constrained.
- **Paraphrasing Chain Ablation**: Replaced the translation step with same-language paraphrasing to verify if the distortion is unique to translation.

## Key Experimental Results

### Exp1: Average FActScore Gradient (Rate of Distortion)

| Language Pair | Llama (News2024) | Mistral (News2024) |
|---|---|---|
| EN↔FR | -0.004 ± 0.003 | -0.007 ± 0.004 |
| EN↔DE | -0.005 ± 0.003 | -0.011 ± 0.006 |
| EN↔NL | -0.005 ± 0.003 | -0.011 ± 0.006 |
| EN↔VN | -0.008 ± 0.005 | -0.027 ± 0.015 |
| EN↔ZH | -0.011 ± 0.006 | -0.024 ± 0.012 |
| EN↔TH | **-0.018 ± 0.009** | **-0.038 ± 0.022** |

### Exp3: Impact of Chain Complexity (FActScore)

| Setup | No. of Languages | No. of Models | Round 10 | Round 100 | Average Gradient |
|---|---|---|---|---|---|
| Setting 1 | 3 | 2 | 0.063 | 0.04 | -0.036 ± 0.02 |
| Setting 2 | 3 | 3 | 0.075 | 0.04 | -0.034 ± 0.02 |
| Setting 3 | 5 | 2 | 0.054 | 0.04 | **-0.038 ± 0.02** |

### Temperature Ablation (EN↔FR, Llama)

| Temperature | Stability after First 2 Rounds | FActScore Trend after 100 Rounds |
|---|---|---|
| 1e-6 | Almost stable | Slight decline, then flattens |
| 0.25 | Slow decline | Continuous slow decline |
| 0.50 | Noticeable decline | Moderate decline |
| 1.00 | Steepest decline | Continuous severe divergence |

### Key Findings
- **Language similarity determines distortion rate**: Latin-script languages (FR/DE/NL) have gradients close to 0, whereas non-Latin-script languages (TH/ZH/VN) display 3x to 10x steeper gradients.
- **Thai is the "worst telephone"**: EN↔TH exhibits the fastest distortion across all datasets and models, with Mistral reaching a gradient of -0.040 on BookSum.
- **More languages > More models**: Setting 3 (5 languages, 2 models) degrades faster than Setting 2 (3 languages, 3 models), indicating that language diversity amplifies distortion more than model diversity.
- **Paraphrasing also distorts**: Same-language paraphrasing chains without translation also exhibit information degradation, though at a slower rate than translation chains.
- **Low temperature + constrained prompts provide effective mitigation**: Setting the temperature to 1e-6 almost halts distortion, and constrained prompts significantly slow down degradation compared to simple prompts.

## Highlights & Insights
- **The "Telephone Game" metaphor is highly intuitive and holds practical importance**: The illustrative transition from a truck to a bus and finally to a car (shown in Table 1 of the paper) clearly illustrates the distortion process. This serves as a warning for any AI workflow utilizing iterative LLM pipelines (e.g., summarization $\rightarrow$ translation $\rightarrow$ summarization).
- **Direct implications for multi-agent systems**: In chains where the output of Agent A is processed by Agent B and then passed to Agent C, each step accumulates distortion. Consequently, multi-agent frameworks must incorporate built-in fact-checking steps.
- **Factuality degradation poses a greater risk than surface-level lexical drift**: A decline in BLEU might simply reflect the use of synonyms, but a decrease in FActScore signifies real factual distortion, posing a severe risk in news, medical, and legal domains.

## Limitations & Future Work
- **Reliance on 7-9B models**: It remains to be explored whether larger models (70B+) or GPT-4 class models are more resilient to such distortion.
- **Limited dataset domains**: The three domains evaluated (books, scripts, news) share similar stylistic characteristics. Distortion in highly specialized domains (e.g., medicine, law) may be more pronounced.
- **Focus on default decoding**: The impact of alternative decoding strategies, such as greedy decoding versus beam search, remains unexamined.
- **Evaluator bias in FActScore**: Using Claude 3.5 Sonnet as the judge may introduce systematic evaluation biases.

## Related Work & Insights
- **vs. Shumailov et al. (2023) Model Collapse**: Model collapse investigates the training loop, whereas this paper targets the inference loop. Together, they provide a complete picture of the risks present in AI-generated content cycles.
- **vs. Perez et al. (2024)**: While Perez et al. examined the evolution of toxicity and positivity in paraphrased chains, this study introduces translation scenarios, multi-model chains, and formal factuality quantification, offering a more extensive perspective.
- **vs. Peterson (2024) Knowledge Collapse**: Knowledge collapse is a broad, macro-level concept, whereas this study offers micro-level, quantifiable evidence of information degradation using concrete translation chains.

## Rating

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LongDPO: Unlock Better Long-form Generation Abilities for LLMs via Critique-augmented Stepwise Information](longdpo_unlock_better_long-form_generation_abilities_for_llms_via_critique-augme.md)
- [\[ACL 2025\] Knockout LLM Assessment: Using Large Language Models for Evaluations through Iterative Pairwise Comparisons](knockout_llm_assessment_using_large_language_models_for_evaluations_through_iter.md)
- [\[ACL 2025\] From Selection to Generation: A Survey of LLM-based Active Learning](from_selection_to_generation_a_survey.md)
- [\[ACL 2025\] An Empirical Study of Iterative Refinements for Non-Autoregressive Translation](an_empirical_study_of_iterative_refinements_for_non-autoregressive_translation.md)
- [\[ACL 2025\] Transforming Podcast Preview Generation: From Expert Models to LLM-Based Systems](transforming_podcast_preview_generation_from_expert_models_to_llm-based_systems.md)

</div>

<!-- RELATED:END -->
