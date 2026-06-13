---
title: >-
  [Paper Note] Can LLMs Act as Historians? Evaluating Historical Research Capabilities of LLMs via the Chinese Imperial Examination
description: >-
  [ACL 2026][LLM Evaluation][ProHist-Bench] This paper constructs ProHist-Bench: centered on the 1300-year history of the Chinese Imperial Examination, featuring 400 expert-level questions handwritten by historians and 10…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "ProHist-Bench"
  - "Imperial Examination"
  - "rubric evaluation"
  - "LLM-as-judge"
  - "historical reasoning"
date: 2026-05-08
content_hash: 646d7df3620d3b84
---

# Can LLMs Act as Historians? Evaluating Historical Research Capabilities of LLMs via the Chinese Imperial Examination

**Conference**: ACL 2026  
**arXiv**: [2604.24690](https://arxiv.org/abs/2604.24690)  
**Code**: https://github.com/inclusionAI/ABench/tree/main/ProHist-Bench  
**Area**: LLM Evaluation / Historical Research  
**Keywords**: ProHist-Bench, Imperial Examination, rubric evaluation, LLM-as-judge, historical reasoning

## TL;DR
This paper constructs ProHist-Bench: centered on the 1300-year history of the Chinese Imperial Examination, featuring 400 expert-level questions handwritten by historians and 10,891 fine-grained rubrics. It evaluates the professional historical research capabilities of 18 SOTA LLMs—the results show that even for the strongest Gemini-3-Pro and Qwen3-235B, the Rubric Scores are only ~28, far below those of open-book historians.

## Background & Motivation

**Background**: LLMs have begun to assist in processing large volumes of digitized archives and generating historical narratives. Relevant benchmarks include HiST-LLM (global history facts), AC-EVAL (ancient Chinese understanding), C-Eval/CMMLU (comprehensive knowledge), and WYWEB (classical Chinese NLP tasks). However, the vast majority remain at the level of "basic historical knowledge + lexical understanding."

**Limitations of Prior Work**: The authors present a typical case—when asked professional questions about the imperial examination, LLMs confidently fabricate records and fail to judge conflicting historical materials. Existing evaluations fail to expose these deep flaws because they mostly consist of multiple-choice or simple QA, where high scores can be achieved via BLEU/ROUGE matching despite poor answer quality.

**Key Challenge**: There is an order-of-magnitude gap between the abilities required for professional historical research (concept definition, fact organization, evidentiary reasoning, viewpoint integration, temporal reframing, etc.) and the abilities evaluated by existing benchmarks (factual recall, textual similarity). The former requires professional training in argumentation chains, historical source citation, cross-dynasty comparison, traditional taboos, and stylistic sensibility, while the latter only requires "rote memorization" and "copying similar words."

**Goal**: (1) Design a benchmark capable of distinguishing "basic historical knowledge" from "professional historical research"; (2) Construct a fine-grained, reproducible rubric evaluation system; (3) Systematically evaluate the true level of current SOTA LLMs on professional historical tasks and locate specific capability shortcomings.

**Key Insight**: The "Imperial Examination" is selected as the anchor theme—it spans 9 dynasties and 1300 years, covering political, economic, social, cultural, and intellectual history. Its historical materials are rich and academic disputes are clear, sufficient to reflect LLM capabilities in broader historical research.

**Core Idea**: Use historian-handwritten questions + 9-dimensional rubrics (including positive and penalty items) + LLM-as-judge Rubric Score metrics to decompose "competence in historical research" into 9 scorable dimensions, exposing where LLMs fail row by row.

## Method

### Overall Architecture
ProHist-Bench consists of three main components: (1) 400 expert questions divided into 4 task types—T1 Terminology Explanation, T2 Factual QA, T3 Historical Reasoning, T4 Celu Generation (Eight-legged essay format); (2) 10,891 rubric criteria (27.23 items/question), covering 9 historical research capability dimensions (R1–R9), each with weighted scores; (3) DeepSeek-R1 serves as the LLM-as-judge to score items 0/1 according to rubrics, aggregated into a Rubric Score (RS). The entire pipeline follows 4-stage quality control: question writing → cross-review → third-party arbitration → 5% spot check. Rubrics are refined through iterations between historians and model outputs.

### Key Designs

1. **Four Task Categories Covering Capability Gradients (T1–T4)**:

    - Function: Evaluates "historical capability" across four levels from memorization to creation to prevent benchmark bias.
    - Mechanism: T1 Terminology Explanation (e.g., explaining "Gongshi") tests concept definition; T2 Factual QA (e.g., anti-cheating systems in the Qing Dynasty) tests fact organization; T3 Historical Reasoning (e.g., discussing the "Six-grade Promotion and Demotion Law") tests comparison and argumentation; T4 Celu Generation is the most unique—requiring the model to play the role of a candidate in the 46th year of Qianlong, writing a 700-word examination essay in the Eight-legged format (Poti/Chengti/Qijiang/Ruti) while adhering to naming taboos. T4 is a representative design for making "historical role-play" a quantifiable task.
    - Design Motivation: Existing benchmarks consist almost entirely of T1+T2 types; T3 and T4 are the real hurdles distinguishing "Wiki-searchers" from "Historians." Experiments show that while BERTScore for T1/T2 is similar (70+), RS gaps in T3 can expand three-fold.

2. **9-Dimensional Rubric Framework + Penalty Items**:

    - Function: Decomposes model responses into 9 independently scorable capability dimensions and accounts for hallucinations/fabrications.
    - Mechanism: R1 Concept Definition (2pt), R2 Fact Organization (3pt), R3 Historical Comparison (3pt), R4 Evidentiary Reasoning (4pt), R5 Comprehensive Evaluation (1pt), R6 Viewpoint Integration (5pt), R7 Academic Expression (5pt); T4 adds R8 Eight-legged Style (3pt) and R9 Temporal Restoration (up to 9pt, including format and style). Penalties include "Fabricated literature -5pt," "Use of taboo words -60pt," "Date conversion error -3pt," etc. RS formula: $$\text{RS}=\max\bigl(0,\frac{\sum I_b w_b + \sum I_p w_p}{\sum w_b}\bigr)$$, normalized to [0, 1].
    - Design Motivation: Traditional BLEU/ROUGE cannot determine if "argumentation holds water," and a single total score masks capability gaps. Fine-grained rubrics provide both specific feedback and locate whether failures are "factual," "argumentative," or "stylistic." High penalties (-5 for fabrication, -60 for taboos) reflect the gravity of "red lines in historiography."

3. **LLM-as-judge Selection + Consistency Calibration**:

    - Function: Automates scoring for thousands of rubrics, avoiding the need for historians for every question.
    - Mechanism: 50 samples were randomly selected; 6 candidate judge models performed 0/1 hit labeling per rubric item. Pearson correlation coefficients were calculated against historian expert labels for both rubric-level and answer-level validation. DeepSeek-R1 achieved the highest average consistency (0.77) and was selected as the judge.
    - Design Motivation: The success of rubric evaluation depends entirely on judge reliability. Calibrating before large-scale evaluation is significantly more rigorous than using GPT-4 directly—this is a reusable methodology for extending rubric frameworks to other historical subfields.

### Loss & Training
No training; purely a diagnostic benchmark. The judge uses DeepSeek-R1 with deterministic hyperparameters; all 18 tested models use a unified prompt template (standard zero-shot), with a comparison of four strategies: Role-playing / Professional / CoT / RAG.

## Key Experimental Results

### Main Results
Comparison of 18 LLMs on T1–T3 (selected):

| Model | BLEU | ROUGE | BERTScore | Rubric Score |
|------|------|-------|-----------|--------------|
| Gemini-3-Pro-Preview | 1.94 | 6.27 | 73.97 | **26.71** |
| Gemini-3-Pro-Preview-Thinking | 2.35 | 5.16 | 73.92 | **26.73** |
| Qwen3-235B-A22B-Thinking | 1.08 | 5.22 | 72.50 | **28.14** |
| DeepSeek-R1-0528 | 1.93 | 6.60 | 73.15 | 26.87 |
| Qwen3-Max | 4.77 | 6.64 | **75.01** | 17.71 |
| GPT-5.2-Thinking | 4.45 | 4.58 | 71.55 | 14.08 |
| Claude-Sonnet-4.5-Thinking | 2.53 | 4.76 | 71.49 | 12.99 |
| Kimi-K2-Thinking | 3.62 | 6.43 | 73.20 | 22.79 |
| GLM-4.6-Thinking | 2.09 | 5.11 | 72.30 | 24.32 |
| Llama-4-Scout-17B-16E | 2.59 | 3.09 | 72.68 | **2.72** |
| gpt-oss-120b | 1.27 | 1.78 | 70.18 | 10.75 |
| gpt-oss-20b | Fail | Fail | Fail | Fail |

**Key Findings**: (1) Even the strongest models reach an RS of only ~28, far from the 100-point limit; (2) BERTScore remains around 70+ for almost all models, **failing completely to distinguish capabilities**, which validates the necessity of RS; (3) Models pre-trained on Chinese corpora (Qwen / DeepSeek / GLM / Kimi) are systematically superior to English-centric models like Llama 4 and gpt-oss; gpt-oss-20b even failed the tasks.

### Ablation Study / Prompt Strategy Comparison
Average RS under 4 prompt strategies:

| Strategy | Closed Average | Open Average | Key Model Comparison |
|------|----------|----------|--------------|
| Historian Role-play | 23.11 | 21.04 | Gemini-3-Pro: 32.94 / Kimi-K2: 31.61 |
| Professional Prompt | **23.76** | 20.99 | Kimi-K2 reached 36.10 (highest) |
| Chain-of-Thought | 19.90 | 18.96 | Generally decreased; CoT was a hindrance |
| RAG (k=10) | 20.19 | 19.46 | Gemini-3-Pro dropped from 32.94 to 26.53 |

### Key Findings
- **CoT hinders historical capability**: The closed-source average for CoT was ~3 points lower than Role/Prof strategies, likely because forced step-by-step reasoning creates more opportunities for hallucinations in professional historical QA; "activating expert identity" is more effective than "forcing steps."
- **RAG fails in this domain**: Increased document retrieval led to lower scores (Gemini-3-Pro RS: 27.35 at k=10 → 25.51 at k=100) because the quality of Chinese classical historical corpora is extremely poor, resulting in retrieved noise.
- **R6 Viewpoint Integration is the universal weakness**: Most models have a hit rate of only 0.02–0.12 in R6, with the maximum at ~0.23, proving LLMs cannot integrate conflicting materials; R3/R4 (comparison and evidentiary reasoning) are also weak.
- **R8 Eight-legged format is the strongest**: Models scored high on rule-based "format generation," proving formalization capability is mature, while substantive argumentation remains weak.
- **LLM vs. Human**: SOTA LLMs approach the level of closed-book historians on T3/T4 but fall far behind open-book historians on T1/T2—implying LLMs are currently acceptable as assistants but insufficient for independent verification.

## Highlights & Insights
- **Pioneering rubric × Historiography evaluation**: While rubric evaluation exists in law/healthcare, this is the first work to systematize it for history, releasing a 9-dimensional design whose methodology can be replicated in law, religious studies, archaeology, and other humanities.
- **Ingenuity of Imperial Examination as evaluation carrier**: It spans dynasties, has official institutional archives, rigid rules (taboos/style), and numerous academic disputes—it measures both facts and argumentation, acting as a tailormade subject for LLM historical evaluation.
- **Constraints of "Role + Dynasty + Taboo" in T4 Celu**: Provides a paradigm for turning "role-play" into a quantifiable task, applicable to legal debate or medical consultation where professional role consistency is required.
- **Judge calibration as an RFP template**: The practice of comparing 6 candidate judges against human Pearson correlation using a 50-sample set should become a standard precursor for LLM-as-judge evaluations.

## Limitations & Future Work
- All LLMs were tested with a unified prompt without model-specific prompt engineering; some models may be undervalued.
- Coverage is limited to Chinese imperial examination history; it has not yet expanded to Western, religious, or archaeological history, so rubric generalizability needs further validation.
- DeepSeek-R1 as judge has a 0.77 correlation with historians, implying a ~23% deviation in automated scoring; absolute score rankings should be treated with caution.
- The high penalty for "Taboo words -60pt" allows certain tasks to be dominated by extreme values, potentially suppressing otherwise decent models.
- Personal Note: With only 400 questions, the sample size is small compared to other benchmarks (e.g., C-Eval ~13k), leading to high variance; small differences between models may lack statistical significance.

## Related Work & Insights
- **vs. HiST-LLM (Hauser 2024)**: HiST-LLM measures global historical facts (broad but shallow); ProHist-Bench focuses on the imperial examination but utilizes deep rubrics; the two are complementary.
- **vs. C-Eval / CMMLU**: These are comprehensive Chinese subject evaluations where history is a subset and entirely multiple-choice; Ours uses open-ended generation + rubrics.
- **vs. WYWEB (Zhou 2023)**: Measures classical Chinese NLP tasks (NER / translation) at the linguistic layer; Ours focuses on the research methodology layer.
- **vs. PLawBench (Shi 2026) / HealthBench (Arora 2025)**: Also rubric-based; this paper introduces this paradigm to history for the first time.

## Rating
- Novelty: ⭐⭐⭐⭐ Rubric-based evaluation has precedents, but applying it cross-disciplinely to historiography with an R1–R9 system is original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 18 SOTA LLMs × 4 prompt strategies × 9 capability dimensions × 4 RAG k-levels, with human baseline comparisons, makes it very robust.
- Writing Quality: ⭐⭐⭐⭐ Task definitions, rubric tables, and case studies are all clear; the appendix is very detailed.
- Value: ⭐⭐⭐⭐⭐ Quantitatively exposes the true gap of LLMs in professional humanities tasks, serving as a reality check for optimism regarding "LLMs for academic research"; the rubric framework is reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Exploring the Capability Boundaries of LLMs in Mastering of Chinese Chouxiang Language](exploring_the_capability_boundaries_of_llms_in_mastering_of_chinese_chouxiang_la.md)
- [\[ACL 2026\] Personalized Benchmarking: Evaluating LLMs by Individual Preferences](personalized_benchmarking_evaluating_llms_by_individual_preferences.md)
- [\[ACL 2026\] TaxPraBen: A Scalable Benchmark for Structured Evaluation of LLMs in Chinese Real-World Tax Practice](taxpraben_a_scalable_benchmark_for_structured_evaluation_of_llms_in_chinese_real.md)
- [\[ACL 2026\] BizCompass: Benchmarking the Reasoning Capabilities of LLMs in Business Knowledge and Applications](bizcompass_benchmarking_the_reasoning_capabilities_of_llms_in_business_knowledge.md)
- [\[ACL 2026\] Language Models Don't Know What You Want: Evaluating Personalization in Deep Research Needs Real Users](language_models_dont_know_what_you_want_evaluating_personalization_in_deep_resea.md)

</div>

<!-- RELATED:END -->
