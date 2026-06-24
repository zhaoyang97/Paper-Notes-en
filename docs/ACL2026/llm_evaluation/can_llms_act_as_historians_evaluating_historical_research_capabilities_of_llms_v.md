---
title: >-
  [Paper Note] Can LLMs Act as Historians? Evaluating Historical Research Capabilities of LLMs via the Chinese Imperial Examination
description: >-
  [ACL 2026][LLM Evaluation][ProHist-Bench] This paper constructs ProHist-Bench: anchored by the 1,300-year history of the Chinese Imperial Examination, it features 400 expert-level questions handwritten by historians and 10,891 fine-grained rubrics to evaluate the professional historical research capabilities of 18 SOTA LLMs. Even the strongest models, Gemini-3-Pro and Qwen3-235B, achieved Rubric Scores of only approximately 28, significantly lower than those of open-book hist…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "ProHist-Bench"
  - "Imperial Examination"
  - "Rubric Evaluation"
  - "LLM-as-judge"
  - "Historical Reasoning"
date: 2026-05-08
content_hash: 06d9897a09afe58d
---

# Can LLMs Act as Historians? Evaluating Historical Research Capabilities of LLMs via the Chinese Imperial Examination

**Conference**: ACL 2026  
**arXiv**: [2604.24690](https://arxiv.org/abs/2604.24690)  
**Code**: https://github.com/inclusionAI/ABench/tree/main/ProHist-Bench  
**Area**: LLM Evaluation / Historical Research  
**Keywords**: ProHist-Bench, Imperial Examination, Rubric Evaluation, LLM-as-judge, Historical Reasoning

## TL;DR
This paper constructs ProHist-Bench: anchored by the 1,300-year history of the Chinese Imperial Examination, it features 400 expert-level questions handwritten by historians and 10,891 fine-grained rubrics to evaluate the professional historical research capabilities of 18 SOTA LLMs. Even the strongest models, Gemini-3-Pro and Qwen3-235B, achieved Rubric Scores of only approximately 28, significantly lower than those of open-book historians.

## Background & Motivation

**Background**: LLMs have begun assisting in the processing of large-scale digitized archives and the generation of historical narratives. Relevant benchmarks include HiST-LLM (global history facts), AC-EVAL (ancient Chinese understanding), C-Eval/CMMLU (comprehensive knowledge), and WYWEB (Classical Chinese NLP tasks). However, the vast majority remain at the level of "basic historical knowledge + lexical understanding."

**Limitations of Prior Work**: The authors present a typical case—when asked professional questions about the Imperial Examination, LLMs confidently fabricate records and fail to adjudicate between conflicting historical sources. Existing evaluations fail to expose these deep-seated flaws because they mostly consist of multiple-choice or simple QA, where high scores can be achieved via BLEU/ROUGE matching despite poor response quality.

**Key Challenge**: There is an order-of-magnitude gap between the capabilities required for professional historical research (concept definition, fact organization, evidentiary reasoning, viewpoint integration, temporal reframing, etc.) and those evaluated by existing benchmarks (factual recall, textual similarity). The former requires professional training in chains of argumentation, historical source citation, cross-dynastic comparison, and awareness of linguistic taboos and style, whereas the latter only requires "memorization" and "copying similar terms."

**Goal**: (1) Design a benchmark capable of distinguishing "basic historical knowledge" from "professional historical research"; (2) Construct a fine-grained, reproducible rubric evaluation system; (3) Systematically evaluate current SOTA LLMs on professional historical tasks and pinpoint specific capability deficiencies.

**Key Insight**: The "Imperial Examination" (Keju) is selected as the anchoring theme—spanning 9 dynasties and 1,300 years, it covers aspects of political, economic, social, cultural, and intellectual history. Its abundant historical materials and clear academic controversies sufficiently reflect LLM capabilities in broader historical research.

**Core Idea**: Use historian-written questions + a 9-dimensional rubric (including positive and penalty items) + an LLM-as-judge Rubric Score metric to decompose "historical research proficiency" into 9 scorable dimensions, exposing the areas where LLMs fail.

## Method

### Overall Architecture
ProHist-Bench decomposes "historical research proficiency" into a quantifiable evaluation pipeline. The input consists of 400 expert questions (categorized into T1 Terminology Explanation, T2 Factual QA, T3 Historical Reasoning, and T4 Eight-Legged Essay/Policy Dissertation) handwritten by historians. After the models provide open-ended responses, they are scored 0/1 against 10,891 fine-grained rubrics (average of 27.23 rubrics per question, covering R1–R9 across 9 dimensions of capability, each with specific weights). Finally, DeepSeek-R1 serves as an LLM-as-judge to aggregate the Rubric Score. The questions underwent a four-stage quality control process (writing → cross-review → third-party arbitration → 5% spot check), while the rubrics were refined iteratively by comparing historian insights with model outputs.

### Key Designs

**1. Four Task Categories Covering Capability Gradients: Distinguishing Memorization from Creation**

Existing benchmarks are almost entirely composed of terminology and factual QA—tasks solvable by "checking a wiki"—which fail to measure historical depth. ProHist-Bench distributes tasks across four gradients: T1 Terminology Explanation (e.g., explaining "Gongshi") measures concept definition; T2 Factual QA (e.g., how the Qing dynasty entrance system prevented cheating) measures factual organization; T3 Historical Reasoning (e.g., discussing the "Six-Grade Promotion and Demotion Law") measures cross-dynastic comparison and argumentation. T4 Policy Dissertation is the most specialized—requiring the model to play the role of a candidate in the 46th year of Qianlong, writing a 700-word examination essay in the eight-legged format (Poti, Chengti, Qijiang, Ruti) while actively observing naming taboos.

T3 and T4 are the real barriers separating "information retrieval" from "historical expertise." In experiments, BERTScore for T1/T2 clustered around 70+ for all models, appearing indistinguishable, whereas the Rubric Score gap in T3 expanded to 3×. T4 further transforms "historical role-playing" into a quantifiable task, representing the most original aspect of the design.

**2. 9-Dimensional Rubric Framework + Penalties: Decomposing Responses into Scorable Dimensions and Penalizing Fabrications**

Traditional BLEU/ROUGE cannot judge whether an "argument is valid," and a single total score hides specific weaknesses. This paper splits each response into 9 dimensions for independent scoring: R1 Concept Definition (2pt), R2 Factual Organization (3pt), R3 Historical Comparison (3pt), R4 Evidentiary Reasoning (4pt), R5 Comprehensive Evaluation (1pt), R6 Viewpoint Integration (5pt), R7 Academic Expression (5pt); T4 adds R8 Eight-Legged Style (3pt) and R9 Temporal Restoration (up to 9pt, including format and style). Simultaneously, heavy penalties are set for violations of historical research conduct: "fabricating literature -5pt," "using taboo words -60pt," "incorrect era conversion -3pt," etc. The final Rubric Score normalizes positive hits and penalties into [0, 1]:

$$\text{RS}=\max\Bigl(0,\ \frac{\sum I_b w_b + \sum I_p w_p}{\sum w_b}\Bigr)$$

This provides fine-grained feedback on whether the error is factual, argumentative, or stylistic, while penalizing serious errors like "fabricating sources" or "taboo violations" with varying weights.

**3. LLM-as-judge Selection + Consistency Calibration: Validating the Judge Before Large-Scale Scoring**

Checking thousands of rubrics against historian standards for every question is cost-prohibitive, necessitated automated LLM scoring. The success of rubric evaluation depends entirely on the reliability of the judge. Therefore, the authors first randomly sampled 50 cases, having 6 candidate judge models perform 0/1 hit labeling per rubric. Pearson correlation coefficients were calculated between these and historian expert labels at both the rubric and answer levels. DeepSeek-R1 was selected as the official judge due to its highest average consistency of 0.77. This "calibrate first, evaluate second" approach is more rigorous than directly using a strong model as a judge and provides a methodology transferable to other humanities fields like law or religious studies.

### Loss & Training
Ours does not involve any training and is a pure evaluation benchmark. The judge model, DeepSeek-R1, uses deterministic hyperparameters to ensure reproducibility. 18 evaluated models use a uniform standard zero-shot prompt template, with additional comparisons across four prompt strategies (Role-playing / Professional / CoT / RAG) to investigate the impact of response methods on historical capability.

## Key Experimental Results

### Main Results
Comparison of 18 LLMs on T1–T3 (Abridged):

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

**Key Observations**: (1) Even the strongest model's RS only reached ~28, far from the 100-point ceiling; (2) BERTScore was ~70+ for nearly all models, **failing completely to distinguish capabilities**, which validates the necessity of RS; (3) Models pre-trained on Chinese corpora (Qwen / DeepSeek / GLM / Kimi) systematically outperformed English-centric models like Llama 4 and gpt-oss.

### Ablation Study
Average RS under 4 prompt strategies:

| Strategy | Closed-source Avg | Open-source Avg | Key Model Comparison |
|------|----------|----------|--------------|
| Historian Role-play | 23.11 | 21.04 | Gemini-3-Pro: 32.94 / Kimi-K2: 31.61 |
| Professional Prompt | **23.76** | 20.99 | Kimi-K2 reached 36.10 (highest) |
| Chain-of-Thought | 19.90 | 18.96 | Universal decline; CoT was a hindrance |
| RAG (k=10) | 20.19 | 19.46 | Gemini-3-Pro dropped from 32.94 to 26.53 |

### Key Findings
- **CoT hurts historical capability**: Closed-source models averaged ~3 points lower with CoT than with Role/Prof strategies. Forced step-by-step reasoning likely creates more opportunities for hallucination; "activating expert persona" is more effective than "forcing steps."
- **RAG failure in this domain**: Scores dropped as retrieved documents increased (Gemini-3-Pro RS: 27.35 at k=10 → 25.51 at k=100), due to the poor quality of digitzed classical Chinese historical corpora, resulting in the retrieval of noisy fragments.
- **R6 Viewpoint Integration is the universal weakness**: Most models achieved a hit rate of only 0.02–0.12 in R6, with the maximum around ~0.23, proving LLMs cannot integrate conflicting historical sources. R3/R4 (comparison and evidentiary reasoning) were also weak.
- **R8 Eight-Legged format performed best**: Models scored high on "format generation" with strong rules, proving formalized capabilities are mature, but substantive argumentation remains weak.
- **LLM vs. Human**: SOTA LLMs approach the level of closed-book historians on T3/T4 but lag far behind open-book historians on T1/T2—implying LLMs are acceptable as assistive tools but insufficient for independent verification.

## Highlights & Insights
- **Pioneering Rubric × Historical Evaluation**: While rubric evaluation exists in law and healthcare, this is the first work to systematize it for history, releasing a complete 9-dimensional rubric framework replicable in other humanities like archeology or religious studies.
- **Ingenious Choice of the Imperial Examination**: The subject spans dynasties, has official institutional archives, rigid rules (taboos/style), and numerous academic disputes—making it an ideal subject for measuring both facts and argumentation.
- **T4 Constraints (Role + Dynasty + Taboo)**: A paradigm for transforming "role-playing" into a quantifiable task, generalizable to legal debates or medical consultations requiring professional role consistency.
- **Judge Calibration as an RFP Template**: The practice of using 50 samples to compare candidate judges against human Pearson correlation coefficients should become a standard preprocessing step for LLM-as-judge evaluations.

## Limitations & Future Work
- All LLMs were tested with a uniform prompt; model-specific prompt engineering was not conducted, potentially underestimating some models.
- The study covers only Chinese Imperial Examination history; expansion to Western history, religious history, or archeological history is needed to validate rubric universality.
- The correlation between DeepSeek-R1 and historians is 0.77, implying a ~23% deviation in automated scoring; absolute score rankings should be interpreted cautiously.
- The -60pt penalty for "taboo words" may allow extreme values to dominate the RS, potentially suppressing otherwise well-performing models.
- The sample size of 400 questions is relatively small compared to other benchmarks (e.g., C-Eval ~13k), leading to higher variance.

## Related Work & Insights
- **vs. HiST-LLM (Hauser 2024)**: HiST-LLM measures global historical facts (broad but shallow); ProHist-Bench focuses on the Imperial Examination but offers deep rubrics. They are complementary.
- **vs. C-Eval / CMMLU**: These are comprehensive Chinese subject evaluations where history is a subset and entirely multiple-choice; Ours is open-ended generation + rubrics.
- **vs. WYWEB (Zhou 2023)**: Measures Classical Chinese NLP tasks (NER/translation) at the linguistic level; Ours targets the research methodology level.
- **vs. PLawBench (Shi 2026) / HealthBench (Arora 2025)**: Similarly rubric-based, but Ours is the first to introduce this paradigm to history.

## Rating
- Novelty: ⭐⭐⭐⭐ Rubric-based evaluation has precedents, but cross-disciplinary application to history with the R1–R9 system is original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 18 SOTA LLMs × 4 prompt strategies × 9 capability dimensions × 4 RAG k-values, plus human baseline comparisons.
- Writing Quality: ⭐⭐⭐⭐ Task definitions, rubric tables, and case studies are all clear; detailed appendices.
- Value: ⭐⭐⭐⭐⭐ Quantitatively exposes the gap in professional humanities tasks; serves as a cautionary note for "LLMs for academic research"; the rubric framework is highly reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] TaxPraBen: A Scalable Benchmark for Structured Evaluation of LLMs in Chinese Real-World Tax Practice](taxpraben_a_scalable_benchmark_for_structured_evaluation_of_llms_in_chinese_real.md)
- [\[ACL 2026\] Exploring the Capability Boundaries of LLMs in Mastering of Chinese Chouxiang Language](exploring_the_capability_boundaries_of_llms_in_mastering_of_chinese_chouxiang_la.md)
- [\[ACL 2026\] BizCompass: Benchmarking the Reasoning Capabilities of LLMs in Business Knowledge and Applications](bizcompass_benchmarking_the_reasoning_capabilities_of_llms_in_business_knowledge.md)
- [\[ICLR 2026\] Sci2Pol: Evaluating and Fine-tuning LLMs' "Science-to-Policy Brief" Generation Capabilities](../../ICLR2026/llm_evaluation/sci2pol_evaluating_and_fine-tuning_llms_on_scientific-to-policy_brief_generation.md)
- [\[ACL 2026\] Are They Lovers or Friends? Evaluating LLMs' Social Reasoning in English and Korean Dialogues](are_they_lovers_or_friends_evaluating_llms39_social_reasoning_in_english_and_kor.md)

</div>

<!-- RELATED:END -->
