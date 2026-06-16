---
title: >-
  [Paper Note] TaxPraBen: A Scalable Benchmark for Structured Evaluation of LLMs in Chinese Real-World Tax Practice
description: >-
  [ACL 2026][LLM Evaluation][Paper Note] Ours proposes TaxPraBen, the first LLM evaluation benchmark for Chinese tax practice, containing 7.3K samples across 14 datasets covering three real-world scenarios: tax risk prevention, inspection analysis, and tax planning. It designs a scalable "structured parsing—field alignment extraction—numerical and text matchi
tags:
  - ACL 2026
  - LLM Evaluation
date: 2026-05-08
content_hash: 6f0125e0e31c9e5b
---
# TaxPraBen: A Scalable Benchmark for Structured Evaluation of LLMs in Chinese Real-World Tax Practice

**Conference**: ACL 2026  
**arXiv**: [2604.08948](https://arxiv.org/abs/2604.08948)  
**Code**: [https://github.com/Yating-Chen/TaxPraBen](https://github.com/Yating-Chen/TaxPraBen)  
**Area**: Information Retrieval  
**Keywords**: Tax Practice, LLM Evaluation Benchmark, Structured Evaluation, Chinese Tax, Bloom's Taxonomy

## TL;DR

Ours proposes TaxPraBen, the first LLM evaluation benchmark for Chinese tax practice, containing 7.3K samples across 14 datasets covering three real-world scenarios: tax risk prevention, inspection analysis, and tax planning. It designs a scalable "structured parsing—field alignment extraction—numerical and text matching" evaluation paradigm. After evaluating 19 LLMs, it was found that closed-source and Chinese-optimized models perform better, while the tax-domain fine-tuned model YaYi2 shows limited improvement.

## Background & Motivation

**Background**: LLMs exhibit excellent performance in general NLP tasks but still show significant deficiencies in highly specialized, knowledge-intensive, and regulation-driven fields such as taxation. Existing domain benchmarks like FinBen (Finance), MedBench (Medicine), and LAiW (Law) have covered multiple vertical domains, but evaluation benchmarks for taxation are extremely scarce.

**Limitations of Prior Work**: (1) Existing tax-related evaluations (e.g., TaxBen) primarily focus on isolated NLP tasks (text classification, generation, reasoning) and fail to reflect the requirement for simultaneous semantic understanding and numerical calculation in real tax practice; (2) Overseas tax data differs significantly from Chinese tax management reality, making it difficult to transfer directly; (3) Some models perform well on isolated tasks but poorly in real-world scenarios requiring the integration of semantic reasoning and numerical calculation, resulting in inflated rankings.

**Key Challenge**: Tax practice requires LLMs to possess both policy semantic understanding and precise numerical calculation capabilities. Traditional NLP benchmarks fail to capture this composite requirement through isolated task evaluation, leading to an overestimation of the models' practical application abilities.

**Goal**: To construct the first comprehensive evaluation benchmark for Chinese tax practice, covering the complete cognitive hierarchy from knowledge memory to understanding and application, filling the gap in evaluating tax practice scenarios.

**Key Insight**: Starting from Bloom's Taxonomy, tax tasks are categorized into three levels: Knowledge Memory (KM), Knowledge Understanding (KU), and Knowledge Application (KA). Three practical tasks closely resembling real work scenarios—tax risk prevention, inspection analysis, and tax planning—are introduced.

**Core Idea**: Design a structured evaluation paradigm through a pipeline of "structured parsing → field alignment extraction → hybrid numerical and text matching" to achieve end-to-end evaluation of LLM tax practice capabilities. This paradigm is scalable to other fields such as law, medicine, and finance.

## Method

### Overall Architecture

The question TaxPraBen aims to answer is: Can an LLM handle real-world Chinese tax work? To this end, it organizes 14 datasets with a total of 7.3K instances around the "memorize—understand—apply" cognitive ladder, including 10 traditional application tasks and 3 practical scenario tasks. Data is aggregated through three pipelines: book OCR (tax examination guides and planning cases), official document downloads (State Taxation Administration policies and regulations), and web crawling (news, risk control reports, inspection cases), all manually verified and assisted by ChatGPT for annotation. The evaluation side uses a "structured parsing → field alignment extraction → hybrid numerical and text matching" paradigm to compress model free-text into standard JSON for automatic scoring.

### Key Designs

**1. Bloom's Taxonomy Task System: Evaluation Across Three Levels from "Memorization" to "Application"**

Focusing only on isolated NLP tasks, as in TaxBen, fails to identify where models struggle in real tax work. TaxPraBen utilizes Bloom's Taxonomy to decompose capabilities into three levels for assessment. Knowledge Memory (KM) tests the accurate recitation of tax laws and policy regulations, represented by the TaxRecite dataset. Knowledge Understanding (KU) tests the identification of key information from tax materials and the comprehension of policy meanings, including TaxSum, TaxTopic, and TaxRead. Knowledge Application (KA) requires models to integrate regulations and calculations in real scenarios, covering 10 datasets such as TaxCalc, TaxSCQ, and TaxMCQ. This complete chain from memory to application exposes weaknesses like "memorizing without calculating" that are masked by traditional benchmarks.

**2. Structured Evaluation Paradigm: Evaluating Semantic Accuracy and Structural Alignment Together**

The output of real tax practice is never a single answer but a hybrid of "explanatory text + key numerical values," which traditional single metrics cannot capture. TaxPraBen establishes a unified JSON output protocol and defines standardized patterns for three types of practice scenarios: TaxRisk extracts risk points and solutions; TaxInspect extracts criminal behaviors, charges, and penalty results; TaxPlan generates planning strategies and calculates tax savings. Since open-source model output formats vary, ChatGPT-3.5 is employed as a structural parser to convert free text into standardized JSON before automatic evaluation. This replaces fragile regular expression matching with a lightweight parsing step, making scoring insensitive to output formatting.

**3. Multi-source Data Fusion and Quality Control: Aligning Datasets with Reality Rather Than Academic Construction**

Existing domain benchmarks often rely on public datasets, which differ from real application scenarios. TaxPraBen uses three data pipelines to cover exam questions, policies, and web cases, attempting to reconstruct the material distribution faced by tax professionals. ChatGPT only serves an auxiliary role for structural information extraction and data augmentation; all data must still pass a human usability check. Instruction annotation is handled by a professional tax team using a 4-point scale, with annotation consistency confirmed by Fleiss' Kappa and Krippendorff's Alpha.

### Loss & Training

TaxPraBen is an evaluation benchmark and does not involve model training. Evaluation metrics are designed based on output types: Accuracy/F1/Macro-F1 for classification tasks, BERTScore/BARTScore for generation tasks, EM Accuracy for structured prediction, and a weighted average of EM Accuracy and BERTScore for hybrid matching tasks to account for both numerical precision and semantic fit.

## Key Experimental Results

### Main Results

**Zero-shot performance of 19 LLMs across tax cognitive levels**

| Model | KM (Memory) | KU (Understanding) | KA (Application) | Type |
|------|----------|----------|----------|------|
| ERNIE-3.5 | 0.667 | 0.599 | 0.475 | Closed-source Chinese |
| Grok3 | 0.519 | 0.579 | 0.482 | Closed-source Multilingual |
| GPT-4o | 0.478 | 0.637 | 0.472 | Closed-source Multilingual |
| ChatGPT | 0.488 | 0.602 | 0.415 | Closed-source Multilingual |
| Qwen2.5 | 0.499 | 0.538 | 0.375 | Open-source Chinese |
| DeepSeek-R1 | 0.461 | 0.455 | 0.324 | Open-source Chinese |
| YaYi2 (Tax fine-tuned) | 0.485 | 0.307 | 0.239 | Domain fine-tuned |
| Mistral-v0.3 | 0.400 | 0.277 | 0.114 | Open-source Multilingual |

### Ablation Study

| Dimension | Finding | Description |
|------|------|------|
| Zero-shot vs. One-shot | 11/19 models improved with one-shot | Some models like GLM4 actually declined |
| NLP-CLS Tasks | Most models performed poorly | Tax classification requires deep domain knowledge |
| NLP-REA Tasks | Difficult for all models | Multi-step numerical reasoning is a bottleneck |
| NLP-GEN Tasks | Relatively best performance | ERNIE-3.5 reached 0.670 in zero-shot |
| Chinese vs. Multilingual LLMs | Chinese models generally superior | Language optimization is critical for tax terminology |

### Key Findings

- Knowledge Application (KA) is the most difficult level, with all models performing worst here, reflecting tax practice's dependence on economic activity context reasoning.
- ERNIE-3.5 leads in KM tasks due to its knowledge-enhanced pre-training strategy, demonstrating the value of domain knowledge fusion.
- Although YaYi2 was fine-tuned on tax data, it underperforms compared to general LLMs, indicating limited effectiveness when fine-tuning data volume and task coverage are insufficient.
- Reasoning tasks (REA) are a challenge for all models, exposing systematic weaknesses of LLMs in numerical calculation and tax logic understanding.

## Highlights & Insights

- Ours marks the first systematic application of Bloom's Taxonomy to the evaluation of the Chinese tax domain, establishing a complete evaluation chain from memory to application.
- The structured evaluation paradigm is highly transferable—the "structured parsing → field alignment → hybrid matching" method can be directly applied to domains such as law, finance, and medicine that require hybrid outputs of "explanatory text + key values."
- Using ChatGPT-3.5 as a lightweight structural parser to replace fragile regular expression matching is a practical engineering trick that reduces evaluation sensitivity to output formats.

## Limitations & Future Work

- Data is sourced from the internet and public books, posing a risk of test data leakage—LLMs may have encountered some test content during pre-training.
- The evaluated models are mostly around 7B parameters, and comparison with larger open-source models is lacking.
- Automatic semantic similarity metrics may not fully reflect human judgment of answer quality.
- Future work could extend to more complex scenarios such as multi-turn conversational tax consultation and tracking cross-year policy changes.

## Related Work & Insights

- **vs. TaxBen**: TaxBen focuses on isolated NLP tasks, whereas TaxPraBen adds three real-world practice scenarios and proposes structured evaluation, making it more aligned with actual applications.
- **vs. FinBen/LAiW**: Benchmarks in finance and law are relatively mature; TaxPraBen fills the gap in the tax domain, and its hybrid evaluation method can benefit these other fields.
- **vs. MMLU/C-Eval**: General benchmarks lack domain depth and cannot evaluate professional numerical reasoning capabilities.

## Rating

- Novelty: ⭐⭐⭐⭐ First Chinese tax practice benchmark; structured evaluation paradigm has general value.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation of 19 models, though lacking comparisons with larger models.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, with detailed descriptions of task classification and data construction.
- Value: ⭐⭐⭐⭐ Fills a gap in tax evaluation; structured evaluation paradigm is transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Beyond Itinerary Planning: A Real-World Benchmark for Multi-Turn and Tool-Using Travel Tasks](beyond_itinerary_planning-a_real-world_benchmark_for_multi-turn_and_tool-using_t.md)
- [\[ACL 2026\] Can LLMs Act as Historians? Evaluating Historical Research Capabilities of LLMs via the Chinese Imperial Examination](can_llms_act_as_historians_evaluating_historical_research_capabilities_of_llms_v.md)
- [\[ACL 2025\] RuleArena: A Benchmark for Rule-Guided Reasoning with LLMs in Real-World Scenarios](../../ACL2025/llm_evaluation/rulearena_rule_guided_reasoning.md)
- [\[ACL 2026\] AgentEval: DAG-Structured Step-Level Evaluation for Agentic Workflows with Error Propagation Tracking](agenteval_dag-structured_step-level_evaluation_for_agentic_workflows_with_error_.md)
- [\[ACL 2026\] SCAN: Structured Capability Assessment and Navigation for LLMs](scan_structured_capability_assessment_and_navigation_for_llms.md)

</div>

<!-- RELATED:END -->
