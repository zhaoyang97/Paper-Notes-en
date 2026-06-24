---
title: >-
  [Paper Note] TaxPraBen: A Scalable Benchmark for Structured Evaluation of LLMs in Chinese Real-World Tax Practice
description: >-
  [ACL 2026][LLM Evaluation][Tax Practice] This paper proposes TaxPraBen, the first LLM evaluation benchmark for Chinese tax practice, consisting of 14 datasets with $7.3K$ samples covering three real-world scenarios: tax risk prevention, audit analysis, and tax planning. It designs a scalable evaluation paradigm of "structured parsing—field alignment extraction—numerical and text matching." Evaluations of 19 LLMs show that closed-source and Chinese-optimized models perform bet…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Tax Practice"
  - "LLM Evaluation Benchmark"
  - "Structured Evaluation"
  - "Chinese Taxation"
  - "Bloom’s Taxonomy"
date: 2026-05-08
content_hash: a7c1429822172e2a
---

# TaxPraBen: A Scalable Benchmark for Structured Evaluation of LLMs in Chinese Real-World Tax Practice

**Conference**: ACL 2026  
**arXiv**: [2604.08948](https://arxiv.org/abs/2604.08948)  
**Code**: [https://github.com/Yating-Chen/TaxPraBen](https://github.com/Yating-Chen/TaxPraBen)  
**Area**: Information Retrieval  
**Keywords**: Tax Practice, LLM Evaluation Benchmark, Structured Evaluation, Chinese Taxation, Bloom’s Taxonomy

## TL;DR

This paper proposes TaxPraBen, the first LLM evaluation benchmark for Chinese tax practice, consisting of 14 datasets with $7.3K$ samples covering three real-world scenarios: tax risk prevention, audit analysis, and tax planning. It designs a scalable evaluation paradigm of "structured parsing—field alignment extraction—numerical and text matching." Evaluations of 19 LLMs show that closed-source and Chinese-optimized models perform better, while the tax-domain fine-tuned model YaYi2 shows limited improvement.

## Background & Motivation

**Background**: LLMs exhibit excellent performance on general NLP tasks but still show significant deficiencies in highly specialized, knowledge-intensive, and regulation-driven fields like taxation. Existing domain benchmarks such as FinBen (finance), MedBench (medicine), and LAiW (law) cover various vertical domains, but benchmarks for tax practice are extremely scarce.

**Limitations of Prior Work**: (1) Existing tax-related evaluations (e.g., TaxBen) primarily focus on isolated NLP tasks (text classification, generation, reasoning) and fail to reflect the dual requirements of semantic understanding and numerical calculation in real tax practice. (2) Overseas tax data differs significantly from Chinese tax management, making direct transfer difficult. (3) Some models perform well on isolated tasks but poorly in real scenarios requiring the simultaneous integration of semantic reasoning and numerical calculation, leading to inflated rankings.

**Key Challenge**: Tax practice requires LLMs to possess both policy semantic understanding and precise numerical calculation capabilities. Traditional NLP benchmarks fail to capture this composite demand through isolated task assessments, leading to an overestimation of actual application capabilities.

**Goal**: To build the first comprehensive evaluation benchmark for Chinese tax practice, covering the complete cognitive hierarchy from knowledge memory to understanding and application, filling the gap in evaluating tax practice scenarios.

**Key Insight**: Starting from Bloom’s Taxonomy, tax tasks are categorized into three levels: Knowledge Memory (KM), Knowledge Understanding (KU), and Knowledge Application (KA). Three practical tasks closely resembling real work scenarios—tax risk prevention, audit analysis, and tax planning—are introduced.

**Core Idea**: A structured evaluation paradigm is designed through a pipeline of "structured parsing → field alignment extraction → numerical and text hybrid matching" to achieve end-to-end evaluation of LLM tax practice capabilities. This paradigm is scalable to other fields such as law, medicine, and finance.

## Method

### Overall Architecture

TaxPraBen aims to answer: Can an LLM handle real Chinese tax work? It organizes 14 datasets with $7.3K$ instances around a cognitive ladder of "recite—understand—apply," including 10 traditional application tasks and 3 realistic practice scenario tasks. Data is aggregated through three pipelines—book OCR (tax exam guides and planning cases), official document downloads (State Taxation Administration policies and regulations), and web crawling (news, risk control reports, audit cases), all manually verified and annotated with ChatGPT assistance. The evaluation side uses a paradigm of "structured parsing → field alignment extraction → numerical and text hybrid matching" to compress free text into standard JSON for automatic scoring.

### Key Designs

**1. Bloom’s Taxonomy Task System: Evaluation Across Three Levels from "Reciting" to "Applying"**

Focusing only on isolated NLP tasks, as TaxBen does, fails to identify where a model falters in real tax work. TaxPraBen utilizes Bloom’s Taxonomy to decompose capabilities into three levels for targeted stress testing. Knowledge Memory (KM) tests the accurate recitation of tax laws and policies (e.g., TaxRecite). Knowledge Understanding (KU) tests identifying key information and understanding policy meanings (e.g., TaxSum, TaxTopic, TaxRead). Knowledge Application (KA) requires models to synthetically apply regulations and calculations in real scenarios, covering 10 datasets such as TaxCalc, TaxSCQ, and TaxMCQ. This complete chain exposes weaknesses like "reciting without calculating" that are hidden by traditional benchmarks.

**2. Structured Evaluation Paradigm: Simultaneous Assessment of Semantic Accuracy and Structural Alignment**

Outputs in real tax practice are never single answers but a mixture of "explanatory text + key numerical values," which traditional single metrics cannot capture. TaxPraBen defines a unified JSON output protocol and standardized schemas for three practice scenarios—TaxRisk (extracting risks and solutions), TaxInspect (extracting criminal behaviors, charges, and penalties), and TaxPlan (generating strategies and calculating tax savings). Given the inconsistent output formats of open-source models, ChatGPT-3.5 is employed as a structured parser to convert free text into normative JSON before automatic evaluation. This replaces fragile regex matching with a lightweight parsing step, making scoring insensitive to output formatting.

**3. Multi-source Data Fusion and Quality Control: Realistic Data Over Academic Construction**

Existing domain benchmarks often rely on public datasets distant from real applications. TaxPraBen uses three data pipelines covering exam questions, policy regulations, and web cases to restore the material distribution faced by tax professionals. ChatGPT serves only an auxiliary role in structured information extraction and data augmentation; all data must pass manual usability checks. Instruction annotation is handled by professional tax teams using a 4-point scale, with consistency confirmed by Fleiss' Kappa and Krippendorff's Alpha to ensure the dataset is both realistic and reliable.

### Loss & Training

TaxPraBen is an evaluation benchmark and does not involve model training. Evaluation metrics are designed based on output types: Accuracy/F1/Macro-F1 for classification, BERTScore/BARTScore for generation, EM Accuracy for structured prediction, and a weighted average of EM Accuracy and BERTScore for hybrid matching tasks to balance numerical precision and semantic fit.

## Key Experimental Results

### Main Results

**Zero-shot Performance of 19 LLMs Across Tax Cognitive Levels**

| Model | KM (Memory) | KU (Understanding) | KA (Application) | Type |
|-------|----------|----------|----------|------|
| ERNIE-3.5 | 0.667 | 0.599 | 0.475 | Closed Chinese |
| Grok3 | 0.519 | 0.579 | 0.482 | Closed Multilingual |
| GPT-4o | 0.478 | 0.637 | 0.472 | Closed Multilingual |
| ChatGPT | 0.488 | 0.602 | 0.415 | Closed Multilingual |
| Qwen2.5 | 0.499 | 0.538 | 0.375 | Open Chinese |
| DeepSeek-R1 | 0.461 | 0.455 | 0.324 | Open Chinese |
| YaYi2 (Tax-FT) | 0.485 | 0.307 | 0.239 | Domain FT |
| Mistral-v0.3 | 0.400 | 0.277 | 0.114 | Open Multilingual |

### Ablation Study

| Dimension | Finding | Description |
|------|------|------|
| Zero-shot vs. One-shot | 11/19 models improved with One-shot | Some models like GLM4 actually declined |
| NLP-CLS Tasks | Most models performed poorly | Tax classification requires deep domain knowledge |
| NLP-REA Tasks | Difficult for all models | Multi-step numerical reasoning is a bottleneck |
| NLP-GEN Tasks | Relatively best performance | ERNIE-3.5 reached 0.670 in zero-shot |
| Chinese vs. Multilingual LLMs | Chinese models generally superior | Language optimization is crucial for tax terminology |

### Key Findings

- Knowledge Application (KA) is the most difficult level, with all models performing worst here, reflecting tax practice's dependence on economic activity context reasoning.
- ERNIE-3.5 leads in KM tasks due to knowledge-enhanced pre-training strategies, demonstrating the value of domain knowledge fusion.
- Although fine-tuned on tax data, YaYi2 performs worse than general LLMs, indicating limited effectiveness when fine-tuning data volume and task coverage are insufficient.
- Reasoning tasks (REA) are a challenge for all models, exposing systematic weaknesses in numerical calculation and tax logic understanding in LLMs.

## Highlights & Insights

- Systematically applied Bloom's Taxonomy to the Chinese tax evaluation for the first time, establishing a complete competency assessment chain from memory to application.
- The structured evaluation paradigm is highly transferable—the "structured parsing → field alignment → hybrid matching" method can be applied to law, finance, and medicine fields requiring "text + numerical" outputs.
- Using ChatGPT-3.5 as a lightweight structured parser instead of fragile regex matching is a practical engineering trick that reduces evaluation sensitivity to output formats.

## Limitations & Future Work

- Data is sourced from the internet and public books, posing a risk of test data leakage—LLMs might have encountered parts of the test content during pre-training.
- Evaluated models are concentrated around 7B parameters; comparison with larger open-source models is lacking.
- Automatic semantic similarity metrics may not fully reflect human judgment of answer quality.
- Future work could extend to more complex scenarios like multi-turn conversational tax consulting and tracking policy changes across fiscal years.

## Related Work & Insights

- **vs. TaxBen**: TaxBen focuses on isolated NLP tasks, while TaxPraBen adds three real practice scenarios and proposes structured evaluation, making it closer to actual applications.
- **vs. FinBen/LAiW**: Finance and law benchmarks are relatively mature; TaxPraBen fills the gap in the tax domain, and its hybrid evaluation method can benefit these other fields.
- **vs. MMLU/C-Eval**: General benchmarks lack domain depth and cannot evaluate professional numerical reasoning capabilities.

## Rating

- Novelty: ⭐⭐⭐⭐ First Chinese tax practice benchmark; structured evaluation paradigm has general value.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation of 19 models, though lacking comparison with larger models.
- Writing Quality: ⭐⭐⭐⭐ Clear structure; detailed descriptions of task classification and data construction.
- Value: ⭐⭐⭐⭐ Fills a gap in tax evaluation; structured evaluation paradigm is transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Beyond Itinerary Planning: A Real-World Benchmark for Multi-Turn and Tool-Using Travel Tasks](beyond_itinerary_planning-a_real-world_benchmark_for_multi-turn_and_tool-using_t.md)
- [\[ACL 2026\] Can LLMs Act as Historians? Evaluating Historical Research Capabilities of LLMs via the Chinese Imperial Examination](can_llms_act_as_historians_evaluating_historical_research_capabilities_of_llms_v.md)
- [\[ACL 2025\] RuleArena: A Benchmark for Rule-Guided Reasoning with LLMs in Real-World Scenarios](../../ACL2025/llm_evaluation/rulearena_rule_guided_reasoning.md)
- [\[ACL 2026\] SCAN: Structured Capability Assessment and Navigation for LLMs](scan_structured_capability_assessment_and_navigation_for_llms.md)
- [\[ACL 2026\] AgentEval: DAG-Structured Step-Level Evaluation for Agentic Workflows with Error Propagation Tracking](agenteval_dag-structured_step-level_evaluation_for_agentic_workflows_with_error_.md)

</div>

<!-- RELATED:END -->
