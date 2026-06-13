---
title: >-
  [Paper Note] TaxPraBen: A Scalable Benchmark for Structured Evaluation of LLMs in Chinese Real-World Tax Practice
description: >-
  [ACL 2026][LLM Evaluation][Tax Practice] This paper proposes TaxPraBen, the first LLM evaluation benchmark oriented towards Chinese tax practice. It includes 14 datasets with a total of 7.3K samples…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Tax Practice"
  - "LLM Evaluation Benchmark"
  - "Structured Evaluation"
  - "Chinese Taxation"
  - "Bloom's Taxonomy"
date: 2026-05-08
content_hash: d24ac93607d16a4b
---

# TaxPraBen: A Scalable Benchmark for Structured Evaluation of LLMs in Chinese Real-World Tax Practice

**Conference**: ACL 2026  
**arXiv**: [2604.08948](https://arxiv.org/abs/2604.08948)  
**Code**: [https://github.com/Yating-Chen/TaxPraBen](https://github.com/Yating-Chen/TaxPraBen)  
**Area**: Information Retrieval  
**Keywords**: Tax Practice, LLM Evaluation Benchmark, Structured Evaluation, Chinese Taxation, Bloom's Taxonomy

## TL;DR

This paper proposes TaxPraBen, the first LLM evaluation benchmark oriented towards Chinese tax practice. It includes 14 datasets with a total of 7.3K samples, covering three real-world scenarios: tax risk prevention and control, audit analysis, and tax planning. The authors design a scalable evaluation paradigm of "structured parsing—field alignment extraction—numerical and text matching." Evaluation of 19 LLMs reveals that closed-source and Chinese-optimized models perform better, while the domain-specific fine-tuned model YaYi2 shows limited improvement.

## Background & Motivation

**Background**: LLMs demonstrate excellent performance in general NLP tasks but still show significant deficiencies in highly specialized, knowledge-intensive, and regulation-driven fields such as taxation. Existing domain benchmarks like FinBen (finance), MedBench (medicine), and LAiW (law) have covered several vertical domains, but benchmarks for taxation are extremely scarce.

**Limitations of Prior Work**: (1) Existing tax-related evaluations (e.g., TaxBen) primarily focus on isolated NLP tasks (text classification, generation, reasoning) and fail to reflect the dual requirements of semantic understanding and numerical calculation in real tax practice. (2) There is a large gap between overseas tax data and Chinese tax management reality, making direct transfer difficult. (3) Some models perform well on isolated tasks but fail in real scenarios requiring the integration of semantic reasoning and numerical calculation, leading to inflated rankings.

**Key Challenge**: Tax practice requires LLMs to possess both policy semantic understanding and precise numerical calculation capabilities. Traditional NLP benchmarks, which evaluate tasks in isolation, cannot capture this composite demand, leading to an overestimation of actual model application capabilities.

**Goal**: Construct the first comprehensive evaluation benchmark for Chinese tax practice, covering the complete cognitive hierarchy from knowledge memory to understanding and then to application, filling the gap in tax practice scenario evaluation.

**Key Insight**: Starting from Bloom's Taxonomy, tax tasks are categorized into three levels: Knowledge Memory (KM), Knowledge Understanding (KU), and Knowledge Application (KA). Three practice tasks closely related to real work scenarios—tax risk prevention, audit analysis, and tax planning—are introduced.

**Core Idea**: Design a structured evaluation paradigm. Through a pipeline of "structured parsing → field alignment extraction → numerical and text mixed matching," end-to-end evaluation of LLMs' tax practice capabilities is achieved. This paradigm is scalable to other fields such as law, medicine, and finance.

## Method

### Overall Architecture

TaxPraBen consists of 14 datasets totaling 7.3K instances, covering 10 traditional application tasks and 3 innovative practice scenario tasks. Data originates from three pipelines: (A) Book data collection—OCR extraction from tax exam guides and tax planning casebooks; (B) Official document downloads—policies and regulations obtained from the State Taxation Administration; (C) Web data processing—crawling news, risk control reports, and audit cases from tax websites. All data underwent manual verification and ChatGPT-assisted labeling.

### Key Designs

1.  **Bloom's Taxonomy Task System**:
    *   Function: Systematically categorizes tax capability evaluation into three cognitive levels.
    *   Mechanism: KM (Knowledge Memory) tests the model's ability to accurately recite tax laws and policies (e.g., TaxRecite dataset); KU (Knowledge Understanding) tests the ability to identify key information and understand policy meanings (e.g., TaxSum, TaxTopic, TaxRead); KA (Knowledge Application) tests the integrated use of regulations and calculation methods in actual scenarios (e.g., TaxCalc, TaxSCQ, TaxMCQ).
    *   Design Motivation: Compared to benchmarks like TaxBen that focus only on isolated NLP tasks, organizing by cognitive levels allows for a systematic evaluation of the complete capability chain from "recitation" to "application."

2.  **Structured Evaluation Paradigm**:
    *   Function: Sophisticated end-to-end tax practice evaluation combining semantic accuracy and structural alignment.
    *   Mechanism: A unified JSON output protocol is defined, designing standardized output modes for three practice scenarios—TaxRisk (extracting risk points and solutions), TaxInspect (extracting criminal behaviors, charges, and penalties), and TaxPlan (generating strategies and calculating tax savings). Since open-source model output formats are often irregular, ChatGPT-3.5 serves as a structured parser to convert free text into standard JSON before automatic evaluation.
    *   Design Motivation: Real tax practice requires hybrid output emphasizing both semantic reasoning and numerical calculation, which traditional single metrics cannot fully evaluate.

3.  **Multi-source Data Fusion and Quality Control**:
    *   Function: Ensures the dataset reflects real tax scenarios rather than purely academic constructs.
    *   Mechanism: Three data pipelines cover exam questions, policies, and web cases. ChatGPT assists in structured information extraction and data augmentation, but all data must pass manual usability checks. Instruction labeling is completed by a professional tax team using a 4-point scale, with consistency ensured via Fleiss' Kappa and Krippendorff's Alpha.
    *   Design Motivation: Prior domain benchmarks often rely on public datasets and lack coverage of real application scenarios.

### Loss & Training

TaxPraBen is an evaluation benchmark and does not involve model training. Evaluation metrics are designed by output type: Accuracy/F1/Macro-F1 for classification tasks, BERTScore/BARTScore for generation tasks, and EM Accuracy for structured prediction. Mixed matching tasks use a weighted average of EM Accuracy and BERTScore.

## Key Experimental Results

### Main Results

**Zero-shot performance of 19 LLMs across tax cognitive levels**

| Model | KM (Memory) | KU (Understanding) | KA (Application) | Type |
| :--- | :--- | :--- | :--- | :--- |
| ERNIE-3.5 | 0.667 | 0.599 | 0.475 | Closed-source Chinese |
| Grok3 | 0.519 | 0.579 | 0.482 | Closed-source Multilingual |
| GPT-4o | 0.478 | 0.637 | 0.472 | Closed-source Multilingual |
| ChatGPT | 0.488 | 0.602 | 0.415 | Closed-source Multilingual |
| Qwen2.5 | 0.499 | 0.538 | 0.375 | Open-source Chinese |
| DeepSeek-R1 | 0.461 | 0.455 | 0.324 | Open-source Chinese |
| YaYi2 (Tax FT) | 0.485 | 0.307 | 0.239 | Domain Fine-tuned |
| Mistral-v0.3 | 0.400 | 0.277 | 0.114 | Open-source Multilingual |

### Ablation Study

| Dimension | Finding | Description |
| :--- | :--- | :--- |
| 0-shot vs 1-shot | 11/19 models improved with 1-shot | Some models like GLM4 actually declined |
| NLP-CLS Tasks | Most models performed poorly | Tax classification requires deep domain knowledge |
| NLP-REA Tasks | Difficult for all models | Multi-step numerical reasoning is the bottleneck |
| NLP-GEN Tasks | Relatively best performance | ERNIE-3.5 reached 0.670 in 0-shot |
| Chinese vs Multilingual LLMs | Chinese models generally superior | Language optimization is crucial for tax terminology |

### Key Findings

*   Knowledge Application (KA) is the most difficult level, with all models performing worst here, reflecting the dependence of tax practice on reasoning within economic activity contexts.
*   ERNIE-3.5 leads in KM tasks due to its knowledge-enhanced pre-training strategy, highlighting the value of domain knowledge integration.
*   YaYi2, despite tax data fine-tuning, performs worse than general LLMs, suggesting limited effectiveness when fine-tuning data volume and task coverage are insufficient.
*   Reasoning tasks (REA) are a challenge for all models, exposing systematic weaknesses in LLMs regarding numerical calculation and tax logic understanding.

## Highlights & Insights

*   This is the first systematic application of Bloom's Taxonomy to the evaluation of the Chinese tax domain, establishing a complete capability chain from memory to application.
*   The structured evaluation paradigm is highly transferable—the "structured parsing → field alignment → mixed matching" method can be directly applied to law, finance, and medicine, where hybrid "explanatory text + key numerical values" output is required.
*   Using ChatGPT-3.5 as a lightweight structured parser instead of fragile regex matching is a practical engineering trick that reduces evaluation sensitivity to output formats.

## Limitations & Future Work

*   Since data comes from the internet and public books, there is a risk of test data leakage—LLMs may have encountered some test content during pre-training.
*   The parameters of evaluated models are concentrated around 7B, lacking comparison with larger open-source models.
*   Automatic semantic similarity metrics may not fully reflect human judgment of answer quality.
*   Future work can expand to more complex scenarios such as multi-turn conversational tax consulting and tracking policy changes across years.

## Related Work & Insights

*   **vs TaxBen**: TaxBen focuses on isolated NLP tasks. TaxPraBen adds three real practice scenarios and proposes structured evaluation, which is closer to actual application.
*   **vs FinBen/LAiW**: Benchmarks for finance and law are more mature; TaxPraBen fills the gap in the tax domain, and its hybrid evaluation method can benefit these other fields.
*   **vs MMLU/C-Eval**: General benchmarks lack domain depth and cannot evaluate professional numerical reasoning capabilities.

## Rating

*   Novelty: ⭐⭐⭐⭐ First Chinese tax practice benchmark; the structured evaluation paradigm has general value.
*   Experimental Thoroughness: ⭐⭐⭐⭐ 19 models evaluated comprehensively, though missing comparisons with larger models.
*   Writing Quality: ⭐⭐⭐⭐ Clear structure with detailed descriptions of task classification and data construction.
*   Value: ⭐⭐⭐⭐ Fills the gap in tax evaluation; the structured evaluation paradigm is transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Beyond Itinerary Planning: A Real-World Benchmark for Multi-Turn and Tool-Using Travel Tasks](beyond_itinerary_planning-a_real-world_benchmark_for_multi-turn_and_tool-using_t.md)
- [\[ACL 2026\] Exploring the Capability Boundaries of LLMs in Mastering of Chinese Chouxiang Language](exploring_the_capability_boundaries_of_llms_in_mastering_of_chinese_chouxiang_la.md)
- [\[ACL 2026\] SCAN: Structured Capability Assessment and Navigation for LLMs](scan_structured_capability_assessment_and_navigation_for_llms.md)
- [\[ACL 2026\] Can LLMs Act as Historians? Evaluating Historical Research Capabilities of LLMs via the Chinese Imperial Examination](can_llms_act_as_historians_evaluating_historical_research_capabilities_of_llms_v.md)
- [\[ACL 2026\] AgentEval: DAG-Structured Step-Level Evaluation for Agentic Workflows with Error Propagation Tracking](agenteval_dag-structured_step-level_evaluation_for_agentic_workflows_with_error_.md)

</div>

<!-- RELATED:END -->
