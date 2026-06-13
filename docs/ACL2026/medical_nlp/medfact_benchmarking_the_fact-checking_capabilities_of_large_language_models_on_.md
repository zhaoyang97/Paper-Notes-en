---
title: >-
  [Paper Note] MedFact: Benchmarking the Fact-Checking Capabilities of Large Language Models on Chinese Medical Texts
description: >-
  [ACL2026][Medical NLP][Chinese medical texts] MedFact constructs an expert-annotated fact-checking benchmark covering real-world Chinese medical texts. Evaluations of 20 LLMs demonstrate that current models can easily ju…
tags:
  - "ACL2026"
  - "Medical NLP"
  - "Chinese medical texts"
  - "fact-checking"
  - "error localization"
  - "medical LLM evaluation"
  - "over-criticism"
date: 2026-05-08
content_hash: d6b57899d5a3676f
---

# MedFact: Benchmarking the Fact-Checking Capabilities of Large Language Models on Chinese Medical Texts

**Conference**: ACL2026  
**arXiv**: [2509.12440](https://arxiv.org/abs/2509.12440)  
**Code**: Project Page https://iflytek-medical-southchina.github.io/MedFact/  
**Area**: Medical NLP  
**Keywords**: Chinese medical texts, fact-checking, error localization, medical LLM evaluation, over-criticism

## TL;DR
MedFact constructs an expert-annotated fact-checking benchmark covering real-world Chinese medical texts. Evaluations of 20 LLMs demonstrate that current models can easily judge "whether an error exists" but struggle to precisely locate errors. While RAG helps, multi-agent collaboration and test-time scaling tend to amplify "over-criticism."

## Background & Motivation
**Background**: Large Language Models (LLMs) have entered clinical applications such as clinical QA, auxiliary diagnosis, patient assessment, medical text classification, and medical RAG. Real-world systems often integrate internet-based or internal medical texts into Retrieval-Augmented Generation (RAG) workflows. Consequently, models must not only answer medical exam questions but also determine the reliability of medical content itself.

**Limitations of Prior Work**: Existing medical evaluations focus largely on QA, relation extraction, or error correction in clinical notes. VeriFact uses synthetic clinical texts, and MEDEC is primarily oriented toward error detection in clinical notes. These setups fail to cover diverse text forms encountered in real-world deployments, such as medical encyclopedias, popular science articles, QA communities, and forged medical rumors.

**Key Challenge**: Medical fact-checking requires both broad medical knowledge and the ability to locate errors in specific segments. A model might judge a passage as "problematic" based on overall tone but misidentify a correct sentence as the source of the error. In medical scenarios, this behavior of "correct conclusion, wrong reasoning" remains unsafe.

**Goal**: The authors aim to construct an uncontaminated, realistic, diverse, and difficulty-stratified Chinese medical fact-checking dataset. This dataset is used to systematically evaluate LLMs on veracity classification, error localization, benefits of RAG, side effects of reasoning strategies, and cross-lingual performance.

**Key Insight**: Instead of scraping data directly from the open web, the paper utilizes non-public medical encyclopedias, consultation platforms, QA pages, and forum content from commercial partners to reduce pre-training contamination risks. AI filtering, physician annotation, and hard sample mining are then applied to create a benchmark capable of truly distinguishing model capabilities.

**Core Idea**: Replace synthetic medical correction tasks with "real-world medical text + physician-annotated error spans + hard sample mining," advancing medical fact-checking evaluation from coarse-grained veracity judgment to fine-grained error localization.

## Method
MedFact is essentially a benchmark construction and evaluation pipeline. It first filters candidate samples suitable for fact-checking from large-scale real Chinese medical texts. Physicians then determine the correctness of the texts, identify where errors occur, and provide corrections. Finally, the test set is derived through model evaluation and hard sample mining. Unlike standard QA benchmarks, the key to MedFact is not asking the model a medical question, but providing a potentially erroneous medical text and requiring it to perform both "veracity detection" and "error localization."

### Overall Architecture
The input consists of 27,116 copyright-compliant real Chinese medical texts from internal encyclopedias, consultation platforms, QA pages, and user forums. The system uses a filtering suite composed of 7 strong models to filter texts based on dimensions such as "too simple, too obscure, or malformed." Subsequently, three rounds of physician feedback improve the RAG-enhanced few-shot filtering prompts, compressing the candidates into 6,405 high-quality texts.

During the annotation phase, 3 licensed medical professionals provide binary veracity labels for each text. If a text is erroneous, they annotate precise error spans and suggested corrections. The authors then perform hard sample mining, deduplication of similar samples, and de-identification rewriting. Finally, physicians review the samples to produce 2,116 MedFact instances.

The evaluation phase includes two tasks: Veracity Classification (VC), judging if a text contains factual errors; and Error Localization (EL), identifying the error span in incorrect texts. The authors evaluate 20 models under strategies such as zero-shot, CoT, MedPrompt, RAG, MAD, MDAgents, and budget forcing. GPT-4o serves as an automated judge for EL; the Cohen's $\kappa$ between GPT-4o and physicians on a random 10% sample is $0.870$.

### Key Designs
1. **AI-Human Hybrid Data Construction**:
    - **Function**: Filters samples from real medical texts that possess medical depth and are suitable for fact-checking evaluation.
    - **Mechanism**: 7 strong models vote to filter based on criteria like "simple, obscure, malformed." After three rounds of physician sampling and review, misjudged samples are added to specialized RAG few-shot prompts. The filtering acceptance rate dropped from 67.69% to 37.00%, and finally to 23.62%, while the agreement rate between models and physicians increased to 96.40%.
    - **Design Motivation**: Pure manual screening is too costly, while pure model screening retains low-quality or biased samples. Iterative feedback ensures data construction is scalable without deviating from professional medical judgment.

2. **Fine-grained Error Taxonomy and Hard Sample Mining**:
    - **Function**: Ensures the benchmark measures more than "basic medical common sense," covering various error mechanisms and difficulty levels.
    - **Mechanism**: Errors are categorized into medical and non-medical types, further subdivided into 8 categories: concepts, terminology, temporal, citation sources, bias, general facts, etc. The authors also remove samples that all models easily correctly identified and use similarity filtering for deduplication, retaining 2,116 instances that better differentiate model performance.
    - **Design Motivation**: Errors in medical texts are often subtle. Without controlling for error types and difficulty, evaluations would be dominated by exaggerated rumors or simple common sense, failing to reflect real deployment risks.

3. **VC + EL Dual-Task Evaluation**:
    - **Function**: Simultaneously checks if the model knows a text is wrong and if it knows exactly where the error is.
    - **Mechanism**: VC calculates Precision, Recall, and F1 using erroneous texts as the positive class. EL examines the identified error spans only when a text is judged as erroneous; a predicted span must match the gold error source to be considered correct.
    - **Design Motivation**: In medical fact-checking, simply flagging a whole paragraph is insufficient. If a model misidentifies correct medical information as an error, subsequent human review or automated correction will be misled.

### Loss & Training
The paper does not train new models; the core contribution is the benchmark construction and evaluation protocol. Inference settings include zero-shot and CoT prompting, with additional tests on MedPrompt, RAG, MAD, MDAgents, and budget forcing. The RAG knowledge base consists of 6,405 expert-annotated source texts retained during construction. Primary metrics are Precision, Recall, and F1 for VC and EL.

## Key Experimental Results

### Main Results
| Model / Setting | VC F1 | EL F1 | Key Information |
| :--- | :--- | :--- | :--- |
| Human | 0.7521 | 0.7012 | Average performance of 3 medical professionals |
| XiaoYi zero-shot | 0.7126 | 0.6758 | One of the strongest medical-specific models |
| XiaoYi CoT | 0.7061 | 0.6858 | Highest EL F1 reported in the paper |
| Doubao-Seed-1.6-thinking zero-shot | 0.7139 | 0.6712 | Strong performance among general models |
| Doubao-Seed-1.6-thinking CoT | 0.7050 | 0.6786 | CoT improves EL but slightly decreases VC |
| DeepSeek-R1 zero-shot | 0.6847 | 0.6051 | High Recall, but localization remains weak |

### Strategy Comparison
| Model / Strategy | VC F1 | EL F1 | Phenomenon |
| :--- | :--- | :--- | :--- |
| DeepSeek-R1 | 0.6847 | 0.6051 | Zero-shot baseline |
| DeepSeek-R1 + RAG top-3 | 0.7369 | 0.6820 | Task-relevant knowledge significantly improves results |
| DeepSeek-R1 + MAD | 0.6829 | 0.6017 | Recall increases but Precision decreases |
| DeepSeek-R1 + MDAgents | 0.6965 | 0.6233 | Slight improvement, but over-criticism remains prominent |
| XiaoYi | 0.7126 | 0.6758 | Zero-shot baseline |
| XiaoYi + RAG top-3 | 0.7484 | 0.7051 | Single metric exceeds human EL F1, but depends on homologous RAG |
| XiaoYi + MAD | 0.6996 | 0.6831 | Multi-agent setups worsen Precision |
| XiaoYi + MDAgents | 0.7059 | 0.6284 | EL F1 significantly lower than RAG top-3 |

### Key Findings
- The dataset scale is 2,116 items, containing 1,058 correct texts and 1,058 texts with a single factual error. Medical errors account for 89.41% of erroneous samples, with conceptual errors being the most frequent (52.65%).
- Models' EL performance is consistently weaker than VC, indicating that "judging an error exists" is much easier than "locating the error." The highest EL F1 of 0.6858 is still lower than the human performance of 0.7012.
- RAG benefits are highly dependent on retrieval source relevance. Homologous top-3 RAG can raise XiaoYi's EL F1 to 0.7051, but authoritative medical data may actually decrease Recall if it does not align closely with the task.
- Multi-agent collaboration and test-time scaling lead to "over-criticism": DeepSeek-R1 + MAD's VC Precision dropped from 0.5488 to 0.5310, while Recall rose from 0.9101 to 0.9565, indicating a tendency to mislabel correct texts as erroneous.
- Cross-lingual experiments show F1 improvements on English translated versions (e.g., Gemini 2.5 Pro from 0.6223 to 0.6745), but the over-criticism trend (high Recall/low Precision) persists.

## Highlights & Insights
- The value of MedFact lies in decomposing medical fact-checking into two levels: veracity classification and error localization. Many models appear near-usable on VC, but EL results reveal a lack of fine-grained medical knowledge.
- "Over-criticism" is one of the most insightful findings. Longer reasoning and more agents do not necessarily improve safety. In fact-checking, additional deliberation may generate far-fetched erroneous hypotheses, ultimately sacrificing Precision.
- The comparison between homologous and authoritative RAG is practical. Medical systems cannot simply dump "authoritative materials" into a retrieval library; they must ensure retrieved content is highly relevant to the claims being checked, or models may be misled by mismatched evidence.
- Hard sample mining in data construction is worth migrating to other high-risk domains. Tasks in law, finance, and drug safety similarly require moving from "questions models can all answer" to "questions where models easily make confident mistakes."

## Limitations & Future Work
- MedFact focuses on Chinese and the Chinese medical context; conclusions may not directly generalize to other languages, medical systems, or clinical norms.
- EL uses GPT-4o as an automated judge; while it shows high agreement with physicians, it may still harbor biases or instability. High-risk samples should involve expert review in the future.
- The benchmark reflects medical knowledge at the time of construction. As guidelines and clinical evidence update, some factual labels may become obsolete, necessitating a dynamic update mechanism.
- Although the dataset is de-identified and restricted to research use, erroneous medical texts still carry a risk of misuse. Future releases should continue to use license constraints and safety instructions.

## Related Work & Insights
- **vs VeriFact**: VeriFact focuses on verifying the factuality of synthetic clinical texts against structured EHRs. MedFact uses real Chinese medical texts, covering more writing styles and scenarios.
- **vs MEDEC**: MEDEC is oriented toward error detection and correction in clinical notes. MedFact emphasizes internet and encyclopedia-style medical content, suitable for evaluating medical RAG and content moderation systems.
- **vs SimpleQA / OpenFactCheck**: General factuality benchmarks measure open-domain factuality but lack medical terminology, treatment boundaries, and error span annotations. MedFact specializes evaluation for medical knowledge-intensive tasks.
- **Insights**: For high-risk RAG systems, evaluation should not focus solely on final answer accuracy but should also include diagnostic metrics such as "evidence relevance," "over-questioning of correct content," and "error localization precision."

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ Solid combination of real-world Chinese medical fact-checking and error localization. The core problem is important, though the overall framework follows standard benchmark construction and evaluation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 20 models, multiple prompting/RAG/multi-agent strategies, cross-lingual analysis, and contamination analysis.
- **Writing Quality**: ⭐⭐⭐⭐☆ Data workflows and error analyses are clear. some large tables are information-dense and require careful reading.
- **Value**: ⭐⭐⭐⭐⭐ High reference value for medical LLMs, medical RAG, and fact-checking systems, particularly in warning that "stronger reasoning" does not simply equate to "higher reliability."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MHGraphBench: Knowledge Graph-Grounded Benchmarking of Mental Health Knowledge in Large Language Models](mhgraphbench_knowledge_graph-grounded_benchmarking_of_mental_health_knowledge_in.md)
- [\[ACL 2026\] Beyond the Leaderboard: Rethinking Medical Benchmarks for Large Language Models](beyond_the_leaderboard_rethinking_medical_benchmarks_for_large_language_models.md)
- [\[ACL 2026\] Text-Attributed Knowledge Graph Enrichment with Large Language Models for Medical Concept Representation](text-attributed_knowledge_graph_enrichment_with_large_language_models_for_medica.md)
- [\[ACL 2026\] RePrompT: Recurrent Prompt Tuning for Integrating Structured EHR Encoders with Large Language Models](reprompt_recurrent_prompt_tuning_for_integrating_structured_ehr_encoders_with_la.md)
- [\[ACL 2026\] MHSafeEval: Role-Aware Interaction-Level Evaluation of Mental Health Safety in Large Language Models](mhsafeeval_role-aware_interaction-level_evaluation_of_mental_health_safety_in_la.md)

</div>

<!-- RELATED:END -->
