---
title: >-
  [Paper Note] MHGraphBench: Knowledge Graph-Grounded Benchmarking of Mental Health Knowledge in Large Language Models
description: >-
  [ACL2026][Medical NLP][Mental Health] MHGraphBench automatically constructs 9 types of multiple-choice tasks from the mental health subgraph of PrimeKG…
tags:
  - "ACL2026"
  - "Medical NLP"
  - "Mental Health"
  - "Knowledge Graph"
  - "PrimeKG"
  - "Relation Judgment"
  - "Two-hop Reasoning"
date: 2026-05-08
content_hash: 9f708c28736a880a
---

# MHGraphBench: Knowledge Graph-Grounded Benchmarking of Mental Health Knowledge in Large Language Models

**Conference**: ACL2026  
**arXiv**: [2605.15589](https://arxiv.org/abs/2605.15589)  
**Code**: None (cache repository not provided)  
**Area**: Medical NLP  
**Keywords**: Mental Health, Knowledge Graph, PrimeKG, Relation Judgment, Two-hop Reasoning

## TL;DR
MHGraphBench automatically constructs 9 types of multiple-choice tasks from the mental health subgraph of PrimeKG, finding that while LLMs achieve near-perfect scores in entity recognition, they remain significantly deficient in drug-disease relation judgment, contraindication boundaries, and two-hop KG reasoning.

## Background & Motivation
**Background**: LLMs are being utilized for medical and mental health-related tasks, including clinical Q&A, consultation assistance, diagnostic suggestions, and knowledge retrieval. Mental health scenarios particularly rely on heterogeneous biomedical knowledge, such as disease associations, drug indications/contraindications, phenotypes, exposure factors, and gene-protein relationships.

**Limitations of Prior Work**: Many medical benchmarks provide broad average accuracy, making it difficult to discern whether a model truly masters structured knowledge related to mental health. Mental health evaluations also frequently focus on diagnosis, consultation quality, or trustworthiness rather than verifiable knowledge graph relation boundaries.

**Key Challenge**: Models may recognize that "Anxiety is a disease" or "A certain substance is a drug," but this does not imply they can determine if a drug is an indication, contraindication, or off-label use for a specific mental disorder, or if it is absent from the graph entirely. This gap from recognition to structured judgment is critical in medical scenarios.

**Goal**: The authors aim to construct a KG-grounded benchmark using a verifiable mental health subgraph from PrimeKG to evaluate LLMs on entity recognition, relation judgment, two-hop reasoning, evidence utilization, and graph coverage.

**Key Insight**: Instead of directly evaluating clinical safety, the paper restricts the problem to "whether the model is consistent with a curated mental health slice of PrimeKG." This boundary makes the benchmark reproducible and interpretable while avoiding over-interpretation of KG results as actual clinical advice.

**Core Idea**: Starting from 42 mental illness seed nodes, a mental health subgraph is extracted, and KG triples are automatically converted into multiple-choice QA. Controlled negative sampling and coverage metrics are used to measure the weaknesses of models in structured mental health knowledge.

## Method
The workflow of MHGraphBench is divided into three steps: defining the mental health domain boundary, extracting the subgraph from PrimeKG, and generating multiple-choice questions from the subgraph. Its key feature is that all answers are supported by KG triples, and negative examples are generated through type matching and "not in subgraph" constraints, allowing every question to be traced back to the graph structure.

### Overall Architecture
The authors manually curated 44 high-precision mental illness candidate seeds, retaining 42 final seeds after removing 2 unsuitable nodes. Based on these seeds, 1-hop seed-touching edges were extracted from PrimeKG, retaining only 7 types of clinically relevant relations: disease_protein, contraindication, indication, off-label use, disease_disease, disease_phenotype_positive, and exposure_disease. After direction normalization, deduplication of symmetric relations, and canonicalization, 4,621 unique triples, 1,847 entities, and 7 relation types were obtained.

In the QA generation stage, the system converts graph facts into 9 task families: Entity Typing, Entity Clustering, Fact Checking, Relation Typing, Relation Prediction, Two-hop Verification, Two-hop Selection, and two evidence-augmented two-hop tasks. All tasks use a letter-only multiple-choice interface, and binary classification tasks utilize A/B instead of Yes/No to reduce lexical bias.

### Key Designs
1.  **PrimeKG Mental Health Subgraph Extraction**:
    - **Function**: Defines a reproducible scope of biomedical knowledge for mental health.
    - **Mechanism**: Extracts 1-hop edges from 42 mental illness seeds, retaining only 7 clinically relevant relations, and fixing relation directions to a schema such as disease $\rightarrow$ gene/protein, drug $\rightarrow$ disease, disease $\rightarrow$ disease, etc. Lexicographical deduplication is performed for symmetric relations like disease_disease.
    - **Design Motivation**: The scope of mental health knowledge is too broad for open-ended questions to be verifiable; the KG subgraph provides structured, traceable gold labels.

2.  **Nine KG-to-QA Task Types**:
    - **Function**: Evaluates entity recognition, relation judgment, and short-chain reasoning respectively.
    - **Mechanism**: ET/EC measures entity typing and clustering; FC measures if a triple is supported by the subgraph; RT measures the relation schema; RP classifies between indication/contraindication/off-label use/none; R1/R2 check compositional reasoning through a two-hop Drug A $\rightarrow$ Disease B $\rightarrow$ Disease C structure.
    - **Design Motivation**: A single accuracy score easily masks capability differences; the task-specific design can locate whether a model "does not recognize the entity" or "recognizes it but cannot judge the relation."

3.  **Coverage Metrics and Format Reliability Analysis**:
    - **Function**: Complements average accuracy by expressing graph-level knowledge coverage and revealing the impact of letter-only output formats on scores.
    - **Mechanism**: The authors define entity, relation, and triple coverage. The score for each triple is calculated as the average of head entity correctness, relation correctness, and tail entity correctness; simultaneously, the model's ability to stably output a single legal option letter is recorded.
    - **Design Motivation**: In multiple-choice evaluations, low scores may stem from missing knowledge or failure to follow the output format; coverage metrics allow for a re-examination of model strengths and weaknesses from a graph structure perspective.

### Loss & Training
Ours does not train models but constructs a benchmark to evaluate 15 models. API model temperature is set to 0 with a maximum completion length of 120, using a strict parser to extract single option letters. Local Hugging Face models utilize forced-choice scoring, selecting answers via the maximum log-prob of different letter surface forms. Benchmark generation uses a fixed random seed of 42.

## Key Experimental Results

### Main Results
| Model | AvgE | RP | AvgS | AvgS+E | AvgAll* | Key Information |
|------|------|------|------|------|------|------|
| GPT-4.1 | 94.73 | 54.96 | 60.79 | 66.46 | 70.28 | Overall strongest; best two-hop performance after evidence augmentation |
| GPT-5.2-chat | 94.07 | 58.63 | 57.88 | 64.33 | 69.32 | Highest single RP score |
| GPT-4o | 94.62 | 53.55 | 58.16 | 65.12 | 69.10 | Highest R1 at 62.08 |
| GPT-5-mini | 95.12 | 57.28 | 55.04 | 62.55 | 68.38 | Highest triple coverage |
| Qwen2.5-32B | 65.53 | 38.43 | 54.75 | 55.66 | 56.09 | Strongest open-source model, but significant gap remains with GPT |

### Ablation Study
| Graph-Level Coverage Metrics | GPT-5-mini | GPT-4o | GPT-4.1 | GPT-5.2 | Qwen2.5-32B |
|------|------|------|------|------|------|
| CovAvg(E) | 77.81 | 77.36 | 77.91 | 63.92 | 61.47 |
| CovDeg(R) | 63.30 | 61.18 | 61.24 | 44.56 | 55.09 |
| Cov(T) | 65.27 | 64.77 | 63.57 | 54.97 | 52.31 |
| Correlation with AvgAll* Rank | Highest Coverage | 2nd Coverage | 1st Accuracy | High Acc, Lower Coverage | 1st OS Acc, Coverage not leading |

### Key Findings
- Top models are very strong in ET/EC: Most GPT series models score above 97% to 98% in ET, with AvgE exceeding 94%, yet RP peaks at only 58.63%.
- The recognition-to-judgment gap is stable. Knowing entity types and relation schemas does not mean a model can reliably distinguish between indication, contraindication, off-label use, and none.
- Two-hop reasoning remains difficult. The AvgS for GPT-4.1 is only 60.79, far below its entity recognition level; evidence augmentation can improve it to an AvgS+E of 66.46, but not all models benefit.
- Evidence augmentation is not a panacea. For Qwen2.5-32B, R1 increased from 50.50 to 61.25, but R2 dropped from 59.00 to 50.08, indicating that short KG snippets may aid verification but interfere with selection.
- Contraindication relations are among the most difficult to analyze at a fine-grained level, which is highly relevant to real-world medical risks.

## Highlights & Insights
- The strongest aspect of this paper is its clear sense of boundaries: it does not claim to test "real clinical safety" but rather the model's consistency with a curated KG slice. This makes the evaluation conclusions more interpretable.
- The inconsistency between average accuracy and graph coverage rankings is insightful. A model might perform well on sampled questions while having low graph-level coverage; benchmarks should not rely solely on aggregate scores.
- Reliability of the constrained multiple-choice format was treated as a finding rather than noise by the authors. In medical evaluation, the inability to stably output a parsable answer is itself a deployment risk.
- KG-grounded negative sampling is highly suitable for structured medical evaluation, but the authors repeatedly emphasize that "absent from the subgraph" is not equivalent to being false in the real world, avoiding common misinterpretations.

## Limitations & Future Work
- The benchmark inherits the coverage limitations of PrimeKG and the subgraph extraction strategy, and does not represent complete psychiatric knowledge, long-term patient context, or individualized treatment decisions.
- All labels are valid relative to the extracted KG subgraph; as medical guidelines update, some edges may become outdated or incomplete.
- The authors did not perform additional expert validation on sampled questions, negative examples, or evidence snippets, thus results rely on KG quality and task generation rules.
- The multiple-choice format conflates knowledge capability with format compliance capability; for some models, low scores may partially stem from parsing failures or option bias.
- Future work could combine KG-grounded evaluation with case-based evaluation closer to real clinical workflows, while still retaining verifiable evidence chains.

## Related Work & Insights
- **vs HealthBench / MedQA**: These benchmarks lean more toward clinical Q&A or health dialogue quality, whereas MHGraphBench focuses on structural judgment of mental health KGs.
- **vs Mental Health Counseling/Diagnosis Benchmarks**: Related works measure diagnosis, counseling, or trustworthiness; this paper measures verifiable biomedical relation boundaries.
- **vs DRKG / PrimeKG Application Research**: Previously, KGs were mostly used for downstream discovery and reasoning; this paper transforms a PrimeKG subgraph into an LLM benchmark and adds coverage analysis.
- **Insights**: Medical LLM evaluations should be decomposed into dimensions like "recognition, relation judgment, short-chain reasoning, evidence integration, and format reliability," otherwise average scores can easily mask safety-critical weaknesses.

## Rating
- Novelty: ⭐⭐⭐⭐☆ KG-to-QA is not a brand-new direction, but the combination of the mental health PrimeKG subgraph, nine-task design, and coverage metrics is solid.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ The 15 models, task grouping, evidence augmentation, and coverage analysis are comprehensive, though expert review and more KG sources are missing.
- Writing Quality: ⭐⭐⭐⭐☆ Boundary explanations are clear, and metric definitions are rigorous; the tables are dense, making the reading cost slightly high.
- Value: ⭐⭐⭐⭐☆ Highly valuable for structured evaluation of mental health LLMs, especially for locating risks in drug relations and two-hop reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Text-Attributed Knowledge Graph Enrichment with Large Language Models for Medical Concept Representation](text-attributed_knowledge_graph_enrichment_with_large_language_models_for_medica.md)
- [\[ACL 2026\] MedFact: Benchmarking the Fact-Checking Capabilities of Large Language Models on Chinese Medical Texts](medfact_benchmarking_the_fact-checking_capabilities_of_large_language_models_on_.md)
- [\[ACL 2026\] MHSafeEval: Role-Aware Interaction-Level Evaluation of Mental Health Safety in Large Language Models](mhsafeeval_role-aware_interaction-level_evaluation_of_mental_health_safety_in_la.md)
- [\[NeurIPS 2025\] Mind the Gap: Aligning Knowledge Bases with User Needs to Enhance Mental Health Retrieval](../../NeurIPS2025/medical_nlp/mind_the_gap_aligning_knowledge_bases_with_user_needs_to_enhance_mental_health_r.md)
- [\[ICLR 2026\] CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of LLMs in Mental Health QA](../../ICLR2026/medical_nlp/counselbench_llm_mental_health_qa.md)

</div>

<!-- RELATED:END -->
