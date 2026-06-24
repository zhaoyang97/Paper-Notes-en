---
title: >-
  [Paper Note] ArgHiTZ at ArchEHR-QA 2025: A Two-Step Divide and Conquer Approach to Patient Question Answering for Top Factuality
description: >-
  [ACL 2025][Medical LLM][Patient Question Answering] A two-step "divide and conquer" approach was proposed for the ArchEHR-QA 2025 shared task: first, key sentences are extracted from electronic health records using a re-ranking model, and then a small medical LLM generates the response. This approach achieved first place in factuality and 8th/30 in overall score without using any external knowledge.
tags:
  - "ACL 2025"
  - "Medical LLM"
  - "Patient Question Answering"
  - "Electronic Health Records"
  - "Shared Task"
  - "Two-Step Method"
  - "Factuality"
date: 2026-05-08
content_hash: 0b40df090519ebb3
---

# ArgHiTZ at ArchEHR-QA 2025: A Two-Step Divide and Conquer Approach to Patient Question Answering for Top Factuality

**Conference**: ACL 2025  
**arXiv**: [2506.12886](https://arxiv.org/abs/2506.12886)  
**Code**: [https://github.com/hitz-zentroa/ArchEHR-ArgHiTZ](https://github.com/hitz-zentroa/ArchEHR-ArgHiTZ)  
**Area**: Medical NLP  
**Keywords**: Patient Question Answering, Electronic Health Records, Shared Task, Two-Step Method, Factuality

## TL;DR
A two-step "divide and conquer" approach was proposed for the ArchEHR-QA 2025 shared task: first, key sentences are extracted from electronic health records using a re-ranking model, and then a small medical LLM generates the response. This approach achieved first place in factuality and 8th/30 in overall score without using any external knowledge.

## Background & Motivation
**Background**: The volume of patient messages in online patient portals of medical institutions continues to grow, placing a massive burden on clinicians. The ArchEHR-QA 2025 task requires automatically answering health questions raised by patients based on their electronic health records (EHR).

**Limitations of Prior Work**: Generating responses directly using end-to-end LLMs requires models to perform both sentence selection and response generation simultaneously, which suffers from low accuracy. Large models (70B) do not necessarily outperform smaller models yet incur significantly higher costs.

**Key Challenge**: Under low-resource settings with only 20 development samples and no training data, how to design a system that can both accurately identify crucial evidence sentences in EHRs and generate high-quality responses for patients.

**Goal**: To optimize evidence selection and response generation separately using a task decomposition strategy, thereby improving factuality and overall quality in a low-resource scenario.

**Key Insight**: Decomposing the task into two steps: first selecting key sentences using the most suitable method (re-ranker or prompting), and second generating responses based on the selected sentences using a medical LLM, which is more controllable than end-to-end generation.

**Core Idea**: Using a similarity-based re-ranker for key sentence extraction and a small medical LLM for response generation, allowing each subtask to leverage the most appropriate approach through task decomposition.

## Method

### Overall Architecture
Input: Patient concern + clinical question reformulations + full EHR text with sentence IDs  
Output: Responses of $\le 75$ words, with each sentence citing EHR sentence IDs  

Comparison of three schemes:
1. **End-to-End Baseline**: Directly prompting the LLM to generate responses and citations in a single step.
2. **Two-Step Prompting**: Using LLM prompts to extract key sentences in the first step $\rightarrow$ generating responses in the second step.
3. **Two-Step Re-ranker**: Using a re-ranking model to extract key sentences in the first step $\rightarrow$ generating responses in the second step.

### Key Designs

1. **End-to-End Baseline (End-to-End)**:

    - Function: Direct LLM output of responses with citations using role prompting + CoT + 1-shot examples.
    - Mechanism: Taking patient narratives, clinical questions, and EHR with IDs as input, with the model simulating a clinician role to generate outputs in one step.
    - Findings: The 8B Aloe model outperformed both Llama 3.3 70B and Aloe 70B, demonstrating that larger models are not necessarily better.

2. **Two-Step Prompting (Two-Step Prompting)**:

    - Function: Extracting a list of key sentences in the first step using LLM prompting, and generating responses based on the key sentences in the second step.
    - Two extraction strategies: (a) List extraction: prompting the model to output all key sentence IDs at once; (b) Sentence-by-sentence classification: independently judging whether each sentence is critical (Yes/No).
    - Multiple prompting strategies tested: basic prompt, role prompting, CoT, and one-shot/few-shot.
    - Optimal combination: Aloe 8B + role prompting (Strict F1 = 0.50).

3. **Two-Step Re-ranker (Two-Step Re-ranker)** (Best Scheme):

    - Function: Sorting EHR sentences based on their relevance to the questions using a similarity re-ranking model in the first step, and filtering key sentences via a threshold.
    - Mechanism: Treating the patient narrative and clinical question as the query, and individual EHR sentences as documents. A re-ranker scores their relevance, with the optimal threshold determined via ROC curves and the Youden index.
    - Evaluated three models—Jina Reranker, BAAI BGE, and Alibaba GTE, with Jina yielding the best performance (F1 = 0.535).
    - The second step similarly utilizes Aloe 8B to generate responses, followed by post-processing to append citations.

### Post-processing
- Restricting responses to under 75 words by truncating selected sentences.
- Adding the most relevant EHR sentence ID citations to each generated sentence based on similarity matching.
- Excluding header-like sentences and constraining the maximum number of citations per sentence.

## Key Experimental Results

### Main Results

| Method | Dev Overall | Dev Factuality | Test Overall | Test Factuality |
|------|-------------|----------------|--------------|-----------------|
| End-to-End (Aloe 8B) | 0.388 | 0.464 | 0.367 | 0.408 |
| Two-Step Prompting | 0.385 | 0.504 | 0.366 | 0.452 |
| **Two-Step Re-Ranker** | **0.421** | **0.558** | **0.440** | **0.605** |
| Baseline (Llama 3.3 70B) | 0.359 | - | 0.307 | - |

### Ablation Study: Comparison of Re-ranking Models

| Model | Strict Prec. | Strict Rec. | Strict F1 |
|------|-------------|-------------|-----------|
| Jina Reranker | 0.427 | **0.717** | **0.535** |
| Alibaba GTE | 0.422 | 0.681 | 0.521 |
| BAAI BGE | **0.507** | 0.507 | 0.507 |

### Key Findings
- The Two-Step Re-ranker method comprehensively outperforms other designs, achieving an Overall score of 0.44 (8th/30th place) on the Test set, and ranking **first in Factuality**.
- Task decomposition itself is not guaranteed to be effective (minimal difference between Two-Step Prompting and End-to-End); the key lies in **pairing each subtask with the right approach**.
- The smaller Aloe 8B model consistently outperforms Llama 3.3 70B and Aloe 70B, suggesting that domain-specific fine-tuning is more critical than parameter scale in medical domains.
- Re-ranking models are superior to LLM prompting for evidence sentence selection.
- There remains room for improvement in Relevance scores, indicating that response generation quality could be further optimized.

## Highlights & Insights
- **Subtask Adaptation**: Rather than simply decomposing the task, the key is selecting the optimal model for each subtask—retrieval models for evidence selection, and generative language models for response generation.
- **Small Models Outperforming Large Models**: In complex tasks requiring medical expertise, summarization, and paraphrasing, domain-specific fine-tuned 8B models outperform general purpose 70B models, demonstrating high utility for resource-constrained scenarios.
- **No External Knowledge Required**: The system relies solely on the provided EHR data without query-to-medical-knowledge-base integration, yet still achieves the top rank in factuality.

## Limitations & Future Work
- With only 20 development samples, tuning thresholds on such a small dataset risks overfitting.
- Tuning/Fine-tuning was not attempted (due to lack of training data constraints), nor was RAG-based integration of external knowledge explored.
- The Relevance score remains relatively low, showing that readability and answering quality still have room to improve.
- Text-to-text models (such as T5) can be explored for response paraphrasing.
- Citation matching relies on simple similarity metrics, which might lack precision.

## Related Work & Insights
- **vs End-to-End LLMs**: Direct end-to-end generation underperforms compared to the two-step approach in Factuality, showing that decoupling evidence selection from response generation helps improve accuracy.
- **vs RAG Architectures**: The Two-Step Re-ranker method is essentially a variant of RAG, but without utilizing external knowledge bases, performing retrieval solely within the given EHR context.
- The concept of task decomposition paired with subtask adaptation can be generalized to other complex NLP tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ The methodology is relatively standard (re-ranking + LLM generation); while novelty is limited, the integration is highly effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Compares multiple baselines and models, though the small development set (20 samples) constrains the depth of analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Methodically structured, with robust experimental design and thorough discussions.
- Value: ⭐⭐⭐⭐ As a shared-task system description paper, the empirical insights regarding methodological combinations offer valuable reference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] AfriMed-QA: A Pan-African, Multi-Specialty, Medical Question-Answering Benchmark Dataset](afrimed_qa_pan_african.md)
- [\[ACL 2025\] Follow-up Question Generation for Enhanced Patient-Provider Conversations](follow-up_question_generation_for_enhanced_patient-provider_conversations.md)
- [\[ACL 2026\] Query Pipeline Optimization for Cancer Patient Question Answering Systems](../../ACL2026/medical_nlp/query_pipeline_optimization_for_cancer_patient_question_answering_systems.md)
- [\[ACL 2025\] A Retrieval-Based Approach to Medical Procedure Matching in Romanian](a_retrieval-based_approach_to_medical_procedure_matching_in_romanian.md)
- [\[ACL 2025\] MedBioRAG: Semantic Search and Retrieval-Augmented Generation with Large Language Models for Medical and Biological QA](medbiorag_semantic_search_and_retrieval-augmented_generation_with_large_language.md)

</div>

<!-- RELATED:END -->
