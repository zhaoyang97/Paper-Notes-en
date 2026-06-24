---
title: >-
  [Paper Note] Faithful Summarization of Consumer Health Queries: A Cross-Lingual Framework with LLMs
description: >-
  [NeurIPS 2025][Medical LLM][Medical Text Summarization] A framework combining TextRank-based extractive sentence selection and medical Named Entity Recognition (NER) is proposed to guide LLMs in generating faithful medical summaries. By fine-tuning LLaMA-2-7B on the English MeQSum and Bengali BanglaCHQ-Summ datasets, consistent improvements in both quality and faithfulness are achieved, with SummaC reaching 0.57 and human evaluation showing that 82% of the summaries retain ke…
tags:
  - "NeurIPS 2025"
  - "Medical LLM"
  - "Medical Text Summarization"
  - "Faithfulness"
  - "Cross-lingual"
  - "TextRank"
  - "Named Entity Recognition"
  - "LLaMA"
date: 2026-05-08
content_hash: f7a03129010f1a0a
---

# Faithful Summarization of Consumer Health Queries: A Cross-Lingual Framework with LLMs

**Conference**: NeurIPS 2025  
**arXiv**: [2511.10768](https://arxiv.org/abs/2511.10768)  
**Code**: Undisclosed  
**Area**: Medical NLP  
**Keywords**: Medical Text Summarization, Faithfulness, Cross-lingual, TextRank, Named Entity Recognition, LLaMA

## TL;DR

A framework combining TextRank-based extractive sentence selection and medical Named Entity Recognition (NER) is proposed to guide LLMs in generating faithful medical summaries. By fine-tuning LLaMA-2-7B on the English MeQSum and Bengali BanglaCHQ-Summ datasets, consistent improvements in both quality and faithfulness are achieved, with SummaC reaching 0.57 and human evaluation showing that 82% of the summaries retain key medical information.

## Background & Motivation

**Growth in Online Health Consultation**: Online consultation platforms driven by the pandemic have become crucial sources of medical information, but long and redundant patient queries impose a burden on healthcare professionals, who must identify core concerns before responding.

**Inadequacy of Traditional Metrics**: Existing summarization evaluations mainly rely on quality metrics like ROUGE/BERTScore to measure lexical/semantic similarity, but **ignore faithfulness**—the factual consistency between the summary and the source text.

**Specific Risks in Medical Scenarios**: Abstractive models often generate intrinsic errors (misrepresenting entities/relations) and extrinsic errors (introducing unsupported facts). In the medical domain, even minor distortions can mislead patients/doctors and jeopardize health outcomes.

**Faithfulness is Underestimated**: Compared to readability and general accuracy, faithfulness is understudied in medical summarization, lacking frameworks specifically designed to enhance it.

**Cross-Lingual Gap**: Cross-lingual evaluation of faithfulness in medical summarization is virtually non-existent, and medical NLP for low-resource languages (such as Bengali) is severely lacking.

## Method

### Overall Architecture

Three-stage pipeline: **Medical NER + TextRank Extraction → LLM Fine-Tuning → Best-of-N Selection**

### Step 1: Preprocessing and Relevant Sentence Extraction

1. Standardize datasets into a question-summary format.
2. Identify overlapping medical entities and negation words to ensure key information is preserved.
3. Apply the TextRank algorithm to extract sentences containing medical entities and query-relevant words.
4. **Core Goal**: Ensure that the input is anchored to medically important content prior to abstractive generation by the LLM.

TextRank is a graph-based ranking algorithm that treats sentences as nodes and inter-sentence similarities as edge weights, iteratively computing sentence importance scores:

$$S(V_i) = (1 - d) + d \cdot \sum_{V_j \in \text{In}(V_i)} \frac{w_{ji}}{\sum_{V_k \in \text{Out}(V_j)} w_{jk}} S(V_j)$$

where $d$ is the damping factor, and $w_{ji}$ is the weight of the edge from sentence $j$ to $i$.

### Step 2: LLM Fine-Tuning

- **Base Model**: LLaMA-2-7B
- **Training Approach**: LoRA (Low-Rank Adaptation) parameter-efficient fine-tuning
- **Input**: Sentences containing medical entities filtered by TextRank
- **Output**: Concise and faithful summaries

### Step 3: Best-of-N Selection

Generate multiple candidate summaries (temperature $t = 0.7$), selecting the optimal one using two strategies:
- **ROUGE-1 Selection**: Maximize lexical coverage (quality-oriented)
- **SummaC Selection**: Maximize factual consistency (faithfulness-oriented)

Temperature sweep $t \in \{0.1, 0.3, 0.5, 0.7, 0.9\}$ reveals a trade-off: lower temperatures favor ROUGE, while higher temperatures favor SummaC, with $t = 0.7$ serving as the equilibrium point.

### Evaluation Metrics

- **Quality Metrics**: ROUGE-1/2/L, BERTScore
- **Faithfulness Metrics**: SummaC (NLI-based factual consistency), AlignScore (semantic alignment)
- **Readability**: Flesch Reading Ease (FRE)

## Key Experimental Results

### MeQSum (English, 1000 samples)

| Setting | R1 | R2 | RL | BERT | Readability | SummaC | AlignScore |
|------|-----|-----|-----|------|--------|--------|------------|
| Zero-shot | 21.97 | 6.48 | 19.98 | 0.60 | 65.16 | 0.28 | 21.80 |
| FT (w/o TR) | 44.23 | 27.36 | 41.55 | 0.71 | 70.21 | 0.31 | 38.45 |
| FT + TR | 47.07 | 29.44 | 44.08 | 0.72 | 70.69 | 0.37 | 45.65 |
| Best-of-3 (R1) | **50.50** | **34.38** | **47.74** | **0.74** | 71.56 | 0.40 | 39.24 |
| Best-of-3 (SummaC) | 48.27 | 31.38 | 45.34 | 0.73 | 71.56 | **0.57** | **45.91** |

### Comparison with SOTA

| Model | R1 | R2 | RL | BERT | SummaC | AlignScore |
|------|-----|-----|-----|------|--------|------------|
| Mixtral-8x7B-Inst. | 32.47 | 36.38 | 16.86 | 0.72 | - | - |
| BioBART + FaMeSumm | 31.76 | 11.71 | 29.64 | 0.74 | 0.46 | - |
| **Ours (Best-of-3)** | **50.50** | **34.38** | **47.74** | **0.74** | **0.57** | **0.46** |

### BanglaCHQ-Summ (Bengali, 2350 samples)

| Setting | R1 | R2 | RL | BERT | SummaC |
|------|-----|-----|-----|------|--------|
| Zero-shot | 19.10 | 8.21 | 18.97 | 0.62 | 0.22 |
| FT (w/o TR) | 28.24 | 14.22 | 24.54 | 0.71 | 0.26 |
| FT + TR | 30.71 | 15.71 | 28.95 | 0.74 | 0.28 |
| Best-of-3 (R1) | **32.35** | **16.32** | **29.09** | **0.76** | 0.29 |
| Best-of-3 (SummaC) | 30.92 | 15.74 | 27.35 | 0.73 | **0.32** |

### Key Findings

1. **TextRank remains consistently effective**: Integrating TextRank consistently improves both quality and faithfulness in both English and Bengali.
2. **Best-of-N strategy is significant**: Selecting the best candidate from 3 options yields substantial improvements over single-pass generation—with ROUGE-1 selection optimizing quality, and SummaC selection optimizing faithfulness.
3. **Successful knowledge transfer**: The framework remains effective from English to Bengali, validating its cross-lingual generalization capability.
4. **Human evaluation validation**: Clinical MD evaluations indicate that 82% of the summaries retain all key medical information while maintaining factual consistency.
5. **Temperature trade-off**: A quality-faithfulness trade-off exists, where $t = 0.7$ acts as the optimal balance point.

## Highlights & Insights

- ⭐⭐⭐⭐ **Focus on Faithfulness**: First to treat faithfulness as the core optimization objective in medical text summarization rather than solely pursuing ROUGE.
- ⭐⭐⭐ **Cross-lingual Validation**: Validated on both English and Bengali (a low-resource language), presenting the first cross-lingual faithfulness evaluation in medical summarization.
- ⭐⭐⭐ **Practical Pipeline**: The TextRank + NER + LoRA fine-tuning + Best-of-N pipeline is simple, practical, and highly reproducible.
- ⭐⭐⭐ **Human Validation**: An 82% pass rate in human evaluation provides credibility beyond automated metrics.
- ⭐⭐ **Clear Ablation**: The incremental contribution of each component is clearly demonstrated through ablation studies.

## Limitations & Future Work

1. **Single LLM**: Evaluated only on LLaMA-2-7B, without testing newer or larger models (e.g., LLaMA-3, Mistral).
2. **Small Dataset Scale**: MeQSum has only 1,000 samples and BanglaCHQ-Summ has 2,350 samples, which limits statistical significance.
3. **Only Two Languages**: Covers only English and Bengali; generalization to other low-resource languages remains unknown.
4. **Limitations of TextRank**: Graph-based ranking based on term frequency might miss semantically crucial but low-frequency information in complex medical narratives.
5. **Limited Evaluation Dimensions**: SummaC and AlignScore are themselves proxy metrics that may fail to capture all types of factual inconsistency.

## Overall Evaluation ⭐⭐⭐

The research direction is correct—treating faithfulness as a core focus in medical summarization is an important contribution. The methodological design is straightforward yet effective (TextRank + NER guidance + Best-of-N). The cross-lingual evaluation is a key highlight. However, the overall scale is relatively small (single LLM, small datasets, only two languages), and the depth of technical contribution is limited—resembling a systematic engineering integration rather than a methodological innovation. Its positioning as a workshop paper (Muslims in ML @ NeurIPS) is appropriate.

## Related Work & Insights

## Related Work & Insights

## Rating

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Empathy Applicability Modeling for General Health Queries](../../ACL2026/medical_nlp/empathy_applicability_modeling_for_general_health_queries.md)
- [\[NeurIPS 2025\] Document Summarization with Conformal Importance Guarantees](document_summarization_with_conformal_importance_guarantees.md)
- [\[ACL 2025\] Evaluation of LLMs in Medical Text Summarization: The Role of Vocabulary Adaptation in High OOV Settings](../../ACL2025/medical_nlp/evaluation_of_llms_in_medical_text_summarization_the_role_of_vocabulary_adaptati.md)
- [\[NeurIPS 2025\] H-DDx: A Hierarchical Evaluation Framework for Differential Diagnosis](h-ddx_a_hierarchical_evaluation_framework_for_differential_diagnosis.md)
- [\[NeurIPS 2025\] CureAgent: A Training-Free Executor-Analyst Framework for Clinical Reasoning](cureagent_a_training-free_executor-analyst_framework_for_clinical_reasoning.md)

</div>

<!-- RELATED:END -->
