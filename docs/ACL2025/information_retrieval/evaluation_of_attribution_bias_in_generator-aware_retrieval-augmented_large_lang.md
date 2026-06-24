---
title: >-
  [Paper Note] Evaluation of Attribution Bias in Generator-Aware Retrieval-Augmented Large Language Models
description: >-
  [ACL 2025][Information Retrieval & RAG][attribution bias] Defines and investigates LLMs' attribution sensitivity and bias toward authorship information in RAG. Through counterfactual evaluation, this study reveals that informing LLMs of document authorship significantly alters attribution quality by 3-18%, and LLMs exhibit an attribution bias toward human authorship.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "attribution bias"
  - "RAG"
  - "counterfactual evaluation"
  - "authorship"
  - "citation generation"
date: 2026-05-08
content_hash: b484e68fd776b59f
---

# Evaluation of Attribution Bias in Generator-Aware Retrieval-Augmented Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2410.12380](https://arxiv.org/abs/2410.12380)  
**Code**: [https://github.com/aminvenv/attrieval](https://github.com/aminvenv/attrieval)  
**Area**: Information Retrieval  
**Keywords**: attribution bias, RAG, counterfactual evaluation, authorship, citation generation

## TL;DR
Defines and investigates LLMs' attribution sensitivity and bias toward authorship information in RAG. Through counterfactual evaluation, this study reveals that informing LLMs of document authorship significantly alters attribution quality by 3-18%, and LLMs exhibit an attribution bias toward human authorship.

## Background & Motivation

**Background**: RAG enhances answer verifiability by prompting LLMs to cite source documents, representing an important direction for reducing hallucinations. Numerous works focus on improving attribution quality.

**Limitations of Prior Work**: Improving attribution may introduce bias—LLMs might trust documents from different sources to varying degrees. Prior studies found that LLMs tend to favor content they generated themselves, but this conclusion may have alternative explanations.

**Key Challenge**: How does the attribution behavior of LLMs change when they are aware of whether a document is human-written or AI-generated? Does this change constitute a systematic bias?

**Goal**: To define and quantify attribution sensitivity and attribution bias of LLMs in RAG.

**Key Insight**: Counterfactual evaluation—observing changes in attribution quality by swapping document authorship labels (true vs. counterfactual).

**Core Idea**: LLMs exhibit a systematic bias toward human authorship labels in RAG—documents labeled as [Human] are more likely to be cited, even when the content is identical.

## Method

### Overall Architecture
Designing three RAG modes (Vanilla/Authorship-Informed/Counterfactual-Authorship) -> Generating answers and attributions using the same set of retrieved documents across all three modes -> Computing Counterfactually-estimated Attribution Sensitivity (CAS) and Counterfactually-estimated Attribution Bias (CAB) through comparison.

### Key Designs

1. **Three RAG Modes**

    - **Vanilla RAG**: Standard RAG without authorship information (baseline).
    - **Authorship-Informed RAG**: Annotating documents with their true author as [Human] / [LLM].
    - **Counterfactual-Authorship RAG**: Swapping the labels—human-written documents are labeled as [LLM], and LLM-written ones are labeled as [Human].
    - **Design Motivation**: Isolating the pure effect of authorship information on attribution by comparing true and counterfactual labels.

2. **Attribution Sensitivity Metric: CAS (Counterfactually-estimated Attribution Sensitivity)**

    - $CAS(Q) = \frac{1}{|Q|} \sum_{q \in Q} |M_{Informed}^q - M_{Vanilla}^q|$
    - Measures the magnitude of change in attribution quality after authorship information is introduced.
    - **Design Motivation**: To quantify the extent to which authorship information influences LLM behavior.

3. **Attribution Bias Metric: CAB (Counterfactually-estimated Attribution Bias)**

    - $CAB(Q) = \frac{\omega}{|Q|} \sum_{q \in Q} (M_{Informed}^q - M_{CF-informed}^q)$
    - Positive values indicate bias toward human authors, while negative values indicate bias toward LLM authors.
    - **Design Motivation**: Eliminating the influence of differences in document content through counterfactual flipping, thereby purely measuring label bias.

4. **Attribution Confidence (AC)**

    - Analyzes whether the probability when LLMs generate citation tokens differs due to different authorship labels.
    - **Design Motivation**: Complementing the analysis from the perspective of internal model confidence.

5. **Synthetic Document Set Construction**

    - Rewriting human documents using Llama3 with low temperature while maintaining the relevance/irrelevance status.
    - Validating synthetic document quality through annotation by two experts.
    - **Design Motivation**: Creating document pairs with equivalent content but different authorship.

### Experimental Setup
- 3 LLMs: Mistral-7B, Llama3-8B, GPT-4
- 2 Datasets: Natural Questions (NQ), MS MARCO
- 4 Document Combinations: Relevant and irrelevant documents are written by human/LLM respectively.

## Key Experimental Results

### Main Results — Comparison of Attribution Quality (NQ Dataset, Precision / Recall)

| Model | Relevant Doc | Irrelevant Doc | Vanilla Prec | Informed Prec | CF-Informed Prec |
|------|---------|-----------|-------------|--------------|-----------------|
| Mistral | LLM | Human | 47.6 | 42.1 | **52.7†** |
| Mistral | Human | LLM | 51.0 | **53.4†** | 44.0 |
| Llama3 | LLM | Human | 49.2 | 45.4 | **57.2†** |
| Llama3 | Human | LLM | 53.5 | **59.9†** | 44.8 |
| GPT-4 | LLM | Human | 63.3 | 59.7 | **65.9†** |
| GPT-4 | Human | LLM | 64.1 | **66.1** | 60.3 |

### Attribution Sensitivity CAS (Higher means more sensitive)

| Model | NQ (LLM-rel/Human-nonrel) | NQ (Human-rel/LLM-nonrel) |
|------|--------------------------|--------------------------|
| Mistral | **16.2†** | **20.1** |
| Llama3 | 13.2† | **17.7†** |
| GPT-4 | 9.7† | 8.7 |

### Attribution Bias CAB (Positive value = bias toward human)

| Model | NQ CAB | MS MARCO CAB | Direction |
|------|--------|-------------|------|
| Mistral | **+5.3** | **+3.1** | Bias toward human |
| Llama3 | **+7.5** | **+4.2** | Bias toward human |
| GPT-4 | **+3.1** | **+2.8** | Bias toward human |

### Key Findings
- **All three LLMs are sensitive to authorship info**: Informing LLMs of authorship alters attribution quality by 3-18% (CAS metric).
- **Consistent bias toward human authors**: When relevant documents are labeled as human-written, attribution precision increases; after counterfactual flipping, precision decreases—indicating that the bias stems from labels rather than content.
- **Llama3 exhibits the largest bias** (highest CAB), while GPT-4 has the smallest but remains significant.
- **Answer correctness remains largely unchanged**: Authorship information primarily influences attribution behavior rather than answer quality.
- **Even without using LLM-generated documents, simply adding [Human]/[LLM] labels induces bias**—demonstrating that the bias is a reaction to the labels rather than differences in content quality.

## Highlights & Insights
- The **counterfactual evaluation framework** is highly ingenious—by swapping labels to eliminate content confounding factors, it achieves a pure measurement of "label bias." This methodology can be transferred to other bias studies.
- The **human authorship bias provides an alternative hypothesis for prior findings**: While previous work suggested that LLMs favor self-generated content, this study finds they may favor content labeled as "human"—with the two directions being opposite, implying a more complex bias mechanism.
- The finding that **document metadata affects LLM trust** has profound implications—in real-world RAG systems, document meta-information (source, author, publication date, etc.) can potentially influence LLM behavior.

## Limitations & Future Work
- Only tests [Human] vs [LLM] authorship types, without exploring finer-grained effects such as specific author names or institutions.
- While of high quality, synthetic documents are not identical to real-world LLM-generated content.
- Limited sample size of 500 queries.
- Directions for improvement: More authorship types (e.g., "experts", "students"), attribution bias studies in real-world scenarios, and bias mitigation methods.

## Related Work & Insights
- **vs Tan et al. (2024)**: They found that LLMs favor self-generated text, whereas this study finds LLMs favor text labeled as human—these findings seem contradictory but act on different mechanisms.
- **vs Gao et al. (2023) ALCE**: ALCE evaluates attribution quality; this work adds the bias dimension on top of it.
- **vs Fairness studies (Ziems et al. 2024)**: Attribution bias is a new dimension of LLM bias.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to define and quantify attribution bias in RAG, with an ingenious counterfactual evaluation design.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 models × 2 datasets × 4 document combinations × 3 RAG modes.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous metric definitions and clear experimental design logic.
- Value: ⭐⭐⭐⭐⭐ Crucial implications for the trustworthiness and fairness of RAG systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] RARE: Retrieval-Augmented Reasoning Enhancement for Large Language Models](rare_retrieval_augmented_reasoning.md)
- [\[ACL 2025\] Enhancing Lexicon-Based Text Embeddings with Large Language Models](enhancing_lexicon-based_text_embeddings_with_large_language_models.md)
- [\[ACL 2025\] Atomic LLM: A Fine-Grained Information Retrieval Evaluation Benchmark for Language Models](atomic_llm_a_fine-grained_information_retrieval_evaluation_benchmark_for_languag.md)
- [\[ACL 2025\] Pandora's Box or Aladdin's Lamp: A Comprehensive Analysis Revealing the Role of RAG Noise in Large Language Models](pandora_box_rag_noise.md)
- [\[ACL 2025\] Parenting: Optimizing Knowledge Selection of Retrieval-Augmented Language Models with Parameter Decoupling and Tailored Tuning](parenting_optimizing_knowledge_selection_of_retrievalaugmented.md)

</div>

<!-- RELATED:END -->
