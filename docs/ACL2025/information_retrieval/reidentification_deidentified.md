---
title: >-
  [Paper Note] Re-identification of De-identified Documents with Autoregressive Infilling
description: >-
  [Information Retrieval & RAG] Proposes a RAG-based re-identification method for de-identified documents: first employs sparse + dense retrieval to find relevant background documents, and then uses an autoregressive infilling model to infer masked personally identifiable information (PII), recovering up to 80% of the masked text across three datasets.
tags:
  - "Information Retrieval & RAG"
date: 2026-05-08
content_hash: e95a7a511561d80d
---

# Re-identification of De-identified Documents with Autoregressive Infilling

- **Conference**: ACL 2025
- **arXiv**: [2505.12859](https://arxiv.org/abs/2505.12859)
- **Code**: Not open-sourced
- **Area**: Other
- **Keywords**: De-identification, Re-identification attacks, RAG, Text infilling, Privacy protection, ColBERT

## TL;DR

Proposes a RAG-based re-identification method for de-identified documents: first employs sparse + dense retrieval to find relevant background documents, and then uses an autoregressive infilling model to infer masked personally identifiable information (PII), recovering up to 80% of the masked text across three datasets.

## Background & Motivation

- **Core Problem**: De-identification protects privacy by masking personally identifiable information (PII). However, there is a lack of effective automated methods to evaluate the robustness of de-identification—specifically, whether an adversary can recover masked content from context and background knowledge.
- **Limitations of Prior Work**:
    - **Human-annotated evaluation**: Relying on human experts for comparison is highly costly and suffers from inconsistency.
    - **Classifier-based attacks**: Manzanares-Salor et al. trained classifiers to directly predict names, but they did not attempt to recover the masked text itself, lacking insights into the intermediate reasoning process.
    - **Morris et al.**: Used models to predict infoboxes to guide masking decisions, but similarly did not try to retrieve the masked content.
- **Design Motivation**: By leveraging LLM capabilities, this work constructs an adversary-simulating RAG system that recovers masked content before inferring identities, thereby providing a more comprehensive security assessment of de-identification methods.

## Method

### Overall Architecture

Given a de-identified document (where PII is replaced by [MASK]), the system executes a three-step re-identification workflow: (1) Sparse retrieval (BM𝒳) to select Top-100 relevant documents from the background knowledge base; (2) Dense retrieval (fine-tuned ColBERT) to extract the most relevant passage for each [MASK]; (3) An infilling model (GLM or Mistral-12B) to infer the original content using the retrieved passage and context. All masks are replaced sequentially until complete.

### Key Designs

1. **Two-stage Retrieval**: Sparse retrieval (BM𝒳) quickly narrows down the scope to 100 documents, and dense retrieval (ColBERT) precisely pinpoints passages containing the masked information. ColBERT is fine-tuned on de-identified Wikipedia biographies using positive (passages containing the original content) and negative examples.
2. **Four-level Background Knowledge Control**: L1 (No Retrieval) $\rightarrow$ L2 (General Knowledge, excluding the original text) $\rightarrow$ L3 (Including other de-identified original texts) $\rightarrow$ L4 (Including the original text of the target document), systematically evaluating the impact of background knowledge on re-identification capability.
3. **Final Identity Inference**: After text infilling, a BERT ranking model matches the recovered document with a candidate list of names to finalize identity lock-on.

### Loss & Training

- ColBERT retriever: Trained using the standard contrastive learning loss (positive/negative passage-query pairs) with a learning rate of $3 \times 10^{-5}$.
- GLM infilling model: Trained on de-identified Wikipedia biographies and corresponding retrieved text pairs with a learning rate of $3 \times 10^{-5}$.
- BERT ranking model: Margin ranking loss with a learning rate of $3 \times 10^{-6}$.

## Experiments

### Main Results (End-to-End Infilling Exact Match / Token Recall)

| Dataset | Model | L1 (No Retrieval) | L2 (General Knowledge) | L3 (With Other Originals) | L4 (With Original) |
|--------|------|-----------|-------------|---------------|-----------|
| Wikipedia | GLM | 6.26 / 12.22 | 9.56 / 15.84 | 9.77 / 16.05 | **80.08 / 82.56** |
| TAB (Court) | GLM | 0.84 / 6.26 | 11.27 / 21.35 | 14.32 / 29.08 | 66.04 / 75.13 |
| TAB (Court) | Mistral | 0.91 / 25.36 | 10.59 / 47.43 | 11.00 / 47.98 | 37.34 / 70.29 |
| Clinical Notes | GLM | 18.31 / 26.71 | 18.92 / 26.36 | 42.31 / 55.40 | **90.87 / 92.68** |

### Ablation Study (Final Identity Inference Top-10 Accuracy)

| Dataset | Model | Masked Document | L1 | L2 | L3 | L4 |
|--------|------|---------|-----|-----|-----|-----|
| TAB | GLM | 28.3 | 32.3 | 31.5 | 29.1 | 61.4 |
| Clinical Notes | GLM | 57.0 | 62.4 | 62.1 | 77.9 | **98.7** |
| TAB | Mistral | 28.3 | 32.3 | 33.1 | 37.0 | 57.5 |
| Clinical Notes | Mistral | 57.0 | 61.1 | 66.1 | 81.2 | 97.0 |

### Key Findings

1. **Background knowledge significantly impacts re-identification capability**: Accuracy jumps from 6% to 80%+ from L1 to L4, indicating that the security of de-identification highly depends on the external knowledge available to the adversary.
2. **Quasi-identifiers are easier to recover than direct identifiers**: Token recall for quasi-identifiers such as locations and dates is much higher than that for direct identifiers like names.
3. **Non-fine-tuned Mistral-12B achieves higher token recall on L1-L3**: The world knowledge of the LLM compensates for the lack of domain adaptation, but its performance in L4 scenarios is worse than that of the specifically fine-tuned GLM.
4. **Clinical notes are the easiest to re-identify**: Highly structured texts with fixed patterns (e.g., patient records) provide adversaries with more exploitable pattern information.
5. **Identity lock-on is effective on small candidate sets**: The Top-10 accuracy for 85 candidate patients in clinical notes is as high as 98.7%, whereas it is only 61.4% for 127 candidates in court cases.

## Highlights & Insights

- First to apply the RAG paradigm to re-identification attacks on de-identified documents, providing a novel perspective on evaluating the robustness of de-identification methods.
- The experimental design of four-level background knowledge (L1-L4) systematically quantifies the privacy risk gradients under different levels of information availability.
- The method can be directly applied to "red-teaming" during the de-identification phase, helping to discover insufficiently masked content.
- Discovers that the non-fine-tuned Mistral-12B already exhibits high token recall at low-to-medium background knowledge levels, revealing the privacy risks brought by the broad knowledge inherent to LLMs.
- The three datasets cover different document types (encyclopedia/legal/medical), enhancing the generalizability of the conclusions.

## Limitations & Future Work

- Only evaluates English text; the de-identification patterns and difficulty of re-identification in other languages may differ.
- The GLM model is relatively small (335M), and Mistral is used only in a zero-shot manner; larger models combined with ICL or fine-tuning could yield further improvements.
- The original versions of Wikipedia and court cases might already be included in the LLM pre-training data, leading to an overestimation of re-identification performance.
- Uses only textual data as background knowledge, without considering structured information sources such as tables and knowledge graphs.
- The de-identification strategy only considers entity masking, without evaluating the robustness of more advanced text-rewriting-based de-identification methods.
- Clinical notes are synthetic data, which might introduce pattern artifacts, making re-identification easier than on real data.

## Related Work & Insights

- **Text De-identification**: Lison et al. 2021 (NER-based masking), Pilán et al. 2022 (TAB benchmark, manually annotating direct/quasi-identifiers), Sánchez & Batet 2016 (text sanitization), Dernoncourt et al. 2017
- **Text Infilling**: GLM (Du et al. 2022, a unified encoder-decoder), Fill-in-the-Middle (Bavarian et al. 2022), Zhu et al. 2019, Donahue et al. 2020
- **RAG**: Lewis et al. 2020 (original RAG paper), ColBERT (Khattab & Zaharia 2020, dense retrieval), Guu et al. 2020 (REALM), Izacard et al. 2023
- **Re-identification Attacks**: Manzanares-Salor et al. 2024 (classifiers directly predicting personal names), Morris et al. 2022/2024 (infobox prediction guiding masking)
- **Privacy Protection**: GDPR data minimization principle, differentially private text rewriting (Igamberdiev & Habernal 2023)

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Applying RAG to privacy attacks presents an insightful and fresh perspective.
- **Practicality**: ⭐⭐⭐⭐⭐ — Directly applicable to evaluating the robustness of de-identification systems.
- **Rigor**: ⭐⭐⭐⭐ — The four-level background knowledge is thoroughly designed, with three datasets covering different scenarios.
- **Overall**: ⭐⭐⭐⭐

<!-- RAG-based re-identification attack on de-identified documents -->
<!-- Four levels of background knowledge systematically quantify privacy risks -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Uncovering Visual-Semantic Psycholinguistic Properties from the Distributional Structure of Text Embedding Space](psycholinguistic_visual_semantic.md)
- [\[ICCV 2025\] LangBridge: Interpreting Image as a Combination of Language Embeddings](../../ICCV2025/information_retrieval/langbridge_interpreting_image_as_a_combination_of_language_embeddings.md)
- [\[ICLR 2026\] Your Language Model Secretly Contains Personality Subnetworks](../../ICLR2026/information_retrieval/your_language_model_secretly_contains_personality_subnetworks.md)
- [\[ACL 2025\] Hypothetical Documents or Knowledge Leakage? Rethinking LLM-based Query Expansion](hypothetical_documents_or_knowledge_leakage_rethinking_llm-based_query_expansion.md)
- [\[CVPR 2025\] VDocRAG: Retrieval-Augmented Generation over Visually-Rich Documents](../../CVPR2025/information_retrieval/vdocrag_retrieval-augmented_generation_over_visually-rich_documents.md)

</div>

<!-- RELATED:END -->
