---
title: >-
  [Paper Note] Structure-aware Domain Knowledge Injection for Large Language Models
description: >-
  [ACL 2025][Knowledge Editing][Domain Knowledge Injection] This paper proposes StructTuning, which automatically extracts the taxonomic knowledge structure of the training corpus and designs a two-stage strategy: Structure-aware Continual Pre-training (SCPT) and Structure-aware Supervised Fine-tuning (SSFT). It achieves 100% of the domain knowledge injection performance of traditional methods while using only 5% of the data.
tags:
  - "ACL 2025"
  - "Knowledge Editing"
  - "Domain Knowledge Injection"
  - "Structured Learning"
  - "Continual Pre-training"
  - "Knowledge Taxonomy"
  - "Data-efficient"
date: 2026-05-08
content_hash: 48ea1be60b6e9abf
---

# Structure-aware Domain Knowledge Injection for Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2407.16724](https://arxiv.org/abs/2407.16724)  
**Code**: [https://github.com/alibaba/struxgpt](https://github.com/alibaba/struxgpt)  
**Area**: Knowledge Editing  
**Keywords**: Domain Knowledge Injection, Structured Learning, Continual Pre-training, Knowledge Taxonomy, Data-efficient

## TL;DR

This paper proposes StructTuning, which automatically extracts the taxonomic knowledge structure of the training corpus and designs a two-stage strategy: Structure-aware Continual Pre-training (SCPT) and Structure-aware Supervised Fine-tuning (SSFT). It achieves 100% of the domain knowledge injection performance of traditional methods while using only 5% of the data.

## Background & Motivation

**Background**: Adapting general LLMs into domain experts is a hot research direction, primarily via two routes: Retrieval-Augmented Generation (RAG) and Domain Knowledge Injection (continual pre-training + supervised fine-tuning). RAG performs poorly when logical reasoning is required and a semantic gap exists between user queries and the knowledge base. Meanwhile, traditional Continual Pre-training (CPT) typically requires massive scale datasets with billions of tokens; for instance, MMedLM utilizes 25.5B tokens for medical domain adaptation.

**Limitations of Prior Work**: Traditional CPT randomly concatenates and chunks the training corpus into fixed lengths, completely ignoring the inherent knowledge structure of the text (such as the table of contents in a textbook). Consequently, LLMs only absorb knowledge shallowly from "data diversity" instead of systematically mastering domain knowledge through "structured learning" like human students.

**Key Challenge**: Human education is structured—students study textbook chapters sequentially, review knowledge points, and apply knowledge through exercises. However, the training process of LLMs lacks this structural nature: the random chunking in the CPT stage discards the knowledge structure, and the QA pairs in the SFT stage do not guide the model to utilize structured knowledge to answer questions.

**Goal**: How to enable LLMs to learn in a structured manner like human students to efficiently inject domain knowledge, significantly reducing the amount of required training data.

**Key Insight**: Simulate the human education process—first study according to "textbook chapters" to associate knowledge points with the knowledge taxonomy, and then learn to invoke structured knowledge to solve practical problems through "exercise questions."

**Core Idea**: Automatically extract the taxonomic knowledge tree of the corpus, train the model to generate content conditioned on the knowledge path during CPT, and explicitly guide the model to reason along the knowledge path to answer questions during SFT.

## Method

### Overall Architecture

StructTuning consists of three core steps: (1) Knowledge Structure Extraction: automatically extract a taxonomic tree structure from the raw corpus using LLMs; (2) SCPT: associate each training chunk with a path node in the knowledge tree, train the language model conditioned on the knowledge path, and periodically ask the model to recall the full knowledge tree; (3) SSFT: generate QA training data containing reasoning paths based on the knowledge tree to teach the model how to answer questions along the knowledge structure.

### Key Designs

1. **Knowledge Structure Extraction**:

    - **Function**: Automatically reconstruct hierarchical taxonomic structures from the raw corpus.
    - **Mechanism**: First, spaCy is used to segment the text by paragraphs into sentences and merge them into fixed-length chunks. Then, Llama3-70B is used to generate summary titles (as "knowledge points") for each chunk. Finally, the list of titles is fed into a specially trained 7B model to identify the hierarchical structure (presented as a mindmap).
    - **Design Motivation**: The entire pipeline requires no human annotation and can scale to large-scale corpora. Experiments demonstrate that the specialized 7B model is capable of extracting sufficiently precise structures, and more powerful models like GPT-3.5 or Llama3-70B do not yield significant improvements.

2. **Structure-aware Continual Pre-training (SCPT)**:

    - **Function**: Train the model to learn textual content conditioned on the knowledge structure.
    - **Mechanism**: Convert the knowledge mindmap into natural language templates and prepend them to each training chunk. The model is trained to predict the chunk content $x^k$ conditioned on the knowledge path $s^k$, i.e., $p(x^k|s^k) = \prod_i p(x_i^k | x_{<i}^k, s^k)$. The knowledge path prefix is excluded from the loss calculation. Twenty diverse templates generated by GPT-4 are utilized. In training, after traversing all knowledge points in each epoch, the model is asked to recall the full knowledge hierarchy $p(\bar{s}) = \prod_k p(s^k)$.
    - **Design Motivation**: Conditional modeling forces the model to associate fragmented text contents with the complete knowledge system, while the "review" process ensures the model retains the global structure.

3. **Structure-aware Supervised Fine-tuning (SSFT)**:

    - **Function**: Teach the model to utilize structured knowledge to answer practical questions.
    - **Mechanism**: Perform random walks on the knowledge tree to sample 1 to $l$ knowledge paths. For a single path, generate knowledge-intensive QA pairs based on the corresponding text. For multiple paths, generate QA pairs that require multi-hop reasoning across multiple knowledge points. Two versions are generated for each QA pair: a raw version and a CoT-style version (prepending the knowledge mindmap path before the answer), and the model is trained on a mixture of both.
    - **Design Motivation**: Explicitly guide the model to first "locate" relevant knowledge paths before reasoning when answering questions, forming a "retrieve $\rightarrow$ reason $\rightarrow$ answer" structured thinking paradigm.

### Loss & Training

SCPT uses the standard language modeling loss but only computes it on the content portion (the loss is not computed for the knowledge path prefix), with a learning rate of 2e-5, and is trained for 3 epochs. SSFT is trained on synthetic QA data for 1 epoch to avoid overfitting. The model is validated across multiple base models, including Llama2-7B, Llama2-13B, Llama3-8B, and InternLM2-7B.

## Key Experimental Results

### Main Results (MMedBench Multiple-Choice Question Accuracy)

| Model | Method | Average Accuracy | Data Volume | Gain |
|--------|------|------|------|------|
| Llama3-8B | Baseline | 62.79 | - | - |
| Llama3-8B | MMed (SOTA) | 67.75 | 25.5B tokens | +4.96 |
| Llama3-8B | StructTuning | 65.36 | 76M tokens (0.3%) | +2.57 |
| Llama3-8B | StructTuning | 67.74 | 1.2B tokens (5%) | +4.95 |

### Ablation Study (Llama2-7B, MMedBench English Subset)

| Configuration | English Accuracy | Average Accuracy | Description |
|------|---------|------|------|
| SFT only | 44.54 | 35.91 | SFT only without pre-training |
| CPT + SFT | 46.27 | 35.49 | Traditional knowledge injection |
| SCPT + SFT | 46.50 | 35.13 | Structured pre-training + ordinary fine-tuning |
| SCPT + SSFT | 49.96 | 35.78 | Full StructTuning |
| SCPT + SSFT* (+ 8K QA) | 49.10 | 38.27 | Significant cross-lingual transfer after adding synthetic QA |
| RAG | 38.12 | 33.95 | Retrieval augmentation performs worse |

### Key Findings

- Achieving over 50% of the SOTA method's (25.5B tokens) performance with only 0.3% of the data (76M tokens); and reaching nearly 100% of the performance with just 5% of the data.
- The performance gain of SSFT compared to traditional SFT is larger than that of SCPT compared to CPT, indicating that teaching the model "how to use knowledge" is more critical than merely "injecting knowledge."
- An interesting cross-lingual knowledge transfer phenomenon emerged in structure-aware training: after training on English textbooks, the accuracy in five other languages also improved significantly.
- RAG performs the worst in this scenario (33.95%) because of the substantial semantic gap between textbook language and practical diagnostic questions.
- The method delivers significant improvements on InternLM2-7B (+4.46%), Llama2-7B (+8.78%), and Llama2-13B (+6.17%), demonstrating excellent generalization across different architectures and scales.

## Highlights & Insights

- **Astonishing data efficiency**: achieving 50% performance with 0.3% of the data and 100% with 5% implies that most data in traditional CPT is absorbed "aimlessly." Structured organization can vastly improve learning efficiency. This insight is transferable to any scenario requiring domain adaptation.
- **Knowledge path guidance in SSFT** is similar to explicit chain-of-thought reasoning, but instead of allowing the model to think freely, it directs it to perform "structured thinking" along the domain knowledge architecture. This idea of embedding domain knowledge graphs into the model's reasoning process is highly inspiring.
- The fitted scaling law curve $p_s \approx -1.11(\log r)^2 + 7.63 \log r + 133.0$ successfully predicts the performance at the 5% data point, providing a quantitative tool to characterize the data efficiency of structured knowledge injection.

## Limitations & Future Work

- Knowledge structure extraction heavily relies on LLMs, and the extraction quality may degrade for unstructured, mixed texts that are not textbook-like (such as web crawls).
- The training pipeline introduces additional computational overhead, including taxonomic structure extraction, QA generation, and inference with Llama3-70B.
- The knowledge path prefixes in SCPT increase the context length, consuming the limited context window.
- The method is currently validated primarily on medicine and long-document QA, leaving more domains like code and mathematics untested.

## Related Work & Insights

- **vs AdaptLLM**: This method appends reading comprehension QA after each CPT chunk, but chunk-level enhancement alone fails to help the model comprehend the global knowledge structure, resulting in only a 0.52% improvement on the MMedBench English subset (46.79% vs. 46.27%).
- **vs RAFT**: The retrieval-augmented fine-tuning method introduces too many irrelevant chunks, which conversely degrades QA reasoning ability (43.60% vs. 46.27%).
- **vs RAG**: In the textbook $\rightarrow$ diagnostic question scenario, RAG performs the worst due to the semantic gap between queries and the knowledge base, whereas knowledge injection methods are more advantageous by internalizing knowledge into the model parameters.

## Rating

- Novelty: ⭐⭐⭐⭐ Maps human educational models (structured learning $\rightarrow$ exercises $\rightarrow$ review) systematically to the LLM training pipeline, showing a clear and novel concept.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Verifies across multiple model architectures and scales, with comprehensive ablations, deep scalability analysis, and detailed method comparisons.
- Writing Quality: ⭐⭐⭐⭐ Rich in charts and figures; the analogy to human education makes the narration easy to comprehend, though several details reside in the appendix.
- Value: ⭐⭐⭐⭐⭐ Resolves the key challenge of low data efficiency during domain adaptation, possessing remarkably high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Neuron-Level Sequential Editing for Large Language Models](neuron-level_sequential_editing_for_large_language_models.md)
- [\[ACL 2025\] Context-Robust Knowledge Editing for Language Models](context-robust_knowledge_editing_for_language_models.md)
- [\[ACL 2025\] Memorizing is Not Enough: Deep Knowledge Injection Through Reasoning](memorizing_is_not_enough_deep_knowledge_injection_through_reasoning.md)
- [\[NeurIPS 2025\] UniEdit: A Unified Knowledge Editing Benchmark for Large Language Models](../../NeurIPS2025/knowledge_editing/uniedit_a_unified_knowledge_editing_benchmark_for_large_language_models.md)
- [\[ACL 2025\] A General Knowledge Injection Framework for ICD Coding](a_general_knowledge_injection_framework_for_icd_coding.md)

</div>

<!-- RELATED:END -->
