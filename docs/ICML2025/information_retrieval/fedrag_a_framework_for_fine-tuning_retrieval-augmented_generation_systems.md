---
title: >-
  [Paper Note] FedRAG: A Framework for Fine-Tuning Retrieval-Augmented Generation Systems
description: >-
  [ICML 2025][Information Retrieval & RAG] FedRAG proposes a fine-tuning framework for RAG systems that supports both centralized and federated architectures. It fills the gap of lacking unified fine-tuning tools in the RAG ecosystem and achieves seamless transition from centralized to federated training through lightweight abstractions.
tags:
  - "ICML 2025"
  - "Information Retrieval & RAG"
date: 2026-05-08
content_hash: 303fd505b951efa0
---

# FedRAG: A Framework for Fine-Tuning Retrieval-Augmented Generation Systems

**Conference**: ICML 2025  
**arXiv**: [2506.09200](https://arxiv.org/abs/2506.09200)  
**Area**: Information Retrieval  

## TL;DR

FedRAG proposes a fine-tuning framework for RAG systems that supports both centralized and federated architectures. It fills the gap of lacking unified fine-tuning tools in the RAG ecosystem and achieves seamless transition from centralized to federated training through lightweight abstractions.

## Background & Motivation

Retrieval-Augmented Generation (RAG) systems mitigate hallucination issues by supplementing the parametric memory of Large Language Models (LLMs) with retrieved relevant information from external knowledge bases. Recent studies demonstrate that fine-tuning the retriever and generator of RAG systems can further enhance performance. However, although the current RAG ecosystem is rich in tools (e.g., LlamaIndex, LangChain), it lacks a framework that simplifies the RAG fine-tuning workflow while deeply integrating with the ecosystem. More importantly, under data privacy constraints, Federated Learning (FL) has become an indispensable tool for improving RAG systems, yet few existing tools can seamlessly convert centralized RAG fine-tuning into federated tasks.

## Method

### Core Design Philosophy

FedRAG follows three main design principles:

1. **Advanced RAG Fine-Tuning**: Comprehensively supports cutting-edge RAG fine-tuning methods and enables easy federated adaptation.
2. **Work With Your Tools**: Deeply integrates with popular frameworks such as HuggingFace, Unsloth, LlamaIndex, etc.
3. **Lightweight Abstractions**: Provides clear and intuitive interfaces to lower the learning curve.

### Overall Architecture

FedRAG adopts a modular design, where key modules include:

- **`core`**: Core type definitions, including the `RAGSystem` class.
- **`generators`**: Generator types (supporting HuggingFace, Unsloth, etc.)
- **`retrievers`**: Retriever types (supporting HF SentenceTransformer, etc.)
- **`knowledge_stores`**: Knowledge stores (supporting Qdrant, etc.)
- **`trainers`**: Trainer types
- **`fl_tasks`**: Federated learning task definitions
- **`evals`**: Evaluation metrics and benchmarks

### RAG System Construction

A `RAGSystem` consists of three parts: `KnowledgeStore`, `Retriever`, and `Generator`. Users can rapidly assemble a RAG system and execute queries through concise APIs.

### Fine-Tuning Methods

FedRAG supports two main categories of fine-tuning methods:

**Generator Fine-Tuning:**
- **RALT (Retrieval-Augmented Language Model Training)**: Fine-tunes with instruction samples containing retrieved context.
- **RAFT (Retrieval-Augmented Fine-Tuning)**: Incorporates LLM-generated Chain-of-Thought (CoT) reasoning paths into instruction samples.
- **ReSearch**: Utilizes reinforcement learning to enable the LLM to learn to generate long CoTs containing search and retrieval operations.

**Retriever Fine-Tuning:**
- **LSR (Language Model Supervised Retriever Training)**: Minimizes the KL divergence between the retrieval score distribution and the generator's conditional probability distribution.

$$\mathcal{L}_{\text{LSR}} = D_{\text{KL}}(P_{\text{retrieval}} \| P_{\text{generator}})$$

### Federated Transition

The core innovation of FedRAG lies in its seamless transition from centralized to federated setups. Users only need to extract the `FL_task` object from the training manager to obtain the federated server and clients, enabling decentralized training via Federated Averaging (FedAvg):

$$\theta_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} \theta_{t+1}^k$$

where $\theta_{t+1}^k$ represents the locally updated parameters of the $k$-th client.

### Evaluation & Benchmarking

FedRAG provides an intuitive benchmarking interface, supporting the execution of specific `Benchmark`s (such as HuggingFace MMLU) using the `Benchmarker` and applying chosen evaluation metrics.

## Experiments

### Main Results

The paper presents lightweight experiments in the appendix, validating that FedRAG can successfully and flexibly execute RAG fine-tuning tasks. The experimental code and containerized images of the knowledge base have been released alongside the paper to facilitate reproducibility.

### Integration Support

| Library | Integrated Content |
|---|---|
| HuggingFace | Generators, Retrievers, Datasets |
| Unsloth | Fast Generator Fine-Tuning |
| Qdrant | Knowledge Store Solutions |
| LlamaIndex | Inference Object Bridging |

### Future Roadmap

| Priority | Development Item |
|---|---|
| High | MCP RAG system and MCP knowledge store integration |
| High | Investigate adaptation efficacy of third-party MCP providers |
| Medium | Support for more fine-tuning methods |

## Highlights

- **First Unified RAG Fine-Tuning Framework**: Supports both centralized and federated architectures, filling a gap in the ecosystem.
- **Minimalist Federated Transition**: Converts centralized training to federated tasks with just a few lines of code.
- **Deep Ecosystem Integration**: Seamlessly interfaces with mainstream tools like HuggingFace, Unsloth, Qdrant, and LlamaIndex.
- **Forward-Looking Design**: Plans to integrate the MCP protocol, aligning with the decentralized AI trend.

## Limitations & Future Work

- The paper mainly focuses on the system framework; the experimental validation is lightweight and lacks large-scale benchmark comparisons.
- Communication efficiency and privacy protection details in federated learning scenarios are not discussed in depth.
- Currently supported fine-tuning methods are limited and do not yet cover all frontier RAG fine-tuning techniques.
- A systematic comparison with other potential competing frameworks is lacking.

## Rating

⭐⭐⭐ (3/5)

As a system tool paper, the design concept is clear and the interface is elegant, filling the gap of fine-tuning tools in the RAG ecosystem. However, the experimental validation is insufficient, making it slightly thin as an ICML paper.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] FlexRAG: A Flexible and Comprehensive Framework for Retrieval-Augmented Generation](../../ACL2025/information_retrieval/flexrag_a_flexible_and_comprehensive_framework_for_retrieval-augmented_generatio.md)
- [\[ICLR 2026\] Supervised Fine-Tuning or Contrastive Learning? Towards Better Multimodal LLM Reranking](../../ICLR2026/information_retrieval/supervised_fine-tuning_or_contrastive_learning_towards_better_multimodal_llm_rer.md)
- [\[ACL 2025\] HASH-RAG: Bridging Deep Hashing with Retriever for Efficient, Fine Retrieval and Augmented Generation](../../ACL2025/information_retrieval/hash-rag_bridging_deep_hashing_with_retriever_for_efficient_fine_retrieval_and_a.md)
- [\[ACL 2026\] GIFT: Guided Fine-Tuning and Transfer for Enhancing Instruction-Tuned Language Models](../../ACL2026/information_retrieval/gift_guided_fine-tuning_and_transfer_for_enhancing_instruction-tuned_language_mo.md)
- [\[NeurIPS 2025\] Improving Consistency in Retrieval-Augmented Systems with Group Similarity Rewards](../../NeurIPS2025/information_retrieval/improving_consistency_in_retrieval-augmented_systems_with_group_similarity_rewar.md)

</div>

<!-- RELATED:END -->
