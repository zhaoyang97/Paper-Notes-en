---
title: >-
  [Paper Note] MuCo: Multi-turn Contrastive Learning for Multimodal Embedding Model
description: >-
  [CVPR 2026][Information Retrieval & RAG][Multimodal Embedding] MuCo proposes a multi-turn dialogue-based contrastive learning framework that leverages the conversational capabilities of MLLMs to process multiple associat…
tags:
  - "CVPR 2026"
  - "Information Retrieval & RAG"
  - "Multimodal Embedding"
  - "Contrastive Learning"
  - "Multi-turn Dialogue"
  - "Retrieval"
  - "Multimodal Large Language Model"
date: 2026-05-08
content_hash: b4c5ad536357c9f8
---

# MuCo: Multi-turn Contrastive Learning for Multimodal Embedding Model

**Conference**: CVPR 2026
**arXiv**: [2602.06393](https://arxiv.org/abs/2602.06393)  
**Code**: [https://github.com/naver-ai/muco](https://github.com/naver-ai/muco)  
**Area**: Information Retrieval
**Keywords**: Multimodal Embedding, Contrastive Learning, Multi-turn Dialogue, Retrieval, Multimodal Large Language Model

## TL;DR

MuCo proposes a multi-turn dialogue-based contrastive learning framework that leverages the conversational capabilities of MLLMs to process multiple associated query-target pairs within a single forward pass, substantially improving training efficiency and achieving state-of-the-art performance on the MMEB and M-BEIR retrieval benchmarks.

## Background & Motivation

**Background**: Universal Multimodal Embedding Models are built upon multimodal large language models (MLLMs) and typically employ contrastive learning to align query-target pair representations across modalities. These models have achieved notable success in tasks such as image-text retrieval and visual question answering retrieval.

**Limitations of Prior Work**: Existing methods are built on a single-turn paradigm, in which each query-target pair is treated as an independent data point. This gives rise to two core problems: (1) computational inefficiency, as each pair requires a separate forward pass; and (2) failure to capture the latent contextual relationships among multiple queries associated with the same context (e.g., the same image).

**Key Challenge**: MLLMs inherently possess multi-turn dialogue capabilities, yet existing multimodal embedding training paradigms make no use of this property. The single-turn paradigm limits the effective batch size and prevents the model from capturing shared contextual information across multiple semantic dimensions associated with the same image.

**Goal**: To design a training framework capable of processing multiple groups of query-target pairs associated with the same image in a single forward pass, extracting multiple embedding representations simultaneously, thereby amplifying the effective batch size and enhancing the coherence of cross-modal representations.

**Key Insight**: The authors observe that MLLMs natively support multi-turn dialogue at inference time, with each turn's response conditioned on a shared context. By treating each query-target pair in embedding learning as one turn of a dialogue, multiple embeddings can be extracted within a single forward pass.

**Core Idea**: Upgrade contrastive learning from the "independent single-turn" paradigm to a "multi-turn dialogue" paradigm—encoding multiple associated queries and targets jointly within a single MLLM forward pass while sharing image context representations, thereby achieving simultaneous gains in training efficiency and representation quality.

## Method

### Overall Architecture

The overall pipeline of MuCo is as follows: given an image and its associated multiple query-target pairs (e.g., descriptions across different dimensions, question-answer pairs of varying granularity), these are organized into a multi-turn dialogue format and fed into an MLLM for a single forward pass. The model extracts the EOS token at the end of each dialogue turn as the embedding for that turn, yielding multiple query embeddings and target embeddings. All embeddings are trained with in-batch negative sampling under the contrastive learning framework.

### Key Designs

1. **Multi-Turn Contrastive Learning**:

    - **Function**: Upgrades traditional single-turn independent query-target contrastive learning to joint training with multi-turn shared context.
    - **Mechanism**: Given an image $I$ and $K$ associated query-target pairs $\{(q_k, t_k)\}_{k=1}^K$, these are organized as a multi-turn dialogue input to the MLLM. The query and target of each turn are encoded into embedding vectors, and all $K$ embeddings are obtained within a single forward pass. This is equivalent to amplifying the effective batch size by a factor of $K$, since each sample contributes $K$ contrastive pairs rather than the single pair in conventional methods.
    - **Design Motivation**: By leveraging KV-cache sharing in the MLLM's dialogue mechanism, repeated encoding of the image context is avoided, substantially reducing computational overhead. Additionally, inter-turn contextual dependencies help the model learn more coherent multi-dimensional representations.

2. **M3T Multimodal Multi-Turn Dataset**:

    - **Function**: Provides training data for multi-turn contrastive learning.
    - **Mechanism**: A multimodal multi-turn dataset, M3T, is constructed comprising 5 million samples. Each sample contains one image and multiple associated query-target pairs, covering diverse task types including image-text retrieval, visual question answering, and image captioning. The dataset is assembled by integrating and augmenting existing data sources, ensuring that multiple pairs associated with the same image span different semantic dimensions.
    - **Design Motivation**: Existing multimodal embedding training datasets are all in single-turn format and cannot be directly used for multi-turn contrastive learning. M3T fills this gap and provides the foundation for large-scale multi-turn training.

3. **Embedding Extraction and Normalization Strategy**:

    - **Function**: Efficiently extracts comparable embedding vectors from the multi-turn outputs of the MLLM.
    - **Mechanism**: An EOS token pooling strategy is adopted, extracting the token representation at the end position of each dialogue turn as the embedding for that turn. The extracted embeddings are L2-normalized before being used in the contrastive loss computation. At inference time, the model supports both single-turn queries (compatible with existing benchmarks) and multi-turn batch queries.
    - **Design Motivation**: EOS pooling naturally corresponds to the end of each turn's response in the dialogue, capturing the most semantically complete representation; normalization ensures geometric consistency in the embedding space.

### Loss & Training

MuCo employs the standard InfoNCE contrastive loss, with the effective batch size amplified by a factor of $K$ (the number of turns per sample). Specifically, contrastive learning with in-batch negative sampling is applied to all query-target pairs extracted across all turns and all samples in the batch. Training is conducted with DeepSpeed for distributed training, supporting both 2B and 7B model scales.

## Key Experimental Results

### Main Results

| Benchmark | Metric | MuCo-2B | MuCo-7B | Prev. SOTA | Gain |
|-----------|--------|---------|---------|------------|------|
| MMEB | Avg Score | 70.1 | 74.2 | ~69 | +1.1 / +5.2 |
| M-BEIR | Recall@10 | — | SOTA | — | Significant |

### Ablation Study

| Configuration | MMEB Score | Notes |
|---------------|-----------|-------|
| Single-turn baseline | ~68 | Traditional single-turn contrastive learning |
| MuCo (K=2) | ~69.5 | 2 dialogue turns per sample |
| MuCo (K=4) | 70.1 | 4 turns per sample, optimal configuration |
| MuCo w/o M3T | ~68.5 | Without M3T dataset |
| MuCo full (7B) | 74.2 | Full configuration with larger model |

### Key Findings

- The performance gain from multi-turn contrastive learning grows with the number of turns $K$ but saturates, with approximately 4 turns being the optimal balance.
- The multi-turn format of the M3T dataset is critical to performance gains; applying the multi-turn framework with single-turn data yields limited improvement.
- The 2B model already achieves 70.1, and the 7B model further improves to 74.2, demonstrating the framework's effectiveness across different model scales.
- Training efficiency is substantially improved: compared to processing $K$ times as many single-turn samples, MuCo reduces image encoding computation by approximately $(K-1)/K$.

## Highlights & Insights

- The notion of **dialogue as batching** is particularly elegant: the MLLM's multi-turn conversational capability is reinterpreted as "batch embedding extraction"—a conceptually simple idea with pronounced empirical impact and virtually no additional architectural modification.
- The shared image context design naturally encourages the model to learn multi-faceted image representations, as different queries attend to different semantic dimensions of the same image, facilitating a richer embedding space.
- This paradigm is transferable to any MLLM-based embedding learning scenario—such as document retrieval or code retrieval—so long as multiple query-target pairs can be constructed for the same context.

## Limitations & Future Work

- Multi-turn training requires multiple high-quality query-target pairs per sample, resulting in non-trivial data construction costs.
- The scale and diversity of the current M3T dataset remain improvable; a larger-scale multi-turn dataset may yield further gains.
- The paper primarily validates on retrieval tasks; the quality of embeddings on generative tasks (e.g., VQA generation, image captioning) remains to be explored.
- Whether the ordering of turns affects embedding quality is not analyzed; the sensitivity to turn permutation warrants further investigation.

## Related Work & Insights

- **vs. E5-V / VLM2Vec**: These methods employ single-turn contrastive learning to train multimodal embeddings, encoding each pair independently. MuCo obtains more contrastive signal under equivalent computation via the multi-turn mechanism, achieving gains in both efficiency and performance.
- **vs. CLIP**: CLIP is the canonical approach for cross-modal contrastive learning but relies on separate image and text encoders. MuCo is built on a unified MLLM, natively supporting multimodal fusion reasoning.
- **vs. UniIR**: UniIR unifies multiple retrieval tasks but remains a single-turn training paradigm. MuCo's multi-turn strategy can serve as a drop-in upgrade for its training efficiency.
- The core principle of the framework—*leveraging a model's existing capabilities (dialogue) to improve training efficiency*—offers broader inspiration for other MLLM fine-tuning scenarios.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The multi-turn dialogue contrastive learning perspective is novel, though the technical implementation is relatively straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validated on two mainstream benchmarks, MMEB and M-BEIR, though ablation analysis could be more fine-grained.
- **Writing Quality**: ⭐⭐⭐⭐ Method presentation is clear with a coherent motivational chain.
- **Value**: ⭐⭐⭐⭐ Provides a general solution for improving multimodal embedding training efficiency with strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Generalized Contrastive Learning for Universal Multimodal Retrieval](../../NeurIPS2025/information_retrieval/generalized_contrastive_learning_for_universal_multimodal_re.md)
- [\[CVPR 2026\] M4-RAG: A Massive-Scale Multilingual Multi-Cultural Multimodal RAG](m4-rag_a_massive-scale_multilingual_multi-cultural_multimodal_rag.md)
- [\[CVPR 2026\] Beyond Global Similarity: Towards Fine-Grained, Multi-Condition Multimodal Retrieval](beyond_global_similarity_towards_fine-grained_multi-condition_multimodal_retriev.md)
- [\[ACL 2026\] FLARE: Task-Agnostic Embedding Model Evaluation via Normalizing Flows](../../ACL2026/information_retrieval/flare_task-agnostic_embedding_model_evaluation_through_a_normalization_process.md)
- [\[ICLR 2026\] HUME: Measuring the Human-Model Performance Gap in Text Embedding Tasks](../../ICLR2026/information_retrieval/hume_measuring_the_human-model_performance_gap_in_text_embedding_tasks.md)

</div>

<!-- RELATED:END -->
