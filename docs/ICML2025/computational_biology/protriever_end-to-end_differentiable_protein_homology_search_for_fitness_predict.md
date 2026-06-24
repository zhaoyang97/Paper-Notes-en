---
title: >-
  [Paper Note] Protriever: End-to-End Differentiable Protein Homology Search for Fitness Prediction
description: >-
  [ICML 2025][Computational Biology][Protein Homology Search] Protriever is proposed as the first end-to-end differentiable protein homology sequence retrieval framework, which jointly trains the retriever and the reader. It achieves sequence-model SOTA performance on protein fitness prediction tasks while being two orders of magnitude faster than traditional MSA retrieval.
tags:
  - "ICML 2025"
  - "Computational Biology"
  - "Protein Homology Search"
  - "End-to-End Differentiable Retrieval"
  - "Protein Fitness Prediction"
  - "Vector Retrieval"
  - "Protein Language Models"
date: 2026-05-08
content_hash: 083096834bd7755d
---

# Protriever: End-to-End Differentiable Protein Homology Search for Fitness Prediction

**Conference**: ICML 2025  
**arXiv**: [2506.08954](https://arxiv.org/abs/2506.08954)  
**Area**: Protein Modeling / Computational Biology  
**Keywords**: Protein Homology Search, End-to-End Differentiable Retrieval, Protein Fitness Prediction, Vector Retrieval, Protein Language Models

## TL;DR

Protriever is proposed as the first end-to-end differentiable protein homology sequence retrieval framework, which jointly trains the retriever and the reader. It achieves sequence-model SOTA performance on protein fitness prediction tasks while being two orders of magnitude faster than traditional MSA retrieval.

## Background & Motivation

- **Protein homology sequence retrieval** is a fundamental step in protein modeling (including fitness prediction, protein design, structure prediction, and protein-protein interactions).
- Traditional workflows rely on a **two-stage pipeline**: first retrieving homologous sequences via multiple sequence alignment (MSA), and then training models on these alignments.
- Traditional MSA-based methods (such as JackHMMER and MMseqs2) suffer from fundamental limitations:
    - **Missing distant homologs**: Distant homologous sequences falling below the alignment significance threshold are missed, leading to the loss of valuable evolutionary context.
    - **Difficulty handling complex indels**: Sequences with extensive insertions, deletions, or structural rearrangements cannot be reliably aligned.
    - **Disconnection from downstream tasks**: Retrieval is based on fixed sequence similarity heuristics rather than being optimized for specific tasks.
    - **Computationally expensive**: Generating MSAs and training models for each protein family individually is highly inefficient, rendering it unsuitable for large-scale applications.
- **Protein Language Models (pLM)**, such as ESM and Tranception, offer alignment-free alternatives; however, single-sequence models often underperform family-specific methods in mutation effect prediction.
- Although **hybrid methods** integrate pLMs and evolutionary information, their retrieval processes remain static and cannot propagate backward to optimize retrieval selection.
- **Key Insight**: Introducing the Retrieval-Augmented Generation (RAG) paradigm from NLP into protein modeling—allowing the model to autonomously learn which homologous sequences are most beneficial for downstream tasks.

## Method

### Overall Architecture

Protriever consists of three core components (as shown in Figure 1):

1. **Retriever**: Encodes query sequences into vectors and retrieves homologous sequences from an index via vector similarity search.
2. **Index**: A pre-computed library of protein sequence embeddings that supports fast approximate nearest neighbor search.
3. **Reader**: Receives the set of retrieved sequences and executes downstream tasks (e.g., autoregressively decoding the query sequence).

During training, the Reader calculates a relevance score $p_{\text{LM}}(\mathbf{q} \mid \mathbf{d}_k)$ for each retrieved document, and gradients are backpropagated to the Retriever to adjust the embedding space.

### Retriever Module

**Initialization**:
- Initialize a Transformer encoder (35M parameters) using pretrained ESM-2 weights.
- Perform average pooling on the final layer output to obtain a 480-dimensional vector representation.
- Compute similarity between sequences using cosine similarity $s(\mathbf{d}, \mathbf{q})$.

**DPR Pre-training**:
- Further pre-train using the Dense Passage Retrieval (DPR) paradigm.
- Objective: Construct an embedding space that maps homologous sequences to high similarity and non-homologous sequences to low similarity.
- Construct training data using BLAST all-vs-all searches on UniRef50.
- Contrastive learning loss: Given a query $\mathbf{q}_i$, maximize positive sample similarity and minimize negative sample similarity.

$$\mathcal{L}_{\text{DPR}} = -\log \frac{e^{s(\mathbf{q}_i, \mathbf{d}_i^+)}}{e^{s(\mathbf{q}_i, \mathbf{d}_i^+)} + \sum_j e^{s(\mathbf{q}_i, \mathbf{d}_{i,j}^-)}}$$

**Hard Negative Mining**:
- Retrieve top-K similar but non-homologous sequences from the index to serve as hard negatives.
- Asynchronously update the index during training to maintain the quality of negative samples.

### Reader Module

- Employ an autoregressive protein language model (ProtGPT2, 36M parameters).
- The Reader reconstructs the query sequence autoregressively, conditioned on the concatenated retrieved sequences.
- Compute the conditional likelihood independently for each retrieved document to serve as the relevance score.
- The Reader loss is a weighted negative log-likelihood.

$$\mathcal{L}_{\text{reader}} = -\sum_{t=1}^{T} \log p(\mathbf{q}_t \mid \mathbf{q}_{<t}, \mathcal{D}_{K_f})$$

### Joint Training

- **End-to-End Gradient Propagation**: Gradients from the Reader are backpropagated to the Retriever through the relevance scores.
- Multiple loss combination strategies: Reader-only loss, joint DPR + Reader loss, and weighted combinations.
- **Sampling Strategies**: Explore different sequence sampling methods (e.g., top-K, temperature-scaled sampling) to optimize performance.

### Accelerating Retrieval

- **Inverted File Index (IVF)**: Partition the vector space into Voronoi cells, searching only the nearest bins.
- **Product Quantization (PQ)**: Compress 480-dimensional vectors into short codes, significantly reducing storage and distance computation overhead.
- Leverage the FAISS library for efficient approximate nearest neighbor search.

## Key Experimental Results

### Table 1: Fitness Prediction Performance on the ProteinGym Benchmark (Spearman rho)

| Method | Category | Retrieval Method | Average Spearman rho |
|------|------|----------|----------------|
| EVE | Family-specific | MSA (JackHMMER) | 0.456 |
| DeepSequence | Family-specific | MSA | 0.440 |
| ESM-1v | Single-sequence pLM | None | 0.421 |
| ESM-2 (650M) | Single-sequence pLM | None | 0.434 |
| Tranception (L) | Hybrid | MSA (At inference time) | 0.462 |
| PoET | Hybrid | MSA (JackHMMER) | 0.467 |
| ProtMamba | Hybrid | MSA | 0.459 |
| **Protriever** | **End-to-end retrieval** | **Vector search** | **0.471** |

> Protriever achieves the best performance among sequence models, outperforming all MSA-retrieval-based methods.

### Table 2: Retrieval Speed Comparison

| Retrieval Method | Database Size | Retrieval Time per Sequence | Relative Speed |
|----------|-----------|---------------|---------|
| JackHMMER | UniRef90 | ~300s | 1x |
| MMseqs2 (CPU) | UniRef90 | ~30s | 10x |
| MMseqs2-GPU | UniRef90 | ~3s | 100x |
| **Protriever (FAISS)** | **UniRef50** | **~0.3s** | **1000x** |

> Vector retrieval is approximately three orders of magnitude faster than JackHMMER and two orders of magnitude faster than MMseqs2.

### Table 3: Ablation Study — Impact of Training Strategies and Components

| Configuration | Spearman rho | Description |
|------|-----------|------|
| Reader only (Random retrieval) | 0.412 | No meaningful retrieval |
| Reader + Frozen ESM-2 retriever | 0.438 | No retriever updates |
| Reader + DPR pre-trained retriever (Frozen) | 0.451 | DPR pre-training is effective |
| Reader + DPR retriever (Fine-tuned) | 0.463 | End-to-end fine-tuning yields further improvement |
| **Protriever (Full joint training)** | **0.471** | Joint training is optimal |

> Ablation study demonstrates: (1) DPR pre-training provides a solid initialization for the retriever; (2) End-to-end joint training yields significant gains compared to a frozen retriever (+0.020).

### Table 4: Different Retrieval Databases and Inference Flexibility

| Inference-time Database | Spearman rho | Description |
|-------------|-----------|------|
| UniRef50 | 0.471 | Same database as training |
| UniRef90 | 0.468 | Larger database, generalizes well |
| BFD | 0.465 | Cross-database generalization |
| Using only top-10 sequences | 0.464 | Effective even with limited retrieval |
| Using top-100 sequences | 0.471 | Slightly better with more retrieved sequences |

> Protriever can flexibly switch databases and the number of retrieved sequences at inference time, demonstrating architecture-agnostic and task-agnostic properties.

## Highlights & Insights

1. **First End-to-End Differentiable Protein Homology Retrieval**: Breaks the traditional two-stage "MSA-to-model" paradigm by jointly optimizing the retriever and reader, enabling the model to autonomously learn which homologous sequences are most valuable for downstream tasks.
2. **Ultra-Fast Retrieval**: Achieves retrieval speeds approximately 1000x faster than JackHMMER through FAISS vector indexing (IVF + PQ), making large-scale protein analysis feasible.
3. **Architecture- and Task-Agnostic**: Both the retriever and the reader can be independently replaced, and different protein databases can be switched at inference time, allowing flexible adaptation to various downstream tasks.
4. **Successful Migration of NLP RAG to Biology**: Inspired by REALM/RAG, it successfully transfers the differentiable retriever-reader framework from NLP to protein sequence modeling with key structural adaptations for the protein domain.
5. **Lightweight and Efficient**: The retriever has only 35M parameters (small ESM-2 model) and the reader has 36M parameters. The total model size is significantly smaller than large language models like ESM-2 (650M).

## Limitations & Future Work

1. **Dependency on Retrieval Databases**: Retrieval quality remains constrained by the coverage of protein sequences in the index, which may limit performance on extremely rare protein families.
2. **Asynchronous Index Updates**: The index is not updated in real-time during training (rebuilt periodically), leading to a potential lag in retrieval consistency.
3. **Validation Limited to Fitness Prediction**: Although claimed to be architecture- and task-agnostic, it has only been validated on a single downstream task: fitness prediction.
4. **Reader Capacity Constraints**: The ProtGPT2 (36M) used as the Reader is relatively small. A larger Reader might yield better performance but at the cost of increased overhead.
5. **Noise Introduced by Approximate Search**: The IVF + PQ compression used to accelerate retrieval introduces approximation errors, which may affect a small number of hard cases.

## Related Work & Insights

- **Alignment Methods**: PSSM, HMM -> EVE, DeepSequence -> family-specific models reliant on MSAs.
- **Protein Language Models**: UniRep -> ESM, ESM-2, ProGen, Tranception, ProtGPT2 -> alignment-free but lacking family specificity.
- **Hybrid Methods**: MSA Transformer, PoET, ProtMamba -> combining pLMs with evolutionary information, but retrieval remains static.
- **Retrieval-Augmented**: DPR, REALM, RAG in NLP -> AIDO.RAG, RSA in the protein domain -> but none have achieved end-to-end joint training.
- **MSA Retrieval Tools**: JackHMMER, MMseqs2, BLAST -> traditional sequence similarity search.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty** ⭐⭐⭐⭐⭐: Achieves the first end-to-end differentiable training of protein homology retrieval, representing a significant conceptual breakthrough.
- **Experimental Thoroughness** ⭐⭐⭐⭐: Thorough comparison on the standard ProteinGym benchmark with detailed ablation studies.
- **Value** ⭐⭐⭐⭐⭐: Increases retrieval speed by 2 to 3 orders of magnitude while maintaining SOTA performance, indicating high industrial application value.
- **Writing Quality** ⭐⭐⭐⭐: Logical structure and detailed methodological descriptions.
- **Limitations**: Validated only on a single downstream task (fitness prediction); the small-scale Reader might limit the performance ceiling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MolParser: End-to-end Visual Recognition of Molecule Structures in the Wild](../../ICCV2025/computational_biology/molparser_end-to-end_visual_recognition_of_molecule_structures_in_the_wild.md)
- [\[NeurIPS 2025\] UniSite: The First Cross-Structure Dataset and Learning Framework for End-to-End Ligand Binding Site Detection](../../NeurIPS2025/computational_biology/unisite_the_first_cross-structure_dataset_and_learning_framework_for_end-to-end_.md)
- [\[ICLR 2026\] Optimal Transport Unlocks End-to-End Learning for Single-Molecule Localization](../../ICLR2026/computational_biology/optimal_transport_unlocks_end-to-end_learning_for_single-molecule_localization.md)
- [\[ICLR 2026\] A Cross-Species Neural Foundation Model for End-to-End Speech Decoding](../../ICLR2026/computational_biology/a_cross-species_neural_foundation_model_for_end-to-end_speech_decoding.md)
- [\[NeurIPS 2025\] Steering Generative Models with Experimental Data for Protein Fitness Optimization](../../NeurIPS2025/computational_biology/steering_generative_models_with_experimental_data_for_protein_fitness_optimizati.md)

</div>

<!-- RELATED:END -->
