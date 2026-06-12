---
title: >-
  [Paper Note] Stable-RAG: Mitigating Retrieval-Permutation-Induced Hallucinations in Retrieval-Augmented Generation
description: >-
  [ACL 2026][Hallucination Detection][Retrieval-Augmented Generation] This paper reveals the high sensitivity of RAG systems to the permutation order of retrieved documents and proposes Stable-RAG: it identifies dominant r…
tags:
  - "ACL 2026"
  - "Hallucination Detection"
  - "Retrieval-Augmented Generation"
  - "Permutation Sensitivity"
  - "Hallucination"
  - "Hidden State Clustering"
  - "Preference Alignment"
date: 2026-05-08
content_hash: c5d77dc5458562d4
---

# Stable-RAG: Mitigating Retrieval-Permutation-Induced Hallucinations in Retrieval-Augmented Generation

**Conference**: ACL 2026  
**arXiv**: [2601.02993](https://arxiv.org/abs/2601.02993)  
**Code**: [GitHub](https://github.com/zqc1023/Stable-RAG)  
**Area**: Hallucination Detection  
**Keywords**: Retrieval-Augmented Generation, Permutation Sensitivity, Hallucination, Hidden State Clustering, Preference Alignment

## TL;DR

This paper reveals the high sensitivity of RAG systems to the permutation order of retrieved documents and proposes Stable-RAG: it identifies dominant reasoning patterns by performing spectral clustering on hidden states induced by document permutations, and then employs DPO alignment to guide hallucinated outputs toward correct answers, achieving dual improvements in accuracy and reasoning consistency across weight QA datasets.

## Background & Motivation

**Background**: RAG is a critical paradigm for mitigating factual hallucinations in LLMs by retrieving external documents to provide evidentiary support. Current RAG research primarily focuses on retrieval quality (how to find better documents) and positional bias (uneven attention in long contexts).

**Limitations of Prior Work**: The authors identify a previously overlooked vulnerability—permutation sensitivity: even when the set of retrieved documents is identical (containing the gold document), merely changing the order can lead the model down different reasoning paths, producing inconsistent answers. On the NQ dataset, the Permutation Success Rate (PSR) remains high even when the gold document is fixed in the first position.

**Key Challenge**: Permutation sensitivity is neither a retrieval quality issue (the document set is identical) nor a long-context positional bias (Top-5 is within a thousand tokens); rather, it stems from the structural instability of LLM internal reasoning dynamics—as network depth increases, document permutations induce an increasing number of divergent reasoning trajectories.

**Goal**: (1) Quantify and understand the internal mechanisms of permutation sensitivity; (2) Design a model-agnostic method to enable RAG to generate consistent and accurate outputs under any document permutation.

**Key Insight**: By visualizing the spectral clustering behavior of hidden states layer-by-layer, the authors found that while shallow hidden states are mixed, deep hidden states clear cluster according to the final answer, and sensitive samples exhibit significantly more clusters than non-sensitive ones. This indicates that permutation effects are amplified in higher layers.

**Core Idea**: Leverage the estimation of permutation sensitivity itself to eliminate permutation-induced hallucinations—perform spectral clustering on the final-layer hidden states of all permutations, decode representative answers from the cluster centers, and construct preference data to align the model via DPO.

## Method

### Overall Architecture

Stable-RAG consists of three phases: (1) Hidden state clustering—running the model on all document permutations for a query $q$ to extract the final-layer hidden state of the last token and performing spectral clustering to identify reasoning patterns; (2) Preference data construction—extracting correct/incorrect answer pairs from the clustering results; (3) DPO alignment—training the model to produce consistent correct outputs under different permutations.

### Key Designs

1.  **Permutation Sensitivity Estimation Based on Spectral Clustering**:

    - **Function**: Identify the model's latent reasoning patterns under different document permutations.
    - **Mechanism**: Extract the final-layer last token hidden states $H \in \mathbb{R}^{N \times d}$ for all $N=n!$ permutations. Construct a cosine distance similarity matrix $A_{ij} = \exp(-\frac{1-\text{cos}(h^{(i)}, h^{(j)})}{\sigma})$ and calculate the normalized graph Laplacian $L = I - D^{-1/2}AD^{-1/2}$. The number of clusters $K$ is adaptively determined via the eigengap. Representative answers are decoded from hidden states closest to the centroids, reducing $N=120$ full decodings to $K$.
    - **Design Motivation**: Exhaustive decoding and annotation for all permutations are prohibitively expensive; spectral clustering captures all reasoning patterns with minimal representative decodings. Quantitative validation shows F1 scores of 83.9% (LLaMA3) and 87.6% (Qwen3).

2.  **Four Categories of Preference Data Construction**:

    - **Function**: Construct training signals for different permutation sensitivity patterns.
    - **Mechanism**: Samples are categorized into four types: FC (Fully Correct, excluded from training), PC (Partially Correct, $y_w$ = most frequent correct answer, $y_l$ = most frequent incorrect answer), FU (Fully Unanswerable, $y_w$ = "I don't know" to encourage abstention), and FA (Fully Answerable but incorrect, $y_w$ = gold answer to encourage correct prediction).
    - **Design Motivation**: Different types of permutation inconsistency require distinct alignment strategies—PC requires stabilizing existing capabilities, FU requires learning to abstain, and FA requires extracting the correct answer from evidence.

3.  **DPO Preference Alignment Training**:

    - **Function**: Train the model to generate consistent outputs across different permutations.
    - **Mechanism**: Use the standard DPO loss $\mathcal{L}_{\text{DPO}} = -\mathbb{E}[\log\sigma(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)})]$, where input $x$ is the concatenation of the query and a specific document permutation.
    - **Design Motivation**: DPO learns directly from preference pairs without requiring a separate reward model and efficiently utilizes high-quality preference data constructed from clustering.

### Loss & Training

Standard DPO loss is used, with hyperparameter $\beta$ controlling preference sharpness. Base models are LLaMA3-8B-Instruct and Qwen3-8B, with DPR and Contriever as retrievers using Top-5 retrieval. Hidden state clustering and preference data construction are performed on the training set using full permutations ($5!=120$ types).

## Key Experimental Results

### Main Results

LLaMA3-8B-Instruct, SubEM (%) / F1 (%):

| Method | NQ (Contriever) | TriviaQA (DPR) | HotpotQA (Contriever) | Average SubEM |
|------|-----------------|----------------|----------------------|------------|
| Vanilla RAG | 40.75 / 42.82 | 67.12 / 68.61 | 30.73 / 34.08 | 45.66 |
| RetRobust | 41.82 / 44.26 | 68.67 / 70.42 | 31.46 / 35.34 | 47.08 |
| ATM | 43.75 / 44.88 | 70.12 / 70.35 | 34.36 / 36.97 | 48.82 |
| **Ours** | **48.14** / 45.80 | **73.43** / **73.76** | **38.91** / **39.87** | **52.34** |

Qwen3-8B, SubEM (%):

| Method | NQ (Contriever) | TriviaQA (DPR) | HotpotQA (Contriever) | Average SubEM |
|------|-----------------|----------------|----------------------|------------|
| Vanilla RAG | 44.65 | 69.62 | 33.14 | 48.08 |
| ATM | 45.47 | 70.06 | 35.12 | 49.24 |
| **Ours** | **46.12** | **71.32** | **35.73** | **50.27** |

### Ablation Study

Based on LLaMA3-8B-Instruct (Contriever, NQ SubEM):

| Component | Average SubEM | Abstention Rate (AR) |
|------|-----------|----------|
| Full Stable-RAG | 52.34 | Moderate |
| w/o PC | 42.51 | 35.1% |
| PC only (no FA/FU) | 51.96 | 0.0% |
| PC + FU (no FA) | 50.87 | 17.3% |

Clustering quality improves with layer depth (LLaMA3-8B, NQ, DPR):

| Layer | Precision | Recall | F1 |
|----|-----------|--------|-----|
| 8 | 69.2 | 71.8 | 69.3 |
| 16 | 81.4 | 82.5 | 81.3 |
| 24 | 82.3 | 83.7 | 82.2 |
| 32 | 84.1 | 85.2 | 83.9 |

### Key Findings

- Permutation sensitivity is a universal phenomenon: even with the gold document in the first position, PSR for various LLaMA versions remains between 40-60%.
- Hidden state clustering quality improves with network depth: F1 for LLaMA3 increases from 69.3% at layer 8 to 83.9% at layer 32.
- The PC (Partially Correct) component contributes the most; removing it drops the average SubEM from 52.34 to 42.51.
- Stable-RAG demonstrates strong generalization across different datasets, retrievers, and Top-K settings.

## Highlights & Insights

- The definition of "permutation sensitivity" is precise—it is distinct from retrieval quality and long-context issues, opening a new dimension for RAG robustness research.
- Layer-wise hidden state visualization intuitively reveals how permutations affect internal reasoning: from mixed shallow layers to divergent deep layers, providing direct inspiration for method design.
- The four categories of preference data (FC/PC/FU/FA) reflect a fine-grained design of training signals.

## Limitations & Future Work

- The computational overhead of full permutations ($n!$) is high (Top-5 requires 120 forward passes); sampling strategies are needed for more retrieved documents.
- Currently evaluated only on QA tasks; permutation sensitivity might manifest differently in scenarios like long-form text generation.
- The combination of training-time and inference-time methods (e.g., multi-permutation decoding + voting at inference) remains unexplored.

## Related Work & Insights

- RetRobust and RAAT train robustness by injecting retrieval noise but do not address permutation issues; ATM considers permutation perturbations but does not explicitly model reasoning trajectories.
- Pos2Distill and Ms-PoE focus on long-context positional bias, whereas permutation sensitivity exists even in short contexts (<1000 tokens).
- Insight: LLM reasoning paths are far more fragile than expected; minor structural changes in input (without altering semantic content) can lead to completely different outputs.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The problem definition is novel and important; permutation sensitivity is a significantly overlooked vulnerability in RAG.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive experiments across three datasets, two retrievers, and two base models, with complete ablation and generalization tests.
- Writing Quality: ⭐⭐⭐⭐ Clear logical chain, excellent visualization charts, and natural derivation of problem motivation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Understanding New-Knowledge-Induced Factual Hallucinations in LLMs: Analysis and Interpretation](understanding_new-knowledge-induced_factual_hallucinations_in_llms_analysis_and_.md)
- [\[ACL 2026\] TPA: Next Token Probability Attribution for Detecting Hallucinations in RAG](tpa_next_token_probability_attribution_for_detecting_hallucinations_in_rag.md)
- [\[ACL 2026\] Mechanisms of Prompt-Induced Hallucination in Vision–Language Models](mechanisms_of_prompt-induced_hallucination_in_vision-language_models.md)
- [\[ICLR 2026\] LUMINA: Detecting Hallucinations in RAG System with Context-Knowledge Signals](../../ICLR2026/hallucination/lumina_detecting_hallucinations_in_rag_system_with_context-knowledge_signals.md)
- [\[ICML 2026\] Building Reliable Long-Form Generation via Hallucination Rejection Sampling](../../ICML2026/hallucination/building_reliable_long-form_generation_via_hallucination_rejection_sampling.md)

</div>

<!-- RELATED:END -->
