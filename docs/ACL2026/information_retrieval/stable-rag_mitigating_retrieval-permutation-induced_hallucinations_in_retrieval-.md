---
title: >-
  [Paper Note] Stable-RAG: Mitigating Retrieval-Permutation-Induced Hallucinations in Retrieval-Augmented Generation
description: >-
  [ACL 2026][Retrieval-Augmented Generation] This paper identifies a critical yet previously overlooked vulnerability in RAG systems—high sensitivity to the ordering of retrieved documents—and proposes Stable-RAG, which applies spectral clustering over hidden states induced by document permutations to identify dominant reasoning patterns, then uses DPO alignment to redirect hallucinated outputs toward correct answers, achieving simultaneous improvements in accuracy and reasoning consistency across three QA datasets.
tags:
  - ACL 2026
  - Retrieval-Augmented Generation
  - permutation sensitivity
  - hallucination
  - hidden-state clustering
  - preference alignment
date: 2026-05-08
content_hash: 8413afa3dea9ad53
---

# Stable-RAG: Mitigating Retrieval-Permutation-Induced Hallucinations in Retrieval-Augmented Generation

**Conference**: ACL 2026
**arXiv**: [2601.02993](https://arxiv.org/abs/2601.02993)
**Code**: [GitHub](https://github.com/zqc1023/Stable-RAG)
**Area**: Information Retrieval / RAG
**Keywords**: Retrieval-Augmented Generation, permutation sensitivity, hallucination, hidden-state clustering, preference alignment

## TL;DR

This paper identifies a critical yet previously overlooked vulnerability in RAG systems—high sensitivity to the ordering of retrieved documents—and proposes Stable-RAG, which applies spectral clustering over hidden states induced by document permutations to identify dominant reasoning patterns, then uses DPO alignment to redirect hallucinated outputs toward correct answers, achieving simultaneous improvements in accuracy and reasoning consistency across three QA datasets.

## Background & Motivation

**State of the Field**: RAG is a key paradigm for mitigating factual hallucinations in LLMs, providing evidence for generation by retrieving external documents. Current RAG research primarily focuses on retrieval quality (finding better documents) and positional bias (uneven attention in long contexts).

**Limitations of Prior Work**: The authors identify a previously neglected vulnerability—*permutation sensitivity*: even when the retrieved document set is identical (including the gold document), merely changing the document order can drive the model into entirely different reasoning paths, yielding inconsistent answers. On the NQ dataset, even with the gold document fixed in the first position, the permutation success rate (PSR) of LLMs remains high.

**Root Cause**: Permutation sensitivity is neither a retrieval quality issue (the document set is identical) nor a long-context positional bias problem (Top-5 retrieval stays within ~1,000 tokens), but rather a structural instability in the LLM's internal reasoning dynamics—as network depth increases, document permutations induce an increasing number of divergent reasoning trajectories.

**Paper Goals**: (1) Quantify and understand the underlying mechanism of permutation sensitivity; (2) Design a model-agnostic method that enables RAG to produce consistent and accurate outputs under arbitrary document orderings.

**Starting Point**: Through layer-by-layer visualization of spectral clustering behavior in hidden states, the authors find that shallow-layer hidden states are mixed, while deep-layer hidden states cluster cleanly by final answer. Sensitive samples exhibit far more clusters than insensitive ones, indicating that permutation effects are progressively amplified in higher layers.

**Core Idea**: Leverage permutation sensitivity estimation itself to eliminate permutation-induced hallucinations—apply spectral clustering to final-layer hidden states across all permutations, decode representative answers from each cluster centroid, construct preference data, and align the model via DPO.

## Method

### Overall Architecture

Stable-RAG proceeds in three stages: (1) **Hidden-state clustering**—run the model over all document permutations for query $q$, extract the hidden state of the last token at the final layer, and apply spectral clustering to identify reasoning patterns; (2) **Preference data construction**—extract correct/incorrect answer pairs from clustering results; (3) **DPO alignment**—train the model to produce consistent, correct outputs across different permutations.

### Key Designs

1. **Permutation Sensitivity Estimation via Spectral Clustering**

   - **Function**: Identify the latent reasoning patterns of the model under different document permutations.
   - **Mechanism**: For $N = n!$ permutations, extract the final-layer last-token hidden states $H \in \mathbb{R}^{N \times d}$, construct a cosine-distance similarity matrix $A_{ij} = \exp\!\left(-\frac{1 - \text{cos}(h^{(i)}, h^{(j)})}{\sigma}\right)$, compute the normalized graph Laplacian $L = I - D^{-1/2}AD^{-1/2}$, and adaptively determine the number of clusters $K$ via the eigengap heuristic. The representative hidden state closest to each cluster centroid is decoded, reducing $N = 120$ full decoding passes to only $K$.
   - **Design Motivation**: Exhaustively decoding all permutations incurs prohibitive computational and annotation costs; spectral clustering captures all reasoning patterns with only a small number of representative decodings. Quantitative validation yields F1 scores of 83.9% (LLaMA3) and 87.6% (Qwen3).

2. **Four-Category Preference Data Construction**

   - **Function**: Construct training signals tailored to different permutation sensitivity patterns.
   - **Mechanism**: Samples are divided into four categories: **FC** (all permutations correct—excluded from training); **PC** (partially correct—$y_w$ = most frequent correct answer, $y_l$ = most frequent incorrect answer); **FU** (all incorrect and unanswerable—$y_w$ = "I don't know" to encourage abstention); **FA** (all incorrect but answerable—$y_w$ = gold answer to encourage correct prediction).
   - **Design Motivation**: Different types of permutation inconsistency require different alignment strategies—PC samples require stabilizing existing capabilities, FU samples require learning to abstain, and FA samples require extracting correct answers from evidence.

3. **DPO Preference Alignment Training**

   - **Function**: Train the model to produce consistent outputs across different permutations.
   - **Mechanism**: Standard DPO loss is applied:
     $$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\!\left[\log\sigma\!\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$
     where the input $x$ is the concatenation of the query and a document permutation.
   - **Design Motivation**: DPO learns directly from preference pairs without training a separate reward model, and can efficiently leverage clustering results to construct high-quality preference data.

### Loss & Training

Standard DPO loss with hyperparameter $\beta$ controlling preference sharpness. Base models are LLaMA3-8B-Instruct and Qwen3-8B; retrievers are DPR and Contriever with Top-5 retrieval. Full permutations ($5! = 120$) are used on the training set for hidden-state clustering and preference data construction.

## Key Experimental Results

### Main Results

LLaMA3-8B-Instruct, SubEM (%) / F1 (%):

| Method | NQ (Contriever) | TriviaQA (DPR) | HotpotQA (Contriever) | Avg. SubEM |
|---|---|---|---|---|
| Vanilla RAG | 40.75 / 42.82 | 67.12 / 68.61 | 30.73 / 34.08 | 45.66 |
| RetRobust | 41.82 / 44.26 | 68.67 / 70.42 | 31.46 / 35.34 | 47.08 |
| ATM | 43.75 / 44.88 | 70.12 / 70.35 | 34.36 / 36.97 | 48.82 |
| **Stable-RAG** | **48.14** / 45.80 | **73.43** / **73.76** | **38.91** / **39.87** | **52.34** |

Qwen3-8B, SubEM (%):

| Method | NQ (Contriever) | TriviaQA (DPR) | HotpotQA (Contriever) | Avg. SubEM |
|---|---|---|---|---|
| Vanilla RAG | 44.65 | 69.62 | 33.14 | 48.08 |
| ATM | 45.47 | 70.06 | 35.12 | 49.24 |
| **Stable-RAG** | **46.12** | **71.32** | **35.73** | **50.27** |

### Ablation Study

Based on LLaMA3-8B-Instruct (Contriever, NQ SubEM):

| Configuration | Avg. SubEM | Abstention Rate AR |
|---|---|---|
| Full Stable-RAG | 52.34 | Moderate |
| w/o PC component | 42.51 | 35.1% |
| PC only (no FA/FU) | 51.96 | 0.0% |
| PC + FU (no FA) | 50.87 | 17.3% |

Clustering quality improves with layer depth (LLaMA3-8B, NQ, DPR):

| Layer | Precision | Recall | F1 |
|---|---|---|---|
| 8 | 69.2 | 71.8 | 69.3 |
| 16 | 81.4 | 82.5 | 81.3 |
| 24 | 82.3 | 83.7 | 82.2 |
| 32 | 84.1 | 85.2 | 83.9 |

### Key Findings

- Permutation sensitivity is pervasive: even with the gold document placed first, PSR remains 40–60% across LLaMA variants.
- Hidden-state clustering quality improves with network depth: F1 rises from 69.3% at layer 8 to 83.9% at layer 32 on LLaMA3.
- The PC (partially correct) component contributes most; removing it causes average SubEM to drop sharply from 52.34 to 42.51.
- Stable-RAG demonstrates strong generalization across datasets, retrievers, and Top-K settings.

## Highlights & Insights

- The definition of "permutation sensitivity" is precise and well-scoped—it is neither a retrieval quality problem nor a long-context problem—opening a new dimension in RAG robustness research.
- Layer-wise hidden-state visualization intuitively reveals how permutations affect internal reasoning: shallow layers are mixed while deep layers diverge, directly motivating the method design.
- The four-category preference data construction strategy (FC/PC/FU/FA) reflects fine-grained training signal design tailored to distinct failure modes.

## Limitations & Future Work

- Full permutation enumeration ($n!$) incurs substantial computational overhead (120 forward passes for Top-5); sampling strategies are needed when scaling to larger retrieved sets.
- Evaluation is currently limited to QA tasks; permutation sensitivity may manifest differently in long-form generation and other settings.
- The combination of training-time and inference-time approaches (e.g., multi-permutation decoding with majority voting at inference) remains unexplored.

## Related Work & Insights

- RetRobust and RAAT improve robustness by injecting retrieval noise during training but do not address permutation issues; ATM considers permutation perturbations but does not explicitly model reasoning trajectories.
- Pos2Distill and Ms-PoE focus on positional bias in long contexts, whereas permutation sensitivity persists even in short contexts (<1,000 tokens).
- **Insight**: LLM reasoning paths are far more fragile than commonly assumed—minor structural changes to the input (without altering semantic content) can lead to entirely different outputs.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The problem formulation is novel and significant; permutation sensitivity is a seriously underexplored vulnerability in the RAG literature.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three datasets, two retrievers, and two base models, with complete ablation and generalization experiments.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear logical flow, outstanding visualizations, and naturally motivated problem formulation.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] MASS-RAG: Multi-Agent Synthesis Retrieval-Augmented Generation](mass-rag_multi-agent_synthesis_retrieval-augmented_generation.md)
- [\[ACL 2026\] Feedback Adaptation for Retrieval-Augmented Generation](feedback_adaptation_for_retrieval-augmented_generation.md)
- [\[ACL 2026\] TPA: Next Token Probability Attribution for Detecting Hallucinations in RAG](tpa_next_token_probability_attribution_for_detecting_hallucinations_in_rag.md)
- [\[ACL 2026\] All Languages Matter: Understanding and Mitigating Language Bias in Multilingual RAG](all_languages_matter_understanding_and_mitigating_language_bias_in_multilingual_.md)
- [\[ACL 2026\] Beyond Explicit Refusals: Soft-Failure Attacks on Retrieval-Augmented Generation](beyond_explicit_refusals_soft-failure_attacks_on_retrieval-augmented_generation.md)

<!-- RELATED:END -->
