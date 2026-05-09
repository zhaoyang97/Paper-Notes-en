---
title: >-
  [Paper Note] A Partition Cover Approach for Tokenization
description: >-
  [NeurIPS 2025][Model Compression][tokenization] This paper reformulates tokenization as a **partition cover** optimization problem, proves it NP-hard, and proposes a polynomial-time greedy algorithm GreedTok that outperforms BPE in both compression rate and downstream tasks when pretraining a 1B-parameter LLM.
tags:
  - NeurIPS 2025
  - Model Compression
  - tokenization
  - BPE
  - partition cover
  - NP-hard
  - greedy algorithm
date: 2026-05-08
content_hash: 12353c306c3b6c03
---

# A Partition Cover Approach for Tokenization

**Conference**: NeurIPS 2025
**arXiv**: [2501.06246](https://arxiv.org/abs/2501.06246)
**Code**: [https://github.com/PreferredAI/pcatt/](https://github.com/PreferredAI/pcatt/)
**Area**: Model Compression
**Keywords**: tokenization, BPE, partition cover, NP-hard, greedy algorithm

## TL;DR
This paper reformulates tokenization as a **partition cover** optimization problem, proves it NP-hard, and proposes a polynomial-time greedy algorithm GreedTok that outperforms BPE in both compression rate and downstream tasks when pretraining a 1B-parameter LLM.

## Background & Motivation
Tokenization is a fundamental component of LLMs, responsible for encoding text into token sequences over a fixed vocabulary. The dominant approach is BPE (Byte-Pair Encoding), which treats tokenization as a compression problem and builds the vocabulary through iterative pairwise merges. Unigram, another approach, prunes top-down from a large candidate set. However, BPE's pairwise merge mechanism imposes an **artificial algorithmic constraint**—many intermediate tokens produced during merging may never appear in the final encoding, wasting vocabulary capacity. Unigram's likelihood objective tends to over-favor whole-word tokens at the expense of informative subword tokens. Theoretical understanding of tokenization remains limited, lacking clear optimization objectives and complexity analyses.

## Core Problem
Can tokenization be understood from a more general optimization perspective, moving beyond BPE's pairwise merging and Unigram's pruning paradigm? Specifically: (1) What is the computational complexity of tokenization? (2) Can a tokenization algorithm be designed that is free from merge-sequence constraints while achieving strong practical performance?

## Method

### Overall Architecture
The authors propose a **Partition Cover** formalization of tokenization: given a corpus $\mathcal{C}=(\mathbf{W}, \text{count})$, a candidate token set $\mathbf{T}$, and a budget $k$, the goal is to select a subset $\mathbf{S} \subseteq \mathbf{T}$ ($|\mathbf{S}| \leq k$) such that the total number of tokens when optimally segmenting all words in the corpus using $\mathbf{S}$ together with the base single-character token set $\mathbf{B}$ is minimized.

The core idea is to treat "whether two adjacent characters are covered by the same token" as the fundamental decision variable, rather than constraining tokens to be produced through stepwise merges as in BPE. This formalization **subsumes all merge-based solutions** as feasible solutions, but admits a strictly larger feasible space—not limited to vocabularies reachable by bottom-up merging.

### Key Designs

1. **NP-hardness Proof**: Tokenization is shown to be NP-hard via a straightforward reduction from vertex cover. For each edge $\{V_i, V_j\}$ in graph $\mathcal{G}=(V,E)$, a word $(@,V_i,@,V_j,@)$ is constructed, with candidate tokens of the form $\{(@,V_i,@)\}$. Selecting a token to cover a word is thus equivalent to selecting a vertex to cover an edge. Each word has length 5 and each token length 3, so a covered word yields a partition of size 3, and an uncovered one of size 5. The proof is clean and intuitive.

2. **Mixed Integer Programming (MIP) Formulation**: Binary variables $m_{i,i+1}^W$ are introduced to indicate whether characters $i$ and $i+1$ in word $W$ are covered by the same token. Minimizing the partition count is reformulated as an equivalent maximization of cover. The key observation is $|W| = \text{partition} + \text{cover}$, so minimizing partition is equivalent to maximizing cover. The MIP encodes token selection constraints, coverage consistency constraints, non-overlap constraints, and more.

3. **Connection to Weighted Maximum Coverage (WMC)**: The MIP naturally relaxes to the Weighted Maximum Coverage problem by dropping the non-overlap and continuity constraints. WMC admits a classical greedy $(1-1/e)$-approximation algorithm GreedWMC. Since the WMC optimum is no less than the tokenization optimum, the GreedWMC solution serves as an empirical upper bound on the approximation ratio of GreedTok.

4. **GreedTok Algorithm**: The algorithm proceeds in two phases. (1) Token set selection: starting from $\mathbf{S}=\emptyset$, greedily add the token yielding the greatest marginal gain in the MIP objective at each round until $|\mathbf{S}|=k$. (2) Encoding: all matching positions of tokens in $\mathbf{S}$ are identified; tokens are applied in the order they were added to $\mathbf{S}$, with later-added, longer tokens allowed to override earlier shorter-token decisions. The selection complexity is $O(|\mathbf{T}|\cdot k \cdot \sum|W|)$, substantially accelerated in practice via lazy evaluation; encoding complexity is $O(|W|^2 \log|W|)$.

### Loss & Training
- The objective function of GreedTok is **neither submodular nor supermodular** (demonstrated via counterexamples), so standard submodular maximization approximation analyses do not apply.
- Empirically, as $k$ increases, the ratio $d_{\text{inst}}$ of GreedTok's objective to GreedWMC's approaches 1, achieving at least a $0.9(1-1/e)$ approximation ratio in practice.

## Key Experimental Results

### Compression Experiments

| Dataset | Metric (avg tokens/word) | GreedTok | BPE | Gain |
|--------|------|------|----------|------|
| UN (k=5000) | tokens/word | 1.163 | 1.194 | 2.54% |
| arXiv (k=5000) | tokens/word | 1.226 | 1.263 | 2.89% |
| wiki (k=10000) | tokens/word | 1.283 | 1.301 | 1.37% |
| PubMed (k=10000) | tokens/word | 1.206 | 1.225 | 1.52% |

GreedTok outperforms BPE by an average of 2.88% and Unigram by 3.43%.

### Pretraining Experiments (1B-parameter LLM)

| Model | Avg Accuracy | Wikitext bits/byte |
|------|-------------|-------------------|
| BPEM (BPE) | 62.0 | 0.7066 |
| GTEP (GreedTok, same data proportion) | 62.5 | 0.7028 |
| GTET (GreedTok, same token count) | **63.2** | **0.6989** |

When trained on the same number of tokens (approximately 20% of DCLM), GreedTok outperforms BPE on both average accuracy across 11 benchmarks and bits/byte on Wikitext. GreedTok reduces the total token count of the full DCLM dataset by approximately 18%.

### Ablation Study
- Token set analysis shows that GreedTok's vocabulary simultaneously inherits the high compression rate of BPE and the high token quality of Unigram (more complete words, Unigram likelihood closer to optimal).
- The approximation ratio $d_{\text{inst}}$ approaches 0.95–1.0 for $k \geq 5000$, indicating that GreedTok nearly finds the optimal solution in practical NLP settings.
- 40 independent sampling experiments on RefinedWeb further confirm the trend $d_{\text{inst}} \to 1$.

## Highlights & Insights
- **Novel perspective**: Tokenization is recast from a "compression/merge" viewpoint to a "coverage" viewpoint, establishing clear connections to classical combinatorial optimization problems such as vertex cover and weighted maximum coverage.
- **Remarkably concise NP-hardness proof**: The direct reduction from vertex cover is significantly simpler than the MAX-2SAT reduction used in the concurrent work WBP (2024).
- **Flexibility of the MIP formulation**: The objective can naturally accommodate additional constraints (e.g., fairness, downstream task objectives), which is not achievable with BPE or Unigram.
- **Practical usability**: A C++ implementation with Python bindings and HuggingFace integration is provided, achieving encoding speeds of 700–800K words/s/thread and constructing a 10K-vocabulary on Wikipedia in 11 minutes.

## Limitations & Future Work
- The theoretical complexity of GreedTok's token selection, $O(|\mathbf{T}|\cdot k \cdot \sum|W|)$, is higher than BPE's $O(k \cdot \sum|W|)$. Although lazy evaluation substantially reduces wall-clock time, computational bottlenecks remain for very large corpora (34 minutes and 160 GB RAM for 14.3M unique words).
- A formal approximation ratio proof for GreedTok is absent; only an empirical $0.9(1-1/e)$ guarantee is provided.
- Pretraining experiments are limited to 1B-parameter models and comparisons with BPE only; validation on larger models or against Unigram in pretraining settings is lacking.
- Low-resource and multilingual scenarios are not evaluated.
- Encoding speed, while usable, remains slower than optimized implementations such as BPE/tiktoken.

## Related Work & Insights
- **vs. BPE**: BPE is constrained to pairwise merge paths, wasting vocabulary capacity on intermediate tokens. GreedTok imposes no such constraint, achieving ~3% better compression and superior pretraining performance.
- **vs. Unigram**: Unigram's top-down pruning biases toward whole-word tokens. GreedTok builds bottom-up, combining Unigram's token quality with BPE's compression efficiency.
- **vs. WBP (2024, concurrent)**: WBP also proves tokenization NP-hard but via a MAX-2SAT reduction, which is more complex. The vertex cover reduction in this work is more concise and intuitive.
- **vs. KV (2024)**: KV proves that the underlying merge problem in BPE is APX-complete, but remains restricted to the merge framework. The formalization in this paper is more general.

The paper demonstrates the value of revisiting foundational NLP components through the lens of combinatorial optimization. The MIP formulation provides a natural framework for incorporating additional objectives (e.g., downstream task signals, fairness constraints) into tokenization. This methodological paradigm—reformulating "taken-for-granted" components in deep learning pipelines as optimization problems—may offer broader inspiration for other NLP and multimodal foundations, such as the design of visual tokenizers.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Redefining tokenization from a coverage perspective is both unique and intellectually deep.
- Experimental Thoroughness: ⭐⭐⭐⭐ Compression experiments are comprehensive; pretraining experiments are reasonably scaled but limited to 1B-parameter models.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are clear, proofs are concise and elegant, and the overall structure is well-organized.
- Value: ⭐⭐⭐⭐ Provides a new theoretical foundation and practical algorithm for tokenization; the MIP framework has considerable extension potential.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] zip2zip: Inference-Time Adaptive Tokenization via Online Compression](zip2zip_inference-time_adaptive_tokenization_via_online_compression.md)
- [\[NeurIPS 2025\] When Worse is Better: Navigating the Compression-Generation Trade-off in Visual Tokenization](when_worse_is_better_navigating_the_compression-generation_tradeoff_in_visual_to.md)
- [\[NeurIPS 2025\] CodeGEMM: A Codebook-Centric Approach to Efficient GEMM in Quantized LLMs](codegemm_a_codebook-centric_approach_to_efficient_gemm_in_quantized_llms.md)
- [\[NeurIPS 2025\] Learning to Factorize and Adapt: A Versatile Approach Toward Universal Spatio-Temporal Foundation Models](learning_to_factorize_and_adapt_a_versatile_approach_toward_universal_spatio-tem.md)
- [\[ICCV 2025\] SAMO: A Lightweight Sharpness-Aware Approach for Multi-Task Optimization with Joint Global-Local Perturbation](../../ICCV2025/model_compression/samo_a_lightweight_sharpness-aware_approach_for_multi-task_optimization_with_joi.md)

<!-- RELATED:END -->
