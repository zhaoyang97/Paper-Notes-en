---
title: >-
  [Paper Note] Flow of Spans: Generalizing Language Models to Dynamic Span-Vocabulary via GFlowNets
description: >-
  [ICLR 2026][Information Retrieval & RAG][GFlowNets] This paper proposes FoSS, the first framework to incorporate GFlowNets into span-level language modeling. By constructing a DAG-structured state space in place of the c…
tags:
  - "ICLR 2026"
  - "Information Retrieval & RAG"
  - "GFlowNets"
  - "dynamic vocabulary"
  - "span generation"
  - "DAG state space"
  - "text generation"
date: 2026-05-08
content_hash: 11cfc66fcd9d9434
---

# Flow of Spans: Generalizing Language Models to Dynamic Span-Vocabulary via GFlowNets

**Conference**: ICLR 2026
**arXiv**: [2602.10583](https://arxiv.org/abs/2602.10583)  
**Code**: [GitHub](https://github.com/sappho-x/Flow-of-Spans)  
**Area**: Information Retrieval
**Keywords**: GFlowNets, dynamic vocabulary, span generation, DAG state space, text generation

## TL;DR

This paper proposes FoSS, the first framework to incorporate GFlowNets into span-level language modeling. By constructing a DAG-structured state space in place of the conventional token-by-token tree structure, FoSS enables more flexible and diverse text generation, achieving up to a 12.5% improvement in MAUVE score.

## Background & Motivation

Standard autoregressive language models generate text token by token using a fixed, finite vocabulary. This paradigm has two fundamental limitations: (1) a fixed vocabulary constrains the granularity of generation; and (2) token-level generation induces a tree-structured state space—where each state has a unique predecessor—restricting the model's ability to explore alternative generation paths.

Recent work has introduced dynamic vocabularies and variable-length generation units (e.g., CoG, kNN-LM), allowing models to sample retrieved text spans. However, these approaches overlook a critical observation: the same sentence can be composed from spans of different lengths (e.g., "ABCDEFGH" can be segmented as "AB-CD-EFGH" or "AB-C-DE-FG-H"), which inherently forms a directed acyclic graph (DAG) rather than a tree. Existing methods do not explicitly model this DAG structure, limiting exploration of compositional paths.

GFlowNets are powerful generative models well suited to efficient exploration and generalization over DAG-structured state spaces. However, prior work applying GFlowNets to language modeling has remained at the token level within tree-structured spaces, failing to fully exploit their potential.

## Method

### Overall Architecture

FoSS formulates text generation as a GFlowNet sampling process over a DAG-structured state space. The model selects variable-length text spans from a dynamic vocabulary and concatenates them sequentially. Since the same text can be reached via different span combinations, the resulting state space naturally forms a DAG.

FoSS defines a complete MDP framework:

- **State**: the concatenation of prefix $c$ and the text generated so far $t_i$
- **Action**: selecting a span (a single word or multi-word phrase) from the dynamic vocabulary
- **Transition**: deterministically appending the selected span to the current state
- **Reward**: a dedicated reward model evaluates the text quality of terminal states

Unlike token-level generation, span-level actions allow the same state to be reached via multiple distinct trajectories. For example, the state "ABCD" can be reached via "AB→ABCD" or "AB→ABC→ABCD". This DAG structure enables GFlowNets to comprehensively explore multiple compositional paths.

### Key Designs

**1. DAG-Inducing Span Segmentation Algorithm**

To construct DAG-structured training trajectories from training data, FoSS proposes a probabilistic span segmentation algorithm. Building on standard forward maximum matching, the method introduces a controlled stochastic early-stopping mechanism: for each candidate span, the algorithm decides with probability $p_r$ whether to extract it as a phrase. By applying different threshold sets $P = \{p_0=0, p_1, \ldots, p_n\}$, the same document yields multiple distinct segmentation trajectories that share common subsequences, naturally inducing a DAG structure.

**2. Policy Network Architecture**

The policy network comprises two components:

- **Prefix Encoder**: a Transformer (initialized from GPT-2) that encodes the current state via causal attention, producing a context vector $h_i$
- **Span Encoder**: a bidirectional Transformer (initialized from BERT) that encodes candidate phrases. For a phrase spanning positions $s$ to $e$ in a retrieved document, start and end position embeddings are transformed via an MLP and concatenated to yield the full phrase representation $v_a$

The forward policy distribution is $P_{\text{SLM}}(a \mid s_i; \theta) \propto \exp(h_i^\top v_a)$, computing a matching score between the context and candidate span via inner product.

The dynamic vocabulary consists of three components: (1) the token-level fixed vocabulary $V$; (2) phrases retrieved from an external corpus $T$ (substrings of length 2–8 tokens); and (3) a termination action.

**3. Backward Policy and Mixed Training**

The backward policy adopts a uniform distribution, assigning equal probability to all possible predecessors of the current state in the dynamic vocabulary. This contrasts with the tree-structured setting, where the backward policy degenerates to a deterministic form ($P_B = 1$).

Training constructs mini-batches from three sources: trajectories sampled online from the forward policy, high-reward trajectories drawn from a reward-prioritized replay buffer, and trajectories constructed from the training set. During the first epoch, only offline data is used; thereafter, offline and online trajectories are mixed with probabilities 0.2 and 0.8, respectively.

### Loss & Training

The **Sub-Trajectory Balance (SubTB)** objective is adopted. For a complete trajectory, the loss sums over all valid sub-trajectory pairs, with an indicator function ensuring that learning signals are computed only at complete sentence boundaries, concentrating supervision on meaningful sentence transitions.

The **reward function** combines two complementary components:

- **Language Model (LM)**: measures fluency as the likelihood of the sequence given the prefix
- **Preference Model (PM)**: a discriminator trained with a Bradley-Terry objective to distinguish human-written text from model-generated text, regularized with score-centering

The final reward is a weighted geometric mean of the two components, with $\alpha$ controlling the trade-off between fluency and preference alignment. Both the LM and PM are fully fine-tuned from GPT-2.

## Key Experimental Results

### Main Results

**Table 1: MAUVE Scores on Open-Domain Text Generation**

| Method | In-Domain Greedy | In-Domain Nucleus | OOD Nucleus | Scaling Nucleus |
|------|:---:|:---:|:---:|:---:|
| Transformer w/ FT | 19.87 | 23.43 | 26.85 | 21.31 |
| kNN-LM | 19.92 | 22.50 | 24.75 | 23.39 |
| CoG | 26.01 | 26.14 | 28.14 | 26.97 |
| GFlowNets-FT | 26.58 | 29.61 | 28.62 | - |
| **FoSS** | **30.78** | **31.65** | **32.17** | **33.79** |

FoSS achieves comprehensive superiority. Under In-Domain Nucleus sampling, it outperforms CoG by 5.51% and GFlowNets-FT by 2.04%.

**Table 2: GPT-4 Pairwise Evaluation (FoSS vs. Baselines)**

| Baseline | FoSS Wins | Tie | FoSS Loses |
|----------|:---:|:---:|:---:|
| Transformer | 53% | 19% | 28% |
| kNN-LM | 67% | 15% | 18% |
| CoG | 42% | 31% | 27% |
| GFlowNets-FT | 55% | 29% | 16% |

**Table 3: Accuracy on Knowledge-Intensive Tasks**

| Method | TruthfulQA | OpenBookQA | ARC-Challenge |
|------|:---:|:---:|:---:|
| Transformer w/ FT | 28.76 | 22.71 | 24.00 |
| CoG | 29.38 | 24.29 | 24.34 |
| **FoSS** | **30.45** | **26.20** | **24.63** |

### Ablation Study

| Variant | In-Domain MAUVE | Diversity | OOD MAUVE |
|------|:---:|:---:|:---:|
| FoSS w/o DAG (tree-degenerated) | 29.61 | 65.72 | 28.62 |
| FoSS w/o PM | 28.25 | 89.91 | 30.49 |
| FoSS w/o LM | 29.09 | 92.77 | 29.79 |
| **FoSS (full)** | **31.65** | 92.48 | **32.17** |

### Key Findings

1. **DAG structure is critical**: Removing spans and reducing the state space to a tree causes a significant drop in MAUVE (31.65 → 29.61) and a dramatic collapse in Diversity (92.48 → 65.72).
2. **LM and PM rewards are complementary**: Removing either component alone degrades MAUVE; using PM alone yields the highest Diversity but a notably lower MAUVE score.
3. **Strong scalability**: Consistent improvements are observed across model size (GPT-2 Base to XL), training data volume, and retrieval corpus size.
4. **Domain adaptation capability**: Updating only the retrieval corpus suffices to surpass in-domain fine-tuned Transformers; FoSS exceeds a fully trained CoG using only 0.47% of the training data.

## Highlights & Insights

- FoSS is the first to combine span-level generation with GFlowNets, constructing a genuine DAG-structured state space.
- The probabilistic span segmentation algorithm generates multiple overlapping segmentation paths for the same text via stochastic early stopping.
- Inference efficiency is comparable to standard Transformers, as span-level generation reduces the number of decoding steps.
- FoSS surpasses fully trained baselines with as little as 0.47% of the training data.

## Limitations & Future Work

1. Experiments are primarily conducted at the GPT-2 scale; performance on larger-scale LLMs remains unclear.
2. The method relies on an external retrieval corpus, requiring pre-encoding of source text collections, which increases deployment complexity.
3. Training involves multiple components (policy network, reward model, retriever), making the pipeline relatively intricate.
4. Evaluation is limited to English-language tasks; cross-lingual generalization has not been investigated.

## Related Work & Insights

This paper combines the DAG exploration capability of GFlowNets with dynamic-vocabulary language modeling, representing a novel interdisciplinary direction. It offers valuable reference for research on LLM generation diversity and retrieval-augmented generation. The span-level generation paradigm may also inspire draft strategy design in speculative decoding.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (first to construct DAG + GFlowNets at the span-level LM)
- Technical Depth: ⭐⭐⭐⭐⭐ (complete MDP formulation + SubTB training + dual-component reward)
- Experimental Thoroughness: ⭐⭐⭐⭐ (multi-dimensional evaluation + GPT-4 judgment + ablation + scalability)
- Practicality: ⭐⭐⭐ (reliance on retrieval corpus raises deployment overhead)
- Writing Quality: ⭐⭐⭐⭐ (clear structure, intuitive illustrations)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] TokMem: One-Token Procedural Memory for Large Language Models](tokmem_one-token_procedural_memory_for_large_language_models.md)
- [\[ICLR 2026\] Query-Level Uncertainty in Large Language Models](query-level_uncertainty_in_large_language_models.md)
- [\[ICLR 2026\] FutureMind: Equipping Small Language Models with Strategic Thinking-Pattern Priors via Adaptive Knowledge Distillation](futuremind_equipping_small_language_models_with_strategic_thinking-pattern_prior.md)
- [\[ICLR 2026\] Hierarchical Concept-based Interpretable Models](hierarchical_concept-based_interpretable_models.md)
- [\[AAAI 2026\] Do Retrieval Augmented Language Models Know When They Don't Know?](../../AAAI2026/information_retrieval/do_retrieval_augmented_language_models_know_when_they_dont_know.md)

</div>

<!-- RELATED:END -->
