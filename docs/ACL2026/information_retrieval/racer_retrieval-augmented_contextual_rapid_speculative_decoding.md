---
title: >-
  [Paper Note] RACER: Retrieval-Augmented Contextual Rapid Speculative Decoding
description: >-
  [ACL 2026][Speculative Decoding] RACER proposes a training-free speculative decoding method that unifies retrieval-based exact pattern matching with logits-based future prediction. It constructs a Logits Tree via a copy-logit strategy and a Retrieval Tree via an LRU-eviction Aho-Corasick automaton, achieving over 2× inference speedup across multiple benchmarks.
tags:
  - ACL 2026
  - Speculative Decoding
  - Retrieval Augmentation
  - Training-Free
  - Aho-Corasick Automaton
  - Inference Acceleration
date: 2026-05-08
content_hash: b975fdc125c063cf
---

# RACER: Retrieval-Augmented Contextual Rapid Speculative Decoding

**Conference**: ACL 2026
**arXiv**: [2604.14885](https://arxiv.org/abs/2604.14885)
**Code**: [https://github.com/hkr04/RACER](https://github.com/hkr04/RACER)
**Area**: Information Retrieval
**Keywords**: Speculative Decoding, Retrieval Augmentation, Training-Free, Aho-Corasick Automaton, Inference Acceleration

## TL;DR

RACER proposes a training-free speculative decoding method that unifies retrieval-based exact pattern matching with logits-based future prediction. It constructs a Logits Tree via a copy-logit strategy and a Retrieval Tree via an LRU-eviction Aho-Corasick automaton, achieving over 2× inference speedup across multiple benchmarks.

## Background & Motivation

**State of the Field**: Autoregressive decoding in LLMs generates one token per step, causing inference latency to grow linearly with sequence length. Speculative decoding, which employs a "draft-then-verify" paradigm to validate multiple tokens in parallel without sacrificing output quality, is among the most promising acceleration approaches.

**Limitations of Prior Work**: Existing training-free methods suffer from two categories of issues: (1) retrieval-based methods (e.g., PLD, REST) rely on exact token matching and fail entirely when no matching continuation exists in context; (2) logits-based methods (e.g., Token Recycling) lack structured guidance, resulting in a narrow prediction range and suboptimal quality. Each category has its advantages, but they remain disjoint.

**Root Cause**: Retrieval provides "seen information" (precise but sparse), while logits provide "unseen information" (flexible but lacking anchors). The two are complementary, yet existing methods fail to integrate them effectively.

**Paper Goals**: Design a lightweight, plug-and-play, training-free speculative decoding method that unifies retrieval and logits as dual signal sources.

**Starting Point**: The authors observe that the copy-logit strategy—reusing the logits at the most recent occurrence of the same token in context—yields higher acceptance rates and sharper distributions than the last-logit strategy (rank-1 acceptance rate exceeding 50%), providing a foundation for building an efficient logits draft tree.

**Core Idea**: An Aho-Corasick automaton maintains n-gram patterns in context as structured retrieval anchors, while copy-logit constructs a layer-wise pruned logits draft tree for flexible extrapolation. Under a fixed capacity budget, the two trees are dynamically allocated and merged into a unified draft tree via trie union.

## Method

### Overall Architecture

At each decoding step, RACER first identifies matching patterns in the current context via the Aho-Corasick automaton and selects retrieval candidates from the most frequent continuations. The remaining capacity is allocated to the Logits Tree for breadth-first expansion based on copy-logit. Finally, both trees are merged into a unified draft tree via trie union and verified in a single forward pass by the target model using tree attention.

### Key Designs

1. **Logits Tree (copy-logit-based Draft Tree)**:

    - **Function**: Generates multi-level token candidates using the target model's own logits information.
    - **Mechanism**: Adopts the copy-logit strategy—for the currently sampled next token $x_t$, the method finds the most recent position $i$ in context where $x_i = x_t$, and reuses its subsequent logits $\mathbf{z}_{i+1}$ as an approximation of $\mathbf{z}_{t+1}$. Experiments show that copy-logit achieves a MAT of 1.87 (vs. 1.57 for last-logit), with rank-1 acceptance exceeding 50%. Based on the heavy-tailed distribution property, a decreasing breadth allocation is designed: $b_{child(i,j)} = \max(1, \lfloor b_i / 2^{j+[i\neq 0]} \rfloor)$, widest at the first layer and progressively pruned at deeper layers.
    - **Design Motivation**: The copy-logit strategy rests on the assumption that "identical tokens in similar contexts exhibit similar semantic continuations," providing more accurate approximations than naively reusing the previous step's logits. The decreasing breadth allocation reflects the empirical observation that acceptance rates decay rapidly with depth.

2. **Retrieval Tree (Aho-Corasick Automaton with LRU Eviction)**:

    - **Function**: Efficiently retrieves repeated n-gram patterns from the generation context, providing structured draft candidates.
    - **Mechanism**: An Aho-Corasick automaton maintains n-grams (maximum length 10) observed in context. A maximum node capacity (10,000) is set, with LRU eviction discarding the least recently used leaf nodes. At matching time, all boundary nodes at depth $\geq 2$ are identified, and top-k continuations with the highest global frequency are selected from their subtrees as retrieval candidates. Failure links are lazily rebuilt at the end of the prefill phase.
    - **Design Motivation**: Suffix arrays and suffix automata grow linearly with context length and cannot evict stale states. The failure links of the Aho-Corasick automaton naturally enrich draft diversity, while LRU eviction ensures stable memory usage.

3. **Unified Integration Strategy**:

    - **Function**: Dynamically balances retrieval and logits candidates under a fixed draft capacity.
    - **Mechanism**: Retrieval candidates are allocated first (structurally reliable but sparse), with remaining capacity assigned to breadth-first expansion of the Logits Tree. Both are merged into a unified draft tree via trie-based union and verified in a single forward pass by the target model under tree attention.
    - **Design Motivation**: Retrieval signals capture nearby repetition patterns and provide sharper predictive guidance for the logits distribution, reducing error accumulation during speculative expansion.

### Loss & Training

RACER requires no training whatsoever. Default hyperparameters: maximum breadth of 8 for the Logits Tree; maximum 10,000 nodes, n-gram length of 10 for the Retrieval Tree; draft capacity of 64 tokens per step. Greedy decoding is used with batch size 1.

## Key Experimental Results

### Main Results

| Model | Method | Spec-Bench Speedup | HumanEval Speedup | MGSM-ZH Speedup | Avg. Speedup |
|------|------|----------------|---------------|--------------|---------|
| Vicuna-7B | PLD | 1.50× | 1.40× | 2.27× | 1.87× |
| Vicuna-7B | LogitSpec | 1.77× | 1.66× | 2.67× | 2.03× |
| Vicuna-7B | Token Recycling | 2.06× | 2.17× | 2.30× | 2.18× |
| Vicuna-7B | **RACER** | **2.21×** | **2.29×** | **2.77×** | **2.42×** |
| Vicuna-33B | **RACER** | **2.20×** | **2.58×** | **2.77×** | **2.52×** |
| Qwen3-8B | EAGLE-3 | 2.14× | 2.44× | 0.86× | 1.81× |
| Qwen3-8B | **RACER** | **2.13×** | 2.24× | **2.26×** | **2.21×** |

### Ablation Study

| Configuration | Spec-Bench MAT | HumanEval MAT | Note |
|------|---------------|---------------|------|
| RACER (full) | 3.00 | 3.11 | Full retrieval + logits integration |
| Logits Tree only | ~2.76 | ~2.83 | No retrieval guidance, similar to Token Recycling |
| Retrieval Tree only | ~1.82 | ~2.06 | No logits extrapolation, similar to REST |

### Key Findings

- RACER consistently outperforms all training-free methods, achieving average speedups of 2.42×–2.52×.
- Compared to EAGLE-3 (which requires an additional draft model), RACER achieves slightly lower MAT but comparable or superior wall-clock speedup due to the absence of an extra model overhead.
- EAGLE-3 fails on Chinese reasoning (MGSM-ZH, speedup <1×), exposing the sensitivity of model-level methods to training data distribution; RACER maintains stable acceleration.
- The copy-logit strategy improves MAT by 0.3 over last-logit (1.87 vs. 1.57), with rank-1 acceptance exceeding 50%.
- The method is robust to hyperparameter choices.

## Highlights & Insights

- The copy-logit strategy embodies a precise observation: the logits distributions following the same token at different positions exhibit high similarity. This idea of "intra-context logits reuse" is simple yet effective and generalizes to any autoregressive model.
- The choice of Aho-Corasick automaton over suffix arrays is elegant: failure links inherently provide pattern generalization, while LRU eviction guarantees bounded memory overhead. This data structure design is worth borrowing in other scenarios requiring online pattern matching.
- The positioning of "retrieval as structural guidance rather than an independent generator" is philosophically more refined than naive combination—retrieval signals serve as anchors and directional cues for logits prediction rather than directly producing candidates.

## Limitations & Future Work

- Evaluation is limited to batch size 1 and greedy decoding; large-batch and sampling-based decoding scenarios remain to be validated.
- The node capacity (10K) and n-gram length (10) of the Aho-Corasick automaton are fixed; adaptive adjustment could further improve performance.
- The potential of combining RACER with model-based methods has not been explored.
- Whether the advantage on non-English languages stems from the language-agnostic nature of retrieval warrants deeper analysis.

## Related Work & Insights

- **vs. Token Recycling**: TR expands the draft tree using only a top-k adjacency matrix without retrieval guidance. RACER's structural anchors from the Aho-Corasick automaton enable more accurate logits expansion, accepting on average 0.4 more tokens per step.
- **vs. EAGLE-3**: EAGLE-3 requires training an additional draft model and achieves higher MAT, but does not necessarily yield superior wall-clock speedup. RACER's zero-training, zero-extra-memory advantages make it more suitable for plug-and-play deployment.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The unified perspective integrating retrieval and logits is novel; the copy-logit and LRU-AC automaton designs are elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers diverse model scales (7B–33B), task types, and languages, with thorough ablations and analyses.
- **Writing Quality**: ⭐⭐⭐⭐ Method description is clear and illustrations are intuitive.
- **Value**: ⭐⭐⭐⭐ Provides a practical training-free inference acceleration solution with high plug-and-play deployment value.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] ChAIRO: Contextual Hierarchical Analogical Induction and Reasoning Optimization for LLMs](chairo_contextual_hierarchical_analogical_induction_and_reasoning_optimization_f.md)
- [\[ACL 2026\] MASS-RAG: Multi-Agent Synthesis Retrieval-Augmented Generation](mass-rag_multi-agent_synthesis_retrieval-augmented_generation.md)
- [\[ACL 2026\] Feedback Adaptation for Retrieval-Augmented Generation](feedback_adaptation_for_retrieval-augmented_generation.md)
- [\[ACL 2026\] Stable-RAG: Mitigating Retrieval-Permutation-Induced Hallucinations in Retrieval-Augmented Generation](stable-rag_mitigating_retrieval-permutation-induced_hallucinations_in_retrieval-.md)
- [\[ICLR 2026\] Token-Guard: Towards Token-Level Hallucination Control via Self-Checking Decoding](../../ICLR2026/information_retrieval/token-guard_towards_token-level_hallucination_control_via_self-checking_decoding.md)

<!-- RELATED:END -->
