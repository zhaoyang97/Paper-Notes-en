---
title: >-
  [Paper Note] RACER: Retrieval-Augmented Contextual Rapid Speculative Decoding
description: >-
  [ACL 2026][LLM Efficiency][Speculative Decoding] RACER proposes a training-free speculative decoding method that unifies retrieval-based exact pattern matching with logit-based future prediction. By constructing a Logits…
tags:
  - "ACL 2026"
  - "LLM Efficiency"
  - "Speculative Decoding"
  - "Retrieval-Augmented"
  - "Training-free"
  - "Aho-Corasick Automaton"
  - "Inference Acceleration"
date: 2026-05-08
content_hash: b6005f31fe575695
---

# RACER: Retrieval-Augmented Contextual Rapid Speculative Decoding

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.14885](https://arxiv.org/abs/2604.14885)  
**Code**: [https://github.com/hkr04/RACER](https://github.com/hkr04/RACER)  
**Area**: Information Retrieval  
**Keywords**: Speculative Decoding, Retrieval-Augmented, Training-free, Aho-Corasick Automaton, Inference Acceleration

## TL;DR

RACER proposes a training-free speculative decoding method that unifies retrieval-based exact pattern matching with logit-based future prediction. By constructing a Logits Tree via a copy-logit strategy and a Retrieval Tree via an LRU-evicted AC automaton, it achieves over $2\times$ inference acceleration across several benchmarks.

## Background & Motivation

**Background**: Autoregressive decoding in LLMs generates one token per step, where inference latency grows linearly with the sequence length. Speculative decoding, employing a "guess-and-verify" strategy to validate multiple tokens in parallel without sacrificing output quality, is one of the most promising acceleration schemes.

**Limitations of Prior Work**: Existing training-free methods suffer from two types of issues: (1) Retrieval-based methods (e.g., PLD, REST) rely on exact token matching and fail completely when no matching continuation exists in the context; (2) Logit-based methods (e.g., Token Recycling) lack structured guidance, resulting in narrow prediction ranges and suboptimal quality. These two categories are complementary yet currently fragmented.

**Key Challenge**: Retrieval provides "seen information" (precise but sparse), while logits provide "unseen information" (flexible but lacking anchors). Although reciprocal, existing methods fail to fuse them effectively.

**Goal**: Design a lightweight, plug-and-play, training-free speculative decoding method that unifies retrieval and logit signal sources.

**Key Insight**: The authors discover that the copy-logit strategy (reusing logits from the most recent occurrence of the same token in the context) yields a higher acceptance rate and sharper distribution (rank-1 exceeding 50%) than the last-logit strategy, providing a foundation for building efficient logit draft trees.

**Core Idea**: An AC automaton is used to maintain n-gram patterns in the context as structured retrieval anchors, while copy-logit is used to construct a logit draft tree with layer-wise pruning for flexible extrapolation. Both sources are dynamically allocated a budget under fixed capacity and merged into a unified draft tree via a trie.

## Method

### Overall Architecture

In each decoding step, RACER first identifies matching patterns in the current context via an AC automaton to select retrieval candidates from the highest-frequency continuations. Subsequently, the remaining capacity is allocated to the Logits Tree for breadth-first expansion based on copy-logit. Finally, both trees are merged into a unified draft tree via a trie and validated by the target model in a single pass using tree attention.

### Key Designs

1.  **Logits Tree (Logit-based Draft Tree)**:

    - **Function**: Utilizes the target model's own logit information to generate multi-level token candidates.
    - **Mechanism**: Employs the copy-logit strategy—for the current sampled next-token $x_t$, find the most recent position $i$ in the context where $x_i = x_t$, and reuse its subsequent logits $\mathbf{z}_{i+1}$ as an approximation for $\mathbf{z}_{t+1}$. Experiments show the MAT of copy-logit is 1.87 (vs. 1.57 for last-logit), with a rank-1 acceptance rate over 50%. Based on heavy-tail distribution characteristics, a decreasing breadth allocation is designed: $b_{child(i,j)} = \max(1, \lfloor b_i / 2^{j+[i\neq 0]} \rfloor)$, where the first layer is widest and deeper layers are progressively pruned.
    - **Design Motivation**: Copy-logit is based on the assumption that "the same tokens have similar semantic continuations in similar contexts," which is more accurate than simply reusing the previous step's logits. Decreasing breadth allocation aligns with the empirical law that acceptance rates decay rapidly with depth.

2.  **Retrieval Tree (AC Automaton with LRU Eviction)**:

    - **Function**: Efficiently retrieves repeated n-gram patterns from the generated context to provide structured draft candidates.
    - **Mechanism**: Uses an Aho-Corasick (AC) automaton to maintain n-grams (max length 10) appearing in the context. A maximum node capacity (10,000) is set, and the least recently used leaf nodes are removed via an LRU eviction strategy. During matching, all boundary nodes with depth $\geq 2$ are identified, and top-k continuations with the highest global frequency are selected from their subtrees as retrieval candidates. Failure links are lazily rebuilt at the end of the prefill stage.
    - **Design Motivation**: Suffix arrays and suffix automata grow linearly with context length and cannot evict outdated states. AC automaton failure links naturally enrich draft diversity, while LRU eviction ensures stable memory usage.

3.  **Unified Integration Strategy**:

    - **Function**: Dynamically balances retrieval and logit candidate sources under a fixed draft capacity.
    - **Mechanism**: Retrieval candidates are prioritized (structurally reliable but sparse), and the remaining capacity is given to the Logits Tree for breadth-first expansion. Both are merged into a unified draft tree via trie-based union and validated by the target model at once using tree attention.
    - **Design Motivation**: Retrieval signals capture near-distance repetition to provide sharper prediction guidance for logit distributions, reducing error accumulation during speculative expansion.

### Loss & Training

RACER is entirely training-free. Default hyperparameters: Logits Tree max breadth 8, Retrieval Tree max 10,000 nodes, n-gram length 10, draft capacity 64 per step. Greedy decoding is used with a batch size of 1.

## Key Experimental Results

### Main Results

| Model | Method | Spec-Bench Accel. | HumanEval Accel. | MGSM-ZH Accel. | Avg. Accel. |
|------|------|----------------|---------------|--------------|---------|
| Vicuna-7B | PLD | 1.50× | 1.40× | 2.27× | 1.87× |
| Vicuna-7B | LogitSpec | 1.77× | 1.66× | 2.67× | 2.03× |
| Vicuna-7B | Token Recycling | 2.06× | 2.17× | 2.30× | 2.18× |
| Vicuna-7B | **RACER** | **2.21×** | **2.29×** | **2.77×** | **2.42×** |
| Vicuna-33B | **RACER** | **2.20×** | **2.58×** | **2.77×** | **2.52×** |
| Qwen3-8B | EAGLE-3 | 2.14× | 2.44× | 0.86× | 1.81× |
| Qwen3-8B | **RACER** | **2.13×** | 2.24× | **2.26×** | **2.21×** |

### Ablation Study

| Config | Spec-Bench MAT | HumanEval MAT | Description |
|------|---------------|---------------|------|
| RACER (Full) | 3.00 | 3.11 | Full integration of retrieval + logits |
| Logits Tree Only | ~2.76 | ~2.83 | No retrieval guidance, similar to Token Recycling |
| Retrieval Tree Only | ~1.82 | ~2.06 | No logit extrapolation, similar to REST |

### Key Findings

- RACER is consistently optimal among all training-free methods, achieving average speedup ratios of 2.42×-2.52×.
- Compared to EAGLE-3 (which requires an additional draft model), RACER has a slightly lower MAT but matches or exceeds actual speedups due to zero extra model overhead.
- EAGLE-3 fails in Chinese reasoning (MGSM-ZH) (speedup <1×), exposing the sensitivity of model-based methods to training data distribution; RACER remains stable in acceleration.
- Copy-logit yields a 0.3 higher MAT than last-logit (1.87 vs. 1.57), with a rank-1 acceptance rate over 50%.
- The method is insensitive to hyperparameters and exhibits good robustness.

## Highlights & Insights

- The copy-logit strategy is a subtle observation—the logit distribution following the same token at different positions remains highly similar. This idea of "in-context logit reuse" is simple yet effective and applicable to any autoregressive model.
- Replacing suffix arrays with AC automata is clever: the failure links provide pattern generalization, and LRU eviction ensures fixed memory overhead. This choice of data structure is worth emulating in other scenarios requiring online pattern matching.
- Positioning "retrieval as structural guidance rather than an independent generator"—retrieval signals do not just generate candidates but provide anchors and direction for logit prediction. this fusion philosophy is more elegant than a simple combination. 

## Limitations & Future Work

- Evaluation was limited to batch size=1 and greedy decoding; large batch and sampling decoding scenarios require verification.
- AC automaton node capacity (10K) and n-gram length (10) are fixed; adaptive adjustment might further improve performance.
- The potential for combination with model-based methods has not been explored.
- Whether the advantages in non-English languages derive from the language-agnostic nature of retrieval deserves in-depth analysis.

## Related Work & Insights

- **vs Token Recycling**: TR only uses a top-k adjacency matrix to expand the draft tree but lacks retrieval guidance. RACER makes logit expansion more accurate via structural anchors provided by the AC automaton, accepting an average of 0.4 more tokens.
- **vs EAGLE-3**: EAGLE-3 requires an additional trained draft model, leading to higher MAT but not necessarily better wall-clock speedup. RACER's advantages of zero training and zero extra memory make it more suitable for plug-and-play deployment.

## Rating

- Novelty: ⭐⭐⭐⭐ The unified perspective of retrieval + logits is novel; copy-logit and LRU-AC automaton designs are elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers multiple model scales (7B-33B), various task categories, and multiple languages; ablation and analysis are exhaustive.
- Writing Quality: ⭐⭐⭐⭐ Clear method description with intuitive illustrations.
- Value: ⭐⭐⭐⭐ Provides a practical training-free inference acceleration scheme with high plug-and-play deployment value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Speculative Verification: Exploiting Information Gain to Refine Speculative Decoding](speculative_verification_exploiting_information_gain_to_refine_speculative_decod.md)
- [\[ACL 2026\] Multi-Drafter Speculative Decoding with Alignment Feedback](multi-drafter_speculative_decoding_with_alignment_feedback.md)
- [\[ACL 2026\] TokenTiming: A Dynamic Alignment Method for Universal Speculative Decoding Model Pairs](tokentiming_a_dynamic_alignment_method_for_universal_speculative_decoding_model_.md)
- [\[NeurIPS 2025\] 3-Model Speculative Decoding (PyramidSD)](../../NeurIPS2025/llm_efficiency/3model_speculative_decoding.md)
- [\[ICML 2026\] MineDraft: A Framework for Batch Parallel Speculative Decoding](../../ICML2026/llm_efficiency/minedraft_a_framework_for_batch_parallel_speculative_decoding.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ACL 2026\] Speculative Verification: Exploiting Information Gain to Refine Speculative Decoding](speculative_verification_exploiting_information_gain_to_refine_speculative_decod.md)
- [\[ACL 2026\] Multi-Drafter Speculative Decoding with Alignment Feedback](multi-drafter_speculative_decoding_with_alignment_feedback.md)
- [\[ACL 2026\] TokenTiming: A Dynamic Alignment Method for Universal Speculative Decoding Model Pairs](tokentiming_a_dynamic_alignment_method_for_universal_speculative_decoding_model_.md)
- [\[ACL 2025\] SAM Decoding: Speculative Decoding via Suffix Automaton](../../ACL2025/llm_efficiency/sam_decoding_speculative_decoding_via_suffix_automaton.md)
- [\[NeurIPS 2025\] 3-Model Speculative Decoding (PyramidSD)](../../NeurIPS2025/llm_efficiency/3model_speculative_decoding.md)

</div>

<!-- RELATED:END -->
