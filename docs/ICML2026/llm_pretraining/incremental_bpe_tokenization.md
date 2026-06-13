---
title: >-
  [Paper Note] Incremental BPE Tokenization
description: >-
  [ICML 2026][LLM Pretraining][BPE Tokenization] This paper proposes the first incremental BPE tokenization algorithm with a strict $\mathcal{O}(\log^2 t)$ worst-case complexity per byte. By utilizing an Aho–Corasick autom…
tags:
  - "ICML 2026"
  - "LLM Pretraining"
  - "BPE Tokenization"
  - "Incremental Algorithm"
  - "Aho–Corasick"
  - "Centroid Decomposition"
  - "Streaming Output"
date: 2026-05-08
content_hash: 3a0a50f3c0838367
---

# Incremental BPE Tokenization

**Conference**: ICML 2026  
**arXiv**: [2605.30813](https://arxiv.org/abs/2605.30813)  
**Code**: https://github.com/ModelTC/mtc-inc-bpe (Available)  
**Area**: NLP Understanding / LLM Efficiency (Tokenizers, Streaming Inference)  
**Keywords**: BPE Tokenization, Incremental Algorithm, Aho–Corasick, Centroid Decomposition, Streaming Output

## TL;DR
This paper proposes the first incremental BPE tokenization algorithm with a strict $\mathcal{O}(\log^2 t)$ worst-case complexity per byte. By utilizing an Aho–Corasick automaton to locate the search space and binary search on Centroid Decomposition to identify the "last token," it serves as a drop-in replacement achieving up to $\sim 3\times$ speedup over Hugging Face tokenizers and eliminating the $\mathcal{O}(n^2)$ degradation of tiktoken on pathological inputs.

## Background & Motivation

**Background**: BPE (Byte Pair Encoding) has become the de facto standard for modern LLMs, including the GPT series, Qwen-3, LLaMA, and DeepSeek. Two major implementations exist: Hugging Face `tokenizers` uses a heap to maintain a global priority queue for processing the entire input, while OpenAI `tiktoken` relies on regex to pre-split the input into small chunks and then runs BPE merging on each. Historically, both are **offline** algorithms that require the complete segment to produce normalized tokenization results.

**Limitations of Prior Work**: This offline nature has two direct consequences. First, the prefill stage must **serially** wait for tokenization before starting inference, preventing the pipelining of tokenization with model forward passes; this latency becomes significant in long-context scenarios. Second, tiktoken exhibits true $\mathcal{O}(n^2)$ behavior on certain pathological inputs (e.g., `'a' × 2^k` long repetitions). Additionally, upstream regex engines may experience stack overflow or crash on ultra-long strings, making the BPE stage a potential attack surface for algorithmic complexity.

**Key Challenge**: Current implementations treat BPE as a global process of "read then merge"—heap methods need all pairs to find the maximum priority, and regex pre-splitting must ensure sentence boundaries are not crossed. This global perspective is naturally at odds with streaming incrementality. Berglund & van der Merwe (2023) theoretically proved that BPE satisfies "prefix consistency" (the tokenization of any prefix is a prefix of the complete tokenization), implying the possibility of incremental processing, but they provided neither an **algorithmic construction** nor a **worst-case complexity bound**.

**Goal**: Construct an incremental algorithm **strictly equivalent to standard BPE** that maintains tokenization results for all prefixes of the current string in worst-case $\mathcal{O}(\log^2 t)$ time per byte ($t$ being the maximum token length). It should support eager output—immediately outputting a token once its boundary can no longer change by any future extension—thereby fully pipelining tokenization.

**Key Insight**: The authors reduce the problem to a core sub-problem: **Given a string $s$ and a new character $c$, find the last token $\theta(sc)$ of the new string**. Due to prefix consistency, knowing $\theta(\cdot)$ allows recursive backtracking to the complete tokenization. Since $\theta(sc)$ must be a suffix token of $sc$, all candidates form a "Suffix-Successor Tree." The authors prove that valid candidates form a **monotonic path** on this tree, compressing the search space from exponential to logarithmic.

**Core Idea**: An Aho–Corasick automaton is used to locate the longest suffix token in $\mathcal{O}(1)$ to frame the search tree. Centroid Decomposition is then used to perform binary search on the tree in $\mathcal{O}(\log t)$ to locate $\theta(sc)$. By using DFS timestamp intervals to verify the "prefix-end token condition" in $\mathcal{O}(1)$, the total complexity reflects $\mathcal{O}(\log^2 t)$ per byte.

## Method

### Overall Architecture

The algorithm reframes BPE tokenization of a string of length $n$ into $n$ incremental updates. Each byte $c$ updates the state from $\theta(s)$ to $\theta(sc)$, implicitly maintaining the tokenization tree of all prefixes.

The overall pipeline consists of: ① **Preprocessing**—Normalizing the vocabulary $\mathcal{V}$ (removing unreachable tokens, establishing a bijection between non-atomic tokens and their merge rules) and constructing the **Successor Forest** (where each non-atomic token points to its successor, i.e., the right part that formed it); ② **Offline Indexing**—Performing a pre-order DFS on the Successor Forest to obtain `dfs_in/dfs_out` timestamps and calculating the "valid interval" $I_t$ for each token; constructing a Centroid Search Tree (CST) for each token $\tau$; building an Aho–Corasick automaton and marking the "search space entry" on each state; ③ **Online Incremental**—Reading byte $c$, transitioning the automaton in $\mathcal{O}(1)$ to get the longest suffix token $\tau(sc)$, entering the corresponding CST on the SufSucTree, and performing logarithmic binary searches to locate the deepest valid node on the monotonic path as $\theta(sc)$.

The entire online stage takes $\mathcal{O}(log^2 t)$ in the worst case per byte, or $\mathcal{O}(n \log^2 t)$ total, significantly lower than the $\mathcal{O}(n^2)$ of tiktoken on pathological inputs.

### Key Designs

1. **Monotonic Path Property — Reducing Global Merging to Tree Search**:

    - **Function**: Characterizes which suffix tokens on the SufSucTree are eligible to be $\theta(sc)$, compressing the "candidate set" from all suffix tokens (linear) to a **unique monotonic path** from the root (atomic token $c$) to $\theta(sc)$.
    - **Mechanism**: The authors formalize the "prefix-end token condition" (Definition 4.1): A candidate $t$ must satisfy two conditions, where $s^{-\operatorname{suc}(t)}$ is the prefix after removing the suffix part of $t$: (i) **Reachability**—$\theta(s^{-\operatorname{suc}(t)})$ must fall within the subtree rooted at $\operatorname{pre}(t)$ in the Successor Forest; (ii) **Priority Dominance**—If $\theta(s^{-\operatorname{suc}(t)}) \neq \operatorname{pre}(t)$, the child $u$ of $\operatorname{pre}(t)$ on the path to that ancestor must have a merge rule priority strictly lower than $t$'s own rule. Theorem 4.2 proves that all $t$ satisfying these conditions on $\operatorname{SufSucTree}(\tau(sc))$ form exactly one monotonic path starting from the root, and $\theta(sc)$ is the deepest node.
    - **Design Motivation**: Existing implementations treat BPE as a global priority queue merge, scanning all pairs at each step, which causes $\mathcal{O}(n \log n)$ or worse. The monotonic path property transforms the problem of determining the last token—which seemingly requires backtracking through history—into a binary search on a statically constructed tree, naturally suited for incremental updates and worst-case analysis.

2. **DFS Timestamps + Valid Intervals — Reducing the Prefix-End Token Condition to $\mathcal{O}(1)$ Interval Detection**:

    - **Function**: Converts the complex decision of checking Successor Forest subtree membership and comparing rule priorities into a simple $\mathcal{O}(1)$ check of whether an "integer falls within an interval."
    - **Mechanism**: A pre-order DFS is performed on the Successor Forest. The key trick is **sorting children by canonical rule priority from lowest to highest** during traversal—ensuring subtrees of higher-priority children have later timestamps. This guarantees that for any non-atomic token $t$, the valid candidate set $\mathcal{C}_t$ corresponds to a continuous timestamp interval $I_t = [L_t, R_t)$, where $L_t = \operatorname{dfs\_in}(\operatorname{pre}(t))$ and $R_t$ is the `dfs_in` of the first child of $\operatorname{pre}(t)$ with rule priority $\geq t$ (or $\operatorname{dfs\_out}(\operatorname{pre}(t))$ if none exist). Online detection becomes a single comparison: `dfs_in(k) ∈ I_t`. Another key corollary: **Valid intervals of sibling nodes in the SufSucTree do not intersect**, ensuring unambiguous direction selection during search.
    - **Design Motivation**: Recalculating the "prefix-end token condition" would introduce an $\mathcal{O}(t)$ factor, preventing the $\mathcal{O}(\log^2 t)$ goal. Linearizing tree relationships into intervals is a classic competitive programming technique, but combining it with "priority-sorted children" encodes both reachability and priority dominance, turning abstract theoretical properties into computable $\mathcal{O}(1)$ predicates.

3. **Aho–Corasick + Centroid Decomposition — Achieving Logarithmic Complexity for Incremental Updates**:

    - **Function**: Locates the search space $\operatorname{SufSucTree}(\tau(sc))$ in $\mathcal{O}(1)$ for each byte and identifies the deepest valid node within it in $\mathcal{O}(\log^2 t)$.
    - **Mechanism**: ① An Aho–Corasick automaton is built for vocabulary $\mathcal{V}$, with precomputed "search space entry" annotations for each state. If the state's token is in $\mathcal{V}$, it is used; otherwise, it inherits from the suffix link. This allows $\mathcal{O}(1)$ access to $\tau(sc)$ when a new character arrives **without traversing suffix links**. The transition table uses square-root tiling for compression while maintaining $\mathcal{O}(1)$ query time. ② A Centroid Search Tree (CST) is built offline for each $\tau \in \mathcal{V}$, with a height of exactly $\mathcal{O}(\log |\tau|)$. Online, the search starts from the CST root. For a centroid $u$, interval detection checks its validity: if **invalid**, the target must be in the parent-side component; if **valid**, $u$ is on the path, and a binary search is performed on $u$'s children in the original SufSucTree (sorted sibling intervals) to find deeper valid children. Each CST step involves an $\mathcal{O}(\log t)$ binary search across a CST depth of $\mathcal{O}(\log t)$, totaling $\mathcal{O}(\log^2 t)$ per byte.
    - **Design Motivation**: Naive top-down traversal of the SufSucTree is $\mathcal{O}(t)$ in the worst case (tree height linear to $|\tau|$). Centroid Decomposition is a standard tool for compressing any tree into an $\mathcal{O}(\log)$ height search structure. Combined with Aho–Corasick for $\mathcal{O}(1)$ search space identification, these components allow for both strict worst-case complexity and incremental updates.

### Loss & Training

This is a pure algorithm/data structure work with no training involved. The Eager Output module (§6) maintains an "Active Frontier" $\mathcal{P}$—a set of candidate last tokens that could become parents of future tokens, bounded by the Aho–Corasick state depth $d(s)$ in the window $[|s|-d(s), |s|]$. Using a monotonic two-pointer approach, tokens are output eagerly when all active paths converge to the same child of the virtual root, enabling full pipelining with model inference. Eager mode introduces approximately 10% throughput overhead compared to non-eager mode.

## Key Experimental Results

### Main Results

As a drop-in replacement for Hugging Face `tokenizers` and OpenAI `tiktoken`, end-to-end throughput speedups were measured across English, Chinese, and Code corpora:

| Backend | Model | English | Chinese | Code |
|------|------|---------|---------|------|
| tokenizers | CodeLlama | **3.13×** | 1.10× | **2.88×** |
| tokenizers | Qwen-3 | 1.05× | 1.04× | 1.08× |
| tokenizers | DeepSeek-3.2 | 1.01× | 0.93× | 1.03× |
| tokenizers | Llama-3.1* | 0.99× | 1.03× | 1.02× |
| tokenizers | GPT-OSS | 1.00× | 1.08× | 1.01× |
| tiktoken | CL100K | 0.96× | **1.59×** | 1.04× |
| tiktoken | O200K | 0.99× | 1.46× | 1.00× |
| tiktoken | P50K | 0.97× | 1.35× | 1.07× |

\* properized dictionary.

### Pathological Input Robustness

Input constructed as `'a' × 2^k`, throughput compared on a logarithmic scale:

| Implementation | Complexity Behavior | Remarks |
|------|-----------|------|
| Ours (Incremental BPE) | Stable $\mathcal{O}(n \log^2 t)$ | Throughput stays mostly flat |
| tiktoken | $\mathcal{O}(n^2)$ significant decay | Throughput drops as input grows |
| O200K model | Regex stage crashes first | Ultra-long inputs trigger upstream errors |
| Ours (Eager Output) | ~10% slower than non-eager | Overhead from active frontier maintenance |

### Key Findings

- **Largest acceleration for CodeLlama (up to 3.13×)**: CodeLlama does not use regex pre-tokenization; BPE processes the entire normalized text. The original `tokenizers` heap method suffers from $\mathcal{O}(n \log n)$ on full input, whereas the advantages of the incremental algorithm are fully realized. Conversely, when pre-tokenization splits input into fine pieces (e.g., English with Qwen-3), implementation constants dominate, leading to negligible or slightly negative gains (0.99×–1.05×). This confirms that pre-tokenization is essentially an engineering patch for BPE complexity that becomes redundant with a guaranteed algorithm.
- **Significant gains for Chinese + tiktoken (1.35×–1.59×)**: Regex splitting for Chinese is naturally coarser, exposing tiktoken's internal BPE bottleneck on large blocks. The strict per-byte bound of this work directly addresses this discrepancy.
- **Pathological inputs are the "killer app"**: While tiktoken's throughput curve shows clear $\mathcal{O}(n^2)$ decay on `'a' × 2^k`, this work remains flat. This is not just a performance issue but a security one, as the authors explicitly list mitigating algorithmic complexity attacks (DoS) as a byproduct.

## Highlights & Insights

- **Precise Translation of Theoretical Properties → Data Structures**: The "prefix consistency" from Berglund & van der Merwe 2023 was an abstract algebraic property. This paper translates it layer-by-layer: Last Token Recursion → SufSucTree → Monotonic Path → DFS Timestamps → Non-intersecting Sibling Intervals → Centroid Decomposition. Each step is a necessary complexity optimization; without any one layer, the $\mathcal{O}(\log^2 t)$ bound would be unreachable. This "theory as a lever" paradigm is highly exemplary.
- **Engineering Beauty of a Drop-in Replacement**: The authors deliberately restricted changes to the BPE stage, leaving segmentation, normalization, and caching untouched. Consequently, benchmarks cleanly isolate the algorithm's contribution, and existing LLM pipelines can switch with zero friction—the ideal way to turn academic algorithms into industry-ready components.
- **Pathological Inputs as Complexity Attack Surface**: Reframing an $\mathcal{O}(n^2)$ tokenizer as an "algorithmic complexity DoS vulnerability" is a visionary perspective. LLM serving systems are exposed to the internet; an attacker could paralyze the tokenization stage with long repeating characters. Linking "strict worst-case complexity" to "security" extends the impact from efficiency to system reliability.

## Limitations & Future Work

- **Regex Preprocessing Bottleneck Remains**: Profiling in Appendix I shows that normalization and regex pre-tokenization have become pipeline bottlenecks. While BPE is now incremental, the upstream stages are still offline; end-to-end streaming tokenization requires these components to be incrementalized as well.
- **Acceleration Concentrated in Specific Scenarios**: In English scenarios with fine-grained pre-tokenization, this method barely breaks even (0.99×), indicating its constant factors are disadvantageous for already shattered inputs. If LLMs continue to rely on regex splitting, the value of this work may be diluted; if the trend follows CodeLlama in removing pre-tokenization, its value will increase.
- **10% Overhead for Eager Output**: A 10% cost is significant for a highly optimized tokenizer. The authors suggest amortizing this via pipeline parallelism, but the actual benefit depends on the downstream inference speed.
- **No Support for SentencePiece Semantics**: The method is formalized under standard BPE merge semantics. Certain non-properizable vocabularies in SentencePiece-style (e.g., Gemma-3, see Appendix A) are not applicable, leaving a gap for future work to cover.

## Related Work & Insights

- **vs Hugging Face `tokenizers` (Heap + Global Priority Queue)**: They use a global heap to pick the highest priority pair, which is offline and log-linear on full input. This work uses incremental updates with logarithmic tree search, achieving strict $\mathcal{O}(\log^2 t)$ per byte and a 3× speedup for CodeLlama.
- **vs OpenAI `tiktoken` (Regex pre-split + In-segment BPE)**: They use regex to limit chunk size to bypass global BPE complexity, but the regex engine can crash on pathological inputs, and in-segment BPE is still $\mathcal{O}(n^2)$ in the worst case. This work provides a strict upper bound at the algorithmic level, removing dependence on regex as an engineering patch.
- **vs rust-gems `bpe` crate (van Antwerpen & Neubeck 2024)**: They also use Aho–Corasick for incremental BPE and are engineering pioneers, but they lack a formal proof of worst-case complexity. This work provides both the theoretical algorithm and rigorous implementation details (see Appendix J for comparison).
- **vs Berglund & van der Merwe 2023**: They provided the formal semantics and prefix consistency of BPE but stopped at algebraic properties, failing to solve the algorithmic problem of how to bound the lookahead required for incremental updates. This work turns those properties into a computable algorithm.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First incremental BPE algorithm with strict $\mathcal{O}(\log^2 t)$ worst-case complexity; introduces new theoretical results like the Monotonic Path Property.
- Experimental Thoroughness: ⭐⭐⭐⭐ Benchmarked 8 tokenizers across 3 corpora with pathological stress tests; however, lacks a direct end-to-end comparison with the rust-gems bpe crate.
- Writing Quality: ⭐⭐⭐⭐⭐ Progressively builds from theory to data structures and complexity analysis; motivations and reductions are crystal clear; includes complete proofs in the appendix.
- Value: ⭐⭐⭐⭐⭐ A drop-in replacement ready for modern LLM serving systems; the connection between complexity guarantees and security elevates the work beyond simple speedup.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Differentiable Hierarchical Visual Tokenization](../../NeurIPS2025/llm_pretraining/differentiable_hierarchical_visual_tokenization.md)
- [\[ICCV 2025\] FlowMo: Flow to the Mode — Mode-Seeking Diffusion Autoencoders for State-of-the-Art Image Tokenization](../../ICCV2025/llm_pretraining/flow_to_the_mode_mode-seeking_diffusion_autoencoders_for_state-of-the-art_image_.md)
- [\[NeurIPS 2025\] Broken Tokens: Your Language Model Can Secretly Handle Non-Canonical Tokenization](../../NeurIPS2025/llm_pretraining/broken_tokens_your_language_model_can_secretly_handle_non-canonical_tokenization.md)
- [\[ICML 2026\] On Training Large Language Models for Long-Horizon Tasks: An Empirical Study of Horizon Length](on_training_large_language_models_for_long-horizon_tasks_an_empirical_study_of_h.md)
- [\[ICML 2026\] Predicting Large Model Test Losses with a Noisy Quadratic System](predicting_large_model_test_losses_with_a_noisy_quadratic_system.md)

</div>

<!-- RELATED:END -->
