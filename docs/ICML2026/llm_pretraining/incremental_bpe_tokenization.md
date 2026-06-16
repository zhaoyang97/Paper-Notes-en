---
title: >-
  [Paper Note] Incremental BPE Tokenization
description: >-
  [ICML 2026][Pretraining][Aho–Corasick] This paper proposes the first incremental BPE tokenization algorithm with a strict $\mathcal{O}(\log^2 t)$ worst-case per-byte complexity. By utilizing an Aho–Corasick automaton to locate the search space and binary search on a Centroid Decomposition of the "Suffix-Successor Tree" to identify the "last token," it serve
tags:
  - ICML 2026
  - Pretraining
  - Aho–Corasick
  - Centroid Decomposition
date: 2026-05-08
content_hash: 6f9107c82c3ecd7a
---
# Incremental BPE Tokenization

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2605.30813](https://arxiv.org/abs/2605.30813)  
**Code**: https://github.com/ModelTC/mtc-inc-bpe (Available)  
**Area**: NLP Understanding / LLM Efficiency (Tokenizers, Streaming Inference)  
**Keywords**: BPE Tokenization, Incremental Algorithm, Aho–Corasick, Centroid Decomposition, Streaming Output

## TL;DR
This paper proposes the first incremental BPE tokenization algorithm with a strict $\mathcal{O}(\log^2 t)$ worst-case per-byte complexity. By utilizing an Aho–Corasick automaton to locate the search space and binary search on a Centroid Decomposition of the "Suffix-Successor Tree" to identify the "last token," it serves as a drop-in replacement achieving up to $\sim 3\times$ speedup over Hugging Face tokenizers. Furthermore, it eliminates the $\mathcal{O}(n^2)$ degradation of tiktoken on pathological inputs.

## Background & Motivation

**Background**: Byte Pair Encoding (BPE) has become the de facto standard for modern LLM tokenization, utilized by the GPT series, Qwen-3, LLaMA, and DeepSeek. The two mainstream implementations are Hugging Face `tokenizers`, which uses a heap to maintain a global priority queue for processing the entire input, and OpenAI `tiktoken`, which relies on regex to pre-split inputs into small segments before running BPE merges on each. Both are essentially **offline** algorithms that require the complete segment to produce normalized tokenization results.

**Limitations of Prior Work**: This offline nature has two direct consequences. First, the prefill stage must **sequentially** wait for tokenization to complete before starting inference, preventing the pipelining of tokenization with model forward passes; this latency becomes significant in long-context scenarios. Second, `tiktoken` exhibits true $\mathcal{O}(n^2)$ behavior on certain pathological inputs (e.g., long repetitions like `'a' × 2^k`); even the upstream regex engine itself may suffer from stack overflows or crashes on extremely long strings, making the BPE stage a potential attack surface for algorithmic complexity exploits.

**Key Challenge**: Existing implementations treat BPE as a global "read-then-merge" process—the heap method needs all pairs to find the maximum priority, and regex pre-splitting must ensure sentence boundaries are not penetrated. This global perspective is naturally at odds with streaming incremental processing. While Berglund & van der Merwe (2023) theoretically proved that BPE satisfies "prefix consistency" (where any prefix tokenization is stably a prefix of the complete tokenization), they did not provide an **algorithmic construction** or **worst-case complexity bounds**.

**Goal**: To construct an incremental algorithm **strictly equivalent to standard BPE** that maintains the tokenization results of all current string prefixes in $\mathcal{O}(\log^2 t)$ worst-case time per byte (where $t$ is the maximum token length). The algorithm also supports eager output—emitting tokens as soon as their boundaries are guaranteed not to change regardless of future extensions, thereby fully pipelining tokenization.

**Key Insight**: The authors reduce the problem to a core sub-problem: **Given string $s$ and new character $c$, find the last token $\theta(sc)$ of the new string $sc$**. Based on prefix consistency, knowing $\theta(\cdot)$ allows for recursive backtracking to obtain the full tokenization. Since $\theta(sc)$ must be a suffix token of $sc$, all candidates constitute a "Suffix-Successor Tree." The authors prove that valid candidates form a **monotonic path** on this tree, compressing the search space from exponential to logarithmic.

**Core Idea**: An Aho–Corasick automaton is used to locate the longest suffix token in $\mathcal{O}(1)$ to frame the search tree. Then, binary search on a Centroid Decomposition (CST) of the tree is performed in $\mathcal{O}(\log t)$ to locate $\theta(sc)$. Each step uses DFS timestamp interval checks to reduce the "prefix-final token condition" to $\mathcal{O}(1)$, resulting in an overall complexity of $\mathcal{O}(\log^2 t)$ per byte.

## Method

### Overall Architecture

The paper transforms BPE tokenization for a string of length $n$ from an offline process into an online incremental process that updates with every byte read. The key observation is that due to prefix consistency, if the **last token** $\theta(sc)$ of the new string $sc$ can be calculated at each step, the entire prefix tokenization can be backtracked. Thus, the algorithm refines the process into $n$ updates of "reading byte $c$ and updating state $\theta(s)$ to $\theta(sc)$," where each update is strictly $\mathcal{O}(\log^2 t)$. This fundamentally avoids the $\mathcal{O}(n^2)$ issues found in `tiktoken`.

The pipeline consists of three stages. **Offline Preprocessing** first normalizes the vocabulary $\mathcal{V}$ (removing unreachable tokens and establishing a bijection between non-atomic tokens and merge rules) and constructs the **Successor Forest**, where each non-atomic token points to its successor (the right part of its merge). **Offline Indexing** performs a preorder DFS on the Successor Forest to obtain `dfs_in/dfs_out` timestamps and valid intervals $I_t$ for each token. It also builds a Centroid Search Tree (CST) for each token $\tau$ and an Aho–Corasick automaton where each state is labeled with a "search space entry." During the **Online Incremental** stage, for each byte $c$ read, the automaton transitions in $\mathcal{O}(1)$ to find the longest suffix token $\tau(sc)$. The algorithm then enters the corresponding CST to perform logarithmic binary search, locating the deepest valid node on the monotonic path, which is $\theta(sc)$.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    V["BPE Vocab 𝒱 + Merge Rules"]
    subgraph PRE["Offline Preprocessing"]
        direction TB
        N["Vocab Normalization<br/>Remove unreachable tokens, build token↔rule bijection"] --> SF["Construct Successor Forest<br/>Non-atomic tokens point to successors"]
    end
    subgraph IDX["Offline Indexing"]
        direction TB
        DFS["Preorder DFS (Children sorted by rule priority)<br/>DFS timestamps + Valid Interval I_t (Design 2)"]
        CST["Construct Centroid Search Tree per token (Design 3)"]
        AC["Construct Aho–Corasick Automata<br/>Annotate search entry per state (Design 3)"]
    end
    subgraph ON["Online Incremental (per byte c)"]
        direction TB
        S1["AC Automaton Step O(1)<br/>Get longest suffix token τ(sc) (Design 3)"]
        S2["Binary Search on CST of SufSucTree<br/>Find deepest valid node along monotonic path (Design 1)<br/>O(1) Valdity via DFS interval check (Design 2)"]
        S3["Locate θ(sc): last token of new string"]
        S1 --> S2 --> S3
    end
    V --> PRE
    PRE --> IDX
    IDX --> ON
    S3 -->|Backtrack θ(·)| OUT["Output current prefix tokenization<br/>Stream out when boundaries are fixed in eager mode"]
```

### Key Designs

**1. Monotonic Path Property: Transforming "Who is the last token" into Monotonic Tree Search**

Existing implementations treat BPE as a global merge with a priority queue, scanning all pairs at each step to find the highest priority, which leads to $\mathcal{O}(n \log n)$ or worse. Identifying which suffix token is $\theta(sc)$ seemingly requires looking back at the entire history. This theorem compresses the candidate set from "all suffix tokens" (linear) to a **unique monotonic path** on the SufSucTree from the root (atomic token $c$) to $\theta(sc)$. The authors formalize the "prefix-final token condition" (Definition 4.1): given $s^{-\operatorname{suc}(t)}$ as the prefix after removing the successor suffix of $t$, candidate $t$ must satisfy (i) **Reachability**: $\theta(s^{-\operatorname{suc}(t)})$ must fall within the subtree rooted at $\operatorname{pre}(t)$ in the Successor Forest; (ii) **Priority Dominance**: if $\theta(s^{-\operatorname{suc}(t)}) \neq \operatorname{pre}(t)$, then the rule priority of $t$ must be strictly higher than the child $u$ of $\operatorname{pre}(t)$ leading to that ancestor. Theorem 4.2 proves all $t$ satisfying these conditions on $\operatorname{SufSucTree}(\tau(sc))$ form exactly one monotonic path from the root, with $\theta(sc)$ being the deepest node. This transforms a history-dependent problem into a binary search on a pre-constructed static tree.

**2. DFS Timestamps + Valid Interval: Reducing the Condition to $\mathcal{O}(1)$ Interval Detection**

While the previous property is elegant, checking it repeatedly would involve Successor Forest membership queries and rule priority comparisons, introducing an $\mathcal{O}(t)$ factor. The authors linearize these tree relationships: they perform a preorder DFS on the Successor Forest, **sorting children by merge rule priority from low to high**. This ensures the timestamps of high-priority children's subtrees appear later. This guarantees that for any non-atomic token $t$, the valid candidate set $\mathcal{C}_t$ corresponds to a continuous interval $I_t = [L_t, R_t)$, where $L_t = \operatorname{dfs\_in}(\operatorname{pre}(t))$ and $R_t$ is the `dfs_in` of the first child of $\operatorname{pre}(t)$ with rule priority $\geq t$. Online detection becomes a simple integer comparison: `dfs_in(k) ∈ I_t`. Additionally, a key corollary is that **valid intervals of siblings in the SufSucTree are disjoint**, allowing unambiguous selection during search.

**3. Aho–Corasick + Centroid Decomposition: Compressing Incremental Updates into Logarithmic Time**

An $\mathcal{O}(1)$ predicate isn't enough—naively traversing the SufSucTree could reach $\mathcal{O}(t)$ depth. Two standard tools are used. First, an Aho–Corasick automaton is built for vocabulary $\mathcal{V}$, with pre-computed "search space entry" labels for each state. When a new character arrives, the automaton transitions in $\mathcal{O}(1)$ to find $\tau(sc)$ **without back-traversing suffix links**. The transition table is compressed via square-root tiling to maintain $\mathcal{O}(1)$ queries. Second, a Centroid Search Tree (CST) with strict $\mathcal{O}(\log |\tau|)$ height is built for each $\tau \in \mathcal{V}$. Online search starts from the CST root, using interval detection for the current centroid $u$. If $u$ is **invalid**, the target must be on the parent side of $u$ in the SufSucTree; if **valid**, $u$ is on the path, and the algorithm checks if a deeper valid child exists via binary search on sibling intervals. Since CST depth is $\mathcal{O}(\log t)$ and each step involves an $\mathcal{O}(\log t)$ binary search, the total is $\mathcal{O}(\log^2 t)$ per byte.

### Loss & Training

This work is purely algorithmic/data-structural and has no training objective. The eager output module (§6) is notable: it allows tokenization to be pipelined with model inference. It maintains an "active frontier" $\mathcal{P}$ of candidate tokens, with a window bounded by the Aho–Corasick state depth $d(s)$. When all active paths converge to the same child of the virtual root, the token boundary is fixed and can be immediately emitted. Eager mode introduces roughly a 10% throughput overhead compared to non-eager mode.

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

Using inputs of `'a' × 2^k`, throughput was compared on a logarithmic scale:

| Implementation | Complexity Behavior | Notes |
|------|-----------|------|
| Ours (Inc. BPE) | Stable $\mathcal{O}(n \log^2 t)$ | Throughput remains flat |
| tiktoken | Significant $\mathcal{O}(n^2)$ decay | Throughput drops as length increases |
| O200K Model | Regex stage failure | Long inputs trigger upstream errors |
| Ours (Eager) | ~10% slower than non-eager | Overhead from active frontier maintenance |

### Key Findings

- **Maximum speedup on CodeLlama (up to 3.13×)**: CodeLlama does not use regex pre-tokenization, so BPE processes full normalized text. The original `tokenizers` heap method suffers from $\mathcal{O}(n \log n)$ on long inputs, while our incremental algorithm's advantages are fully realized. Conversely, when inputs are pre-split finely (e.g., English + Qwen-3), implementation constants dominate, and our gains are marginal or slightly negative (0.99×–1.05×).
- **Significant gains in Chinese + tiktoken (1.35×–1.59×)**: Chinese regex splitting is naturally coarser, exposing the BPE bottleneck in `tiktoken`. Our strict per-byte bound directly addresses this.
- **Pathological inputs as a "killer app"**: `tiktoken` shows clear $\mathcal{O}(n^2)$ decay on concentrated repetitions, while ours remains flat. This is not just a performance issue but a security one, serving as a mitigation against DoS attacks via algorithmic complexity.

## Highlights & Insights

- **Theoretical Properties to Data Structures**: The "prefix consistency" from Berglund & van der Merwe (2023) is an abstract algebraic property; this paper translates it into "Last Token Recursion → SufSucTree → Monotonic Path → DFS Intervals → Centroid Decomposition." Every step is necessary to reach $\mathcal{O}(\log^2 t)$.
- **Engineering Aesthetics of a Drop-in Replacement**: The authors restricted changes to the BPE stage, leaving segmentation, normalization, and caching untouched. This cleanly isolates the algorithmic contribution and allows for zero-friction adoption in existing LLM pipelines.
- **Complexity as an Attack Surface**: Reframing $\mathcal{O}(n^2)$ tokenization as an algorithmic complexity DoS vulnerability is insightful. Since LLM serving systems are exposed to the internet, attackers could freeze the tokenization stage with long repetitive strings. Binding "strict worst-case complexity" to "security" extends the impact beyond mere efficiency.

## Limitations & Future Work

- **Regex Bottleneck Remains**: Profiling shows that normalization and regex pre-tokenization have become the new bottlenecks. While BPE is now incremental, the upstream stages are still offline.
- **Concentrated Acceleration**: In fine-grained pre-tokenization English scenarios, the algorithm is roughly equal to or slightly slower than existing ones, indicating the overhead of the new algorithm when inputs are already split into small pieces.
- **Eager Output Overhead**: 10% overhead is notable in highly optimized tokenizers. The authors suggest amortization through pipeline parallelism, but actual benefits depend on downstream inference speed.
- **Lack of SentencePiece Support**: The method is formalized under standard BPE merge semantics. SentencePiece-style models (like Gemma-3) with certain "un-properizable" vocabularies are currently not supported.

## Related Work & Insights

- **vs Hugging Face `tokenizers`**: Their global priority queue is offline and log-linear on full input. Ours is incremental and strictly $\mathcal{O}(\log^2 t)$ per byte, resulting in up to 3× speedup for CodeLlama.
- **vs OpenAI `tiktoken`**: They use regex to bypass global BPE complexity, but the BPE stage is still $\mathcal{O}(n^2)$ in the worst case. Ours provides algorithmic guarantees without relying on regex "patches."
- **vs rust-gems `bpe` crate**: They also use Aho–Corasick for incremental BPE but lack formal worst-case complexity proofs. This paper provides both rigorous theory and engineering.
- **vs Berglund & van der Merwe 2023**: They provided formal semantics and prefix consistency but did not solve the algorithmic problem of bounding the lookahead required for incremental updates.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First incremental BPE with strict $\mathcal{O}(\log^2 t)$ complexity.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive testing across 8 models and 3 corpora, though direct comparison with the `rust-gems` BPE crate is missing.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear progression from theory to data structure to complexity analysis.
- Value: ⭐⭐⭐⭐⭐ Drop-in replacement ready for modern LLM systems, addressing both efficiency and security.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Adversarial Tokenization](../../ACL2025/llm_pretraining/adversarial_tokenization.md)
- [\[NeurIPS 2025\] Differentiable Hierarchical Visual Tokenization](../../NeurIPS2025/llm_pretraining/differentiable_hierarchical_visual_tokenization.md)
- [\[ACL 2025\] Tokenization is Sensitive to Language Variation](../../ACL2025/llm_pretraining/tokenization_is_sensitive_to_language_variation.md)
- [\[ACL 2025\] Splintering Nonconcatenative Languages for Better Tokenization](../../ACL2025/llm_pretraining/splintering_nonconcatenative_languages_for_better_tokenization.md)
- [\[ACL 2025\] Incorporating Domain Knowledge into Materials Tokenization](../../ACL2025/llm_pretraining/incorporating_domain_knowledge_into_materials_tokenization.md)

</div>

<!-- RELATED:END -->
