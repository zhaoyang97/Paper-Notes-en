---
title: >-
  [Paper Note] KnapSpec: Self-Speculative Decoding via Adaptive Layer Selection as a Knapsack Problem
description: >-
  [ICML 2026][LLM Efficiency][Self-Speculative Decoding] KnapSpec reformulates draft layer selection in Self-Speculative Decoding (SSD) as a 0/1 knapsack problem. It decouples Attention and MLP modules…
tags:
  - "ICML 2026"
  - "LLM Efficiency"
  - "Self-Speculative Decoding"
  - "Layer Selection"
  - "Knapsack Problem"
  - "Long-Context Inference"
  - "Dynamic Programming"
date: 2026-05-08
content_hash: 781e172678deec86
---

# KnapSpec: Self-Speculative Decoding via Adaptive Layer Selection as a Knapsack Problem

**Conference**: ICML 2026  
**arXiv**: [2602.20217](https://arxiv.org/abs/2602.20217)  
**Code**: https://github.com/kaist-flexml-lab/knapspec (Available)  
**Area**: LLM Efficiency  
**Keywords**: Self-Speculative Decoding, Layer Selection, Knapsack Problem, Long-Context Inference, Dynamic Programming

## TL;DR
KnapSpec reformulates draft layer selection in Self-Speculative Decoding (SSD) as a 0/1 knapsack problem. It decouples Attention and MLP modules, utilizes context-length-dependent hardware latency as "weight," and employs hidden state cosine similarity (with the first rigorous proof provided) as "value." Through parallel dynamic programming (DP), it adaptively identifies sub-networks that maximize Tokens-per-Time at each step, achieving up to 1.47× real-world wall-clock speedup on Qwen3/Llama3 in long-context scenarios without additional training.

## Background & Motivation

**Background**: LLM inference bottlenecks are increasingly severe. Speculative Decoding (SD) has become a mainstream acceleration paradigm where a "small draft model predicts, and a large target model verifies." Self-Speculative Decoding (SSD) further eliminates the independent draft model by selecting a sub-network from the target model as the draft, avoiding the complexity of training and aligning two sets of weights. Representative methods include LayerSkip's early exiting, Bayesian search in Draft&Verify/SWIFT, and cosine-based DP in CLaSp.

**Limitations of Prior Work**: Existing SSD methods treat Transformer layers as "indivisible atoms" or "constant-latency black boxes." While these static heuristics suffice for short contexts, they fail as context grows: Attention latency increases linearly with sequence length ($t_{\mathtt{Attn}}=\Theta(n)$), while MLP latency remains constant ($t_{\mathtt{MLP}}=\Theta(1)$). "Layer skipping" schemes optimized for the prefill stage become inefficient during the Attention-dominant decode stage, causing acceleration gains to collapse.

**Key Challenge**: The optimal trade-off between draft latency and acceptance rate **shifts dynamically with context length**. However, prior methods search for a static, global layer subset and bind Attention/MLP together, artificially restricting the search space. Furthermore, their objective functions (e.g., TPL, cosine) are decoupled from real wall-clock speed. Additionally, the use of cosine similarity as a proxy (e.g., in CLaSp) has lacked rigorous theoretical support, remaining merely "empirically effective."

**Goal**: (i) Design a layer selection framework that truly targets wall-clock throughput and is aware of context length and hardware latency; (ii) Decouple Attention and MLP to expand the search space; (iii) Provide a mathematical foundation for the proxy relationship between "cosine similarity and acceptance rate."

**Key Insight**: The authors observe that since each Attention/MLP layer has its own "latency cost" and "value contribution to final output," selecting a set of layers to maximize total value under a latency budget is naturally a **0/1 Knapsack Problem**. The knapsack problem has a standard DP solution, and the intermediate DP table provides optimal solutions for all budgets simultaneously, avoiding repeated searches.

**Core Idea**: Formulate SSD layer selection as a 0/1 knapsack problem where hardware latency is the weight and cosine similarity is the value. At each decoding step, use parallel DP to instantly recalculate the optimal draft sub-network, then perform a grid search over the DP candidate set to select the configuration that maximizes TPT.

## Method

### Overall Architecture

Given a target model with $L$ layers, each containing an Attention sub-layer and an MLP sub-layer, the authors flatten them into $2L$ independent "items": $f^{(2i-1)}:=f^{(i)}_{\mathtt{Attn}}$ and $f^{(2i)}:=f^{(i)}_{\mathtt{MLP}}$. A draft model $f^{(S)}$ is a sub-network composed of a subset $S\subseteq [2L]$ in their original order. The pipeline consists of three steps:

1.  **Pre-processing (One-time)**: Run micro-benchmarks on the target hardware to fit Attention latency as a function of context length $n$, $t_{\mathtt{Attn}}(n)$, and measure MLP latency as a constant $t_{\mathtt{MLP}}$.
2.  **Dynamic Decision per Speculation Step**: Take the hidden states $X\in\mathbb{R}^{r\times d}$ of $r$ tokens generated in the last $m=5$ steps as a reference. Calculate integer weights $(w_{\mathtt{Attn}}, w_{\mathtt{MLP}})$ based on current context length. Run parallel DP to obtain a candidate set $\mathcal{A}=\{S_k\}_{k\in[K]}$ (each budget $k$ corresponds to an optimal $S_k$). Perform a grid search over $\mathcal{A}\times[D]$ to maximize TPT, yielding $(S^*, \gamma^*)$.
3.  **Speculation & Verification**: Autoregressively draft $\gamma^*$ tokens using $f^{(S^*)}$, followed by a single parallel verification by the target model, integrated with dynamic early termination (top-1 confidence $\tau_{\text{conf}}=0.7$).

The entire process **requires no extra training or parameters**, and the output distribution is strictly equivalent to the target model (inheriting acceptance/rejection sampling guarantees).

### Key Designs

1.  **TPT (Tokens-per-Time) — Wall-clock Aligned Objective**:
    *   **Function**: Upgrades the Tokens-per-Layer (TPL) metric from DEL to a throughput metric based on actual latency, serving as the final optimization goal.
    *   **Mechanism**: The expected tokens produced per speculation step is $\frac{1-\alpha_S^{\gamma+1}}{1-\alpha_S}$ (mean of a truncated geometric distribution, where $\alpha_S$ is the acceptance rate). The divisor is the total latency: $\gamma\, t_{\mathtt{Draft}}(S) + t_{\mathtt{Target}}$, where $t_{\mathtt{Draft}}(S)=n_{\mathtt{Attn}}(S)\cdot t_{\mathtt{Attn}} + n_{\mathtt{MLP}}(S)\cdot t_{\mathtt{MLP}}$ and $t_{\mathtt{Target}}=L(t_{\mathtt{Attn}}+t_{\mathtt{MLP}})$. Thus, $\text{TPT}(S,\gamma)=\frac{1-\alpha_S^{\gamma+1}}{1-\alpha_S}\cdot\frac{1}{\gamma\, t_{\mathtt{Draft}}(S)+t_{\mathtt{Target}}}$.
    *   **Design Motivation**: When $t_{\mathtt{Attn}}=t_{\mathtt{MLP}}$, TPT reduces to TPL. However, on real hardware, the gap between them widens with context length. Results show that the Pearson correlation and $R^2$ between best-TPT and actual throughput are significantly higher than for acceptance rate, validating that time-based metrics are better predictors of speedup.

2.  **Knapsack Formulation + Parallel DP — Reducing NP-hard Search to $O(nL)$**:
    *   **Function**: Searches for approximate optimal layer subsets $\{S_k\}$ across all latency budgets in each decoding step in $O(nL)$ time and $O(L)$ VRAM.
    *   **Mechanism**: Perform **integer weight normalization**: let $\Delta=\min(t_{\mathtt{Attn}}(n), t_{\mathtt{MLP}})$, then $w_{\mathtt{Attn}}=\lfloor t_{\mathtt{Attn}}(n)/\Delta\rceil$ and $w_{\mathtt{MLP}}=\lfloor t_{\mathtt{MLP}}/\Delta\rceil$. Use proxy $\cos(f(X), f^{(S)}(X))$ to replace the non-differentiable $\alpha_S$, forming the knapsack problem: $\max_{S\subseteq[2L]} \cos(f(X), f^{(S)}(X))$ s.t. $n_{\mathtt{Attn}}(S)w_{\mathtt{Attn}}+n_{\mathtt{MLP}}(S)w_{\mathtt{MLP}}=k$. The DP table $g[i,j]\in\mathbb{R}^{r\times d}$ stores the optimal hidden state for the first $i$ layers with skipped weight $j$. The transition compares "executing layer $i$" ($h_{\mathtt{e}}=f^{(i)}(g[i-1,j])$) with "skipping layer $i$" ($h_{\mathtt{s}}=g[i-1,j-w_i]$) based on cosine score. The $j$ dimension is computed in **batch parallel** on GPU. Backtracking $g[2L, \cdot]$ yields $\mathcal{A}$.
    *   **Design Motivation**: The original search space $2^{2L}D$ is exponentially infeasible. Decoupling Attention/MLP doubles the items to $2L$ but introduces new degrees of freedom (e.g., keeping Attention but skipping its MLP). DP allows covering all budgets in one pass. Pruning with threshold $\tau=0.5$ and a $K/2$ upper bound reduces overhead to the same order of magnitude as a standard AR decode step.

3.  **Lemma 4.1 — Cosine Similarity as a Rigorous Proxy for Acceptance Rate**:
    *   **Function**: Provides theoretical legitimacy for using cosine similarity as the "value," addressing a long-standing mathematical gap in methods like CLaSp.
    *   **Mechanism**: For LM head word vectors $w_1,\dots,w_V$, let the target hidden state $x$ predict $i^*=\arg\max_i\langle w_i, x\rangle$. Define margin $\xi(x)=\langle w_{i^*},x\rangle - \max_{j\neq i^*}\langle w_j, x\rangle$. The paper proves: if $\|x'\|_2=\|x\|_2$ and $\cos(x,x')\geq 1-\frac{\xi(x)^2}{2\|x\|_2^2 \max_{j\neq i^*}\|w_{i^*}-w_j\|_2^2}$, then $\arg\max_i\langle w_i, x\rangle = \arg\max_i\langle w_i, x'\rangle$, ensuring greedy token consistency.
    *   **Design Motivation**: The equal norm assumption is approximately satisfied by RMSNorm in modern LLMs. The lemma proves that sufficiently high cosine similarity **guarantees** that the draft and target models select the same token under greedy decoding.

### Loss & Training

Ours is completely training-free. Runtime hyperparameters: pruning threshold $\tau=0.5$, dynamic early exit threshold $\tau_{\text{conf}}=0.7$, maximum draft length $D=10$, and historical reference window $m=5$. Latency coefficients $(t_{\mathtt{Attn}}(n), t_{\mathtt{MLP}})$ are measured once per hardware during pre-processing.

## Key Experimental Results

### Main Results

Evaluated on Qwen3 (4B-32B) and Llama3 (1B-70B) for **long-context generation** (AIME, MMLU-Pro) and **long-context input** (GovReport, PG19, BookSum).

| Model | Task | Metric | AR | SWIFT | CLaSp | KnapSpec |
|------|------|------|----|----|----|----|
| Qwen3-32B | AIME24 | Speedup | 1.00× | 1.23× | 1.30× | **1.43×** |
| Qwen3-32B | MMLU-Pro | TPT | 23.15 | 21.75 | 23.90 | **34.62** |
| Llama3.1-70B | GovReport | Speedup | 1.00× | 1.33× | 1.22× | **1.47×** |
| Llama3.1-8B | AIME24 | Speedup | 1.00× | 1.05× | 1.11× | **1.28×** |
| Llama3.1-8B | AIME24 | $\alpha$ | — | 62.1% | 91.7% | **97.0%** |

KnapSpec achieved the highest TPT and speedup across **all 48 configurations**, with a maximum wall-clock speedup of 1.47×. While acceptance rates are comparable to CLaSp, throughput is significantly higher due to cheaper draft sub-networks.

### Ablation Study

| Config | Observation |
|------|---------|
| TPT vs TPL/acc-rate | TPT has the highest PCC and $R^2$ with real throughput. |
| Decoupling vs Binding | Decoupling Attn/MLP adds +0.1–0.2× speedup on average. |
| Pruning ($\tau=0.5$) | Saves significant DP time without performance degradation. |
| Nucleus Sampling ($T=0.7$) | Speed advantages remain stable under sampling. |

### Key Findings

- **Attention is the bottleneck in long contexts**: Methods unaware of context length (e.g., SWIFT) can be slower than AR on long-input tasks (speedup < 1×), whereas KnapSpec maintains 1.1–1.5×.
- **Dynamic selection is essential**: Global optimal subsets degrade quickly as context grows; per-step knapsack decisions are necessary.
- **TPT is the correct objective**: CLaSp often has a higher acceptance rate, but its TPT is lower due to heavier draft sub-networks.

## Highlights & Insights

- **Clean Formalization**: Modeling SSD as a knapsack problem (Items=Layers, Weight=Latency, Value=Cosine) provides both optimal structure and an efficient $O(nL)$ DP algorithm.
- **Mathematical Foundation**: Lemma 4.1 completes the mathematical proof for the entire family of cosine-based SSD papers (CLaSp, ASD, DEL).
- **Hardware-Awareness**: TPT aligns with wall-clock time, and weights are derived from hardware profiling, making the pipeline deployment-ready.
- **Transferable Insights**: The decoupling of Attention/MLP can be applied to other layer-selection scenarios like conditional early exiting or KV cache retention.

## Limitations & Future Work

- **Offline Profiling Dependency**: Changing GPUs or batch sizes requires re-measuring latency coefficients; online light-weight calibration could be explored.
- **DP Overhead**: For tiny models (e.g., Llama3.2-1B), the relative overhead of DP is larger, narrowing speedup gains to 1.06–1.13×.
- **Greedy Assumption**: Lemma 4.1 covers greedy decoding. While experimental results show it works for sampling, a rigorous theoretical guarantee for nucleus sampling is still needed.

## Related Work & Insights

- **vs CLaSp**: CLaSp also uses cosine + DP but binds Attention/MLP, uses fixed layer budgets, and is context-unaware. KnapSpec upgrades all these and adds theoretical proof.
- **vs SWIFT / Draft&Verify**: These use Bayesian optimization which is slow and context-insensitive. KnapSpec uses DP for an exact solution at each step.
- **vs DEL**: DEL optimizes TPL and is restricted to prefix-based early exiting. KnapSpec generalizes TPL to TPT and allows a larger search space.
- **vs LayerSkip / Kangaroo**: These require training/fine-tuning. KnapSpec is training-free and plug-and-play.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Knapsack formalization, module decoupling, and context-awareness combined with the first rigorous cosine-acceptance proof.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 models, 6 tasks, 4 baselines, covering both greedy/sampling modes and TPT correlation analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear progression from problem to theory to experiment.
- Value: ⭐⭐⭐⭐⭐ Training-free, achieves 1.47× speedup in long-context tasks, and provides a framework reusable for future SSD research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SpecBound: Adaptive Bounded Self-Speculation with Layer-wise Confidence Calibration](../../ACL2026/llm_efficiency/specbound_adaptive_bounded_self-speculation_with_layer-wise_confidence_calibrati.md)
- [\[ICML 2026\] MineDraft: A Framework for Batch Parallel Speculative Decoding](minedraft_a_framework_for_batch_parallel_speculative_decoding.md)
- [\[NeurIPS 2025\] OmniDraft: A Cross-Vocabulary Online Adaptive Drafter for On-Device Speculative Decoding](../../NeurIPS2025/llm_efficiency/omnidraft_a_cross-vocabulary_online_adaptive_drafter_for_on-device_speculative_d.md)
- [\[ACL 2026\] Speculative Verification: Exploiting Information Gain to Refine Speculative Decoding](../../ACL2026/llm_efficiency/speculative_verification_exploiting_information_gain_to_refine_speculative_decod.md)
- [\[ACL 2026\] Multi-Drafter Speculative Decoding with Alignment Feedback](../../ACL2026/llm_efficiency/multi-drafter_speculative_decoding_with_alignment_feedback.md)

</div>

<!-- RELATED:END -->
