---
title: >-
  [Paper Note] LycheeDecode: Accelerating Long-Context LLM Inference via Hybrid-Head Sparse Decoding
description: >-
  [ICLR 2026][LLM Efficiency][Long-context inference] LycheeDecode is proposed to accelerate long-context LLM decoding by fine-grainedly partitioning attention heads into a small number of retrieval heads (performing full…
tags:
  - "ICLR 2026"
  - "LLM Efficiency"
  - "Long-context inference"
  - "sparse attention"
  - "attention head specialization"
  - "KV cache optimization"
  - "HardKuma distribution"
date: 2026-05-08
content_hash: 19a5c9b8230e4540
---

# LycheeDecode: Accelerating Long-Context LLM Inference via Hybrid-Head Sparse Decoding

**Conference**: ICLR 2026
**arXiv**: [2602.04541](https://arxiv.org/abs/2602.04541)
**Code**: [https://github.com/](https://github.com/)
**Area**: LLM Efficiency
**Keywords**: Long-context inference, sparse attention, attention head specialization, KV cache optimization, HardKuma distribution

## TL;DR
LycheeDecode is proposed to accelerate long-context LLM decoding by fine-grainedly partitioning attention heads into a small number of retrieval heads (performing full attention to select critical tokens) and a large number of sparse heads (reusing the selected tokens for sparse computation). Head roles are learned end-to-end via the Hard Kumaraswamy distribution, achieving 2.7× speedup at 128K context length with no performance degradation.

## Background & Motivation
Long-context LLMs (e.g., Gemini-2.5 and Qwen2.5-1M supporting up to 1M tokens) have become mainstream, yet autoregressive decoding suffers from linearly growing KV caches, creating severe memory and latency bottlenecks. Existing sparse attention methods fall into two categories: **eviction-based** (SnapKV, H2O, etc., permanently discarding tokens) and **selection-based** (TidalDecode, SeerAttention, etc., dynamically selecting token subsets).

A key observation is that recent works (TidalDecode, OmniKV) find critical tokens to be highly consistent across adjacent layers, motivating a **layer-level sharing strategy** where all heads within a layer share the same set of critical tokens. However, this assumption is overly coarse: heatmap analysis reveals that the top-k overlap rate varies dramatically across heads within the same layer (e.g., head 14 at 0% and head 24 at 100% overlap between adjacent layers), indicating that uniform layer-level sharing suppresses functional diversity across heads.

The root cause is that **layer-level sharing granularity is too coarse and neglects attention head specialization**. The starting point is to refine the sharing granularity from the layer level to the head level, allowing different heads to play distinct roles. The core idea is that a small number of retrieval heads identify critical tokens via full attention, while the majority of sparse heads reuse those tokens for efficient computation.

## Method

### Overall Architecture
LycheeDecode is a head-level sparse decoding framework comprising two core components:
1. **Head role assignment**: Attention heads are divided into Retrieval Heads and Sparse Heads.
2. **HardKuma head-type learning**: Hard Kumaraswamy distributions are used to learn head role assignments end-to-end.

### Key Designs

1. **Retrieval Heads**: Perform standard dense attention over the full sequence, selecting a set of top-k critical token indices $\mathcal{S}_h^{(l+1)} = \text{argsTopK}(A_h^{(l)}, k)$ and passing this set to the head with the same index in the next layer. All heads in the first layer default to retrieval heads to initialize token sets.

2. **Sparse Heads**: Inherit the token set $\mathcal{S}_h^{(l)}$ from the previous layer and compute attention only over that subset: $O_h^{(l)} = \text{softmax}\left(\frac{q_h^{(l)} (K_h^{(l)}[\mathcal{S}_h^{(l)}])^T}{\sqrt{d_k}}\right) V_h^{(l)}[\mathcal{S}_h^{(l)}]$, forwarding the token set unchanged. This substantially reduces both computation and KV cache loading overhead.

3. **HardKuma Head Specialization**: Head role assignment is inherently a discrete optimization problem (binary variables). DuoAttention learns continuous variables and rounds them post-training, causing train–inference inconsistency. The Hard Kumaraswamy distribution is introduced instead: (1) samples from a uniform distribution are transformed via the Kuma inverse CDF; (2) the result is linearly stretched to an interval $(p, q)$ with $p<0, q>1$; (3) hard clipping to $[0,1]$ is applied, naturally concentrating outputs near 0 and 1 while remaining fully differentiable. Each head learns parameters $\alpha_h^{(l)}, \beta_h^{(l)}$; at inference, a head is assigned as a retrieval head if $\mathbb{E}[z_h^{(l)}] > 0.5$.

### Loss & Training
During training, each head computes both sparse and full attention maps simultaneously, linearly interpolated by the HardKuma sample $z_h^{(l)}$:
$$\tilde{A}_h^{(l)} = z_h^{(l)} \cdot A_{R,h}^{(l)} + (1 - z_h^{(l)}) \cdot A_{S,h}^{(l)}$$

The loss combines **distillation and a Lagrangian sparsity constraint**:
- Distillation loss: L2 distance between the student (hybrid attention) and teacher (full attention) logits.
- Sparsity constraint: $\min_{\alpha,\beta} \max_{\lambda \geq 0} \mathcal{L}_{\text{distill}} + \lambda \cdot (\mathbb{E}[\|\mathbf{z}\|_0] - N_{\text{target}})$

$\mathbb{E}[\|\mathbf{z}\|_0]$ admits a closed-form solution, and $\lambda$ is adjusted automatically via gradient ascent, eliminating the need for manual hyperparameter search. Training requires only 3,000 steps on a single A100 GPU (a few hours).

## Key Experimental Results

### Main Results (LongBench Long-Context Understanding)

| Method (Budget) | MFQA | NrtQA | Qasper | 2Wiki | HotQA | QMSum | TrQA | PRe | Avg |
|---|---|---|---|---|---|---|---|---|---|
| Full Attention (Llama3-8B) | 30.76 | 5.52 | 14.56 | 13.32 | 11.50 | 19.43 | 86.56 | 77.00 | 32.33 |
| TidalDecode (4096) | 30.94 | 6.19 | 13.85 | 14.40 | 13.71 | 19.48 | 86.30 | 78.00 | 32.86 |
| **LycheeDecode (4096)** | **30.11** | **5.85** | **14.39** | **12.86** | **12.66** | **19.30** | **86.78** | **82.58** | **33.07** |
| Full Attention (Qwen3-8B) | 25.84 | 3.43 | 10.96 | 11.97 | 11.74 | 20.90 | 90.21 | 89.08 | 33.02 |
| TidalDecode (4096) | 23.57 | 2.99 | 10.79 | 11.47 | 11.31 | 20.01 | 88.94 | 85.00 | 31.76 |
| **LycheeDecode (4096)** | **24.90** | **3.32** | **10.88** | **12.74** | **11.68** | **20.71** | **90.34** | **93.25** | **33.48** |

On mathematical reasoning (DeepSeek-R1-Distill-Qwen-7B), LycheeDecode + Cache Correction achieves 46.7% on AIME24 (vs. 40.0% for Full Attention), with an average score of 44.9 surpassing Full Attention's 43.0.

### Ablation Study (Head Identification Method Comparison)

| Method | Passkey Retrieval | HotpotQA |
|---|---|---|
| Direct Optimize (DuoAttention) | 32.06 | 31.02 |
| Hard Concrete | 32.13 | 30.25 |
| **HardKuma (Ours)** | **33.07** | **31.11** |

Among different sparsity strategies (Top-k / Top-p / Threshold / Ratio), the Ratio method yields the best overall performance at equivalent sparsity levels.

### Key Findings
- LycheeDecode outperforms layer-level sharing TidalDecode on both Llama3 and Qwen3, validating the superiority of head-level over layer-level strategies.
- End-to-end decoding speedup reaches 2.7× at 128K context length, with kernel-level speedup up to 7× (8/8 sparse head configuration).
- HardKuma is more stable than both DuoAttention's direct optimization and the Hard Concrete distribution.
- Inference performance even surpasses Full Attention; the authors hypothesize that head specialization helps filter irrelevant context noise.
- End-to-end speedup is maintained across multiple batch sizes, demonstrating practical applicability.

## Highlights & Insights
- **Head-level granularity** is the primary contribution of this work; heatmap visualizations and LongBench comparisons compellingly demonstrate that head functional diversity should not be suppressed by a uniform sharing strategy.
- The Retrieval–Sparse collaboration mechanism establishes an efficient information propagation pipeline: retrieval heads periodically refresh critical tokens to maintain context adaptability, while sparse heads reuse results to ensure computational efficiency.
- The HardKuma distribution elegantly resolves the end-to-end learning problem for discrete variables, offering a more principled alternative to continuous relaxation followed by rounding.
- Training overhead is minimal (a few hours on a single A100), making the method highly practical.
- The hybrid-head block-sparse kernel implemented in TileLang achieves genuine end-to-end acceleration.

## Limitations & Future Work
- The number of retrieval heads is fixed at 32; methods for automatically determining the optimal count remain unexplored.
- Head identification is slightly weaker on short-answer scenarios (e.g., HotpotQA), and performance under sparse supervision signals warrants further investigation.
- Evaluation is limited to 7B–8B scale models; effectiveness at larger scales (70B+) is unknown.
- No direct comparison with Native Sparse Attention (Qwen3's built-in sparsity) is provided.
- The block-sparse kernel implementation relies on TileLang; portability is not discussed.

## Related Work & Insights
- Compared to DuoAttention: DuoAttention also distinguishes retrieval and streaming heads, but each head makes decisions independently without the retrieval → sparse cooperative propagation mechanism.
- Compared to TidalDecode: TidalDecode shares at the layer level, whereas LycheeDecode shares at the head level, offering finer granularity.
- The approach is composable with KV cache quantization and compression methods for further memory reduction.
- The head specialization + sparsity paradigm may generalize to expert assignment in MoE architectures.
- The method is equally applicable to multimodal long-sequence scenarios (e.g., video understanding, multi-document dialogue).

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of head-level sharing and HardKuma is novel, though the retrieval/sparse head classification concept is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage of long-context understanding, mathematical reasoning, efficiency benchmarks, and ablations across Llama3 and Qwen3.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, rich figures and tables, and well-motivated presentation.
- Value: ⭐⭐⭐⭐ Practically significant for long-context LLM inference acceleration, with a low-overhead training procedure.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Understanding and Improving Length Generalization in Hierarchical Sparse Attention Models](understanding_and_improving_length_generalization_in_hierarchical_sparse_attenti.md)
- [\[ICLR 2026\] When Does Divide and Conquer Work for Long Context LLM? A Noise Decomposition Framework](when_does_divide_and_conquer_work_for_long_context_llm_a_noise_decomposition_fra.md)
- [\[ICLR 2026\] SwingArena: Adversarial Programming Arena for Long-context GitHub Issue Solving](swingarena_competitive_programming_arena_for_long-context_github_issue_solving.md)
- [\[ICLR 2026\] Semantic Parallelism: Redefining Efficient MoE Inference via Model-Data Co-Scheduling](semantic_parallelism_redefining_efficient_moe_inference_via_model-data_co-schedu.md)
- [\[ICLR 2026\] One-Prompt Strikes Back: Sparse Mixture of Experts for Prompt-based Continual Learning](one-prompt_strikes_back_sparse_mixture_of_experts_for_prompt-based_continual_lea.md)

</div>

<!-- RELATED:END -->
