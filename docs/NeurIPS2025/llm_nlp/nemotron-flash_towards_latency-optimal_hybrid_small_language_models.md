---
title: >-
  [Paper Note] Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models
description: >-
  [NeurIPS 2025][LLM (Other)][Hybrid Operators] Nemotron-Flash constructs a latency-optimal family of small language models through systematic optimization of depth-width ratios, evolutionary search over hybrid operator combinations (DeltaNet + Mamba2 + Attention), and weight-normalization-based training. Compared to Qwen3-1.7B/0.6B, it achieves 1.3×/1.9× latency reduction alongside a +5.5% average accuracy improvement.
tags:
  - "NeurIPS 2025"
  - "LLM (Other)"
  - "Hybrid Operators"
  - "Depth-Width Ratio"
  - "Weight Normalization"
  - "Evolutionary Search"
date: 2026-05-08
content_hash: f348d7a09999ca3b
---

# Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2511.18890](https://arxiv.org/abs/2511.18890)  
**Code**: [Hugging Face Model Card](https://huggingface.co/nvidia/Nemotron-Flash-1B)  
**Area**: SLM Design, Latency Optimization
**Keywords**: Hybrid Operators, Depth-Width Ratio, Weight Normalization, Evolutionary Search

## TL;DR
Nemotron-Flash constructs a latency-optimal family of small language models through systematic optimization of depth-width ratios, evolutionary search over hybrid operator combinations (DeltaNet + Mamba2 + Attention), and weight-normalization-based training. Compared to Qwen3-1.7B/0.6B, it achieves 1.3×/1.9× latency reduction alongside a +5.5% average accuracy improvement.

## Background & Motivation
Existing SLM designs primarily pursue parameter efficiency (parameter-optimal); however, reducing parameter count does not proportionally reduce actual device latency. For instance, deep-thin architectures adopted by MobileLLM and SmolLM exhibit high parameter efficiency, yet incur greater latency during real GPU inference due to their large layer counts. Furthermore, given the proliferation of efficient attention operators (Mamba, DeltaNet, GLA, etc.), the community lacks systematic exploration of their synergistic effects within hybrid models, and existing hybrid model operator combinations rely on manual heuristics.

This paper argues that SLM design should treat **actual device latency** as the primary optimization objective rather than parameter count, and provides a generalizable methodology along two axes: architectural design and training strategy.

## Method

### Overall Architecture
The paper improves the accuracy–latency trade-off of SLMs along three dimensions:
1. **Depth-width ratio optimization**: Determining the optimal model depth and width under a given latency budget
2. **Hybrid operator search**: Evolutionary search to discover complementary operator combinations
3. **Training improvements**: Weight normalization + learnable Meta Tokens

### Depth-Width Ratio Optimization
The authors train a series of Llama models (depths 6/12/18/24/30, with width varied at each depth) on 100B tokens of the Smollm-corpus, yielding the following key findings:

- **Deep models are parameter-efficient but latency-inefficient**: Deep-thin models are superior on the accuracy–parameter curve but inferior on the accuracy–latency curve.
- **An optimal depth-width ratio exists**: For example, under a 3-second latency budget, depth 12 achieves the best accuracy.
- **The optimal depth-width ratio increases with the latency budget**: A larger latency budget permits deeper models.

To more precisely identify the sweet-spot depth-width ratio, the authors extend existing scaling laws by decoupling model size $P$ into depth $D$ and width $W$:

$$\mathcal{L}(D, W, N) = \mathcal{L}_0 + aD^{-\alpha} + bW^{-\beta} + cN^{-\gamma}$$

where $a, b, c$ control the contribution weights of each dimension, and $\alpha, \beta, \gamma$ control the rates of diminishing returns. Experiments show that this scaling law extrapolates to unseen depth-width configurations with a PPL error within 5.3%.

### Hybrid Operator Search
**Operator evaluation**: Seven operators—Mamba, Mamba2, GLA, DeltaNet, Gated DeltaNet, RWKV7, and Sliding Window Attention (SWA)—are uniformly evaluated on a 500M model. DeltaNet and Gated DeltaNet are found to lie on the PPL–latency Pareto frontier.

**Operator combination exploration**: Different operators are paired with Mamba2 or Attention to construct hybrid models:
- DeltaNet/Gated DeltaNet + Mamba2 combinations yield the best results, outperforming pure models on both PPL and CR accuracy.
- Gains from pairing with Attention are inconsistent.
- Performance gaps between individual operators narrow within hybrid models, owing to complementary memory mechanisms.

**Evolutionary search framework**:
- **Search proxy**: Spearman correlation between short-training PPL and full-training PPL reaches 88.8%, establishing a reliable proxy.
- **Search space**: Candidate operators are DeltaNet, Attention, and Mamba2; the search covers operator ratios across three stages (early/middle/late), the number of FFN blocks per operator type, and repetition counts.
- **Search algorithm**: Aging Evolution with tournament selection, single-factor mutation, and short-training evaluation per iteration.
- **Search result**: The discovered latency-friendly architecture alternates [DeltaNet-FFN-Mamba2-FFN] and [Attention-FFN-Mamba2-FFN] blocks.

Compared to pure-model baselines at equivalent latency, the searched hybrid architecture outperforms on both PPL and CR accuracy (51.04% vs. 50.38% for the next-best DeltaNet baseline).

### Weight Normalization
Standard training produces weight matrices with large-magnitude outliers, causing relative weight updates to diminish at low learning rates in later training and leading to learning stagnation. Inspired by nGPT, the authors project weights onto the unit-norm sphere after each training iteration:

- Case-1 (matrices acting on hidden features): $\mathbf{W}_{i,:} \leftarrow \mathbf{W}_{i,:} / \|\mathbf{W}_{i,:}\|_2$
- Case-2 (matrices whose outputs are added back to the hidden layer): $\mathbf{W}_{:,j} \leftarrow \mathbf{W}_{:,j} / \|\mathbf{W}_{:,j}\|_2$

This yields an average CR accuracy improvement of +1.20% and PPL reduction of 0.66 across Llama, DeltaNet, and Mamba2 architectures. Compared to the full nGPT scheme, this approach eliminates the 20%+ training overhead introduced by activation normalization layers while achieving comparable task performance.

### Meta Token
A set of 256 learnable tokens is prepended to the input sequence, simultaneously alleviating the attention sink problem (for softmax attention) and providing a learned cache initialization for linear attention. This consistently improves accuracy by +0.45% with negligible overhead.

## Model Configurations

| Model | Parameters | Hidden Dim | Blocks | Operators | Structure |
|-------|-----------|------------|--------|-----------|-----------|
| Nemotron-Flash-1B | 0.96B | 2048 | 12 | 24 | Block-1 × 4 + Block-2 × 2, alternating |
| Nemotron-Flash-3B | 2.7B | 3072 | 18 | 36 | Block-1 × 6 + Block-2 × 3, alternating |

Block-1 = DeltaNet-FFN-Mamba2-FFN; Block-2 = Attention-FFN-Mamba2-FFN. Training uses 4.5T tokens on 256 H100 GPUs with Adam (no weight decay) and a cosine learning rate of 1e-3.

## Key Experimental Results

### Nemotron-Flash vs. SOTA Base Models (H100, decode 8k tokens, BS=1)

| Model | Params | Depth | Latency (s) | Max-BS Throughput (tok/s) | MMLU | CR | Math | Coding | Recall | Avg |
|-------|--------|-------|-------------|--------------------------|------|----|------|--------|--------|-----|
| Qwen2.5-0.5B | 0.5B | 24 | 22.81 | 2,382 | 47.6 | 47.5 | 32.7 | 32.1 | 65.4 | 45.2% |
| Qwen3-0.6B | 0.6B | 28 | 27.55 | 160 | 52.4 | 48.9 | 36.9 | 24.3 | 62.9 | 44.1% |
| **NF-1B** | 0.96B | 12 | **14.45** | **7,289** | 44.6 | **54.5** | 34.9 | **37.9** | 67.1 | **49.6%** |
| Qwen3-1.7B | 1.7B | 28 | 36.20 | 157 | 62.5 | 57.2 | 53.7 | 43.8 | 66.4 | 55.5% |
| Qwen2.5-3B | 3B | 36 | 49.40 | 459 | 65.6 | 58.9 | 53.8 | 49.5 | 73.0 | 59.0% |
| **NF-3B** | 2.7B | 18 | **28.71** | **2,939** | 61.2 | **61.0** | **57.6** | **53.3** | 73.3 | **61.0%** |

- NF-1B vs. Qwen3-0.6B: +5.5% accuracy, 1.9× lower latency, 45.6× higher throughput
- NF-3B vs. Qwen2.5-3B: +2.0% accuracy, 1.7× lower latency, 6.4× higher throughput
- NF-3B contains only 2 full attention layers yet achieves Recall 73.3%, demonstrating that full KV cache coverage across all layers is unnecessary

### Instruct Model Comparison

| Model | Params | Latency (s) | Throughput (tok/s) | MMLU | GPQA | GSM8K | IFEval | Avg |
|-------|--------|-------------|-------------------|------|------|-------|--------|-----|
| Qwen2.5-1.5B | 1.5B | 34.50 | 687 | 59.7 | 30.1 | 56.0 | 46.8 | 48.2% |
| Qwen3-1.7B | 1.7B | 36.20 | 157 | 60.2 | 28.3 | 64.9 | 31.3 | 46.2% |
| **NF-3B-Inst** | 2.7B | 28.71 | 2,939 | 60.3 | 29.5 | **69.5** | **52.0** | **52.8%** |

### Ablation Study: Weight Normalization (1B model, 100B tokens)

| Model | Setting | Wiki PPL | CR Accuracy |
|-------|---------|----------|-------------|
| Llama 1B | w/o wnorm | 18.67 | 53.81% |
| Llama 1B | w/ wnorm | 18.03 | 54.85% (+1.04) |
| DeltaNet 1B | w/o wnorm | 18.86 | 53.46% |
| DeltaNet 1B | w/ wnorm | 18.19 | 54.39% (+0.93) |
| Mamba2 1B | w/o wnorm | 18.44 | 53.30% |
| Mamba2 1B | w/ wnorm | 17.88 | 54.71% (+1.41) |

## Highlights & Insights
1. **Parameter efficiency ≠ latency efficiency**: This work is the first to systematically quantify that deep-thin models are inferior to models with an appropriate depth-width ratio on the accuracy–latency trade-off.
2. **Extended scaling law**: Decoupling the scaling law into depth and width as independent variables provides a principled approach for identifying the sweet-spot depth-width ratio.
3. **Golden hybrid operator pairing**: The alternating DeltaNet-Mamba2 combination achieves Pareto optimality in both latency and accuracy; evolutionary search is more reliable than manual design.
4. **Short-training proxy is effective**: An 88.8% Spearman correlation substantially reduces the cost of evolutionary search.
5. **Weight normalization is simple yet effective**: It saves 20% of training overhead compared to nGPT with nearly identical performance, and generalizes across architectures.
6. **Full attention is not necessary**: NF-3B achieves state-of-the-art Recall performance with only 3 full attention layers, supporting the argument for KV cache savings in hybrid attention models.

## Limitations & Future Work
- The search space is limited to 3 stages × 3 operator types; a larger search space may yield better architectures.
- Latency evaluation is conducted only on A100/H100; optimal architectures may differ across hardware platforms (GPUs, TPUs, edge devices).
- The sliding window is fixed at 512, which may be insufficient for tasks requiring long-range dependencies.
- The quality of scaling law fitting depends on the coverage of training points; extrapolation accuracy at extreme depth-width ratios remains uncertain.

## Related Work & Insights
- **SLM design**: MobileLLM emphasizes the parameter efficiency of deep-thin architectures; MiniCPM proposes scaling law guidance for SLM training. This paper counters the direct adoption of deep-thin models by using latency as the evaluation metric.
- **Efficient attention**: Mamba/Mamba2 proposes selective state space models; DeltaNet/GLA proposes linear attention variants, both of which exhibit limited recall capacity when used in isolation.
- **Hybrid models**: Jamba, Hymba, and similar works manually combine Mamba and Attention layers; this paper automates that process via evolutionary search.

## Rating
⭐⭐⭐⭐⭐ (5/5)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Q♯: Provably Optimal Distributional RL for LLM Post-Training](qsharp_provably_optimal_distributional_rl_for_llm_post-training.md)
- [\[ACL 2025\] Mixture of Small and Large Models for Chinese Spelling Check](../../ACL2025/llm_nlp/mixture_of_small_and_large_models_for_chinese_spelling_check.md)
- [\[ACL 2025\] PiFi: Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models](../../ACL2025/llm_nlp/plugin_finetuning_bridge.md)
- [\[ACL 2025\] Are Optimal Algorithms Still Optimal? Rethinking Sorting in LLM-Based Pairwise Ranking with Batching and Caching](../../ACL2025/llm_nlp/are_optimal_algorithms_still_optimal_rethinking_sorting_in_llm-based_pairwise_ra.md)
- [\[ICLR 2026\] COSMOS: A Hybrid Adaptive Optimizer for Efficient Training of Large Language Models](../../ICLR2026/llm_nlp/cosmos_a_hybrid_adaptive_optimizer_for_efficient_training_of_large_language_mode.md)

</div>

<!-- RELATED:END -->
