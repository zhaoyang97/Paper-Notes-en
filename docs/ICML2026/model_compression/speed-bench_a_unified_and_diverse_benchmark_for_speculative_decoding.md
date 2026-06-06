---
title: >-
  [Paper Note] SPEED-Bench: A Unified and Diverse Benchmark for Speculative Decoding
description: >-
  [ICML 2026][Model Compression][Speculative Decoding] SPEED-Bench is a unified benchmark for Speculative Decoding (SD) that reveals real-world deployment behaviors often masked by "small data + single batch + HuggingFace"…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Speculative Decoding"
  - "Inference Acceleration"
  - "Evaluation Benchmark"
  - "Throughput-Latency"
  - "Production Engines"
date: 2026-05-08
content_hash: 2bc15c1cb3f11923
---

# SPEED-Bench: A Unified and Diverse Benchmark for Speculative Decoding

**Conference**: ICML 2026  
**arXiv**: [2604.09557](https://arxiv.org/abs/2604.09557)  
**Code**: HuggingFace datasets open-sourced (Paper footnote 1)  
**Area**: Model Compression / LLM Efficiency  
**Keywords**: Speculative Decoding, Inference Acceleration, Evaluation Benchmark, Throughput-Latency, Production Engines

## TL;DR
SPEED-Bench is a unified benchmark for Speculative Decoding (SD) that reveals real-world deployment behaviors often masked by "small data + single batch + HuggingFace" evaluations. It achieves this through a *Qualitative split* (880 samples maximizing semantic diversity) and a *Throughput split* (large-batch data organized by 1k–32k input length buckets covering three entropy levels), coupled with a measurement framework compatible with vLLM / TensorRT-LLM / SGLang.

## Background & Motivation

**Background**: Speculative decoding has become a mainstream method to accelerate LLM autoregressive inference—using a lightweight draft model to predict $\gamma$ tokens at once, followed by parallel verification by a target model in a single forward pass. By leveraging the fact that modern GPUs are often memory-bound ("moving weights is slower than computation"), it achieves near-lossless speedup. From vanilla SD to Medusa and EAGLE3, to cutting-edge models like Qwen3-Next / DeepSeek-R1 / Nemotron-3 that integrate multi-token prediction (MTP) heads natively, the community has accumulated a comprehensive set of drafter designs.

**Limitations of Prior Work**: Evaluating these technologies remains in a "fragmented" stage. The authors identify four specific gaps: (1) Acceptance rates are highly sensitive to text distribution and entropy, yet inconsistent datasets across papers prevent baseline comparisons; (2) Mainstream papers still use research-grade runtimes like HuggingFace, which do not reflect real-world overheads from CUDA Graphs, continuous batching, and kernel fusion in production engines like vLLM / TensorRT-LLM; (3) Most experiments focus on latency at batch size = 1, whereas industrial deployment prioritizes high-throughput scenarios where systems shift from memory-bound to compute-bound, often leading to significantly overestimated SD gains; (4) Input Sequence Lengths (ISL) are rarely tested above 8k, despite real workloads like coding assistants already entering the long-context regime.

**Key Challenge**: SD performance is **data-dependent**, but existing benchmarks (such as MT-Bench and SpecBench) lack the sample size and intra-class diversity to stably measure this dependency—most categories in SpecBench have only 10 samples with an average ISL < 100 tokens, and its multilingual subset is 100% "Translate German to English:" templates, inherently failing to reflect modern LLM input distributions.

**Goal**: To deliver a **single, reproducible** evaluation suite that answers two questions: (a) How stable is the drafter's acceptance rate across diverse semantic domains? (b) What is the actual end-to-end SD speedup ratio under real serving configurations with different batch sizes and ISLs?

**Key Insight**: The benchmark is divided into "Qualitative" and "Throughput" components. The former uses semantic embedding for redundancy-reduced sampling to minimize sample size while maximizing diversity. The latter ensures sufficient samples in each ISL bucket to plot stable throughput-latency Pareto curves, explicitly integrating with production engines rather than standalone runtimes.

**Core Idea**: Replace the previous "small-scaled patchwork + HuggingFace" evaluation method with a **compact dataset maximizing semantic diversity + a unified measurement framework in production engines**, ensuring that reported SD paper figures correspond to observable acceleration in industrial deployment.

## Method

### Overall Architecture
SPEED-Bench consists of three components: ① Qualitative split—880 selected samples from 18 public sources organized into 11 semantic categories (Coding / Humanities / Math / Multilingual / QA / RAG / Reasoning / Roleplay / STEM / Summarization / Writing), specifically for measuring Acceptance Rate (AR) and Acceptance Length (AL); ② Throughput split—organized by 5 fixed ISL buckets (1k / 2k / 8k / 16k / 32k) × 3 entropy levels (Low / Mixed / High), with 1536 samples per bucket to plot throughput-latency Pareto curves; ③ Measurement Framework—an asyncio client distributing token sequences to SGLang / vLLM / TensorRT-LLM / SpecBench, back-calculating AR based on "tokens per chunk" in streaming responses, and recording TTFT, step latency, User TPS, and Output TPS. The key principle is: **External factors (tokenizer, BOS, chat template) are normalized for parity, while internal factors (kernel / scheduler / continuous batching) are preserved to reflect real deployment.**

### Key Designs

1. **Qualitative split with Maximized Semantic Diversity**:
    - **Function**: Selects only 80 samples per category while covering a much broader semantic space than SpecBench, pushing the trade-off between "computational cost vs. representativeness" to the Pareto frontier.
    - **Mechanism**: Each prompt is embedded as a unit vector $x_i$ using OpenAI `text-embedding-3-large`. The goal is to select a subset $S$ of size $|S|=k$ from $N$ candidates to minimize the sum of pairwise cosine similarities $\mathcal{L}(S) = \sum_{i \in S} \sum_{j \in S, j \neq i} x_i^\top x_j$. As this is NP-hard, the paper uses a "Greedy + Local Search Refinement" (Greedy + LSR) algorithm. It starts from a random point, iteratively adds points using $i^\ast = \arg\min_{i \notin S} \sum_{j \in S} x_i^\top x_j$, and then repeatedly attempts swaps between $i_{out} \in S$ and $i_{in} \notin S$, executing the swap only if $\Delta < 0$ to escape local minima.
    - **Design Motivation**: Exhaustive Cartesian products of 18 sources are too costly, and random sampling leaves significant redundancy; this algorithm reduces average semantic similarity by 40% compared to SpecBench (83% for multilingual) while providing metadata for fine-grained analysis.

2. **Throughput split Bucketed by ISL × Entropy**:
    - **Function**: Replaces synthetic benchmarks with real long-context workloads that produce stable Pareto curves.
    - **Mechanism**: Samples are truncated or padded to fixed ISL buckets (1k / 2k / 8k / 16k / 32k) using the `o200k_base` tokenizer, and categorized into Low / Mixed / High entropy levels, totaling 1536 samples per bucket. An analytical proxy for "domain speedup" is also provided: $\text{Speedup} = (t_{ar} \cdot AL) / t_{sd}$, where $t_{ar}$ and $t_{sd}$ are per-step latencies for autoregressive and SD inference respectively, decoupling the data-dependent AL from system-level per-step latency.
    - **Design Motivation**: Empirical findings show that synthetic tokens trigger "trivial response" (inflating AR) and "topic latching" (hallucinating coherent text, deflating AR). Furthermore, they cause MoE routers to collapse, leading to inaccurate baseline measurements.

3. **Unified Measurement Framework in Production Engines**:
    - **Function**: Enables comparable measurements of AR / AL / TTFT / step latency / User TPS / Output TPS across different runtimes.
    - **Mechanism**: Uses an asyncio client to simulate multi-user serving. Acceptance length is back-calculated from chunk sizes in streaming responses. The paper formally defines core metrics, including the expected form of acceptance length: $\text{AL} = \mathbb{E}[L_t] = 1 + \sum_{i=1}^{\gamma} \prod_{j=1}^{i} \text{AR}_j$, where $\text{AR}_i$ is the conditional acceptance rate of the $i$-th draft token.
    - **Design Motivation**: Speculative decoding speedups measured with HuggingFace often differ vastly from production deployments. The framework is compatible with SpecBench, preserving research assets while refocusing on deployment feasibility.

### Loss & Training
This work does not involve training drafters; all results are derived from benchmark evaluations. Hyperparameters include $k=80$ (samples per Qualitative category), ISL buckets = {1k, 2k, 8k, 16k, 32k}, draft length $\gamma$, batch size, and temperature $T \in \{0, 1\}$.

## Key Experimental Results

### Main Results
Evaluation conducted on the Qualitative split covering five LLMs × four drafter types, on NVIDIA B200 with batch size = 32, draft length = 3, and $T=0$:

| Model | Drafter | Mean Acceptance Length (Mean AL) | Avg Speedup (T=0) | Avg Speedup (T=1) |
|-------|---------|--------------------------------|-------------------|-------------------|
| Llama 3.3 70B | N-Gram | 1.41 | 0.88× | 0.85× |
| Llama 3.3 70B | Vanilla SD | 2.44 | 1.60× | 1.15× |
| Llama 3.3 70B | EAGLE3 | 2.44 | 1.90× | 1.75× |
| GPT-OSS 120B | EAGLE3 | 2.25 | 1.34× | 1.06× |
| GPT-OSS 120B | Native MTP | 2.55 | 1.45× | — |
| DeepSeek R1 | Vanilla SD | 2.43 | 1.17× | 1.06× |
| Qwen3-Next | Native MTP | 2.81 | 1.20× | 1.18× |

Key Observations: (a) N-Gram yields a speedup < 1× for most LLMs, indicating heuristic drafters are detrimental at moderate concurrency (BS=32); (b) Raising temperature from 0 to 1 generally reduces speedup by 0.1–0.5×.

### Ablation Study (Semantic Diversity and Benchmark Sensitivity)

| Config | Avg Semantic Similarity | Comparison vs. SpecBench | Description |
|--------|--------------------------|-------------------------|-------------|
| Original SpecBench | Baseline | — | Dominated by MT-Bench, low intra-class diversity |
| Same sources + Random sampling | Lower in most categories | Data source validation | Proves the 18 new sources are inherently better |
| Same sources + Greedy only (No LSR) | Further reduction | Lower bound of algorithm contribution | Superior to random |
| Same sources + Greedy + LSR (Ours) | 40% lower vs. SpecBench | Best in all categories | Local swaps escape local minima |

### Key Findings
- **Production Engines vs. HuggingFace**: Speedups on vLLM / TensorRT-LLM are often partially offset by "system-level optimizations."
- **Synthetic Tokens "Mislead" Evaluation**: Large gaps between real long prompts and synthetic ones can reverse conclusions about drafter superiority.
- **Optimal Draft Length Coupled with Batch Size**: The optimal $\gamma$ shifts significantly downward as batch size increases from $BS=1$ to $BS=32$.
- **Side Effects of Vocabulary Pruning**: Common "vocab pruning" in SOTA drafters is nearly harmless on low-diversity data but exposes a collapse in acceptance rates for long-tail tokens on SPEED-Bench.

## Highlights & Insights
- **Formalizing "Evaluation Methodology" as a contribution**: Shifting focus from new algorithms to standardized evaluation ("what, how, and where to evaluate").
- **Sample Selection via Semantic Embedding + Greedy Exchange**: Minimizing pairwise similarity is a robust alternative to vague "coverage" concepts.
- **Analytical Speedup Formula**: Decoupling per-step latency from algorithmic AL allows for speedup prediction across domains using minimal metrics.

## Limitations & Future Work
- The Python client is limited by the GIL for $BS > 256$, suggesting a need for multi-processing or a Rust client.
- The Qualitative split is designed for evaluation, not for training drafters.
- Lack of coverage for security/privacy scenarios and speculative attacks.
- Discrete ISL buckets require interpolation for intermediate workloads.

## Related Work & Insights
- **vs. SpecBench (Xia et al., 2024)**: SPEED-Bench upgrades data sources, sampling, and production engine integration while maintaining backward compatibility.
- **vs. Drafter Papers (EAGLE3, Medusa, etc.)**: Provides the first unified platform to compare these disparate methods consistently.
- **vs. LongSpec / MagicDec**: The 32k ISL bucket provides a rigorous testing ground for long-context speculative decoding.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

## Rating
- Novelty: To be evaluated
- Experimental Thoroughness: To be evaluated
- Writing Quality: To be evaluated
- Value: To be evaluated

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding](lk_losses_direct_acceptance_rate_optimization_for_speculative_decoding.md)
- [\[ACL 2026\] SSSD: Simply-Scalable Speculative Decoding](../../ACL2026/model_compression/sssd_simply-scalable_speculative_decoding.md)
- [\[AAAI 2026\] Steering Pretrained Drafters during Speculative Decoding](../../AAAI2026/model_compression/steering_pretrained_drafters_during_speculative_decoding.md)
- [\[NeurIPS 2025\] Traversal Verification for Speculative Tree Decoding](../../NeurIPS2025/model_compression/traversal_verification_for_speculative_tree_decoding.md)
- [\[ACL 2026\] Calibrated Speculative Decoding: Frequency-Guided Candidate Selection for Efficient Inference](../../ACL2026/model_compression/calibrated_speculative_decoding_frequency-guided_candidate_selection_for_efficie.md)

</div>

<!-- RELATED:END -->
