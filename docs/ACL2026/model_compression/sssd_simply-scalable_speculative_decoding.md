---
title: >-
  [Paper Note] SSSD: Simply-Scalable Speculative Decoding
description: >-
  [ACL2026][Model Compression][Speculative decoding] Ours proposes SSSD, a training-free speculative decoding method that combines lightweight n-gram matching with hardware-aware speculative length adjustment. It achieves…
tags:
  - "ACL2026"
  - "Model Compression"
  - "Speculative decoding"
  - "LLM inference acceleration"
  - "n-gram matching"
  - "training-free"
  - "hardware-aware"
date: 2026-05-08
content_hash: 708df73b660de150
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# SSSD: Simply-Scalable Speculative Decoding

**Conference**: ACL2026
**arXiv**: [2411.05894](https://arxiv.org/abs/2411.05894)
**Code**: [GitHub](https://github.com/huawei-csl/sssd_speculator)
**Area**: model_compression
**Keywords**: Speculative decoding, LLM inference acceleration, n-gram matching, training-free, hardware-aware

## TL;DR

Ours proposes SSSD, a training-free speculative decoding method that combines lightweight n-gram matching with hardware-aware speculative length adjustment. It achieves up to 2.9× inference acceleration without requiring any draft model training or deployment, demonstrating superior robustness over training-based methods in language/domain transfer and long-context scenarios.

## Background & Motivation

Speculative Decoding (SD) is a popular technique for accelerating LLM inference, yet existing methods face two primary deployment bottlenecks: (1) Training-based methods (e.g., EAGLE, Medusa) require training and deploying additional draft models, necessitating retraining when the target model or application domain changes, which increases maintenance complexity; (2) existing model-free methods (e.g., Prompt Lookup Decoding, REST) have limited speculation quality and are only effective for specific tasks. In real production systems, ease of integration and cost efficiency are as critical as raw latency improvements. Furthermore, prior work shows that training-based draft models generalize poorly to non-English languages, leading to unfairness in inference speed. SSSD aims to fill the gap of "high performance and easy deployment."

## Method

### Overall Architecture

SSSD treats the prompt and self-generated text as a unified n-gram source and integrates it with a large text datastore. The prefix of the final token is matched across both sources, and the retrieved continuations are used to construct a tree-structured candidate set for verification. The entire draft generation process runs on the CPU and does not consume GPU resources.

### Key Designs

1.  **Dual-Source Candidate Fusion**: It fuses candidates from the prompt/self-output (stored in a trie structure, updated in real-time) and an external large datastore (built on a suffix array). The two sources provide complementary candidates—the prompt source is sensitive to the current context, while the datastore source covers broader linguistic patterns. Weighted tree merging is performed, with decay calibration applied to avoid over-confidence in the prompt source.
2.  **Datastore Management**: A multi-sub-index architecture is adopted (each sub-index contains 512M tokens, $\approx$ 4GB), supporting asynchronous reconstruction and an oldest-replacement policy. Logarithmic search based on suffix arrays keeps retrieval latency low and nearly independent of datastore size. It supports cold starts—achieving 1.1–1.23× acceleration even starting from an empty datastore.
3.  **Hardware-Aware Speculative Length**: Based on the Roofline model, the optimal speculative length is dynamically selected as $s_q = I_{\text{knee}} / b$ (where $I_{\text{knee}}$ is the peak FLOPS/bandwidth ratio and $b$ is the batch size). SSSD uses a larger $s_q$ (more speculation) for small batches and reduces $s_q$ for large batches, automatically adapting to hardware resource utilization.

### Loss & Training

SSSD is entirely training-free. Weight fusion parameters for the n-gram models are determined via a simple grid search and are transferable across different models and tasks. It supports both greedy decoding and speculative sampling modes.

## Key Experimental Results

### Main Results

End-to-end evaluation on Llama-3.1-8B integrated into SGLang:

| Method | MT-Bench | MATH-500 | HumanEval | MT-Bench (German) | Need Training |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Autoregressive | 1.00× | 1.00× | 1.00× | 1.00× | No |
| EAGLE-2 | ~1.6× | ~1.5× | ~1.5× | ~1.3× | Yes |
| EAGLE-3 | ~1.8× | ~1.6× | ~1.7× | ~1.4× | Yes |
| PLD | ~1.2× | ~1.1× | ~1.2× | ~1.1× | No |
| REST | ~1.4× | ~1.3× | ~1.3× | ~1.2× | No |
| **SSSD** | **~1.7×** | **~1.8×** | **~1.6×** | **~1.6×** | **No** |

Long context (PG-19, 40k tokens):

| Method | Llama-3.1-8B | Llama-3.3-70B |
| :--- | :--- | :--- |
| Lookahead | 1.08× | 1.15× |
| EAGLE-3 | 0.80× | 1.09× |
| **SSSD** | **1.23×** | **1.26×** |

DeepSeek-R1-Distill-Llama-8B (MATH-500): SSSD reaches 2.29× speedup ($batch=1$) and is the only method to maintain acceleration under large batch sizes.

### Ablation Study

| Content | Key Findings |
| :--- | :--- |
| Candidate Source Decomposition | Contributions from the prompt source and datastore source overlay almost perfectly, verifying complementarity. |
| Datastore Cold Start | An empty datastore yields 1.1–1.23× speedup; after 1000 dialogues, it reaches 1.6–1.8×. |
| Model Generation vs. Datasets | The correct token prediction rate for model-generated data is 35% higher. |
| Cross-lingual Adaptation | English datastores provide initial gains for new languages, converging to monolingual performance as data accumulates. |
| Hardware-aware $s_q$ Tuning | Automatically adapts to different batch sizes without manual tuning. |

### Key Findings

- SSSD consistently outperforms all n-gram methods and EAGLE-2 across all tests, approaching or exceeding EAGLE-3 in most scenarios.
- Acceleration is more significant for non-English languages (1.6–1.8× vs. 1.4× in English), helping mitigate cross-lingual latency inequality caused by tokenizers.
- SSSD demonstrates greater advantages in long-context and agentic workloads (up to 1.9× throughput improvement) because its drafting cost is independent of context length.

## Highlights & Insights

- **Deployment Friendliness**: Zero training, zero extra GPU memory, and zero model alignment requirements. Drafts run on the CPU, making it plug-and-play for existing serving systems.
- **Cold Start Capability**: Acceleration begins from an empty datastore and adaptively improves with usage, fitting progressive scenarios in real deployments.
- **System-Algorithm Co-optimization**: Treating inference acceleration as a joint algorithm-system problem; the choice of $s_q$ guided by the Roofline model is an elegant engineering insight.
- **Cross-Lingual Fairness**: Since no language-specific training is required, it provides greater speedup for low-resource languages, reducing language bias in inference efficiency.

## Limitations & Future Work

- Gains are limited on MoE models because increasing speculative length activates more experts, increasing memory loading overhead.
- MLA attention (e.g., DeepSeek-V2) uses compute-intensive FlashMLA kernels, which are less compatible with speculative decoding.
- The drafting stage relies on CPU performance and host memory, which may be restricted in CPU-constrained deployment environments.
- Using historical output as a candidate source raises privacy considerations, although it does not introduce additional information leakage.

## Related Work & Insights

- **EAGLE / EAGLE-3** (Li et al., 2024/2025): Training-based speculative heads; high performance but high deployment complexity.
- **REST** (He et al., 2024): Retrieval-based speculation using external corpora; SSSD improves upon its datastore design (interval sampling, no pruning, multi-sub-indices).
- **Prompt Lookup Decoding** (Saxena, 2023): Simple n-gram matching using only the prompt; SSSD significantly improves candidate quality by incorporating a datastore.
- Insight: In the field of inference acceleration, "good enough + minimal deployment" is often of more practical value than "optimal but complex."

## Rating

| Dimension | Score (1-10) |
| :--- | :--- |
| Novelty | 7 |
| Experimental Thoroughness | 9 |
| Writing Quality | 8 |
| Value | 9 |
| Total Score | 8.3 |

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Steering Pretrained Drafters during Speculative Decoding](../../AAAI2026/model_compression/steering_pretrained_drafters_during_speculative_decoding.md)
- [\[ACL 2026\] Calibrated Speculative Decoding: Frequency-Guided Candidate Selection for Efficient Inference](calibrated_speculative_decoding_frequency-guided_candidate_selection_for_efficie.md)
- [\[ICML 2026\] SPEED-Bench: A Unified and Diverse Benchmark for Speculative Decoding](../../ICML2026/model_compression/speed-bench_a_unified_and_diverse_benchmark_for_speculative_decoding.md)
- [\[NeurIPS 2025\] Traversal Verification for Speculative Tree Decoding](../../NeurIPS2025/model_compression/traversal_verification_for_speculative_tree_decoding.md)
- [\[ICML 2026\] LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding](../../ICML2026/model_compression/lk_losses_direct_acceptance_rate_optimization_for_speculative_decoding.md)

</div>

<!-- RELATED:END -->
