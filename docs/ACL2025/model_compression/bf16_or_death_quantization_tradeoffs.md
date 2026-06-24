---
title: >-
  [Paper Note] "Give Me BF16 or Give Me Death"? Accuracy-Performance Trade-Offs in LLM Quantization
description: >-
  [ACL 2025][Model Compression][LLM quantization] This is the most comprehensive empirical study of LLM quantization to date, conducting over 500k evaluations of FP8/INT8/INT4 on the entire Llama-3.1 family (8B/70B/405B). It finds that FP8 is nearly lossless, INT8 incurs only a 1-3% drop, and INT4 is surprisingly competitive, while providing recommendations for selecting quantization formats in different deployment scenarios.
tags:
  - "ACL 2025"
  - "Model Compression"
  - "LLM quantization"
  - "FP8"
  - "INT8"
  - "INT4"
  - "inference benchmark"
  - "vLLM"
date: 2026-05-08
content_hash: 04763ce974aaade3
---

# "Give Me BF16 or Give Me Death"? Accuracy-Performance Trade-Offs in LLM Quantization

**Conference**: ACL 2025  
**arXiv**: [2411.02355](https://arxiv.org/abs/2411.02355)  
**Code**: None  
**Area**: Model Compression / LLM Efficiency  
**Keywords**: LLM quantization, FP8, INT8, INT4, inference benchmark, vLLM  

## TL;DR
This is the most comprehensive empirical study of LLM quantization to date, conducting over 500k evaluations of FP8/INT8/INT4 on the entire Llama-3.1 family (8B/70B/405B). It finds that FP8 is nearly lossless, INT8 incurs only a 1-3% drop, and INT4 is surprisingly competitive, while providing recommendations for selecting quantization formats in different deployment scenarios.

## Background & Motivation

**Background**: LLM quantization has become the most dominant inference acceleration technique. Main quantization formats include W8A8-FP (Hopper GPU), W8A8-INT (Ampere GPU), and W4A16-INT (4-bit weight). However, there is a lack of systematic benchmarking regarding the accuracy-performance trade-offs across different formats.

**Limitations of Prior Work**: (1) Prior research (e.g., Lee et al., 2024b) claimed that W8A8-INT performs significantly worse than FP8, leading to community misunderstandings (such as skepticism toward quantized 405B models); (2) most evaluations only use academic benchmarks, which do not reflect real-world deployment scenarios; (3) suboptimal hyperparameters in studies lead to misleading conclusions (e.g., that AWQ outperforms GPTQ); (4) there is a lack of comprehensive analysis that combines inference performance (latency/throughput).

**Key Challenge**: The community exhibits a "BF16 or nothing" bias regarding quantization—remaining uncertain whether quantization is good enough, which leads to wasteful deployments.

**Goal**: To provide a data-driven guide for selecting quantization formats: how much accuracy is actually lost under each format, and which format is optimal across different hardware/scenarios?

**Key Insight**: Building a comprehensive evaluation framework covering academic, real-world, and textual similarity benchmarks, and testing inference performance on 3 types of GPUs (A6000/A100/H100).

**Core Idea**: Answering "how much does quantization actually lose" through over 500k evaluations—the answer is: far less than generally expected.

## Method

### Overall Architecture
Rather than proposing a new methodology, this paper is a systematic empirical study. The framework covers: three quantization formats (W8A8-FP/W8A8-INT/W4A16-INT) $\times$ the entire Llama-3.1 family (8B/70B/405B) $\times$ multi-dimensional evaluations (academic benchmarks/real-world tasks/textual similarity/inference performance) $\times$ three GPU architectures.

### Key Designs

1. **Evaluation Framework**:

    - **Academic Benchmarks**: Open LLM Leaderboard V1 (GSM/MMLU/ARC/Winogrande/HellaSwag/TruthfulQA) + V2 (MMLU-Pro/GPQA/BBH/MuSR/MATH/IFEval)
    - **Real-world Benchmarks**: Arena-Hard-Auto (500 complex prompts), HumanEval/HumanEval+ (code generation), RULER (long context 4k-128k)
    - **Textual Similarity**: ROUGE-1/ROUGE-L/BERTScore/STS to analyze the semantic and structural consistency between the outputs of the quantized model and the original model under the same prompts.

2. **Quantization Algorithm Optimization**:

    - **W8A8-FP**: Dynamic per-token activation quantization + symmetric weight quantization, requiring no calibration data.
    - **W8A8-INT**: GPTQ symmetric weight quantization + dynamic per-token activation quantization + SmoothQuant (necessary for 70B).
    - **W4A16-INT**: GPTQ + MSE optimal clipping + group size of 128 + high-quality calibration data (OpenPlatypus).
    - **Correcting prior AWQ vs GPTQ misunderstandings**: GPTQ, when paired with MSE clipping and high-quality calibration data, outperforms AWQ on real-world tasks.

3. **Inference Performance Evaluation**:

    - 7 deployment scenarios: synchronous/asynchronous, varying concurrency levels.
    - vLLM framework tested across 3 GPU types: A6000/A100/H100.

## Key Experimental Results

### Main Results: Accuracy on Academic and Real-world Benchmarks

| Quantization Format | Llama-3.1-8B Leaderboard V1/V2 | Arena-Hard | HumanEval |
|---------|-------------------------------|-----------|-----------|
| BF16 | 74.06 / 27.62 | 25.8 | 67.3 |
| W8A8-FP | ≈BF16 (within error margin) | ≈BF16 | ≈BF16 |
| W8A8-INT | 73.x / 27.x (1-3%↓) | 24.x | 65.x |
| W4A16-INT(GPTQ) | 73.11 / 26.53 | 24.0 | 67.1 |
| W4A16-INT(AWQ) | 72.69 / 27.40 | 22.3 | 63.0 |

### Ablation Study: GPTQ vs AWQ

| Dimension | GPTQ | AWQ | Notes |
|---------|------|-----|------|
| Academic Benchmark Average | ≈AWQ | ≈GPTQ | Roughly on par |
| Arena-Hard | +1.7 | Baseline | GPTQ is better on real-world tasks |
| HumanEval | +4.1 | Baseline | GPTQ is significantly better for code generation |
| MBPP | +3.0 | Baseline | Re-verified on code generation |

### Key Findings
- **FP8 is nearly lossless**: Across all model scales and benchmarks, FP8 results are equivalent to BF16 within the evaluation error margin. It only requires RTN quantization with no calibration data needed.
- **INT8 is far better than previously reported**: The previously claimed drop of 10+% is actually only 1-3%; the key lies in the correct usage of SmoothQuant and high-quality calibration data.
- **INT4 is surprisingly good**: W4A16-INT is comparable to or even better than W8A8-INT in accuracy, while requiring only half the memory for weights.
- **GPTQ > AWQ on real-world tasks**: MSE optimal clipping + high-quality calibration are crucial, overturning the community assumption that AWQ is superior.
- **Textual Similarity**: The textual output of large models post-quantization remains almost identical to the original (BERTScore >0.95); smaller models exhibit moderate structural variation but preserve semantics.
- **Deployment Recommendations**: Use W4A16 for synchronous scenarios (lowest latency), W8A8 for asynchronous high-throughput scenarios (maximum throughput), and determine based on specific ratios for hybrid scenarios.

## Highlights & Insights
- **The lesson of "hyperparameters matter"**: Many previous studies' conclusions regarding quantization formats were actually due to poorly tuned hyperparameters (e.g., using default absmax instead of MSE clipping in GPTQ, or using C4 instead of high-quality data). This serves as a reminder to ensure strong baselines during comparative experiments.
- **Empirically-driven deployment guide**: Derived from over 500k actual evaluations rather than theoretical analysis, directly guiding industrial deployments.
- **Dismantling the "BF16 or nothing" myth**: Presenting a wealth of evidence showing that quantization is not intimidating—FP8 is lossless and INT8 is nearly lossless—lowering the psychological barrier for adopting quantization in the community.
- **Textual similarity analysis**: Going beyond traditional accuracy metrics, analyzing the structural and semantic consistency of generated text offers a new dimension for evaluating the impact of quantization.

## Limitations & Future Work
- Only the Llama-3.1 family was evaluated; findings might differ for other architectures (e.g., Qwen, Mistral).
- The FP8 conclusions are only applicable to Hopper-architecture GPUs; older GPUs cannot utilize this format.
- More extreme low-bit (2-bit/3-bit) quantization was not evaluated.
- Inference performance tests were based on a specific version of vLLM (0.6.4); framework optimizations might impact the conclusions.
- Fine-tuning scenarios for quantized models were not considered.

## Related Work & Insights
- **vs Lee et al. (2024b)**: The closest previous study, which claimed that W8A8-INT was significantly inferior to FP8. This paper narrows the gap from 10+ points to 0.7 points through correct hyperparameter tuning.
- **vs AWQ (Lin et al., 2024)**: AWQ is widely recommended in academic papers, but this study proves that GPTQ + MSE clipping + high-quality data is superior on real-world tasks.
- **vs SmoothQuant (Xiao et al., 2022)**: SmoothQuant addresses the problem of outliers in activation quantization, and this paper confirms that it is necessary for large models (e.g., 70B) in W8A8-INT.

## Rating
- Novelty: ⭐⭐⭐ Not a methodological innovation, but rather an empirical contribution with unprecedented comprehensiveness.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 500k evaluations, spanning academic, real-world, textual similarity, and inference performance dimensions across three GPU architectures.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, explicit findings, and practical deployment recommendations.
- Value: ⭐⭐⭐⭐⭐ Direct guiding value for industrial quantization deployment decisions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Navigating the Accuracy-Size Trade-Off with Flexible Model Merging](../../ICLR2026/model_compression/navigating_the_accuracy-size_trade-off_with_flexible_model_merging.md)
- [\[ACL 2025\] MoQAE: Mixed-Precision Quantization for Long-Context LLM Inference via Mixture of Quantization-Aware Experts](moqae_mixed_precision_kv_cache.md)
- [\[ACL 2025\] Compression in Transformer Language Models Has a Surprising Relationship with Performance](compression_in_transformer_language_models_has_a_surprising_relationship_with_pe.md)
- [\[ICLR 2026\] Multi-View Encoders for Performance Prediction in LLM-Based Agentic Workflows](../../ICLR2026/model_compression/multi-view_encoders_for_performance_prediction_in_llm-based_agentic_workflows.md)
- [\[ICML 2025\] FlatQuant: Flatness Matters for LLM Quantization](../../ICML2025/model_compression/flatquant_flatness_matters_for_llm_quantization.md)

</div>

<!-- RELATED:END -->
