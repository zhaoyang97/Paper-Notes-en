---
title: >-
  [Paper Note] EffiVLM-Bench: A Comprehensive Benchmark for Evaluating Training-Free Acceleration in Large Vision-Language Models
description: >-
  [ACL 2025][Multimodal Efficiency][VLM Acceleration] Proposes EffiVLM-Bench, a unified evaluation framework to systematically evaluate training-free acceleration methods (token compression + parameter compression) for LVLMs across four dimensions: performance, generalization, faithfulness, and efficiency. Spanning 3 cutting-edge models and 17 benchmark tasks, it reveals the Pareto-optimal trade-offs of various methods under different compression rates.
tags:
  - "ACL 2025"
  - "Multimodal Efficiency"
  - "VLM Acceleration"
  - "Training-Free Compression"
  - "Token Compression"
  - "Parameter Quantization"
  - "Pareto Optimality"
date: 2026-05-08
content_hash: adee340b18b7483c
---

# EffiVLM-Bench: A Comprehensive Benchmark for Evaluating Training-Free Acceleration in Large Vision-Language Models

**Conference**: ACL 2025  
**arXiv**: [2506.00479](https://arxiv.org/abs/2506.00479)  
**Code**: [Project Page](https://effivlm-bench.github.io/)  
**Area**: Multimodal VLMs / Model Acceleration  
**Keywords**: VLM Acceleration, Training-Free Compression, Token Compression, Parameter Quantization, Pareto Optimality

## TL;DR

Proposes EffiVLM-Bench, a unified evaluation framework to systematically evaluate training-free acceleration methods (token compression + parameter compression) for LVLMs across four dimensions: performance, generalization, faithfulness, and efficiency. Spanning 3 cutting-edge models and 17 benchmark tasks, it reveals the Pareto-optimal trade-offs of various methods under different compression rates.

## Background & Motivation

**Background**: Large Vision-Language Models (LVLMs) have achieved outstanding success in multimodal AI tasks. However, their immense computational and memory overhead severely hinders practical deployment. To enhance efficiency, researchers have explored training-free acceleration methods—which reduce inference costs without requiring retraining—categorized mainly into token compression (eliminating redundant tokens) and parameter compression (reducing parameter counts through pruning/quantization).

**Limitations of Prior Work**: Existing evaluations of acceleration methods suffer from three key deficiencies: (1) **Outdated model architectures**—evaluations are often confined to older models like LLaVA/LLaVA-v1.5, failing to consider the latest LVLMs with dynamic resolution processing mechanisms (e.g., Qwen2-VL, InternVL2.5); (2) **Limited benchmarks**—typically restricted to general VQA tasks, while overlooking more challenging tasks such as OCR and long-text generation; (3) **Single evaluation metric**—focusing solely on absolute performance (accuracy) while neglecting dimensions such as generalization and faithfulness, as well as lacking a systematic exploration of the performance-efficiency Pareto-optimal trade-offs.

**Key Challenge**: The lack of a unified evaluation framework to comprehensively understand the performance and trade-offs of different acceleration techniques across diverse scenarios, which hinders method selection and practical deployment.

**Goal** → To provide a unified, comprehensive, and scalable evaluation benchmark for the research area of training-free LVLM acceleration.

**Key Insight**: Define four orthogonal evaluation dimensions (performance, generalization, faithfulness, and efficiency) covering the latest model architectures and diverse task scenarios to systematically compare mainstream acceleration methods.

**Core Idea**: End the fragmented evaluation landscape of LVLM acceleration methods with a four-dimensional unified framework, providing a basis for method selection in practical deployment through Pareto frontier analysis.

## Method

### Overall Architecture

EffiVLM-Bench defines four core evaluation dimensions covering 17 benchmark tasks (from document understanding to mathematical reasoning), 3 cutting-edge LVLMs (LLaVA-OneVision-7B, Qwen2-VL-7B, InternVL2.5-38B), and two major classes of acceleration methods (token compression and parameter compression), providing a comprehensive comparison under different compression rates such as 1%, 10%, and 40%.

### Key Designs

1. **Four-Dimensional Evaluation Indicator System**:

    - Function: Comprehensively measures the quality of compression methods from four complementary perspectives.
    - Mechanism:
        - **Overall Performance (OP)**: $$OP^{m,c} = \sqrt{\frac{1}{B}\sum_{b=1}^{B}\mathbb{E}\left(\frac{EM_b^{m,c}}{EM_b^m}\right)^2}$$, which is the root-mean-square average of the ratio of evaluation metrics of the compressed model to the original model across benchmarks.
        - **Generalization (OG)**: The coefficient of variation of performance across benchmarks and models; a lower value indicates more stable behavior of the compression method.
        - **Faithfulness (OL)**: $$OL^c = \mathbb{E}_{b,m}[\mathbb{I}(P_b^{m,c}, P_b^m)]$$, measuring the consistency between the predictions of the compressed model and the original model to ensure the compression does not introduce new biases.
        - **Efficiency (OE)**: The speedup ratio based on actual inference time (rather than theoretical metrics like FLOPs), directly reflecting real-world latency.
    - Design Motivation: Relying solely on accuracy hides behavioral shifts introduced by compression—high accuracy but low faithfulness indicates that the model "gives the correct answer but for different reasons".

2. **Comprehensive Method Coverage and Compression Rate Gradients**:

    - Function: Compares the performance of the two major categories of methods across multiple compression rates under unified conditions.
    - Mechanism: **Token compression** includes token pruning (FastV dynamic pruning, VisionZip vision encoder pruning, PruMerge+ prune-then-merge) and KV cache compression (StreamingLLM sliding window attention, H2O heavy-hitter eviction, SnapKV/PyramidKV layered strategies, LOOK-M/VL-Cache multimodal-aware compression); **parameter compression** includes weight pruning (EcoFLAP, Wanda, SparseGPT) and quantization (AWQ int4, GPTQ int4). All methods are tested under 1%, 10%, 40%, and 100% token retention rates or corresponding parameter compression levels.
    - Design Motivation: The strengths and weaknesses of different methods may invert at different compression rates, requiring a full-spectrum comparison to identify the Pareto-optimal frontier.

### Loss & Training

EffiVLM-Bench evaluates **training-free methods**, thus involving no additional training processes. The evaluated methods achieve compression through the following strategies: token pruning selects tokens to keep based on attention scores or text-visual relevance; KV cache compression leverages attention sparsity to select fewer key-value pairs to reduce memory; weight pruning removes redundant parameters based on weight importance metrics (such as Wanda's weight-times-activation metric); quantization converts full-precision weights to low-precision formats such as int4/int8.

## Key Experimental Results

### Main Results (Token Compression Comparison on LLaVA-OneVision-7B)

| Method | Retention Rate | DocVQA | ChartQA | OCRBench | MMMU | MMBench | OP |
|------|--------|--------|---------|----------|------|---------|-----|
| Original | 100% | 87 | 80.00 | 595 | 45.44 | 83.12 | 1.00 |
| FastV | 40% | 80 | 69.20 | 488 | 46.33 | 81.22 | 0.94 |
| VisionZip | 40% | 72 | 67.04 | 500 | 46.11 | 80.43 | 0.93 |
| PruMerge+ | 40% | 49 | 51.40 | 382 | 45.55 | 80.88 | 0.88 |
| FastV | 10% | 48 | 43.16 | 190 | 45.33 | 70.12 | 0.76 |
| VisionZip | 10% | 56 | 49.88 | 352 | 44.11 | 78.86 | 0.84 |
| FastV | 1% | 8 | 14.00 | 27 | 40.89 | 27.69 | 0.48 |
| VisionZip | 1% | 35 | 35.16 | 194 | 42.56 | 74.10 | 0.75 |

### Ablation Study (Cross-Model Generalization Comparison, 40% Retention Rate)

| Method | LLaVA-OV-7B OP | Qwen2-VL-7B OP | InternVL2.5-38B OP |
|------|---------------|----------------|-------------------|
| FastV | 0.94 | 0.92 | 0.91 |
| VisionZip | 0.93 | 0.93 | 0.89 |
| PruMerge+ | 0.88 | 0.91 | 0.85 |

### Key Findings

- **Token compression performance is highly dependent on tasks and models**: Under a 40% retention rate, all three token pruning methods maintain an OP > 0.88, but under extreme 1% compression, the difference is massive—VisionZip still maintains OP = 0.75, whereas FastV plummets to 0.48.
- **Document understanding tasks are the most vulnerable**: DocVQA and ChartQA are highly sensitive to token compression (at 1%, FastV's DocVQA score drops from 87 to 8) since these tasks require precise, localized visual information.
- **Quantization is the most practical acceleration solution**: AWQ int4 quantization provides roughly $2\times$ speedup while preserving close-to-original performance (OP > 0.95), and is insensitive to task types.
- **Multimodal-aware strategies for KV cache compression outperform general LLM methods**: VL-Cache and LOOK-M outperform general-purpose cache compression methods directly ported from LLMs at the same compression rates by differentiating the distinct roles of visual and textual tokens.
- **Pareto frontier analysis**: In the performance-efficiency trade-off, token pruning methods dominate the Pareto frontier at low compression rates (40%+), whereas quantization methods are superior at high compression rates.

## Highlights & Insights

- The design concept of the **four-dimensional evaluation framework** is highly elegant—particularly the faithfulness (OL) metric, which covers a blind spot of traditional evaluation: a method might "guess new answers correctly" on some samples but "lose correct answers" on others, which cannot be identified by looking at accuracy alone.
- **Pareto frontier analysis** directly provides selection recommendations for practical deployment—allowing users to choose methods directly from the Pareto curve based on their latency budgets.
- The **comparison under 1% extreme compression** reveals the robustness floor of each method; VisionZip's advantage under extremely low budgets stems from its compression at the vision encoder level rather than the LLM level.
- The **modular design** supports future extensions to new models and methods.

## Limitations & Future Work

- Only three LVLMs are covered, leaving out larger-scale models (e.g., 72B/110B) or more architectural variants.
- The combined effects of multiple compression methods have not been evaluated—jointly applying token and parameter compression may yield synergistic or conflicting effects.
- Efficiency is measured solely by inference time, leaving out other equally important deployment metrics like memory footprint and throughput.
- Lacks fine-grained evaluation of generation quality—the quality of long-text generation scenarios in LVLMs (such as detailed descriptions) might degrade much earlier than the drop in accuracy.
- Does not cover the impact of recent attention architecture variants (e.g., GQA, MQA) on the effectiveness of compression methods.

## Related Work & Insights

- In token compression, LLaVA-PruMerge (Shang 2024) first introduced pruning and merging during the vision encoder stage, and VL-Cache (Tu 2024) introduced a modality-aware cache allocation strategy; in parameter compression, SparseGPT (Frantar 2023) and AWQ (Lin 2024) represent two mainstream pathways.
- The unified evaluation framework methodology of this paper can be generalized to other domains—such as LLM inference acceleration and speech model compression.
- **Insights**: (1) Practical deployment should prioritize AWQ int4 quantization as a baseline acceleration solution; (2) Multimodal-aware token compression strategies are critical differentiators in high-compression scenarios; (3) Evaluation frameworks should be multi-dimensional, as a single metric can easily mislead method selection.

## Rating

⭐⭐⭐⭐ Fills the gap of lacking a unified evaluation framework in the field of LVLM acceleration. The design of the four-dimensional metrics is reasonable, the experimental coverage is comprehensive, and the Pareto frontier analysis provides direct guiding value for practical deployment. However, model coverage is somewhat limited, and combined compression strategies have not been explored.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] CoreMatching: A Co-adaptive Sparse Inference Framework with Token and Neuron Pruning for Comprehensive Acceleration of Vision-Language Models](../../ICML2025/vlm_efficiency/corematching_a_co-adaptive_sparse_inference_framework_with_token_and_neuron_prun.md)
- [\[ICLR 2026\] VisionTrim: Unified Vision Token Compression for Training-Free MLLM Acceleration](../../ICLR2026/vlm_efficiency/visiontrim_unified_vision_token_compression_for_training-free_mllm_acceleration.md)
- [\[AAAI 2026\] Filter, Correlate, Compress: Training-Free Token Reduction for MLLM Acceleration](../../AAAI2026/vlm_efficiency/filter_correlate_compress_training-free_token_reduction_for_.md)
- [\[ICML 2026\] CLIP Tricks You: Training-free Token Pruning for Efficient Pixel Grounding in Large Vision-Language Models](../../ICML2026/vlm_efficiency/clip_tricks_you_training-free_token_pruning_for_efficient_pixel_grounding_in_lar.md)
- [\[ACL 2025\] Hierarchical Safety Realignment: Lightweight Restoration of Safety in Pruned Large Vision-Language Models](hierarchical_safety_realignment_lightweight_restoration_of_safety_in_pruned_larg.md)

</div>

<!-- RELATED:END -->
