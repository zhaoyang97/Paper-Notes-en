---
title: >-
  [Paper Note] Unlocking the Edge: Multi-LoRA On-Device Deployment and Acceleration
description: >-
  [ACL 2026][On-device LLM deployment] This paper presents an on-device LLM deployment framework for Samsung Galaxy S24/S25, achieving dynamic task switching by treating LoRA weights as runtime inputs, reducing style-variant generation latency by 6× via multi-stream concurrent token generation, and accelerating decoding by 2.3× through draft-model-free Dynamic Self-Speculative Decoding—yielding an overall 4–6× optimization across 9 languages and 8 tasks.
tags:
  - ACL 2026
  - On-device LLM deployment
  - Multi-LoRA
  - speculative decoding
  - concurrent token generation
  - INT4 quantization
date: 2026-05-08
content_hash: 0c8cba620180301a
---

# Unlocking the Edge: Multi-LoRA On-Device Deployment and Acceleration

**Conference**: ACL 2026
**arXiv**: [2604.18655](https://arxiv.org/abs/2604.18655)
**Code**: None (Samsung internal system)
**Area**: Multilingual Translation
**Keywords**: On-device LLM deployment, Multi-LoRA, speculative decoding, concurrent token generation, INT4 quantization

## TL;DR

This paper presents an on-device LLM deployment framework for Samsung Galaxy S24/S25, achieving dynamic task switching by treating LoRA weights as runtime inputs, reducing style-variant generation latency by 6× via multi-stream concurrent token generation, and accelerating decoding by 2.3× through draft-model-free Dynamic Self-Speculative Decoding—yielding an overall 4–6× optimization across 9 languages and 8 tasks.

## Background & Motivation

**Background**: Deploying LLMs on mobile devices offers privacy, low latency, and offline capabilities, but is subject to strict constraints on memory, latency, and runtime flexibility. LoRA is the dominant parameter-efficient fine-tuning method; however, the conventional approach of statically merging weights precludes dynamic task switching on-device.

**Limitations of Prior Work**: (1) Server-side deployments support flexible LoRA loading and switching, whereas on-device inference requires frozen computation graphs that cannot be recompiled or dynamically updated. (2) Style-variant generation (e.g., simultaneously producing formal, polite, and humorous replies) requires eight sequential forward passes, incurring 8× latency. (3) Autoregressive token-by-token decoding is the primary latency bottleneck on-device, and existing speculative decoding approaches require additional draft models that compete for scarce memory.

**Key Challenge**: Flexible model development (multi-LoRA, multi-task) versus the immutability of on-device inference graphs—necessitating a fundamental re-engineering of the on-device adaptable LLM architecture.

**Goal**: Achieve real-time, multilingual, multi-task LLM inference on the Qualcomm NPU of commercial smartphones (Galaxy S24/S25).

**Key Insight**: Three levels of innovation—(1) LoRA weights as runtime inputs to a frozen graph; (2) multi-stream token generation with shared KV-cache; (3) prefix-tuning-based draft-model-free speculative decoding.

**Core Idea**: Encapsulate all task-specific knowledge in lightweight LoRA weights supplied as external inputs to the inference graph, and combine concurrent decoding with self-speculative strategies to support multiple tasks on a single frozen on-device model.

## Method

### Overall Architecture

LLaMA models with 1B/3B parameters are quantized to INT4 and deployed on Qualcomm SM8650/SM8750 NPUs. Among three Multi-LoRA schemes, the "LoRA weights as input" approach is selected for its optimal latency and memory profile, complemented by CTG (Concurrent Token Generation) and DS2D (Dynamic Self-Speculative Decoding).

### Key Designs

1. **LoRA Weights as Runtime Inputs**:

    - **Function**: Enables plug-and-play switching of arbitrary LoRA adapters on a single frozen inference graph.
    - **Mechanism**: A base LLM graph with LoRA weight placeholders is constructed; at inference time, LoRA weights are passed alongside tokens as inputs. All LoRA adapters must share the same dimensions, and runtime replacement is achieved via the placeholder mechanism.
    - **Design Motivation**: Three schemes are compared—multiple graphs with shared weights (good memory, requires graph switching), a single graph with multiple LoRAs and masks (good latency, high memory), and weights as inputs (optimal on both dimensions). The third scheme offers the best scalability for multi-task scenarios.

2. **Concurrent Token Generation (CTG)**:

    - **Function**: Compresses 8-style-variant generation from 8 sequential forward passes into a single forward pass.
    - **Mechanism**: All tasks share the frozen graph and KV-cache; 8 distinct output streams are generated simultaneously by modifying the mask scheme applied at the first token sampling step. Since stylistic differences are typically determined by the first token, divergence only needs to occur at that point.
    - **Design Motivation**: Applications such as Smart Reply require presenting 8 response options simultaneously. CTG achieves a 6× reduction in latency and memory without any modification to the model binary or inference graph.

3. **Dynamic Self-Speculative Decoding (DS2D)**:

    - **Function**: Enables semi-autoregressive decoding acceleration without an additional draft model.
    - **Mechanism**: Prefix tuning augments the LLM with $m$ forecast embeddings, enabling the model to predict $1+m$ tokens in a single step (the first token from the frozen LLM distribution ensures consistency; the remaining tokens are low-fidelity drafts requiring verification). A tree-branching structure expands candidate sequences, and the optimal branch configuration is selected to match hardware-friendly input sizes (multiples of 32).
    - **Design Motivation**: Conventional speculative decoding requires a draft model that consumes additional on-device memory. The prefix-tuning-based approach introduces negligible additional parameters and is fully compatible with the frozen inference graph.

### Loss & Training

Training details for the base model and LoRA adapters are not disclosed (Samsung proprietary). DS2D forecast embeddings are trained via prefix tuning. INT4 quantization employs QAT (Quantization-Aware Training) with a mixed-precision strategy.

## Key Experimental Results

### Main Results

**3B LLM Performance on Galaxy S25 Ultra**

| Task | Decoding Time w/o DS2D (ms) | Decoding Time w/ DS2D (ms) | Speedup |
|------|-----------------------------|----------------------------|---------|
| Correction | 50.17 | 22.30 | 2.25× |
| Composer | 53.23 | 28.57 | 1.86× |
| Style | 50.42 | 25.21 | 2.00× |

### Ablation Study

**CTG Latency Analysis (1B Model)**

| Streams | Prefill (ms) | AR (ms) | Total (ms) | Formula |
|---------|-------------|---------|-----------|---------|
| 1 stream × 8 | 40 | 23 | 174 | (23×8)+40 |
| 8 streams concurrent | 40 | 23 | 63 | 23+40 |

### Key Findings

- The LoRA-as-input scheme outperforms the other two approaches on both memory and latency—peak memory for the 3B model is only 2.5 GB.
- CTG reduces 8-stream generation from 174 ms to 63 ms (2.76× speedup) without any model modification.
- DS2D achieves 1.86–2.25× decoding speedup across different tasks.
- Task accuracy across 6 languages remains above 90% after INT4 quantization.

## Highlights & Insights

- The LoRA-as-runtime-input design is both elegant and effective—it shifts adaptability from compile time to runtime, a principle broadly applicable to any frozen-graph deployment scenario.
- The CTG insight that "stylistic differences are typically driven by the first token" is practically well-motivated; in real products, stylistically distinct replies do indeed diverge at their opening tokens.
- This is one of the rare papers to provide a complete engineering-level account of on-device LLM deployment, offering direct reference value for industry practitioners.

## Limitations & Future Work

- Validation is limited to Samsung proprietary hardware and models.
- Training details for the base model and LoRA adapters are not disclosed, limiting reproducibility.
- The tree-branch search in DS2D introduces additional engineering complexity.
- No direct comparison is made against other on-device LLM frameworks (e.g., MLC-LLM, llama.cpp).

## Related Work & Insights

- **vs. QLoRA**: QLoRA focuses on quantization during training; this work addresses dynamic LoRA switching at deployment time.
- **vs. Medusa/Eagle**: These methods require additional speculative heads or draft models; DS2D relies only on lightweight prefix tuning.
- **vs. MobiLlama**: MobiLlama targets model efficiency; this work addresses the full engineering stack for on-device deployment optimization.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The LoRA-as-input and CTG designs are novel; DS2D is an improvement upon existing methods.
- **Experimental Thoroughness**: ⭐⭐⭐ Real-device performance data on commercial hardware is valuable, but comparisons with external methods are absent.
- **Writing Quality**: ⭐⭐⭐ Engineering details are rich, though academic writing could be improved.
- **Value**: ⭐⭐⭐⭐ Directly applicable practical value for on-device LLM deployment.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] MERIT: Multilingual Semantic Retrieval with Interleaved Multi-Condition Query](../../NeurIPS2025/multilingual_mt/merit_multilingual_semantic_retrieval_with_interleaved_multi-condition_query.md)
- [\[AAAI 2026\] Consensus-Aligned Neuron Efficient Fine-Tuning Large Language Models for Multi-Domain Machine Translation](../../AAAI2026/multilingual_mt/consensus-aligned_neuron_efficient_fine-tuning_large_language_models_for_multi-d.md)
- [\[CVPR 2026\] MMTIT-Bench: A Multilingual and Multi-Scenario Benchmark with Cognition-Perception-Reasoning Guided Text-Image Machine Translation](../../CVPR2026/multilingual_mt/mmtit-bench_a_multilingual_and_multi-scenario_benchmark_with_cognition-perceptio.md)
- [\[ACL 2026\] A Multilingual Dataset and Empirical Validation for the Mutual Reinforcement Effect in Information Extraction](a_multilingual_dataset_and_empirical_validation_for_the_mutual_reinforcement_eff.md)
- [\[ACL 2026\] Mitigating Extrinsic Gender Bias for Bangla Classification Tasks](mitigating_extrinsic_gender_bias_for_bangla_classification_tasks.md)

<!-- RELATED:END -->
