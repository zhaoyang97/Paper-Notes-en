---
title: >-
  [Paper Note] Unlocking the Edge: Multi-LoRA On-Device Deployment and Acceleration
description: >-
  [ACL 2026][Multilingual & Machine Translation][On-device LLM deployment] Ours proposes an on-device LLM deployment framework for Samsung Galaxy S24/S25. It enables dynamic task switching by using LoRA weights as runtime…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "On-device LLM deployment"
  - "Multi-LoRA"
  - "Speculative decoding"
  - "Concurrent token generation"
  - "INT4 quantization"
date: 2026-05-08
content_hash: 470af43470d3dd45
---

# Unlocking the Edge: Multi-LoRA On-Device Deployment and Acceleration

**Conference**: ACL 2026  
**arXiv**: [2604.18655](https://arxiv.org/abs/2604.18655)  
**Code**: None (Samsung internal system)  
**Area**: Multilingual Translation  
**Keywords**: On-device LLM deployment, Multi-LoRA, Speculative decoding, Concurrent token generation, INT4 quantization

## TL;DR

Ours proposes an on-device LLM deployment framework for Samsung Galaxy S24/S25. It enables dynamic task switching by using LoRA weights as runtime inputs, reduces latency for style variants by 6x through multi-stream concurrent token generation, and accelerates decoding by 2.3x via Dynamic Self-Speculative Decoding without a draft model, achieving relative 4-6x overall optimization across 9 languages and 8 tasks.

## Background & Motivation

**Background**: Deploying LLMs on mobile devices provides privacy, low latency, and offline capabilities but faces strict constraints regarding memory, latency, and runtime flexibility. LoRA is a mainstream efficient fine-tuning method, but traditional practices involve static weight merging, which prevents dynamic task switching on-device.

**Limitations of Prior Work**: (1) While servers can flexibly load/switch LoRA, devices must use frozen inference graphs that cannot be recompiled or dynamically loaded; (2) Generating style variants (e.g., simultaneous formal/polite/humorous replies) requires 8 sequential runs, resulting in 8x latency; (3) Auto-regressive decoding is the primary bottleneck for on-device latency, and existing speculative decoding requires additional draft models that consume scarce memory.

**Key Challenge**: Flexible model development (multi-LoRA/multi-use case) vs. the immutability of on-device inference graphs—a fundamental redesign of the engineering architecture for adaptable on-device LLMs is required.

**Goal**: To achieve real-time, multilingual, multi-use case LLM inference on Qualcomm NPUs of commercial smartphones (Galaxy S24/S25).

**Key Insight**: Innovations at three levels—(1) LoRA weights as runtime inputs for frozen graphs; (2) Multi-stream token generation utilizing shared KV-cache; (3) Draft-model-free speculative decoding based on prefix tuning.

**Core Idea**: Encapsulate all use-case-specific knowledge into lightweight LoRA weights as external inputs to the inference graph. Combined with concurrent decoding and self-speculation strategies, this enables multi-use case on-device LLMs on a single frozen model.

## Method

### Overall Architecture

Based on LLaMA models with 1B/3B parameters, deployed on Qualcomm SM8650/SM8750 NPUs after INT4 quantization. Among three Multi-LoRA candidates, the "LoRA weights as input" scheme was selected (optimal latency and memory). Acceleration is achieved through Concurrent Token Generation (CTG) and Dynamic Self-Speculative Decoding (DS2D).

### Key Designs

1.  **LoRA Weights as Runtime Inputs**:

    - **Function**: Enables plug-and-play switching of any LoRA on a single frozen inference graph.
    - **Mechanism**: A base LLM graph is created with placeholders for LoRA weights. During inference, LoRA weights are passed as inputs alongside tokens. All LoRAs must have the same dimension to allow runtime replacement via the placeholder mechanism.
    - **Design Motivation**: Comparing three schemes—multi-graph with shared weights (good memory but requires graph switching), single-graph multi-LoRA with masking (good latency but large memory), and weights-as-input (best of both)—the third scheme offers superior scalability for multi-use case scenarios.

2.  **Concurrent Token Generation (CTG)**:

    - **Function**: Compresses the generation of 8 style variants from 8 sequential runs into a single forward pass.
    - **Mechanism**: Utilizes the shared frozen graph and KV-cache across all use cases. By modifying the masking scheme for the initial token sampling, 8 different output streams are generated simultaneously. Since stylistic differences are typically driven by the first token, branching only needs to occur during the first token sampling.
    - **Design Motivation**: Applications like Smart-Reply require providing 8 options simultaneously. CTG achieves a 6x reduction in latency and memory without modifying model binaries or inference graphs.

3.  **Dynamic Self-Speculative Decoding (DS2D)**:

    - **Function**: Accelerates decoding via semi-autoregressive methods without an auxiliary draft model.
    - **Mechanism**: Adds $m$ forecast embeddings to the LLM via prefix tuning, enabling the model to predict $1+m$ tokens in a single step (the first is from the frozen LLM distribution to ensure consistency, while the others are low-fidelity drafts requiring verification). A tree-based branching structure expands candidates, choosing an optimal branch configuration to match hardware-friendly input sizes (multiples of 32).
    - **Design Motivation**: Traditional speculative decoding draft models require extra memory (scarce on-device), whereas prefix-tuning-based methods add negligible parameters and remain fully compatible with frozen graphs.

### Loss & Training

Details for base model and LoRA training are not disclosed (Samsung proprietary). DS2D forecast embeddings are trained via prefix tuning. INT4 quantization utilizes Quantization-Aware Training (QAT) with a mixed-precision strategy.

## Key Experimental Results

### Main Results

**3B LLM Performance on GS25 Ultra**

| Use Case | Decoding Time w/o DS2D (ms) | Decoding Time w/ DS2D (ms) | Speedup |
| :--- | :--- | :--- | :--- |
| Correction | 50.17 | 22.30 | 2.25x |
| Composer | 53.23 | 28.57 | 1.86x |
| Style | 50.42 | 25.21 | 2.00x |

### Ablation Study

**CTG Latency Analysis (1B Model)**

| Streams | Prefill (ms) | AR (ms) | Total Time (ms) | Formula |
| :--- | :--- | :--- | :--- | :--- |
| 1 stream × 8 | 40 | 23 | 174 | $(23 \times 8) + 40$ |
| 8 streams concurrent | 40 | 23 | 63 | $23 + 40$ |

### Key Findings

- The LoRA-as-input scheme outperforms the other two in both memory and latency—peak memory for the 3B model is only 2.5GB.
- CTG compresses 8-stream generation from 174ms to 63ms (2.76x Gain) without any model modifications.
- DS2D achieves 1.86-2.25x decoding speedup across different use cases.
- Task accuracy across 6 languages remains at 90%+ after INT4 quantization.

## Highlights & Insights

- The design of LoRA weights as runtime inputs is elegant and efficient—shifting adaptability from "compile-time" to "runtime" provides a reference for any frozen-graph deployment scenario.
- The insight that "stylistic differences are usually driven by the first token" used in CTG is highly practical—in real products, different style replies indeed diverge at the beginning.
- This is a rare paper that fully describes on-device LLM deployment from an engineering implementation perspective, offering direct reference value for the industry.

## Limitations & Future Work

- Validated only on Samsung's own hardware and proprietary models.
- Training details for base models and LoRAs are not disclosed, limiting reproducibility.
- Tree-branch searching in DS2D increases engineering complexity.
- Lacks direct comparisons with other on-device LLM solutions (e.g., MLC-LLM, llama.cpp).

## Related Work & Insights

- **vs. QLoRA**: Focuses on training-time quantization, while Ours focuses on dynamic LoRA switching during deployment.
- **vs. Medusa/Eagle**: Requires additional speculative heads or draft models, whereas DS2D only requires lightweight prefix tuning.
- **vs. MobiLlama**: Focuses on model efficiency, while Ours focuses on full-stack engineering optimization for on-device deployment.

## Rating

- **Novelty**: ⭐⭐⭐⭐ LoRA-as-input and CTG designs are novel; DS2D improves upon existing methods.
- **Experimental Thoroughness**: ⭐⭐⭐ Real-world performance data on commercial devices is valuable, but lacks comparison with external methods.
- **Writing Quality**: ⭐⭐⭐ Rich in engineering details, but academic writing could be improved.
- **Value**: ⭐⭐⭐⭐ Directly useful for on-device LLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] RouteLMT: Learned Sample Routing for Hybrid LLM Translation Deployment](routelmt_learned_sample_routing_for_hybrid_llm_translation_deployment.md)
- [\[ACL 2026\] FairQE: Multi-Agent Framework for Mitigating Gender Bias in Translation Quality Estimation](fairqe_multi-agent_framework_for_mitigating_gender_bias_in_translation_quality_e.md)
- [\[ACL 2026\] Efficient Low-Resource Language Adaptation via Multi-Source Dynamic Logit Fusion](efficient_low-resource_language_adaptation_via_multi-source_dynamic_logit_fusion.md)
- [\[ACL 2026\] Alexandria: A Multi-Domain Dialectal Arabic Machine Translation Dataset for Culturally Inclusive and Linguistically Diverse LLMs](alexandria_a_multi-domain_dialectal_arabic_machine_translation_dataset_for_cultu.md)
- [\[ACL 2026\] TransLaw: A Large-Scale Dataset and Multi-Agent Benchmark Simulating Professional Translation of Hong Kong Case Law](translaw_a_large-scale_dataset_and_multi-agent_benchmark_simulating_professional.md)

</div>

<!-- RELATED:END -->
