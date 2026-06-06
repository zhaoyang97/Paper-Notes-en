---
title: >-
  [Paper Note] From Static Inference to Dynamic Interaction: A Survey of Streaming Large Language Models
description: >-
  [ACL 2026][LLM/NLP][Streaming LLMs] This paper presents the first systematic survey of Streaming Large Language Models (Streaming LLMs). It proposes a unified definition based on data flow and interaction concurrency…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "Streaming LLMs"
  - "Real-time Interaction"
  - "Incremental Encoding"
  - "Full-duplex"
  - "Speculative Decoding"
date: 2026-05-08
content_hash: c13228efc3db96b9
---

# From Static Inference to Dynamic Interaction: A Survey of Streaming Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2603.04592](https://arxiv.org/abs/2603.04592)  
**Code**: [GitHub](https://github.com/EIT-NLP/Awesome-Streaming-LLMs)  
**Area**: LLM Systems / Streaming Inference  
**Keywords**: Streaming LLMs, Real-time Interaction, Incremental Encoding, Full-duplex, Speculative Decoding

## TL;DR

This paper presents the first systematic survey of Streaming Large Language Models (Streaming LLMs). It proposes a unified definition based on data flow and interaction concurrency, categorizing existing methods into a three-stage progressive hierarchy: Output-streaming, Sequential-streaming, and Concurrent-streaming, covering methodologies and applications across text, speech, and video modalities.

## Background & Motivation

**Background**: Standard LLMs adopt a "one-time full-input reading" static inference paradigm—encoding the complete input into KV caches before autoregressive decoding. While effective for benchmarks, this fundamentally limits applicability in dynamic scenarios such as real-time translation, streaming video understanding, and interactive tool agents.

**Limitations of Prior Work**: (1) Information in the real world arrives incrementally, accumulates over time, and may be unbounded in length, which the static paradigm cannot handle; (2) Definitions of "Streaming LLM" are currently fragmented—concepts like autoregressive decoding, incremental/chunked encoding, and full-duplex interaction are often conflated under the same label; (3) There is a lack of large-scale pre-training data that supports real-time interaction, partial input supervision, and fine-grained temporal alignment.

**Key Challenge**: There is a fundamental mismatch between the offline, all-context design of LLMs and the online, incremental, interactive data streams of the real world—models need to dynamically decide when to respond, when to wait for more information, and when to terminate.

**Goal**: To propose a unified definition and systematic taxonomy of Streaming LLMs, clarify terminological ambiguities, and provide a structured research roadmap for this emerging field.

**Key Insight**: Streaming LLMs are defined along two dimensions—data flow and interaction concurrency—rather than being categorized solely by modality or architecture.

**Core Idea**: Streaming LLMs constitute a three-stage progressive hierarchy: (1) Output-streaming: static input + streaming output (standard autoregressive); (2) Sequential-streaming: streaming input + streaming output (generation after incremental encoding); (3) Concurrent-streaming: simultaneous streaming input and output (full-duplex interaction), with each level introducing new challenges atop the previous one.

## Method

### Overall Architecture

The survey is structured around the three-stage taxonomy: Output-streaming focuses on streaming generation mechanisms and efficiency; Sequential-streaming focuses on incremental encoding and context management; Concurrent-streaming additionally introduces architectural adaptation and interaction strategies. Specific methods for text, speech, and video modalities are discussed within each stage.

### Key Designs

1.  **Output-Streaming LLMs**:
    *   **Function**: Achieve efficient streaming output based on static inputs.
    *   **Mechanism**: Three types of generation mechanisms—(a) Token-level: standard autoregressive (GPT, LlamaGen); (b) Block-level: semi-autoregressive (SoT parallel token generation) and block diffusion (SSD-LM); (c) Refined: multi-scale (VAR from coarse to fine) and global diffusion (LLaDA iterative denoising). Efficient technologies include speculative decoding (small draft model + large verification model), layer skipping (AdaInfer dynamic skipping of redundant layers), and memory optimization (StreamingLLM's attention sink mechanism to retain initial token KV caches).
    *   **Design Motivation**: The low efficiency of token-by-token autoregressive generation is the main bottleneck for streaming applications; throughput must be increased while maintaining quality.

2.  **Sequential-Streaming LLMs**:
    *   **Function**: Process incrementally arriving input streams and generate output at appropriate timings.
    *   **Mechanism**: Two core challenges—(a) Incremental encoding: segmenting continuous inputs into "atomic encoding" (existing discrete units like subwords, ViT patches) and "fragment encoding" (requiring segmentation, such as fixed-interval or semantic-driven CTC/DiSeg); (b) Context management: KV cache compression (token eviction like H2O, quantization like KVQuant, merging like CaM), retrieval-augmented memory (MemWalker hierarchical navigation of long context), and State Space Models (Mamba’s linear complexity suited for streaming input).
    *   **Design Motivation**: Streaming input implies the full context is unavailable during encoding, requiring decisions under incomplete information; the linear growth of KV caches for long sequences makes memory consumption infeasible.

3.  **Concurrent-Streaming LLMs**:
    *   **Function**: Achieve full-duplex interaction with simultaneous input and output.
    *   **Mechanism**: Two core challenges—(a) Architectural adaptation: single-channel architectures interleave input/output into a single sequence (e.g., SpeechGPT, MiniOmni), while dual-channel architectures use independent channels (e.g., Moshi’s dual Transformer for inner/outer streams); (b) Interaction strategies: explicit strategies based on special tokens (using `<wait>/<speak>` tokens to control timing), implicit strategies based on heuristics (triggering generation when output probability exceeds a threshold), and learned strategies (learning optimal interaction timing via RL).
    *   **Design Motivation**: Full-duplex interaction is the natural mode of human conversation—people listen while speaking, but the "turn-taking" mode of traditional LLMs cannot achieve this parallelism.

### Loss & Training

As a survey paper, specific loss functions are not proposed. The paper outlines key streaming training challenges: the lack of streaming pre-training data, the design of partial input supervision, and the high cost of temporal alignment annotations.

## Key Experimental Results

### Main Results

The survey organizes key technical comparisons:

**Comparison of the Three Streaming Paradigms**

| Paradigm | Input Mode | Output Mode | Core Challenge | Representative Methods |
| :--- | :--- | :--- | :--- | :--- |
| Output-streaming | Static | Streaming | Efficient Generation | Speculative Decoding, SoT |
| Sequential-streaming | Streaming | Streaming (Delayed) | Incremental Encoding + Context Management | StreamingLLM, H2O, Mamba |
| Concurrent-streaming | Streaming | Streaming (Simultaneous) | Full-duplex Architecture + Interaction Strategy | Moshi, MiniOmni, OmniChat |

**Comparison of KV Cache Compression Methods**

| Method Category | Representative Work | Mechanism | Pros/Cons |
| :--- | :--- | :--- | :--- |
| Token Eviction | H2O, FastGen | Evict unimportant tokens based on attention scores | Simple/efficient but may lose critical info |
| Quantization | KVQuant, KIVI | Reduce KV cache precision | High compression but possible accuracy loss |
| Merging | CaM, D2O | Merge KV representations of similar tokens | Retains info but higher compute overhead |

### Key Findings

*   The three-stage taxonomy clearly illustrates the progression of technical challenges—Concurrent-streaming builds on the previous two, adding architectural adaptation and interaction strategies.
*   Speech streaming is currently the most mature direction (Moshi, SpeechGPT-Gen), followed by text, with video streaming being the least mature.
*   KV cache management is a common bottleneck across all streaming scenarios—the discovery of "attention sinks" in StreamingLLM (initial tokens are critical for attention scores) has driven this direction.
*   The primary open problem in full-duplex interaction is learning the strategy for "when to respond"—heuristic/threshold methods are simple but brittle, while RL methods are promising but difficult to train.

## Highlights & Insights

*   The three-stage progressive taxonomy is highly clear—unifying confusing terminology from a data flow perspective and providing a structured roadmap for future research.
*   The comparison between single-channel and dual-channel architectures reveals the core design trade-off in full-duplex interaction: single-channel is simple but suffers from input-output interference, while dual-channel allows parallelism but doubles parameter counts.
*   The systematic categorization of "when to respond" (explicit tokens, implicit thresholds, learned strategies) covers the full spectrum from simple to complex.

## Limitations & Future Work

*   Evaluation standards for Streaming LLMs are not yet unified—there is no consensus on how to synthesize latency, quality, and interactivity.
*   The scarcity of streaming pre-training data is a fundamental bottleneck, especially for full-duplex interactive scenarios.
*   Unified architectures for multimodal concurrent streaming (simultaneous speech, video, and text processing) remain an open problem.
*   Safety considerations are insufficient—once a streaming generation is output, it is difficult to retract, making the cost of errors higher than in static inference.

## Related Work & Insights

*   **vs Xiao et al. (2023) StreamingLLM**: StreamingLLM is a specific method (attention sink + sliding window); this survey categorizes it as a KV cache management method under Sequential-streaming.
*   **vs Speculative Decoding Surveys**: Speculative decoding is an efficient generation technique within Output-streaming; this survey places it within the broader Streaming LLM framework.
*   **vs Speech Dialogue Model Surveys**: Speech dialogue models focus mainly on Concurrent-streaming; this survey encompasses text and video scenarios simultaneously.

## Rating

*   Novelty: ⭐⭐⭐⭐ The three-stage progressive taxonomy and unified definition provide significant conceptual contributions.
*   Experimental Thoroughness: ⭐⭐ Survey paper with no original experiments.
*   Writing Quality: ⭐⭐⭐⭐⭐ Clear categorization, excellent diagram design, and comprehensive coverage.
*   Value: ⭐⭐⭐⭐⭐ The first systematic survey of Streaming LLMs, providing a much-needed structural framework for a rapidly evolving field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] PersonaArena: Dynamic Simulation for Evaluating and Enhancing Persona-Level Role-Playing in Large Language Models](personaarena_dynamic_simulation_for_evaluating_and_enhancing_persona-level_role-.md)
- [\[ACL 2026\] Foresight Optimization for Strategic Reasoning in Large Language Models](foresight_optimization_for_strategic_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Repeated Sequences Reveal Gaps between Large Language Models and Natural Language](repeated_sequences_reveal_gaps_between_large_language_models_and_natural_languag.md)
- [\[ACL 2026\] Adam's Law: Textual Frequency Law on Large Language Models](adam39s_law_textual_frequency_law_on_large_language_models.md)
- [\[ICML 2026\] ANCHOR: Abductive Network Construction with Hierarchical Orchestration for Reliable Probability Inference in Large Language Models](../../ICML2026/llm_nlp/anchor_abductive_network_construction_with_hierarchical_orchestration_for_reliab.md)

</div>

<!-- RELATED:END -->
