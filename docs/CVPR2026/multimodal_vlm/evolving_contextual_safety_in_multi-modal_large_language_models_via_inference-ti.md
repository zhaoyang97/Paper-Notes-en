---
title: >-
  [Paper Note] Evolving Contextual Safety in Multi-Modal Large Language Models via Inference-Time Self-Reflective Memory
description: >-
  [CVPR 2026][Multimodal VLM][MLLM safety] This paper proposes MM-SafetyBench++ and the EchoSafe framework, which accumulates safety insights by maintaining a self-reflective memory bank at inference time, enabling MLLMs to distinguish visually similar scenarios with different safety intents based on context—improving contextual safety without any training.
tags:
  - CVPR 2026
  - Multimodal VLM
  - MLLM safety
  - contextual safety
  - self-reflective memory
  - inference-time defense
  - safety benchmark
date: 2026-05-08
content_hash: 1ec80582f8c92cdd
---

# Evolving Contextual Safety in Multi-Modal Large Language Models via Inference-Time Self-Reflective Memory

**Conference**: CVPR 2026
**arXiv**: [2603.15800](https://arxiv.org/abs/2603.15800)
**Code**: [https://EchoSafe-mllm.github.io](https://EchoSafe-mllm.github.io)
**Area**: Multimodal VLM
**Keywords**: MLLM safety, contextual safety, self-reflective memory, inference-time defense, safety benchmark

## TL;DR
This paper proposes MM-SafetyBench++ and the EchoSafe framework, which accumulates safety insights by maintaining a self-reflective memory bank at inference time, enabling MLLMs to distinguish visually similar scenarios with different safety intents based on context—improving contextual safety without any training.

## Background & Motivation
**Background**: MLLMs demonstrate strong performance on multimodal reasoning tasks but face significant safety risks. Existing defense methods primarily focus on detecting and refusing jailbreak attacks.

**Limitations of Prior Work**: Existing methods tend to exhibit over-defensive behavior—rejecting even benign queries. For instance, a model may refuse to answer "How should I use this knife?" upon seeing a kitchen knife, even though the user is simply asking about cooking.

**Key Challenge**: There exists a fundamental trade-off between safety and utility. Over-defensiveness ensures safety but compromises helpfulness, while relaxed defenses risk producing harmful outputs.

**Goal**: (a) The lack of a systematic benchmark for evaluating contextual safety; (b) How to enable models to understand contextual differences and make appropriate safety decisions without training.

**Key Insight**: Humans form abstract cognitive patterns by accumulating past experiences, allowing flexible responses to similar yet distinct situations. Inspired by this, the paper proposes maintaining an "experiential memory bank" during inference.

**Core Idea**: Dynamically accumulate and retrieve safety insights at inference time via a self-reflective memory bank, enabling the model's safety behavior to continuously evolve.

## Method

### Overall Architecture
EchoSafe is a training-free framework built on two core mechanisms: (1) self-reflective memory construction—extracting safety insights from each interaction and storing them in a memory bank; (2) memory-retrieval-augmented inference—retrieving the most relevant past safety experiences when a new query arrives and integrating them into the prompt to guide contextually aware safety reasoning.

### Key Designs

1. **MM-SafetyBench++ Benchmark**:

    - Function: Constructs a safe counterpart for each unsafe image-text pair by flipping user intent through minimal modifications.
    - Mechanism: Preserves the underlying contextual semantics while altering only the safety intent, forming safe-unsafe paired samples.
    - Design Motivation: Addresses three shortcomings of existing benchmarks: exclusive focus on refusal behavior, insufficient difficulty, and coarse evaluation metrics.

2. **Self-Reflective Memory Bank**:

    - Function: Dynamically accumulates contextual safety insights during inference.
    - Mechanism: After each interaction, the model reflects on its own safety reasoning and extracts safety-relevant patterns, stored as memory entries.
    - Design Motivation: Simulates the human cognitive process of learning from experience, allowing safety capabilities to continuously evolve with use.

3. **Contextual Safety Retrieval Reasoning**:

    - Function: Retrieves the most relevant memory entries and integrates them into the prompt upon receiving a new query.
    - Mechanism: Retrieves past safety scenarios based on semantic similarity and uses them as in-context examples to guide the model.
    - Design Motivation: Enables more contextually aware safety reasoning grounded in prior safety judgments.

### Evaluation Metrics
The paper introduces the harmonic mean of Contextual Correctness Rate (CCR) and Quality Score (QS), jointly measuring the unsafe refusal rate and the response rate on safe queries.

## Key Experimental Results

### Main Results

| Model | Method | Illegal Activity CCR/QS | Hate Speech CCR/QS | Physical Harm CCR/QS | Fraud CCR/QS |
|-------|--------|------------------------|-------------------|---------------------|-------------|
| GPT-5 | Baseline | 91.9/4.6 | 93.1/4.6 | 94.9/4.8 | 85.9/4.3 |
| GPT-5-Mini | Baseline | 92.2/4.5 | 92.7/4.5 | 96.4/4.8 | 88.4/4.4 |
| Gemini-2.5-Pro | Baseline | 76.4/3.6 | 79.8/3.7 | 63.3/3.0 | 68.9/3.3 |
| LLaVA-1.5-7B | Baseline | 7.9/0.4 | 16.8/0.7 | 8.1/0.4 | — |
| LLaVA-1.5-7B | +EchoSafe | Significant gain | Significant gain | Significant gain | Significant gain |

### Ablation Study

| Configuration | CCR | QS | Notes |
|--------------|-----|----|-------|
| Full EchoSafe | Best | Best | Complete framework |
| w/o memory retrieval | Degraded | Degraded | Reduces to zero-shot without retrieval |
| w/o self-reflection | Degraded | Degraded | Lacks experience accumulation |
| Random memory | Degraded | Degraded | Irrelevant memories fail to guide reasoning |

### Key Findings
- Open-source models lag far behind closed-source models in contextual safety; LLaVA-1.5-7B achieves single-digit CCR scores.
- EchoSafe consistently improves contextual safety across multiple models while preserving helpfulness on general tasks.
- Continuous accumulation in the memory bank causes safety performance to improve as interactions increase, demonstrating the "evolving" property.
- Computational overhead is reasonable, making the framework suitable for practical deployment.

## Highlights & Insights
- The **safe-unsafe pairing design** is particularly elegant: by flipping intent through minimal modifications, it precisely evaluates a model's contextual understanding rather than simple safe/unsafe binary classification.
- The **training-free** design enables direct application to any MLLM without retraining or fine-tuning.
- The "continuous evolution" property of the self-reflective memory fundamentally distinguishes EchoSafe from existing methods, as safety capability grows with use.
- The contextual reasoning paradigm is transferable to other tasks requiring fine-grained understanding.

## Limitations & Future Work
- Growth in memory bank size may introduce retrieval efficiency and storage challenges.
- The quality of self-reflection depends on the model's inherent safety judgment capability, potentially limiting effectiveness for weaker models.
- The benchmark primarily covers visual-text pairs and does not address more complex multi-turn dialogue safety scenarios.
- Quality control and deduplication mechanisms for memory entries have room for further optimization.

## Related Work & Insights
- **vs. ECSO/AdaShield**: Prior prompt engineering methods guide safety reasoning through fixed templates; EchoSafe achieves more flexible contextual adaptation via dynamic memory retrieval.
- **vs. safety fine-tuning methods (VLGuard, etc.)**: Fine-tuning is constrained by training data; EchoSafe continuously adapts to new scenarios without any training.
- The notion of contextual safety can inspire other multimodal safety research, such as safety judgment in video understanding.

## Rating
- Novelty: ⭐⭐⭐⭐ The formalization of contextual safety and the memory-driven framework design are novel contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 4 safety benchmarks, 4 general benchmarks, and 3 representative MLLMs.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated and the method description is intuitive.
- Value: ⭐⭐⭐⭐ Contextual safety is a critical issue for MLLM deployment; both the benchmark and the method offer practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] EvoLMM: Self-Evolving Large Multimodal Models with Continuous Rewards](evolmm_self_evolving_lmm_continuous_rewards.md)
- [\[CVPR 2026\] Scaling Test-Time Robustness of Vision-Language Models via Self-Critical Inference Framework](scaling_test-time_robustness_of_vision-language_models_via_self-critical_inferen.md)
- [\[AAAI 2026\] SDEval: Safety Dynamic Evaluation for Multimodal Large Language Models](../../AAAI2026/multimodal_vlm/sdeval_safety_dynamic_evaluation_for_multimodal_large_language_models.md)
- [\[CVPR 2026\] Decoupling Stability and Plasticity for Multi-Modal Test-Time Adaptation](decoupling_stability_and_plasticity_for_multi-modal_test-time_adaptation.md)
- [\[CVPR 2026\] EMO-R3: Reflective Reinforcement Learning for Emotional Reasoning in Multimodal Large Language Models](emo-r3_reflective_reinforcement_learning_for_emotional_reasoning_in_multimodal_l.md)

</div>

<!-- RELATED:END -->
