---
title: >-
  [Paper Note] LoopLLM: Transferable Energy-Latency Attacks in LLMs via Repetitive Generation
description: >-
  [AAAI 2026][LLM/NLP][energy-latency attack] This paper proposes LoopLLM, a framework that launches energy-latency attacks by inducing LLMs into repetitive generation modes. Through repetition-inducing prompt optimization…
tags:
  - "AAAI 2026"
  - "LLM/NLP"
  - "energy-latency attack"
  - "repetitive generation"
  - "adversarial suffix"
  - "low-entropy loop"
  - "cross-model transferability"
date: 2026-05-08
content_hash: 0d27b77b659f2cfb
---

# LoopLLM: Transferable Energy-Latency Attacks in LLMs via Repetitive Generation

**Conference**: AAAI 2026
**arXiv**: [2511.07876](https://arxiv.org/abs/2511.07876)
**Code**: [https://github.com/neuron-insight-lab/LoopLLM](https://github.com/neuron-insight-lab/LoopLLM)
**Area**: LLM/NLP
**Keywords**: energy-latency attack, repetitive generation, adversarial suffix, low-entropy loop, cross-model transferability

## TL;DR

This paper proposes LoopLLM, a framework that launches energy-latency attacks by inducing LLMs into repetitive generation modes. Through repetition-inducing prompt optimization and token-aligned ensemble optimization, LoopLLM achieves over 90% of maximum output length across 12 open-source and 2 commercial LLMs, with approximately 40% improvement in cross-model transferability.

## Background & Motivation

**Background**: LLM inference consumes substantial computational resources, with inference energy accounting for 90% of total LLM lifecycle energy consumption. Current security research predominantly focuses on integrity (jailbreaking) and confidentiality, with insufficient attention to availability (energy-latency attacks).

**Limitations of Prior Work**: Existing energy-latency attack methods (e.g., LLMEffiChecker, Engorgio) extend output length by delaying EOS token generation, but as output grows, controlling EOS via input becomes increasingly difficult, limiting attack effectiveness to approximately 20% of maximum length. Furthermore, existing methods rely on white-box gradient optimization and suffer severe overfitting to the source model, resulting in poor cross-model transferability.

**Key Challenge**: The delayed-EOS strategy cannot fundamentally alter output structure and fails to reliably trigger maximum-length outputs; a gap exists between the model-specificity of white-box optimization and the black-box requirements of real-world scenarios.

**Goal**: To design a more effective energy-latency attack that reliably forces LLMs to generate up to the maximum length limit while achieving strong cross-model transferability.

**Key Insight**: An observation that repetitive generation can trigger low-entropy decoding loops — once a model begins generating previously appeared content, the autoregressive mechanism reinforces repetition, forming a loop until the maximum length is reached.

**Core Idea**: Rather than suppressing EOS, the approach **induces repetitive generation**, exploiting the inherent vulnerability of autoregressive models to lock them into low-entropy loops. Introducing a small amount of repetition in the output rapidly reduces output entropy, proving more effective than input-level repetition.

## Method

### Overall Architecture

LoopLLM consists of two core components: (1) repetition-inducing prompt optimization — constructing adversarial suffixes containing cyclic segments, optimized via a cycle loss to encourage the model to reproduce cyclic patterns in its output; and (2) token-aligned ensemble optimization — aggregating gradients from multiple proxy models sharing the same tokenizer to improve cross-model transferability.

### Key Designs

1. **Repetition-Inducing Prompt Optimization**:

    - **Initialization**: Randomly sample tokens to form a cyclic segment (length $c$), then repeat-concatenate to total length $L$ as the initial suffix $\mathbf{t}_s$
    - **Cycle Loss**: Encourages the model to generate tokens from the cyclic segment at every output position
    - Formula: $\mathcal{L}_{cycle} = -\frac{1}{N}\sum_{i=1}^{N}\log\sum_{j=1}^{c}\mathcal{P}_i^{t_j}(\mathbf{x}_{adv})$
    - Softmax probabilities are used instead of logits for better measurement of relative confidence
    - **Gradient token search**: For each position in the suffix, one-hot gradients are computed over the full vocabulary, the top-$K$ candidate replacements are selected, $B$ are randomly sampled, and the one minimizing loss is adopted as the update

2. **Token-Aligned Ensemble Optimization**:

    - Mechanism: Multiple proxy models sharing the same tokenizer (e.g., variants of the Llama3 family) are used to ensure one-hot vectors are aligned in both dimensionality and token-index mapping
    - Gradients from $M$ models are aggregated: $\sum_{j=1}^{M}\nabla_{e_{t_i}}\mathcal{L}_{cycle}^{(j)}$
    - The candidate suffix minimizing the aggregated loss is selected
    - Avoids overfitting to a single model and discovers adversarial patterns with cross-model generalizability

3. **Key Mechanistic Insight**:

    - Input-level repetition is ineffective against instruction-aligned LLMs (the model tends to ignore it)
    - However, even a small amount of repetition in the output rapidly reduces output entropy and triggers a low-entropy loop
    - Therefore, it is necessary not only to embed repetitive patterns in the input but also to induce the model to reproduce them in the output

### Loss & Training

- Cycle Loss: A non-targeted strategy that only increases the probability of tokens within the cyclic segment across all output positions, without enforcing specific tokens at specific positions
- Optimization: Direct gradient descent is infeasible in discrete token space; GCG-style one-hot gradient search is used instead
- Early stopping: Optimization halts when output entropy stabilizes at a low level
- Attack scenarios: White-box (direct optimization on the target model) and black-box (proxy model + transfer)

## Key Experimental Results

### Main Results

**White-box attack (6 large models)**:

| Model | Max Length | Normal Avg-len | LoopLLM-t Avg-len | LoopLLM-t ASR |
|------|---------|---------------|-------------------|--------------|
| Llama2-13B | 8192 | 298 | 7439 | 91% |
| GLM4-9B | 4096 | 188 | 3730 | 90% |
| Llama3-8B | 4096 | 353 | 3892 | 94% |
| Vicuna-7B | 2048 | 233 | 1507 | 68% |
| Llama2-7B | 2048 | 309 | 1930 | 92% |
| Mistral-7B | 2048 | 248 | 1700 | 79% |

For comparison, LLMEffiChecker achieves a maximum ASR of only 23%, and Engorgio achieves at most 6%.

**Cross-model transfer (black-box)**:
- Transferability improves by approximately 40% when transferring to DeepSeek-V3 and Gemini 2.5 Flash
- Token-aligned ensemble optimization (LoopLLM-t) outperforms naive ensemble (LoopLLM-p)

### Ablation Study

- Cyclic segment length $c$ and total suffix length $L$ both affect attack performance
- Token alignment is a critical prerequisite for effective gradient aggregation (shared tokenizer required)
- Input-level repetition alone is insufficient to attack instruction-aligned models; output-level repetition must be induced

### Key Findings

- **Repetitive generation is more effective than suppressing EOS**: LoopLLM achieves 90%+ of maximum length, whereas EOS-delay methods reach only approximately 20%
- **Low-entropy loops are an inherent vulnerability of autoregressive models**: Once repetition appears in the output, the autoregressive mechanism self-reinforces it
- **Shared tokenizer is key to cross-model transferability**: Token alignment makes gradient aggregation semantically meaningful across different models
- **Instruction alignment does not prevent repetition attacks**: Aligned models can ignore input-level repetition, but cannot resist output-level repetition induced by optimized adversarial suffixes

## Highlights & Insights

- The mechanistic insight linking repetitive generation to low-entropy loops is highly precise, effectively transforming a "generation defect" into an "attack vector"
- The attack advantage is overwhelming: 90% vs. 20% represents a qualitative rather than incremental difference
- Token-aligned ensemble optimization is an elegant engineering design that resolves the semantic alignment problem in gradient aggregation by constraining proxy models to share a tokenizer
- Successful transfer to commercial models (DeepSeek-V3, Gemini) validates the practical threat

## Limitations & Future Work

- The assumption of a shared tokenizer constrains the selection range of proxy models
- The length of adversarial suffixes and their visibly garbled patterns make them susceptible to input-level filtering
- The study focuses exclusively on energy/latency and does not analyze the impact on generation quality
- Defense strategies are not discussed (e.g., the effectiveness of simple countermeasures such as repetition detection, output length limiting, and entropy monitoring)
- Attacks against closed-source models rely on transferability and are not consistently stable

## Related Work & Insights

- Sponge Examples (Shumailov et al. 2021) first introduced the concept of energy-latency attacks
- Engorgio uses a parameterized surrogate distribution to track long-sequence prediction trajectories, but is limited to text completion settings
- The gradient token search strategy from GCG (Zou et al. 2023) is adopted and extended to ensemble optimization in LoopLLM
- This work reveals a fundamental vulnerability of autoregressive models, posing new challenges for availability protection in LLM inference systems

## Rating

⭐⭐⭐⭐ (4/5)

The mechanistic insight is deep, the attack effectiveness is significant, and the method is concise and efficient. Exploiting the inherent vulnerability of autoregressive models through repetitive generation is a more fundamental approach than delaying EOS. The experiments cover 14 models comprehensively. Weaknesses include the absence of defense discussion and the limited stealthiness of adversarial suffixes. This work represents an important contribution to availability security research for AI systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] VSPO: Validating Semantic Pitfalls in Ontology via LLM-Based CQ Generation](vspo_validating_semantic_pitfalls_in_ontology_via_llm-based_cq_generation.md)
- [\[AAAI 2026\] C3TG: Conflict-aware, Composite, and Collaborative Controlled Text Generation](c3tg_conflict-aware_composite_and_collaborative_controlled_text_generation.md)
- [\[NeurIPS 2025\] Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models](../../NeurIPS2025/llm_nlp/nemotron-flash_towards_latency-optimal_hybrid_small_language_models.md)
- [\[AAAI 2026\] Guess or Recall? Training CNNs to Classify and Localize Memorization in LLMs](guess_or_recall_training_cnns_to_classify_and_localize_memorization_in_llms.md)
- [\[AAAI 2026\] Understanding Syllogistic Reasoning in LLMs from Formal and Natural Language Perspectives](understanding_syllogistic_reasoning_in_llms_from_formal_and_natural_language_per.md)

</div>

<!-- RELATED:END -->
