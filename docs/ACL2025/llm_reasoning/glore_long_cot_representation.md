---
title: >-
  [Paper Note] Unlocking General Long Chain-of-Thought Reasoning Capabilities of Large Language Models via Representation Engineering
description: >-
  [ACL 2025][Reasoning][Long Chain-of-Thought] It is discovered from the perspective of representation space that LLMs encode long CoT reasoning as a general capability clearly distinct from vanilla CoT. This work proposes GLoRE (General Long CoT Reasoning via Representation Engineering)—which unlocks long CoT capabilities through contrastive reasoning pattern injection and domain-specific representation steering, outperforming SFT methods in both in-domain and cross-domain sce…
tags:
  - "ACL 2025"
  - "Reasoning"
  - "Long Chain-of-Thought"
  - "Representation Engineering"
  - "Cross-Domain Reasoning"
  - "Training-Free Method"
  - "Slow Thinking"
date: 2026-05-08
content_hash: c8f01e11c396751e
---

# Unlocking General Long Chain-of-Thought Reasoning Capabilities of Large Language Models via Representation Engineering

**Conference**: ACL 2025  
**arXiv**: [2503.11314](https://arxiv.org/abs/2503.11314)  
**Code**: [https://github.com/txy77/GLoRE](https://github.com/txy77/GLoRE)  
**Area**: LLM Reasoning  
**Keywords**: Long Chain-of-Thought, Representation Engineering, Cross-Domain Reasoning, Training-Free Method, Slow Thinking  

## TL;DR
It is discovered from the perspective of representation space that LLMs encode long CoT reasoning as a general capability clearly distinct from vanilla CoT. This work proposes GLoRE (General Long CoT Reasoning via Representation Engineering)—which unlocks long CoT capabilities through contrastive reasoning pattern injection and domain-specific representation steering, outperforming SFT methods in both in-domain and cross-domain scenarios without training.

## Background & Motivation
**Background**: Slow-thinking reasoning models (o1, DeepSeek-R1) have significantly improved LLM performance on complex reasoning tasks through long chain-of-thought (long CoT)—detailed reasoning including planning, verification, and backtracking. Research shows that long CoT capability can be efficiently triggered with few-shot examples and transferred across tasks.

**Limitations of Prior Work**: (a) It remains unclear whether long CoT is a "general and independent" capability in LLMs or is task-specific; (b) Cross-domain transfer performs poorly in certain domains—transferring well in mathematics but poorly in other domains like physics; (c) High-quality long CoT data is not easily constructed for all domains.

**Key Challenge**: Long CoT capability appears to be general (triggerable with small data + transferable), yet domain transfer is incomplete—suggesting there are both general and domain-specific components that need to be understood in isolation.

**Goal**: To understand and leverage the generality and domain-specificity of long CoT from the perspective of representation engineering, unlocking long CoT capability across domains without training.

**Key Insight**: By analyzing the internal long CoT representations of LLMs using representation engineering, it is discovered that long CoT and vanilla CoT occupy distinct regions in the representation space, and this distinction is consistent across domains. Based on this, it is proposed to directly manipulate representations to switch reasoning modes.

**Core Idea**: Long CoT is an independent "region" in the representation space of LLMs—which can be activated without training by directly pushing the model into this region via representation injection.

## Method

### Overall Architecture
GLoRE accomplishes representation injection in two steps: (1) **Contrastive Reasoning Pattern Injection**—calculates the representation difference (contrastive vector) between long CoT and vanilla CoT, and injects it into the activations of middle layers of the LLM to push the model from the "vanilla CoT region" to the "long CoT region"; (2) **Domain-Specific Representation Injection**—extracts domain representation vectors from a few examples of the target domain and injects them to precisely guide the model from the general long CoT region to the specific space of the target domain.

### Key Designs

1. **Representation Space Analysis (Discovery)**:

    - **Function**: To prove that long CoT is an independent and general capability in the representation space.
    - **Key Findings**: (a) Long CoT representations cluster in specific regions across layers, clearly separated from vanilla CoT; (b) The contrastive directions of long/vanilla CoT across different domains are similar—indicating that the direction to "switch to long CoT" is general.
    - **Design Motivation**: These two findings support the feasibility of "switching reasoning modes through representation injection".

2. **Contrastive Reasoning Pattern Injection**:

    - **Function**: To push the model from vanilla CoT to long CoT.
    - **Mechanism**: Collect a small number of paired long/vanilla CoT examples, calculate the representation differences in each intermediate layer ($\Delta h = h_{long} - h_{vanilla}$), and add this difference vector to the activations of the corresponding layers during inference.
    - **Design Motivation**: Similar to the approach of InstructionRepresentation—general reasoning pattern switching does not require parameter modification, but only a "push" in the representation space.

3. **Domain-Specific Representation Steering**:

    - **Function**: To fine-tune the general long CoT to the target domain.
    - **Mechanism**: Extract domain representation vectors from target domain questions (no need for answers, just questions), and inject them to guide the model to focus on domain-specific knowledge and reasoning patterns.
    - **Design Motivation**: General long CoT performs well in math but poorly in physics—because physics requires different domain knowledge. Domain-specific injection bridges this gap.

### Loss & Training
- **Completely training-free**—pure inference-time representation manipulation.
- Requires only a small number of labeled long/vanilla CoT pairs to calculate contrastive vectors.
- Supports multiple LLM backbones (Qwen2.5-7B, LLaMA3.1-8B).

## Key Experimental Results

### Main Results (Qwen2.5-7B-Instruct)

| Method | Math (In-domain) | Physics (Cross-domain) | Chemistry (Cross-domain) | Description |
|------|----------|----------|----------|------|
| Vanilla CoT | Baseline | Baseline | Baseline | Short reasoning chain |
| Prompt Engineering (Simulating long CoT) | Slight improvement | Slight improvement | Slight improvement | Does not guide deep thinking |
| SFT (Math long CoT data) | High | Medium | Medium | Training-based method |
| **GLoRE (Training-free)** | **Highest** | **High** | **High** | Outperforms SFT |

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| Contrastive Reasoning Injection Only | Significant improvement | General long CoT switching is effective |
| + Domain-Specific Injection | Further improvement | Domain adjustment is necessary |
| Different Injection Layers | Middle layers optimal | Consistent with representation engineering literature |
| Number of Contrastive Samples | 10-50 is sufficient | Highly data-efficient |

### Key Findings
- **GLoRE outperforms SFT training-free**—manipulating representations alone is better than fine-tuning on thousands of long CoT samples.
- Long CoT is indeed a general capability of LLMs—the similarity of contrastive directions across different domains is as high as 0.8+.
- Domain-specific representation is crucial for cross-domain transfer—general injection alone brings limited gains in non-mathematical domains such as physics.
- Middle layers (~50% depth) are the optimal locations for injection—too shallow has insufficient impact, too deep may disrupt existing information.
- Only 10-50 contrastive samples are needed—extremely high data efficiency.

## Highlights & Insights
- The discovery that **"long CoT is an independent region in the representation space"** is of fundamental significance—indicating that LLMs already possess the potential for slow thinking and only need to be "guided" to the correct representation region.
- **Outperforming SFT without training** demonstrates that representation engineering is more efficient than parameter updates in activating existing capabilities—since the capability is already present in the parameters, and only requires the correct "activation signal".
- The **binary decomposition of general and domain-specific parts** provides a clear framework for understanding long CoT—contrastive vectors represent "how to think" (general), while domain representations represent "what knowledge to think with" (specific).
- This method allows dynamic switching during inference—injected when deep thinking is needed, and not injected when it is not, making it flexible and controllable.
- It provides important inspiration for the research and development of o1-like models—large-scale RL training is not necessarily required; representation engineering is also a viable path.

## Limitations & Future Work
- Calculating the contrastive vector still requires a small number of paired long/vanilla CoT samples—although very few, it is not completely zero-shot.
- The interpretability of representation engineering is still limited—the lack of theoretical explanation for why middle layers perform best.
- The performance on ultra-large models (70B+) has not been verified.
- Injection strength requires hyperparameter tuning—excessive strength may lead to output degradation.

## Related Work & Insights
- **vs SFT on long CoT**: SFT modifies parameters to learn long CoT; GLoRE modifies activations to activate existing capabilities—different levels.
- **vs Prompt Engineering ("Think step by step in detail")**: Prompts only guide at the input level with limited impact; GLoRE directly manipulates at the representation level—deeper level.
- **vs Disentangling Memory & Reasoning**: That work uses tokens to separate memory and reasoning; this work uses representation vectors to separate short/long thinking—complementary perspectives.
- **vs RepE/Activation Steering (Zou et al.)**: Uses representation engineering to control safety/truthfulness; this work is the first to use it to control reasoning depth—a new application.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to understand long CoT from the perspective of representation space, offering dual innovations in both findings and methods.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ In-domain + cross-domain + ablation + representation analysis + multiple models.
- Writing Quality: ⭐⭐⭐⭐⭐ Perfect logical chain of Discovery -> Hypothesis -> Method -> Verification.
- Value: ⭐⭐⭐⭐⭐ Fundamental contribution to understanding and activating LLM reasoning capabilities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can Large Language Models Detect Errors in Long Chain-of-Thought Reasoning?](can_large_language_models_detect_errors_in_long_chain-of-thought_reasoning.md)
- [\[ACL 2025\] Enhancing Chain-of-Thought Reasoning with Critical Representation Fine-tuning](enhancing_chain-of-thought_reasoning_with_critical_representation_fine-tuning.md)
- [\[ACL 2025\] DRT: Deep Reasoning Translation via Long Chain-of-Thought](drt_deep_reasoning_translation_via_long_chain-of-thought.md)
- [\[ACL 2025\] Chain-of-Reasoning: Towards Unified Mathematical Reasoning in Large Language Models via a Multi-Paradigm Perspective](chain_of_reasoning_unified_math.md)
- [\[ACL 2025\] Large Language and Reasoning Models are Shallow Disjunctive Reasoners](large_language_and_reasoning_models_are_shallow_disjunctive_reasoners.md)

</div>

<!-- RELATED:END -->
