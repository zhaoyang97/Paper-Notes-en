---
title: >-
  [Paper Note] ReflectDiffu: Reflect between Emotion-intent Contagion and Mimicry for Empathetic Response Generation via a RL-Diffusion Framework
description: >-
  [ACL 2025][Dialogue Systems][Empathetic Dialogue] Proposes a lightweight empathetic dialogue framework called ReflectDiffu, which integrates emotion contagion (capturing emotion), an intent twice mechanism (Exploring-Sampling-Correcting to map emotion to behavioral intent), and diffusion model generation, comprehensively outperforming existing baselines and Llama-3.1-8B in terms of relevance, controllability, and informativeness.
tags:
  - "ACL 2025"
  - "Dialogue Systems"
  - "Empathetic Dialogue"
  - "Emotion Contagion"
  - "Intent Mimicry"
  - "Diffusion Model"
  - "Reinforcement Learning"
date: 2026-05-08
content_hash: 78586d3be2cb681a
---

# ReflectDiffu: Reflect between Emotion-intent Contagion and Mimicry for Empathetic Response Generation via a RL-Diffusion Framework

**Conference**: ACL 2025  
**arXiv**: [2409.10289](https://arxiv.org/abs/2409.10289)  
**Code**: None  
**Area**: Dialogue Systems  
**Keywords**: Empathetic Dialogue, Emotion Contagion, Intent Mimicry, Diffusion Model, Reinforcement Learning  

## TL;DR

Proposes a lightweight empathetic dialogue framework called ReflectDiffu, which integrates emotion contagion (capturing emotion), an intent twice mechanism (Exploring-Sampling-Correcting to map emotion to behavioral intent), and diffusion model generation, comprehensively outperforming existing baselines and Llama-3.1-8B in terms of relevance, controllability, and informativeness.

## Background & Motivation

**Background**: Empathetic dialogue generation requires identifying emotional states and generating appropriate emotional responses. Existing methods either rely on external knowledge enhancement (commonsense reasoning, causal inference) or use LLM+CoT, but the former is poorly controllable and the latter incurs high computational overhead.

**Limitations of Prior Work**: They ignore the cognitive interaction mechanism between emotion and intent—the theories of emotion contagion and empathetic mimicry in psychology indicate that empathetic behavior is a chain of "perceiving the other side's emotion $\rightarrow$ generating intent $\rightarrow$ executing action".

**Key Challenge**: Lightweight models lack deep emotion-to-intent mapping capability, while LLMs are capable but too heavy. How can small models be enabled to perform reflective mapping of emotion-to-intent effectively?

**Goal**: To design a psychology-inspired lightweight framework that translates emotional decisions into precise intentional actions through a reflection mechanism.

**Key Insight**: Operationalize the sociological theories of emotion contagion and empathetic mimicry into computational modules—emotion contagion to perceive user emotions, and intent mimicry to map emotions to response intent.

**Core Idea**: Use emotion contagion to perceive emotions, the RL-guided intent twice mechanism to decide actions, and the diffusion model to generate responses.

## Method

### Overall Architecture
Three core components: (1) Emotion mimicry module—enhances emotion perception with emotion contagion, and locates key elements using emotion cause masking; (2) Intent twice mechanism—Exploring (exploring intent space) - Sampling (sampling intent) - Correcting (RL correction); (3) Diffusion decoder—generates a response guided by the intent.

### Key Designs

1. **Emotion Contagion Encoder**:

    - **Function**: Capture the process of emotion transmission during dialogue.
    - **Mechanism**: Mine emotional causes from the text using an Emotion Cause Annotator (ERA) to generate inference masks; perform fine-grained emotion classification using Contrastive-Experts.
    - **Design Motivation**: Understand "why this emotion exists" before deciding "how to respond".

2. **Intent Twice Mechanism**:

    - **Function**: Map the identified emotions to concrete response intents.
    - **Mechanism**:
        - **Exploring**: Query predefined top-3 intent references according to the emotion group (e.g., "sad" $\rightarrow$ acknowledging/consoling/encouraging).
        - **Sampling**: The policy network samples a specific intent from the reference intents.
        - **Correcting**: RL reward signals correct the intent choice—$\text{Reward} = \text{BARTScore(response quality)} + \text{Emotion matching degree}$.
    - **Design Motivation**: "Twice" means coarse-to-fine—first narrowing down the range via an emotion-intent mapping table, and then precisely selecting via RL.

3. **Diffusion Response Decoder**:

    - **Function**: Generate empathetic responses guided by the intent.
    - **Mechanism**: Use DDPM to progressively denoise response token embeddings from noise, with the intent vector served as conditional input.
    - **Design Motivation**: The diversity of diffusion models is superior to autoregressive generation, which is suitable for empathetic dialogues that require flexible expression.

### Loss & Training
- Multi-task training: Emotion classification + Intent prediction + RL reward + Diffusion denoising.
- Trained on the EmpatheticDialogues dataset.
- Self-annotated emotion-intent mapping table (collecting the top-3 common intents under each emotion).

## Key Experimental Results

### Main Results

| Method | BLEU-1 | BARTScore | Acc_emo | Acc_intent | PPL(↓) | Dist-2 |
|------|--------|----------|---------|-----------|--------|--------|
| CAB (Prev. SOTA) | ~14 | ~-3.5 | ~34 | - | ~50 | ~2.0 |
| Llama-3.1-8B CoT | ~16 | ~-3.6 | ~17 | ~32 | ~17 | ~2.3 |
| **Ours** | **~16.3** | **~-3.3** | **~41** | **~80** | **~35** | **~3.0** |

### Ablation Study

| Configuration | Effect | Explanation |
|------|------|------|
| w/o ERA | Decreased emotion accuracy | Emotion Cause Annotator is crucial |
| w/o C-Experts | Emotional classification degrades | Contrastive-Experts module is necessary |
| w/o Intent Twice | Significant drop in intent accuracy and relevance | Core component |
| w/o EMU (Diffusion) | Decreased diversity | Diffusion model increases diversity |

### Key Findings
- ReflectDiffu achieves 80.32% intent accuracy—far exceeding all baselines, proving that the intent twice mechanism is highly effective.
- 2.4x higher emotion accuracy than Llama-3.1-8B CoT—lightweight model + proper mechanism > LLM + general prompt.
- The diffusion model contributes 47.4% of the Dist-2 gain—showing a distinct diversity advantage.
- Human evaluation comprehensively wins across three dimensions: empathy, relevance, and fluency.

## Highlights & Insights

- **Psychological theory-driven framework design**—Directly operationalizes the theories of emotion contagion and empathetic mimicry into computational modules, providing a solid theoretical foundation. It designs modules based on validated sociological empathetic mechanisms rather than pure intuition.
- **Lightweight beats LLMs**—Demonstrates that domain-specific mechanism design can compensate for model scale gaps. ReflectDiffu with fewer parameters outperforms Llama-3.1-8B CoT on multiple metrics.
- **Clever design of the intent twice mechanism's** Exploring-Sampling-Correcting—It successfully combines retrieval (narrowing down to top-3 intents), sampling (exploring from candidates), and RL (modifying based on rewards) to refine selection step-by-step.
- **The diversity advantage of diffusion models** is validated in empathetic dialogue—Distinct-2 improved by 47.4%, illustrating that the diffusion denoising process inherently supports diverse emotional expressions.
- The emotion-intent mapping table (Table 1) itself holds educational/psychological reference value.

## Limitations & Future Work

- The emotion-intent mapping table is predefined and only covers the top-3 intents, which may lack flexibility—the optimal intent in certain scenarios might exceed the mapping range.
- Validated only on a single dataset, EmpatheticDialogues; generalization cross-domain and cross-culture remains unknown.
- Inference speed of diffusion models is slower than autoregressive models—multi-step denoising adds latency.
- The quality of training data for ERA (Emotion Cause Annotator) directly affects system performance—annotation bias propagates.
- No comparison with the latest ChatGPT/GPT-4 class models—only compared against Llama-3.1-8B.

## Related Work & Insights

- **vs CAB/MISC**: These methods enhance with external knowledge but do not model the inner connection between emotion and intent; ReflectDiffu performs explicit mapping using a reflection mechanism.
- **vs LLM+CoT**: LLMs perform empathy based on general capabilities but are uncontrollable; ReflectDiffu controls precisely through the intent mechanism.

## Rating

- Novelty: ⭐⭐⭐⭐ Unique combination of psychology-inspired design + RL + diffusion.
- Experimental Thoroughness: ⭐⭐⭐⭐ Automated + human evaluation + ablation, but limited to a single dataset.
- Writing Quality: ⭐⭐⭐ Complex method, high usage of terminology, moderate readability.
- Value: ⭐⭐⭐⭐ Meaningful contribution to empathetic dialogue.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] STRIDE-ED: A Strategy-Grounded Stepwise Reasoning Framework for Empathetic Dialogue Systems](../../ACL2026/dialogue/stride-ed_a_strategy-grounded_stepwise_reasoning_framework_for_empathetic_dialog.md)
- [\[ACL 2025\] UniConv: Unifying Retrieval and Response Generation for Large Language Models in Conversations](uniconv_retrieval_response_gen.md)
- [\[ACL 2026\] Author-in-the-Loop Response Generation and Evaluation: Integrating Author Expertise and Intent in Responses to Peer Review](../../ACL2026/dialogue/author-in-the-loop_response_generation_and_evaluation_integrating_author_experti.md)
- [\[ACL 2025\] Dynamic Label Name Refinement for Few-Shot Dialogue Intent Classification](dynamic_label_name_refinement_for_few-shot_dialogue_intent_classification.md)
- [\[NeurIPS 2025\] LatentGuard: Controllable Latent Steering for Robust Refusal of Attacks and Reliable Response Generation](../../NeurIPS2025/dialogue/latentguard_controllable_latent_steering_for_robust_refusal_of_attacks_and_relia.md)

</div>

<!-- RELATED:END -->
