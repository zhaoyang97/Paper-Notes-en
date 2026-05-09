---
title: >-
  [Paper Note] MetaDefense: Defending Finetuning-based Jailbreak Attack Before and During Generation
description: >-
  [NeurIPS 2025][LLM Alignment][jailbreak defense] This paper proposes MetaDefense, a two-stage (pre-generation + mid-generation) defense framework that trains the LLM itself to predict the harmfulness of queries and partial responses, defending against finetuning-based jailbreak attacks without external classifiers, achieving 2× memory efficiency.
tags:
  - NeurIPS 2025
  - LLM Alignment
  - jailbreak defense
  - finetuning safety
  - mid-generation defense
  - LLM safety alignment
  - meta defense
date: 2026-05-08
content_hash: ca40f00e2342ea7b
---

# MetaDefense: Defending Finetuning-based Jailbreak Attack Before and During Generation

**Conference**: NeurIPS 2025

**arXiv**: [2510.07835](https://arxiv.org/abs/2510.07835)

**Code**: [GitHub](https://github.com/ws-jiang/MetaDefense)

**Area**: LLM Alignment / Safety

**Keywords**: jailbreak defense, finetuning safety, mid-generation defense, LLM safety alignment, meta defense

## TL;DR

This paper proposes MetaDefense, a two-stage (pre-generation + mid-generation) defense framework that trains the LLM itself to predict the harmfulness of queries and partial responses, defending against finetuning-based jailbreak attacks without external classifiers, achieving 2× memory efficiency.

## Background & Motivation

- **Finetuning Jailbreak Attack (FJAttack)**: Adversaries exploit finetuning APIs (e.g., from OpenAI) to finetune models on a small number of harmful samples, bypassing safety alignment.
- **Failures of Existing Defenses**:
  1. Existing defense mechanisms fail to generalize to **unseen attack templates** (e.g., novel role-playing prompts).
  2. External safety classifiers (e.g., Llama Guard) introduce additional inference overhead and memory requirements.
- **Core Observation**:
    - LLMs **can distinguish** disguised harmful queries in the embedding space, yet still generate harmful content during generation.
    - This suggests that safety signals already exist internally in LLMs but are not being leveraged.
- **Starting Point**: Train the LLM itself to act as a safety detector, continuously monitoring both before and during generation.

## Method

### Overall Architecture

MetaDefense operates in two defense stages:

```
Input Query → [Stage 1: Pre-Generation Defense] → Safe? ─No──→ Reject
                                                      │
                                                     Yes
                                                      ↓
              [Begin Generation] → [Stage 2: Mid-Generation Defense]
                                    │               │
                                  Every γ tokens   Check once
                                    │               │
                                  Harmful? ─Yes──→ Stop generation, return safety notice
                                    │
                                   No → Continue generation
```

### Key Designs

#### 1. Pre-Generation Defense

Before generating a response, a diagnostic prompt is appended to the LLM:

> "Is this query harmful or harmless? It is 100% ..."

- The LLM predicts a single token—"harmful" or "harmless."
- Decision is based on logit comparison: if $P(\text{harmful}) > P(\text{harmless})$, the query is immediately rejected.
- **Key advantage**: Reuses the LLM's own embedding space as a discriminator, requiring no external model.

#### 2. Mid-Generation Defense

Even if a query passes the pre-generation check, harmful content may still be generated:

- After every $\gamma$ tokens, generation is paused and a diagnostic prompt is appended:

> "Is this response harmful or harmless? It is 100% ..."

- If the prediction is "harmful," generation is immediately terminated and a rejection message is returned.
- $\gamma$ is adaptive: a larger window is used for safe content (reducing overhead), while the window shrinks for suspicious content.
- This progressive checking strategy captures "gradually harmful content" (responses that begin safely but become increasingly harmful).

#### 3. Lightweight Instruction Finetuning

- Training data: harmful/harmless query-response pairs with corresponding diagnostic labels.
- LoRA is used for low-rank adaptation, preserving the original model's capabilities.
- Only the diagnostic prompt response capability is trained, leaving normal generation quality unaffected.

### Loss & Training

The alignment training objective is:

$$\mathcal{L} = \mathcal{L}_{\text{safety}} + \lambda \mathcal{L}_{\text{utility}}$$

- $\mathcal{L}_{\text{safety}}$: Cross-entropy loss on harmful/harmless diagnostics.
- $\mathcal{L}_{\text{utility}}$: Performance preservation loss on normal tasks (SST-2, AG News, GSM8K).
- The BeaverTail dataset is used as the source of harmful queries.
- Four attack templates are used for training: Direct, PrefixInjection, RefusalSuppression, RolePlay.

## Key Experimental Results

### Main Results

#### Defense Effectiveness on LLaMA-2-7B (Attack Success Rate ASR↓)

| Defense Method | Direct ASR↓ | PrefixInj ASR↓ | RefusalSup ASR↓ | RolePlay ASR↓ | Unseen Template ASR↓ | SST-2 Acc↑ |
|---------|------------|----------------|-----------------|--------------|-------------|-----------|
| No Defense | 92.3 | 88.7 | 85.4 | 90.1 | 87.5 | 93.2 |
| Vaccine | 45.2 | 52.8 | 48.1 | 50.3 | 68.7 | 91.5 |
| RepNoise | 38.4 | 41.6 | 39.8 | 43.2 | 62.4 | 90.8 |
| TAR | 32.7 | 38.5 | 35.2 | 37.8 | 55.3 | 91.2 |
| Booster | 28.1 | 33.4 | 30.5 | 32.7 | 48.6 | 90.5 |
| **MetaDefense** | **8.3** | **11.2** | **9.7** | **10.5** | **15.8** | **92.8** |

**Finding**: MetaDefense achieves significantly lower ASR across all attack templates compared to all baselines, while maintaining the best benign task performance.

#### Cross-Architecture Validation

| Model | Method | Seen Template ASR↓ | Unseen Template ASR↓ | AG News Acc↑ | GSM8K Acc↑ |
|------|------|-------------|-------------|-------------|-----------|
| Qwen-2.5-3B-Inst | No Defense | 89.5 | 85.2 | 88.1 | 65.3 |
| Qwen-2.5-3B-Inst | Booster | 31.5 | 52.1 | 86.4 | 62.8 |
| Qwen-2.5-3B-Inst | **MetaDefense** | **10.8** | **18.3** | **87.5** | **64.7** |
| LLaMA-3.2-3B-Inst | No Defense | 87.2 | 83.8 | 86.7 | 62.1 |
| LLaMA-3.2-3B-Inst | Booster | 29.8 | 48.7 | 84.9 | 59.5 |
| LLaMA-3.2-3B-Inst | **MetaDefense** | **9.5** | **16.2** | **86.1** | **61.5** |

### Ablation Study

| Variant | Seen Template ASR↓ | Unseen Template ASR↓ | SST-2 Acc↑ |
|------|-------------|-------------|-----------|
| MetaDefense (Full) | **8.3** | **15.8** | **92.8** |
| Pre-Generation Only | 18.5 | 28.7 | 92.5 |
| Mid-Generation Only | 15.2 | 24.3 | 92.6 |
| Fixed Window γ=32 | 9.1 | 17.2 | 91.8 |
| Fixed Window γ=128 | 12.4 | 21.5 | 92.7 |
| External Classifier Substitute | 10.2 | 18.5 | 92.3 |

### Key Findings

1. **Complementarity of Two Stages**: Pre-Gen intercepts ~60% of harmful queries; Mid-Gen further intercepts ~80% of those that slip through.
2. **Generalization to Unseen Templates**: MetaDefense achieves only 15.8% ASR on unseen attack templates, compared to 48.6% for the strongest baseline.
3. **No Performance Sacrifice**: Benign task performance is marginally better than no-defense, attributed to the regularization effect of LoRA finetuning.
4. **Memory Efficiency**: No external classifier is required; memory overhead is limited to LoRA parameters (~0.1%), saving 2× memory versus external safety classifiers.
5. **Adaptive Window Effectiveness**: Adaptive $\gamma$ achieves a better balance between safety and efficiency compared to fixed-window variants.
6. **Cross-Architecture Consistency**: The method is effective across LLaMA-2, LLaMA-3.2, and Qwen-2.5 architectures.

## Highlights & Insights

- **Self-Defense Paradigm**: The idea of using the LLM itself as a safety detector is elegant, avoiding the overhead of external models.
- **Novelty of Mid-Generation Defense**: Most prior work focuses solely on pre-generation checking; MetaDefense is the first to systematically monitor safety during the generation process.
- **Insight into Embedding Space**: The paper reveals that LLMs can already distinguish harmful content in the embedding space, but explicit training is required to activate this capability.
- **Engineering Practicality**: LoRA finetuning with no external dependencies results in a low deployment barrier.

## Limitations & Future Work

1. **Adversarial Attacks**: Adversaries aware of MetaDefense's mechanism may design attacks that evade the diagnostic prompts.
2. **Latency Overhead**: Mid-generation checks introduce additional inference latency (one extra forward pass per $\gamma$ tokens).
3. **Model Scale**: Validation is limited to 3B–7B models; effectiveness on 70B+ models remains unknown.
4. **Non-Finetuning Attacks**: MetaDefense is specifically designed for FJAttack; its effectiveness against other attack types such as prompt injection is unclear.
5. **False Rejection Rate**: The paper provides limited discussion of false rejections for legitimate but sensitive topics (e.g., medical discussions).

## Related Work & Insights

- **Vaccine** (Huang et al., 2024): Injects safety "vaccines" during the alignment stage.
- **RepNoise** (Rosati et al., 2024): Adds noise in the representation space as a defense.
- **Booster** (Huang et al., 2024): Enhances safety alignment through finetuning strategies.
- **Llama Guard** (Meta, 2024): An external safety classifier serving as an alternative to MetaDefense.
- **Insight**: The self-defense paradigm can be extended to other safety scenarios, such as hallucination detection and bias identification.

## Rating

| Dimension | Score (1–5) | Notes |
|------|-----------|------|
| Novelty | 4.5 | Pre+Mid two-stage self-defense paradigm is novel |
| Technical Depth | 4 | Embedding space analysis combined with systematic defense design |
| Experimental Thoroughness | 4.5 | 3 models × 4 attacks × multiple baselines, with detailed ablations |
| Value | 4.5 | LoRA deployment, no external dependencies, directly applicable |
| Writing Quality | 4 | Clear structure with compelling motivation |
| **Overall** | **4.3** | Excellent work in LLM safety defense |

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Attack via Overfitting: 10-shot Benign Fine-tuning to Jailbreak LLMs](attack_via_overfitting_10-shot_benign_fine-tuning_to_jailbreak_llms.md)
- [\[ICCV 2025\] Heuristic-Induced Multimodal Risk Distribution Jailbreak Attack for Multimodal Large Language Models](../../ICCV2025/llm_alignment/heuristic-induced_multimodal_risk_distribution_jailbreak_attack_for_multimodal_l.md)
- [\[NeurIPS 2025\] Short-length Adversarial Training Helps LLMs Defend Long-length Jailbreak Attacks](short-length_adversarial_training_helps_llms_defend_long-length_jailbreak_attack.md)
- [\[NeurIPS 2025\] SafePTR: Token-Level Jailbreak Defense in Multimodal LLMs via Prune-then-Restore Mechanism](safeptr_token-level_jailbreak_defense_in_multimodal_llms_via_prune-then-restore_.md)
- [\[NeurIPS 2025\] GASP: Efficient Black-Box Generation of Adversarial Suffixes for Jailbreaking LLMs](gasp_efficient_black-box_generation_of_adversarial_suffixes_for_jailbreaking_llm.md)

<!-- RELATED:END -->
