---
title: >-
  [Paper Note] Where to Start Alignment? Diffusion Large Language Model May Demand a Distinct Position
description: >-
  [AAAI 2026 (Oral)][Reinforcement Learning][Diffusion language model] This paper presents the first systematic safety analysis of diffusion large language models (dLLMs), revealing that—unlike autoregressive LLMs—**middle tokens** are more critical to safety in dLLMs, and that attackers are fundamentally constrained by the model's inherent sequential generation tendency from manipulating these positions. Based on this asymmetry, the paper proposes MOSA (Middle-tOken Safety Alignment) as a defense method.
tags:
  - AAAI 2026 (Oral)
  - Reinforcement Learning
  - Diffusion language model
  - safety alignment
  - middle tokens
  - jailbreak defense
date: 2026-05-08
content_hash: 6a697fc50f681297
---

# Where to Start Alignment? Diffusion Large Language Model May Demand a Distinct Position

**Conference**: AAAI 2026 (Oral)  
**arXiv**: [2508.12398](https://arxiv.org/abs/2508.12398)  
**Code**: None  
**Area**: AI Safety / Large Language Model Alignment  
**Keywords**: Diffusion language model, safety alignment, middle tokens, jailbreak defense, reinforcement learning

## TL;DR

This paper presents the first systematic safety analysis of diffusion large language models (dLLMs), revealing that—unlike autoregressive LLMs—**middle tokens** are more critical to safety in dLLMs, and that attackers are fundamentally constrained by the model's inherent sequential generation tendency from manipulating these positions. Based on this asymmetry, the paper proposes MOSA (Middle-tOken Safety Alignment) as a defense method.

## Background & Motivation

### Safety Symmetry in Autoregressive LLMs

In conventional autoregressive LLMs (e.g., Llama-3, GPT-4), safety alignment exhibits a phenomenon known as "Shallow Safety Alignment (SSA)": safety fine-tuning primarily concentrates on the first few tokens of a response. This leads to a **symmetric competition** in which both attackers and defenders compete for control over initial tokens:
- Attackers: force the model to begin with affirmative prefixes (e.g., "Sure, here is...")
- Defenders: reinforce the model to begin with refusal prefixes (e.g., "I cannot...")

### The Paradigm Shift in dLLMs

Diffusion large language models (e.g., LLaDA, DREAM) employ a fundamentally different inference mechanism: starting from a fully masked sequence and iteratively predicting content over multiple rounds. In principle, dLLMs can fill tokens at arbitrary positions without strict left-to-right constraints.

**Core Question**: Does the conventional "first-token-centric" safety analysis still apply to dLLMs?

### Three Core Findings

Through systematic experiments, the authors uncover a distinctive **safety asymmetry** in dLLMs:

**Finding 1: Middle tokens are more critical to safety than initial tokens**
- Prefilling phrases at different positions reveals that when an affirmative phrase is prefilled at the initial position, the model tends to recover and produce a refusal; however, when a procedural phrase is prefilled at a middle position (tokens 40–160), the model abandons the safe opening and continues with harmful content.
- Jailbreak success rate increases significantly as the prefill position shifts toward the middle.

**Finding 2: Attackers struggle to manipulate middle tokens**
- GCG attacks (considered an upper bound on manipulation capability) yield a 33% attack success rate at initial token positions but only 2% at middle positions.
- Optimization loss remains persistently high at middle positions, reflecting a fundamental barrier to attacker manipulation.

**Finding 3: dLLMs exhibit an inherent sequential generation preference**
- Despite architectural freedom for non-sequential generation, dLLMs exhibit a strong left-to-right generation tendency in practice.
- The average position of newly unmasked tokens correlates nearly linearly with decoding steps.
- This preference is independent of input type (benign and adversarial prompts exhibit identical behavior).

## Method

### Overall Architecture

MOSA operates under a reinforcement learning paradigm, aligning middle tokens of responses with predefined safe refusal sequences through contrastive reward signals. The overall framework comprises: contrastive reward computation + KL divergence penalty + lightweight LoRA fine-tuning.

### Key Designs

#### 1. **Middle Token Window Definition**

Tokens 20 through 60 are designated as the middle token window. The window boundaries are motivated as follows:
- **Lower bound (token 20)**: sufficiently distant from the initial token region where attacker influence is strong
- **Upper bound (token 60)**: early enough to truncate the response and limit potential harm

**Design Motivation**: This design exploits the safety asymmetry—defenders can directly intervene in middle tokens during training, whereas attackers are constrained by the sequential generation preference and cannot easily reach these positions.

#### 2. **Contrastive Reward Function**

A positive sample set $S_{safe\_set}$ (safe refusal sentences, e.g., "therefore, I cannot answer this question," including the EOS token) and a negative sample set $S_{harmful\_set}$ are defined.

At each training step, one positive sample $s_{pos}$ and one negative sample $s_{neg}$ are randomly selected; all contiguous segments within the target window are searched to compute the maximum log-likelihood:

$$R_{pos} = \text{GetMaxScore}(s_{pos}, L_\theta, [k_{start}, k_{end}])$$
$$R_{neg} = \text{GetMaxScore}(s_{neg}, L_\theta, [k_{start}, k_{end}])$$
$$R_{contrastive} = R_{pos} - R_{neg}$$

where GetMaxScore iterates over all contiguous spans within the window whose length matches the target sentence and returns the maximum log-likelihood value.

**Key Design**: Positive samples include an EOS token as a "circuit breaker"—even if the initial tokens are compromised, the safe sentence in the middle can truncate output length via EOS, bounding the scope of harm.

#### 3. **KL Divergence Regularization**

The complete reward function is:

$$\mathcal{L}(\theta) = \mathbb{E}_{x \sim D}\left[R(y|x) - \beta \cdot D_{KL}(P_\theta(y|x) \| P_{ref}(y|x))\right]$$

where $\beta$ controls the KL penalty strength (set to 0.05), preventing the policy model from deviating too far from the reference model and losing general capabilities.

**Design Motivation**: Naive safety alignment may cause performance degradation on normal tasks. The KL penalty ensures that alignment is "precise"—triggered only when harmful prompts are encountered, without affecting normal usage.

### Loss & Training

- **Base model**: LLaDA-8B-Instruct
- **Training data**: 3,000 harmful questions randomly sampled from SORRY-Bench
- **Fine-tuning**: LoRA (r=32, α=64), trained for 1 epoch only
- **Optimizer**: AdamW, learning rate 5e-5, gradient clipping 0.01
- **Computational cost**: Dual A100-40GB GPUs, approximately 12 minutes to complete training
- Reward rises rapidly within ~500 steps and stabilizes at 15–18

## Key Experimental Results

### Main Results

Attack success rate (ASR%, lower is better) against 8 jailbreak attacks on AdvBench:

| Attack Method | Base Model | Initial Alignment | MOSA (Ours) |
|---------------|-----------|-------------------|-------------|
| Avatar | 74.5 | 23.5 | **14.3** |
| TAP | 79.1 | 29.6 | **4.5** |
| Speakeasy | 69.8 | 22.4 | **8.1** |
| AOS | 65.2 | 32.4 | **6.5** |
| PAL | 72.8 | 36.4 | **6.2** |
| EPT | 78.4 | 28.5 | **3.8** |
| DIA | 66.7 | 34.3 | **4.2** |
| AdvPrefix | 79.5 | 29.8 | **6.8** |

Results on HarmBench are similar; MOSA reduces ASR to single digits for most attacks.

### General Capability Retention

| Model | GSM8K | MMLU | HumanEval |
|-------|-------|------|-----------|
| Base Model | 69.8 | 66.4 | 32.8 |
| Initial Alignment | 67.4 | 68.2 | 29.6 |
| **MOSA** | **68.3** | **65.9** | **30.4** |

MOSA has minimal impact on general capabilities, remaining on par with the base model.

### Ablation Study / Adaptive Attacks

| Configuration | TAP ASR | EPT ASR | AdvPrefix ASR |
|---------------|---------|---------|---------------|
| Initial Alignment (first-token only) | ~29% | ~28% | ~30% |
| MOSA (middle-token alignment) | **5.1%** | **3.8%** | **4.5%** |
| Adaptive attack (targeting middle tokens) | 5.1% | 3.8% | 4.5% |

Adaptive attack ASR is equally low as that of standard attacks, demonstrating that MOSA does not merely shift the vulnerable position but leverages an architectural safety advantage.

### Key Findings

1. The safety-critical position in dLLMs shifts from the "first token" to the "middle tokens."
2. Attackers are constrained by the sequential generation preference and have extremely limited ability to manipulate middle tokens (GCG attack success rate: only 2%).
3. This safety asymmetry is a general property of the dLLM paradigm—the same phenomenon is observed on Dream 7B and MMaDA.
4. MOSA is a lightweight fine-tuning approach that does not alter the model's inherent sequential generation preference, thereby preserving the safety asymmetry.
5. The "middle-token alignment" strategy **does not apply to autoregressive LLMs**—because in AR models each token causally depends on preceding ones, middle tokens cannot be aligned independently of the beginning.

## Highlights & Insights

1. **First dLLM safety analysis**: Fills an important research gap, as dLLMs represent an emerging paradigm urgently requiring safety investigation.
2. **Discovery of safety asymmetry**: This core insight is highly original—attackers and defenders have asymmetric capabilities in dLLMs, which stands in stark contrast to AR-LLMs.
3. **EOS circuit breaker design**: Embedding EOS tokens in safe sentences simultaneously ensures safety and controls output length, achieving two objectives at once.
4. **Highly efficient training**: Achievable in 12 minutes with 3,000 data samples, offering strong practical utility.
5. **Implications beyond safety**: The authors suggest that an "anchor-then-fill" strategy—first generating critical middle formulas and then completing the surrounding content—may serve as a general approach to unlocking dLLM potential.

## Limitations & Future Work

1. Defense against narrative-wrapping attacks (e.g., Avatar, Speakeasy) is relatively weaker, due to insufficient diversity of disguised malicious prompts in the training data.
2. The middle token window (tokens 20–60) is manually specified and does not adapt to different tasks or models.
3. Validation is limited to LLaDA-8B; the approach has not been tested on larger-scale dLLMs or commercial dLLMs (e.g., Gemini Diffusion).
4. Future directions: investigating the linear representation of "benign/harmful" concepts in dLLM activation space and developing more robust defenses.

## Related Work & Insights

- **Relationship to AR-LLM safety research**: This work reveals that dLLMs require an entirely different safety paradigm and cannot simply adopt strategies developed for AR models.
- **Contrast with SSA (Shallow Safety Alignment)**: MOSA constitutes "deep safety alignment," operating at the genuinely safety-critical region of the model.
- **Broader insight**: The sequential generation preference of dLLMs is itself a compelling phenomenon—why do non-autoregressive models "learn" autoregressive behavior? This may be a natural consequence of the training objective, as predicting adjacent tokens exhibits lower variance.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First safety analysis of dLLMs; the discovery of safety asymmetry is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 8 attacks × 2 benchmarks, with adaptive attacks and general capability evaluation fully covered.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Analysis proceeds in well-structured layers with excellent figure-text coordination and a clear logical chain.
- **Value**: ⭐⭐⭐⭐ — Directly applicable to safe deployment of dLLMs, though coverage of dLLM variants remains limited.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Language Model Distillation: A Temporal Difference Imitation Learning Perspective](language_model_distillation_a_temporal_difference_imitation_learning_perspective.md)
- [\[AAAI 2026\] Formal Verification of Diffusion Auctions](formal_verification_of_diffusion_auctions.md)
- [\[ICLR 2026\] AWM: Accurate Weight-Matrix Fingerprint for Large Language Models](../../ICLR2026/reinforcement_learning/awm_accurate_weight-matrix_fingerprint_for_large_language_models.md)
- [\[ICLR 2026\] VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models](../../ICLR2026/reinforcement_learning/verifybench_benchmarking_reference-based_reward_systems_for_large_language_model.md)
- [\[ICLR 2026\] Robust Multi-Objective Controlled Decoding of Large Language Models](../../ICLR2026/reinforcement_learning/robust_multi-objective_controlled_decoding_of_large_language_models.md)

<!-- RELATED:END -->
