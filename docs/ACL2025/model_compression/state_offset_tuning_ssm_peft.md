---
title: >-
  [Paper Note] State-offset Tuning: State-based Parameter-Efficient Fine-Tuning for State Space Models
description: >-
  [ACL 2025][Model Compression][state space model] This paper proposes State-offset Tuning, a novel family of "state-based" PEFT methods for SSMs (such as Mamba). By directly injecting a trainable state offset $h'$ at each time step rather than virtual tokens used in Prefix-Tuning, it overcomes the issue of limited expressivity of prompt-based approaches on SSMs, consistently outperforming LoRA and Prefix-Tuning with fewer parameters.
tags:
  - "ACL 2025"
  - "Model Compression"
  - "state space model"
  - "Mamba"
  - "PEFT"
  - "state-based tuning"
  - "parameter-efficient fine-tuning"
date: 2026-05-08
content_hash: 990805bfd6ae0247
---

# State-offset Tuning: State-based Parameter-Efficient Fine-Tuning for State Space Models

**Conference**: ACL 2025  
**arXiv**: [2503.03499](https://arxiv.org/abs/2503.03499)  
**Code**: [https://github.com/furiosa-ai/ssm-state-tuning](https://github.com/furiosa-ai/ssm-state-tuning)  
**Area**: Model Compression  
**Keywords**: state space model, Mamba, PEFT, state-based tuning, parameter-efficient fine-tuning

## TL;DR
This paper proposes State-offset Tuning, a novel family of "state-based" PEFT methods for SSMs (such as Mamba). By directly injecting a trainable state offset $h'$ at each time step rather than virtual tokens used in Prefix-Tuning, it overcomes the issue of limited expressivity of prompt-based approaches on SSMs, consistently outperforming LoRA and Prefix-Tuning with fewer parameters.

## Background & Motivation
**Background**: SSMs (e.g., Mamba) are emerging as sub-quadratic alternatives to Transformers, but PEFT methods for SSMs remain under-explored.

**Limitations of Prior Work**:
   - Prompt Tuning and Prefix-Tuning, while effective on Transformers, perform poorly on SSMs because virtual tokens can only affect the initial state $h_0$, and their influence decays exponentially as the time step $t$ increases ($\bar{A}^t h_0$).
   - Although LoRA is effective, it does not exploit the unique architectural characteristics of SSMs.

**Key Challenge**: The recurrent nature of SSMs causes the influence of prompt-based methods to decay over time, whereas Prefix-Tuning in Transformers directly impacts every step in every layer.

**Goal**: To design PEFT methods for SSMs that utilize their unique architectural layout.

**Key Insight**: Directly modify the hidden states of SSMs, instead of indirectly influencing them via external prompts.

**Core Idea**: Add a trainable offset $h'$ directly to the hidden state at each time step, eliminating the temporal decay issue of Prefix-Tuning.

## Method

### Overall Architecture
"State-based methods" are defined as a category of PEFT methods that directly modify the inner state features of SSMs. Two State-offset Tuning variants are proposed: (1) $h$-offset: added to the hidden states as $\hat{y}_t = y_t + C_t h'$; (2) $y$-offset: added directly to the outputs as $\hat{y}_t = y_t + y'$.

### Key Designs

1. **Analysis of Prefix-Tuning Limitations on SSMs**:

    - Function: Mathematically proving that Prefix-Tuning is equivalent to Initial State Tuning.
    - Mechanism: The effect of virtual tokens is $\bar{A}^t h_{\text{prefix}}$, which decays exponentially with $t$ and can only affect the initial state.
    - Design Motivation: Explains why prompt-based PEFT performs poorly on SSMs: the influence decays too rapidly.

2. **State-offset Tuning (h-offset)**:

    - Function: Adds a position-independent trainable offset to the SSM output at each time step.
    - Mechanism: $\hat{y}_t = y_t + C_t h'$, where $h' \in \mathbb{R}^H$ represents trainable parameters (shared across channels).
    - Design Motivation: Eliminates the temporal decay of $\bar{A}^t$, meaning the offset has a uniform impact at every step. Parameter count is only $D \cdot H$.
    - Difference from Initial State Tuning: Initial State tuning yields an effect of $C_t (\prod \bar{A}_i) h'$ (decaying), whereas State-offset yields $C_t h'$ (non-decaying).

3. **State-offset Tuning (y-offset)**:

    - Function: Directly adds an offset to the scalar output of the SSM.
    - Mechanism: $\hat{y}_t = y_t + y'$, with a parameter count of under $D$ (one scalar per channel).
    - More extremely parameter-efficient, though slightly less expressive.

## Key Experimental Results

### Main Results

**Mamba-2.8B, Various Downstream Tasks**:

| Method | Parameter Size | Average Performance |
|------|:-----:|:------:|
| Full FT | 100% | Benchmark Upper Bound |
| LoRA | ~0.5% | Medium |
| Prefix-Tuning | ~0.3% | Poor |
| Initial State Tuning | ~0.01% | Medium-Low |
| **State-offset (h)** | **~0.01%** | **Outperforms LoRA** |
| **State-offset (y)** | **~0.001%** | **Close to LoRA** |

State-offset achieves superior or competitive performance with significantly fewer parameters than LoRA.

### Key Findings
- Prompt-based methods on SSMs indeed underperform compared to Transformers, validating the theoretical analysis on temporal decay.
- The uniform impact of State-offset is crucial; eliminating the decay significantly boosts performance.
- h-offset outperforms y-offset, as it retains interaction with inputs through the $C_t$ matrix.

## Highlights & Insights
- **Pioneering definition of the "state-based PEFT" family for SSMs**: Designing PEFT methods derived directly from SSM structural characteristics rather than adaptions from Transformers.
- **Clear theoretical analysis**: The proof showing that Prefix-Tuning is equivalent to Initial State Tuning is concise and convincing, directly explaining its ineffectiveness on SSMs.
- **Minimalist design**: State-offset only adds $D \times H$ parameters ($h'$) per layer, making it one of the most parameter-efficient PEFT methods for SSMs.

## Limitations & Future Work
- **Only evaluated on Mamba**: The applicability to other SSM variants (such as RWKV or RetNet) remains untested.
- **Position-independent offsets**: Applying the same $h'$ at all time steps might limit adaptability to position-sensitive tasks.
- **Potentially combinable with LoRA**: Applying State-offset in the SSM blocks alongside LoRA in linear layers may yield superior results.

## Related Work & Insights
- **vs. LoRA on SSMs**: LoRA does not exploit the structure of SSMs, whereas State-offset is directly derived from the state equations of SSMs.
- **vs. Prefix-Tuning**: Prefix-Tuning on SSMs is equivalent to Initial State Tuning and suffers from decay, whereas State-offset suffers from no decay.
- With the growing adoption of SSM models (such as Mamba-2 and Jamba), state-based PEFT is expected to play an increasingly important role.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Pioneering work designing PEFT from SSM architectural properties, with entirely original theory and methodology.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across various SSM configurations and downstream tasks.
- Writing Quality: ⭐⭐⭐⭐⭐ Elegant theoretical analysis and clear mathematical proofs.
- Value: ⭐⭐⭐⭐ PEFT for SSMs is a nascent direction; State-offset provides a simple and effective foundation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Parameter-Efficient Fine-Tuning of State Space Models](../../ICML2025/model_compression/parameter-efficient_fine-tuning_of_state_space_models.md)
- [\[ACL 2025\] C3A: Parameter-Efficient Fine-Tuning via Circular Convolution](parameter-efficient_fine-tuning_via_circular_convolution.md)
- [\[ACL 2025\] Trans-PEFT: Transferable Parameter-Efficient Fine-Tuning on Evolving Base Models](trans_peft_transferable.md)
- [\[ACL 2025\] Quaff: Quantized Parameter-Efficient Fine-Tuning under Outlier Spatial Stability Hypothesis](quaff_quantized_peft.md)
- [\[CVPR 2025\] MambaIC: State Space Models for High-Performance Learned Image Compression](../../CVPR2025/model_compression/mambaic_state_space_models_for_high-performance_learned_image_compression.md)

</div>

<!-- RELATED:END -->
