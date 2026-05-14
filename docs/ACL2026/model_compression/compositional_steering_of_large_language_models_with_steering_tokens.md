---
title: >-
  [Paper Note] Compositional Steering of Large Language Models with Steering Tokens
description: >-
  [ACL 2026][Model Compression][Compositional Steering] This paper proposes compositional steering tokens that compress behavioral instructions into input-space embedding vectors via self-distillation…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "Compositional Steering"
  - "Steering Tokens"
  - "Self-Distillation"
  - "Multi-Behavior Control"
  - "Zero-Shot Composition"
date: 2026-05-08
content_hash: 3bd93e0c2f9d8a31
---

# Compositional Steering of Large Language Models with Steering Tokens

**Conference**: ACL 2026
**arXiv**: [2601.05062](https://arxiv.org/abs/2601.05062)
**Code**: None
**Area**: Model Compression
**Keywords**: Compositional Steering, Steering Tokens, Self-Distillation, Multi-Behavior Control, Zero-Shot Composition

## TL;DR

This paper proposes compositional steering tokens that compress behavioral instructions into input-space embedding vectors via self-distillation, and trains a dedicated composition token <and> to capture the general concept of "composition." The approach demonstrates strong generalization to unseen behavior combinations, unseen behaviors, and unseen numbers of behaviors to compose.

## Background & Motivation

**Background**: LLM deployment requires satisfying multiple simultaneous behavioral constraints (e.g., language, length, format). Fine-tuning is computationally expensive and may degrade general capabilities; arbitrary combinations of $N$ behaviors imply $2^N$ fine-tuning runs. Instruction-based steering is flexible but brittle—semantically equivalent prompts yield inconsistent behavior.

**Limitations of Prior Work**: (1) Activation-space steering methods (e.g., CAA) compose behaviors via vector addition, but directly combining independently trained modules is disruptive. (2) Gist tokens address only single-behavior compression and do not tackle compositional generalization. (3) Existing compositional steering lacks rigorous evaluation—most work provides only anecdotal evidence or omits baseline comparisons.

**Key Challenge**: Independently trained behavioral representations interfere when composed, yet training separately for each combination leads to combinatorial explosion. A representation that learns the concept of "composition" itself—rather than each specific combination—is needed.

**Goal**: Learn a universal composition token <and> that generalizes to unseen behavior combinations, including combinations involving unseen behaviors and unseen numbers of behaviors.

**Key Insight**: Placing behavioral representations in the input space (rather than the activation space) facilitates better zero-shot composition; freezing behavior tokens during composition token training forces <and> to learn a behavior-agnostic composition function.

**Core Idea**: Composition is treated as a learnable universal operator rather than a behavior-pair-specific adjustment.

## Method

### Overall Architecture

Training proceeds in two stages. (1) Each behavior's steering token <b> is trained independently via self-distillation—a teacher receives the instruction text while a student receives the steering token, with training minimizing KL divergence between their output distributions. (2) The LLM and behavior tokens are frozen; a composition token <and> is then trained on pairs of behaviors, where the teacher receives two instructions and the student receives $\texttt{<b}_i\texttt{><and><b}_j\texttt{>}$. At inference, the input sequence is $[\mathbf{E}_x, \mathbf{e}_{b_i}, \mathbf{e}_{\text{<and>}}, \mathbf{e}_{b_j}]$.

### Key Designs

1. **Input-Space Steering Tokens**:

    - **Function**: Compress behavioral instructions into a single input embedding vector.
    - **Mechanism**: A steering token $\mathbf{e}_b \in \mathbb{R}^d$ resides in the model's input embedding space (not intermediate activations) and is trained via self-distillation by minimizing $\text{KL}(P_{\text{teacher}}(y|x, I_b) \| P_{\text{student}}(y|x, \texttt{<b>}))$. Ten instruction paraphrases are used to prevent overfitting.
    - **Design Motivation**: Input-space behavioral representations are better suited for composition than activation-space ones, where direct vector addition tends to cause destructive interference.

2. **Universal Composition Token <and>**:

    - **Function**: Learn the concept of "composition" independent of specific behaviors.
    - **Mechanism**: $\mathbf{e}_{\text{<and>}}$ is trained on behavior pairs with behavior tokens frozen—this ensures that <and> learns the composition function itself rather than adjustments to individual behaviors. Zero initialization (to avoid bias toward any behavior) and orthogonal regularization (to prevent collapse into the behavior token space) are applied.
    - **Design Motivation**: If <and> were allowed to modify behavior tokens during training, it would learn only specific combinatorial adjustments rather than a general compositional capability.

3. **Orthogonal Regularization**:

    - **Function**: Prevent the composition token from collapsing into the behavior token representation space.
    - **Mechanism**: $\mathcal{L}_{\text{orth}} = \sum_{b \in \mathcal{B}_{\text{seen}}} \left(\frac{\mathbf{e}_{\text{<and>}} \cdot \mathbf{e}_b}{\|\mathbf{e}_{\text{<and>}}\| \cdot \|\mathbf{e}_b\|}\right)^2$, with the total loss $\mathcal{L} = \mathcal{L}_{\text{dist}} + \lambda \cdot \mathcal{L}_{\text{orth}}$.
    - **Design Motivation**: Ablation experiments confirm that orthogonal regularization is critical for zero-shot compositional generalization.

### Loss & Training

The training objective combines self-distillation loss and orthogonal regularization. The LLM is fully frozen; only $|\mathcal{B}|+1$ vectors of dimension $d$ are learned. A temperature of $T=10.0$ encourages matching the full output distribution. Behavior tokens are semantically initialized (as the mean of instruction token embeddings); the composition token is zero-initialized.

## Key Experimental Results

### Main Results

**2-Behavior Composition Accuracy (%) on Qwen3-8B**

| Method | Seen Combinations | Unseen Combinations | Order Variance ↓ |
|--------|-------------------|---------------------|------------------|
| CAA (Activation Steering) | 1.6 | 0.5 | - |
| LM-Steer | 18.1 | 13.4 | - |
| LoRA DARE | 81.5 | 44.8 | - |
| Instruction Steering | 86.2 | 67.3 | 12.3 |
| **Steering Tokens** | **90.5** | **75.5** | **4.8** |
| **Steering Tokens + Instructions** | **92.0** | **80.3** | **3.5** |

### Ablation Study

| Configuration | Seen Combinations | Unseen Combinations | Notes |
|---------------|-------------------|---------------------|-------|
| No <and> token (direct concatenation) | Degraded | Significantly degraded | Composition token is critical |
| No orthogonal regularization | Slightly degraded | Noticeably degraded | Orthogonality important for generalization |
| Random initialization of <and> | Slightly degraded | Degraded | Zero initialization is superior |
| Trained on 2-behavior only | — | Generalizes to 3-behavior | Compositional concept transfers |

### Key Findings

- Steering tokens substantially outperform activation-space steering on both seen and unseen combinations (CAA: 1.6% vs. steering tokens: 90.5%).
- The composition token successfully generalizes to: unseen behavior combinations, combinations involving unseen behaviors, and 3-behavior compositions (despite training only on 2-behavior combinations).
- The hybrid approach (steering tokens + instructions) achieves the best performance across all settings, indicating complementarity between the two control signals.
- Composition accuracy and robustness improve with model scale (4B → 8B → 14B).
- Steering tokens exhibit substantially lower order variance than instruction steering, reflecting more stable behavioral representations.

## Highlights & Insights

- The idea of "learning a composition operator rather than memorizing each combination" is both concise and powerful—analogous to learning a function versus storing a lookup table.
- Freezing behavior tokens during <and> training is the critical design decision that enables generalization rather than overfitting.
- The complementarity between steering tokens and instructions is noteworthy—compressed representations and natural language provide distinct types of control signals.

## Limitations & Future Work

- Evaluation is restricted to automatically verifiable constraints (length, format, language); subjective behaviors such as style and tone are not covered.
- Each behavior requires independently trained steering tokens, resulting in training costs that scale linearly with the number of behaviors.
- The composition token is trained only on 2-behavior combinations; performance on larger compositions may degrade.
- The approach depends on the quality of self-distillation—if the teacher (instruction steering) itself fails to follow instructions reliably, the student cannot be trained effectively.

## Related Work & Insights

- **vs. CAA / Rimsky et al.**: Activation-space steering suffers severe interference during composition (1.6%), while input-space steering tokens decisively outperform it.
- **vs. Gist Tokens**: Gist tokens compress only single instructions and do not address the compositional generalization problem.
- **vs. LoRA Merging**: LoRA DARE is competitive on seen combinations (81.5%) but generalizes poorly to unseen ones (44.8%).

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The concept of a universal composition operator and the frozen-training design are highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Seven models, 15 behaviors, multiple composition settings, and over 1M evaluations—remarkably comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is clear, the method is elegant, and the experimental design is rigorous.
- **Value**: ⭐⭐⭐⭐⭐ Provides a concise and effective new paradigm for multi-behavior controllable generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] From Weights to Activations: Is Steering the Next Frontier of Adaptation?](from_weights_to_activations_is_steering_the_next_frontier_of_adaptation.md)
- [\[AAAI 2026\] Steering Pretrained Drafters during Speculative Decoding](../../AAAI2026/model_compression/steering_pretrained_drafters_during_speculative_decoding.md)
- [\[ICLR 2026\] Steering MoE LLMs via Expert (De)Activation](../../ICLR2026/model_compression/steering_moe_llms_via_expert_deactivation.md)
- [\[ACL 2026\] SeLaR: Selective Latent Reasoning in Large Language Models](selar_selective_latent_reasoning_in_large_language_models.md)
- [\[ACL 2026\] JudgeMeNot: Personalizing Large Language Models to Emulate Judicial Reasoning in Hebrew](judgemenot_personalizing_large_language_models_to_emulate_judicial_reasoning_in_.md)

</div>

<!-- RELATED:END -->
