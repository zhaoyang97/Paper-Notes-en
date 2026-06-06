---
title: >-
  [Paper Note] Compositional Steering of Large Language Models with Steering Tokens
description: >-
  [ACL 2026][Interpretability][Compositional steering] This paper proposes compositional steering tokens, which compress behavioral instructions into embedding vectors in the input space via self-distillation. By training…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Compositional steering"
  - "steering tokens"
  - "self-distillation"
  - "multi-behavior control"
  - "zero-shot composition"
date: 2026-05-08
content_hash: 52ac7b66414852e4
---

# Compositional Steering of Large Language Models with Steering Tokens

**Conference**: ACL 2026  
**arXiv**: [2601.05062](https://arxiv.org/abs/2601.05062)  
**Code**: None  
**Area**: Model Compression  
**Keywords**: Compositional steering, steering tokens, self-distillation, multi-behavior control, zero-shot composition

## TL;DR

This paper proposes compositional steering tokens, which compress behavioral instructions into embedding vectors in the input space via self-distillation. By training a dedicated compositional token `<and>` to capture the universal concept of "composition," the method demonstrates strong generalization capabilities across unseen behavior combinations, unseen behaviors, and unseen numbers of compositions.

## Background & Motivation

**Background**: LLM deployment requires simultaneously satisfying multiple behavioral constraints (e.g., language, length, format). Fine-tuning is computationally expensive and may damage general capabilities, and $N$ behaviors imply $2^N$ possible combinations for fine-tuning. Instruction steering is flexible but fragile—semantically equivalent prompts often produce inconsistent behaviors.

**Limitations of Prior Work**: (1) Activation space steering methods (e.g., CAA) combine behaviors through vector addition, but directly combining independently trained modules is disruptive; (2) Gist tokens only handle single-behavior compression, leaving the composition problem unsolved; (3) Existing compositional steering lacks rigorous evaluation—most provide only anecdotal evidence or lack baseline comparisons.

**Key Challenge**: Independently trained behavior representations produce interference when combined, yet training for every specific combination leads to combinatorial explosion. A representation is needed that can learn the concept of "composition" itself, rather than every specific combination.

**Goal**: To learn a universal compositional token `<and>` that generalizes across unseen behavior combinations, including unseen behaviors and unseen numbers of compositions.

**Key Insight**: Behavior representations are placed in the input space (rather than the activation space) to support better zero-shot composition. During the training of the compositional token, behavior tokens are frozen to force `<and>` to learn a behavior-agnostic composition function.

**Core Idea**: Composition equals a learnable universal operator, rather than specific adjustments for every pair of behaviors.

## Method

### Overall Architecture

Ours utilizes a two-stage training process: (1) Each behavior's steering token `<b>` is trained independently via self-distillation—the teacher receives the instruction text while the student receives the steering token, minimizing KL divergence; (2) The LLM and behavior tokens are frozen, and the compositional token `<and>` is trained on behavior pairs—the teacher receives two instructions while the student receives `[bi, <and>, bj]`. During inference, the composition is represented as $[\mathbf{E}_x, \mathbf{e}_{b_i}, \mathbf{e}_{\text{<and>}}, \mathbf{e}_{b_j}]$.

### Key Designs

1. **Input Space Steering Tokens**:

    - **Function**: Compresses behavioral instructions into a single input embedding vector.
    - **Mechanism**: The steering token $\mathbf{e}_b \in \mathbb{R}^d$ exists in the model's input embedding space (not the intermediate activation space) and is trained via self-distillation: minimize $\text{KL}(P_{\text{teacher}}(y|x, I_b) \| P_{\text{student}}(y|x, \texttt{<b>}))$. Ten instruction rewrites are used to prevent overfitting.
    - **Design Motivation**: Input space behavior representations are more suitable for composition than activation space representations—direct addition in the activation space easily causes interference.

2. **Universal Compositional Token `<and>`**:

    - **Function**: Learns the behavior-agnostic concept of "composition."
    - **Mechanism**: The $\mathbf{e}_{\text{<and>}}$ is trained on behavior pairs while behavior tokens are frozen—ensuring that `<and>` learns the composition function itself rather than modifying individual behaviors. It utilizes zero initialization (to avoid bias towards any behavior) and orthogonal regularization (to prevent collapsing into the behavior token space).
    - **Design Motivation**: If `<and>` could modify behavior tokens during training, it would only learn adjustments for specific combinations rather than a general compositional capability.

3. **Orthogonal Regularization**:

    - **Function**: Prevents the representations of compositional tokens and behavior tokens from collapsing.
    - **Mechanism**: $\mathcal{L}_{\text{orth}} = \sum_{b \in \mathcal{B}_{\text{seen}}} (\frac{\mathbf{e}_{\text{<and>}} \cdot \mathbf{e}_b}{\|\mathbf{e}_{\text{<and>}}\| \cdot \|\mathbf{e}_b\|})^2$, with the final loss being $\mathcal{L} = \mathcal{L}_{\text{dist}} + \lambda \cdot \mathcal{L}_{\text{orth}}$.
    - **Design Motivation**: Ablation studies prove that orthogonal regularization is crucial for zero-shot composition generalization.

### Loss & Training

Self-distillation loss combined with orthogonal regularization. The LLM is completely frozen; only $|\mathcal{B}|+1$ vectors of dimension $d$ are learned. A temperature of $T=10.0$ encourages matching the full probability distribution. Behavior tokens are initialized semantically (mean of instruction token embeddings), while the compositional token is zero-initialized.

## Key Experimental Results

### Main Results

**2-Behavior Composition Accuracy on Qwen3-8B (%)**

| Method | Seen Combi. | Unseen Combi. | Order Variance ↓ |
|:---|:---:|:---:|:---:|
| CAA (Activation Steering) | 1.6 | 0.5 | - |
| LM-Steer | 18.1 | 13.4 | - |
| LoRA DARE | 81.5 | 44.8 | - |
| Instruction Steering | 86.2 | 67.3 | 12.3 |
| **Steering Tokens (Ours)** | **90.5** | **75.5** | **4.8** |
| **Steering Tokens + Instructions** | **92.0** | **80.3** | **3.5** |

### Ablation Study

| Config | Seen Combi. | Unseen Combi. | Explanation |
|:---|:---:|:---:|:---|
| No `<and>` token (Direct Concat) | Decrease | Sharp Decrease | Compositional token is critical |
| No Orthogonal Regularization | Slight Decrease | Significant Decrease | Orthogonality is important for generalization |
| Randomly Init `<and>` | Slight Decrease | Decrease | Zero initialization is superior |
| Only 2-behavior Training | - | Generalize to 3-behavior | Composition concept is generalizable |

### Key Findings

- Steering tokens significantly outperform activation steering methods on both seen and unseen combinations (CAA: 1.6% vs. Steering Tokens: 90.5%).
- The compositional token successfully generalizes to: unseen behavior combinations, combinations containing unseen behaviors, and 3-behavior combinations (even though only 2-behavior combinations were trained).
- The hybrid method of Steering Tokens + Instructions is optimal across all settings, showing complementarity.
- Compositional accuracy and robustness improve as the model scale increases (4B → 8B → 14B).
- The order variance of steering tokens is much lower than that of instruction steering, indicating more stable behavior representation.

## Highlights & Insights

- The idea of "learning a compositional operator instead of every combination" is robust and simple—similar to learning a function versus memorizing a table.
- Freezing behavior tokens while training `<and>` is a critical design decision that ensures generalization rather than overfitting.
- The complementarity between steering tokens and instructions is surprising—compressed representations and natural language provide different types of control signals.

## Limitations & Future Work

- Evaluation is limited to automatically verifiable constraints (length, format, language); subjective behaviors (e.g., style, tone) are not covered.
- Each behavior requires independent training of a steering token; training costs increase linearly with the number of behaviors.
- The compositional token was only trained on 2-behavior combinations; performance may degrade for combinations of many more behaviors.
- Dependency on self-distillation quality—if the teacher (instruction steering) does not follow instructions, the student cannot learn effectively.

## Related Work & Insights

- **vs. CAA/Rimsky et al.**: Activation space steering suffers from heavy interference during composition (1.6%), whereas input space steering tokens perform decisively better.
- **vs. Gist token**: Gist tokens only compress single instructions and do not solve the composition problem.
- **vs. LoRA merging**: LoRA DARE is competitive on seen combinations (81.5%) but generalizes poorly to unseen combinations (44.8%).

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The concept of a universal compositional operator and the frozen training design are highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Extremely comprehensive with seven models, 15 behaviors, various composition settings, and over 1 million evaluations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear motivation, elegant methodology, and rigorous experimental design.
- **Value**: ⭐⭐⭐⭐⭐ Provides a simple and effective new paradigm for multi-behavior controllable generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] FineSteer: A Unified Framework for Fine-Grained Inference-Time Steering in Large Language Models](finesteer_a_unified_framework_for_fine-grained_inference-time_steering_in_large_.md)
- [\[CVPR 2026\] Language Models Can Explain Visual Features via Steering](../../CVPR2026/interpretability/language_models_can_explain_visual_features_via_steering.md)
- [\[ACL 2026\] From Weights to Activations: Is Steering the Next Frontier of Adaptation?](from_weights_to_activations_is_steering_the_next_frontier_of_adaptation.md)
- [\[ICML 2026\] The Cylindrical Representation Hypothesis for Language Model Steering](../../ICML2026/interpretability/the_cylindrical_representation_hypothesis_for_language_model_steering.md)
- [\[ACL 2026\] Knowledge Vector of Logical Reasoning in Large Language Models](knowledge_vector_of_logical_reasoning_in_large_language_models.md)

</div>

<!-- RELATED:END -->
