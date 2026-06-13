---
title: >-
  [Paper Note] DeCoVec: Building Decoding Space based Task Vector for Large Language Models via In-Context Learning
description: >-
  [ACL 2026][LLM/NLP][Task Vector] This paper proposes DeCoVec (Decoding Space based Task Vector), a training-free and non-invasive framework that constructs task vectors in the decoding space by contrasting the output log…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "Task Vector"
  - "Decoding Space"
  - "In-Context Learning"
  - "Training-free LLM Guiding"
  - "logit manipulation"
date: 2026-05-08
content_hash: 8479b8d25b66db73
---

# DeCoVec: Building Decoding Space based Task Vector for Large Language Models via In-Context Learning

**Conference**: ACL 2026  
**arXiv**: [2604.11129](https://arxiv.org/abs/2604.11129)  
**Code**: [GitHub](https://github.com/szu-tera/DeCoVec)  
**Area**: Robotics  
**Keywords**: Task Vector, Decoding Space, In-Context Learning, Training-free LLM Guiding, logit manipulation

## TL;DR
This paper proposes DeCoVec (Decoding Space based Task Vector), a training-free and non-invasive framework that constructs task vectors in the decoding space by contrasting the output logit distributions of few-shot and zero-shot prompts. These vectors are injected into the decoding process to guide generation, achieving an average accuracy improvement of up to 5.50 over standard few-shot baselines on TruthfulQA, Math-500, and AQUA-RAT.

## Background & Motivation

**Background**: Task vectors—directions in high-dimensional space that encode specific task behaviors—have emerged as promising tools for guiding LLMs. Existing methods operate in two spaces: (1) weight-space task vectors (requiring fine-tuning); (2) activation-space task vectors (requiring invasive manipulation of internal hidden states).

**Limitations of Prior Work**: (1) Weight-space methods require full fine-tuning for each task, which is computationally expensive; (2) Activation-space methods require complex optimization or auxiliary training to manipulate hidden states, making them structurally invasive; (3) Both categories limit flexibility and scalability.

**Key Challenge**: Effectively guiding the task behavior of LLMs without modifying model parameters or invading internal structures.

**Goal**: To construct task vectors in the decoding space (output logit layer) to achieve training-free and non-invasive LLM guidance.

**Key Insight**: In-Context Learning (ICL) alters the output distribution of LLMs, and this distributional change inherently encodes task information. This change can be directly captured in the output logit space as a task vector.

**Core Idea**: Task vector = few-shot logit - zero-shot logit. This difference vector is injected into the normal decoding process to guide generation.

## Method

### Overall Architecture
Given a test query: (1) Construct a zero-shot context and a few-shot ICL context; (2) Use the model to calculate logit vectors for both contexts; (3) Compute the task vector as the difference $\mathbf{v}_\mathcal{T}^t = \mathbf{z}_{\text{icl}}^t - \mathbf{z}_{\text{zs}}^t$; (4) Inject the task vector into the base decoding logits using a scaling factor $\lambda$: $\tilde{\mathbf{z}}^t = \mathbf{z}_{\text{de}}^t + \lambda \cdot \mathbf{v}_\mathcal{T}^t$.

### Key Designs

1.  **Decoding Space Task Vector Construction**:
    - **Function**: Captures task-specific semantic signals in the output logit space.
    - **Mechanism**: The zero-shot context represents the task-agnostic state of the model, while the few-shot ICL context represents the task-aware state. The logit difference encodes task-level features activated by ICL. Since both share the same generation prefix $y^{1:t}$, the vocabulary distributions are strictly aligned, avoiding sequence length mismatch issues.
    - **Design Motivation**: Task vectors in weight and activation spaces require either fine-tuning or internal invasion. The logit space is the final output interface of the model, making operations transparent and controllable.

2.  **Token-level Online Steering**:
    - **Function**: Injects task signals into the decoding process on a token-by-token basis.
    - **Mechanism**: Three forward passes are calculated at each decoding step: (1) logit $\mathbf{z}_{\text{de}}^t$ for the base context; (2) logit $\mathbf{z}_{\text{zs}}^t$ for the zero-shot context; (3) logit $\mathbf{z}_{\text{icl}}^t$ for the steering ICL context. Final output = base logit + $\lambda$ × task vector.
    - **Design Motivation**: Token-level operations ensure that task signals are dynamically aligned with the current generation context rather than being a static global bias.

3.  **Decoupling of Two ICL Contexts**:
    - **Function**: Decouples the construction of the task vector from the base decoding.
    - **Mechanism**: Independent sampling strategies are used to construct the steering context (for task vector calculation) and the decode context (for base decoding). Different sets of exemplars can be used for each.
    - **Design Motivation**: Prevents the bias in exemplar selection from affecting both the base decoding and the task vector simultaneously.

## Key Experimental Results

### Main Results (7 LLMs, 0.5B-9B)

| Method | TruthfulQA MC1/MC2/MC3 | Math-500 | AQUA-RAT | Average Δ |
|------|----------------------|----------|----------|-------|
| Zero-shot | Baseline | Baseline | Baseline | - |
| Few-shot (Random) | +Slight | +Slight | +Slight | - |
| Few-shot (KATE) | +Moderate | +Moderate | +Moderate | - |
| **Ours (DeCoVec)** | **+Significant** | **+Significant** | **+Significant** | **+5.50** |

### Ablation Study

| Configuration | Description |
|------|------|
| λ=0 (No task vector) | Degenerates to standard few-shot |
| λ too large | Task signal is too strong, potentially distorting semantics |
| λ moderate (0.5-1.5) | Optimal range, stable improvement |
| Different k (No. of shots) | 3-5 shots is optimal |

### Key Findings
- **DeCoVec consistently outperforms few-shot baselines across all 7 models**, with a maximum average accuracy gain of 5.50.
- **Task vectors encode high-level task semantics rather than surface patterns**: Analysis shows that the vectors amplify token probabilities associated with correct reasoning.
- **Effective suppression of generation degradation and logical flaws**: Error analysis indicates that DeCoVec reduces logical errors in mathematical reasoning.
- **Robust to exemplar ordering**: Unlike standard ICL which is sensitive to exemplar order, DeCoVec remains stable.
- **No additional input token cost**: DeCoVec operates in the logit space and does not increase the input context length.

## Highlights & Insights
- **Constructing task vectors in the decoding space** is a conceptual breakthrough—moving task vectors from "inside the model" to the "model output interface," making the method entirely non-invasive.
- The discovery that **"ICL logit differences encode task semantics"** has theoretical significance for understanding ICL mechanisms.
- While the **overhead of three forward passes** is higher than standard decoding, it is much lighter than fine-tuning or training auxiliary models.

## Limitations & Future Work
- Requires three forward passes per decoding step, resulting in inference latency approximately 3x the standard.
- The optimal value of $\lambda$ varies by task and model, requiring some hyperparameter tuning.
- Validated on models in the 0.5B-9B range; effectiveness on larger models (70B+) remains unknown.
- Validated only on knowledge and reasoning tasks; effectiveness on generative tasks (e.g., dialogue, translation) remains to be explored.
- The interpretability of task vectors requires further research.

## Related Work & Insights
- **vs. Weight-space Task Vectors (Ilharco et al.)**: Requires fine-tuning and lacks flexibility. DeCoVec is training-free.
- **vs. Activation-space Task Vectors (In-Context Vector)**: Requires manipulation of internal hidden states and is invasive. DeCoVec is non-invasive.
- **vs. Contrastive Decoding**: Contrastive decoding uses "expert vs. amateur" logit differences to improve quality; DeCoVec uses "exemplar vs. no-exemplar" differences to inject task knowledge. The logic is similar, but the goals differ.

## Rating
- Novelty: ⭐⭐⭐⭐ Decoding space task vector is a new concept, though the core mechanism of logit difference injection is straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐ 7 models and 3 benchmarks with in-depth analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear methodology; comparison tables with existing work are helpful.
- Value: ⭐⭐⭐⭐ A lightweight plug-and-play solution that offers insights into ICL understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] UCS: Estimating Unseen Coverage for Improved In-Context Learning](ucs_estimating_unseen_coverage_for_improved_in-context_learning.md)
- [\[ACL 2026\] OOD Proxy Demonstration Retrieval Scheme for Robust In-Context Learning](toward_robust_in-context_learning_leveraging_out-of-distribution_proxies_for_tar.md)
- [\[ACL 2026\] MoRI: Learning Motivation-Grounded Reasoning for Scientific Ideation in Large Language Models](mori_learning_motivation-grounded_reasoning_for_scientific_ideation_in_large_lan.md)
- [\[NeurIPS 2025\] Unifying Attention Heads and Task Vectors via Hidden State Geometry in In-Context Learning](../../NeurIPS2025/llm_nlp/unifying_attention_heads_and_task_vectors_via_hidden_state_geometry_in_in-contex.md)
- [\[AAAI 2026\] LILAD: Learning In-context Lyapunov-stable Adaptive Dynamics Models](../../AAAI2026/llm_nlp/lilad_learning_in-context_lyapunov-stable_adaptive_dynamics_models.md)

</div>

<!-- RELATED:END -->
