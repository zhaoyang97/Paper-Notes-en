---
title: >-
  [Paper Note] SITE: Soft Head Selection for Injecting ICL-Derived Task Embeddings
description: >-
  [ACL 2026][Interpretability][Attention head selection] SITE proposes a gradient-based soft attention head selection method that identifies task-relevant attention heads to effectively inject ICL-derived task embeddings.…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Attention head selection"
  - "task embeddings"
  - "in-context learning"
  - "activation patching"
  - "parameter efficiency"
date: 2026-05-08
content_hash: 688777e2536325ab
---

# SITE: Soft Head Selection for Injecting ICL-Derived Task Embeddings

**Conference**: ACL 2026
**arXiv**: [2507.20906](https://arxiv.org/abs/2507.20906)
**Code**: [https://github.com/SNU-DRL/Soft_Injection](https://github.com/SNU-DRL/Soft_Injection)
**Area**: Interpretability / Parameter-Efficient Adaptation
**Keywords**: Attention head selection, task embeddings, in-context learning, activation patching, parameter efficiency

## TL;DR

SITE proposes a gradient-based soft attention head selection method that identifies task-relevant attention heads to effectively inject ICL-derived task embeddings. Across 12 LLMs (4B–70B), SITE substantially outperforms ICL and existing embedding injection methods while achieving performance comparable to PEFT with far fewer trainable parameters.

## Background & Motivation

**Background**: Task adaptation for LLMs has three main paradigms: parameter-efficient fine-tuning (PEFT, e.g., LoRA) yields strong performance but requires training; in-context learning (ICL) requires no training but increases inference cost; embedding injection methods extract task embeddings from ICL activations and inject them at inference time.

**Limitations of Prior Work**: ICL-driven embedding injection is conceptually appealing but in practice fails to demonstrate consistent advantages over PEFT or ICL. Existing methods (e.g., FV, TV, MTV, I2CL) rely on heuristic rules or restricted search spaces to determine extraction and injection locations, and are mostly evaluated only on simple classification tasks.

**Key Challenge**: Task-relevant information is distributed unevenly across attention heads and varies across tasks — randomly patching heads leads to severe performance fluctuations, yet existing methods lack an efficient head selection mechanism.

**Goal**: To develop an ICL-driven embedding injection method that approaches PEFT performance with far fewer parameters while substantially surpassing ICL.

**Key Insight**: Formalize attention head selection as a continuous optimization problem, learning per-head importance parameters via gradient descent (soft selection) to efficiently identify injection locations for task embeddings.

**Core Idea**: Learnable soft selection parameters linearly interpolate between original activations and task embeddings, optimizing only $L \times H$ scalar parameters (~1K) to precisely identify task-relevant heads and perform efficient injection.

## Method

### Overall Architecture

A three-stage pipeline: (1) **Construct task embeddings** — extract last-token activations from each attention head across $M$ few-shot prompts and average them; (2) **Optimize soft head selection parameters** — minimize cross-entropy loss on zero-shot inference via gradient descent; (3) **Zero-shot inference** — inject task embeddings at the last-token position of the input prompt; subsequent autoregressive decoding is not further intervened.

### Key Designs

1. **Task Embedding Construction**:

    - Function: Extract embeddings encoding task information from few-shot ICL activations.
    - Mechanism: For $M$ few-shot prompts each containing $N$ input-output demonstrations, extract the last-token activation $\mathbf{t}_m^{(l,h)}$ at each attention head per layer, then average across $M$ prompts to obtain the task embedding $\mathbf{t}^{(l,h)} = \frac{1}{M}\sum_m \mathbf{t}_m^{(l,h)}[-1,:]$.
    - Design Motivation: Averaging reduces instance-specific noise while preserving task-level information.

2. **Soft Head Selection Parameter Optimization**:

    - Function: Efficiently identify the most important attention heads for each task.
    - Mechanism: Introduce a learnable matrix $\mathbf{A} \in [0,1]^{L \times H}$, where each $\alpha^{(l,h)}$ controls the degree of task embedding injection. During zero-shot inference, the last-token activation is replaced by a linear interpolation: $\mathbf{o}^{(l,h)} \leftarrow (1-\alpha^{(l,h)}) \cdot \mathbf{o}^{(l,h)} + \alpha^{(l,h)} \cdot \mathbf{t}^{(l,h)}$. The LLM is frozen; only $\mathbf{A}$ (~1K parameters) is optimized using the Adam optimizer for 400 steps. $\alpha$ is parameterized via sigmoid to ensure values lie in $[0,1]$.
    - Design Motivation: Continuous optimization replaces discrete search or reinforcement learning, yielding higher efficiency. Optimizing only injection locations rather than embedding content keeps parameter count minimal (1.02K vs. LoRA's 3407K).

3. **Single-Token Injection Inference**:

    - Function: Task adaptation at inference time with minimal intervention.
    - Mechanism: Injection is performed only once at the last-token position of the initial input prompt; injected information is written to the KV cache, and subsequent autoregressive decoding proceeds without further intervention.
    - Design Motivation: Compared to methods that inject at multiple token positions, single-point injection reduces intervention complexity and adverse effects on generation.

### Loss & Training

The optimization objective is the cross-entropy loss under zero-shot inference. Checkpoints are selected every 50 steps using a validation set. No regularization or model-specific hyperparameter tuning is applied.

## Key Experimental Results

### Main Results

**Llama-3.1-8B Average across Four Benchmarks**

| Method | Type | Trainable Params | FV (57 tasks) | ANLI | MMLU-Pro | BBH | Avg |
|--------|------|-----------------|---------------|------|---------|-----|-----|
| LoRA | PEFT | 3407K | 86.76 | 45.82 | 41.04 | 60.39 | 58.50 |
| 10-shot ICL | ICL | 0 | 76.76 | 43.96 | 36.47 | 47.17 | 51.09 |
| I2CL | Emb | 0.13K | 79.89 | 28.01 | 27.14 | 50.60 | 46.41 |
| **SITE (M=50)** | Emb | **1.02K** | **90.02** | **47.31** | **38.78** | **58.04** | **58.54** |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| SITE M=50 | 58.54 avg | Best |
| SITE M=1 | 57.50 avg | Slight drop; relatively insensitive to $M$ |
| Random head patching | Unstable | Performance highly dependent on selected heads |
| Low-$\alpha$ head patching | 6.2 avg | Performance collapse; validates selection effectiveness |
| High-$\alpha$ head patching | 57.3 avg | Approaches SITE performance |

### Key Findings

- SITE surpasses LoRA on the FV benchmark (90.02 vs. 86.76) and on ANLI, achieving PEFT-level performance with only 0.03% of the parameters.
- Consistently outperforms 10-shot ICL by 10.2–14.3 percentage points across 12 LLMs (4B–70B).
- Optimized soft selection parameters exhibit a near-binary distribution, indicating that the task-relevance of attention heads is essentially an "all-or-nothing" phenomenon.
- Cross-task activation patching analysis reveals that similar tasks share important attention heads, while dissimilar tasks have non-overlapping important heads — strong evidence of task specificity.
- A performance gap with PEFT remains on MMLU-Pro and BBH, suggesting that ICL-derived task embeddings have limited expressive capacity for complex reasoning.

## Highlights & Insights

- Achieving 3.4M-parameter-level performance with only 1K parameters is a striking result — the core insight is that **injection location matters more than injection content**.
- The near-binary selection parameters and cross-task head sharing analysis provide novel mechanistic interpretability insights, confirming that attention heads have task-specific functions.
- The method's minimal design (no regularization, no model-specific tuning, 400-step training) makes it highly reproducible and easy to deploy.

## Limitations & Future Work

- A performance gap with LoRA persists on complex reasoning benchmarks (MMLU-Pro, BBH).
- Each task requires independently optimizing a set of selection parameters; scalability in multi-task scenarios remains to be validated.
- Injection at only the last token position may limit the expressiveness of task information.
- Task embeddings are fixed and cannot adapt to intra-task variation (e.g., samples of varying difficulty).

## Related Work & Insights

- **vs. FV/TV**: These methods use heuristic search or activation patching to determine injection locations; SITE achieves this more efficiently via gradient optimization.
- **vs. LoRA**: LoRA modifies model weights; SITE modifies only the activations of selected heads. The parameter count differs by 3,000× yet performance is comparable.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The formalization of soft head selection and the insight that "location matters more than content" are highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 12 models, four benchmarks, comprehensive activation patching analysis, and cross-task analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ — Method presentation is clear and experimental logic is rigorous.
- Value: ⭐⭐⭐⭐⭐ — Provides an extremely parameter-efficient task adaptation solution and novel understanding of attention head functionality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Style over Story: Measuring LLM Narrative Preferences via Structured Selection](style_over_story_measuring_llm_narrative_preferences_via_structured_selection.md)
- [\[ICLR 2026\] Cross-Modal Redundancy and the Geometry of Vision-Language Embeddings](../../ICLR2026/interpretability/cross-modal_redundancy_and_the_geometry_of_vision-language_embeddings.md)
- [\[AAAI 2026\] Unsupervised Feature Selection Through Group Discovery](../../AAAI2026/interpretability/unsupervised_feature_selection_through_group_discovery.md)
- [\[NeurIPS 2025\] Interpretable Next-token Prediction via the Generalized Induction Head](../../NeurIPS2025/interpretability/interpretable_next-token_prediction_via_the_generalized_induction_head.md)
- [\[AAAI 2026\] Finding the Translation Switch: Discovering and Exploiting the Task-Initiation Features in LLMs](../../AAAI2026/interpretability/finding_the_translation_switch_discovering_and_exploiting_the_task-initiation_fe.md)

</div>

<!-- RELATED:END -->
