---
title: >-
  [Paper Note] SITE: Soft Head Selection for Injecting ICL-Derived Task Embeddings
description: >-
  [ACL 2026][Interpretability][Attention Head Selection] SITE proposes a soft attention head selection method based on gradient optimization to effectively inject ICL-derived task embeddings by identifying task-relevant at…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Attention Head Selection"
  - "Task Embeddings"
  - "In-Context Learning"
  - "Activation Patching"
  - "Parameter-Efficient"
date: 2026-05-08
content_hash: ba7738065589975e
---

# SITE: Soft Head Selection for Injecting ICL-Derived Task Embeddings

**Conference**: ACL 2026 Findings  
**arXiv**: [2507.20906](https://arxiv.org/abs/2507.20906)  
**Code**: [https://github.com/SNU-DRL/Soft_Injection](https://github.com/SNU-DRL/Soft_Injection)  
**Area**: Interpretability / Parameter-Efficient Adaptation  
**Keywords**: Attention Head Selection, Task Embeddings, In-Context Learning, Activation Patching, Parameter-Efficient

## TL;DR

SITE proposes a soft attention head selection method based on gradient optimization to effectively inject ICL-derived task embeddings by identifying task-relevant attention heads. It significantly outperforms ICL and existing embedding methods on 12 LLMs (4B-70B) while achieving comparable performance to PEFT with far fewer trainable parameters.

## Background & Motivation

**Background**: LLM task adaptation primarily follows three paradigms: Parameter-Efficient Fine-Tuning (PEFT, e.g., LoRA) offers high performance but requires training; In-Context Learning (ICL) requires no training but increases inference costs; and embedding injection methods extract task embeddings from ICL activations to inject them during inference.

**Limitations of Prior Work**: ICL-driven embedding injection methods are conceptually attractive but fail to demonstrate consistent advantages over PEFT or ICL in practice. Existing methods (e.g., FV, TV, MTV, I2CL) rely on heuristic rules or restricted search spaces to determine extraction and injection locations, and most are evaluated only on simple classification tasks.

**Key Challenge**: Task-relevant information is unevenly distributed across attention heads and varies by task—randomly selecting heads for patching leads to drastic performance fluctuations, but existing methods lack efficient head selection mechanisms.

**Goal**: To develop an ICL-driven embedding injection method that achieves near-PEFT performance with fewer parameters while significantly outperforming ICL.

**Key Insight**: Formalize attention head selection as a continuous optimization problem, learning importance parameters for each head (soft selection) via gradient descent to identify efficient task embedding injection locations.

**Core Idea**: Use learnable soft selection parameters to linearly interpolate between original activations and task embeddings, optimizing only $L \times H$ scalar parameters (approximately 1K) to achieve precise task-relevant head identification and efficient injection.

## Method

### Overall Architecture

The process consists of three stages: (1) Construct Task Embeddings—extract and average last-token activations for each attention head from $M$ few-shot prompts; (2) Optimize Soft Head Selection Parameters—minimize zero-shot inference cross-entropy loss via gradient descent; (3) Zero-shot Inference—inject task embeddings at the first token position of the input, with no further intervention during subsequent decoding.

### Key Designs

1. **Task Embedding Construction**:

    - **Function**: Extract embeddings encoding task information from few-shot ICL activations.
    - **Mechanism**: For $M$ few-shot prompts containing $N$ input-output examples, extract the last-token activation $\mathbf{t}_m^{(l,h)}$ for each attention head in every layer. Averaging across the $M$ prompts yields the task embedding $\mathbf{t}^{(l,h)} = \frac{1}{M}\sum_m \mathbf{t}_m^{(l,h)}[-1,:]$.
    - **Design Motivation**: Averaging reduces instance-specific noise and preserves task-level information.

2. **Soft Head Selection Parameter Optimization**:

    - **Function**: Efficiently identify the most important attention heads for each task.
    - **Mechanism**: Introduce a learnable matrix $\mathbf{A} \in [0,1]^{L \times H}$, where each $\alpha^{(l,h)}$ controls the degree of task embedding injection. During zero-shot inference, the last-token activation is replaced by linear interpolation: $\mathbf{o}^{(l,h)} \leftarrow (1-\alpha^{(l,h)}) \cdot \mathbf{o}^{(l,h)} + \alpha^{(l,h)} \cdot \mathbf{t}^{(l,h)}$. The LLM is frozen, and only $\mathbf{A}$ (approximately 1K parameters) is optimized using the Adam optimizer for 400 steps. $\alpha$ is parameterized via sigmoid to ensure a value range of $[0,1]$.
    - **Design Motivation**: Continuous optimization replaces discrete search or reinforcement learning for higher efficiency; optimizing only injection locations rather than embedding content results in minimal parameters (1.02K vs. LoRA 3407K).

3. **Single-token Injection Inference**:

    - **Function**: Inference-time task adaptation with minimal intervention.
    - **Mechanism**: Injection occurs only once at the last-token position of the initial input prompt. The injected information is written into the KV cache, and subsequent autoregressive decoding proceeds without further intervention.
    - **Design Motivation**: Compared to methods injecting at multi-token positions, single-point injection reduces intervention complexity and adverse effects on generation.

### Loss & Training

The optimization objective is the cross-entropy loss under zero-shot inference. Checkpoints are selected using a validation set every 50 steps. No regularization or model-specific hyperparameter tuning is employed.

## Key Experimental Results

### Main Results

**Llama-3.1-8B Average across Four Benchmarks**

| Method | Type | Trainable Params | FV (57 tasks) | ANLI | MMLU-Pro | BBH | Avg |
|------|------|-----------|---------------|------|---------|-----|-----|
| LoRA | PEFT | 3407K | 86.76 | 45.82 | 41.04 | 60.39 | 58.50 |
| 10-shot ICL | ICL | 0 | 76.76 | 43.96 | 36.47 | 47.17 | 51.09 |
| I2CL | Emb | 0.13K | 79.89 | 28.01 | 27.14 | 50.60 | 46.41 |
| **Ours (M=50)** | Emb | **1.02K** | **90.02** | **47.31** | **38.78** | **58.04** | **58.54** |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| SITE M=50 | 58.54 avg | Optimal |
| SITE M=1 | 57.50 avg | Slight decrease, insensitive to M |
| Random Head Patching | Unstable | Performance highly dependent on selected heads |
| Low-α Head Patching | 6.2 avg | Performance drop, validates selection effectiveness |
| High-α Head Patching | 57.3 avg | Close to SITE |

### Key Findings

- SITE outperforms LoRA on the FV benchmark (90.02 vs. 86.76) and ANLI, achieving PEFT-level performance with only 0.03% of the parameters.
- Consistently outperforms 10-shot ICL by 10.2-14.3 percentage points across 12 LLMs (4B-70B).
- The optimized soft selection parameters exhibit a near-binary distribution, indicating that the task relevance of attention heads is "all-or-nothing."
- Cross-task activation patching analysis reveals that similar tasks share important attention heads, while important heads for dissimilar tasks do not overlap—demonstrating strong task specificity.
- A performance gap relative to PEFT remains on MMLU-Pro and BBH, suggesting that ICL-derived task embeddings have limited expressiveness for complex reasoning.

## Highlights & Insights

- Achieving the performance of 3.4M parameters with only 1K parameters is a striking result—the core insight is that "injection location is more important than injection content."
- Near-binary selection parameters and cross-task head sharing analysis provide new mechanistic interpretability insights—attention heads indeed possess task-specific functions.
- The minimalist design of the method (no regularization, no model-specific tuning, 400-step training) makes it highly reproducible and deployable.

## Limitations & Future Work

- Performance gaps persist compared to LoRA on benchmarks requiring complex reasoning (MMLU-Pro, BBH).
- Each task requires an independent set of choice parameters to be optimized, and scalability in multi-task scenarios remains to be verified.
- Injection only at the last token position may limit the representation of task information.
- Task embeddings remain fixed and cannot adapt to intra-task changes (e.g., samples of varying difficulty).

## Related Work & Insights

- **vs FV/TV**: These methods use heuristic search or activation patching to determine injection locations; SITE utilizes more efficient gradient optimization.
- **vs LoRA**: LoRA modifies model weights, whereas SITE only modifies the activations of specific heads; the parameter count difference is 3000x, yet performance is comparable.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The formalization of soft head selection and the insight that "location is more important than content" are highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 12 models, four benchmarks, full activation patching analysis, and cross-task analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ The method is clearly articulated and the experimental logic is rigorous.
- Value: ⭐⭐⭐⭐⭐ Provides an extreme parameter-efficient task adaptation scheme and a new understanding of attention head functionality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Style over Story: Measuring LLM Narrative Preferences via Structured Selection](style_over_story_measuring_llm_narrative_preferences_via_structured_selection.md)
- [\[ICLR 2026\] Bridging Explainability and Embeddings: BEE Aware of Spuriousness](../../ICLR2026/interpretability/bridging_explainability_and_embeddings_bee_aware_of_spuriousness.md)
- [\[ACL 2026\] Learning What Matters: Dynamic Dimension Selection and Aggregation for Interpretable Vision-Language Reward Modeling](learning_what_matters_dynamic_dimension_selection_and_aggregation_for_interpreta.md)
- [\[ICLR 2026\] Cross-Modal Redundancy and the Geometry of Vision-Language Embeddings](../../ICLR2026/interpretability/cross-modal_redundancy_and_the_geometry_of_vision-language_embeddings.md)
- [\[ACL 2026\] Linear Probes Detect Task Format, Not Reasoning Mode in Language Model Hidden States](linear_probes_detect_task_format_not_reasoning_mode_in_language_model_hidden_sta.md)

</div>

<!-- RELATED:END -->
