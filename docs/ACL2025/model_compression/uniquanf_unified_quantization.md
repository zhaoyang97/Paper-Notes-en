---
title: >-
  [Paper Note] UniQuanF: Unifying Uniform and Binary-coding Quantization for Accurate Compression of Large Language Models
description: >-
  [ACL 2025][Model Compression][quantization] UniQuanF unifies the strengths of Uniform Quantization (UQ, high optimizability but low representational capacity) and Binary-Coding Quantization (BCQ, high representational capacity but low optimizability). Through unified initialization, local periodic mapping, and a unification theorem, it achieves highly accurate LLM quantization without any extra deployment overhead, yielding up to 4.60% improvement on GSM8K.
tags:
  - "ACL 2025"
  - "Model Compression"
  - "quantization"
  - "binary-coding"
  - "uniform quantization"
  - "LLM compression"
  - "non-uniform levels"
date: 2026-05-08
content_hash: f53734aa470177dc
---

# UniQuanF: Unifying Uniform and Binary-coding Quantization for Accurate Compression of Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2506.03781](https://arxiv.org/abs/2506.03781)  
**Code**: [https://github.com/snudm-starlab/UniQuanF](https://github.com/snudm-starlab/UniQuanF)  
**Area**: Model Compression / LLM Efficiency  
**Keywords**: quantization, binary-coding, uniform quantization, LLM compression, non-uniform levels

## TL;DR
UniQuanF unifies the strengths of Uniform Quantization (UQ, high optimizability but low representational capacity) and Binary-Coding Quantization (BCQ, high representational capacity but low optimizability). Through unified initialization, local periodic mapping, and a unification theorem, it achieves highly accurate LLM quantization without any extra deployment overhead, yielding up to 4.60% improvement on GSM8K.

## Background & Motivation

**Background**: LLM quantization is a core technology for deployment acceleration. Two main paradigms exist: (1) Uniform Quantization (UQ, e.g., FlexRound/OmniQuant) — quantization levels are uniformly distributed, with mature optimization methods (strong optimizability), but unable to adapt to non-uniform weight distributions (weak representational capacity); (2) Binary-coding Quantization (BCQ, e.g., Alternating) — generates non-uniform quantization levels by adding or subtracting scale factors (strong representational capacity), but lacks precise optimization methods (weak optimizability).

**Limitations of Prior Work**: The uniformly spaced quantization levels in UQ cannot match the non-uniform weight distribution in LLMs. Meanwhile, the only BCQ-based method applicable to LLMs (Alternating) does not consider the input distribution, resulting in poor optimization accuracy. Both paradigms have inherent deficiencies, and no existing method simultaneously leverages the strengths of both.

**Key Challenge**: The contradiction between representational capacity (adapting non-uniform quantization levels to weight distributions) and optimizability (utilizing gradients to precisely optimize quantization parameters).

**Goal**: To unify UQ and BCQ, simultaneously obtaining the advantages of both.

**Key Insight**: Analysis reveals that the optimizability of UQ originates from its transformation process $\mathcal{T}$, whereas the representational capacity of BCQ originates from its mapping process $\mathcal{M}$ — these two can be effectively combined.

**Core Idea**: Embed the differentiable transformation process of UQ into the non-uniform mapping process of BCQ, and eliminate the extra deployment overhead of the unified framework via a unification theorem.

## Method

### Overall Architecture
The quantization process is unified as: $\hat{w} = \mathcal{D}(\mathcal{M}(\mathcal{T}(w; \Theta); \Theta); \Theta)$, where $\mathcal{T}$ represents the transformation, $\mathcal{M}$ the mapping, and $\mathcal{D}$ the dequantization (inverse transformation). UniQuanF incorporates $\mathcal{T}_F$ from FlexRound (trainable rounding transformation) + $\mathcal{M}_B^*$ from BCQ (non-uniform mapping) + $\mathcal{D}_R$ from UQ (dequantization).

### Key Designs

1. **UniQuan Unified Framework**:

    - Function: Combines the transformation process of UQ and the mapping process of BCQ into a unified quantization scheme.
    - Mechanism: The optimizability of UQ derives from the transformation process $\mathcal{T}$ (e.g., trainable rounding offsets in FlexRound), while the representational capacity of BCQ comes from the mapping process $\mathcal{M}$ (which maps weights to non-uniform quantization levels). UniQuan = $\mathcal{D}_R \circ \mathcal{M}_B \circ \mathcal{T}_F$.
    - Design Motivation: The functions of these two processes do not overlap, allowing them to be directly combined. UQ's $\mathcal{T}$ provides precisely optimized weight transformation for BCQ, and BCQ's $\mathcal{M}$ provides more flexible quantization levels for the transformed weights.

2. **Unified Initialization**:

    - Function: Provides joint initialization for FlexRound and Alternating parameters in UniQuanF.
    - Mechanism: Employs RTN grid search first to find high-quality UQ parameters, and then converts the UQ quantization levels into BCQ scale factors to initialize Alternating.
    - Design Motivation: Directly combining them without proper initialization (as in UniQuanF-0) leads to slow convergence and poor accuracy.

3. **Local & Periodic Mapping**:

    - Function: Accelerates the slow mapping process in BCQ.
    - Mechanism: (1) Local mapping — updates the mapping relationship only for weights modified by the transformation process, instead of performing a global re-mapping; (2) Periodic mapping — performs mapping periodically rather than at every optimization step.
    - Design Motivation: The original Alternating mapping in BCQ is extremely slow (requiring traversing all weights). The local and periodic strategies accelerate this process to an acceptable level.

4. **Unification Theorem**:

    - Function: Proves that the optimized UniQuanF can be equivalently converted into the standard BCQ format, eliminating extra deployment overhead.
    - Mechanism: Merges the two-step inference $\mathcal{D}_R(\mathcal{R}_B(C; \Theta_B); \Theta_R)$ into a single step $\mathcal{R}_B(C; \Theta_B^*)$ by absorbing UQ's $\Theta_R$ parameters into BCQ's $\Theta_B^*$ via algebraic transformations.
    - Key Significance: UniQuanF enjoys the dual benefits of UQ + BCQ during optimization, while retaining identical execution during deployment as pure BCQ (same memory, same computation overhead, same inference kernel).

## Key Experimental Results

### Main Results: Multi-benchmark Comparison on the Llama Family

| Method | Type | GSM8K | MMLU | ARC-c | Average |
|------|------|-------|------|-------|------|
| FP16 | - | Baseline | Baseline | Baseline | Baseline |
| FlexRound | UQ | Lower | Baseline | Baseline | Baseline |
| Alternating | BCQ | Even Lower | Lower | Lower | Lower |
| **UniQuanF** | UQ+BCQ | **Highest** | **Highest** | **Highest** | **+4.60% on GSM8K** |

### Ablation Study

| Configuration | Performance | Description |
|------|------|------|
| Full UniQuanF | Best | Unified initialization + Local periodic mapping + Unification theorem |
| UniQuanF-0 (No init, no accel) | Poor | Slow convergence, worse accuracy than FlexRound |
| w/o Unified Initialization | Significant Drop | Initialization quality significantly impacts final accuracy |
| w/o Local Mapping | Slow Training | High computational cost of global mapping |
| FlexRound Only | Good | Strong optimizability but constrained quantization levels |
| Alternating Only | Poor | High representational capacity but coarse optimization |

### Key Findings
- **UniQuanF achieves the most significant gains in mathematical reasoning**: Up to 4.60% improvement on GSM8K, demonstrating that mathematical tasks are more sensitive to quantization accuracy.
- **Unification theorem guarantees zero deployment overhead**: Employs double the parameters during optimization, but equivalently decomposes to standard BCQ during deployment with zero memory or computational overhead.
- **Non-uniform quantization levels are more critical at low bits**: The representational advantages of BCQ are more pronounced in 2-bit/3-bit scenarios.
- **Unified initialization is crucial**: Starting with a strong initialization from UQ is much more effective than randomly initializing BCQ parameters.

## Highlights & Insights
- **Orthogonal Decomposition of Optimizability and Representational Capacity**: Decomposing the quantization process into transformation (source of optimizability) and mapping (source of representational capacity) as two independent dimensions, revealing they are combinable rather than mutually exclusive. This analytical framework itself offers a theoretical contribution.
- **Practicality of the Unification Theorem**: Utilizing UniQuan during training to obtain superior quantization quality, and seamlessly converting to standard BCQ during deployment — effectively gaining extra accuracy for "free".
- **"Resurrecting" BCQ**: Although BCQ was mostly overlooked in the LLM era, this work demonstrates that its non-uniform quantization levels are indeed superior to UQ, highlightingly showing that prior struggles were solely due to the lack of proper optimization methods.

## Limitations & Future Work
- BCQ inference kernels are still in early stages and may not be as mature or efficient as UQ kernels.
- Validation is primarily conducted on the Llama family, lacking coverage over diverse model architectures.
- The period hyperparameter for local periodic mapping requires manual tuning.
- Lacks comparison against state-of-the-art vector quantization methods (e.g., AQLM/QuIP#).

## Related Work & Insights
- **vs FlexRound (Lee et al., 2023)**: FlexRound is a state-of-the-art UQ method that optimizes quantization via trainable rounding offsets. UniQuanF builds upon this by incorporating BCQ's non-uniform mapping for further improvement.
- **vs Alternating (Xu et al., 2018)**: This is a classic BCQ approach but neglects input distribution. UniQuanF injects the input-aware optimization of FlexRound into Alternating.
- **vs OmniQuant (Shao et al., 2024)**: OmniQuant is another UQ optimization technique. The unified framework of UniQuanF can theoretically support combination with any UQ method.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The theoretical framework unifying UQ and BCQ is elegant; the unification theorem ensuring zero-overhead deployment is a major highlight.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive experiments across multiple models and benchmarks, with rigorous ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear formalization and smooth logical flow from the generalized framework to concrete instantiation.
- Value: ⭐⭐⭐⭐ Provides a fresh theoretical perspective on the design of quantization methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] EfficientQAT: Efficient Quantization-Aware Training for Large Language Models](efficientqat.md)
- [\[ACL 2025\] L4Q: Parameter Efficient Quantization-Aware Fine-Tuning on Large Language Models](l4q_parameter_efficient_quantization_aware_finetuning.md)
- [\[ICML 2025\] SlimLLM: Accurate Structured Pruning for Large Language Models](../../ICML2025/model_compression/slimllm_accurate_structured_pruning_for_large_language_models.md)
- [\[ACL 2025\] 500xCompressor: Generalized Prompt Compression for Large Language Models](500xcompressor_generalized_prompt_compression_for_large_language_models.md)
- [\[ACL 2025\] Wanda++: Pruning Large Language Models via Regional Gradients](wanda_pruning_large_language_models_via_regional_gradients.md)

</div>

<!-- RELATED:END -->
