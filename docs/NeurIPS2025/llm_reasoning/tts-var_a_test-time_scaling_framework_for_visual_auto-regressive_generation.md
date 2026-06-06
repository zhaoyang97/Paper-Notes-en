---
title: >-
  [Paper Note] TTS-VAR: A Test-Time Scaling Framework for Visual Auto-Regressive Generation
description: >-
  [NeurIPS 2025][LLM Reasoning][test-time scaling] This paper proposes TTS-VAR — the first test-time scaling framework specifically designed for Visual Auto-Regressive (VAR) models. It formulates image generation as a path…
tags:
  - "NeurIPS 2025"
  - "LLM Reasoning"
  - "test-time scaling"
  - "visual auto-regressive"
  - "VAR"
  - "path searching"
  - "image generation"
date: 2026-05-08
content_hash: 5b37b0582e8c0bda
---

# TTS-VAR: A Test-Time Scaling Framework for Visual Auto-Regressive Generation

**Conference**: NeurIPS 2025
**arXiv**: [2507.18537](https://arxiv.org/abs/2507.18537)  
**Code**: [GitHub](https://github.com/ali-vilab/TTS-VAR)  
**Area**: LLM Inference
**Keywords**: test-time scaling, visual auto-regressive, VAR, path searching, image generation

## TL;DR
This paper proposes TTS-VAR — the first test-time scaling framework specifically designed for Visual Auto-Regressive (VAR) models. It formulates image generation as a path searching problem and achieves an 8.7% improvement on GenEval (0.69 → 0.75) with Infinity 2B by combining adaptive descending batch sizes, early-stage clustering-based diversity search, and late-stage resampling-based potential selection. With $N=2$, TTS-VAR already surpasses Best-of-N at $N=8$.

## Background & Motivation

**Background**: Test-time scaling (TTS) has achieved remarkable success in LLMs (e.g., CoT, tree search) and has also been explored as a path searching problem in diffusion models. However, VAR (Visual Auto-Regressive) models — which generate images by progressively predicting the "next scale" in a coarse-to-fine multi-scale manner — still lack a dedicated TTS framework.

**Limitations of Prior Work**: (a) TTS methods for diffusion models cannot be directly transferred — additional inference steps break the KV Cache mechanism in VAR, causing exponential complexity growth; (b) Applying reward functions at early VAR scales leads to misjudgments, as the consistency between early intermediate images and final image quality is very low, potentially eliminating promising candidates prematurely; (c) Simple Best-of-N strategies are inefficient.

**Key Challenge**: The causal generation nature of VAR — once a token is generated it cannot be modified and directly affects all subsequent tokens — makes early decisions critical, yet early scales are difficult to evaluate accurately.

**Key Insight**: Exploit the inherent structure of VAR's multi-scale generation: early scales encode structural information (layout/composition) while late scales encode fine details, and apply distinct strategies accordingly.

**Core Idea**: Maintain diversity via clustering at early scales (without scoring), and select the optimal candidate via reward at late scales (where consistency is high), coupled with adaptive descending batch sizes to fully exploit the low cost of early scales.

## Method

### Overall Architecture
The VAR image generation process is formulated as a path searching problem across 13 scales: (1) **adaptive descending batch size** — maintain large batches at early scales to generate more candidates; (2) **coarse-scale clustering search** — when reducing batch size, use DINOv2 features with K-Means++ clustering to preserve structural diversity; (3) **fine-scale resampling selection** — at scales where intermediate image quality is consistent with the final output, use a reward function to guide selection of the best candidates.

### Key Designs

1. **Adaptive Descending Batch Size**:

    - Function: Adopt a decreasing batch size schedule during inference: $\{8N, 8N, 6N, 6N, 6N, 4N, 2N, 2N, 2N, 1N, 1N, 1N, 1N\}$
    - Design Motivation: FLOPs and memory consumption at early VAR scales are extremely low (short token sequences) but grow exponentially at later scales. Concentrating large batches at low-cost early stages incurs minimal overhead.
    - Effect: Explores more generation possibilities at nearly the same total cost compared to fixed batch sizes.

2. **Clustering-Based Diversity Search**:

    - Function: When reducing batch size at early scales (scale 2, 5), apply K-Means++ clustering on DINOv2 semantic features and select the sample nearest to each cluster centroid.
    - Core Observations: (a) The correlation between reward scores of early intermediate images and final images is low (<0.3), so direct scoring eliminates good candidates; (b) However, structural information (layout/composition) is already clearly discernible at scale 2.
    - Feature Extraction: DINOv2 self-supervised features → PCA dimensionality reduction → K-Means++ clustering.
    - Effect: Preserves structural diversity and prevents all candidates from collapsing to similar layouts.

3. **Resampling-Based Potential Selection**:

    - Function: At late scales (scale 6, 9), score candidates using ImageReward and resample from a multinomial distribution based on potential scores.
    - Potential Score Design: Four strategies are compared — VALUE (current score), DIFF (adjacent difference), MAX (historical maximum), and SUM (cumulative sum).
    - Key Finding: VALUE performs best — the current-scale reward score alone suffices; DIFF performs worst due to unstable growth rates.
    - Resampling Frequency: Performed only once each at scale 6 and scale 9; increasing frequency yields marginal gains at high cost.
    - Theoretical Basis: The objective is to shift the generation distribution $p_\theta(x)$ toward $p_{\theta'}(x) \propto p_\theta(x) \exp(\lambda \cdot r_\phi(x,c))$.

### Why a Scale-Differentiated Strategy Is Necessary
- Intermediate-state consistency experiments show that reward score correlation with final quality is <0.3 for scales 0–5, rising rapidly to 0.6–0.8 at scale 6+.
- Applying resampling at scale 3 actually degrades final scores (prematurely eliminating promising candidates).
- This is fundamentally different from diffusion models, where iterative denoising allows correction; in VAR, tokens are immutable once generated.

## Key Experimental Results

### Main Results (GenEval)

| Method | Two Obj. | Counting | Color Attri. | Overall |
|--------|---------|----------|-------------|---------|
| Infinity 2B | 0.835 | 0.592 | 0.615 | 0.695 |
| +IS (N=8) | 0.897 | 0.622 | 0.655 | 0.718 |
| +BoN (N=8) | 0.920 | 0.676 | 0.670 | 0.736 |
| **+TTS-VAR (N=2)** | **0.928** | **0.711** | **0.678** | **0.740** |
| **+TTS-VAR (N=8)** | **0.950** | **0.741** | **0.680** | **0.753** |
| Infinity 8B | 0.887 | 0.729 | 0.675 | 0.765 |
| +TTS-VAR (N=4) | 0.930 | 0.804 | 0.760 | 0.819 |

### User Study

| Metric | Baseline | IS | BoN | **TTS-VAR** |
|--------|---------|-----|-----|-----------|
| Image Quality | 13.3% | 7.9% | 13.3% | **65.4%** |
| Plausibility | 13.7% | 8.6% | 8.6% | **69.2%** |
| Prompt Consistency | 1.3% | 1.9% | 2.5% | **94.3%** |

### Ablation Study

| Component | GenEval (N=4) | Note |
|-----------|-------------|------|
| BoN only | 0.724 | Final selection only |
| +Resampling | 0.728 | Add late-stage resampling |
| +Clustering | 0.730 | Add early-stage clustering |
| **+Both (full)** | **0.744** | Clustering + resampling, best |

### Key Findings
- **N=2 surpasses BoN at N=8**: TTS-VAR exceeds Best-of-N using only 25% of the samples, demonstrating high efficiency.
- **Effective on 8B models**: Infinity 8B improves from 0.765 to 0.819, confirming the generalizability of the framework.
- **Overwhelming advantage in user study**: 94.3% of users prefer TTS-VAR on prompt consistency.

## Highlights & Insights
- **First TTS framework for VAR**: Fills an important gap in improving VAR model generation quality; the method is general and plug-and-play.
- **Necessity of scale-differentiated processing**: The paper rigorously validates the intuition that "early scales cannot be scored; only late scales enable reliable selection," supported by quantitative consistency curves.
- **Structural diversity > early quality scoring**: A counter-intuitive finding — preserving diversity at early stages is more effective than selecting the best candidates, because rewards are unreliable at early scales.
- **Significant efficiency advantage**: Exploits the low cost of early VAR scales to achieve substantial improvements with minimal additional overhead.

## Limitations & Future Work
- **Dependence on an external reward model**: Biases in ImageReward propagate to selection outcomes, potentially limiting effectiveness in scenarios poorly covered by the reward model.
- **Fixed clustering features**: DINOv2 + PCA is used without exploration of whether different feature extractors are needed for different tasks or styles.
- **Validated only on the Infinity series**: The effectiveness on other VAR architectures (e.g., original VAR, LlamaGen) remains unverified.
- **Future directions**: (1) Learning adaptive scale-switching strategies instead of manually setting scales 6/9; (2) Training lightweight proxy reward models to reduce inference overhead; (3) Exploring hybrid scale strategies combining clustering and resampling.

## Related Work & Insights
- **vs. Diffusion model TTS (Ma et al.)**: Diffusion models support noise-adding/denoising search at arbitrary steps, while VAR cannot backtrack — TTS-VAR replaces early-stage scoring with clustering, a clever adaptation.
- **vs. PARM (Guo et al.)**: PARM leverages a unified understanding-generation model for image-level CoT self-correction but requires additional training; TTS-VAR is purely inference-time and training-free.
- **vs. Best-of-N**: BoN selects only at the final step, discarding intermediate process information; TTS-VAR continuously searches and filters throughout the generation process.
- **Insight**: The multi-scale structure of VAR is naturally suited to stage-wise strategies. The principle of "choosing different strategies based on signal reliability" is generalizable to other hierarchical generative models.

## Rating
- Novelty: ⭐⭐⭐⭐ First VAR TTS framework; the scale-differentiated clustering + resampling design is insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐ GenEval/T2I-CompBench benchmarks + user study + detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Method motivation is clear; experimental analysis is thorough (consistency curve analysis is particularly strong).
- Value: ⭐⭐⭐⭐ A plug-and-play inference enhancement for VAR with high practical value for autoregressive image generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Video-T1: Test-Time Scaling for Video Generation](../../ICCV2025/llm_reasoning/video-t1_test-time_scaling_for_video_generation.md)
- [\[NeurIPS 2025\] Atom of Thoughts for Markov LLM Test-Time Scaling](atom_of_thoughts_for_markov_llm_testtime_scaling.md)
- [\[NeurIPS 2025\] Rethinking Optimal Verification Granularity for Compute-Efficient Test-Time Scaling](rethinking_optimal_verification_granularity_for_compute-efficient_test-time_scal.md)
- [\[NeurIPS 2025\] LIMOPro: Reasoning Refinement for Efficient and Effective Test-time Scaling](limopro_reasoning_refinement_for_efficient_and_effective_test-time_scaling.md)
- [\[NeurIPS 2025\] SolverLLM: Solving Optimization Problems via Test-Time Scaling with LLM-Guided Search](solverllm_leveraging_test-time_scaling_for_optimization_problem_via_llm-guided_s.md)

</div>

<!-- RELATED:END -->
