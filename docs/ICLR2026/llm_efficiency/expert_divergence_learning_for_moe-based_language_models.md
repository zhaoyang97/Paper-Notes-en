---
title: >-
  [Paper Note] Expert Divergence Learning for MoE-based Language Models
description: >-
  [ICLR 2026][LLM Efficiency][Mixture of Experts] This paper addresses the expert homogenization problem in MoE training by maximizing the Jensen-Shannon divergence of routing distributions across different data domains…
tags:
  - "ICLR 2026"
  - "LLM Efficiency"
  - "Mixture of Experts"
  - "expert homogenization"
  - "routing diversity"
  - "Jensen-Shannon divergence"
  - "domain specialization"
date: 2026-05-08
content_hash: acc681a0183864b3
---

# Expert Divergence Learning for MoE-based Language Models

**Conference**: ICLR 2026
**arXiv**: [2603.00054](https://arxiv.org/abs/2603.00054)  
**Code**: Not released  
**Area**: LLM Efficiency / MoE
**Keywords**: Mixture of Experts, expert homogenization, routing diversity, Jensen-Shannon divergence, domain specialization

## TL;DR
This paper addresses the expert homogenization problem in MoE training by maximizing the Jensen-Shannon divergence of routing distributions across different data domains, encouraging distinct expert subsets to be activated for different domains. The approach improves expert specialization and language modeling performance on a 15B-A1.5B model.

## Background & Motivation
**Background**: Mixture-of-Experts (MoE) models achieve high parameter counts with low computation through sparse activation, but training frequently suffers from "expert homogenization"—different experts learn highly similar functions, wasting parameter capacity.

**Limitations of Prior Work**: Existing methods such as load balancing losses only ensure uniform expert utilization, without guaranteeing that different experts learn distinct skills. Experts may be used uniformly yet remain functionally equivalent.

**Key Challenge**: Load balancing and functional specialization are fundamentally different objectives—uniform utilization does not imply distinct expertise.

**Core Idea**: Different data domains should activate different combinations of experts. Expert specialization can be encouraged by maximizing the JS divergence between inter-domain routing distributions.

## Method

### Overall Architecture
An expert divergence loss $\mathcal{L}_{ED}$ is added to the standard MoE training objective (language modeling loss $\mathcal{L}_{LM}$ + load balancing loss $\mathcal{L}_{LB}$):
$$\mathcal{L}_{final} = \mathcal{L}_{LM} + \alpha \mathcal{L}_{LB} + \beta \mathcal{L}_{ED}$$

### Key Designs
1. **Three-level aggregation**: Hierarchical aggregation of routing probabilities at the Token→Sequence→Domain levels

    - Token level: each token obtains a probability distribution over $N$ experts via the router, denoted $p(x_t)$
    - Sequence level: $\bar{p}_s = \frac{1}{T}\sum_{t=1}^T p(x_t)$, averaging all token distributions within a sequence
    - Domain level: $\bar{p}_j = \frac{1}{|\mathcal{B}_j|}\sum_{s \in \mathcal{B}_j} \bar{p}_s$, grouped and averaged by domain label

2. **JS divergence maximization**: $\mathcal{L}_{ED} = \frac{1}{\binom{M_B}{2}}\sum_{j<k} -\log(D_{JS}(\bar{p}_j || \bar{p}_k) + \epsilon)$

    - Maximizes the Jensen-Shannon divergence between routing distributions of all domain pairs
    - The negative log transformation amplifies gradients for small divergence values, preventing gradient vanishing

3. **Domain labeling schemes**: Two granularities

    - 3-Class: three coarse domains—English / Chinese / Math (derived directly from data sources)
    - 49-Class: a classifier maps English → 24 topics, Chinese → 24 topics, and Math → 1 category, yielding 49 fine-grained domains

### Theoretical Motivation — Diversity Decomposition
- **Decomposition Theorem (Proposition 1)**: Total routing diversity $D_{total} = D_{inter} + D_{intra}$
    - $D_{inter}$: inter-domain divergence — the degree to which different domains activate different experts
    - $D_{intra}$: intra-domain divergence — the degree to which tokens within the same domain activate different experts
- **Proposition 2**: $\mathcal{L}_{ED}$ directly increases $D_{inter}$, redistributing global diversity toward inter-domain differences
- Standard $\mathcal{L}_{LB}$ only attends to $D_{total}$ without controlling its allocation, whereas $\mathcal{L}_{ED}$ provides finer directional guidance
- The two losses are complementary: $\mathcal{L}_{LB}$ ensures sufficient total diversity, while $\mathcal{L}_{ED}$ channels that diversity into inter-domain differences, promoting expert specialization

## Key Experimental Results

### Main Results (three model scales, pre-trained from scratch on 100B tokens)

| Model | Method | CEval | MMLU | CMMLU | ARC-e | ARC-c | RACE-m | RACE-h | Avg. |
|------|------|-------|------|-------|-------|-------|--------|--------|------|
| 15B-A1.5B | Standard MoE | 28.0 | 25.8 | 25.6 | 47.4 | 28.2 | 50.5 | 43.6 | 35.59 |
| 15B-A1.5B | **+ED (49-class)** | **28.9** | **27.1** | **26.3** | **48.6** | **28.5** | **51.7** | **45.5** | **36.65** |
| 8B-A0.8B | Standard MoE | 25.8 | 24.5 | 25.0 | 43.2 | 23.6 | 42.7 | 36.5 | 31.61 |
| 8B-A0.8B | **+ED (49-class)** | **26.1** | **25.2** | **25.2** | **44.1** | **24.9** | **44.3** | **38.2** | **32.57** |
| 3B-A0.3B | Standard MoE | 23.8 | 23.1 | 24.2 | 35.0 | 22.6 | 37.8 | 32.1 | 28.37 |
| 3B-A0.3B | +ED (49-class) | 24.5 | 23.4 | 24.5 | 36.2 | 22.8 | 37.5 | 32.8 | 28.81 |

### Training Dynamics and Expert Analysis

| Analysis Dimension | Finding |
|---------|------|
| LM loss | All ED configurations converge to lower $\mathcal{L}_{LM}$; all $\beta$ settings outperform the baseline |
| Domain granularity | 49-class > 3-class > baseline; finer-grained domain labels yield greater benefit |
| Expert specialization | Layer 4 exhibits substantially higher specialization than other layers (middle-layer experts are most differentiated) |
| Computational overhead | Negligible additional training cost (only inter-domain divergence computation per batch is required) |
| Scaling effect | Performance gains increase with model scale (15B > 8B > 3B) |

### Key Findings
- Load balancing $\neq$ functional specialization: uniform utilization does not guarantee distinct expertise
- The ED loss guides experts to develop differentiated routing strategies across domains, forming an organized expert team
- The 49-class fine-grained domain classification outperforms the 3-class scheme, indicating that the informativeness of domain labels directly affects specialization quality

## Highlights & Insights
- **Paradigm shift from balance to specialization**: Standard MoE training focuses on load balancing ($D_{total}$), whereas this paper targets functional specialization ($D_{inter}$)—a more fundamental objective
- **Leveraging domain labels**: Pre-training data domain labels serve as free supervisory signals for guiding expert specialization, requiring zero additional annotation cost
- **Choice of JS divergence**: The symmetric and bounded JS divergence is more appropriate than KL divergence for measuring differences between routing distributions
- **Theoretical clarity**: The diversity decomposition theorem elegantly reveals the complementary relationship between $\mathcal{L}_{LB}$ and $\mathcal{L}_{ED}$

## Limitations & Future Work
- Domain labels are required; the method does not directly apply in fully unlabeled settings (though automatic labeling via a classifier is feasible, as demonstrated in this work)
- Validation is limited to three model scales (3B/8B/15B) and a relatively small training budget (100B tokens)
- The domain classification granularity (49 vs. 3) must be set manually; adaptive determination of optimal granularity remains an open problem
- Interaction effects with shared-expert architectures (e.g., DeepSeek-MoE) have not been explored
- Whether domain-label-guided fine-tuning after pre-training can retroactively induce specialization is left unexplored

## Related Work & Insights
- **vs. DeepSeek-MoE**: DeepSeek employs shared experts to capture common knowledge and mitigate routing expert redundancy, whereas this work uses inter-domain divergence maximization to directly drive routing expert differentiation—the two approaches are orthogonal and potentially composable
- **vs. ERNIE 4.5**: ERNIE enforces orthogonality of router weight matrices (unsupervised), while this work uses domain labels (supervised) to guide specialization—the supervised approach proves more effective
- **vs. Qiu et al. (global LB)**: Global batch load balancing enhances overall diversity; this work further steers the allocation of that diversity
- **Insight**: The "divide and conquer" design intent of MoE requires explicit support from the training objective; otherwise, experts degrade into "redundant generalists"

## Supplementary Analysis
- Core insight: load balancing only encourages global routing diversity without directing how that diversity is distributed — $\mathcal{L}_{ED}$ uses domain labels to redirect diversity toward inter-domain differences
- The Divergence Decomposition ($D_{total} = D_{inter} + D_{intra}$) is elegant — $\mathcal{L}_{LB}$ promotes $D_{total}$, while $\mathcal{L}_{ED}$ steers toward $D_{inter}$
- The superiority of 49-class over 3-class suggests that finer domain granularity enables more refined division of labor among experts
- Performance gains scale positively with model size (3B < 8B < 15B), indicating that larger models have greater untapped potential that benefits from structured specialization
- Computational overhead is nearly zero — $\mathcal{L}_{ED}$ is computed solely from existing routing logits via JSD

## Rating
- Novelty: ⭐⭐⭐⭐ Expert specialization via inter-domain divergence maximization is a novel angle; the theoretical decomposition is elegant
- Experimental Thoroughness: ⭐⭐⭐⭐ Three model scales + two domain classification granularities + expert behavior analysis
- Writing Quality: ⭐⭐⭐⭐ Problem analysis is clear; theoretical motivation is complete
- Value: ⭐⭐⭐⭐ Practically useful for MoE training; domain label utilization incurs low cost

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Ultra-Fast Language Generation via Discrete Diffusion Divergence Instruct](ultra-fast_language_generation_via_discrete_diffusion_divergence_instruct.md)
- [\[NeurIPS 2025\] Advancing Expert Specialization for Better MoE](../../NeurIPS2025/llm_efficiency/advancing_expert_specialization_for_better_moe.md)
- [\[ICLR 2026\] DND: Boosting Large Language Models with Dynamic Nested Depth](dnd_boosting_large_language_models_with_dynamic_nested_depth.md)
- [\[ICLR 2026\] One-Prompt Strikes Back: Sparse Mixture of Experts for Prompt-based Continual Learning](one-prompt_strikes_back_sparse_mixture_of_experts_for_prompt-based_continual_lea.md)
- [\[ICLR 2026\] EvoEngineer: Mastering Automated CUDA Kernel Code Evolution with Large Language Models](evoengineer_mastering_automated_cuda_kernel_code_evolution_with_large_language_m.md)

</div>

<!-- RELATED:END -->
