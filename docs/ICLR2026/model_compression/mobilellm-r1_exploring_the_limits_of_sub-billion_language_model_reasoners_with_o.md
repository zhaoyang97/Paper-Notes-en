---
title: >-
  [Paper Note] MobileLLM-R1: Exploring the Limits of Sub-Billion Language Model Reasoners with Open Training Recipes
description: >-
  [ICLR 2026][Model Compression][small model reasoning] Through careful data selection and an adaptive mixing strategy, MobileLLM-R1-950M is pretrained on only 4.2T tokens (11.7% of Qwen3's token budget) and matches or surpasses Qwen3-0.6B on reasoning benchmarks such as AIME, while fully open-sourcing both data sources and training recipes.
tags:
  - ICLR 2026
  - Model Compression
  - small model reasoning
  - data selection
  - influence scores
  - data mixing
  - on-device deployment
date: 2026-05-08
content_hash: a14fe64c49676925
---

# MobileLLM-R1: Exploring the Limits of Sub-Billion Language Model Reasoners with Open Training Recipes

**Conference**: ICLR 2026
**arXiv**: [2509.24945](https://arxiv.org/abs/2509.24945)
**Code**: [GitHub](https://github.com/facebookresearch/MobileLLM-R1)
**Area**: Model Compression
**Keywords**: small model reasoning, data selection, influence scores, data mixing, on-device deployment

## TL;DR
Through careful data selection and an adaptive mixing strategy, MobileLLM-R1-950M is pretrained on only 4.2T tokens (11.7% of Qwen3's token budget) and matches or surpasses Qwen3-0.6B on reasoning benchmarks such as AIME, while fully open-sourcing both data sources and training recipes.

## Background & Motivation
Large-model reasoning (the o1 paradigm) is reshaping the AI landscape, yet large models remain infeasible for on-device deployment—long chain-of-thought reasoning further aggravates KV-cache memory pressure. Two popular assumptions prevail: (1) reasoning ability emerges only in sufficiently large models; and (2) reasoning requires massive training data. Assumption 1 has been challenged by sub-billion models such as Qwen3-0.6B, but Assumption 2 has largely gone unquestioned.

**Core Problem**: Given strict capacity constraints, what is the most effective recipe for endowing small models with strong reasoning ability?

**Key Challenge**: Small models are extremely sensitive to noise—data noise can easily overwhelm their limited capacity, and neurons must encode more overlapping knowledge, increasing interference risk. Data quality and selection therefore matter far more for small models than for large ones.

**Core Idea**: Identify beneficial data sources via capability-aware leave-one-out (LOO) analysis, perform adaptive cross-capability data mixing using influence scores, and iteratively refine the mixture through data–model co-evolution during mid-training.

## Method

### Overall Architecture
A three-stage training pipeline: **Pretraining** (4.2T tokens, influence-weighted mixing) → **Mid-training** (100B tokens per stage, data–model co-evolution) → **Post-training** (SFT + reasoning SFT).

### Key Designs
1. **Capability-Aware Data Selection (LOO Analysis)**

   - *Function*: Identify each pretraining data source's contribution to reasoning capabilities.
   - *Mechanism*: One dataset is excluded at a time during training, and the change in NLL on three capability probe sets (Code / Math / Knowledge) is tracked. Influence is defined as $\Delta\mathcal{L}(\mathcal{D}_j, \mathcal{D}^P) = \mathbb{E}[\ell(z;\hat{\theta}_{-j}) - \ell(z;\hat{\theta})]$.
   - *Key Findings*: FineWeb-Edu acts as a cross-domain "glue"—removing it degrades all capabilities. StarCoder's contribution to mathematics exceeds OpenWebMath's contribution to code (counter-intuitive). Wikipedia contributes little to code or math.

2. **Cross-Capability Influence-Based Data Mixing**

   - *Function*: Compute optimal sampling weights for each data source based on influence scores.
   - *Mechanism*: The AutoMixer framework is used to efficiently approximate influence scores $\mathcal{I}(x_i, x_{\text{test}}; \theta) \approx -\nabla\mathcal{L}(x_{\text{test}})^\top H^{-1} \nabla\mathcal{L}(x_i)$. Joint influence aggregates contributions across capabilities and training stages, which are then converted into dataset-level weights $w_g = \frac{\rho_g}{\sum \rho_{g'}}$.
   - *Design Motivation*: Quantified cross-domain influence replaces heuristic uniform sampling; the resulting mixture ratios consistently outperform uniform sampling on unseen benchmarks.

3. **Mid-Training Data–Model Co-Evolution**

   - *Function*: Iteratively refine data mixing during mid-training.
   - *Mechanism*: At each stage, influence scores are computed for each sample using the current model; only samples with positive influence are retained, $\mathcal{D}_t = \{x_i : I(x_i; \theta_t) > 0\}$, and dataset weights are updated accordingly. The process iterates until the influence of most samples approaches zero (convergence, typically within 2 stages).
   - *Design Motivation*: As model capabilities evolve, a fixed data mixture becomes suboptimal; the process is framed as iterative denoising.

### Loss & Training
- **Pretraining**: Standard next-token prediction.
- **Post-training** consists of two sequential stages: Tulu-3-SFT (instruction alignment) → OpenMathReasoning + OpenCodeReasoning + OpenScienceReasoning (reasoning SFT).
- **Key Finding**: Sequential two-stage post-training outperforms joint training.

## Key Experimental Results

### Main Results

| Model | Params | Training Tokens | MATH | GSM8K | AIME | LCBv6 |
|-------|--------|----------------|------|-------|------|-------|
| OLMo-2-1.48B | 1.48B | 4T+ | ~20 | ~50 | 0.6 | 11.4 |
| SmolLM-2-1.7B | 1.7B | 11T | ~15 | ~40 | 0.3 | 7.7 |
| Qwen3-0.6B | 0.6B | 36T | ~55 | ~65 | ~10 | ~12 |
| **MobileLLM-R1-950M** | 0.95B | 4.2T | **57.8** | **68.5** | **15.5** | **13.7** |
| MobileLLM-R1-360M | 0.36B | — | 19.2 | 23.8 | — | 4.0 |
| MobileLLM-R1-140M | 0.14B | — | 4.8 | 3.7 | — | 1.1 |

### Ablation Study

| Configuration | MATH | GSM8K | LCBv6 | Notes |
|---------------|------|-------|-------|-------|
| Math SFT only (M) | 57.4 | 68.2 | 0.0 | Code capability lost |
| Code SFT only (C) | 16.2 | 31.0 | 12.0 | Math capability lost |
| M+C+S (sequential) | 57.8 | 68.5 | 13.7 | Best balance |
| M+C+S (joint) | 56.2 | 53.1 | 14.9 | Sharp GSM8K drop |
| w/o Tulu-3 stage | 56.2 | 68.2 | 13.1 | Instruction alignment helps |
| Raw mid-training data | Lower | Lower | — | Performance dip observed |
| Filtered mid-training data | Higher | More stable | — | Noise removal is effective |

### Key Findings
- Matching Qwen3's reasoning performance with only 11.7% of its token budget demonstrates that data quality far outweighs quantity.
- AIME scores of 15.5 vs. 0.6 (OLMo) and 0.3 (SmolLM) underscore the critical importance of pretraining data selection for small models.
- StarCoder contributes more to mathematics than OpenWebMath contributes to code—code provides strong positive transfer to reasoning.
- Data–model co-evolution during mid-training converges within 2 stages, with the influence distribution compressing toward zero.

## Highlights & Insights
- The concept of "benchmark-free, self-evolving data optimization"—optimizing data mixtures without consulting any downstream benchmark—is notably novel.
- Cross-domain influence relationships revealed by LOO analysis (e.g., positive transfer from code to math) offer actionable guidance for data selection.
- The convergence behavior of data–model co-evolution is elegant: the influence distribution progressively collapses toward zero, indicating that the data's information has been fully absorbed.
- The fully open-sourced training recipe and data sources make the work highly reproducible.

## Limitations & Future Work
- LOO analysis requires training a separate model for each data source, which is computationally expensive.
- Influence score computation relies on AutoMixer's Hessian approximation, which may introduce errors.
- Models smaller than 360M still exhibit very weak reasoning ability, suggesting a scale lower bound.
- The post-training stage reuses existing SFT datasets without applying equivalent influence-based filtering.

## Related Work & Insights
- **vs. Qwen3-0.6B**: Comparable performance achieved with 11.7% of the token budget, demonstrating substantial potential for data efficiency.
- **vs. OLMo-2**: MATH accuracy is more than 5× higher; the core gap lies in pretraining data quality.
- **vs. SmolLM-2**: MATH accuracy is more than 2× higher with fewer parameters.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Influence-driven data mixing and co-evolution are genuinely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive LOO analysis, stage-wise ablations, and cross-model comparisons.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear methodology, well-crafted figures, and deep insights.
- **Value**: ⭐⭐⭐⭐⭐ — Highly instructive for small model training; fully open-source.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] PASER: Post-Training Data Selection for Efficient Pruned Large Language Model Recovery](paser_post-training_data_selection_for_efficient_pruned_large_language_model_rec.md)
- [\[ICLR 2026\] Is Finer Better? The Limits of Microscaling Formats in Large Language Models](is_finer_better_the_limits_of_microscaling_formats_in_large_language_models.md)
- [\[ICLR 2026\] InftyThink: Breaking the Length Limits of Long-Context Reasoning in Large Language Models](inftythink_breaking_the_length_limits_of_long-context_reasoning_in_large_languag.md)
- [\[ICLR 2026\] Pedagogically-Inspired Data Synthesis for Language Model Knowledge Distillation](pedagogically-inspired_data_synthesis_for_language_model_knowledge_distillation.md)
- [\[ICLR 2026\] The Unseen Frontier: Pushing the Limits of LLM Sparsity with Surrogate-Free ADMM](the_unseen_frontier_pushing_the_limits_of_llm_sparsity_with_surrogate-free_admm.md)

<!-- RELATED:END -->
