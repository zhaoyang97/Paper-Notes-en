---
title: >-
  [Paper Note] LLEMA: Evolutionary Search with LLMs for Multi-Objective Materials Discovery
description: >-
  [ICLR 2026][LLM/NLP][Materials Discovery] This paper proposes LLEMA, a framework that integrates the scientific prior knowledge of LLMs with chemistry-rule-guided evolutionary search and memory-driven iterative optimization, substantially outperforming generative and pure-LLM baselines across 14 multi-objective materials discovery tasks.
tags:
  - ICLR 2026
  - LLM/NLP
  - Materials Discovery
  - LLM Evolutionary Search
  - Multi-Objective Optimization
  - Surrogate Models
  - Memory Evolution
date: 2026-05-08
content_hash: 0ca5420ffb02cfe1
---

# LLEMA: Evolutionary Search with LLMs for Multi-Objective Materials Discovery

**Conference**: ICLR 2026
**arXiv**: [2510.22503](https://arxiv.org/abs/2510.22503)
**Code**: [https://github.com/scientific-discovery/LLEMA](https://github.com/scientific-discovery/LLEMA)
**Area**: LLM/NLP
**Keywords**: Materials Discovery, LLM Evolutionary Search, Multi-Objective Optimization, Surrogate Models, Memory Evolution

## TL;DR

This paper proposes LLEMA, a framework that integrates the scientific prior knowledge of LLMs with chemistry-rule-guided evolutionary search and memory-driven iterative optimization, substantially outperforming generative and pure-LLM baselines across 14 multi-objective materials discovery tasks.

## Background & Motivation

Materials discovery requires searching vast chemical and structural spaces while simultaneously satisfying multiple conflicting optimization objectives. Traditional machine learning approaches are constrained by the availability of large-scale annotated data. Although LLMs encode rich scientific prior knowledge, existing LLM-based methods exhibit several critical limitations:

**Reliance on prompt engineering or unguided generation**: Candidate materials produced are theoretically plausible yet thermodynamically unstable or unsynthesizable.

**Single-objective optimization**: Existing methods typically optimize a single property (e.g., band gap or conductivity), whereas real-world materials design is inherently a multi-objective problem.

**Absence of feedback mechanisms**: No closed-loop optimization exists that incorporates successes and failures into subsequent generation steps.

The core motivation of LLEMA is to combine the scientific knowledge of LLMs with the systematic exploration capability of evolutionary search, enabling multi-objective materials discovery while satisfying synthesizability constraints.

## Method

### Overall Architecture

LLEMA comprises four core components executed iteratively over $N$ rounds:

1. **Material Candidate Generation (A)**: An LLM generates candidate materials conditioned on task descriptions and property constraints.
2. **Crystallographic Representation (B)**: Generated materials are converted into structured CIF format (including lattice parameters, atomic species, and fractional coordinates).
3. **Physicochemical Property Prediction (C)**: Surrogate models predict material properties such as formation energy and band gap.
4. **Fitness Evaluation and Feedback (D)**: Multi-objective scoring combined with updates to success and failure memory pools.

### Key Designs

**Chemistry-rule-guided generation**: Rather than relying solely on prompting, LLEMA injects chemical design rules $\mathcal{R}$ into the generation prompt, including isoelectronic substitution, stoichiometry-preserving replacement, and oxidation-state consistency. These rules act as evolutionary operators, steering the search toward chemically valid regions.

**Multi-island evolutionary strategy**: The candidate population is partitioned into $m=5$ independent "islands," each maintaining its own success pool $\mathbb{M}^+$ and failure pool $\mathbb{M}^-$. At each iteration, islands are selected via Boltzmann sampling:

$$P_i = \frac{\exp(s_i / \tau_c)}{\sum_j \exp(s_j / \tau_c)}$$

where $s_i$ is the average score of island $i$ and $\tau_c$ is a temperature parameter. This parallel exploration strategy prevents premature convergence.

**Hierarchical property prediction**: Databases such as Materials Project are first queried for exact or approximate matches; for out-of-distribution candidates, pretrained surrogate models including CGCNN and ALIGNN provide predictions without requiring retraining.

### Loss & Training

The multi-objective scoring function is defined as:

$$S(\mathcal{T}, \mathcal{C}; \mathcal{M}_j) = \sum_{i=1}^{k} w_i \cdot \Phi_i(f_i(\mathcal{M}_j), c_i)$$

where $\Phi_i$ is a normalized reward function measuring the degree to which constraint $c_i$ is satisfied. Candidates are assigned to the success pool (all hard constraints satisfied) or the failure pool based on their scores, and top-$k$ sampling from both pools is used to construct prompts for the next iteration.

## Key Experimental Results

### Main Results

Hit Rate (HR) and Stability (Stab) comparisons across 14 industrially relevant materials discovery tasks:

| Method | Wide-Bandgap Semiconductor HR/Stab | SAW Acoustic Substrate HR/Stab | Solid Electrolyte HR/Stab | Transparent Conductor HR/Stab |
|---|---|---|---|---|
| CDVAE | 0.04/0.04 | 0.29/0.00 | 0.04/0.04 | 0.00/0.00 |
| MatterGen | 6.56/4.15 | 26.27/0.00 | 5.33/3.11 | 9.38/0.00 |
| LLMatDesign | 4.19/1.13 | 47.59/0.13 | 2.51/2.44 | 0.04/0.04 |
| LLEMA (Mistral) | 17.08/10.71 | 31.58/6.80 | 31.79/20.78 | 43.87/18.48 |
| **LLEMA (GPT)** | **33.62/22.42** | **59.88/10.74** | **46.17/25.37** | **39.11/14.85** |

LLEMA achieves substantially higher hit rates and stability scores on the vast majority of tasks, with particularly pronounced advantages in **stability**—indicating that the discovered candidates not only satisfy property constraints but are also thermodynamically feasible.

### Ablation Study

Incremental component contributions aggregated over four tasks:

| Method | Hit Rate↑ | Stability↑ | Memorization Rate↓ |
|---|---|---|---|
| Pure LLM | 4.4 | 1.8 | 95.3 |
| + Memory Feedback | 15.1 | 20.1 | 58.3 |
| + Mutation & Crossover | 29.8 | 21.5 | 25.3 |
| LLEMA (Full) | **30.2** | **27.6** | **16.6** |

Each component contributes distinctly: memory feedback substantially improves the hit rate (+10.7), evolutionary search further increases it (+14.7) while significantly suppressing memorization, and chemistry rules ultimately raise stability to 27.6 with the lowest memorization rate.

### Key Findings

1. **Surrogate models are indispensable**: Removing CGCNN and ALIGNN causes hit rate and stability to collapse to near zero, as the evolutionary process receives no effective reward signal.
2. **LLMs exhibit severe memorization**: Directly generated materials share 95.3% overlap with Materials Project entries; LLEMA reduces this to 16.6%.
3. **Pareto frontiers are dominated entirely by LLEMA**: In the wide-bandgap semiconductor and hard ceramic tasks, all Pareto-optimal solutions originate from LLEMA.

## Highlights & Insights

- **Explicit modeling of synthesizability constraints** is the core advantage distinguishing LLEMA from all baselines, ensuring that discovered materials are not only property-optimal but also synthesizable.
- The **dual success/failure memory pool** design is particularly elegant—success samples provide positive guidance while failure samples impose negative constraints, analogous to preference learning in RLHF.
- The paradigm of **LLMs as knowledge engines combined with evolutionary search as an optimization framework** may generalize to a broader range of scientific discovery settings.
- The proposed 14-task benchmark spanning electronics, energy, and optics fills a gap in multi-objective materials discovery evaluation.

## Limitations & Future Work

1. Dependence on surrogate model accuracy—predictions from CGCNN/ALIGNN may be unreliable for out-of-distribution candidates.
2. Iterative LLM queries incur substantial computational cost, particularly when using GPT-4o-mini.
3. Lack of experimental validation—all "discovered" materials remain at the computational level without wet-lab verification.
4. Chemistry rules are manually designed and may introduce domain bias, limiting exploration of entirely novel chemical spaces.

## Related Work & Insights

- **LLMatDesign** (Jia et al., 2024): LLM combined with self-reflection for single-objective materials design, lacking evolutionary mechanisms and multi-objective capability.
- **MatterGen** (Zeni et al., 2025): Diffusion model-based materials generation, complementary to LLEMA's LLM-driven approach.
- **FunSearch** (Romera-Paredes et al., 2024): LLM combined with evolutionary search for mathematical discovery; LLEMA extends this paradigm to materials science.
- Key insight: LLMs are better suited as "knowledge providers" rather than "end-to-end optimizers" in scientific discovery, and their value is maximized when coupled with structured search strategies.

## Rating

- **Novelty**: 7/10 — The framework design is sound, but each individual component (evolutionary search, surrogate models, memory pools) is established technology.
- **Technical Depth**: 7/10 — Strong engineering integration, but limited theoretical analysis.
- **Experimental Thoroughness**: 9/10 — Large-scale evaluation across 14 tasks with comprehensive ablation studies.
- **Writing Quality**: 8/10 — Clear structure with rich figures and tables.
- **Value**: 8/10 — Highly generalizable framework with open-sourced code and datasets.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Toward Safer Diffusion Language Models: Discovery and Mitigation of Priming Vulnerabilities](toward_safer_diffusion_language_models_discovery_and_mitigation_of_priming_vulne.md)
- [\[ICLR 2026\] Unsupervised Evaluation of Multi-Turn Objective-Driven Interactions](unsupervised_evaluation_of_multi-turn_objective-driven_interactions.md)
- [\[ICLR 2026\] d²Cache: Accelerating Diffusion-Based LLMs via Dual Adaptive Caching](d2cache_accelerating_diffusion-based_llms_via_dual_adaptive_caching.md)
- [\[ICLR 2026\] Near-Optimal Online Deployment and Routing for Streaming LLMs](near-optimal_online_deployment_and_routing_for_streaming_llms.md)
- [\[ICLR 2026\] GASP: Guided Asymmetric Self-Play For Coding LLMs](gasp_guided_asymmetric_self-play_for_coding_llms.md)

</div>

<!-- RELATED:END -->
