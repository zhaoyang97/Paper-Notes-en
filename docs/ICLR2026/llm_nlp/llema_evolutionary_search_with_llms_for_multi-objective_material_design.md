---
title: >-
  [Paper Note] LLEMA: Evolutionary Search with LLMs for Multi-Objective Materials Discovery
description: >-
  [ICLR 2026][LLM/NLP][materials discovery] This paper proposes LLEMA, a framework that integrates LLM scientific knowledge with chemistry-rule-guided evolutionary search and memory-driven iterative optimization…
tags:
  - "ICLR 2026"
  - "LLM/NLP"
  - "materials discovery"
  - "LLM evolutionary search"
  - "multi-objective optimization"
  - "crystal structure generation"
  - "surrogate models"
date: 2026-05-08
content_hash: 20d4d8be2d5f19c5
---

# LLEMA: Evolutionary Search with LLMs for Multi-Objective Materials Discovery

**Conference**: ICLR 2026  
**arXiv**: [2510.22503](https://arxiv.org/abs/2510.22503)  
**Code**: [github.com/scientific-discovery/LLEMA](https://github.com/scientific-discovery/LLEMA)  
**Area**: LLM NLP  
**Keywords**: materials discovery, LLM evolutionary search, multi-objective optimization, crystal structure generation, surrogate models  

## TL;DR

This paper proposes LLEMA, a framework that integrates LLM scientific knowledge with chemistry-rule-guided evolutionary search and memory-driven iterative optimization, achieving superior hit rates, stability, and Pareto front quality across 14 multi-objective materials discovery tasks.

## Background & Motivation

Materials discovery requires searching a vast combinatorial space of chemical compositions and crystal structures while simultaneously satisfying multiple, often conflicting, objectives. The traditional discovery process is resource-intensive and slow, and existing approaches face the following challenges:

1. **Traditional generative models** (CDVAE, G-SchNet, DiffCSP, MatterGen) require task-specific retraining, lack generalization capability, and do not leverage the extensive prior knowledge embedded in LLMs.
2. **Existing LLM-based methods** (e.g., LLMatDesign) rely on prompt engineering or unguided material generation, producing candidates that are theoretically plausible but often thermodynamically unstable or unsynthesizable.
3. **Single-objective limitation**: Most methods reduce materials discovery to a single-objective task, whereas real-world scenarios are inherently multi-objective (e.g., thermoelectric materials must simultaneously optimize electrical conductivity and thermal resistance).

LLEMA is the first framework to simultaneously possess all four properties: **domain knowledge integration, multi-objective optimization, rule-guided generation, and evolutionary optimization**.

## Method

### Overall Architecture

LLEMA comprises four core components (see Figure 1):

- **(A) Materials candidate generation**: The LLM generates candidates based on task descriptions and property constraints.
- **(B) Crystallographic representation**: Generated materials are converted into structured CIF files.
- **(C) Physicochemical property prediction**: Properties such as band gap and formation energy are predicted.
- **(D) Fitness evaluation and feedback**: Constraint satisfaction is assessed, and results are iteratively fed back via success/failure memory pools.

### Problem Formulation

The materials discovery task $\mathcal{T}$ is modeled as a constrained multi-objective optimization problem:

$$m^* = \arg\max_{m \in \mathcal{M}} \sum_i w_i f_i(m)$$

where each constraint $c_i$ may take the form of an interval, lower-bound, or upper-bound constraint:

$$c_i: f_i(m) \in [l_i, u_i] \quad \text{or} \quad c_i: f_i(m) \geq l_i \quad \text{or} \quad c_i: f_i(m) \leq u_i$$

### Key Designs

**Hypothesis Generation**:

At each iteration $n$, the LLM $\pi_\theta$ samples a batch of candidate materials from prompt $\mathbf{p}_n$, which consists of four components:

1. **Task specification**: Natural-language objectives and property constraints (e.g., "wide-bandgap semiconductor, bandgap ≥ 2.5 eV").
2. **Chemistry-informed design principles**: Rules such as isoelectronic substitution, stoichiometry preservation, and phase stability, serving as evolutionary operators.
3. **Demonstration samples**: Positive and negative examples sampled from the success pool $\mathbb{M}^+$ and failure pool $\mathbb{M}^-$.
4. **Crystallographic representation**: The LLM outputs crystal configurations in JSON format (chemical formula, lattice parameters, atomic coordinates).

**Physicochemical Property Prediction**:

- A hierarchical prediction system first queries the Materials Project database for exact matches.
- Out-of-distribution candidates are evaluated using surrogate models (CGCNN, ALIGNN).
- This yields a property vector $f(m) \in \mathbb{R}^d$.

**Fitness Evaluation and Memory Management**:

A composite scoring function is defined as:

$$S(\mathcal{T}, \mathcal{C}; \mathcal{M}_j) = \sum_{i=1}^k w_i \cdot \Phi_i(f_i(\mathcal{M}_j), c_i)$$

- Candidates satisfying all constraints ($S \geq 0$) are added to the success pool $\mathbb{M}^+$.
- Candidates violating constraints are added to the failure pool $\mathbb{M}^-$.

**Multi-Island Evolutionary Strategy**:

- The population is divided into $m=5$ independent islands.
- Islands are selected via Boltzmann sampling: $P_i = \frac{\exp(s_i/\tau_c)}{\sum_j \exp(s_j/\tau_c)}$
- Within each island, top-$k$ sampling from $\mathbb{M}^+$ and $\mathbb{M}^-$ constructs the prompt for the next iteration.

### Loss & Training

Users need only provide a CSV file containing task names and property constraints; the framework then automatically:

- Constructs prompts from the CSV and iteratively generates candidates.
- Applies surrogate models using publicly available pretrained weights (no retraining required).
- Prunes candidates violating hard constraints by assigning them low scores.

## Key Experimental Results

### Main Results: 14 Materials Discovery Tasks

Hit rate (H.R.) and stability (Stab.) are evaluated across 14 tasks spanning five domains: electronics, energy, coatings, optics, and aerospace.

| Method | Wide-Bandgap H.R/Stab | SAW/BAW H.R/Stab | Solid-State H.R/Stab | Piezo H.R/Stab | Transparent H.R/Stab |
|------|:-:|:-:|:-:|:-:|:-:|
| CDVAE | 0.04/0.04 | 0.29/0.00 | 0.04/0.04 | 42.19/0.00 | 0.00/0.00 |
| MatterGen | 6.56/4.15 | 26.27/0.00 | 5.33/3.11 | 21.64/0.00 | 9.38/0.00 |
| LLMatDesign | 4.19/1.13 | 47.59/0.13 | 2.51/2.44 | 32.16/1.38 | 0.04/0.04 |
| LLEMA (Mistral) | 17.08/10.71 | 31.58/6.80 | 31.79/20.78 | 67.11/4.84 | 43.87/18.48 |
| **LLEMA (GPT)** | **33.62/22.42** | **59.88/10.74** | **46.17/25.37** | 63.46/3.22 | 39.11/14.85 |

LLEMA substantially outperforms all baselines across nearly all tasks, with a particularly pronounced advantage in stability—baseline methods may generate candidates that satisfy property constraints but are thermodynamically unstable.

### Ablation Study: Component Contributions

| Method | Hit Rate↑ | Stability↑ | Memorization Rate↓ |
|------|:-:|:-:|:-:|
| LLM (direct generation) | 4.4 | 1.8 | 95.3 |
| + Memory feedback | 15.1 | 20.1 | 58.3 |
| + Mutation & crossover | 29.8 | 21.5 | 25.3 |
| **LLEMA (full)** | **30.2** | **27.6** | **16.6** |

### Key Findings

1. **Evolutionary optimization substantially reduces memorization**: The Materials Project repetition rate drops from 95.3% under pure LLM generation to 16.6% under LLEMA.
2. **Surrogate models are indispensable**: Removing surrogate models causes both hit rate and stability to collapse to <5%, and the search degenerates into trivial repetition.
3. **Convergence dynamics**: The proportion of valid candidates increases from ~27% at iteration 250 to ~33% at iteration 1000.
4. **Pareto front advantage**: For wide-bandgap semiconductor and hard/rigid ceramic tasks, all Pareto-optimal solutions originate from LLEMA.
5. **Discovered candidates align with domain expert research**: For example, ZrAl₂O₅ and Hf₀.₅Zr₀.₅O₂ correspond to well-known high-$k$ dielectric material families.

## Highlights & Insights

- **Chemical knowledge encoded as evolutionary operators**: Domain knowledge such as substitution rules, stoichiometry conservation, and oxidation state consistency is translated into operators that guide LLM generation, rather than being reduced to simple prompt engineering.
- **Multi-island evolutionary strategy**: Inspired by works such as FunSearch, the parallel island structure balances exploration and exploitation.
- **High practical utility**: New tasks can be initiated with only a CSV file; surrogate models use pretrained weights without retraining.
- **Benchmark contribution**: The paper introduces a benchmark of 14 industrially relevant multi-objective materials discovery tasks, each with well-defined physical constraints.

## Limitations & Future Work

1. Reliance on surrogate models (CGCNN, ALIGNN) for property prediction means that prediction errors accumulate and can misdirect the search.
2. Experimental validation is absent—newly discovered materials are verified only computationally and have not been synthesized in the laboratory.
3. Iterative LLM queries incur high costs; 250 iterations require a substantial number of API calls.
4. Chemistry rules are currently designed manually by domain scientists; automated rule discovery is a natural direction for extension.
5. Evaluation is limited to GPT-4o-mini and Mistral-Small; stronger LLMs may yield further performance gains.

## Related Work & Insights

- **Relationship to FunSearch (Romera-Paredes et al., 2024)**: LLEMA's multi-island evolutionary strategy is directly inspired by FunSearch, extending it from program synthesis to materials discovery.
- **Complementarity with MatterGen**: MatterGen performs inverse design via conditional sampling with diffusion models, whereas LLEMA employs LLM reasoning combined with evolutionary search; the two approaches are complementary.
- **Implications for AI4Science**: This work demonstrates how to integrate broad LLM knowledge with domain-specific constraints, and the paradigm is transferable to drug design, catalyst discovery, and related fields.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The first unified framework for materials discovery combining LLM evolutionary search, chemistry rules, and multi-objective optimization.
- **Technical Depth**: ⭐⭐⭐⭐ — The multi-layered framework incorporates surrogate models, multi-island evolution, and memory management.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 14 tasks, multiple baselines, comprehensive ablations, and qualitative analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure and rigorous problem formulation.
- **Value**: ⭐⭐⭐⭐ — Low barrier to entry (CSV only), though dependent on surrogate model quality.
- **Overall Rating**: ⭐⭐⭐⭐ (8/10)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Unsupervised Evaluation of Multi-Turn Objective-Driven Interactions](unsupervised_evaluation_of_multi-turn_objective-driven_interactions.md)
- [\[ACL 2026\] When Gradients Collide: Failure Modes of Multi-Objective Prompt Optimization for LLM Judges](../../ACL2026/llm_nlp/when_gradients_collide_failure_modes_of_multi-objective_prompt_optimization_for_.md)
- [\[ICLR 2026\] Toward Safer Diffusion Language Models: Discovery and Mitigation of Priming Vulnerabilities](toward_safer_diffusion_language_models_discovery_and_mitigation_of_priming_vulne.md)
- [\[NeurIPS 2025\] AceSearcher: Bootstrapping Reasoning and Search for LLMs via Reinforced Self-Play](../../NeurIPS2025/llm_nlp/acesearcher_bootstrapping_reasoning_and_search_for_llms_via_reinforced_self-play.md)
- [\[ICLR 2026\] WebOperator: Action-Aware Tree Search for Autonomous Agents in Web Environment](weboperator_action-aware_tree_search_for_autonomous_agents_in_web_environment.md)

</div>

<!-- RELATED:END -->
