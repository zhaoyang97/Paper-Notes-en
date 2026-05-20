---
title: >-
  [Paper Note] GraphFLA: Augmenting Biological Fitness Prediction Benchmarks with Landscape Features
description: >-
  [NeurIPS 2025][Medical Imaging][Fitness landscape] GraphFLA is an efficient fitness landscape analysis framework that computes 20 biologically meaningful landscape features (ruggedness / epistasis / navigability / neutra…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "Fitness landscape"
  - "protein engineering"
  - "landscape features"
  - "model diagnostics"
  - "combinatorial mutations"
date: 2026-05-08
content_hash: bcb118143ac5dd85
---

# GraphFLA: Augmenting Biological Fitness Prediction Benchmarks with Landscape Features

**Conference**: NeurIPS 2025
**arXiv**: [2510.24826](https://arxiv.org/abs/2510.24826)  
**Code**: [https://github.com/COLA-Laboratory/GraphFLA](https://github.com/COLA-Laboratory/GraphFLA)  
**Area**: Bioinformatics / Benchmark Methodology
**Keywords**: Fitness landscape, protein engineering, landscape features, model diagnostics, combinatorial mutations

## TL;DR
GraphFLA is an efficient fitness landscape analysis framework that computes 20 biologically meaningful landscape features (ruggedness / epistasis / navigability / neutrality) across 5,300+ real-world landscapes (ProteinGym / RNAGym / CIS-BP), revealing that model performance is highly dependent on landscape topology—e.g., VenusREM outperforms ProSST on highly navigable landscapes but underperforms it on highly epistatic ones—while processing one million mutants in just 20 seconds (vs. 5 hours for MAGELLAN).

## Background & Motivation

**Background**: ProteinGym (217 DMS tasks) and RNAGym (31 tasks) serve as standard benchmarks for protein/RNA fitness prediction. Eighty-nine models compete on these benchmarks, typically ranked by mean Spearman correlation.

**Limitations of Prior Work**: (a) Aggregate scores mask substantial task-level variance—VenusREM achieves the highest mean but ranks first on only 14/217 tasks, while 44/89 models rank first on at least one task; (b) there is a lack of quantitative features describing "why each task is difficult"—only coarse metadata such as species and sequence length are available; (c) the only existing tool, MAGELLAN, is implemented in C only and is infeasible for datasets exceeding 100K mutants.

**Key Challenge**: Model performance depends on task-level properties (landscape topology), yet benchmarks lack such features, making it impossible to diagnose "which type of landscape requires which type of model."

**Goal**: To provide an efficient landscape analysis toolkit that computes interpretable meta-features for every benchmark task, shifting model comparison from "who achieves the highest average score" to "who performs best on which type of landscape."

**Key Insight**: The fitness landscape is a classical concept in evolutionary biology—treating mutation space as a topographic map in which the fitness of each mutant corresponds to its "elevation." Properties such as ruggedness, epistasis, and navigability govern the difficulty of both evolution and engineering.

**Core Idea**: Efficiently compute 20 landscape topological features → annotate 5,300+ fitness prediction tasks with meta-labels → explain why different models perform differently across tasks.

## Method

### Overall Architecture
Mutation data (sequences + fitness) → **Efficient landscape construction** (implicit neighbor generation, near-linear complexity) → **Computation of 20 landscape features** (4 categories: ruggedness / epistasis / navigability / neutrality) → **Applications**: correlation analysis with model performance; construction of a landscape-aware model comparison framework.

### Key Designs

1. **Efficient Landscape Construction**:

    - Function: Construct a directed fitness landscape graph from millions of mutants.
    - Mechanism: Instead of $O(n^2)$ all-pairs distance computation, single-mutation neighbors are generated directly for each mutant (linear complexity), with graph operations handled by the igraph C backend. One million mutants are processed in 20 seconds using 2 GB of memory.
    - Design Motivation: MAGELLAN requires 5 hours and runs out of memory at 100K mutants; GraphFLA supports up to $10^7$ mutants.

2. **20 Biologically Meaningful Landscape Features**:

    - Function: Quantify four categories of topological landscape properties.
    - Mechanism: **Ruggedness** (5 features): proportion of local optima $\phi_{lo}$, roughness-slope ratio, autocorrelation—measuring the "bumpiness" of the landscape. **Epistasis** (9 features): sign / magnitude / reciprocal epistasis, diminishing returns—measuring nonlinear interactions among mutations. **Navigability** (5 features): fitness-distance correlation (FDC), global optimum reachability—measuring whether greedy search can reach the global optimum. **Neutrality** (1 feature): proportion of zero-effect mutations.
    - Design Motivation: More than 100 candidate features were identified via an LLM-assisted review of 1,673 papers; 20 were selected based on frequency, biological relevance, coverage, and computational feasibility.

3. **155 Combinatorially Complete Empirical Landscape Datasets**:

    - Function: Collect combinatorially complete empirical landscapes for feature validation.
    - Mechanism: 155 landscapes (DNA 55 / protein 63 / RNA 37) were collected from 61 papers, comprising a total of 2.2 million sequences. Combinatorial completeness means all possible mutational combinations have been experimentally measured.
    - Design Motivation: Combinatorially complete landscapes enable precise topological analysis without estimation errors introduced by missing data.

### Loss & Training
- This is an analysis framework; no model training is involved.
- Applied to: ProteinGym (217 DMS tasks), RNAGym (31 tasks), CIS-BP (5,016 TF-binding landscapes).
- Total: 8,338 landscapes, 174 million mutants.

## Key Experimental Results

### Main Results (Correlation between Landscape Features and Model Performance)

Evaluated on Evo2-7b across 155 landscapes:
- 10 features exhibit |correlation| > 0.6 (strong)
- 6 features exhibit $0.3 < |\rho| < 0.6$ (moderate)

Key patterns:
- High ruggedness (low $\rho_a$) → lower model performance
- High reciprocal sign epistasis $\epsilon_{reci}$ → harder to predict
- Low navigability (FDC > 0) → models struggle
- High neutrality → unpredictable performance

### Landscape-Aware Model Comparison

| Comparison | Low-Epistasis Landscapes | High-Epistasis Landscapes |
|---|---|---|
| VenusREM vs ProSST | ProSST wins | VenusREM wins |
| Zero-shot vs supervised | Zero-shot wins | Supervised wins |
| Kermut (supervised) vs VenusREM | +0.53 Spearman $\rho$ for Kermut at FDC = 0.23 | — |

### Robustness Analysis

| Test | Result |
|---|---|
| Remove 10–50% of data | All features stable except global reachability |
| Add $0.2\sigma$ noise | All metrics consistent |
| Random mutation sampling bias | Features highly consistent |

### Key Findings
- Aggregate scores mask landscape-dependent performance differences—VenusREM is "best" overall but ranks first on only 14/217 tasks.
- Landscape features provide interpretable performance diagnostics—"Why is this protein hard to predict? Because epistasis is high."
- Zero-shot models outperform supervised models only on highly navigable landscapes—when the landscape is "flat," training data are unnecessary.
- Features are robust to missing data and noise, making them applicable in real-world settings where data quality is imperfect.

## Highlights & Insights
- **Paradigm shift from "who scores highest on average" to "who performs best on which landscape"**: This represents a major methodological advance in benchmark design—no longer summarizing everything with a single number.
- **The 20 landscape features serve as a "map" for protein engineering**: They inform researchers how rugged and complex the space they are exploring is, enabling selection of appropriate tools.
- **A 100× efficiency improvement makes large-scale analysis tractable**: MAGELLAN 5h → GraphFLA 20s, enabling analysis of 8,000+ landscapes.

## Limitations & Future Work
- The 155 combinatorially complete landscapes were collected from 61 papers, which may introduce publication bias.
- Definitions of certain features vary across implementations (e.g., diminishing returns epistasis $\epsilon_{DR}$).
- Feature estimation on incomplete landscapes may exhibit systematic bias.

## Related Work & Insights
- **vs. MAGELLAN**: The only previously available landscape analysis tool, but not scalable; GraphFLA is more than 100× faster.
- **vs. ProteinGym leaderboard**: The leaderboard reports only aggregate scores; GraphFLA provides task-level diagnostics.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Systematically introducing fitness landscape analysis into ML benchmarks represents an entirely new perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8,338 landscapes + 155 combinatorially complete datasets + robustness analysis.
- Writing Quality: ⭐⭐⭐⭐ Feature definitions are clear; application analyses are insightful.
- Value: ⭐⭐⭐⭐⭐ Provides indispensable landscape analysis infrastructure for protein/RNA engineering benchmarks.

## Key Experimental Results

### Model Performance vs. Landscape Features

| Landscape Feature | Effect on Model Performance |
|---|---|
| Higher ruggedness (high $\phi_{lo}$) | Accuracy decreases for all models |
| More epistasis (high $\epsilon_{sign}$) | Accuracy decreases for all models |
| Higher neutrality (high $\eta$) | Accuracy decreases for most models |
| Higher navigability (high FDC) | Accuracy increases for most models |

### Landscape Preferences of Different Models

| Model | Strengths | Weaknesses |
|---|---|---|
| VenusREM | Low-epistasis landscapes | High-ruggedness landscapes |
| ESM-1v | High-navigability landscapes | Neutral landscapes |
| MSA Transformer | Large-scale landscapes | Few-shot settings |

### Key Findings
- **Model performance is strongly dependent on landscape topology**—rugged and highly epistatic landscapes are harder for all models (answering Q1).
- **Different models excel on different landscape types**—even models with similar average scores exhibit entirely different landscape preferences (answering Q2).
- **GraphFLA successfully reproduces qualitative and quantitative conclusions from 61 papers**—validating the reliability of the framework.
- **Robust to missing data, noise, and biased sampling**—verified through synthetic landscape experiments.
- **The 5,016 TF-binding landscapes in CIS-BP** (174 million mutants) demonstrate scalability at extreme scale.

## Highlights & Insights
- The insight that **"not all tasks are equally important"** has broad implications for benchmark design—stratified evaluation by landscape difficulty is more informative than aggregate scores.
- **Landscape features are biologically meaningful meta-features**—they directly reflect evolutionary predictability rather than being arbitrary statistics.
- **GraphFLA processes one million mutants in 20s vs. 5h+ for competitors**—a substantial engineering achievement.

## Limitations & Future Work
- The 20 features may not fully capture all factors that determine model performance.
- Combinatorially complete landscapes are scarce (155 total); most benchmark data consist of randomly sampled mutants.
- The causal relationship between landscape features and model predictive performance has not been rigorously established.

## Related Work & Insights
- **vs. MAGELLAN**: The only existing tool, implemented in C with poor scalability. GraphFLA uses Python + C backend and is up to 1,000× faster.
- **vs. FLIP / ProteinGym / RNAGym**: These benchmarks provide only tasks. GraphFLA augments them with meta-features.
- **vs. landscape analysis in optimization (e.g., flacco)**: Designed for continuous black-box optimization. GraphFLA targets discrete sequence–fitness settings.

## Rating
- Novelty: ⭐⭐⭐⭐ Methodological innovation in augmenting fitness benchmarks with landscape features.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5,300+ landscapes × multi-modality × multi-model × reproduction validation × robustness analysis.
- Writing Quality: ⭐⭐⭐⭐ Taxonomy and feature tables are clearly presented.
- Value: ⭐⭐⭐⭐⭐ Significant contributions to both protein engineering and benchmark methodology.

### Supplementary Technical Details
- The 20 features were selected from an LLM-assisted review of 1,673 papers based on five criteria: frequency, biological relevance, coverage, and computational feasibility.
- GraphFLA processes 1 million mutants in 20 seconds; MAGELLAN exceeds 5 hours (a speedup of 900×+).
- The 5,016 TF-binding landscapes in the CIS-BP database contain 174 million total mutants—demonstrating scalability at extreme scale.
- The 155 combinatorially complete landscapes span DNA (55) / protein (63) / RNA (37), totaling 2.2 million sequences.
- Landscape construction uses a directed graph representation: nodes = mutants, edges = single-mutation steps pointing toward higher fitness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Steering Generative Models with Experimental Data for Protein Fitness Optimization](steering_generative_models_with_experimental_data_for_protein_fitness_optimizati.md)
- [\[NeurIPS 2025\] CrossNovo: Bidirectional Representations Augmented Autoregressive Biological Sequence Generation](bidirectional_representations_augmented_autoregressive_biological_sequence_gener.md)
- [\[NeurIPS 2025\] Toward a Vision-Language Foundation Model for Medical Data: Multimodal Dataset and Benchmarks for Vietnamese PET/CT Report Generation](toward_a_vision-language_foundation_model_for_medical_data_multimodal_dataset_an.md)
- [\[ICLR 2026\] Augmenting Representations with Scientific Papers](../../ICLR2026/medical_imaging/augmenting_representations_with_scientific_papers.md)
- [\[NeurIPS 2025\] Multiscale Guidance of Protein Structure Prediction with Heterogeneous Cryo-EM Data](multiscale_guidance_of_protein_structure_prediction_with_heterogeneous_cryo-em_d.md)

</div>

<!-- RELATED:END -->
