---
title: >-
  [Paper Note] DiSciPLE: Learning Interpretable Programs for Scientific Visual Discovery
description: >-
  [CVPR 2025][Remote Sensing][Interpretable Program Synthesis] The DiSciPLE framework is proposed to automatically synthesize interpretable Python programs for visual data analysis using an LLM-guided evolutionary algorithm. It achieves SOTA on scientific tasks such as population density estimation, reducing error by 35% compared to recent baselines while remaining fully interpretable.
tags:
  - "CVPR 2025"
  - "Remote Sensing"
  - "Interpretable Program Synthesis"
  - "Evolutionary Algorithms"
  - "LLM-Guided Search"
  - "Scientific Discovery"
  - "Remote Sensing Analysis"
date: 2026-05-08
content_hash: c25051174354f6c5
---

# DiSciPLE: Learning Interpretable Programs for Scientific Visual Discovery

**Conference**: CVPR 2025  
**arXiv**: [2502.10060](https://arxiv.org/abs/2502.10060)  
**Code**: [https://disciple.cs.columbia.edu](https://disciple.cs.columbia.edu)  
**Area**: Remote Sensing / Explainable AI  
**Keywords**: Interpretable Program Synthesis, Evolutionary Algorithms, LLM-Guided Search, Scientific Discovery, Remote Sensing Analysis

## TL;DR
The DiSciPLE framework is proposed to automatically synthesize interpretable Python programs for visual data analysis using an LLM-guided evolutionary algorithm. It achieves SOTA on scientific tasks such as population density estimation, reducing error by 35% compared to recent baselines while remaining fully interpretable.

## Background & Motivation

**Background**: Scientific disciplines (remote sensing, ecology, climate science, etc.) heavily rely on visual data to make predictions (e.g., estimating population density from satellite imagery). However, scientists require not only accurate predictions but also an understanding of the underlying mechanisms. Although explainable models like Concept Bottleneck Models (CBMs) are readable, their expressivity is restricted to a simple bag-of-concepts.

**Limitations of Prior Work**: LLM-based code-generation methods like ViperGPT are effective for standard computer vision tasks but fail on novel scientific tasks due to the lack of domain-specific knowledge in LLMs. Black-box deep learning models are accurate but lack interpretability, while symbolic regression methods cannot handle high-dimensional visual inputs.

**Key Challenge**: The trade-off between interpretability and expressivity: simple interpretable models (e.g., linear classifiers) are insufficiently accurate, whereas highly accurate deep learning models are uninterpretable.

**Goal**: How to automatically discover programs that are both accurate and interpretable for processing scientific visual data.

**Key Insight**: Reframe program search as an evolutionary algorithm problem, substituting traditional random crossover and mutation operations with LLMs to search the program space more intelligently by leveraging their coding capabilities and common-sense knowledge.

**Core Idea**: Utilize an LLM-driven evolutionary algorithm to search for interpretable Python programs that interface neural network primitives (e.g., open-vocabulary segmentation models) with mathematical and logical operations.

## Method

### Overall Architecture
Input: dataset $\mathcal{D}$, evaluation metric $\mathcal{M}$, a library of primitive functions $\mathcal{F}$ (including open-vocabulary segmentation models, mathematical/logical/image operations), and a textual task description. Output: an interpretable Python program $P: X \to Y$. Pipeline: Initial program population is generated zero-shot by the LLM $\to$ Parent selection based on fitness $\to$ Crossover/mutation executed by the LLM $\to$ Program critic $\to$ Program simplifier $\to$ Iterative evolution.

### Key Designs

1. **LLM-Driven Evolutionary Search**:

    - **Function**: Intelligently traverse the program space.
    - **Mechanism**: The initial population is generated zero-shot by the LLM based on the task description (e.g., "Given a satellite image, write a function to estimate population density") to ensure a non-random starting point. The crossover operation passes two parent programs along with their scores to the LLM, prompting it to synthesize a new program combining the strengths of both. Mutation operations introduce random modifications to programs with a certain probability. The common-sense knowledge of the LLM makes crossover and mutation significantly more effective than traditional symbolic methods.
    - **Design Motivation**: Random search in traditional evolutionary algorithms is extremely inefficient in high-dimensional program spaces. The coding capability and common-sense knowledge of LLMs can drastically accelerate convergence.

2. **Program Critic**:

    - **Function**: Provide fine-grained, stratified evaluative feedback.
    - **Mechanism**: Instead of assessing only the global score of the program, evaluation is stratified by data partitions (e.g., different land-use types). Subsets with poor performance are fed back to the LLM, guiding targeted improvements. For instance, if a program performs well in urban areas but poorly in rural ones, the LLM is instructed to specifically refine the logic for rural areas.
    - **Design Motivation**: Global scores fail to inform the LLM about which specific parts of the program require improvement.

3. **Program Simplifier**:

    - **Function**: Preserve interpretability and eliminate redundancy.
    - **Mechanism**: Dead code and redundant operations are removed via Abstract Syntax Tree (AST) analysis, and features with negligible contributions to the output are pruned through regression weight thresholding. This ensures programs do not become bloated and unreadable as they accumulate modifications during evolution.
    - **Design Motivation**: The evolutionary process continually accumulates code fragments; without simplification, programs grow increasingly verbose and difficult to interpret.

### Loss & Training
The fitness function utilizes task-specific metrics (e.g., L2-Log error for population density). The process begins with a population size M, evolves for T generations, where each generation applies crossover (mandatory) + mutation (with probability $\rho_m$) + critic + simplifier to each offspring.

## Key Experimental Results

### Main Results

**Population Density Estimation (Satellite Images)**:

| Method | L2-Log Error ↓ | Interpretable |
|------|------------|--------|
| Deep Learning Baseline | 0.3974 | ✗ |
| Concept Bottleneck | ~0.5 | ✓ (Limited) |
| LLM Zero-shot | 0.84 | ✓ |
| **DiSciPLE** | **0.2607** | **✓** |

**34 Demographic Indicators**:

| Method | L1 Error ↓ |
|------|---------|
| Deep Learning / Mean Baseline | 0.8527 |
| **DiSciPLE** | **0.8159** |

### Ablation Study

| Configuration | L2-Log Error ↓ |
|------|------------|
| Base version (w/o critic/simplifier) | 0.3159 |
| + Feature Set Expansion | 0.2906 |
| + Program Critic | 0.2873 |
| + Program Simplifier | **0.2607** |
| Remove LLM Common-Sense | 0.8401 |
| Remove Problem Context | 0.4498 |

### Key Findings
- LLM common-sense knowledge is critical: removing it causes the error to jump from 0.2607 to 0.8401, demonstrating that the domain intuition provided by the LLM is fundamental to the search's success.
- The contribution of the program simplifier is underestimated: it not only maintains interpretability but also enhances generalization capability by removing noisy features.
- DiSciPLE significantly outperforms deep learning models in out-of-distribution (OOD) generalization, signifying that the interpretable programs learn more robust patterns.
- Experiments across 34 demographic indicators demonstrate the broad applicability of the proposed method, showing it is not limited to a single task.

## Highlights & Insights
- **LLM as an Intelligent Search Engine**: Replacing the random operations of traditional evolutionary algorithms with LLMs is an elegant design; the programming capability and common-sense knowledge of LLMs yield much higher search efficiency than random search.
- **Interpretability Without Sacrificing Accuracy**: DiSciPLE's synthesized programs are more accurate than deep models (35% lower error), challenging the stereotype that "interpretability must compromise performance."
- **Democratizing Scientific Discovery**: The framework allows non-ML experts to automatically discover visual patterns by simply providing task descriptions and data, lowering the barrier to entry for scientific discovery.

## Limitations & Future Work
- Dependency on the code generation capability of LLMs, making program quality constrained by LLM performance.
- The library of primitive functions requires manual design, needing different sets of primitives for different domains.
- High computational cost of evolutionary search (due to multiple LLM calls and program evaluations).
- Validation is limited to remote sensing and demographics, leaving applicability to other scientific domains unexplored.

## Related Work & Insights
- **vs ViperGPT/VisProg**: These methods rely directly on LLM zero-shot code generation, which performs poorly on scientific tasks (error of 0.84). DiSciPLE iteratively refines the code through evolutionary search.
- **vs Concept Bottleneck Models (CBMs)**: Concept bottleneck models are restricted to simple linear functions, whereas the programs generated by DiSciPLE can contain complex logic and nested calls.
- **vs Symbolic Regression (SR)**: Traditional SR methods cannot handle high-dimensional image inputs, which DiSciPLE resolves by introducing vision foundation models as primitives.
- Holds significant value for automating scientific discovery and AI for Science.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ LLM-guided evolutionary search for synthesizing interpretable programs establishes a brand-new paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on three real-world scientific tasks with detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Strong motivation and clear problem formulation.
- Value: ⭐⭐⭐⭐⭐ Opens a new direction for interpretability in AI for Science.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GeoViS: Geospatially Rewarded Visual Search for Remote Sensing Visual Grounding](../../CVPR2026/remote_sensing/geovis_geospatially_rewarded_visual_search_for_remote_sensing_visual_grounding.md)
- [\[CVPR 2025\] Meta-Learning Hyperparameters for Parameter Efficient Fine-Tuning](meta-learning_hyperparameters_for_parameter_efficient_fine-tuning.md)
- [\[ICML 2026\] The Perception-Physics Paradox: Probing Scientific Alignment with TC-Bench](../../ICML2026/remote_sensing/the_perception-physics_paradox_probing_scientific_alignment_with_tc-bench.md)
- [\[CVPR 2025\] Learning Occlusion-Robust Vision Transformers for Real-Time UAV Tracking](learning_occlusion-robust_vision_transformers_for_real-time_uav_tracking.md)
- [\[CVPR 2025\] Hierarchical Dual-Change Collaborative Learning for UAV Scene Change Captioning](hierarchical_dual-change_collaborative_learning_for_uav_scene_change_captioning.md)

</div>

<!-- RELATED:END -->
