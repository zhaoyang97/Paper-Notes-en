---
title: >-
  [Paper Note] Adaptive Kernel Design for Bayesian Optimization Is a Piece of CAKE with LLMs
description: >-
  [NeurIPS 2025][LLM/NLP][Bayesian Optimization] This paper proposes CAKE (Context-Aware Kernel Evolution), which leverages LLMs as crossover and mutation operators within a genetic algorithm framework to adaptively generate and evolve GP kernel expressions during Bayesian optimization. Combined with the BAKER ranking mechanism that balances model fit (BIC) and expected improvement (EI), CAKE consistently outperforms both fixed-kernel and adaptive-kernel baselines on tasks including hyperparameter optimization, controller tuning, and photonic chip design.
tags:
  - NeurIPS 2025
  - LLM/NLP
  - Bayesian Optimization
  - Gaussian Process
  - Kernel Design
  - LLM
  - Genetic Algorithm
date: 2026-05-08
content_hash: 6fbf66c7f80efc81
---

# Adaptive Kernel Design for Bayesian Optimization Is a Piece of CAKE with LLMs

**Conference**: NeurIPS 2025
**arXiv**: [2509.17998](https://arxiv.org/abs/2509.17998)
**Code**: [https://github.com/richardcsuwandi/cake](https://github.com/richardcsuwandi/cake)
**Area**: Bayesian Optimization / LLM Applications
**Keywords**: Bayesian Optimization, Gaussian Process, Kernel Design, LLM, Genetic Algorithm

## TL;DR
This paper proposes CAKE (Context-Aware Kernel Evolution), which leverages LLMs as crossover and mutation operators within a genetic algorithm framework to adaptively generate and evolve GP kernel expressions during Bayesian optimization. Combined with the BAKER ranking mechanism that balances model fit (BIC) and expected improvement (EI), CAKE consistently outperforms both fixed-kernel and adaptive-kernel baselines on tasks including hyperparameter optimization, controller tuning, and photonic chip design.

## Background & Motivation

**State of the Field**: Bayesian optimization (BO) relies on Gaussian processes (GPs) as surrogate models, where the choice of kernel determines the exploration–exploitation trade-off. Most BO frameworks default to SE or Matérn-5/2 kernels.

**Limitations of Prior Work**: Fixed kernel assumptions may not align with the true characteristics of the objective function (e.g., periodicity, non-stationarity), leading to slow convergence or suboptimal solutions. Existing adaptive kernel methods are either constrained to a limited kernel dictionary or require complex structural search procedures.

**Root Cause**: BO operates under a limited evaluation budget (typically tens to hundreds of queries), leaving very few data points available for kernel selection. Conventional optimization methods cannot function effectively in such few-shot settings.

**Paper Goals**: How can expressive kernel functions be adaptively constructed and evolved during the BO process, such that a well-matched kernel can be identified quickly under few-shot conditions?

**Starting Point**: LLMs are inherently capable of few-shot learning and in-context reasoning, enabling them to "understand" structural patterns in observational data and express these patterns through kernel function grammars.

**Core Idea**: Treat LLMs as intelligent operators within a genetic algorithm, prompting them to propose and evolve kernel expressions conditioned on observed data.

## Method

### Overall Architecture
A population of kernel functions $\mathbb{K}$ is maintained. At each BO iteration: (1) LLMs perform crossover and mutation to generate new kernels; (2) BIC evaluates the goodness of fit for each kernel; (3) BAKER integrates BIC-based weights with expected improvement to select the best kernel; (4) the selected kernel determines the next query point.

### Key Designs

1. **LLM as Genetic Operator**:

    - **Function**: GPT-4o-mini is used as crossover and mutation operators to propose new kernel expressions conditioned on the observed data context.
    - **Mechanism**: *Crossover* — two parent kernels are sampled from the population according to fitness, and the LLM is prompted to combine them using operators (+, ×) to produce a new kernel. *Mutation* — the best kernel is selected, and the LLM is prompted to replace one of its base kernels with another. The base kernel set = {SE, PER, LIN, RQ, Matérn-3/2, Matérn-5/2}.
    - **Design Motivation**: The in-context learning capability of LLMs is analogous to implicit Bayesian inference, enabling the inference of structural properties (periodicity, linear trends, etc.) from a small number of data points more efficiently than random search. Additionally, LLMs verbalize their reasoning, enhancing interpretability.

2. **BAKER Ranking (BIC-Acquisition Kernel Ranking)**:

    - **Function**: Jointly considers model fit quality and acquisition function value to select the optimal kernel.
    - **Mechanism**: Each kernel's weight is computed as $w_k = \frac{\exp(-\text{BIC}_k)}{\sum_{k'} \exp(-\text{BIC}_{k'})}$, and the optimal kernel is selected as $k^* = \arg\max_{k} w_k \cdot \alpha(\mathbf{x}_{t,k}; \mathcal{D}, k)$, where $\alpha$ denotes normalized EI.
    - **Design Motivation**: Some kernels may achieve a good fit (low BIC) while proposing query points with little expected improvement. BAKER avoids this pitfall.

3. **Kernel Grammar and Population Management**:

    - **Function**: Recursively constructs the kernel space based on a generalized kernel grammar and maintains a top-$n_p$ population.
    - **Mechanism**: $\mathbb{K}_i = \{\mathcal{T}_j(k_1, k_2) | k_1, k_2 \in \mathbb{K}_{i-1}\} \cup \mathbb{K}_{i-1}$; crossover is applied $n_c=5$ times, mutation probability $p_m=0.7$, and population size $n_p=10$.
    - **Design Motivation**: Population-based evolution preserves diversity and prevents premature convergence to a single kernel structure.

### Loss & Training
- LLMs are used entirely in an in-context manner without fine-tuning.
- GP hyperparameters are fit via MLE.
- BIC serves as the kernel fitness measure: $\text{BIC} = -2 \log \hat{L} + p \log n$.
- EI (Expected Improvement) is the default acquisition function.

## Key Experimental Results

### Main Results (HPOBench: 60 tasks = 12 datasets × 5 ML models)

| Method | Average Rank | Notes |
|--------|-------------|-------|
| CAKE (Ours) | **1st** | Highest average rank across 60 tasks |
| ABO | 2nd | Kernel selection via BO |
| CKS | 3rd | Greedy kernel structure search |
| EGP | 4th | GP ensemble |
| DGP | 5th | Deep GP |
| Fixed (M5) | ~6th | Fixed Matérn-5/2 kernel |

### Ablation Study

| Configuration | Performance | Notes |
|---------------|-------------|-------|
| CAKE (full) | Best | Complete method |
| w/o BAKER (BIC-only selection) | Degraded | Low BIC does not imply high EI |
| w/o Mutation | Moderate degradation | Mutation increases kernel diversity |
| w/o LLM (random crossover/mutation) | Significant degradation | LLM reasoning is the critical factor |

### Key Findings
- **CAKE yields the greatest advantage in early optimization stages (when data is scarce)**: This confirms that LLM few-shot reasoning is most impactful under data-limited conditions.
- **Learned kernel expressions are interpretable**: For example, "LIN + (PER × SE)" encodes "local periodicity with a linear trend," and the LLM provides natural language explanations for why a given structure is selected.
- **Performance gaps are substantial and consistent across tasks**: Fixed-kernel methods exhibit inconsistent performance across tasks (e.g., Matérn-5/2 performs well on SVM/RF but poorly on LR/XGB), while CAKE maintains a uniform advantage.

## Highlights & Insights
- **LLM + genetic algorithm as a novel kernel search paradigm**: This approach requires neither gradients nor a predefined kernel dictionary; LLMs autonomously propose and interpret kernel structures. The framework is transferable to any machine learning problem involving structural search.
- **BAKER balances fit and improvement**: A simple yet effective mechanism that resolves potential inconsistencies between BIC and EI.
- **Zero-shot kernel design**: Without prior knowledge of the objective function, the LLM infers functional structure from a handful of evaluation points and proposes appropriate kernels, demonstrating LLM potential as a "scientific hypothesis generator."

## Limitations & Future Work
- **Dependence on commercial LLM APIs**: The use of GPT-4o-mini requires multiple API calls per BO iteration, incurring non-negligible cost and latency.
- **Limited kernel grammar**: Only two composition operators (+/×) and six base kernels are used; more complex constructions (e.g., convolutional kernels, deep kernels) remain unexplored.
- **Uncontrolled LLM outputs**: LLMs occasionally generate invalid kernel expressions, necessitating error handling.
- **Scalability**: Performance on high-dimensional objectives (d > 20) has not been sufficiently validated.
- **Fair comparison with vanilla BO**: CAKE consumes additional computational resources through LLM inference.

## Related Work & Insights
- **vs. CKS (Compositional Kernel Search)**: CKS uses greedy search to traverse the kernel grammar space, whereas CAKE employs LLM-guided proposals to avoid combinatorial explosion.
- **vs. ABO (Automated BO)**: ABO treats kernel selection as a meta-BO problem; CAKE leverages LLM in-context reasoning for more efficient kernel adaptation.
- **vs. Deep GP**: DGP increases model expressiveness by stacking GP layers at the cost of greater computational complexity; CAKE achieves increased expressiveness at the kernel expression level with lower overhead.

## Rating
- Novelty: ⭐⭐⭐⭐ Using LLMs as kernel evolution operators is a novel idea, though the overall paradigm follows the common LLM-for-optimization pattern.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 60 hyperparameter optimization tasks, controller tuning, and photonic chip design.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear; the food-themed naming (CAKE/BAKER) is memorable.
- Value: ⭐⭐⭐⭐ Directly useful for BO practitioners, though reliance on LLM APIs limits applicability in certain settings.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] PRESTO: Preimage-Informed Instruction Optimization for Prompting Black-Box LLMs](presto_preimage-informed_instruction_optimization_for_prompting_black-box_llms.md)
- [\[NeurIPS 2025\] AutoDiscovery: Open-ended Scientific Discovery via Bayesian Surprise](autodiscovery_open-ended_scientific_discovery_via_bayesian_surprise.md)
- [\[ACL 2026\] ChatHLS: Towards Systematic Design Automation and Optimization for High-Level Synthesis](../../ACL2026/llm_nlp/chathls_towards_systematic_design_automation_and_optimization_for_high-level_syn.md)
- [\[NeurIPS 2025\] System Prompt Optimization with Meta-Learning](system_prompt_optimization_with_meta-learning.md)
- [\[NeurIPS 2025\] EvoRefuse: Evaluating and Mitigating LLM Over-Refusal via Evolutionary Prompt Optimization](evorefuse_evolutionary_prompt_optimization_for_evaluation_and_mitigation_of_llm_.md)

<!-- RELATED:END -->
