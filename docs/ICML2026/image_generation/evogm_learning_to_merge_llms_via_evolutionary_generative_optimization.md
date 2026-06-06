---
title: >-
  [Paper Note] EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization
description: >-
  [ICML 2026][Image Generation][Model Merging] EvoGM reformulates the search for "task vector merging coefficients $\bm{\lambda}$" from evolutionary search with hand-crafted mutation operators into a learnable generative t…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Model Merging"
  - "Task Vectors"
  - "Evolutionary Algorithms"
  - "Generative Optimization"
  - "Cycle Consistency"
date: 2026-05-08
content_hash: 149a6ce25c34b5b5
---

# EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization

**Conference**: ICML 2026  
**arXiv**: [2605.29295](https://arxiv.org/abs/2605.29295)  
**Code**: https://github.com/JiangTao97/evogm (Yes)  
**Area**: Model Compression / LLM Merging / Evolutionary Search  
**Keywords**: Model Merging, Task Vectors, Evolutionary Algorithms, Generative Optimization, Cycle Consistency

## TL;DR
EvoGM reformulates the search for "task vector merging coefficients $\bm{\lambda}$" from evolutionary search with hand-crafted mutation operators into a learnable generative task. It employs a pair of MLP generators with cycle-consistency to learn the high-performance distribution from historical winner/loser pairs. By utilizing an outer loop for "basis shift" to periodically refresh the expert pool, the method achieves an average improvement of approximately 1.4% over SOTA PSO-Merging on 8 GLUE tasks and significantly outperforms existing methods on 10 unseen tasks with Qwen2.5-1.5B.

## Background & Motivation

**Background**: The cost of full-parameter fine-tuning for LLMs is increasingly prohibitive. Model merging has emerged as a mainstream paradigm for "training-free capability reuse" by directly combining multiple expert models in the parameter space. In task arithmetic, each expert is represented as a task vector $\bm{\tau}_i = \bm{\theta}_i - \bm{\theta}_{pre}$ relative to the pre-trained model. These are combined linearly via a coefficient vector $\bm{\lambda} \in \mathbb{R}^N$ to obtain $\bm{\theta}(\bm{\lambda}) = \bm{\theta}_{pre} + \sum_i \lambda_i \bm{\tau}_i$, reducing the merging problem to a low-dimensional ($N = $ number of experts) black-box optimization.

**Limitations of Prior Work**: Current approaches suffer from structural flaws. Heuristic-based methods like TA, TIES, DARE, and DELLA use "pruning + scaling + averaging," which are not learnable with respect to validation targets. Search-based methods like CMA, PSO-Merging, and Model Swarm utilize validation signals but rely on hand-crafted random mutations, **completely ignoring the performance landscape of the coefficient space**. In practical deployments with small validation sets and limited evaluation budgets (each fitness evaluation requires a full inference), the search efficiency of random perturbations is too low, often trapping the model in local optima.

**Key Challenge**: Under sparse and expensive fitness feedback, random mutations waste samples and fail to capture structural inductive biases. Meanwhile, historical search trajectories accumulate significant signals regarding which $\bm{\lambda}$ values are superior; **these signals are wasted by random operators**.

**Goal**: Upgrade evolutionary merging from "random perturbation" to "learning a proposal distribution from history." This involves solving three sub-problems: (i) extracting preference signals of "good vs. bad" from history; (ii) training a generator to map "bad" to "good" without collapsing to a single point; and (iii) preventing early saturation under a fixed expert basis.

**Key Insight**: Drawing from generative-model-based optimization in sparse-feedback scenarios, the coefficient search is treated as a data-driven generative task. Validation accuracy serves as the reward, and the set of "winners" provides samples of the high-performance distribution.

**Core Idea**: A **bidirectional MLP generator** is used to learn a "loser $\to$ winner" mapping to replace random mutations. Cycle-consistency is applied to prevent collapse, and an **outer loop "basis shift"** periodically treats the elite merge as a new expert to evolve both the search space and model capabilities.

## Method

### Overall Architecture
EvoGM replaces the **mutation operator** in the traditional "sample $\to$ evaluate $\to$ select $\to$ mutate" evolutionary loop with a learned generator and wraps this in an "expert basis refresh" layer. The complete pipeline is a nested loop:

- **Input**: Pre-trained base $\bm{\theta}_{pre}$, $N$ LoRA fine-tuned experts $\{\bm{\theta}_i\}_{i=1}^N$ (task vectors $\bm{\tau}_i = \bm{\theta}_i - \bm{\theta}_{pre}$), validation set $\mathcal{D}_{val}$, population size $P=2N$, outer rounds $R$, and inner iterations $T$ per round.
- **Outer Loop (Rounds)**: For each round, a set $\{\bm{\tau}_i\}$ is fixed. After an inner search, the top-$N$ coefficients $\bm{\lambda}^{(i)}$ are used to synthesize new experts $\bm{\theta}_i^{(r+1)} = \bm{\theta}(\bm{\lambda}^{(i)})$, and $\bm{\tau}_i^{(r+1)}$ are recalculated as the basis for the next round (**basis shift**).
- **Inner Loop (Iterations)**: The population $\mathcal{P}$ is initialized via a hybrid strategy (uniform average, one-hot, random uniform), evaluated on $\mathcal{D}_{val}$, and recorded in history $\mathcal{H}$. In each iteration, $\mathcal{H}$ is partitioned into a winner set $\mathcal{H}^+$ and a loser set $\mathcal{H}^-$ based on the top-$\rho$ (default $\rho=0.3$) fitness. **A pair of generators** $G_{-\to+}, G_{+\to-}$ is trained. $G_{-\to+}$ "pushes" the current population toward the high-performance region. The population is then re-evaluated, merged into $\mathcal{H}$, and the top-$P$ by fitness are selected for the next generation.
- **Output**: The $\bm{\lambda}^*$ with the highest historical fitness and its corresponding $\bm{\theta}(\bm{\lambda}^*)$.

The search is performed entirely in $\mathbb{R}^N$ ($N$ is the number of experts; $N=8$ or $10$ in the paper), requiring no gradient backpropagation through LLM parameters.

### Key Designs

1.  **Bidirectional MLP Generator + Winner-Loser Pair Training**:
    *   **Function**: Explicitly learns the "bad solution $\to$ good solution" transformation as a differentiable mapping, upgrading the mutation operator from random noise to a proposal with structural priors.
    *   **Mechanism**: Uses two 5-layer MLPs with tanh output layers to constrain $\bm{\lambda}$ to $[-1, 1]^N$. The forward generator is $G_{-\to+}: \mathcal{H}^- \to \mathcal{H}^+$, and the backward generator is $G_{+\to-}: \mathcal{H}^+ \to \mathcal{H}^-$. Training data is constructed via the Cartesian product $\mathcal{H}^- \times \mathcal{H}^+$ (implemented by independent mini-batch sampling). The optimization target pulls toward the winner mean $\bm{\mu}^+ = \frac{1}{|\mathcal{H}^+|}\sum_{\bm{\lambda}^+ \in \mathcal{H}^+} \bm{\lambda}^+$: $\mathcal{L}_{opt} = \mathbb{E}_{\bm{\lambda}^- \in \mathcal{H}^-}[\|G_{-\to+}(\bm{\lambda}^-) - \bm{\mu}^+\|_2^2]$.
    *   **Design Motivation**: Fitting a reward model directly in sparse-feedback optimization often leads to high variance; winner-loser preference signals are more stable. The backward generator provides a cycle constraint to prevent the generator from collapsing to a single point.

2.  **Cycle-Consistency Regularization to Prevent Mode Collapse**:
    *   **Function**: Ensures the composite mapping of the two generators is approximately identity, maintaining population geometric diversity while increasing fitness.
    *   **Mechanism**: Applies cycle constraints in both directions: $\mathcal{L}_{cyc} = \mathbb{E}_{\bm{\lambda}^-}[\|G_{+\to-}(G_{-\to+}(\bm{\lambda}^-)) - \bm{\lambda}^-\|_2^2] + \mathbb{E}_{\bm{\lambda}^+}[\|G_{-\to+}(G_{+\to-}(\bm{\lambda}^+)) - \bm{\lambda}^+\|_2^2]$. The total loss is $\mathcal{L}_{total} = \alpha_c \mathcal{L}_{cyc} + \alpha_o \mathcal{L}_{opt}$. If $G_{-\to+}$ maps all losers to a single winner center, $G_{+\to-}$ cannot reconstruct the diverse original losers, causing $\mathcal{L}_{cyc}$ to explode.
    *   **Design Motivation**: Minimizing only $\mathcal{L}_{opt}$ turns the generator into a constant function $\bm{\mu}^+$, leading to zero diversity. Borrowing the fixed-point idea from CycleGAN as an implicit regularizer is more stable than explicit entropy terms. Ablation shows removing cycle loss significantly hurts performance on reasoning tasks (NLGraph).

3.  **Outer Basis Shift: Evolving Expert Basis with Search**:
    *   **Function**: Breaks through the space spanned by fixed experts, avoiding the performance plateau where coefficient adjustments no longer yield improvements.
    *   **Mechanism**: At the end of each outer round $R$, the top-$N$ coefficients $\{\bm{\lambda}^{(i)}\}_{i=1}^N$ from $\mathcal{H}$ are instantiated into new models $\bm{\theta}_i^{(r+1)} = \bm{\theta}(\bm{\lambda}^{(i)})$. New task vectors $\bm{\tau}_i^{(r+1)} = \bm{\theta}_i^{(r+1)} - \bm{\theta}_{pre}$ are computed. The subsequent inner loop searches for $\bm{\lambda}$ within this **new basis spanned by elite merges**. Default configuration is $R=2, T=3$.
    *   **Design Motivation**: The upper bound of $\bm{\lambda}^*$ is restricted by the original experts. Since linear combinations of task vectors are themselves new directions, promoting them back to the basis is equivalent to "changing the coordinate system." This step often causes a temporary fluctuation followed by a secondary surge in the convergence curve.

### Loss & Training
The generator loss is $\mathcal{L}_{total} = \alpha_c \mathcal{L}_{cyc} + \alpha_o \mathcal{L}_{opt}$. In each inner iteration, the generators are reset and retrained to track the latest distribution of $\mathcal{H}$. EvoGM performs no gradient updates on LLM parameters; the computational overhead is concentrated on candidate validation evaluations.

## Key Experimental Results

### Main Results

**Seen tasks**: FLAN-T5-base merging 8 single-task experts on GLUE.

| Dataset (GLUE) | Metric | EvoGM | PSO-Merging (SOTA) | Gain |
|---|---|---|---|---|
| CoLA | acc | 71.1 | 68.2 | +2.9 |
| MNLI | acc | 82.8 | 83.8 | -1.0 |
| QQP | acc | 84.8 | 83.6 | +1.2 |
| RTE | acc | 82.2 | 81.2 | +1.0 |
| STS-B | acc | **80.9** | 71.9 | **+9.0** |
| **AVG (8 tasks)** | **acc** | **82.4** | 81.2 | **+1.2** |

**Unseen tasks**: 10 Tulu-v2 LoRA experts on Qwen2.5-1.5B for multi-task merging, evaluated on 8 benchmarks (knowledge/reasoning/safety).

| Benchmark (Test) | EvoGM | PSO-Merging | Model Swarm | Single Best |
|---|---|---|---|---|
| MMLU | **0.625** | 0.595 | 0.606 | 0.586 |
| MMLU-Pro | 0.224 | 0.232 | 0.236 | 0.232 |
| HellaSwag | **0.594** | 0.587 | 0.587 | 0.572 |
| GSM8K | **0.434** | 0.354 | 0.332 | 0.325 |
| NLGraph | **0.537** | 0.465 | 0.373 | 0.376 |
| TruthQA | **0.441** | 0.421 | 0.392 | 0.384 |
| AbstainQA | 0.121 | **0.147** | 0.095 | 0.119 |

EvoGM ranks first on 5 out of 8 test benchmarks. The improvements on reasoning tasks (GSM8K/NLGraph) are most significant (+8 to +14 relative points), suggesting learned proposal distributions are particularly effective for "hard" tasks.

### Ablation Study
Configuration: 4-expert merge, population 8, $R=2, T=2$.

| Configuration | Key Observation | Description |
|---|---|---|
| Full EvoGM | Best across all tasks | Full bidirectional generator + cycle + basis shift. |
| Single-Generator | Consistent drop | Removing the backward generator prevents cycle constraints, degrading search. |
| w/o Rounds | Significant drop on NLGraph/MMLU | Using 5 continuous inner iterations instead of "rounds + shift" proves refreshing the basis is necessary. |
| w/o Cycle Loss | Largest drop on NLGraph | Only $\mathcal{L}_{opt}$ remains; generator collapses to winner center, losing diversity needed for reasoning. |

### Key Findings
- **Cycle loss is critical**: Its removal leads to severe performance drops in reasoning tasks (like NLGraph) that require exploration, confirming that "preventing collapse > directly fitting reward."
- **Basis shift is not cosmetic**: Training curves show a "dip and surge" pattern at iteration 3 (round switch point), where basis shift kicks the search out of local optima.
- **Population sensitivity vs. Iteration stability**: Increasing population from 10 to 30 improves test accuracy by 0.0305; increasing iterations from 3 to 8 yields marginal gains, showing candidate coverage is the bottleneck.
- **Advantages in low-budget settings**: By default, only 200 samples are used for validation and 1000 for testing. Under sparse feedback, generative proposals are far more sample-efficient than random perturbations.

## Highlights & Insights
- **"Learning the evolutionary operator" is a clean paradigm shift**: Traditionally, mutation in EA/PSO is hand-crafted (Gaussian, polynomial). EvoGM replaces this with a learnable conditional generator using only weak winner/loser preference signals—a "reward-free + preference-only" setup valuable for expensive black-box optimizations.
- **CycleGAN as a subtle tool for low-dim search**: Applying cycle constraints in $\mathbb{R}^N$ adds negligible computation but replaces hard-to-tune entropy/KL hyperparameters with a constraint that has clear geometric meaning.
- **Basis shift reveals hidden dimensions of model merging**: Previous search-based merging treated task vectors as immutable structures. This paper posits that "elite merges are better task vectors," allowing the search space to evolve.
- **Transferability to expensive black-box search**: This template (winner-loser + cycle-generator) could be adapted for NAS, prompt optimization, or hyperparameter search in continuous low-dimensional spaces.

## Limitations & Future Work
- **Dimensionality constraints**: $N$ is limited to the number of experts (up to 10 in the paper). Whether the MLP structure scales to layer-wise coefficients (hundreds/thousands of dimensions) is unverified.
- **Evaluation overhead**: Each generation still requires full validation on 200 samples per candidate. Total wall-clock time remains limited by LLM inference; no specific speedup ratio over PSO-Merging was reported for fixed budgets.
- **Expert pool quality assumption**: Basis shift assumes elite merges are "better bases." If original experts are highly redundant, the new basis might not extend the capability upper bound.
- **Hyperparameter sensitivity**: Defaults for $\alpha_c / \alpha_o$, $\rho$, and the tanh range $[-1, 1]$ (which excludes task negation $| \lambda_i | > 1$) lack robustness analysis across different model families or task difficulties.

## Related Work & Insights
- **vs. PSO-Merging / Model Swarm**: Similar population-based search, but those use hand-crafted mutation/crossover. EvoGM's structural proposals are significantly better for complex landscapes like NLGraph.
- **vs. CMA (Akiba et al., 2025)**: CMA expands search to layer-wise dimensions and uses CMA-ES for adaptive covariance. EvoGM focuses on sample efficiency in the original space; these are orthogonal and potentially combinable.
- **vs. TIES / DARE / DELLA**: These are heuristic-based and ignore validation signals. EvoGM's performance on GLUE (+3 points over DELLA) highlights the ceiling of validation-guided search.
- **vs. AdaMerging**: AdaMerging uses unsupervised entropy minimization with some gradients. EvoGM is gradient-free and requires no unlabelled test data, making it more suitable for pure black-box deployment.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing learnable proposals, cycle consistency, and basis shift to model merging is a clear and effective methodological combination.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers FLAN-T5 and Qwen, seen and unseen tasks, and 10+ baselines, though wall-clock time comparisons are missing.
- Writing Quality: ⭐⭐⭐⭐ Clear algorithms and diagrams; logical derivations and readable formulas.
- Value: ⭐⭐⭐⭐ Significant for practical deployment with small validation sets and limited budgets.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Threshold-Guided Optimization for Visual Generative Models](threshold-guided_optimization_for_visual_generative_models.md)
- [\[ICML 2026\] A Systematic Investigation of RL-Jailbreaking in LLMs](a_systematic_investigation_of_rl-jailbreaking_in_llms.md)
- [\[ICML 2026\] $f$-Trajectory Balance: A Loss Family for Tuning GFlowNets, Generative Models, and LLMs with Off- and On-Policy Data](f-trajectory_balance_a_loss_family_for_tuning_gflownets_generative_models_and_ll.md)
- [\[ICML 2026\] Esoteric Language Models: A Family of Any-Order Diffusion LLMs](esoteric_language_models_a_family_of_any-order_diffusion_llms.md)
- [\[ICML 2026\] A Diffusive Classification Loss for Learning Energy-based Generative Models](a_diffusive_classification_loss_for_learning_energy-based_generative_models.md)

</div>

<!-- RELATED:END -->
