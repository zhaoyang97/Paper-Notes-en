---
title: >-
  [Paper Note] Raising the Bar: Investigating the Values of Large Language Models via Generative Evolving Testing
description: >-
  [ICML 2025][Social Computing][Value Alignment Evaluation] This paper proposes the GETA framework, which integrates Computerized Adaptive Testing (CAT) from psychometrics with Automatic Item Generation (AIG). Utilizing a variational IRT model and an LLM-driven item generator, GETA dynamically probes the value boundaries of LLMs to address the "evaluation chronoeffect" (data leakage and difficulty saturation) inherent in static benchmarks.
tags:
  - "ICML 2025"
  - "Social Computing"
  - "Value Alignment Evaluation"
  - "Adaptive Testing"
  - "Item Response Theory"
  - "Automatic Item Generation"
  - "Evaluation Chronoeffect"
date: 2026-05-08
content_hash: 8bb2eb81165b6b26
---

# Raising the Bar: Investigating the Values of Large Language Models via Generative Evolving Testing

**Conference**: ICML 2025  
**arXiv**: [2406.14230](https://arxiv.org/abs/2406.14230)  
**Code**: None  
**Area**: Social Computing  
**Keywords**: Value Alignment Evaluation, Adaptive Testing, Item Response Theory, Automatic Item Generation, Evaluation Chronoeffect

## TL;DR

This paper proposes the GETA framework, which integrates Computerized Adaptive Testing (CAT) from psychometrics with Automatic Item Generation (AIG). Utilizing a variational IRT model and an LLM-driven item generator, GETA dynamically probes the value boundaries of LLMs to address the "evaluation chronoeffect" (data leakage and difficulty saturation) inherent in static benchmarks.

## Background & Motivation

The evaluation of LLM value alignment faces a core challenge—the **evaluation chronoeffect**:

**Data Leakage**: Static benchmarks (e.g., RealToxicityPrompts, ETHICS) are highly susceptible to leaking into the training corpora of newer models, leading to artificially inflated safety scores. For instance, successive versions of GPT exhibit progressively decreasing toxicity on RealToxicityPrompts, yet they continue to produce substantial harmful outputs when tested on newly compiled datasets.

**Difficulty Saturation**: Due to the rapid iteration of LLMs, static test suites often fail to offer sufficient difficulty to differentiate performance across models, resulting in ceiling effects.

**Psychometric Analogy**: This is conceptually equivalent to the "separation of teaching and testing" dilemma in human education—once a specific exam becomes the direct learning objective, its power of discrimination plummets.

Limitations of Prior Work:
- **Static Evaluation (SE)**: Calculates metrics directly on frozen datasets, which is static and defenseless against leakage and difficulty saturation.
- **Computerized Adaptive Testing (CAT)**: While capable of adaptive item selection, it still relies on a static candidate item pool and struggles to transcend predefined difficulty boundaries.
- **Red-Teaming Methods** (e.g., GPTFuzzer, SAP): Primarily designed to expose individual vulnerabilities rather than provide systematic difficulty control and joint capability estimation.

## Method

### Overall Architecture

GETA (Generative Evolving Testing of vAlues) consists of three core components that are jointly trained:

1. **Variational Item Response Theory Model (VIRT)**: Employs variational inference (instead of MLE) to jointly estimate candidate ability $a_i$ and item parameters $d_j=(b_j, c_j)$.
2. **LLM-driven Item Generator**: Based on LLaMA-3-8B; generates novel evaluation items corresponding to specified difficulty parameters.
3. **Selective Generation and Iterative Evolution Strategy**: Dynamically adjusts item difficulty, filters generated tests, and updates the generator model during the testing pipeline.

### Key Designs

1. **Variational IRT (VIRT) Estimator**

    - **Value Estimator** $q_\theta(a_i | \mathbf{y}_i, \mathbf{d})$: Built upon a two-layer Transformer encoder, it infers the target LLM's value conformity $\hat{a}_i^t$ based on its historical response accuracy sequence and the corresponding item parameters.
    - **Item Parameter Estimator** $q_\phi(d_j | \mathbf{y}_{\cdot,j})$: Infers the objective item difficulty $b_j$ and discrimination $c_j$ based on the response patterns of all candidate LLMs to that item.
    - The probability of a correct response is simulated using the standard 2PL-IRT model: $p(y_{i,j}=1|a_i, b_j, c_j) = \frac{1}{1+\exp(-c_j(a_i - b_j))}$
    - Compared to traditional Maximum Likelihood Estimation (MLE), variational inference demonstrates robust stability under sparse data regimes and easily integrates into a joint variational learning framework with the generator.

2. **LLM Item Generator and Joint Training**

    - The generator $p_\omega(x|d)$ leverages LLaMA-3-8B, fine-tuned via Prefix Adapter + LoRA. It maps specific difficulty inputs $d=(b,c)$ to corresponding evaluation text.
    - The joint training objective is formulated as the maximization of the ELBO for joint distribution $p(\mathbf{x}, \mathbf{y})$:
        - First term: IRT Likelihood—predicting response correctness based on ability and item parameters.
        - Second term: Generator Loss—learning the direct conditional mapping from difficulty coordinates to natural language items.
        - Last two terms: KL-divergence regularization—imposing standard Gaussian priors on both the ability and difficulty parameters.
    - This joint formulation enables mutual advancement: VIRT yields accurate difficulty annotations for generating diverse items, while the generator expands the item bank to out-of-distribution regions to refine IRT coordinates.

3. **Selective Generation and Iterative Evolution**

   The core mechanism promotes **self-evolution** of the generator during the assessment run to probe beyond the difficulty limits of static databases:
    - **Optimal Difficulty Estimation**: For step $t$, the system solves for the optimal difficulty $d^* = (b^*, c^*)$ that maximizes Fisher information corresponding to the current ability pool estimate $\hat{a}^t$, yielding the analytical solution $b^* = \hat{a}^t$ (matching test difficulty directly to examinee capability).
    - **Fault-tolerant Sampling**: Difficulty coordinates are sampled in the neighborhood $[d^*-\epsilon, d^*+\epsilon]$ around $d^*$ to generate $k_1=100$ items.
    - **Filtering and Evolution**: Once candidates reply to these newly generated items, their true difficulty $\hat{d}$ is re-estimated via $q_\phi$:
      - If $|\hat{d} - d^*| < \delta_1$: The item matches target specifications and is used to update the candidate's ability estimates.
      - If $|\hat{d} - d^*| > \delta_2$: The model-generated item deviates severely from expected parameterization. Such items are stored in dataset $\mathcal{D}$ for future model reinforcement.
    - Once the size of $\mathcal{D}$ reaches $t \times k_2$, a scheduled round of generator fine-tuning is triggered using $\mathcal{D}$ to **expand the range of printable difficulties**.

### Loss & Training

The global objective function (Eq. 4) is decomposed into four components:

$$\mathcal{L}(\theta, \phi, \omega) = \mathbb{E}_{\hat{p}(x,y) + \hat{p}(y)q(x|y)} [\mathcal{L}_\mathcal{I}(\theta, \phi) + \beta \mathcal{L}_\mathcal{G}(\omega)] - \beta \mathbb{E}_{\hat{p}(y)}[H[q(x|y)]]$$

- **Variational IRT Loss** $\mathcal{L}_\mathcal{I}$: IRT Likelihood + KL Regularization terms.
- **Generator Loss** $\mathcal{L}_\mathcal{G}$: Standard conditional language modeling loss (generating contextual queries conditioned on target metrics).
- **Entropy Regularization** $H[q(x|y)]$: Maximizes Shannon entropy to prompt diversity in item output texts.
- **Training Pipeline**: Initialize VIRT and generator models on static seed data $\to$ Proceed into the active testing phase $\to$ For each round, spawn new queries, update student profiles, filter out deviated items $\to$ Periodically fine-tune the generator module.

Key hyperparameters: $T{=}10$ test rounds, $k_1{=}100$ (candidate items per round), $k_2{=}640$ (evolution threshold), $\beta{=}0.1$, $\epsilon{=}0.5$, $\delta_2{=}0.5$.

## Key Experimental Results

### Main Results

Data: 15k test items (5k for each of the 3 categories), compiled from 12 distinct datasets including BBQ, ETHICS, RealToxicityPrompts, and HarmfulQA. 8 candidate LLMs.

| Evaluation Method | Va-L (Higher is better) | Va-I | Va-O | Overall |
|---------|----------------|------|------|---------|
| SE (Static) | 0.30 | 0.55 | 0.49 | — |
| CAT | 0.41 | 0.79 | 0.68 | — |
| NCAT | 0.30 | 0.50 | 0.44 | — |
| **GETA** | **0.95** | **0.97** | **0.84** | **0.88** |

GETA consistently outperforms all baseline methods across concurrent validity variables, showing its most prominent advantage in the highly representative Va-L (correlation with authoritative leaderboards) benchmark (+0.54 compared to CAT).

### Ablation Study

| Configuration | Va-L | Va-I | Va-O | Overall | Description |
|------|------|------|------|---------|------|
| GETA (Full) | 0.890 | 0.944 | 0.793 | 0.875 | Full framework |
| w/o VIRT | 0.431 | 0.527 | 0.505 | 0.488 | Replacing variational inference with MLE causes validity to crash |
| w/o AIG | 0.864 | 0.878 | 0.834 | 0.859 | Removing the generator and using a static test bank drops performance by 2% overall |
| w/o Both | 0.643 | 0.847 | 0.786 | 0.759 | Degrades to vanilla CAT, dropping by 13.3% |
| w/o Update | 0.866 | 0.949 | 0.790 | 0.868 | No iterative evolution, dropping Va-L by 2.4% |
| w/o Transf. | 0.764 | 0.868 | 0.704 | 0.778 | Replacing Transformer with RNN |

### Key Findings

1. **VIRT Is Pivotal**: Removing variational inference leads to a critical drop in validity of nearly 40%, illustrating that variational inference is significantly superior to MLE under low-sample regimes (e.g., 5k per type).
2. **Evolutionary Updating Chiefly Boosts Va-L**: Since global safety leaderboards consistently feature harder queries, GETA's co-evolutionary system continuously adapts to keep alignment congruent with current industry metrics.
3. **GETA Uncovers Counter-intuitive yet True Findings**: For instance, LLaMA2-70B shows inferior social bias performance compared to LLaMA2-7B (80.91% vs 39.67% biased outputs) because larger models show excessive instruction follower patterns (even when instructions contain implicit biases).
4. **Adaptive Scaling Works**: While multiple top models achieve indistinguishable safety scores on static benchmarks, items produced by GETA accurately pinpoint individual model boundaries.
5. **Robustness to Candidate Size**: Even when testing a sparse set of 4 candidate LLMs, GETA retains massive validity (Va-I=0.999, Va-O=0.980).

## Highlights & Insights

- **Deep Integration of Psychometrics and AI Evaluation**: The first framework to unify CAT + IRT + AIG + Language Modeling into a singular theoretical model, offering a paradigm shift in AI evaluation.
- **Addressing a Real and Critical Problem**: The evaluation chronoeffect is a pressing reality. The paper leverages empirical evidence from GPT version updates to powerfully motivate the research.
- **Outstanding Efficiency**: Yields evaluation conclusions consistent with large-scale leaderboards using only ~150 adaptive test items.
- **Incredible Joint Training Design**: VIRT supplies high-fidelity difficulty labels to the generator, which in turn offers out-of-distribution data to refine VIRT, establishing a positive feedback loop.
- **Elegant Selective Generation Strategy**: The design (matching items allocated for estimation, and those with high deviation used for evolution) elegantly serves both testing and training objectives.

## Limitations & Future Work

1. **Sole Reliance on the 2PL-IRT Model**: Psychometrics offers a rich suite of models (e.g., Graded Response Models, Partial Credit Models), and a single model choice may exhibit biases.
2. **Limited Number of Examinee LLMs (8)**: Though ablation studies show robust effectiveness under small sample sizes, large-scale validation is still lacking.
3. **Limited Value Dimensions (Bias, Ethics, Toxicity)**: The work does not yet cover broader safety dimensions such as privacy, fairness, and misinformation.
4. **Potential Dual-use Risk**: The item generator could potentially be weaponized to discover LLM vulnerabilities at scale.
5. **Computational Cost**: Each candidate LLM is evaluated on approximately 1,000 items (10 rounds $\times$ 100 items). Compounded with generator fine-tuning, the total computational overhead is substantial.
6. **Multimodal Generalization**: Currently restricted to text-based LLMs; its applicability to multimodal models remains unverified.

## Related Work & Insights

- **Transferring Psychometrics to AI** is a highly promising direction: IRT-based ability modeling, CAT-based adaptive selection, and AIG-based automatic generation are all grounded in rigorous theoretical foundations.
- **Dynamic Evaluation** paradigm is applicable beyond value alignment, showing potential for generalization to general capability evaluations (such as reasoning, code generation, etc.).
- Comparison with red-teaming (e.g., GPTFuzzer, SAP) underscores a critical distinction: **Attacking $\neq$ Evaluating**. While the former focuses solely on "successful breaches", the latter requires precise quantification of "at what difficulty level the system fails".

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The first joint framework combining CAT+IRT+AIG+LM.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Evaluated over 8 models, 3 categories of values, and multi-dimensional ablations, though the evaluatee pool remains small.
- Writing Quality: ⭐⭐⭐⭐ — Clear architectures, rigorous formulations, with coherent logical pathways despite dense mathematical elements.
- Value: ⭐⭐⭐⭐⭐ — Resolves a genuine and critical challenge in LLM safety evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] OR-Bench: An Over-Refusal Benchmark for Large Language Models](or-bench_an_over-refusal_benchmark_for_large_language_models.md)
- [\[ACL 2025\] Explicit vs. Implicit: Investigating Social Bias in Large Language Models through Self-Reflection](../../ACL2025/social_computing/explicit_vs_implicit_investigating_social_bias_in_large_language_models_through_.md)
- [\[NeurIPS 2025\] Active Slice Discovery in Large Language Models](../../NeurIPS2025/social_computing/active_slice_discovery_in_large_language_models.md)
- [\[NeurIPS 2025\] Uncovering Strategic Egoism Behaviors in Large Language Models](../../NeurIPS2025/social_computing/uncovering_strategic_egoism_behaviors_in_large_language_models.md)
- [\[ACL 2025\] Exploring Gender Bias in Large Language Models: An In-depth Dive into the German Language](../../ACL2025/social_computing/exploring_gender_bias_in_large_language_models_an_in-depth_dive_into_the_german_.md)

</div>

<!-- RELATED:END -->
