---
title: >-
  [Paper Note] Uncovering Bias Paths with LLM-guided Causal Discovery: An Active Learning and Dynamic Scoring Approach
description: >-
  [AAAI2026][LLM Safety][Causal Discovery] This paper proposes a hybrid causal discovery framework that integrates LLM semantic priors with statistical signals. Through an active learning strategy and a dynamic scoring mechanism, the framework prioritizes querying the most informative variable pairs, effectively recovering fairness-critical causal paths (e.g., sex→education→income) under noise and confounding conditions, substantially outperforming classical CD methods and naïve LLM-based approaches.
tags:
  - AAAI2026
  - LLM Safety
  - Causal Discovery
  - LLM-guided
  - Fairness Auditing
  - Active Learning
  - Dynamic Scoring
  - Bias Paths
date: 2026-05-08
content_hash: 8f226f7c144c6aef
---

# Uncovering Bias Paths with LLM-guided Causal Discovery: An Active Learning and Dynamic Scoring Approach

**Conference**: AAAI2026
**arXiv**: [2506.12227](https://arxiv.org/abs/2506.12227)
**Authors**: Khadija Zanna, Akane Sano (Rice University)
**Code**: Not released
**Area**: AI Safety
**Keywords**: Causal Discovery, LLM-guided, Fairness Auditing, Active Learning, Dynamic Scoring, Bias Paths

## TL;DR

This paper proposes a hybrid causal discovery framework that integrates LLM semantic priors with statistical signals. Through an active learning strategy and a dynamic scoring mechanism, the framework prioritizes querying the most informative variable pairs, effectively recovering fairness-critical causal paths (e.g., sex→education→income) under noise and confounding conditions, substantially outperforming classical CD methods and naïve LLM-based approaches.

## Background & Motivation

### A Causal Perspective on Fairness Auditing

Bias in machine learning models deployed in high-stakes domains such as hiring, lending, education, and healthcare has attracted growing concern. Such bias often arises not through the direct effect of sensitive attributes (e.g., gender, race) on outcomes, but through indirect structural paths—where sensitive attributes influence final decisions via proxy variables or confounded relationships. Conventional fairness auditing relies on statistical disparity metrics (e.g., demographic parity), which cannot reveal the causal propagation mechanisms underlying bias, potentially rendering interventions ineffective or misleading. Causal Discovery (CD) offers tools to identify such paths, enabling the distinction between genuine causal effects and spurious associations introduced by confounding or measurement artifacts.

### Limitations of Prior Work

Classical CD methods (e.g., PC algorithm, GES) rely on conditional independence tests and strong assumptions (e.g., faithfulness), and tend to fail under noisy data, latent confounding, or incomplete metadata. Optimization-based methods (NOTEARS, DAGMA) are computationally expensive and highly sensitive to hyperparameters. Recent work has explored LLMs as auxiliary tools for CD, leveraging semantic knowledge to infer causal directions from variable metadata. However, two critical issues persist: (1) naïve use of LLMs may over- or under-attribute causality, particularly when sensitive attributes are involved; (2) BFS-based LLM query strategies (Jiralerspong et al. 2024) treat all variable pairs uniformly, wasting query budgets on non-informative relationships, while early errors propagate through the discovery process.

### Core Motivation

In fairness-sensitive settings, real-world datasets lack known ground-truth causal structures, making systematic evaluation of CD methods extremely challenging. This paper is motivated by two goals: (1) to design an adaptive query strategy that efficiently recovers fairness-relevant causal paths by balancing semantic and statistical signals; and (2) to construct a semi-synthetic benchmark with known bias paths to support rigorous evaluation.

## Core Problem

How can causal paths from sensitive attributes (e.g., sex, race) to outcomes (e.g., income) be recovered efficiently and accurately under noise, confounding, and data corruption? Classical CD methods exhibit insufficient precision in this setting, while naïve LLM approaches tend to over-predict. The core challenge is to design a hybrid framework that achieves both accuracy and query efficiency.

## Method

### Overall Architecture

The proposed framework augments the BFS-based CD pipeline with two key components: active learning and dynamic scoring. The overall procedure is: (1) identify independent root variables; (2) compute dynamic scores for each unqueried variable pair; (3) prioritize querying the highest-scoring pair via the LLM; (4) if the LLM returns "Yes" and adding the edge does not create a cycle, insert the directed edge into the causal graph; (5) repeat until the maximum number of iterations is reached or all scores fall below a threshold.

### Key Design 1: Dynamic Scoring Mechanism

Each unqueried variable pair $(x, y)$ receives a composite score integrating statistical signals, model confidence, and query history:

$$S(x,y) = w_{\text{stat}} \cdot \text{StatScore}(x,y) + w_{\text{conf}} \cdot \text{LLMConf}(x,y) + w_{\text{hist}} \cdot \text{HistScore}(x,y)$$

The individual components are defined as follows:

**Statistical Score**: Combines mutual information (MI) and partial correlation (PCorr) to capture both linear and nonlinear dependencies:

$$\text{StatScore}(x,y) = \frac{\text{MI}(x,y) + \text{PCorr}(x,y)}{2}$$

PCorr is computed as a conditional partial correlation conditioned on the currently discovered parent set, following the iterative conditioning procedure of the PC algorithm.

**LLM Confidence Score**: A sigmoid transformation of token-level log-probabilities, reflecting the LLM's certainty regarding the causal relationship of the variable pair:

$$\text{LLMConf}(x,y) = \frac{1}{1 + e^{-\text{confidence}}}$$

**Query History Score**: Penalizes repeated queries and encourages broader exploration:

$$\text{HistScore}(x,y) = \frac{1.5}{1 + \text{query\_count}(x,y)}$$

The weights $w_{\text{stat}}, w_{\text{conf}}, w_{\text{hist}}$ are tuned via Bayesian optimization (Gaussian Process surrogate + gp_minimize).

### Key Design 2: Active Learning Query Strategy

At each step, the highest-scoring variable pair is selected for querying:

$$(x^*, y^*) = \arg\max_{(x,y) \in \text{Unqueried}} S(x,y)$$

The LLM operates in a multi-turn dialogue mode, maintaining context awareness over prior reasoning. If the response is "Yes" and no cycle is introduced, a directed edge is added, constructing the adjacency matrix:

$$A(i,j) = \begin{cases} 1 & \text{if } X_i \rightarrow X_j \text{ is predicted} \\ 0 & \text{otherwise} \end{cases}$$

### Key Design 3: Fairness Path Analysis

After learning the causal graph, all directed paths from each sensitive attribute $S$ to the outcome $Y$ are enumerated and classified as:

- **Direct paths**: $S \rightarrow Y$
- **Indirect paths**: $S \rightarrow \cdots \rightarrow Y$ (mediated through intermediate variables)
- **Spurious paths**: involving $S$ but not reaching $Y$

Causal effect decomposition quantifies bias as $TE = DE + IE$, and a normalized bias coefficient is computed:

$$C_{\text{bias}} = \frac{TE}{\text{Var}(Y)}$$

### Key Design 4: Semi-Synthetic Benchmark Construction

A semi-synthetic causal graph with 15 variables is constructed based on the UCI Adult dataset. Direct edges and indirect paths from race and sex to income (e.g., sex→education→income) are injected, along with noise, data corruption, and a latent confounding variable $U \sim \mathcal{N}(0,1)$, providing a controlled ground-truth evaluation environment.

## Key Experimental Results

### Main Results: Structural Recovery Performance Across Three Datasets

| Method | Dataset | Accuracy ↑ | F1 ↑ | Precision | Recall | NHD ↓ | Predicted Edges |
|--------|---------|-----------|------|-----------|--------|-------|----------------|
| PC | Adult (15n, 28e) | 0.239 | 0.382 | 0.352 | 0.420 | 0.193 | 33 |
| GES | Adult | 0.296 | 0.473 | 0.368 | 0.580 | 0.203 | 44 |
| NOTEARS | Adult | 0.021 | 0.039 | 0.035 | 0.045 | 0.260 | 27 |
| DAGMA | Adult | 0.099 | 0.180 | 0.141 | 0.250 | 0.283 | 50 |
| LLM Pairwise | Adult | 0.307 | 0.470 | 0.331 | 0.813 | 0.253 | 69 |
| LLM BFS | Adult | 0.299 | 0.456 | 0.332 | 0.750 | 0.305 | 64 |
| **Proposed** | **Adult** | **0.413** | **0.585** | **0.792** | 0.464 | **0.109** | 17 |
| PC | Child (20n, 25e) | 0.146 | 0.255 | 0.273 | 0.239 | 0.097 | 22 |
| NOTEARS | Child | 0.216 | 0.356 | 0.403 | 0.319 | 0.080 | 20 |
| **Proposed** | **Child** | **0.364** | **0.533** | **0.601** | 0.479 | 0.082 | 20 |
| PC | Neuropathic (221n) | 0.041 | 0.078 | 0.092 | 0.068 | 0.025 | 563 |
| LLM BFS | Neuropathic | 0.000 | 0.000 | 0.000 | 0.000 | 0.903 | 43 |
| **Proposed** | **Neuropathic** | **0.073** | **0.136** | **0.690** | 0.075 | 0.109 | 84 |

### Fairness Path Recovery

Ground-truth: 2 direct paths (sex→income, race→income), 25 indirect paths, TE=4.89, $C_{\text{bias}}=28.46$.

| Method | Direct Paths | Indirect Paths | Spurious Paths | $C_{\text{bias}}$ |
|--------|-------------|---------------|---------------|-----------|
| PC | 0 | Few | Many | Low |
| GES | 0 | Moderate | Many | Low |
| LLM Pairwise | 3 (includes spurious age path) | Excessive | Many | Inflated |
| **Proposed** | **2 (sex and race only)** | Partially recovered | **Very few** | Near ground-truth |

The proposed method is the only approach that correctly recovers both true direct paths (sex→income, race→income) while excluding the spurious age→income path.

### Hyperparameter Sensitivity Analysis

A Random Forest regression analysis of Bayesian optimization trials reveals that the maximum number of iterations has the largest impact on F1, followed by the scoring threshold and LLM temperature. Anti-correlations among scoring weights indicate competitive trade-offs—semantic guidance dominates in small graphs, while MI/PCorr statistical signals take precedence in larger graphs (e.g., Neuropathic).

## Highlights & Insights

- **Adaptive Query Strategy**: The dynamic scoring mechanism adaptively balances semantic versus statistical signals according to graph structure and data quality—relying on LLM semantic judgments during early exploration and shifting toward statistical signal verification in later stages, reflecting a progressive transition from semantic exploration to empirical refinement.
- **Precise Fairness Path Recovery**: The proposed method is the only approach to correctly identify the sex→income and race→income direct paths while avoiding the spurious age→income path, demonstrating strong fairness diagnostic capability.
- **High Precision with Low False Positives**: On the Adult dataset, Precision reaches 0.792 (far exceeding all baselines), with only 17 edges predicted (ground-truth: 28), reflecting a conservative yet reliable strategy.
- **Cross-Scale Generalization**: The method maintains optimal F1 across graphs ranging from 15 to 221 nodes, remaining effective on the large-scale Neuropathic network where LLM BFS fails entirely.
- **Semi-Synthetic Benchmark**: A reproducible benchmark with controlled ground-truth is provided for fairness-sensitive CD evaluation.

## Limitations & Future Work

- **Low Recall**: The conservative strategy yields a Recall of only 0.464 on the Adult dataset, potentially missing certain fairness-relevant indirect paths.
- **LLM Dependency and Reproducibility**: GPT-4-based inference introduces stochasticity and version sensitivity; inconsistencies with results reported by Jiralerspong et al. highlight the absence of standardized evaluation protocols for LLM-based CD.
- **Semi-Synthetic Data Limitations**: The causal graph treats income as a terminal node, ignoring downstream feedback effects and temporal dynamics.
- **Computational Cost**: Bayesian optimization combined with LLM queries incurs high overhead on large graphs; the Neuropathic dataset is significantly affected by token-limit constraints.
- **Risk of LLM Social Bias**: Social biases encoded in LLMs may lead to inferences that reflect stereotypes rather than true causal mechanisms.
- **Observational CD Only**: Interventional or experimental causal discovery methods (e.g., do-calculus, instrumental variables) are not considered.

## Related Work & Insights

- **PC / GES**: Classical constraint- and score-based methods achieve F1 of only 0.38–0.47 under noise and fail to recover direct fairness paths.
- **NOTEARS / DAGMA**: Continuous optimization methods perform worst on fairness-relevant structures (F1 < 0.18) and are highly sensitive to hyperparameters.
- **LLM Pairwise (Kıcıman et al. 2023)**: Achieves high Recall (0.813) but severely over-predicts (69 edges vs. 28 ground-truth), introducing spurious fairness paths that inflate $C_{\text{bias}}$.
- **LLM BFS (Jiralerspong et al. 2024)**: Fixed-order querying is sensitive to early errors and fails completely on large-scale graphs (Neuropathic F1=0).
- **Takayama et al. 2024**: LLM-augmented statistical method, but with a single prior injection step, lacking the dynamic adaptivity of the proposed approach.
- The core contribution lies in the dynamic scoring + active learning query prioritization mechanism, which substantially reduces NHD while maintaining high precision.

**Broader Implications**:

- **Path-Level Interpretability over Aggregate Statistics**: Summary metrics such as $C_{\text{bias}}$ can be inflated by spurious paths; path-level analysis provides more reliable fairness diagnoses, with important implications for real-world auditing.
- **Complementarity of Semantic Priors and Statistical Signals**: LLM semantic guidance is especially valuable in the early stages of graph discovery when statistical signals are weak; as evidence accumulates, statistical signals progressively take over. This adaptive balancing strategy generalizes to other LLM-assisted scientific discovery tasks.
- **Fairness Auditing as a Tool**: The framework can serve as an algorithmic auditing tool for high-risk domains such as hiring and lending, enabling non-technical stakeholders to trace bias propagation paths.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Introducing active learning and dynamic scoring into LLM-guided CD is a meaningful contribution, though the overall framework represents an incremental improvement over BFS-based methods.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three datasets, multiple baselines, hyperparameter sensitivity analysis, and fairness path evaluation provide comprehensive coverage, though the low-Recall issue is not deeply analyzed.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with clear motivation and rigorous formalism; some discussions are somewhat verbose.
- **Value**: ⭐⭐⭐⭐ — The intersection of fairness, causal discovery, and LLMs addresses a practically significant problem, and the semi-synthetic benchmark contributes to the community; however, LLM reproducibility concerns limit broader adoption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] BiasBusters: Uncovering and Mitigating Tool Selection Bias in Large Language Models](../../ICLR2026/llm_safety/biasbusters_uncovering_and_mitigating_tool_selection_bias_in_large_language_mode.md)
- [\[AAAI 2026\] CATFormer: When Continual Learning Meets Spiking Transformers With Dynamic Thresholds](catformer_when_continual_learning_meets_spiking_transformers_with_dynamic_thresh.md)
- [\[AAAI 2026\] Principles2Plan: LLM-Guided System for Operationalising Ethical Principles into Plans](principles2plan_llm-guided_system_for_operationalising_ethical_principles_into_p.md)
- [\[AAAI 2026\] PSM: Prompt Sensitivity Minimization via LLM-Guided Black-Box Optimization](psm_prompt_sensitivity_minimization_via_llm-guided_black-box_optimization.md)
- [\[ACL 2026\] Who Gets Which Message? Auditing Demographic Bias in LLM-Generated Targeted Text](../../ACL2026/llm_safety/who_gets_which_message_auditing_demographic_bias_in_llm-generated_targeted_text.md)

</div>

<!-- RELATED:END -->
