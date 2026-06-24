---
title: >-
  [Paper Note] IRIS: An Iterative and Integrated Framework for Verifiable Causal Discovery
description: >-
  [ACL 2025][Causal Inference][causal discovery] The IRIS framework is proposed. Requiring only a set of initial variable names as input, it automatically retrieves documents, extracts variable values to construct structured data, and builds causal graphs via hybrid causal discovery (GES statistical algorithm + LLM causal verification). Additionally, it iteratively expands the variable set using a missing variable proposal component, relaxing the acyclicity and causal sufficien…
tags:
  - "ACL 2025"
  - "Causal Inference"
  - "causal discovery"
  - "LLM"
  - "hybrid method"
  - "missing variable proposal"
  - "iterative framework"
  - "value extraction"
date: 2026-05-08
content_hash: 5fe49cb447c60862
---

# IRIS: An Iterative and Integrated Framework for Verifiable Causal Discovery

**Conference**: ACL 2025  
**arXiv**: [2510.09217](https://arxiv.org/abs/2510.09217)  
**Code**: [https://github.com/WilliamsToTo/iris](https://github.com/WilliamsToTo/iris)  
**Institution**: Monash University, Microsoft Research India  
**Area**: Causal Inference  
**Keywords**: causal discovery, LLM, hybrid method, missing variable proposal, iterative framework, value extraction

## TL;DR

The IRIS framework is proposed. Requiring only a set of initial variable names as input, it automatically retrieves documents, extracts variable values to construct structured data, and builds causal graphs via hybrid causal discovery (GES statistical algorithm + LLM causal verification). Additionally, it iteratively expands the variable set using a missing variable proposal component, relaxing the acyclicity and causal sufficiency assumptions of traditional methods. IRIS comprehensively outperforms 0-shot, CoT, and RAG baselines in F1 score across 6 datasets: Cancer, Diabetes, Obesity, ADNI, and Insurance.

## Background & Motivation

**Statistical causal discovery relies on high-quality tabular data**: Traditional algorithms such as PC, GES, and NOTEARS require pre-collected structured observational data, which is expensive and time-consuming to obtain. In fields like biology, economics, and healthcare, acquiring high-quality causal discovery data requires significant human and material resources, greatly limiting the practical application of causal discovery in fields like NLP.

**LLMs only parrot known relationships from training data**: LLM-based methods (e.g., Pairwise-LLM, BFS-LLM) excel at identifying highly frequent causal relationships in their training data (e.g., "smoking → lung cancer"). However, empirical studies by Zečević et al. (2023) and Feng et al. (2024b) show that LLMs perform poorly when discovering rare or undocumented new causal relationships, essentially acting as "causal parrots" that rely on memory rather than reasoning.

**Assumptions of statistical methods are overly idealized**: Most algorithms assume causal sufficiency (no confounders/unobserved variables) and acyclicity (no feedback loops in the causal graph). However, feedback loops are ubiquitous in the real world, such as poverty cycles (poverty → limited education → low-paying jobs → poverty) and predator-prey cycles (predator increase → prey decrease → predator decrease). These assumptions are severely disconnected from real-world scenarios.

**Variable sets must be predefined and are non-extensible**: Statistical algorithms require a predefined set of random variables as input and cannot automatically identify crucial variables that might be missing during the discovery process. For instance, in cancer research, if "air pollution" is not included in the initial variables, traditional methods will never discover its causal effects.

**LLM causal predictions lack verification mechanisms**: Statistical methods rely on strict mathematical verifiability (such as conditional independence tests), whereas LLM causal judgments lack transparent verification, making it difficult to quantify the reliability of their outputs.

**The complementary potential of both methods has not been fully exploited**: Statistical methods can discover novel, unknown relationships from data but require structured data; LLMs can leverage pre-trained knowledge to identify known relationships without needing data. IRIS aims to combine the strengths of both—statistical methods provide discovery capability, while LLMs provide knowledge verification.

## Method

### Overall Architecture

IRIS (Iterative Retrieval and Integrated System for Real-Time Causal Discovery) consists of four core components connected in a pipeline: **Document Retrieval and Variable Value Extraction** → **Hybrid Causal Discovery** (Statistical Branch + LLM Verification Branch → Graph Merger) → **Missing Variable Proposal** → **Iterative Expansion**. The input to the entire pipeline is only an initial set of variables $\mathbb{Z}=\{z_1, z_2, \ldots, z_N\}$ (e.g., {smoking, cancer, pollution, diet}), and the final output is an expanded causal graph $\mathcal{G}=(\mathbb{Z}_m, \mathbb{R})$, where $\mathbb{Z}_m$ contains the initial variables along with newly discovered variables, and $\mathbb{R}$ represents all causal edges.

### Component 1: Document Retrieval and Variable Value Extraction

- **Goal**: Automatically collect relevant documents starting from variable names and extract variable values to construct structured tabular data $\mathbb{X}$ (rows = documents, columns = variables).
- **Document Retrieval**: A query-construction strategy using stepwise removal is employed via the Google Search API. It begins with an AND combination of all variable names (e.g., "smoking" AND "cancer" AND "pollution"), gradually reducing the number of variables to single-variable queries. The retrieval quota $k$ for multi-variable queries is set higher to ensure documents are relevant to most variables. Meanwhile, synonyms of variables are used to enhance coverage, and retrieval continues until the total number of documents reaches a predefined threshold.
- **Variable Value Extraction**: For each document $d_i$ and each variable $z_j$, a prompt $l$ containing variable descriptions (names + meaning of values) is designed to guide the LLM to extract values using multi-step chain-of-thought reasoning: $o_{ij} = \bm{M}(l(d_i, z_j))$. In the constructed table $\mathbb{X}$, each element $v_{ij}$ represents the extracted value of variable $z_j$ in document $d_i$.
- **Design Motivation**: Multi-variable combined queries prioritize documents relevant to all variables, while stepwise degradation guarantees sufficient coverage for each variable. Replacing manual annotation with LLM extraction significantly reduces data collection costs.

### Component 2: Hybrid Causal Discovery

- **Statistical Branch**: Runs statistical causal discovery algorithms (PC / GES / NOTEARS) on the structured data $\mathbb{X}$ to discover causal relationships via conditional independence tests and other statistical methods, outputting a causal graph $\hat{\mathcal{G}_s}$. In experiments, GES yields the best average performance.
- **LLM Verification Branch**: Formulates potential causal relationships for each pair of variables as claims (e.g., "smoking causes lung cancer") and retrieves evidence documents containing terms from both variables across 7 academic domains (jstor.org, springer.com, ieee.org, ncbi.nlm.nih.gov, sciencedirect.com, scholar.google.com, arxiv.org). The LLM determines whether each document supports, refutes, or is not related to the claim. Relationships supported by the majority of documents are integrated into the causal graph $\hat{\mathcal{G}_v}$.
- **Graph Merger**: Builds on the statistical graph $\hat{\mathcal{G}_s}$ by adding high-confidence relationships from $\hat{\mathcal{G}_v}$ and removing edges strongly refuted by academic literature. The merging strategy is guided by two rationale: (1) structured data $\mathbb{X}$ might be noisy; (2) relationships widely supported or refuted by credible academic literature can be treated as established knowledge. This hybrid strategy naturally allows feedback loops in the causal graph.

### Component 3: Missing Variable Proposal (MVP)

- **Variable Abstraction**: The LLM analyzes the content of each retrieved document to identify new candidate variables that might influence or be influenced by the initial variables.
- **Double-Filtering Mechanism**:
    - *Causal Relation Verification (VCR)*: For each candidate new variable, evidence is retrieved from academic websites using the method in Section 3.3 to verify its causal relationship with the initial variables.
    - *Statistical Measure (PMI)*: Obtains document co-occurrence counts via the Google Search API to calculate pointwise mutual information $PMI(z_i, z_j) = \log \frac{o(z_i, z_j)}{o(z_i) \cdot o(z_j)}$. The top-$k$ variables with the highest aggregated PMI scores are incorporated into $\mathbb{Z}_m$.
- **Iterative Expansion**: Integrates new variables from $\mathbb{Z}_m$ into the original variable set and re-executes the data collection → value extraction → causal discovery pipeline, allowing the causal graph to grow continuously.

## Key Experimental Results

### Table 1: Evaluation of the Complete Framework across 6 Datasets (Precision / Recall / F1↑ / NHD Ratio↓)

| Method | Cancer | Resp. Disease | Diabetes | Obesity | ADNI | Insurance |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 0-shot | 0.64/0.32/0.43/0.57 | 0.67/0.36/0.47/0.53 | 0.70/0.46/0.56/0.45 | 0.57/0.33/0.42/0.58 | 0.47/0.29/0.36/0.64 | 0.35/0.38/0.36/0.65|
| CoT | 0.67/0.38/0.48/0.54 | 0.64/0.40/0.49/0.51 | 0.66/0.48/0.55/0.46 | 0.59/0.38/0.46/0.54 | 0.46/0.31/0.37/0.62 | 0.41/0.38/0.39/0.61|
| RAG | 0.70/0.44/0.54/0.49 | 0.64/0.45/0.53/0.47 | 0.73/0.47/0.57/0.43 | 0.62/0.45/0.52/0.49 | 0.50/0.34/0.40/0.60 | 0.44/0.40/0.42/0.57|
| **IRIS** | **0.89/0.57/0.70/0.30** | **0.67/0.55/0.60/0.40** | **0.76/0.50/0.60/0.39** | **0.67/0.58/0.62/0.38** | **0.50/0.36/0.42/0.58** | **0.61/0.46/0.53/0.47**|

> Paired t-tests confirm that the differences between IRIS and the baselines are statistically significant in both F1 and NHD Ratio ($p \leq 0.05$). The Insurance dataset was successfully expanded from 27 initial variables to 35 variables and 67 edges, demonstrating the scalability of IRIS. The core advantage of IRIS lies in recall—while all baselines can sometimes approach IRIS in precision, none can match its recall, demonstrating that the hybrid strategy indeed discovers more true causal relationships.

### Table 2: Evaluation of the Variable Value Extraction Component (Precision / Recall / F1)

| Method | AppleGastronome | Neuropathic |
|------|:---:|:---:|
| COAT (GPT-4o) | 0.74 / 0.76 / 0.75 | 0.72 / 0.80 / 0.79 |
| IRIS (Llama-3.1-8b) | 0.71 / 0.72 / 0.71 | 0.76 / 0.82 / 0.79 |
| IRIS (GPT-3.5) | 0.75 / 0.77 / 0.76 | 0.71 / 0.89 / 0.79 |
| **IRIS (GPT-4o)** | **0.79 / 0.82 / 0.79** | **0.73 / 1.00 / 0.84** |

> Under the same LLM (GPT-4o), the value extraction method of IRIS outperforms COAT, and performance scales stably with stronger LLMs (Llama → GPT-3.5 → GPT-4o). GPT-4o achieves 100% recall on Neuropathic. Binary variables (Neuropathic, 0/1) are easier to extract than ternary variables (AppleGastronome, -1/0/1).

### Table 3: Success Rate of Missing Variable Proposal (MVP)

| Method | Cancer | Resp. | Diabetes | Obesity | ADNI | Insurance |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 0-shot | 0.40 | 0.25 | 0.50 | 0.25 | 0.25 | 0.22 |
| CoT | 0.40 | 0.50 | 0.50 | 0.75 | 0.25 | 0.30 |
| RAG | 0.60 | 0.75 | 0.75 | 0.75 | 0.38 | 0.41 |
| **MVP (Full)** | **0.80** | **0.75** | **1.00** | **1.00** | **0.50** | **0.59** |
| − VCR | 0.60 | 0.75 | 0.50 | 0.75 | 0.25 | 0.48 |
| − Stats | 0.60 | 0.75 | 0.75 | 1.00 | 0.38 | 0.52 |
| ↔ Llama | 0.40 | 0.50 | 0.25 | 0.50 | 0.13 | 0.45 |

> Evaluation is performed by simulating missing variables: each variable is sequentially removed from the complete causal graph, and we check whether MVP can recover the removed variable in the proposed set. Diabetes and Obesity achieve a 100% success rate.

## Key Findings

- **Hybrid methods comprehensively outperform single methods**: By combining statistical methods to discover new relationships and LLMs to verify known ones, IRIS improves the F1 score on Cancer from 0.54 (RAG) to 0.70 and reduces the NHD Ratio from 0.49 to 0.30.
- **The dominance of IRIS lies primarily in recall**: Although baselines (especially RAG) occasionally approach IRIS in precision, their recall lags far behind. This confirms that the hybrid strategy discovers a larger number of true causal edges.
- **There is no "silver bullet" statistical algorithm**: GES performs best on average (improving F1 by 0.09 over PC), but NOTEARS completely fails on Diabetes/Obesity (F1=0, NHD=1), indicating that the choice of statistical algorithm must adapt to data characteristics.
- **Both filtering steps in MVP are indispensable**: Removing VCR drops the success rate on Diabetes from 1.00 to 0.50, and removing Stats drops ADNI from 0.50 to 0.38, verifying the complementarity of the two signals.
- **GPT-4o far outperforms Llama on expert-knowledge-intensive tasks**: The performance gap on the ADNI dataset is particularly pronounced (MVP success rate of 0.50 vs. 0.13), as Alzheimer's-related knowledge is extremely scarce in smaller models' pre-training datasets.
- **High scalability**: The framework remains highly effective across scales, ranging from 4 variables (Cancer) to 27 variables (Insurance), the latter of which was successfully expanded to 35 variables and 67 edges.

## Highlights & Insights

- **A paradigm shift to "variable names only"**: Traditional causal discovery requires pre-collected high-quality tabular data; IRIS simplifies the input to a list of variable names. The entire pipeline (retrieval → extraction → discovery → expansion) is fully automated, lowering the barrier to entry from "data scientists + domain experts" to "any researcher who can define a set of variables."
- **Restricting the search to academic domains ensures credible literature verification**: Searching specifically within seven authoritative academic platforms (jstor, springer, ieee, ncbi, sciencedirect, scholar.google, arxiv) provides a simple yet highly effective mechanism for quality control.
- **The iterative expansion mechanism enables causal graphs to grow incrementally from core variables**: This is highly suited for exploratory research scenarios—researchers can start with a few known variables and let the system automatically uncover "unknown unknowns."
- **The hybrid strategy is naturally compatible with feedback loops**: The output causal graphs are no longer constrained by the acyclicity assumptions of traditional methods, making them capable of representing real-world feedback mechanisms like poverty cycles or predator-prey dynamics.
- **Independent mathematical formulation and experimental validation for each component**: IRIS avoids being an end-to-end black box; value extraction, causal discovery, and variable proposal each undergo separate baseline comparisons and ablation studies.

## Limitations & Future Work

- **High computational overhead**: The number of LLM queries scales as $O(N^2)$ with the number of variables $N$ (since all variable pairs must be checked), leading to an average runtime of approximately 15 hours (around 3x that of zero-shot). Although the paper notes that all LLM queries can be parallelized, the API costs and latency remain non-negligible in practice.
- **Reliance on commercial services**: The Google Search API (for document retrieval and PMI counting) and GPT-4o (for value extraction, relationship verification, and variable abstraction) are irreplaceable core components, meaning reproducibility is constrained by API stability and usage fees.
- **LLM-extracted variable values contain noise**: Particularly, ternary classification (-1/0/1) is more error-prone than binary classification (AppleGastronome F1 of 0.79 vs. Neuropathic F1 of 0.84), and this noise propagates to the statistical causal discovery branch.
- **Coverage bias in retrieved documents**: Google Search results tend to favor highly frequent, mainstream topics, potentially suffering from poor coverage of niche or rare causal relationships.
- **Performance bottlenecks in highly complex domains**: The overall F1 score on ADNI (Alzheimer's Disease) is only 0.42, showing that current methods still struggle to efficiently discover and verify causal relationships in expert-knowledge-intensive fields.
- **Energy consumption and environmental impact of LLM inference**: The paper acknowledges this concern but does not offer a specific mitigation strategy.

## Related Work & Insights

- **Three major schools of statistical causal discovery**: Constraint-based (e.g., PC, which eliminates non-causal edges via conditional independence tests), score-based (e.g., GES, which greedily searches for the optimal DAG structure), and functional-based (e.g., NOTEARS, which reformulates DAG learning as a continuous optimization problem). These methods are mathematically verifiable but limited by data quality, causal sufficiency, and acyclicity assumptions.
- **LLM-based causal discovery**: Pairwise-LLM (Feng et al., 2024b) pairwise evaluates the causal direction between variables; BFS-LLM (Jiralerspong et al., 2024) structures the reasoning sequence using breadth-first search. These perform reasonably on simple datasets but degrade drastically on complex ones.
- **Hybrid LLM + statistical methods**: COAT (Liu et al., 2024) uses LLMs for extracting variable values followed by the PC algorithm for discovering relationships, serving as a direct precursor to this work. However, COAT requires manual document collection, does not expand the variable set, and is restricted to the PC algorithm. IRIS augments this with automatic retrieval, hybrid discovery (dual statistical + validation branches), and iterative expansion.
- **Causal relation verification**: The claim verification paradigm from Si et al. (2024) and Wadden et al. (2022) is directly adopted by IRIS—framing causal relationships as claims and looking up supporting/refuting evidence in academic literature.
- **Unobserved variable discovery**: Traditional approaches, such as Tetrad-based methods (Silva et al., 2006) and higher-order moment methods (Chen et al., 2022), focus solely on specific types of unobserved variables (e.g., latent confounders). In contrast, the MVP component of IRIS addresses more generalized types of missing variables.

## Rating

- ⭐⭐⭐⭐⭐ **Novelty**: First end-to-end framework to achieve fully automated causal discovery starting solely from initial variable names, presenting a significant paradigm shift from "data-dependent" discovery.
- ⭐⭐⭐⭐ **Experimental Thoroughness**: Comprehensive evaluation across 6 datasets, independent evaluation of 3 components, ablation studies, comparison of multiple LLMs (GPT-4o / GPT-3.5 / Llama), and paired t-tests.
- ⭐⭐⭐⭐ **Writing Quality**: The problem definitions are highly clear, the system workflow (Figure 1) is intuitive, and every component is detailed with independent algorithmic pseudocode and mathematical formulation.
- ⭐⭐⭐⭐ **Value**: Drastically lowers the barrier to causal discovery from "requiring complete structured datasets" to "requiring only variable names", making it highly accessible for researchers in biomedical and social sciences.
- ⭐⭐⭐ **Reproducibility**: Heavy reliance on the Google Search API and GPT-4o. The combination of closed-source models and paid APIs introduces challenges for perfect reproducibility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] On the Reliability of Large Language Models for Causal Discovery](llm_causal_discovery_reliability.md)
- [\[CVPR 2026\] A Polynomial Chaos Framework for Causal Discovery in Nonlinear Uncertain Systems](../../CVPR2026/causal_inference/a_polynomial_chaos_framework_for_causal_discovery_in_nonlinear_uncertain_systems.md)
- [\[ICLR 2026\] Efficient Ensemble Conditional Independence Test Framework for Causal Discovery](../../ICLR2026/causal_inference/efficient_ensemble_conditional_independence_test_framework_for_causal_discovery.md)
- [\[ACL 2025\] FitCF: A Framework for Automatic Feature Importance-guided Counterfactual Example Generation](fitcf_a_framework_for_automatic_feature_importance-guided_counterfactual_example.md)
- [\[NeurIPS 2025\] Differentiable Structure Learning and Causal Discovery for General Binary Data](../../NeurIPS2025/causal_inference/differentiable_structure_learning_and_causal_discovery_for_general_binary_data.md)

</div>

<!-- RELATED:END -->
