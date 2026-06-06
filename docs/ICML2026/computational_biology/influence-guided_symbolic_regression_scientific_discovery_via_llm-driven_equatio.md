---
title: >-
  [Paper Note] Influence-Guided Symbolic Regression: Scientific Discovery via LLM-Driven Equation Search with Granular Feedback
description: >-
  [ICML 2026][Computational Biology][Symbolic Regression] IGSR decomposes symbolic regression into a two-step cycle: "LLM proposes basis functions $\psi_j$ + term-wise influence score $\Delta_j$ pruning." By embedding this…
tags:
  - "ICML 2026"
  - "Computational Biology"
  - "Symbolic Regression"
  - "Influence Score"
  - "LLM Equation Discovery"
  - "MCTS"
  - "Interpretable Modeling"
date: 2026-05-08
content_hash: 367dffed7649e76a
---

# Influence-Guided Symbolic Regression: Scientific Discovery via LLM-Driven Equation Search with Granular Feedback

**Conference**: ICML 2026  
**arXiv**: [2605.29184](https://arxiv.org/abs/2605.29184)  
**Code**: https://github.com/DrShushen/IGSR (Available)  
**Area**: Computational Biology  
**Keywords**: Symbolic Regression, Influence Score, LLM Equation Discovery, MCTS, Interpretable Modeling

## TL;DR
IGSR decomposes symbolic regression into a two-step cycle: "LLM proposes basis functions $\psi_j$ + term-wise influence score $\Delta_j$ pruning." By embedding this cycle into MCTS to search the combinatorial space, it achieves the best MSE and symbolic recall across 6 biomedical benchmarks and LLM-SRBench. Furthermore, it identified a novel relationship between DNA methylation and RNA Pol II pausing validated via wet-lab experiments.

## Background & Motivation

**Background**: Traditional symbolic regression (e.g., GP-SR, PySR, SINDy) performs evolution or sparse regression on preset operator libraries. While they produce closed-form formulas, they struggle with high-dimensional inputs where $d \gg 20$. Recent LLM-driven equation discovery methods (e.g., D3, ICSR, LLM-SR, LaSR) leverage scientific priors of LLMs to "propose" basis functions, pushing symbolic regression into complex scenarios like biology, epidemiology, and pharmacokinetics.

**Limitations of Prior Work**: Existing LLM-based equation discovery methods rely on **global scalar signals** (typically global MSE or code execution errors) as feedback. This informs the LLM whether a formula is "good or bad" but fails to indicate **which term contributes and which detracts**. Consequently, the search degrades into trial-and-error, heavily dependent on the LLM's generative priors rather than the data itself.

**Key Challenge**: Generation (creative proposal) and selection (rigorous pruning) are coupled within a single scalar loss. The LLM simultaneously handles "proposing new terms" and "judging whether old terms should remain." It excels at the former but fares poorly at the latter, often hallucinating and deleting statistically significant terms as irrelevant.

**Goal**: (1) Provide LLMs with **term-wise** fine-grained credit assignment signals; (2) **Decouple** generation and selection—letting the LLM handle creation while statistics handle selection; (3) Efficiently balance exploration and exploitation within the combinatorial search space.

**Key Insight**: The authors restrict the model class to a **linear model** of basis functions: $f(\mathbf{x}) = \sum_{j=1}^M w_j \psi_j(\mathbf{x})$. Consequently, the marginal contribution of each $\psi_j$ is naturally quantifiable. By defining $\Delta_j$ as the increase in validation MSE after removing a specific term, a direct, inexpensive, and principled signal is obtained.

**Core Idea**: Use **term-wise influence scores $\Delta_j$ instead of global MSE** for feedback, embedding the "propose-and-prune" cycle into MCTS to explore the basis function combinatorial space.

## Method

### Overall Architecture
IGSR aims to discover sparse closed-form models $f(\mathbf{x}) = \sum_j w_j \psi_j(\mathbf{x})$, where $\psi_j$ are arbitrarily complex non-linear basis functions proposed by an LLM, and $w_j$ are fitted via OLS. The core is a three-stage **propose-and-prune cycle**: ① "Propose": An LLM agent reads the context (variable descriptions, active terms, historical records of retained/discarded terms, and their MSE impacts) to generate candidate $\psi_j$; ② An expanded set of old and new terms is fitted via OLS to obtain $\mathbf{w}$, and term-wise influence scores $\Delta_j$ are calculated on the validation set; ③ Terms are ranked by $\Delta_j$ to retain the Top-$K$ items, with results logged in a history buffer. This cycle is wrapped in an MCTS tree where each node represents an equation state. Expanding a node involves running the propose-and-prune cycle, with node rewards defined as $-\mathrm{MSE}_{\mathrm{val}}$, using UCT to balance exploration and exploitation.

### Key Designs

1.  **Term-wise Influence Score $\Delta_j$**:
    - **Function**: Provides **structure-aware fine-grained credit assignment** for the search process, informing the LLM of the specific contribution of each term.
    - **Mechanism**: On a fitted linear model, the weight of the $j$-th term is set to zero ($w_j \to 0$) while keeping other coefficients fixed. $\Delta_j$ is defined as the increment in validation MSE: $\Delta_j = \mathrm{MSE}_{\mathrm{val}}(\mathbf{w}_{-j}) - \mathrm{MSE}_{\mathrm{val}}(\mathbf{w})$. This is an application of leave-one-term-out analysis in the **structural dimension** (as opposed to the data point dimension). Computation requires only one OLS solution and simple algebra, incurring near-zero overhead.
    - **Design Motivation**: Global MSE only indicates "good/bad" without localizing the source of contribution. $\Delta_j$ transforms model selection from "guessing" to "testing," remaining reliable even under collinearity or interaction (epistasis-like) signals.

2.  **Propose-and-Prune Cycle (Decoupling Generation and Selection)**:
    - **Function**: Separates the LLM's creative proposals from data-driven term selection, preventing the LLM from hallucinating while pruning.
    - **Mechanism**: It defaults to **deterministic pruning**, where the LLM only generates candidates and the Top-$K$ are kept based on $\Delta_j$. An optional **Agentic pruning** (IGSR-Agent) allows a second LLM agent to read the $(\psi_j, w_j, \Delta_j)$ triplets, using $\Delta_j \approx 0 \Rightarrow \text{drop}$ as the primary heuristic alongside semantic rationality judgments. The proposal prompt includes the history buffer, enabling in-context learning to avoid repeating previous failures.
    - **Design Motivation**: The deterministic version is hallucination-free, reproducible, and computationally cheap. The Agentic version trades some stability for the injection of domain knowledge. Experiments show that the **deterministic version is the most robust default configuration**.

3.  **MCTS Embedded Search (Avoiding Local Optima)**:
    - **Function**: Systematically balances exploration and exploitation in the combinatorially explosive equation space.
    - **Mechanism**: Each node is an equation state. Multiple successors are generated via random LLM sampling (e.g., one branch explores trigonometric terms, another explores interaction terms). Node rewards are $-\mathrm{MSE}_{\mathrm{val}}$, and the UCT formula $\bar r_i + c\sqrt{\ln N / n_i}$ guides expansion. A **heuristic MCTS** is used, where immediate rewards are backpropagated without full rollouts to focus the budget on breadth.
    - **Design Motivation**: Single-chain linear refinement easily gets stuck in specific functional form biases. MCTS allows parallel exploration of different hypotheses.

### Loss & Training
This is a search process rather than end-to-end training. OLS fitting uses the training set, while $\Delta_j$ and MCTS rewards are calculated on the validation set. LLM backends tested include GPT-4o and GPT-4o-mini. A uniform 300k token budget is used for LLM-SRBench. The sparsity limit $K$ is the primary hyperparameter.

## Key Experimental Results

### Main Results
Six biomedical benchmarks (Lung Cancer variants, COVID-19, RNA Polymerase, Warfarin), 25 seeds, GPT-4o:

| Dataset | IGSR MSE | Best White-box Baseline MSE | Notes |
| :--- | :--- | :--- | :--- |
| Lung Cancer | 5.64e-5 | ICL 0.0557 (3 orders of magnitude difference) | Clean Tumor Growth ODE |
| LC + Chemo | 0.0013 | ICSR 0.688 | Coupled ODE with Chemotherapy |
| LC + Chemo+Radio | 0.0141 | LaSR 3.97 | Complex triple-drug dynamics |
| COVID-19 | 5.01e-8 | ICL 9.35e-8 | Comparable to black-box RNN |
| RNA Polymerase | 0.0111 | ICL 0.0119 | 263-dimensional genomic data |
| Warfarin | 0.565 | ICSR 0.497 | Only case where IGSR was 2nd |
| **Average Rank** | **1.17** | ICL 3.83 | 1st in 5/6 benchmarks |

On LLM-SRBench (128 problems), IGSR achieved the best average rank in NMSE, Acc$_{0.1}$, Term Recall, and Symbolic Accuracy across both ID and OOD sets, also outperforming AFE baselines (e.g., AutoFeat, CAAFE).

### Ablation Study
| Configuration | Outcome | Explanation |
| :--- | :--- | :--- |
| Full IGSR | Best | Complete model. |
| No MCTS | Slow convergence, local optima | Confirms necessity of search structure. |
| No $\Delta_j$ feedback | Rank dropped from 1.17 to 3.83 | Influence score is the core gain source. |
| No History Buffer | Repeated failed terms | In-context memory is essential. |
| IGSR-Agent vs IGSR | Slightly worse, hallucinations | Selection does not strictly require an LLM. |

### Key Findings
- **Fine-grained signals are critical**: Degrading IGSR to use only global loss (ICL) causes performance to regress to baseline levels, indicating $\Delta_j$, not just MCTS, is the primary driver.
- **Deterministic pruning outperforms LLM pruning**: IGSR-Agent is less stable, confirming that "selection should be left to statistics."
- **Wet-lab Validation**: In RNA Pol II pausing modeling, IGSR not only replicated known mechanisms but also hypothesized a new relationship between DNA methylation and Pol II pausing, which was subsequently **supported by wet-lab sequencing experiments**.

## Highlights & Insights
- **Structural Leave-one-out Analysis**: Traditional influence functions study the impact of data points; IGSR applies this to basis functions, obtaining statistics that directly map to "whether to keep a term" at almost zero cost.
- **Decoupling is Generic for Agent Design**: Using LLMs for creative proposals is effective, but entrusting them with scoring/selection introduces hallucinations. Replacing LLM judgments with cheap statistical metrics is a winning strategy for avoiding over-engineering.
- **Scientific Loop**: By moving from "benchmark scoring" to "real biological discovery," IGSR provides a template for AI-driven scientific hypothesis generation in fields like pharmacology and material science.

## Limitations & Future Work
- The model class is locked to a **linear combination** of basis functions $\sum w_j \psi_j$, limiting depth in nested or cyclic dynamics (though $\psi_j$ itself can be non-linear).
- Influence scores are essentially **conditional leave-one-out** metrics, which may underestimate contributions under high collinearity.
- Highly dependent on the LLM's proposal quality; advantages may diminish in "pure math" SR without scientific priors.
- Future Work: Upgrade $\Delta_j$ to group-wise or SHAP-like attribution; incorporate multi-objective rewards (accuracy + simplicity + physical consistency).

## Related Work & Insights
- **vs LLM-SR / D3 / LaSR**: These use LLMs but only provide scalar feedback. IGSR's differentiator is the **structure-aware** feedback from $\Delta_j$, turning search from trial-and-error into directed pruning.
- **vs PySR / SINDy**: Traditional SR evolves within preset operator libraries. IGSR allows the LLM to "think of" $\psi_j$ based on scientific semantics, a critical capability in high-dimensional settings.
- **vs SHAP / LIME**: While all provide attribution, SHAP/LIME are post-hoc explanations for black boxes. $\Delta_j$ is an **active** feedback signal driving the search directly and efficiently.

## Rating
- Novelty: ⭐⭐⭐⭐ Influence functions are not new, but their systematic use as feedback for LLM-based SR and the "generation-selection decoupling" principle are significant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive benchmarks, LLM-SRBench, AFE comparisons, and real-world wet-lab validation.
- Writing Quality: ⭐⭐⭐⭐ Clear differentiation in Table 1, well-defined algorithms, and extensive appendix addressing collinearity and data leakage.
- Value: ⭐⭐⭐⭐⭐ Elevates LLM-based SR to the level of genuine scientific discovery. The design patterns are highly relevant for LLM-agent researchers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] TadA-Bench: A Million-Variant Benchmark for Future-Round Discovery Toward Agentic Protein Engineering](tada-bench_a_million-variant_benchmark_for_future-round_discovery_toward_agentic.md)
- [\[ICLR 2026\] AFD-INSTRUCTION: A Comprehensive Antibody Instruction Dataset with Functional Annotations for LLM-Based Understanding and Design](../../ICLR2026/computational_biology/afd-instruction_a_comprehensive_antibody_instruction_dataset_with_functional_ann.md)
- [\[NeurIPS 2025\] Random Search Neural Networks for Efficient and Expressive Graph Learning](../../NeurIPS2025/computational_biology/random_search_neural_networks_for_efficient_and_expressive_graph_learning.md)
- [\[ICML 2026\] Learning Protein Structure-Function Relationships through Knowledge-guided Representation Decomposition](learning_protein_structure-function_relationships_through_knowledge-guided_repre.md)
- [\[ICLR 2026\] Protein Counterfactuals via Diffusion-Guided Latent Optimization](../../ICLR2026/computational_biology/protein_counterfactuals_via_diffusion-guided_latent_optimization.md)

</div>

<!-- RELATED:END -->
