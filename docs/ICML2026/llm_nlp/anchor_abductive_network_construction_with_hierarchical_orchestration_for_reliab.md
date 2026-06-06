---
title: >-
  [Paper Note] ANCHOR: Abductive Network Construction with Hierarchical Orchestration for Reliable Probability Inference in Large Language Models
description: >-
  [ICML 2026][LLM/NLP][abductive reasoning] ANCHOR utilizes "bottom-up abduction + hierarchical clustering" to construct a dense factor space. It performs coarse-to-fine retrieval for downstream conditions to obtain sparse…
tags:
  - "ICML 2026"
  - "LLM/NLP"
  - "abductive reasoning"
  - "Bayesian inference"
  - "LLM uncertainty"
  - "causal Bayesian network"
  - "hierarchical factor space"
date: 2026-05-08
content_hash: 99d714ce3587d3ec
---

# ANCHOR: Abductive Network Construction with Hierarchical Orchestration for Reliable Probability Inference in Large Language Models

**Conference**: ICML 2026  
**arXiv**: [2605.10328](https://arxiv.org/abs/2605.10328)  
**Code**: Not disclosed  
**Area**: LLM Reasoning / Probabilistic Inference / Causal Bayesian Networks  
**Keywords**: abductive reasoning, Bayesian inference, LLM uncertainty, causal Bayesian network, hierarchical factor space

## TL;DR
ANCHOR utilizes "bottom-up abduction + hierarchical clustering" to construct a dense factor space. It performs coarse-to-fine retrieval for downstream conditions to obtain sparse relevant factor sets, then aggregates posteriors by combining Naïve Bayes with a latent-variable Causal Bayesian Network constructed on-the-fly by the LLM. This significantly reduces "unknown" predictions and improves probability calibration in high-stakes LLM decision scenarios.

## Background & Motivation

**Background**: High-stakes decisions, such as emergency response and infrastructure planning, require reliable conditional probability $P(O_i|C)$ estimates from LLMs. Leading approaches (e.g., BIRD) adopt a two-stage "abduction + Bayesian" framework: the LLM first generates a discrete factor set $F=\{F_1,\dots,F_N\}$ and their values from a scenario Scen, then uses Naïve Bayes to marginalize $P(O_i|C)=\sum_f P(O_i|f)\prod_j P(f_j|C)$.

**Limitations of Prior Work**: A dilemma exists: (a) forward abduction often generates sparse factor spaces, causing downstream conditions $u$ to map to zero factors, leading the model to output "unknown"; (b) forcibly expanding the factor set introduces noise and spurious correlations (e.g., "cold weather" and "wearing thick clothes" are highly correlated), violating the conditional independence assumption of Naïve Bayes.

**Key Challenge**: There is a trade-off between factor space coverage (avoiding unknown) and independence (avoiding spurious correlations). Additionally, numerical confidence provided by LLMs is often overconfident and uninterpretable, making it unsuitable for direct use as a probability.

**Goal**: (1) Construct a factor space that is both dense and structured to balance coverage and noise; (2) Design a reliable "condition → relevant factor" retrieval mechanism; (3) Explicitly model latent variable dependencies between factors during probabilistic inference to mitigate the distortion caused by the Naïve Bayes independence assumption.

**Key Insight**: Reverse the traditional "top-down abduction" into **bottom-up abduction**—first generate a large number of supporting/opposing sentences and then extract factors, finally organizing them into a two-level hierarchy using clustering and LLM-based thematic naming. Use the LLM to perform online latent variable structure inference to create a query-level Causal Bayesian Network (CBN) specifically for the current condition $u$.

**Core Idea**: Transform "abduction → factor extraction → retrieval → probability aggregation" into an end-to-end four-stage pipeline. In each stage, let the LLM handle what it does best (generation, extraction, naming, causal discovery, flexible priors), while delegating probabilistic computation to two lightweight models, NB and CBN, for final weighted fusion.

## Method

### Overall Architecture
Input: Scenario Scen + condition $u$ + two candidate hypotheses $O_1, O_2$.
Pipeline:
(1) **Factor Space Construction**: Bottom-up iterative generation → MiniLM embedding → UMAP dimensionality reduction → HDBSCAN clustering → LLM thematic naming → formation of a two-level hierarchy $\tilde{F}$;
(2) **Context-Aware Mapping**: Perform cluster-level KNN coarse retrieval + factor-level KNN fine retrieval on $\tilde{F}$ → filter via self-consistency voting + refine with reflection prompts → obtain a sparse factor set $F^*(u)$;
(3) **Probabilistic Inference**: Use LLM to flexibly provide factor-level posteriors $\phi_f=P(O_1|f)$ and latent variable parameters → construct Naïve Bayes (Outcome → factors) and Causal Bayesian Network (Outcome → latent variables → factors);
(4) **Aggregation**: Weighted fusion of posteriors from both networks to obtain calibrated probabilities. Proactively abstain when $|F^*(u)|=0$ or $\max_i P(O_i|C)<\tau$.

### Key Designs

1. **Bottom-up Abduction + Hierarchical Clustering for Dense Factor Space**:
    - **Function**: Reverses "structure first, factors later" into "mass factor generation first, then structural merging" to alleviate the sparsity of forward abduction.
    - **Mechanism**: Iterate for $T_{max}$ rounds starting from an empty set $F^{(0)}=\emptyset$. Each round involves: (a) few-shot prompts for the LLM to generate $b$ multi-perspective supporting/opposing sentences; (b) extracting factors into $F$, removing semantic duplicates, and checking for convergence. Theoretically, the error rate of factors collected per round under self-consistency voting is upper-bounded by $\exp(-2m(q-0.5)^2)$. After obtaining $F$, use MiniLM for embedding → UMAP for reduction → HDBSCAN for clustering (no $K$ required) → LLM to name clusters (e.g., "Economic Feasibility") and prune redundancies. Each factor is labeled as supports $O_1$ / supports $O_2$ / neutral, forming hierarchy $\tilde{F}$.
    - **Design Motivation**: Single-round forward abduction is limited by prompts. Free generation + posterior structuring decouples "completeness" from "organization." The structured output can be reused across multiple queries to avoid redundant inference.

2. **Coarse-to-Fine Hierarchical Retrieval + Self-Consistency Reflection Refinement**:
    - **Function**: Maps downstream condition $u$ to a high-precision, low-recall-bias factor subset in $\tilde{F}$.
    - **Mechanism**: Construct cluster prototype embeddings $\tilde{C}_j=\alpha\cdot e_{theme}+(1-\alpha)\cdot \frac{1}{|F_j|}\sum_{f\in F_j} e_f$ (hybrid of theme semantics and member mean). Use KNN to find top-$K_1$ clusters → find top-$K_2$ factors within each cluster → union serves as high-recall candidates $F_{cand}(u)$. Refine in two steps: (i) Invoke LLM $R$ times to pick factors directly supported by $u$; factors with votes $v_f(u)=\sum_r \mathbf{1}[f\in m^{(r)}(u)]$ above threshold $\gamma$ form $F_{vote}(u)$; (ii) Use a reflection prompt to explicitly remove remaining irrelevant factors, yielding $F^*(u)$.
    - **Design Motivation**: Brute-force retrieval in dense spaces is computationally intensive. Coarse filtering by semantic clusters ensures millisecond response times. The two-stage refinement uses "voting" and "reflection" prompts to suppress hallucinations and retrieval noise complementarily.

3. **NB + Latent Variable CBN Dual Network with Flexible Parameters + Posterior Aggregation**:
    - **Function**: Explicitly models latent dependencies between factors while retaining the simplicity of NB, outputting better-calibrated probabilities.
    - **Mechanism**: (a) **NB Model**: Root node Outcome ($O_1 / O_2$) connects to each factor $f_j$; query LLM for $\phi_f=P(O_1|f)$ and approximate $P(f|O_1)\approx\phi_f, P(f|O_2)\approx 1-\phi_f$ using symmetric priors. (b) **CBN Model**: Use LLM as a causal discovery engine to output latent variables $L=\{L_1,\dots,L_k\}$ and factor groupings; graph structure is Outcome → $L_i$ → corresponding $f_j$. LLM provides conditional tables like $P(L_i=1|O_k), P(f_j|L_i,O_k)$. (c) **Aggregation**: Perform inference on $P^{NB}(O_i|C)$ and $P^{CBN}(O_i|C)$, then apply weighted fusion.
    - **Design Motivation**: Pure NB assumes factor independence (ignoring correlations within "economic factors"). CBN latent variables act as common parents to absorb intra-category correlations without the data cost of training. LLM-provided flexible priors are naturally suited for "unsupervised decision-making." NB and CBN have different bias directions, allowing for denoising through fusion.

### Loss & Training
Ours does not require training neural networks; all parameters are obtained flexibly via LLM. Hyperparameters include $K_1, K_2$ for clustering, $\alpha$ for cluster prototypes, $R$ for self-consistency, $\gamma$ for voting, $\tau$ for abstaining, $T_{max}$, $N_{target}$, and NB-CBN fusion weights. Experiments used GPT-4 series / Qwen; prompts are in Appendix D.

## Key Experimental Results

### Main Results
The authors claim SOTA performance on preference-based pairwise benchmarks (multiple LLM-driven decision tasks) consistent with BIRD. Representative metrics:

| Method | "Unknown" rate ↓ | Alignment with Human Pref. ↑ | Inference Time ↓ | Token Usage ↓ |
|------|-------------------|--------------------|----------|--------------|
| Direct LLM Estimation | Low | Low (Overconfident) | Low | Low |
| BIRD (Forward Abduction + NB) | High (Sparse) | Medium | Medium | Medium |
| BIRD + Expanded Factors | Medium | Medium-Low (Noise) | High | High |
| **ANCHOR (Ours)** | **Significantly Lower** | **SOTA** | **Significantly Lower** | **Significantly Lower** |

### Ablation Study

| Configuration | Phenomenon | Interpretation |
|------|------|------|
| Only bottom-up + NB | Unknown rate drops vs. BIRD, but biased | Dense coverage solves sparsity |
| + Hierarchical Retrieval | High recall but low precision | Retrieval alone is insufficient; needs refinement |
| + Self-consistency Voting | Precision recovers | Voting removes random noise factors |
| + Reflection Prompt | Further removes irrelevant factors | Two-stage refinement is complementary |
| Pure NB Inference | Biased on correlated factors | Independence assumption fails |
| Pure CBN Inference | Unstable structure, prone to over-param | Single network sensitive to latent mismatch |
| **NB + CBN Weighted** | Best calibration | Complementary denoising |

### Key Findings
- Reducing "unknown" while lowering inference costs is the most significant engineering contribution. Once constructed, the structured factor space is reusable across queries. Single query retrieval + inference only requires $O(K_1 K_2)$ LLM calls, drastically reducing token usage compared to BIRD.
- The number of self-consistency rounds $R$ is sensitive to the recall-precision trade-off; reflection prompts are more effective than simply increasing $R$, indicating "structured criticism" is more informative than repetitive sampling.
- Latent variables are inferred online rather than globally learned; each query has a custom CBN structure, avoiding "mismatched shared latent structures."

## Highlights & Insights
- **Clean Task Division**: Generative, extractive, naming, and causal discovery tasks are assigned to the "LLM expert," while probabilistic calculations are handled by the NB+CBN graphical models. This "Probabilistic Engine + LLM Knowledge Base" division is an excellent paradigm for substituting expert knowledge with LLMs.
- **Reusable Structure vs. One-time Reasoning**: The factor hierarchy $\tilde{F}$ is built once, and downstream queries rely on retrieval—amortizing expensive LLM reasoning into cheap vector retrieval, which is highly engineering-friendly.
- **Abstain as a First-class Citizen**: Explicitly treats "unknown" as a valid output rather than an error. In high-risk scenarios, "better no answer than a wrong one" is more responsible.
- **On-the-fly Query-level CBN**: Traditional causal inference requires stable structures; Ours allows CBN to vary per query, effectively performing "on-demand causal inference." This could extend to dialogue systems or medical decision-making.

## Limitations & Future Work
- Dependency on flexible LLM parameters means that if the LLM's conditional probabilities $\phi_f$ are systematically biased (overconfident or reflecting training data bias), the entire framework follows; lacks independent verification of $\phi_f$ calibration.
- CBN structures generated online have no formal sanity checks, risking "hallucinated latent variables"; the paper does not provide a fallback for nonsensical causal graphs.
- Convergence of bottom-up abduction depends on $T_{max}$ and $N_{target}$; quality varies with LLM diversity, and niche scenarios might remain sparse.
- Benchmarks depend on preference-based pairwise metrics without ground-truth probabilities, making it hard to judge if the numerical output is truly calibrated beyond a proxy for human preference.
- NB+CBN fusion weights are manually specified; an adaptive scheme is missing.

## Related Work & Insights
- **vs. BIRD (Feng et al. 2025)**: BIRD uses forward abduction + single NB, prone to sparsity and independence violations. ANCHOR addresses both by reversing abduction, adding hierarchy, and using CBN.
- **vs. CoT / ToT / Belief Graph**: Chain/Tree of Thought and Belief Graphs are reactive decompositions performed per query; ANCHOR is proactive, pre-building a reusable factor space for higher efficiency.
- **vs. Graph RAG / Hierarchical RAG**: Traditional structured RAG indexes existing documents; ANCHOR generates knowledge sources (factors) from scratch and organizes them, better for decision scenarios lacking domain documentation.
- **vs. LLM Uncertainty (verbalized confidence / sampling)**: Asking LLMs "how confident are you" is unreliable. ANCHOR externalizes uncertainty through explicit probabilistic graphs, providing better interpretability.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of bottom-up abduction, on-the-fly CBN, and NB-CBN fusion is a cohesive and novel synthesis.
- Experimental Thoroughness: ⭐⭐⭐ Main tables and ablations are comprehensive in the appendix, but comparisons against ground-truth probabilities and large-scale cross-domain generalization tests are missing.
- Writing Quality: ⭐⭐⭐⭐ Clear logical chain from motivation to limitations, with well-integrated formulas and flowcharts.
- Value: ⭐⭐⭐⭐ Simultaneously addresses "reducing unknown," "calibration," and "cost reduction" for LLM high-stakes decisions, showing high practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] From Static Inference to Dynamic Interaction: A Survey of Streaming Large Language Models](../../ACL2026/llm_nlp/from_static_inference_to_dynamic_interaction_a_survey_of_streaming_large_languag.md)
- [\[ICML 2026\] Margin-Adaptive Confidence Ranking for Reliable LLM Judgement](margin-adaptive_confidence_ranking_for_reliable_llm_judgement.md)
- [\[ICML 2026\] dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching](dllm-cache_accelerating_diffusion_large_language_models_with_adaptive_caching.md)
- [\[ICML 2026\] Resting Neurons, Active Insights: Robustify Activation Sparsity for Large Language Models](resting_neurons_active_insights_robustify_activation_sparsity_for_large_language.md)
- [\[ICML 2026\] Rare Event Analysis of Large Language Models](rare_event_analysis_of_large_language_models.md)

</div>

<!-- RELATED:END -->
