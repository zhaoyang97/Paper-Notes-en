---
title: >-
  [Paper Note] Interpretable Semantic Gradients in SSD: A PCA Sweep Approach and a Case Study on AI Discourse
description: >-
  [ACL 2026][Interpretability][Supervised Semantic Differential] This paper proposes a PCA sweep procedure for Supervised Semantic Differential (SSD)—a method that "estimates semantic gradients of text embeddings using ind…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Supervised Semantic Differential"
  - "PCA Sweep"
  - "Semantic Gradient"
  - "Narcissism"
  - "Researcher Degrees of Freedom"
date: 2026-05-08
content_hash: d5ecb520e187a6f6
---

# Interpretable Semantic Gradients in SSD: A PCA Sweep Approach and a Case Study on AI Discourse

**Conference**: ACL 2026  
**arXiv**: [2603.13038](https://arxiv.org/abs/2603.13038)  
**Code**: TBD  
**Area**: Interpretability / Psycholinguistics / Embedding Analysis  
**Keywords**: Supervised Semantic Differential, PCA Sweep, Semantic Gradient, Narcissism, Researcher Degrees of Freedom

## TL;DR
This paper proposes a PCA sweep procedure for Supervised Semantic Differential (SSD)—a method that "estimates semantic gradients of text embeddings using individual difference variables." It employs joint diagnostics of interpretability and stability (rather than predictive accuracy) to select the PCA dimension $K$. In a case study involving 349 AI-themed short texts and narcissism questionnaires, the sweep-selected $K=15$ yielded stable semantic gradients of Admiration—"optimistic collaboration vs. distrustful mockery"—while a high-dimensional counterfactual ($K=120$) produced chaotic and uninterpretable clusters.

## Background & Motivation

**Background**: The Semantic Differential paradigm in psychology has a 70-year tradition of using polar adjectives (e.g., warm/cold) to measure the connotative meaning of concepts; modern distributed semantics transfers this to word vector spaces. Supervised Semantic Differential (SSD, Plisiecki et al. 2025) combines both—aggregating each author's text into a Personal Concept Vector, performing linear regression against personality scores, and normalizing coefficients as "semantic gradients." It then clusters the positive and negative poles to estimate how personality differences modulate relational meaning in language.

**Limitations of Prior Work**: A critical unresolved degree of freedom in the SSD pipeline is the PCA step before regression. The selection of $K$ (number of retained principal components) is often unprincipled and relies on researcher aesthetics. This falls into the trap of "undisclosed flexibility in analysis" (Simmons et al. 2011), leaving a back door for false-positive conclusions and compromising the reliability of semantic gradient interpretations.

**Key Challenge**: A $K$ that is too small loses semantic structure, while a $K$ that is too large captures noise dimensions in small corpora and biases the regression direction. The conventional approach of "selecting $K$ by predictive accuracy" conflicts with the interpretability goals of SSD—the most predictive $K$ is not necessarily the most semantically coherent.

**Goal**: (1) Propose a $K$-selection procedure independent of predictive accuracy specifically for interpretability methods; (2) Demonstrate this procedure on real-world psychological cases to prove it yields interpretable and falsifiable semantic gradients.

**Key Insight**: Treat $K$-selection as a joint optimization of "stability of semantic structure" and "interpretability of clusters." A robust $K$ should ensure that semantic clusters are coherent and that gradient directions remain nearly invariant across adjacent $K$ values.

**Core Idea**: Scan $K \in \{1, 3, \dots, 119\}$. For each $K$, execute the full SSD pipeline and record "detrended interpretability" (interpretability, removing the trend of log-variance-explained) and "unit change" (stability, cosine distance between gradients of adjacent $K$). Apply local smoothing to these diagnostics and select the minimum $K$ with the highest joint score.

## Method

### Overall Architecture

Input: Document collection $\{d_i\}$ paired with continuous outcomes $y_i$ (personality scores), a fixed embedding model, and an optional lexicon. Intermediate process: (1) For each candidate $K$, construct Personal Concept Vectors $\mathbf{x}_i \in \mathbb{R}^D$, project to $\tilde{\mathbf{x}}_i \in \mathbb{R}^K$ via PCA, fit $y_i = \alpha + \boldsymbol{\beta}^\top \tilde{\mathbf{x}}_i + \epsilon_i$, and normalize coefficients to obtain the gradient $\hat{\boldsymbol{\beta}}_K$; (2) Project back to the original embedding space, take the top-100 neighbors of each pole for clustering, and select the number of clusters $k \in [2,5]$ via silhouette scores; (3) Calculate three diagnostics: representation (cumulative variance explained), interpretability (cluster coherence + cosine alignment between cluster centroids and the gradient, weighted by cluster size), and stability ($\Delta_K = 1 - \cos(\hat{\boldsymbol{\beta}}_K, \hat{\boldsymbol{\beta}}_{K-1})$); (4) Detrend interpretability signals by log-variance-explained, apply local AUC-K smoothing to interpretability and stability, and select the minimum $K$ with the maximum joint score. Output: The selected $K^*$, the corresponding gradient $\hat{\boldsymbol{\beta}}_{K^*}$, and the polar clusters.

### Key Designs

1.  **Anti-predictive accuracy $K$-selection criterion**:
    - **Function**: Aligns $K$ selection with the stability and interpretation of semantic structures rather than regression fit.
    - **Mechanism**: Calculates representation, interpretability, and stability metrics **after** regression fitting without relying on $R^2$ or $F$ statistics. Interpretability is detrended against log-variance-explained to remove the "pseudo-increase" (where higher dimensions naturally yield tighter clusters); stability monitors the cosine difference of gradients at adjacent $K$. This preserves the essence of SSD as an interpretability method—not rewarding $K$ for "better fit," but for "more consistent semantic structure."
    - **Design Motivation**: Selecting $K$ via $R^2$ favors high dimensions (overfitted regressions). Counterfactual analysis shows that $K=120$ yields a higher $R^2$ than $K=15$ (0.234 vs. 0.19) but results in collapsed semantic clusters.

2.  **Plateau-sensitive smoothing (AUC-K)**:
    - **Function**: Avoids selecting isolated local maxima, favoring broad and stable "plateaus."
    - **Mechanism**: Averages interpretability and stability curves within a local neighborhood (radius 3) to obtain AUC-K values, then standardizes via z-scores. The joint score is defined as $\text{joint\_score}_K = \frac{1}{2}(\text{interp\_auck}_K + \text{stab\_auck}_K)$, picking $K^* = \min \arg\max_K \text{joint\_score}_K$.
    - **Design Motivation**: Interpretability curves are highly volatile at low $K$. Smoothing identifies regions of consistent quality and rewards parsimony.

3.  **Detrending Interpretability**:
    - **Function**: Removes the artifact where high-dimensional representations naturally make neighbors appear tighter.
    - **Mechanism**: After calculating a weighted aggregation of cluster coherence and centroid-gradient alignment, perform regression detrending using $\log(\text{cumulative variance explained})$, taking the residuals. This "interpretability" reflects signal beyond trivial dimensional growth.
    - **Design Motivation**: Raw interpretability metrics are monotonically pulled higher by $K$. Detrending normalizes metrics to a fair baseline, allowing the true plateau to emerge.

### Loss & Training
Only a linear regression $y_i = \alpha + \boldsymbol{\beta}^\top \tilde{\mathbf{x}}_i + \epsilon_i$ is trained using standard OLS for each candidate $K$. Embeddings use Dolma 300-d GloVe with SIF weighting ($a=10^{-3}$) and top principal component removal. Clustering uses silhouette scores to select $k \in [2,5]$. The total sweep computation time is proportional to $|K| \times$ single SSD duration, remaining computationally inexpensive.

## Key Experimental Results

### Main Results (349 AI-themed texts + Narcissism NARQ questionnaire)

| Trait | Selected $K$ | $R^2_{\text{adj}}$ | $F$ | $p$ | $r$ | $\|\hat{\boldsymbol{\beta}}\|$ |
|---|---|---|---|---|---|---|
| Admiration (ADM) | **15** | **0.19** | 6.32 | $<10^{-10}$ | 0.47 | 5.58 |
| Rivalry (RIV) | 23 | 0.03 | 1.43 | 0.095 | 0.30 | 5.38 |

ADM sweep diagnostics: The interpretability curve rises sharply at low $K$ and peaks at $K=15$ (explaining 50% of PCV variance), where the stability curve also hits a plateau. ADM polar clusters were qualitatively categorized into four themes:

| Pole | Size | Theme (Top Words + Excerpt) |
|---|---|---|
| + | 14 | Cultivation & enrichment: cultivate, rediscover, rejuvenated — "AI's potential is boundless..." |
| + | 86 | Innovation & collaboration: innovation, partnership, empower — "AI is transforming our world..." |
| − | 56 | Deception & threat: misleading, dishonest, unfair — "woke, evasive..." |
| − | 44 | Ridicule & contempt: ridiculous, absurd, laughable — "the stupid programmers..." |

### Ablation Study (Counterfactual $K=120$ Proposal)

| Configuration | $R^2_{\text{adj}}$ | $p$ | Semantic Coherence |
|---|---|---|---|
| Sweep $K=15$ | 0.19 | $<10^{-10}$ | 4 clear thematic clusters |
| Counterfactual $K=120$ | **0.234** (Higher) | $2.17 \times 10^{-5}$ | Scattered clusters (mixed brands, brands, animals) |

$K=120$ achieved a higher $R^2$ but cluster contents included "inclusive, asean, maldives, rolex, brics" or "birds, rabbits, squirrels," which were irrelevant to the AI topic. This proves high-dimensional PCA forces noise dimensions into the gradient direction, boosting scores while collapsing interpretability.

### Key Findings
- **$R^2$ High ≠ Semantics Good**: Directly confirmed by counterfactuals. $K=120$ had an $R^2$ 0.04 higher than $K=15$, yet its clusters were uninterpretable—a cautionary tale for using predictive accuracy to select hyperparameters in interpretability research.
- The sweep-selected $K=15$ corresponds to the 50% variance point of PCVs, where interpretability peaks and stability plateaus align—providing strong evidence that the sweep procedure identifies the "sweet spot."
- On ADM, the semantic polarity of "optimistic collaboration vs. distrustful mockery" aligns with psychological theory (Admiration involves agentic, status-seeking tendencies favoring positive narratives), demonstrating that SSD + sweep can produce publishable psychological insights.
- For RIV, the sweep selected $K=23$ but the regression was not significant ($p=0.095$). The procedure avoids "manufacturing" significance where no signal exists.

## Highlights & Insights
- The methodological reminder to "align diagnostics with evaluation goals" is profound. Since SSD is an interpretive method, hyperparameter selection should rely on interpretability/stability rather than predictive accuracy. This principle can be generalized to all mixed-method research (topic models, explanatory PCA, etc.).
- Detrending and AUC-K plateau smoothing effectively solve "mechanical metric inflation" and "local noise peaks."
- The counterfactual $K=120$ experiment is highly educational; by showing a "better $R^2$" that fails semantically, it proves that the sweep's strength lies in structure rather than raw numbers.
- Directly linking individual differences (narcissism) to embedding gradients provides a template for quantitative psychological and sociological research in the LLM era.

## Limitations & Future Work
- The case study is limited to 349 texts, a single language (English), one platform (Prolific), and one topic (AI). Generalizability remains to be tested.
- The sweep only addresses the $K$ degree of freedom; embedding model choice and word window sizes remain unprincipled.
- The use of "full-text PCVs" rather than lexicon-based ones was justified by the specific prompt, but might blur fine-grained semantics.
- Psychological interpretations remain correlational. The ADM "optimistic narrative preference" could partly result from LLM sycophancy biases + user rewriting, requiring further experimental dissociation.

## Related Work & Insights
- **vs. Plisiecki et al. 2025 (Original SSD)**: Extends the pipeline by upgrading $K$ selection from "researcher intuition" to "principled scanning."
- **vs. Mimno et al. 2011 (Topic Coherence)**: Adapts the concept of coherence to evaluate the "interpretability" of regression gradients.
- **vs. Garg et al. 2018 / Kozlowski et al. 2019 (Embedding Bias)**: While they study social norms in embedding spaces, this work uses gradients and individual differences for more granular linguistic personality research.
- **vs. Simmons et al. 2011 (False-positive Psychology)**: Acts as a tool-level response to the call for reducing undisclosed analytical flexibility.

## Rating
- Novelty: ⭐⭐⭐ (Individual components are not new, but their combination for SSD is novel.)
- Experimental Thoroughness: ⭐⭐⭐ (Limited sample size; sufficient PoC for a methodological paper.)
- Writing Quality: ⭐⭐⭐⭐ (Clear motivation, concise formulas, and honest discussion of boundaries.)
- Value: ⭐⭐⭐⭐ (An immediately usable tool for SSD users and a methodological model for embedding-based psychological research.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Understanding or Memorizing? A Case Study of German Definite Articles in Language Models](understanding_or_memorizing_a_case_study_of_german_definite_articles_in_language.md)
- [\[ACL 2026\] A Structured Clustering Approach for Inducing Media Narratives](a_structured_clustering_approach_for_inducing_media_narratives.md)
- [\[ACL 2026\] Rhetorical Questions in LLM Representations: A Linear Probing Study](rhetorical_questions_in_llm_representations_a_linear_probing_study.md)
- [\[ACL 2026\] NOSE: Neural Olfactory-Semantic Embedding with Tri-Modal Orthogonal Contrastive Learning](nose_neural_olfactory-semantic_embedding_with_tri-modal_orthogonal_contrastive_l.md)
- [\[ACL 2026\] Constructing Interpretable Features from Compositional Neuron Groups](constructing_interpretable_features_from_compositional_neuron_groups.md)

</div>

<!-- RELATED:END -->
