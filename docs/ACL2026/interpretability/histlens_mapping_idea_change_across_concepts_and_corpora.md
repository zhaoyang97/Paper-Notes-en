---
title: >-
  [Paper Note] HistLens: Mapping Idea Change across Concepts and Corpora
description: >-
  [ACL 2026][Interpretability][Conceptual History Analysis] The HistLens framework is proposed, utilizing Sparse Autoencoders (SAE) to decompose conceptual representations into interpretable semantic basis vectors. It trac…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Conceptual History Analysis"
  - "Sparse Autoencoders"
  - "Diachronic Semantic Change"
  - "Cross-corpus Comparison"
  - "Implicit Concept Computation"
date: 2026-05-08
content_hash: a4d9eeed0210637c
---

# HistLens: Mapping Idea Change across Concepts and Corpora

**Conference**: ACL 2026  
**arXiv**: [2604.11749](https://arxiv.org/abs/2604.11749)  
**Code**: [https://github.com/LeoJ-xy/HistLens](https://github.com/LeoJ-xy/HistLens)  
**Area**: Computational Humanities / Interpretability  
**Keywords**: Conceptual History Analysis, Sparse Autoencoders, Diachronic Semantic Change, Cross-corpus Comparison, Implicit Concept Computation

## TL;DR
The HistLens framework is proposed, utilizing Sparse Autoencoders (SAE) to decompose conceptual representations into interpretable semantic basis vectors. It tracks the diachronic evolution trajectories of multiple concepts across various corpora in a shared coordinate system and supports implicit concept computation, providing quantifiable and comparable analysis tools for digital humanities and conceptual history research.

## Background & Motivation

**Background**: Significant progress has been made in computational diachronic semantics and discourse analysis, including lexical semantic change detection, topic evolution modeling, and stance/framing analysis. However, integrating these methods into a scalable, comparable, and interpretable research paradigm for conceptual semantic evolution remains challenging.

**Limitations of Prior Work**: (1) Insufficient scalability and comparability—much work focuses on single concepts or corpora, making it difficult to directly compare analysis results across different concepts or sources, failing to answer core questions like "whether multiple concepts co-evolve"; (2) Inadequate characterization of implicit concepts—existing methods rely on keywords and surface co-occurrence, failing to capture concepts expressed through stable discourse patterns without explicit mention, which leads to misinterpreting conceptual shifts as lexical substitutions.

**Key Challenge**: Conceptual evolution research requires balancing interpretability, comparability, and the capture of implicit expressions, which current computational methods cannot simultaneously satisfy.

**Goal**: To build a unified diachronic conceptual history analysis framework based on an interpretable sparse feature space.

**Key Insight**: SAEs are utilized to decompose LLM hidden representations into interpretable semantic basis vectors, redefining conceptual queries as tracking the activation dynamics of these vectors. Anchoring different concepts in the same coordinate system enables natural comparability.

**Core Idea**: Conceptual evolution is modeled as the activation recombination of interpretable basis vectors in a shared SAE semantic space—concepts do not simply disappear or appear; rather, their internal semantic components are re-weighted under historical pressure.

## Method

### Overall Architecture
Inputs consist of timestamped text units, which are encoded by a frozen LLM and mapped to a sparse feature space via a pre-trained SAE. Multi-level diachronic analysis is performed in this space: constructing a Concept-Corpus Atlas → Single-concept Decomposition → Multi-concept Comparison → Cross-corpus Contrast → Implicit Concept Computation. All analyses share the same SAE basis vector coordinate system.

### Key Designs

1. **SAE Sparse Representation and Basis Vector Selection**:

    - **Function**: Decomposes dense neural representations into interpretable sparse features, forming a shared semantic coordinate system.
    - **Mechanism**: Each sentence is encoded by a frozen LLM (Llama-3.1-8B-Instruct), using the hidden state of the 29th residual stream layer, which is mapped to a sparse vector $\mathbf{z}_{i,j} = f_{\text{SAE}}(\mathbf{h}_{i,j})$ via a pre-trained OpenSAE, and aggregated to text-level via max pooling. Cumulative drift $D_k = \sum_{s=2}^{S} |\mu_{k,s} - \mu_{k,s-1}|$ is calculated for each basis vector to select those with the most change, which are then assigned semantic labels by human experts based on high-activation texts.
    - **Design Motivation**: SAE alleviates the polysemy superposition problem of dense representations, and the shared basis vectors across concepts ensure cross-concept and cross-corpus comparability.

2. **Concept-Corpus Atlas and Diachronic Statistics**:

    - **Function**: Computes reproducible navigation statistics for each (concept, corpus) pair to locate key time nodes.
    - **Mechanism**: Corpora are partitioned by year. Three statistics are calculated: Peak Year (year with highest concept activation), Turning Point (largest change between adjacent slices, with signed intensity $I$), and Diversity $H$ (normalized entropy of basis vector contributions). These statistics provide reproducible anchors for selecting cases for downstream in-depth analysis.
    - **Design Motivation**: Compact quantitative signals replace the subjectivity of manual case selection, systematizing the entry point for analysis.

3. **Implicit Concept Computation**:

    - **Function**: Distinguishes and quantifies explicit lexical expressions versus implicit discursive practices of concepts.
    - **Mechanism**: High-activation text sets $\mathcal{I}_{c,r}$ are split into explicit and implicit subsets $\mathcal{I}_{c,r}^{\text{Imp}}$ based on whether they contain canonical keywords. The implicit realization ratio is calculated as $\bar{r}_{c,r} = \sum_{i \in \mathcal{I}^{\text{Imp}}} m_i^{(c)} / \sum_{i \in \mathcal{I}} m_i^{(c)}$. A high $\bar{r}$ indicates the concept is primarily expressed through implicit discourse patterns rather than explicit vocabulary.
    - **Design Motivation**: In conceptual history, concepts are often expressed through stable discourse strategies rather than canonical terms; ignoring implicit expressions introduces source selection bias.

### Loss & Training
The framework involves no training; both the LLM and SAE are frozen. It is a pure analysis pipeline.

## Key Experimental Results

### Main Results
Analysis of four concepts ("Individual", "Society", "Nation", "World") in two modern Chinese periodicals, *New Youth* and *The Guide Weekly*:

| Concept | Corpus | Implicit Ratio $\bar{r}$ | Diversity $H$ | Peak Year | Turning Point (Year, $I$) |
|------|------|-------------------|-----------|--------|-----------------|
| Individual | New Youth | 0.920 | 0.741 | 1920 | (1918, +0.226) |
| Nation | New Youth | 0.921 | 0.743 | 1924 | (1918, +0.116) |
| Society | New Youth | 0.595 | 0.368 | 1922 | (1918, -0.213) |
| World | New Youth | 0.900 | 0.683 | 1926 | (1918, +0.230) |
| Individual | Guide Weekly | 0.963 | 0.763 | 1923 | (1923, +0.067) |

### Ablation / Cross-layer Robustness

| Config | Description |
|------|------|
| Layer 29 (Ours) | Layer used for primary results |
| Layer 06/14/22 | Cross-layer robustness analysis, with consistent patterns |

### Key Findings
- The "Individual" concept is not a semantically homogeneous object; it can be decomposed into three independently evolving semantic threads: "Agentic Subjectivity", "Individualist Discourse", and "Property/Economic Individuality".
- Four concepts in *New Youth* shared the strongest turning point in 1918 ($|I|=0.116-0.230$), while turning points in *The Guide Weekly* were delayed to 1923-1926 with lower intensity.
- The proportion of implicit conceptual practices is generally high (0.595-0.963), indicating that concepts are extensively expressed through discourse patterns of non-canonical vocabulary.
- Cross-corpus comparison reveals that the "World" concept shares a semantic skeleton of revolution/class struggle in both corpora, but emphasizes intellectual debate in *New Youth* and organizational mobilization in *The Guide Weekly*.

## Highlights & Insights
- Redefining conceptual evolution as "activation recombination of interpretable semantic components" is a profound insight—concepts do not simply appear or disappear, but their internal components are re-weighted under historical pressure.
- The SAE space provides a natural infrastructure for comparability, allowing cross-concept and cross-corpus comparisons without retraining concept-specific spaces, which offers significant methodological value.
- Implicit concept computation provides a systematic detection tool for source selection bias and misinterpretation of conceptual changes.

## Limitations & Future Work
- Semantic labeling of SAE basis vectors relies on human expert interpretation, introducing subjectivity.
- The framework was validated only on modern Chinese periodical corpora; generalizability to other languages and eras remains to be verified.
- Basis vector drift ranking might miss important but slowly changing semantic components.
- Future work could explore expanding the framework to compare concepts over longer time spans and multilingual corpora.

## Related Work & Insights
- **vs Dynamic Topic Models (DTM)**: DTM learns topic-level evolution but lacks interpretable semantic decomposition; HistLens provides finer-grained analysis at the basis vector level.
- **vs Diachronic Word Vector Contrast**: Traditional methods suffer from anisotropy and robustness issues; SAE sparse features are more stable.
- **vs Semantic Difference Keyword Methods**: Such methods focus on lexical-level "semantic battlegrounds"; HistLens captures implicit conceptual expressions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Using SAE for conceptual history analysis is a brand-new direction, and implicit concept computation is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Detailed multi-concept and multi-corpus analysis, though limited to one language and era.
- Writing Quality: ⭐⭐⭐⭐⭐ Integration of computational methods and humanistic interpretation is excellent.
- Value: ⭐⭐⭐⭐⭐ Provides critical methodological infrastructure for digital humanities.
- Overall: ⭐⭐⭐⭐⭐ A model for interdisciplinary fusion, deeply integrating computation and humanities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] IDEA: An Interpretable and Editable Decision-Making Framework for LLMs via Verbal-to-Numeric Calibration](idea_an_interpretable_and_editable_decision-making_framework_for_llms_via_verbal.md)
- [\[ACL 2026\] Follow the Flow: On Information Flow Across Textual Tokens in Text-to-Image Models](follow_the_flow_on_information_flow_across_textual_tokens_in_text-to-image_model.md)
- [\[ICLR 2026\] Concepts' Information Bottleneck Models](../../ICLR2026/interpretability/concepts_information_bottleneck_models.md)
- [\[ICLR 2026\] Evolution of Concepts in Language Model Pre-Training](../../ICLR2026/interpretability/evolution_of_concepts_in_language_model_pre-training.md)
- [\[AAAI 2026\] LLM Circuit Analyses Are Consistent Across Training and Scale](../../AAAI2026/interpretability/llm_circuit_analyses_consistent_across_training_and_scale.md)

</div>

<!-- RELATED:END -->
