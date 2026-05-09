---
title: >-
  [Paper Note] HistLens: Mapping Idea Change across Concepts and Corpora
description: >-
  [ACL 2026][conceptual history analysis] This paper proposes HistLens, a framework that leverages sparse autoencoders (SAEs) to decompose concept representations into interpretable semantic basis vectors, enabling the tracking of diachronic evolution trajectories across multiple concepts and corpora within a shared coordinate system. The framework supports implicit concept computation and provides a quantifiable, comparable analytical tool for digital humanities and conceptual history research.
tags:
  - ACL 2026
  - conceptual history analysis
  - sparse autoencoders
  - diachronic semantic change
  - cross-corpus comparison
  - implicit concept computation
date: 2026-05-08
content_hash: 3a582dca1a709135
---

# HistLens: Mapping Idea Change across Concepts and Corpora

**Conference**: ACL 2026
**arXiv**: [2604.11749](https://arxiv.org/abs/2604.11749)
**Code**: [https://github.com/LeoJ-xy/HistLens](https://github.com/LeoJ-xy/HistLens)
**Area**: Digital Humanities / Interpretability
**Keywords**: conceptual history analysis, sparse autoencoders, diachronic semantic change, cross-corpus comparison, implicit concept computation

## TL;DR
This paper proposes HistLens, a framework that leverages sparse autoencoders (SAEs) to decompose concept representations into interpretable semantic basis vectors, enabling the tracking of diachronic evolution trajectories across multiple concepts and corpora within a shared coordinate system. The framework supports implicit concept computation and provides a quantifiable, comparable analytical tool for digital humanities and conceptual history research.

## Background & Motivation

**State of the Field**: Computational diachronic semantics and discourse analysis have made significant advances in recent years, encompassing lexical semantic change detection, topic evolution modeling, and stance and framing analysis. However, integrating these methods into a scalable, comparable, and interpretable paradigm for studying conceptual semantic evolution remains challenging.

**Limitations of Prior Work**: (1) Insufficient scalability and comparability — a large body of work focuses on single concepts or single corpora, making it difficult to directly compare analytical results across different concepts and sources, and unable to address core questions such as "do multiple concepts co-evolve?"; (2) Inadequate characterization of implicit concepts — existing methods rely on keywords and surface co-occurrence patterns, failing to capture concepts that are not explicitly mentioned but expressed through stable discourse patterns, leading to misinterpretation of conceptual change as lexical substitution.

**Root Cause**: Research on conceptual evolution requires balancing interpretability, comparability, and the capture of implicit expression, yet existing computational methods cannot simultaneously satisfy all three requirements.

**Paper Goals**: To construct a unified, multi-concept, multi-corpus conceptual history analysis framework grounded in an interpretable sparse feature space.

**Starting Point**: SAEs are employed to decompose the hidden representations of LLMs into interpretable semantic basis vectors, recasting concept queries as a problem of tracking the activation dynamics of these basis vectors. Different concepts are anchored in the same coordinate system, enabling natural cross-concept comparability.

**Core Idea**: Conceptual evolution is modeled as the activation reorganization of interpretable basis vectors within a shared SAE semantic space — concepts do not simply appear or disappear, but rather undergo a reweighting of their internal semantic components under historical pressures.

## Method

### Overall Architecture
The input consists of timestamped text units, which are encoded by a frozen LLM and mapped into a sparse feature space via a pretrained SAE. Multi-level diachronic analyses are conducted in this space: constructing a concept–corpus atlas → single-concept decomposition → multi-concept comparison → cross-corpus contrast → implicit concept computation. All analyses share the same SAE basis vector coordinate system.

### Key Designs

1. **SAE Sparse Representation and Basis Vector Selection**:

    - Function: Decompose dense neural representations into interpretable sparse features, forming a shared semantic coordinate system.
    - Mechanism: Each sentence is encoded by a frozen LLM (Llama-3.1-8B-Instruct); the hidden state from the residual stream at layer 29 is extracted and mapped to a sparse vector $\mathbf{z}_{i,j} = f_{\text{SAE}}(\mathbf{h}_{i,j})$ via a pretrained OpenSAE, then aggregated to a text-level representation via max pooling. Basis vectors are selected by computing the cumulative drift $D_k = \sum_{s=2}^{S} |\mu_{k,s} - \mu_{k,s-1}|$ for each vector and retaining those with the largest change; human experts then assign semantic labels based on highly activated texts.
    - Design Motivation: SAEs mitigate the polysemy superposition problem of dense representations, and the sharing of a single set of basis vectors across concepts ensures cross-concept and cross-corpus comparability.

2. **Concept–Corpus Atlas and Diachronic Statistics**:

    - Function: Compute reproducible navigation statistics for each (concept, corpus) pair to locate key temporal nodes.
    - Mechanism: The corpus is segmented by year, and three statistics are computed: peak year (the year with the highest concept activation), turning point (the year with the largest change between adjacent slices, with signed intensity $I$), and diversity $H$ (the normalized entropy of basis vector contribution shares). These statistics provide reproducible anchors for systematically selecting cases for downstream in-depth analysis.
    - Design Motivation: Compact quantitative signals replace the subjectivity of manually selecting case studies, making the analytical entry point more systematic.

3. **Implicit Concept Computation**:

    - Function: Distinguish and quantify the explicit lexical expression and implicit discourse practice of concepts.
    - Mechanism: The high-activation text set $\mathcal{I}_{c,r}$ is split into an explicit subset and an implicit subset $\mathcal{I}_{c,r}^{\text{Imp}}$ based on whether texts contain the canonical vocabulary of the concept. The implicit realization ratio is computed as $\bar{r}_{c,r} = \sum_{i \in \mathcal{I}^{\text{Imp}}} m_i^{(c)} / \sum_{i \in \mathcal{I}} m_i^{(c)}$. A high $\bar{r}$ indicates that the concept is primarily expressed through implicit discourse patterns rather than explicit lexical items.
    - Design Motivation: In conceptual history research, concepts are often expressed through stable discourse strategies rather than canonical terminology; neglecting implicit expression introduces source selection bias.

### Loss & Training
The framework involves no training; both the LLM and SAE are used in a frozen state as a purely analytical pipeline.

## Key Experimental Results

### Main Results
Four concepts — "individual," "society," "nation," and "world" — are analyzed across two modern Chinese periodical corpora, *New Youth* and *Guide Weekly*:

| Concept | Corpus | Implicit Ratio $\bar{r}$ | Diversity $H$ | Peak Year | Turning Point (Year, $I$) |
|---------|--------|--------------------------|--------------|-----------|--------------------------|
| Individual | New Youth | 0.920 | 0.741 | 1920 | (1918, +0.226) |
| Nation | New Youth | 0.921 | 0.743 | 1924 | (1918, +0.116) |
| Society | New Youth | 0.595 | 0.368 | 1922 | (1918, −0.213) |
| World | New Youth | 0.900 | 0.683 | 1926 | (1918, +0.230) |
| Individual | Guide Weekly | 0.963 | 0.763 | 1923 | (1923, +0.067) |

### Ablation Study / Cross-Layer Robustness

| Configuration | Description |
|---------------|-------------|
| Layer 29 (main) | Layer used for primary results |
| Layer 06/14/22 | Cross-layer robustness analysis; patterns remain consistent |

### Key Findings
- The concept of "individual" is not a semantically homogeneous object; it can be decomposed into three independently evolving semantic threads: "agentive action," "individualism discourse," and "property/economic individuality."
- In *New Youth*, all four concepts share the strongest turning point in 1918 ($|I|=0.116$–$0.230$), whereas the turning points in *Guide Weekly* are delayed to 1923–1926 with weaker intensity.
- The proportion of implicit concept realization is universally high (0.595–0.963), indicating that concepts are largely expressed through discourse patterns rather than canonical vocabulary.
- Cross-corpus comparison reveals that the concept of "world" shares a semantic skeleton of revolution/class struggle across both corpora, but skews toward intellectual debate in *New Youth* and toward organizational mobilization in *Guide Weekly*.

## Highlights & Insights
- Reconceptualizing conceptual evolution as "the activation reorganization of interpretable semantic components" is a profound insight — concepts do not simply appear or disappear, but rather their internal components are reweighted under historical pressures.
- The SAE space provides a natural comparability infrastructure, enabling cross-concept and cross-corpus comparison without retraining concept-specific spaces — a methodological contribution of considerable value.
- Implicit concept computation offers a systematic means of detecting source selection bias and misinterpretations of conceptual change.

## Limitations & Future Work
- Semantic labeling of SAE basis vectors depends on human expert interpretation, introducing subjectivity.
- Validation is limited to modern Chinese periodical corpora; generalizability to other languages and historical periods remains to be demonstrated.
- Ranking basis vectors by drift magnitude may overlook semantically important but slowly changing components.
- Future work could explore extending the framework to conceptual comparisons across longer time spans and multilingual corpora.

## Related Work & Insights
- **vs. Dynamic Topic Models (DTM)**: DTM learns topic-level evolution but does not provide interpretable semantic decomposition; HistLens offers finer-grained analysis at the basis vector level.
- **vs. Temporal Word Vector Comparison**: Traditional methods suffer from anisotropy and robustness issues; SAE sparse features are more stable.
- **vs. Semantic Difference Keyword Methods**: Such methods focus on lexical-level "semantic battlegrounds," whereas HistLens can capture implicit conceptual expression.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Applying SAEs to conceptual history analysis is an entirely new direction; implicit concept computation is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-concept, multi-corpus analysis is detailed, but limited to a single language and historical period.
- Writing Quality: ⭐⭐⭐⭐⭐ The integration of computational methods and humanistic interpretation is exemplary.
- Value: ⭐⭐⭐⭐⭐ Provides important methodological infrastructure for the digital humanities.
- Overall: ⭐⭐⭐⭐⭐ A model of interdisciplinary integration, representing deep synthesis of computation and the humanities.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] IDEA: An Interpretable and Editable Decision-Making Framework for LLMs via Verbal-to-Numeric Calibration](idea_an_interpretable_and_editable_decision-making_framework_for_llms_via_verbal.md)
- [\[ICLR 2026\] Concepts' Information Bottleneck Models](../../ICLR2026/interpretability/concepts_information_bottleneck_models.md)
- [\[ICLR 2026\] Evolution of Concepts in Language Model Pre-Training](../../ICLR2026/interpretability/evolution_of_concepts_in_language_model_pre-training.md)
- [\[CVPR 2026\] Reallocating Attention Across Layers to Reduce Multimodal Hallucination](../../CVPR2026/interpretability/reallocating_attention_across_layers_to_reduce_multimodal_hallucination.md)
- [\[AAAI 2026\] LLM Circuit Analyses Are Consistent Across Training and Scale](../../AAAI2026/interpretability/llm_circuit_analyses_consistent_across_training_and_scale.md)

<!-- RELATED:END -->
