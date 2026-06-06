---
title: >-
  [Paper Note] ClimateCause: Complex and Implicit Causal Structures in Climate Reports
description: >-
  [ACL 2026][Causal Inference][Causal discovery] ClimateCause constructs the first expert-annotated dataset (874 causal relations) targeting complex and implicit causal structures in climate reports. It supports nested cau…
tags:
  - "ACL 2026"
  - "Causal Inference"
  - "Causal discovery"
  - "climate change"
  - "implicit causality"
  - "nested causality"
  - "IPCC reports"
date: 2026-05-08
content_hash: 72b1e1754ac8a8f2
---

# ClimateCause: Complex and Implicit Causal Structures in Climate Reports

**Conference**: ACL 2026  
**arXiv**: [2604.14856](https://arxiv.org/abs/2604.14856)  
**Code**: [GitHub](https://github.com/laallein/ClimateCause)  
**Area**: Causal Inference / Datasets  
**Keywords**: Causal discovery, climate change, implicit causality, nested causality, IPCC reports

## TL;DR
ClimateCause constructs the first expert-annotated dataset (874 causal relations) targeting complex and implicit causal structures in climate reports. It supports nested causality, multi-event decomposition, correlation direction, and spatio-temporal context annotation. It also proposes a readability metric based on the semantic complexity of causal graphs. LLM benchmarking reveals that causal chain reasoning remains a significant challenge.

## Background & Motivation

**Background**: Textual causal discovery datasets (e.g., BioCause, BECauSE, CNC) are primarily derived from news and social media, focusing on explicit and direct causal relations. Existing datasets lack annotations for implicit causality (inferred through semantics rather than explicit trigger words), nested causality (where one causal relation is embedded within a cause or effect), and multi-event decomposition.

**Limitations of Prior Work**: Causal relationships in climate change are inherently complex—causal networks are multi-layered, restricted by spatio-temporal contexts, and involve uncertainty and confounding factors. Existing datasets cannot represent this complexity, particularly the multiple nested causalities found in abbreviations like CO2-FFI (CO2 emissions from fossil fuel combustion and industrial processes).

**Key Challenge**: The complexity of causal structures in scientific reports far exceeds the expressive capacity of current NLP resources, leading to insufficient evaluation of LLMs in causal reasoning tasks.

**Goal**: To build a high-quality annotated dataset covering implicit, nested, and complex causal structures, and to explore its application in readability metrics and LLM causal reasoning benchmarks.

**Key Insight**: Extract statements from the IPCC Sixth Assessment Report (AR6) and have linguistics and argumentation experts annotate the causal relationships.

**Core Idea**: Noun phrase reconstruction + Multi-event decomposition + Nested causality and spatio-temporal context annotation → Construction of semantically rich causal graphs.

## Method

### Overall Architecture
75 statements were extracted from the IPCC AR6 Synthesis Report and annotated by two expert annotators following detailed guidelines. The annotation process included: Causal existence judgment → Trigger word identification and explicit/implicit classification → Noun phrase reconstruction of causes/effects → Multi-event decomposition → Nested structure labeling → Correlation and relationship type annotation → Spatio-temporal context annotation. This resulted in 874 causal relations.

### Key Designs

1.  **Noun Phrase Reconstruction and Multi-event Decomposition**:
    - **Function**: Standardizes causes and effects into comparable canonical forms to support causal graph construction.
    - **Mechanism**: Reconstructs original expressions of causes/effects into noun phrase forms (e.g., "Unsustainable agricultural expansion increases ecosystem vulnerability" → cause: unsustainable agricultural expansion, effect: increased ecosystem vulnerability). For causes/effects containing multiple events (e.g., "damaged terrestrial, freshwater, cryospheric ecosystems"), they are decomposed into independent causal pairs. The "Belongs_to" and "Combined" fields distinguish between "exemplification" and "joint action."
    - **Design Motivation**: Existing datasets retain mixed representations, which cannot be used for precise event matching in causal graphs or pair-wise verification.

2.  **Implicit and Nested Causal Annotation**:
    - **Function**: Captures causal relationships expressed through semantics rather than explicit trigger words.
    - **Mechanism**: Implicit causality, such as "anthropogenic greenhouse gas emissions" (implying humans → greenhouse gas emissions), lacks trigger words but is inferable via semantics. Nested causality, such as in CO2-FFI, embeds fossil fuel combustion → CO2 emissions and industrial processes → CO2 emissions. These are marked with a "Nested" field and extracted as independent causal relations.
    - **Design Motivation**: A large number of causal relations in scientific reports are expressed implicitly through terminology semantics and domain knowledge; ignoring them significantly underestimates the complexity of causal networks.

3.  **Readability Metric based on Causal Graph Semantic Complexity**:
    - **Function**: Quantifies the cognitive complexity of causal relationships within a statement.
    - **Mechanism**: Proposes complexity indices across five dimensions: common cause/effect structural complexity $C^{com}$, exemplification expansion complexity $C^{ex}$, nested causal complexity $C^{nest}$ (including a $T_i \log T_i$ penalty for multi-layer nesting), correlation direction complexity $C^{corr}$, and relationship type complexity $C^{pol}$. The total complexity $C(s)$ is obtained by an equal-weighted summation after min-max normalization.
    - **Design Motivation**: Traditional readability metrics (e.g., Flesch Reading Ease) are based on word and sentence length and fail to capture the cognitive load of causal reasoning. This metric helps evaluate the comprehensibility of scientific reports for non-expert readers.

### Loss & Training
Ours is a dataset and analysis work and does not involve model training. LLM benchmarking uses zero-shot and few-shot prompting.

## Key Experimental Results

### Main Results

| Metric | Value |
|------|------|
| Annotated Statements | 75 (63 with causal relations) |
| Number of Causal Relations | 874 |
| Number of Unique Relations | 653 (after removing quantifiers) |
| Unique Trigger Words | 95 |
| Explicit vs. Implicit | Primarily explicit, but the proportion of implicit is significantly higher than in existing datasets |
| Positive vs. Negative Relations | Primarily positive |

### Ablation Study

| LLM Task | Challenge | Description |
|---------|------|------|
| Correlation Inference | Moderate | LLMs perform reasonably well in judging positive/negative correlations |
| Causal Chain Reasoning | Difficult | Multi-hop causal reasoning is a key bottleneck for LLMs |

### Key Findings
- 57.33% of statements contain semantically complex causal structures ($C(s)>0$), with the highest complexity reaching 1.821.
- Statement length is significantly positively correlated with causal complexity ($r=0.590, p<0.01$).
- Nested causal relationships are entirely positive correlations; negative correlations appear only in explicit relationships ($\chi^2=26.53, p<0.01$).
- LLM performance on causal chain reasoning is far worse than on correlation inference, indicating that multi-hop causal reasoning is a significant capability deficit in current LLMs.

## Highlights & Insights
- **Readability metrics for causal structures** provide a novel and practical approach—helping organizations like the IPCC evaluate report comprehensibility for policymakers and guiding report simplification.
- The dataset's annotation design is very thorough, particularly the distinction between "Belongs_to/Combined" and the annotation of spatio-temporal contexts, demonstrating how causal annotation can go beyond simple (cause, effect) tuples.
- The concept of nested causality can be extended to other professional fields (e.g., medical reports, legal documents), which are also full of term-level implicit causality.

## Limitations & Future Work
- The dataset size is small (75 statements, 874 causal relations), limiting the possibility of LLM fine-tuning.
- Data is sourced only from the IPCC AR6 Synthesis Report, covering limited climate topics.
- Annotation relies heavily on domain knowledge; inter-annotator agreement in the first round was very low (trigger word identification $\kappa=-0.075$), indicating high annotation difficulty.
- The equal-weighted summation of dimensions in the readability metric is a simplifying assumption and has not been cognitively validated.

## Related Work & Insights
- **vs BioCause**: A biomedical causal dataset featuring cross-sentence causality but lacking nested causality and spatio-temporal context annotations.
- **vs BECauSE 2.0**: Causal datasets from news sources have trigger word annotations but lack implicit causality.
- **vs PolarIs3CAUS/PolarIs4CAUS**: Also in the climate domain but sourced from social media and smaller in scale; ClimateCause is derived from authoritative scientific reports.

## Rating
- Novelty: ⭐⭐⭐⭐ Implicit/nested causal annotation and causal readability metrics are both new contributions.
- Experimental Thoroughness: ⭐⭐⭐ Dataset analysis is thorough, but the scale is small and LLM benchmarking is preliminary.
- Writing Quality: ⭐⭐⭐⭐ The description of the annotation design is clear and detailed, though the readability metric section is notation-heavy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] I-CAM-UV: Integrating Causal Graphs over Non-Identical Variable Sets Using Causal Additive Models with Unobserved Variables](../../AAAI2026/causal_inference/i-cam-uv_integrating_causal_graphs_over_non-identical_variable_sets_using_causal.md)
- [\[ACL 2026\] Learning Invariant Modality Representation for Robust Multimodal Learning from a Causal Inference Perspective](learning_invariant_modality_representation_for_robust_multimodal_learning_from_a.md)
- [\[ACL 2026\] iTAG: Inverse Design for Natural Text Generation with Accurate Causal Graph Annotations](itag_inverse_design_for_natural_text_generation_with_accurate_causal_graph_annot.md)
- [\[ICML 2026\] Controllable Generative Sandbox for Causal Inference](../../ICML2026/causal_inference/controllable_generative_sandbox_for_causal_inference.md)
- [\[AAAI 2026\] Sparse Additive Model Pruning for Order-Based Causal Structure Learning](../../AAAI2026/causal_inference/sparse_additive_model_pruning_for_order-based_causal_structure_learning.md)

</div>

<!-- RELATED:END -->
