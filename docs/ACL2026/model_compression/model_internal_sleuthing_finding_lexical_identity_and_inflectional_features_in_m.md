---
title: >-
  [Paper Note] Model Internal Sleuthing: Finding Lexical Identity and Inflectional Features in Modern Language Models
description: >-
  [ACL 2026][Model Compression][linguistic probing] This paper systematically probes 25 Transformer language models (ranging from BERT Base to Qwen2.5-7B) and finds that lexical identity (lexeme) is linearly decodable in e…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "linguistic probing"
  - "lexical identity"
  - "inflectional features"
  - "representational geometry"
  - "cross-lingual analysis"
date: 2026-05-08
content_hash: b5527b71a1a1e7bc
---

# Model Internal Sleuthing: Finding Lexical Identity and Inflectional Features in Modern Language Models

**Conference**: ACL 2026
**arXiv**: [2506.02132](https://arxiv.org/abs/2506.02132)
**Code**: [https://github.com/ml5885/model_internal_sleuthing](https://github.com/ml5885/model_internal_sleuthing)
**Area**: Model Compression / NLP Understanding
**Keywords**: linguistic probing, lexical identity, inflectional features, representational geometry, cross-lingual analysis

## TL;DR
This paper systematically probes 25 Transformer language models (ranging from BERT Base to Qwen2.5-7B) and finds that lexical identity (lexeme) is linearly decodable in early layers but decays with depth, while inflectional features remain stably readable across all layers and occupy compact, controllable subspaces.

## Background & Motivation

**Background**: Probing is a central methodology for understanding the internal linguistic representations of Transformers. Early work on BERT and GPT-2 established a hierarchical view in which surface features are encoded in lower layers, syntactic information in middle layers, and semantic content in higher layers.

**Limitations of Prior Work**: Nearly all prior probing studies focus on first-generation models (BERT, GPT-2). Modern LLMs differ substantially in architecture (encoder vs. decoder), training data scale (billions vs. trillions of tokens), and post-training adaptation, leaving the validity of earlier conclusions unverified.

**Key Challenge**: Our understanding of how modern large language models encode fundamental linguistic information — lexical identity versus grammatical inflection — remains grounded in outdated experiments on small models, constituting a significant knowledge gap.

**Goal**: (1) Systematically probe the encoding patterns of lexical identity and inflectional features across 25 modern models; (2) Analyze multiple dimensions including representational geometry, attention vs. residual stream, activation steering, and pretraining dynamics.

**Key Insight**: Lexical identity (lexeme, e.g., *walk* and *walked* sharing a base form) is semantically oriented, while inflectional features (e.g., plural, past tense) are grammatically oriented. This pairing enables a clean disentanglement of how models balance meaning and form.

**Core Idea**: Employ linear and nonlinear probes together with selectivity metrics, representational geometry analysis, and activation steering experiments to comprehensively characterize the encoding trajectories of lexical and inflectional information in modern LLMs.

## Method

### Overall Architecture
For 25 pretrained models spanning 3 architecture types and 6 languages, residual stream activations are extracted from each layer. Linear regression probes and MLP probes are trained to predict lexeme identity and inflectional features respectively. Analysis is conducted across multiple dimensions: selectivity, linear separability gap, effective dimensionality, and activation steering.

### Key Designs

1. **Dual-Probe Framework with Selectivity Metrics**:

    - Function: Distinguish whether the model genuinely encodes linguistic information or whether the probe is merely memorizing.
    - Mechanism: Linear regression and MLP probes are both trained; control tasks with randomized labels are constructed in parallel. Selectivity $\text{Sel}_\ell = \text{Acc}^\text{real}_\ell - \text{Acc}^\text{control}_\ell$ measures genuine linguistic signal; the linear separability gap $\text{Gap}_\ell = \text{Sel}^\text{nonlin}_\ell - \text{Sel}^\text{linear}_\ell$ measures whether nonlinear probes yield genuine information gains or merely capture spurious correlations.
    - Design Motivation: High accuracy does not necessarily imply that linguistic information is truly encoded — it may reflect memorization due to excessive probe capacity. Selectivity metrics effectively filter out this spurious signal.

2. **Representation Geometry Analysis**:

    - Function: Reveal compression and expansion patterns in the model's intermediate representation spaces.
    - Mechanism: The linear effective dimensionality of each layer's activations is computed — i.e., the number of PCA components required to explain a fixed proportion of variance. GPT-2, Qwen2.5, and Pythia exhibit sharp mid-layer dimensionality collapse (absolute activation values spiking to ~8000), whereas Llama and OLMo maintain smooth compression.
    - Design Motivation: Changes in effective dimensionality directly correlate with probe performance and steering efficacy — layers exhibiting dimensionality collapse show markedly reduced steerability.

3. **Inflection Steering**:

    - Function: Causally verify whether inflectional features occupy a controllable low-dimensional subspace.
    - Mechanism: For each pair of inflectional categories (e.g., singular vs. plural), a mean-difference vector is computed and added to the hidden states at varying intensities $\lambda$. A linear probe then measures the class-flip rate following intervention. Results show that even moderate intervention strength ($\lambda=5$) produces substantial probability shifts.
    - Design Motivation: Moving from correlation to causation — probe results demonstrate only that information *exists*; steering experiments demonstrate that it is *controllable*, which has direct implications for representation engineering.

### Loss & Training
Linear probes use ridge regression (closed-form solution). MLP probes are two-layer ReLU networks (hidden dimension 64), trained with standard cross-entropy loss.

## Key Experimental Results

### Main Results

| Attribute | Model Type | Early-Layer Accuracy | Deep-Layer Accuracy | Selectivity Trend |
|-----------|-----------|---------------------|--------------------|--------------------|
| Lexeme | Encoder | 0.8–1.0 | Large drop | Near zero |
| Lexeme | Small decoder | 0.8–1.0 | Gradual drop | Near zero |
| Lexeme | Large decoder | 0.8–1.0 | Remains relatively high | Near zero |
| Inflection | All | 0.9–1.0 | 0.9–1.0 | 0.4–0.6 (positive) |

### Ablation Study

| Analysis Dimension | Key Finding | Notes |
|-------------------|-------------|-------|
| Linear vs. nonlinear | Gap < 0 (globally) | MLP's additional capacity captures spurious correlations rather than genuine linguistic structure |
| Residual stream vs. attention | Residual stream substantially outperforms attention | Mid-layer lexeme: residual 0.6–0.9 vs. attention 0.2–0.4 |
| Cross-lingual | Turkish shows fastest decay | Lexeme accuracy drops from 0.95 to 0.25, due to morphological complexity |
| Pretraining dynamics | Inflection stabilizes early; lexeme continues evolving | Inflection converges within a few checkpoints; lexeme representation continues reshaping in later stages |

### Key Findings
- High early-layer accuracy for lexeme information is accompanied by near-zero selectivity, indicating it is primarily driven by surface correlations (e.g., subword overlap) rather than genuine lexical structure.
- Inflectional information maintains positive selectivity (0.4–0.6) throughout model depth, indicating it is a "genuinely encoded" linguistic property.
- Frequency strongly correlates with probe accuracy — rare lexemes and rare inflectional forms are the dominant sources of error.
- DeBERTa-v3 exhibits a sharp drop in steering efficacy at approximately 75% of model depth, suggesting an architecture-specific representational constraint.

## Highlights & Insights
- **Systematic application of selectivity metrics** is the paper's most notable methodological contribution: by reporting both task accuracy and control comparisons, it effectively addresses the long-standing "memorization artifact" problem in probing research. This paradigm is directly transferable to any probing study.
- The progression from correlation to causation is methodologically coherent: probes establish that information exists, steering demonstrates that it is controllable, and pretraining dynamics reveal when it is acquired.
- The coverage of 25 models × 6 languages is unprecedented in scale, lending the conclusions broad generalizability.

## Limitations & Future Work
- Decoder models use the final subword token as the word representation, which may not be optimal across all architectures.
- Probes can only detect associations rather than causal mechanisms; steering experiments also measure only classifier changes rather than downstream generation effects.
- Syncretism is not addressed (e.g., English infinitive and non-past verb forms are identical in surface form).
- The approach could be extended to larger-scale models (70B+) and additional linguistic features such as syntactic dependencies and semantic roles.

## Related Work & Insights
- **vs. Jawahar et al. (2019) / Tenney et al. (2019)**: These works established the hierarchical linguistic encoding view on BERT; the present paper systematically verifies and updates these findings across 25 modern models.
- **vs. Acs et al. (2024)**: Their multilingual morphosyntactic probing study was limited to mBERT and XLM-RoBERTa; the present paper extends the scope to modern decoder models and incorporates representational geometry analysis.

## Rating
- Novelty: ⭐⭐⭐⭐ Not an entirely new paradigm, but unprecedented in scale and depth of analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across 25 models × 6 languages × multiple analytical dimensions.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured, fluent narrative, and rich in figures and tables.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Reason Only When Needed: Efficient Generative Reward Modeling via Model-Internal Uncertainty](reason_only_when_needed_efficient_generative_reward_modeling_via_model-internal_.md)
- [\[ACL 2026\] Compositional Steering of Large Language Models with Steering Tokens](compositional_steering_of_large_language_models_with_steering_tokens.md)
- [\[ACL 2026\] SeLaR: Selective Latent Reasoning in Large Language Models](selar_selective_latent_reasoning_in_large_language_models.md)
- [\[ACL 2026\] CLAG: Adaptive Memory Organization via Agent-Driven Clustering for Small Language Model Agents](clag_adaptive_memory_organization_via_agent-driven_clustering_for_small_language.md)
- [\[ACL 2026\] JudgeMeNot: Personalizing Large Language Models to Emulate Judicial Reasoning in Hebrew](judgemenot_personalizing_large_language_models_to_emulate_judicial_reasoning_in_.md)

</div>

<!-- RELATED:END -->
