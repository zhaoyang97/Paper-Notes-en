---
title: >-
  [Paper Note] Model Internal Sleuthing: Finding Lexical Identity and Inflectional Features in Modern Language Models
description: >-
  [ACL 2026][Interpretability][Linguistic Probing] This paper systematically performs probe analysis on 25 Transformer language models (ranging from BERT Base to Qwen2.5-7B)…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Linguistic Probing"
  - "Lexical Identity"
  - "Inflectional Features"
  - "Representation Geometry"
  - "Cross-lingual Analysis"
date: 2026-05-08
content_hash: 4bb40a685340cb21
---

# Model Internal Sleuthing: Finding Lexical Identity and Inflectional Features in Modern Language Models

**Conference**: ACL 2026  
**arXiv**: [2506.02132](https://arxiv.org/abs/2506.02132)  
**Code**: [https://github.com/ml5885/model_internal_sleuthing](https://github.com/ml5885/model_internal_sleuthing)  
**Area**: Model Compression / NLP Understanding  
**Keywords**: Linguistic Probing, Lexical Identity, Inflectional Features, Representation Geometry, Cross-lingual Analysis

## TL;DR
This paper systematically performs probe analysis on 25 Transformer language models (ranging from BERT Base to Qwen2.5-7B), discovering that lexical identity (lexeme) is linearly decodable in early layers but decays with depth, while inflectional features (inflection) remain stable and readable across all layers and occupy a compact, controllable subspace.

## Background & Motivation

**Background**: Probing research is a core method for understanding the internal linguistic representations of Transformers. Early work on BERT and GPT-2 established a hierarchical understanding that "different layers encode different linguistic levels"—lower layers encode surface features, middle layers encode syntax, and higher layers encode semantics.

**Limitations of Prior Work**: Previous probing studies focused almost entirely on first-generation models (BERT, GPT-2). However, modern LLMs have undergone significant changes in architecture (Encoder/Decoder), training data scale (billions vs. trillions of tokens), and post-training adaptation. Whether early conclusions still hold remains unverified.

**Key Challenge**: The understanding of how modern large language models encode basic linguistic information (lexical identity vs. grammatical inflection) is still built on outdated experiments with small models, creating a serious knowledge gap.

**Goal**: (1) Systematically probe the encoding patterns of lexical identity and inflectional features across 25 modern models; (2) Analyze multiple dimensions including representation geometry, attention vs. residual stream, activation steering, and pre-training dynamics.

**Key Insight**: Two attributes—lexical identity (lexeme, e.g., "walk/walked" sharing a lemma) and inflectional features (e.g., plural, past tense)—were selected to decouple how models balance "meaning" and "form," with the former related to semantics and the latter to grammar.

**Core Idea**: Use linear/non-linear probes + selectivity metrics + representation geometry analysis + activation steering experiments to comprehensively characterize the encoding trajectories of lexical and inflectional information in modern LLMs.

## Method

### Overall Architecture
For 25 pre-trained models (3 architectures, 6 languages), residual stream activations are extracted from each layer. Linear regression probes and MLP probes are trained to predict lexemes and inflectional features, respectively. The analysis is conducted through multiple lenses: selectivity, linear separability gap, effective dimensionality, and activation steering.

### Key Designs

1.  **Dual-probe + Selectivity Metric System**:
    - **Function**: Distinguish whether the model truly encodes linguistic information or if the probe is simply memorizing.
    - **Mechanism**: Train both linear regression and MLP probes, while constructing control tasks with random labels. Selectivity $\text{Sel}_\ell = \text{Acc}^\text{real}_\ell - \text{Acc}^\text{control}_\ell$ measures the true linguistic signal. The linear separability gap $\text{Gap}_\ell = \text{Sel}^\text{nonlin}_\ell - \text{Sel}^\text{linear}_\ell$ measures whether non-linear probes provide true information gain or merely capture spurious correlations.
    - **Design Motivation**: High accuracy does not necessarily mean linguistic information is truly encoded—it might be memorization due to excessive probe capacity. Selectivity metrics effectively filter these pseudo-signals.

2.  **Representation Geometry Analysis**:
    - **Function**: Reveal the compression/expansion patterns of the representation space in middle layers.
    - **Mechanism**: Calculate the linear effective dimensionality of activations per layer—the number of PCA components required to explain a fixed proportion of variance. Sharp mid-layer dimensional collapse was found in GPT-2, Qwen2.5, and Pythia (where absolute activation values soar to ~8000), while Llama and OLMo maintain smooth compression.
    - **Design Motivation**: Changes in effective dimensionality are directly related to probe performance and steering effectiveness—steering efficacy drops significantly in layers with dimensional collapse.

3.  **Inflection Steering**:
    - **Function**: Causally verify whether inflectional features occupy a controllable low-dimensional subspace.
    - **Mechanism**: Calculate mean difference vectors for each pair of inflectional categories (e.g., singular vs. plural) and add them to hidden states with varying intensity $\lambda$. Measure the category flip rate after intervention using a linear probe. Results show even moderate intensity ($\lambda = 5$) produces significant probability shifts.
    - **Design Motivation**: Moving from correlation to causality—probe results only prove information "existence," while steering experiments prove the information is "manipulatable," which has practical implications for representation engineering.

### Loss & Training
Linear probes use Ridge regression (closed-form solution). MLP probes are two-layer ReLU networks (64-dimensional hidden layer). All are trained using standard cross-entropy loss.

## Key Experimental Results

### Main Results

| Property | Model Type | Early Layer Acc | Deep Layer Acc | Selectivity Trend |
| :--- | :--- | :--- | :--- | :--- |
| Lexeme | Encoder | 0.8-1.0 | Sharp Decrease | Near Zero |
| Lexeme | Small Decoder | 0.8-1.0 | Slow Decrease | Near Zero |
| Lexeme | Large Decoder | 0.8-1.0 | Stays High | Near Zero |
| Inflection | All | 0.9-1.0 | 0.9-1.0 | 0.4-0.6 (Pos) |

### Ablation Study

| Analysis Dimension | Key Findings | Description |
| :--- | :--- | :--- |
| Linear vs. Non-linear | Gap < 0 (Global) | MLP extra capacity captures spurious correlations rather than true linguistic structure |
| Residual vs. Attention | Residual >> Attention | Mid-layer Lexeme: Residual 0.6-0.9 vs. Attention 0.2-0.4 |
| Cross-lingual | Turkish decays fastest | Lexeme accuracy drops from 0.95 to 0.25 due to morphological complexity |
| Training Dynamics | Inflection stable early | Inflection converges in few checkpoints; Lexemes continue reshaping later |

### Key Findings
- High early accuracy of lexeme information accompanied by near-zero selectivity implies it is driven primarily by surface correlations (e.g., subword overlap) rather than true lexical structure.
- Inflectional information maintains positive selectivity (0.4-0.6) across the entire model depth, indicating it is a "truly encoded" linguistic property.
- Frequency is strongly correlated with probe accuracy—rare lexemes and rare inflectional forms are primary error sources.
- DeBERTa-v3 exhibits a sudden drop in steering effectiveness at approximately 75% depth, suggesting unique architectural representation constraints.

## Highlights & Insights
- **Systematic application of selectivity metrics** is the primary methodological highlight: reporting control comparisons alongside accuracy effectively addresses the long-standing "memorization pseudo-signal" issue in probing research. This paradigm is directly transferable to any probing experiment.
- The verification flow from "correlation" to "causality" via activation steering is comprehensive: first using probes to detect information existence, then using steering to prove manipulability, and finally using pre-training dynamics to track when the information forms.
- The scale of covering 25 models across 6 languages is unprecedented, lending strong universality to the conclusions.

## Limitations & Future Work
- Decoder models use the last subword token as the word representation, which may not be optimal for all architectures.
- Probes only detect correlations rather than causal mechanisms; steering experiments also only measure classifier changes rather than effects on downstream generation.
- Syncretism (ambiguity) is not handled (e.g., identical forms for infinitive and non-past verbs in English).
- Future work could extend to larger models (70B+) and more linguistic features (syntactic dependency, semantic roles, etc.).

## Related Work & Insights
- **vs. Jawahar et al. (2019) / Tenney et al. (2019)**: While they established the perception of hierarchical linguistic encoding in BERT, this paper systematically verifies and updates these conclusions across 25 modern models.
- **vs. Acs et al. (2024)**: While they performed multilingual morphosyntactic probing, they were limited to mBERT and XLM-RoBERTa; this work extends to modern decoder models and incorporates representation geometry analysis.

## Rating
- Novelty: ⭐⭐⭐⭐ Not a completely new paradigm, but unprecedented in scale and depth.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive across 25 models, 6 languages, and multiple dimensions.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, fluent narrative, and rich visualizations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Internal Planning in Language Models: Characterizing Horizon and Branch Awareness](../../ICLR2026/interpretability/internal_planning_in_language_models_characterizing_horizon_and_branch_awareness.md)
- [\[ICLR 2026\] Universal Properties of Activation Sparsity in Modern Large Language Models](../../ICLR2026/interpretability/universal_properties_of_activation_sparsity_in_modern_large_language_models.md)
- [\[AAAI 2026\] Finding the Translation Switch: Discovering and Exploiting the Task-Initiation Features in LLMs](../../AAAI2026/interpretability/finding_the_translation_switch_discovering_and_exploiting_the_task-initiation_fe.md)
- [\[CVPR 2026\] Language Models Can Explain Visual Features via Steering](../../CVPR2026/interpretability/language_models_can_explain_visual_features_via_steering.md)
- [\[ACL 2026\] Dual Alignment Between Language Model Layers and Human Sentence Processing](dual_alignment_between_language_model_layers_and_human_sentence_processing.md)

</div>

<!-- RELATED:END -->
