---
title: >-
  [Paper Note] The Rise of Parameter Specialization for Knowledge Storage in Large Language Models
description: >-
  [NeurIPS 2025][LLM/NLP][Parameter Specialization] This paper systematically analyzes 20 open-source LLMs and finds that stronger models exhibit higher degrees of parameter specialization in MLP value vectors — i.e.…
tags:
  - "NeurIPS 2025"
  - "LLM/NLP"
  - "Parameter Specialization"
  - "Knowledge Storage"
  - "MLP Interpretability"
  - "Knowledge Superposition"
  - "Large Language Models"
date: 2026-05-08
content_hash: 42f3f4a4a1f0336d
---

# The Rise of Parameter Specialization for Knowledge Storage in Large Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2505.17260](https://arxiv.org/abs/2505.17260)
**Code**: Not released
**Area**: LLM/NLP
**Keywords**: Parameter Specialization, Knowledge Storage, MLP Interpretability, Knowledge Superposition, Large Language Models

## TL;DR

This paper systematically analyzes 20 open-source LLMs and finds that stronger models exhibit higher degrees of parameter specialization in MLP value vectors — i.e., semantically similar knowledge tends to be concentrated in a small subset of parameter vectors. Causal experiments further confirm a causal relationship between this specialization degree and model performance on knowledge tasks.

## Background & Motivation

Large language models leverage vast parameter spaces to encode enormous amounts of knowledge, yet from a microscopic perspective, **how knowledge is more effectively stored and compressed within limited parameters** remains an underexplored question.

Prior work has interpreted the MLP layers of Transformers as key-value memories (Geva et al., 2021), where the up-projection matrix serves as keys and the down-projection matrix as values, with each value vector acting as a basic unit of knowledge storage. However, since the number of data features far exceeds the number of model parameters, each parameter inevitably overlaps with multiple types of knowledge — a phenomenon known as Knowledge Superposition.

The core problem addressed in this paper is: **How do LLMs of varying capability differ in their knowledge storage patterns? Is there a causal relationship between the degree of parameter specialization and model performance?**

## Method

### Overall Architecture

The study proceeds in two stages:
1. **Observational Analysis**: Systematically measuring the correlation between parameter specialization and model performance across 20 LLMs.
2. **Causal Verification**: Confirming the causal relationship between parameter specialization and performance improvement through controlled experiments.

### Key Designs

**Key-Value Decomposition of MLP Output**: The output of the MLP at layer $l$ of a Transformer can be expressed as a linear combination of value vectors, where each coefficient score controls the contribution weight of the corresponding value vector.

**Knowledge Vectors Masking**: To identify parameter vectors associated with specific concepts, the method runs related and unrelated questions for each concept, computes the differential activation scores across layers, and selects the top-$k$ value vectors with the largest score differences for masking (setting their coefficients to zero).

**Parameter Specialization Score (PSS)**:

$$\text{PSS} = \frac{|\text{General Score after surgery} - \text{Concept Specific Score after surgery}|}{\text{General Score before surgery}}$$

A higher PSS indicates greater specialization of parameter vectors for specific knowledge; a lower PSS indicates more severe knowledge superposition.

### Dataset Construction

The **SpecWiki** dataset is constructed as follows:
- 525 entity concepts (persons, events, locations, etc.) are selected from Wikipedia.
- Concepts are divided into high-frequency, medium-frequency, and low-frequency tiers based on page view counts.
- 10 multiple-choice questions and open-ended generation questions are designed for each concept.
- Questions and options are generated using GPT-4o.

### Loss & Training (Causal Verification Experiments)

Four fine-tuning strategies are designed to verify causality on LLaMA2-7B and Qwen2-7B:
- **FT-FV** (Full Vectors): Fine-tunes all MLP parameter vectors.
- **FT-PV** (Partial Vectors): Fine-tunes only the top-$k/8$ highly activated parameter vectors.
- **FT-CV** (Complementary Vectors): Fine-tunes only the complementary set of parameter vectors.
- **FT-RV** (Random Vectors): Fine-tunes an equal number of randomly selected parameter vectors.

## Key Experimental Results

### Main Results

Analysis results across 20 open-source LLMs:

- The **Pearson correlation between PSS and SpecWiki performance is 0.92; Spearman correlation is 0.93**.
- More recently released models and models with stronger MMLU performance tend to exhibit higher PSS.

**Effect of Model Scale**:

| Model | Accuracy_MCQ | PSS |
|-------|-------------|-----|
| Qwen1.5-0.5B | 0.61 | 0.019 |
| Qwen1.5-1.8B | 0.61 | 0.044 |
| Qwen1.5-4B | 0.73 | 0.075 |
| Qwen1.5-7B | 0.75 | 0.121 |
| Qwen1.5-14B | 0.82 | 0.184 |
| Gemma2-2B | 0.72 | 0.057 |
| Gemma2-9B | 0.86 | 0.138 |

### Ablation Study (Causal Verification)

**Fine-tuning Strategy Comparison** (10 high-frequency concepts on LLaMA2-7B):

| Method | Accuracy_MCQ | Accuracy_OEG | PSS | Semantic Entropy↓ | LID↓ |
|--------|-------------|-------------|-----|-------------------|------|
| Original | 0.60 | 0.51 | 0.67 | 0.67 | 11.23 |
| FT-FV | 0.63 | 0.54 | 0.65 | 0.62 | 11.12 |
| **FT-PV** | **0.67** | **0.59** | **0.72** | **0.50** | **7.89** |
| FT-CV | 0.62 | 0.51 | 0.63 | 0.62 | 11.12 |
| FT-RV | 0.58 | 0.49 | 0.65 | 0.65 | 11.07 |

FT-PV outperforms all baselines across every metric: highest knowledge accuracy, highest specialization degree, and lowest hallucination.

**Evolution of Specialization During Pre-training** (OLMo-2-7B, 10 checkpoints):
- Early training (10K–210K steps): Both PSS and accuracy remain near zero.
- Mid training (310K–510K steps): Accuracy begins to improve while PSS remains below 0.1.
- Late training (610K–910K steps): Parameter specialization emerges markedly, accompanied by substantial accuracy gains.

### Key Findings

1. **More advanced models within the same model family exhibit higher peak PSS**, a finding consistent across the LLaMA, Qwen, Mistral, and Gemma families.
2. **Concept frequency influences specialization**: High-frequency concepts correspond to higher PSS, while low-frequency concepts exhibit both lower accuracy and lower specialization.
3. **Unexpected finding**: Masking a small proportion of concept-relevant vectors (5%–10%) actually improves model performance on unrelated questions.

## Highlights & Insights

1. **First systematic quantification and cross-model comparison of parameter specialization**, filling an important gap in the microscopic understanding of knowledge storage in LLMs.
2. **Elegant causal experimental design**: The four-group comparison of FT-PV vs. FT-FV vs. FT-CV vs. FT-RV precisely isolates the variable of interest.
3. **Pre-training dynamics**: Specialization emerges in the late stages of training, analogous to other emergent capabilities observed in LLMs.
4. **Practical implication**: Fine-tuning high-activation vectors not only yields the best performance but also effectively reduces hallucination.

## Limitations & Future Work

1. **Only MLP layers are analyzed**: Knowledge may also be stored in attention modules, which are not examined.
2. **Model scale constraint**: Due to GPU limitations, experiments are conducted on models up to 14B parameters.
3. **Coarse analysis granularity**: The analysis operates at the level of parameter vectors rather than individual neurons.
4. **SpecWiki dataset scope**: The dataset is grounded in Wikipedia encyclopedic knowledge and does not cover procedural or reasoning-based knowledge.
5. **No exploration of connections to MoE architectures**: Mixture-of-Experts models naturally improve performance through parameter specialization via expert routing, yet this connection is not discussed.

## Related Work & Insights

- **Knowledge Neurons** (Dai et al., 2022) and **Knowledge Editing** (Meng et al., 2022) provide the foundation for knowledge localization.
- **Knowledge Superposition** (Elhage et al., 2022) reveals the prevalence of parameter reuse.
- The work is related to but distinct from **Sparse Autoencoders** in terms of its approach to disentangling features.
- The findings have direct implications for **knowledge distillation** and **model compression**.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First systematic quantification of parameter specialization in LLMs with causal validation; a novel and well-motivated perspective.
- **Technical Depth**: ⭐⭐⭐⭐ — Large-scale analysis across 20 models combined with carefully designed causal verification experiments.
- **Experimental Quality**: ⭐⭐⭐⭐ — Comprehensive coverage across model families, scales, pre-training dynamics, frequency analysis, and causal experiments.
- **Practicality**: ⭐⭐⭐ — Findings offer meaningful insights (e.g., the FT-PV strategy), though directly deployable methods remain limited.
- **Overall**: ⭐⭐⭐⭐ (7.5/10)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] GeoCAD: Local Geometry-Controllable CAD Generation with Large Language Models](geocad_local_geometry-controllable_cad_generation_with_large_language_models.md)
- [\[NeurIPS 2025\] Solving Inequality Proofs with Large Language Models](solving_inequality_proofs_with_large_language_models.md)
- [\[NeurIPS 2025\] Large Language Models Miss the Multi-Agent Mark](large_language_models_miss_the_multi-agent_mark.md)
- [\[NeurIPS 2025\] Scaling Up Active Testing to Large Language Models](scaling_up_active_testing_to_large_language_models.md)
- [\[AAAI 2026\] LoKI: Low-damage Knowledge Implanting of Large Language Models](../../AAAI2026/llm_nlp/loki_low-damage_knowledge_implanting_of_large_language_models.md)

</div>

<!-- RELATED:END -->
