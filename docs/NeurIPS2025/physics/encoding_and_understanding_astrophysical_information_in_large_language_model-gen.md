---
title: >-
  [Paper Note] Encoding and Understanding Astrophysical Information in Large Language Model-Generated Summaries
description: >-
  [NeurIPS 2025][Physics & Scientific Computing][LLM embeddings] This work investigates whether LLM embeddings encode physically meaningful quantities derived from X-ray astronomical observations—specifically hardness ratios, power-law indices, and variability indices. Results show that structured prompt design improves clustering purity of physical attributes by 5.9%–57.5%, and sparse autoencoders reveal that LLMs infer physical parameters not explicitly stated by recognizing…
tags:
  - "NeurIPS 2025"
  - "Physics & Scientific Computing"
  - "LLM embeddings"
  - "astrophysics"
  - "sparse autoencoders"
  - "X-ray astronomy"
  - "physical encoding"
date: 2026-05-08
content_hash: aa455fb1a526a137
---

# Encoding and Understanding Astrophysical Information in Large Language Model-Generated Summaries

**Conference**: NeurIPS 2025
**arXiv**: [2511.14685](https://arxiv.org/abs/2511.14685)  
**Code**: None  
**Area**: Physics / LLM Scientific Reasoning
**Keywords**: LLM embeddings, astrophysics, sparse autoencoders, X-ray astronomy, physical encoding

## TL;DR
This work investigates whether LLM embeddings encode physically meaningful quantities derived from X-ray astronomical observations—specifically hardness ratios, power-law indices, and variability indices. Results show that structured prompt design improves clustering purity of physical attributes by 5.9%–57.5%, and sparse autoencoders reveal that LLMs infer physical parameters not explicitly stated by recognizing object types.

## Background & Motivation

**Background**: LLMs have demonstrated cross-domain generalization capabilities, yet it remains unclear whether their embeddings encode physical attributes directly measured from scientific observations—rather than merely textual descriptions. Astrophysics provides an ideal testbed: X-ray sources are characterized by precise physical measurements (from the Chandra Source Catalog) and rich literature descriptions.

**Limitations of Prior Work**: (a) The scientific reasoning capabilities of LLMs are difficult to quantitatively assess; (b) it is unclear how prompt design affects the encoding of physical information; (c) the semantic pathways through which physical concepts are represented in LLM internal representations remain opaque.

**Key Challenge**: Do LLMs genuinely "understand" physics, or have they merely learned statistical correlations? How can physical information be disentangled from embedding spaces?

**Goal**: (1) Does prompt design affect how LLMs encode physical quantities? (2) Which aspects of language are most important for encoding physical information?

**Key Insight**: GPT-4o-mini is used to generate textual summaries for 4,000 X-ray sources from NASA ADS papers; ada-002 is used to produce embeddings; the clustering quality of physical attributes in embedding space is then measured.

**Core Idea**: KNN purity is used to measure the encoding quality of physical attributes in embeddings, while sparse autoencoders are employed to trace semantic pathways—revealing that LLMs infer physical parameters through object-type recognition.

## Method

### Overall Architecture
A three-step pipeline: (1) structured source descriptions are generated from astronomical papers using GPT-4o-mini; (2) embeddings are encoded via ada-002; (3) physical attribute clustering is evaluated using KNN purity, and semantic features are analyzed using a sparse autoencoder (SAE).

### Key Designs

1. **Prompt Engineering (Baseline vs. Optimized)**:

    - Function: Compare the effect of different prompt designs on physical encoding.
    - Mechanism: The baseline prompt simply requests a summary of physical attributes; the optimized prompt includes structured formatting instructions, rules for handling missing information, and explicit guidance to avoid uninformative text.
    - Design Motivation: Different prompts direct the LLM's attention to different information, directly influencing the physical content encoded in the resulting embeddings.

2. **KNN Purity Evaluation**:

    - Function: Quantify the clustering quality of physical attributes in embedding space.
    - Mechanism: For each source, $K=10$ nearest neighbors are identified, and the proportion of neighbors sharing similar physical attributes is computed. Higher purity indicates better encoding of physical attributes.
    - Design Motivation: Directly measures whether physical information is preserved in the embeddings.

3. **Sparse Autoencoder (SAE) Feature Analysis**:

    - Function: Reveal the semantic pathways through which physical information is encoded in LLM embeddings.
    - Mechanism: A pretrained SAE extracts monosemantic features from ada-002 embeddings; the analysis identifies which tokens activate which features; Claude/Gemini is used to annotate the semantic meaning of the most strongly activated features.
    - Design Motivation: SAE provides an interpretable feature decomposition that exposes the physical reasoning mechanisms of LLMs.

### Loss & Training
The SAE is trained with standard sparsity constraints. The main experiments involve no model training—the pipeline is purely evaluative.

## Key Experimental Results

### Main Results

| Physical Quantity | Baseline Prompt Purity | Optimized Prompt Purity | Gain |
|---|---|---|---|
| Hardness Ratio | 0.7998 | 0.8468 | +5.9% |
| Power-Law Index $\gamma$ | 0.8185 | 0.9418 | +15.1% |
| Variability Index | 0.6346 | 0.9994 | **+57.5%** |

### SAE Feature Analysis

| SAE Feature ID | Semantic Label | Associated Physical Quantity |
|---|---|---|
| Feature 1212 | "Scientific inference and likelihood" | Source classification |
| Feature 839 | "X-ray binary burst periodicity" | Variability |
| Feature 2047 | "Non-thermal emission spectrum" | Power-law index |

### Key Findings
- **Physical encoding without explicit numerical values**: The high clustering purity of the power-law index $\gamma$ (0.94) arises from object-type descriptions rather than numerical values—the model encodes physical properties through a causal reasoning chain of "QSO-type → non-thermal spectrum → specific $\gamma$ range."
- **Substantial impact of prompt design**: The variability index purity increases from 0.63 to 0.9994 (+57.5%), demonstrating that the degree of prompt structuring directly controls the retention of physical information.
- **SAE reveals compositional reasoning**: Rather than directly memorizing physical values, the model infers them indirectly through semantic pathways such as object-type recognition and spectral feature association.

## Highlights & Insights
- **Semantic pathways for physical encoding**: LLMs encode physical information through hierarchical reasoning of the form "object type → physical characteristics → parameter inference," which goes beyond simple statistical correlation.
- **Prompts as physical information selectors**: Structured prompt design is equivalent to selecting which physical dimensions the LLM "attends to."
- **SAE as a tool for scientific interpretability**: This work introduces a new instrument for understanding the scientific reasoning processes of LLMs.

## Limitations & Future Work
- The study is limited to X-ray astrophysics; generalization to other scientific domains (e.g., particle physics, chemistry) has not been validated.
- It is not possible to definitively distinguish between "understanding physics" and "learning statistical correlations present in training data."
- Evaluation relies on a single embedding model (ada-002); other models may exhibit different behavior.

## Related Work & Insights
- **vs. Traditional Scientific ML**: Conventional approaches directly model physical equations, whereas this work explores the implicit physical encoding capabilities of LLMs.
- **Insights**: The SAE + KNN evaluation framework is generalizable to assessing domain-specific knowledge encoding in LLMs across any scientific field.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic evaluation of astrophysical information encoding in LLM embeddings; SAE-based analysis is novel.
- Experimental Thoroughness: ⭐⭐⭐ Dataset of 4,000 sources is moderate in scale, but limited to a single scientific domain.
- Writing Quality: ⭐⭐⭐⭐ Cross-disciplinary exposition is clear; figures and tables are intuitive.
- Value: ⭐⭐⭐⭐ Introduces a new methodology for evaluating LLM knowledge encoding in the context of AI for Science.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] L2D: Large Language Models to Diffusion Finetuning](../../ICML2025/physics/large_language_models_to_diffusion_finetuning.md)
- [\[ICML 2025\] Liger: Linearizing Large Language Models to Gated Recurrent Structures](../../ICML2025/physics/liger_linearizing_large_language_models_to_gated_recurrent_structures.md)
- [\[NeurIPS 2025\] Transfer Learning Beyond the Standard Model](transfer_learning_beyond_the_standard_model.md)
- [\[NeurIPS 2025\] Simulation-Based Inference for Neutrino Interaction Model Parameter Tuning](simulation-based_inference_for_neutrino_interaction_model_parameter_tuning.md)
- [\[ICML 2026\] Softplus Attention with Re-weighting Boosts Length Extrapolation in Large Language Models](../../ICML2026/physics/softplus_attention_with_re-weighting_boosts_length_extrapolation_in_large_langua.md)

</div>

<!-- RELATED:END -->
