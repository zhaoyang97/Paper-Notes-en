---
title: >-
  [Paper Note] Cross-model Transferability among Large Language Models on the Platonic Representations of Concepts
description: >-
  [ACL 2025][LLM (Other)][steering vectors] Proposes the L-Cross Modulation method, which transfers concept steering vectors (SVs) from one LLM to another via simple linear transformations to achieve behavioral control. Three key findings are identified: (1) cross-model SV transfer is effective; (2) different concepts share the same transformation matrix; (3) SVs of smaller models can control larger models (weak-to-strong transfer).
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "steering vectors"
  - "cross-model transfer"
  - "representation alignment"
  - "linear transformation"
  - "LLM interpretability"
date: 2026-05-08
content_hash: 64146f03f70d2c7e
---

# Cross-model Transferability among Large Language Models on the Platonic Representations of Concepts

**Conference**: ACL 2025  
**arXiv**: [2501.02009](https://arxiv.org/abs/2501.02009)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: steering vectors, cross-model transfer, representation alignment, linear transformation, LLM interpretability

## TL;DR
Proposes the L-Cross Modulation method, which transfers concept steering vectors (SVs) from one LLM to another via simple linear transformations to achieve behavioral control. Three key findings are identified: (1) cross-model SV transfer is effective; (2) different concepts share the same transformation matrix; (3) SVs of smaller models can control larger models (weak-to-strong transfer).

## Background & Motivation
**Background**: Steering vectors (SVs) represent the directional behavior of concepts within LLMs and can be used to control generation (e.g., guiding the generation of harmful/harmless content). However, existing research is limited to individual LLMs.

**Limitations of Prior Work**: Different LLMs have distinct representation spaces, preventing the direct cross-model application of SVs—for instance, an SV extracted from Llama cannot be directly applied to Qwen.

**Key Challenge**: If different LLMs learn fundamentally different representations of the same concept, cross-model control would be unfeasible. However, the Platonic Representation Hypothesis suggests that different networks converge toward a shared statistical model of reality.

**Goal**: To verify whether concept representations share an underlying structure across different LLMs and whether cross-model transfer can be achieved via simple transformations.

**Key Insight**: Analogous to Plato's Allegory of the Cave—different LLMs are like different "prisoners" observing different "shadows" of the same reality, where a linear transformation serves as the bridge between these "shadows."

**Core Idea**: Concept representations in different LLMs possess a linearly transformable shared structure, enabling weak-to-strong cross-model control.

## Method

### Overall Architecture
(1) Encode sentences using source and target LLMs on a shared corpus $\rightarrow$ (2) Apply ordinary least squares (OLS) to solve for the linear transformation matrix $\mathbf{T}$ that maps source representations to target representations $\rightarrow$ (3) Apply $\mathbf{T}$ to transform the source model's SV and inject it into the target model's hidden states for behavioral control.

### Key Designs
1. **Learning the Linear Transformation Matrix $\mathbf{T}$**:

    - Function: Solve the OLS to minimize $\|\lambda_\mathcal{D}^{m_t} - \lambda_\mathcal{D}^{m_s} \mathbf{T}'\|$
    - Closed-form solution: $\mathbf{T} = (\lambda^{m_s\top}\lambda^{m_s})^\dagger \lambda^{m_s\top}\lambda^{m_t}$
    - Design Motivation: Linear transformations preserve the fundamental relationships between concepts (involving only rotation and scaling), which helps verify the universality of representations.

2. **Corpus Selection**:

    - Concept-related contrastive texts $Y_W$ can be used (more precise).
    - Concept-unrelated general texts can also be used (more general)—experiments show that $\mathbf{T}$ trained on general corpora still transfers effectively across concepts.

3. **Weak-to-Strong Transfer**:

    - SVs extracted from Qwen-0.5B can effectively control the behavior of Qwen-7B after transformation.
    - This implies that smaller models have already captured the core directions of concepts.

## Key Experimental Results

### RQ1: Cross-model Transfer Effectiveness (11 Concepts)

| Concept | No Mod | Self Mod | L-Cross (Qwen→Llama2) |
|------|--------|----------|----------------------|
| Harmfulness (↑) | 0% | 90%+ | ~90% (close to Self Mod) |
| Happiness (↑) | Low | High | Close to Self |
| Refusal (↓) | Baseline | Large change | Effective change |

### RQ2: Cross-concept Generalization of the Transformation Matrix
- $\mathbf{T}_{Y_{W_1}}$ learned from concept $W_1$ data can effectively transfer SVs of concept $W_2$.
- Even a $\mathbf{T}$ learned from a general corpus (concept-unrelated text) is effective.
- This suggests that the SVs of different concepts share an underlying cross-model transformation structure.

### RQ3: Weak-to-Strong Transfer

| Source Model | Target Model | Effect |
|--------|---------|------|
| Qwen-0.5B | Qwen-7B | Effective (close to same-model Self Mod) |

### Key Findings
- Linear transformations are sufficient to align the concept spaces of different LLMs (without requiring complex non-linear mapping).
- The same transformation matrix is effective across multiple concepts $\rightarrow$ the relational structures between concepts are consistent across models.
- Transfer from small to large models is effective $\rightarrow$ the core directions of concepts are shared across models of different scales.

## Highlights & Insights
- The analogy to Plato's Allegory of the Cave is highly vivid—different LLMs observe different "shadows" of the same "reality," yet the underlying structure remains consistent.
- Linear transferability provides empirical support for the validity of the Platonic Representation Hypothesis at the conceptual level of LLMs.
- Weak-to-strong transfer has direct implications for AI safety: smaller models can be utilized as "concept probes" to control larger models.

## Limitations & Future Work
- Only validated across three LLM families (Llama2/3.1, Qwen2).
- Linear transformations may not apply to vastly different architectures.
- The scaling factor $\beta$ still requires manual adjustment.
- Evaluation in complex scenarios such as multi-turn dialogues has not yet been conducted.

## Related Work & Insights
- **vs Activation Steering (CAA/RepE)**: Extends from single-model to cross-model scenarios, representing the first systematic study.
- **vs Platonic Representation Hypothesis**: Provides empirical support at the conceptual level.
- **vs Weak-to-Strong Generalization (Burns et al.)**: Proves that weak-to-strong transfer is viable within the dimension of concept control.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Cross-model concept transfer and weak-to-strong control provide an entirely new perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ 11 concepts, progress through 3 RQs, and comparisons across multiple models.
- Writing Quality: ⭐⭐⭐⭐⭐ Elegant analogy to Plato's allegory, with clear narrative logic.
- Value: ⭐⭐⭐⭐⭐ Profound impact on LLM interpretability and AI safety research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Representations of Fact, Fiction and Forecast in Large Language Models: Epistemics and Attitudes](representations_of_fact_fiction_and_forecast_in_large_language_models_epistemics.md)
- [\[ACL 2025\] From Neurons to Semantics: Evaluating Cross-Linguistic Alignment Capabilities of Large Language Models via Neurons Alignment](from_neurons_to_semantics_evaluating_cross-linguistic_alignment_capabilities_of_.md)
- [\[ACL 2025\] On the Acquisition of Shared Grammatical Representations in Bilingual Language Models](on_the_acquisition_of_shared_grammatical_representations_in_bilingual_language_m.md)
- [\[ACL 2025\] Language-Codec: Bridging Discrete Codec Representations and Speech Language Models](language_codec_bridging_discrete_codec_speech_language_models.md)
- [\[ACL 2025\] NeKo: Cross-Modality Post-Recognition Error Correction with Tasks-Guided Mixture-of-Experts Language Model](neko_cross-modality_post-recognition_error_correction_with_tasks-guided_mixture-.md)

</div>

<!-- RELATED:END -->
