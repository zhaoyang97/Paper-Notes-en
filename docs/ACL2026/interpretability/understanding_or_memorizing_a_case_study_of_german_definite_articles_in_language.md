---
title: >-
  [Paper Note] Understanding or Memorizing? A Case Study of German Definite Articles in Language Models
description: >-
  [ACL 2026][Interpretability][grammatical encoding] This paper employs the Gradiend gradient-based interpretability method to investigate whether language models predict German definite articles (der/die/das/den/dem/des)…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "grammatical encoding"
  - "memorization vs. generalization"
  - "German articles"
  - "gradient-based interpretability"
  - "Gradiend"
date: 2026-05-08
content_hash: ab593475a2b61816
---

# Understanding or Memorizing? A Case Study of German Definite Articles in Language Models

**Conference**: ACL 2026
**arXiv**: [2601.09313](https://arxiv.org/abs/2601.09313)
**Code**: None
**Area**: Interpretability
**Keywords**: grammatical encoding, memorization vs. generalization, German articles, gradient-based interpretability, Gradiend

## TL;DR

This paper employs the Gradiend gradient-based interpretability method to investigate whether language models predict German definite articles (der/die/das/den/dem/des) by leveraging abstract grammatical rules or surface-level memorization, finding that models rely at least partially on memorized associations rather than strict rule-based encoding.

## Background & Motivation

**Background**: Modern language models achieve near-perfect performance on grammatical agreement tasks, yet whether their internal mechanisms encode abstract grammatical rules (e.g., systematic relations among gender, number, and case) or merely memorize high-frequency token–context associations remains a central question in interpretability research.

**Limitations of Prior Work**: Existing probing studies can only demonstrate that grammatical features are "recoverable from model representations," but cannot establish that these features "causally drive" model predictions. Consequently, high probing accuracy does not imply that a model genuinely understands grammatical rules.

**Key Challenge**: The German definite article system provides an ideal testbed — the same article can correspond to multiple gender–case combinations (e.g., *der* serves as both masculine nominative and feminine dative/genitive), and this syncretism allows researchers to distinguish rule-based from memory-based behavior: if the model encodes rules, an intervention targeting a specific gender–case transformation should affect only that grammatical dimension; if it relies on memorization, the intervention will spill over to unrelated grammatical settings that share the same surface article form.

**Goal**: To empirically determine, via gradient intervention experiments, whether German definite article prediction is rule-driven or memory-driven.

**Key Insight**: The Gradiend method — a gradient-based encoder–decoder interpretability framework — is used to learn parameter update directions for specific gender–case article transformations, and these directions are then tested for generalization to unrelated grammatical settings.

**Core Idea**: If the update direction learned for "masculine nominative *der* → feminine nominative *die*" simultaneously affects unrelated grammatical settings such as "feminine dative *der* → feminine dative *die*," this indicates that the model relies on surface-level memorization rather than abstract rules at these positions.

## Method

### Overall Architecture

The German definite article system (3 genders × 4 cases = 12 slots, 6 article forms) is selected as a controlled experimental system. For each pair of gender–case slots (e.g., masculine nominative ↔ feminine nominative), Gradiend is used to learn a one-dimensional feature direction. The analysis proceeds from three perspectives: encoder value distributions, post-intervention article probability changes, and overlap of update directions in parameter space.

### Key Designs

1. **Gradiend Gradient Feature Learning**

    - *Function*: Learns a compressed one-dimensional feature direction for a specific article transformation.
    - *Mechanism*: Given a transformation between two gender–case slots $z_1, z_2$ (e.g., masculine nominative *der* ↔ feminine nominative *die*), gradients are collected in both directions — factual-target gradient $\nabla^F$ and alternative-target gradient $\nabla^A$. An encoder is trained to compress the gradient difference $\nabla^\Delta = \nabla^F - \nabla^A$ into a scalar $h \in [-1, 1]$ (+1 for one direction, −1 for the other), and a decoder reconstructs the gradient from the scalar. Training data from all non-target slots are treated as identity pairs (factual = alternative), enforcing $h \approx 0$.
    - *Design Motivation*: The one-dimensional bottleneck ensures that the most dominant update direction is captured, while the identity-pair constraint ensures that the learned direction is specific to the target transformation rather than a global perturbation.

2. **Three Complementary Analytical Perspectives**

    - *Function*: Assess rule-based encoding vs. memorization from mutually complementary angles.
    - *Mechanism*: (a) *Encoder analysis* — examines whether gradients from non-target slots are also encoded to non-zero values (under rule-based encoding, these should be zero); (b) *Probability intervention* — applies the learned direction to model parameters and checks whether article probability changes are confined to the target slot (LR), systematically extend to slots sharing the same gender or case dimension (GR), or spill over to unrelated slots sharing the same surface article form (SO); (c) *Top-$k$ parameter overlap* — compares the most important parameters across different transformations to assess whether they overlap substantially.
    - *Design Motivation*: The three perspectives provide complementary evidence from representation space, functional behavior, and parameter space, respectively; any single perspective may be subject to confounding factors.

3. **Learning Rate Selection and Language Modeling Preservation**

    - *Function*: Ensures that intervention effects do not constitute spurious signals caused by model degradation.
    - *Mechanism*: Multiple learning rates $\alpha$ are swept during intervention; only candidates that maintain at least 99% of the language modeling score on a neutral dataset are retained, and the $\alpha^*$ that maximizes the target article probability on the target dataset is selected. SuperGLEBer benchmark scores are additionally reported to confirm that the model's overall capabilities are unaffected.
    - *Design Motivation*: Large parameter updates may alter predictions by disrupting language modeling ability rather than reflecting genuine grammatical mechanisms.

### Loss & Training

Gradiend is trained with an MSE reconstruction loss $\|\text{dec}(\text{enc}(\nabla^A W_m)) - \nabla^\Delta W_m\|_2^2$, using the alternative-target gradient as input, since the factual-target gradient is typically close to zero and thus insufficiently informative.

## Key Experimental Results

### Main Results

Gradiend variants (19 in total) are trained on 6 models: GermanBERT, GBERT, ModernGBERT, EuroBERT, GermanGPT-2, and LLaMA.

| Model | Encoder Correlation | Spillover |
|---|---|---|
| GermanBERT | 90–98% | Pronounced: interventions for *der*→*die* also affect feminine dative/genitive |
| GBERT | 95–99% | Pronounced: similar pattern |
| ModernGBERT | 81–95% | Moderate spillover |
| EuroBERT | 50–73% | Weaker but still significant |
| GermanGPT-2 | 51–71% | Inconsistent patterns |
| LLaMA | 50–67% | Least spillover, potentially reflecting a trend in larger models |

### Ablation Study

| Analysis | Key Finding | Notes |
|---|---|---|
| Probability intervention | Spillover pattern (SO) occurs frequently | *der*→*die* intervention increases *die* probability for feminine dative/genitive (also using *der*) |
| Top-1000 parameter overlap | 40–60% overlap within same-article groups | Different transformations share a large proportion of parameters, far above the random baseline |
| Cross-article-group overlap | 20–30% | Considerable parameter overlap even across transformations involving different articles |
| Control group | Baseline-level overlap | Gradiend trained on shuffled data yields only baseline overlap |

### Key Findings

- **Widespread spillover**: Parameter update directions learned for specific gender–case transformations significantly affect unrelated slots that share the same surface article form, which is inconsistent with the pure rule-based encoding hypothesis.
- **High parameter overlap**: The most important parameters across different article transformations overlap substantially (40–60%), far exceeding the random baseline, indicating that the model does not allocate independent parameter subsets to distinct grammatical relations.
- **Encoder models are more "memory-dependent"**: Encoder models such as GermanBERT and GBERT exhibit the strongest spillover, possibly because bidirectional attention facilitates the exploitation of surface co-occurrence associations.
- **Larger models may memorize less**: LLaMA (3.2B) is the only model that does not exhibit spillover at certain critical slots, suggesting that larger models may tend toward more abstract encoding.
- **Language modeling capability preserved**: SuperGLEBer scores remain essentially unchanged before and after intervention (70.7 → 70.1–70.2), confirming that the observed effects are not attributable to model degradation.

## Highlights & Insights

- **Elegant experimental design**: The syncretism of German articles (*der* can be masculine nominative or feminine dative/genitive) is exploited to construct a natural controlled experiment; this "same surface form, different underlying grammatical function" setup is replicable in other languages.
- **Causal evidence**: The work moves beyond the correlational evidence of traditional probing by providing causal evidence through parameter intervention — the pattern of changes in model predictions directly reveals internal encoding mechanisms.
- **Triangulated three-perspective analysis**: The encoder analysis, probability intervention, and parameter overlap perspectives yield consistent conclusions, substantially strengthening the evidentiary force.

## Limitations & Future Work

- Only one grammatical phenomenon — German definite articles — is studied; whether the conclusions generalize to the morphological systems of other morphologically rich languages remains to be verified.
- Decoder models (GPT-2, LLaMA) require custom MLM prediction heads, which may introduce noise that affects the conclusions.
- The one-dimensional bottleneck of Gradiend may oversimplify what is in practice a multidimensional grammatical encoding.
- The influence of article–noun co-occurrence frequency in training data on the degree of memorization vs. generalization is not explored.

## Related Work & Insights

- **vs. Linear probing**: Probing can only demonstrate that information "exists within" representations; Gradiend intervention can demonstrate that information "causally drives" predictions, providing stronger evidence.
- **vs. Finlayson et al. (2021)**: Their work modifies internal representations to study subject–verb agreement, whereas this paper modifies model parameters to study article prediction — the methodologies are complementary but operate at different levels (representation space vs. parameter space).

## Rating

- Novelty: ⭐⭐⭐⭐ The experimental design exploiting German article syncretism is highly ingenious.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Six models, 19 variants, three-perspective analysis, rigorous statistical testing.
- Writing Quality: ⭐⭐⭐⭐ Clear, though non-German readers may face a non-trivial learning curve.
- Value: ⭐⭐⭐⭐ Provides important causal evidence for the memorization-vs.-rules debate in LM grammatical encoding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ChemVLR: Prioritizing Reasoning in Perception for Chemical Vision-Language Understanding](chemvlr_prioritizing_reasoning_in_perception_for_chemical_vision-language_unders.md)
- [\[ACL 2026\] Rhetorical Questions in LLM Representations: A Linear Probing Study](rhetorical_questions_in_llm_representations_a_linear_probing_study.md)
- [\[ACL 2026\] Tracing Relational Knowledge Recall in Large Language Models](tracing_relational_knowledge_recall_in_large_language_models.md)
- [\[ACL 2026\] Experiments or Outcomes? Probing Scientific Feasibility in Large Language Models](experiments_or_outcomes_probing_scientific_feasibility_in_large_language_models.md)
- [\[ACL 2026\] Towards Intrinsic Interpretability of Large Language Models: A Survey of Design Principles and Architectures](towards_intrinsic_interpretability_of_large_language_modelsa_survey_of_design_pr.md)

</div>

<!-- RELATED:END -->
