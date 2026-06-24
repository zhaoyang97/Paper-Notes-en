---
title: >-
  [Paper Note] Geometric Signatures of Compositionality Across a Language Model's Lifetime
description: >-
  [ACL 2025][LLM (Other)][compositionality] By linking the degree of dataset compositionality with the non-linear intrinsic dimension ($I_d$) and linear effective dimension ($d$) of language model representations, this work reveals a form-meaning dichotomy: non-linear $I_d$ encodes meaningful compositional semantic complexity, whereas linear $d$ encodes surface word-form complexity. This correspondence is established during training alongside the emergence of linguistic capabil…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "compositionality"
  - "intrinsic dimension"
  - "manifold hypothesis"
  - "representation geometry"
  - "training dynamics"
  - "form-meaning dichotomy"
date: 2026-05-08
content_hash: 9c6785e9f4b7b9e7
---

# Geometric Signatures of Compositionality Across a Language Model's Lifetime

**Conference**: ACL 2025  
**arXiv**: [2410.01444](https://arxiv.org/abs/2410.01444)  
**Code**: [jinhl9/llm-compositionality-lifetime](https://github.com/jinhl9/llm-compositionality-lifetime)  
**Area**: LLM/NLP  
**Keywords**: compositionality, intrinsic dimension, manifold hypothesis, representation geometry, training dynamics, form-meaning dichotomy  

## TL;DR

By linking the degree of dataset compositionality with the non-linear intrinsic dimension ($I_d$) and linear effective dimension ($d$) of language model representations, this work reveals a form-meaning dichotomy: non-linear $I_d$ encodes meaningful compositional semantic complexity, whereas linear $d$ encodes surface word-form complexity. This correspondence is established during training alongside the emergence of linguistic capabilities.

## Background & Motivation

**Compositionality of Language**: Language can generate an infinite number of sentences using a finite vocabulary and a small set of syntactic rules. That is, although language seems high-dimensional, it can be explained by relatively few degrees of freedom. If language models are good language modelers, their internal representations should reflect the "relative conciseness" arising from the compositionality of language.

**Manifold Hypothesis**: Existing work has found that LMs compress inputs onto a non-linear manifold where the intrinsic dimension ($I_d$) is far lower than the ambient dimension, but an explicit connection between the **degree of compositionality** and the **geometric complexity of representations** has not yet been established.

**Linear vs. Non-linear Dimensions**: Prior work has leveraged either PCA effective dimension or non-linear $I_d$ separately, but has not systematically compared their distinct roles in encoding linguistic structures. These two types of metrics may encode complementary linguistic information.

**Training Dynamics**: When do LMs learn compositional semantics during pre-training? Recent work (e.g., $I_d$ phase transitions in BERT) provides clues, but a systematic investigation on causal LMs and natural language inputs is lacking.

**Surface Complexity vs. Semantic Complexity**: Word shuffling preserves surface lexical distribution properties but disrupts phrase-level semantics, providing an ablation method to disentangle "form" from "meaning."

**Ours Contributions**: Constitutes the first systematic demonstration on controlled compositional datasets that the degree of input compositionality maps to the geometric complexity of representation manifolds, and reveals a dichotomy where non-linear $I_d$ and linear $d$ correspond to "meaning" and "form", respectively.

## Method

### Overall Architecture

Research Pipeline: Construct controlled datasets with tunable degrees of compositionality $\rightarrow$ Extract LM representations from different layers and training stages $\rightarrow$ Compute the non-linear intrinsic dimension $I_d$ (using TwoNN) and linear effective dimension $d$ (using PCA) separately $\rightarrow$ Correlate geometric complexity with dataset compositionality (approximated by Kolmogorov complexity). Simultaneously, ablate semantics via word shuffling to disentangle the contributions of form and meaning to the dimensions.

### Module 1: Controlled Compositionality Datasets

- Design a synthetic grammar with 12 semantic categories, each containing 50 words, to generate grammatically correct sentences of 17 words.
- Control compositionality via a **coupling factor $k$**: when $k=1$, the 12 categories are sampled independently (12 degrees of freedom); when $k=2$, bigrams are sampled jointly (6 degrees of freedom); when $k=4$, there are only 3 degrees of freedom.
- Key ablation: For each $k$, construct a **shuffled version** (randomly permuting word order) that retains the unigram distribution but destroys phrase-level semantics.
- Compositionality metric: Use the file size after gzip compression to approximate the Kolmogorov complexity (KC).

### Module 2: Intrinsic Dimension Estimation

- **Non-linear $I_d$**: TwoNN estimator—assuming that points on the manifold follow a locally homogeneous Poisson process, it fits $I_d$ using the distribution of the ratio of distances to the first and second nearest neighbors, $\mu = r_2/r_1$. Maximum likelihood estimation is performed over all data points for each layer.
- **Linear $d$**: PCA variance thresholding, where the number of principal components retaining 99% of the variance is used as the effective dimension.
- Representation extraction: Extract representations of the **last token** in each layer of the Transformer residual stream (as it is the only token that can attend to the full context under causal attention).

### Module 3: Models and Training Dynamics Analysis

- Model selection: Pythia suite (from 14M to 12B, using publicly available intermediate training checkpoints), Llama-3-8B, and Mistral-v0.1-7B.
- Training dynamics: Utilize the 143 intermediate checkpoints of Pythia to track the evolution of $I_d$ and $d$ over training steps.
- Linguistic capability evaluation: Evaluate checkpoints on multiple zero-shot tasks (LAMBADA, PIQA, WinoGrande, ARC, etc.) as proxy metrics for "compositional understanding capabilities".

### Training & Evaluation Strategies

- For each setting (with $k \times \{\text{coherent, shuffled}\}$), randomly sample 5 data groups of 10,000 sequences each, reporting the mean $\pm$ standard deviation.
- Validate on both the controlled datasets and The Pile (natural language).
- Use Spearman's rank correlation $\rho$ to measure the strength of association between dimensions and KC.

## Key Experimental Results

### Table 1: Spearman Correlation Between Dimensions and Kolmogorov Complexity

| Metric | 14M | 70M | 160M | 410M | 1.4B | 6.9B | 12B | Llama | Mistral |
|------|-----|-----|------|------|------|------|-----|-------|---------|
| $I_d$  | -0.20 | -0.06 | -0.20 | -0.05 | 0.04 | 0.01 | 0.05 | -0.36 | 0.00 |
| $d$    | 0.90* | 0.47† | -0.50† | 0.96* | 0.96* | 0.92* | 0.86* | 1.0* | 1.0* |

Core findings: Linear $d$ is highly correlated with surface complexity (gzip KC) ($\rho > 0.86$), whereas non-linear $I_d$ shows almost no correlation with KC. This suggests that $I_d$ encodes compositional semantic information that goes beyond surface form.

### Table 2 (Summary of Figure 3): Dimensional Ordering Under Different Coupling Factors $k$

| Setting | $I_d$ Ordering | $d$ Ordering | Effect of Shuffling on $I_d$ | Effect of Shuffling on $d$ |
|------|----------|--------|-----------------|----------------|
| coherent | $k=1 > k=2 > k=3 > k=4$ | $k=1 > k=2 > k=3 > k=4$ | — | — |
| shuffled | $k=1 \approx k=2 \approx k=3 \approx k=4$ (collapsed) | $k=1 > k=2 > k=3 > k=4$ (preserved) | Significantly decreases | Increases instead |

Core findings: After shuffling destroys semantics, $I_d$ collapses to an extremely low range (shuffling feature collapse), but $d$ does not decrease and instead increases—providing direct evidence of the form-meaning dichotomy.

### Key Findings

- **Phase Transition Synchronization**: At training step $t \approx 10^3$, $I_d$ undergoes a drastic redistribution (first decreasing, then recovering), which is precisely synchronized with the sudden burst of the model's linguistic capabilities on zero-shot tasks.
- **Robustness to Model Scale**: $I_d$ does not depend on the hidden dimension $D$ (remaining $O(10)$ across Pythia 14M–12B), whereas $d$ scales linearly with $D$ ($R > 0.99$). This indicates that $I_d$ captures the intrinsic degrees of freedom of the data rather than model capacity.
- **Training Dynamics**: Shuffling feature collapse first emerges at $t \approx 10^3$, precisely when the model starts learning semantic features; before this point, the differentiation of $I_d$ across $k$ exists in both coherent and shuffled data (reflecting architectural inductive bias), after which it is only retained in coherent data (reflecting learned semantic features).

## Highlights & Insights

- **Novel Perspective**: Constitutes the first quantitative link between compositionality—a core property of language—and the geometric complexity of representation spaces.
- **Disentangling Form and Meaning**: The dichotomous discovery of $I_d$ encoding semantics and $d$ encoding form is remarkably elegant, echoing interdisciplinary theories of intrinsic versus embedding dimensionality in neuroscience.
- **Ingenious Experimental Design**: By precisely tuning the degree of compositionality via the coupling factor $k$ alongside shuffling ablation, the experimental setup establishes clear causal relationships.
- **Analysis of Training Dynamics**: Tracking geometric features using public Pythia checkpoints reveals that phase transitions synchronize temporally with the emergence of linguistic capabilities.

## Limitations & Future Work

- Constrained by computation, this study only examines a limited set of syntactic structures and does not cover complex structures such as recursive nesting.
- The model scale is capped at 8B parameters; generalization to larger models remains to be validated.
- Dimensionality metrics only describe "how complex" the features are, but cannot reveal "what the features are"—disentangling and interpreting non-linear features remains an open problem.
- The use of gzip to approximate Kolmogorov complexity has limitations; it cannot perceive semantics, and its distinction between coherent and shuffled data relies entirely on word co-occurrence patterns.

## Related Work & Insights

- **Representation Geometry**: Cai et al. (2021) and Cheng et al. (2023) investigate the $I_d$ of language representations; Hernandez & Andreas (2021) find linear low-dimensional structures for linguistic categories such as part-of-speech.
- **Measuring Compositionality**: Sathe et al. (2024) provide system-level definitions of compositionality; Elmoznino et al. (2025) propose a complexity-based theory of compositionality.
- **Training Dynamics**: Chen et al. (2024) discover $I_d$ phase transitions in BERT synchronized with syntactic acquisition; Lubana et al. (2025) propose percolation models on formal languages.
- **High- vs. Low-Dimensional Representations**: Recanatesi et al. (2021) present a dual-pressure theory of predictive coding (expansion into linear space + compression onto low-$I_d$ manifolds), which is perfectly corroborated by ours experiments.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to establish a quantitative link between compositionality and representation geometry; the discovery of the form-meaning dichotomy is highly original and profound.
- **Technical Depth**: ⭐⭐⭐⭐ — Rigorous experimental design (controlled datasets + natural data, multiple model scales + training dynamics, shuffling ablations) and clear theoretical motivations.
- **Utility**: ⭐⭐⭐ — Primarily provides theoretical insights with limited immediate application scenarios, but offers geometric tools for understanding and improving LM representations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — From Yoshua Bengio's group; rigorous writing logic, intuitive figures, and an exceptionally clear and fluent narrative.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] AgentGym: Evolving Large Language Model-based Agents across Diverse Environments](agentgym_evaluating_and_training_large_language_model-based_agents_across_divers.md)
- [\[ICML 2026\] A Geometric Relation of the Error Introduced by Sampling a Language Model's Output Distribution to its Internal State](../../ICML2026/llm_nlp/a_geometric_relation_of_the_error_introduced_by_sampling_a_language_models_outpu.md)
- [\[NeurIPS 2025\] Strassen Attention, Split VC Dimension and Compositionality in Transformers](../../NeurIPS2025/llm_nlp/strassen_attention_split_vc_dimension_and_compositionality_in_transformers.md)
- [\[ACL 2025\] Convert Language Model into a Value-based Strategic Planner](convert_language_model_into_a_value-based_strategic_planner.md)
- [\[ACL 2025\] Training Language Model to Critique for Better Refinement](training_language_model_to_critique_for_better_refinement.md)

</div>

<!-- RELATED:END -->
