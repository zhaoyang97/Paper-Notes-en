---
title: >-
  [Paper Note] Lossless Vocabulary Reduction for Auto-Regressive Language Models
description: >-
  [ICLR 2026][LLM Pretraining][Vocabulary Reduction] This paper establishes a **lossless vocabulary reduction** theoretical framework that efficiently converts any auto-regressive language model into an equivalent model us…
tags:
  - "ICLR 2026"
  - "LLM Pretraining"
  - "Vocabulary Reduction"
  - "Auto-Regressive Language Models"
  - "Tokenization"
  - "Model Ensembling"
  - "Lossless Conversion"
date: 2026-05-08
content_hash: 2c6105cbe804d4fc
---

# Lossless Vocabulary Reduction for Auto-Regressive Language Models

**Conference**: ICLR 2026
**arXiv**: [2510.08102](https://arxiv.org/abs/2510.08102)
**Code**: N/A
**Area**: LLM Pretraining
**Keywords**: Vocabulary Reduction, Auto-Regressive LM, Tokenization, Model Ensemble, Maximal Common Vocabulary

## TL;DR

This paper proposes a theoretical framework for **Lossless Vocabulary Reduction (LVR)**, which converts any auto-regressive language model into an exactly equivalent model operating over an arbitrary sub-vocabulary via nested tokenization. Building on the **Maximal Common Vocabulary (MCV)**, the framework enables efficient ensembling of language models with heterogeneous tokenization schemes, with effectiveness validated on GSM8K, MATH, translation, and other tasks.

## Background & Motivation

**Background**: Tokenization is a core component of language models. Various tokenization algorithms—BPE, SentencePiece, Unigram, etc.—are widely used. Each language model maintains its own vocabulary $V$, over which auto-regressive models predict the next-token distribution $p(v_t | v_{<t})$ token by token.

**Limitations of Prior Work**: Vocabularies across different language models are typically incompatible (e.g., Qwen uses 151,643 tokens, Falcon uses 131,072 tokens), meaning their next-token distributions are defined over entirely different probability spaces. This renders techniques requiring token-distribution-level collaboration—such as model ensembling and cross-model knowledge distillation—inapplicable to models with mismatched vocabularies.

**Key Challenge**: Existing solutions are either restricted to ensembling models with identical vocabularies, or resort to byte-level reduction (collapsing all models to a byte-level vocabulary). The former severely limits model selection; the latter results in excessively long generation sequences due to the tiny vocabulary, making inference highly inefficient. No framework exists that is simultaneously lossless, efficient, and general.

**Goal**: How can an arbitrary auto-regressive language model be converted **losslessly** into an equivalent model operating over an arbitrary sub-vocabulary? And how can an optimal common vocabulary be identified across multiple models—one that maximizes generation efficiency while preserving losslessness?

**Key Insight**: The problem is formalized from a probabilistic perspective as a distributional equivalence problem under nested tokenization. The paper proves that, given a sub-vocabulary satisfying a coverage condition, a unique equivalent next-token distribution exists, and provides a recursive construction algorithm.

**Core Idea**: Lossless conversion from a large vocabulary to an arbitrary smaller vocabulary is achieved via exact probability decomposition under nested tokenization. A Maximal Common Vocabulary (MCV) then unifies the output space of heterogeneous models to enable ensembling.

## Method

### Overall Architecture

Given an auto-regressive language model $M$ (vocabulary $V$) and a target sub-vocabulary $V_{\text{sub}}$, the LVR framework converts $M$ into an equivalent model $M'$ using $V_{\text{sub}}$, such that $P_M(x) = P_{M'}(x)$ holds for **any text** $x$. The core construction pipeline:

1. **Nested Tokenization**: Text is first tokenized under $V$; each token's byte string is then re-tokenized under $V_{\text{sub}}$.
2. **Recursive Probability Decomposition**: Probabilities of split tokens are exactly redistributed over multi-step generation paths in the sub-vocabulary.
3. **K-LVR Approximation Algorithm**: Efficient inference is achieved via top-$K$ truncation and caching of relative cover sets.
4. **MCV Ensembling**: Vocabularies of multiple models are intersected to form a maximal common vocabulary; each model is reduced to this common vocabulary prior to ensembling.

### Key Designs

1. **Nested Tokenization & Exact Probability Decomposition**:

    - **Function**: Exactly converts the next-token distribution defined over a large vocabulary into an equivalent distribution over a smaller vocabulary.
    - **Mechanism**: A nested tokenization $\tau_{V \to V_{\text{sub}}}$ is defined, re-tokenizing each token in $V$ using $V_{\text{sub}}$. For tokens that are "split" (e.g., "abc" in $V$ becomes "a"+"b"+"c" in $V_{\text{sub}}$), their probability must be propagated exactly across multi-step generation paths. The paper derives a **recursive decomposition formula**: at each intermediate step, not only must the probability of the split token be distributed, but the contributions of other tokens that share the same prefix must also be correctly accounted for.
    - **Design Motivation**: A naive approach of "cutting probability by prefix" (Naive Restriction) violates the normalization property. Exact computation requires the **relative cover set** $C_{V, V_{\text{sub}}}$ to correctly derive the conditional probability of each sub-vocabulary token in a given context.

2. **Maximal Common Vocabulary (MCV)**:

    - **Function**: Constructs an optimal common target vocabulary for ensembling models with heterogeneous vocabularies.
    - **Mechanism**: Given vocabularies $V_1, V_2, \ldots, V_n$ of multiple models, the MCV is defined as the largest set containing all tokens shared across vocabularies, constructed via intersection and BPE merge rule constraints. All models are losslessly reduced to the MCV and their next-token distributions are then averaged or mixed within a shared output space.
    - **Design Motivation**: Compared to byte-level reduction (yielding only 256 tokens), the MCV retains high-frequency subwords shared across models, substantially reducing sequence length. For example, the MCV of Qwen2.5-3B and Falcon3-7B contains thousands of shared tokens—far exceeding the 256-token byte-level vocabulary.

3. **K-LVR Approximate Inference Algorithm (Algorithm 2)**:

    - **Function**: Efficiently computes the next-token distribution of the reduced model at inference time.
    - **Mechanism**: The exact LVR (Algorithm 1) requires iterating over all tokens in $V$, with complexity proportional to $|V|$. Algorithm 2 applies **top-$K$ truncation** to consider only the $K$ highest-probability tokens, and **caches intermediate computations** of relative cover sets for acceleration. Experiments show that the size of the relative cover set stabilizes around $K$ after a few steps and does not grow with sequence length.
    - **Design Motivation**: Practical deployment requires a trade-off between efficiency and accuracy. $K=1$ suffices for exact greedy decoding simulation; $K \geq 250$ is needed to approximate the full distribution for stochastic sampling; $K \geq 10$ suffices for greedy decoding in ensembling scenarios.

### Loss & Training

- **No additional training required**: LVR is a purely inference-time probabilistic transformation, involving no parameter learning or gradient updates.
- **Theoretical guarantee**: Under exact LVR (Algorithm 1), the conversion satisfies text-level distributional equivalence $P_M(x) = P_{M'}(x)$—a rigorous mathematical theorem with complete existence, uniqueness, and constructive proofs.
- **Approximation strategy**: K-LVR introduces approximation via top-$K$ truncation; larger $K$ yields closer approximation to exact LVR. $K=300$ covers nearly all scenarios with negligible accuracy loss in the paper's experiments.
- **Ensembling strategy**: After reducing all models to the MCV, their next-token distributions are directly averaged arithmetically, without learning additional ensemble weights.

## Key Experimental Results

### Main Results

Losslessness and ensembling effectiveness are evaluated on GSM8K and MATH benchmarks (greedy decoding, $K=300$):

| Model / Configuration | GSM8K Acc (%) | MATH Acc (%) | Notes |
|---|---|---|---|
| Qwen2.5-3B (original vocabulary) | 79.1 | 42.4 | Original model baseline |
| Qwen2.5-3B (K-LVR, N-bytes ≥ 3) | ~79 | ~42 | Losslessness verified: comparable to original |
| Falcon3-7B (original vocabulary) | 77.9 | 30.2 | Original model baseline |
| Falcon3-7B (K-LVR, N-bytes ≥ 3) | ~78 | ~30 | Losslessness verified: comparable to original |
| Naive Restriction (Qwen) | Large drop | Large drop | Naive truncation severely fails |
| **MCV Ensemble (Qwen + Falcon)** | **82.6** | **44.2** | MCV ensembling significantly surpasses both single models |
| Byte-level Ensemble | ~80 | ~41 | Lower efficiency and suboptimal performance |

### Ablation Study

Effect of hyperparameter $K$ on K-LVR approximation quality (Qwen2.5-3B):

| $K$ | Greedy Acc (GSM8K) | Distribution Distance (Random Sampling) | Notes |
|---|---|---|---|
| $K = 1$ | ~79% (matches original) | Large | Top-1 sufficient for greedy decoding |
| $K = 10$ | ~79% | Moderate | Sufficient for ensemble greedy decoding |
| $K = 100$ | ~79% | Small | Begins to approximate exact distribution |
| $K = 250$ | ~79% | Negligible | Sufficient for random sampling |
| $K = 300$ | ~79% | Negligible | Default setting in the paper |

### Key Findings

- **Losslessness verified**: K-LVR under $N$-byte reduction ($N \geq 3$) achieves accuracy on GSM8K and MATH identical to the original model, confirming that the theoretical guarantee holds in practice.
- **Naive Restriction fails critically**: Directly truncating the probability distribution to a sub-vocabulary causes accuracy collapse, underscoring the necessity of exact probability decomposition.
- **MCV ensembling is effective**: Cross-vocabulary ensembling consistently outperforms individual models across multiple model pairs (Qwen+Falcon, Qwen+OLMo2, OLMo2+Falcon, Phi2+Llama3.1, Phi2+Yi1.5), with BLEU improvements also observed on translation tasks (En↔Fr, En↔De).
- **Computational overhead is stable**: The size of the relative cover set stabilizes around $K$ after the first few steps and does not grow with sequence length, making inference overhead effectively constant.
- **Corner case in 2-byte reduction**: Initial experiments showed accuracy degradation for Falcon3-7B under 2-byte reduction, later traced to a corner case bug in the tokenization implementation; accuracy was restored after the fix.

## Highlights & Insights

- **Outstanding theoretical contribution**: This is a mathematically complete framework—providing existence, uniqueness, and constructive proofs for the vocabulary reduction problem, transforming what seemed a heuristic-only problem into an algorithm with strict guarantees. All four reviewers acknowledged the theoretical rigor.
- **Breaking the vocabulary barrier**: Different-vocabulary LLMs have long been regarded as "non-interoperable"; this work fundamentally removes that obstacle, opening pathways for model ensembling, knowledge distillation, and unified evaluation.
- **Training-free inference-time method**: Unlike learned vocabulary reduction, which requires retraining, LVR can be applied directly to any existing pretrained model in a plug-and-play manner.
- **Elegant MCV design**: The maximal common vocabulary, constructed via intersection and BPE merge rule constraints, achieves an optimal balance between byte-level reduction (extremely inefficient) and the full vocabulary (incompatible).
- **Practical K-LVR**: Top-$K$ approximation translates the theoretical framework into a deployable algorithm with clear guidance for selecting $K$ (greedy: $K=1$; sampling: $K \geq 250$).

## Limitations & Future Work

- **Reduced generation efficiency**: Reducing to a smaller vocabulary results in longer token sequences for the same text (MCV vocabulary < original vocabulary), increasing inference steps and latency—especially significant for model pairs with very small common vocabularies.
- **K-LVR is not strictly lossless**: Algorithm 2 introduces top-$K$ truncation, which no longer guarantees exact distributional equivalence; a theoretical bound on the approximation error is also absent (noted by reviewer zQB1).
- **Limited model scale in experiments**: Current experiments are primarily conducted on 3B–13B models; effectiveness and computational overhead on larger-scale models (70B+) require further investigation.
- **Restricted to auto-regressive models**: The framework relies on the sequential nature of auto-regressive generation and cannot be directly extended to bidirectional models (e.g., BERT) or other generation paradigms such as diffusion models.
- **Common vocabulary may degenerate**: If two models' tokenization schemes differ substantially (e.g., byte-level vs. word-level), the MCV may approach byte-level, diminishing efficiency gains.
- **No comparison with learned methods**: Although the objectives differ, reviewers suggested comparing against training-based vocabulary compression methods in terms of efficiency for a more complete positioning of the proposed approach.

## Related Work & Insights

- **Tokenization algorithms**: BPE (Sennrich et al., 2016), SentencePiece, Unigram LM, and others each have their own trade-offs; this work provides a unified solution for cross-tokenization collaboration, eliminating tokenization choice as a barrier to model interoperability.
- **Byte-level reduction**: Collapsing all models to a byte-level vocabulary is the simplest common-space solution but is extremely inefficient; LVR can be viewed as a theoretical generalization and efficiency improvement over byte-level reduction.
- **Model ensembling**: Traditional ensembling requires a shared output space; this work breaks that constraint and opens a new direction for heterogeneous LLM ensembling.
- **Cross-vocabulary knowledge distillation**: Existing work aligns teacher and student model vocabularies using heuristics, lacking theoretical guarantees; the LVR framework can be applied in a principled manner to distillation scenarios, though parallel inference efficiency remains a challenge.
- **Inspiration**: Could a "universal vocabulary" be designed in advance to maximize the MCV across mainstream LLMs? This may provide a theoretical foundation for future tokenization standardization.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First complete theoretical framework for vocabulary reduction, with existence + uniqueness + constructive proofs; the MCV concept is novel and practical.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers multiple model pairs and diverse tasks with thorough sensitivity analysis; limited model scale and absence of wall-clock time comparisons are noted.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical sections are rigorous and clear with intuitive examples; reviewers noted some presentation improvements (visualizations added in the revised version).
- Value: ⭐⭐⭐⭐ — A directional contribution that breaks the vocabulary barrier, with broad implications for model ensembling, distillation, and evaluation; practical deployment still requires engineering adaptation.

---
title: >-
  [Paper Notes] Lossless Vocabulary Reduction for Auto-Regressive Language Models
description: >-
  [ICLR 2026][Vocabulary Reduction] Establishes a **lossless vocabulary reduction** theoretical framework that efficiently converts any auto-regressive language model into an equivalent model using an arbitrary smaller vocabulary without accuracy loss, enabling effective collaboration (e.g., model ensembling) between language models with different tokenization schemes.
tags:
  - ICLR 2026
  - Vocabulary Reduction
  - Auto-Regressive Language Models
  - Tokenization
  - Model Ensembling
  - Lossless Conversion
---

# Lossless Vocabulary Reduction for Auto-Regressive Language Models

**Conference**: ICLR 2026
**arXiv**: [2510.08102](https://arxiv.org/abs/2510.08102)
**Code**: N/A
**Area**: NLP / Language Models
**Keywords**: Vocabulary Reduction, Auto-Regressive Language Models, Tokenization, Model Ensembling, Lossless Conversion

## TL;DR
This paper establishes a **lossless vocabulary reduction** theoretical framework that efficiently converts any auto-regressive language model into an equivalent model using an arbitrary smaller vocabulary without accuracy loss, enabling effective collaboration (e.g., model ensembling) between language models with different tokenization schemes.

## Background & Motivation
Tokenization is one of the core components in language model development. Auto-regressive language models generate text token by token—predicting the probability distribution over the next token given the preceding token sequence—so the tokenization scheme directly affects generation efficiency and quality.

**Key Challenge**: Each language model has its own vocabulary, and different models typically use different vocabularies (e.g., GPT uses BPE, LLaMA uses SentencePiece). This creates a fundamental problem: **language models with different vocabularies cannot directly collaborate at the level of next-token distributions**.

**Concrete Scenario**: Model ensembling is a classical approach to improving language model performance, but traditional ensembling requires models to share the same output space. When two models use different tokenization schemes, their next-token distributions are defined over entirely different vocabularies and cannot be directly averaged or mixed. Existing solutions either restrict ensembling to same-vocabulary models or apply lossy approximate conversions with limited effectiveness.

**Key Insight**: If any language model can be converted "losslessly" into an equivalent model using a smaller vocabulary, different models can first be converted to a shared minimal common vocabulary and then ensembled—this is the core idea of the paper.

## Method

### Overall Architecture
- **Input**: An arbitrary auto-regressive language model $M$ (with vocabulary $V$) and a target vocabulary $V'$ ($V' \subset V$ or $V'$ is a subset of $V$)
- **Output**: An equivalent model $M'$ using vocabulary $V'$, such that $P_M(x) = P_{M'}(x)$ for any text $x$
- **Core constraint**: The conversion is lossless—the string-level probability distribution of the new model is identical to that of the original model

### Key Designs
1. **Theoretical Framework for Vocabulary Reduction**:

    - **Mechanism**: Converting a model from a large vocabulary $V$ to a small vocabulary $V'$ hinges on handling tokens that exist in $V$ but not in $V'$.
    - **Design Motivation**: Suppose a token "abc" in the large vocabulary is absent from $V'$, but "a", "b", and "c" are all present. In the new model $M'$, the probability previously assigned to generating "abc" in a single step must be "redistributed" across the multi-step sequence of first generating "a", then "b", then "c".
    - **Key theorem**: The paper proves that this redistribution can be performed exactly and losslessly—provided the target vocabulary $V'$ satisfies a coverage condition, a unique equivalent next-token distribution exists.

2. **Exact Decomposition of Conditional Probabilities**:

    - Probabilities of split tokens must be exactly redistributed across multi-step generation paths.
    - **Key challenge**: At each intermediate step, the model must not only redistribute the probability of the split token, but also correctly account for other tokens that originally began with the same sub-sequence.
    - The paper derives an exact recursive decomposition formula that preserves all probabilistic properties (normalization, non-negativity).

3. **Maximal Common Vocabulary (MCV)**:

    - **Core concept**: Given vocabularies $V_1, V_2, \ldots, V_n$ of multiple models, the "maximal common vocabulary" is defined as the largest common subset across all vocabularies in a well-defined sense.
    - **Application**: All models can be losslessly reduced to this common vocabulary and then ensembled within the same output space.
    - **Design Motivation**: The maximal common vocabulary minimizes information loss by retaining the largest set of tokens that all models can directly represent.

### Loss & Training
- **No training required**: The key contribution of this paper is a theoretical framework and exact probability transformation formulas, with no additional training involved.
- Vocabulary reduction is a **deterministic inference-time algorithm**, requiring no parameter learning.
- The computational cost of the transformation is primarily determined by the probability redistribution for split tokens, with complexity proportional to the number of removed tokens and their maximum length.

## Key Experimental Results

### Main Results
The framework's effectiveness is validated on model ensembling tasks using language models with different tokenization schemes.

| Dataset | Metric | Ours | Direct Ensemble (same vocab required) | Single Model |
|--------|------|---------|-------------------|--------|
| Language Modeling Benchmarks | Perplexity | Equivalent to theoretical optimum | Same vocabulary only | Higher PPL |
| Text Generation Quality | Multiple metrics | Effective improvement | Infeasible (cross-vocabulary) | Baseline |

### Ablation Study

| Configuration | Key Metric | Notes |
|------|---------|------|
| Original model | PPL_orig | Baseline perplexity |
| Reduced model | PPL_reduced | Exactly matches original (losslessness verified) |
| Reduced to minimal vocabulary | Generation speed decreases | More tokens needed, but probabilities are exact |
| Cross-vocabulary ensemble | PPL decreases | Validates effective ensembling of models with different tokenization |

### Key Findings
- **Completely lossless**: The string-level probabilities of the reduced model are identical to those of the original, confirming the theoretical guarantee.
- **Cross-vocabulary ensembling is feasible**: By reducing to a common vocabulary, successful ensembling of models with different tokenization schemes is demonstrated.
- **Ensembling is highly effective**: Even with different tokenization schemes, ensembled performance substantially surpasses individual models.
- **Computational overhead is manageable**: Although a smaller vocabulary requires more steps to generate the same text, the additional overhead for probability computation itself is small.

## Highlights & Insights
- **Significant theoretical contribution**: This is an elegant theoretical framework providing a complete mathematical solution to the vocabulary reduction problem—with existence, uniqueness, and constructive proofs.
- **Breaking the vocabulary barrier**: Models with different tokenization schemes have long been considered "incomparable"; this paper lifts that restriction.
- **No additional training required**: Vocabulary reduction is a purely probabilistic transformation with no need for additional training data or computation.
- **Opens new research directions**: Model ensembling is only one application; the framework can also be applied to cross-model knowledge distillation, unified evaluation, and more.
- **High mathematical rigor**: The theorems and proofs in the paper are complete, meeting the high standards expected of theoretical work.

## Limitations & Future Work
- **Reduced generation efficiency**: Reducing to a smaller vocabulary requires more token steps to generate the same length of text, increasing inference latency.
- **Limited experimental scale**: Validation is primarily conducted on medium-sized models; effects and efficiency on larger-scale models (70B+) require further experiments.
- **Restricted to auto-regressive models**: Whether the framework can be extended to non-auto-regressive models (e.g., BERT-class) or other generation paradigms (e.g., diffusion models) remains unexplored.
- **Practical deployment challenges**: In production systems, multi-step inference after reduction may require specialized decoding algorithm support.
- **Common vocabulary may be very small**: If two models' tokenization schemes differ substantially, the common vocabulary may degenerate to the character level, leading to very low efficiency.

## Related Work & Insights
- **Tokenization research**: BPE (Sennrich et al., 2016), SentencePiece, Unigram LM, and others each have their own trade-offs; this paper provides a unified solution for cross-tokenization collaboration.
- **Model ensembling**: Traditional ensembling requires a shared output space; this work breaks that constraint.
- **Knowledge distillation**: The framework can be used to "translate" knowledge from a large model to a smaller model with a different vocabulary.
- **Probabilistic perspective**: The paper is fundamentally concerned with establishing exact mappings between different probability spaces, with connections to measure-theoretic transformations in probability theory.
- **Inspiration**: Could a "universal vocabulary" be designed such that all models can be efficiently reduced to it? This may become a theoretical foundation for future tokenization standardization.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Steering Language Models with Weight Arithmetic](steering_language_models_with_weight_arithmetic.md)
- [\[ACL 2026\] Compact Example-Based Explanations for Language Models](../../ACL2026/llm_pretraining/compact_example-based_explanations_for_language_models.md)
- [\[NeurIPS 2025\] The Curse of Depth in Large Language Models](../../NeurIPS2025/llm_pretraining/the_curse_of_depth_in_large_language_models.md)
- [\[NeurIPS 2025\] Scaling Embedding Layers in Language Models](../../NeurIPS2025/llm_pretraining/scaling_embedding_layers_in_language_models.md)
- [\[NeurIPS 2025\] Scalable Fingerprinting of Large Language Models](../../NeurIPS2025/llm_pretraining/scalable_fingerprinting_of_large_language_models.md)

</div>

<!-- RELATED:END -->
