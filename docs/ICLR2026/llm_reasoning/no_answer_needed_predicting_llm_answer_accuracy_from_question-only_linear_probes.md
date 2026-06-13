---
title: >-
  [Paper Note] No Answer Needed: Predicting LLM Answer Accuracy from Question-Only Linear Probes
description: >-
  [ICLR 2026][LLM Reasoning][Linear Probes] Prior to answer generation, a linear probe (difference-of-means) trained solely on residual stream activations at the question-processing stage can predict whether a model's fort…
tags:
  - "ICLR 2026"
  - "LLM Reasoning"
  - "Linear Probes"
  - "Correctness Direction"
  - "LLM Internal Representations"
  - "Self-Assessment"
  - "Linear Representation Hypothesis"
  - "Confidence"
date: 2026-05-08
content_hash: 92c1258377ed89de
---

# No Answer Needed: Predicting LLM Answer Accuracy from Question-Only Linear Probes

**Conference**: ICLR 2026  
**arXiv**: [2509.10625](https://arxiv.org/abs/2509.10625)  
**Code**: [ivanvmoreno/correctness-model-internals](https://github.com/ivanvmoreno/correctness-model-internals)  
**Area**: LLM Reasoning  
**Keywords**: Linear Probes, Correctness Direction, LLM Internal Representations, Self-Assessment, Linear Representation Hypothesis, Confidence

## TL;DR

Prior to answer generation, a linear probe (difference-of-means) trained solely on residual stream activations at the question-processing stage can predict whether a model's forthcoming answer will be correct. This "pre-generation correctness direction," trained on TriviaQA, generalizes across multiple factual knowledge datasets (AUROC 0.68–0.88) but fails to generalize to mathematical reasoning (GSM8K), revealing a structural separation between representations of *factual correctness* and *reasoning correctness* within the model's internals.

## Background & Motivation

### Linear Representation Hypothesis

Prior work has demonstrated that LLM internal activations encode information beyond what is observable in the output: statement truthfulness, deceptive behavior, and hallucinations can all be detected via linear probes. This paper extends the paradigm to **self-correctness prediction**—whether a model "knows" it is about to answer correctly or incorrectly.

### Key Distinctions from Prior Work

**Pre-generation rather than post-hoc**: Predictions are made before any token is generated, without requiring the full answer.

**Free-form question answering**: Not restricted to multiple-choice; applicable to open-ended QA.

**Simple linear probes**: Uses difference-of-means directions rather than complex nonlinear models, with the aim of verifying linear separability.

**Cross-domain generalization**: The primary goal is not to maximize prediction accuracy, but to verify whether correctness exists as a **unified linear feature direction**.

### Comparison with Confidence Estimation Methods

- Token-level logits and self-verbalization (asking the model to state its confidence) depend on model generation.
- External assessors rely on model-agnostic input features (e.g., question embeddings).
- The proposed method directly exploits the model's internal states, occupying a position between these two paradigms.

## Method

### Overall Architecture

1. Given a question $x$, extract residual stream activations $h^{(l)}(x)$ (last token of the prompt, layer $l$).
2. The model generates an answer $y$ (temperature=0); correctness is evaluated.
3. A linear classifier is learned: $f_w(h^{(l)}(x)) \approx \mathbf{1}\{\text{Correct}(x, M(x))\}$.

### Learning the Correctness Direction

Activations are grouped by correctness, and centroids are computed for each group:

$$\mu_{\text{true}} = \frac{1}{|\mathcal{D}_{\text{correct}}|} \sum_{x \in \mathcal{D}_{\text{correct}}} h^{(l)}(x), \quad \mu_{\text{false}} = \frac{1}{|\mathcal{D}_{\text{incorrect}}|} \sum_{x \in \mathcal{D}_{\text{incorrect}}} h^{(l)}(x)$$

Correctness direction: $w = \mu_{\text{true}} - \mu_{\text{false}}$

Correctness score:

$$\text{score}(h) = \frac{(h - \mu)^\top w}{\|w\|}$$

where $\mu = \frac{1}{2}(\mu_{\text{false}} + \mu_{\text{true}})$. Discriminative power is evaluated directly via AUROC, without threshold selection.

### Optimal Layer Selection

3-fold cross-validation is performed across all layers of each model on TriviaQA:
- Early layers perform poorly.
- **Middle layers** (roughly from the midpoint to the later portion of model depth) reach saturation.
- The selected optimal layer is fixed for all subsequent evaluations.

### Key Designs

- **Highly efficient training**: A single computation of $d$-dimensional mean vectors; <3 minutes on CPU.
- **No sigmoid or threshold**: Scores are kept as continuous values and evaluated via AUROC.
- **3-shot prompting**: Reduces formatting errors; few-shot examples have no significant effect on performance.

## Key Experimental Results

### Experimental Setup

- **6 models**: Llama 3.1 8B, Llama 3.3 70B Instruct, Qwen 2.5 7B, DeepSeek R1 Distill Qwen 32B, Mistral 7B v0.3, Ministral 8B.
- **6 datasets**: TriviaQA (60K), Cities (10K), Notable People (16K), Medals (9K), Math Operations (6K), GSM8K (8K).
- All datasets use **open-ended QA format**; no multiple-choice.

### Main Results: Cross-Domain Generalization AUROC

All directions are trained on TriviaQA and evaluated on each dataset:

| Model | TriviaQA | N.People | Cities | Math Ops | Medals | GSM8K |
|-------|----------|----------|--------|----------|--------|-------|
| Llama 3.1 8B — Assessor | 0.852 | 0.630 | 0.663 | 0.528 | 0.623 | 0.558 |
| Llama 3.1 8B — Verb.Conf | 0.502 | 0.499 | 0.500 | 0.623 | 0.500 | 0.540 |
| **Llama 3.1 8B — Direction** | 0.804 | **0.722** | **0.732** | **0.858** | **0.680** | 0.534 |
| Llama 3.3 70B — Assessor | 0.759 | 0.583 | 0.672 | 0.449 | 0.568 | 0.573 |
| **Llama 3.3 70B — Direction** | **0.826** | **0.708** | **0.880** | **0.835** | **0.770** | 0.499 |
| Qwen 2.5 7B — Assessor | 0.807 | 0.723 | 0.708 | 0.400 | 0.622 | 0.584 |
| **Qwen 2.5 7B — Direction** | 0.758 | **0.800** | **0.842** | **0.837** | 0.586 | 0.601 |
| Mistral 7B — Assessor | 0.846 | 0.673 | 0.710 | 0.493 | 0.638 | 0.559 |
| **Mistral 7B — Direction** | 0.796 | **0.760** | **0.880** | **0.782** | **0.645** | 0.579 |

Core observations:
- The Direction method outperforms the Assessor and Verbalized Confidence baselines on nearly all OOD datasets.
- **All methods approach random chance on GSM8K** (~0.5)—the factual correctness direction does not transfer to mathematical reasoning.
- The largest model (70B) exhibits the greatest advantage on difficult datasets such as Medals.

### Sample Efficiency Analysis

| Training Samples | Average AUROC |
|-----------------|---------------|
| 160 | Already robust performance |
| 2,560 | Matches full 48,540-sample result |
| 48,540 (full) | Only marginal improvement |

The high sample efficiency provides strong support for the linear representation hypothesis. Larger models require fewer samples to converge.

### Ablation Study: Cross-Dataset Direction Transfer

Directions trained on different datasets are tested on held-out datasets:
- The TriviaQA direction generalizes most strongly across domains due to its diversity, which mitigates dataset-specific patterns.
- Directions trained on smaller datasets sometimes transfer across domains, but inconsistently.
- Cosine similarities between directions from different datasets are largely near zero, with a few exceptions (Cities and Notable People are relatively aligned, and both align with the TriviaQA direction).

### "I Don't Know" Behavior Analysis

Some models produce IDK responses despite being prompted to answer. These responses are located at the **extreme negative end of the correctness direction**:

- Correctness scores: IDK responses < incorrect answers < correct answers.

This indicates that the correctness direction also serves as a **confidence axis**: models choose to abstain only when their internal state reflects extremely low confidence.

### Qualitative Analysis of Extremes (Mistral 7B, Notable People)

| Type | Low Score | High Score |
|------|-----------|------------|
| Incorrect answers | IDK responses / highly deviant answers | Near-miss errors off by only 1–2 years |
| Correct answers | Lesser-known individuals | Charles Darwin (1809), Albert Einstein (1879) |

High-confidence correct answers correspond to highly prominent figures, which is intuitively consistent.

### Key Findings

1. **Linear separability confirmed**: LLMs do encode an **anticipatory correctness signal** in intermediate layers.
2. **Structural separation of factual vs. reasoning correctness**: Factual retrieval and arithmetic reasoning likely rely on **distinct internal verification mechanisms**.
3. **Scale effect**: The 70B model exhibits the strongest and most consistent correctness signal.
4. **Confidence–abstention alignment**: The correctness direction strongly correlates with the model's spontaneous abstention behavior.

## Highlights & Insights

1. **Profound findings from a minimal method**: Using only difference-of-means (no trainable parameters), this work reveals the internal mechanism underlying LLM self-assessment.
2. **Strong evidence for the linear representation hypothesis**: Correctness demonstrably exists as a linear direction in activation space.
3. **Factual vs. reasoning dichotomy**: This is an important negative result—it implies that a single "knowing" dimension is insufficient, and that different types of knowledge require distinct representations.
4. **Practical safety value**: A low-cost internal failure warning signal applicable to early stopping, fallback mechanisms, or human-AI collaboration.
5. **Remarkable sample efficiency**: A robust correctness direction can be obtained with as few as 160 samples.

## Limitations & Future Work

1. **Binary correctness labels**: Answer ambiguity and partial correctness are not accounted for.
2. **Linear probes may underestimate predictive power**: Nonlinear classifiers may reveal richer signals.
3. **Limited model diversity**: 6 models, only one at the 70B scale; MoE and closed-source models are not covered.
4. **Optimal layer selection based on a single dataset** (TriviaQA): May not capture the globally optimal layer for all models.
5. **Temperature fixed at 0**: Correctness uncertainty arising from generation stochasticity is not considered.

## Related Work & Insights

- **Burns et al. (2022)**: The CCS method probes for truthfulness directions; this paper extends from truthfulness to **self-correctness**.
- **Burger et al. (2024)**: A similar difference-of-means approach applied to statement truthfulness; this paper applies it to the pre-generation stage.
- **Ferrando et al. (2025)**: Uses SAE latents to distinguish correct from incorrect responses, but is limited to small Gemma models.
- **Kadavath et al. (2022)**: Tested similar probes on older proprietary models without open-sourcing.
- **Insight**: Combining correctness directions with other internal signals (e.g., probes on intermediate steps of reasoning chains) could yield a more comprehensive internal uncertainty estimation framework.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Pre-generation correctness prediction is an important and novel angle.
- **Technical Depth**: ⭐⭐⭐ — The method is intentionally minimal, but deeper theoretical explanation is lacking.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 6 models × 6 datasets, with multiple baselines and qualitative analysis.
- **Value**: ⭐⭐⭐⭐ — Low-cost failure detection has direct deployment value.
- **Overall Recommendation**: ⭐⭐⭐⭐ — A concise and compelling set of findings; the factual/reasoning correctness separation carries significant implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Predicting LLM Reasoning Performance with Small Proxy Models](predicting_llm_reasoning_performance_with_small_proxy_models.md)
- [\[ICLR 2026\] Predicting LLM Reasoning Performance with Small Proxy Model](predicting_llm_reasoning_performance_with_small_proxy_model.md)
- [\[ICLR 2026\] When Shallow Wins: Silent Failures and the Depth-Accuracy Paradox in Latent Reasoning](when_shallow_wins_silent_failures_and_the_depth-accuracy_paradox_in_latent_reaso.md)
- [\[ICLR 2026\] Harder Is Better: Boosting Mathematical Reasoning via Difficulty-Aware GRPO and Multi-Aspect Question Reformulation](harder_is_better_boosting_mathematical_reasoning_via_difficulty-aware_grpo_and_m.md)
- [\[ACL 2026\] Self-Consistency from Only Two Samples: CoT-PoT Ensembling for Efficient LLM Reasoning](../../ACL2026/llm_reasoning/self-consistency_from_only_two_samples_cot-pot_ensembling_for_efficient_llm_reas.md)

</div>

<!-- RELATED:END -->
