---
title: >-
  [Paper Note] The Trilemma of Truth in Large Language Models
description: >-
  [NeurIPS 2025][veracity probing] This paper proposes sAwMIL (Sparse-Aware Multiple Instance Learning), a three-class probing framework that combines MIL and conformal prediction to classify LLM internal activations into true/false/neither, revealing that truth and falsity signals are not encoded as simple bidirectional opposites but as distributed representations spanning a multi-dimensional subspace.
tags:
  - NeurIPS 2025
  - veracity probing
  - multiple-instance learning
  - conformal prediction
  - truth direction
  - LLM internals
date: 2026-05-08
content_hash: 186cffdb7285599f
---

# The Trilemma of Truth in Large Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2506.23921](https://arxiv.org/abs/2506.23921)
**Code**: [GitHub](https://github.com/carlomarxdk/trilemma-of-truth)
**Area**: LLM NLP / Interpretability / Veracity Probing
**Keywords**: veracity probing, multiple-instance learning, conformal prediction, truth direction, LLM internals

## TL;DR
This paper proposes sAwMIL (Sparse-Aware Multiple Instance Learning), a three-class probing framework that combines MIL and conformal prediction to classify LLM internal activations into true/false/neither, revealing that truth and falsity signals are not encoded as simple bidirectional opposites but as distributed representations spanning a multi-dimensional subspace.

## Background & Motivation
**Background**: Existing veracity probing methods (e.g., mean-difference, CCS, TTPD) separate "true" and "false" directions in LLM internal activations via linear probes, performing binary classification based on the last-token representation.

**Limitations of Prior Work**:
   - They assume symmetric encoding of truth and falsity ($P(\phi|K_\mathcal{M}) = 1 - P(\neg\phi|K_\mathcal{M})$), an assumption unsupported by empirical evidence.
   - They assume LLMs have knowledge of all facts, ignoring cases where the model simply lacks certain knowledge.
   - Using only the last-token representation discards signals from critical positions within the sentence (e.g., factual actualization points).
   - Output scores are uncalibrated and cannot serve as reliable confidence estimates.
   - Binary classification (true/false) cannot handle cases where the model does not know.

**Key Challenge**: Binary logic fails to accurately characterize the internal knowledge states of LLMs — a model may regard a statement as neither true nor false.

**Key Insight**: Introduce three-valued logic (true/false/neither) and apply a MIL mechanism to attend to the most informative tokens in a sentence, rather than defaulting to the last token.

**Core Idea**: Multiple instance learning enables probes to automatically discover token positions that carry veracity signals within a sentence; conformal prediction is used to quantify uncertainty, yielding a three-class classification scheme.

## Method

### Overall Architecture
The input is the intermediate-layer activation $h_i(\boldsymbol{x}) \in \mathbb{R}^{L \times d}$ (representations of all tokens) obtained by encoding a declarative statement through an LLM; the output is a three-class probability distribution $\{p_{true}, p_{false}, p_{neither}\}$. The pipeline consists of three steps: (1) training three one-vs-all sbMIL probes, (2) integrating their outputs into multi-class probabilities via softmax regression, and (3) calibrating the outputs with conformal prediction.

### Key Designs

1. **Sparse-Aware Multiple Instance Learning (sAwMIL)**:

    - **Function**: Treats the entire sentence as a "bag" and each token's activation as an "instance," automatically identifying which tokens carry veracity signals.
    - **Mechanism**: Two-stage training. In the first stage, MIL-SVM identifies the highest-scoring instance in positive bags and computes an $\eta$-quantile threshold. In the second stage, only tokens above the threshold that belong to the actualized part of the sentence are used to train a standard SVM.
    - **Design Motivation**: Only factually critical tokens (e.g., "Latvia" in "The city of Riga is in Latvia") carry veracity signals; the prefix ("The city of Riga is in") contains no decisive information. MIL automatically localizes these key tokens without requiring manual annotation.

2. **Intra-bag Label Mechanism**:

    - **Function**: Partitions the sentence into a pre-actualized segment $\boldsymbol{x}^p$ (label 0) and an actualized segment $\boldsymbol{x}^a$ (label 1).
    - **Mechanism**: After $\eta$-quantile filtering, intra-bag labels further restrict training to high-scoring tokens within the actualized region only.
    - **Design Motivation**: Prevents MIL from mistakenly treating noisy signals from non-critical positions as veracity signals.

3. **One-vs-All → Multi-class Integration**:

    - Three independent sAwMIL probes are trained: is-true, is-false, and is-neither.
    - Their outputs are integrated via softmax regression: $p_k = \frac{\exp(z_k)}{\sum_j \exp(z_j)}$, where $z_k = g_i^k(\boldsymbol{x}) \cdot \alpha_k + \beta_k$.
    - The final output is a calibrated three-class probability distribution.

4. **Conformal Prediction**:

    - **Function**: Converts raw SVM decision scores into prediction sets with statistical coverage guarantees.
    - **Mechanism**: Constructs nonconformity scores to ensure prediction sets cover the true label with probability $1-\alpha$.
    - **Design Motivation**: SVM decision scores are not calibrated probabilities; direct sigmoid compression is also unreliable.

## Key Experimental Results

### Main Results: Probe Comparison (Average over 16 LLMs and 3 Datasets)

| Method | Type | Correlation MCC | Generalization MCC |
|---|---|---|---|
| Zero-shot prompting | Black-box | ~0.35 | ~0.25 |
| MD+CP (last token) | Binary+CP | ~0.30 | ~0.15 |
| TTPD+CP (last token) | Binary+CP | ~0.32 | ~0.18 |
| sPCA+CP (last token) | Binary+CP | ~0.30 | ~0.17 |
| SVM+CP (last token) | Multi-class | ~0.55 | ~0.30 |
| **sAwMIL (full bag)** | **Multi-class+MIL** | **~0.60** | **~0.45** |

- Binary probes degrade substantially under the full-bag setting, indicating sensitivity to noisy tokens.
- sAwMIL maintains stable or improved performance under the full-bag setting, demonstrating effective noise filtering via MIL.

### Truth–Falsity Direction Analysis

| Metric | SVM | sAwMIL |
|---|---|---|
| Cosine similarity: is-true vs. is-false | ~-0.5 | ~-0.3 |
| Spearman correlation: is-true vs. is-false | ~-0.6 | ~-0.4 |
| Effective rank of direction matrix | 1.73±0.012 | 1.93±0.004 |

- If truth and falsity were strictly symmetric, cosine similarity would approach -1 and effective rank would approach 1.
- The observed effective rank close to 2 demonstrates that truth and falsity directions span a two-dimensional subspace and are not antipodal on a single axis.

### Key Findings
- **Binary probes are unreliable**: Methods such as MD and TTPD perform worse than zero-shot prompting under generalization evaluation.
- **The neither class is critical**: Binary probes frequently misclassify neither statements as true or false with high confidence.
- **Truth–falsity encoding is asymmetric**: The effective rank of 1.93 for sAwMIL indicates that truth and falsity are independently encoded within LLMs.
- **Key token positions matter**: Relying solely on the last token discards signals at factual actualization points.

## Highlights & Insights
- **Introduction of three-valued logic**: Extends LLM veracity probing from binary to ternary classification; the "neither" class elegantly handles cases of model ignorance.
- **Automatic key-token localization via MIL**: No manual annotation of signal-bearing tokens is required — the data speaks for itself — yielding greater robustness than fixed last-token approaches.
- **Systematic critique of five assumptions**: The paper systematically examines assumptions prevalent in the existing literature and provides a corrected formulation for each — a research paradigm transferable to other probing tasks.

## Limitations & Future Work
- Evaluation is limited to binary relational statements (city–country, drug–indication); complex statements involving multiple relations or entities remain untested.
- The neither class is constructed using synthetic entities (e.g., fictitious city names such as "Staakess"); it is unclear whether this genuinely reflects a model's "don't know" state.
- Model scale is restricted to 3–14B parameters; behavior at larger scales (70B+) may differ.
- Probes remain linear; nonlinear probes may capture more complex veracity signals.

## Related Work & Insights
- **vs. Mean-Difference Probe (Marks et al.)**: Uses only the last token with binary classification, cannot handle neither, and generalizes poorly; sAwMIL employs the full sentence with three-class classification and generalizes substantially better.
- **vs. CCS (Burns et al.)**: Employs unsupervised contrastive learning to identify truth–falsity directions, but still assumes binary symmetry; sAwMIL provides direct empirical evidence against this assumption.
- **vs. TTPD (Burger et al.)**: Although multi-directional encodings are observed, the probe remains binary; sAwMIL is the first complete three-class solution.
- Implications for LLM safety and hallucination detection: Real-time detection of "neither" signals in model internals could enable proactive abstention during generation.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of three-valued logic and MIL is novel; the critique of five assumptions is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 16 models × 3 datasets with diverse baselines and in-depth directional analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Argumentation is logically clear; the assumption–critique–solution structure is well-executed.
- Value: ⭐⭐⭐⭐ Makes an important contribution to understanding internal knowledge representations in LLMs, though further validation is needed for practical applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Emergence of Linear Truth Encodings in Language Models](emergence_of_linear_truth_encodings_in_language_models.md)
- [\[NeurIPS 2025\] Table as a Modality for Large Language Models](table_as_a_modality_for_large_language_models.md)
- [\[NeurIPS 2025\] Auditing Meta-Cognitive Hallucinations in Reasoning Large Language Models](auditing_meta-cognitive_hallucinations_in_reasoning_large_language_models.md)
- [\[NeurIPS 2025\] Probabilistic Token Alignment for Large Language Model Fusion](probabilistic_token_alignment_for_large_language_model_fusion.md)
- [\[NeurIPS 2025\] Encoding and Understanding Astrophysical Information in Large Language Model-Generated Summaries](encoding_and_understanding_astrophysical_information_in_large_language_model-gen.md)

</div>

<!-- RELATED:END -->
