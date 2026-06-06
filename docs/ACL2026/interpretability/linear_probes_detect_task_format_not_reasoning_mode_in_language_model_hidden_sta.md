---
title: >-
  [Paper Note] Linear Probes Detect Task Format, Not Reasoning Mode in Language Model Hidden States
description: >-
  [ACL 2026][Interpretability][Linear Probes] This paper demonstrates through probes on Qwen3-14B, residual deconfounding, trace-anchor…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Linear Probes"
  - "Format Confounding"
  - "Representational Geometry"
  - "Causal Turn"
  - "Reasoning Mode"
date: 2026-05-08
content_hash: 85e0400bca657cc7
---

# Linear Probes Detect Task Format, Not Reasoning Mode in Language Model Hidden States

**Conference**: ACL 2026  
**arXiv**: [2606.02907](https://arxiv.org/abs/2606.02907)  
**Code**: https://github.com/SubramanyamSahoo/Linear-Probes-Detect-Task-Format-Not-Reasoning-Mode  
**Area**: Interpretability / Linear Probes / LLM Reasoning Analysis  
**Keywords**: Linear Probes, Format Confounding, Representational Geometry, Causal Turn, Reasoning Mode

## TL;DR
This paper demonstrates through probes on Qwen3-14B, residual deconfounding, trace-anchor, and causal steering experiments that while linear probes seemingly distinguish deductive, inductive, and abductive reasoning with 100% accuracy, they actually detect data sources and task formats rather than reasoning modes in hidden states.

## Background & Motivation
**Background**: Linear probes are widely used in mechanistic interpretability to determine whether model hidden states encode specific attributes. If a probe predicts "reasoning type" with high accuracy, many works interpret this as the formation of distinct reasoning circuits or mode-specific representations within the model.

**Limitations of Prior Work**: Reasoning mode evaluations often map different benchmarks directly to specific reasoning types—for instance, LogiQA for deductive, ARC-Challenge for inductive, and $\alpha$NLI for abductive reasoning. This setup binds the reasoning label to surface formats such as dataset source, number of options, prompt style, and output length. High probe accuracy may simply reflect the identification of the dataset source.

**Key Challenge**: While hidden states do contain significant linearly separable information, "predictability" does not equate to "causal involvement in reasoning." Without controlling for format confounding, it is impossible to determine whether a model truly employs three distinct reasoning mechanisms or has merely learned the distributional differences of the input sources.

**Goal**: The authors aim to perform a systematic stress test on reasoning-mode probing: first replicating high probe accuracy and clean geometric separation, then progressively removing format information to verify whether this geometry corresponds to actual reasoning modes through behavioral and causal experiments.

**Key Insight**: The paper deliberately adopts a standard multi-source reasoning setup to construct a balanced three-way classification dataset, then poses a straightforward question: how much of the original 100% probe signal remains once source identity, answer option count, and response length are controlled?

**Core Idea**: To shift the evidentiary standard for linear probing from "high accuracy implies internal structure" back to "high accuracy must first pass format deconfounding and random-direction causal controls."

## Method
The methodology does not propose a new model but rather an experimental pipeline for the adversarial validation of probing conclusions. It first allows traditional probes to achieve strong surface results, then utilizes residual analysis, trace agreement, and causal steering to scrutinize these results from representational, behavioral, and causal perspectives.

### Overall Architecture
The dataset comprises 750 samples, with 250 per category: LogiQA 2.0 represents deductive, ARC-Challenge represents inductive, and $\alpha$NLI represents abductive reasoning. The model used is Qwen3-14B (40 layers, hidden dimension 5120, bfloat16). The authors unify the prompts and set `DISABLE_THINKING=True` during inference to remove thinking blocks, preventing the thinking mode's linguistic style from acting as an additional confounded.

For each sample, hidden states from the last input token across all layers, generated text, and output confidence are extracted; geometric analysis is restricted to correctly answered samples. Subsequently, L2-regularized logistic regression linear probes are trained at each layer using 5-fold stratified cross-validation to predict reasoning-mode labels, followed by manifold geometry analysis at the optimal layer. Finally, format residuals, trace-anchor similarity, and activation steering are used to test the functional relevance of the geometry.

### Key Designs
1.  **Multi-source Reasoning Probes and Geometric Analysis**:
    - **Function**: Replicates the surface phenomenon where hidden states perfectly distinguish reasoning modes.
    - **Mechanism**: Linear probes are trained to predict the three classes (D/I/A), achieving 100% balanced accuracy at layer 32; simultaneously, intrinsic dimensionality, local curvature, inter-mode separation, and hull contamination are calculated.
    - **Design Motivation**: Rebuttal is only persuasive if standard probes first achieve strong results. The paper rejects the probing conclusion not because of poor performance, but because the high performance is fully explained by confounding factors.

2.  **Four-stage Format Confounding Deconstruction**:
    - **Function**: Distinguishes whether the probe is identifying reasoning mode or task format.
    - **Mechanism**: The same probe is used to predict dataset source; then only option count is used for prediction; then analysis is restricted to 4-choice samples from LogiQA and ARC; finally, a format feature vector $f_i = [source\ one\text{-}hot, n_{options}, |y_i|]$ is constructed to regress out format info from hidden states via Ridge regression, yielding residuals $r_i = h_i - \hat{h}_i$ for re-probing.
    - **Design Motivation**: Source, option count, and response length are the most common yet overlooked confounding variables in multi-source benchmarks. Residual analysis provides a rigorous counterfactual test.

3.  **Behavioral and Causal Random Control**:
    - **Function**: Validates whether geometric directions truly shift reasoning modes rather than any perturbation producing similar effects.
    - **Mechanism**: Trace-anchor analysis computes cosine similarity between generated traces and three anchor descriptions to see if overt reasoning shifts; steering experiments construct directions via mode centroid differences and compare them against $N_{rand}=20$ random directions, with empirical $p$-values calculated via Laplace correction.
    - **Design Motivation**: Even if geometric separation exists, it must be functionally linked to behavior. Random direction control is crucial to avoid misinterpreting general perturbations as mode-specific steering.

### Loss & Training
The paper does not train the language model; it trains linear probes for analysis. Probes are logistic regression models with L2 regularization ($C=1.0$) and 5-fold cross-validation. The steering intensity $\alpha^*$ is determined via coherence sweep and Otsu thresholding rather than manual tuning; the number of random directions is set to 20.

## Key Experimental Results

### Main Results
The original probe results are extremely strong, but after deconfounding format via residuals, they collapse to near-random levels. This constitutes the core evidence of the paper.

| Target of Analysis | Original Hidden States | De-formatted Residual States | Chance Level |
| :--- | :--- | :--- | :--- |
| Reasoning Mode (D/I/A) Probe | 100.0% | ~33.5% | 33.3% |
| Dataset Source Probe | 100.0% | ~33.5% | 33.3% |
| Optimal Layer | Layer 32 | Layer 32 (Residual) | N/A |
| Original Geometric Phenomenon | Clean clusters | No mode/source separability | N/A |

### Ablation Study
The four-stage confounding analysis shows how the original "reasoning mode" separation is gradually explained away by format.

| Stage | Setting | Result | Interpretation |
| :--- | :--- | :--- | :--- |
| Stage 1 | Predict source using the same linear probe | 100% accuracy | Mode and source labels are informationally equivalent |
| Stage 2 | Use only option count (2 vs. 4) | 33.3% mode accuracy | Option count directly identifies the $\alpha$NLI prior |
| Stage 3 | Retain only 4-choice LogiQA + ARC | Near perfect | Lexical and task styles remain separable despite identical option counts |
| Stage 4 | Regress out source one-hot, option count, and response length | ~33.5% | Original linear separation is primarily format-driven |

### Key Findings
- Probes achieve 100% balanced accuracy at layer 32 with perfect precision, recall, and F1; permutation tests show $p < 0.0002$.
- The geometry at layer 32 appears convincing: intrinsic dimensionality for deductive, inductive, and abductive modes are 20.6, 28.5, and 33.6, respectively, with convex hull contamination below 1.5%.
- Behaviorally, Qwen3-14B achieves 86.0% accuracy across tasks, but trace-mode agreement is only 42.5%, barely above the 33.3% chance level.
- Source accuracy varies significantly (LogiQA 2.0 at 73.2%, ARC-Challenge at 93.6%, $\alpha$NLI at 91.2%), suggesting clear differences in task difficulty and source distribution.
- In steering experiments, targeted steering accuracy recovery is 40.0% vs. a random direction mean of 31.7%; empirical $p=0.286$ and Cohen's $d < 0.5$ indicate that the centroid direction lack significant mode-specific causal effects.

## Highlights & Insights
- The strength of the paper lies in moving beyond a general warning about confounding to constructing a complete chain of counter-evidence: linking original probes, source probes, residual probes, trace behavior, and causal steering.
- Setting `DISABLE_THINKING=True` is a subtle but vital detail. it prevents the linguistic style of thinking blocks from further externalizing "reasoning modes," focusing the experiment on whether the input format itself causes the separation.
- The paper serves as a practical warning for interpretability research: if labels correspond 1-to-1 with data sources, high probe accuracy only indicates that hidden states preserve source information, not that they encapsulate the target concept.
- The use of random direction steering controls should be widely adopted. Many steering works only report that the target direction is effective without proving it is more specialized than random perturbations.

## Limitations & Future Work
- Experiments only cover Qwen3-14B. While format confounding stems from experimental design rather than the model, the "unified reasoning strategy" conclusion needs replication on Llama, Mistral, and GPT series.
- Residual analysis is conservative; regressing out source one-hots might simultaneously remove some genuine reasoning signals. Future work could regress only option count and response length.
- Trace-anchor analysis relies on handwritten descriptions and cosine similarity, which might miss fine-grained reasoning differences. Stronger behavioral analysis could track explicit operations like syllogisms or hypothesis elimination.
- Structural differences (e.g., $\alpha$NLI is 2-choice while others are 4-choice) are prominent. Future reasoning mode benchmarks should unify option counts, prompt templates, and output formats.

## Related Work & Insights
- **vs. Traditional Linear Probes**: While traditional probes emphasize linear readability, this work emphasizes that probes must pass deconfounding and causal controls to prove more than mere correlation.
- **vs. Reasoning Circuit Work**: If reasoning mode labels are derived from different benchmarks, circuit explanations might be tracking benchmark style rather than the latent reasoning algorithm.
- **vs. Activation Steering**: This work does not propose stronger steering but argues that target directions must be contrasted with random directions to discuss causal roles.
- **Inspiration for Future Work**: When constructing datasets for reasoning interpretability, sub-types should be labeled within the same data source/format, or paired samples should be used to maintain surface consistency.

## Rating
- Novelty: ⭐⭐⭐⭐ (Not a new model, but a very complete probe validation pipeline.)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Robust evidence across representation, behavior, and causality, though model coverage is limited.)
- Writing Quality: ⭐⭐⭐⭐⭐ (Defines clear problems and maintains logical clarity.)
- Value: ⭐⭐⭐⭐⭐ (High methodological value for both interpretability and reasoning benchmark design.)

## Related Papers

- [\[ICLR 2026\] Beyond Linear Probes: Dynamic Safety Monitoring for Language Models](../../ICLR2026/interpretability/beyond_linear_probes_dynamic_safety_monitoring_for_language_models.md)
- [\[ICLR 2026\] Hidden Breakthroughs in Language Model Training](../../ICLR2026/interpretability/hidden_breakthroughs_in_language_model_training.md)
- [\[ACL 2026\] AdaptiveK: Complexity-Driven Sparse Autoencoders for Interpretable Language Model Representations](adaptivek_complexity-driven_sparse_autoencoders_for_interpretable_language_model.md)
- [\[ACL 2026\] Rhetorical Questions in LLM Representations: A Linear Probing Study](rhetorical_questions_in_llm_representations_a_linear_probing_study.md)
- [\[ACL 2026\] Knowledge Vector of Logical Reasoning in Large Language Models](knowledge_vector_of_logical_reasoning_in_large_language_models.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Beyond Linear Probes: Dynamic Safety Monitoring for Language Models](../../ICLR2026/interpretability/beyond_linear_probes_dynamic_safety_monitoring_for_language_models.md)
- [\[ICLR 2026\] Hidden Breakthroughs in Language Model Training](../../ICLR2026/interpretability/hidden_breakthroughs_in_language_model_training.md)
- [\[ACL 2026\] Knowledge Vector of Logical Reasoning in Large Language Models](knowledge_vector_of_logical_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Dual Alignment Between Language Model Layers and Human Sentence Processing](dual_alignment_between_language_model_layers_and_human_sentence_processing.md)
- [\[ACL 2026\] AdaptiveK: Complexity-Driven Sparse Autoencoders for Interpretable Language Model Representations](adaptivek_complexity-driven_sparse_autoencoders_for_interpretable_language_model.md)

</div>

<!-- RELATED:END -->
