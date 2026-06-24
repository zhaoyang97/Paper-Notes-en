---
title: >-
  [Paper Note] Diversity Explains Inference Scaling Laws: Through a Case Study of Minimum Bayes Risk Decoding
description: >-
  [ACL 2025][LLM Pretraining][MBR Decoding] Reinterprets MBR decoding from the theoretical perspective of bias-diversity decomposition: estimation error $MSE = Bias - Diversity$, indicating that increasing diversity (the diversity of pseudo-references) is the key to improving MBR performance. It further extends this to general inference methods through information theory, revealing that diversity is the theoretical root of the inference scaling law (improving performance by inc…
tags:
  - "ACL 2025"
  - "LLM Pretraining"
  - "MBR Decoding"
  - "Bias-Diversity Decomposition"
  - "Inference Scaling Laws"
  - "Ensemble Learning"
  - "information theory"
date: 2026-05-08
content_hash: efd3bbcc232de78e
---

# Diversity Explains Inference Scaling Laws: Through a Case Study of Minimum Bayes Risk Decoding

**Conference**: ACL 2025  
**arXiv**: [2410.15021](https://arxiv.org/abs/2410.15021)  
**Code**: [https://github.com/naist-nlp/mbr-bias-diversity](https://github.com/naist-nlp/mbr-bias-diversity)  
**Area**: LLM Pre-training  
**Keywords**: MBR Decoding, Bias-Diversity Decomposition, Inference Scaling Laws, Ensemble Learning, information theory

## TL;DR
Reinterprets MBR decoding from the theoretical perspective of bias-diversity decomposition: estimation error $MSE = Bias - Diversity$, indicating that increasing diversity (the diversity of pseudo-references) is the key to improving MBR performance. It further extends this to general inference methods through information theory, revealing that diversity is the theoretical root of the inference scaling law (improving performance by increasing samples yields diminishing returns), and empirically validates this on machine translation, summarization, and image captioning tasks.

## Background & Motivation

### Background

**Background**: MBR decoding is a core method to improve generation quality during LLM inference by utilizing multiple sampled candidates. Popular methods like Self-Consistency and Complex SC can be viewed as variants of MBR (Bertsch et al., 2023). In practice, increasing the number of samples continuously improves performance but with diminishing returns — known as the inference scaling law.

**Limitations of Prior Work**: (a) Numerous empirical findings exist for MBR but lack a unified theoretical explanation — e.g., why quality of evaluation metrics matters, why diversity of pseudo-references is crucial, and how the number of samples affects performance. (b) The inference scaling law has only empirical observations without theoretical analysis. (c) The relationship between bias (evaluation metric quality) and diversity (sample diversity) remains unclarified.

**Key Challenge**: Intuitively, both bias (evaluation metrics aligning with humans) and diversity (diverse sampling) should be optimized simultaneously, but a fundamental trade-off exists between them in practice. This work aims to formalize this trade-off.

**Goal**: To provide a unified theoretical framework for MBR decoding and inference scaling laws.

**Key Insight**: Drawing inspiration from the bias-diversity decomposition in ensemble learning (Krogh & Vedelsby, 1994), MBR decoding is formulated as a form of ensemble learning.

**Core Idea**: The estimation error of MBR can be decomposed into $Bias - Diversity$, where the improvement in diversity is identified as the theoretical root of the inference scaling law.

## Method

### Overall Architecture
1. Define the mean squared error (MSE) between MBR estimates and human evaluation.
2. Decompose the MSE into Bias and Diversity terms (Theorem 1).
3. Analyze the Bias-Diversity trade-off (Theorems 2-3).
4. Extend to an information-theoretic framework to explain the inference scaling law (Theorems 4-8).
5. Empirically validate on MT, summarization, and image captioning tasks.

### Key Designs

1. **Bias-Diversity Decomposition (Theorem 1)**:

    - $MSE(\hat{\mathbf{u}}, \bar{\mathbf{u}}) = \underbrace{\mathbb{E}[(\hat{u}_i - f_\theta(h_i, y_j))^2]}_{\text{Bias}} - \underbrace{\mathbb{E}[(\bar{u}_i - f_\theta(h_i, y_j))^2]}_{\text{Diversity}}$
    - Bias: Closeness of the evaluation function to human evaluation (the smaller, the better).
    - Diversity: Variability of evaluation quality yielded by different pseudo-references (the larger, the better, which reduces MSE).
    - Key point: The Diversity term is negative, meaning larger diversity leads to smaller MSE.

2. **Limitations of Diversity (Theorem 2) and Trade-off (Theorem 3)**:

    - Theorem 2: When Bias $\to$ 0, Diversity $\to$ 0 (no benefit can be gained from diversity when the metric is perfect).
    - Theorem 3: Bias and Diversity share a component $\Omega$, leading to a fundamental trade-off.
    - Implication: Lower-quality metrics might actually benefit more from diversity (explaining why BLEU competes with COMET in certain settings).

3. **Information-Theoretic Extension and Inference Scaling Law (Theorems 4-8)**:

    - Theorem 4: Prediction error is bounded from above and below by $H(\hat{\mathcal{H}}) - I(\mathcal{X}_{1:|\mathcal{Y}|}; \hat{\mathcal{H}})$.
    - Maximizing mutual information $I$ requires maximizing Relevancy + Information-Theoretic Diversity.
    - Theorem 7: Under conditional independence of samples given the correct output, $I$ is submodular $\to$ increasing samples yields monotonic improvements but with diminishing returns $\to$ inference scaling law.
    - Theorem 8: Error bounds satisfy supermodularity (non-increasing) $\to$ error decreases and converges.

4. **Interpretation of MBR as Ensemble Learning**:

    - Each pseudo-reference $y_j$ corresponds to an "evaluator" $f_\theta(h, y_j)$.
    - The averaging operation in MBR is equivalent to ensembling multiple evaluators.
    - The Law of Large Numbers explains the effect of increasing pseudo-references.

## Key Experimental Results

### Experimental Setup
- **Tasks**: WMT22 En-De/En-Ja translation, CNN/DailyMail summarization, MS-COCO image captioning.
- **Sampling Methods**: 5 types (epsilon, top-k, top-p, typical, ancestral).
- **Evaluation Metrics / Utility**: MetricX, CometKiwi, BLEURT, COMET, etc.

### Main Empirical Findings

1. **Bias-Diversity Validation**:
    - In machine translation, higher sampling temperatures increase Diversity (high-diversity sampling like ancestral) while also increasing Bias.
    - When COMET serves as the utility, Bias is low and Diversity is also low; MetricX shows slightly higher Bias but also higher Diversity.
    - The final performance is determined by the net effect of $Bias - Diversity$.

2. **Inference Scaling Law Validation**:
    - Increasing sample size monotonically improves COMET/MetricX scores but with diminishing returns (logarithmic convergence).
    - Aligns with the submodularity/supermodularity predictions of Theorems 7-8.

3. **Diversity Model Perturbation Experiment (Section 5.5)**:
    - In addition to changing pseudo-references, diversity can also be increased by perturbing the evaluation model parameters $\theta$.
    - Parameter perturbation is more efficient at increasing diversity than increasing pseudo-references.

### Key Findings
- Diversity is the core driving force behind inference scaling — it is not the sample "quantity" itself, but the sample "diversity".
- The Bias-Diversity trade-off explains why lower-quality metrics sometimes perform exceptionally well.
- Submodularity (diminishing returns) theoretically explains the marginal decrease in inference scaling.

## Highlights & Insights
- **Theoretical Unification**: Eight theorems unify the perspectives of MBR decoding, inference scaling laws, and ensemble learning, consolidating scattered empirical findings into a single framework.
- **Explaining Counter-intuitive Results**: The competitiveness of BLEU vs. COMET and the effectiveness of ancestral sampling are successfully explained by the Bias-Diversity trade-off.
- **Practical Insights**: Perturbing model parameters might be more efficient than increasing samples, pointing towards a new direction for inference-time compute optimization.

## Limitations & Future Work
1. The theoretical analysis is based on MSE as the error metric, whereas practical tasks prioritize ranking rather than absolute values.
2. Submodularity relies on the conditional independence assumption (samples are independent conditioned on the correct output), whereas actual LLM samplings may exhibit correlations.
3. Experimental tasks are skewed towards traditional NLP (translation, summarization) and do not cover reasoning-heavy scenarios like logic/math.
4. The deployment overhead of parameter perturbation to increase diversity (requiring multiple model variants) is not fully discussed.

## Related Work & Insights
- Complementary to the inference scaling work of Snell et al. (2025): while the latter provides empirical observations, this work provides the theoretical foundation.
- Relationship with Self-Consistency (Wang et al., 2023): SC is a special case of MBR, making the theories in this work directly applicable.
- Insights: Future research on inference optimization should focus on "how to maximize diversity rather than purely increasing sample size" — potentially through multi-model ensembles, sampling strategy optimization, or diversity-aware candidate selection.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (first systematic theoretical framework to explain MBR + inference scaling laws)
- Theoretical Depth: ⭐⭐⭐⭐⭐ (8 theorems and formal proofs, solid theoretical contribution)
- Experimental Thoroughness: ⭐⭐⭐⭐ (multi-task validation, but lacks reasoning tasks)
- Value: ⭐⭐⭐⭐ (theories directly guide inference optimization directions)
- Overall Recommendation: ⭐⭐⭐⭐⭐ (a milestone work in the theory of inference scaling)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Scaling Inference-Efficient Language Models](../../ICML2025/llm_pretraining/scaling_inference-efficient_language_models.md)
- [\[ICLR 2026\] How Text Quality Interventions Reshape Neural Scaling Laws for LLMs: Empirical Study](../../ICLR2026/llm_pretraining/how_text_quality_interventions_reshape_neural_scaling_laws_for_llms_empirical_st.md)
- [\[ACL 2025\] Training Dynamics Underlying Language Model Scaling Laws: Loss Deceleration and Zero-Sum Learning](training_dynamics_underlying_language_model_scaling_laws_loss_deceleration_and_z.md)
- [\[NeurIPS 2025\] Conformal Risk Training: End-to-End Optimization of Conformal Risk Control](../../NeurIPS2025/llm_pretraining/conformal_risk_training_end-to-end_optimization_of_conformal_risk_control.md)
- [\[NeurIPS 2025\] Gemstones: A Model Suite for Multi-Faceted Scaling Laws](../../NeurIPS2025/llm_pretraining/gemstones_a_model_suite_for_multi-faceted_scaling_laws.md)

</div>

<!-- RELATED:END -->
