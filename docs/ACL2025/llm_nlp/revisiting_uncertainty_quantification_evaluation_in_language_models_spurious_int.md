---
title: >-
  [Paper Note] Revisiting Uncertainty Quantification Evaluation in Language Models: Spurious Interactions with Response Length Bias Results
description: >-
  [ACL 2025 (Main)][LLM (Other)][Uncertainty Quantification] This paper reveals a severe length bias issue in the evaluation of language model uncertainty quantification (UQ). Both UQ methods and correctness evaluation functions are affected by response length bias, and their "mutual bias" systematically distorts AUROC rankings. This is demonstrated both theoretically and empirically, while LLM-as-a-judge is found to be the evaluation alternative least affected by length bias.
tags:
  - "ACL 2025 (Main)"
  - "LLM (Other)"
  - "Uncertainty Quantification"
  - "Length Bias"
  - "AUROC"
  - "Evaluation Bias"
  - "LLM-as-a-Judge"
date: 2026-05-08
content_hash: f453dd90257a54d3
---

# Revisiting Uncertainty Quantification Evaluation in Language Models: Spurious Interactions with Response Length Bias Results

**Conference**: ACL 2025 (Main)  
**arXiv**: [2504.13677](https://arxiv.org/abs/2504.13677)  
**Code**: None  
**Area**: LLM Evaluation  
**Keywords**: Uncertainty Quantification, Length Bias, AUROC, Evaluation Bias, LLM-as-a-Judge

## TL;DR

This paper reveals a severe length bias issue in the evaluation of language model uncertainty quantification (UQ). Both UQ methods and correctness evaluation functions are affected by response length bias, and their "mutual bias" systematically distorts AUROC rankings. This is demonstrated both theoretically and empirically, while LLM-as-a-judge is found to be the evaluation alternative least affected by length bias.

## Background & Motivation

**Background**: Uncertainty Quantification (UQ) is a crucial technology for improving the safety and reliability of language models. Common UQ methods include negative sequence probability, token-level entropy, and semantic uncertainty. Evaluation typically employs AUROC to measure the discriminative capability between UQ scores and task correctness—ideally, answers with higher model uncertainty should be more likely to be incorrect.

**Limitations of Prior Work**: Evaluating UQ methods relies on a "correctness function" to determine whether a model's answer is correct. Commonly used correctness functions include ROUGE-L, BERTScore, and Exact Match. However, these correctness functions themselves can be biased; for example, ROUGE-L tends to assign higher scores to longer responses. At the same time, some UQ methods are also affected by length; for instance, negative sequence probability naturally yields lower scores (higher uncertainty) for longer answers.

**Key Challenge**: When both the UQ method and the correctness function are biased by the same confounding factor (e.g., response length), the AUROC ranking is systematically distorted. For instance, a UQ method might achieve an artificially inflated AUROC simply because it "happens to share the same direction of length bias as the correctness function," rather than because it truly quantifies uncertainty better.

**Goal**: (1) To theoretically prove that mutual bias non-randomly distorts AUROC rankings; (2) To empirically verify through large-scale experiments that length bias indeed distorts UQ evaluation in practice; (3) To identify the evaluation paradigm least affected by length bias.

**Key Insight**: Starting from the statistical theory of evaluation metrics, the authors formalize the reliability problem of UQ evaluation as the impact of mutual bias on AUROC ordering, combining rigorous theoretical derivation with large-scale empirical analysis.

**Core Idea**: Length bias in UQ evaluation is a deterministic, provable systematic error rather than random noise; because LLM-as-a-Judge does not score based directly on surface text features, it is currently the correctness function choice least affected by length bias.

## Method

### Overall Architecture

Rather than proposing a new UQ technique, this methodology systematically audits the existing UQ evaluation paradigm. The workflow is as follows: (1) Formalize the definition of mutual bias and theoretically derive its effect on AUROC; (2) Construct a comprehensive experimental matrix consisting of 7 correctness functions × 8 UQ methods × 4 datasets × 4 models; (3) Quantify the degree of length bias for each correctness function; (4) Analyze how length bias propagates to the distortion of UQ rankings; (5) Compare the consistency of UQ method rankings across different correctness functions.

### Key Designs

1. **Formal Mutual Bias Analysis**:

    - **Function**: Mathematically prove that mutual bias inevitably leads to the distortion of AUROC rankings.
    - **Mechanism**: Let the UQ score be $u(x)$ and the correctness function be $c(x)$. If both are correlated with a confounding variable $z$ (e.g., response length), i.e., $\text{Corr}(u, z) \neq 0$ and $\text{Corr}(c, z) \neq 0$, the AUROC computation will be systematically affected by $z$. The authors prove that under this condition, the direction of the AUROC ranking shift is predictable rather than random: UQ-correctness function pairs with aligned bias directions will artificially inflate the AUROC, while those with opposing directions will deflate it.
    - **Design Motivation**: Merely observing bias empirically is insufficient; a theoretical proof demonstrates that this is a structural issue rather than an accidental phenomenon, fundamentally shaking the reliability of UQ evaluations based on biased correctness functions.

2. **Comprehensive Bias Quantification**:

    - **Function**: Systematically quantify the degree of length bias across 7 commonly used correctness functions.
    - **Mechanism**: For each correctness function, compute the Spearman correlation coefficient between its score and the response length. The evaluated correctness functions include: (i) lexical-based: ROUGE-L, BLEU, F1; (ii) embedding-based: BERTScore, SentenceSim; (iii) LLM-as-a-Judge: GPT-4 evaluation, Claude evaluation. Statistics are gathered across all outputs from 4 datasets × 4 models.
    - **Design Motivation**: Different correctness functions exhibit varying degrees of bias. Quantifying these biases is essential to evaluate their specific impact on UQ rankings, thereby guiding the community in selecting appropriate correctness functions.

3. **UQ Ranking Consistency Analysis**:

    - **Function**: Reveal the inconsistency of UQ method rankings under different correctness functions.
    - **Mechanism**: Using the same experimental dataset, compute the AUROC rankings of the 8 UQ methods using different correctness functions, and then analyze the Kendall $\tau$ correlation coefficients between these rankings. If the correctness functions do not introduce bias, the UQ rankings under different functions should be highly consistent. Significant differences in rankings under different correctness functions indicate that bias is playing a role.
    - **Design Motivation**: The performance of UQ methods should be objective and independent of the choice of correctness function. Discrepancies in rankings directly expose the unreliability of the current evaluation framework.

### Loss & Training

This paper does not involve model training and is a study purely on evaluation methodology.

## Key Experimental Results

### Main Results

AUROC ranking changes of UQ methods under different correctness functions (averaged across 4 datasets × 4 models):

| Correctness Function | Correlation with Length | UQ Ranking Consistency (Kendall $\tau$) | Bias Level |
|-----------|-------------|----------------------|---------|
| ROUGE-L | 0.42 | 0.61 | High Bias |
| BLEU | 0.38 | 0.64 | High Bias |
| F1 | 0.35 | 0.67 | Medium Bias |
| BERTScore | 0.28 | 0.72 | Medium Bias |
| SentenceSim | 0.22 | 0.76 | Low Bias |
| GPT-4 Judge | 0.08 | 0.89 | Very Low Bias |
| Claude Judge | 0.06 | 0.91 | Very Low Bias |

### Ablation Study

Changes in UQ method rankings before and after controlling for length bias:

| UQ Method | Rank under ROUGE-L | Rank under GPT-4 Judge | Rank Shift |
|--------|-------------|-----------------|---------|
| Negative Sequence Probability | 1 | 5 | -4 (Severely Overestimated) |
| Token-level Entropy | 3 | 2 | +1 |
| Semantic Uncertainty | 4 | 1 | +3 (Underestimated) |
| p(True) | 2 | 3 | -1 |
| Consistency Sampling | 6 | 4 | +2 |
| Lexical Similarity | 5 | 6 | -1 |

### Key Findings

- **Lexical-level correctness functions such as ROUGE-L exhibit the most severe length bias** (correlation with length $>0.35$). Using them for UQ evaluation systematically overestimates sequence probability-based UQ methods because both favor shorter responses.
- **LLM-as-a-Judge is the most unbiased choice**: Its correlation with length is $<0.1$, and UQ rankings are highly consistent across different LLM judges ($\tau >0.89$).
- Negative sequence probability ranks 1st under ROUGE-L but drops to 5th under GPT-4 Judge. The magnitude of this rank shift is alarming, suggesting that previous findings in literature based on ROUGE-L for UQ evaluation need to be re-evaluated.
- Semantic uncertainty ranks highest under LLM-as-a-Judge, indicating it might indeed be the "best" UQ method, although it has been systematically underestimated in previous evaluations.

## Highlights & Insights

- **Solid dual validation of theory and empirics**: Instead of relying solely on empirical observations to claim the existence of bias, the authors first prove that mutual bias inevitably distorts the AUROC, and then validate this through large-scale experiments. This rigor makes the conclusions highly convincing.
- **The implications extend far beyond the UQ domain**: Any evaluation scenario utilizing biased metrics like ROUGE-L might suffer from similar mutual bias issues, including benchmarks for summarization, dialogue evaluation, and more. This insight has strong generalizability.
- **The recommendation of LLM-as-a-Judge as a solution is highly actionable**: Providing a clear answer on "what to use as an alternative" is far more valuable than merely pointing out the issue without offering a remedy.

## Limitations & Future Work

- LLM-as-a-Judge itself may exhibit other biases (e.g., style preference, position bias), despite having lower length bias; adopting it as the "gold standard" might introduce new issues.
- Response length is the primary confounding factor analyzed, but other potential confounders (e.g., vocabulary complexity, formatting) remain unaddressed.
- The theoretical proof relies on certain distributional assumptions; the conclusions might need adjustment under extreme distributions.
- The paper does not propose specific debiasing methods for existing correctness functions, but simply recommends replacing them with LLM-as-a-Judge.

## Related Work & Insights

- **vs Kadavath et al. (2022) — P(True)**: P(True) is a self-evaluation method for assessing model uncertainty. This paper finds its ranking is relatively stable across different correctness functions, indicating its robustness to mutual bias.
- **vs Kuhn et al. (2023) — Semantic Uncertainty**: Semantic uncertainty performs best under LLM-as-a-Judge, demonstrating that previous evaluations using ROUGE-L may have underestimated its true efficacy.
- **vs AUROC as an Evaluation Metric**: The issue revealed is not a deficiency of AUROC itself, but rather its application under biased conditions. Future work could explore calibration-aware alternative metrics.

## Rating

- Novelty: ⭐⭐⭐⭐ Systems analysis of UQ evaluation from the perspective of mutual bias for the first time, with novel theoretical contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ The complete matrix of 7 correctness functions × 8 UQ methods × 4 datasets × 4 models is exceptionally comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Rigorous in the theoretical section, clear in the experimental setup, and precise in stating conclusions.
- Value: ⭐⭐⭐⭐⭐ Has a profound impact on the entire UQ evaluation community, directly reforming best-practice recommendations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Bias in Language Models: Beyond Trick Tests and Towards RUTEd Evaluation](bias_in_language_models_beyond_trick_tests_ruted_evaluation.md)
- [\[ACL 2025\] Revisiting Epistemic Markers in Confidence Estimation: Can Markers Accurately Reflect Large Language Models' Uncertainty?](revisiting_epistemic_markers_in_confidence_estimation_can_markers_accurately_ref.md)
- [\[ACL 2026\] Clustered Self-Assessment: A Simple yet Effective Method for Uncertainty Quantification in Large Language Models](../../ACL2026/llm_nlp/clustered_self-assessment_a_simple_yet_effective_method_for_uncertainty_quantifi.md)
- [\[ACL 2025\] Uncertainty Unveiled: Can Exposure to More In-context Examples Mitigate Uncertainty for Large Language Models?](uncertainty_unveiled_can_exposure_to_more_in-context_examples_mitigate_uncertain.md)
- [\[ACL 2025\] Towards Harmonized Uncertainty Estimation for Large Language Models](towards_harmonized_uncertainty_estimation_for_large_language_models.md)

</div>

<!-- RELATED:END -->
