---
title: >-
  [Paper Note] Normalized AOPC: Fixing Misleading Faithfulness Metrics for Feature Attribution Explainability
description: >-
  [ACL 2025][Interpretability][Feature Attribution] This paper reveals that the widely used AOPC (Area Over the Perturbation Curve) faithfulness metric yields misleading conclusions when comparing across different models (due to the vast differences in upper and lower bounds of AOPC for distinct models). It proposes Normalized AOPC (NAOPC) to eliminate inter-model incomparability using min-max normalization. Experiments demonstrate that normalization can fundamentally reverse m…
tags:
  - "ACL 2025"
  - "Interpretability"
  - "Feature Attribution"
  - "AOPC"
  - "Faithfulness Evaluation"
  - "Normalized Metrics"
date: 2026-05-08
content_hash: 955c7dcb69261d10
---

# Normalized AOPC: Fixing Misleading Faithfulness Metrics for Feature Attribution Explainability

**Conference**: ACL 2025  
**arXiv**: [2408.08137](https://arxiv.org/abs/2408.08137)  
**Code**: [https://github.com/JoakimEdin/naopc](https://github.com/JoakimEdin/naopc)  
**Area**: NLP Understanding / Interpretability  
**Keywords**: Feature Attribution, Interpretability, AOPC, Faithfulness Evaluation, Normalized Metrics

## TL;DR
This paper reveals that the widely used AOPC (Area Over the Perturbation Curve) faithfulness metric yields misleading conclusions when comparing across different models (due to the vast differences in upper and lower bounds of AOPC for distinct models). It proposes Normalized AOPC (NAOPC) to eliminate inter-model incomparability using min-max normalization. Experiments demonstrate that normalization can fundamentally reverse model faithfulness rankings.

## Background & Motivation

**Background**: Feature attribution methods (such as attention weights, gradient-based methods, etc.) are used to explain the predictions of deep neural networks. Faithfulness—whether the attribution accurately reflects the model's internal mechanism—is commonly measured by two variants of AOPC: comprehensiveness (larger output change after removing important features = better) and sufficiency (smaller output change after removing unimportant features = better).

**Limitations of Prior Work**: The upper and lower bounds of AOPC vary drastically across different models. The authors found that one model has a mean upper bound of 0.3, while another has 0.8. This means that even if two models use the exact same attribution method, their AOPC scores will be completely different, making cross-model comparison meaningless.

**Key Challenge**: The AOPC score is simultaneously influenced by two factors: the faithfulness of the attribution method (which we want to measure) and the characteristics of the model itself (e.g., how many features the model relies on, and the interaction pattern between features). These two factors are entangled.

**Goal**: Eliminate the interference of model characteristics on AOPC, ensuring that faithfulness evaluation only reflects the quality of the attribution method.

**Key Insight**: A simple toy example is used to prove the existence of the issue—in linear models, varying the number of dependent features leads to differences in AOPC; in non-linear models, different interaction patterns like OR/AND gates also shift the upper and lower bounds.

**Core Idea**: Compute the exact upper and lower bounds of AOPC for each model-input combination, and then apply min-max normalization to project them into $[0,1]$.

## Method

### Overall Architecture
NAOPC adds a step to the standard AOPC calculation: it first identifies the minimum AOPC (optimal sufficiency) and maximum AOPC (optimal comprehensiveness) for a specific model and input, and then uses min-max normalization to standardize the AOPC score into the $[0,1]$ range. Two versions are proposed: an exact version (exhaustive search) and an approximate version (beam search).

### Key Designs

1. **NAOPC Normalization Formula**:

    - **Function**: Standardizes AOPC scores to a uniform, comparable scale.
    - **Mechanism**: $\text{NAOPC}(f, x, r) = \frac{\text{AOPC}(f, x, r) - \text{AOPC}_\downarrow(f, x)}{\text{AOPC}_\uparrow(f, x) - \text{AOPC}_\downarrow(f, x)}$, where $\text{AOPC}_\downarrow$ and $\text{AOPC}_\uparrow$ represent the lower and upper bounds of AOPC for the given model and input, respectively.
    - **Design Motivation**: After normalization, all model scores fall within the range of $[0, 1]$, where 0 represents the worst (equivalent to random ranking) and 1 represents the optimal (equivalent to ideal ranking), eliminating the interference of model characteristics.

2. **NAOPC_exact (Exact Version)**:

    - **Function**: Exhaustively searches all $N!$ feature permutations to find the exact upper and lower bounds.
    - **Mechanism**: Traverses all possible feature perturbation sequences, calculates the AOPC score for each sequence, and takes the maximum and minimum values as the bounds.
    - **Design Motivation**: Serves as the gold standard to validate the accuracy of the approximate method. The time complexity is $O(N!)$, which is suitable only for short sequences ($\le 12$ features).

3. **NAOPC_beam (Beam Search Approximate Version)**:

    - **Function**: Efficiently approximates the upper and lower bounds using beam search.
    - **Mechanism**: Maintains $B$ candidate feature sequences, gradually expands them, and retains the top $B$ sequences with the highest/lowest scores at each step. The time complexity is $O(B \cdot N^2)$, and the beam size is adaptively selected (doubling from 1 until the bounds stabilize).
    - **Design Motivation**: Scalability of NAOPC to long sequences. Experiments show that the correlation coefficient between beam search and the exact version reaches $0.99+$.

### Loss & Training
NAOPC is an evaluation metric and does not involve training. It is a post-hoc correction for evaluating the attribution methods of existing models.

## Key Experimental Results

### Main Results

Question 1: Do the upper and lower bounds of AOPC indeed differ across different models?

| Model | Dataset | AOPC Lower Bound | AOPC Upper Bound | Range |
|------|--------|----------|----------|------|
| BERT | SST2 | ~0.03 | ~0.30 | 0.27 |
| RoBERTa | SST2 | ~0.05 | ~0.65 | 0.60 |
| LSTM | SST2 | ~0.10 | ~0.80 | 0.70 |

Conclusion: The upper and lower bounds differ significantly (ranging from 0.27 to 0.70), confirming the unreliability of cross-model AOPC comparisons.

### Ablation Study

Question 2: Does NAOPC change model faithfulness rankings?

| Metric | Ranking Conclusion |
|------|---------|
| Original AOPC Comprehensiveness | Adversarially Trained Model > Standard Model (consistent with prior work) |
| **NAOPC Comprehensiveness** | **Standard Model > Adversarially Trained Model (conclusion reversed!)** |

Question 3: Approximation Accuracy of NAOPC_beam

| Metric | Pearson Correlation |
|------|----------------|
| NAOPC_beam vs NAOPC_exact (Comp.) | 0.994 |
| NAOPC_beam vs NAOPC_exact (Suff.) | 0.997 |

### Key Findings
- The upper and lower bounds of AOPC can differ by up to 2-3 times across different models, confirming that cross-model comparison is fundamentally unreliable.
- After normalization, several conclusions from prior work are refuted: the claim that adversarial training "improves faithfulness" no longer holds.
- The correlation coefficient between NAOPC_beam and the exact version is $>0.99$, indicating that the beam search approximation is sufficiently accurate.
- The paper lists 11 top-tier conference papers that used cross-model AOPC comparisons, whose conclusions may need to be re-evaluated.
- The number of features a model relies on and the feature interaction patterns (OR vs AND) are the two core factors influencing the upper and lower bounds of AOPC.

## Highlights & Insights
- **Highly persuasive toy example**: Using 4 simple functions (2 linear, 2 logic gates) clearly demonstrates the fundamental flaw of AOPC, allowing readers to grasp the severity of the problem immediately. This approach of revealing issues using minimal counterexamples is exemplary.
- **Broad impact**: Conclusions of 11 top-tier papers might need re-evaluation, including the widely accepted view that adversarial training improves model interpretability.
- **Simple and effective approach**: Min-max normalization is the most intuitive solution, and the beam search approximation makes it practically applicable. It is released as a PyPI package, lowering the barrier to adoption.
- **Transferability**: The philosophy of NAOPC can be transferred to other evaluation scenarios that use perturbation curves, such as visual saliency map evaluation.

## Limitations & Future Work
- Beam search may still be slow on extremely long sequences, and the approximation quality depends on the choice of beam size.
- The method is only validated on classification tasks; whether feature attribution evaluation for generative tasks has similar issues remains unexplored.
- Normalization assumes that upper and lower bounds can be accurately estimated, but beam search might underestimate the upper bound or overestimate the lower bound.
- Feature attribution evaluation for modern LLMs (GPT, Llama, etc.) is not covered.

## Related Work & Insights
- **vs Original AOPC (DeYoung et al. 2020)**: The original AOPC ignores the differences in upper and lower bounds across models, whereas NAOPC corrects this fundamental flaw through normalization.
- **vs Other Faithfulness Metrics**: Whether this issue exists in other perturbation-based metrics (such as deletion/insertion metrics) warrants further discussion.
- This paper serves as a reminder to meticulously check whether proposed evaluation metrics are confounded by unrelated factors during metric design.

## Rating
- Novelty: ⭐⭐⭐⭐ The identified problem is valuable, though the solution (normalization) is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐ 5 datasets and 4 models are used, but validation in large language model scenarios is lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ Excellent toy example and clear logical reasoning.
- Value: ⭐⭐⭐⭐ Holds significant cautionary value for the interpretability evaluation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Faithfulness Under the Distribution: A New Look at Attribution Evaluation](../../ICLR2026/interpretability/faithfulness_under_the_distribution_a_new_look_at_attribution_evaluation.md)
- [\[ICLR 2026\] Missingness Bias Calibration in Feature Attribution Explanations](../../ICLR2026/interpretability/missingness_bias_calibration_in_feature_attribution_explanations.md)
- [\[AAAI 2026\] Distribution-Based Feature Attribution for Explaining the Predictions of Any Classifier](../../AAAI2026/interpretability/distribution-based_feature_attribution_for_explaining_the_predictions_of_any_cla.md)
- [\[ACL 2025\] Bias Attribution in Filipino Language Models: Extending a Bias Interpretability Metric for Application on Agglutinative Languages](bias_attribution_in_filipino_language_models_extending_a_bias_interpretability_m.md)
- [\[ACL 2025\] Enhancing Automated Interpretability with Output-Centric Feature Descriptions](output_centric_interpretability.md)

</div>

<!-- RELATED:END -->
