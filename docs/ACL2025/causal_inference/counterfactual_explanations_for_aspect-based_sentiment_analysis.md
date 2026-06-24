---
title: >-
  [Paper Note] Counterfactual Explanations for Aspect-Based Sentiment Analysis
description: >-
  [ACL 2025][Causal Inference][Counterfactual Explanations] This paper proposes a method for generating counterfactual explanations for aspect-based sentiment analysis (ABSA). By finding the minimal text edits that flip the sentiment polarity of a specific aspect, it provides intuitive causal explanations for ABSA model predictions.
tags:
  - "ACL 2025"
  - "Causal Inference"
  - "Counterfactual Explanations"
  - "Aspect-Based Sentiment Analysis"
  - "Explainable AI"
  - "Sentiment Polarity Flipping"
date: 2026-05-08
content_hash: 4a8eccc4524c62ca
---

# Counterfactual Explanations for Aspect-Based Sentiment Analysis

**Conference**: ACL 2025  
**Area**: Causal Inference  
**Keywords**: Counterfactual Explanations, Aspect-Based Sentiment Analysis, Explainable AI, Causal Inference, Sentiment Polarity Flipping

## TL;DR
This paper proposes a method for generating counterfactual explanations for aspect-based sentiment analysis (ABSA). By finding the minimal text edits that flip the sentiment polarity of a specific aspect, it provides intuitive causal explanations for ABSA model predictions.

## Background & Motivation

**Background**: Aspect-Based Sentiment Analysis (ABSA) aims to identify the sentiment polarity of a target aspect in a text. For example, in the restaurant review "The food was great but the service was too slow," the sentiment for "food" is positive, while the sentiment for "service" is negative. Although deep learning models have achieved strong performance on ABSA tasks, their predictions lack interpretability, making it difficult for both users and developers to understand why a model makes a specific judgment.

**Limitations of Prior Work**: Existing interpretability methods for ABSA are primarily attribution methods based on attention weights or gradients (e.g., attention visualization, Integrated Gradients), which answer "which words are most important to the prediction." However, this "importance-based" explanation is not intuitive enough—knowing that "too slow" is important for a negative prediction does not tell the user "what should it be changed to for the prediction to change." Counterfactual explanations ("if the review said the service was prompt, the prediction would become positive") align better with human causal reasoning, but research on counterfactual explanations in the ABSA scenario remains almost non-existent.

**Key Challenge**: Generating counterfactual explanations for ABSA is far more challenging than for general text sentiment analysis: edits must flip the sentiment of the target aspect while keeping the sentiment of non-target aspects unchanged. For instance, in "The food was good but the service was bad," generating a counterfactual for "service" must not alter the evaluation of "food." This aspect-specificity constraint significantly complicates the problem.

**Goal**: The goal is to design a counterfactual explanation generation method for ABSA that can: (1) find the minimal edits to flip the target aspect's sentiment; (2) keep the sentiment of non-target aspects unchanged; (3) generate linguistically natural and fluent counterfactual texts.

**Key Insight**: The authors formulate ABSA counterfactual explanation as a constrained optimization problem: minimizing the amount of text modification under the constraints of target aspect sentiment flipping, non-target aspect sentiment preservation, and language fluency.

**Core Idea**: Through an aspect-aware counterfactual search algorithm, the method searches in the semantic space for text variants that simultaneously satisfy the three constraints of sentiment flipping, aspect preservation, and minimal modification, thereby providing high-quality counterfactual explanations for ABSA models.

## Method

### Overall Architecture
The input is a text segment, a target aspect, and the predictions of an ABSA model. The output is a modified counterfactual text—the sentiment of this text toward the target aspect is opposite to the original prediction, while other aspects remain as unchanged as possible. The method consists of three stages: (1) candidate edit position identification; (2) aspect-aware substitution generation; (3) search and ranking.

### Key Designs

1. **Aspect-Relevance-Aware Edit Position Identification**:

    - **Function**: Identify which tokens are most relevant to the target aspect's sentiment and should thus be selected as edit positions.
    - **Mechanism**: A combination of two signals is used to determine edit positions: (a) the ABSA model's attention weights on the target aspect (identifying tokens most relevant to the target aspect's sentiment judgment); (b) the aspect-sentiment dependency path (finding tokens on the path connecting aspect terms and sentiment words using dependency parsing). Tokens whose combined score from both signals exceeds a threshold are marked as candidate edit positions. Aspect terms themselves are excluded from modification (as an ABSA explanation should explain "what evaluation led to this sentiment" rather than changing the aspect itself).
    - **Design Motivation**: Not all tokens are relevant to the sentiment of a specific aspect, and random modifications would lead to unnecessary changes. Aspect relevance awareness ensures that only words directly associated with the target aspect's sentiment judgment are modified.

2. **Constraint-Satisfying Substitution Generation**:

    - **Function**: Generate candidate substitutions for each candidate edit position that satisfy multiple constraints.
    - **Mechanism**: A Masked Language Model (MLM, such as BERT) is used to generate candidate substitutions. Then, constraint checks are applied to each candidate: (a) Sentiment Flipping Constraint—the predicted sentiment of the target aspect should flip after substitution; (b) Aspect Preservation Constraint—the predicted sentiment of non-target aspects must not change; (c) Semantic Minimal Modification Constraint—the semantic distance between the substitution and the original word should not exceed a threshold $\delta$; (d) Language Fluency Constraint—the perplexity of the modified sentence should not exceed $k$ times that of the original sentence. Qualified substitution plans are obtained by filtering out candidates that fail to satisfy these constraints.
    - **Design Motivation**: The four constraints correspond to the four core requirements of counterfactual explanations: validity (flipping the prediction), specificity (affecting only the target aspect), minimality (minimal modification), and plausibility (natural language).

3. **Beam Search and Counterfactual Ranking**:

    - **Function**: Select the optimal counterfactual explanation from multiple candidate counterfactuals.
    - **Mechanism**: When multiple edit positions are available, beam search is used to explore different combinations of edits. The scoring function jointly considers the edit amount (the fewer, the better), semantic preservation (cosine similarity between original and counterfactual texts), and language fluency (perplexity). The counterfactual with the highest overall score is selected as the final output. If modifying a single token suffices to flip the sentiment, single-token edits are prioritized.
    - **Design Motivation**: The value of a counterfactual explanation lies in the "minimal necessary modification"—the fewer the changes, the clearer the explanation. Beam search finds the most concise counterfactual while ensuring constraints are satisfied.

### Loss & Training
This method is an inference-time approach and does not require additional training. Both the targeted ABSA model and the MLM are pre-trained or already fully trained models.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | MICE | Polyjuice | Random Substitution |
|--------|------|---------|------|-----------|---------|
| SemEval-14 Laptop | Flip Rate ↑ | 91.3% | 84.6% | 78.2% | 52.1% |
| SemEval-14 Laptop | Aspect Preservation Rate ↑ | 88.7% | 72.4% | 65.8% | 71.2% |
| SemEval-14 Restaurant | Flip Rate ↑ | 93.1% | 86.3% | 80.5% | 55.4% |
| SemEval-14 Restaurant | Aspect Preservation Rate ↑ | 90.2% | 75.1% | 68.3% | 73.6% |
| SemEval-14 Restaurant | Average Edit Distance ↓ | 2.1 | 3.8 | 4.5 | 2.3 |
| MAMS | Flip Rate ↑ | 87.6% | 79.8% | 73.1% | 48.3% |

### Ablation Study

| Configuration | Flip Rate | Aspect Preservation Rate | Edit Distance | Description |
|------|--------|-----------|---------|------|
| Full model | 91.3% | 88.7% | 2.1 | Full method |
| w/o Aspect Preservation Constraint | 93.5% | 61.2% | 1.8 | Flip rate increases slightly, but preservation rate plummets |
| w/o Aspect Relevance | 85.4% | 82.1% | 3.4 | Imprecise edit positions |
| w/o Fluency Constraint | 90.8% | 87.9% | 2.0 | Fluency decreases, but other metrics remain close |
| Attention-only Localization | 87.2% | 85.3% | 2.5 | No dependency parsing |

### Key Findings
- The aspect preservation constraint is the key distinction between the proposed method and general counterfactual methods; without it, the aspect preservation rate plummets from 88.7% to 61.2%.
- Aspect-relevance-aware position identification enables more precise editing, reducing the edit distance from 3.4 to 2.1 while simultaneously improving the flip rate.
- In complex sentences with multiple interwoven aspects (MAMS dataset), the advantages of the proposed method are even more pronounced.

## Highlights & Insights
- Introducing the aspect preservation constraint is the core innovation. This marks the essential difference between ABSA counterfactuals and general counterfactuals, and is where its practical value lies.
- The dual-path edit position identification (attention + dependency parsing) fully utilizes the complementarity of different signals: attention captures model focus, while dependency parsing captures linguistic structures.
- The counterfactuals generated by this method can be directly used to debug and improve ABSA models: if a generated counterfactual is implausible, it suggests that the model might have learned spurious correlations.

## Limitations & Future Work
- The method depends on the differentiability of the target ABSA model (requiring attention weights), making it inapplicable to black-box models.
- The range of substitutions generated by the MLM is limited by its vocabulary, restricting the method's ability to handle cases that require phrase-level or syntactic-level changes to flip sentiment.
- The human comprehensibility evaluation of the counterfactual explanations is insufficient; more user studies are required.
- Future work can explore LLM-based generative counterfactual methods to overcome the limitations of token-level substitutions.

## Related Work & Insights
- **vs MICE (Ross et al.)**: MICE generates general counterfactual explanations without considering aspect preservation; the aspect-aware constraints proposed in this work make it more accurate in ABSA scenarios.
- **vs Polyjuice**: Polyjuice is a general-purpose counterfactual generator that produces diverse outputs, but they may not necessarily satisfy the aspect preservation constraint.
- **vs LIME for ABSA**: LIME provides feature importance explanations while this work provides counterfactual explanations. The two are complementary: LIME tells you "what matters" and counterfactuals tell you "what to change."

## Rating
- Novelty: ⭐⭐⭐⭐ Novelty in introducing counterfactual explanations to ABSA and designing aspect preservation constraints.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple datasets and detailed ablation, but lacks user studies.
- Writing Quality: ⭐⭐⭐⭐ Clear problem formulation and logical, complete constraint design.
- Value: ⭐⭐⭐⭐ Tangible progress in advancing explainable AI research for ABSA.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CoA-Reasoning: Explorations on Counterfactual Analysis in Physical Reasoning of LVLMs](coa-reasoning_explorations_on_counterfactual_analysis_in_physical_reasoning_of_l.md)
- [\[ICLR 2026\] Counterfactual Explanations on Robust Perceptual Geodesics](../../ICLR2026/causal_inference/counterfactual_explanations_on_robust_perceptual_geodesics.md)
- [\[CVPR 2025\] FG-VCE: Towards Fine-Grained Interpretability — Counterfactual Explanations for Misclassification with Saliency Partition](../../CVPR2025/causal_inference/towards_fine-grained_interpretability_counterfactual_explanations_for_misclassif.md)
- [\[NeurIPS 2025\] Performative Validity of Recourse Explanations](../../NeurIPS2025/causal_inference/performative_validity_of_recourse_explanations.md)
- [\[CVPR 2026\] MaskDiME: Adaptive Masked Diffusion for Precise and Efficient Visual Counterfactual Explanations](../../CVPR2026/causal_inference/maskdime_adaptive_masked_diffusion_for_precise_and_efficient_visual_counterfactu.md)

</div>

<!-- RELATED:END -->
