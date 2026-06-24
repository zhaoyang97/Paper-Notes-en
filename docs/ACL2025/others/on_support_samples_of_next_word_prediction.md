---
title: >-
  [Paper Note] On Support Samples of Next Word Prediction
description: >-
  [ACL 2025][Data-centric interpretability] Based on the representer theorem, this paper investigates the role of training samples in next-word prediction of language models, identifying two types of support samples (facilitating prediction and suppressing prediction). It demonstrates that being a support sample is an intrinsic property of the sample itself (predictable prior to training), while non-support samples remain critical for representation learning.
tags:
  - "ACL 2025"
  - "Data-centric interpretability"
  - "support samples"
  - "representer theorem"
  - "language model interpretability"
  - "next word prediction"
date: 2026-05-08
content_hash: c2c4b81edba4ae2e
---

# On Support Samples of Next Word Prediction

**Conference**: ACL 2025  
**arXiv**: [2506.04047](https://arxiv.org/abs/2506.04047)  
**Code**: [github](https://github.com/liyuqian44/On-Support-Samples-of-Next-Word-Prediction)  
**Area**: Others  
**Keywords**: Data-centric interpretability, support samples, representer theorem, language model interpretability, next word prediction

## TL;DR

Based on the representer theorem, this paper investigates the role of training samples in next-word prediction of language models, identifying two types of support samples (facilitating prediction and suppressing prediction). It demonstrates that being a support sample is an intrinsic property of the sample itself (predictable prior to training), while non-support samples remain critical for representation learning.

## Background & Motivation

The interpretability of decision-making in language models is an important research topic. Existing explanation methods mainly follow two directions:

**Mechanistic interpretability**: focuses on neuron activation patterns and circuits.

**Data-centric interpretability**: traces model decisions back to the training data.

This paper focuses on the latter, addressing the core problem: "When the model decides to predict token $v$, which training samples contribute the most?"

Among existing methods, counterfactual approaches (such as the influence function) require calculating the Hessian matrix, which is computationally expensive for large models. This paper adopts a more efficient representer theorem method, directly decomposing the predictor parameters into a weighted combination of training samples, which naturally provides a measure of importance for each sample.

## Method

### Overall Architecture

The language model is decomposed into a representation function $\phi(\mathbf{x})$ and a token prediction function $f(\mathbf{x})$. The representer theorem is used to express the prediction head parameter $\theta_v$ as a weighted sum of representations of all training samples, where the weights act as the importance coefficients of the samples.

### Key Designs

1. **Representer Theorem and Definition of Support Samples**:

    - Based on the Crammer-Singer theorem, when the parameter $\theta$ is a stationary point of the loss function:
    $$\theta_v = \frac{1}{2N\lambda} \sum_{i=1}^{N} (\mathbb{1}(\mathbf{y}_i = v) - p(v|\mathbf{x}_i)) \phi(\mathbf{x}_i)$$
    - The coefficient $\alpha_i = \mathbb{1}(\mathbf{y}_i = v) - p(v|\mathbf{x}_i)$ measures the importance of the sample.
    - Samples with large $|\alpha_i|$ are defined as "support samples" ($|α_i| ≥ τ$, where $τ=0.9$), while the rest are non-support samples.
    - A large $\alpha_i$ implies that the sample is difficult to learn: the prediction probability for the correct label is small.

2. **Two Types of Support Samples**:

    - **Type-1 (Facilitating)**: $\mathbf{y}_i = v$, but the model has low prediction confidence ($\alpha_i > 0$). These pull the predictor toward predicting $v$.
    - **Type-2 (Suppressing)**: $\mathbf{y}_i \neq v$, but the model incorrectly predicts $v$ with high confidence ($\alpha_i < 0$). These push the predictor away from predicting $v$.
    - Leave-one-out (removal) experiments confirm: removing Type-1 leads to prediction failure for $v$; removing Type-2 leads to perfect prediction for $v$ but negatively impacts other tokens.

3. **Intrinsicality of Support Samples**: A simple classifier (linear/MLP) is used to determine whether a sample is a support sample. Features include:

    - The last-layer hidden vector
    - Concatenation of hidden vectors from all layers
    - Gradient features (randomly projected to 4096 dimensions)
    - Features extracted at three training stages (initialization, early, final)

4. **The Role of Non-Support Samples**:

    - Analyzing the proportion of non-support samples in each layer through layer-wise probing.
    - It is found that higher layers contain more non-support samples, indicating they are crucial for intermediate representation learning.

### Loss & Training

- Language Model: GPT-2 architecture (117M parameters), 12-layer Transformer, trained from scratch on the wikitext-2 dataset (2.37M samples).
- Also validated on larger models (345M/774M/1.5B) and a larger dataset (wikitext-103, 117M samples).
- Binary Classifier: Linear and MLP, with a train/validation/test ratio of 8:1:1.

## Key Experimental Results

### Main Results

Basic statistics of support samples:

| Metric | Value |
|------|------|
| Total training samples | 2.37M |
| Support samples | 1.29M (54%) |
| More than half of the samples are important to the predictor | - |
| 42% of tokens contribute most of the support samples | - |
| 58% of tokens have fewer than 10 support samples | - |

Support sample deletion experiment (taking $v$="hens" as an example):

| Removed Set | Full Set Loss | Target Token Loss |
|---------|---------|---------------|
| No removal | 3.28 | 0.24 |
| Remove all support samples | 4.45 | 2.37 |
| Remove only Type-1 | 3.35 | 16.73 |
| Remove only Type-2 | 4.75 | 0.00 |
| Randomly remove same amount | 3.75 | 0.27 |

### Ablation Study

Non-support sample removal experiment (training LM heads):

| Sampling Method | Test Loss | Description |
|---------|---------|------|
| No removal | 5.08 | Baseline |
| Hard removal (all non-support) | 5.64 | Overfitting, introducing many new support samples |
| Soft sampling (weighted by $\alpha$) | 5.13 | Close to baseline, optimal strategy |
| Random removal of same amount | 5.18 | Suboptimal |

Non-support sample removal experiment (full model training):

| Sampling Method | Test Loss | Description |
|---------|---------|------|
| No removal | 5.08 | Baseline |
| Hard removal | 6.57 | Severe deterioration |
| Random removal of same amount | 5.47 | Outperforms soft sampling |
| Soft sampling | 5.69 | Inferior to random |

Support sample prediction accuracy:

| Feature / Stage | Initialization ($\theta^0$) | Final Checkpoint |
|----------|-------------|-----------|
| Gradient features + MLP | **~80%** | ~85% |
| Last hidden layer + MLP | ~55% | ~65% |
| All hidden layers + MLP | ~58% | ~66% |

### Key Findings

1. **Surprisingly high proportion of support samples (54%)**: Over half of the training samples contribute significantly to the predictor parameters, indicating that the patterns learned by language models are far from concise.
2. **Highly skewed distribution of support samples**: A few tokens (such as punctuation) require almost no support samples, while semantically rich words (verbs have 86% support samples, whereas punctuation has only 20%) require substantial support.
3. **Type-2 support samples are indispensable**: After removing Type-2 samples, the target token loss drops to 0 (perfect prediction), but other tokens are negatively affected. Type-2 samples act as "guardrails" to prevent the model from being overconfident in predicting $v$.
4. **Different strategies for training head vs. full model**: When training LM heads, non-support samples can be removed (using soft sampling), but they are indispensable during full model training.
5. **Support samples are predictable prior to training**: Using only the gradient features of the randomly initialized model, one can predict the support/non-support labels with 80% accuracy.
6. **Non-support samples are more important in higher layers**: Layer-wise probing shows a sharp increase in non-support samples at Layer 6 (the middle layer), suggesting a qualitative change in the model's representational capacity at this stage.

## Highlights & Insights

- **Integration of theoretical elegance and experimental validation**: The representer theorem provides a concise mathematical framework, while the removal experiments offer intuitive causal validation.
- **Redefining "memorization"**: In traditional definitions, "memorized" samples are those predicted correctly with high confidence. However, this paper demonstrates that these are precisely non-support samples; support samples are the ones truly "memorized" to make predictions.
- **Token relationship network**: The directed network formed by Type-2 support relationships reveals adversarial relationships among tokens. High-frequency tokens (such as commas, @, and "and") are the primary providers of Type-2 support.
- **Discovery of intrinsic properties**: Whether a sample is a support sample can be determined with 80% accuracy prior to training, which is a surprising finding.

## Limitations & Future Work

1. The boundary between support and non-support samples is not sharp (binary classification based on the threshold $τ=0.9$ is an approximation), and the classifier cannot achieve 100% accuracy even at the final checkpoint.
2. Experiments are primarily based on a smaller-scale GPT-2 (117M). Although some validations were performed on larger models, the generalizability to modern large language models remains to be verified.
3. The representer theorem requires the parameters to be stationary points of the loss function, which may not be completely satisfied during actual training.
4. The analysis is limited to support relationships at the prediction-head level; sample contributions within deeper Transformer blocks have not been analyzed.

## Related Work & Insights

- Compared to the Influence Function (Koh & Liang, 2017), the representer theorem method is computationally less expensive, but has a narrower scope of application (limited to the prediction head).
- The "forgetting events" of Toneva et al. (2019) share similarities with the support samples in this paper, as both focus on samples that are difficult to learn.
- The definition of "memorized samples" by Tirumala et al. (2022) is precisely a subset of the non-support samples in this paper (Claim 3).
- Research on data selection/pruning typically focuses only on "retaining hard samples" (Type-1). This paper finds that Type-2 is equally important, providing a more comprehensive perspective.
- This may inspire new training data selection strategies: retaining all support samples along with an appropriate amount of non-support samples.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Applies the representer theorem to LM interpretability, identifying two types of support samples and their intrinsic properties, offering a novel perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Rigorously designed experiments with multi-angle validation, though the model scale is relatively small.
- Writing Quality: ⭐⭐⭐⭐⭐ Theorems and experiments are closely interwoven, with tight logic and an engaging narrative.
- Value: ⭐⭐⭐⭐⭐ Opens up a new direction for data-centric LM interpretability, providing practical guidance for training data selection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Cautious Next Token Prediction](cautious_next_token_prediction.md)
- [\[ACL 2025\] Subword Models Struggle with Word Learning, but Surprisal Hides It](subword_models_struggle_with_word_learning_but_surprisal_hides_it.md)
- [\[ACL 2025\] MockConf: A Student Interpretation Dataset: Analysis, Word- and Span-level Alignment and Baselines](mockconf_a_student_interpretation_dataset_analysis_word-_and_span-level_alignmen.md)
- [\[ACL 2025\] Advancing Sequential Numerical Prediction in Autoregressive Models](advancing_sequential_numerical_prediction_in_autoregressive_models.md)
- [\[ACL 2025\] Generating Plausible Distractors for Multiple-Choice Questions via Student Choice Prediction](distractor_gen_multiple_choice.md)

</div>

<!-- RELATED:END -->
