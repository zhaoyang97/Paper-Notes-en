---
title: >-
  [Paper Note] Prediction Hubs are Context-Informed Frequent Tokens in LLMs
description: >-
  [ACL 2025][LLM (Other)][High-dimensional space] This paper presents the first systematic analysis of the hubness phenomenon in autoregressive LLMs. It theoretically proves that the probability distance used in LLM prediction is unaffected by the distance concentration effect. Empirically, it finds that prediction hubs are context-modulated high-frequency tokens (constituting "benign hubs"), whereas using Euclidean distance to compare LLM representations leads to harmful nuisa…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "High-dimensional space"
  - "hubness phenomenon"
  - "prediction hubs"
  - "probability distance"
  - "token frequency distribution"
date: 2026-05-08
content_hash: 8a2f9d0fda9e5a76
---

# Prediction Hubs are Context-Informed Frequent Tokens in LLMs

**Conference**: ACL 2025  
**arXiv**: [2502.10201](https://arxiv.org/abs/2502.10201)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: High-dimensional space, hubness phenomenon, prediction hubs, probability distance, token frequency distribution

## TL;DR

This paper presents the first systematic analysis of the hubness phenomenon in autoregressive LLMs. It theoretically proves that the probability distance used in LLM prediction is unaffected by the distance concentration effect. Empirically, it finds that prediction hubs are context-modulated high-frequency tokens (constituting "benign hubs"), whereas using Euclidean distance to compare LLM representations leads to harmful nuisance hubs.

## Background & Motivation

**Background**: Hubness is a prevalent phenomenon in high-dimensional data—a small number of data points appear in the $k$-nearest neighbors of a large number of other points, while most points are rarely selected as neighbors. This phenomenon has been observed in various fields, such as time series, image processing, word vectors, sentence embeddings, and cross-modal embeddings, and is generally considered a harmful nuisance that needs to be mitigated through various methods (e.g., Local Scaling, Mutual Proximity, etc.).

**Limitations of Prior Work**: Autoregressive LLMs operate in high-dimensional representation spaces, but whether hubness affects LLM computations has never been systematically studied. This knowledge gap is concerning—if LLM predictions are interfered with by nuisance hubs, it could lead to systematic prediction biases.

**Key Challenge**: The emergence of hubness in high-dimensional space is typically related to the "concentration of distances" phenomenon—as dimensionality increases, the distances between all points tend to become equal, making nearest-neighbor relations meaningless. However, is the actual prediction operation of LLMs (the softmaxed dot product of the context vector and the unembedding matrix) also affected by this?

**Goal**: (1) Analyze hubness in LLM predictions both theoretically and empirically; (2) Distinguish between "benign hubs" and "harmful hubs"; (3) Provide guidance for researchers using LLM representations for other distance calculations.

**Key Insight**: The key insight is that a distinction must be made between two scenarios—distance calculations when the model itself makes predictions (probability distance) vs. manual comparison of LLM representations by researchers using metrics like Euclidean distance. The former is an in-built model operation, while the latter is externally imposed.

**Core Idea**: Hubs in LLM predictions are context-modulated high-frequency tokens, reflecting the highly skewed frequency distribution of natural language. They serve as a beneficial "guessing heuristic" that should not be eliminated. In contrast, external Euclidean distance comparisons do indeed introduce nuisance hubs.

## Method

### Overall Architecture

The study is divided into theoretical and empirical analyses. Theoretically, "probability distance" $d(\mathbf{x}_i, \mathbf{y}_j) = 1 - p(\mathbf{y}_j|\mathbf{x}_i)$ is defined as a metric for internal comparison in LLMs, and it is proven to be unaffected by the distance concentration effect. Empirically, the hubness in three comparison scenarios—context-to-vocabulary (prediction), context-to-context, and vocabulary-to-vocabulary—is analyzed across 5 LLMs and 3 datasets.

### Key Designs

1. **Theoretical analysis of probability distance (Theorem 1)**:

    - **Function**: To prove that the prediction operation in LLMs does not cause distance concentration, and thus does not generate nuisance hubs.
    - **Mechanism**: Define the probability distance as $d(\mathbf{x}_i, \mathbf{y}_j) = 1 - p(\mathbf{y}_j|\mathbf{x}_i)$, where $p(\mathbf{y}_j|\mathbf{x}_i)$ is the predicted probability of token $\mathbf{y}_j$ given context $\mathbf{x}_i$. Using the theorem by Durrant & Kabán (2009), the authors prove that as long as the probability distribution does not converge to a uniform distribution, the distance variance $\text{Var}[d(\mathbf{x},\mathbf{y})]$ does not converge to 0, thereby avoiding distance concentration. The key derivation highlights that the variance equals the expectation of the $L_2$ distance from the probability distribution to the uniform distribution, which is non-zero as long as the model possesses discriminative predictive capacity.
    - **Design Motivation**: Theorem 1 provides an elegant theoretical guarantee that LLMs are not interfered with by nuisance hubness when performing next-token prediction.

2. **Empirical characterization of prediction hubs**:

    - **Function**: To verify that prediction hubs indeed exist but are benign, and to reveal their relationship with token frequency.
    - **Mechanism**: Analysis is conducted on 50K sequences from 3 datasets (Bookcorpus, Pile10k, WikiText-103) across 5 LLMs (OPT-6.7B, Llama-3-8B, Pythia-6.9B, OLMo-7B, Mistral-7B). A hub is defined as when $N_k(x) \geq 100$ for $k=10$. The findings demonstrate: (a) All models exhibit $k$-skewness $> 40$, showing that hubs are abundant; (b) Hub tokens correspond to high-frequency words such as "\n", "the", ",", ".", and "and"; (c) The Spearman correlation between the $k$-occurrence of hubs and token frequency lies between 0.63 and 0.79; (d) This frequency correlation is **context-modulated**—when predicting contexts from a specific corpus, the $k$-occurrence of hubs shows the highest correlation with the token frequency of that specific corpus.
    - **Design Motivation**: Quantitatively validating the intuition that hubs correspond to high-frequency tokens and discovering context-modulated frequency sensitivity proves that this is a useful prediction strategy rather than a nuisance interference.

3. **Detection of nuisance hubs under Euclidean distance**:

    - **Function**: To verify that comparing LLM representations using Euclidean distance generates harmful hubs, serving as a warning to practitioners.
    - **Mechanism**: Context-to-context and vocabulary-to-vocabulary comparisons are evaluated using Euclidean distance, normalized Euclidean distance, and softmaxed dot product. The results indicate: (a) Euclidean distance between contexts indeed exhibits distance concentration (characterized by a gap in the distance distribution that does not extend to 0); (b) Hubs appear in completely semantically unrelated neighborhoods (e.g., contexts from scientific papers appearing in the nearest neighbors of novel texts); (c) The context is more complex for vocabulary-to-vocabulary comparisons—Pythia and OPT show distance concentration, whereas OLMo, Mistral, and Llama do not; (d) Despite this, all vocabulary hubs in all models are "junk tokens" (such as special characters, padding tokens) and are unrelated to frequency.
    - **Design Motivation**: Many researchers are accustomed to comparing LLM representations using cosine similarity or Euclidean distance. This paper's results show that such practices risk hubness and require mitigation techniques.

### Training Dynamics Analysis

Using the public training checkpoints of Pythia, the formation of hubs is analyzed. Hubs exist from the early stages of training (indicating a potential intrinsic bias in the model), but the correlation between hubs and token frequency gradually strengthens as training progresses (Spearman correlation increases from 0.59 to 0.71 on Pile10k), suggesting that the frequency-sensitive prediction strategy is learned during training.

## Key Experimental Results

### Main Results: Spearman correlation between prediction hubs and token frequency

| Model | Corpus | Same-corpus frequency | Cross-corpus frequency | k-skewness |
|------|--------|----------------|------------------|------------|
| Pythia | Pile10k | 0.71 | 0.25 (Bookcorpus) | >40 |
| Pythia | WikiText-103 | 0.70 | 0.28 (Bookcorpus) | >40 |
| Pythia | Bookcorpus | 0.72 | 0.46 (WikiText) | >40 |
| Mistral | Pile10k | 0.79 | 0.29 (Bookcorpus) | >40 |
| Opt | Pile10k | 0.76 | 0.31 (Bookcorpus) | >40 |
| Olmo | Pile10k | 0.74 | 0.27 (Bookcorpus) | >40 |
| Llama | Pile10k | 0.69 | 0.29 (Bookcorpus) | >40 |

### Hub Prediction Accuracy Comparison

| Model | Corpus | Overall Accuracy | Hub Accuracy | Non-hub Accuracy |
|------|--------|-----------|----------|--------------|
| Pythia | Pile10k | 0.37 | 0.39 | 0.28 |
| Llama | Pile10k | 0.37 | 0.40 | 0.31 |
| Mistral | Pile10k | 0.35 | 0.38 | 0.27 |
| Opt | Pile10k | 0.34 | 0.37 | 0.26 |
| Olmo | Pile10k | 0.36 | 0.39 | 0.29 |

### Key Findings

- **Prediction hubs are benign**: When the model predicts a hub token, the accuracy (~38-40%) is significantly higher than when predicting non-hub tokens (~26-31%), showing that predicting high-frequency tokens is an effective strategy.
- **Context modulation is key**: Hubs are not static; instead, they dynamically adjust based on the domain of the input text—the same-corpus frequency correlation is about 0.4 higher than the cross-corpus frequency correlation.
- **Learned during training**: The frequency-sensitive hub strategy is learned step-by-step during the training process, and saturates faster on corpora similar to the training data.
- **Euclidean distance is harmful**: Hubs generated when comparing contexts or vocabulary using Euclidean distance are "junk" tokens (special characters, space sequences, etc.) that appear in completely unrelated neighborhoods—acting as typical nuisance hubs.
- **Heterogeneity exists across models**: In 3 out of 5 models, the Euclidean distance of the unembedding matrix does not display distance concentration (an unexpected finding), but nuisance hubs still persist.

## Highlights & Insights

- **Elegant integration of theory and empirical findings**: Theorem 1 theoretically rules out the chain from distance concentration to nuisance hubs, but empirical analysis reveals that hubs still exist, albeit from a different source (token frequency distribution rather than distance concentration). This three-step process of "theory-empirical analysis-explanation" is highly solid.
- **Conceptual contribution of distinguishing "benign hubs" and "harmful hubs"**: Prior hubness research almost universally treated hubs as nuisances to be eliminated. This work is the first to argue that hubs can be beneficial in certain scenarios, redefining our understanding of hubness.
- **Direct guidance for practitioners**: The conclusion is very clear—there is no need to worry about hubness for next-token prediction, but if one intends to use cosine/Euclidean distance to compare LLM representations for downstream tasks (like semantic search, clustering), hubness mitigation techniques must be applied. Such actionable recommendations are highly valuable.

## Limitations & Future Work

- Empirical findings are based solely on five 7B-class models; whether the same patterns hold true for larger-scale models (70B+) and different architectures (e.g., MoE) remains to be validated.
- An unexpected phenomenon was discovered where the Euclidean distance of the unembedding matrix in 3 out of 5 models did not display distance concentration, but the underlying reason is not deeply explained.
- There is a lack of understanding concerning the causal mechanism—specifically, the detailed dynamical process of how hubs form during training.
- The analysis can be extended to multimodal models (e.g., CLIP) to compare their hub behaviors with those of text-only LLMs.
- The quantitative relationship between hubness and model performance (downstream task accuracy) was not explored.

## Related Work & Insights

- **vs Radovanovic et al. (2010)**: The seminal work on hubness defined it as a universal phenomenon in high-dimensional data and treated it as a nuisance. This work is the first to demonstrate that hubs can be benign in LLM scenarios, amending the fundamental assumptions in this field.
- **vs CLIP/Cross-modal retrieval**: CLIP uses normalized Euclidean distance to compare text and image embeddings and is known to suffer from hubness. This paper's analysis theoretically explains why—CLIP uses Euclidean distance, whereas LLM prediction uses probability distance, representing a fundamental difference with respect to hubness.
- **vs Stolfo et al. (2024) — Confidence Regulation Neurons**: That study identified specialized neurons in LLMs that boost the probability of high-frequency tokens. This paper's discovery of frequency-sensitive hubs aligns perfectly with this, corroborating the same mechanism from a different angle.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic analysis of hubness in LLMs; the theoretical proof and the discovery of benign hubs represent major conceptual contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 models × 3 datasets × 3 distance metrics, supplemented by training dynamics analysis, providing comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear theory-empirical-conclusion logic, explaining complex phenomena in an accessible yet profound manner.
- Value: ⭐⭐⭐⭐ Significantly advances the understanding of LLM representation spaces, with direct actionable guidance for practitioners using LLM embeddings for downstream tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Text-to-Distribution Prediction with Quantile Tokens and Neighbor Context](../../ACL2026/llm_nlp/text-to-distribution_prediction_with_quantile_tokens_and_neighbor_context.md)
- [\[ACL 2025\] Collaborative Performance Prediction for Large Language Models](collaborative_performance_prediction_for_large_language_models.md)
- [\[ACL 2025\] Leveraging In-Context Learning for Political Bias Testing of LLMs](leveraging_in-context_learning_for_political_bias_testing_of_llms.md)
- [\[ACL 2025\] Open-Set Living Need Prediction with Large Language Models](open-set_living_need_prediction_with_large_language_models.md)
- [\[ACL 2025\] ArithmAttack: Evaluating Robustness of LLMs to Noisy Context in Math Problem Solving](arithmattack_evaluating_robustness_of_llms_to_noisy_context_in_math_problem_solv.md)

</div>

<!-- RELATED:END -->
