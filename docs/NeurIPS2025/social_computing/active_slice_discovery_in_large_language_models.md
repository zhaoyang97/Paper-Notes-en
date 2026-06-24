---
title: >-
  [Paper Note] Active Slice Discovery in Large Language Models
description: >-
  [NeurIPS 2025 (Workshop: Reliable ML from Unreliable Data)][Social Computing][slice discovery] This paper proposes the **Active Slice Discovery** problem framework, integrating active learning into LLM error slice discovery. By combining uncertainty sampling with LLM internal representations (raw embeddings or SAE features), the method achieves slice detection accuracy comparable to fully supervised settings using only 2–10% of labeled data.
tags:
  - "NeurIPS 2025 (Workshop: Reliable ML from Unreliable Data)"
  - "Social Computing"
  - "slice discovery"
  - "active learning"
  - "LLM interpretability"
  - "toxicity classification"
  - "sparse autoencoder"
date: 2026-05-08
content_hash: 903ec5116d573024
---

# Active Slice Discovery in Large Language Models

**Conference**: NeurIPS 2025 (Workshop: Reliable ML from Unreliable Data)
**arXiv**: [2511.20713](https://arxiv.org/abs/2511.20713)  
**Code**: To be released (promised by the authors)  
**Area**: Social Computing
**Keywords**: slice discovery, active learning, LLM interpretability, toxicity classification, sparse autoencoder

## TL;DR

This paper proposes the **Active Slice Discovery** problem framework, integrating active learning into LLM error slice discovery. By combining uncertainty sampling with LLM internal representations (raw embeddings or SAE features), the method achieves slice detection accuracy comparable to fully supervised settings using only 2–10% of labeled data.

## Background & Motivation

**LLMs exhibit systematic error patterns**: LLMs consistently fail on specific data subsets (error slices)—for example, misidentifying toxic comments targeting certain demographic groups in toxicity classification. Discovering these error slices is critical for model auditing and improvement.

**Traditional slice discovery is fully unsupervised**: Existing methods (Domino, Spotlight, etc.) cluster error samples without any slice-level annotations, but unsupervised settings are inherently difficult and yield limited performance.

**Manual sample-level annotation is prohibitively expensive**: Labeling every sample in a dataset with its slice membership requires substantial human effort, which is infeasible in practice.

**Active learning can reduce annotation requirements**: By strategically selecting the most informative samples for human judgment of slice membership, strong performance can be achieved with minimal annotations.

**LLM internal representations may encode slice information**: Hidden-layer embeddings and Sparse Autoencoder (SAE) activations may encode sufficient semantic information to distinguish different error slices.

**Lack of systematic study**: Despite extensive prior work on both active learning and slice discovery individually, their combination under active slice discovery has not been formally studied.

## Method

### Overall Architecture

Given a trained classifier $f_\theta$, a small seed set $\mathcal{D}_s$ with slice annotations, and a large dataset $\mathcal{D}$ with only classification labels, the goal is to learn a slice membership function $\phi: \mathcal{X} \times \mathcal{Y} \to \{0,1\}^k$. The iterative procedure proceeds as follows:

1. Train a slice classifier on the current labeled set
2. Select the most informative unlabeled samples via query strategy $A$
3. Query an oracle (human annotator) for the slice membership of selected samples
4. Add new annotations to $\mathcal{D}_s$ and repeat until budget $K$ is exhausted

### Key Design 1: Feature Representation

- **Raw Layer Embeddings**: The hidden output of the second-to-last layer of Llama-3.1-8B is used as the representation vector for each sample.
- **SAE Sparse Activations**: Activations from the Llama Scope SAE trained on the last layer of Llama-3.1-8B serve as features. SAE features are more interpretable and yield more stable training curves.

### Key Design 2: Active Learning Query Strategies

Three categories of query strategies are compared:
- **Uncertainty strategies** (best performing): Least Confidence, Prediction Entropy, Breaking Ties — select samples for which the model is most uncertain
- **Diversity strategies**: Embedding K-Means, Discriminative Active Learning, Lightweight Coreset — select the most diverse samples in representation space
- **Baseline**: Random Sampling

### Key Design 3: Slice Classifier

- **MLP (Multi-Layer Perceptron)**: Achieves a higher accuracy ceiling (85.8%) but requires careful hyperparameter tuning.
- **Linear SVM**: Requires no complex tuning; achieves 83.0% accuracy with SAE features and is more deployment-friendly.

### Loss & Training

Slice classification is formulated as a binary classification problem, trained with standard cross-entropy loss for the MLP or hinge loss for the SVM. The evaluation metric is slice membership classification accuracy:

$$\text{Acc}_j = \mathbb{E}_{x,y,\mathbf{s}}\left[\mathbf{1}\left[\phi_j(x,y) = s_j\right]\right]$$

## Key Experimental Results

**Dataset**: Jigsaw Toxicity Dataset
**Base Model**: Llama-3.1-8B
**Active Learning Library**: Small-Text

### Main Results (Table 1: "disagree" slice)

| Configuration | Classifier | Best Accuracy | Annotations Required |
|---|---|---|---|
| Raw Embedding | MLP + AL | **85.8%** | 250 / 12,504 (2%) |
| Raw Embedding | SVM + LC | 81.0% | 3,500 |
| SAE Features | MLP + AL | 82.2% | 1,460 / 12,416 |
| SAE Features | SVM + LC | 83.0% | 1,000 |

### Key Findings

1. **Remarkable annotation efficiency**: MLP + Embedding + AL achieves 85.8% accuracy with only 2% of annotations (250 samples), reducing annotation cost by 98% compared to full supervision.
2. **Slice type affects difficulty**: Identity-based slices (female, christian) can be detected with high accuracy using only a few hundred annotations; reaction-based slices (disagree, sad) require more annotations (>1,000) due to weaker lexical cues.
3. **SAE features vs. raw embeddings**: SAE features yield smoother and more stable training curves and are less sensitive to the choice of query strategy. Detection accuracy on the disagree slice improves from 0.80 (embedding) to 0.83 (SAE).
4. **Uncertainty strategies consistently win**: Least Confidence, Prediction Entropy, and Breaking Ties significantly outperform diversity strategies and random sampling across both representation types.

## Highlights & Insights

- **Novel problem formulation**: This is the first work to formally define the active slice discovery problem, transforming "discovering where LLMs fail" from passive observation to active exploration, with strong practical relevance.
- **98% annotation savings**: The best configuration approaches fully supervised performance with only 2% of labels, demonstrating that error slices possess identifiable structure in representation space.
- **Interpretability advantage of SAE**: Incorporating sparse autoencoders as intermediate representations balances performance with interpretability, representing an effective application of mechanistic interpretability tools to downstream tasks.
- **Flexible and composable pipeline**: The framework supports free combination of different LLMs, representations, classifiers, and query strategies, enabling practitioners to select the configuration best suited to their scenario.

## Limitations & Future Work

1. **Validated on a single dataset**: All experiments are limited to the Jigsaw toxicity classification task; generalizability to other NLP tasks (QA, summarization, code generation, etc.) remains untested.
2. **Slices must be predefined**: The current framework assumes that the number of slices and seed samples are known in advance, leaving the discovery of unknown slice types from scratch unaddressed.
3. **No direct comparison with advanced unsupervised slice discovery methods**: The performance of methods such as Domino and DISCERN on the same data is not evaluated.
4. **Limited to classification tasks**: Defining and detecting error slices in generative tasks is considerably more complex and is not addressed.
5. **Workshop paper scope limitations**: Experimental scale and analytical depth are constrained by venue format; ablation studies and more fine-grained analyses are absent.

## Related Work & Insights

- **Slice Discovery (unsupervised)**: Domino (Eyuboglu et al.), Spotlight (D'Eon et al.), SliceLine (Sagadeeva & Burtsev) — this work extends these by introducing active annotation.
- **LLM Interpretability**: Sparse Autoencoder (Cunningham et al.; Templeton et al.) — SAE is employed here as a feature extractor.
- **Active Learning**: Settles (2009) survey; Desai et al. (2025) on uncertainty strategies for text classification — these strategies are directly adopted in this work.
- **Most closely related work**: Hua et al. (2023) use slice discovery to guide active learning for training better classifiers. Their objective is the inverse of this paper's: they elicit task labels $y$, whereas this work elicits slice labels $\mathbf{s}$.

## Rating

- Novelty: ⭐⭐⭐⭐ — Novel and practically meaningful problem formulation; first work to integrate active learning with slice discovery.
- Experimental Thoroughness: ⭐⭐⭐ — Single dataset and model, limited by workshop scope, but covers key dimensions.
- Writing Quality: ⭐⭐⭐⭐ — Problem formalization is clear and experimental results are presented intuitively.
- Value: ⭐⭐⭐⭐ — Provides an efficient and practical tooling framework for LLM auditing; the 98% annotation savings represent significant practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Uncovering Strategic Egoism Behaviors in Large Language Models](uncovering_strategic_egoism_behaviors_in_large_language_models.md)
- [\[ICML 2025\] OR-Bench: An Over-Refusal Benchmark for Large Language Models](../../ICML2025/social_computing/or-bench_an_over-refusal_benchmark_for_large_language_models.md)
- [\[NeurIPS 2025\] DATE-LM: Benchmarking Data Attribution Evaluation for Large Language Models](date-lm_benchmarking_data_attribution_evaluation_for_large_language_models.md)
- [\[ACL 2025\] Exploring Gender Bias in Large Language Models: An In-depth Dive into the German Language](../../ACL2025/social_computing/exploring_gender_bias_in_large_language_models_an_in-depth_dive_into_the_german_.md)
- [\[ACL 2025\] A Survey on Proactive Defense Strategies Against Misinformation in Large Language Models](../../ACL2025/social_computing/a_survey_on_proactive_defense_strategies_against_misinformation_in_large_languag.md)

</div>

<!-- RELATED:END -->
