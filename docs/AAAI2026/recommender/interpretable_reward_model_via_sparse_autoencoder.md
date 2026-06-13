---
title: >-
  [Paper Note] Interpretable Reward Model via Sparse Autoencoder
description: >-
  [AAAI 2026][Recommender Systems][Reward Model] This paper proposes SARM (Sparse Autoencoder-enhanced Reward Model), which integrates a pretrained sparse autoencoder into a reward model to map hidden-layer activations int…
tags:
  - "AAAI 2026"
  - "Recommender Systems"
  - "Reward Model"
  - "Sparse Autoencoder"
  - "Interpretability"
  - "RLHF"
  - "Preference Steering"
date: 2026-05-08
content_hash: dce072412f49e6a4
---

# Interpretable Reward Model via Sparse Autoencoder

**Conference**: AAAI 2026
**arXiv**: [2508.08746](https://arxiv.org/abs/2508.08746)  
**Code**: [https://github.com/schrieffer-z/sarm](https://github.com/schrieffer-z/sarm)  
**Area**: Recommender Systems / LLM Alignment
**Keywords**: Reward Model, Sparse Autoencoder, Interpretability, RLHF, Preference Steering

## TL;DR

This paper proposes SARM (Sparse Autoencoder-enhanced Reward Model), which integrates a pretrained sparse autoencoder into a reward model to map hidden-layer activations into an interpretable, sparse, monosemantic feature space. This design enables feature-level reward attribution and dynamic preference steering, while achieving the highest overall score among all models on RewardBench 2.

## Background & Motivation

### State of the Field

RLHF is the dominant paradigm for LLM alignment, in which a reward model (RM) serves as a proxy for human preferences to guide policy optimization. A typical RM consists of an LLM with a scalar value head that outputs a scalar reward score for a given input–response pair. However, the accuracy, reliability, and **interpretability** of the RM directly affect the alignment quality of downstream models.

### Limitations of Prior Work

**Lack of interpretability**: The scalar reward signal is inherently opaque; it cannot explain why a particular response receives a high or low score. This makes it difficult to verify whether the model is genuinely aligned with human values or merely exploiting spurious correlations in the training data.

**Inflexible preference steering**: Once trained, a conventional RM is static and cannot dynamically adapt to changing user preferences. This rigidity, compounded by opacity, severely limits practical applicability.

### Attempts at Multi-Dimensional RMs and Their Shortcomings

Prior work (e.g., ArmoRM, HelpSteer2) has explored multi-dimensional reward modeling to improve interpretability: regression layers are trained on annotated multi-dimensional data (e.g., helpfulness and verbosity scores) to produce multi-dimensional scores that are then aggregated via weighted summation. However, two key limitations remain:

**Absence of feature-level interpretability**: Each dimension is itself opaque, making it impossible to attribute decisions to interpretable features.

**High annotation cost**: Human scoring across multiple dimensions is required, which is poorly scalable and inherently subjective.

### Core Insight

Sparse autoencoders (SAEs) have been shown to decompose the hidden-layer activations of LLMs into **monosemantic**, interpretable features. Integrating an SAE into an RM enables direct attribution of reward scores to these interpretable features, while fine-grained preference steering can be achieved by modifying the value head weights.

## Method

### Overall Architecture

SARM adopts a two-stage training pipeline:
1. **Stage 1: Sequence-level SAE Pretraining** — An SAE is trained on general-purpose corpora to extract interpretable features.
2. **Stage 2: Reward Modeling** — The SAE encoder is integrated into the RM, and the value head is trained.

### Key Designs

#### 1. **Sequence-Level SAE Pretraining**

**Distinction from conventional token-level SAEs**:

Prior work trains SAEs on token-level activations to extract token-level features. Reward modeling, however, concerns **overall response quality**, which calls for more abstract, sequence-level features.

Drawing on findings from Anthropic, the activation of the last token in a sentence exhibits distinctive activation patterns. Accordingly, SARM trains the SAE **exclusively on the activations of the last token of each sentence**.

**Procedure**:

Given an input sequence $\mathbf{T}$, the hidden states at layer $l$ of the RM are obtained as:

$$\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_{\text{last}}] = \text{RM}_\theta^l(\mathbf{T})$$

The activation of the last token $\mathbf{x}_{\text{last}}$ is extracted and encoded via a TopK SAE:

$$\mathbf{z} = \text{TopK}(\mathbf{W}_{\text{enc}}(\mathbf{x}_{\text{last}} - \mathbf{b}_{\text{pre}}))$$

$$\hat{\mathbf{x}}_{\text{last}} = \mathbf{W}_{\text{dec}} \mathbf{z} + \mathbf{b}_{\text{pre}}$$

where $M = 16 \times d$ (feature dimensionality is 16 times the hidden dimension) and sparsity $k = \frac{3}{64} d$.

Training minimizes the reconstruction error: $\mathcal{L} = \|\mathbf{x} - \hat{\mathbf{x}}\|_2^2$

**Training details**: 50M sequences (~1B tokens) from OpenWebText2 are used; activations are extracted from the layer at depth $\frac{1}{2}$ of the model (balancing representation quality and computational efficiency).

#### 2. **Reward Modeling**

The pretrained SAE encoder is inserted at layer $l$ of the RM; **all layers beyond layer $l$ are discarded**, and a learnable linear value head is applied directly to the sparse feature vector:

$$r_{(x,y)} = h(\mathbf{z}) = \sum_{i=1}^{M} z_i \cdot w_i$$

- $z_i$: activation magnitude of feature $i$ (produced by the frozen SAE encoder)
- $w_i$: learnable weight of the value head

**Training objective**: Standard Bradley-Terry loss:

$$\mathcal{L}(\theta) = -\mathbb{E}_{(x, y_c, y_r) \sim \mathcal{D}} [\log \sigma(r_\theta(x, y_c) - r_\theta(x, y_r))]$$

Only preference data (chosen vs. rejected) is required — **no multi-dimensional annotation is needed**.

During training, the SAE encoder parameters are frozen; only the first $l$ backbone layers and the final linear value head are trained.

#### 3. **Interpretability and Preference Steering**

**Feature attribution**:
Due to TopK sparsity, only a small number of features are activated ($z_i > 0$) at each inference step, allowing the reward score to be directly decomposed into the contributions of individual interpretable features.

**Examples of positively weighted features**:
- Feature 58353: Captures structured analytical content (computation, programming, mathematical reasoning)
- Feature 60427: Captures ethical considerations (privacy, respect, responsible communication)
- These features have positive value head weights $w_i$

**Examples of negatively weighted features**:
- Feature 13950: Captures demeaning or offensive tone
- Feature 17289: Activates in contexts involving unethical advice such as hacking or credit card theft
- These features have negative value head weights $w_i$

**Dynamic preference steering**:
Because SAE features are approximately orthogonal and monosemantic, modifying value head weights $w_i$ enables fine-grained control over RM preferences:
- Increasing $w_i$: Amplifies feature $i$'s contribution to the reward
- Decreasing $w_i$: Suppresses feature $i$'s influence
- Since $w_i$ does not affect the activation $z_i$, samples in which feature $i$ is not activated are unaffected

### Loss & Training

- SAE pretraining: Reconstruction loss only; Adam optimizer; lr = 5e-4
- RM training: Bradley-Terry loss; trained for 3 epochs on Skywork-Reward-Preference-80K-v0.2; batch size 512; lr = 4e-6
- Decoder columns are normalized to unit norm every 10 steps

## Key Experimental Results

### Main Results

Performance comparison on RewardBench 2 (higher Overall is better):

| Model | Params | Overall | Factuality | Precise IF | Math | Safety | Focus | Ties |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ArmoRM-8B | 7.5B | 66.5 | 65.7 | 41.9 | 66.1 | 82.2 | 76.6 | 66.3 |
| Skywork-8B | 7.5B | 71.8 | 69.7 | 40.6 | 60.1 | 94.2 | 94.1 | 71.7 |
| Tulu-70B | 70B | 72.2 | 80.8 | 36.9 | 67.8 | 86.9 | 77.8 | 83.1 |
| GPT-4o | — | 64.9 | 56.8 | 33.1 | 62.3 | 86.2 | 72.9 | 78.2 |
| GPT-4.1 | — | 72.3 | 82.9 | 39.7 | 65.2 | 87.3 | 73.4 | 85.4 |
| Claude Sonnet 4 | — | 71.2 | 76.1 | 35.9 | 70.5 | 89.1 | 76.0 | 79.4 |
| SARM-2B | 2.0B | 62.5 | 55.6 | 35.6 | 60.7 | 84.9 | 82.4 | 56.0 |
| SARM-3B | 2.7B | 64.2 | 58.6 | 34.4 | 62.8 | 87.3 | 86.3 | 55.6 |
| **SARM-4B** | **4.3B** | **73.6** | 68.5 | **42.5** | 63.9 | 91.3 | **96.0** | 79.6 |

**SARM-4B achieves the highest overall score of 73.6 among all models — including the 70B open-source model and closed-source models — with only 4.3B parameters**, reaching 96.0 on the Focus dimension.

### Ablation Study

| Configuration | Params | Overall | Safety | Focus | Ties | Note |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| Random SAE Init | (4+0.3)B | 68.4 | 88.9 | 88.2 | 64.9 | No pretrained SAE |
| Token-level SAE Pretraining | (4+0.3)B | 71.5 | 92.9 | 92.5 | 72.5 | Token-level features |
| **SARM-4B** | **(4+0.3)B** | **73.6** | 91.3 | **96.0** | **79.6** | Sequence-level features |

1. Random SAE initialization → 68.4: Confirms that SARM's gains stem from the structured features extracted by the SAE rather than parameter count alone.
2. Token-level SAE → 71.5: Sequence-level pretraining outperforms token-level by 2.1 points, validating the suitability of sequence-level features for reward modeling.
3. Both components are indispensable: Optimal performance requires the combination of a pretrained SAE and the sequence-level strategy.

### Preference Steering Experiment

After steering the weights of safety-related features:
- **Target set T** (safety queries + chosen responses): The reward distribution shifts noticeably to the right, indicating that the RM correctly assigns higher rewards to safe responses.
- **Complement set C** (remaining samples): The reward distribution remains nearly unchanged, confirming that the steering is precise and does not affect unrelated attributes.

### Key Findings

1. **Interpretability does not compromise performance**: SARM is simultaneously interpretable and state-of-the-art, challenging the conventional interpretability–performance trade-off.
2. **Remarkable parameter efficiency**: SARM-4B at 4.3B parameters outperforms Tulu-3 at 70B and closed-source GPT-4.1.
3. **Features carry semantic meaning**: Positively weighted features (mathematical reasoning, ethical considerations) receive positive weights, while negatively weighted features (offensive tone, illegal advice) receive negative weights — semantic meaning and weight polarity are highly consistent.
4. **Preference steering is precise and controllable**: Modifying a single feature weight affects only samples in which that feature is activated, leaving all other samples unaffected.

## Highlights & Insights

1. **Elegant architectural design**: The streamlined pipeline of "pretrained SAE + frozen encoder + learnable value head" simultaneously achieves interpretability and high performance.
2. **Insight behind sequence-level SAE**: Leveraging the distinctive activation patterns of sentence-final tokens for sequence-level feature extraction is better suited to global quality assessment than token-level approaches.
3. **Bold design choice of discarding later layers**: Inserting the SAE at an intermediate layer and discarding all subsequent layers suggests that the useful information for reward modeling is concentrated in the middle layers.
4. **Causal controllability**: Beyond post-hoc explanation, the framework supports causal intervention by modifying weights — a capability that multi-dimensional RMs cannot offer.

## Limitations & Future Work

1. **Dead latents**: Some SAE features are rarely activated, reducing the effective number of interpretable features below $M$.
2. **Reliance on GPT-4o for feature interpretation**: The quality of automated feature explanations is bounded by GPT-4o's capabilities and incurs non-trivial cost.
3. **Prior on intermediate layer selection**: The layer is fixed at depth $\frac{1}{2}$; while ablations are provided, the optimal layer position may vary across models.
4. **Validation limited to Llama-3**: Generalizability to other architectures (e.g., Mistral, Qwen) has not been verified.
5. **Feature interactions are not modeled**: The current value head performs linear aggregation and does not capture nonlinear interactions among features.
6. **Preference steering validated on the safety dimension only**: Precision of steering across additional dimensions remains to be verified.

## Related Work & Insights

- **Anthropic's SAE work** (Claude Scaling, Towards Monosemanticity): SARM extends the interpretability capability of SAEs from "understanding LLMs" to "controlling RMs."
- **Llama Scope / Gemma Scope**: Infrastructure for layer-wise SAE training.
- **ArmoRM / HelpSteer2**: Pioneers of multi-dimensional reward modeling, though they require costly multi-dimensional annotation.
- **TopK SAE**: Balances sparsity and reconstruction quality by explicitly controlling the number of activations.
- Key insight: SAEs can serve not only as interpretability tools for LLMs but also as interfaces for model controllability.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Integrating SAEs into RMs is a genuinely novel direction; sequence-level pretraining is an original contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Strong RewardBench 2 results with complete ablations, though preference steering is validated on the safety dimension only.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The exposition is clear, with motivation, method, and experiments forming a coherent narrative.
- **Value**: ⭐⭐⭐⭐⭐ — A breakthrough contribution to RM interpretability and controllability with significant implications for RLHF safety research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] GoalRank: Group-Relative Optimization for a Large Ranking Model](../../ICLR2026/recommender/goalrank_group-relative_optimization_for_a_large_ranking_model.md)
- [\[NeurIPS 2025\] Inference-Time Reward Hacking in Large Language Models](../../NeurIPS2025/recommender/inference-time_reward_hacking_in_large_language_models.md)
- [\[NeurIPS 2025\] Measuring What Matters: Construct Validity in Large Language Model Benchmarks](../../NeurIPS2025/recommender/measuring_what_matters_construct_validity_in_large_language_model_benchmarks.md)
- [\[AAAI 2026\] Tokenize Once, Recommend Anywhere: Unified Item Tokenization for Multi-domain LLM-based Recommendation](tokenize_once_recommend_anywhere_unified_item_tokenization_for_multi-domain_llm-.md)
- [\[AAAI 2026\] Length-Adaptive Interest Network for Balancing Long and Short Sequence Modeling in CTR Prediction](length-adaptive_interest_network_for_balancing_long_and_short_sequence_modeling_.md)

</div>

<!-- RELATED:END -->
