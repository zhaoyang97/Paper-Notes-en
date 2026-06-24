---
title: >-
  [Paper Note] SparseRM: A Lightweight Preference Modeling with Sparse Autoencoder
description: >-
  [AAAI 2026][Interpretability][Sparse Autoencoder] SparseRM leverages sparse autoencoders (SAE) to extract preference-relevant directions from LLM intermediate representations, constructing a lightweight reward model via projection vectors. With fewer than 1% trainable parameters, it surpasses most mainstream reward models and demonstrates stronger generalization in online iterative alignment frameworks.
tags:
  - "AAAI 2026"
  - "Interpretability"
  - "Sparse Autoencoder"
  - "Reward Model"
  - "Preference Modeling"
  - "LLM Alignment"
date: 2026-05-08
content_hash: d2250b15f1d0b908
---

# SparseRM: A Lightweight Preference Modeling with Sparse Autoencoder

**Conference**: AAAI 2026
**arXiv**: [2511.07896](https://arxiv.org/abs/2511.07896)  
**Code**: [github.com/ldc111521/SparseRM](https://github.com/ldc111521/SparseRM)  
**Area**: Model Compression
**Keywords**: Sparse Autoencoder, Reward Model, Preference Modeling, LLM Alignment, Interpretability

## TL;DR
SparseRM leverages sparse autoencoders (SAE) to extract preference-relevant directions from LLM intermediate representations, constructing a lightweight reward model via projection vectors. With fewer than 1% trainable parameters, it surpasses most mainstream reward models and demonstrates stronger generalization in online iterative alignment frameworks.

## Background & Motivation

### State of the Field
Reward models (RM) are a core component of LLM post-training, serving as proxies for human preference evaluation to guide model alignment. In both traditional RLHF and emerging online iterative alignment frameworks, RMs play an indispensable role—assessing response quality, constructing preference pairs, and steering policy optimization.

### Limitations of Prior Work

**High training cost**: Conventional RMs require full LLM fine-tuning (even LoRA demands substantial parameters), incurring significant computational and memory overhead.

**Strong data dependency**: Large-scale human-annotated preference data is required, which is difficult to obtain in resource-constrained settings.

**Distribution shift**: RMs are trained on supervised preference data, yet face policy-generated data during online alignment, leading to poor generalization due to distribution mismatch.

### Root Cause
LLM intermediate representations already encode rich preference-relevant features (e.g., truthfulness, safety), typically associated with a small number of linear directions in the representation space. The question is whether these existing representations can be directly exploited, rather than learning preferences through expensive fine-tuning.

### Starting Point
SAE is used to decompose LLM representations into interpretable sparse directions; preference-relevant directions are filtered out, preference scores are computed via projection, and only an extremely lightweight reward head is trained.

## Method

### Overall Architecture
SparseRM consists of three steps:
1. **Identifying preference-relevant directions**: Decompose LLM representations using SAE and filter preference-relevant latent variables by activation frequency differences.
2. **Computing projection vectors**: Project representations onto the selected directions to obtain preference-aware vectors.
3. **Preference modeling**: Score the projection vectors using a single-layer MLP reward head.

### Key Designs

#### 1. **Preference-Relevant Direction Identification**
- **Input**: Given a preference dataset $\{x_i, y_w^i, y_l^i\}$, pass chosen and rejected responses through model $\mathcal{M}$ to extract hidden states $\mathbf{z}_w$, $\mathbf{z}_l$ at target layer $L$.
- **SAE decomposition**: Feed hidden states into the SAE encoder to obtain sparse latent representations $\mathbf{f}_w$, $\mathbf{f}_l$.
- **Activation frequency computation**:
  $$\mu_w^j = \frac{1}{|\mathcal{D}_w|} \sum_{\mathbf{z}_w} \mathbb{I}(f_j(\mathbf{z}_w) > 0), \quad \mu_l^j = \frac{1}{|\mathcal{D}_l|} \sum_{\mathbf{z}_l} \mathbb{I}(f_j(\mathbf{z}_l) > 0)$$
- **Separation scoring**:
  $$\nabla_j = \mu_w^j - \mu_l^j, \quad \Delta_j = \mu_l^j - \mu_w^j$$
- **Top-K latent variable selection**: Select K latent variables with the highest separation scores from both the chosen and rejected sets; the corresponding decoder directions form the positive/negative preference subspaces $\mathbf{F}_w$, $\mathbf{F}_l$.
- **Design Motivation**: Latent variables exhibiting large activation frequency differences between chosen and rejected samples correspond to directions most discriminative of preference.

#### 2. **Projection Vector Computation**
- For input representation $\mathbf{z}$, compute inner products with each preference direction:
  $$\mathbf{p}_w = [\langle \mathbf{z}, \mathbf{d}_{j_w^1}\rangle, \ldots, \langle \mathbf{z}, \mathbf{d}_{j_w^k}\rangle]$$
  $$\mathbf{p}_l = [\langle \mathbf{z}, \mathbf{d}_{j_l^1}\rangle, \ldots, \langle \mathbf{z}, \mathbf{d}_{j_l^k}\rangle]$$
- Concatenate to form the preference-aware projection vector: $\mathbf{v}_p = [\mathbf{p}_w \mid \mathbf{p}_l]$
- **Key Insight**: Directly using SAE sparse activations yields inferior performance (sparse vectors have limited representational capacity); projection vectors better preserve preference information.

#### 3. **Preference Modeling and Loss Function**
- Reward head: single-layer MLP (hidden dimension 512)
- **Margin loss**: $\mathcal{L}_{\text{margin}} = \max(0, \gamma - (s_w - s_l))$
- Why margin loss over BCE: Humans are better at relative comparisons than absolute scoring; margin loss directly optimizes the score gap between preference pairs.
- Why not BT loss: Margin loss consistently outperforms both BCE and Bradley-Terry loss across three datasets.

#### 4. **Integration into Online Iterative Alignment Framework**
- At each iteration, the policy model generates candidate response pairs.
- SparseRM evaluates the preference score $(s_w, s_l)$ for each pair.
- Inconsistent samples where $s_w < s_l$ are filtered out.
- The retained high-quality preference pairs are used for DPO training.

### Loss & Training
- **Extremely low parameter count**: Only the reward head is trained (256-dim input, 512-dim hidden), accounting for less than 1% of LLM parameters.
- Existing open-source SAEs (GemmaScope, LlamaScope) are used directly without training a new SAE.
- Alignment training: DPO with LoRA fine-tuning, 3 epochs per round, 5 iterative rounds.

## Key Experimental Results

### Main Results (RM Accuracy + Alignment Performance)

| Backbone | Method | SafeRLHF | Red-Teaming | TQA MC1 | TQA MC2 | Trainable Params |
|----------|--------|----------|-------------|---------|---------|-----------------|
| Gemma-2-2B-it | WO RM | 73.4 | 61.8 | 56.1 | 69.8 | — |
| Gemma-2-2B-it | StandardRM | 77.9 | 65.2 | 56.7 | 70.5 | 100% |
| Gemma-2-2B-it | GRAM | 79.0 | 65.8 | 60.0 | 73.9 | 100% |
| Gemma-2-2B-it | **SparseRM** | **79.5** | 67.0 | 59.3 | 73.1 | **<1%** |
| Gemma-2-9B-it | StandardRM | 78.7 | 59.3 | 62.5 | 77.7 | 100% |
| Gemma-2-9B-it | GRAM | 79.3 | 60.7 | 64.7 | 77.9 | 100% |
| Gemma-2-9B-it | **SparseRM** | **79.9** | 60.4 | **65.2** | **78.5** | **<1%** |

### Ablation Study

| RM Input | SafeRLHF | Red-Teaming | TruthfulQA | Notes |
|----------|----------|-------------|------------|-------|
| SAE activations | 92.4 | 88.4 | 91.4 | Limited representational capacity of sparse activations |
| Random directions | 93.0 | 88.0 | 90.7 | Lower bound with random directions |
| **Top-K directions (Ours)** | **94.4** | **90.2** | **93.6** | Projection vectors are optimal |

| Loss Function | SafeRLHF | Red-Teaming | TruthfulQA | Notes |
|--------------|----------|-------------|------------|-------|
| BT Loss | 94.0 | 88.7 | 91.4 | Standard RM loss |
| BCE Loss | 85.7 | 83.1 | 86.3 | Absolute labels perform poorly |
| **Margin Loss** | **94.4** | **90.2** | **93.6** | Consistently optimal |

### SparseRM vs. DenseRM

| Method | RM Accuracy (SafeRLHF) | Alignment (SafeRLHF) | Notes |
|--------|----------------------|---------------------|-------|
| DenseRM | **94.7** | 78.7 | Slightly better in-distribution, but poor out-of-distribution generalization |
| SparseRM | 94.4 | **79.9** | Comparable RM accuracy, superior alignment performance |

### Key Findings
1. SparseRM surpasses or matches most mainstream RMs with fewer than 1% of parameters.
2. **Projection vectors outperform SAE activations**: Direct use of sparse activations offers limited representational capacity; inner-product projection better preserves preference information.
3. **Margin loss consistently outperforms BT and BCE**: Consistent with the relative comparison nature of human preference.
4. **SparseRM generalizes better than DenseRM**: Although DenseRM achieves slightly higher in-distribution accuracy, SparseRM is more robust to distribution shift in alignment tasks.
5. t-SNE visualization reveals higher separation between positive and negative samples in the sparse space, enabling better noise filtering.
6. $K=128$ is the optimal number of latent variables; too few provides insufficient coverage while too many introduces noise.
7. Interpretability analysis confirms that top latent variables correspond to preference-relevant semantics such as "judging correctness."

## Highlights & Insights
1. **Extreme parameter efficiency**: <1% parameters achieve parity with full fine-tuning, challenging the convention that RMs require large parameter counts.
2. **Distribution robustness**: The preference subspace extracted by SAE is more robust to distribution shift, which is critical for online alignment.
3. **Interpretability**: Each preference direction can be semantically decoded via Neuronpedia; for example, latent 4128 corresponds to "WRONG, untrue remarks."
4. **Seamless integration with existing SAE ecosystems**: Direct use of GemmaScope/LlamaScope lowers the barrier to adoption.

## Limitations & Future Work
1. Relies on the quality and coverage of pretrained SAEs; not applicable to models lacking open-source SAEs.
2. Layer selection requires empirical search (e.g., only 3 layers of Gemma-2-9B-it have SAEs).
3. Evaluation is limited to safety/truthfulness dimensions; effectiveness on more complex preference axes (e.g., helpfulness, creativity) remains unverified.
4. The expressive capacity of the single-layer MLP reward head may be insufficient for more challenging tasks.
5. The optimal value of $K$ may vary across tasks; an automated selection mechanism is lacking.

## Related Work & Insights
- The capability of SAE as an LLM interpretability tool has been widely validated; this paper extends its application to preference modeling.
- The linear representation hypothesis provides the theoretical foundation for SparseRM.
- The simplicity of DPO makes it well-suited for integration with lightweight RMs; compatibility with more complex RL methods such as PPO remains to be verified.
- The comparative analysis with DenseRM reveals an important principle: out-of-distribution generalization of RMs is more important than in-distribution accuracy.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — First application of SAE to preference modeling; the approach is original and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three benchmarks, three backbones, and detailed ablations, though task diversity is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear logic with in-depth DenseRM comparative analysis.
- Value: ⭐⭐⭐⭐⭐ — High practical value with <1% parameters; advances efficient alignment research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Data Whitening Improves Sparse Autoencoder Learning](data_whitening_improves_sparse_autoencoder_learning.md)
- [\[CVPR 2026\] Improving Sparse Autoencoder with Dynamic Attention](../../CVPR2026/interpretability/improving_sparse_autoencoder_with_dynamic_attention.md)
- [\[ICML 2026\] Rational Sparse Autoencoder](../../ICML2026/interpretability/rational_sparse_autoencoder.md)
- [\[ICML 2026\] PolySAE: Modeling Feature Interactions in Sparse Autoencoders via Polynomial Decoding](../../ICML2026/interpretability/polysae_modeling_feature_interactions_in_sparse_autoencoders_via_polynomial_deco.md)
- [\[AAAI 2026\] Probing Preference Representations: A Multi-Dimensional Evaluation and Analysis Method for Reward Models](probing_preference_representations_a_multi-dimensional_evaluation_and_analysis_m.md)

</div>

<!-- RELATED:END -->
