---
title: >-
  [Paper Note] Redefining Experts: Interpretable Decomposition of Language Models for Toxicity Mitigation
description: >-
  [NeurIPS 2025][Social Computing][toxicity mitigation] This paper proposes EigenShift, a method that performs SVD decomposition on the final output projection layer of LLMs to identify semantic directions (eigen-choices) associated with toxic generation, and suppresses toxicity by selectively attenuating the corresponding singular values. On LLaMA-2, EigenShift reduces toxicity by 58% while increasing perplexity by only 3.62, achieving a favorable balance between safety and fl…
tags:
  - "NeurIPS 2025"
  - "Social Computing"
  - "toxicity mitigation"
  - "eigenvalue decomposition"
  - "interpretability"
  - "language model safety"
  - "neuron experts"
date: 2026-05-08
content_hash: e188bb8784761ada
---

# Redefining Experts: Interpretable Decomposition of Language Models for Toxicity Mitigation

**Conference**: NeurIPS 2025
**arXiv**: [2509.16660](https://arxiv.org/abs/2509.16660)  
**Code**: [GitHub](https://github.com/flamenlp/EigenShift)  
**Area**: Robotics
**Keywords**: toxicity mitigation, eigenvalue decomposition, interpretability, language model safety, neuron experts

## TL;DR

This paper proposes EigenShift, a method that performs SVD decomposition on the final output projection layer of LLMs to identify semantic directions (eigen-choices) associated with toxic generation, and suppresses toxicity by selectively attenuating the corresponding singular values. On LLaMA-2, EigenShift reduces toxicity by 58% while increasing perplexity by only 3.62, achieving a favorable balance between safety and fluency.

## Background & Motivation

Large language models have achieved remarkable success in generating fluent text, yet their tendency to produce toxic content remains a central challenge in AI safety. Existing toxicity mitigation methods primarily operate on individual neuron activations (so-called "concept experts"), but suffer from fundamental limitations:

**Unstable neuron activations**: The AUROC of individual neurons typically falls between 0.50 and 0.55, nearly equivalent to a random classifier. When the AUROC threshold is raised from 0.50 to 0.55, the proportion of "expert" neurons in BERT drops sharply from 11.13% to 3.68%.

**Confusion between detection and generation**: Existing methods conflate "toxicity-detection experts" and "toxicity-generation experts." Suppressing detection experts causes the model to lose its ability to recognize toxicity, leading to catastrophic forgetting.

**Aggressive interventions degrade fluency**: The Det-0 method reduces toxicity to 0% but causes perplexity to spike from 6.23 to 43,517, with a TPH score of only 0.03%.

The paper is organized around three research questions:
- **RQ1**: Are individual neurons reliable indicators of toxicity?
- **RQ2**: Can layer-level or structural representations capture toxicity more robustly?
- **RQ3**: Can interpretable components beyond layers and neurons be identified?

## Method

### Overall Architecture

EigenShift consists of three stages: (1) sampling generations from the base model and labeling toxic tokens with a classifier; (2) applying SVD to the output projection matrix to identify toxicity-related eigen-choices; (3) selectively attenuating the corresponding singular values and reconstructing the output layer weights.

### Key Designs

1. **From Neurons to Layer-Level Experts**

   For each Transformer layer, hidden representations $H_l \in \mathbb{R}^{N \times d}$ are extracted and clustered via k-means ($k=2$), after which AUROC is used to evaluate the alignment between cluster assignments and toxicity labels. Experiments show that layer-level AUROC improves from 54.66% to 63.32% on the Jigsaw dataset, an improvement of 15.84 percentage points.

   **Design Motivation**: Individual neurons perform linear transformations that produce scalar outputs $o = wx + b$, lacking the dimensionality to encode rich semantic information. Layer-level embeddings ($\text{dim} \gg 1$) can capture complex semantic relationships. Moreover, stochastic training techniques such as dropout prevent any single neuron from consistently encoding a specific semantic concept.

2. **Eigen-Choices: Semantic Decision Axes**

   SVD is applied to the final output layer weight matrix $W \in \mathbb{R}^{V \times d}$:

    $W = U \Sigma V^T, \quad B = U, \quad A = \Sigma V^T$

   Here, $V^T \in \mathbb{R}^{d \times d}$ defines an orthonormal basis for the semantic subspace of hidden states; the diagonal entries of $\Sigma$ weight each semantic direction; and $U \in \mathbb{R}^{V \times d}$ maps semantic directions to vocabulary tokens. Each column vector $v_i$ of $V$ corresponds to an "eigen-choice"—a fundamental semantic axis along which the model makes decisions during text generation.

   For a hidden state $h$, the activation along the $i$-th eigen-choice is $a_i = v_i^T h$. It is hypothesized that certain eigenvectors $v_{\text{toxic}}$ are systematically associated with toxic generation.

3. **Toxicity Direction Detection and EigenShift Intervention**

   The directional influence of each eigenvector is computed over toxic and non-toxic samples:

    $\Delta_i = \mathbb{E}_{h_\Phi \sim \text{Toxic}}[v_i^T h_\Phi] - \mathbb{E}_{h_\Psi \sim \text{Non-Toxic}}[v_i^T h_\Psi]$

   Eigenvectors are ranked by $\Delta_i$, and the top-$k$ (e.g., the 99.9th percentile) high-influence eigenvectors are selected as the toxicity-aligned direction set $\mathcal{T}$.

   **Intervention strategy — singular value attenuation**: Singular values in $\mathcal{T}$ are scaled by a decay factor $\alpha < 1$:

    $\sigma_i' = \alpha \cdot \sigma_i, \quad \text{for } i \in \mathcal{T}$

    $W' = U \Sigma' V^T$

   This effectively reduces the model's amplification capacity along toxicity-related semantic directions. The setting $\alpha = 0.9$, $k = 1024$ achieves the best balance between toxicity reduction and perplexity preservation.

### Loss & Training

EigenShift **requires no training or fine-tuning**. The SVD decomposition is a one-time operation, and the Frobenius reconstruction loss is negligible (only $8 \times 10^{-5}$ for LLaMA-7B). The intervention modifies only the output layer weight matrix, leaving the rest of the model unchanged. Toxicity is evaluated on the RealToxicPrompts benchmark, and perplexity is measured on a Wikipedia corpus.

**Newly proposed evaluation metric — TPH Score**:

$$\text{TPH}(T, P) = \frac{2 \cdot T \cdot \frac{1}{1+|P|}}{T + \frac{1}{1+|P|}}$$

where $T$ denotes the percentage reduction in toxicity and $P$ denotes the percentage change in perplexity. The metric takes the harmonic mean of both, providing a unified measure of safety and fluency.

## Key Experimental Results

### Main Results

Performance of different intervention methods across five LLMs:

| Model | Method | Toxicity (%) | Perplexity | TPH Score (%) |
|---|---|---|---|---|
| LLaMA-2 | No intervention | 11.13 | 6.23 | - |
| LLaMA-2 | Det-0 | 0% (↓100%) | 43517 (↑∞) | 0.03 |
| LLaMA-2 | Damp | 0.13% (↓98%) | 741.65 (↑∞) | 1.67 |
| LLaMA-2 | Aura | 3.59% (↓67%) | 19.3 (↑210%) | 43.73 |
| LLaMA-2 | **EigenShift** | **4.71% (↓58%)** | **9.84 (↑58%)** | **60.37** |
| Falcon | No intervention | 9.74 | 8.99 | - |
| Falcon | EigenShift | 3.24% (↓79%) | 9.33 (↑3.78%) | **78.86** |
| MPT-7B | No intervention | 11.13 | 6.8 | - |
| MPT-7B | Aura | 2.83% (↓99.75%) | 7.66 (↑12.65%) | **93.94** |
| MPT-7B | EigenShift | 2.33% (↓79%) | 6.9 (↑1.47%) | 87.74 |

### Ablation Study

Layer-level vs. neuron-level expert detection (AUROC comparison):

| Model | Dataset | Neuron AUROC | Layer AUROC | Gain |
|---|---|---|---|---|
| BERT | Jigsaw | 54.37 | 63.42 | ↑16.67% |
| BART | Jigsaw | 53.95 | 63.37 | ↑17.44% |
| Llama-3.1 | Jigsaw | 54.99 | 63.22 | ↑14.96% |
| Chinese BERT | ToxiCN | 55.71 | 60.42 | ↑8.45% |
| GLM-4 | ToxiCN | 57.56 | 61.92 | ↑7.57% |

Fragility of "expert" neurons:

| Model | AUROC > 0.50 | AUROC > 0.51 | AUROC > 0.55 |
|---|---|---|---|
| BERT | 11.13% | 8.82% | 3.68% |
| BART | 19.91% | 14.60% | 5.85% |
| Llama | 18.64% | 15.42% | 7.08% |
| Mistral | 22.97% | 19.30% | 9.46% |

### Key Findings

1. The vast majority of "expert" neurons have AUROC scores between 0.50 and 0.55, nearly indistinguishable from random; a slight increase in threshold causes most of them to disappear.
2. Layer-level representations significantly outperform neuron-level analysis for toxicity detection in both English and Chinese.
3. Toxicity-detection experts are consistently located in the middle layers of the network (normalized depth 0.7–0.9).
4. EigenShift achieves the best TPH Score on all models except MPT-7B, and increases perplexity by only 3.78% on Falcon.
5. Qualitative case studies show that EigenShift can substitute offensive terms with neutral ones (e.g., "rap*d" → "assault") while preserving semantic intent.

## Highlights & Insights

- **Conceptual clarity**: The paper explicitly distinguishes "detection experts" from "generation experts," explaining why prior methods lead to catastrophic forgetting.
- **Training-free**: A one-time SVD decomposition followed by singular value attenuation incurs negligible computational overhead.
- **Well-designed TPH metric**: The harmonic mean unifies safety and fluency evaluation, avoiding the false sense of security obtained by "destroying the model for zero toxicity."
- **Interpretability of eigen-choices**: Each eigenvector corresponds to a semantic decision axis (e.g., "rudeness," "bluntness"), transforming LLMs from black boxes into interpretable collections of semantic axes.

## Limitations & Future Work

- Only the final `lm_head` layer is analyzed; SVD decomposition of intermediate layers and the evolution of semantic directions across depth remain unexplored.
- Evaluation is limited to models with at most 7B parameters; behavior on larger models is unknown.
- The interpretation of eigen-choices as "semantic axes" is largely qualitative, lacking fine-grained automated semantic annotation.
- EigenShift underperforms Aura on MPT-7B, indicating that the method's adaptability varies across different architectures.
- The effectiveness of the approach for mitigating toxic generation in non-English languages has not been sufficiently validated.

## Related Work & Insights

- The paper contrasts with neuron-level interventions such as "Whispering Experts" and identifies their fundamental limitations.
- Unlike methods requiring external models such as PPLM and FUDGE, EigenShift is entirely intrinsic to the model.
- The approach is complementary to representation surgery methods but more precisely targets the generation side.
- The eigen-choice framework can be extended to control arbitrary semantic concepts, including hate speech, vulgarity, and cultural sensitivity.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The combination of SVD decomposition and singular value attenuation is original and theoretically well-grounded.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Five LLMs, multiple baselines, and cross-lingual analysis, though scale is limited.
- **Writing Quality**: ⭐⭐⭐⭐ The three-RQ structure is well-organized with a clear and progressive logical flow.
- **Value**: ⭐⭐⭐⭐⭐ Provides a lightweight, training-free, and interpretable intervention method for LLM safety.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Auto-Search and Refinement: An Automated Framework for Gender Bias Mitigation in LLMs](auto-search_and_refinement_an_automated_framework_for_gender_bias_mitigation_in_.md)
- [\[ICLR 2026\] From Five Dimensions to Many: Large Language Models as Precise and Interpretable Psychological Profilers](../../ICLR2026/social_computing/from_five_dimensions_to_many_large_language_models_as_precise_and_interpretable_.md)
- [\[NeurIPS 2025\] Uncovering Strategic Egoism Behaviors in Large Language Models](uncovering_strategic_egoism_behaviors_in_large_language_models.md)
- [\[NeurIPS 2025\] DATE-LM: Benchmarking Data Attribution Evaluation for Large Language Models](date-lm_benchmarking_data_attribution_evaluation_for_large_language_models.md)
- [\[ACL 2025\] MDiT-Bench: Evaluating the Dual-Implicit Toxicity in Large Multimodal Models](../../ACL2025/social_computing/mdit-bench_evaluating_the_dual-implicit_toxicity_in_large_multimodal_models.md)

</div>

<!-- RELATED:END -->
