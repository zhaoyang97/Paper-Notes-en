---
title: >-
  [Paper Note] SwiftTS: A Swift Selection Framework for Time Series Pre-trained Models via Multi-task Meta-Learning
description: >-
  [ICLR 2026][Time Series][Pre-trained model selection] SwiftTS is proposed as the first model selection framework for time series pre-trained models. It employs a dual-encoder architecture to independently embed patch-lev…
tags:
  - "ICLR 2026"
  - "Time Series"
  - "Pre-trained model selection"
  - "dual encoder"
  - "meta-learning"
  - "time series forecasting"
  - "horizon adaptation"
date: 2026-05-08
content_hash: 3f6300f5b2630b20
---

# SwiftTS: A Swift Selection Framework for Time Series Pre-trained Models via Multi-task Meta-Learning

**Conference**: ICLR 2026  
**arXiv**: [2510.23051](https://arxiv.org/abs/2510.23051)  
**Code**: [GitHub](https://github.com/decisionintelligence/SwiftTS)  
**Area**: Time Series / Model Selection  
**Keywords**: Pre-trained model selection, dual encoder, meta-learning, time series forecasting, horizon adaptation

## TL;DR

SwiftTS is proposed as the first model selection framework for time series pre-trained models. It employs a dual-encoder architecture to independently embed patch-level temporal features of datasets and model meta-information (architecture / topology / function), computes compatibility scores via patch-level cross-attention, and incorporates horizon-adaptive mixture-of-experts together with cross-domain/cross-horizon meta-learning. On 14 datasets × 8 models, it achieves an average weighted Kendall $\tau_\omega = 0.442$, substantially outperforming all baselines.

## Background & Motivation

**Background**: Time series foundation models (TimesFM, MOIRAI, Chronos, etc.) have proliferated rapidly, spanning encoder-only, decoder-only, and encoder-decoder paradigms. No single model is optimal across all tasks — given a new dataset, which model should be selected? Fine-tuning each candidate is prohibitively expensive (reaching $3.46 \times 10^6$ seconds on the Traffic dataset).

**Limitations of Prior Work**:
- Existing model selection methods (RankME, LogME, LEEP, etc.) are primarily designed for CV → they do not account for temporal dependencies or sequential patterns.
- Feature analysis methods require a forward pass through each candidate model to extract features → computational cost scales linearly with the model pool size.
- Time series pre-trained models are highly heterogeneous in architecture and training paradigm → no unified feature extractor is applicable.
- The performance of a single model can vary significantly across different forecasting horizons → existing methods neglect the horizon dimension.

**Key Challenge**: Model selection requires understanding "whether a given dataset and model are compatible," yet (1) different models lack comparable feature representations, and (2) the compatibility relationship varies with the forecasting horizon.

**Goal**: How can the optimal pre-trained model be efficiently selected for a new dataset × new horizon without executing any candidate model?

**Key Insight**: Learn matching patterns from historical (dataset × model × horizon) performance triplets → learn-to-rank rather than feature analysis → use independent encoders to represent data and models separately → match via attention mechanisms.

**Core Idea**: Cast the model selection problem as "learning to find matches between data and model embeddings" via meta-learning.

## Method

### Overall Architecture: Dual Encoder + Cross-Attention + Meta-Learning

The framework is trained on a meta-dataset $\mathcal{D}_{\text{meta}} = \{D^i, Z, H^i, \boldsymbol{r}^i\}_{i=1}^N$, where each sample consists of a downstream dataset $D^i$, a shared model pool $Z$, a horizon $H^i$, and ranking scores $\boldsymbol{r}^i$. The model learns a scoring function $\hat{r}_k = f(\phi_k, D, H)$ to predict the performance of model $\phi_k$ on dataset $D$ under horizon $H$.

### Key Design 1: Temporal-Aware Data Encoder + Knowledge-Injected Model Encoder

**Data Encoder**: The time series $X \in \mathbb{R}^{L \times C}$ is divided into $P = \lfloor L/S \rfloor$ patches, linearly projected into $d$-dimensional embeddings, and fed through self-attention with positional encoding to capture long-range dependencies:

$$E_{\text{sa}} = \text{softmax}\left(\frac{E_{\text{inp}} W_Q^{sa} (E_{\text{inp}} W_K^{sa})^T}{\sqrt{d_k}}\right) E_{\text{inp}} W_V^{sa}$$

For large datasets, $B$ time series are repeatedly sampled and aggregated → yielding a compact data embedding $E_d \in \mathbb{R}^{P \times d}$.

**Model Encoder**: Three knowledge representations are fused to encode candidate model $\phi_k$:
- **Meta-information embedding** $\boldsymbol{v}_a^k$: architecture type (encoder/decoder/enc-dec), parameter count, GMACs complexity, hidden dimension, pre-training domain.
- **Topology embedding** $\boldsymbol{v}_t^k$: the model architecture is represented as a DAG → unsupervised graph embedding obtained via graph2vec.
- **Functional embedding** $\boldsymbol{v}_c^k$: fixed Gaussian noise $\epsilon \sim \mathcal{N}(0, I)$ is fed as input and the output is recorded → different models implement different functions → outputs are distinguishable.

The three embeddings are concatenated and projected: $\boldsymbol{E}_m = \sigma([\boldsymbol{v}_a, \boldsymbol{v}_t, \boldsymbol{v}_c] W_m^T)$

**Design Motivation**: Independent encoding of data and model avoids requiring each candidate model to process the target dataset → eliminates the fundamental cost of feature analysis. The functional embedding captures input–output behavior via "black-box probing" → models can be distinguished even without access to their internal structure.

### Key Design 2: Patchwise Cross-Attention Compatibility Scoring

The model embedding $E_m$ serves as the query, while the data embedding $E_d$ serves as keys and values in cross-attention:

$$E_{\text{ca}} = \text{softmax}\left(\frac{E_m W_Q^{ca} (E_d W_K^{ca})^T}{\sqrt{d_k}}\right) E_d W_V^{ca}$$

This allows each model to attend to the temporal regions of the data that best align with its characteristics → fine-grained matching rather than global similarity.

The training objective combines ranking regularization and prediction accuracy:

$$\mathcal{L}_{\text{total}} = \underbrace{-\sum_{k=1}^K p_k(\hat{\boldsymbol{r}}) \log q_k(\boldsymbol{r})}_{\text{ranking loss}} + \lambda \cdot \underbrace{\sum_{k=1}^K \|\boldsymbol{r}_k - \hat{\boldsymbol{r}}_k\|_2^2}_{\text{prediction loss}}$$

### Key Design 3: Horizon-Adaptive Experts + Cross-Task Meta-Learning

**Horizon-Adaptive Mixture of Experts**: A lightweight router dynamically assigns weights to $G$ experts based on the target horizon $H$:

$$\boldsymbol{w} = \text{softmax}(\text{Router}(H; \theta_s)), \quad \hat{\boldsymbol{r}} = \sum_{g=1}^G w_g \cdot \text{MLP}_g(E_{\text{ca}})$$

Different horizons activate different experts → the same framework handles multiple horizons without retraining.

**Transferable Cross-Task Learning**: A meta-learning paradigm is adopted to enhance OOD generalization:
- **Inner loop**: rapid adaptation on the support set $\theta_i' = \theta - \alpha \nabla_\theta \mathcal{L}_{\text{supp}}(\mathcal{T}_i; \theta)$
- **Outer loop**: update of meta-parameters on the query set $\theta \leftarrow \theta - \gamma \nabla_\theta \sum_{\mathcal{T}_i} \mathcal{L}_{\text{query}}(\mathcal{T}_i; \theta_i')$

Two sampling strategies are employed: (1) cross-dataset sampling — support and query sets are drawn from different datasets → promotes cross-domain generalization; (2) cross-horizon sampling — support and query sets use different horizons → enhances horizon-level adaptability.

## Key Experimental Results

### Main Results: Average Weighted Kendall $\tau_\omega$ across 14 Datasets × 4 Horizons

| Method | H=96 | H=192 | H=336 | H=720 | Avg. | #Top-1 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| RankME | 0.008 | -0.046 | -0.201 | -0.238 | -0.119 | 0 |
| LogME | 0.020 | -0.027 | -0.066 | -0.090 | -0.041 | 3 |
| Etran | 0.318 | 0.212 | -0.004 | 0.073 | 0.150 | 8 |
| DISCO | 0.003 | 0.023 | 0.066 | 0.040 | 0.033 | 4 |
| Model Spider | 0.319 | 0.301 | 0.294 | 0.271 | 0.296 | 6 |
| zero-shot | 0.031 | 0.104 | 0.114 | 0.251 | 0.125 | 5 |
| **SwiftTS** | **0.470** | **0.453** | **0.411** | **0.432** | **0.442** | **28** |

### Top-k Selection Probability and Overall Ranking Correlation

| Method | Pr(top-1) | Pr(top-2) | Pr(top-3) | $\tau_\omega$ |
|------|:---:|:---:|:---:|:---:|
| RankME | 0.000 | 0.000 | 0.196 | -0.119 |
| Model Spider | 0.304 | 0.482 | 0.571 | 0.296 |
| **SwiftTS** | **0.339** | **0.500** | **0.607** | **0.442** |

### Ablation Study on Model Embeddings

| $\boldsymbol{v}_a$ | $\boldsymbol{v}_t$ | $\boldsymbol{v}_c$ | H=96 | H=192 | H=336 | H=720 | Avg. |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ✓ | | | 0.341 | 0.283 | 0.331 | 0.401 | 0.339 |
| | | ✓ | 0.365 | 0.401 | 0.317 | 0.397 | 0.370 |
| ✓ | ✓ | | 0.361 | 0.383 | 0.315 | 0.417 | 0.369 |
| **✓** | **✓** | **✓** | **0.470** | **0.453** | **0.411** | **0.432** | **0.442** |

### Key Findings

- SwiftTS achieves an average $\tau_\omega = 0.442$ → 49% higher than the second-best method Model Spider (0.296) → markedly superior cross-dataset and cross-horizon generalization.
- SwiftTS wins 28/56 Top-1 rankings → the remaining methods each win at most 6–8 → comprehensively leading in selection accuracy.
- Efficiency: model selection on ETTh1 requires only ~1,000–4,000 seconds vs. $4.97 \times 10^4$ seconds for full fine-tuning of all candidates → 10–50× speedup.
- The functional embedding $\boldsymbol{v}_c$ ($\tau_\omega = 0.370$) contributes most, followed by meta-information $\boldsymbol{v}_a$ ($0.339$) → a model's "behavior" is more informative than its "description."
- Cross-task meta-learning uniformly improves $\tau_\omega$ across all horizons → the source of OOD robustness.
- Feature analysis methods (RankME/LogME) frequently yield negative correlation on TS models → these CV-oriented methods are unsuitable for heterogeneous TS models.

## Highlights & Insights

- **"First TS model selection method"**: Model selection for CV has been studied for years, yet the TS domain has remained unexplored → SwiftTS fills this gap, with problem formulation itself constituting a contribution.
- **Aesthetic symmetry of the dual encoder**: Data and model are each independently embedded → matched via attention → candidate models are not required to process the target data → the fundamental overhead of feature analysis is avoided.
- **"Black-box probing" for functional embeddings**: Feeding Gaussian noise to a model and observing its output → different models produce different outputs → models can be distinguished without knowledge of their internal structure → elegant in its simplicity.
- **Natural fit of meta-learning**: Model selection is inherently a "learning-to-learn" problem → cross-dataset/cross-horizon sampling under the MAML paradigm naturally promotes generalization.

## Limitations & Future Work

- The model pool comprises only 8 pre-trained models → scalability remains to be verified as TS foundation models continue to proliferate.
- Functional embeddings require one forward pass per candidate model (albeit with only Gaussian noise as input) → the approach is not entirely zero-cost.
- The framework is limited to time series forecasting → other TS tasks such as classification and anomaly detection are not addressed.
- Constructing the meta-training dataset requires pre-collecting fine-tuned performance of all models on all datasets → the cold-start cost is substantial.

## Related Work & Insights

- **vs. Model Spider (Zhang et al., 2023)**: Also a learning-to-rank approach → but does not account for temporal characteristics or the horizon dimension → achieves only $\tau_\omega = 0.296$ on TS vs. SwiftTS's 0.442.
- **vs. LogME/RankME (feature analysis)**: Produce negative correlation on heterogeneous TS models → identified by SwiftTS as methods inapplicable to the TS domain.
- **vs. TSFM-Bench (Li et al., 2025)**: Provides ground-truth fine-tuned performance → SwiftTS leverages this data for training → the two are complementary.
- **Insight**: Can SwiftTS be extended into a "TS model recommendation system" → integrating automatic data augmentation and online learning → continuously updating selection strategies during deployment?

## Rating

⭐⭐⭐⭐ (4/5)

Overall assessment: SwiftTS is the first model selection framework for TS pre-trained models, featuring a well-designed combination of dual encoder, patch-level cross-attention, horizon-adaptive experts, and meta-learning. The extensive experiments across 14 datasets × 8 models are thorough. However, the model pool is relatively small and the approach is not entirely zero-cost. While the technical design is solid, the primary innovation is methodological rather than theoretical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] TSPulse: Tiny Pre-Trained Models with Disentangled Representations for Rapid Time Series](tspulse_tiny_pre-trained_models_with_disentangled_representations_for_rapid_time.md)
- [\[ICLR 2026\] TSRating: Rating Quality of Diverse Time Series Data by Meta-learning from LLM Judgment](tsrating_time_series_quality_llm.md)
- [\[ICLR 2026\] Rating Quality of Diverse Time Series Data by Meta-learning from LLM Judgment](rating_quality_of_diverse_time_series_data_by_meta-learning_from_llm_judgment.md)
- [\[ICLR 2026\] Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting](learning_recursive_multi-scale_representations_for_irregular_multivariate_time_s.md)
- [\[ICLR 2026\] FeDaL: Federated Dataset Learning for General Time Series Foundation Models](fedal_federated_dataset_learning_for_general_time_series_foundation_models.md)

</div>

<!-- RELATED:END -->
