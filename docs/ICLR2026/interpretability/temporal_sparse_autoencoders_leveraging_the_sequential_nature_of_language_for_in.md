---
title: >-
  [Paper Note] Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability
description: >-
  [ICLR 2026][Interpretability][Paper Note] Ours proposes Temporal SAEs (T-SAEs), which introduce a temporal contrastive loss to encourage high-level features to maintain consistent activation across adjacent tokens. This achieves disentanglement of semantic and syntactic features under self-supervised training without explicit semantic signals, restoring smooth
tags:
  - ICLR 2026
  - Interpretability
date: 2026-05-08
content_hash: aa0bd3cd84b07fd4
---
# Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability

**Conference**: ICLR 2026 Oral  
**arXiv**: [2511.05541](https://arxiv.org/abs/2511.05541)  
**Code**: [github.com/AI4LIFE-GROUP/temporal-saes](https://github.com/AI4LIFE-GROUP/temporal-saes)  
**Area**: Model Compression / Interpretability  
**Keywords**: Sparse Autoencoders, Temporal Consistency, Semantic Disentanglement, Contrastive Learning, Interpretability

## TL;DR

Ours proposes Temporal SAEs (T-SAEs), which introduce a temporal contrastive loss to encourage high-level features to maintain consistent activation across adjacent tokens. This achieves disentanglement of semantic and syntactic features under self-supervised training without explicit semantic signals, restoring smoother and more coherent semantic concepts without sacrificing reconstruction quality.

## Background & Motivation

- Existing SAEs often recover features on LLMs that are **token-level, local, and unstable** syntactic patterns (e.g., "The at the sentence start", "period at the sentence end").
- Root cause: SAEs treat tokens as independent samples, **ignoring the sequential structure of language**.
- Key properties of human language:
    - **Semantics** (high-level) change smoothly over time (e.g., a discussion about "plant biology").
    - **Syntax** (low-level) changes rapidly at specific tokens (e.g., "capitalized first letter", "plural nouns").
- A method is needed for SAEs to leverage this temporal structure to discover more meaningful high-level semantic features.

## Method

### Overall Architecture

T-SAE splits the feature dimensions of a standard sparse autoencoder into two segments—treating the first $h$ dimensions as "high-level" features that drift smoothly with semantics, and the remaining $m-h$ dimensions as "low-level" features that jitter with tokens. A temporal contrastive loss is used to inject the sequential structure of language into the training process. The entire Mechanism is fully self-supervised: high-level features are forced to maintain consistent activation across adjacent tokens, while low-level features naturally fit the rapidly changing syntactic signals in the reconstruction residual. Ultimately, this disentangles semantics and syntax without sacrificing reconstruction quality.

### Key Designs

**1. Data Generation Model: Writing "Slow Semantics, Fast Syntax" as an Optimizable Prior**

Existing SAEs treat each token as an independent sample, losing the most fundamental structure in language where "semantics remain largely unchanged within a segment, while syntax jumps token-by-token." T-SAE first models language generation as $\tau_t = \phi(\tau^{t-1}, \mathbf{h}_t, \mathbf{l}_t)$, where high-level variables $\mathbf{h}_t$ (semantics, intent) are approximately time-invariant within a sequence, and low-level variables $\mathbf{l}_t$ (syntax, lexical choice) vary per token. Based on this, two assumptions are made: the temporal consistency assumption requires $\mathbf{h}_t \approx \mathbf{h}_{t'}$ in the same sequence, and the hierarchical representation assumption requires $\mathbf{h}_t$ alone to reconstruct activation $\mathbf{x}_t$ to $\epsilon$ precision, while $\mathbf{l}_t$ only supplements the residual. These assumptions translate "what should be slow and what should be fast" into two optimizable objectives in the loss, providing a clear inductive bias for disentanglement.

**2. Hierarchical Matryoshka Reconstruction: Let High-level Features Carry the Reconstruction Backbone**

To implement the hierarchical representation assumption, T-SAE does not allow high-level features to be merely insignificant decorations. Instead, it uses a Matryoshka loss to simultaneously constrain reconstructions using "only the first $h$ dimensions" and "all $m$ dimensions": $\mathcal{L}_{\text{matr}}(\mathbf{x}_t) = \underbrace{\|\mathbf{x}_t - \mathbf{W}_{0:h}^{\text{dec}} \mathbf{f}_{0:h}(\mathbf{x}_t) + \mathbf{b}^{\text{dec}}\|_2^2}_{\mathcal{L}_H} + \underbrace{\|\mathbf{x}_t - \mathbf{W}^{\text{dec}} \mathbf{f}(\mathbf{x}_t) + \mathbf{b}^{\text{dec}}\|_2^2}_{\mathcal{L}_L}$. $\mathcal{L}_H$ forces the first $h$ dimensions to independently complete the bulk of the reconstruction (corresponding to the semantic backbone), while $\mathcal{L}_L$ lets the full set of features complete the details (corresponding to the syntactic residual). The default split is 20:80 for high and low dimensions, ensuring the division of labor—high-level for "what is this passage about" and low-level for "what does this token look like"—is fixed by the loss structure.

**3. Temporal Contrastive Loss: Targeting High-level Features to Smooth Semantic Curves**

Reconstruction constraints alone do not guarantee temporal consistency for high-level features. Therefore, T-SAE adds an InfoNCE-style bidirectional contrastive loss to the high-level features $\mathbf{z}_t$, treating adjacent tokens in the same sequence as positive samples and other sequences in the same batch as negative samples: $\mathcal{L}_{\text{contr}} = -\frac{1}{N}\sum_{i=1}^N \log \frac{\exp(s(\mathbf{z}_t^{(i)}, \mathbf{z}_{t-1}^{(i)}))}{\sum_j \exp(s(\mathbf{z}_t^{(i)}, \mathbf{z}_{t-1}^{(j)}))} - \frac{1}{N}\sum_{j=1}^N \log \frac{\exp(s(\mathbf{z}_t^{(j)}, \mathbf{z}_{t-1}^{(j)}))}{\sum_i \exp(s(\mathbf{z}_t^{(i)}, \mathbf{z}_{t-1}^{(j)}))}$. It encourages high-level features to be "similar for neighbors and distinct across samples," which is equivalent to forcing semantics to drift continuously within a sequence. Since this loss only acts on the high-level segment, low-level features are liberated to freely fit fluctuating syntactic signals. The total loss is the sum: $\mathcal{L} = \sum_i \mathcal{L}_{\text{matr}}(\mathbf{x}_t^{(i)}) + \alpha \mathcal{L}_{\text{contr}}$, where $\alpha$ regulates the trade-off between semantic smoothness and reconstruction accuracy, all without any explicit semantic labels.

## Key Experimental Results

### Core Performance Metrics

| | FVE ↑ | Cos Sim ↑ | Frac Alive ↑ | Smoothness (High/Low) | Autointerp ↑ |
|--|-------|----------|-------------|----------------------|-------------|
| **T-SAE** (Pythia-160m) | 0.94 | 0.93 | 0.87 | **0.09** / 0.17 | 0.81 |
| Matryoshka SAE | 0.95 | 0.94 | 0.89 | 0.12 / 0.13 | 0.83 |
| BatchTopK SAE | 0.95 | 0.94 | 0.84 | 0.13 / — | 0.85 |
| **T-SAE** (Gemma2-2b) | 0.75 | 0.88 | 0.78 | **0.10** / 0.15 | 0.83 |
| Matryoshka SAE | 0.75 | 0.89 | 0.76 | 0.15 / 0.12 | 0.83 |

### Probing Accuracy for Semantic/Context/Syntax (MMLU)

| Probing Task | T-SAE High-level | T-SAE Low-level | Matryoshka | BatchTopK |
|---------|-----------|-----------|-----------|----------|
| Semantic (k=5) | **Best** | Low | Mid | Mid |
| Context (k=5) | **Best** | Low | Mid | Mid |
| Syntax (k=5) | Mid | **Best** | High | High |

### Ablation Study

| Variant | FVE | Frac Alive | Smoothness(High) | Semantic | Context | Syntax |
|------|-----|-----------|-----------------|------|-------|------|
| Random Contrast (non t-1) | 0.0 | -0.05 | 0.0 | -0.02 | +0.11 | -0.10 |
| 50:50 Partition | -0.01 | +0.01 | 0.0 | +0.02 | +0.09 | — |
| Naive Similarity Loss | Better Recon | — | — | Worse Sem | Worse Ctxt | — |

### Steering Experiments

T-SAE high-level features **Pareto dominate** all baseline SAEs in steering tasks:
- Higher steering success rate + higher output coherence.
- Baselines suffer from token repetition failures during high-intensity steering, whereas T-SAE does not.

### Key Findings

1. T-SAE high-level features are significantly smoother (0.09 vs. 0.12-0.15), exhibiting clear semantic phase transitions across sequences.
2. **Clear Disentanglement**: High-level captures semantics/context, low-level captures syntax $\rightarrow$ this separation does not exist in standard Matryoshka SAEs.
3. Reconstruction quality is nearly unaffected (FVE: 0.94 vs. 0.95).
4. Analyzing the HH-RLHF dataset with T-SAE revealed **spurious correlations** in labeling (rejected responses are longer and more formal).
5. Steering effect and stability of high-level features are far superior to existing SAEs.

## Highlights & Insights

- **Linguistics-Driven Design**: The distinction between smoothly changing semantics vs. locally changing syntax is derived from classical linguistics.
- **Purely Self-supervised Semantic Structure**: Clear semantic clusters emerge without any semantic labels.
- **Unlocking Sequence-level Interpretability**: While existing SAEs only provide token-level explanations, T-SAE enables sequence-level semantic tracking for the first time.
- **Practical Discovery**: Spurious length correlations in the HH-RLHF dataset $\rightarrow$ a warning for safety alignment data quality.
- **Fundamental Steering Advantage**: High-level features change semantic encoding rather than simply increasing the frequency of specific tokens.

## Limitations & Future Work

- The high-level/low-level partition ratio (default 20:80) requires manual setting.
- Contrast is only applied between adjacent tokens; longer-range dependencies require additional design (ablations show random timestep contrast has different properties).
- Training cost is slightly higher than baseline SAEs.
- Validation was limited to Pythia-160m and Gemma2-2b; larger models require additional experiments.

## Related Work

- Sparse Autoencoders: Bricken et al. 2023, Matryoshka SAE, BatchTopK SAE
- Temporal Representation Learning: CPC (Contrastive Predictive Coding), Slow Feature Analysis
- Semantic-Syntactic Separation: LDA (Topic Models), Griffiths et al. 2004 (HMM+LDA)

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The application of temporal consistency priors in SAEs is original and elegant.
- **Technical Depth**: ⭐⭐⭐⭐ — Data generation model + contrastive loss design is clear.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive through probing, visualization, steering, safety case studies, and ablations.
- **Value**: ⭐⭐⭐⭐⭐ — Unlocks sequence-level interpretability and more effective steering capabilities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Ensembling Sparse Autoencoders](../../ICML2026/interpretability/ensembling_sparse_autoencoders.md)
- [\[ICLR 2026\] Towards Understanding the Nature of Attention with Low-Rank Sparse Decomposition](towards_understanding_the_nature_of_attention_with_low-rank_sparse_decomposition.md)
- [\[ICLR 2026\] Toward Faithful Retrieval-Augmented Generation with Sparse Autoencoders](toward_faithful_retrieval-augmented_generation_with_sparse_autoencoders.md)
- [\[ICLR 2026\] AbsTopK: Rethinking Sparse Autoencoders For Bidirectional Features](abstopk_rethinking_sparse_autoencoders_for_bidirectional_features.md)
- [\[ICML 2026\] Sparse Autoencoders are Topic Models](../../ICML2026/interpretability/sparse_autoencoders_are_topic_models.md)

</div>

<!-- RELATED:END -->
