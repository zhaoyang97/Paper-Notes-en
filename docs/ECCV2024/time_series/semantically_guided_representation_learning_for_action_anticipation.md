---
title: >-
  [Paper Note] Semantically Guided Representation Learning For Action Anticipation
description: >-
  [ECCV2024][Time Series][Action Anticipation] The S-GEAR framework is proposed, which learns visual action prototypes and utilizes the semantic associations of language models to guide the geometric relationships among these prototypes. This enables the model to comprehend the semantic interconnectedness among actions, thereby enhancing action anticipation performance. S-GEAR achieves SOTA or highly competitive results across four benchmarks: Epic-Kitchens 55/100, EGTEA Gaze+…
tags:
  - "ECCV2024"
  - "Time Series"
  - "Action Anticipation"
  - "Prototype Learning"
  - "Semantically Guided"
  - "Vision-Language Prototypes"
  - "Geometric Association Transfer"
date: 2026-05-08
content_hash: fe2fda009b11c12a
---

# Semantically Guided Representation Learning For Action Anticipation

**Conference**: ECCV2024  
**arXiv**: [2407.02309](https://arxiv.org/abs/2407.02309)  
**Code**: [github.com/ADiko1997/S-GEAR](https://github.com/ADiko1997/S-GEAR)  
**Area**: Self-Supervised  
**Keywords**: Action Anticipation, Prototype Learning, Semantically Guided, Vision-Language Prototypes, Geometric Association Transfer

## TL;DR
The S-GEAR framework is proposed, which learns visual action prototypes and utilizes the semantic associations of language models to guide the geometric relationships among these prototypes. This enables the model to comprehend the semantic interconnectedness among actions, thereby enhancing action anticipation performance. S-GEAR achieves SOTA or highly competitive results across four benchmarks: Epic-Kitchens 55/100, EGTEA Gaze+, and 50 Salads.

## Background & Motivation

**Background**: Action anticipation aims to predict future activities from partially observed event sequences, which is a critical capability for applications such as autonomous driving and wearable assistants. Existing methods primarily process temporal information using sequential models such as LSTMs or causal Transformers.

**Limitations of Prior Work**:
   - Traditional methods focus on extracting better visual and temporal representations but fail to explicitly model the semantic connection relationships between actions that go beyond the immediate video context.
   - Cognitive science research indicates that semantic interconnectedness is fundamental to human anticipation of future behaviors—humans form reliable predictions by relating actions to objects, intentions, and potential outcomes.
   - Modeling semantic relationships among actions solely from video faces two major challenges: (a) handling extremely long sequences is required to capture sufficient co-occurrence context; (b) the distribution of actions in videos is highly imbalanced.

**Mechanism**: Utilizing the property that language models naturally encode semantic relationships between concepts, the geometric associations (rather than features themselves) between action labels in the language space are transferred to the visual prototype space. This enables the visual prototypes to acquire semantic awareness while retaining visual cues.

## Method

### Overall Architecture
S-GEAR contains four core components: (1) a ViT visual encoder to extract frame-level features; (2) a Temporal Context Aggregator (TCA) module to fuse temporal context; (3) a Prototype Attention (PA) module to interact features with learnable prototypes; and (4) a Causal Transformer decoder to predict future representations. Additionally, it incorporates a semantically guided strategy based on dual visual/language prototypes.

### Visual Encoder
- Given an input video segment $V_o = \{f_0, \ldots, f_{T-1}\}$, a ViT-B/16 is used to split each frame into $P$ patch tokens.
- Learnable positional encodings and a CLS token are added, which are processed by Transformer blocks to obtain frame-level features $I_t = \phi(S_t)$.

### Temporal Context Aggregator (TCA)
- Inspired by causal Transformers, but considering all patch tokens within a frame (rather than only the CLS token).
- Causal masked attention is applied over the frame sequence, enabling the current frame to receive the full fine-grained context of past frames.
- Outputs causally enhanced intermediate features $\bar{I} \in \mathbb{R}^{T \times (P+1) \times d}$.

### Prototype Attention (PA)
- Runs in parallel with the TCA, taking the frame's CLS token $I_t^0$ as query and visual prototypes as keys and values.
- Aggregates prototype information most relevant to the current frame through an attention mechanism, producing semantically enhanced features $\tilde{I} \in \mathbb{R}^{T \times d}$.
- The final fusion is weighted via a learnable weight $\lambda$: $\hat{I} = \lambda \bar{I}^0 + (1-\lambda)\tilde{I}$.

### Causal Transformer Decoder
- An autoregressive decoder $\Omega$ processes the fused features $\hat{I}$ and generates future feature sequences $\zeta$ based on masked self-attention.
- For $t=T-1$, $z_t$ is the predicted future action feature.

### Semantic Guidance Strategy (Core Innovation)

**Dual Prototype Definition**:
- **Language Prototypes** $\rho_\ell \in \mathbb{R}^{K \times d}$: Action labels (verb + noun) are encoded using a Sentence Transformer and kept frozen to serve as a reference for semantic relationships.
- **Visual Prototypes** $\rho_v \in \mathbb{R}^{K \times d}$: Learnable parameters initialized using exemplar feature encodings from a pre-trained action recognition model.

**Common Communication Space (Indirect Alignment)**:
- Key Insight: Instead of directly aligning visual and language features (which would lose visual cues), the **relative positional relationships** within their respective prototype spaces are aligned.
- Visual relative representation: $r_k^{z_t} = \cos(z_t, \rho_v[k])$, yielding the cosine similarity vector of action features with all visual prototypes.
- Language relative representation: $r_k^{\text{enc}(y)_t} = \cos(\text{enc}(y)_t, \rho_\ell[k])$, yielding the similarity vector of label encodings with all language prototypes.
- Semantic Loss: $\mathcal{L}_{Sem} = |r^{z_t} - r^{\text{enc}(y)_t}|$, which prompts the geometric relationships among actions in the visual space to mimic those in the language space.

**Lasso Regularization**: $\mathcal{L}_{reg} = ||z_t - \rho_v[k]||_2^2$, pulling the action representation toward its corresponding category's visual prototype.

**Classification Head (Cosine Attention)**:
- Computes the cosine similarity between the predicted feature $z_{T-1}$ and all visual prototypes.
- Converted to weights via softmax, then prototypes are aggregated with weights: $\bar{z}_{T-1} = \text{softmax}(r^{z_{T-1}}) \cdot \rho_v$.
- Fused via a learnable sigmoid gate: $\hat{z}_{T-1} = \sigma(\alpha) z_{T-1} + (1-\sigma(\alpha)) \bar{z}_{T-1}$.
- Finally, a linear layer + softmax outputs class probabilities.

### Loss & Training
$$\mathcal{L}_{tot} = \lambda_1 \mathcal{L}_{Sem} + \lambda_2 \mathcal{L}_{Cls} + \lambda_3 \mathcal{L}_{Past} + \lambda_4 \mathcal{L}_{Feat}$$

- $\mathcal{L}_{Cls}$: Cross-entropy loss for future action classification.
- $\mathcal{L}_{Past}$: Loss for past action classification of observed frames using the causal decoder.
- $\mathcal{L}_{Feat}$: Distance loss between predicted future frame features and actual next-frame features.

## Key Experimental Results

### Datasets

| Dataset | Type | Scale | Action Classes |
|--------|------|------|----------|
| Epic-Kitchens 55 | Egocentric kitchen | 432 videos / ~40K segments | 2,747 |
| Epic-Kitchens 100 | Egocentric kitchen | 700 videos / ~90K segments | 4,053 |
| EGTEA Gaze+ | Egocentric kitchen | 28 hours | 106 |
| 50 Salads | Third-person salad preparation | 50 videos | 17 |

### Main Results
- **EK55 Multimodal** (RGB+Obj+Flow): Top-1 Acc 22.7 (+3.5), Top-5 Acc 43.2 (+2.0)
- **EK100 Single Modality** (RGB, ViT-B): Action Top-5 Recall 18.3 (+0.7 vs RAFTformer-16), without requiring spatio-temporal pre-training
- **EK100 S-GEAR-2B Fusion**: Action Top-5 Recall 19.6 (+0.5 vs RAFTformer-2B)
- **EGTEA Gaze+**: Achieves Top-1 Acc 45.7 (+2.7) using only RGB, Top-5 Acc 71.9 (+0.4 vs HRO using three modalities)
- **50 Salads**: Outperforms SOTA in 5 out of 8 settings, with a Top-1 Acc improvement of up to 3.5 absolute percentage points

### Ablation Study

| Configuration | Action |
|------|--------|
| Baseline (ViT+CT) | 15.2 |
| + Semantic Prototype Learning | 17.8 (+2.6) |
| + TCA | 16.7 |
| + PA + Sem | 18.0 |
| S-GEAR (TCA+PA+Sem) | 18.3 |
| TCA+PA but using language prototype $\rho_\ell$ (no semantic transfer) | 17.4 |

- Semantic prototype learning contributes the most (+2.6).
- Directly using language prototypes is inferior to learning visual prototypes (17.4 vs 18.3), validating the superiority of indirect geometric relationship transfer.
- Using only a 10% prototype subset achieves 17.8 (vs 18.3 of 100%), which can reduce computation by 90%.

## Highlights & Insights

1. **Novel Semantic Transfer Paradigm**: Instead of directly aligning visual and language features, it aligns their geometric associations in their respective prototype spaces. This avoids the loss of modality-specific information in cross-modal alignment.
2. **Cognitive Science Inspired**: Drawing inspiration from human cognitive processes of semantic interconnectedness, this capability is encoded into visual models.
3. **Cross-Scenario Generalization**: Effective across first-person/third-person, short-term/long-term prediction, and datasets of different scales.
4. **RGB-Only Outperforming Multimodal**: On EGTEA Gaze+, using only a single RGB modality outperforms HRO, which utilizes three modalities.
5. **Computational Efficiency**: A small subset of prototypes can be used to approximate the full prototypes, significantly reducing computational overhead.

## Limitations & Future Work

1. **Lack of Built-In Multimodal Mechanism**: Current multimodal results rely on late fusion, without integrating multimodal information at the architecture level.
2. **Semantic Relationships Lack Temporal Ordering**: Currently, action co-occurrence relationships are modeled without explicitly considering the chronological order of actions. Incorporating ordering could mitigate the uncertainty of future predictions.
3. **Dependence on Pre-Trained Language Model Quality**: Experiments show that the quality of semantic modeling by different Sentence Transformers directly impacts results (STSB outperforms BERT).
4. **Large Frame Dimensions**: ViT-B uses 384×384 input, which is more computationally expensive than the 224×224 used by similar methods.

## Related Work & Insights

| Method | Core Difference |
|------|----------|
| AVT | Also a ViT+CT architecture, but without semantic guidance; S-GEAR achieves Top-1 +3.3 on EK55 ViT-B |
| DCR | A curriculum learning method; S-GEAR achieves Top-1 +2.0 on EK55 TSN features |
| MeMViT/RAFTformer | Uses a stronger MViTv2 encoder and Kinetics pre-training; S-GEAR matches them using a simpler ViT-B+IN21K |
| HRO | Stores long-term action prototypes but requires three modalities; S-GEAR surpasses it using RGB only |
| CLIP-based methods | Directly aligns visual-language spaces; S-GEAR only transfers geometric relationships without space alignment |

## Insights & Connection

1. **Geometric relationship transfer vs. feature alignment** is a valuable general paradigm—when feature semantics vary widely between two spaces, transferring topological structures is more flexible than aligning features.
2. The success of prototype learning in action anticipation can be generalized to other temporal prediction tasks (e.g., trajectory prediction, event prediction).
3. Fusing Markov chains/graph structures of action sequences into semantic guidance could be explored to make up for the limitation of ignoring temporal order.

## Rating
- Novelty: ⭐⭐⭐⭐ — The idea of geometric association transfer instead of direct feature alignment is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Four datasets, single/multiple modalities, multiple backbones, and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, standardized charts and tables.
- Value: ⭐⭐⭐⭐ — Opens up a new direction for researching action semantic interconnectedness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] iTimER: Reconstruction Error-Guided Irregularly Sampled Time Series Representation Learning](../../AAAI2026/time_series/beyond_observations_reconstruction_error-guided_irregularly_sampled_time_series_.md)
- [\[ICLR 2026\] Towards Generalizable PDE Dynamics Forecasting via Physics-Guided Invariant Learning](../../ICLR2026/time_series/towards_generalizable_pde_dynamics_forecasting_via_physics-guided_invariant_lear.md)
- [\[ICLR 2026\] GTM: A General Time-series Model for Enhanced Representation Learning](../../ICLR2026/time_series/gtm_a_general_time-series_model_for_enhanced_representation_learning_of_time-series.md)
- [\[NeurIPS 2025\] Universal Spectral Tokenization via Self-Supervised Panchromatic Representation Learning](../../NeurIPS2025/time_series/universal_spectral_tokenization_via_self-supervised_panchromatic_representation_.md)
- [\[AAAI 2026\] C3RL: Rethinking the Combination of Channel-independence and Channel-mixing from Representation Learning](../../AAAI2026/time_series/c3rl_rethinking_the_combination_of_channel-independence_and_channel-mixing_from_.md)

</div>

<!-- RELATED:END -->
