---
title: >-
  [Paper Note] Limited-Resource Adapters Are Regularizers, Not Linguists
description: >-
  [Model Compression][Low-resource machine translation] This paper combines adapter souping (weight averaging) with cross-attention fine-tuning for low-resource Creole machine translation. While the method yields significant improvements (up to +8 BLEU), linguistic relatedness does not meaningfully co-vary with adapter performance—randomly initialized, untrained adapters perform equally well. This indicates that the role of adapters in this setting is essentially **parameter re…
tags:
  - "Model Compression"
  - "Low-resource machine translation"
  - "Adapters"
  - "Regularization"
  - "Cross-lingual transfer"
  - "Creoles"
date: 2026-05-08
content_hash: 7d6987e56b9f58ee
---

# Limited-Resource Adapters Are Regularizers, Not Linguists

| Conference | Area | arXiv | Code |
|------|------|-------|------|
| ACL2025 | Model Compression / Low-resource NLP | [2505.24525](https://arxiv.org/abs/2505.24525) | — |

**Keywords**: Low-resource machine translation, Adapters, Regularization, Cross-lingual transfer, Creoles

## TL;DR

This paper combines adapter souping (weight averaging) with cross-attention fine-tuning for low-resource Creole machine translation. While the method yields significant improvements (up to +8 BLEU), linguistic relatedness does not meaningfully co-vary with adapter performance—randomly initialized, untrained adapters perform equally well. This indicates that the role of adapters in this setting is essentially **parameter regularization rather than linguistic information transfer**.

## Background & Motivation

### Translation Challenges for Low-Resource Languages

Most languages globally suffer from data scarcity, and reliable machine translation (MT) remains elusive. Cross-lingual transfer learning is a dominant strategy but has yet to bridge the translation quality gap between high- and low-resource languages.

### Peculiarities of Creoles

Creoles represent a unique class of low-resource languages:
- Hundreds of millions of speakers worldwide (e.g., Haitian Creole, Papiamento, Sango)
- Emerged from colonial-era language contact, tracing back to European and African languages
- Relate to different family trees across diverse linguistic dimensions
- Under-researched in language technology, despite clear demands

### Existing Approaches in Adapter Methods

The two-step method proposed by Üstün et al. (2021):
1. Train monolingual denoising adapters for source and target languages respectively
2. Freeze all parameters and fine-tune only the decoder's cross-attention (CA-FT)

Chronopoulou et al. (2023a) proposed adapter souping—averaging multiple domain adapters in the weight space.

The innovation of this study lies in **combining source/target language adapters, cross-attention fine-tuning, and souping**.

## Method

### Experimental Design

**Target Languages**: Haitian Creole (hat), Papiamento (pap), Sango (sag)

**Five strategies for choosing transfer languages**:

1. **Phylogeny**:
    - Indo-European (IE) relatives: French, Spanish, Portuguese, etc.
    - Niger-Congo (NC) relatives: Yoruba, Wolof, etc.
2. **Inter-Creole Transfer**: Commonalities shared among Creoles
3. **Typological Feature Transfer** (lang2vec): Distances based on syntactic features
4. **Model Representation Transfer** (NLLB representations): Language embedding similarity from NLLB-200
5. **Subword Evenness Transfer** (SuE): Uniformity of subword tokenization length

### Control Experiments

- **Unrelated Language Group**: Uralic, Dravidian, CJK (Chinese, Japanese, Korean)
- **Random Adapter**: Untrained, randomly initialized adapters (init)
    - Replacing Creole adapters
    - Souping with Creole adapters

### Implementation Details

- **Base Model**: Distilled 600M version of NLLB-200 (12-layer encoder/decoder, 16 attention heads, 1024 dimensions)
- **Adapter**: Train bottleneck adapters on 10K monolingual data from MADLAD-400
- **CA-FT**: Fine-tune cross-attention of the decoder using 10K parallel data from NLLB-OPUS
- **Evaluation**: Evaluate BLEU and chrF on FLORES-200

### Adapter Souping Formula

$$\theta_{soup} = \frac{1}{l} \sum_{i=1}^{l} \theta_i$$

When souping with init adapters, the weight ratio of the Creole adapter to the init adapter is 1:3 (simulating souping with three other adapters).

## Key Experimental Results

### Main Results (BLEU, Creole → English)

| Experimental Condition | hat→eng | pap→eng | sag→eng |
|----------|---------|---------|---------|
| Base Model (CA-FT) | 33.37 | 38.97 | 10.89 |
| s and t Adapters | 32.33 | 40.04 | 11.40 |
| **Untrained s Adapter** | **37.07** | **45.01** | **14.91** |
| IE Transfer | 36.44 | 46.35 | 12.46 |
| NC Transfer | 36.06 | 46.69 | 12.29 |
| Creole Transfer | 35.25 | 46.23 | 12.76 |
| lang2vec | 36.54 | 47.04 | 13.07 |
| NLLB Vec | 35.80 | 46.91 | 12.80 |
| SuE | 36.36 | 47.03 | 13.12 |
| **Untrained Souping** | **37.42** | **46.34** | **13.41** |
| Uralic | 37.06 | 47.00 | 13.58 |
| CJK | 36.41 | 47.17 | 13.33 |
| Dravidian | 36.55 | 47.27 | 13.27 |

### Key Findings

**Choice of transfer language is inconsequential**—two key pieces of evidence:

1. **Unrelated languages ≈ Principled choices**: Performance of the Uralic, Dravidian, and CJK control groups is on par with transfer languages chosen based on phylogeny, typology, or model representations.
2. **Untrained adapter ≈ Trained adapter**: Randomly initialized adapters perform comparably to, or even better than, "meaningful" language adapters.

→ **Conclusion**: The gains of adapters originate from the regularization effect, rather than cross-lingual information transfer.

### Catalan Validation Experiments

To rule out the idiosyncrasies of Creoles, validation was conducted on Catalan (which has clear close relatives: Spanish, Portuguese, Occitan):

| Experiment | 800 samples | 10K samples |
|------|----------|---------|
| Base Model (CA-FT) | 45.45 | 45.53 |
| s and t Adapters | 38.58 | 41.92 |
| spa+por+oci Souping | 41.87 | 43.74 |
| **Untrained Souping** | **43.97** | **44.75** |

Even for a language with clear close relatives, all adapter methods fail to outperform the baseline, and random adapters come closest to recovering baseline performance—further supporting the regularization hypothesis.

### Analysis of Regularization Evidence

1. **Gradient Norm and Validation Loss**: CA-FT without regularization exhibits higher gradient norms and validation losses, consistent with overfitting.
2. **Parameter Variance**: Parameter variance of souped adapters is significantly lower than that of individually pre-trained Creole adapters (Figure 2), supporting the regularization effect.
3. **Human Evaluation**: Manual evaluation of 33 samples by native Haitian speakers shows that Untrained Souping outperforms IE Transfer in terms of grammatical errors.

## Highlights & Insights

1. **Counter-intuitive Core Finding**: Linguistic information in adapters may be irrelevant for cross-lingual transfer learning—presenting a significant challenge to the NLP community's understanding of language transfer.
2. **Regularization Perspective**: Reinterprets adapter souping as a regularizer that adds noise, echoing classic regularization techniques like dropout and noise injection.
3. **Rigorous Experimental Design**: Six principled transfer strategies + three unrelated language controls + random adapter baseline + Catalan cross-validation; multi-layered validation lends high credibility to the conclusions.
4. **Practical Implications for Low-resource MT**: If the role of adapters is regularization, practitioners do not need to laboriously search for the "optimal transfer language" in practical applications; using random adapters suffices.
5. **Ethics Considerations**: Thoughtfully discusses the needs of Creole communities and the social impact of MT technology.

## Limitations & Future Work

1. **Only applicable to a few Creoles supported by NLLB-200**, making large-scale validation difficult.
2. **Training data primarily comes from religious domains** (Bible translations), which does not represent general language usage.
3. **Regularization hypothesis is hard to rigorously prove**: The authors acknowledge that a mathematically rigorous proof is beyond the scope of this paper.
4. **Small sample size**: The human evaluation consists of only 33 samples per condition, limiting statistical power.
5. **Only 600M distilled models are used**; behavior on larger models might differ.

## Related Work & Insights

- **Parameter-efficient Fine-tuning**: LoRA, $(IA)^3$, bottleneck adapters (Houlsby et al., 2019; Pfeiffer et al., 2020)
- **Cross-lingual Transfer**: Phylogenetic adapters (Faisal and Anastasopoulos, 2022; Chronopoulou et al., 2023b), transfer language selection (Pires et al., 2019; Pelloni et al., 2022)
- **Adapter Souping**: Weight-space averaging from Wortsman et al. (2022) used for domain adaptation
- **Creole NLP**: Exploration of Creole transfer learning by Lent et al. (2022a, 2024), Robinson et al. (2022, 2023)

## Rating

⭐⭐⭐⭐ (4/5)

The greatest value of this paper lies in its counter-intuitive finding: the cross-lingual transfer effect of adapters may stem purely from regularization rather than linguistic information. The experimental design is comprehensive and strictly controlled, supporting the core argument from multiple angles. It has direct guiding significance for low-resource MT practice. The primary limitations lie in the scale constraints (only 3 Creoles, 600M model) and the lack of a rigorous mathematical proof for the regularization hypothesis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Random Initialization of Gated Sparse Adapters (RIGSA)](../../ICML2025/model_compression/random_initialization_of_gated_sparse_adapters.md)
- [\[ICML 2025\] Neutral Residues: Revisiting Adapters for Model Extension](../../ICML2025/model_compression/neutral_residues_revisiting_adapters_for_model_extension.md)
- [\[ICML 2025\] Come Together, But Not Right Now: A Progressive Strategy to Boost Low-Rank Adaptation](../../ICML2025/model_compression/come_together_but_not_right_now_a_progressive_strategy_to_boost_low-rank_adaptat.md)
- [\[NeurIPS 2025\] A*-Thought: Efficient Reasoning via Bidirectional Compression for Low-Resource Settings](../../NeurIPS2025/model_compression/a-thought_efficient_reasoning_via_bidirectional_compression_for_low-resource_set.md)
- [\[NeurIPS 2025\] REP: Resource-Efficient Prompting for Rehearsal-Free Continual Learning](../../NeurIPS2025/model_compression/rep_resource-efficient_prompting_for_rehearsal-free_continual_learning.md)

</div>

<!-- RELATED:END -->
