---
title: >-
  [Paper Note] ATLAS: Adaptive Transfer Scaling Laws for Multilingual Pretraining, Finetuning, and Decoding the Curse of Multilinguality
description: >-
  [ICLR 2026][scaling laws] This paper proposes the Adaptive Transfer Scaling Law (ATLAS), which decomposes effective data volume into three components—target language, transfer languages, and other languages—and introduces a data repetition saturation function. Evaluated across 774 multilingual training experiments (10M–8B parameters, 400+ languages), ATLAS substantially outperforms existing scaling laws, improving multilingual $R^2$ from 0.67 to 0.98, and systematically quantifies the cross-lingual transfer matrix, capacity constraints underlying the curse of multilinguality, and the computational crossover point between pretraining and finetuning.
tags:
  - ICLR 2026
  - scaling laws
  - multilingual
  - cross-lingual transfer
  - curse of multilinguality
  - pretraining vs finetuning
date: 2026-05-08
content_hash: dbe45fe9257be81f
---

# ATLAS: Adaptive Transfer Scaling Laws for Multilingual Pretraining, Finetuning, and Decoding the Curse of Multilinguality

**Conference**: ICLR 2026
**arXiv**: [2510.22037](https://arxiv.org/abs/2510.22037)
**Code**: Not released
**Area**: Multilingual Translation
**Keywords**: scaling laws, multilingual, cross-lingual transfer, curse of multilinguality, pretraining vs finetuning

## TL;DR
This paper proposes the Adaptive Transfer Scaling Law (ATLAS), which decomposes effective data volume into three components—target language, transfer languages, and other languages—and introduces a data repetition saturation function. Evaluated across 774 multilingual training experiments (10M–8B parameters, 400+ languages), ATLAS substantially outperforms existing scaling laws, improving multilingual $R^2$ from 0.67 to 0.98, and systematically quantifies the cross-lingual transfer matrix, capacity constraints underlying the curse of multilinguality, and the computational crossover point between pretraining and finetuning.

## Background & Motivation

### Limitations of Existing Scaling Laws
Scaling law research has focused almost exclusively on English. The Chinchilla Scaling Law (CSL) models the effect of model size $N$ and data volume $D$ on loss via two power-law terms, but suffers from several shortcomings:

**No support for data repetition**: Low-resource languages (e.g., Hindi, Swahili) have extremely limited data and require multiple training epochs; CSL cannot model the diminishing returns of repeated data.

**Cross-lingual transfer is ignored**: Monolingual scaling laws only account for target-language token counts and cannot leverage positive or negative transfer from other languages.

**Data-Constrained Scaling Law (DCSL)** accounts for data repetition but requires abundant observations both before and after the 1-epoch boundary for a two-stage fitting procedure. Collecting more than one epoch of data for high-resource languages (English, French) is costly, while low-resource languages may lack sufficient observations even before the first epoch.

### Practical Needs
Developers of multilingual models face three core questions that lack systematic answers:
- What are the transfer relationships among languages? Which language pairs are mutually beneficial, and which interfere?
- How much additional compute is required as the number of supported languages grows? (A quantitative characterization of the curse of multilinguality.)
- Given a fixed compute budget, is it more efficient to pretrain from scratch or to finetune from a multilingual checkpoint?

## Method

### Core Formula of ATLAS
ATLAS builds on Chinchilla's functional form but replaces the raw data volume $D$ with an **effective data volume** $\mathcal{D}_{\text{eff}}$:

$$\mathcal{L}(N, \mathcal{D}_{\text{eff}}) = E + \frac{A}{N^\alpha} + \frac{B}{\mathcal{D}_{\text{eff}}^\beta}$$

The effective data volume is decomposed into three terms:

$$\mathcal{D}_{\text{eff}} = \underbrace{\mathcal{S}_{\lambda_t}(D_t; U_t)}_{\text{Target language}} + \underbrace{\sum_{i \in \mathcal{K}} \tau_i \mathcal{S}_{\lambda_i}(D_i; U_i)}_{\text{Transfer languages}} + \underbrace{\tau_{\text{other}} \mathcal{S}_{\lambda_{\text{other}}}(D_{\text{other}}; U_{\text{other}})}_{\text{Other languages}}$$

### Saturation Function (Handling Data Repetition)
For each data source, a saturation function $\mathcal{S}_\lambda$ models the diminishing returns of repeated training passes:

$$\mathcal{S}_\lambda(D; U) = \begin{cases} D, & D \leq U \text{ (}\leq\text{1 epoch)} \\ U\left[1 + \frac{1 - \exp(-\lambda(D/U - 1))}{\lambda}\right], & D > U \text{ (}>\text{1 epoch)} \end{cases}$$

Here $U$ denotes the number of unique tokens in that language, and $\lambda$ is a shared repetition decay parameter. Data volume grows linearly within the first epoch and saturates exponentially beyond it.

### Cross-Lingual Transfer Matrix (38×38)
A Bilingual Transfer Score (BTS) is defined to measure the effect of a source language $s$ on a target language $t$:

$$\text{BTS}_{s \to t} = -\frac{\sigma_{\text{bi}}(L_t(d_{\text{mono}})) - 2d_{\text{mono}}}{d_{\text{mono}}}$$

where $d_{\text{mono}}$ is a preset target step count (42B tokens), and $\sigma_{\text{bi}}$ computes the number of tokens the bilingual model requires to reach the same loss. BTS = 0 indicates no transfer, BTS > 0 positive transfer, and BTS < 0 negative interference.

BTS values are measured for 80 language pairs and estimated for the remaining pairs using auxiliary training signals ($R^2 = 0.85$), yielding a complete $38 \times 38$ transfer matrix.

### Capacity Modeling for the Curse of Multilinguality
The per-target-language loss is modeled as a function of the number of languages $K$, model size $N$, and target-language data $D_t$:

$$L(K, N, D_t) = L_\infty + A \frac{K^\phi}{N^\alpha} + B \frac{K^\psi}{D_t^\beta}$$

Here $\phi > 0$ reflects capacity pressure as the number of languages increases, and $\psi < 0$ captures positive cross-lingual transfer (i.e., the required data per language grows sub-linearly). The formula reduces to Chinchilla when $K = 1$.

### Pretraining vs. Finetuning Crossover
Loss curves for training from scratch and for finetuning from a Unimax checkpoint are compared to identify a crossover point: pretraining from scratch surpasses finetuning after approximately 144B–283B tokens. The crossover point scales with model size $N$ as $C = 1113708 \times N^{1.65}$.

## Key Experimental Results

### Experimental Scale
- **774 independent training runs** on the MADLAD-400 dataset (400+ languages)
- Model scales: 10M–8B parameters across 20 size tiers
- 280 monolingual + 240 bilingual + 120 multilingual mixture + 134 finetuning models
- Vocabulary-insensitive loss evaluated on 48 languages

### Scaling Law Fit Quality (Table 1)

| Scaling Law | $R^2$ (Overall) | $R^2(N)$ | $R^2(D)$ | $R^2(C)$ | $R^2(M)$ |
|-------------|----------------|----------|----------|----------|----------|
| Chinchilla (multilingual) | 0.64 | -0.99 | 0.72 | 0.66 | 0.61 |
| Multilingual SL (He et al.) | 0.67 | -0.65 | 0.73 | 0.67 | 0.70 |
| **ATLAS (full)** | **0.98** | **0.89** | **0.96** | **0.98** | **0.82** |

ATLAS substantially outperforms prior methods across all generalization dimensions in the multilingual setting, most notably improving extrapolation $R^2(N)$ for the largest models from $-0.99$ to $0.89$.

### Key Findings on Cross-Lingual Transfer
- **English is the most broadly positive transfer source**, appearing among the top-5 most helpful source languages for 19 of 30 target languages.
- French (16/30), Spanish (13/30), and Hebrew (11/30) follow closely.
- **Shared writing system** correlates with higher transfer: mean BTS of $-0.23$ for same-script pairs vs. $-0.39$ for different-script pairs ($p < .001$).
- Transfer is **asymmetric**: global Pearson correlation $r = -0.11$, meaning "A helps B" does not imply "B helps A."
- Same-family, same-script pairs (e.g., French–Spanish, Russian–Ukrainian) exhibit high symmetry; cross-family, cross-script pairs (e.g., Chinese–Persian, Russian–Vietnamese) exhibit high asymmetry.

### Quantitative Results on the Curse of Multilinguality
- Fitted parameters: $\phi = 0.11$ (mild capacity curse) and $\psi = -0.04$ (slight positive transfer).
- **Compute budget for expanding language coverage**: scaling from $K$ to $r \cdot K$ languages requires scaling the compute budget by $C \cdot r^{0.97}$.
- Expanding to $4K$ languages requires a 2.74× increase in total tokens and a 1.4× increase in model size.
- Increasing model size $N$ mitigates the curse of multilinguality more effectively than increasing data volume $D$ ($|\partial S / \partial \log N| > |\partial S / \partial \log D|$).

### Pretraining vs. Finetuning
- For a 2B-parameter model, finetuning a Unimax checkpoint is more efficient up to 144B–283B tokens.
- Beyond this threshold, pretraining from scratch becomes superior.
- The crossover occurs earliest for English (due to its low 5% sampling ratio in Unimax); other languages exhibit crossovers at approximately 1.4%.

## Highlights & Insights

1. **Effective data decomposition as the key innovation**: Splitting multilingual training data into target-language, transfer-language, and other-language components—each with its own learned weight and saturation parameter—enables the model to precisely capture the contribution of each data source. This design is conceptually simple yet remarkably effective ($R^2$ improves from 0.67 to 0.98).
2. **Practical value of the transfer matrix**: BTS scores for 1,444 language pairs constitute the largest empirical resource of this kind and can directly guide language mixture strategies in multilingual training.
3. **Actionable formula for the curse of multilinguality**: The iso-loss formula for scaling from $K$ to $rK$ languages provides practitioners with a clear budget planning tool.
4. **Script family matters more than language family**: Shared writing systems have a stronger influence on transfer than shared linguistic family, suggesting that subword vocabulary overlap is the primary mechanism underlying positive transfer.
5. **Transfer asymmetry**: This finding cautions practitioners against intuitively assuming reciprocal transfer; empirical measurement is necessary.

## Limitations & Future Work

1. **Evaluation limited to perplexity**: All experiments measure vocabulary-insensitive loss only; the predictive power of the scaling law on downstream tasks (e.g., translation, question answering, classification) remains unvalidated.
2. **Single data source**: Only MADLAD-400 (CommonCrawl) is used; data from different domains or quality levels may alter transfer relationships.
3. **Uniform sampling assumption**: The curse-of-multilinguality model assumes uniform sampling across languages, whereas practical deployments typically require non-uniform allocation.
4. **Unimax checkpoint specificity**: The pretraining–finetuning crossover depends on the training mixture and duration of the Unimax checkpoint; different base models may yield different crossover points.
5. **Model-size dependence of the transfer matrix**: BTS values are measured on 2B-parameter models; transfer relationships may differ at other scales, though some analysis is provided.
6. **Underrepresentation of low-resource languages**: Despite data coverage of 400+ languages, in-depth analysis remains focused on approximately 50 languages.

## Related Work & Insights

| Method | Core Idea | Key Difference from ATLAS |
|--------|-----------|--------------------------|
| **Chinchilla** (Hoffmann 2022) | English monolingual $L = E + A/N^\alpha + B/D^\beta$ | No support for data repetition; no cross-lingual transfer modeling |
| **DCSL** (Muennighoff 2024) | Repetition-aware, two-stage fitting | Requires sufficient pre- and post-epoch observations; unfriendly to multilingual settings |
| **MSL** (He 2024) | Models multilinguality via language-family sampling ratios | Groups by language family only; ATLAS learns per-language transfer weights |
| **BiMix** (Ge 2024) | Bivariate data-mixture scaling law | Focuses on English domains; does not address multilinguality |
| **Llama-3** (Dubey 2024) | Briefly mentions multilingual scaling laws | Only 8% non-English tokens; far smaller in scale and depth than this work |

ATLAS's core advantages are: (1) unified single-stage fitting, (2) fine-grained cross-lingual transfer modeling, and (3) the largest multilingual scaling experiment conducted to date.

## Rating
- Novelty: ⭐⭐⭐⭐ The effective data decomposition combined with the saturation function is elegantly simple; the transfer matrix and curse-of-multilinguality modeling are both significant contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 774 experiments, 400+ languages, 10M–8B parameters—unprecedented in scale, with rigorous multi-dimensional generalization validation.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with complete derivations and highly informative figures.
- Value: ⭐⭐⭐⭐⭐ Directly actionable for engineering multilingual model training pipelines; the transfer matrix and iso-loss formula are immediately usable.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Zero-Shot Performance Prediction for Probabilistic Scaling Laws](../../NeurIPS2025/multilingual_mt/zero-shot_performance_prediction_for_probabilistic_scaling_laws.md)
- [\[ICLR 2026\] Multilingual Routing in Mixture-of-Experts](multilingual_routing_in_mixture-of-experts.md)
- [\[ICLR 2026\] SASFT: Sparse Autoencoder-guided Supervised Finetuning to Mitigate Unexpected Code-Switching in LLMs](sasft_sparse_autoencoder-guided_supervised_finetuning_to_mitigate_unexpected_cod.md)
- [\[NeurIPS 2025\] Enhancing Multilingual LLM Pretraining with Model-Based Data Selection](../../NeurIPS2025/multilingual_mt/enhancing_multilingual_llm_pretraining_with_model-based_data_selection.md)
- [\[ACL 2026\] Language on Demand, Knowledge at Core: Composing LLMs with Encoder-Decoder Translation Models for Extensible Multilinguality](../../ACL2026/multilingual_mt/language_on_demand_knowledge_at_core_composing_llms_with_encoder-decoder_transla.md)

<!-- RELATED:END -->
