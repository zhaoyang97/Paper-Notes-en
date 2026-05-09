---
title: >-
  [Paper Note] First is Not Really Better Than Last: Evaluating Layer Choice and Aggregation Strategies in Language Model Data Influence Estimation
description: >-
  [ICLR 2026][LLM/NLP][Influence Functions] Through theoretical analysis and empirical experiments, this paper demonstrates that the widely accepted claim that "the first layer (embedding) is best suited for influence estimation" is unreliable. The work finds that intermediate attention layers are more effective, proposes two novel cross-layer aggregation strategies—Rank and Vote—along with a Noise Detection Rate (NDR) proxy metric, and achieves significant improvements in detecting harmful training samples in LLMs.
tags:
  - ICLR 2026
  - LLM/NLP
  - Influence Functions
  - Data Attribution
  - Layer Analysis
  - LLM Interpretability
  - Training Data Quality
date: 2026-05-08
content_hash: ea8a424163d6ea0e
---

# First is Not Really Better Than Last: Evaluating Layer Choice and Aggregation Strategies in Language Model Data Influence Estimation

**Conference**: ICLR 2026
**arXiv**: [2511.04715](https://arxiv.org/abs/2511.04715)
**Code**: Available
**Area**: LLM/NLP (Other)
**Keywords**: Influence Functions, Data Attribution, Layer Analysis, LLM Interpretability, Training Data Quality

## TL;DR
Through theoretical analysis and empirical experiments, this paper demonstrates that the widely accepted claim that "the first layer (embedding) is best suited for influence estimation" is unreliable. The work finds that intermediate attention layers are more effective, proposes two novel cross-layer aggregation strategies—Rank and Vote—along with a Noise Detection Rate (NDR) proxy metric, and achieves significant improvements in detecting harmful training samples in LLMs.

## Background & Motivation

**Background**: Influence functions are a key tool for measuring the impact of training data on model decisions (TracIn, DataInf, Cosine, etc.). Due to the massive parameter counts of modern LLMs, influence is typically computed over a subset of layers for practical feasibility.

**Limitations of Prior Work**: Yeh et al. (2022) concluded, based on the "cancellation effect" hypothesis, that the first layer (word embedding) is most suitable for influence estimation. However, this conclusion was validated only on small-scale models (RoBERTa) with a single method (TracIn), and the reliability of the cancellation effect itself has never been rigorously examined.

**Key Challenge**: The cancellation effect metric $C(W)$ aggregates the norms of parameter subsets to measure gradient cancellation, but this aggregation can mask extreme cancellation at individual parameters, making the metric an unreliable predictor of a layer's actual influence performance. Moreover, standard mean aggregation may reduce discriminative ability due to cancellation effects across layers.

**Goal**: (RQ1) Is the cancellation effect reliable? (RQ2) Which layers are best suited for influence estimation? (RQ3) How can influence scores be better aggregated across layers? (RQ4) Does a proxy metric exist that can evaluate influence methods without retraining?

**Key Insight**: The paper begins by theoretically constructing a counterexample to the cancellation effect, then conducts large-scale experiments across multiple models and datasets to systematically evaluate layer selection and aggregation strategies.

**Core Idea**: Intermediate attention layers are more suitable than embedding layers for influence estimation; Vote aggregation substantially outperforms mean aggregation; and NDR serves as a reliable proxy metric that requires no retraining.

## Method

### Overall Architecture

The input consists of a noisy training dataset, a validation dataset, and a pretrained LLM. The pipeline proceeds in five stages: (1) inject synthetic noise into the training data; (2) fine-tune on the noisy data and select the best checkpoint; (3) compute influence values across all trainable layers; (4) partition the model into WE (embedding), four groups of attention layers, and CL (classification head), and aggregate influence scores per training sample; (5) remove the 30% lowest-influence samples, retrain, and evaluate on test accuracy.

### Key Designs

1. **Theoretical Refutation of the Cancellation Effect**

    - *Function*: Proves that the cancellation effect proposed by Yeh et al. is an unreliable criterion for layer selection.
    - *Mechanism*: Theorem 5.1 constructs a counterexample showing that there exists a validation point $\bar{x}_3$ for which the influence score $\Delta I_{\theta,\omega}$ including high-cancellation weights $\omega$ achieves greater discriminability between noisy and clean samples than $\Delta I_{\theta}$ computed using only low-cancellation weights $\theta$. Spearman correlation analysis further reveals that $C$ is nearly uncorrelated with downstream performance ($\rho$ close to 0).
    - *Design Motivation*: Directly challenges a foundational assumption in the field, opening the door to a systematic reassessment of layer selection.

2. **Rank Aggregation Strategy**

    - *Function*: Replaces raw influence scores with rank-based aggregation across layers, eliminating the dominance of extreme values.
    - *Mechanism*: For each validation sample and each layer, training samples are ranked by influence score; the ranks are summed ($\operatorname{Rank}(I') = \sum_{x',l} \sum_{y} \mathbb{I}(I'(y,...) < I'(\cdot,...))$), considering only validation samples that are correctly predicted.
    - *Design Motivation*: In standard mean aggregation, large differences in influence magnitude across layers allow a few extreme values to dominate. Rank-based aggregation eliminates this scale discrepancy.

3. **Vote Aggregation Strategy**

    - *Function*: Each validation sample and layer votes for the $k$ lowest-ranked training samples, with vote counts decreasing by rank.
    - *Mechanism*: $\operatorname{Vote}_k(I') = -\sum_{x',l} \max(k - \operatorname{rank}, 0)$, where $k$ is set equal to the number of training samples to be filtered. Only samples at the very bottom of the ranking receive votes, avoiding interference from mid-ranked samples.
    - *Design Motivation*: The Rank method is still influenced by very low or very high rankings. Vote addresses this by truncating attention to only the most harmful samples. Experiments show $k \in [10, 50]$ yields the best performance.

4. **Noise Detection Rate (NDR) Proxy Metric**

    - *Function*: Proposes a proxy metric for evaluating influence methods without requiring retraining.
    - *Mechanism*: NDR measures the proportion of noisy samples among the bottom $k\%$ of training samples by influence rank. AUC measures the degree to which noisy samples are skewed toward the bottom of the ranking. Unlike the cancellation effect, which shows near-zero correlation with downstream performance, NDR achieves Spearman correlations of 0.5–0.9 with actual downstream performance.
    - *Design Motivation*: Avoids the expensive retraining experiments that would otherwise be required each time a new influence method is evaluated.

### Loss & Training

Standard cross-entropy fine-tuning with LoRA is used, with checkpoint selection at the lowest validation loss. Each configuration is repeated with 10 random seeds. Pairwise comparisons between configurations are conducted, with win rate and Pareto front analysis used for evaluation.

## Key Experimental Results

### Main Results

| Model | Best Layer | Worst Layer | Best Method + Layer | Win Rate |
|-------|-----------|-------------|---------------------|----------|
| RoBERTa-Large | Attn 18–23 | CL | DataInf + Attn 18–23 | 0.70 |
| Qwen-2.5 1.5B | Attn 07–13 | CL | Cosine + Attn 07–13 | Top-1 |
| Mistral 7B | Attn 08–15 | CL | DataInf + Attn 08–15 | 0.71 |
| Llama-3.2 1B | Attn 04–07 | CL | DataInf + Attn 04–07 | 0.64 |

The best layer outperforms the worst layer (CL) by 10–15% in post-filtering accuracy.

### Ablation Study

| Aggregation Strategy | Performance | Notes |
|---------------------|-------------|-------|
| Mean (baseline) | Baseline | Standard cross-layer mean |
| Rank | Moderate improvement | Eliminates scale discrepancy; outperforms Vote in some scenarios |
| Vote ($k$=10–50) | Significant improvement | TracIn CL win rate increases from 0.10 to top-1; DataInf 00–07 win rate reaches 0.84 |

| Proxy Metric | Correlation with Downstream Performance (Spearman $\rho$) |
|-------------|----------------------------------------------------------|
| Cancellation Effect $C$ | −0.3 to 0.2 (weak/none) |
| NDR@30% (Mean) | 0.4 to 0.7 (moderate to strong) |
| NDR@30% (Vote) | 0.8 to 0.9 (strong) |

### Key Findings
- **Intermediate attention layers consistently outperform both embedding layers and the classification head** across models and methods, refuting the conclusion of Yeh et al.
- **The classification head (CL) is the worst choice across all models**, likely because CL is overly sensitive to noise.
- **Vote aggregation can elevate originally poor configurations to top-1 performance** (e.g., TracIn CL on Mistral improves from rank 12 to rank 1).
- **DataInf and Cosine generally outperform TracIn**, particularly at intermediate layers of larger models.
- **Llama-3.2 1B is the most challenging model**—no influence method outperforms random filtering, possibly due to characteristics specific to that model's training.

## Highlights & Insights
- **Combining a theoretical counterexample with large-scale empirical evidence** to overturn a classic assumption is a methodologically exemplary approach: Theorem 5.1 shows that the assumption *can* fail, and experiments show that it *does* fail in practice.
- **Vote aggregation is remarkably simple yet highly effective**: by tuning a single hyperparameter $k$, originally unusable layers (such as CL) become competitive, suggesting that the bottleneck in influence estimation may lie in aggregation strategy rather than the influence method itself.
- **NDR as a proxy metric has substantial practical value**: researchers can rapidly assess the effectiveness of influence methods without retraining, significantly reducing experimental costs.
- Correlation analysis of inter-layer influence scores reveals three layer groups (early/middle/late), a structure consistent with findings from knowledge editing methods (ROME/MEMIT).

## Limitations & Future Work
- Experiments are conducted only on the GLUE benchmark; generative tasks and in-context learning settings are not explored.
- The complete failure of influence functions on Llama-3.2 1B is not convincingly explained.
- The hyperparameter $k$ for Vote requires tuning, and Vote degrades performance for the Cosine method.
- The noise injection scheme (label flipping) is relatively simple; more complex data quality issues such as backdoor attacks are not examined.
- Only LoRA fine-tuning is tested; behavior under full-parameter fine-tuning may differ.

## Related Work & Insights
- **vs. Yeh et al. (2022)**: Directly challenges their core conclusion. Yeh et al. derived the superiority of embedding layers from a small model and a single method; this paper demonstrates that intermediate layers are superior across 4 models, 5 methods, and 8 datasets.
- **vs. ROME/MEMIT**: Knowledge editing methods also identify intermediate MLP layers as encoding the most factual information, consistent with this paper's finding that intermediate layers yield the strongest influence signals—providing independent, complementary support for the hypothesis that "intermediate layers are most information-rich."
- **vs. Li et al. (2025)**: Li et al. find that influence functions perform poorly on LLMs, but use default settings (mean aggregation over all layers). This paper shows that selecting the right layers combined with Vote aggregation can substantially improve results.

## Rating
- Novelty: ⭐⭐⭐⭐ — Overturns a classical assumption; introduces Rank/Vote aggregation and the NDR proxy metric across multiple dimensions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 4 models × 5 methods × 8 datasets × 10 seeds; exceptionally comprehensive.
- Writing Quality: ⭐⭐⭐⭐ — RQ-driven structure is clear, though the paper is somewhat lengthy.
- Value: ⭐⭐⭐⭐ — Provides direct practical guidance for influence function research, particularly regarding layer selection and aggregation strategies.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Evaluating Text Creativity across Diverse Domains: A Dataset and Large Language Model Evaluator](evaluating_text_creativity_across_diverse_domains_a_dataset_and_large_language_m.md)
- [\[AAAI 2026\] Do Not Merge My Model! Safeguarding Open-Source LLMs Against Unauthorized Model Merging](../../AAAI2026/llm_nlp/do_not_merge_my_model_safeguarding_open-source_llms_against_unauthorized_model_m.md)
- [\[NeurIPS 2025\] Triplets Better Than Pairs: Towards Stable and Effective Self-Play Fine-Tuning for LLMs](../../NeurIPS2025/llm_nlp/triplets_better_than_pairs_towards_stable_and_effective_self-play_fine-tuning_fo.md)
- [\[ICLR 2026\] Weight Decay may matter more than μP for Learning Rate Transfer in Practice](weight_decay_may_matter_more_than_mup_for_learning_rate_transfer_in_practice.md)
- [\[ICLR 2026\] WebDevJudge: Evaluating (M)LLMs as Critiques for Web Development Quality](webdevjudge_mllm_web_development.md)

<!-- RELATED:END -->
