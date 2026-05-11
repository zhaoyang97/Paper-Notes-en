---
title: >-
  [Paper Note] Token-level Data Selection for Safe LLM Fine-tuning
description: >-
  [ICLR 2026][LLM Pretraining][LLM safety] This paper proposes TOSS (Token-level data Selection for Safe LLM fine-tuning), the first token-level data selection framework that evaluates the safety risk of each token via the…
tags:
  - "ICLR 2026"
  - "LLM Pretraining"
  - "LLM safety"
  - "fine-tuning safety"
  - "token-level selection"
  - "data curation"
  - "safety-utility tradeoff"
date: 2026-05-08
content_hash: a2de346953f28f0b
---

# Token-level Data Selection for Safe LLM Fine-tuning

**Conference**: ICLR 2026
**arXiv**: [2603.01185](https://arxiv.org/abs/2603.01185)
**Code**: [github.com/Polly-LYP/TOSS](https://github.com/Polly-LYP/TOSS)
**Area**: LLM Pre-training
**Keywords**: LLM safety, fine-tuning safety, token-level selection, data curation, safety-utility tradeoff

## TL;DR

This paper proposes TOSS (Token-level data Selection for Safe LLM fine-tuning), the first token-level data selection framework that evaluates the safety risk of each token via the loss difference between a safety-degraded reference model and a utility-oriented reference model, achieving a superior safety-utility tradeoff compared to sample-level methods.

## Background & Motivation

Fine-tuning LLMs on custom datasets is standard practice for domain adaptation, yet **the fine-tuning process can severely erode a model's safety alignment**. Existing defenses all operate at the **sample level**:

**Data mixing** (Bianchi et al., 2023): Injects safety data into the custom dataset, but excessive safety data causes the model to over-refuse.

**Sample filtering** (SEAL, Shen et al., 2024): Identifies and discards entire samples deemed unsafe, but discards valuable downstream task information in the process.

**Core finding**: Safety degradation is not a sample-level problem but a **token-level problem**. Token-level diagnostic analysis reveals that:
- The most pronounced distributional shift occurs at **the first few tokens of the response**—the model replaces safe refusal prefixes with prefixes that comply with harmful instructions.
- However, harm is not limited to initial tokens: intermediate and later tokens also exhibit significant deviation toward the safety-degraded model.
- Even superficially benign data can erode safety alignment at the token level.
- Simple fixed-position token masking (e.g., masking the first 5 tokens) improves safety but hurts utility.

A **fine-grained token-level selection mechanism** is therefore needed—one that accurately identifies and removes harmful tokens while retaining tokens critical for task adaptation.

## Method

### Overall Architecture

The TOSS framework consists of three stages: reference model training → token evaluation → token-level selective fine-tuning.

### Key Design 1: Reference Model Training

Two specialized reference models are constructed:

**Safety-degraded model** $f_{\theta^h}$: Trained on a harmful reference dataset $\mathcal{D}^h$ to learn harmful next-token prediction patterns:

$$\mathcal{L}_{f_{\theta^h}} = \frac{1}{\sum_{i=1}^H L_i} \sum_{i=1}^H \sum_{j=1}^{L_i} -\log P(y_{i,j}^h | \boldsymbol{x}_i^h, \boldsymbol{y}_{i,:j-1}^h; \theta)$$

**Utility-oriented model** $f_{\theta^u}$: Trained on a high-quality utility reference dataset $\mathcal{D}^u$ to learn the downstream task data distribution.

### Key Design 2: Token Evaluation

The core metric evaluates the safety risk of each token via a loss difference:

$$\mathcal{S}(y_{i,j}^{\text{cus}}) = -\log P(y_{i,j}^{\text{cus}}|\boldsymbol{x}_i^{\text{cus}}, \boldsymbol{y}_{i,:j-1}^{\text{cus}}; \theta^u) + \log P(y_{i,j}^{\text{cus}}|\boldsymbol{x}_i^{\text{cus}}, \boldsymbol{y}_{i,:j-1}^{\text{cus}}; \theta^h)$$

**Intuition**: A high-scoring token has high probability under the safety-degraded model (low loss) and low probability under the utility-oriented model (high loss), indicating a high safety risk.

The score decomposes into the sum of two competing components:
- **Utility-related score**: Measures alignment of the token with the desired task distribution.
- **Safety-related score**: Measures alignment of the token with harmful patterns.

### Key Design 3: Global Ranking and Token Masking

All tokens in the custom dataset are globally ranked, and the top $d \times 100\%$ highest-scoring tokens are discarded:

$$m_{i,j} = \begin{cases} 0 & \text{if } \mathcal{S}(y_{i,j}^{\text{cus}}) \text{ is in the global top } d\times100\% \\ 1 & \text{otherwise} \end{cases}$$

Global ranking outperforms intra-sample local ranking because the proportion of harmful tokens in harmful samples is non-uniform.

### Loss & Training

Token-level selective fine-tuning loss:

$$\mathcal{L}^{\text{cus}} = \frac{1}{\sum_{i=1}^N L_i} \sum_{i=1}^N \sum_{j=1}^{L_i} -m_{i,j} \log P(y_{i,j}^{\text{cus}} | \boldsymbol{x}_i^{\text{cus}}, \boldsymbol{y}_{j-1}^{\text{cus}}; \theta)$$

### TOSS-Pro: Progressive Refinement

Iteratively enhances the capability of the safety-degraded model:
1. Compute token-level scores using the current safety-degraded model $f_{\theta_t^h}$ and the fixed utility model $f_{\theta^u}$.
2. Select $k$ samples corresponding to the highest-scoring tokens and add them to the harmful dataset.
3. Update the safety-degraded model on the expanded harmful dataset $\mathcal{D}_{t+1}^h = \mathcal{D}_t^h \cup \mathcal{D}_t^s$.
4. Repeat for $T$ iterations; use the refined model for final token selection.

## Key Experimental Results

### Main Results

| Method | Llama-3-8B (HH / HEx-PHI / SLIMORCA / AVG) | Llama-2-7B (HH / HEx-PHI / SLIMORCA / AVG) |
|--------|---------------------------------------------|---------------------------------------------|
| Standard SFT | 50 / 50 / 50 / 50 | 50 / 50 / 50 / 50 |
| SafeInstr | 51.5 / 64.6 / 50.5 / 55.5 | 48.2 / 51.3 / 53.1 / 50.9 |
| DSIR | 67.4 / 60.8 / 53.8 / 60.7 | 63.7 / 57.0 / 52.0 / 57.6 |
| SEAL | 58.2 / 68.8 / 57.4 / 61.5 | 58.6 / 50.3 / 52.5 / 53.8 |
| **TOSS** | **88.8 / 87.5 / 68.4 / 81.6** | **83.2 / 69.9 / 57.3 / 70.1** |
| **TOSS-Pro** | **88.9 / 93.8 / 68.9 / 83.8** | **87.0 / 74.4 / 60.7 / 74.0** |

Compared to SEAL, TOSS achieves up to 30% improvement in safety and up to 11% improvement in utility. TOSS-Pro further improves safety by up to 6% over TOSS.

### Transferability Experiments

Token selections made on Llama-3-8B-Instruct are directly applied to Llama-3.2-1B/3B (sharing the same tokenizer):

| Method | Llama-3.2-1B AVG | Llama-3.2-3B AVG |
|--------|-------------------|-------------------|
| Standard SFT | 50 | 50 |
| SEAL | 56.3 | 53.7 |
| **TOSS** | **63.9** | **68.1** |

Token-level selection needs to be performed only once and can be reused across models sharing the same tokenizer.

### Ablation Study

| Ablation | Finding |
|----------|---------|
| Global vs. local ranking | Global ranking is significantly superior; the proportion of harmful tokens in harmful samples is non-uniform. |
| Token-level vs. sample-level | Token-level selection outperforms sample-level on both safety and utility. |
| Safety-degraded model only | Safety improves but utility drops substantially—tokens critical for task adaptation are discarded. |
| Utility-oriented model only | Utility is acceptable but safety shows no improvement—harmful tokens cannot be identified. |
| Random vs. metric-based sample selection (TOSS-Pro) | Random selection is ineffective or even degrades performance; precise selection of informative samples is essential. |
| Number of TOSS-Pro iterations | 1–2 iterations yield consistent improvement in safety performance. |

### Key Findings

1. **Safety degradation is a token-level problem**: Harmful and beneficial signals are interleaved within the same sample.
2. **The complementarity of the two reference models is essential**: Removing either one leads to significant degradation in safety or utility.
3. **Global ranking outperforms local ranking**: Harmful tokens are highly unevenly distributed across samples.
4. **Progressive refinement outperforms one-shot selection**: Iterative selection of higher-quality harmful samples continuously improves identification precision.

## Highlights & Insights

1. **"The fundamental unit of safety degradation is the token, not the sample"**—this core hypothesis is thoroughly validated through diagnostic analysis and represents a key methodological breakthrough.
2. **The loss-difference metric elegantly unifies safety and utility into a single objective**: high score = the safety-degraded model "prefers" the token + the utility model "dislikes" it = should be discarded.
3. **The progressive refinement in TOSS-Pro exploits a bootstrapping effect**: a better safety-degraded model → more accurate token identification → higher-quality harmful samples → an even better safety-degraded model.
4. **Cross-tokenizer transferability** gives the method significant practical value—token selection is performed once on a large model and directly reused by smaller models.

## Limitations & Future Work

1. **Requires constructing additional harmful and utility reference datasets**: Although the required amount is small (~10%), domain knowledge is still needed.
2. **The token discard ratio $d$ is fixed at 0.1**: Different datasets may require different ratios.
3. **Training the safety-degraded model** raises inherent ethical considerations, as it requires explicitly training a "harmful" model.
4. **Evaluation relies on GPT-4o as judge**: This may introduce evaluation bias.
5. **Experiments are validated only on the Llama family**: Other architectures such as Mistral and Qwen are not tested.
6. **Differences across harmful content types are not discussed**: Token-level characteristics may vary across safety categories.

## Related Work & Insights

- **SEAL** (Shen et al., 2024): Sample-level data selection baseline and the direct predecessor that TOSS improves upon.
- **SafeInstr** (Bianchi et al., 2023): Data mixing approach.
- **DSIR** (Xie et al., 2023): Importance resampling-based sample selection.
- **TokenTune** (Simoulin et al., 2024): Token-level activation pruning (focused on efficiency rather than safety).
- **DPO/RLHF**: Training-stage safety alignment methods, complementary to TOSS.

The core insight from TOSS: **the granularity of data curation determines the upper bound of the safety-utility tradeoff**. The substantial performance gains achieved by moving from sample-level to token-level granularity suggest that future work may benefit from pushing further to sub-token or semantic unit levels.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First work to systematically diagnose and address fine-tuning safety degradation at the token level.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-model, multi-benchmark evaluation with comprehensive ablations and transferability validation.
- Writing Quality: ⭐⭐⭐⭐ — Clear logical structure with a complete loop from diagnostic analysis to method design to experimental validation.
- Value: ⭐⭐⭐⭐⭐ — Establishes a new paradigm for safe fine-tuning, substantially outperforms existing methods, and releases code publicly.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning](pre-training_llm_without_learning_rate_decay_enhances_supervised_fine-tuning.md)
- [\[ICLR 2026\] Common Corpus: The Largest Collection of Ethical Data for LLM Pre-Training](common_corpus_ethical_data_for_llm_pretraining.md)
- [\[ICLR 2026\] Understanding the Emergence of Seemingly Useless Features in Next-Token Predictors](understanding_the_emergence_of_seemingly_useless_features_in_next-token_predicto.md)
- [\[ICLR 2026\] A Law of Data Reconstruction for Random Features (and Beyond)](a_law_of_data_reconstruction_for_random_features_and_beyond.md)
- [\[ICLR 2026\] Predicting Training Re-evaluation Curves Enables Effective Data Curriculums](predicting_training_re-evaluation_curves_enables_effective_data_curriculums_for_.md)

</div>

<!-- RELATED:END -->
