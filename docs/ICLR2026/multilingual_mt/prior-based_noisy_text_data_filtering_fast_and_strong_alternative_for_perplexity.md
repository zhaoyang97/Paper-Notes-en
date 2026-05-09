---
title: >-
  [Paper Note] Prior-based Noisy Text Data Filtering: Fast and Strong Alternative For Perplexity
description: >-
  [ICLR 2026][Data Filtering] This paper proposes a text data filtering method based on token term frequency priors, detecting anomalous documents by computing the mean and standard deviation of token priors within each document. The approach achieves over 1000× speedup compared to PPL-based filtering while delivering superior downstream performance.
tags:
  - ICLR 2026
  - Data Filtering
  - Pretraining Data Quality
  - Token Frequency Prior
  - Perplexity Alternative
  - Efficient Data Selection
date: 2026-05-08
content_hash: 1addbbabf1207b7e
---

# Prior-based Noisy Text Data Filtering: Fast and Strong Alternative For Perplexity

**Conference**: ICLR 2026
**arXiv**: [2509.18577](https://arxiv.org/abs/2509.18577)
**Code**: [https://github.com/ybseo-ac/prior_filter](https://github.com/ybseo-ac/prior_filter)
**Area**: Multilingual Translation
**Keywords**: Data Filtering, Pretraining Data Quality, Token Frequency Prior, Perplexity Alternative, Efficient Data Selection

## TL;DR
This paper proposes a text data filtering method based on token term frequency priors, detecting anomalous documents by computing the mean and standard deviation of token priors within each document. The approach achieves over 1000× speedup compared to PPL-based filtering while delivering superior downstream performance.

## Background & Motivation
LLM pretraining relies on massive web corpora, which are heavily noisy and require careful filtering. The evolution of data filtering methods follows: rule-based → classifier-based (FastText) → n-gram-based (DSIR) → **PPL-based filtering** (current SOTA).

PPL-based filtering trains a small reference model (137M), runs inference over the entire corpus to obtain PPL scores, and discards samples that deviate most from the median. Despite being state-of-the-art, it suffers from two critical limitations:

**High computational cost**: Processing 6B tokens requires 216 GPU hours (reference model training + inference), which is highly uneconomical given the ever-growing scale of web data.

**Poor reliability**: Small models yield inaccurate PPL estimates for noisy or OOD data, potentially misclassifying repetitive or patterned noise as valid text.

The core insight is that PPL can be decomposed under a Bayesian framework as $\log \text{PPL} \propto \pi_{\text{likelihood}} + \pi_{\text{prior}}$, where the prior term $p(x_i)$ is context-independent and can be approximated by simple token frequency statistics, bypassing model inference entirely. This is further motivated by two linguistic observations: (1) token frequency serves as a one-dimensional representation of lexical role (high-frequency → function words; low-frequency → content words), and (2) well-formed sentences exhibit stable lexical density. Together, these observations support using statistical properties of token priors to detect anomalous documents.

## Method

### Overall Architecture
1. Estimate token priors from the corpus: $p_{\text{prior}}(x) = \frac{f_D(x)}{\sum_{x' \in V} f_D(x')}$
2. Compute two document-level statistics: prior mean $\mu_d$ and prior standard deviation $\sigma_d$
3. Filter anomalous documents by their deviation from the corpus median

### Key Designs

1. **Three-Region Analysis of Token Priors**: Sorting tokens by frequency reveals three distinct clusters — high-frequency (function words such as "the", "is"), mid-frequency (content words such as "Phone", "tackles"), and low-frequency (noise tokens such as garbled text and non-dominant-language characters). This empirically validates the linguistic hypothesis that token frequency is a one-dimensional encoding of lexical role.

2. **Dual-Metric Filtering Mechanism**:

   - **Prior mean** $\mu_d = \mathbb{E}_{x_i \in d}[\log p_{\text{prior}}(x_i)]$: Reflects the overall token composition of a document; abnormally high values indicate documents consisting entirely of function words (e.g., blank lines), while abnormally low values indicate documents dominated by noise characters.
   - **Prior standard deviation** $\sigma_d = \text{std}_{x_i \in d}[p_{\text{prior}}(x_i)]$: Reflects the uniformity of the token distribution; abnormally low values indicate documents with homogeneous token types (e.g., noun lists lacking sentence structure).
   - The two metrics capture complementary anomaly types: $\mu_d$ corresponds to the $\pi_{\text{prior}}$ term in PPL, while $\sigma_d$ approximates the structural regularity captured by $\pi_{\text{likelihood}}$.

3. **Multilingual Adaptability**: When a small proportion of Chinese text is mixed into an English corpus (<1%), it is almost entirely classified as anomalous; when the proportion exceeds 20%, the prior filter automatically treats Chinese as a valid language. This property requires no manually specified reference dataset, unlike DSIR.

4. **Subsampling Acceleration**: Estimating token frequencies from only 1% of the corpus yields an anomaly set nearly identical to that obtained from the full corpus, requiring approximately 70 seconds in total.

### Filtering Pipeline
The method computes $\delta_\mu(d) = |\mu_d - M_\mu|$ and $\delta_\sigma(d) = |\sigma_d - M_\sigma|$ (where $M$ denotes the corpus median), discards documents with the largest deviations under the constraint $|F_\mu| = |F_\sigma|$, and retains 50% of the data.

## Key Experimental Results

### Main Results (Dolma Corpus, 20 Downstream Benchmarks)

| Method | Type | Time | Avg | World Knowledge | Commonsense | Language | Symbolic | Reading |
|---|---|---|---|---|---|---|---|---|
| No-filter | Rule | - | 5.78 | 5.52 | 0.44 | 6.14 | 13.22 | 3.59 |
| FastText | Classifier | 3.6 hrs | 7.09 | 6.71 | 6.11 | 6.89 | 11.93 | 3.82 |
| DSIR | n-gram | 4 hrs | 7.56 | 7.03 | 6.84 | 7.31 | 12.67 | 3.97 |
| PPL-based | Model | **216 GPU hrs** | 8.22 | 9.98 | 11.91 | 7.34 | 7.91 | 3.96 |
| **Prior-based (Ours)** | Statistical | **0.25 hrs** | **9.20** | 9.53 | 11.27 | 10.31 | 11.13 | 3.79 |

Results with a 1.5B model show that the prior-based method achieves the highest average score of 9.20 in only 0.25 hours, outperforming PPL-based filtering (8.22, 216 GPU hours) in both speed and performance.

### Ablation Study (Symbolic Language: Pile-github)

| Method | Time | Avg | CS Algo | Dyck | Operators | Math QA | GSM8k | SVAMP |
|---|---|---|---|---|---|---|---|---|
| No-filter | - | 9.51 | 35.75 | 12.30 | 5.71 | 1.15 | 0.15 | 2.00 |
| PPL-based | 224 GPU hrs | 11.21 | 37.42 | 20.60 | 7.14 | 2.09 | 0.00 | 0.00 |
| **Prior-based** | **0.26 hrs** | **12.03** | 38.86 | 21.30 | 9.04 | 1.17 | 0.15 | 1.67 |

The method remains effective on symbolic languages such as code and mathematics, validating the generalization of the linguistic prior across language types.

### Key Findings
- The overlap between anomaly sets identified by PPL-based and prior-based filtering is approximately 50% (at a filtering ratio of $e=0.10$), confirming that the prior serves as an effective approximation to PPL.
- PPL-based filtering performs worst on symbolic problem-solving tasks, as small models produce unstable likelihood estimates for code and mathematical data.
- Prior-based filtering consistently outperforms PPL at larger scales (3B model, 12B tokens).
- Using only 1% subsampling yields nearly identical filtering results, with a total runtime of approximately one minute.

## Highlights & Insights
- **Simplicity outperforms complexity**: Replacing expensive model inference with the most elementary token frequency statistics yields superior results, revealing that the instability of the likelihood term in PPL is a liability rather than an asset.
- The method requires no model, no GPU, and can complete filtering decisions over trillion-token corpora in minutes using only CPUs.
- It supports incremental updates: when new data arrives, only the token frequency table needs to be updated, with no need to retrain a reference model.
- The linguistic motivation (Al-Kindi frequency analysis → lexical density) is elegant and theoretically well-grounded.
- The multilingual adaptability is of significant practical value, enabling unsupervised handling of mixed-language data.
- The methodological presentation is exceptionally clear: the derivation from the Bayesian decomposition of PPL forms a complete and coherent logical chain.

## Limitations & Future Work
- The method relies on linguistic properties and is **not applicable to non-textual modalities such as images** (as stated by the authors).
- Rare but valuable documents (e.g., papers containing uncommon technical terminology) may be incorrectly discarded, requiring manual supplementation of the corresponding subsets.
- The chunk size is fixed at 512 tokens; adaptability to very short or very long documents is not discussed.
- Validation is conducted exclusively with the GPT-2 tokenizer; robustness across other tokenizers warrants further investigation.

## Related Work & Insights
- **Relation to PPL-based filtering** (Ankner et al., 2024): Prior-based filtering approximates PPL while bypassing the unstable likelihood term.
- **Comparison with DSIR**: DSIR requires a manually specified reference dataset, whereas prior-based filtering is fully self-adaptive.
- The method is well-suited for online data selection in continual pretraining, given that token frequency estimates can be updated incrementally.
- The approach offers broader methodological inspiration: replacing costly model inference with simple statistical features for data quality assessment.
- The work raises a wider open question: what should the "gold standard" for data quality estimation be? PPL may not be the optimal answer.

## Rating
- Novelty: ⭐⭐⭐⭐ Frequency analysis is not new, but the Bayesian decomposition perspective and dual-metric design demonstrate genuine insight.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 20 benchmarks + symbolic language + multilingual + large-scale validation + efficiency comparison — extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain from linguistic motivation → method derivation → experimental validation is compact and elegant.
- Value: ⭐⭐⭐⭐⭐ 1000× speedup with better performance delivers direct engineering value for LLM pretraining data pipelines.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] ASSESS: A Semantic and Structural Evaluation Framework for Statement Similarity](assess_a_semantic_and_structural_evaluation_framework_for_statement_similarity.md)
- [\[ICLR 2026\] Multilingual Routing in Mixture-of-Experts](multilingual_routing_in_mixture-of-experts.md)
- [\[ICLR 2026\] ATLAS: Adaptive Transfer Scaling Laws for Multilingual Pretraining, Finetuning, and Decoding the Curse of Multilinguality](atlas_adaptive_transfer_scaling_laws_for_multilingual_pretraining_finetuning_and.md)
- [\[ICLR 2026\] SASFT: Sparse Autoencoder-guided Supervised Finetuning to Mitigate Unexpected Code-Switching in LLMs](sasft_sparse_autoencoder-guided_supervised_finetuning_to_mitigate_unexpected_cod.md)
- [\[AAAI 2026\] STELLAR: Scene Text Editor for Low-Resource Languages and Real-World Data](../../AAAI2026/multilingual_mt/stellar_scene_text_editor_for_low-resource_languages_and_real-world_data.md)

<!-- RELATED:END -->
