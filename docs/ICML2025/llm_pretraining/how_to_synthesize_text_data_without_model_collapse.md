---
title: >-
  [Paper Note] How to Synthesize Text Data without Model Collapse?
description: >-
  [ICML 2025][LLM Pretraining][Model Collapse] Token-level Editing (ToEdit) is proposed, which performs local resampling at the token level on human data (instead of fully generating synthetic data). This is theoretically proven to guarantee a finite upper bound on the test error, thereby avoiding model collapse. Its effectiveness is empirically validated across three training stages: pre-training, continual pre-training, and fine-tuning.
tags:
  - "ICML 2025"
  - "LLM Pretraining"
  - "Model Collapse"
  - "Synthetic Data"
  - "Token Editing"
  - "Data Distribution"
  - "Semi-synthetic Data"
date: 2026-05-08
content_hash: f8aadca0d14d516a
---

# How to Synthesize Text Data without Model Collapse?

**Conference**: ICML 2025  
**arXiv**: [2412.14689](https://arxiv.org/abs/2412.14689)  
**Code**: [GitHub](https://github.com/Xuekai-Zhu/toedit)  
**Area**: LLM Pre-training  
**Keywords**: Model Collapse, Synthetic Data, Token Editing, Data Distribution, Semi-synthetic Data

## TL;DR

Token-level Editing (ToEdit) is proposed, which performs local resampling at the token level on human data (instead of fully generating synthetic data). This is theoretically proven to guarantee a finite upper bound on the test error, thereby avoiding model collapse. Its effectiveness is empirically validated across three training stages: pre-training, continual pre-training, and fine-tuning.

## Background & Motivation

With the ubiquity of generative AI, synthetic data on the internet is continuously increasing. Future GPT-{n} models will inevitably be trained on mixed datasets containing both synthetic and human data. Model collapse is the core risk in this scenario—iterative training on self-generated data leads to a progressive degradation in model performance.

This paper focuses on two key questions:

**What is the impact of synthetic data on language model pre-training?** The authors find that even non-iterative, direct mixing of synthetic data harms pre-training performance (non-iterative model collapse).

**How can data be synthesized without triggering model collapse?** The authors propose a token-level editing strategy to generate "semi-synthetic data" to circumvent distribution collapse.

Prior theoretical works (Shumailov et al., 2024; Dohmatob et al., 2024a) have proven that iterative training leads to a linear growth in test error: $E_{test} = \frac{\sigma^2 d}{T-d-1} \times n$. Gerstgrasser et al. (2024) demonstrated that data accumulation can break the collapse, but a practical data synthesis scheme is still lacking. The core innovation of this work is that, instead of "pure synthesis", it performs controlled token-level edits on the original human data while providing theoretical guarantees.

## Method

### Overall Architecture

The core mechanism of ToEdit can be divided into three steps:

1. **Infer token probabilities using a prior language model**: Given a human text sequence $x = (x_1, \dots, x_t)$, a pre-trained LM (e.g., Llama-3-8B) is used to compute the conditional probability $P(x_i | x_1, \dots, x_{i-1})$ for each token.
2. **Identify "overly easy" tokens**: If the conditional probability of a token $P(x_i | \text{context}) \geq p$ (where $p$ is a threshold), it indicates that the token is too easy for the model to predict, carries low information content, and poses a risk of over-concentration.
3. **Resampling and replacement**: These high-confidence tokens are resampled and replaced with new tokens $\tilde{x}_i$ sampled from the prior distribution, while other tokens are kept intact.

This process requires only a **single forward-pass inference** without involving autoregressive generation, which is highly computationally efficient (achievable on a single RTX 4090).

### Key Designs

#### Token Editing Formula

$$
x_i' = \begin{cases} x_i, & \text{if } P(x_i | x_1, \dots, x_{i-1}) < p \\ \tilde{x}_i, & \text{if } P(x_i | x_1, \dots, x_{i-1}) \geq p \end{cases}
$$

where $\tilde{x}_i$ is the token resampled according to the conditional probability distribution, and $p$ is the threshold controlling the editing intensity.

**Intuitive Interpretation**: The token probabilities in human text exhibit a U-shaped distribution—where both high-probability (easy to predict) and low-probability (difficult to predict) tokens are heavily concentrated. ToEdit only replaces the tokens on the high-probability end, preserving the long-tailed features of the low-probability end, thereby maintaining distribution coverage.

#### Why Not Use Pure Synthetic Data? — Four Findings

The authors reveal the fundamental flaws of synthetic data through systematic experiments:

- **Finding I**: Mixing synthetic data harms pre-training. On GPT-2 (124M), the synthetic data ratio is negatively correlated with performance—specifically, the average PPL of 100% synthetic data spikes from 20.99 to 51.93.
- **Finding II**: Synthetic data distribution lacks the long tail and has a narrowed coverage range. Estimating the PPL distribution with Llama-3-8B, human data covers [1, 100+], whereas synthetic data is concentrated only within [0, 14], covering only the top 25% of human data.
- **Finding III**: N-gram features of synthetic data are over-concentrated. Hashing uni-grams/bi-grams into 10,000 buckets, the responses of synthetic data are concentrated in a few buckets, lacking the broad coverage of human data.
- **Finding IV**: Data selection cannot correct the distribution shift. Even when using DSIR importance sampling to filter synthetic data, performance still fluctuates at the level of the original synthetic data and fails to align with the human data distribution.

#### Operational Matrix Formalization

In the theoretical analysis framework, token editing is formalized as operations with a diagonal matrix $M_i$:

$$
\tilde{Y}_n = M_{n-1} \hat{Y}_n + (1 - M_{n-1}) \tilde{Y}_{n-1}
$$

$M_i$ is a diagonal matrix, where diagonal elements are 0 or 1, determining which data points are edited (1) or preserved (0). This ensures that each iteration only modifies a portion of the data rather than replacing it entirely.

### Loss & Training

#### Theoretical Guarantee: Finite Upper Limit of Test Error

**Theorem 2**: Under the token editing setting, after $n+1$ rounds of iterative editing, the test error satisfies:

$$
E_{test}(\hat{w}_{n+1}) \leq \frac{2\sigma^2 d}{T - d - 1}
$$

This is a **finite upper bound independent of the iteration round $n$**. In contrast, the error in model collapse is $\frac{\sigma^2 d}{T-d-1} \times n$, which grows linearly with $n$.

Furthermore, if the editing operation satisfies the decay condition $\|M_i\| = \|M_{i-1}\| \eta$ ($\eta \in (0,1)$), the upper bound can be tighter:

$$
E_{test}(\hat{w}_{n+1}) \leq \frac{\sigma^2 d}{T-d-1} + \sigma^2 \sqrt{\mathbb{E}[\text{tr}((X^\top X)^{-2})]} \cdot \frac{\sqrt{\mathbb{E}[\text{tr}(M_1)]}}{1-\eta}
$$

**Core Logic**: Since only a portion of tokens is modified in each round (controlled by $M_i$), the original distribution coverage is preserved, and noise does not accumulate across iterations.

#### Implementation Details

- **Prior Model**: Llama-3-8B is used as the probability estimator.
- **Editing Threshold**: $p = 0.99$ (only replacing tokens with conditional probability $\geq 0.99$).
- **Sampling Strategy**: top-k ($k=8$).
- **Inference Engine**: vLLM fast inference, enabling data editing on a single RTX 4090.
- **No Autoregressive Generation**: Only a single forward pass is required.

## Key Experimental Results

### Main Results

The experiments cover three training stages: pre-training from scratch, continual pre-training, and supervised fine-tuning (SFT).

**Non-iterative model collapse (Pre-training GPT-2 124M from scratch)**:

| Data Ratio | Average PPL on 22 Subdomains ↓ | Comparison with Pure Human Data |
|---|---|---|
| 100% Human | 20.99 / 22.59 | Baseline |
| 25% Synthetic | 22.06 / 23.91 | +1.07 / +1.32 |
| 50% Synthetic | 23.48 / 25.09 | +2.49 / +2.50 |
| 75% Synthetic | 27.60 / 28.64 | +6.61 / +6.05 |
| 100% Synthetic | 51.93 / 47.87 | +30.94 / +25.28 |

**Continual Pre-training (Biomedicine Domain)**:

| Model | MQP | ChemProt | PubMedQA | RCT | USMLE | Avg |
|---|---|---|---|---|---|---|
| OLMo-1B Baseline | 52.59 | 17.2 | 51.40 | 32.70 | 28.90 | 36.63 |
| OLMo-1B CPT | 52.29 | 21.00 | 58.50 | 34.90 | 27.49 | 38.83 |
| OLMo-1B + ToEdit | **54.59** | **22.40** | **65.00** | 34.50 | 27.96 | **40.89** |
| Llama-3-8B Baseline | 66.80 | 28.59 | 60.8 | 73.85 | 40.61 | 54.13 |
| Llama-3-8B CPT | 72.29 | 29.4 | 69.1 | 72.65 | 36.76 | 56.04 |
| Llama-3-8B + ToEdit | **76.39** | **30.2** | 65.3 | 73.30 | 37.23 | **56.48** |

**SFT (Instruction Fine-tuning Llama-3-8B)**:

| Task | Original Avg | +ToEdit Avg | Gain |
|---|---|---|---|
| Natural Instructions | 69.34 | 69.70 | +0.36 |
| CoT | 69.01 | 69.26 | +0.25 |
| FLANv2 | 70.18 | 70.65 | +0.47 |
| Open Assistant | 69.19 | 69.44 | +0.25 |
| OSS-Instruct (Code) | 45.76 | 46.13 | +0.37 |
| Evol-Instruct (Code) | 46.62 | 46.92 | +0.30 |

### Ablation Study

| Configuration | Avg (Biomedicine) | Description |
|---|---|---|
| $p \geq 0.99$ | **38.69** | Default threshold, replacing approx 27% high-probability tokens |
| $p \geq 0.99$ | 38.48 | More conservative, replacing fewer tokens |
| $p \leq 0.1$ | 35.72 | Replacing low-probability tokens, poor performance |
| $p \leq 0.01$ | 37.46 | Replacing extremely-low-probability tokens |
| Top-k (k=8) | Baseline | Default, computationally efficient |
| Top-p | Comparable | Dynamic sampling range, higher overhead |
| Rejection Sampling | Comparable | Multi-round computation, highest overhead |
| k=8 vs k=64 | Small difference | Increasing k yields limited gains |

### Key Findings

1. **Clear negative correlation between synthetic data ratio and performance**: The PPL of 100% synthetic data is 2.5 times that of pure human data.
2. **ToEdit is consistently effective across three training stages**: Pre-training +0.36, continual pre-training averaging over +2, and SFT yielding +0.25~0.47.
3. **High-confidence token replacement outperforms low-confidence replacement**: $p \geq 0.99$ significantly outperforms $p \leq 0.1$, validating the design motivation—replacing "easy" tokens that the model has already mastered yields gains.
4. **U-shaped distribution of token probabilities**: Approximately 27.1% of tokens have probabilities in the range [0.9, 1.0), and 34.7% in [0.0, 0.1), showing concentration at both ends.
5. **Consistency between theory and practice**: The finite upper bound guarantees that multi-round iterations will not collapse, which is empirically confirmed by the experiments.

## Highlights & Insights

- **Precise Problem Definition**: It distinguishes between "iterative model collapse" and "non-iterative model collapse", with the latter being more realistic for actual training scenarios (i.e., direct mixing rather than iterative generation).
- **Highly Simple and Efficient Method**: It requires no autoregressive generation, no additional training, only a single forward pass paired with top-k sampling. It can be run on a single RTX 4090, translating to exceptionally low engineering costs.
- **Solid Theoretical Foundation**: The upper bound of test error is rigorously proven within a linear regression framework, reducing it from $O(n)$ to $O(1)$.
- **Inspiring Concept of "Semi-synthetic Data"**: Rather than "generating instances to replace human data", the core idea of "making minimal edits to human data" preserves distribution coverage while improving data quality.
- **Systematic and Robust Statistical Findings**: The four findings thoroughly explain why pure synthetic data fails from three dimensions: distribution, features, and data selection.

## Limitations & Future Work

1. **Theoretical Framework Limited to Linear Models**: Real-world LLMs are highly non-linear Transformers, leaving the generalizability of the upper bound proof from the linear regression framework to actual scenarios open to question.
2. **Limited Margin of Improvement**: The performance gain during the SFT stage is only 0.25~0.47 points, and the pre-training improvement is also modest (32.75→33.11), requiring further validation of its practical utility.
3. **Dependency on Strong Prior Models**: The method requires models of the caliber of Llama-3-8B to estimate token probabilities, which remains a barrier for resource-constrained scenarios.
4. **Choice of Threshold $p$**: Ablation studies show that performance fluctuates with different $p$ values, and there is a lack of an adaptive selection strategy.
5. **Lack of Empirical Validation on Multi-round Iterative Editing**: Although the theoretical proof guarantees the upper bound for multi-round iterations, the empirical experiments only evaluate single-round editing.

## Related Work & Insights

- **Model Collapse Theories**: Shumailov et al. (2024) and Dohmatob et al. (2024a,b,c) established the theoretical foundations; Gerstgrasser et al. (2024) proposed that data accumulation can break the collapse. This work advances these ideas by proposing a practical token-level editing scheme.
- **Synthetic Data Quality**: While efforts like Cosmopedia and the Phi series rely heavily on high-quality synthetic data, this paper reveals that the distribution deficiencies of pure synthetic data are fundamental.
- **Data Selection**: Methods such as DSIR attempt to filter data via importance sampling, but experiments in this paper show that this is ineffective against the distribution shift of synthetic data.
- **Insights**: The philosophy of "minimal editing" can be extended to other data augmentation scenarios—instead of generating entirely new data from scratch, making controlled, local modifications to original data may be superior.

## Rating

| Dimension | Score (1-5) | Description |
|---|---|---|
| Novelty | 4 | The concept of "token-level editing" rather than "pure synthesis" is both novel and practical |
| Theoretical Depth | 4 | Rigorous proof under the linear regression framework, though non-linear scenarios remain unproven |
| Experimental Thoroughness | 4 | Covers three training stages, multiple models, and multiple domains, with a complete ablation study |
| Practicality | 3.5 | Simple and deployable method, but the margin of improvement is relatively small |
| Writing Quality | 4 | Clear logic, with the four Findings progressing step-by-step |
| **Overall** | **4** | Solid work with an important problem, simple methodology, and comprehensive theory and experiments |

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] How Text Quality Interventions Reshape Neural Scaling Laws for LLMs: Empirical Study](../../ICLR2026/llm_pretraining/how_text_quality_interventions_reshape_neural_scaling_laws_for_llms_empirical_st.md)
- [\[ICML 2025\] Chameleon: A Flexible Data-mixing Framework for Language Model Pretraining and Finetuning](chameleon_a_flexible_data-mixing_framework_for_language_model_pretraining_and_fi.md)
- [\[ICLR 2026\] How to Train Data-Efficient LLMs](../../ICLR2026/llm_pretraining/how_to_train_data-efficient_llms.md)
- [\[NeurIPS 2025\] Neural Collapse under Gradient Flow on Shallow ReLU Networks for Orthogonally Separable Data](../../NeurIPS2025/llm_pretraining/neural_collapse_under_gradient_flow_on_shallow_relu_networks_for_orthogonally_se.md)
- [\[CVPR 2025\] MR-PLIP: Multi-Resolution Pathology-Language Pre-training Model with Text-Guided Visual Representation](../../CVPR2025/llm_pretraining/mr_plip_multi_resolution_pathology.md)

</div>

<!-- RELATED:END -->
