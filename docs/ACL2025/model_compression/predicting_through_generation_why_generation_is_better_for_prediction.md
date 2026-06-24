---
title: >-
  [Paper Note] Predicting Through Generation: Why Generation Is Better for Prediction
description: >-
  [ACL2025][Model Compression][PredGen] This paper proves from an information-theoretic perspective that token-level generation retains more mutual information than pooled representations. It proposes the PredGen framework, which addresses the exposure bias and format mismatch issues in generative prediction through scheduled sampling and a task adapter. Additionally, a Writer-Director Alignment Loss is designed to unify the generation and prediction objectives.
tags:
  - "ACL2025"
  - "Model Compression"
  - "PredGen"
  - "Generation for Prediction"
  - "Data Processing Inequality"
  - "Exposure Bias"
  - "Task Adapter"
date: 2026-05-08
content_hash: 2a8bcf51edd7e33c
---

# Predicting Through Generation: Why Generation Is Better for Prediction

**Conference**: ACL2025  
**arXiv**: [2502.17817](https://arxiv.org/abs/2502.17817)  
**Code**: [GitHub](https://github.com/Kowsher/PredGen)  
**Area**: Other  
**Keywords**: PredGen, Generation for Prediction, Data Processing Inequality, Exposure Bias, Task Adapter  

## TL;DR

This paper proves from an information-theoretic perspective that token-level generation retains more mutual information than pooled representations. It proposes the PredGen framework, which addresses the exposure bias and format mismatch issues in generative prediction through scheduled sampling and a task adapter. Additionally, a Writer-Director Alignment Loss is designed to unify the generation and prediction objectives.

## Background & Motivation

### Problem Background
LLMs have demonstrated powerful capabilities in NLP tasks. However, in prediction tasks (classification, regression), they typically employ pooled representations (e.g., [CLS] token or mean pooling) combined with a classification head. This pooling operation irreversibly discards positional and sequential information, limiting the model's ability to capture fine-grained dependencies.

### Core Motivation
- LLMs are inherently pre-trained via next-token prediction, aligning the generative paradigm naturally with their learning objectives.
- The pooling operation is a deterministic compression, which inevitably leads to information loss according to the Data Processing Inequality (DPI).
- Reformulating prediction tasks as generation tasks can retain more target-related mutual information.
- However, directly utilizing generation for prediction faces two major challenges: exposure bias and format mismatch.

### Value
This work presents a concise yet profound perspective—"generation is better than prediction"—with a rigorous proof from information theory, while solving practical problems through engineering innovations.

## Method

### Theoretical Foundation: Why Generation Is Better

**Theorem 1 (Mutual Information Does Not Increase Under Deterministic Compression)**: Let $\mathbf{X}$ be the input, $\mathbf{Z}$ be the hidden representation, $\mathbf{Z_p} = g(\mathbf{Z})$ be the pooled representation (where $g$ is a deterministic function such as first-token or mean pooling), and $\mathbf{Y}$ be the target, then:

$$I(\mathbf{Y}; \mathbf{Z}) \geq I(\mathbf{Y}; \mathbf{Z_p})$$

**Proof Core**: Based on the Data Processing Inequality (DPI), a deterministic function does not increase mutual information. Since $\mathbf{Z_p}$ is derived from $\mathbf{Z}$ via a deterministic process, the conditional entropy satisfies $H(\mathbf{Y}|\mathbf{Z}) \leq H(\mathbf{Y}|\mathbf{Z_p})$, and therefore the mutual information satisfies $I(\mathbf{Y}; \mathbf{Z_p}) \leq I(\mathbf{Y}; \mathbf{Z})$.

**Empirical Validation**: Using the MINE method to estimate mutual information, it is verified across multiple datasets that PredGen consistently retains higher mutual information. Token-level mutual information analysis shows that the predicted token (e.g., "positive") is highly correlated with semantically relevant words (e.g., "funny" 0.47, "pretty" 0.34).

### PredGen Framework

#### 1. Reformulating Prediction as Generation
The target $\mathbf{P}$ (e.g., 13.4) of a prediction task is represented as a token sequence $\mathbf{Y} = ['1','3','.','4']$, which the model generates autoregressively:
$$P(\mathbf{Y}|\mathbf{X};\theta) = \prod_{t=1}^m P(Y_t|\mathbf{X}, \mathbf{Y}_1, ..., \mathbf{Y}_{t-1}; \theta)$$

#### 2. Scheduled Sampling to Solve Exposure Bias
Core problem: Standard training conditions on ground-truth tokens, whereas inference relies on the model's own generation, causing small errors to accumulate.

Solution: Use the model's own predicted token (instead of the ground-truth) with probability $p$, where $p$ is gradually increased during training:
$$\tilde{\mathbf{Y}} = \begin{cases} \mathbf{Y} & \text{with probability } (1-p) \\ \tilde{\mathbf{Y}} & \text{with probability } p \end{cases}$$

#### 3. Task Adapter to Solve Format Mismatch
The generator produces discrete tokens, but certain tasks require continuous values or structured outputs. A Task Adapter $\mathcal{T}$ maps the hidden representations of the generated tokens to the final prediction:
$$\hat{\mathbf{P}} = \mathcal{T}(\mathbf{Z}[n:n+m])$$
where $\mathbf{Z}[n:n+m]$ is the hidden representation corresponding to the generated tokens.

### Loss & Training: Writer-Director Alignment Loss (WDAL)

A "writer-director" analogy is adopted: the Writer (generator) is responsible for generating tokens, and the Director (task adapter) is responsible for mapping them to the task format.

$$L_{\text{WDAL}} = \max(L_W^2, L_D^2) \cdot \exp\left(-|\log L_W - \log L_D|\right)$$

- $L_W$: Writer Loss (cross-entropy loss, measuring generation quality)
- $L_D$: Director Loss (task-specific loss, measuring prediction precision)
- $\max(L_W^2, L_D^2)$: **Authority term**, focusing on the component with the larger error
- $\exp(-|\log L_W - \log L_D|)$: **Alignment penalty term**, ensuring that the two losses remain balanced

When there is a large discrepancy between the two losses, the penalty term decreases while the max term increases, keeping the total loss high and driving the model to improve its weaker components.

## Key Experimental Results

### Experimental Setup
- **Models**: Llama2-7B, Llama2-13B, Llama2-8B
- **PEFT Methods**: LoRA, AdaLoRA, RoCoFT, DoRA
- **Classification Datasets**: BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA (Metric: Accuracy)
- **Regression Datasets**: WASSA, SICK, STSB, LCP, CLEAR, Humicroedit (Metrics: MSE, MAE)

### Main Results (Classification Tasks, Llama2-7B + LoRA)

| Method | BoolQ | HellaSwag | WinoGrande | ARC-e | Average |
|------|-------|-----------|------------|-------|------|
| Predictor | 66.29 | 88.53 | 70.49 | 75.27 | 73.49 |
| Generator | 68.09 | 90.86 | 77.54 | 79.54 | 76.63 |
| **PredGen** | **73.82** | **93.14** | **83.21** | **84.79** | **79.67** |

Key Findings:
1. **PredGen consistently outperforms Predictor and Generator**: Across all model $\times$ PEFT combinations, PredGen improves average accuracy by approximately 6 percentage points (vs. Predictor).
2. **Generator outperforms Predictor**: This validates the theoretical claim that generation is superior to classification.
3. **Gains increase with model size**: On Llama2-13B, PredGen achieves an average of 82.71% (vs. Predictor 76.20%), showing a more pronounced improvement.
4. **Robustness across PEFT methods**: PredGen maintains its advantage under different PEFT methods.

### Regression Task Results
PredGen also consistently outperforms baselines across multiple regression benchmarks. Especially on tasks requiring precise numerical predictions (such as STS-B similarity scoring), the Task Adapter effectively bridges the gap between discrete tokens and continuous values.

### Mutual Information Verification
MINE estimation on datasets like SST-2 and PIQA shows:
- The mutual information between PredGen's hidden representations and the target > Generator > Predictor.
- This validates the theoretical predictions of DPI.

## Highlights & Insights

1. **Theoretical elegance**: The information-theoretic basis that generation outperforms pooling is rigorously proven using the Data Processing Inequality, making the argumentation concise and powerful.
2. **Ingenious WDAL loss design**: The Writer-Director analogy is intuitive, and the log-sum-exp stabilization technique effectively addresses numerical instability issues.
3. **End-to-end framework**: Scheduled sampling, task adapter, and alignment loss are seamlessly integrated without requiring modifications to the model architecture.
4. **Token-level mutual information visualization**: It clearly demonstrates how generated tokens capture the semantic dependencies of the input.

## Limitations & Future Work

1. Autoregressive generation increases inference latency (requiring multiple tokens to be generated instead of a single forward pass).
2. The probability scheduling strategy for scheduled sampling requires hyperparameter tuning.
3. For numerical values with a large number of tokens (e.g., long floating-point numbers), generation accuracy may still be limited.
4. Validation was primarily conducted on the Llama2 series, without extending to more architectures (such as encoder-only models).

## Related Work & Insights

- **LLM Prediction**: Traditional classification head fine-tuning (BERT [CLS]), in-context learning (GPT-3 few-shot).
- **Generative Prediction**: T5 unifying all tasks into a text-to-text format, zero-shot inference of the GPT series.
- **Exposure Bias**: Scheduled Sampling, Reward Augmented Maximum Likelihood.
- **Information Theory and Deep Learning**: Information Bottleneck Theory, MINE mutual information estimation.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐⭐ |
| Overall Rating | ⭐⭐⭐⭐⭐ |

> This paper presents an excellent integration of theory and practice. The core insight—"generation retains more information, and is thus more suitable for prediction"—is rigorously proven via DPI, and the practical challenges are elegantly addressed by the PredGen framework. The design of the WDAL loss function is novel, and the Writer-Director analogy is highly impressive. It stands out as an outstanding work among the ACL2025 papers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] When Worse is Better: Navigating the Compression-Generation Trade-off in Visual Tokenization](../../NeurIPS2025/model_compression/when_worse_is_better_navigating_the_compression-generation_tradeoff_in_visual_to.md)
- [\[ACL 2025\] BrainECHO: Semantic Brain Signal Decoding through Vector-Quantized Spectrogram Reconstruction for Whisper-Enhanced Text Generation](brainecho_semantic_brain_signal_decoding_through_vector-quantized_spectrogram_re.md)
- [\[ACL 2025\] UniICL: An Efficient ICL Framework Unifying Compression, Selection, and Generation](uniicl_icl_framework.md)
- [\[ACL 2025\] SCOPE: Optimizing Key-Value Cache Compression in Long-context Generation](scope_optimizing_key-value_cache_compression_in_long-context_generation.md)
- [\[ACL 2025\] ClaimPKG: Enhancing Claim Verification via Pseudo-Subgraph Generation with Lightweight Specialized LLM](claimpkg_enhancing_claim_verification_via_pseudo-subgraph_generation_with_lightw.md)

</div>

<!-- RELATED:END -->
