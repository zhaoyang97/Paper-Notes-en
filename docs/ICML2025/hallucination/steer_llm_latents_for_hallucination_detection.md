---
title: >-
  [Paper Note] Steer LLM Latents for Hallucination Detection
description: >-
  [ICML 2025][Hallucination Detection][steering vector] Proposes Truthfulness Separator Vector (TSV), a lightweight steering vector that reshapes the LLM representation space at inference time to enhance the separation between truthful and hallucinated outputs, achieving performance close to full supervision with only 32 labeled exemplars.
tags:
  - "ICML 2025"
  - "Hallucination Detection"
  - "steering vector"
  - "optimal transport"
  - "pseudo-labeling"
  - "TSV"
date: 2026-05-08
content_hash: f9892a02c5d222c5
---

# Steer LLM Latents for Hallucination Detection

**Conference**: ICML 2025  
**arXiv**: [2503.01917](https://arxiv.org/abs/2503.01917)  
**Code**: -  
**Area**: Hallucination Detection  
**Keywords**: steering vector, hallucination detection, optimal transport, pseudo-labeling, TSV  

## TL;DR

Proposes Truthfulness Separator Vector (TSV), a lightweight steering vector that reshapes the LLM representation space at inference time to enhance the separation between truthful and hallucinated outputs, achieving performance close to full supervision with only 32 labeled exemplars.

## Background & Motivation

### Background

**Background**: LLM hallucination is a significant risk for safe deployment.

### Key Challenge

**Key Challenge**: Existing latent-space-based methods rely on pre-trained LLM embeddings, but these embeddings are optimized for **linguistic coherence** rather than **factual accuracy**.

### Limitations of Prior Work

**Limitations of Prior Work**: Truthful and hallucinated contents heavily overlap in pre-trained embeddings (see T-SNE visualization).

### Proposed Approach

**Proposed Approach**: Fine-tuning LLMs is computationally expensive and alters model parameters.

### Supplementary Notes

**Supplementary Notes**: **Core Problem**: How to reshape the latent space to distinguish hallucinations without modifying model parameters?

## Method

### 1. Truthfulness Separator Vector (TSV)

Define a trainable vector $\mathbf{v} \in \mathbb{R}^d$, which is added to the hidden state of an intermediate layer $l$ during inference:

$$\mathbf{h}^{(l)} \leftarrow \mathbf{h}^{(l)} + \lambda \mathbf{v}$$

where $\lambda$ controls the steering intensity. TSV is shared across all token positions and influences the final-layer embeddings through subsequent non-linear transformations.

### 2. Initial Training Phase

The final-layer embeddings are modeled using the von Mises-Fisher distribution, with the class-conditional probability defined as:

$$p(c|\mathbf{r}^{\mathbf{v}}) = \frac{\exp(\kappa \boldsymbol{\mu}_c^\top \mathbf{r}^{\mathbf{v}})}{\sum_{c'} \exp(\kappa \boldsymbol{\mu}_{c'}^\top \mathbf{r}^{\mathbf{v}})}$$

where $\mathbf{r}^{\mathbf{v}}$ is the normalized final embedding, and $\boldsymbol{\mu}_c$ represents the class prototype.

The training objective is to maximize the log-likelihood on the exemplar set $\mathcal{D}_E$:

$$\mathcal{L} = -\frac{1}{|\mathcal{D}_E|}\sum_{i=1}^{|\mathcal{D}_E|}\sum_{c \in \mathcal{C}} q(c|\mathbf{r}_i^{\mathbf{v}}) \log p(c|\mathbf{r}_i^{\mathbf{v}})$$

### 3. Enhanced Training Phase

#### Pseudo-Label Assignment based on Optimal Transport

For the unlabeled data $\mathcal{D}_U$, pseudo-labels are assigned by solving the optimal transport problem using the Sinkhorn algorithm:

$$\min_{\mathbf{Q} \in [0,1]^{M \times 2}} -\sum_{m,c} \mathbf{Q}_{m,c} \log \mathbf{P}_{m,c} - \epsilon H(\mathbf{Q})$$

Subject to constraints where row sums equal $1/M$ (total probability per sample is 1) and column sums match the class distribution prior $\mathbf{w}$.

#### Confidence Filtering

Only the $K$ pseudo-labeled samples with the lowest prediction uncertainty are selected for training:

$$\mathcal{D}_S = \{\mathcal{D}_U^j \mid j \in \text{TopK}_{i}(-\Omega_i)\}$$

where $\Omega_i$ represents the uncertainty measured by cross-entropy.

## Key Experimental Results

### Main Results: TruthfulQA (AUROC)

| Method | LLaMA-3.1-8B |
|------|-------------|
| CCS | 58.1 |
| SAPLMA | 63.2 |
| Probing (supervised) | 71.3 |
| HaloScope | 71.4 |
| **TSV (32 exemplars)** | **84.2** |
| Fully supervised upper bound | 85.5 |

- TSV improves upon the SOTA by **+12.8%** AUROC.
- Using only 32 labeled exemplars, TSV performs close to the fully supervised upper bound (84.2% vs 85.5%).

### Cross-Dataset Generalization

The TSV trained on TruthfulQA is evaluated on TriviaQA and HaluEval:
- It remains competitive, demonstrating robust out-of-distribution generalization.

### Ablation Study

| Component | AUROC |
|------|-------|
| TSV (Initial Training Only) | 79.8 |
| + OT Pseudo-Labeling | 82.5 |
| + Confidence Filtering | **84.2** |
| Without TSV (Direct Embedding) | 71.3 |

- Each component makes a clear contribution.
- Best intervention layer: Mid-level layers (around the 16th layer out of 32 layers total).

## Highlights & Insights

- This work is the first to apply steering vectors to hallucination **detection** (rather than generation mitigation), filling an important gap.
- Optimal transport pseudo-label assignment accounts for class imbalance, outperforming simple thresholding approaches.
- Remarkably low annotation requirement (32 exemplars) achieves performance close to full supervision, offering high practical value.
- TSV can be applied after generation is completed without modifying model parameters.
- Modeling with the von Mises-Fisher distribution aligns perfectly with the unit-norm property of post-RMSNorm embeddings.

## Limitations & Future Work

- The selection of $\lambda$ and intervention layer $l$ requires tuning on a validation set.
- The TSV may need to be retrained for different LLMs.
- Evaluated only on closed-book QA tasks; its effectiveness in open-ended generation scenarios remains unexplored.
- The class distribution prior $\mathbf{w}$ for pseudo-labeling is derived from the exemplar set, which may not be completely accurate.
- There is currently a lack of deep theoretical explanation for why TSV can effectively separate these spaces.

## Rating

⭐⭐⭐⭐⭐ — The method is lightweight, elegant, and achieves impressive results. Obtaining performance close to the fully supervised upper bound with only 32 labeled exemplars is of significant importance to the field of hallucination detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MultiHaluDet: Multilingual Hallucination Detection via LLM Hidden State Probing](../../ACL2026/hallucination/multihaludet_multilingual_hallucination_detection_via_llm_hidden_state_probing.md)
- [\[ICML 2026\] Zero-source LLM Hallucination Detection with Human-like Criteria Probing](../../ICML2026/hallucination/zero-source_llm_hallucination_detection_with_human-like_criteria_probing.md)
- [\[ACL 2025\] HalluLens: LLM Hallucination Benchmark](../../ACL2025/hallucination/hallulens_llm_hallucination_benchmark.md)
- [\[ACL 2026\] Rethinking Evaluation for LLM Hallucination Detection: A Desiderata, A New RAG-based Benchmark, New Insights](../../ACL2026/hallucination/rethinking_evaluation_for_llm_hallucination_detection_a_desiderata_a_new_rag-bas.md)
- [\[ACL 2026\] Logical Consistency as a Bridge: Improving LLM Hallucination Detection via Label Constraint Modeling between Responses and Self-Judgments](../../ACL2026/hallucination/logical_consistency_as_a_bridge_improving_llm_hallucination_detection_via_label_.md)

</div>

<!-- RELATED:END -->
