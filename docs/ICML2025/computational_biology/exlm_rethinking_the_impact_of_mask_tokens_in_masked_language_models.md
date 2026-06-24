---
title: >-
  [Paper Note] ExLM: Rethinking the Impact of [MASK] Tokens in Masked Language Models
description: >-
  [ICML 2025][Computational Biology][Masked Language Models] This paper provides the first systematic analysis of the impact of [MASK] tokens on performance in MLMs, revealing that **corrupted semantics** exerts a more detrimental effect than **unreal tokens**. Based on this finding, ExLM is proposed: by expanding each [MASK] into multiple hidden states and modeling their dependencies with transition matrices, it effectively mitigates the semantic multimodality problem…
tags:
  - "ICML 2025"
  - "Computational Biology"
  - "Masked Language Models"
  - "Corrupted Semantics"
  - "Multimodality"
  - "States Expansion"
  - "DAG Alignment"
date: 2026-05-08
content_hash: 3bd77786b2d4d54a
---

# ExLM: Rethinking the Impact of [MASK] Tokens in Masked Language Models

**Conference**: ICML 2025  
**arXiv**: [2501.13397](https://arxiv.org/abs/2501.13397)  
**Code**: None (implemented based on Fairseq)  
**Area**: Computational Biology  
**Keywords**: Masked Language Models, Corrupted Semantics, Multimodality, States Expansion, DAG Alignment

## TL;DR

This paper provides the first systematic analysis of the impact of [MASK] tokens on performance in MLMs, revealing that **corrupted semantics** exerts a more detrimental effect than **unreal tokens**. Based on this finding, ExLM is proposed: by expanding each [MASK] into multiple hidden states and modeling their dependencies with transition matrices, it effectively mitigates the semantic multimodality problem, yielding significant improvements on both text and molecular modeling tasks.

## Background & Motivation

Pre-training of MLMs (e.g., BERT) learns contextual representations by replacing input tokens with [MASK]. However, introducing [MASK] tokens brings two issues:

**Unreal token problem**: The context contains a large number of [MASK] symbols unique to pre-training, leading to inconsistency with real text.

**Corrupted semantics problem**: Masked tokens result in incomplete contextual semantics, potentially causing polysemous ambiguity.

Prior works (such as ELECTRA, MAE-LM, etc.) primarily focus on the first problem, while systematic research on the impact of corrupted semantics is lacking. Crucially, these two factors are coupled with the mask ratio, making it difficult to analyze their individual impacts.

The core motivation of this paper is to: **decouple these two factors, quantitatively compare their impact on MLM performance, and design a better pre-training method based on the findings**.

## Method

### Overall Architecture

The core idea of ExLM is divided into two steps:

**Step 1: Pilot Analysis (Repeated MLM)** — A decoupling experiment is designed to quantify the impact of corrupted semantics. Specifically, before entering the MLM, each token is repeated $k$ times and then masked with probability $p$. In this setup:
- The ratio of unreal tokens remains $p$.
- The ratio of corrupted semantics becomes $p^k$ (since corrupted semantics only occurs when all $k$ copies of a token are masked).

By keeping $p$ fixed and varying $k$, the ratio of unreal tokens can be maintained while altering the level of corrupted semantics, and vice versa. Experimental results on the MNLI task show that:
- When the ratio of corrupted semantics is fixed, as the mask ratio increases, the performance only slightly drops from 83.6 to 82.8.
- When the mask ratio is fixed, as the ratio of corrupted semantics increases, the performance drops significantly from 82.8 to 79.6.

Conclusion: **Corrupted semantics has a much greater impact on MLM performance than unreal tokens**.

**Step 2: ExLM Design** — Based on these findings, an enhanced context-aware MLM is designed to handle the multimodality problem caused by corrupted semantics.

### Key Designs

ExLM contains two core components:

#### 1. States Expansion

For each [MASK] token in the input, its embedding is duplicated $k$ times to form an expanded input sequence:

$$\mathbf{X'} = [\mathbf{e}_{x_1}, \mathbf{e}_{x_2}, \ldots, \mathbf{e}_{[\text{MASK}]}^{(1)}, \ldots, \mathbf{e}_{[\text{MASK}]}^{(k)}, \ldots, \mathbf{e}_{x_n}]$$

The expanded sequence is then fed into the Transformer Encoder to obtain corresponding hidden states. By expanding states, the model possesses a larger semantic space to capture different potential semantics, effectively addressing *intra-token multimodality* (the ambiguity of a single token).

#### 2. 2D RoPE Position Encoding

To distinguish different clones of the same [MASK], a two-dimensional Rotary Position Encoding is introduced. The $k$ clones of [MASK] at the original position $i$ are assigned positions $(i,1), (i,2), \ldots, (i,k)$, while non-masked tokens maintain $(j,0)$. The first dimension encodes the sequence position, and the second dimension distinguishes the clone index.

#### 3. Semantic Dependency Modeling via Transition Matrix

The semantic dependencies between expanded states are modeled as a Directed Acyclic Graph (DAG). Specifically, a transition matrix $\mathbf{E}$ is obtained via an attention-like calculation:

$$\mathbf{E} = \text{softmax}\left(\frac{\mathbf{QK}^{\top}}{\sqrt{d}} + \mathbf{M}\right)$$

where $\mathbf{Q} = \mathbf{H}\mathbf{W}_Q$, $\mathbf{K} = \mathbf{H}\mathbf{W}_K$, and $\mathbf{M}$ is an upper triangular mask matrix ensuring the DAG structure. Each state also computes a token probability distribution through a prediction head:

$$\mathbf{P} = \text{softmax}(\mathbf{H}\mathbf{W}_P^{\top})$$

The transition matrix effectively captures *inter-token multimodality* (semantic dependency between different masked tokens); for example, when the first [MASK] is "terrible", the second [MASK] is more likely to be "sorry".

### Loss & Training

#### States Alignment

Since the number of expanded hidden states exceeds the number of target tokens, alignment between states and targets must be established. This is framed as a DAG decoding problem:

$$\mathcal{L}_{SA} = -\log P_{\theta}(\mathbf{Y}|\mathbf{X'}) = -\log \sum_{\mathbf{A} \in \Gamma} P_{\theta}(\mathbf{Y}, \mathbf{A}|\mathbf{X'})$$

This is solved efficiently using dynamic programming. Define $f_{i,u}$ as the cumulative probability of all paths reaching state $u$ that have generated the first $i$ target tokens:

$$f_{i,u} = \sum_{v < u} f_{i-1,v} \times \mathbf{E}_{v,u} \times \mathbf{P}_u(y_i)$$

The final objective is $\mathcal{L}_{SA} = -\log f_{M,L}$. The time complexity is $O(M \times L^2)$, which can be further optimized to $O(M)$ via CUDA parallelization.

Pre-training data: English Wikipedia + BookCorpus for text; 19 million SMILES for molecules. The model architecture matches BERT-base (12 layers, 768 dimensions, 128M parameters), and $k=4$ is used as the default expansion factor.

## Key Experimental Results

### Main Results

**Text Tasks (GLUE + SQuAD 2.0, dev set):**

| Model | MNLI-m/mm | QQP | QNLI | SST-2 | CoLA | RTE | MRPC | STS-B | Mean | SQuAD EM | SQuAD F1 |
|------|-----------|-----|------|-------|------|-----|------|-------|------|----------|----------|
| BERT | 84.5/- | 91.3 | 91.7 | 93.2 | 58.9 | 68.6 | 87.3 | 89.5 | 83.1 | 73.7 | 76.3 |
| RoBERTa* | 85.9/85.8 | 91.6 | 92.3 | 93.7 | 64.3 | 75.5 | 88.7 | 89.5 | 85.2 | 78.3 | 81.5 |
| TUPE | 86.2/86.2 | 91.3 | 92.2 | 93.3 | 63.6 | 73.6 | 89.9 | 89.2 | 84.9 | - | - |
| **ExLM** | **86.9/86.7** | **92.0** | **93.1** | **93.9** | **64.6** | **78.8** | **89.6** | **90.5** | **86.2** | **82.0** | **84.6** |
| ExLM_LARGE | 87.8/87.5 | 92.2 | 93.8 | 94.5 | 65.3 | 79.1 | 90.4 | 91.2 | 86.9 | 82.6 | 85.0 |

ExLM achieves the best performance in 7 out of 8 GLUE tasks, raising the Mean score from 85.2 (RoBERTa) to 86.2 (+1.0) and SQuAD F1 from 81.5 to 84.6 (+3.1).

**Molecular Property Prediction (MoleculeNet, ROC-AUC):**

| Model | BACE | BBBP | Tox21 | ToxCast | SIDER | ClinTox | MUV | Avg |
|------|------|------|-------|---------|-------|---------|-----|-----|
| D-MPNN | 80.9 | 71.0 | 75.9 | 57.0 | 78.6 | 90.6 | 65.5 | 74.2 |
| SMILES-BERT* | 77.8 | 68.6 | 75.1 | 61.2 | 75.1 | 89.8 | 64.9 | 73.2 |
| GraphMVP | 81.2 | 72.4 | 75.9 | 63.9 | 77.7 | 79.1 | 63.1 | 73.3 |
| **ExLM** | 79.6 | **72.8** | **78.2** | **64.5** | **78.8** | **91.6** | **66.9** | **76.1** |

ExLM achieves the best performance in 5 out of 7 datasets, with an average score of 76.1, significantly outperforming SMILES-BERT with the same architecture (73.2, +2.9).

### Ablation Study

| Configuration | MNLI | QNLI | QQP | RTE | Avg | Description |
|------|------|------|-----|-----|-----|------|
| Vanilla MLM | 83.6 | 90.0 | 90.3 | 54.7 | 79.6 | Baseline |
| Vanilla MLM++ | 84.4 | 91.2 | 90.6 | 56.3 | 80.7 | Equal training cost MLM |
| ExLM w/o Transitions | 83.8 | 90.9 | 91.1 | 55.6 | 80.4 | Without transition matrix |
| ExLM w/o 2D RoPE | 84.6 | 91.1 | 91.3 | 56.7 | 80.9 | Without 2D position encoding |
| ExLM w/ Sparse DAG | 84.4 | 91.2 | 91.3 | 56.9 | 81.0 | Sparse DAG |
| **ExLM** | **85.1** | **91.4** | **91.3** | **57.6** | **81.4** | Full Model |

### Key Findings

1. **Transition Matrix > 2D RoPE**: Removing the transition matrix (-1.0 avg) has a greater impact than removing 2D RoPE (-0.5 avg), indicating that state dependency modeling is core.
2. **Reasonable Efficiency**: ExLM ($k=4$) takes about 1.9 times the training time of MLM (104.2h vs 54.7h on A100), but the equal-cost Vanilla MLM++ still underperforms compared to ExLM.
3. **$k=4$ is Optimal**: Performance continuously improves as $k$ goes from 2 to 4, with a slight drop at $k=8$ due to excessively long sequences.
4. **Robust to High Mask Ratios**: ExLM's performance degradation under high mask ratios is significantly smaller than MLM, demonstrating the effectiveness of enhanced semantic modeling.
5. **Entropy Analysis**: The prediction entropy of ExLM is significantly lower than that of MLM, indicating effective mitigation of semantic multimodality.

## Highlights & Insights

- **Ingenious Decoupled Experimental Design**: Repeated MLM elegantly decouples unreal tokens and corrupted semantics through token repetition, serving as the most innovative analytical tool in this paper.
- **Closed-Loop from Analysis to Method**: Discovering the problem first via experiments (corrupted semantics being the primary cause) and then designing a targeted solution (state expansion + dependency modeling) creates a clear logical progression.
- **Cross-Domain Validation**: Effectiveness is verified on text (GLUE/SQuAD) and molecules (SMILES/MoleculeNet), which are highly distinct domains, proving the universality of the method.
- **Intuitive Case Studies**: DAG visualization clearly demonstrates how ExLM uses different states to capture various semantic possibilities and their dependencies.

## Limitations & Future Work

1. **Increased Training Cost**: Expanded states lead to longer sequence lengths, with training time close to 2x at $k=4$, making it difficult to directly scale to larger models.
2. **Only Verified at BERT-scale**: Not validated on larger-scale models (beyond BERT-Large) or more massive pre-training datasets.
3. **How to use at inference time?**: The paper mainly focuses on the pre-training phase; the handling of expanded states during fine-tuning is not fully discussed.
4. **Only Applicable to Encoder MLMs**: The applicability to decoder-only or encoder-decoder architectures has not been explored.
5. **Strong DAG Decoding Assumption**: Imposure of a directed acyclic graph structure may limit the modeling of certain cyclic dependencies.

## Related Work & Insights

- **Complementary to ELECTRA (Clark, 2020)**: ELECTRA tackles the unreal tokens issue while ExLM addresses corrupted semantics, showing potential for combination.
- **DAG + DP Paradigm Migration**: Transferring the DAG + DP paradigm from DA-Transformer (Huang et al., 2022) in NAT to MLM pre-training demonstrates the feasibility of method transfer across different tasks.
- **Inspiration for MAE-like Vision Pre-training**: The issue of semantic corruption under high mask ratios also exists in vision; ExLM's approach could potentially be adapted to improve MAE.

## Rating

- Novelty: ⭐⭐⭐⭐ — The decoupling experimental design is highly ingenious; the overall method is an innovative combination of existing components.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers both text and molecule domains, with comprehensive ablation, visualization, and efficiency analyses.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear logic from analysis to methodology, with high-quality tables and figures.
- Value: ⭐⭐⭐⭐ — Provides deep insights into MLM pre-training, though somewhat limited by the current popularity of the encoder-only paradigm.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Steering Protein Language Models](steering_protein_language_models.md)
- [\[NeurIPS 2025\] KLASS: KL-Guided Fast Inference in Masked Diffusion Models](../../NeurIPS2025/computational_biology/klass_kl-guided_fast_inference_in_masked_diffusion_models.md)
- [\[ICML 2025\] CFP-Gen: Combinatorial Functional Protein Generation via Diffusion Language Models](cfp-gen_combinatorial_functional_protein_generation_via_diffusion_language_model.md)
- [\[NeurIPS 2025\] Understanding and Enhancing Mask-Based Pretraining towards Universal Representations](../../NeurIPS2025/computational_biology/understanding_and_enhancing_mask-based_pretraining_towards_universal_representat.md)
- [\[ICML 2025\] Elucidating the Design Space of Multimodal Protein Language Models](elucidating_the_design_space_of_multimodal_protein_language_models.md)

</div>

<!-- RELATED:END -->
