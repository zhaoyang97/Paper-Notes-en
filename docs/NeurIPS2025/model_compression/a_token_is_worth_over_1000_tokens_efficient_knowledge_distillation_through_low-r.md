---
title: >-
  [Paper Note] A Token is Worth over 1,000 Tokens: Efficient Knowledge Distillation through Low-Rank Clone
description: >-
  [NeurIPS 2025][Model Compression][knowledge distillation] This paper proposes Low-Rank Clone (LRC), which compresses teacher weights into student weights via learnable low-rank projection matrices (soft pruning)…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "knowledge distillation"
  - "low-rank projection"
  - "small language models"
  - "activation cloning"
  - "efficient pretraining"
date: 2026-05-08
content_hash: 8dddb0c57ded80d1
---

# A Token is Worth over 1,000 Tokens: Efficient Knowledge Distillation through Low-Rank Clone

**Conference**: NeurIPS 2025
**arXiv**: [2505.12781](https://arxiv.org/abs/2505.12781)
**Code**: GitHub + HuggingFace (mentioned in the paper)
**Area**: Model Compression / Knowledge Distillation
**Keywords**: knowledge distillation, low-rank projection, small language models, activation cloning, efficient pretraining

## TL;DR
This paper proposes Low-Rank Clone (LRC), which compresses teacher weights into student weights via learnable low-rank projection matrices (soft pruning), while aligning intermediate activations of both attention and FFN modules (activation cloning). A 1.7B model trained on only 20B tokens surpasses Qwen3-1.7B trained on 36T tokens (64.98 vs. 63.17), achieving a **1,000× improvement in training efficiency**.

## Background & Motivation

**Background**: Training high-performance small language models (SLMs) remains extremely costly; for instance, Llama-3.2-3B requires 9T tokens and Qwen3-1.7B requires 36T tokens. Knowledge distillation is a key approach to accelerating this process.

**Limitations of Prior Work**: (a) **Information loss**: Hard pruning permanently removes neurons/layers, discarding valuable information from the teacher (LLM-Pruner drops from 63.25 to 48.98 after 50% pruning); (b) **Inefficient alignment**: Feature-based distillation requires additional projection matrices to align intermediate representations of different dimensions, which are difficult to learn effectively during training; (c) **Wasted activations**: Existing methods primarily align attention scores while neglecting the information-rich FFN activations.

**Key Challenge**: How to maximize knowledge transfer from teacher to student under an extremely limited training budget?

**Key Insight**: Rather than training the student's original weights directly, the method trains a set of low-rank projection matrices that temporarily generate student weights from teacher weights at each forward pass.

**Core Idea**: Low-rank projection matrices simultaneously achieve soft pruning (weight compression) and alignment (the same matrices both generate weights and align activations), thereby eliminating information loss and alignment overhead.

## Method

### Overall Architecture
Given a teacher model (e.g., Qwen2.5-3B-Instruct), LRC trains a set of low-rank projection matrices $\mathbf{W}_m^p \in \mathbb{R}^{d^T \times d^S}$. At each forward pass: (1) teacher weights are projected to obtain student weights → (2) both models perform forward passes independently → (3) intermediate activations are aligned via KL divergence, LM loss, and clone loss. Only the projection matrices and RMSNorm parameters are trained (<1% of total parameters).

### Key Designs

1. **Low-Rank Projection**:

    - Function: Compresses each teacher weight matrix to the student's dimension.
    - Mechanism: For the 7 weight matrices $\{q,k,v,o,up,gate,down\}$ in each layer, $\mathbf{W}_{m,i}^S = \mathbf{W}_{m,i}^T \cdot \mathbf{W}_{m,i}^p$; similarly applied to embeddings and the LM head.
    - Design Motivation: **Soft pruning**—no teacher information is discarded; instead, the method learns an optimal compression mapping that preserves more teacher knowledge.
    - Distinction from hard pruning: Hard pruning is irreversible and incurs large information loss, whereas low-rank projection is learnable and achieves better knowledge retention.

2. **Activation Clone**:

    - Function: Aligns all intermediate activations between teacher and student.
    - Mechanism: Collects the linear projection outputs of $\{q,k,v,up,gate\}$ and the module outputs of attention/FFN blocks, and aligns them via MSE.
    - Key formula: $\mathcal{L}_{clone} = \sum_i^l [\mathcal{E}(\mathbf{o}_{attn,i}^S, \mathbf{o}_{attn,i}^T \mathbf{W}_{o,i}^p) + \mathcal{E}(\mathbf{o}_{ffn,i}^S, \mathbf{o}_{ffn,i}^T \mathbf{W}_{down,i}^p) + \sum_m \mathcal{E}(\mathbf{h}_{m,i}^S, \mathbf{h}_{m,i}^T)]$
    - Design Motivation: FFN activations contain rich semantic information that has been largely overlooked in prior work.

3. **Alignment-Free Design**:

    - Function: Reuses projection matrices for both weight generation and activation alignment.
    - Mechanism: As proven by Lemma 1, if intermediate activations are perfectly cloned, the student's output equals the teacher's output transformed by the same projection matrix—**a single set of projection matrices serves simultaneously for compression and alignment**.
    - Elegance: Eliminates the need for additional alignment projection modules required in conventional feature-based distillation.

### Loss & Training
- Total loss: $\mathcal{L} = \mathcal{L}_{KL} + \mathcal{L}_{LM} + \alpha \mathcal{L}_{clone}$
- KL divergence aligns logits; LM loss supervises next-token prediction; clone loss aligns intermediate activations.
- Only projection matrices and RMSNorm parameters are trained (<1% of parameters); teacher weights are frozen.
- Training requires only 10–20B tokens.

## Key Experimental Results

### Main Results (~1.7B Model Comparison)

| Model | Training Tokens | ARC-E | ARC-C | MMLU | Avg. |
|-------|----------------|-------|-------|------|------|
| Qwen3-1.7B | 36T | 62.96 | 36.86 | 55.44 | 63.17 |
| SmolLM2-1.7B | 11T | 62.58 | 34.30 | 48.50 | 60.50 |
| **LRC-1.7B** (Qwen2.5-3B teacher) | **20B** | 65.74 | 37.37 | 54.93 | **64.98** |

### Ablation Study

| Configuration | Avg. Accuracy | Note |
|--------------|--------------|------|
| Full LRC | **Best** | Complete method |
| w/o FFN activation clone | Notable drop | FFN activation cloning is critical |
| w/o Attention activation clone | Slight drop | Attention activations also contribute |
| KL+LM only (no clone loss) | Large drop | Activation cloning is essential |
| Hard pruning + distillation (Minitron) | Far below LRC | Severe information loss |

### Key Findings
- **1,000× training efficiency**: LRC surpasses Qwen3-1.7B (trained on 36T tokens) using only 20B tokens—a striking efficiency gap.
- **FFN activations are an overlooked resource**: Ablation experiments confirm that FFN clone contributes more than attention clone, contrary to the focus of existing methods.
- **Soft pruning >> hard pruning**: Low-rank projection retains all teacher information, far outperforming hard pruning + distillation approaches such as Minitron.
- **Alignment without extra parameters**: The theoretical guarantee of Lemma 1 makes LRC inherently free of alignment projection overhead.

## Highlights & Insights
- **The paradigm of "student weights = teacher weights × projection matrix"**: This idea is remarkably concise yet highly effective—rather than training student weights directly, the method learns a mapping from teacher to student. Conceptually, this resembles LoRA (low-rank adaptation) but operates in the opposite direction (compression rather than adaptation).
- **First explicit empirical validation of FFN activation importance**: Prior feature-based distillation methods have focused primarily on attention; the ablation experiments in this paper provide strong evidence for the critical role of FFN activations, a finding with broad implications for the distillation community.
- **Theoretical justification for the alignment-free design**: Lemma 1 not only simplifies the method but also provides a mathematical explanation for why projection matrices can simultaneously serve two distinct purposes.

## Limitations & Future Work
- **Simultaneous loading of teacher and student during training**: Although only projection matrices are trained, the teacher weights must be loaded for the forward pass, leading to high memory consumption.
- **Limited compression ratio**: The dimensional compression $d^T \to d^S$ has an upper bound; the effectiveness of more aggressive compression ratios remains unexplored.
- **Dependence on teacher quality**: The student's performance ceiling is determined by the teacher; effectiveness with weaker teachers has not been validated.
- **Architecture-specific design**: The projection matrix formulation relies on the Transformer structure and does not generalize to other architectures.
- **Future directions**: (1) Offline storage of teacher activations to reduce memory usage; (2) Support for cross-architecture distillation; (3) Exploration of more aggressive compression ratios (e.g., 7B→1B).

## Related Work & Insights
- **vs. Minitron**: Minitron applies hard pruning + distillation, whereas LRC employs soft pruning + activation cloning, achieving substantially better information retention.
- **vs. TinyBERT/MiniLM**: These methods align only attention representations; LRC aligns all intermediate activations, particularly those of the FFN.
- **vs. SliceGPT**: SliceGPT uses PCA-based linear projection (non-learnable), whereas LRC uses learnable low-rank projection, offering greater adaptability.
- **vs. LoRA**: The directions are opposite but the underlying intuition is shared—LoRA adds low-rank updates on top of original weights, while LRC applies low-rank projection to teacher weights.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The paradigm of learning projection matrices rather than weights is highly novel; the alignment-free design is theoretically grounded.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple teachers, ablation studies, and comparisons with various SLMs yield convincing results.
- Writing Quality: ⭐⭐⭐⭐ Problem analysis is clear, method description is detailed, and Lemma proofs are rigorous.
- Value: ⭐⭐⭐⭐⭐ A 1,000× efficiency improvement holds significant practical value and has the potential to reshape the SLM training paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Beyond Higher Rank: Token-wise Input-Output Projections for Efficient Low-Rank Adaptation](beyond_higher_rank_token-wise_input-output_projections_for_efficient_low-rank_ad.md)
- [\[NeurIPS 2025\] Accurate and Efficient Low-Rank Model Merging in Core Space](accurate_and_efficient_low-rank_model_merging_in_core_space.md)
- [\[NeurIPS 2025\] Single-Teacher View Augmentation: Boosting Knowledge Distillation via Angular Diversity](single-teacher_view_augmentation_boosting_knowledge_distillation_via_angular_div.md)
- [\[NeurIPS 2025\] RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models](reflora_refactored_low-rank_adaptation_for_efficient_fine-tuning_of_large_models.md)
- [\[NeurIPS 2025\] QSVD: Efficient Low-Rank Approximation for Unified Query-Key-Value Weight Compression](qsvd_efficient_low-rank_approximation_for_unified_query-key-value_weight_compres.md)

</div>

<!-- RELATED:END -->
