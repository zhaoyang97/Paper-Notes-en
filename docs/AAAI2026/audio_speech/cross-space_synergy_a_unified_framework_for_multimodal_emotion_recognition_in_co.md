---
title: >-
  [Paper Note] Cross-Space Synergy: A Unified Framework for Multimodal Emotion Recognition in Conversation
description: >-
  [AAAI 2026][Audio & Speech][multimodal emotion recognition] This paper proposes the Cross-Space Synergy (CSS) framework, which simultaneously addresses two major challenges in multimodal emotion recognition in conversati…
tags:
  - "AAAI 2026"
  - "Audio & Speech"
  - "multimodal emotion recognition"
  - "high-order fusion"
  - "gradient conflict"
  - "Pareto optimization"
  - "conversational emotion"
date: 2026-05-08
content_hash: cf94f56bcf14b9bd
---

# Cross-Space Synergy: A Unified Framework for Multimodal Emotion Recognition in Conversation

**Conference**: AAAI 2026
**arXiv**: [2512.03521](https://arxiv.org/abs/2512.03521)  
**Code**: N/A  
**Area**: Audio & Speech / Multimodal Emotion Recognition
**Keywords**: multimodal emotion recognition, high-order fusion, gradient conflict, Pareto optimization, conversational emotion

## TL;DR

This paper proposes the Cross-Space Synergy (CSS) framework, which simultaneously addresses two major challenges in multimodal emotion recognition in conversation (MERC)—insufficient fusion expressiveness and multi-objective gradient conflicts—via Synergistic Polynomial Fusion (SPF) in the representation space and a Pareto Gradient Modulator (PGM) in the gradient space.

## Background & Motivation

**Background**: Multimodal Emotion Recognition in Conversation (MERC) requires integrating textual, acoustic, and visual signals to infer a speaker's emotional state. Existing approaches include attention-based models (DialogueRNN, Ada2I, SDT), multi-stage fusion frameworks (LMF, MulT), and graph-based methods (DialogueGCN).

**Limitations of Prior Work**: Shallow fusion strategies (e.g., concatenation, bilinear pooling) fail to capture nonlinear, high-order cross-modal interactions, while deeper networks or more complex interaction mechanisms introduce gradient conflicts and training instability.

**Key Challenge**: A fundamental trade-off exists between expressiveness and training stability—shallow interactions lead to insufficient modeling, whereas deep fusion introduces chaotic optimization dynamics.

**Goal**: To simultaneously enhance the expressiveness of cross-modal representations and the stability of multi-objective training within a unified framework.

**Key Insight**: MERC is reconceptualized as a cross-space synergy problem: the representation space requires stronger nonlinear modeling capacity, while the gradient space requires coordination among conflicting learning objectives.

**Core Idea**: High-order cross-modal fusion is achieved via low-rank tensor decomposition in the representation space, while Pareto-optimal gradient directions are used to balance multi-objective conflicts in the gradient space; the two components work synergistically.

## Method

### Overall Architecture

CSS consists of three components:
1. **Representation Encoding**: A two-stage encoder that is both speaker-aware and context-aware.
2. **Representation-Space Fusion (SPF)**: High-order multimodal fusion based on low-rank tensor combinations.
3. **Gradient-Space Optimization (PGM)**: Dynamic multi-objective gradient coordination based on Pareto optimality.

### Key Designs

1. **Two-Stage Representation Encoding**:

    - **Modality-aware Encoding**: An independent Transformer self-attention encoder is applied to each modality to capture intra-modal contextual dependencies, yielding $\mathbf{h}_{\text{Ma}}^{(m)}$.
    - **Interaction-aware Encoding**: A gated cross-attention mechanism is introduced. For each query modality $m$, the influence of other modalities is modulated via a sigmoid gate: $\mathbf{g}^{(\tilde{m})} = \sigma(\mathbf{W}^{(\tilde{m})}\mathbf{q}^{(m)} + \mathbf{b}^{(\tilde{m})})$, and non-query modal information is aggregated via the Hadamard product. **Design Motivation**: The gating mechanism adaptively filters irrelevant or noisy modal signals.
    - The outputs of both stages are concatenated and projected into a unified space: $\tilde{\mathbf{h}}^{(m)} = \mathbf{W}^{(m)}[\mathbf{h}_{\text{Ma}}^{(m)}; \mathbf{h}_{\text{Ia}}^{(m)}] + \mathbf{b}^{(m)}$.

2. **Synergistic Polynomial Fusion (SPF)**:

    - Inspired by Polynomial Tensor Pooling (PTP) and CP decomposition, SPF models $p$-order multilinear interactions.
    - **Modality-specific Projection**: Rather than using shared projections (as in prior work), independent linear projections $\mathbf{z}_j^{(m)} = \phi(\mathbf{W}_j^{(m)}\tilde{\mathbf{h}}^{(m)})$ are applied for each modality $m$ and interaction order $j$.
    - **Static Gating + Bias Anchoring**: Learnable scalar weights $\lambda_j^{(m)}$ regulate each modality's contribution, while a bias vector $\mathbf{w}_j$ serves as a stabilizing anchor to compensate for the multiplicative amplification effect of high-order interactions.
    - Gated fusion: $\mathbf{z}_j = \sum_m \lambda_j^{(m)}\mathbf{z}_j^{(m)} + \mathbf{w}_j$, followed by element-wise product aggregation: $\mathbf{z} = \mathbf{z}_1 \odot \mathbf{z}_2 \odot \cdots \odot \mathbf{z}_p$.
    - **Signed Square-Root Transform**: $\hat{\mathbf{z}} = \text{sign}(\mathbf{z}) \odot \sqrt{|\mathbf{z}|}$, which controls the dynamic range of the fused tensor, reduces amplitude variance while preserving directional information, and promotes more stable gradient propagation.

3. **Pareto Gradient Modulator (PGM)**:

    - Three supervision signals are treated as competing objectives: the multimodal classification loss $\mathcal{L}_1$, a unimodal regularization loss $\mathcal{L}_2$ (hard-label supervision for each unimodal branch), and a distillation loss $\mathcal{L}_3$ (soft-label guidance from the fusion branch to each unimodal branch).
    - The L2-normalized gradients of each task are computed to construct a $3 \times 3$ inner-product matrix $Q = GG^\top$.
    - A QP problem is solved: $\min_\gamma \frac{1}{2}\gamma^\top Q \gamma$, subject to $\gamma \geq 0, \mathbf{e}_3^\top \gamma = 1$, yielding Pareto-optimal task weights.
    - Unlike MGDA, PGM operates on normalized gradients with a fixed QP dimension of 3, incurring negligible computational overhead.

### Loss & Training

The final loss is a dynamically weighted combination: $\mathcal{L} = \gamma_1 \cdot \mathcal{L}_1 + \gamma_2 \cdot \mathcal{L}_2 + \gamma_3 \cdot \mathcal{L}_3$

Where:
- $\mathcal{L}_1$: mask-aware cross-entropy (multimodal fusion classification)
- $\mathcal{L}_2$: cross-entropy for each unimodal branch (hard labels)
- $\mathcal{L}_3$: KL divergence distillation loss (temperature-scaled fusion branch → unimodal branches)
- Weights $\gamma_1, \gamma_2, \gamma_3$ are dynamically computed per mini-batch by PGM.

## Key Experimental Results

### Main Results

**IEMOCAP Dataset** (6 emotion classes):

| Method | ACC | w-F1 |
|--------|-----|-------|
| DialogueRNN | 65.43 | 64.29 |
| MM-DFN | 67.84 | 67.85 |
| M3NET | 69.56 | 69.52 |
| GraphSmile | 70.98 | 71.00 |
| SDT | 74.12 | 74.34 |
| **CSS (Ours)** | **75.42** | **75.66** |

Compared to the strongest baseline SDT, CSS achieves a 1.30% gain in accuracy and 1.32% gain in weighted F1.

**MELD Dataset** (7 emotion classes):

| Method | ACC | w-F1 |
|--------|-----|-------|
| DialogueRNN | 60.27 | 57.95 |
| M3NET | 67.32 | 65.91 |
| MPT-HCL | 65.86 | 65.02 |
| SDT | — | — |
| **CSS (Ours)** | **Best** | **Best** |

### Ablation Study

- Removing SPF → performance drops, demonstrating the necessity of high-order fusion.
- Removing PGM (replaced with fixed weights) → training instability and large performance variance.
- Replacing modality-specific projections with shared projections → performance degradation, highlighting the importance of preserving modality-specific characteristics.
- Removing the signed square-root transform → unstable gradient propagation.

### Key Findings

- SPF achieves high performance even with low fusion complexity (low rank $r$), indicating that high-order interactions matter more than network depth.
- PGM significantly improves training stability and reduces performance variance across different runs.
- The most pronounced improvements are observed on harder emotion categories such as *happy* and *frustrated*.

## Highlights & Insights

- The paper reframes MERC from "how to fuse" to a dual-space synergy problem spanning the representation space and the gradient space, providing a more complete perspective.
- SPF's design—combining modality-specific projections, static gating, and bias anchoring—achieves asymmetric, noise-aware high-order interactions while remaining compact.
- PGM exploits the fixed number of three tasks to reduce the overhead of multi-objective optimization to a negligible level.

## Limitations & Future Work

- Validation is limited to IEMOCAP and MELD; generalization to additional datasets remains unexplored.
- The interaction order $p$ in SPF is a hyperparameter; no adaptive selection mechanism is provided.
- PGM assumes three fixed objectives and does not readily extend to a larger number of auxiliary tasks.
- The impact of feature extractor selection (e.g., pretrained language models for text) on model performance is not sufficiently discussed.

## Related Work & Insights

- SDT (self-distillation) is the most direct baseline; CSS extends it by incorporating high-order fusion and gradient coordination.
- MGDA (Multiple Gradient Descent Algorithm) provides the theoretical foundation for PGM; CSS substantially reduces overhead via a fixed-dimension QP.
- PTP (Polynomial Tensor Pooling) inspired SPF; CSS extends it to a modality-specific formulation.
- **Insight**: The multi-objective optimization framework is generalizable to other multimodal learning tasks such as visual question answering and cross-modal retrieval.

## Rating

- Novelty: ⭐⭐⭐⭐ The dual-space synergy perspective is novel; SPF and PGM each make independent contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two mainstream datasets with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are clear and motivations are well articulated.
- Value: ⭐⭐⭐⭐ The framework is highly generalizable; SPF and PGM can be applied independently in other settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Dynamic Parameter Memory: Temporary LoRA-Enhanced LLM for Long-Sequence Emotion Recognition in Conversation](../../ICLR2026/audio_speech/dynamic_parameter_memory_temporary_lora-enhanced_llm_for_long-sequence_emotion_r.md)
- [\[AAAI 2026\] Let the Model Learn to Feel: Mode-Guided Tonality Injection for Symbolic Music Emotion Recognition](let_the_model_learn_to_feel_mode-guided_tonality_injection_f.md)
- [\[AAAI 2026\] Do LLMs Feel? Teaching Emotion Recognition with Prompts, Retrieval, and Curriculum Learning](do_llms_feel_teaching_emotion_recognition_with_prompts_retrieval_and_curriculum_.md)
- [\[AAAI 2026\] CCFQA: A Benchmark for Cross-Lingual and Cross-Modal Speech and Text Factuality Evaluation](ccfqa_a_benchmark_for_cross-lingual_and_cross-modal_speech_and_text_factuality_e.md)
- [\[CVPR 2026\] UniM: A Unified Any-to-Any Interleaved Multimodal Benchmark](../../CVPR2026/audio_speech/unim_a_unified_any-to-any_interleaved_multimodal_benchmark.md)

</div>

<!-- RELATED:END -->
