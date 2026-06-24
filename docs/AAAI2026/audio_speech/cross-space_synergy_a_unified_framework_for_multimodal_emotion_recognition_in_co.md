---
title: >-
  [Paper Note] Cross-Space Synergy: A Unified Framework for Multimodal Emotion Recognition in Conversation
description: >-
  [AAAI 2026][Audio & Speech][Multimodal Emotion Recognition] The proposed Cross-Space Synergy (CSS) framework simultaneously addresses the two key challenges of insufficient fusion expressiveness and multi-objective gradient conflicts in multimodal conversational emotion recognition using a two-pronged approach: Synergistic Polynomial Fusion (SPF) in the representation space and a Pareto Gradient Modifier (PGM) in the gradient space.
tags:
  - "AAAI 2026"
  - "Audio & Speech"
  - "Multimodal Emotion Recognition"
  - "High-Order Fusion"
  - "Gradient Conflict"
  - "Pareto Optimization"
  - "Conversational Emotion"
date: 2026-05-08
content_hash: a0789da42c7b68e2
---

# Cross-Space Synergy: A Unified Framework for Multimodal Emotion Recognition in Conversation

**Conference**: AAAI 2026  
**arXiv**: [2512.03521](https://arxiv.org/abs/2512.03521)  
**Code**: None  
**Area**: Audio & Speech / Multimodal Emotion Recognition  
**Keywords**: Multimodal Emotion Recognition, High-Order Fusion, Gradient Conflict, Pareto Optimization, Conversational Emotion

## TL;DR

The proposed Cross-Space Synergy (CSS) framework simultaneously addresses the two key challenges of insufficient fusion expressiveness and multi-objective gradient conflicts in multimodal conversational emotion recognition using a two-pronged approach: Synergistic Polynomial Fusion (SPF) in the representation space and a Pareto Gradient Modifier (PGM) in the gradient space.

## Background & Motivation

**Background**: Multimodal Emotion Recognition in Conversation (MERC) requires integrating textual, acoustic, and visual signals to infer the emotional states of speakers. Existing approaches include attention-based models (DialogueRNN, Ada2I, SDT), multi-stage fusion frameworks (LMF, MulT), and graph-based methods (DialogueGCN).

**Limitations of Prior Work**: Shallow fusion (e.g., concatenation, bilinear) fails to capture non-linear, high-order cross-modal interactions, whereas deepening the networks or adopting more complex interaction mechanisms introduces gradient conflicts and training instability.

**Key Challenge**: A fundamental trade-off between expressiveness and training stability—shallow interactions lead to under-modeling, while deep fusion introduces chaotic optimization dynamics.

**Goal**: Simultaneously enhance the expressiveness of cross-modal representations and the stability of multi-objective training within a unified framework.

**Key Insight**: Reconceptualize MERC as a cross-space synergy problem—the representation space requires stronger non-linear modeling capabilities, and the gradient space needs to coordinate conflicts among multiple learning objectives.

**Core Idea**: Leverage low-rank tensor decomposition to achieve high-order cross-modal fusion (representation space) and utilize Pareto-optimal gradient directions to balance multi-objective conflicts (gradient space), with both components working synergistically.

## Method

### Overall Architecture

CSS consists of three components:
1. **Representation Encoding**: A speaker-aware + context-aware two-stage encoder.
2. **Representation Space Fusion (SPF)**: High-order multimodal fusion based on low-rank tensor combinations.
3. **Gradient Space Optimization (PGM)**: Dynamic multi-objective gradient coordination based on Pareto optimality.

### Key Designs

1. **Two-Stage Representation Encoding**:

    - **Modality-aware Encoding**: Apply a Transformer self-attention encoder to each modality independently to capture intra-modal contextual dependencies $\mathbf{h}_{\text{Ma}}^{(m)}$.
    - **Interaction-aware Encoding**: Introduce a gated cross-attention mechanism. For each query modality $m$, modulate the influence of other modalities using a sigmoid gate: $\mathbf{g}^{(\tilde{m})} = \sigma(\mathbf{W}^{(\tilde{m})}\mathbf{q}^{(m)} + \mathbf{b}^{(\tilde{m})})$, and then aggregate non-query modality information via Hadamard product. **Design Motivation**: The gating mechanism adaptively filters out irrelevant or noisy modality signals.
    - The outputs of the two stages are concatenated and projected into a unified space: $\tilde{\mathbf{h}}^{(m)} = \mathbf{W}^{(m)}[\mathbf{h}_{\text{Ma}}^{(m)}; \mathbf{h}_{\text{Ia}}^{(m)}] + \mathbf{b}^{(m)}$

2. **Synergistic Polynomial Fusion (SPF)**:

    - Inspired by Polynomial Tensor Pooling (PTP) and CP decomposition, this models a $p$-order multi-way multiplicative interaction.
    - **Modality-specific Projection**: Instead of using shared projections (as done in prior work), unique linear projections are applied for each modality $m$ and each interaction order $j$: $\mathbf{z}_j^{(m)} = \phi(\mathbf{W}_j^{(m)}\tilde{\mathbf{h}}^{(m)})$.
    - **Static Gating + Bias Anchoring**: Dynamically adjust the contribution of each modality via learnable scalar weights $\lambda_j^{(m)}$, and introduce a bias vector $\mathbf{w}_j$ as a stable anchor to compensate for the multiplicative amplification effect of high-order interactions.
    - Gated Fusion: $\mathbf{z}_j = \sum_m \lambda_j^{(m)}\mathbf{z}_j^{(m)} + \mathbf{w}_j$, followed by element-wise product aggregation $\mathbf{z} = \mathbf{z}_1 \odot \mathbf{z}_2 \odot \cdots \odot \mathbf{z}_p$.
    - **Signed Square Root Transform**: $\hat{\mathbf{z}} = \text{sign}(\mathbf{z}) \odot \sqrt{|\mathbf{z}|}$, controlling the dynamic range of the fused tensor to reduce magnitude variance while preserving directional information, facilitating more stable gradient propagation.

3. **Pareto Gradient Modifier (PGM)**:

    - Treat three supervision signals as competing objectives: multimodal classification loss $\mathcal{L}_1$, unimodal regularization loss $\mathcal{L}_2$ (supervising individual unimodal branches with hard labels), and distillation loss $\mathcal{L}_3$ (guiding unimodal branches with soft labels from the fusion branch).
    - Compute the L2-normalized versions of the gradients for each task to construct a $3 \times 3$ inner product matrix $Q = GG^\top$.
    - Solve the QP problem: $\min_\gamma \frac{1}{2}\gamma^\top Q \gamma$, subject to $\gamma \geq 0, \mathbf{e}_3^\top \gamma = 1$, to obtain the Pareto-optimal task weights.
    - Unlike MGDA, PGM operates on normalized gradients, keeping the QP dimension fixed at 3, which incurs negligible overhead.

### Loss & Training

The final loss is a dynamically weighted combination: $\mathcal{L} = \gamma_1 \cdot \mathcal{L}_1 + \gamma_2 \cdot \mathcal{L}_2 + \gamma_3 \cdot \mathcal{L}_3$

Where:
- $\mathcal{L}_1$: mask-aware cross-entropy (multimodal fusion classification)
- $\mathcal{L}_2$: cross-entropy for each unimodal branch (hard labels)
- $\mathcal{L}_3$: KL divergence distillation loss (temperature-scaled fusion branch $\rightarrow$ unimodal branches)
- Weights $\gamma_1, \gamma_2, \gamma_3$ are dynamically computed by PGM at each mini-batch.

## Key Experimental Results

### Main Results

**IEMOCAP Dataset** (6 emotion classes):

| Method | ACC | w-F1 |
|------|-----|------|
| DialogueRNN | 65.43 | 64.29 |
| MM-DFN | 67.84 | 67.85 |
| M3NET | 69.56 | 69.52 |
| GraphSmile | 70.98 | 71.00 |
| SDT | 74.12 | 74.34 |
| **CSS (Ours)** | **75.42** | **75.66** |

Compared to the strongest baseline SDT, accuracy is improved by 1.30%, and weighted F1 is improved by 1.32%.

**MELD Dataset** (7 emotion classes):

| Method | ACC | w-F1 |
|------|-----|------|
| DialogueRNN | 60.27 | 57.95 |
| M3NET | 67.32 | 65.91 |
| MPT-HCL | 65.86 | 65.02 |
| SDT | — | — |
| **CSS (Ours)** | **Best** | **Best** |

### Ablation Study

- Removing SPF $\rightarrow$ performance drops, proving the necessity of high-order fusion.
- Removing PGM (replacing it with fixed weights) $\rightarrow$ training instability and large performance fluctuations.
- Replacing modality-specific projection with shared projection $\rightarrow$ performance drops, illustrating the importance of preserving modality characteristics.
- Removing the signed square root transform $\rightarrow$ unstable gradient propagation.

### Key Findings

- SPF achieves high performance even under low fusion complexity (low rank $r$), showing that high-order interaction is more critical than network depth.
- PGM significantly improves training stability, reducing performance variance across different runs.
- The improvements are most prominent on difficult emotion categories such as "happy" and "frustrated".

## Highlights & Insights

- Recontextualizes the MERC problem from "how to fuse" to a dual-space "representation space + gradient space" synergy problem, offering a more complete perspective.
- The designs of modality-specific projection, static gating, and bias anchoring in SPF achieve asymmetric, noise-aware high-order interactions while keeping the model compact.
- PGM utilizes the fact that the number of tasks is fixed at 3 to keep the overhead of multi-objective optimization extremely low.

## Limitations & Future Work

- Evaluation is limited to only two datasets (IEMOCAP and MELD), lacking extensive validation on generalization across more datasets.
- The interaction order $p$ of SPF is a hyperparameter; no adaptive selection mechanism is provided.
- PGM assumes three fixed objectives, making it less straightforward to scale to more auxiliary tasks.
- The selection and impact of feature extractors for different modalities (e.g., pre-trained language models for text) are not thoroughly discussed.

## Related Work & Insights

- SDT (self-distillation) is the most direct baseline: CSS builds on top of it by adding high-order fusion and gradient coordination.
- MGDA (Multiple-Gradient Descent Algorithm) is the theoretical foundation of PGM, but CSS greatly reduces the computational overhead via a fixed-dimension QP.
- PTP (Polynomial Tensor Pooling) inspired SPF; CSS extends it to a modality-specific version.
- Insight: The multi-objective optimization framework can be extended to other multimodal learning tasks (such as visual question answering and cross-modal retrieval).

## Rating

- Novelty: ⭐⭐⭐⭐ The dual-space synergy perspective is novel, and both SPF and PGM make independent contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on two mainstream datasets with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Mathematical formulations are clear, and motivations are well articulated.
- Value: ⭐⭐⭐⭐ The framework has strong generalizability; SPF and PGM can be applied independently to other scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] AudioX: A Unified Framework for Anything-to-Audio Generation](../../ICLR2026/audio_speech/audiox_a_unified_framework_for_anything-to-audio_generation.md)
- [\[AAAI 2026\] Let the Model Learn to Feel: Mode-Guided Tonality Injection for Symbolic Music Emotion Recognition](let_the_model_learn_to_feel_mode-guided_tonality_injection_f.md)
- [\[ICLR 2026\] Learnable Fractional Superlets with a Spectro-Temporal Emotion Encoder for Speech Emotion Recognition](../../ICLR2026/audio_speech/learnable_fractional_superlets_with_a_spectro-temporal_emotion_encoder_for_speec.md)
- [\[AAAI 2026\] CCFQA: A Benchmark for Cross-Lingual and Cross-Modal Speech and Text Factuality Evaluation](ccfqa_a_benchmark_for_cross-lingual_and_cross-modal_speech_and_text_factuality_e.md)
- [\[AAAI 2026\] Do LLMs Feel? Teaching Emotion Recognition with Prompts, Retrieval, and Curriculum Learning](do_llms_feel_teaching_emotion_recognition_with_prompts_retrieval_and_curriculum_.md)

</div>

<!-- RELATED:END -->
