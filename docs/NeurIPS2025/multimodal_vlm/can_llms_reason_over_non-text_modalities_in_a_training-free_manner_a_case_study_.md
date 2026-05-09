---
title: >-
  [Paper Note] Can LLMs Reason Over Non-Text Modalities in a Training-Free Manner? A Case Study with In-Context Representation Learning
description: >-
  [NeurIPS 2025][Multimodal VLM][In-context representation learning] This paper proposes In-Context Representation Learning (ICRL), the first training-free framework that injects representations from non-text-modality foundation models (FMs) into a text-only LLM for few-shot reasoning. Two strategies are introduced: PCA-based text-level injection and optimal transport (OT)-based embedding alignment, enabling cross-modal knowledge utilization without any parameter updates.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - In-context representation learning
  - training-free multimodal reasoning
  - optimal transport alignment
  - foundation models
  - few-shot learning
date: 2026-05-08
content_hash: 0906ad4672025f04
---

# Can LLMs Reason Over Non-Text Modalities in a Training-Free Manner? A Case Study with In-Context Representation Learning

**Conference**: NeurIPS 2025
**arXiv**: [2509.17552](https://arxiv.org/abs/2509.17552)
**Code**: [GitHub](https://github.com/ztlmememe/LLMxFM_ICRL)
**Area**: Multimodal VLM
**Keywords**: In-context representation learning, training-free multimodal reasoning, optimal transport alignment, foundation models, few-shot learning

## TL;DR

This paper proposes In-Context Representation Learning (ICRL), the first training-free framework that injects representations from non-text-modality foundation models (FMs) into a text-only LLM for few-shot reasoning. Two strategies are introduced: PCA-based text-level injection and optimal transport (OT)-based embedding alignment, enabling cross-modal knowledge utilization without any parameter updates.

## Background & Motivation

### 1. State of the Field

LLMs have substantially expanded their capabilities through test-time computation and tool integration. Multimodal LLMs (MLLMs) such as LLaVA and InstructBLIP enable text-based LLMs to process non-text modalities (e.g., images) by training projection layers or fine-tuning the LLM itself.

### 2. Limitations of Prior Work

- Existing MLLM approaches require **additional supervised training** for modality alignment (e.g., training projection layers or fine-tuning the LLM), incurring high computational costs.
- They depend on specialized **paired datasets**, limiting immediate adaptation to new domains or modalities.
- Multi-agent frameworks exploit only the final outputs of external models, discarding the rich information encoded in intermediate representations.

### 3. Root Cause

Can a text-only LLM leverage the internal representations of non-text foundation models **at inference time without any training**?

### 4. Paper Goals

To investigate the feasibility of integrating non-text FM representations into a text-only LLM in a training-free manner, and to analyze the underlying mechanisms.

### 5. Starting Point

Traditional ICL text–label pairs $(x_i, y_i)$ are replaced with FM representation–label pairs $(r_i, y_i)$, exploiting the few-shot learning capability of LLMs to adaptively process non-text representations.

### 6. Core Idea

Dimensionality-reduced FM representations replace text inputs in ICL. Optimal transport aligns their distribution to the LLM embedding space, enabling zero-training cross-modal reasoning.

## Method

### Overall Architecture

The ICRL framework supports two injection levels:
1. **Text-level injection**: FM representations are dimensionality-reduced and embedded into the prompt as strings.
2. **Embedding-level injection**: FM representations are projected and directly replace token embeddings at the LLM embedding layer.

### Key Designs

#### Module 1: PCA Text-Level Injection

**Function**: Embeds high-dimensional FM representations into the prompt text via dimensionality reduction.

**Mechanism**: PCA is applied to FM representations $\mathbf{H} \in \mathbb{R}^{N \times d_{FM}}$, reducing them to $d_{Reduced} \ll d_{FM}$ dimensions (default: 20), after which the reduced vectors are serialized into a string $S_{pca}$ and incorporated into the prompt.

**Design Motivation**: This is the most straightforward approach. Although information is partially lost, experiments demonstrate that LLMs can effectively interpret numerical strings and use them for reasoning.

#### Module 2: Optimal Transport Alignment (OT-Embed / OT-PCA)

**Function**: Aligns the distribution of projected FM representations to the LLM embedding distribution.

**Mechanism**: A randomly initialized linear projector $P: \mathbb{R}^{d_{FM}} \to \mathbb{R}^{d_{LLM}}$ is applied, followed by OT-based distribution alignment. For each dimension $j$, a shift and scale are computed:
$$shift_j = \bar{v}_j - \bar{u}_j, \quad scale_j = \frac{\sigma_{t,j}}{\sigma_{p,j}}$$

The final aligned representation is:
$$OT(\mathcal{D}_{proj}, \mathcal{D}_{tar}) = scale \cdot \mathbf{H}_{proj} + shift$$

Two variants are proposed:
- **OT-Embed**: The target distribution is derived from LLM embeddings of the raw text (e.g., SMILES strings).
- **OT-PCA**: The target distribution is derived from LLM embeddings of PCA-serialized strings.

**Design Motivation**: Direct projection may cause distributional mismatch. OT aligns the mean and variance to ensure that FM representations fall within the normal statistical range of LLM embeddings.

#### Module 3: Theoretical Guarantees for Linear Projectors

**Function**: Formally establishes that randomly initialized linear projectors (without nonlinear activations) preserve geometric structure.

**Theorem 1 (Norm Concentration)**:
$$|\|\mathbf{W}\mathbf{u} + \mathbf{b}\|^2 - (\|\mathbf{b}\|^2 + \|\mathbf{u}\|^2)| \leq \epsilon_1 (\|\mathbf{b}\|^2 + \|\mathbf{u}\|^2)$$
where $\epsilon_1 = O(\sqrt{\log(1/\delta_1)/d})$.

**Theorem 2 (Cosine Similarity Preservation)**:
$$|\cos(\mathbf{W}\mathbf{u}, \mathbf{W}\mathbf{v}) - \cos(\mathbf{u}, \mathbf{v})| \leq \epsilon_2$$

**Corollary 1**: Nonlinear activations (ReLU, sigmoid) distort vector angles and inflate similarity estimates, leading to information loss.

**Design Motivation**: This theoretically justifies that linear projectors are better suited than MLPs for preserving the geometric structure of FM representations.

### Loss & Training

**Completely training-free**:
- The projector is randomly initialized with no gradient updates.
- OT shift/scale parameters are computed once (< 2 seconds on CPU).
- Inference uses Llama-3.1-70B-Instruct under the standard ICL protocol.

## Key Experimental Results

### Main Results: ICRL Injection Methods Comparison (RMSE ↓)

| Dataset | ICL (Text) | PCA | Zero-Pad | Random Noise | Random Proj | OT-Embed | OT-PCA |
|---------|-----------|-----|----------|--------------|-------------|----------|--------|
| ESOL | 1.16 | **1.11** | 1.73 | 1.41 | 1.69 | 1.19 | 1.24 |
| Caco2 | **0.83** | 0.95 | 1.04 | 1.03 | 1.03 | 0.89 | **0.88** |
| LD50 | **0.99** | 1.06 | 1.28 | 1.21 | 1.29 | 1.18 | 1.14 |
| AstraZeneca | **1.37** | **1.39** | 1.55 | 1.50 | 1.54 | 1.46 | 1.47 |

Text-level PCA achieves the best performance on most datasets; among embedding-level methods, the OT variants are consistently superior.

### Combined ICRL + ICL (Pearson r ↑)

| Dataset | ICL baseline | Zero-Pad+ICL | Ran-Noi+ICL | OT-Embed+ICL | OT-PCA+ICL |
|---------|-------------|--------------|-------------|--------------|------------|
| ESOL | 0.465 | 0.526 | **0.540** | 0.508 | **0.542** |
| Caco2 | 0.411 | 0.410 | 0.420 | **0.429** | 0.394 |
| AqSolDB | 0.596 | **0.606** | 0.597 | 0.569 | 0.589 |

OT-PCA+ICL achieves a **16.6%** improvement in Pearson r on ESOL.

### Ablation Study

**Effect of activation functions**:

| Projector Type | ESOL RMSE | Caco2 RMSE |
|----------------|-----------|------------|
| Linear (no activation) | **Best** | **Best** |
| +ReLU | Degraded | Degraded |
| +GELU | Degraded | Degraded |

**PCA dimensionality**: In pure ICRL mode, increasing PCA dimensions does not improve performance (and may even degrade it), suggesting that LLMs have limited capacity to process long numerical sequences.

**Cost comparison**:

| Method | Type | Resources | Training Time | ESOL RMSE |
|--------|------|-----------|---------------|-----------|
| MolecularGPT | Instruction fine-tuning | 4×A800 | < 1 day | 1.471 |
| GIMLET | Pretraining + fine-tuning | 2–4 GPUs | ~1 day | 1.132 |
| GPT-MolBERTa | Pretraining + fine-tuning | 2–4 GPUs | ~2 weeks | **0.477** |
| **OT-PCA (Ours)** | **Training-free** | **CPU only** | **~2 seconds** | 1.140 |

### Key Findings

1. **High inter-representation similarity degrades performance**: A narrow FM feature space causes projected embeddings to be highly similar, preventing the LLM from distinguishing between samples.
2. **Two operating modes**:
   - Pure ICRL: The LLM enters a *task-learning mode*, relying on few-shot demonstrations for prediction.
   - ICRL + text: The LLM enters a *task-retrieval mode*, where injected representations function more like "pause tokens" that increase deliberation time.
3. **Better FM features + text do not necessarily yield better results**: OT methods perform best in isolation, but simpler methods (e.g., random noise) outperform them when combined with text — because LLM attention is predominantly focused on the SMILES text input.

## Highlights & Insights

1. **First training-free framework for non-text modality integration**: No gradient updates are required; distribution alignment alone enables LLMs to exploit FM representations.
2. **Theoretical grounding**: The paper formally proves that linear projectors preserve geometric structure better than nonlinear MLPs, providing principled guidance for projector design.
3. **Optimal transport alignment**: Simple and efficient (< 2 seconds on CPU), with each FM representation occupying only one token position.
4. **Discovery of dual operating modes**: Reveals that ICRL representations play distinct roles depending on whether text input is present (information carrier vs. pause token).
5. **Cross-modal generalization**: Preliminary results demonstrate feasibility on visual (ViT) and speech (wav2vec2) modalities.

## Limitations & Future Work

1. Overall ICRL performance remains substantially below supervised fine-tuning methods (e.g., GPT-MolBERTa: ESOL 0.477 vs. ICRL: 1.140).
2. Evaluation is primarily conducted in the molecular domain (SMILES), where text naturally encodes structural information; validation on purely non-text modalities remains limited.
3. Pure ICRL requires more than 10 in-context examples to outperform random prediction, indicating limited few-shot sample efficiency.
4. OT alignment operates via first-order moment matching (mean/variance); more sophisticated distribution alignment (e.g., Sinkhorn) may yield further improvements.
5. The paper lacks in-depth analysis of LLM internal attention patterns (e.g., which layers are most sensitive to ICRL representations).

## Related Work & Insights

- **Relation to Vector-ICL**: Vector-ICL requires pretraining or fine-tuning to train its projector; ICRL is entirely training-free and serves as a lightweight alternative.
- **Relation to Flamingo/LLaVA**: These MLLMs require large-scale paired data for training; ICRL is applicable to scenarios where paired data is scarce (e.g., molecular property prediction, protein modeling).
- **Connection to pause tokens**: When text input is present, ICRL representations are used as pause tokens, echoing findings from the "Think before you speak" line of work.
- **Practical implications**: In domains lacking domain-specific pretrained multimodal LLMs (e.g., sensor data, biomedical signals), ICRL offers a readily deployable lightweight solution.

## Rating

⭐⭐⭐⭐ (4/5)

The paper addresses a novel problem formulation (training-free cross-modal reasoning) with solid theoretical analysis (geometric preservation via linear projection) and insightful mechanistic findings (dual operating modes). Experiments, though primarily conducted in the molecular domain, are comprehensive and systematic. The main limitation is the substantial performance gap relative to supervised methods, and the practical trade-offs of the training-free advantage vary across settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] VT-FSL: Bridging Vision and Text with LLMs for Few-Shot Learning](vt-fsl_bridging_vision_and_text_with_llms_for_few-shot_learning.md)
- [\[NeurIPS 2025\] What Can RL Bring to VLA Generalization? An Empirical Study](what_can_rl_bring_to_vla_generalization_an_empirical_study.md)
- [\[NeurIPS 2025\] Training-free Online Video Step Grounding](training-free_online_video_step_grounding.md)
- [\[NeurIPS 2025\] Can Multi-Modal LLMs Provide Live Step-by-Step Task Guidance?](can_multi-modal_llms_provide_live_step-by-step_task_guidance.md)
- [\[NeurIPS 2025\] In-Context Compositional Learning via Sparse Coding Transformer](in-context_compositional_learning_via_sparse_coding_transformer.md)

</div>

<!-- RELATED:END -->
