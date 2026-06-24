---
title: >-
  [Paper Note] DenseLoRA: Dense Low-Rank Adaptation of Large Language Models
description: >-
  [ACL 2025][LLM (Other)][Parameter-Efficient Fine-Tuning] This paper proposes DenseLoRA, which introduces a cross-layer shared Encoder-Decoder for the joint compression and reconstruction of hidden representations. It replaces two redundant low-rank matrices in LoRA with a single small, dense low-rank matrix for adaptation. With only 0.01% of trainable parameters, it achieves 83.8% accuracy on LLaMA3-8B, surpassing LoRA which achieves 80.8% with 0.70% parameters.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Parameter-Efficient Fine-Tuning"
  - "LoRA"
  - "Low-Rank Adaptation"
  - "Representation Fine-Tuning"
  - "Parameter Redundancy"
date: 2026-05-08
content_hash: 42a2edd38357366b
---

# DenseLoRA: Dense Low-Rank Adaptation of Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2505.23808](https://arxiv.org/abs/2505.23808)  
**Code**: [https://github.com/mulin-ahu/DenseLoRA](https://github.com/mulin-ahu/DenseLoRA)  
**Area**: LLM/NLP  
**Keywords**: Parameter-Efficient Fine-Tuning, LoRA, Low-Rank Adaptation, Representation Fine-Tuning, Parameter Redundancy

## TL;DR
This paper proposes DenseLoRA, which introduces a cross-layer shared Encoder-Decoder for the joint compression and reconstruction of hidden representations. It replaces two redundant low-rank matrices in LoRA with a single small, dense low-rank matrix for adaptation. With only 0.01% of trainable parameters, it achieves 83.8% accuracy on LLaMA3-8B, surpassing LoRA which achieves 80.8% with 0.70% parameters.

## Background & Motivation
LoRA significantly reduces the number of trainable parameters via low-rank matrix decomposition ($\Delta W = BA$) and is currently the most popular parameter-efficient fine-tuning method. However, research indicates there is substantial weight redundancy in LoRA's low-rank matrices: many parameter increments approach zero during training, playing a marginal role in adaptation.

Existing LoRA variants (such as AdaLoRA, DoRA, etc.) attempt to address redundancy by selectively identifying important weights, but they remain constrained by the traditional dual low-rank matrix framework. This paper poses a fundamental question: **Is it possible to develop a low-rank adaptation method that utilizes a denser structure to achieve better performance with fewer parameters?**

The Core Idea is to refine not only the weight matrices but also the hidden representations themselves. Inspired by representation fine-tuning, this study integrates low-rank adaptation with representation compression.

## Method

### Overall Architecture
The adaptation process of DenseLoRA is structured as a three-stage pipeline: (1) an Encoder compresses the hidden representations; (2) a dense low-rank matrix M adapts the compressed representations; (3) a Decoder reconstructs the adapted representations back to the original dimension. The key innovation is that the Encoder-Decoder is shared across all adapted layers, while each layer maintains an independent adaptation matrix M.

### Key Designs
1. **Encoder Compression Module**:

    - Uses a fully connected network $W_e \in \mathbb{R}^{r \times k}$ to compress the hidden representation $h \in \mathbb{R}^k$ to a low-dimensional representation $h' \in \mathbb{R}^r$
    - Followed by an activation function $\sigma(\cdot)$
    - Initialized using Kaiming initialization
    - Shared across all adapted layers to reduce parameter redundancy

2. **Dense Low-Rank Adaptation Matrix**:

    - Each layer employs an independent square matrix $M \in \mathbb{R}^{r \times r}$ for adaptation
    - Unlike LoRA's $B \times A$ (the product of two matrices), DenseLoRA uses a single small dense square matrix
    - Although it is a small $r \times r$ matrix, because it shares the compression and reconstruction capabilities of the Encoder-Decoder, it actually learns a more effective adaptation
    - Initialized using Kaiming initialization

3. **Decoder Reconstruction Module**:

    - Uses $W_d \in \mathbb{R}^{d \times r}$ to reconstruct the adapted representation back to the original dimension
    - Followed by an activation function
    - Zero-initialized (to ensure no interference with forward propagation at the beginning of training)
    - Shared across layers, similar to the Encoder

4. **Parameter Complexity Analysis**:

    - LoRA: $|\Theta| = l \times (d+k) \times r$ (where $l$ is the number of adapted layers)
    - DenseLoRA: $|\Theta| = (d+k+l \times r) \times r$
    - Practical Comparison: For LLaMA2-7B with $r=16$, LoRA requires 28M parameters, whereas DenseLoRA requires only 0.9M, achieving a 30x compression

### Loss & Training
Overall adaptation formulation: $\hat{h} = W_0 h + Decoder(M \cdot Encoder(h))$

Standard cross-entropy loss is employed for fine-tuning. The Encoder is initialized with Kaiming initialization, and the Decoder is initialized with zeros to ensure stability at the start of training. Training is executed on 4×NVIDIA 3090 24GB GPUs.

## Key Experimental Results

### Main Results - Commonsense Reasoning (LLaMA3-8B)

| Method | Params (%) | BoolQ | PIQA | HellaS. | WinoG. | ARC-e | ARC-c | OBQA | Avg. |
|------|----------|-------|------|---------|--------|-------|-------|------|------|
| LoRA | 0.70 | 70.8 | 85.2 | 91.7 | 84.3 | 84.2 | 71.2 | 79.0 | 80.8 |
| VeRA | 0.01 | 62.2 | 81.6 | 54.5 | 6.18 | 84.4 | 67.2 | 64.6 | 67.7 |
| LoKr | 0.01 | 65.1 | 81.6 | 92.0 | 82.1 | 89.2 | 76.7 | 80.9 | 80.9 |
| DoRA | 0.71 | 74.6 | 89.3 | 95.5 | 85.6 | 90.5 | 80.4 | 85.8 | 85.2 |
| **DenseLoRA**(r=16) | **0.01** | 72.3 | 87.5 | 93.5 | 85.2 | 89.8 | 78.2 | 84.0 | **83.8** |
| DenseLoRA(r=32) | 0.02 | 74.3 | 88.0 | 94.5 | 86.0 | 89.7 | 78.7 | 85.6 | 84.6 |
| DenseLoRA(r=64) | 0.06 | 74.1 | 88.9 | 95.0 | 87.0 | 90.0 | 79.2 | 85.6 | 85.0 |

### Mathematical Reasoning (LLaMA3-8B)

| Method | Params (%) | GSM8K | AQUA | AddSub | SVAMP | Avg. |
|------|----------|-------|------|--------|-------|------|
| LoRA | 0.70 | 47.1 | 18.1 | 90.6 | 71.9 | 56.9 |
| DenseLoRA(r=32) | 0.02 | 45.5 | 20.5 | 73.5 | 92.1 | 57.5 |
| DenseLoRA(r=64) | 0.06 | 47.2 | 19.7 | 92.4 | 74.5 | 58.5 |

### Ablation Study

| Configuration | Key Metric (Avg.) | Description |
|------|--------------|------|
| DenseLoRA, QKV modules only | 82.3 | MHA layer adaptation |
| DenseLoRA, UD modules only | 83.8 | Better adaptation performance on MLP layers |
| DenseLoRA QKVUD | 84.6 | Optimal configuration |
| LoRA QKVUD + No DenseLoRA | 80.8 | Traditional LoRA |
| DenseLoRA with 10% training data | 81.1 | Surpasses LoRA trained on 100% data (80.8) |

### Key Findings
- DenseLoRA surpasses LoRA by 3 percentage points (83.8% vs 80.8%) using only 1/70 of the parameters (0.01% vs 0.70%).
- At $r=64$, DenseLoRA achieves 85.0%, which is close to DoRA (85.2%) but requires only 1/6 of its parameters.
- The advantage is even more pronounced in low-resource scenarios: DenseLoRA trained on 10% data (81.1%) outperforms LoRA trained on 100% data (80.8%).
- Adaptation of MLP layers is more critical than attention layers: adapting only the UD modules achieves 83.8%.
- It is equally effective on mathematical reasoning tasks: 58.5% at $r=64$ compared to 56.9% for LoRA.

## Highlights & Insights
- Integrating representation fine-tuning with low-rank adaptation is highly creative, breaking out of the conventional framework of "optimizing AB matrices."
- The cross-layer shared Encoder-Decoder design is highly elegant: it simultaneously reduces the parameter count and maintains consistency in the compressed representations.
- The parameter efficiency improvement is remarkable: achieving a 30-70x parameter compression while delivering superior performance.
- Excellent performance in low-resource scenarios (10% data) indicates that DenseLoRA possesses stronger generalization capabilities.
- The finding that MLP layer adaptation is more crucial than attention layer adaptation is somewhat counter-intuitive, warranting further investigation.

## Limitations & Future Work
- The experiments primarily focus on commonsense and mathematical reasoning, lacking validation on NLG tasks (e.g., summarization, translation).
- Sharing the Encoder-Decoder across all layers might be less flexible in deep models with substantial variance between layers.
- Zero-initialization of the Decoder implies weak adaptation signals in the early training phases, potentially affecting convergence speed.
- Evaluation was limited to 7B/8B models, lacking experiments on larger scales (70B+).
- Research idea: explore adaptive rank assignment for each layer, rather than utilizing a fixed rank $r$ across all layers.
- During inference, DenseLoRA cannot be directly merged into the original weight matrices like LoRA (due to the presence of non-linear activation functions and Encoder/Decoder architectures), introducing extra inference latency. This represents a significant practical limitation.

## Related Work & Insights
- Related to the ReFT (Representation Fine-tuning) family, but DenseLoRA deeply integrates representation fine-tuning and low-rank adaptation.
- LoKr uses Kronecker product decomposition but suffers from higher computational costs, whereas DenseLoRA's computational overhead is comparable to LoRA.
- VeRA also employs shared matrices but exhibits a substantial performance gap (67.7% vs 83.8%), demonstrating the effectiveness of DenseLoRA's three-stage design.
- NoRA adopts nested structures and SVD; the formulation differs, but the objective remains similar.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Combining representation fine-tuning with low-rank adaptation is novel, though the Encoder-Decoder design itself is not entirely new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Ablation studies are comprehensive with various rank configurations and module combinations, though the coverage of task types and model scales could be broader.
- **Writing Quality**: ⭐⭐⭐⭐ The methodology is clearly described, and the parameter analysis is thorough, though more discussion on inference latency could be integrated.
- **Value**: ⭐⭐⭐⭐ The substantial improvement in parameter efficiency holds significant practical value; however, the inference latency issue limits certain deployment scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] TableLoRA: Low-rank Adaptation on Table Structure Understanding for Large Language Models](table_lora_structure_understanding.md)
- [\[ACL 2025\] GORP: Continual Gradient Low-Rank Projection Fine-Tuning for LLMs](gorp_continual_gradient_projection.md)
- [\[ACL 2025\] Cultural Learning-Based Culture Adaptation of Language Models](cultural_learning-based_culture_adaptation_of_language_models.md)
- [\[ACL 2025\] Efficient Ensemble for Fine-tuning Language Models on Multiple Datasets](efficient_ensemble_for_fine-tuning_language_models_on_multiple_datasets.md)
- [\[ACL 2025\] RoCoFT: Efficient Finetuning of Large Language Models with Row-Column Updates](rocoft_efficient_finetuning_of_large_language_models_with_row-column_updates.md)

</div>

<!-- RELATED:END -->
