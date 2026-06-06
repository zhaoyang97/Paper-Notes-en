---
title: >-
  [Paper Note] In-Context Compositional Learning via Sparse Coding Transformer
description: >-
  [NeurIPS 2025][Multimodal VLM][Sparse Coding] Inspired by sparse coding, this work reinterprets the Transformer attention mechanism as projection onto encoding and decoding dictionaries…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "Sparse Coding"
  - "Attention Mechanism"
  - "Compositional Learning"
  - "In-Context Learning"
  - "Transformer"
date: 2026-05-08
content_hash: 1bfa780d8ef87b6e
---

# In-Context Compositional Learning via Sparse Coding Transformer

**Conference**: NeurIPS 2025
**arXiv**: [2511.20194](https://arxiv.org/abs/2511.20194)  
**Code**: N/A  
**Area**: Multimodal VLM
**Keywords**: Sparse Coding, Attention Mechanism, Compositional Learning, In-Context Learning, Transformer

## TL;DR

Inspired by sparse coding, this work reinterprets the Transformer attention mechanism as projection onto encoding and decoding dictionaries, explicitly represents compositional rules via sparse coefficients, and transfers compositional rules from in-context tasks to target tasks using a lifting scheme.

## Background & Motivation

In-context compositional learning requires a model to infer compositional rules from in-context examples and apply them to target problems. Standard Transformers face two fundamental limitations in such tasks:

**Softmax produces dense attention weights**: This leads to indiscriminate global mixing of information, making it impossible to represent compositional structures in the input.

**Lack of rule transfer mechanism**: The model cannot effectively extract and reuse local compositional rules from in-context examples.

Core observation: When the target task input is zero (unobserved), the softmax attention of a standard Transformer degenerates to a uniform distribution $\frac{1}{N}\mathbf{1}$, and the output becomes a simple average of all inputs — directly causing ambiguous predictions.

## Method

### Overall Architecture

The multi-head attention (MHA) is decomposed into three steps:

1. **Encoding**: Project inputs onto an encoding dictionary $\phi(\mathbf{X})$ to obtain coefficients.
2. **Sparsification**: Apply a sparsity constraint to the coefficients.
3. **Decoding**: Linearly combine the decoding dictionary $\psi(\mathbf{X})$ using sparse coefficients to produce the output.

Mathematical formulation:

$$\text{MHA}(\mathbf{X}) = \sum_{h=1}^{H} \sigma(\mathbf{X} \underbrace{\phi^{(h)}(\mathbf{X})}_{\text{encoding dict.}}) \underbrace{\psi^{(h)}(\mathbf{X})}_{\text{decoding dict.}}$$

where the encoding dictionary is $\phi^{(h)}(\mathbf{X}) = \mathbf{W}_{qk}^{(h)} \mathbf{X}^\top$ and the decoding dictionary is $\psi^{(h)}(\mathbf{X}) = \mathbf{X} \mathbf{W}_{vo}^{(h)}$.

### Key Designs

#### 1. Sparse Coefficients

Soft-thresholding replaces softmax as the nonlinearity $\sigma(\cdot)$:

$$\text{prox}(\mathbf{S}) = \text{sign}(\mathbf{S}) \odot \max(|\mathbf{S}| - \xi, 0)$$

The sparse coefficients $\boldsymbol{\alpha} = \sigma(\mathbf{X} \phi(\mathbf{X}))$ retain the most informative components, suppress redundant interactions, and align representations more closely with underlying compositional rules.

Critical analysis — when the target input $\mathbf{X}_L = \mathbf{0}$:

- **Standard Transformer**: $\boldsymbol{\alpha}_L = \text{softmax}(\mathbf{0}) = \frac{1}{N}\mathbf{1}$ → output is global average → ambiguous
- **Proposed method**: $\boldsymbol{\alpha}_L = \text{prox}(\mathbf{0}) = \mathbf{0}$ → honestly outputs zero → requires coefficient transfer

#### 2. Target Task Coefficient Estimation (Coefficient Transfer)

Inspired by the lifting scheme, target coefficients are estimated as a linear combination of in-context coefficients:

$$\boldsymbol{\alpha}_L \leftarrow \boldsymbol{\alpha}_L + \sum_{i=1}^{L-1} \lambda_i \boldsymbol{\alpha}_i$$

where $\lambda_i$ are learnable parameters. The updated output is:

$$\mathbf{Z}_L = \boldsymbol{\alpha}_L \psi(\mathbf{X}) + \sum_{i=1}^{L-1} \lambda_i \boldsymbol{\alpha}_i \psi(\mathbf{X})$$

This is highly parameter-efficient: only $L-1$ learnable parameters are added per layer (equal to the number of in-context tasks), introducing near-zero overhead.

#### 3. Basis Function Variants

Nonlinearities (e.g., ReLU) can be introduced into the basis functions $\phi(\cdot), \psi(\cdot)$ of the encoding/decoding dictionaries to enhance expressiveness:

$$\phi(\mathbf{X}) = \text{ReLU}(\mathbf{W}_{qk}^{(h)} \mathbf{X}), \quad \psi(\mathbf{X}) = \text{ReLU}(\mathbf{W}_{vo}^{(h)} \mathbf{X})$$

Experiments show that applying ReLU to both sides achieves the best performance (73.6% vs. 71.7%).

### Loss & Training

- Synthetic datasets: MSE loss, Adam optimizer, lr=0.001, 200 epochs, batch size 128
- S-RAVEN: Cross-entropy loss, Adam, lr=0.001, weight decay 0.1, 1 epoch, batch size 128
- RAVEN: MSE loss, Adam, lr=0.0001, 2000 epochs, batch size 256
- RMS normalization is applied to all models

## Key Experimental Results

### Main Results (S-RAVEN)

| Method | 4L/10M | 4L/20M | 4L/40M | 8L/10M | 8L/20M | 8L/40M |
|--------|--------|--------|--------|--------|--------|--------|
| Transformer | 51.6 | 55.7 | 58.1 | 59.8 | 63.3 | 65.1 |
| HYLA | 55.0 | 68.6 | 73.2 | 72.5 | 77.1 | 79.3 |
| **Ours** | **63.1** | **73.9** | **76.3** | **72.6** | **78.2** | **82.7** |

### RAVEN Dataset

Under the strict criterion of PSNR > 40, nearly 0% of test samples from the standard Transformer meet the threshold, whereas the proposed method maintains approximately 40% — a substantial gap.

### Ablation Study

| Threshold $\xi$ | 0.003 | 0.01 | 0.03 | 0.1 | 0.3 |
|-----------------|-------|------|------|-----|-----|
| Sparsity (%) | 18.53 | 57.82 | 90.45 | 97.82 | 99.38 |

| Basis Function Configuration | Accuracy |
|------------------------------|----------|
| Linear $\phi$, Linear $\psi$ | 71.7 |
| ReLU $\phi$, Linear $\psi$ | 72.3 |
| Linear $\phi$, ReLU $\psi$ | 72.9 |
| ReLU $\phi$, ReLU $\psi$ | **73.6** |

### Key Findings

1. **Sparsity is critical for compositional learning**: Dense attention in standard Transformers completely fails on compositional generalization tasks (producing ambiguous outputs), while sparse attention yields sharp and structured predictions.
2. **Coefficient transfer is indispensable**: Without it, even with sparsity, the target task output remains zero.
3. **Effective with limited data**: A 4-layer model trained on 10M samples outperforms an 8-layer standard Transformer trained on 40M samples.
4. **Applicable to language models**: When integrated into Llama-7B, the method surpasses the base model on commonsense reasoning with approximately a hundred parameters, compared to 50M+ for LoRA.

## Highlights & Insights

1. **Theoretical elegance**: A rigorous correspondence between attention mechanisms and sparse coding is established — not merely an analogy but a mathematical reformulation.
2. **Incisive problem analysis**: The root cause of standard Transformer degradation to uniform averaging when target inputs are zero is precisely identified.
3. **Extreme parameter efficiency**: Only $L-1$ parameters per layer (typically < 10) are added; coefficient transfer incurs virtually zero overhead.
4. **Controllable sparsity**: Attention sparsity can be precisely controlled via the threshold $\xi$.

## Limitations & Future Work

1. Validation is limited to small-scale synthetic datasets (S-RAVEN, RAVEN); generalization to large-scale real-world tasks remains undemonstrated.
2. Language model experiments (Llama-7B) do not yet match LoRA/DoRA performance, though the parameter count is significantly smaller.
3. The gradient properties of soft-thresholding during backpropagation may constrain training of deeper networks.
4. The linear combination assumption in coefficient transfer may be insufficient for complex nonlinear compositional rules.

## Related Work & Insights

- **HYLA**: A strong baseline on S-RAVEN that employs hybrid linear attention but does not explicitly model compositional rules.
- **Sparse attention** (Longformer, BigBird, etc.): Targets computational complexity reduction, whereas this work aims to enhance compositional reasoning — fundamentally different motivations.
- **Insight**: The sparse coding perspective may offer a new framework for understanding the in-context learning capabilities of large language models.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The idea of reformulating attention mechanisms from a sparse coding perspective is highly original.
- Experimental Thoroughness: ⭐⭐⭐ — Synthetic dataset validation is thorough, but large-scale real-world task validation is lacking.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical derivations and analyses are clearly presented.
- Value: ⭐⭐⭐⭐ — Offers a new perspective for understanding and improving compositional reasoning in Transformers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] TOMCAT: Test-time Comprehensive Knowledge Accumulation for Compositional Zero-Shot Learning](tomcat_test-time_comprehensive_knowledge_accumulation_for_compositional_zero-sho.md)
- [\[NeurIPS 2025\] Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models](sparse_autoencoders_learn_monosemantic_features_in_visionlan.md)
- [\[NeurIPS 2025\] Advancing Compositional Awareness in CLIP with Efficient Fine-Tuning](advancing_compositional_awareness_in_clip_with_efficient_fin.md)
- [\[NeurIPS 2025\] Can LLMs Reason Over Non-Text Modalities in a Training-Free Manner? A Case Study with In-Context Representation Learning](can_llms_reason_over_non-text_modalities_in_a_training-free_manner_a_case_study_.md)
- [\[ICLR 2026\] EgoHandICL: Egocentric 3D Hand Reconstruction with In-Context Learning](../../ICLR2026/multimodal_vlm/egohandicl_egocentric_3d_hand_reconstruction_with_in-context_learning.md)

</div>

<!-- RELATED:END -->
