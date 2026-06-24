---
title: >-
  [Paper Note] Language Fusion for Parameter-Efficient Cross-lingual Transfer (FLARE)
description: >-
  [ACL 2025][Multilingual & Machine Translation][cross-lingual transfer] FLARE fuses layer-wise representations of the source (English) and target languages via lightweight linear/non-linear transformations within the low-rank bottleneck of LoRA adapters. It achieves parameter-efficient cross-lingual transfer without requiring extra parameters, improving QA exact match by 4.9% on Llama 3.1.
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "cross-lingual transfer"
  - "LoRA"
  - "language fusion"
  - "adapter bottleneck"
  - "multilingual NLU"
date: 2026-05-08
content_hash: decbf2d2b5a3bf4d
---

# Language Fusion for Parameter-Efficient Cross-lingual Transfer (FLARE)

**Conference**: ACL 2025  
**arXiv**: [2501.06892](https://arxiv.org/abs/2501.06892)  
**Code**: [https://github.com/pnborchert/FLARE](https://github.com/pnborchert/FLARE)  
**Area**: Multilingual Translation  
**Keywords**: cross-lingual transfer, LoRA, language fusion, adapter bottleneck, multilingual NLU

## TL;DR
FLARE fuses layer-wise representations of the source (English) and target languages via lightweight linear/non-linear transformations within the low-rank bottleneck of LoRA adapters. It achieves parameter-efficient cross-lingual transfer without requiring extra parameters, improving QA exact match by 4.9% on Llama 3.1.

## Background & Motivation
**Background**: Multilingual Pre-trained Language Models (mPLMs) are dominated by English corpora, leading to undertrained representation spaces for non-English languages, resulting in significantly lower downstream performance compared to English.

**Limitations of Prior Work**:
   - Input-level fusion (concatenating source + target sequences) doubles sequence length, leading to quadratic growth in attention computation.
   - X-Mixup only performs cross-attention alignment at a single transformer layer and requires extra parameters.
   - Translate-test loses cultural details and semantics.
   - Translate-train utilizing standard LoRA does not fully leverage source language information.

**Key Challenge**: Improving cross-lingual transfer typically requires processing bilingual inputs or additional modules, which increases computational complexity and conflicts with the goal of parameter-efficient fine-tuning.

**Key Insight**: LoRA already compresses representations into a low-rank bottleneck. This compressed space can be leveraged to fuse bilingual information with almost zero additional overhead.

**Core Idea**: Fuse the source and target language representations using simple element-wise operations after the down-projection and before the up-projection inside LoRA.

## Method

### Overall Architecture
The workflow of FLARE: (1) First, fine-tune the base model using standard LoRA on English task data; (2) During fine-tuning on the target language (using machine-translated parallel data), extract layer-wise representations $V^S$ of the English (source language) input using the base model (without the fusion adapter); (3) Pass the target language input through the model equipped with the fusion adapter, combining the source representation $v_{i+1}^S W^{down}$ and target representation $v_i^T W^{down}$ within each layer's LoRA bottleneck via a fusion function $\phi$; (4) The fused low-rank representation is added to the frozen attention output after up-projection.

### Key Designs

1. **Bottleneck Fusion**:

    - **Function**: Fuse bilingual representations in the low-rank space of LoRA ($r \ll d$).
    - **Mechanism**: The source representation is $S = v^S W^{down}$ and the target representation is $T = v^T W^{down}$, where sharing the down-projection ensures they are in the same space. The fusion function $\phi$ includes: addition $S+T$, multiplication $S \circ T$, ReLU variants (add+relu, mul+relu), and cross-attention.
    - **Design Motivation**: Reuse existing down/up projections of LoRA to avoid extra parameters (except for cross-attention). Fusing in the low-rank space requires much lower computational cost than fusing in the original high-dimensional space.

2. **Layer-wise Representation Extraction and Shift**:

    - **Function**: Fuse the $i+1$-th layer representation of the source language from the base model with the $i$-th layer representation of the target language.
    - **Mechanism**: $h = \phi(v_{i+1}^S W^{down}, v_i^T W^{down})$, where the source language "leads by one layer".
    - **Design Motivation**: The $i+1$-th layer has already been processed by that transformer block, containing richer task-specific information that can "guide" the target language's $i$-th layer learning.

3. **FLARE MT Variant**:

    - **Function**: Replace the source language representation of the mPLM with the encoder representation of a machine translation model (latent translation).
    - **Mechanism**: Directly map the target language to a "latent translation" $v^T = \mathcal{M}(x^T)$ using an NLLB encoder, followed by fusion after a linear projection.
    - **Design Motivation**: Avoid the forward pass of the source language in the mPLM, further reducing computational cost.

### Comparison of Fusion Functions

| Fusion Function | XNLI | TyDiQA | NusaX | Requires Extra Parameters |
|---------|------|--------|-------|-------------|
| add | 80.53 | 40.31 | 78.73 | No |
| mul | 79.59 | 36.23 | 77.73 | No |
| add+relu | 80.99 | 40.93 | 79.18 | No |
| cross-attention | 80.66 | 39.15 | 77.72 | Yes (Few) |

add+relu performs best without requiring extra parameters.

## Key Experimental Results

### Main Results
Average performance and rankings of FLARE vs various baselines across four mPLMs × three tasks:

| Method | XLM-R Large | mT5-XL | Llama 3.1 8B | Gemma 2 9B |
|------|------------|--------|-------------|-----------|
| LoRA | 65.88 / rank 3.33 | 68.99 / rank 3.67 | 55.55 / rank 3.33 | 52.37 / rank 3.67 |
| X-Mixup | 64.69 / rank 4.67 | 68.82 / rank 4.00 | - | - |
| Input-level fusion | 65.41 / rank 3.00 | 68.84 / rank 4.33 | 55.86 / rank 3.33 | 52.54 / rank 3.00 |
| **FLARE** | **67.03 / rank 1.33** | **69.89 / rank 1.33** | **57.42 / rank 1.33** | **53.54 / rank 1.67** |

FLARE ranks first on average across all models.

### QA Task Improvement (TyDiQA Exact Match)

| Model | LoRA | FLARE | Gain |
|------|------|-------|------|
| Llama 3.1 8B | 15.88 | 20.77 | **+4.9%** |
| Gemma 2 9B | 4.21 | 6.38 | **+2.2%** |
| mT5-XL | 46.76 | 48.94 | +2.2% |
| XLM-R Large | 40.14 | 40.93 | +0.8% |

### Ablation Study

| Configuration | Avg Performance | Description |
|------|----------------|------|
| FLARE (add+relu) | 67.03 | Full model (XLM-R) |
| w/o fusion (standard LoRA) | 65.88 | Removing fusion drops performance by 1.15 |
| FLARE MT | 65.89 | Replaces mPLM representations with MT encoder, close performance |
| Input-level fusion | 65.41 | Concatenates input sequences, underperforms FLARE |
| X-Mixup | 64.69 | Single-layer cross-attention, worst performance |

### Key Findings
- **QA tasks benefit the most**: Decoder-only models (Llama, Gemma) show the most significant improvement in QA (+4.9%), as generative QA depends more heavily on cross-lingual understanding.
- **add+relu is the best fusion function**: Simple yet effective, where ReLU helps filter information from unaligned tokens.
- **FLARE MT is feasible**: Replacing the mPLM's source language representation with a much smaller MT encoder (600M) results in minimal performance loss while being more efficient.
- **Low-resource languages benefit more**: The most significant improvements are observed in NusaX (11 Indonesian languages).

## Highlights & Insights
- **A novel perspective on the LoRA bottleneck**: Beyond dimension reduction, it can serve as a bridge for cross-lingual information fusion. This concept can be transferred to other cross-modal fusion scenarios (e.g., vision-language), leveraging LoRA infrastructure to achieve lightweight modal fusion.
- The **"one-layer-ahead" source representation** design is intuitively similar to using a "teacher signal to guide a student" in knowledge distillation, utilizing deeper source representations to guide the learning of the target language.
- **FLARE MT demonstrates a possibility**: The latent representations of an MT encoder can be directly injected into an LLM, avoiding discretization loss during translation.

## Limitations & Future Work
- Relies on machine translation to generate parallel corpora; hence, MT quality directly impacts effectiveness.
- The improvement of decoder-only models on classification tasks is not as pronounced as on QA.
- Performance in extremely low-resource scenarios (without any parallel corpora) has not been validated.
- The choice of the fusion function remains empirical, and more optimal, learnable fusion methods may exist.

## Related Work & Insights
- **vs X-Mixup**: X-Mixup fuses representations at a single layer using cross-attention, which requires extra parameters and is sensitive to layer selection. In contrast, FLARE fuses within the LoRA bottlenecks of all layers, requires no extra parameters, and is more stable.
- **vs Input-level fusion**: Concatenating inputs doubles the sequence length, whereas FLARE processes them independently and fuses them in the low-rank space, requiring significantly less computation.
- **vs AdaMergeX**: AdaMergeX merges the weights of the English task adapter and the target language adapter. FLARE dynamically fuses representations during training, enabling the learning of finer-grained interactions.

## Rating
- Novelty: ⭐⭐⭐⭐ Fusing cross-lingual representations within the LoRA bottleneck is a simple and effective new idea.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 models × 3 tasks × multiple baselines × fusion function ablation.
- Writing Quality: ⭐⭐⭐⭐ Clear method description, intuitive illustrations, and sound experimental design.
- Value: ⭐⭐⭐⭐ Practically valuable for cross-lingual PEFT, especially in generative tasks like QA.

## Related Work & Insights
- Refer to the Related Work section of the original paper for detailed comparisons.

## Rating
- Novelty: ⭐⭐⭐⭐ Language fusion within LoRA.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-task and multi-model.
- Value: ⭐⭐⭐⭐ Practical parameter-efficient cross-lingual method.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Cross-Lingual Optimization for Language Transfer in Large Language Models](cross-lingual_optimization_for_language_transfer_in_large_language_models.md)
- [\[ACL 2025\] Semantic Aware Linear Transfer by Recycling Pre-trained Language Models for Cross-Lingual Transfer](semantic_aware_linear_transfer_by_recycling_pre-trained_language_models_for_cros.md)
- [\[ACL 2025\] Translation and Fusion Improves Zero-shot Cross-lingual Information Extraction](translation_and_fusion_improves_cross-lingual_information_extraction.md)
- [\[ACL 2025\] Cross-Lingual Transfer of Cultural Knowledge: An Asymmetric Phenomenon](cross-lingual_transfer_of_cultural_knowledge_an_asymmetric_phenomenon.md)
- [\[ACL 2025\] Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs](mid_layer_crosslingual_alignment.md)

</div>

<!-- RELATED:END -->
