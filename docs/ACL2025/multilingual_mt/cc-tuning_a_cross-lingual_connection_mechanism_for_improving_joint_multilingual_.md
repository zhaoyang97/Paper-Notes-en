---
title: >-
  [Paper Note] CC-Tuning: A Cross-Lingual Connection Mechanism for Improving Joint Multilingual Supervised Fine-Tuning
description: >-
  [ACL 2025 Main Conference (Long Paper)][Multilingual & Machine Translation][Cross-Lingual Connection] This paper proposes CC-Tuning, a multilingual fine-tuning paradigm that explicitly establishes cross-lingual connections in the hidden space. It enhances non-English capabilities by fusing feed-forward activations of English and non-English inputs, and employs a Transform Matrix during inference to simulate this cross-lingual connection.
tags:
  - "ACL 2025 Main Conference (Long Paper)"
  - "Multilingual & Machine Translation"
  - "Cross-Lingual Connection"
  - "Multilingual Fine-Tuning"
  - "Hidden Space Interaction"
  - "Decision Maker"
  - "Representation Transformation"
date: 2026-05-08
content_hash: 6086cbd0eed01236
---

# CC-Tuning: A Cross-Lingual Connection Mechanism for Improving Joint Multilingual Supervised Fine-Tuning

**Conference**: ACL 2025 Main Conference (Long Paper)  
**arXiv**: [2506.00875](https://arxiv.org/abs/2506.00875)  
**Code**: None  
**Area**: Multilingual Translation  
**Keywords**: Cross-Lingual Connection, Multilingual Fine-Tuning, Hidden Space Interaction, Decision Maker, Representation Transformation

## TL;DR

This paper proposes CC-Tuning, a multilingual fine-tuning paradigm that explicitly establishes cross-lingual connections in the hidden space. It enhances non-English capabilities by fusing feed-forward activations of English and non-English inputs, and employs a Transform Matrix during inference to simulate this cross-lingual connection.

## Background & Motivation

**Background**: Current large language models are predominantly pre-trained on English corpora, leading to imbalanced multilingual capabilities where English performance is strong while other languages lag behind. Multilingual Supervised Fine-Tuning (Multilingual SFT) is a common approach to enhance non-English capabilities, but existing methods primarily operate at the data-level, such as utilizing translation for data augmentation or knowledge distillation.

**Limitations of Prior Work**: Data-level methods (e.g., translating English data into target languages or distilling multilingual data using stronger models) can only introduce implicit cross-lingual alignment. Models are left to learn associations between languages on their own during training, which is passive and insufficient. These methods overlook a deeper possibility: directly performing cross-lingual information interaction within the model's internal representation space.

**Key Challenge**: LLMs can activate rich knowledge and reasoning capabilities when processing English inputs, but these capabilities cannot be invoked to the same extent when processing non-English inputs. The root cause is not the lack of multilingual data, but the absence of an effective cross-lingual information transmission mechanism within the model.

**Goal**: To design a fine-tuning method that establishes direct cross-lingual connections in the hidden space of the model, allowing non-English inputs to "borrow" the powerful capabilities activated by English inputs within the model.

**Key Insight**: It is observed that when LLMs process English and non-English inputs with the same semantic meaning, the activation patterns produced by the Feed-Forward Network (FFN) differ—English activations typically contain richer knowledge signals. If these beneficial English activations can be "injected" into the non-English computational process during training, cross-lingual knowledge transfer can be directly achieved in the hidden space.

**Core Idea**: Establish cross-lingual connections by fusing English and non-English FFN activations during training, filter beneficial activations via a trainable Decision Maker, and simulate this connection using a Transform Matrix in monolingual scenarios during inference.

## Method

### Overall Architecture

The core of CC-Tuning is an "asymmetric training-inference" design. In the training phase, for each non-English training sample, its corresponding English version is provided. FFN activations of both languages are fused at each layer of the model, with the fusion weights determined by a Decision Maker. In the inference phase, since English reference inputs are no longer available, a trained Transform Matrix is used to project the non-English activations into the activation space "as if English auxiliary signals were present", thereby simulating the cross-lingual connection effect from training.

### Key Designs

1. **Cross-Lingual Activation Fusion**:

    - **Function**: Injects beneficial signals from English activations into the non-English computational process during training.
    - **Mechanism**: For the feed-forward network of each Transformer layer, the activation vectors of the English and non-English inputs, $h_{en}$ and $h_{non-en}$, are computed simultaneously. They are then fused via weighted addition to obtain the final activation $h_{fused} = \alpha \cdot h_{en} + (1-\alpha) \cdot h_{non-en}$, where $\alpha$ is dynamically determined by the Decision Maker. Consequently, the non-English path can directly "borrow" beneficial activation signals from the English path.
    - **Design Motivation**: Compared to data-level translation augmentation, activation fusion in the hidden space offers a more direct and fine-grained approach to cross-lingual knowledge transfer. Since not all English activations are helpful to non-English tasks, a Decision Maker is required for filtering.

2. **Decision Maker**:

    - **Function**: Dynamically evaluates whether the English activations at each layer and each dimension are beneficial for the non-English inputs.
    - **Mechanism**: The Decision Maker is a lightweight, trainable module (such as a linear layer or a gating network) that takes the activation differences between English and non-English inputs as features and outputs fusion weights $\alpha \in [0,1]$ for each dimension. A weight close to 1 indicates that the English activation in that dimension is helpful to the non-English input and should be introduced, while a weight close to 0 indicates the original non-English activation should be retained.
    - **Design Motivation**: Indiscriminately fusing all English activations may introduce noise or even cause negative transfer (e.g., English-specific syntactic patterns interfering with non-English generation). The Decision Maker learns to identify truly beneficial cross-lingual signals.

3. **Transform Matrix**:

    - **Function**: Simulates the cross-lingual connection effect during inference.
    - **Mechanism**: During inference, only non-English inputs are available, with no English reference to compute the fusion. To address this, a Transform Matrix $W$ is learned during training such that $h_{non-en} \cdot W \approx h_{fused}$, mapping monolingual activations to the fused activation space via a linear transformation. During inference, $W$ is applied directly to the non-English activations without requiring English inputs.
    - **Design Motivation**: To resolve the training-inference discrepancy. The Transform Matrix acts as a "compressed representation" of the cross-lingual connection effect during training, enabling the benefits of cross-lingual connection during inference without requiring bilingual inputs.

### Loss & Training

The training objective consists of two components: (1) The standard multilingual SFT loss, which performs regular language modeling training using the fused activations; (2) The alignment loss of the Transform Matrix, which minimizes $\|h_{non-en} \cdot W - h_{fused}\|^2$ to ensure the transformation matrix accurately approximates the fusion effect. The Decision Maker and the Transform Matrix are trained jointly with the model parameters in an end-to-end manner.

## Key Experimental Results

### Main Results

| Method | MGSM (Math) | XCOPA (Common) | XStoryCloze | XNLI | XWinograd | Average |
|------|-----------|------------|-------------|------|-----------|------|
| Vanilla SFT | 38.5 | 62.3 | 71.8 | 55.2 | 64.1 | 58.4 |
| Translation-Augmented SFT | 42.1 | 65.8 | 74.2 | 58.6 | 67.3 | 61.6 |
| Knowledge Distillation SFT | 43.5 | 66.2 | 75.1 | 59.8 | 68.0 | 62.5 |
| **CC-Tuning** | **46.2** | **68.5** | **77.3** | **62.1** | **70.5** | **64.9** |

### Ablation Study

| Configuration | Average Score | Description |
|------|-------|------|
| CC-Tuning (Full) | 64.9 | Full model |
| w/o Decision Maker | 61.8 | Without decision maker, direct equal-weight fusion, drops by 3.1% |
| w/o Transform Matrix | 59.2 | No transformation during inference, training-inference discrepancy, drops by 5.7% |
| Transform Matrix Only | 62.0 | Skip fusion training and only learn the transform matrix, drops by 2.9% |
| CC-Tuning + Translation Augmentation | 66.3 | Data-level and hidden space-level methods are complementary, gaining an additional 1.4% |

### Key Findings

- CC-Tuning outperforms Vanilla SFT across all 6 benchmarks and 22 languages, with an average improvement of approximately 6.5 percentage points.
- The contribution of the Decision Maker is significant—fusing English activations indiscriminately degrades performance for some languages, demonstrating the necessity of selective fusion.
- The Transform Matrix is critical during inference—without it, performance drops below that of data augmentation methods, indicating that training-inference consistency is vital.
- CC-Tuning is complementary to data-level augmentation methods (such as translation-based augmentation), and combining them yields further performance improvements.
- Low-resource languages benefit the most from CC-Tuning, showing the largest gains.

## Highlights & Insights

- **Cross-lingual connection in the hidden space**: Distinguishing itself from traditional data-level methods, CC-Tuning establishes direct cross-lingual connections in the internal representation space of the model. This concept can be transferred to other scenarios with "unbalanced capabilities", such as injecting representation signals of stronger tasks into the computational process of weaker tasks.
- **Asymmetric training-inference design**: Using bilingual inputs during training to obtain the optimal fused signal and approximating it with a Transform Matrix during inference—this "rich information during training, efficient approximation during inference" paradigm is ingenious and can be applied to various teacher-student scenarios.
- **Complementarity with data augmentation**: Proves that hidden-space methods and data-level methods can work synergistically, offering guidance for optimal strategy combinations in practice.

## Limitations & Future Work

- The Transform Matrix uses a linear transformation to approximate the fusion effect, which may not be precise enough for highly non-linear cross-lingual relationships.
- The training process requires English-non-English reference data, meaning high-quality parallel corpora or translations are needed, which increases data preparation costs.
- The interpretability of the Decision Maker is limited, making it difficult to intuitively understand which dimensions or layers of English activations it chooses to select.
- Future work could explore more powerful non-linear transformations to replace the Transform Matrix, as well as utilizing a few English examples during inference to achieve more precise cross-lingual connections.

## Related Work & Insights

- **vs xTune**: xTune introduces cross-lingual noise to inputs during fine-tuning to enhance robustness, but it remains essentially a data-level operation. CC-Tuning operates directly in the hidden space, making information transfer more direct.
- **vs MAD-X**: MAD-X achieves multilingual transfer through adapters, but each language requires an independent adapter. The cross-lingual connection mechanism of CC-Tuning is language-agnostic and does not require training separate modules for each language.
- **vs Translation-augmented fine-tuning**: Conventional methods translate English SFT data into target languages to expand training sets. Experiments with CC-Tuning demonstrate that this data-level approach is complementary to connections in the hidden space.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Establishes explicit cross-lingual connections in the hidden space for the first time; the asymmetric training-inference design utilizing a Decision Maker + Transform Matrix is highly creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive experiments across 6 benchmarks and 22 languages, robust ablation analysis, and comprehensive comparison with multiple baseline methods.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, detailed technical descriptions, and well-organized experiments.
- Value: ⭐⭐⭐⭐⭐ Provides a brand-new technical route for multilingual LLM fine-tuning, highly deserving of its status as an ACL main conference long paper.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SIFT-50M: A Large-Scale Multilingual Dataset for Speech Instruction Fine-Tuning](sift-50m_a_large-scale_multilingual_dataset_for_speech_instruction_fine-tuning.md)
- [\[ACL 2025\] Cross-Lingual Representation Alignment Through Contrastive Image-Caption Tuning](cross-lingual_representation_alignment_through_contrastive_image-caption_tuning.md)
- [\[ACL 2025\] Statement-Tuning Enables Efficient Cross-lingual Generalization in Encoder-only Models](statement-tuning_enables_efficient_cross-lingual_generalization_in_encoder-only_.md)
- [\[ACL 2026\] Exploring Two-Phase Continual Instruction Fine-tuning for Multilingual Adaptation in Large Language Models](../../ACL2026/multilingual_mt/exploring_continual_fine-tuning_for_enhancing_language_ability_in_large_language.md)
- [\[ACL 2025\] Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs](mid_layer_crosslingual_alignment.md)

</div>

<!-- RELATED:END -->
