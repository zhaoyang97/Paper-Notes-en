---
title: >-
  [Paper Note] UniCode: Learning a Unified Codebook for Multimodal Large Language Models
description: >-
  [ECCV 2024][Multimodal VLM][Unified Codebook] UniCode proposes learning a unified codebook to tokenize both visual and textual signals simultaneously. It progressively aligns the visual tokenizer's codebook with the LLM's vocabulary through a language-driven iterative training paradigm. Additionally, it introduces an in-context image decompression pre-training task to enhance image generation quality, enabling MLLMs to achieve multimodal understanding and generation without r…
tags:
  - "ECCV 2024"
  - "Multimodal VLM"
  - "Unified Codebook"
  - "Visual Quantization"
  - "Multimodal Generation"
  - "Vector Quantization"
  - "Image Generation"
date: 2026-05-08
content_hash: 7c9c113fbf0c44bd
---

# UniCode: Learning a Unified Codebook for Multimodal Large Language Models

**Conference**: ECCV 2024  
**arXiv**: [2403.09072](https://arxiv.org/abs/2403.09072)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: Unified Codebook, Visual Quantization, Multimodal Generation, Vector Quantization, Image Generation

## TL;DR

UniCode proposes learning a unified codebook to tokenize both visual and textual signals simultaneously. It progressively aligns the visual tokenizer's codebook with the LLM's vocabulary through a language-driven iterative training paradigm. Additionally, it introduces an in-context image decompression pre-training task to enhance image generation quality, enabling MLLMs to achieve multimodal understanding and generation without requiring extra alignment modules.

## Background & Motivation

**Background**: Current MLLMs primarily map visual signals into the textual space of LLMs through lightweight projectors. However, this paradigm can only generate text and cannot produce non-linguistic content such as images.

**Limitations of Prior Work**:
   - **Paradigm A** (vis enc + text tok, e.g., LLaVA): Aligns vision and text using a projector, but can only output text.
   - **Paradigm B** (vis tok + text tok, e.g., Unified-IO 2): Concatenates the visual codebook with the text vocabulary. However, expanding the vocabulary causes a dramatic explosion in parameter size and faces the "codebook collapse" problem (where the model relies excessively on a small number of codes). Furthermore, bridging the distribution discrepancy across modalities remains difficult.

**Key Challenge**: To equip MLLMs with multimodal generation capabilities, a shared token space capable of representing both vision and text is required. However, the distribution of the visual codebook differs significantly from that of the textual vocabulary, making their unification a key challenge.

**Goal**: Is it possible to learn a unified codebook that directly allows the LLM vocabulary to quantize visual signals?

**Key Insight**: Instead of expanding the codebook, the codebook of the visual tokenizer is progressively aligned and converged with the existing vocabulary of the LLM, sharing the exact same set of codes.

**Core Idea**: Smoothing the visual codebook toward the LLM vocabulary using Exponential Moving Average (EMA) (language-driven iterative training), accompanied by an in-context image decompression task to enhance generation capabilities.

## Method

### Overall Architecture

Pipeline: Image → Visual Encoder → Feature Map $Z_0$ → Vector Quantization (using the unified codebook) → Code Map $M$ → Stacked Quantization Compression → Aggregated Embeddings → LLM → Output (text or visual tokens) → Visual Decoder → Reconstructed Image

Two-stage training: Stage I (Unified Codebook Learning) → Stage II (Multimodal Instruction Tuning)

### Key Designs

1. **Visual Tokenization and Stacked Quantization**:

    - **Function**: Compresses images into discrete token sequences, making them processable by the LLM.
    - **Mechanism**: Using the VQ-VAE framework, the encoder $\mathbb{E}$ extracts the feature map $Z_0 \in \mathbb{R}^{h \times w \times c}$, and then finds the nearest neighbor in the codebook $\mathbb{C}$ for each vector $z$: $Q(z;\mathbb{C}) = \arg\min_{k} \|z - e(k)\|_2^2$
    - To address the issue where high-resolution code maps lead to excessively long LLM sequences, stacked quantization (e.g., HQ-VAE) is adopted to compress the $h \times w$ code map into an $\hat{h} \times \hat{w} \times D$ multi-layer stacked map, where the final embedding for each position is the aggregation of the D-layer quantized vectors: $\hat{z}_{ij} = \mathcal{F}_{d=1}^{D} e(\hat{M}_{i,j,d})$
    - **Design Motivation**: Directly using a $256 \times 256$ resolution code map would require the LLM to process excessively long sequences. Stacked quantization significantly reduces the token count while maintaining information fidelity.

2. **Language-Driven Iterative Training (Core Innovation)**:

    - **Function**: Learns a unified codebook suitable for both visual quantization and text processing.
    - **Mechanism**: Alternately trains the visual tokenizer and the LLM, smoothing the visual tokenizer's codebook toward the LLM vocabulary using EMA:
    $\mathbb{C}' = \lambda \mathbb{C} + (1-\lambda) \mathbb{C}_L$
      where $\mathbb{C}_L$ is the LLM vocabulary embedding, and $\lambda$ is the decay rate. The key point is: only the LLM's codebook is used to update the visual tokenizer's codebook, and no reverse updates are performed on the LLM.
    - **Design Motivation**:
        - The **Frozen LLM codebook scheme** leads to poor reconstruction quality due to the lack of a synchronization mechanism between the encoder/decoder and the frozen codebook.
        - The **Dual alternative training scheme** causes the LLM's language capability to collapse, because the rate of change of the visual codebook is much faster than that of the textual codebook, and frequent replacements disrupt the LLM's internal consistency.
        - The **Language-driven scheme** balances both ends: EMA ensures that the visual codebook slowly converges toward the LLM vocabulary, while avoiding any disturbance to the LLM's own training.
    - **Ablation Validation**: The dual scheme collapses in VQA performance (VQA-v2 drops from 53.1 to 9.3), the frozen scheme yields poor generation quality (FID of 34.45 vs. 6.72), and the language-driven scheme excels in both.

3. **In-Context Image Decompression Pre-training Task**:

    - **Function**: "Decompresses" compressed visual embeddings into a multi-layer code map, serving as a pre-training task to enhance the LLM's capability to understand visual tokens.
    - **Mechanism**: Given the compressed quantized embedding $\hat{Z} \in \mathbb{R}^{\hat{h} \times \hat{w}}$ as input, autoregressively predicts the unfolded code sequence $\{u_1, u_2, ..., u_{\hat{h} \times \hat{w} \times D}\}$:
    $\max_\theta \sum_{l=1}^{\hat{h} \times \hat{w} \times D} \log P_\Theta(u_l | u_{<l}; \hat{Z})$
      Segmenting the image into $T$ segments, in-context learning is performed using a multi-turn conversation format.
    - **Design Motivation**: Stacked quantization makes the alignment between aggregated embeddings and the LLM vocabulary more complex. The decompression task forces the LLM to learn and understand the internal structure of the compressed visual representation, preventing premature convergence.

### Loss & Training

- **Stage I (Unified Codebook Learning)**: Alternately trains the visual tokenizer (image reconstruction task, using LCS-558K) and the LLM (textual instruction data). The visual codebook is updated using the LLM's codebook via EMA at regular step intervals.
- **Stage II (Multimodal Instruction Tuning)**: Freezes the visual encoder and decoder, fine-tuning only the LLM. It uses a combination of Mixed-665K + CC3M (text-to-image) + image decompression data.
- Loss Function: Standard NLL objective $\mathcal{L}(\Theta) = -\sum_{j=1}^{L} \log P_\Theta(y_j | \mathcal{I}, \hat{y}_{1:j-1})$, calculating the loss only on answer tokens.

## Key Experimental Results

### Main Results

| Benchmark | Metric | UniCode | UniCode+ | LLaVA-1.5 | Emu | Description |
|-----------|------|---------|----------|-----------|-----|------|
| VQA-v2 | Acc | 53.1 | **56.2** | 79.1 | 52.0 | Outperforms comparable generative model Emu |
| VizWiz | Acc | 46.2 | **47.1** | 47.8 | 34.2 | Close to LLaVA-1.5 |
| ScienceQA | Acc | 62.9 | **65.4** | 68.4 | - | Competitive |
| POPE | Acc | 71.8 | **77.6** | 86.4 | - | Benchmark with the largest gap |
| ImageNet | FID↓ | **6.72** | - | - | - | SOTA in class-conditional generation |
| LSUN-Cat | FID↓ | **8.07** | - | - | - | Strong unconditional generation |
| LSUN-Church | FID↓ | **6.96** | - | - | - | Outperforms StyleGAN2 (3.86) |
| CC3M | FID↓ | **11.54** | - | - | - | Text-conditioned generation |

UniCode is a model capable of both understanding and generation, achieving state-of-the-art results among similar models with fewer parameters (104M visual tokenizer vs. 1B for Emu) and less data.

### Ablation Study

| Configuration | VQA-v2 | VizWiz | POPE | ImageNet FID↓ | Description |
|------|--------|--------|------|---------------|------|
| vis enc+text tok | 52.3 | 45.4 | 69.7 | - | Only understanding, no generation |
| vis tok+text tok | 49.0 | 44.5 | 65.4 | 9.82 | Separate codebooks, poor performance |
| **unified tok** | **53.1** | **46.2** | **71.8** | **6.72** | Unified codebook, optimal on both ends |
| frozen codebook | 44.2 | 35.1 | 63.9 | 34.45 | Frozen codebook causes generation to collapse |
| dual training | 9.3 | 5.2 | 13.2 | 8.87 | Bidirectional updates cause VQA collapse |
| **iterative (ours)** | **53.1** | **46.2** | **71.8** | **6.72** | Our best paradigm |
| w/o ImgDecomp | - | - | - | 7.08 | Without decompression |
| **w/ ImgDecomp** | - | - | - | **6.72** | With decompression, -0.36 |

### Key Findings

- **Unified codebook outperforms separate codebooks**: Achieves comprehensive success in both VQA and image generation, as the shared token space allows visual tokens to interact with the LLM more naturally.
- **Codebook learning paradigm is crucial**: Dual training leads to catastrophic forgetting (VQA-v2: 53.1 $\to$ 9.3), while the frozen scheme yields extremely poor generation quality (FID 34.45 vs. 6.72).
- **Visual tokenizer quality directly impacts downstream performance**: Upgrading progressively from VQ-GAN $\to$ RQ-VAE $\to$ HQ-VAE improves VQA-v2 from 49.1 $\to$ 49.8 $\to$ 53.1, indicating that the UniCode framework can continuously scale with advancements in visual tokenizers.
- **Image decompression task** effectively reduces FID (ImageNet 7.08 $\to$ 6.72) by increasing training complexity to prevent premature convergence.
- **Resolution alignment is critical**: Performance decreases significantly when training on 256×256 but testing on 320×320, due to inconsistency in the image regions represented by each element of the code map.

## Highlights & Insights

- **Unified codebook is a new paradigm**: Unlike expanding the vocabulary, UniCode demonstrates that vision and text can share the same codebook without additional alignment modules. This represents an elegant solution for multimodal I/O.
- **Insight on the language-driven update direction**: The key insight is that the visual codebook changes much faster than the LLM vocabulary. Thus, synchronization must flow unidirectionally from the LLM to the visual tokenizer, not vice versa.
- **Ingenious use of EMA**: Standardizing cross-modal codebook synchronization via EMA (drawing an analogy from traditional batch normalization statistics) serves as a simple yet powerful technical contribution.
- **Image decompression as a pre-training task**: Converting the decompression and reconstruction of visual tokens into an in-context learning task for the LLM cleverly allows it to learn to "interpret" compressed visual representations.
- **Scalability of the framework**: Compatible with various visual quantization schemes like VQ-GAN, RQ-VAE, and HQ-VAE. Upgrading the tokenizer directly translates to overall performance gains.

## Limitations & Future Work

- A notable gap remains in VQA performance compared to LLaVA-1.5 (POPE: 71.8 vs. 86.4), primarily limited by the quality of the visual tokenizer and the scale of the training data.
- The visual tokenizer is trained on only 558K images, which is far fewer than CLIP's 400M, suggesting massive potential for improvement by scaling up the dataset.
- Image reconstruction quality on LCS-558K (diverse scenes) is significantly lower than on ImageNet, indicating a need to strengthen generalization.
- Training and testing resolutions must be identical, leading to insufficient flexibility.
- There is an implicit limit on the size of the unified codebook—LLM vocabularies are typically around 32K, and whether this is sufficient to fully cover visual semantics remains to be validated.
- Only image generation is supported; it has not yet been extended to other modalities like video or audio.

## Related Work & Insights

- **vs. Emu**: Emu uses a 1B parameter visual encoder + 82M pre-training data for multimodal generation. UniCode surpasses its VQA performance using only 104M parameters and no additional pre-training data.
- **vs. Unified-IO 2**: Unified-IO 2 expands the vocabulary to concatenate visual and text codes, requiring 1B image-text pairs for training. UniCode shares the same vocabulary, requiring significantly fewer resources.
- **vs. LQAE**: LQAE uses a frozen BERT vocabulary for visual quantization but suffers from poor reconstruction quality. UniCode achieves unification while maintaining reconstruction quality through iterative training.
- **vs. SPAE**: SPAE uses a multi-layer pyramid tokenizer to align with a frozen LLM, which requires an exponential number of tokens. UniCode is more efficient using stacked quantization + decompression tasks.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The concept of a unified codebook is novel; both language-driven iterative training and image decompression are original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ The ablation studies are highly systematic (evaluating three paradigms, three codebook learning methods, and various visual tokenizers), but the gap compared to SOTA models like LLaVA-1.5 limits its persuasiveness.
- Writing Quality: ⭐⭐⭐⭐☆ The methodology is described clearly, and the comparison diagrams of the three paradigms and three codebook learning schemes are highly intuitive.
- Value: ⭐⭐⭐⭐☆ Points to a promising research direction (unified multimodal codebook), but current performance still needs substantial improvements to become practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CodeBind: Decoupled Representation Learning for Multimodal Alignment with Unified Compositional Codebook](../../ACL2026/multimodal_vlm/codebind_decoupled_representation_learning_for_multimodal_alignment_with_unified.md)
- [\[ECCV 2024\] BLINK: Multimodal Large Language Models Can See but Not Perceive](blink_multimodal_large_language_models_can_see_but_not_perceive.md)
- [\[ECCV 2024\] Uni3DL: Unified Model for 3D and Language Understanding](uni3dl_a_unified_model_for_3d_vision-language_understanding.md)
- [\[ECCV 2024\] Attention Prompting on Image for Large Vision-Language Models](attention_prompting_on_image_for_large_visionlanguage_models.md)
- [\[ECCV 2024\] Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models](vary_scaling_up_the_vision_vocabulary_for_large_visionlanguag.md)

</div>

<!-- RELATED:END -->
