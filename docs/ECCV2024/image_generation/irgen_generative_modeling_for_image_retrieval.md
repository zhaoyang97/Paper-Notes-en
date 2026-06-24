---
title: >-
  [Paper Note] IRGen: Generative Modeling for Image Retrieval
description: >-
  [ECCV 2024][Image Generation][Image Retrieval] Redefines image retrieval as a generative modeling task, proposing IRGen—a sequence-to-sequence model that converts images into short sequences of discrete semantic tokens via a semantic image tokenizer, and then autoregressively generates the identifier of the query image's nearest neighbor, achieving end-to-end differentiable retrieval and reaching state-of-the-art (SOTA) performance across three standard benchmarks.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Image Retrieval"
  - "Autoregressive Models"
  - "Generative Retrieval"
  - "Semantic Tokenizer"
  - "Sequence-to-Sequence"
date: 2026-05-08
content_hash: 20f86e7d1319a052
---

# IRGen: Generative Modeling for Image Retrieval

**Conference**: ECCV 2024  
**arXiv**: [2303.10126](https://arxiv.org/abs/2303.10126)  
**Code**: [GitHub](https://github.com/yakt00/IRGen)  
**Area**: Image Generation  
**Keywords**: Image Retrieval, Autoregressive Models, Generative Retrieval, Semantic Tokenizer, Sequence-to-Sequence

## TL;DR

Redefines image retrieval as a generative modeling task, proposing IRGen—a sequence-to-sequence model that converts images into short sequences of discrete semantic tokens via a semantic image tokenizer, and then autoregressively generates the identifier of the query image's nearest neighbor, achieving end-to-end differentiable retrieval and reaching state-of-the-art (SOTA) performance across three standard benchmarks.

## Background & Motivation

Traditional image retrieval systems consist of two independent phases:

**Feature Representation Learning**: Encoding images into feature vectors.

**Approximate Nearest Neighbor (ANN) Search**: Efficiently searching for the most similar database images in the feature space.

These two phases are optimized independently, requiring meticulous hyperparameter tuning to coordinate them during deployment. Although joint learning attempts exist (such as deep quantization/hashing), they still rely on post-processing non-exhaustive searches and fail to achieve true end-to-end optimization.

**Key Insight**: Generative modeling has achieved great success in fields such as translation, dialogue, and image generation, with the core idea of unifying tasks into sequence generation problems. Image retrieval can similarly be redefined as: **given a query image, generate the identifier sequence of its nearest neighbor database image**.

This introduces two key technical challenges:
1. How to convert an image into a **sufficiently short** sequence of semantic tokens (to ensure search efficiency).
2. How to ensure these tokens contain **sufficient semantic information** (to guarantee retrieval accuracy).

Existing image tokenizers (such as VQ-VAE, RQ-VAE) are designed for image generation and are unsuitable for retrieval tasks due to extremely long sequences (e.g., 256 tokens) and encoding low-level details rather than semantic information.

## Method

### Overall Architecture

IRGen consists of two core components:

1. **Semantic Image Tokenizer**: Converts an image into a short sequence of discrete semantic tokens to serve as its identifier.
2. **Encoder-Decoder**: A standard Transformer architecture that takes a query image as input and autoregressively generates the identifier of the nearest neighbor image.

### Key Designs

#### 1. Semantic Image Tokenizer

**Why are existing tokenizers inadequate?**

| Property | VQ-VAE / RQ-VAE | IRGen Tokenizer |
|------|------------------|-------------|
| Objective | Pixel-level reconstruction | Semantic-level classification |
| Input Features | Spatial patch embeddings | Global class token |
| Sequence Length | 256 (8×8×4) | **Only M** (e.g., 4-8) |
| Encoded Content | Low-level details, textures | High-level semantic information |

**Core Design Idea:**

**(a) Using Global Features instead of Spatial Features**

Instead of using the spatial tokens of ViT (which yield long sequences), the global **class token** $\mathbf{f}_{cls}$ is used as the image representation. This compresses the sequence length from spatial tokens to a single token, which is then expanded into $M$ discrete codes using residual quantization.

**(b) Residual Quantization (RQ)**

Using $M$ codebooks, each containing $L$ elements. Recursively map $\mathbf{f}_{cls}$ into $M$ ordered discrete codes $\{l_1, l_2, \ldots, l_M\}$:

$$l_m = \arg\min_{l \in [L]} \|\mathbf{r}_{m-1} - \mathbf{c}_{ml}\|_2^2$$
$$\mathbf{r}_m = \mathbf{r}_{m-1} - \mathbf{c}_{ml_m}$$

The initial residual is $\mathbf{r}_0 = \mathbf{f}_{cls}$. This recursive process is naturally compatible with autoregressive generation—each token depends on the preceding ones.

**(c) Semantic Supervision instead of Reconstruction Supervision**

Key Innovation: Train the tokenizer using a **classification loss** instead of a pixel reconstruction loss! Classification losses are applied to both the original embedding and the reconstructed embeddings at various levels:

$$\mathcal{L} = \mathcal{L}_{cls}(\mathbf{f}_{cls}) + \lambda_1\sum_{m=1}^{M}\mathcal{L}_{cls}(\hat{\mathbf{f}}_{cls}^{\le m}) + \lambda_2\sum_{m=1}^{M}\|\mathbf{r}_m\|_2^2$$

Where $\hat{\mathbf{f}}_{cls}^{\le m} = \sum_{i=1}^{m}\mathbf{c}_{il_i}$ represents the embedding reconstructed using the first $m$ codes.
- The first term: ensures the semantic quality of the original feature.
- The second term: ensures that prefix codes at all levels contain semantic information (hierarchical semantics).
- The third term: minimizes the quantization residuals.

Training employs an alternating optimization scheme to update the codebooks and network parameters, using a straight-through estimator (STE) for quantization gradients.

#### 2. Encoder-Decoder Autoregressive Retrieval

Once discrete identifiers are established, a sequence-to-sequence model is trained:

**Training Process:**
- Input: Image pairs $(x_1, x_2)$, where $x_2$ is the nearest neighbor of $x_1$.
- Encoder $\mathbb{E}$ (ViT-Base) encodes $x_1$ into a query embedding.
- Decoder $\mathbb{D}$ (standard Transformer with causal self-attention + cross-attention + MLP) autoregressively predicts the identifier sequence of $x_2$.

Training Objective—maximize the conditional likelihood of the identifier:

$$p(l_1, \ldots, l_M | x_1, \theta) = \prod_{m=1}^{M} p(l_i | x_1, l_1, \ldots, l_{m-1}, \theta)$$

The model is trained using a softmax cross-entropy loss over a vocabulary of $M$ discrete tokens.

**Note**: The decoder operates entirely on discrete variables and does not involve any visual content—allowing the model to easily scale using mature Transformer infrastructures.

#### 3. Beam Search Retrieval

**Top-1 Retrieval**: Greedy decoding, generating the token with the highest probability step by step.

**Top-K Retrieval**: Using beam search (beam size = K).

Specific process:
1. Initialize the beam with the start token.
2. Expand candidate sequences, generating all possible next tokens for each candidate.
3. Rank them based on sequence probability (the product of predicted token probabilities) and retain the top-K.
4. Repeat until the final token is decoded.

**Prefix Tree Constraints**: Not all generated identifiers correspond to valid images in the database. A prefix tree containing all valid codes is constructed, constraining the search process to consider only valid next tokens, significantly boosting efficiency.

**Core Difference between Beam Search and ANN Search**:
- ANN: Calculates the query-database distance using a predefined distance metric (e.g., Euclidean or cosine).
- Beam Search: Computes likelihood scores via a **differentiable neural network** conditioned on the query—allowing the entire retrieval pipeline to be optimized end-to-end.

### Loss & Training

**Tokenizer Training:**
- Classification loss $\mathcal{L}_{cls}$ (softmax cross-entropy) + quantization residual minimization.
- Alternating optimization of the codebooks and the encoder network.

**Retrieval Model Training:**
- Autoregressive cross-entropy loss: predicting the next token of the nearest neighbor's identifier.
- Training data: mining neighbor image pairs from the database.

## Key Experimental Results

### Main Results

Precision comparison on three standard benchmarks (all methods use identical data preprocessing and comparable model sizes):

| Method | Search Method | In-shop P@1 | In-shop P@10 | CUB200 P@1 | CUB200 P@2 | Cars196 P@1 | Cars196 P@2 |
|------|----------|:---:|:---:|:---:|:---:|:---:|:---:|
| FT-CLIP | Linear Scan | 91.4 | 66.8 | 79.2 | 77.6 | 88.4 | 87.7 |
| CGD | Linear Scan | 83.2 | 47.8 | 76.7 | 75.5 | 87.1 | 86.1 |
| IRT_R | Linear Scan | 92.7 | 59.6 | 79.3 | 77.7 | 75.6 | 73.1 |
| FT-CLIP | SPANN | 90.2 | 62.9 | 78.5 | 77.6 | 88.6 | 88.1 |
| **IRGen** | **Beam Search**| **92.4** | **87.0** | **82.7** | **82.7** | **90.1** | **89.9** |

Key metric improvements (vs. the best baseline, including linear scan):
- **In-shop P@10**: +20.2% (87.0 vs. 66.8)
- **CUB200 P@2**: +6.0% (82.7 vs. 77.7)
- **Cars196 P@2**: +2.4% (89.9 vs. 87.7)

Particularly noteworthy is the massive gain on **In-shop P@10**—demonstrating that IRGen's precision does not experience rapid decay at larger K values (from P@1=92.4 to P@10=87.0), unlike traditional methods which drop precipitously (FT-CLIP drops from 91.4 to 66.8).

### Ablation Study

Precision comparison of different identifier types (In-shop dataset):

| Identifier Type | P@1 | P@10 | P@20 |
|-----------|:---:|:---:|:---:|
| Random Codes | 76.9 | 60.3 | 56.8 |
| Hierarchical K-means | 86.1 | 67.2 | 62.1 |
| RQ-VAE (Reconstruction Supervision) | 85.2 | 68.0 | 63.5 |
| **Semantic Tokenizer (Ours)** | **92.4** | **87.0** | **86.6** |

The semantic tokenizer substantially outperforms other identifier designs, validating the critical importance of semantic supervision.

### Key Findings

1. **Generative retrieval can outperform linear scan**: Highly counter-intuitive, as non-exhaustive search methods are generally considered incapable of matching exhaustive search. IRGen breaks this barrier via end-to-end optimization.
2. **The 'Plateau Effect' of P@K**: IRGen's P@10/P@20/P@30 barely decrease (87.0/86.6/86.5), indicating that beam search produces highly robust, high-quality nearest neighbors.
3. **Semantic vs. Reconstruction Supervision**: The tokenizer trained under classification loss vastly outperforms RQ-VAE's pixel reconstruction loss, verifying the significance of semantic priors for retrieval.
4. **Scaling to Million-scale Datasets**: Maintains superiority on ImageNet and Places365, demonstrating the extensibility of the method.
5. **Potential Elimination of Re-ranking**: The massive precision gain suggests that the traditionally indispensable re-ranking phase in practical workflows might be bypassed entirely.

## Highlights & Insights

1. **Pioneering Unified Paradigm**: Redefines image retrieval as a sequence generation task for the first time, achieving end-to-end differentiability and eliminating explicit boundaries between feature extraction, compression, and indexing.
2. **Deep Analogy with NLP**: Just as GPT models in NLP generate the next token, IRGen generates identifier tokens of the target image—achieving methodological unification across modalities.
3. **Elegant Use of Global Class Token**: Compresses the sequence length from spatial tokens (e.g., 256) to just $M$ (typically 4-8). This not only accelerates inference (quadratic reduction in time complexity) but also inherently encodes high-level semantics.
4. **Graceful Constraint via Prefix Tree**: Cleverly formulates validity verification as path constraints on a tree structure, avoiding the generation of invalid identifiers during beam search.
5. **Conceptual Simplicity**: All components are standard Transformers, directly enabling scaling using established mature technologies.

## Limitations & Future Work

1. **Autoregressive Inference Speed**: Although sequences are remarkably short, the inherent sequential nature of autoregressive generation remains an efficiency bottleneck.
2. **Training Data Construction**: Requires pre-mining of nearest neighbor image pairs as training data, relying on existing retrieval pipelines.
3. **Codebook Size vs. Expressive Power Trade-off**: The choice of $L$ and $M$ directly impacts the uniqueness and discriminative capability of identifiers.
4. **Dynamic Database Updates**: Adding or removing images in the database requires updating the prefix tree and potentially retraining components.
5. Future work can explore extending this framework to cross-modal retrieval (Text-to-Image) and unified multi-modal retrieval.

## Related Work & Insights

- **DSI / NCI**: Generative document retrieval methods that utilize hierarchical K-means for document identifiers; IRGen demonstrates that semantic-supervised identifiers are far more effective.
- **RQ-VAE**: Residual quantization targeted for image generation; IRGen shifts it from a reconstruction objective to classification, adapting it for retrieval.
- **JPQ / DPQ**: Jointly learn embeddings and compression, but still rely on linear scan; IRGen achieves true non-exhaustive end-to-end retrieval via beam search.
- **GPT**: Autoregressive language architecture; IRGen's decoder directly inherits this universal structure.
- **Insight**: The potential of generative modeling as a unified framework is far from fully exploited; image retrieval is only the beginning.

## Rating

- **Novelty**: ★★★★★ — Redefines image retrieval as a generative task, representing a paradigm-level innovation.
- **Experimental Thoroughness**: ★★★★★ — Three standard benchmarks + two million-scale datasets + extensive ablations + comparisons across various search strategies.
- **Writing Quality**: ★★★★★ — Clear problem motivation, systematic description of method, and rigorous experimental structure.
- **Value**: ★★★★☆ — Strong conceptual breakthrough, but autoregressive inference speed and dynamic database updates pose practical engineering challenges.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Breaking the Modality Barrier: Generative Modeling for Accurate Molecule Retrieval from Mass Spectra](../../AAAI2026/image_generation/breaking_the_modality_barrier_generative_modeling_for_accurate_molecule_retrieva.md)
- [\[ECCV 2024\] MacDiff: Unified Skeleton Modeling with Masked Conditional Diffusion](macdiff_unified_skeleton_modeling_with_masked_conditional_diffusion.md)
- [\[NeurIPS 2025\] GenIR: Generative Visual Feedback for Mental Image Retrieval](../../NeurIPS2025/image_generation/genir_generative_visual_feedback_for_mental_image_retrieval.md)
- [\[ECCV 2024\] NL2Contact: Natural Language Guided 3D Hand-Object Contact Modeling with Diffusion Model](nl2contact_natural_language_guided_3d_hand-object_contact_modeling_with_diffusio.md)
- [\[CVPR 2025\] VLog: Video-Language Models by Generative Retrieval of Narration Vocabulary](../../CVPR2025/image_generation/vlog_video-language_models_by_generative_retrieval_of_narration_vocabulary.md)

</div>

<!-- RELATED:END -->
