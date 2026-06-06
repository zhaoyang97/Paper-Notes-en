---
title: >-
  [Paper Note] Improved Masked Image Generation with Knowledge-Augmented Token Representations
description: >-
  [AAAI 2026][Image Generation][Masked image generation] This paper proposes KA-MIG, a framework that mines three types of token-level semantic prior knowledge graphs from training data (co-occurrence graph…
tags:
  - "AAAI 2026"
  - "Image Generation"
  - "Masked image generation"
  - "knowledge graph"
  - "discrete tokens"
  - "prior knowledge augmentation"
  - "graph convolutional network"
date: 2026-05-08
content_hash: 65bdb7b13e7ff29d
---

# Improved Masked Image Generation with Knowledge-Augmented Token Representations

**Conference**: AAAI 2026
**arXiv**: [2511.12032](https://arxiv.org/abs/2511.12032)  
**Code**: [https://github.com/GuotaoLiang/KA-MIG](https://github.com/GuotaoLiang/KA-MIG)  
**Area**: Image Generation
**Keywords**: Masked image generation, knowledge graph, discrete tokens, prior knowledge augmentation, graph convolutional network

## TL;DR

This paper proposes KA-MIG, a framework that mines three types of token-level semantic prior knowledge graphs from training data (co-occurrence graph, semantic similarity graph, and position-token incompatibility graph), learns augmented token representations via a graph-aware encoder, and injects them into existing MIG models through a lightweight addition-subtraction fusion mechanism, consistently improving generation quality across multiple backbone networks.

## Background & Motivation

Masked image generation (MIG), exemplified by MaskGIT, achieves a favorable balance between sampling speed and quality through parallel decoding. The pipeline encodes images into discrete token sequences via VQ-VAE → trains a transformer to predict masked tokens → iteratively samples to generate complete token sequences.

However, MIG still lags behind diffusion models. Existing improvements have focused primarily on **decoding/sampling strategies** (e.g., Token-Critic, DPC, Self-Guidance, Halton sampling), with almost no attention paid to enhancing the model's **internal representational capacity**.

The root problem identified by the authors: **existing MIG methods rely entirely on the transformer itself to learn semantic dependencies among tokens**, which is challenging because:

**Individual tokens lack explicit semantic meaning**: VQ-VAE codebook entries are merely vectors in latent space that are not directly interpretable.

**Token sequences are typically long** (e.g., 256 tokens/image), making complex relationships in long sequences difficult to capture effectively.

Core motivation: **Given that tokens themselves lack semantics, can implicit structural patterns among tokens be mined from large-scale training data and injected into the model as prior knowledge?**

## Method

### Overall Architecture

KA-MIG consists of three stages:
1. **Graph construction**: Building three prior knowledge graphs from training data
2. **Graph-aware encoder**: Learning augmented token and position representations using GCN
3. **Lightweight fusion mechanism**: Injecting prior knowledge into the MIG transformer via addition and subtraction operations

### Key Designs

#### 1. **Construction of Three Prior Knowledge Graphs**

**(a) Co-occurrence Graph $\mathcal{G}_{co}$ (Positive Prior)**

Captures frequently co-occurring token pairs within a local neighborhood, reflecting latent spatial-semantic correlations.

- Construction: Statistics over token sequences of all training images, recording the frequency of co-occurrence between token pairs within first-order neighborhoods (horizontal, vertical, diagonal directions)
- A weighted undirected graph is built; low-frequency edges are pruned to reduce noise
- Intuition: If token A and token B frequently appear in adjacent positions, they likely encode semantically related visual patterns

**(b) Semantic Similarity Graph $\mathcal{G}_s$ (Positive Prior)**

Identifies tokens that are semantically similar (akin to "synonyms") in the context of image synthesis.

- **Core assumption**: If two tokens exhibit similar **positional distributions** across a large number of images, they likely express similar semantics
- A positional distribution vector of length $N$ is constructed for each token (each entry represents the frequency of the token appearing at a specific position)
- Jensen-Shannon divergence is used to measure distributional similarity
- Each token retains its top-2 most similar tokens, forming a directed graph

**The validation experiment is highly compelling**: replacing token (1013) with its most similar token (463) yields a reconstructed image visually indistinguishable from the original (PSNR=35.78); replacing it with the least similar token (149) severely degrades quality (PSNR=18.97).

**(c) Position-Token Incompatibility Graph $\mathcal{G}_p^c$ (Negative Prior)**

Identifies tokens that should not appear at specific spatial positions under a given category.

- For each category $c$, all training images are scanned to record tokens that **never** appear at a given position
- Example: For the "airplane" category, tokens encoding ground/grass textures almost never appear in the upper half of the image
- Helps the model avoid semantically unreasonable spatial-token combinations

#### 2. **Graph-aware Encoder**

**Positive prior processing**: Two independent 3-layer GCNs extract global token representations:
$$C_{co} = f_{\theta_{co}}(\mathcal{G}_{co}, C), \quad C_s = f_{\theta_s}(\mathcal{G}_s, C)$$
where $C$ denotes the VQ-VAE codebook embeddings.

**Negative prior processing**: For each position $i$ under category $c$, the mean embedding of incompatible tokens is aggregated:
$$p_i^c = \frac{1}{|\mathcal{I}_{i,j}|}\sum_{t \in \mathcal{I}_{i,j}} C_t W$$
yielding positional embeddings $P^c \in \mathbb{R}^{N \times d}$ that encode spatial constraints.

#### 3. **Lightweight Fusion Mechanism**

**Additive fusion (positive prior)**: Augments unmasked token representations before each transformer layer:
$$Z_{\overline{M}}^l = Z_{\overline{M}}^l + f_{pos}^l(C_{co}[Z_{\overline{M}}]) + f_{pos}^l(C_s[Z_{\overline{M}}])$$

**Subtractive fusion (negative prior)**: Suppresses incompatible token features at masked positions in each layer:
$$Z_M^l = Z_M^l - \alpha f_{neg}^l(P^c)$$

Both $f_{pos}$ and $f_{neg}$ are implemented with zero convolution, ensuring no interference with existing knowledge at the start of training.

### Loss & Training

- Standard MIG training objective (negative log-likelihood of masked tokens)
- Backbone is frozen; **only classification layers and newly added parameters are fine-tuned**
- Graph features can be precomputed and stored; inference involves only lightweight addition and subtraction operations
- Validated on three backbones: MaskGIT, AutoNAT, and TiTok

## Key Experimental Results

### Main Results

**ImageNet-256 Class-conditional Generation**

| Model | Type | Params | FID↓ | IS↑ | Prec↑ | Rec↑ |
|------|------|--------|------|-----|-------|------|
| MaskGIT | MIG | 227M | 6.18 | 182.1 | 0.80 | 0.52 |
| **MaskGIT-KA** | **MIG** | **245M** | **5.69** | **170.2** | **0.81** | **0.50** |
| AutoNAT | MIG | 194M | 2.68 | 278.8 | - | - |
| **AutoNAT-KA** | **MIG** | **211M** | **2.45** | 274.1 | 0.82 | 0.56 |
| TiTok-b64 | MIG | 177M | 2.48 | 214.7 | - | - |
| **TiTok-b64-KA** | **MIG** | **194M** | **2.40** | **217.0** | 0.78 | 0.60 |
| TiTok-s128 | MIG | 177M | 1.97 | 281.8 | - | - |
| **TiTok-s128-KA** | **MIG** | **194M** | **1.90** | 271.9 | 0.78 | 0.61 |
| VAR-d20 | AR | 600M | 2.57 | 302.6 | 0.83 | 0.56 |
| LDM-4 | Diff. | 400M | 3.60 | 247.7 | - | - |

**MS-COCO Text-to-Image Generation**

| Method | FID↓ | CLIP-Score↑ |
|------|------|------------|
| MaskGen | 22.27 | 25.58 |
| **MaskGen + KA (Ours)** | **21.01** | **26.10** |

### Ablation Study

| Configuration | FID↓ | IS↑ | Note |
|------|------|-----|------|
| AutoNAT (baseline) | 2.68 | 278.8 | |
| + $\mathcal{G}_s$ only | 2.49 | 279.6 | Semantic similarity graph contributes most to FID |
| + $\mathcal{G}_p$ only | 2.51 | **285.6** | Position incompatibility graph yields the largest IS gain |
| + $\mathcal{G}_{co}$ only | 2.51 | 282.1 | Co-occurrence graph also effective |
| + $\mathcal{G}_s$ + $\mathcal{G}_p$ | 2.46 | 279.9 | Pairwise combinations yield further gains |
| + $\mathcal{G}_{co}$ + $\mathcal{G}_p$ | 2.46 | 280.7 | |
| + $\mathcal{G}_{co}$ + $\mathcal{G}_s$ | 2.48 | 277.4 | |
| + All three (KA-MIG) | **2.45** | 274.1 | Best FID |

**Efficiency Analysis**:

| Graph Type | Online Parameters | Precomputed Parameters | Online TFLOPs |
|--------|-------------|-----------|------------|
| $\mathcal{G}_{co}$ | +16M | +0.79M | ~0 |
| $\mathcal{G}_s$ | +16M | +0.79M | ~0 |
| $\mathcal{G}_p$ | +15M | +196M | +0.06 |

Optimal strategy: precompute $\mathcal{G}_{co}$ and $\mathcal{G}_s$ (lightweight), compute $\mathcal{G}_p$ online (avoiding per-class storage overhead).

### Key Findings

1. **All three graphs are individually effective and mutually complementary**: each yields improvements in isolation, with further gains when combined.
2. **$\mathcal{G}_s$ contributes most to FID**: learning interchangeable token patterns enhances both robustness and diversity.
3. **Longer sequences benefit more**: MaskGIT/AutoNAT (256 tokens) gain more than TiTok (64/128 tokens), as token dependencies are more complex in longer sequences.
4. **Only ~20M additional parameters**: the lightweight design incurs minimal inference overhead.
5. AutoNAT-KA (2.45 FID) **outperforms larger models such as LlamaGen-XL (2.62) and VAR-d20 (2.57)**.

## Highlights & Insights

- **Precise problem formulation**: The paper observes that MIG improvements have almost exclusively targeted sampling strategies and is the first to systematically address internal representational capacity.
- **Data-driven prior knowledge mining** is highly practical: no external annotations or hand-crafted rules are required; knowledge is extracted purely from statistical patterns in training data.
- The assumption that **"similar positional distributions imply semantic similarity"** is simple yet effective, supported by compelling validation experiments (token substitution reconstruction).
- **Additive fusion for positive priors and subtractive fusion for negative priors** offers clear design intuition and straightforward implementation.
- **Fully decoupled from the backbone**: graph features can be precomputed, and only a small number of parameters need fine-tuning with the backbone frozen, making the approach highly practical.

## Limitations & Future Work

- All three graphs are **static** (derived from training data statistics) and are not dynamically updated during training.
- The class-conditional nature of $\mathcal{G}_p$ leads to substantial storage overhead (196M parameters across 1,000 classes); more compact representations could be explored.
- Improvements on short-sequence models such as TiTok are relatively limited; applicability to next-generation compact VQ methods remains to be seen.
- The combination of all three graphs does not always yield the best IS score (274 vs. baseline 278), suggesting possible information redundancy.
- More expressive graph network architectures (e.g., GAT, GraphSAGE) are not explored; the current 3-layer GCN may be insufficient.
- Systematic evaluation at higher resolutions (512×512) is absent.

## Related Work & Insights

- Complementary to MaskGIT-SAG (self-guided sampling) and Halton sampling: those improve sampling while KA-MIG improves representations, and the two can be used jointly.
- The co-occurrence modeling paradigm from graph neural networks in recommender systems is cleverly transferred to the visual token domain.
- The "negative prior" (position-token incompatibility graph) conceptually parallels hard negative mining in contrastive learning.
- The approach may inspire similar token prior knowledge injection in autoregressive image generation models (e.g., LlamaGen, VAR).
- The use of zero convolution draws on the ControlNet design philosophy, ensuring that newly added modules do not interfere with pretrained model knowledge at initialization.

## Rating

- Novelty: ⭐⭐⭐⭐ — The construction of prior knowledge graphs is innovative, though the broader idea of mining statistical patterns from data is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three backbone networks, detailed ablations, efficiency analysis, and visual validation; very comprehensive.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with abundant illustrations.
- Value: ⭐⭐⭐⭐ — Opens a new direction for MIG improvement; the lightweight plug-and-play design is highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] TruthfulRAG: Resolving Factual-level Conflicts in Retrieval-Augmented Generation with Knowledge Graphs](truthfulrag_resolving_factual-level_conflicts_in_retrieval-augmented_generation_.md)
- [\[NeurIPS 2025\] Can Knowledge-Graph-based Retrieval Augmented Generation Really Retrieve What You Need?](../../NeurIPS2025/image_generation/can_knowledge-graph-based_retrieval_augmented_generation_really_retrieve_what_yo.md)
- [\[AAAI 2026\] Laytrol: Preserving Pretrained Knowledge in Layout Control for Multimodal Diffusion Transformers](laytrol_preserving_pretrained_knowledge_in_layout_control_fo.md)
- [\[ICML 2026\] WISE: A World Knowledge-Informed Semantic Evaluation for Text-to-Image Generation](../../ICML2026/image_generation/wise_a_world_knowledge-informed_semantic_evaluation_for_text-to-image_generation.md)
- [\[CVPR 2026\] BiGain: Unified Token Compression for Joint Generation and Classification](../../CVPR2026/image_generation/bigain_unified_token_compression_for_joint_generation_and_classification.md)

</div>

<!-- RELATED:END -->
