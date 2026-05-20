---
title: >-
  [Paper Note] UniTok: A Unified Tokenizer for Visual Generation and Understanding
description: >-
  [NeurIPS 2025][Multimodal VLM][unified tokenizer] This paper proposes UniTok, a unified tokenizer for visual generation and understanding that overcomes the representation capacity bottleneck of discrete tokens via Multi…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "unified tokenizer"
  - "VQVAE"
  - "CLIP"
  - "multi-codebook quantization"
  - "visual generation and understanding"
date: 2026-05-08
content_hash: 8ad381c958e818e2
---

# UniTok: A Unified Tokenizer for Visual Generation and Understanding

**Conference**: NeurIPS 2025
**arXiv**: [2502.20321](https://arxiv.org/abs/2502.20321)  
**Code**: [GitHub](https://github.com/FoundationVision/UniTok)  
**Area**: Multimodal VLM
**Keywords**: unified tokenizer, VQVAE, CLIP, multi-codebook quantization, visual generation and understanding

## TL;DR

This paper proposes UniTok, a unified tokenizer for visual generation and understanding that overcomes the representation capacity bottleneck of discrete tokens via Multi-Codebook Quantization (MCQ). UniTok achieves simultaneous state-of-the-art records of 0.38 rFID and 78.6% zero-shot accuracy on ImageNet, and can be seamlessly integrated into MLLMs to enable both generation and understanding.

## Background & Motivation

Unified multimodal large language models (e.g., GPT-4o) require a single tokenizer suitable for both visual generation and understanding. Existing approaches face a fundamental tension: CLIP tokenizers excel at understanding tasks but produce continuous high-dimensional features incompatible with autoregressive generation, while VQVAE tokenizers support discrete generation but lack semantic modeling capacity, leading to poor understanding performance.

An intuitive solution is to incorporate CLIP supervision into VQVAE training, but in practice this causes severe convergence issues and understanding performance far below the CLIP baseline. **Prior work attributed this to a conflict between semantic loss and pixel-level reconstruction loss** — but is this conclusion correct?

Through systematic analysis, the authors arrive at a **counterintuitive finding**: reconstruction supervision and semantic supervision are not inherently in conflict. The true cause of performance degradation is **insufficient representation capacity of the discrete token space**. Specifically, two key operations create an information bottleneck:

**Token down-projection (Factorization)**: Compressing 768-dimensional features to 16 dimensions for codebook index lookup severely degrades token expressiveness.

**Discretization**: Mapping continuous features to small codebooks (typically 4K–16K entries) causes substantial information loss.

This stands in stark contrast to language tokenizers, which commonly employ vocabularies exceeding 200K entries.

## Method

### Overall Architecture

UniTok is trained jointly with a VQVAE reconstruction loss and a CLIP contrastive loss. Image features extracted by the encoder are quantized via a multi-codebook quantization mechanism — the feature vector is partitioned into chunks, each quantized independently within a dedicated sub-codebook — before being passed to the decoder for image reconstruction. The quantized features are also pooled for image–text contrastive learning.

### Key Designs

1. **Multi-Codebook Quantization (MCQ)**: The core innovation. The latent vector $f \in \mathbb{R}^d$ is evenly partitioned into $n$ chunks $\{f_1, f_2, \ldots, f_n\}$, each quantized independently in its own sub-codebook:

    $\hat{f} = \text{Concat}(\mathcal{Q}(Z_1, f_1), \mathcal{Q}(Z_2, f_2), \ldots, \mathcal{Q}(Z_n, f_n))$

   The key advantage is exponential growth in vocabulary size: increasing the number of sub-codebooks from 1 to 4 (each with 16K entries) expands the theoretical vocabulary from $2^{14}$ to $2^{56}$. Simultaneously, the latent dimensionality grows linearly with the number of sub-codebooks (e.g., from 16 to 64 dimensions), further enhancing representation capacity. Since each sub-codebook remains small, MCQ avoids the optimization pitfalls of large codebooks (low utilization, dead codes).

   A key distinction from Residual Quantization (RQ): RQ employs a coarse-to-fine sequential quantization scheme, whereas MCQ adopts a divide-and-conquer strategy. In high-dimensional latent spaces, MCQ holds a substantial advantage — at 64-dimensional latents, MCQ's quantization error is 15–45× lower than RQ's.

2. **Attention Projection**: Replaces conventional linear/convolutional projection layers for token down-projection. The standard multi-head attention operation of concatenating head outputs is replaced with average pooling for channel compression. Despite its simplicity, this design effectively enhances token expressiveness after down-projection and stabilizes training.

3. **Unified MLLM Integration**: Building on the Liquid framework, images are encoded as $H \times W \times K$ discrete codes (where $K$ is the number of sub-codebooks). An MLP projector maps UniTok code embeddings into the MLLM token space. During encoding, every $K$ consecutive codes are merged into a single visual token; during generation, each token autoregressively predicts the next group of $K$ codes via a deep transformer head.

### Loss & Training

The unified loss is: $\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda_{\text{contra}} \mathcal{L}_{\text{contra}}$

where the reconstruction loss $\mathcal{L}_{\text{recon}} = \mathcal{L}_R + \lambda_{VQ}\mathcal{L}_{VQ} + \lambda_P\mathcal{L}_P + \lambda_G\mathcal{L}_G$ comprises pixel reconstruction, VQ commitment, perceptual (LPIPS), and adversarial losses. The contrastive weight is set to $\lambda_{\text{contra}}=1$.

- Architecture: ViTamin-L/16 hybrid backbone; 8 sub-codebooks × 4096 entries × 8 dimensions.
- Discriminator initialized from DINOv2-S.
- Trained for 1 epoch on DataComp-1B (1.28 billion image–text pairs) at 256×256 resolution.
- MLLM uses Llama-2-7B, pretrained on 70M samples and fine-tuned on 3M samples.

## Key Experimental Results

### Main Results

**ImageNet Reconstruction FID and Zero-Shot Classification Accuracy**

| Method | Type | #Tokens | rFID↓ | Zero-Shot Acc. |
|--------|------|---------|-------|----------------|
| VQGAN | VQVAE | 256 | 4.98 | — |
| VAR | VQVAE | 680 | 0.90 | — |
| CLIP | CLIP | 256 | — | 76.2% |
| SigLIP | CLIP | 256 | — | 80.5% |
| VILA-U | Unified | 256 | 1.80 | 73.3% |
| **UniTok** | **Unified** | 256 | **0.38** | **78.6%** |

**Unified MLLM VQA Benchmark Comparison**

| Method | LLM | Token Type | VQAv2 | GQA | TextVQA | MME |
|--------|-----|------------|-------|-----|---------|-----|
| Chameleon | 34B | Discrete | 69.6 | — | — | — |
| Liquid | Gemma-7B | Discrete | 71.3 | 58.4 | 42.4 | 1119 |
| VILA-U | Llama-2-7B | Discrete | 75.3 | 58.3 | 48.3 | 1336 |
| **UniTok** | Llama-2-7B | Discrete | **76.8** | **61.1** | **51.6** | **1448** |

### Ablation Study

| Supervision | rFID↓ | gFID↓ | VQAv2 | TextVQA | MME |
|-------------|-------|-------|-------|---------|-----|
| Contrastive only | — | — | 68.95 | 49.89 | 1373 |
| Reconstruction only | 0.82 | 3.59 | 56.33 | 43.65 | 902 |
| Reconstruction + Contrastive | 0.72 | 3.26 | 69.14 | 49.22 | 1333 |

**MCQ vs. RQ (64-dim latent)**

| Quantization | Code Shape | rFID↓ | Zero-Shot Acc. |
|-------------|------------|-------|----------------|
| RQ | 16×16×8 | 3.46 | 58.8% |
| **MCQ** | 16×16×8 | **0.55** | **63.7%** |

**Number of Sub-Codebooks Ablation**

| Codebook Config / Vocabulary | rFID↓ | Zero-Shot Acc. |
|-----------------------------|-------|----------------|
| 1×16384 / $2^{14}$ | 1.50 | 41.0% |
| 2×8192 / $2^{26}$ | 0.98 | 43.9% |
| 4×4096 / $2^{48}$ | 0.54 | 44.7% |
| 8×2048 / $2^{88}$ | 0.33 | 46.1% |

### Key Findings

- Reconstruction supervision and semantic supervision are **not inherently in conflict** — joint training achieves understanding performance comparable to pure CLIP training (69.14 vs. 68.95 VQAv2), with rFID even superior to reconstruction-only training (0.72 vs. 0.82).
- The performance bottleneck lies in **discretization**, not loss conflict — transitioning from CLIP to VQ-CLIP, down-projection and discretization each independently cause substantial VQA performance degradation.
- MCQ achieves 15–45× lower quantization error than RQ in 64-dimensional high-dimensional latent spaces.
- Increasing the number of sub-codebooks consistently benefits both reconstruction and understanding without inducing codebook utilization issues.
- In CFG-free generation, UniTok reduces gFID from 14.6 to 2.5, indicating that semantic supervision produces a more structured code distribution.

## Highlights & Insights

- **Core Insight**: The bottleneck of unified tokenizers is not a "generation vs. understanding loss conflict" but rather insufficient representation capacity of the discrete token space. This finding corrects a widespread misconception in the field.
- The divide-and-conquer strategy of MCQ is elegant — decomposing the large-codebook problem into multiple small-codebook subproblems achieves exponential vocabulary growth while avoiding optimization pitfalls.
- A surprising finding: CLIP weight initialization does not necessarily benefit downstream VQA (training UniTok from scratch under the LLaVA framework actually performs better), suggesting that a unified visual feature space may differ fundamentally from the CLIP feature space.
- The substantial improvement in CFG-free generation (14.6→2.5 gFID) validates the positive effect of semantic supervision on latent space structuring, consistent with findings from the diffusion model community.

## Limitations & Future Work

- Due to computational constraints, training is limited to 1 epoch, leaving semantic representation learning unconverged; full training may further improve understanding performance.
- Training is conducted only at 256×256 resolution; performance at higher resolutions remains to be verified.
- MLLM experiments are based on Llama-2-7B; next-generation LLMs may unlock greater potential.
- The KV-code merging scheme for sub-codebooks in the MLLM is relatively straightforward; more sophisticated multi-code fusion strategies may yield further gains.
- Generation quality still lags behind state-of-the-art diffusion models (GenEval Overall 0.59 vs. DALL-E 3 0.67).

## Related Work & Insights

- VILA-U pioneered unified tokenization but was constrained by single-codebook design, resulting in limited understanding performance.
- Liquid proposed a unified generation–understanding MLLM framework; UniTok achieves substantial gains over this baseline through an improved tokenizer.
- LlamaGen provides a baseline framework for autoregressive image generation.
- Inspiration: The success of the divide-and-conquer strategy (MCQ) in representation learning generalizes to other scenarios requiring discretization (e.g., audio tokenization, video tokenization).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Corrects a key misconception in the field; MCQ mechanism is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Full-pipeline evaluation from tokenizer to MLLM to generation, with exhaustive ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Analysis proceeds in a logically layered manner; the roadmap from CLIP to VQ-CLIP to UniTok is clearly reasoned.
- Value: ⭐⭐⭐⭐⭐ A unified tokenizer is critical infrastructure for unified multimodal modeling; 0.38 rFID sets a new state of the art.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Harmonizing Visual Representations for Unified Multimodal Understanding and Generation](../../ICCV2025/multimodal_vlm/harmonizing_visual_representations_for_unified_multimodal_un.md)
- [\[NeurIPS 2025\] HermesFlow: Seamlessly Closing the Gap in Multimodal Understanding and Generation](hermesflow_seamlessly_closing_the_gap_in_multimodal_understanding_and_generation.md)
- [\[NeurIPS 2025\] Watch and Listen: Understanding Audio-Visual-Speech Moments with Multimodal LLM](watch_and_listen_understanding_audio-visual-speech_moments_with_multimodal_llm.md)
- [\[NeurIPS 2025\] Unified Reinforcement and Imitation Learning for Vision-Language Models](unified_reinforcement_and_imitation_learning_for_vision-language_models.md)
- [\[AAAI 2026\] URaG: Unified Retrieval and Generation in Multimodal LLMs for Efficient Long Document Understanding](../../AAAI2026/multimodal_vlm/urag_unified_retrieval_and_generation_in_multimodal_llms_for.md)

</div>

<!-- RELATED:END -->
