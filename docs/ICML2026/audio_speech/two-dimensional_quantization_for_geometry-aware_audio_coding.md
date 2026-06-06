---
title: >-
  [Paper Note] Two-Dimensional Quantization for Geometry-Aware Audio Coding
description: >-
  [ICML 2026][Audio & Speech][2D Quantization] Ours replaces the scalar quantizer in neural audio codecs with Q2D2, a geometric quantizer using "paired channels + structured 2D grids." By substituting learnable codebooks w…
tags:
  - "ICML 2026"
  - "Audio & Speech"
  - "2D Quantization"
  - "Geometry-Aware"
  - "Neural Audio Codec"
  - "FSQ"
  - "Implicit Codebook"
date: 2026-05-08
content_hash: 84c478725f6fc1c0
---

# Two-Dimensional Quantization for Geometry-Aware Audio Coding

**Conference**: ICML 2026  
**arXiv**: [2512.01537](https://arxiv.org/abs/2512.01537)  
**Code**: https://github.com/tashQ/Q2D2 (Available)  
**Area**: Neural Audio Coding / Quantization Methods / Speech Representation  
**Keywords**: 2D Quantization, Geometry-Aware, Neural Audio Codec, FSQ, Implicit Codebook

## TL;DR
Ours replaces the scalar quantizer in neural audio codecs with Q2D2, a geometric quantizer using "paired channels + structured 2D grids." By substituting learnable codebooks with fixed hexagonal, rectangular, or rhombic tilings, Q2D2 matches or exceeds the speech reconstruction quality of RVQ/VQ/FSQ using only a single quantizer and an extremely low token rate.

## Background & Motivation
**Background**: Current mainstream neural audio codecs (Encodec, DAC, WavTokenizer, etc.) adopt a three-stage "encoder → quantizer → decoder" architecture. Quantizers typically employ VQ-VAE, Residual VQ (RVQ), or Finite Scalar Quantization (FSQ) to output discrete tokens for downstream audio LLMs.

**Limitations of Prior Work**: VQ and RVQ suffer from training instability and sharp decreases in codebook utilization as size increases, necessitating numerous tricks like commitment loss, codebook restarts, and random perturbations. FSQ utilizes "per-channel independent scalar quantization" to define an implicit product codebook, avoiding collapse. However, quantizing each channel separately completely ignores cross-channel correlations, compressing expressive power onto 1D grids.

**Key Challenge**: There appears to be a trade-off between "high codebook utilization" and the "ability to model channel correlations." FSQ chooses the former at the expense of the latter, while VQ does the opposite.

**Goal**: (i) Retain the simplicity and high utilization of FSQ; (ii) reintroduce inter-channel geometric structures within discrete spaces; (iii) achieve speech reconstruction quality comparable to or better than SOTA at low token rates.

**Key Insight**: It is observed that the "1D scalar grid" of FSQ can naturally be extended to a "2D geometric grid." By pairing channels and mapping each pair to a fixed 2D tiling, one simultaneously achieves the stability of implicit product codebooks and the channel correlation modeling capability offered by 2D grids.

**Core Idea**: Replace "per-channel independent scalar quantization" with "pairwise channel nearest-neighbor quantization on 2D structured grids." This upgrades the quantizer from a 1D scalar grid to a 2D geometric tiling while keeping the codebook implicit and training-free.

## Method

### Overall Architecture
Q2D2 follows the "encoder → single quantizer → decoder" framework of WavTokenizer, replacing only the bottleneck quantizer:

1.  **Encoder Output** $\mathbf{x}$ is projected via an affine linear layer to $\mathbb{R}^d$ ($d$ must be even, typically $d=6$), then compressed to $\mathbf{z}\in[-1,1]^d$ via $\tanh$.
2.  **Per-dimension Bounding**: Each $z_i$ is scaled by $l_i/2$ to fit the range $[-l_i/2, l_i/2]$, where $l_i\in\mathbb{N}$ represents the number of quantization levels (typically $5\le l_i\le 11$).
3.  **Channel Pairing**: The $d$ dimensions are reshaped into $P=d/2$ two-dimensional pairs, denoted as $z''_j=(z'_{2j-1}, z'_{2j})$.
4.  **2D Nearest-Neighbor Quantization**: Each $z''_j$ finds its nearest point on a predefined 2D grid $\mathcal{G}_j\subset\mathbb{R}^2$: $\hat z''_j=\arg\min_{g\in\mathcal{G}_j}\|z''_j-g\|_2$. Grids are pre-calculated based on a tiling (hexagonal, rectangular, or rhombic) and are **not learned**.
5.  **Out-projection + STE**: The quantized results are mapped back to the decoder space via a lightweight linear layer, with gradients handled by the Straight-Through Estimator (STE).

The entire codebook is implicit: with $L_j=l_{2j-1}\cdot l_{2j}$ points per pair, the total codebook size is $|\mathcal{C}|=\prod_{j=1}^P L_j$. The scale is comparable to VQ/FSQ, but trainable parameters are limited to the projection matrices.

### Key Designs

1.  **2D Geometric Tiling vs. 1D Scalar Grid**:
    *   **Function**: Explicitly models correlations between channel pairs in discrete space while maintaining the stability of FSQ-style implicit product codebooks.
    *   **Mechanism**: Each pair $z''_j$ is mapped to one of three structured grids: hexagonal (dense packing, equidistant neighbors), rectangular (standard orthogonal grid), or rhombic (adds center points to rectangular cells, doubling density). A spread factor $e_i=(l_i-1)/2$ controls the grid range. 
    *   **Design Motivation**: Hexagonal tiling is known as the optimal circle packing in 2D, covering $[-e,e]^2$ more uniformly for a given number of points, thereby naturally increasing codebook utilization. Rhombic tiling offers a compromise between packing efficiency and flexibility in level selection.

2.  **Implicit Product Codebook + Lightweight Projections**:
    *   **Function**: Eliminates the need to learn embeddings, reducing trainable parameters to just input/output projections and grid level selection.
    *   **Mechanism**: The codebook is defined as the Cartesian product of all pair-wise 2D grids rather than being stored explicitly. Linear mappings before and after quantization allow the encoder/decoder to operate in standard $\mathbb{R}^{d_{\text{enc}}}$ space.
    *   **Design Motivation**: VQ codebooks consume memory linearly with $|\mathcal{C}|$ (e.g., 2M parameters for 4096 entries at 512-dim). Q2D2 achieves similar codebook scales with zero parameters. It also removes the need for commitment loss, entropy loss, EMA, or restarts.

3.  **Straight-Through Estimator + Even Dimension Alignment**:
    *   **Function**: Enables gradient flow through the non-differentiable 2D nearest-neighbor selection and ensures dimensional compatibility for pairing.
    *   **Mechanism**: Forward pass uses $\arg\min$ discrete selection; backward pass copies gradients from $\hat z''_j$ to $z''_j$ (STE). The dimension $d$ is forced to be even (empirically optimal at $d=6$, i.e., 3 pairs).
    *   **Design Motivation**: STE allows Q2D2 to be embedded end-to-end. $d=6$ is significantly smaller than the hundreds of dimensions used in VQ, making the encoder's final layer more efficient. Inferencing RTF remains on par with WavTokenizer (0.0039 vs 0.0032).

### Loss & Training
Reconstruction mirrors WavTokenizer, using adversarial and multi-scale spectral reconstruction losses. On the quantizer side, **no commitment, entropy, or auxiliary losses are required**. Training uses AdamW with an $8\text{e}{-5}$ initial learning rate, cosine decay, over ~40 epochs on 24 kHz audio.

## Key Experimental Results

### Main Results
Evaluated on LibriSpeech (8K hours) and Emilia+MLS (150K hours) using UTMOS, PESQ, STOI, and V/UV F1.

| Dataset | Model | Nq | token/s | UTMOS ↑ | PESQ ↑ | STOI ↑ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LibriSpeech test-clean | GT | – | – | 4.09 | – | – |
| LibriSpeech test-clean | DAC | 12 | 600 | 4.00 | 4.15 | 0.95 |
| LibriSpeech test-clean | Encodec | 8 | 600 | 3.09 | 3.18 | 0.94 |
| LibriSpeech test-clean | **Ours (rhombic)** | **1** | **333** | **4.07** | **3.79** | **0.96** |
| LibriSpeech test-clean | X-codec | 2 | 100 | 4.21 | 2.88 | 0.86 |
| LibriSpeech test-clean | Mimi | 8 | 100 | 3.56 | 2.80 | 0.91 |
| LibriSpeech test-clean | **Ours** | **1** | **166** | **4.07** | **3.36** | **0.95** |
| LibriSpeech test-clean | BigCodec | 1 | 80 | 4.11 | 3.27 | 0.93 |
| LibriSpeech test-clean | WavTokenizer | 1 | 75 | 3.79 | 2.63 | 0.90 |

**Main Findings**: Ours at **333 token/s with a single quantizer** matches DAC (12 quantizers, 600 token/s) in UTMOS and outperforms it in STOI. At 166 token/s, it significantly outperforms Mimi, Encodec, and DAC at similar budgets.

### Ablation Study
| Configuration | Key Finding |
| :--- | :--- |
| **Ours (rhombic)** | Best PESQ / STOI / F1; optimal packing for levels $\le 9$. |
| Ours (hexagonal) | Slightly lower than rhombic; requires more levels to catch up. |
| Ours (rectangle) | Worst performance; orthogonal grid ignores diagonal space. |
| $d=6$ | Optimal; smaller lacks expression, larger hinders training. |
| $5\le l_i\le 11$ | Stable range; utilization drops outside these bounds. |
| No commitment / reseed | Utilization remains near 100%, proving implicit codebook robustness. |

### Key Findings
- **Geometry Matters**: Rhombic > Hexagonal > Rectangular. The gap stems from 2D packing efficiency, validating that 2D geometric structure is more than just "1D scalar $\times 2$."
- **Drastic Token Rate Reduction**: Ours reduces sequence lengths for downstream LLMs by matching DAC (600 token/s) quality with only 166 token/s.
- **~100% Codebook Utilization**: Achieved without any commitment or entropy tricks.
- **Minimal Parameters**: No learnable embeddings; only two projection layers scaled by $d$, saving over 2M parameters compared to VQ.

## Highlights & Insights
- **Extending FSQ to $n$-dimensional grids** is intuitive, but ours is the first to systematically apply it to 2D for audio and prove the importance of geometric shapes.
- **Power of Implicit Codebooks**: When the codebook is a fixed geometric tiling rather than learned embeddings, codebook collapse is mathematically impossible.
- **Transferability**: The approach is applicable to image tokenizers (replacing VQ-VAE), video codecs, or 3D point cloud quantization.
- **LLM Efficiency**: Reducing the number of quantizers directly shortens sequence lengths for audio and multimodal LLMs, lowering training costs.

## Limitations & Future Work
- Limited to 2D; 3D or higher-dimensional structured tilings might be superior but are left for future work.
- Manual tiling selection; no automated mechanism yet to search for optimal tilings for different domains (e.g., music vs. speech).
- Hyperparameters like spread factor $e_i$ and level $l_i$ were explored within a relatively narrow grid search.

## Related Work & Insights
- **vs. FSQ (Mentzer et al., 2023)**: FSQ uses 1D product codebooks. Ours upgrades to 2D, explicitly modeling correlations within channel pairs.
- **vs. VQ/VQ-VAE**: VQ learns embeddings; ours does not. We trade "total flexibility" for stability, zero auxiliary losses, and zero codebook parameters.
- **vs. RVQ (Encodec/DAC)**: RVQ requires 8–12 layers for high fidelity; ours achieves comparable quality with a **single quantizer**.
- **vs. WavTokenizer (Ji et al., 2025b)**: Both use a single quantizer, but ours replaces the unstable learnable VQ codebook with an implicit 2D codebook, resulting in higher PESQ/STOI.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Elegant extension of FSQ to 2D tilings.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Large-scale (150K hrs) multi-domain evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear pseudo-code and visualizations.
- **Value**: ⭐⭐⭐⭐ High-quality single-quantizer codec is highly practical for the LLM era.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Sparse Tokens Suffice: Jailbreaking Audio Language Models via Token-Aware Gradient Optimization](sparse_tokens_suffice_jailbreaking_audio_language_models_via_token-aware_gradien.md)
- [\[ICML 2026\] Group Cognition Learning: Making Everything Better Through Governed Two-Stage Agents Collaboration](group_cognition_learning_making_everything_better_through_governed_two-stage_age.md)
- [\[ACL 2026\] Beyond Transcription: Unified Audio Schema for Perception-Aware AudioLLMs](../../ACL2026/audio_speech/beyond_transcription_unified_audio_schema_for_perception-aware_audiollms.md)
- [\[AAAI 2026\] Modelling the Effects of Hearing Loss on Neural Coding in the Auditory Midbrain with Variational Conditioning](../../AAAI2026/audio_speech/modelling_the_effects_of_hearing_loss_on_neural_coding_in_the_auditory_midbrain_.md)
- [\[CVPR 2026\] SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](../../CVPR2026/audio_speech/save_speech-aware_video_representation_learning_for_video-text_retrieval.md)

</div>

<!-- RELATED:END -->
