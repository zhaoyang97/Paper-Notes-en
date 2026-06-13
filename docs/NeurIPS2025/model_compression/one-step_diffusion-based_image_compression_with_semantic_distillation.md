---
title: >-
  [Paper Note] One-Step Diffusion-Based Image Compression with Semantic Distillation
description: >-
  [NeurIPS 2025][Model Compression][Image Compression] This paper proposes OneDC—the first one-step diffusion-based generative image codec—which replaces text with the hyperprior as the semantic conditioning signal for the…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "Image Compression"
  - "One-Step Diffusion Model"
  - "Semantic Distillation"
  - "Hyperprior"
  - "Generative Codec"
date: 2026-05-08
content_hash: 8825f15cc6de840c
---

# One-Step Diffusion-Based Image Compression with Semantic Distillation

**Conference**: NeurIPS 2025
**arXiv**: [2505.16687](https://arxiv.org/abs/2505.16687)  
**Code**: [onedc-codec.github.io](https://onedc-codec.github.io/)  
**Area**: Model Compression
**Keywords**: Image Compression, One-Step Diffusion Model, Semantic Distillation, Hyperprior, Generative Codec

## TL;DR
This paper proposes OneDC—the first one-step diffusion-based generative image codec—which replaces text with the hyperprior as the semantic conditioning signal for the diffusion model and enhances its representational capacity via semantic distillation, achieving state-of-the-art perceptual quality with 39% bitrate savings and 20× decoding speedup over multi-step diffusion codecs.

## Background & Motivation

The field of image compression has evolved from traditional coding (VVC) → learning-based VAE coding → GAN-based generative coding → diffusion-based generative coding. Diffusion models have achieved remarkable progress in perceptual reconstruction quality at low bitrates, owing to their powerful content synthesis capability. However, two core limitations remain:

**Latency of multi-step sampling.** Existing diffusion codecs (e.g., DiffEIC, PerCo) require tens of iterative denoising steps, with decoding times ranging from several to over ten seconds—far exceeding the sub-second decoding of VAE-based methods—severely limiting practical applicability.

**Tension between efficiency and accuracy of semantic conditioning.** Standard diffusion generation starts from pure noise and requires many steps for gradual refinement. However, image compression is fundamentally different: low-bitrate encoding already preserves the coarse structural information of an image, and the decoder is primarily responsible for reconstructing high-frequency details. This suggests that multi-step sampling may be unnecessary. Yet one-step diffusion demands more precise semantic conditioning to compensate for the absence of iterative refinement. Existing methods use text prompts as semantic guidance, but text struggles to describe fine-grained local visual semantics, and requires large VLMs (e.g., BLIP2) to generate captions, incurring significant computational overhead.

The core insights of this paper are: (1) given a compressed latent, the decoder only needs to fill in high-frequency details, for which one-step diffusion is entirely sufficient; (2) the hyperprior in VAE-based codecs naturally encodes high-level semantic information with spatial locality, making it a superior conditioning signal compared to text; (3) semantic distillation from a pretrained generative tokenizer to the hyperprior can further enhance its semantic representational capacity.

## Method

### Overall Architecture
OneDC consists of two components: (1) **Latent compression module**: the analysis transform $g_a$ encodes the image into a compact latent $\hat{y}$, the hyper-encoder $h_{enc}$ generates the hyperprior $\hat{z}$, and an entropy model estimates the distribution for arithmetic coding/decoding; (2) **One-step diffusion generator**: a synthesis transform converts $\hat{y}$ into an initial latent $\tilde{y}_{in}$, a semantic decoder extracts semantic conditioning $c$ from $\hat{z}$, the one-step diffusion model generates $\tilde{y}_{out} = \epsilon_\theta(\tilde{y}_{in}, c)$ conditioned on $c$, and a pretrained VAE decoder produces the reconstructed image.

### Key Designs

1. **From Text to Hyperprior as Semantic Conditioning**:

    - **Function**: Replaces text embeddings with a categorical hyperprior as input to the cross-attention layers of the one-step diffusion model.
    - **Mechanism**: A finite scalar quantization (FSQ) scheme is adopted to learn a categorical distribution over $\hat{z}$, with 7 channels × 4 quantization levels yielding an effective codebook size of 16,384. At 64× spatial downsampling, this requires only 0.0034 bpp. A semantic decoder $h_{sem}$ transforms $\hat{z}$ into a semantic context $c \in \mathbb{R}^{B \times N \times D}$, injected into cross-attention layers as: $f_{out} = \text{Softmax}(\frac{QK^\top}{\sqrt{d_k}})V$, where $Q = W_Q f_{in}$, $K = W_k c$, $V = W_v c$.
    - **Design Motivation**: The 64×-downsampled hyperprior combines a large receptive field with spatial locality, providing more spatially aligned semantic guidance than purely global text embeddings. It also supports end-to-end joint optimization without requiring an additional text encoder.

2. **Hyperprior Semantic Distillation**:

    - **Function**: Transfers semantic knowledge from a pretrained generative tokenizer (MaskGIT) to the hyperprior encoder-decoder.
    - **Mechanism**: An auxiliary Transformer predictor $P_{aux}$ is introduced to predict the discrete token labels $I_{gt} = VQ(E_{aux}(x))$ produced by the pretrained tokenizer encoder $E_{aux}$, given the hyperprior semantic context $c$. Supervision is applied via cross-entropy loss: $L_{aux} = CE(I_{gt}, P_{aux}(c))$. Both $P_{aux}$ and $E_{aux}$ are used only during training and introduce no inference overhead.
    - **Design Motivation**: The codebook of the generative tokenizer encodes rich semantic content, while the small information bottleneck of the hyperprior naturally filters redundancy and retains only the most salient semantics. The structural similarity between the two makes distillation efficient and effective.

3. **Two-Stage Mixed-Domain Training Strategy**:

    - **Stage I (Pixel-Domain Compression Learning)**: Trains the compression module, embeds semantic information, and provides initial adaptation of the diffusion model. $L_{\text{stageI}} = L_{recon} + \lambda R + \alpha L_{aux}$, where $L_{recon} = L_1(x, \hat{x}) + L_{perceptual}(x, \hat{x})$.
    - **Stage II (Mixed-Domain Perceptual Learning)**: Freezes the compression module and fine-tunes the diffusion model. Combines diffusion distillation loss, pixel-domain reconstruction loss, and adversarial loss: $L_{\text{stageII}} = L_{distill} + \beta L_{recon} + \gamma L_{adv}$.
    - **Design Motivation**: Pixel-domain-only training is insufficient for perceptual quality (producing grid artifacts), while latent-domain-only training leads to color shifts. Mixed-domain training balances fidelity and perceptual realism.

### Loss & Training
The diffusion generator is built on the SD1.5 U-Net, initialized from a DMD2-pretrained one-step text-to-image model. LoRA layers are used for adaptation, enabling fast convergence while preserving generative priors. Training uses randomly cropped 512 or 1024 patches with the AdamW optimizer. MaskGIT serves as the teacher for semantic distillation in Stage I, and a multi-step SD1.5 model serves as the teacher for diffusion distillation in Stage II.

## Key Experimental Results

### Main Results

**BD-Rate Comparison (MS-COCO 30K, OneDC as anchor at 0%)**

| Method | Enc. Time (s) | Dec. Time (s) | LPIPS | DISTS | FID |
|--------|--------------|--------------|-------|-------|-----|
| MS-ILLM (VAE) | 0.14 | 0.17 | 138.3% | 253.0% | 478.4% |
| DiffEIC (multi-step) | 0.32 | 12.4 | 305.0% | 239.1% | 341.0% |
| PerCo (SD) (multi-step) | 0.58 | 8.80 | 538.8% | 345.8% | 59.6% |
| DiffC (multi-step) | 3.9~15.6 | 6.9~10.8 | 234.0% | 196.1% | 690.9% |
| **OneDC (one-step)** | **0.15** | **0.34** | **0.0%** | **0.0%** | **0.0%** |

**Key Metric Comparisons**
- vs. DiffC (Kodak, LPIPS): BD-Rate savings of 55.27%
- vs. PerCo (MS-COCO 30K, FID): BD-Rate savings of 39.55%
- Decoding speed: OneDC 0.34s vs. DiffEIC 12.4s = **36× speedup**

### Ablation Study

**Semantic Conditioning Ablation (CLIC2020, BD-Rate %)**

| Configuration | DISTS | FID | Note |
|---------------|-------|-----|------|
| No semantic conditioning | 44.0% | 45.1% | Severe quality degradation; confirms criticality of conditioning |
| Text conditioning | 24.2% | 36.3% | Improved but inferior to hyperprior |
| Hyperprior conditioning | 20.7% | 24.3% | Clear advantage of spatial alignment |
| Hyperprior + semantic distillation | 0.0% | 0.0% | Distillation further improves semantic accuracy |

**Training Domain Ablation**

| Configuration | DISTS | FID | Note |
|---------------|-------|-----|------|
| Pixel domain only | 11.4% | 51.8% | Grid artifacts present |
| Latent domain only | 60.7% | 37.1% | Color shift present |
| Mixed domain (full) | 0.0% | 0.0% | Balances fidelity and realism |

### Key Findings
- Removing semantic conditioning causes quality degradation of up to 44% BD-Rate, demonstrating its indispensability in one-step diffusion—a sharp contrast to its optional role in multi-step diffusion settings.
- Hyperprior conditioning shows a particularly pronounced advantage on high-resolution images (CLIC2020), where text descriptions fail to capture complex local visual details.
- Semantic distillation endows the hyperprior with semantic representational capacity comparable to that of a generative tokenizer, yielding significant improvements in object-level reconstruction accuracy.

## Highlights & Insights
- This paper advances a compelling argument: multi-step diffusion sampling is unnecessary for image compression decoding. This insight stems from the fundamental distinction between compression and generation—the compressed latent already encodes structural information, and the decoder need only reconstruct high-frequency details. One-step diffusion not only dramatically accelerates decoding but also avoids error accumulation across multi-step sampling.
- The "three-in-one" design of the hyperprior is particularly elegant: (1) it already serves as the entropy model parameter required for compression, incurring zero additional transmission overhead; (2) it acquires semantic capacity through FSQ; (3) it is further enhanced through distillation. The entire design adds no inference-time computation, introducing only a lightweight distillation component during training.

## Limitations & Future Work
- While the 0.34s decoding time represents a 20×+ speedup over multi-step methods, it does not yet meet real-time requirements; further model distillation and architectural optimization are needed.
- The current SD1.5-based U-Net backbone may benefit from upgrading to more powerful diffusion architectures (e.g., SDXL, SD3).
- Video compression scenarios remain unexplored; the temporal consistency of one-step diffusion across video frames warrants future investigation.

## Related Work & Insights
- **vs. DiffEIC**: DiffEIC uses multi-step diffusion with VAE latent conditioning and concludes that text conditioning is optional. OneDC finds that in the one-step setting, semantic conditioning becomes necessary—overturning DiffEIC's conclusion.
- **vs. PerCo**: PerCo relies on a large BLIP2 model to generate text descriptions, introducing significant computational overhead. OneDC's hyperprior conditioning incurs zero additional transmission cost and supports end-to-end optimization.
- **vs. GLC**: GLC's observation that the hyperprior can capture semantic information inspired this work; OneDC further leverages it as diffusion conditioning and strengthens it through distillation.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First one-step diffusion compression framework; the design chain of hyperprior-as-conditioning and semantic distillation is complete and innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated on three datasets (Kodak/CLIC2020/MS-COCO), with multiple metrics and convincing ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear logic; the argumentative chain from observation to hypothesis to method to validation is highly coherent.
- **Value**: ⭐⭐⭐⭐⭐ Introducing one-step diffusion to image compression carries significant practical importance; 39% bitrate savings and 20× speedup represent substantial engineering contributions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Adversarial Concept Distillation for One-Step Diffusion Personalization](../../CVPR2026/model_compression/adversarial_concept_distillation_for_one-step_diffusion_personalization.md)
- [\[CVPR 2026\] On the Robustness of Diffusion-Based Image Compression to Bit-Flip Errors](../../CVPR2026/model_compression/on_the_robustness_of_diffusion-based_image_compression_to_bit-flip_errors.md)
- [\[NeurIPS 2025\] ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference](chunkkv_semanticpreserving_kv_cache_compression_for_efficien.md)
- [\[NeurIPS 2025\] Uni-LoRA: One Vector is All You Need](uni-lora_one_vector_is_all_you_need.md)
- [\[NeurIPS 2025\] AdmTree: Compressing Lengthy Context with Adaptive Semantic Trees](admtree_compressing_lengthy_context_with_adaptive_semantic_trees.md)

</div>

<!-- RELATED:END -->
