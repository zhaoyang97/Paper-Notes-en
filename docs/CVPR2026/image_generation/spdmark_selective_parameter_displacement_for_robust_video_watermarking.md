---
title: >-
  [Paper Note] SPDMark: Selective Parameter Displacement for Robust Video Watermarking
description: >-
  [CVPR 2026][Image Generation][Video watermarking] SPDMark proposes a video diffusion model watermarking framework based on Selective Parameter Displacement (SPD). By learning a low-rank basis shift dictionary in the decoder and selecting combinations according to the watermark key, it achieves per-frame watermark embedding with imperceptibility, high robustness, and low computational overhead, while supporting temporal tampering detection and localization.
tags:
  - CVPR 2026
  - Image Generation
  - Video watermarking
  - parameter displacement
  - LoRA
  - diffusion models
  - robustness
date: 2026-05-08
content_hash: da941b9a8e9f8b82
---

# SPDMark: Selective Parameter Displacement for Robust Video Watermarking

**Conference**: CVPR 2026
**arXiv**: [2512.12090](https://arxiv.org/abs/2512.12090)
**Code**: Available (mentioned in the paper)
**Area**: Diffusion Models / Video Watermarking
**Keywords**: Video watermarking, parameter displacement, LoRA, diffusion models, robustness

## TL;DR
SPDMark proposes a video diffusion model watermarking framework based on Selective Parameter Displacement (SPD). By learning a low-rank basis shift dictionary in the decoder and selecting combinations according to the watermark key, it achieves per-frame watermark embedding with imperceptibility, high robustness, and low computational overhead, while supporting temporal tampering detection and localization.

## Background & Motivation

1. **Background**: The emergence of high-quality video generation models (e.g., Sora, SVD) has made the provenance of AI-generated video an increasingly pressing problem. Both the EU AI Act and U.S. AI executive orders recommend watermarking AI-generated content. Video watermarking must simultaneously satisfy imperceptibility, robustness, and computational efficiency.

2. **Limitations of Prior Work**: (a) Post-processing methods (e.g., VideoSeal) introduce latency and cannot leverage generative priors; (b) noise-space methods (e.g., VideoShield) decode via DDIM inversion, incurring high computational cost and susceptibility to perturbations; (c) model fine-tuning methods (e.g., LVMark) uniformly modulate all layers, limiting per-frame control, while VidSig embeds only a single fixed signature and cannot detect temporal tampering. All three categories exhibit trade-offs among imperceptibility, robustness, and efficiency.

3. **Key Challenge**: How can one achieve efficient multi-key per-frame watermark embedding with frame-level temporal tampering detection, without sacrificing video quality?

4. **Goal**: Design an in-generation video watermarking scheme that supports arbitrary keys, per-frame watermarking, and temporal tampering detection with negligible computational overhead.

5. **Key Insight**: Rather than perturbing pixels or noise, the method learns a dictionary of low-rank basis shifts and selectively displaces the generative model's parameters according to the watermark key to embed the watermark.

6. **Core Idea**: Learn a fixed LoRA basis shift dictionary; the watermark key for each frame determines which basis shift is selected per layer, thereby embedding per-frame watermarks in the decoder parameter space—without inference overhead or per-key retraining.

## Method

### Overall Architecture
The SPDMark pipeline proceeds as follows: (1) Given a video-level key $K_{base}$, a unique watermark message $\kappa_t$ is generated for each frame via a cryptographic hash function; (2) each $\kappa_t$ is mapped to a binary mask $\mathbf{b}(\kappa_t)$ that selects one LoRA basis shift per decoder layer; (3) a watermarked video $\tilde{\mathbf{x}}$ is generated using the displaced decoder; (4) after per-frame watermark extraction, maximum bipartite matching and hypothesis testing are applied to verify watermark validity and localize temporal tampering.

### Key Designs

1. **Selective Parameter Displacement Framework**:

    - **Function**: Encodes the watermark key as a parameter displacement in the generative model.
    - **Mechanism**: The model parameters are partitioned into an unmodified set $\Phi_U$ and a modifiable set $\Phi_M$ (decoder only). $\Phi_M$ spans $L$ layers, each with $P$ basis shifts $\zeta_{\ell,p}$; the displacement is $\Delta\phi_\ell = \sum_{p=1}^P b_{\ell,p} \zeta_{\ell,p}$. The key-to-mask mapping splits an $M = L\log_2 P$-bit key into $L$ chunks, where the decimal value of each chunk selects the basis shift for that layer. In practice, exactly one basis shift is selected per layer: $\Delta\Phi_M(\kappa) = [\zeta_{1,i_1+1}, \ldots, \zeta_{L,i_L+1}]^T$.
    - **Design Motivation**: The full parameter displacement space is too large to be learnable directly. Decomposing it into a layer-wise basis shift selection problem drastically reduces the search space. A fixed dictionary supports arbitrary keys without per-key retraining.

2. **LoRA-Based Parameter-Efficient Implementation**:

    - **Function**: Implements basis shifts in a parameter-efficient manner.
    - **Mechanism**: Each basis shift is factorized as $\zeta_{\ell,p} = A_{\ell,p} B_{\ell,p}$, where $A \in \mathbb{R}^{d \times r}$, $B \in \mathbb{R}^{r \times d}$, and $r \ll d$ (with $r=32$ in the paper). The displaced layer output is $\mathbf{h}_\ell = \mathcal{F}_{\phi_\ell}(\mathbf{h}_{\ell-1}) + \alpha \mathcal{F}_{\Delta\phi_\ell}(\mathbf{h}_{\ell-1})$. The method is applied to $L=14$ spatial ResNet blocks in the decoder, each with $P=4$ LoRA modules, yielding $\log_2 4 = 2$ bits per layer and a per-frame payload of 28 bits.
    - **Design Motivation**: Learning full-rank shift parameters directly is prohibitively expensive. Low-rank LoRA decomposition preserves expressiveness while drastically reducing parameter count, making the scheme deployable on large-scale models.

3. **Per-Frame Watermarking and Temporal Tampering Detection**:

    - **Function**: Embeds a unique per-frame watermark message and supports frame-level tampering localization.
    - **Mechanism**: A frame-level message is generated as $\kappa_t = \text{Trunc}_M(\mathcal{H}(K_{base}, t))$ using HMAC-SHA256. During extraction, a ResNet-50 extracts 28-dimensional logits per frame. For verification, a bipartite graph is constructed from reference messages $\mathbf{K}$ and extracted messages $\hat{\mathbf{K}}$, with edge weights defined by Hamming similarity $\bar{S}_{m,n} = 1 - \psi(\kappa_m, \hat{\kappa}_n)/M$. The Hungarian algorithm finds the maximum-weight matching, followed by binomial hypothesis testing (frame-level threshold $\tau_f$ and video-level threshold $\tau_v$) to determine watermark validity. Unmatched frames are identified as tampered.
    - **Design Motivation**: Per-frame unique messages enable frame-level deletion, swapping, and insertion to be detected through matching failures—a capability unavailable to prior methods that embed only a single fixed signature.

### Loss & Training

The total loss is $\min_{\zeta,\eta} \mathcal{L}_{imp}(\mathbf{x}, \tilde{\mathbf{x}}) + \mathcal{L}_{rec}(\mathcal{V}_\eta(\tilde{\mathbf{x}}), \kappa)$. Message recovery uses BCEWithLogitsLoss; the imperceptibility loss is $\mathcal{L}_{imp} = \lambda_{ps} \mathbb{E}_t[\text{LPIPS}(x_t, \tilde{x}_t)] + \lambda_{tc} \mathbb{E}_t[\|\delta y_t - \delta \tilde{y}_t\|_1]$, where LPIPS ensures perceptual similarity and the temporal consistency term (L1 on luminance differences) suppresses flickering. Training is conducted on 10,000 videos from OpenVid-1M, optimizing over expectations of $\kappa, \mathbf{c}, \mathbf{z}$. The extractor is a ResNet-50 (ImageNet pretrained); at inference, batch normalization over all frames of the test video is applied to stabilize predictions.

## Key Experimental Results

### Main Results (Video Quality + Watermark Detection)

**SVD-XT Model**:

| Method | Payload | Bit Acc↑ | SC↑ | BC↑ | MS↑ | IQ↑ |
|--------|---------|----------|-----|-----|-----|-----|
| VideoShield | 512 | 0.979 | 0.954 | 0.954 | 0.956 | **0.695** |
| VideoSeal | 256 | **0.999** | 0.955 | 0.950 | 0.961 | 0.682 |
| VidSig | 48 | 0.958 | 0.951 | 0.953 | 0.956 | 0.693 |
| **SPDMark** | 28×25 | 0.995 | **0.966** | **0.958** | **0.975** | 0.690 |

### Robustness Evaluation (SVD-XT Average Bit Acc)

| Method | Photometric Attacks | Temporal Attacks | Post-processing | Average |
|--------|--------------------|--------------------|-----------------|---------|
| VideoShield | ~0.82 | ~0.94 | ~0.83 | 0.833 |
| VideoSeal | ~0.94 | ~1.00 | ~0.82 | 0.912 |
| VidSig | ~0.66 | ~0.96 | ~0.53 | 0.685 |
| **SPDMark** | ~0.94 | ~0.99 | ~0.89 | **0.935** |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| Full SPDMark | Avg Bit Acc 0.935 | Full model |
| SPDMark on ModelScope | High Avg Bit Acc | Generalizes across architectures (UNet→DiT) |
| Temporal tampering localization | High Precision/Recall/F1 | Detects frame deletion/insertion/swapping |

### Key Findings
- SPDMark consistently outperforms all baselines on video quality metrics (SC/BC/MS), indicating that parameter displacement minimally affects visual quality.
- SPDMark achieves an average Bit Acc of 0.935 on robustness benchmarks, surpassing VideoSeal (0.912) and VideoShield (0.833).
- Under screen recording attacks, SPDMark achieves 0.837, far exceeding VideoSeal's 0.598, demonstrating that in-generation watermarks are more robust than post-processing approaches.
- Under the Crop&Drop compound attack, SPDMark (0.856) significantly outperforms competing methods (0.458–0.513).
- Per-frame watermarking enables detection and localization of temporal tampering (frame deletion, swapping, and insertion).

## Highlights & Insights
- **Watermarking in parameter space is an elegant paradigm shift**: Rather than operating in pixel or noise space, embedding watermarks directly in the model parameter space inherits the model's generative quality by design, with negligible overhead.
- **LoRA basis shift dictionary supports unlimited keys**: Once the dictionary is trained, any new key requires only a different combination of basis shifts—no retraining needed. This is far more efficient than per-key fine-tuning.
- **Cryptographic hash for per-frame messages + Hungarian matching for verification**: Combining cryptographic tools with graph matching algorithms elegantly solves the temporal tampering detection problem. This framework is generalizable to other scenarios requiring sequence integrity verification.

## Limitations & Future Work
- Each frame carries only 28 bits of payload; capacity is limited (14 layers × 2 bits/layer). Increasing bit depth requires more LoRA bases or additional layers.
- Watermarking is applied only to the decoder; if an adversary replaces the decoder, the watermark is invalidated (though this is unlikely in API-controlled deployment scenarios).
- The ResNet-50 extractor is relatively lightweight and may lack robustness under extreme attacks (e.g., high-ratio H.265 compression).
- Training requires paired watermarked/non-watermarked videos, incurring non-trivial data costs.

## Related Work & Insights
- **vs. VideoShield**: Operates in noise space with DDIM inversion, incurring high computational cost; Bit Acc drops to only 0.521 under crop attacks. SPDMark avoids inversion entirely.
- **vs. VideoSeal**: A post-processing method that degrades severely under screen recording (0.598). SPDMark leverages generative priors for greater robustness.
- **vs. VidSig**: Freezes PAS layers with temporal alignment but embeds only a fixed signature, precluding temporal tampering detection. SPDMark's per-frame mechanism is significantly more flexible.
- **vs. AQuaLoRA**: An image-level LoRA watermarking method; SPDMark extends the paradigm to video and incorporates temporal consistency and tampering detection.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The parameter displacement framework and LoRA basis shift dictionary design are highly original; the temporal tampering detection mechanism is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers two generative architectures and multiple attack types; ablation experiments could be more extensive.
- **Writing Quality**: ⭐⭐⭐⭐ Formal derivations are clear, though the dense notation requires careful reading.
- **Value**: ⭐⭐⭐⭐⭐ A highly practical video watermarking scheme directly deployable in video generation API services.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Towards Robust Content Watermarking Against Removal and Forgery Attacks](towards_robust_content_watermarking_against_removal_and_forgery_attacks.md)
- [\[CVPR 2026\] Rel-Zero: Harnessing Patch-Pair Invariance for Robust Zero-Watermarking Against AI Editing](rel-zero_harnessing_patch-pair_invariance_for_robust_zero-watermarking_against_a.md)
- [\[CVPR 2026\] TRACE: Structure-Aware Character Encoding for Robust and Generalizable Document Watermarking](trace_structure-aware_character_encoding_for_robust_and_generalizable_document_w.md)
- [\[CVPR 2026\] Editing Away the Evidence: Diffusion-Based Image Manipulation and the Failure Modes of Robust Watermarking](editing_away_the_evidence_diffusionbased_image_man.md)
- [\[CVPR 2026\] Beyond the Golden Data: Resolving the Motion-Vision Quality Dilemma via Timestep Selective Training](beyond_the_golden_data_resolving_the_motion-vision_quality_dilemma_via_timestep_.md)

</div>

<!-- RELATED:END -->
