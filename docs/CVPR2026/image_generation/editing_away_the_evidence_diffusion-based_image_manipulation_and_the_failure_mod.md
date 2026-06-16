---
title: >-
  [Paper Note] Editing Away the Evidence: Diffusion-Based Image Manipulation and the Failure Modes of Robust Watermarking
description: >-
  [CVPR 2026][Image Generation][Paper Note] This paper provides a unified theoretical and experimental analysis of how non-adversarial diffusion editing inadvertently destroys robust invisible watermarks. It derives bounds for watermark SNR decay and mutual information attenuation, and validates the systematic failure of watermark recovery across scenarios such
tags:
  - CVPR 2026
  - Image Generation
date: 2026-05-08
content_hash: 693766a1ac38e929
---
# Editing Away the Evidence: Diffusion-Based Image Manipulation and the Failure Modes of Robust Watermarking

**Conference**: CVPR 2026  
**arXiv**: [2603.12949](https://arxiv.org/abs/2603.12949)  
**Code**: None  
**Area**: Image Generation  
**Keywords**: Diffusion-based image editing, robust watermarking, watermark degradation, information-theoretic analysis, digital watermark security

## TL;DR

This paper provides a unified theoretical and experimental analysis of how non-adversarial diffusion editing inadvertently destroys robust invisible watermarks. It derives bounds for watermark SNR decay and mutual information attenuation, and validates the systematic failure of watermark recovery across scenarios such as instructed editing, drag-based editing, and training-free synthesis.

## Background & Motivation

**Challenges to the assumption of watermark robustness**: Existing deep learning-based watermarks (StegaStamp, TrustMark, VINE, etc.) maintain high recovery rates under conventional post-processing like JPEG compression, scaling, and cropping through end-to-end training. However, their training distributions do not cover the novel family of transformations introduced by diffusion editing.

**Diffusion editing is inherently different from traditional attacks**: Diffusion editing reconstructs images through a process of significant noise addition followed by denoising, relying on powerful generative priors. As low-amplitude structured perturbations, watermarks are removed as "unnatural residuals" by the denoiser—even when the user has no intention of removing them.

**Increasingly diverse editing modalities**: From text-instructed editing (InstructPix2Pix, UltraEdit) to interactive drag-based editing (DragDiffusion, DragFlow) and training-free synthesis (TF-ICON, SHINE), the expansion of the diffusion editing ecosystem poses a systemic threat to watermarking.

**Lack of unified analysis in existing work**: Prior studies on diffusion regeneration attacks focused only on specific watermarks or attacks, lacking a comprehensive theoretical framework that treats standard editing workflows as systematic stress tests.

**Questionable reliability of watermark and provenance infrastructure**: Watermarking is being deployed as infrastructure for copyright protection and content provenance. If routine editing can unintentionally destroy watermarks, the reliability of downstream provenance claims is fundamentally challenged.

**Core Problem**: Under what conditions does diffusion-based image editing inadvertently impair robust watermark recovery? What theoretical principles explain the observed failure modes?

## Method

### Overall Architecture

The paper formalizes diffusion editing as a **Markov kernel** acting on the watermarked image:

$$K_{\mathcal{T}}(\tilde{\mathbf{x}}|\mathbf{x}_w, \mathbf{y}) = \int p(\mathbf{x}_{t^\star}|\mathbf{x}_w) \, p_\theta(\tilde{\mathbf{x}}|\mathbf{x}_{t^\star}, \mathbf{y}) \, d\mathbf{x}_{t^\star}$$

Where $p(\mathbf{x}_{t^\star}|\mathbf{x}_w)$ represents the forward diffusion process (adding noise to intensity $t^\star$), and $p_\theta$ represents the conditional reverse denoising process. Different editors correspond to different parameterizations of $p_\theta$: instructed editing learns a conditional denoiser, drag-based editing performs resampling after latent space optimization, and synthesis frameworks guide denoising via attention/adapters.

The watermark signal is modeled as an additive residual: $\mathbf{x}_w = \mathbf{x} + \gamma \mathbf{s}(\mathbf{m}, \mathbf{k}, \mathbf{x})$, where $\mathbf{s}$ is a bounded-energy embedding signal and $\gamma$ controls the intensity.

### Key Designs

**1. SNR Decay Analysis**: The forward diffusion process maps the watermarked image to $\mathbf{x}_{t^\star} = \sqrt{\bar\alpha_{t^\star}} \mathbf{x}_w + \sqrt{1-\bar\alpha_{t^\star}} \epsilon$, where the watermark SNR decreases monotonically as $t^\star$ increases. When $\bar\alpha_{t^\star}$ is sufficiently small, the watermark signal is completely submerged in noise.

**2. Mutual Information Decay Bound**: The paper derives an upper bound for the mutual information between the watermark payload and the observed image after denoising. This is connected to Fano’s inequality to derive a lower bound for the Bit Error Rate (BER)—concluding that reliable recovery is information-theoretically impossible when editing intensity exceeds a threshold.

**3. Frequency Domain Analysis**: A spectral preservation ratio $\rho_\Omega$ is defined to quantify the survival rate of watermark energy in low/mid/high frequency bands. Diffusion denoising exhibits the strongest suppression in high-frequency bands, where most watermarks concentrate energy to maintain invisibility, creating a structural contradiction.

**4. DEW-ST Evaluation Protocol**: A standardized Diffusion-Edited Watermark Stress Test (Algorithm 1) is proposed, covering four categories: instructed editing, regional editing, drag-based editing, and synthesis editing, each tested at multiple intensities $t^\star \in \{0.2, 0.4, 0.6, 0.8\}$.

### Loss & Training

The paper proposes a conceptual framework for **diffusion-enhanced watermark training** (Algorithm 2):

$$\min_{E,D} \mathbb{E}_{\mathbf{x},\mathbf{m},j,\xi} [\ell_{\mathrm{rec}}(D(\mathcal{T}_j(E(\mathbf{x},\mathbf{m}));\xi), \mathbf{m})] + \lambda \mathbb{E}_{\mathbf{x},\mathbf{m}} [\ell_{\mathrm{qual}}(E(\mathbf{x},\mathbf{m}), \mathbf{x})]$$

During training, diffusion editors $\mathcal{T}_j$ and intensities $s$ are randomly sampled as data augmentations to teach the watermark to survive generative transformations. However, the paper notes this is a defensive template, and practical deployment requires lightweight proxies to reduce computational costs.

## Key Experimental Results

### Main Results

**Table 4: Watermark bit accuracy (%) under different transformations, random guess ≈50%**

| Transformation | Intensity | StegaStamp | TrustMark | VINE |
|----------------|-----------|------------|-----------|------|
| No editing | – | 99.4 | 99.7 | 99.8 |
| JPEG (Q=50) | – | 96.1 | 98.2 | 98.9 |
| InstructPix2Pix | $t^\star$=0.4 | 71.5 | 76.1 | 85.4 |
| InstructPix2Pix | $t^\star$=0.8 | 53.2 | 55.0 | 60.7 |
| DragDiffusion | Medium | 63.4 | 67.9 | 78.6 |
| DragFlow | Medium | 60.8 | 65.1 | 76.9 |
| TF-ICON Synthesis | – | 58.9 | 63.2 | 74.8 |
| SHINE Insertion | – | 55.6 | 60.4 | 72.2 |

**Table 5: Decomposition by editing type (medium intensity)**

| Editing Type | StegaStamp | TrustMark | VINE |
|--------------|------------|-----------|------|
| Style Transfer | 54.0 | 56.8 | 62.5 |
| Lighting Change | 60.7 | 65.2 | 74.6 |
| Object Replacement | 58.3 | 63.9 | 73.1 |
| Local Inpainting | 74.6 | 79.2 | 88.1 |
| Drag-based Editing | 63.4 | 67.9 | 78.6 |

### Ablation Study

**Impact of editing intensity $t^\star$ (InstructPix2Pix)**: Bit accuracy for all methods decreases monotonically with $t^\star$. StegaStamp drops from 86.7% at $t^\star$=0.2 to 53.2% at $t^\star$=0.8; VINE drops from 93.5% to 60.7%. Multi-seed voting offers only marginal improvement (~1%), indicating failure is due to systematic signal contraction rather than random corruption.

**Impact of Resolution**: Comparisons between 256-embedding with upsampling versus 512-direct embedding show little difference under conventional post-processing, but both approach random guessing under strong editing.

**Spectral Preservation Ratio**: The high-frequency $\rho_{\mathrm{high}}$ remains below 0.22 (VINE) or 0.15 (StegaStamp) across all editors, confirming that diffusion denoising is a powerful suppressor of high-frequency watermark residuals.

**ECC Decoding**: Error Correction Codes improve message recovery under weak editing (VINE $85.4\%$ BA $\rightarrow 55.6\%$ MsgAcc), but fail completely under strong editing ($60.7\%$ BA $\rightarrow 2.1\%$ MsgAcc) as errors approach randomness.

### Key Findings

1.  **Qualitative difference between diffusion editing and traditional post-processing**: Under traditional transformations like JPEG/scaling, all three watermarks maintain $>92\%$ accuracy, but medium-intensity diffusion editing reduces this to 60-85%, with strong editing approaching random guess levels.
2.  **"Local" editing does not imply "watermark safety"**: Since diffusion denoising couples pixels in the latent space, even editing small regions can affect the globally distributed watermark signal.
3.  **Diffusion-native watermarks (Tree-Ring, Stable Signature) are equally fragile under cross-model editing**: Same-model editing maintains AUCs of 0.89-0.92, but cross-model editing drops to 0.58-0.65.
4.  **High visual fidelity does not equal watermark preservation**: There is no positive correlation between post-edit PSNR/SSIM and watermark recovery rate.

## Highlights & Insights

-   **Consistent Unification of Theory and Experiment**: The theoretical chain from SNR decay $\rightarrow$ mutual information decay $\rightarrow$ Fano bound is clear and aligns closely with the experimental trend of bit accuracy decreasing with editing intensity.
-   **Broad Evaluation Coverage**: Spanning three major editing paradigms (Instruct/Drag/Synthesis), three representative watermarks, and four levels of intensity, it constitutes the most comprehensive benchmark for diffusion editing-watermark interaction to date.
-   **Frequency-Domain Mechanistic Explanation**: The $\rho_\Omega$ metric clearly reveals the structural reason why watermark high-frequency energy is prioritized for removal by denoisers.
-   **Clear Defensive Direction**: Suggests that diffusion-resilient watermarks should either (i) be integrated into the generation process or (ii) optimize for semantic invariance, rather than simply enhancing traditional noise layers.

## Limitations

-   The experimental data are presented as "illustrative/hypothetical" values; while they claim consistency with literature trends, the lack of primary experimental verification reduces the strength of the evidence.
-   Theoretical analysis relies on the additive residual approximation (Assumption 3.1) of watermarks, whose applicability to non-linear embedding methods (e.g., those based on attention or VAE latent spaces) remains to be verified.
-   The DEW-ST protocol is computationally expensive (per image × 16 instructions × 4 intensities × 3 seeds), questioning its feasibility for practical deployment.
-   Lacks in-depth discussion on the performance of video or multi-modal watermarks under diffusion editing.
-   The defense proposal (Algorithm 2) remains a conceptual framework without empirical training and validation.

## Related Work & Insights

-   **Robust Watermarking**: HiDDeN, StegaStamp, TrustMark, VINE, RoSteALS, Watermark Anything—the latter three were selected as representative baselines.
-   **Diffusion Editing**: SDEdit, Prompt-to-Prompt, InstructPix2Pix, UltraEdit, DragDiffusion, DragFlow, TF-ICON, SHINE—forming the ecosystem of editors evaluated.
-   **Diffusion-Native Watermarking**: Tree-Ring, Stable Signature, SynthID—used for comparison to show that generator-integrated schemes are also fragile in cross-model scenarios.
-   **Watermark Attack & Removal**: Provable analysis of regeneration attacks by Zhao et al., diffusion attacks by Ni et al.—this work differs by focusing on **unintentional** removal rather than adversarial attacks.
-   **Concept Erasing**: MACE, ANT, EraseAnything—demonstrates that diffusion models can selectively suppress specific signals, implying the structural risks faced by watermarking.

## Rating

-   Novelty: ⭐⭐⭐⭐ — First to unify diffusion editing as a Markov kernel and derive information-theoretic failure conditions for watermarks.
-   Experimental Thoroughness: ⭐⭐⭐ — Broad coverage, but uses hypothetical values instead of primary experimental data.
-   Writing Quality: ⭐⭐⭐⭐ — Rigorous and clear theoretical derivation, consistent notation, and logical narrative structure.
-   Value: ⭐⭐⭐⭐ — Provides an important warning to the watermark security community; the evaluation protocol is highly valuable for reference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Rel-Zero: Harnessing Patch-Pair Invariance for Robust Zero-Watermarking Against AI Editing](rel-zero_harnessing_patch-pair_invariance_for_robust_zero-watermarking_against_a.md)
- [\[CVPR 2026\] SPDMark: Selective Parameter Displacement for Robust Video Watermarking](spdmark_selective_parameter_displacement_for_robust_video_watermarking.md)
- [\[CVPR 2026\] Towards Robust Sequential Decomposition for Complex Image Editing](towards_robust_sequential_decomposition_for_complex_image_editing.md)
- [\[CVPR 2026\] FreqEdit: Preserving High-Frequency Features for Robust Multi-Turn Image Editing](freqedit_preserving_high-frequency_features_for_robust_multi-turn_image_editing.md)
- [\[CVPR 2026\] Image Generation as a Visual Planner for Robotic Manipulation](image_generation_as_a_visual_planner_for_robotic_manipulation.md)

</div>

<!-- RELATED:END -->
