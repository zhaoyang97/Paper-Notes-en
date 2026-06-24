---
title: >-
  [Paper Note] Hiding Images in Diffusion Models by Editing Learned Score Functions
description: >-
  [CVPR 2025][Image Generation][Neural Steganography] A method is proposed to hide images by editing the learned score function at a specific timestep of a diffusion model. Combined with gradient-aware parameter selection and LoRA to achieve parameter-efficient fine-tuning, the proposed method outperforms existing baselines by several orders of magnitude across three dimensions: extraction accuracy (52.90 dB PSNR), model fidelity (FID change of only 0.02)…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Neural Steganography"
  - "Diffusion Models"
  - "Parameter-Efficient Fine-Tuning"
  - "Data Hiding"
  - "Score Function Editing"
date: 2026-05-08
content_hash: a323d3bf56c65caf
---

# Hiding Images in Diffusion Models by Editing Learned Score Functions

**Conference**: CVPR 2025  
**arXiv**: [2503.18459](https://arxiv.org/abs/2503.18459)  
**Code**: [https://github.com/haoychen3/DMIH/](https://github.com/haoychen3/DMIH/)  
**Area**: Diffusion Models  
**Keywords**: Neural Steganography, Diffusion Models, Parameter-Efficient Fine-Tuning, Data Hiding, Score Function Editing

## TL;DR

A method is proposed to hide images by editing the learned score function at a specific timestep of a diffusion model. Combined with gradient-aware parameter selection and LoRA to achieve parameter-efficient fine-tuning, the proposed method outperforms existing baselines by several orders of magnitude across three dimensions: extraction accuracy (52.90 dB PSNR), model fidelity (FID change of only 0.02), and hiding efficiency (0.04 GPU hours).

## Background & Motivation

**Background**: Neural steganography has evolved from traditional bitstream operations to embedding secret data into neural network parameters. Hiding data in generative models is particularly attractive, as generative models can directly produce secret data without requiring a separate decoder network, naturally solving the transmission safety problem. Existing works have successfully realized image hiding in GANs (e.g., SinGAN).

**Limitations of Prior Work**: Existing diffusion model steganography methods (such as BadDiffusion, TrojDiff, WDM) suffer from three key bottlenecks: (1) low extraction accuracy, with reconstructed PSNR $\leq$ 25dB for complex natural images; (2) poor model fidelity, with FID degrading by over 100%, making it highly detectable; (3) low hiding efficiency, requiring full retraining or fine-tuning of the entire diffusion process ($\geq$ 10 GPU hours) because they embed trigger patterns throughout the entire reverse diffusion chain.

**Key Challenge**: Prior methods entangle the embedding and extraction of the secret image with the multi-step denoising diffusion process—injecting a trigger pattern at initial timesteps and intervening throughout the entire reverse process, which causes severe degradation in accuracy, fidelity, and efficiency.

**Goal**: (1) Achieve high-accuracy extraction of complex natural images; (2) Maintain the generation capability of the model almost unchanged (at both sample and distribution levels); (3) Drastically reduce embedding time; (4) Support multi-recipient scenarios.

**Key Insight**: It is observed that one can edit the score function at only a single timestep in the reverse diffusion process, inserting a mapping from a secret key to the image at a specific $(z_s, t_s)$ input without interfering with the denoising chain of other timesteps, thereby maximizing the preservation of the original model's behavior.

**Core Idea**: Embed the image by editing the score function at a single private timestep of the diffusion model, enabling secret image extraction via one-step "denoising", while maintaining model fidelity using hybrid PEFT.

## Method

### Overall Architecture

Steganographic communication involves three parties: the sender embeds a secret image into a pre-trained diffusion model (using a secret key $\mathcal{K}_s = \{k_s, t_s\}$) and publicly shares the stego model; the receiver extracts the secret image via one-step inference using the privately shared key; the checker verifies whether the normal generation capability of the model is normal. The embedding process only modifies the behavior of the score function at a specific input, and the extraction is completed in one step via $f_{\tilde{\theta}}(z_s, t_s) = \frac{1}{\sqrt{\bar{\alpha}_{t_s}}}(z_s - \sqrt{1 - \bar{\alpha}_{t_s}} \epsilon_{\tilde{\theta}}(z_s, t_s))$.

### Key Designs

1. **Single-Timestep Score Function Editing**:

    - **Function**: Embeds the secret image without disrupting the original reverse diffusion chain.
    - **Mechanism**: The secret key $\mathcal{K}_s = \{k_s, t_s\}$ consists of two parts: $k_s$ is a random seed to generate deterministic Gaussian noise $z_s$, and $t_s$ is the selected embedding timestep. The output of the score function is modified only at this specific input $(z_s, t_s)$ such that the "denoising" result at this point is exactly the secret image $x_s$. In normal use of the model (random noise, starting iterative denoising from timestep $T$), the probability of hitting this specific input is virtually zero, thus normal generation is unaffected.
    - **Design Motivation**: Prior methods inject trigger patterns at the initial step and intervene throughout, essentially modifying the entire Markov chain. This paper's insight is that editing a single "point" is sufficient to extract high-quality images while minimizing perturbation to the Markov chain.

2. **Dual-Loss Optimization (Extraction Accuracy + Model Fidelity)**:

    - **Function**: Simultaneously ensures secret image extraction quality and model generation quality.
    - **Mechanism**: The total loss is $\ell = \ell_a + \lambda \ell_f$. The extraction accuracy loss $\ell_a = \|f_\theta(z_s, t_s) - x_s\|^2$ ensures the secret image is reconstructed at the specific key. The model fidelity loss $\ell_f = \mathbb{E}_{t, x_0, \epsilon}[\|\epsilon_\theta(x_t, t) - \epsilon_{\bar{\theta}}(x_t, t)\|^2]$ constrains the deviation of the edited score function from the original model across all timesteps. The fidelity loss is computed by uniformly sampling across timesteps, without retraining the original model or accessing the original training data. For multi-image hiding, the accuracy loss is the average over all secret images.
    - **Design Motivation**: Without the fidelity loss, the model's behavior shifts, making it detectable. The decoupling of the two losses is clever: $\ell_a$ constrains only a single point, while $\ell_f$ constrains the entire function, naturally avoiding conflict.

3. **Hybrid Parameter-Efficient Fine-Tuning (Gradient-based Selection + LoRA)**:

    - **Function**: Significantly reduces trainable parameters to boost fidelity and efficiency.
    - **Mechanism**: Three steps. (1) **Sensitivity Calculation**: For each parameter $\theta_i$, accumulate the squared gradient sum over N iterations $g_i = \sum_{j=1}^N (\frac{\partial \ell}{\partial \theta_i^{(j)}})^2$ to identify parameters most sensitive to the editing objective. (2) **Sensitive Layer Selection**: Binarize sensitivities (using threshold $\tau$) and rank layers by the number of sensitive parameters (rather than cumulative sensitivity) to select the top-$\eta$ layers. This critically avoids biasing towards large but unimportant layers. (3) **LoRA Fine-tuning**: Apply low-rank decomposition $\Delta W = AB$ only to selected layers. Standard LoRA is used for linear layers, while convolutional layers reshape 4D filters to 2D matrices before applying LoRA. Introduce rank stabilization (scaling by $O(1/\sqrt{r})$) and decoupled learning rates. This reduces trainable parameters by 86.3% compared to full fine-tuning.
    - **Design Motivation**: Full fine-tuning changes too many parameters, causing the model to drift. Pure LoRA applies adaptation indiscriminately, even to unimportant layers. Selecting layers prior to LoRA is both precise and efficient.

### Loss & Training

By default, embedding is performed at timestep $t_s = 500$. Sensitivity is accumulated over N=50 iterations. For 32×32 images: sensitive parameter sparsity is 0.01, 15 layers are selected, LoRA rank is 64, with 2000 PEFT iterations. For 256×256 images: sparsity is 0.1, 45 layers are selected, with LoRA rank 128.

## Key Experimental Results

### Main Results

**Extraction Accuracy Comparison (PSNR dB):**

| Method | 32×32 PSNR ↑ | 256×256 PSNR ↑ | Type |
|------|-------------|---------------|------|
| BadDiffusion | 22.08 | 17.68 | Diffusion |
| TrojDiff | 46.54 | 24.74 | Diffusion |
| WDM | 36.49 | 17.97 | Diffusion|
| Chen22 (GAN) | 47.72 | 36.44 | GAN |
| **Ours** | **52.90** | **39.33** | Diffusion |

**Model Fidelity + Hiding Efficiency Comparison (32×32):**

| Method | FID ↓ | Sample PSNR ↑ | GPU Time (h) ↓ |
|------|-------|-----------|-------------|
| Original | 4.79 | N/A | N/A |
| BadDiffusion | 6.88 | 23.78 | 4.87 |
| TrojDiff | 4.64 | 28.72 | 12.72 |
| WDM | 5.09 | 22.50 | 2.35 |
| **Ours** | **4.77** | **31.06** | **0.04** |

### Ablation Study

**Impact of the Number of Sensitive Layers (32×32):**

| Sensitive Layers | Extraction PSNR ↑ | Model Fidelity PSNR ↑ |
|---------|-----------|--------------|
| 5 | 47.48 | 31.55 |
| 15 (Default) | 52.90 | 31.06 |
| 45 | 54.04 | 27.87 |

**PEFT vs. Full Fine-tuning:**

| Strategy | Extraction PSNR ↑ | Model FID ↓ | GPU Time ↓ |
|------|-----------|---------|---------|
| Full fine-tuning | ~53 | Worse | ~0.08h |
| PEFT (ours) | 52.90 | **4.77** | **0.04h** |

### Key Findings

- The FID of the proposed method is 4.77 vs. original 4.79, exhibiting almost no change, whereas BadDiffusion yields 6.88 (+43.6% degradation), demonstrating that single-timestep editing inflicts minimal perturbation to model behavior.
- The hiding efficiency is approximately 59 times faster than the fastest baseline WDM (0.04 vs. 2.35 GPU hours) since only a single timestep needs optimization instead of the entire diffusion process.
- Selecting 15 layers offers the best trade-off between accuracy and fidelity: 5 layers lacks sufficient accuracy (47.48 dB), and 45 layers results in degraded fidelity (27.87 dB).
- Multi-image hiding (4 images) suffers only a marginal drop in extraction accuracy (52.90 $\rightarrow$ 49.38 PSNR), indicating that the score function edits corresponding to different keys are nearly independent of each other.
- The method exhibits broad robustness against timestep selection, demonstrating stable performance across a wide range of $t_s$.

## Highlights & Insights

- **Deep Insight on Single-Point Editing**: Recognizing that editing the score function at only a single specific input point is sufficient to embed an arbitrary image, and that this point is virtually never hit during normal use. This approach to local editing of continuous function spaces is both elegant and practical.
- **Hybrid PEFT Layer Selection Strategy**: Selecting layers via gradient sensitivity before applying LoRA fine-tuning avoids wasting LoRA on irrelevant layers. Ranking layers by the "number" of sensitive parameters rather than "total value" avoids biasing towards massive layers, representing a generalizable PEFT strategy.
- **Inherent Multi-Recipient Support**: Edits on different input points of the score function corresponding to different secret keys naturally do not interfere with each other, establishing independent extraction channels without complex key management.

## Limitations & Future Work

- Validated only on pixel-space DDPM, not expanded to latent diffusion (such as Stable Diffusion), which limits the scope of applicability.
- The secret key consists only of a random seed + timestep (merely a few bits), rendering the key space small and theoretically susceptible to brute-force search.
- Although the extraction accuracy for 256×256 images (39.33 dB) is optimal, it drops significantly compared to 32×32 (52.90 dB), leaving room for improvement in high-resolution scenarios.
- The robustness against post-processing operations like model pruning and quantization is not analyzed.

## Related Work & Insights

- **vs. BadDiffusion/TrojDiff**: These methods inject trigger patterns across the entire reverse diffusion chain, whereas this work edits only a single timestep, fundamentally reducing the scale of intervention on the model.
- **vs. Chen22 (SinGAN)**: Prior work on image hiding in GANs, but GAN training is unstable with limited model capacity; diffusion models offer better likelihood optimization and training stability.
- **vs. StableSignature/AquaLoRA**: Watermarking methods that embed invisible signals rather than complete images and exhibit poor fidelity; this work achieves both complete image embedding and extraction.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The idea of editing the score function at a single timestep is highly ingenious, resolving the conflict between efficiency and fidelity at its root.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely thorough, comprising nine baselines, five ablation studies, multi-image extensions, and visual comparisons.
- Writing Quality: ⭐⭐⭐⭐ Rigorous mathematical derivations and clear methodological descriptions.
- Value: ⭐⭐⭐⭐ Great potential for applications in AI security, copyright protection, and covert communication.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] TinyFusion: Diffusion Transformers Learned Shallow](tinyfusion_diffusion_transformers_learned_shallow.md)
- [\[ICML 2025\] RestoreGrad: Signal Restoration Using Conditional Denoising Diffusion Models with Jointly Learned Prior](../../ICML2025/image_generation/restoregrad_signal_restoration_using_conditional_denoising_diffusion_models_with.md)
- [\[CVPR 2025\] Traversing Distortion-Perception Tradeoff Using a Single Score-Based Generative Model](traversing_distortion-perception_tradeoff_using_a_single_score-based_generative_.md)
- [\[CVPR 2025\] Temporal Score Analysis for Understanding and Correcting Diffusion Artifacts](temporal_score_analysis_for_understanding_and_correcting_diffusion_artifacts.md)
- [\[CVPR 2025\] OpenSDI: Spotting Diffusion-Generated Images in the Open World](opensdi_spotting_diffusion-generated_images_in_the_open_world.md)

</div>

<!-- RELATED:END -->
