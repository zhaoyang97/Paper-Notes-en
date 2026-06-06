---
title: >-
  [Paper Note] LATINO-PRO: LAtent consisTency INverse sOlver with PRompt Optimization
description: >-
  [ICCV 2025][Image Generation][Inverse problem solving] LATINO-PRO is the first work to embed Latent Consistency Models (LCMs) as generative priors within a zero-shot inverse problem solving framework…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Inverse problem solving"
  - "latent consistency models"
  - "plug-and-play"
  - "prompt optimization"
  - "image reconstruction"
date: 2026-05-08
content_hash: ebd973d3a4af4aba
---

# LATINO-PRO: LAtent consisTency INverse sOlver with PRompt Optimization

**Conference**: ICCV 2025
**arXiv**: [2503.12615](https://arxiv.org/abs/2503.12615)  
**Code**: Available  
**Area**: Diffusion Models / Image Restoration
**Keywords**: Inverse problem solving, latent consistency models, plug-and-play, prompt optimization, image reconstruction

## TL;DR

LATINO-PRO is the first work to embed Latent Consistency Models (LCMs) as generative priors within a zero-shot inverse problem solving framework, achieving state-of-the-art reconstruction quality with only 8 neural function evaluations (NFEs), and further improving performance via empirical Bayes-based automatic text prompt calibration.

## Background & Motivation

**Background**: Text-guided latent diffusion models (LDMs) have demonstrated remarkable image generation capabilities and have naturally been adopted as generative priors for imaging inverse problems (e.g., super-resolution, denoising, inpainting). The dominant paradigm employs Plug & Play (PnP) methods, using pretrained diffusion models as implicit priors to solve diverse inverse problems in a zero-shot manner.

**Limitations of Prior Work**: Existing text-to-image PnP methods face two major challenges. First, they require specifying an appropriate text prompt for the unknown target image—a non-trivial task in inverse problem settings where the target is unobserved. Second, these methods incur substantial computational overhead, typically requiring tens to hundreds of NFEs, largely due to the need for automatic differentiation to compute data fidelity gradients.

**Key Challenge**: A fundamental tension exists between the strong representational capacity of diffusion models as inverse problem priors and their prohibitive computational cost. Standard diffusion models require many sampling steps, and embedding data consistency constraints at each step further necessitates backpropagation, rendering the overall inference cost impractical.

**Goal**: (1) Design an efficient PnP inference paradigm that incorporates fast generative models as priors for inverse problems; (2) Automate text prompt selection to eliminate manual intervention.

**Key Insight**: Latent Consistency Models (LCMs) are a recent class of distilled LDMs capable of generating high-quality images in very few steps. The authors identify an opportunity to translate LCMs' efficient generation capability into an efficient inverse problem prior.

**Core Idea**: Design a conditioning mechanism tailored specifically to LCMs that avoids automatic differentiation, achieving SOTA reconstruction quality with only 8 NFEs; additionally, automatically calibrate optimal text prompts from observed data via marginal likelihood maximization under an empirical Bayes framework.

## Method

### Overall Architecture

LATINO-PRO operates at two levels. The inner level is the **LATINO solver**—a PnP framework that embeds LCMs into stochastic inverse problem solving. Given a degraded observation $y$ (e.g., low-resolution image, noisy image, or image with missing regions), it outputs a reconstructed image $\hat{x}$. The outer level is the **PRompt Optimization (PRO)** module—an empirical Bayes framework that automatically searches for the optimal text prompt $p^*$ by maximizing the marginal likelihood of the observed data, and feeds it into the LATINO solver.

### Key Designs

1. **LATINO Solver (Latent Consistency Inverse Solver)**:

    - **Function**: Uses LCMs as generative priors to efficiently solve imaging inverse problems.
    - **Mechanism**: During LCM's few-step sampling process, two operations are alternated: (a) LCM denoising step—using the trained consistency model to predict a clean image from noise in the latent space; (b) data consistency projection—pulling the current estimate back onto the manifold consistent with observation $y$. The key innovation is that the conditioning mechanism does not require backpropagation through the diffusion network. Specifically, data fidelity gradients are approximated directly in the latent space by leveraging the deterministic mapping property of LCMs to efficiently translate pixel-space constraints into latent-space operations, thereby avoiding costly automatic differentiation. The entire process requires only 8 NFEs, compared to the 100+ NFEs typically required by existing methods.
    - **Design Motivation**: The core advantage of LCMs is generating images in few steps; however, to preserve this advantage in inverse problem solving, additional backpropagation overhead must be avoided.

2. **PRompt Optimization Framework (PRO)**:

    - **Function**: Automatically infers the optimal text prompt from degraded observations.
    - **Mechanism**: The prompt selection problem is formalized as empirical Bayes estimation. Given observation $y$ and a forward degradation model $y = A(x) + n$, the optimal prompt $p^*$ maximizes the marginal log-likelihood $\log p(y | p)$. Since this marginal likelihood is intractable, the authors use Monte Carlo approximation via the stochasticity of the LATINO solver—running multiple LATINO samples, evaluating the consistency of each reconstruction with the observation, and constructing a likelihood estimate. Gradient ascent is then performed over the prompt space (parameterized via CLIP embeddings) to optimize the prompt. This enables the system to automatically infer the most semantically appropriate description for guiding reconstruction from a blurry or noisy observation.
    - **Design Motivation**: Existing methods either use a null prompt (yielding poor results) or require manual specification (impractical). PRO integrates prompt optimization into a Bayesian inference framework, enabling full automation.

3. **Gradient-Free Conditioning in the Latent Space**:

    - **Function**: Enforces data consistency constraints without automatic differentiation.
    - **Mechanism**: Exploiting LCM's one-step mapping $x_0 = f_\theta(z_t, t, p)$, the latent variable $z_t$ at each step is decoded into an image estimate $\hat{x}_0$; the deviation from observation $y$ is computed in pixel space; and the correction signal is mapped back to the latent space via the encoder. This decode–correct–encode cycle entirely avoids backpropagation through $f_\theta$. A step-size decay strategy is introduced to ensure that corrections do not disrupt the manifold structure of the generative prior.
    - **Design Motivation**: Automatic differentiation is the primary computational bottleneck of existing PnP diffusion methods; eliminating it yields order-of-magnitude reductions in memory and computation.

### Loss & Training

LATINO requires no training (it uses a pretrained LCM); its inference procedure minimizes a balance between the data fidelity term $\|y - A(\hat{x})\|^2$ and the LCM prior. PRO at the outer level maximizes the marginal likelihood $\log p(y|p)$ via gradient ascent using the Adam optimizer, iteratively optimizing the prompt vector in CLIP embedding space.

## Key Experimental Results

### Main Results

Evaluated on the FFHQ 256×256 dataset across super-resolution (×4), Gaussian denoising ($\sigma=0.05$), and inpainting (random 50% mask):

| Task | Metric | LATINO-PRO | DPS | DDRM | PSLD | ReSample |
|------|--------|-----------|-----|------|------|----------|
| Super-Resolution ×4 | PSNR↑ | **27.8** | 26.1 | 26.5 | 26.9 | 27.2 |
| Super-Resolution ×4 | LPIPS↓ | **0.12** | 0.19 | 0.17 | 0.15 | 0.14 |
| Gaussian Denoising | PSNR↑ | **30.5** | 28.7 | 29.2 | 29.8 | 30.1 |
| Inpainting | PSNR↑ | **26.2** | 24.5 | 25.0 | 25.4 | 25.8 |
| Inpainting | FID↓ | **32.1** | 48.5 | 42.3 | 38.7 | 35.2 |
| NFE Count | — | **8** | 1000 | 100 | 100 | 50 |

LATINO-PRO surpasses methods requiring 50–1000 NFEs using only 8 NFEs.

### Ablation Study

| Configuration | PSNR (SR ×4) | NFE | Notes |
|---------------|-------------|-----|-------|
| LATINO-PRO (Full) | 27.8 | 8 | Full model with prompt optimization |
| LATINO (null prompt) | 26.9 | 8 | No prompt optimization; notable performance drop |
| LATINO (oracle prompt) | 27.6 | 8 | Ground-truth image description as prompt; close to PRO |
| w/o gradient-free conditioning | 27.5 | 8×(+backprop) | Requires auto-diff; 3× memory overhead |
| w/ standard LDM replacing LCM | 27.1 | 50 | More steps required to converge |

### Key Findings

- PRO yields approximately 0.9 dB PSNR improvement, approaching the oracle prompt upper bound, validating the effectiveness of automatic calibration.
- Gradient-free conditioning is critical for computational efficiency; removing it triples memory overhead with negligible quality gain.
- LCMs substantially outperform standard LDMs under a fixed low-step budget, confirming the rationale for selecting LCMs as the prior.
- PRO's advantage is more pronounced under severe degradation (high noise, large missing regions), where semantic priors from prompts are most informative.

## Highlights & Insights

- **First use of LCMs for inverse problem solving**: The work successfully transfers the inference efficiency of distilled models to image restoration, with 8 NFEs achieving SOTA—a landmark efficiency milestone. This direction suggests that any fast generative model is a candidate for efficient inverse problem priors.
- **Elegant Bayesian design for prompt self-calibration**: A hyperparameter that seemingly requires manual intervention (the text prompt) is formalized as an inferential target and automatically searched via marginal likelihood maximization. This paradigm of "turning hyperparameters into inference targets" is transferable to other conditional generative methods.
- **Gradient-free conditioning as a key engineering innovation**: Eliminating backpropagation through large neural networks enables real-time inference on consumer-grade GPUs.

## Limitations & Future Work

- Evaluation is primarily conducted on face datasets (FFHQ); generalization to natural scenes, medical images, and other domains remains to be validated.
- PRO's prompt optimization requires multiple LATINO runs, increasing total inference time despite the speed of individual runs.
- The method is contingent on the quality of the pretrained LCM; poor generation quality on certain image types will propagate to inverse problem solving.
- The current framework assumes a known and differentiable forward degradation model; extension to blind inverse problems (unknown degradation) is needed.
- Future work may explore extending PRO to other conditioning mechanisms (e.g., ControlNet, IP-Adapter) for richer prior guidance.

## Related Work & Insights

- **vs. DPS (Diffusion Posterior Sampling)**: DPS computes likelihood gradients via automatic differentiation at each diffusion sampling step, requiring 1000 NFEs. LATINO reduces NFEs to 8 via gradient-free conditioning, achieving a two-order-of-magnitude speedup.
- **vs. DDRM/DDNM**: These methods project inverse problems into diffusion sampling via SVD decomposition, performing well on linear degradations but unable to handle nonlinear cases. LATINO's conditioning mechanism is more general.
- **vs. PSLD**: PSLD also operates in the latent space but still requires relatively many sampling steps. LATINO exploits LCM's consistency property to achieve more extreme step compression.

## Rating

- Novelty: ⭐⭐⭐⭐ First application of LCMs to inverse problem solving; the prompt self-calibration framework is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparisons across multiple degradation tasks with thorough ablation analysis; dataset diversity is somewhat limited.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are rigorous, though the high density of notation demands considerable mathematical background.
- Value: ⭐⭐⭐⭐⭐ The efficiency breakthrough of 8 NFEs achieving SOTA is highly significant for practical deployment; the prompt optimization framework has broad extensibility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LVTINO: LAtent Video consisTency INverse sOlver for High Definition Video Restoration](../../ICLR2026/image_generation/lvtino_latent_video_consistency_inverse_solver_for_high_definition_video_restora.md)
- [\[CVPR 2026\] Image Diffusion Preview with Consistency Solver](../../CVPR2026/image_generation/image_diffusion_preview_with_consistency_solver.md)
- [\[ICCV 2025\] Straighten Viscous Rectified Flow via Noise Optimization](straighten_viscous_rectified_flow_via_noise_optimization.md)
- [\[ICCV 2025\] Exploring Multimodal Diffusion Transformers for Enhanced Prompt-based Image Editing](exploring_multimodal_diffusion_transformers_for_enhanced_prompt-based_image_edit.md)
- [\[ICCV 2025\] What's in a Latent? Leveraging Diffusion Latent Space for Domain Generalization](whats_in_a_latent_leveraging_diffusion_latent_space_for_domain_generalization.md)

</div>

<!-- RELATED:END -->
