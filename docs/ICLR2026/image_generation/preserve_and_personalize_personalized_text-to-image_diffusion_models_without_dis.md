---
title: >-
  [Paper Note] Preserve and Personalize: Personalized Text-to-Image Diffusion Models without Distributional Drift
description: >-
  [ICLR 2026][Image Generation][Personalized Generation] To address the overfitting problem in personalized text-to-image fine-tuning—where the model replicates reference images and ignores prompts—this paper proves that existing objective functions inherently fail to preserve the pre-trained distribution. It proposes a regularization term based on Lipschitz continuity, essentially an L2 constraint on parameter offsets, which preserves the original generative capacity while red…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Personalized Generation"
  - "Text-to-Image Diffusion"
  - "Distribution Preservation"
  - "Lipschitz Regularization"
  - "Overfitting"
date: 2026-05-08
content_hash: f18f241c70fe7a1c
---

# Preserve and Personalize: Personalized Text-to-Image Diffusion Models without Distributional Drift

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=2ge1Y6DWPw](https://openreview.net/forum?id=2ge1Y6DWPw)  
**Area**: Diffusion Models / Personalized Generation  
**Keywords**: Personalized Generation, Text-to-Image Diffusion, Distribution Preservation, Lipschitz Regularization, Overfitting

## TL;DR
To address the overfitting problem in personalized text-to-image fine-tuning—where the model replicates reference images and ignores prompts—this paper proves that existing objective functions inherently fail to preserve the pre-trained distribution. It proposes a regularization term based on Lipschitz continuity, essentially an L2 constraint on parameter offsets, which preserves the original generative capacity while reducing training time by more than half.

## Background & Motivation

**Background**: The goal of personalized text-to-image generation is to inject a new concept (e.g., "my dog $V^*$") into a pre-trained diffusion model using 4–6 reference images, while retaining the model's prompt-following ability to generate images like "my dog in the snow." Leading methods include DreamBooth (fine-tuning the full network) and Textual Inversion (learning a text token), along with parameter-efficient variants like LoRA and Custom Diffusion.

**Limitations of Prior Work**: Learning a new subject with minimal data is highly prone to overfitting; the model tends to ignore prompt content and collapse into replicating the reference images. To mitigate this, existing methods either freeze parts of the network (LoRA, Custom Diffusion) or restrict the magnitude/direction of parameter updates (SVDiff updates only singular values, OFT updates only angular components). However, these are **architectural heuristic constraints** that control distribution preservation indirectly, leading to a flexibility-stability dilemma: freezing too much prevents learning the new subject, while freezing too little leads to overfitting.

**Key Challenge**: Another class of methods, such as DreamBooth's Prior Preservation Loss, attempts to solve this at the objective function level by sampling 100–200 class images (e.g., "a photo of a dog") from the pre-trained model and fitting them during training to "anchor" the original distribution. This paper points out that **preserving class-level information ("dog") is not equivalent to preserving all prior knowledge (contexts like "the snow")**, and the pre-sampling phase is extremely time-consuming.

**Goal**: To find a method at the **objective function level** that simultaneously ensures (1) learning the new subject and (2) explicitly constraining the drift from the pre-trained distribution, without relying on additional sampled data.

**Key Insight**: The authors attribute the problem to a "misalignment between the training objective and the true personalization goal" and theoretically quantify this misalignment. Since standard objectives drive the model distribution toward the adaptive distribution $p_{adapt}$, they propose placing a provable bound on the "amount of drift" in the parameter space.

**Core Idea**: Use the **Lipschitz continuity** of the diffusion network with respect to its parameters to provide an upper bound on the KL divergence of the output distribution. This bound is then relaxed into an L2 regularization term on the difference between current and pre-trained parameters—a simple one-line L2 constraint supported by theory that completely eliminates the need for pre-sampling.

## Method

### Overall Architecture
The method begins by proving "why existing objectives are wrong" and then provides a bounded alternative objective. The pipeline is minimalist: a pre-trained denoising network $\epsilon_{\theta_{base}}$ is copied as the trainable $\epsilon_{\theta_{per}}$, while the original network is frozen. During personalization, the denoising loss is calculated as usual using the target prompt. The **only modification** is adding a term $\lambda\sum_i\|\theta^i_{per}-\theta^i_{base}\|_2^2$ to the loss, penalizing the L2 distance of each parameter from its pre-trained value. The personalized model is obtained directly after training without any prior sampling data. Since the method is an algorithmic improvement rather than a multi-module pipeline, it is best explained via formulas.

The standard conditional denoising loss is:

$$\mathcal{L}_{denoise}=\mathbb{E}_{z,c,\epsilon,t}\big\|\epsilon-\epsilon_\theta(z_t,c,t)\big\|_2^2$$

DreamBooth’s traditional combined objective adds a prior preservation loss using class prompts (requiring pre-sampled $z'_t$):

$$\mathcal{L}_{total}=\mathbb{E}\big[\|\epsilon-\epsilon_\theta(z_t,c_{target},t)\|_2^2+\lambda_{prior}\|\epsilon-\epsilon_\theta(z'_t,c_{class},t)\|_2^2\big]$$

Ours replaces the second term with a penalty on parameter distance, requiring neither $z'_t$ nor class prompts.

### Key Designs

**1. Diagnosing "Objective-Goal Misalignment": Why Standard Objectives Fail to Preserve the Pre-trained Distribution**

This section explains why existing methods overfit. Let $p^*$ be the reference distribution and $p_{\theta_{base}}$ be the pre-trained model distribution, assuming they are sufficiently close. Suppose the quality $\gamma$ of the adaptive data distribution $p_{adapt}$ on a set $D$ is much larger than its quality $\delta$ in $p^*$ ($\gamma\gg\delta$, meaning personalized data is heavily biased toward a few samples). **Theorem 1** proves: when a model uses $\mathcal{L}_{denoise}$ on $p_{adapt}$ and converges such that $p_{\theta_t}\to p_{adapt}$, it necessarily follows that $D_{KL}(p^*\|p_{\theta_{base}})<D_{KL}(p^*\|p_{\theta_t})$. This implies fine-tuning inherently moves the model further from the original distribution (Remark 1). Crucially, **Corollary 1** states that even with prior preservation samples (Eq. 3), the mixture distribution $p'_{per}$ still suffers from few-shot imbalance ($M\ll N$), and Theorem 1 still holds (Remark 2). This theoretically exposes the limitation of Prior Preservation Loss: preserving the class $\neq$ preserving all priors.

**2. Lipschitz Regularization: Provable Bounds on Distribution Drift via Parameter Distance**

Since the issue lies in the objective, a bounded constraint is introduced at the objective level. **Theorem 2** provides the core inequality: if the denoising network $\epsilon_\theta$ is Lipschitz continuous with respect to parameters $\theta$, then for any two sets of parameters $\theta_1,\theta_2$, there exists a constant $\lambda>0$ such that:

$$D_{KL}(p_{\theta_1}\|p_{\theta_2})\le\lambda\cdot\|\theta_1-\theta_2\|^k$$

The proof logic follows a chain of Lipschitz property propagation: the composition of Lipschitz functions remains Lipschitz; the attention mechanism has a finite Lipschitz constant on compact input domains; hence $\epsilon_\theta$ is Lipschitz w.r.t. $\theta$. By Tweedie’s formula, the score $s_\theta=-\epsilon_\theta/\sigma_t$ remains Lipschitz (differing only by a scalar). The probability-flow ODE integrates the score into $\log p_\theta$, and since integration is a linear operator, it preserves the bound. Finally, the triangle inequality yields the expression above. **Remark 3** sets $k=2$, resulting in $\lambda\cdot\|\theta_1-\theta_2\|^2\le\lambda'\cdot\|\theta_1-\theta_2\|_2^2$. Thus, the Lipschitz constraint is relaxed into a standard L2 regularization. While L2 is common, **using it as a proxy for Lipschitz continuity to constrain distribution drift in personalization is novel**: it implies that by suppressing parameter movement, the KL divergence of the output distribution relative to the pre-training is also suppressed.

**3. Removing Pre-sampling + Single $\lambda$ Knob: Gains in Efficiency and Controllability**

Mapping "distribution preservation" to parameter distance offers two practical benefits. First, **no additional data is required**. Traditional prior preservation requires sampling 100–200 class images, which dominates training time. Ours looks only at the difference between $\theta_{per}$ and $\theta_{base}$, eliminating the pre-sampling bottleneck (Algorithm 1). On SDXL-CD, training time drops to less than 1/5 of the baseline. Second, **a single hyperparameter $\lambda$ quantifies the strength of distribution preservation**: a larger $\lambda$ restricts parameter movement more, staying closer to the pre-trained distribution (higher CLIP-T, stronger prompt following); a smaller $\lambda$ fits the target subject better (higher DINO/CLIP-I), turning the "adaptation-preservation" conflict into a continuously adjustable knob.

### Loss & Training
The final optimized loss during the personalization phase is (Algorithm 1, line 9):

$$\mathcal{L}=\|\epsilon-\epsilon_{\theta_{per}}(\tilde{z}_t,t,c)\|_2^2+\lambda\sum_i\|\theta^i_{per}-\theta^i_{base}\|_2^2$$

where $\tilde{z}_t=\sqrt{\bar\alpha_t}\,z+\sqrt{1-\bar\alpha_t}\,\epsilon$ is the noisy latent and $\theta_{base}$ is frozen throughout as an anchor. This regularization can be seamlessly added to various personalization strategies like DreamBooth, Custom Diffusion, or DreamBooth-LoRA as a plug-and-play replacement.

## Key Experimental Results

Evaluation uses the DreamBooth benchmark (30 subjects, up to 6 images each, 25 evaluation prompts, 3000 total generated images). Backbones include SD-1.5, SD-XL, and SD-3.0. Metrics are DINO, CLIP-I (subject fidelity), and CLIP-T (text alignment).

### Main Results
Gains across backbones after replacing baseline objectives with ours (selected from Table 1):

| Backbone | Method | DINO ↑ | CLIP-T ↑ | CLIP-I ↑ |
|------|------|--------|----------|----------|
| SD-1.5 | DB | 0.6028 | 0.2793 | 0.7881 |
| SD-1.5 | DB + Ours | 0.6394 (+0.0366) | 0.2976 (+0.0183) | 0.7948 (+0.0067) |
| SD-1.5 | CD + Ours | 0.5638 (+0.0070) | 0.3158 (+0.0004) | 0.7550 (+0.0011) |
| SD-XL | DB-LoRA + Ours | 0.6819 (+0.0257) | 0.3014 (−0.0085) | 0.8103 (+0.0133) |
| SD-3.0 | DB-LoRA + Ours | 0.6147 (+0.0104) | 0.3106 (+0.0008) | 0.7869 (+0.0046) |

The key takeaway is that our objective often **improves both metrics (fidelity + alignment) simultaneously**, or improves one without degrading the other—whereas these usually trade off. On SD-1.5 + DB, DINO increases by +0.0366 and CLIP-T by +0.0183, showing it preserves pre-trained knowledge while accelerating the learning of new concepts.

Comparison with various baselines (Table 2, combined ranking of DINO/CLIP-I/CLIP-T):

| Method | DINO ↑ | CLIP-T ↑ | CLIP-I ↑ | Rank |
|------|--------|----------|----------|------|
| Ours | 0.6394 | 0.2976 | 0.7948 | 1 |
| IP-Adapter | 0.6304 | 0.2635 | 0.8318 | 2 |
| DreamBooth | 0.6028 | 0.2793 | 0.7881 | 3 |
| OFT | 0.6320 | 0.2370 | 0.7850 | 4 |
| SVDiff | 0.3839 | 0.3194 | 0.6886 | 8 |

Ours ranks first overall; other baselines often excel in one metric while failing in another (e.g., SVDiff has the highest CLIP-T but the lowest DINO).

### Ablation Study

| Analysis | Phenomenon | Explanation |
|------|------|------|
| $\lambda$ scanning | $\lambda\uparrow \to$ CLIP-T$\uparrow$, DINO/CLIP-I$\downarrow$ | High $\lambda$ favors prompts/priors; low $\lambda$ favors subject fidelity. |
| Lipschitz validation | $\lambda\uparrow \to \Delta\theta$ and $\Delta\epsilon$ decrease | Output changes are bounded by parameter changes (supports Theorem 2). |
| Training efficiency | SDXL-CD time <1/5; others >2$\times$ speedup | Removing pre-sampling eliminates the main bottleneck. |

### Key Findings
- **$\lambda$ is a single knob for "adaptation-preservation"**: Large $\lambda$ ensures high preservation/low adaptation; small $\lambda$ ensures high adaptation but risks collapse or artifacts.
- **Lipschitz assumption holds in diffusion models**: Parameter drift $\Delta\theta$ and output drift $\Delta\epsilon$ scale proportionally with $\lambda$, validating the theoretical claim.
- **Efficiency gains are striking for parameter-efficient baselines**: As these methods train fast, pre-sampling overhead reflects a larger percentage of total time; removing it yields significant speedups.
- **2D toy experiments** confirm that standard objectives (Eq. 2) fail to preserve the original distribution, Prior Preservation (Eq. 3) only preserves the selected class, while our regularization maintains the overall structure of all classes.

## Highlights & Insights
- **Shifting "Distribution Preservation" from Architecture to Objective**: Theorem 1 proves existing objectives fail in principle, while Theorem 2 provides a provable bound. This closed-loop diagnosis and solution is more robust than empirical layer-freezing.
- **L2 backed by Lipschitz Theory**: The final implementation is a simple L2 on parameter offsets, but it is interpreted as a relaxation of Lipschitz constraints—giving "simple but principled" a new meaning.
- **Undervalued Engineering Benefit of Removing Pre-sampling**: Prior preservation's hidden cost is pre-sampling. Using parameter distance bypasses this, making it faster and independent of external data.
- **Quantifiable KL Drift via Single Hyperparameter**: $\lambda$ explicitly quantifies how much prior is retained, providing a clean control interface for personalization.

## Limitations & Future Work
- The authors acknowledge that the method **does not guarantee improvements across every single metric/prompt**, as different subjects and backbones require varying levels of adaptation.
- Regularization is **applied uniformly** across all parameters (constant $\lambda$), without distinguishing parameter importance. Ideally, parameters would be weighted based on importance (e.g., Fisher Information $F_i$ as in EWC), but $F_i$ requires pre-training data which is unavailable in standard personalization scenarios.
- Observation: The theory relies on strong assumptions like "universal approximation + convergence," and the mapping between theoretical bounds and practical $\lambda$ remains qualitative.

## Related Work & Insights
- **vs. DreamBooth / Prior Preservation Loss**: DreamBooth uses class prompts to anchor the distribution; this paper proves that only preserves class-level info and is slow. Ours uses L2 on parameters: no sampling, faster, and more comprehensive preservation.
- **vs. SVDiff / OFT**: These restrict updates to specific components (singular values or angles) to stabilize generation, limiting flexibility. Ours provides a continuous constraint at the objective level.
- **vs. IP-Adapter / BLIP-Diffusion**: These require massive data to train adapter networks; ours is a few-shot, lightweight, plug-and-play objective improvement.

## Rating
- Novelty: ⭐⭐⭐⭐ Attributes personalization overfitting to "objective misalignment" and uses Lipschitz theory for a provable bound.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three backbones, multiple baselines, and ablation studies, though lacking large-scale user studies in the main text.
- Writing Quality: ⭐⭐⭐⭐ High; clear transition from theory to motivation.
- Value: ⭐⭐⭐⭐ Plug-and-play, significantly faster, and offers single-knob control.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Directional Textual Inversion for Personalized Text-to-Image Generation](directional_textual_inversion_for_personalized_text-to-image_generation.md)
- [\[CVPR 2026\] DCoAR: Deep Concept Injection into Unified Autoregressive Models for Personalized Text-to-Image Generation](../../CVPR2026/image_generation/dcoar_deep_concept_injection_into_unified_autoregressive_models_for_personalized.md)
- [\[ICLR 2026\] Latent Diffusion Model without Variational Autoencoder](latent_diffusion_model_without_variational_autoencoder.md)
- [\[ICLR 2026\] Learning an Image Editing Model without Image Editing Pairs](learning_an_image_editing_model_without_image_editing_pairs.md)
- [\[ECCV 2024\] Removing Distributional Discrepancies in Captions Improves Image-Text Alignment](../../ECCV2024/image_generation/removing_distributional_discrepancies_in_captions_improves_image-text_alignment.md)

</div>

<!-- RELATED:END -->
