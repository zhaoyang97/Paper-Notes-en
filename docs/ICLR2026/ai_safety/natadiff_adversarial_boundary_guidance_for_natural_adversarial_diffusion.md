---
title: >-
  [Paper Note] NatADiff: Adversarial Boundary Guidance for Natural Adversarial Diffusion
description: >-
  [ICLR 2026][AI Safety][Diffusion Model] NatADiff utilizes diffusion models to guide sampling trajectories toward the "boundary between the true class and the adversarial class." Rather than producing constrained adversarial samples with perturbations, it generates "natural adversarial samples" that naturally blend adversarial semantic cues. This maintains wh
tags:
  - ICLR 2026
  - AI Safety
  - Diffusion Model
  - time-travel sampling
date: 2026-05-08
content_hash: e8e05b3683d0e2a3
---
# NatADiff: Adversarial Boundary Guidance for Natural Adversarial Diffusion

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=uWvLZqxjmx](https://openreview.net/forum?id=uWvLZqxjmx)  
**Code**: To be confirmed  
**Area**: AI Safety / Adversarial Attacks / Diffusion Generation  
**Keywords**: Natural adversarial samples, adversarial boundary guidance, diffusion models, attack transferability, time-travel sampling  

## TL;DR
NatADiff utilizes diffusion models to guide sampling trajectories toward the "boundary between the true class and the adversarial class." Rather than producing constrained adversarial samples with perturbations, it generates "natural adversarial samples" that naturally blend adversarial semantic cues. This maintains white-box attack success rates while significantly enhancing cross-architecture transferability, producing a distribution closer to real-world test-time errors.

## Background & Motivation
**Background**: Adversarial sample research has long been dominated by two paradigms: constrained attacks (PGD/AutoAttack, adding $\ell_p$ invisible perturbations to clean images) and unconstrained attacks (allowing arbitrary perturbation magnitude as long as the result stays near the natural image manifold). Recent generative attacks (GAN/Diffusion) attempt to "create" adversarial samples by directly injecting classifier gradients into the sampling process.

**Limitations of Prior Work**: (1) GAN-based methods are sensitive to sampling path perturbations, lack theoretical grounding, and often degrade image quality; (2) Directly injecting non-robust classifier gradients into diffusion trajectories (e.g., AdvDiff/AdvClass) essentially creates **constrained adversarial samples**—because these samples often lie within the $\epsilon$-neighborhood of natural images. When the diffusion model pulls the sample back to the manifold and the classifier gradient pushes it toward the nearest adversarial pocket, the resulting "tug-of-war" creates minute perturbations clinging to the manifold, leading to poor transferability. (3) Few studies have addressed "natural adversarial samples," the strongest category of unconstrained attacks.

**Key Challenge**: The real danger lies in **natural adversarial samples (test-time errors)**—legitimate inputs that are unperturbed and exist naturally but are misclassified (e.g., a shark on a beach being misclassified). These exhibit high transferability and bypass most adversarial defenses due to the lack of detectable perturbations. However, existing generative attacks produce samples with a distribution far from these real test-time errors.

**Goal**: To enable diffusion models to generate natural adversarial samples that are **highly transferable and distributionally close to real test-time errors**.

**Core Idea**: The paper observes a mechanism—**natural adversarial samples are highly transferable because different classifiers rely on the same "incorrect contextual cues" to take shortcuts for classification**. Thus, these structural cues from the adversarial class are actively injected into the image: **guiding the diffusion trajectory toward the intersection of the "true class $\cap$ adversarial class."** This ensures the image is perceived as the true class by humans while carrying enough adversarial features to trigger misclassification.

## Method

### Overall Architecture
NatADiff uses Stable Diffusion 1.5 as its backbone, performing guided sampling in latent space. The goal is to adjust the denoising direction at each step to "stay on the true class manifold while moving toward the adversarial class boundary." The pipeline consists of four cooperating components: estimating a clean $\hat{x}_0$ using the Tweedie formula for the classifier, "smoothing out" constrained perturbation gradients with differentiable image transformations, pulling the trajectory toward class boundaries using adversarial boundary guidance, and preserving image quality with time-travel sampling, finally supporting untargeted attacks via similarity targeting.

```mermaid
flowchart TD
    A[zT ~ N(0,I)] --> B[Tweedie Estimate x̂0]
    B --> C[Differentiable Transformation T<br/>Normalized Adversarial Gradient g]
    C --> D[Adversarial Boundary Guidance<br/>Combine vy and vy∩ỹ + Classifier Gradient]
    D --> E[Time-travel Sampling<br/>Repeated Forward/Backward Re-sampling]
    E --> F{argmax = ỹ?}
    F -- No --> G[Increase μ,s and Retry] --> D
    F -- Yes --> H[VAE Decode Output Adversarial Image]
```

### Key Designs

**1. Adversarial Boundary Guidance: Pulling trajectories toward class intersections.** This is the core of the paper. Standard adversarial classifier guidance (AdvClass) merely adds a victim classifier gradient $s\nabla_{x_t}\log p(\tilde{y}|x_t)$ to the classifier-free guidance, which only creates constrained samples. NatADiff introduces a new direction vector pointing to the "intersection." Let $v_y=\epsilon_{\theta^\star}(x_t,t,y)-\epsilon_{\theta^\star}(x_t,t)$ be the direction toward the true class $y$, and $v_{y\cap\tilde{y}}=\epsilon_{\theta^\star}(x_t,t,y\cap\tilde{y})-\epsilon_{\theta^\star}(x_t,t)$ be the direction toward the "intersection of the true and adversarial classes" (implemented by feeding the prompt `"<adversarial class> and <true class>"` to the diffusion model). The guided score becomes:

$$\nabla_{x_t}\log\bar{p}(x_t|y,\tilde{y}) = -\frac{1}{\beta(t)}\Big(\epsilon_{\theta^\star}(x_t,t) + (\omega-\mu\omega)v_y + \mu\rho\, v_{y\cap\tilde{y}}\Big) + s\nabla_{x_t}\log p(\tilde{y}|x_t).$$

The parameter $\mu\in[0,1]$ controls the intensity of moving "toward the intersection." When $\mu$ is sufficiently large, the trajectory approaches the class boundary, absorbing enough adversarial elements to cause misclassification while remaining the true class to human eyes. $\mu=0$ degrades to standard AdvClass. The intuition is: since $\bar{p}(x_t|y)$ is already an amplification of guidance on the learned manifold, the model should extract more information to paint structural cues of the adversarial class into the image rather than relying on fragile pixel perturbations.

**2. Reducing Adversarial Gradient: Smoothing perturbation shortcuts with image transformations.** Constrained adversarial attacks are fragile against rotations, cropping, and translations. NatADiff exploits this by applying a set of differentiable transformations $T=\{T_1,T_2,\dots\}$ to the current $\hat{x}_0$ estimate before calculating the adversarial gradient. This "averages out" local perturbation signals, forcing true adversarial semantic features to emerge. The normalized gradient is:

$$\nabla_{x_t}\log p(\tilde{y}|x_t) = g(x_t)/\|g(x_t)\|_2, \quad g(x_t)=\nabla_{x_t}\log\sigma_{\tilde{y}}\!\Big(\tfrac{1}{|T|}\textstyle\sum_{i=1}^{|T|}h(T_i(\hat{x}_0(x_t)))\Big),$$

where $h$ returns the victim classifier logits and $\sigma_{\tilde{y}}$ provides the target class probability. To address the issue that off-the-shelf classifiers are not trained on noisy samples, the Tweedie formula $\hat{x}_0(x_t)=(x_t-\beta(t)\epsilon_{\theta^\star}(x_t,t,y))/\alpha(t)$ is used to estimate a clean image first.

**3. Time-travel Sampling: Preserving quality to prevent manifold deviation.** Strong guidance can easily push samples off the image manifold, causing artifacts. NatADiff borrows from RePaint/FreeDoM, performing $R$ "forward-noise and backward-denoise" steps within selected time intervals, giving the diffusion model a chance to recover from poor trajectories. For computational efficiency, this is only enabled on a subset of steps. An adaptive search loop is used: if the decoded result is not classified as $\tilde{y}$, it retries with increased $\mu$ and $s$.

**4. Similarity Targeting: Extending the method to untargeted attacks.** Untargeted attacks are generally stronger but require dynamically choosing the "easiest" target class. NatADiff assumes that borrowing adversarial features from semantically similar classes is easier. It uses a CLIP text encoder $C_{enc}$ to map class names to a shared embedding space and selects the candidate with the highest cosine similarity to the true class:

$$\tilde{y} = \arg\max_{y\in Y_{cand}} \frac{C_{enc}(y_i)\cdot C_{enc}(y)}{\|C_{enc}(y_i)\|_2\,\|C_{enc}(y)\|_2}.$$

## Key Experimental Results

Setup: ImageNet 1000 classes, SD1.5 backbone + 200 DDIM steps, ~103 seconds per sample on one RTX 4090, 2000 samples per group. Victim models include CNNs (RN-50/Inc-v3/RN-152/adversarially trained AdvRes, AdvInc) and Transformers (ViT-H/Max-ViT/Swin-B/DeIT). Metrics: ASR (Attack Success Rate), IS, FID-Val (quality relative to ImageNet-Val), FID-A (similarity to real natural adversarial samples in ImageNet-A).

### Main Results (ASR %, Selected RN-50 and ViT-H surrogates)

| Surrogate | Attack | White-box* | Avg ASR | IS↑ | FID-Val↓ | FID-A↓ |
|-----------|--------|-----------|---------|-----|---------|--------|
| RN-50     | PGD    | 99.4      | 17.6    | -   | -       | -      |
| RN-50     | ACA    | 78.8      | 52.9    | 23.9| 65.0    | 77.9   |
| RN-50     | AdvClassᵁ | 99.9      | 45.7    | 38.5| 50.2    | 92.7   |
| RN-50     | **NatADiffᵀ** | 96.9      | 56.8    | 26.0| 66.5    | **77.3** |
| RN-50     | **NatADiffᵁ** | 99.3      | **68.2** | 43.2| 51.4    | 95.9   |
| ViT-H     | ACA    | 75.8      | 53.2    | 25.5| 64.2    | 80.9   |
| ViT-H     | AdvClassᵁ | 98.7      | 42.8    | 39.2| 48.5    | 98.8   |
| ViT-H     | **NatADiffᵀ** | 98.5      | **73.2** | 15.3| 88.0    | 93.5   |
| ViT-H     | **NatADiffᵁ** | 99.6      | 69.7    | 31.9| 53.9    | 96.2   |

(*White-box means the surrogate and victim models match; T = random target, U = similarity untargeted.) Core takeaway: NatADiff's white-box ASR is on par with SOTA, but its **average transfer ASR leads significantly** (68.2 on RN-50 vs ACA 52.9 / AdvClass 45.7), and adversarial training (AdvRes/AdvInc) offers almost no additional robustness against it.

### Ablation Study

| Component Removed | Consequence |
|-------------------|-------------|
| Remove $v_{y\cap\tilde{y}}$ ($\mu=0$, reverts to AdvClass) | Trajectory fails to move toward natural adversarial samples; transferability drops sharply. |
| Use raw gradient without image transformation | Reverts to constrained adversarial samples; visible adversarial features decrease. |
| Disable time-travel sampling | Degraded image quality / samples fall off the manifold. |

### Key Findings
- **Source of Transferability Verified**: NatADiff achieves high transferability by injecting classifier-independent structural adversarial cues (different surrogate models generate similar adversarial features) rather than relying on a single surrogate gradient. It is the only method that does not rely solely on surrogate classifier gradients.
- **Targeted vs. Untargeted Quality-Naturalness Trade-off**: Targeted NatADiff has lower FID-A (more like real natural adversarial samples) but worse IS/FID-Val. The untargeted version is the opposite. Reproducing the "mix" of features found in real test-time errors is challenging for the diffusion backbone and prone to artifacts.
- **ViT-H is the Hardest Target**: As the largest and most modern model, it learns more robust feature representations, resulting in the lowest transfer ASR. Targeted ViT-H attacks introduce artifacts that artificially inflate ASR, highlighting the value of similarity targeting in finding "model weaknesses."

## Highlights & Insights
- **Paradigm Shift**: Redefines adversarial attacks from "adding perturbations" to "semantic guidance along class boundaries," connecting generative attacks for the first time with Hendrycks' natural adversarial example phenomena (shortcut learning / contextual cues).
- **Clever $v_{y\cap\tilde{y}}$ design**: Requires no new models; uses the prompt `"A and B"` to let Stable Diffusion synthesize the class-intersection direction, painting adversarial semantics into the image rather than applying perturbations.
- **FID-A as an Evaluation Dimension**: Using the distance to real ImageNet-A to quantify how much a sample "looks like a real test-time error" provides more practical significance than ASR alone.

## Limitations & Future Work
- **High Cost**: ~103 seconds per sample (200-step DDIM + time-travel loops + external search loop), much slower than PGD-style single-forward-pass attacks, making large-scale deployment difficult.
- **Quality-Naturalness Trade-off**: Systematic trade-offs exist between targeted and untargeted configurations across IS, FID-Val, and FID-A; no single configuration is globally optimal.
- **Backbone Dependencies**: Generation quality is constrained by the SD1.5 manifold and CLIP similarity; classes outside the backbone's distribution may not produce credible samples.
- **Defensive Aspect**: While the paper notes that natural adversarial samples bypass common defenses, it does not systematically propose a closed-loop "how to use NatADiff for adversarial training" to improve robustness.

## Related Work & Insights
- **Hierarchy of Attacks** ($A_N\subseteq A_C\subseteq A_U$): Based on Szegedy, Song, and Hendrycks, this work operates at the strongest $A_N$ end.
- **Generative Attack Genealogy**: Evolves from GAN attacks → AdvDiff/AdvClass (Dai 2024, direct gradient injection) → ACA (Chen 2023b, latent perturbation with source image constraints) → DiffAttack. NatADiff differentiates itself through "source-free, free synthesis + class-boundary guidance."
- **Mechanistic Foundation**: Shortcut learning (Geirhos 2020) and invariant risk minimization (Arjovsky 2020) explain why multiple classifiers share the same incorrect cues, leading to high transferability.
- **Technical Components**: Tweedie formula (Efron 2011), time-travel/universal guidance (RePaint, FreeDoM, Bansal 2024), CLIP (Radford 2021).
- **Insight**: Shifting "adversarialism" from pixel space to "semantic/class boundary space" is valuable for robustness diagnostics, data augmentation, and creating test sets closer to real-world failure modes.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Connects "class boundary semantic guidance" with the natural adversarial example mechanism; the $v_{y\cap\tilde{y}}$ design is elegant and theoretically sound.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 9 victim models across CNNs/Transformers, 6 SOTA baselines, and multi-dimensional metrics (ASR+IS+FID-Val+FID-A); lacks only closed-loop defense verification and testing on larger backbones.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear definitions, solid derivation, and intuitive figures comparing different attack morphologies.
- **Value**: ⭐⭐⭐⭐ — Provides a viable solution for generating "highly transferable and realistic test-time error" samples; high computational cost is the main barrier.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GROW: Watermark Generation with Progressive Guidance for Diffusion Models](../../CVPR2026/ai_safety/grow_watermark_generation_with_progressive_guidance_for_diffusion_models.md)
- [\[ICLR 2026\] On the Interaction of Compressibility and Adversarial Robustness](on_the_interaction_of_compressibility_and_adversarial_robustness.md)
- [\[ICLR 2026\] SCOPED: Score–Curvature Out-of-Distribution Proximity Evaluator for Diffusion](scoped_scorecurvature_out-of-distribution_proximity_evaluator_for_diffusion.md)
- [\[ICLR 2026\] Concept-based Adversarial Attack: a Probabilistic Perspective](concept-based_adversarial_attack_a_probabilistic_perspective.md)
- [\[ICLR 2026\] FERD: Fairness-Enhanced Data-Free Adversarial Robustness Distillation](ferd_fairness-enhanced_data-free_adversarial_robustness_distillation.md)

</div>

<!-- RELATED:END -->
