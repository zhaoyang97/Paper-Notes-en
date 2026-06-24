---
title: >-
  [Paper Note] Black-box Membership Inference Attacks on the Pre-training Data of Image-generation Models
description: >-
  [CVPR 2026][Image Generation][Membership Inference Attack] Focusing on closed-source text-to-image diffusion models, this paper proposes **SD-MIA**: instead of traditional methods that add noise to images and check denoising capabilities, it **perturbs text instructions** and monitors the stability of reconstructed images to determine whether an image was in the **pre-training data**. Under pure black-box constraints (text-in, image-out only)…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Membership Inference Attack"
  - "Pre-training Data Auditing"
  - "Cross-modal Perturbation"
  - "Black-box Attack"
  - "Diffusion Models"
date: 2026-05-08
content_hash: e987ddc13be65f5b
---

# Black-box Membership Inference Attacks on the Pre-training Data of Image-generation Models

**Conference**: CVPR 2026  
**arXiv**: [2605.27020](https://arxiv.org/abs/2605.27020)  
**Code**: https://github.com/wanghl21/SD-MIA (Available)  
**Area**: Diffusion Models / Membership Inference / Data Privacy  
**Keywords**: Membership Inference Attack, Pre-training Data Auditing, Cross-modal Perturbation, Black-box Attack, Diffusion Models

## TL;DR
Focusing on closed-source text-to-image diffusion models, this paper proposes **SD-MIA**: instead of traditional methods that add noise to images and check denoising capabilities, it **perturbs text instructions** and monitors the stability of reconstructed images to determine whether an image was in the **pre-training data**. Under pure black-box constraints (text-in, image-out only), it achieves an AUC up to 10 points higher than gray-box baselines that access internal features.

## Background & Motivation

**Background**: Copyright and privacy disputes surrounding diffusion models have sparked numerous Membership Inference Attacks (MIA), aiming to determine if a "copyrighted image" was used to train a generative model. The mainstream paradigm is uniform: add random noise to a suspect image, have the model reconstruct it, and use "reconstruction quality" as a signal—members are seen before and reconstruct accurately, while non-members yield blurry results.

**Limitations of Prior Work**: Existing evaluations are almost entirely built on **fine-tuned data**—where an open-source model is fine-tuned on a small, randomly split dataset, leading to severe overfitting (strong memory) and high attack accuracy. However, in real-world deployments, most training data is ingested during the **large-scale pre-training** phase, where the model's memory of a single image is far weaker. When "image-space noise" methods are applied to pre-training data, the detection signal decays sharply, nearly degrading to random chance.

**Key Challenge**: Why does image perturbation fail in pre-training scenarios? This paper provides a structural attribution—modern diffusion pipelines contain two "signal killers": ① The VAE encoder is **locally contractive** ($\|J_{f_v}(x)\|_2 \ll 1$), compressing fine-grained image perturbations $\delta x$ into the latent space until they nearly vanish; ② subsequent **random denoising trajectories** completely drown out residual micro-perturbations. Consequently, both members and non-members exhibit similarly stable reconstructions under image perturbation, erasing the signal. Even attempts to fix this using intermediate noise prediction (gray-box) are unfeasible since commercial APIs only provide final outputs.

**Goal**: Under "text-in, image-out" pure black-box constraints, find a membership signal that remains discriminative for **pre-training data**.

**Key Insight**: The authors observe that text and images follow **entirely different computational paths** in diffusion pipelines—text embeddings act as conditions that are **never added with noise**, stably guiding the entire denoising trajectory. During training, the model internalizes a **locally overfitted** "text-to-visual" mapping for pre-training samples, forming what is called **representation-region collapse**: a cluster of semantically similar text variants are funneled into the same visual pattern.

**Core Idea**: Shift the perturbation from the image to the text. For **member** images, small text perturbations still fall within the collapse region, resulting in stable reconstructions close to the original image. For **non-member** images, no collapse region exists, and text perturbations push the condition to different areas of the representation space, producing significantly divergent outputs. This **structural asymmetry** serves as a reliable membership signal—replacing "denoising capability after image perturbation" with "reconstruction consistency after text perturbation."

## Method

### Overall Architecture

SD-MIA addresses the problem: given a suspect image $x$ and its text description $c$, determine if $(x,c)$ was seen during pre-training under black-box constraints. The process is a **"Perturb Text → Reconstruct → Measure Consistency → Pooling into Score"** pipeline. First, an LLM generates three granularities of perturbations for text $c$. Each perturbed description is fed back into the diffusion model for repeated sampling to reconstruct $\hat{x}$. CLIP then measures the cross-modal correlation between $x$ and $\hat{x}$ as a **proxy signal** for unobservable generation probability. Finally, top-$K\%$ maximum correlation pooling is applied across multiple random reconstructions and subtracted from the no-perturbation baseline to obtain a membership score approximating the "text-perturbation-induced probability curvature change $\delta_c p$." A score closer to 0 (stable) indicates a member; a larger score indicates a non-member.

```mermaid
graph TD
    A["Suspect Sample<br/>Image x + Text c"] --> B["Cross-modal Perturbation Insight<br/>Perturb Text instead of Image"]
    B --> C["Multi-perspective Text Perturbation<br/>Token/Style/Semantic tiers"]
    C -->|Query model 10 times per description| D["Black-box T2I<br/>Reconstruct x̂"]
    D --> E["Max Cross-modal Correlation Estimation<br/>CLIP Correlation + top-K% Pooling"]
    E -->|sf = sf(x,ĉ) - sf(x,c)| F["Membership Determination<br/>Stable→Member / Divergent→Non-member"]
```

### Key Designs

**1. Cross-modal Perturbation Insight: Shifting the signal from image space to text space**

This is the foundation of the paper, targeting the failure of image perturbations on pre-training data. The authors use a first-order expansion to write the probability change induced by image perturbation as $\delta_x p \approx |\nabla_{\mathbf{z}} p(\mathbf{z},\mathbf{c};\theta^*)\cdot\delta\mathbf{z}|$, then substitute the local contractivity of the VAE $\|\delta\mathbf{z}\|_2 \lesssim \|J_{f_v}(x)\|_2\,\|\delta x\|_2$. Since $\|J_{f_v}(x)\|_2 \ll 1$, for both members and non-members, $|\delta_x p(x_m)-\delta_x p(x_n)| \approx \xi\cdot\delta x \to 0$, crushing the signal. Conversely, on the text side, embeddings are never noisy, and perturbation $\delta\mathbf{c}$ acts directly on the condition: $\delta_c p \approx |\nabla_{\mathbf{c}} p(\mathbf{z},\mathbf{c};\theta^*)\cdot\delta\mathbf{c}|$. Under representation-region collapse, the gradient for member pairs $\|\nabla_{\mathbf{c}} p(\mathbf{z}_m,\mathbf{c}_m;\theta^*)\|_2 \approx 0$, while non-member pairs do not satisfy this, thus:

$$|\delta_c p(x_m)-\delta_c p(x_n)| \approx \|\nabla_{\mathbf{c}} p(\mathbf{z}_n,\mathbf{c}_n;\theta^*)\cdot\delta\mathbf{c}_n\|_2 \gg 0$$

Empirical tests on SD v1.5 confirm that member/non-member distributions overlap under image perturbation but are clearly separable under text perturbation.

**2. Multi-perspective Text Perturbation: Creating controlled embedding shifts in a black-box**

Since the black-box cannot access internal embeddings to manipulate $\mathbf{c}$ directly, the authors use an LLM (GPT-5) for natural language rewriting as an **indirect but structured** embedding perturbation. They design three tiers spanning a displacement spectrum: **token perspective** (lexical/syntactic rewriting), **style perspective** (changing style, description density, or narrative framework), and **semantic perspective** (controlled attribute changes, e.g., replacing objects). To prevent excessive drift, each perturbed description $\hat{c}$ must satisfy $\mathrm{sim}(f_e(c), f_e(\hat{c})) \ge \tau$, with thresholds $\tau_t=0.9$, $\tau_s=0.8$, and $\tau_c=0.6$ respectively.

**3. Maximum Cross-modal Correlation Estimation: Proxying unobservable probability with CLIP and suppressing randomness via max pooling**

Under black-box constraints, $p(x|c;\theta^*)$ is unqueryable, and diffusion sampling is stochastic. The authors solve this in two steps. First, **cross-modal correlation as a proxy**: for each $(x,\hat{x})$, a captioning model (BLIP2) generates descriptions $d_x, d_{\hat{x}}$, and the correlation is calculated as:

$$s(x,\hat{c}) = \big(h_v(x)\oplus h_t(d_x)\big)\cdot\big(h_v(\hat{x})\oplus h_t(d_{\hat{x}})\big)$$

Second, utilizing **replicability asymmetry**, they use **max correlation pooling**: non-members almost never replicate $x$ under resampling, while members have a non-negligible probability. Thus, they take the top-$K\%$ average of $N$ random reconstructions: $s^t = \frac{1}{n}\sum_{j=1}^{n} s(x, \hat{c}^t_{R_j}),\ n=\lfloor N\cdot K\%\rfloor$. The final score is $s_f = s_f(x,\hat{c}) - s_f(x,c)$.

### Loss & Training
SD-MIA is a **training-free / fine-tuning-free** inference framework. Key settings: CLIP ViT-L/14 for embeddings; BLIP2-opt-6.7b for proxy descriptions; GPT-5 for text perturbations; thresholds $\tau_t=0.9, \tau_s=0.8, \tau_c=0.6$; 10 samples per description; results averaged over 5 random seeds.

## Key Experimental Results

### Main Results

Evaluated on **LAION-mi** (aligned member/non-member distribution) and **FlickrMIA-25** (newly released images as non-members). Metrics are AUC (%) and TPR@5% FPR. Results from Table 1 (Balanced 1:1, AUC):

| Method | Access | SD v1.2 | SD v1.4 | SD v1.5 | SD v3.5 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Loss | Black-box | 51.59 | 52.91 | 53.75 | 42.10 |
| PIA | Gray-box | 52.66 | 49.52 | 48.16 | 50.62 |
| CLiD | Gray-box | 49.26 | 53.71 | 51.88 | 58.15 |
| DRC (Strongest Gray-box) | Gray-box | 54.66 | 55.83 | 54.61 | 60.44 |
| Reconstruction | Black-box | 59.66 | 60.99 | 60.30 | 46.74 |
| **SD-MIA** | **Black-box** | **66.28** | **66.23** | **65.92** | **66.93** |

Ours leads across the board, even outperforming the strongest gray-box DRC by up to 10 points. On SD v3.5, where others drop to near random chance (~50%), SD-MIA holds steady at 66.93%.

### Ablation Study

| Configuration | Conclusion | Note |
| :--- | :--- | :--- |
| Full (token+style+semantic) | Optimal | Complimentary perspectives ensure stability |
| Token tier only | Good on some models | Sensitive to micro-memory |
| Style / Semantic only | Positive contribution | Weaker than combined tiers |
| Using paired description | Slightly better | Better signal with ground truth |
| Using BLIP proxy | Effective | Still outperforms DRC without original text |

### Key Findings
- **Tri-perspective necessity**: While token-level is effective on some models, only the combined tiers ensure stability across different model architectures.
- **Set-level auditing reaches 95%+**: Performance increases monotonically with set size $L$. At $L=30$, AUC exceeds 95% due to accumulated consistency signals.
- **Effective on closed-source APIs**: SD-MIA outperforms SOTA black-box baselines on DALL·E-3, Gemini-2.0, and GPT-4o.
- **Robustness to image distortion**: SD-MIA is far more stable than Reconstruction under Gaussian blur, noise, or cropping.

## Highlights & Insights
- **Perspective shift to "perturbing the other modality"**: While others focus on image denoising, this paper proves image perturbations are stifled by VAE contraction and random paths, shifting focus to the **noise-free text condition**.
- **Structural properties**: Effectively utilizes representation-region collapse and replicability asymmetry to design max-pooling, amplifying tail signals of member replication.
- **LLM as proxy for embedding manipulation**: A clever trick to achieve controlled $\|\delta\mathbf{c}\|$ in a black-box environment via natural language rewriting and CLIP thresholds.
- **Fair evaluation protocol**: Critiques older protocols (like using MS-COCO as non-members for LAION members) and emphasizes domain-aligned, time-disjoint testing.

## Limitations & Future Work
- **Dependency on external models**: Relies heavily on GPT-5, BLIP2, and CLIP. Biases or inaccuracies in these models could undermine the proxy signal $s$.
- **Query cost**: Multiple perturbations and samples per instruction could be expensive on commercial APIs.
- **Absolute AUC remains modest**: Instance-level AUC ~66% is still far from high-confidence industrial auditing; high reliability currently requires set-level aggregation ($L=30$).

## Related Work & Insights
- **vs. Image-space Black-box (Reconstruction/Loss)**: SD-MIA rescues signals that these methods lose due to VAE contraction, especially on weakly-memorized pre-training data.
- **vs. Gray-box (DRC/PIA/CLiD)**: These require internal features unavailable in commercial APIs. SD-MIA proves cross-modal consistency is more general and powerful.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Enhancing Membership Inference Attacks on Diffusion Models from a Frequency-Domain Perspective](../../ICML2026/image_generation/enhancing_membership_inference_attacks_on_diffusion_models_from_a_frequency-doma.md)
- [\[CVPR 2026\] CSF: Black-box Fingerprinting via Compositional Semantics for Text-to-Image Models](csf_black-box_fingerprinting_via_compositional_semantics_for_text-to-image_model.md)
- [\[CVPR 2026\] Rethinking UMM Visual Generation: Masked Modeling for Efficient Image-Only Pre-training](rethinking_umm_visual_generation_masked_modeling_for_efficient_image-only_pre-tr.md)
- [\[CVPR 2026\] BlackMirror: Black-Box Backdoor Detection for Text-to-Image Models via Instruction-Response Deviation](blackmirror_black-box_backdoor_detection_for_text-to-image_models_via_instructio.md)
- [\[NeurIPS 2025\] Transferable Black-Box One-Shot Forging of Watermarks via Image Preference Models](../../NeurIPS2025/image_generation/transferable_black-box_one-shot_forging_of_watermarks_via_image_preference_model.md)

</div>

<!-- RELATED:END -->
