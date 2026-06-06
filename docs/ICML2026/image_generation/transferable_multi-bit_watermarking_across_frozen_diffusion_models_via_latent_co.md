---
title: >-
  [Paper Note] Transferable Multi-Bit Watermarking Across Frozen Diffusion Models via Latent Consistency Bridges
description: >-
  [ICML2026][Image Generation][Multi-bit watermarking] DiffMark continuously injects a learned latent space perturbation $\delta$ into each denoising step of a frozen diffusion model…
tags:
  - "ICML2026"
  - "Image Generation"
  - "Multi-bit watermarking"
  - "Frozen diffusion models"
  - "Latent space perturbation"
  - "LCM differentiable bridge"
  - "Cross-model transferable"
date: 2026-05-08
content_hash: bfe3635f94d81984
---

# Transferable Multi-Bit Watermarking Across Frozen Diffusion Models via Latent Consistency Bridges

**Conference**: ICML2026  
**arXiv**: [2603.20304](https://arxiv.org/abs/2603.20304)  
**Code**: To be confirmed  
**Area**: AI Security / Diffusion Model Watermarking  
**Keywords**: Multi-bit watermarking, Frozen diffusion models, Latent space perturbation, LCM differentiable bridge, Cross-model transferable

## TL;DR
DiffMark continuously injects a learned latent space perturbation $\delta$ into each denoising step of a frozen diffusion model, allowing the watermark signal to accumulate on the final state latent variable $z_0$. By utilizing a Latent Consistency Model (LCM) as a differentiable training bridge to bypass the backpropagation of 50-step DDIM, the method achieves 64-bit extraction in 16.4 ms via a single forward pass, providing a plug-and-play solution across models without retraining.

## Background & Motivation
**Background**: Currently, mainstream diffusion model watermarking follows two paths: sampling-based (Tree-Ring / RingID / Shallow Diffuse), which embeds watermarks in the initial noise $z_T$ or intermediate latents and relies on 50-step DDIM inversion for detection; and fine-tuning-based (Stable Signature / AquaLoRA), which binds watermarks to model weights by fine-tuning the UNet or using LoRA, followed by a lightweight decoder for one-time multi-bit extraction.

**Limitations of Prior Work**: The inversion detection in sampling-based methods requires running the UNet for $N=50$ steps per image, which is unacceptable for platform-level throughput. Most of these only support 0-bit (presence/absence) and cannot perform user attribution. Furthermore, noise patterns must be regenerated for each unique key. Although fine-tuning-based methods support multi-bit single-pass decoding, the watermark is strictly tied to a specific checkpoint, requiring retraining for every new SD variant—making unified governance impossible in the open-source ecosystem.

**Key Challenge**: Regulators require a "cross-model, attributable, and platform-scale verifiable" watermarking infrastructure. Existing methods compromise between "latency/bits" and "model transferability." The root cause lies in anchoring the watermark to $z_T$ (forcing inversion) or UNet weights (forcing retraining).

**Goal**: (i) Extract $L$ bits in a single forward pass at the detector, avoiding $N$-step inversion; (ii) Ensure the watermark is transparent to the frozen UNet, allowing a single set of encoder-decoder weights to work across the SD family; (iii) Enable arbitrary keys for each image without retraining.

**Key Insight**: The authors observe that since cumulative perturbations are amplified along the denoising trajectory and ultimately reside in $z_0$, the watermark does not need to be hidden in $z_T$. Instead, it can be injected as a "constant additive perturbation $\delta$" before each denoising step. Consequently, the decoder only needs to examine $z_0$, making inversion unnecessary. Since $\delta$ depends only on a lightweight encoder $E_\phi(s)$, it is decoupled from UNet weights, inherently supporting cross-model use and per-image keys.

**Core Idea**: The watermark is redefined from a "pattern in noise" or a "fingerprint in weights" to a "persistent latent perturbation $\delta$ added at every step." An LCM is then used to compress the 50-step DDIM into a 4-step differentiable path to backpropagate gradients for $\delta$, enabling end-to-end learning of an encoder-decoder pair while the UNet remains completely frozen.

## Method

### Overall Architecture
The inference side of DiffMark is straightforward: given an $L=64$ bit key $s$, a lightweight encoder $E_\phi$ maps it to a latent perturbation $\delta = E_\phi(s) \in \mathbb{R}^{4 \times h \times w}$. Standard 50-step DDIM sampling proceeds as usual, with the only modification being the addition of $\delta$ to the current latent before each UNet call: $\tilde z_{t_k} = z_{t_k} + \delta$. The $\tilde z_{t_k}$ is then fed into the frozen $\epsilon_\theta$ to predict noise for the DDIM update. The final $z_0$ is decoded into an image. During detection, the image is passed through a VAE encoder to retrieve $z_0$, which is fed into a lightweight decoder $D_\psi$ for a single forward pass to obtain $\hat s = D_\psi(z_0) \in \mathbb{R}^{L \times 2}$, followed by a bit-wise argmax to recover the key.

The training side employs two parallel paths: a "short and differentiable" LCM path ($K=4$ steps) to backpropagate encoder gradients, and a "long and high-fidelity" DDIM path ($N=50$ steps, with stop-gradient on $\delta$ and per-step injection scaled by $1/N$ to match cumulative perturbation) to provide the decoder with the $z_0^{ddim}$ seen during real inference. The UNet in the LCM path only allows gradients to pass through without updating weights; thus, the resulting $(E_\phi, D_\psi)$ can be used with any SD-family model at zero cost.

### Key Designs

1.  **Persistent Delta Injection**:
    - **Function**: Encodes an $L$-bit key into an additive latent space perturbation $\delta$, which is added to the current latent before each DDIM denoising step, allowing the watermark signal to accumulate into $z_0$ along the sampling trajectory.
    - **Mechanism**: Replaces the original latent with $\tilde z_{t_k} = z_{t_k} + \delta$ before feeding it to the UNet (including CFG), followed by standard DDIM updates: $z_{t_{k+1}} = \sqrt{\bar\alpha_{t_{k+1}}}\frac{\tilde z_{t_k} - \sqrt{1-\bar\alpha_{t_k}}\hat\epsilon_{t_k}}{\sqrt{\bar\alpha_{t_k}}} + \sqrt{1-\bar\alpha_{t_{k+1}}}\hat\epsilon_{t_k}$. To ensure $\tilde z_{t_k}$ stays within the UNet's training distribution, a magnitude loss $\mathcal{L}_{mag} = (\sigma(\delta) - \sigma_{target})^2$ and KL divergence $\mathcal{L}_{KL}$ are applied to pull the encoder's variational output toward a standard Gaussian, forcing $\|\delta\| \ll \|z_T\|$.
    - **Design Motivation**: Completely bypasses DDIM inversion by anchoring the watermark to $z_0$ rather than $z_T$, requiring only one forward pass for decoding. Since $\delta$ depends on the key rather than image content, it naturally supports per-image keys.

2.  **Dual-Path with LCM Bridge**:
    - **Function**: Provides backpropagatable gradient signals for the encoder under the constraint of a frozen UNet, while supervising the decoder on the actual 50-step DDIM output.
    - **Mechanism**: The LCM path uses $K=4$ LCM forward steps to push the perturbation $\delta$ to $z_0^{lcm}$. The backward chain $\mathcal{L}_{lcm} \to D_\psi \to z_0^{lcm} \to 4\,\text{LCM steps} \to \delta \to E_\phi$ is fully differentiable, with gradients passing through but not modifying the UNet. The parallel DDIM path uses $N=50$ standard steps to obtain high-fidelity $z_0^{ddim}$, applying stop-gradient to $\delta$ and scaling per-step injection to $\delta/N$ to match the LCM path's accumulation. The loss $\mathcal{L}_{ddim} = \mathcal{L}_{CE}(D_\psi(z_0^{ddim}), s)$ updates only the decoder.
    - **Design Motivation**: The computational graph of 50-step DDIM is prohibitive in terms of memory and gradient stability. The 4-step LCM distillation serves as a "short differentiable approximation." However, since LCM fidelity is lower than DDIM, relying solely on LCM would cause the decoder to fail on the inference distribution. The dual-path approach decouples the two goals: "encoder learning where to place" and "decoder learning how to read."

3.  **Multi-Stage Curriculum**:
    - **Function**: Addresses the conflict between reconstruction (requiring large $\|\delta\|$) and imperceptibility (requiring $\|\delta\| \to 0$) during joint optimization.
    - **Mechanism**: Groups losses into a reconstruction set $\mathcal{G}_{rec} = \{\mathcal{L}_{lcm}, \mathcal{L}_{ddim}\}$ and an imperceptibility set $\mathcal{G}_{imp} = \{\mathcal{L}_{lafid}, \mathcal{L}_{prvl}, \mathcal{L}_{freq}, \mathcal{L}_{neg}\}$. Each loss is assigned an activation gate $g_i(t) = \mathbb{1}[t \geq \tau_i]$, with the schedule satisfying $\max_{i \in \mathcal{G}_{rec}} \tau_i \leq \min_{j \in \mathcal{G}_{imp}} \tau_j$. Total loss is $\mathcal{L}(t) = \sum_i g_i(t) \cdot w_i(t) \cdot \mathcal{L}_i$. Here, $\mathcal{L}_{lafid}$ constrains $z_0$ displacement, $\mathcal{L}_{prvl}$ spreads watermark energy, $\mathcal{L}_{freq}$ pushes perturbations to high frequencies, and $\mathcal{L}_{neg}$ applies negative entropy to non-watermarked images to force uniform decoder output, suppressing false positives.
    - **Design Motivation**: If both loss groups are enabled simultaneously, imperceptibility terms would push $\delta$ to 0 before the watermark signal is established. Strictly establishing a decodable watermark before refining imperceptibility is the only stable path for such adversarial objectives.

### Loss & Training
In the pre-training phase, the encoder-decoder is jointly trained for 50,000 steps independent of the DM using $\mathcal{L}_{CE}$ (per-bit cross-entropy) + orthogonality loss $\mathcal{L}_{orth} = \frac{1}{B(B-1)}\sum_{i \neq j} \frac{\langle\delta_i, \delta_j\rangle_F}{\|\delta_i\|_F \|\delta_j\|_F}$ (batch 64, AdamW, encoder lr $3 \times 10^{-4}$, decoder lr $1 \times 10^{-4}$), ensuring unique keys map to orthogonal perturbations robust to $\delta + \epsilon$ noise. This is followed by 10,000 steps of joint fine-tuning with SD v1.5 + LCM_Dreamshaper_v7 ($K=4$) (batch 16, encoder lr $5 \times 10^{-5}$, decoder lr $3 \times 10^{-4}$), utilizing curriculum gating for $\mathcal{G}_{rec}$ and $\mathcal{G}_{imp}$. Inference uses standard 50-step DDIM with zero extra overhead compared to the base model.

## Key Experimental Results

### Main Results
Comparison with 6 baselines (StegaStamp / Stable Signature / AquaLoRA / Tree-Ring / RingID / Shallow Diffuse) on DiffusionDB across detection accuracy, consistency, and quality:

| Method | Type | Plug&Play | bits | Bit Acc | TPR@0.1%FPR | PSNR↑ | FID↓ | CLIP-FID↓ |
|------|------|-----------|------|---------|-------------|-------|------|-----------|
| StegaStamp | Post-proc | ✗ | 100 | 0.9994 | 1.0 | 11.34 | 54.82 | 10.26 |
| Stable Signature | FT | ✗ | 48 | 0.9950 | 0.9900 | 16.23 | 46.81 | 4.61 |
| AquaLoRA | FT | ✗ | 48 | 0.9355 | 0.9910 | 20.59 | 32.32 | 1.98 |
| Tree-Ring | Sampling | ✓ | 0 | — | 1.0 | 11.02 | 47.09 | 4.65 |
| RingID | Sampling | ✓ | 11 | — | 1.0 | 10.74 | 47.18 | 4.77 |
| Shallow Diffuse | Sampling | ✓ | 0 | — | 1.0 | 11.01 | 43.37 | 4.10 |
| **DiffMark** | Sampling | ✓ | **64** | 0.9381 | 1.0 | 11.01 | **38.07** | **2.20** |

**Cross-model Transferability**: Trained only on SD 1.5, zero-shot testing on SD-2.1 / DreamShaper 8 / Realistic Vision 5.1 / OpenJourney v4 showed stable bit accuracy between 93.3–95.5%, proving generalizability across the SD family. **Detection Latency**: DiffMark takes 16.4 ms/img on an L40S GPU, compared to 754.9 ms for Tree-Ring and 753.2 ms for RingID, a $45\times$ speedup. **Ownership Attribution**: 64-bit capacity achieves 100% Top-1 accuracy for $10^6$ users and $\geq 99.97\%$ for $10^8$ users.

### Ablation Study

| Config | Key Finding | Description |
|------|---------|------|
| K=2 / 4 / 8 LCM Steps | K=4 for main results; K=2 is faster with similar accuracy. | K=2 is likely the optimal default; increasing K provides no extra benefit. |
| L=48 / 64 / 128 / 256 bits | Training collapses at L=128; $\mathcal{L}_{orth}$ fails diversity. | L=64 is the sweet spot; L=256 causes LPIPS to spike to 0.508. |
| Random vs Fixed key (DiffMark) | BER distributions are nearly identical. | Generalizes to $2^{64}$ key space without local interpolation. |
| Random vs Fixed key (AquaLoRA) | BER jumps from 6.42% to 28.16%. | Highlights overfitting of fine-tuning methods to training keys. |
| Robustness (13 Attacks) | Avg 0.70 TPR@0.1%FPR; 1.00 for brightness/JPEG. | Fails (0.00) on rotation/cropping/Adv-KLVAE8 due to latent corruption. |

### Key Findings
- The LCM path acts as a "gradient bridge" rather than a "sampler." Ablation shows $K=2$ is sufficient, and more steps only slow down training.
- Curriculum training, specifically delaying imperceptibility losses until after decodability is achieved, is critical for stability.
- Latent-space watermarking systematically fails against geometric transformations (rotation/cropping) and gray-box adversarial attacks on VAE, which is a structural limitation of this paradigm.
- A 64-bit capacity maintains 99.97% Top-1 attribution at a $10^8$ user scale, a critical threshold for usable governance infrastructure.

## Highlights & Insights
- Anchoring the watermark to $z_0$ instead of $z_T$ solves latency, key flexibility, and cross-model transferability simultaneously. This proves that hiding watermarks in initial noise is not a necessary assumption.
- The dual-path paradigm (LCM for "differentiable short path" + DDIM for "high-fidelity inference path") is a general template for training small modules within frozen large models.
- The "decode first, hide later" curriculum reveals a rule: when joint-training adversarial objectives, a stable attractor for the "hard goal" must be established before introducing "soft goals."
- Section 5 maps technical capabilities directly to regulatory requirements (EU AI Act / SB-53), showing how single-pass 64-bit decoding translates to "user attribution" and "auditability."

## Limitations & Future Work
- **Limitations**: Structural failure under rotation, blur, and random cropping as these disrupt the latent representation $z_0 = \mathcal{E}(x) \cdot f_s$.
- **Further Constraints**: (i) Only tested on SD 1.x/2.x; architecture differences in SDXL/Flux/SD3 remain unverified. (ii) Efficacy of "same $\delta$ at every step" on ultra-short trajectories (1-4 steps) is unknown. (iii) Fixed 64-bit key space requires retraining for variable lengths.
- **Future Work**: Injecting $\delta$ using joint space-frequency transforms for geometric invariance; utilizing time-dependent $\delta_t = E_\phi(s, t)$; and training a robust VAE adapter to defend against gray-box attacks.

## Related Work & Insights
- **vs Sampling-based (Tree-Ring, etc.)**: These write to $z_T$, requiring 50-step inversion. DiffMark writes to $\delta$ injected at each step, requiring only one VAE + one decoder forward pass, achieving $45\times$ acceleration and multi-bit capacity.
- **vs Fine-tuning-based (Stable Signature, etc.)**: These modify weights, binding them to specific checkpoints. DiffMark is cross-model (93.3% accuracy on unseen models) and avoids the key-overfitting seen in methods like AquaLoRA.
- **vs StegaStamp (Post-generation)**: StegaStamp processes pixels after generation, harming quality (PSNR 11.34). DiffMark embeds during generation, yielding superior quality metrics.
- **Insight**: Using LCM as a "differentiable gradient bridge" can be extended to any task requiring end-to-end learning of lightweight condition modules (e.g., conditional encoders or reward heads) on frozen diffusion models while avoiding memory explosion.

## Rating
- Novelty: ⭐⭐⭐⭐ (Combining $z_0$ anchoring with LCM bridges is a clean, effective shift.)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Covers many baselines, models, and attacks.)
- Writing Quality: ⭐⭐⭐⭐⭐ (Excellent policy-to-technology argumentation.)
- Value: ⭐⭐⭐⭐⭐ ($45\times$ speedup and transferability make this a viable infrastructure candidate.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Discrete Diffusion Samplers and Bridges: Off-Policy Algorithms and Applications in Latent Spaces](discrete_diffusion_samplers_and_bridges_off-policy_algorithms_and_applications_i.md)
- [\[ICML 2026\] Structured Diffusion Bridges: Inductive Bias for Denoising Diffusion Bridges](structured_diffusion_bridges_inductive_bias_for_denoising_diffusion_bridges.md)
- [\[NeurIPS 2025\] OmniCast: A Masked Latent Diffusion Model for Weather Forecasting Across Time Scales](../../NeurIPS2025/image_generation/omnicast_a_masked_latent_diffusion_model_for_weather_forecasting_across_time_sca.md)
- [\[CVPR 2026\] SLICE: Semantic Latent Injection via Compartmentalized Embedding for Image Watermarking](../../CVPR2026/image_generation/slice_semantic_latent_injection_via_compartmentalized_embedding_for_image_waterm.md)
- [\[ICML 2026\] Latent Diffusion Pretraining for Crystal Property Prediction](latent_diffusion_pretraining_for_crystal_property_prediction.md)

</div>

<!-- RELATED:END -->
