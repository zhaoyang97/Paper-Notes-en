---
title: >-
  [Paper Note] Training-Free Coverless Multi-Image Steganography with Access Control
description: >-
  [ICML 2026][AI Safety][Coverless Steganography] The authors propose MIDAS, a training-free coverless multi-image steganography framework based on pre-trained diffusion models. By replacing the traditional Noise Flip with…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Coverless Steganography"
  - "Multi-image Steganography"
  - "Access Control"
  - "Diffusion Models"
  - "Random Basis"
date: 2026-05-08
content_hash: 9baf9aa11b26c79e
---

# Training-Free Coverless Multi-Image Steganography with Access Control

**Conference**: ICML 2026  
**arXiv**: [2603.09390](https://arxiv.org/abs/2603.09390)  
**Code**: https://github.com/Minyeol/MIDAS  
**Area**: AI Security / Information Hiding / Diffusion Models  
**Keywords**: Coverless Steganography, Multi-image Steganography, Access Control, Diffusion Models, Random Basis

## TL;DR
The authors propose MIDAS, a training-free coverless multi-image steganography framework based on pre-trained diffusion models. By replacing the traditional Noise Flip with an orthogonal Random Basis, it achieves fine-grained access control based on private keys. Combined with Latent Vector Fusion to eliminate stitching boundaries, it enables multi-image hiding and anti-steganalysis without transmitting any secret-related side information.

## Background & Motivation

**Background**: Image steganography primarily follows two paradigms. **Modification-based** methods (e.g., Baluja, HiNet, DeepMIH, IIS, AIS) directly encode secret images into the pixels or wavelet coefficients of a cover image. While quality is high, once the cover is leaked, they are easily detected by steganalysis. **Coverless Image Steganography (CIS)** uses generative models to directly synthesize stego images (where no modified cover exists), possessing natural anti-steganalysis capabilities. CRoSS, DiffStega, and DStyleStego are representative training-free CIS schemes.

**Limitations of Prior Work**: Existing training-free CIS methods rarely support access control. Naively extending them to multi-image scenarios leads to two failures: (1) **Reconstruction quality collapses** when single-image designs are scaled N times. (2) Directly concatenating the noisy latents of N secrets causes the diffusion reverse process to fail at **smoothing cross-boundary transitions**, resulting in visible stitching seams in stego images (as shown in Fig. 1 for DiffStega* / CRoSS*). More critically, even with an incorrect $K_{priv}$, DiffStega can reconstruct recognizable images, **lacking access control security**.

**Key Challenge**: CIS aims to simultaneously possess four capabilities: "training-free + multi-image + access control + no side information." Existing works either have high training costs (Chen 2025, Qin 2025 self-train generators), require extra secret-related side information for every communication (DStyleStego, HIS), or fail to guarantee access control—no existing method satisfies all four conditions.

**Goal**: Construct a truly training-free multi-image access control CIS on public diffusion models: (a) Fuse $N$ secret images into one stego image, (b) Ensure only users with the correct $K_{priv}^i$ can recover the $i$-th image while wrong keys yield meaningless results, (c) Transmit no supplementary secret-related information, and (d) Resist steganalysis.

**Key Insight**: The authors observe that previous private key mechanisms like Noise Flip use simple diagonal sign-flip matrices $M_d = \text{diag}(e), e\in\{-1,1\}^d$. Their search space is too regular, and they insufficiently suppress structural residual information in noisy latents. Replacing them with an **orthogonal random basis $Q_d(\mathcal{K},\gamma)$ derived from a seed** maintains reversibility ($Q^T Q = I$ for perfect reconstruction) and theoretically ensures that the information leakage rate $R_L$ approaches zero as the intensity $\gamma\to 1$.

**Core Idea**: Use seed-driven orthogonal matrix (Random Basis) encryption and Latent Vector Fusion with a shared reference latent to replace naive concatenation, unifying "encryption" and "seam elimination" within the same mathematical structure.

## Method

### Overall Architecture
MIDAS operates entirely within the latent space ($C\times H\times W$) of pre-trained Stable Diffusion v1.5, divided into Hiding and Reconstruction stages.

**Hiding Stage** (Sender): $N$ secret images $I_{sec}^i$ → Perform DDIM forward after downsampling to obtain noisy latents $\mathbf{z}_{sec}^i \in \mathbb{R}^{C\times H/N_1\times W/N_2}$ (where $N_1 N_2 = N$) → Encrypt with private key $\mathcal{K}_{priv}^i$ via Random Basis into $\mathbf{z}_{prot}^i$ → Concatenate to obtain $\mathbf{z}_{prot}\in\mathbb{R}^{C\times H\times W}$ → Apply Latent Vector Fusion with public key $\mathcal{K}_{pub}$ to mix it with reference latent $\mathbf{z}_{ref}$ into $\mathbf{z}_{pub}$ → Perform DDIM reverse + public prompt $\mathcal{P}_{pub}$ to render stego image $I_{stego}$. The $\mathbf{z}_{ref}$ is deterministically generated by a Reference Generator (RefGen) from $(\mathcal{K}_{pub}, \mathcal{P}_{pub})$, requiring no separate transmission.

**Reconstruction Stage** (Receiver): Received $\tilde{I}_{stego}$ (possibly corrupted) → DDIM inversion yields $\tilde{\mathbf{z}}_{pub}$ → Reverse Latent Vector Fusion with public key to get $\hat{\mathbf{z}}_{prot}$ → Use personal private key $\mathcal{K}_{priv}^i$ to decrypt $N$ segments (only the $i$-th segment yields a meaningful latent) → Perform joint denoise on the entire latent before splitting → VAE decode to obtain $\hat{I}_{sec}^i(i)$.

### Key Designs

1.  **Random Basis Private Key Encryption**:
    *   **Function**: Scatters each secret latent using a seed-derived orthogonal matrix to achieve encryption and suppress structural residuals.
    *   **Mechanism**: For any $d$-dimensional vector $\mathbf{z}$, it is encrypted as $\mathbf{z}_{enc} = M_d \mathbf{z}$, where $M_d = Q_d(\mathcal{K},\gamma)$ is an orthogonal matrix derived from seed $\mathcal{K}$ and intensity $\gamma$. Due to orthogonality, $\mathbf{z} = M_d^T \mathbf{z}_{enc}$ allows perfect restoration. Theorem 3.1 states the information leakage rate $R_L \approx O\left(\frac{-\log\Delta+\log m}{m} + (1-\gamma)(-\log\Delta+1)\right)$. With $m\approx 10^6$ and $\Delta\approx 10^{-7}$ (float32), the first term is negligible, and the second term approaches zero as $\gamma\to 1$.
    *   **Design Motivation**: To replace the Noise Flip used in DiffStega. Random Basis uses true rotations to break the spatial correlation of the latent, offering provable leakage control.

2.  **Latent Vector Fusion**:
    *   **Function**: Applies a **global** orthogonal transformation to the concatenated multi-image latent and mixes it with a reference latent to fundamentally eliminate stitching seams.
    *   **Mechanism**: Defined as $\mathbf{z}_{pub} = \sqrt{\alpha}\, M_D \mathbf{z}_{prot} + \sqrt{1-\alpha}\, \mathbf{z}_{ref}$, where $M_D = Q_D(\mathcal{K}_{pub}, \gamma_{fuse})$ with dimension $D = C\times H\times W$. $M_D$ globally scatters spatial information across multiple segments to break concatenation boundaries, while weighted mixing with $\mathbf{z}_{ref}$ injects "natural image" priors.
    *   **Design Motivation**: The authors found that noisy latents retain residual structures of secret images. Direct concatenation causes the DDIM reverse process to fail at smoothing across segments. Actively scattering the spatial structure and substituting the natural image distribution is critical for SOTA visual quality.

3.  **RefGen (Control-Image-Free Reference Generation)**:
    *   **Function**: Generates the reference image $I_{ref}$ deterministically from public resources $(\mathcal{K}_{pub}, \mathcal{P}_{pub})$, reproducible locally by both sender and receiver without transmission.
    *   **Mechanism**: An independent pre-trained diffusion model (PicX_real) is used with $\mathcal{K}_{pub}$ as the initial noise seed and $\mathcal{P}_{pub}$ as the prompt to perform deterministic sampling.
    *   **Design Motivation**: Unlike DiffStega which uses ControlNet conditions (which might leak structural information if transmitted), MIDAS removes the ControlNet path. It uses Random Basis + Latent Vector Fusion to handle both "high-quality conditional generation" and "secret information embedding."

### Loss & Training
The method is **completely training-free**, updating no model parameters. The pipeline utilizes SD v1.5 with EDICT exact inversion and the DDIM sampler. Main hyperparameters include $\gamma_{priv}, \gamma_{fuse}, \alpha$.

## Key Experimental Results

### Main Results
Datasets: Stego260 (Yu 2023) + UniStega (Yang 2024), evaluated with $N=2$. Metrics include stego quality (MANIQA↑), stego diversity (PSNR↓/SSIM↓ with secret images), correct key reconstruction quality, and incorrect key reconstruction quality (PSNR↓ indicating better security).

| Setting | Method | MANIQA↑ | Stego-PSNR↓ | CLIP↑ | Correct key Reconstruction PSNR↑ | Wrong key Reconstruction PSNR↓ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| N=2 | CRoSS* | 0.406 | 15.55 | 26.07 | 17.61 | 15.27 |
| N=2 | DiffStega* | 0.399 | 17.07 | 26.95 | 21.91 | 18.14 |
| N=2 | **MIDAS** | **0.434** | **9.89** | **30.13** | **23.90** | **9.96** |
| N=4 | CRoSS* | 0.418 | 13.45 | 24.60 | 13.19 | 12.73 |
| N=4 | DiffStega* | 0.364 | 16.16 | 27.37 | 19.23 | 17.53 |
| N=4 | **MIDAS** | **0.479** | **9.00** | **30.17** | **22.28** | **9.40** |

Note that for $N=4$, MIDAS stego quality (MANIQA 0.479) is actually higher than for $N=2$ (0.434), whereas baselines decline significantly as $N$ increases. This highlights the **multi-image scalability** of MIDAS. Furthermore, a ~14 dB PSNR gap (23.9 vs 9.96) between correct and wrong key reconstruction indicates robust access control.

### Robustness & Anti-Steganalysis

| Attack | IIS | AIS | CRoSS* | DiffStega* | **MIDAS** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Clean (PSNR↑) | 41.36 | 30.77 | 17.61 | 21.91 | **23.90** |
| Gaussian noise σ=5 | 12.44 | 14.65 | 16.30 | 20.08 | **20.05** |
| JPEG Q=70 | 10.05 | 9.44 | 16.93 | **20.42** | 19.92 |
| Gaussian blur σ=2 | 10.68 | 10.00 | 15.63 | 19.38 | **19.69** |

Modification-based methods (IIS/AIS) perform well under clean settings but collapse to ~10 dB under any channel noise; MIDAS maintains ~20 dB across three common distortions.

**Anti-Steganalysis** (XuNet / SiaStegNet): Detection rates for IIS/AIS are >90%, and CRoSS*/DiffStega* are >85%. MIDAS detection rates are approximately 20% lower than all baselines, approaching the level of random guessing.

### Key Findings
*   **Random Basis vs Noise Flip**: Ablations show Random Basis significantly outperforms Noise Flip in both stego and reconstruction quality.
*   **Latent Vector Fusion is Critical**: Removing this step reverts the system to the stitching seam issues observed in DiffStega*.
*   **Security at $\gamma_{priv}=0.4$**: Although the theoretical $(1-\gamma)$ term is non-zero, wrong-key reconstruction quality already drops to ~10 dB at this level.
*   **Extreme Capacity N=8**: MIDAS remains functional when sharing one stego image among 8 secret images.

## Highlights & Insights
*   **Unified Algebra**: Unifies "encryption" and "seam elimination" under the same orthogonal matrix structure.
*   **Provable Security**: Theorem 3.1 provides an asymptotic leakage bound $R_L$, giving steganographic security an explainable scaling behavior.
*   **Removed ControlNet Dependency**: Shifting from publicly transmitted control images to deterministic seed-based generation represents a more "cryptographically pure" design.
*   **Architecture Agnostic**: The method can be migrated to any latent generative model (e.g., SD3, Flux) without retraining.

## Limitations & Future Work
*   **Latency**: High inference latency due to DDIM inversion and EDICT; sampling acceleration is needed.
*   **Assumption on N**: Current logic requires $N = N_1 \times N_2$, limiting secret counts to regular grids; arbitrary $N$ requires flexible patch packing.
*   **Prompt Selection**: The impact of semantic conflict between the public prompt and secrets is not fully quantified.
*   **Dynamic Scheduling**: Potential for scheduling $\alpha$ across timesteps rather than using a fixed value.

## Related Work & Insights
*   **vs CRoSS**: CRoSS is single-image and requires prompt transmission; MIDAS is multi-image and uses seeds.
*   **vs DiffStega**: DiffStega relies on ControlNet and Noise Flip; MIDAS upgrades Noise Flip to Random Basis and adds Latent Vector Fusion.
*   **vs HIS**: HIS uses modification-based techniques on generated stego images; MIDAS remains entirely coverless.
*   **vs IIS / AIS**: These are modification-based; while having high clean PSNR, they exhibit weak robustness and anti-steganalysis capabilities compared to MIDAS.

## Rating
*   Novelty: ⭐⭐⭐⭐
*   Experimental Thoroughness: ⭐⭐⭐⭐
*   Writing Quality: ⭐⭐⭐⭐
*   Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] One-to-More: High-Fidelity Training-Free Anomaly Generation with Attention Control](../../CVPR2026/ai_safety/one-to-more_high-fidelity_training-free_anomaly_generation_with_attention_control.md)
- [\[ICML 2026\] SORA: Free Second-Order Attacks in Fast Adversarial Training](sora_free_second-order_attacks_in_fast_adversarial_training.md)
- [\[ICML 2026\] Rethinking Evaluation Paradigms in IBP-based Certified Training](rethinking_evaluation_paradigms_in_ibp-based_certified_training.md)
- [\[ICML 2026\] TimeGuard: Channel-wise Pool Training for Backdoor Defense in Time Series Forecasting](timeguard_channel-wise_pool_training_for_backdoor_defense_in_time_series_forecas.md)
- [\[ICML 2026\] Demystifying the Optimal Fair Classifier in Multi-Class Classification](demystifying_the_optimal_fair_classifier_in_multi-class_classification.md)

</div>

<!-- RELATED:END -->
