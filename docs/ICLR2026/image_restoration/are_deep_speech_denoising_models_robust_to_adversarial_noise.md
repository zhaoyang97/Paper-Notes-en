---
title: >-
  [Paper Note] Are Deep Speech Denoising Models Robust to Adversarial Noise?
description: >-
  [ICLR 2026][Image Restoration][Adversarial Attack] This paper presents the first systematic evaluation of the robustness of four SOTA deep speech denoising (DNS) models against adversarial noise. By generating perceptually imperceptible adversarial perturbations via PGD attacks constrained by psychoacoustic masking, the authors demonstrate that Demucs, Full-SubNet+, FRCRN, and MP-SENet can be made to produce completely unintelligible gibberish. The evaluation covers diverse acoustic conditions and human listening studies, while also revealing the limitations of targeted attacks, universal perturbations, and cross-model transferability.
tags:
  - ICLR 2026
  - Image Restoration
  - Adversarial Attack
  - Speech Denoising
  - Psychoacoustic Masking
  - DNS
  - Adversarial Robustness
  - PGD
date: 2026-05-08
content_hash: bd335d05ce214c2a
---

# Are Deep Speech Denoising Models Robust to Adversarial Noise?

**Conference**: ICLR 2026
**arXiv**: [2503.11627](https://arxiv.org/abs/2503.11627)
**Code**: [GitHub](https://github.com/audiolabs/speech-denoising-adversarial) (UMass Amherst + Dolby Labs)
**Area**: Image Restoration
**Keywords**: Adversarial Attack, Speech Denoising, Psychoacoustic Masking, DNS, Adversarial Robustness, PGD

## TL;DR
This paper presents the first systematic evaluation of the robustness of four SOTA deep speech denoising (DNS) models against adversarial noise. By generating perceptually imperceptible adversarial perturbations via PGD attacks constrained by psychoacoustic masking, the authors demonstrate that Demucs, Full-SubNet+, FRCRN, and MP-SENet can be made to produce completely unintelligible gibberish. The evaluation covers diverse acoustic conditions and human listening studies, while also revealing the limitations of targeted attacks, universal perturbations, and cross-model transferability.

## Background & Motivation

**State of the Field**: Deep DNS models (e.g., Demucs, Full-SubNet+, FRCRN, MP-SENet) have achieved remarkable progress on objective metrics such as PESQ and STOI, and are widely deployed in communication devices including smartphones, video conferencing systems, and hearing aids. While these models perform well under standard conditions, their adversarial robustness remains almost entirely unstudied.

**Limitations of Prior Work**: (a) Adversarial robustness has been extensively studied in computer vision, but the speech denoising domain is largely unexplored — existing work covers only individual models or single attack types and lacks human perceptual validation; (b) DNS models are increasingly deployed in safety-critical scenarios (hearing aids, emergency communications), where silent adversarial attacks pose genuine threats; (c) conventional $L_p$-norm constraints are insufficient to guarantee perceptual imperceptibility in audio — the frequency and temporal masking properties of the human auditory system require psychoacoustic modeling.

**Root Cause**: DNS models achieve ever-higher scores on standard benchmarks, yet it remains unclear whether small, inaudible perturbations can entirely destroy their denoising capability.

**Starting Point**: Drawing on the psychoacoustic model used in MP3 encoding to constrain adversarial perturbations for imperceptibility, this work systematically evaluates the vulnerability of four representative DNS architectures under diverse acoustic conditions (SNR, reverberation, OTA).

**Core Idea**: Psychoacoustic masking-constrained PGD attacks are used to generate adversarial noise that is imperceptible to the human ear yet causes SOTA DNS models to output gibberish, with human listening experiments confirming the attack's effectiveness.

## Method

### Overall Architecture
Clean speech $x$ mixed with ambient noise $n$ is correctly processed by a DNS model to yield clean output. When an adversarial perturbation $\delta$ is added — i.e., the input becomes $x + n + \delta$ — the DNS model instead produces unintelligible gibberish. The attack objective is to find $\delta$ such that (a) $\delta$ is imperceptible to human listeners (psychoacoustic constraint), and (b) the intelligibility of the DNS output is minimized (STOI approaching zero).

### Key Designs

1. **Attack Objective — STOI Loss Minimization**

   - **Function**: Minimize the Short-Time Objective Intelligibility (STOI) of the DNS output.
   - **Mechanism**: STOI computes the frame-wise normalized cross-correlation between the clean reference and the denoised output and averages across frames. Its negation serves as the loss function, optimized via PGD gradient descent over the adversarial perturbation $\delta$.
   - **Design Motivation**: STOI is highly correlated with human speech intelligibility and more directly reflects "comprehensibility" than perceptual quality metrics such as PESQ. Minimizing STOI is equivalent to maximizing speech unintelligibility.
   - **Key Detail**: STOI is differentiable, enabling direct backpropagation of gradients to the input perturbation.

2. **Psychoacoustic Imperceptibility Constraint**

   - **Function**: Ensure that the adversarial perturbation remains below the auditory masking threshold so that the perturbed signal sounds identical to the original.
   - **Mechanism**: The ISO MPEG-1 Psychoacoustic Model 2 (the standard model used in MP3 encoding) is employed to compute the masking threshold $T(k)$ for each frequency bin $k$. An additional 12 dB safety margin is applied to ensure sufficient imperceptibility, constraining the perturbation's power spectral density: $\mathrm{PSD}(\delta, k) \leq T(k) - 12\,\mathrm{dB}$.
   - **Additional Consideration**: Pre-masking (~2 ms) and post-masking (~200 ms) temporal effects are incorporated, further relaxing the temporal-domain constraint to exploit the auditory system's temporal masking properties.
   - **Design Motivation**: Traditional $L_\infty$ constraints do not reflect human auditory perception — the ear is more sensitive at low frequencies and tolerates larger perturbations at high frequencies. The psychoacoustic model precisely captures both frequency and temporal masking, providing a perceptually grounded bound rather than a fixed-norm constraint.

3. **PGD Optimization Procedure**

   - **Function**: Iterative gradient projection to solve the constrained optimization problem.
   - **Mechanism**: Standard PGD — gradient descent step followed by projection onto the psychoacoustic constraint set. Each update is $\delta \leftarrow \delta - \alpha \cdot \mathrm{sign}(\nabla)$, followed by per-frequency-bin clipping to remain below the masking threshold.
   - **Key Detail**: 200 PGD steps with tuned learning rate; gradient clipping is applied for Full-SubNet+ to handle its known gradient explosion issue.
   - **Initialization**: Zero initialization is more stable than random initialization.

4. **Four Evaluated DNS Models**

   - **Demucs** (Meta): Time-domain U-Net with LSTM encoder-decoder architecture; largest parameter count.
   - **Full-SubNet+ (FSN+)**: Frequency-domain full-band/sub-band network; known to suffer from gradient explosion (obfuscated gradients).
   - **FRCRN** (Alibaba): Frequency-recurrent CRN with complex spectral processing; moderate parameter count.
   - **MP-SENet**: Mask-enhanced network predicting both amplitude and phase simultaneously; most recent architecture.

### Evaluation Conditions
- **Acoustic Conditions**: Five SNR levels (70 dB / 30 dB / 10 dB / 5 dB / 0 dB) with and without reverberation, plus simulated over-the-air (OTA) transmission.
- **Human Evaluation**: (a) Transcription test — listeners attempt to transcribe denoised outputs, with WER computed; (b) ABX test — listeners identify the adversarial signal among three audio samples, verifying perceptual imperceptibility.
- **Objective Metrics**: STOI, PESQ, ViSQOL, and SI-SDR are reported comprehensively.

## Key Experimental Results

### Main Results — Untargeted Attack (70 dB SNR, No Reverberation)

| Model | STOI (Before) | STOI (After) | PESQ (Before) | PESQ (After) |
|-------|--------------|-------------|--------------|-------------|
| Demucs | 0.97 | 0.12 | 3.5 | 1.1 |
| FSN+ | 0.96 | 0.35 | 3.3 | 1.3 |
| FRCRN | 0.97 | 0.08 | 3.5 | 1.0 |
| MP-SENet | 0.96 | 0.15 | 3.4 | 1.1 |

### Attack Effectiveness Across Acoustic Conditions

| Condition | Demucs STOI | FRCRN STOI | MP-SENet STOI | Notes |
|-----------|-------------|------------|---------------|-------|
| 70 dB SNR, no reverb | 0.12 | 0.08 | 0.15 | Ideal conditions |
| 10 dB SNR, no reverb | 0.15 | 0.11 | 0.18 | Moderate noise |
| 5 dB SNR + reverb | 0.20 | 0.14 | 0.22 | Difficult conditions |
| Simulated OTA | 0.25 | 0.18 | 0.28 | Closest to real deployment |

### Human Evaluation Results
- **Transcription Test**: WER exceeds 95% on attacked outputs; participants are largely unable to comprehend any lexical content, confirming that the outputs are genuine gibberish.
- **ABX Imperceptibility Test**: Listener accuracy in identifying the adversarial signal is approximately 55% (near the 50% chance level), confirming that the perturbations are perceptually imperceptible.
- The conservative 12 dB safety margin is empirically validated as effective — more reliable than relying on the masking threshold alone.

### Key Findings

1. **All four DNS models are successfully attacked**: STOI drops from ~0.97 to 0.08–0.35, with outputs degrading to completely unintelligible noise.
2. **FSN+'s apparent robustness is illusory**: Its higher post-attack STOI (0.35 vs. 0.08–0.15 for other models) stems from PGD optimization difficulties caused by gradient explosion (obfuscated gradients), not genuine robustness — a known form of spurious defense circumventable by adaptive attacks (cf. Carlini et al.).
3. **Model size does not correlate with robustness**: Demucs has the largest parameter count yet is equally vulnerable; FRCRN has moderate capacity yet is the most susceptible. Gradient flow stability, not model capacity, is the determining factor.
4. **Attacks generalize across acoustic conditions**: From ideal (70 dB SNR, no reverb) to challenging (low SNR + reverb) and simulated OTA scenarios, attacks remain consistently effective, with only moderate degradation in severity.

### Negative Results (Equally Important Findings)

| Attack Type | Objective Metrics | Subjective Evaluation | Analysis |
|-------------|------------------|----------------------|----------|
| Targeted attack (force specific output speech) | Partial success | Listeners cannot perceive target content | Speech perception is high-dimensional and nonlinear; low-level feature matching does not imply intelligibility matching |
| Universal perturbation (single $\delta$ for all inputs) | Fails | STOI decreases only marginally | Spectral diversity across utterances is too large; the psychoacoustic constraint set is too small for a universal solution |
| Cross-model transfer attack | Negligible transfer | Other models unaffected | Gradient directions differ substantially across architectures; white-box attacks are highly model-specific |

### Defense Exploration
- **Gaussian Noise Injection**: Adding small Gaussian noise to the DNS input partially mitigates attacks (STOI recovers from ~0.08 to ~0.5), but at the cost of noticeably degraded quality during normal use — partial but insufficient protection.
- **Adversarial Training**: Noted as worthy of exploration but not pursued in depth due to the high training cost of DNS models.
- **Input Transformation Defenses**: Randomizing the input may help but introduces additional latency.

## Highlights & Insights

- **Elegant application of psychoacoustic masking constraints**: Directly repurposing Psychoacoustic Model 2 from MP3 encoding is a well-grounded and engineering-practical choice. The combination of the 12 dB safety margin with pre- and post-masking temporal effects yields imperceptibility that is rigorously validated through human experiments — this approach surpasses naive $L_\infty$ constraints for audio and sets a new standard for imperceptibility constraints in audio adversarial attack research.
- **Honest reporting of negative results**: The objective–subjective mismatch in targeted attacks and the failure of universal perturbations and transfer attacks are analyzed and discussed in detail, which is highly valuable in adversarial robustness research — it delineates the true capability boundaries of such attacks and avoids threat inflation.
- **Gradient explosion ≠ robustness**: FSN+'s apparent resilience is a textbook case of obfuscated gradients, echoing lessons from Athalye et al. (2018) and Carlini (2023) — defense evaluation must employ adaptive attacks, and gradient masking does not constitute genuine security.
- **Comprehensive and practical threat model**: From ideal conditions (70 dB SNR, no reverb) to realistic deployments (5 dB SNR + reverb + OTA transmission), the evaluation spans a complete threat assessment spectrum; simulated OTA transmission is an important addition for real-world relevance.
- **Model scale does not determine security**: The larger Demucs is no safer than smaller models; gradient flow characteristics are the key determinant of adversarial robustness — a practically relevant insight for security-aware DNS model design and architecture selection.

## Rating
- **Novelty**: 4/5 — First systematic multi-model adversarial robustness evaluation for DNS; psychoacoustic constraints applied to DNS attacks are a novel contribution; the PGD framework itself is not new.
- **Experimental Thoroughness**: 5/5 — Four models × multiple acoustic conditions × human evaluation (transcription + ABX) × thorough negative result analysis; the experimental design is comprehensive and rigorous.
- **Writing Quality**: 5/5 — Clear and well-structured; both positive and negative results are thoroughly discussed; the threat model is precisely and completely defined.
- **Value**: 4/5 — Raises a credible alarm regarding DNS model security; however, defense solutions remain at an exploratory stage and require follow-up work.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Learning to Translate Noise for Robust Image Denoising](../../CVPR2026/image_restoration/learning_to_translate_noise_for_robust_image_denoising.md)
- [\[ICLR 2026\] Activation Steering for Masked Diffusion Language Models](activation_steering_for_masked_diffusion_language_models.md)
- [\[ICLR 2026\] Horizon Imagination: Efficient On-Policy Rollout in Diffusion World Models](horizon_imagination_efficient_on-policy_rollout_in_diffusion_world_models.md)
- [\[ICLR 2026\] wd1: Weighted Policy Optimization for Reasoning in Diffusion Language Models](wd1_weighted_policy_optimization_for_reasoning_in_diffusion_language_models.md)
- [\[CVPR 2026\] NEC-Diff: Noise-Robust Event–RAW Complementary Diffusion for Seeing Motion in Extreme Darkness](../../CVPR2026/image_restoration/nec-diff_noise-robust_event-raw_complementary_diffusion_for_seeing_motion_in_ext.md)

<!-- RELATED:END -->
