---
title: >-
  [Paper Note] De-AntiFake: Rethinking the Protective Perturbations Against Voice Cloning Attacks
description: >-
  [ICML 2025][AI Safety][Adversarial Purification] This paper presents the first systematic evaluation of the vulnerability of protective perturbation-based voice cloning (VC) defense methods against adversarial purification. It proposes PhonePuRe, a two-stage "Purification-Refinement" framework that utilizes a phoneme-guided diffusion model to effectively eliminate protective perturbations. This allows voice cloning models to accurately replicate speaker characteristics again…
tags:
  - "ICML 2025"
  - "AI Safety"
  - "Adversarial Purification"
  - "Voice Cloning Defense"
  - "Diffusion Models"
  - "Phoneme Guidance"
  - "Protective Perturbations"
date: 2026-05-08
content_hash: 6ae663367d04c28a
---

# De-AntiFake: Rethinking the Protective Perturbations Against Voice Cloning Attacks

**Conference**: ICML 2025  
**arXiv**: [2507.02606](https://arxiv.org/abs/2507.02606)  
**Code**: [Yes](https://de-antifake.github.io)  
**Area**: AI Security  
**Keywords**: Adversarial Purification, Voice Cloning Defense, Diffusion Models, Phoneme Guidance, Protective Perturbations

## TL;DR

This paper presents the first systematic evaluation of the vulnerability of protective perturbation-based voice cloning (VC) defense methods against adversarial purification. It proposes PhonePuRe, a two-stage "Purification-Refinement" framework that utilizes a phoneme-guided diffusion model to effectively eliminate protective perturbations. This allows voice cloning models to accurately replicate speaker characteristics again, revealing the fundamental limitations of existing defense schemes.

## Background & Motivation

Voice cloning (VC) technology has advanced rapidly in recent years, capable of generating highly realistic synthetic speech from only a few seconds of target speaker audio. While this technology has broad positive applications in virtual assistants and speech-assistive devices, it also poses severe security risks. Attackers can leverage VC technology for phone fraud, bypassing speaker verification systems, and even copyright infringement. Real-world fraud cases utilizing VC technology have already occurred, such as forging a CFO's voice to steal 25 million USD.

To counter the threats posed by VC, researchers have proposed various protective perturbation methods. These methods inject human-imperceptible adversarial perturbations into the speech to prevent VC models from accurately replicating speaker characteristics. Representative works include AttackVC, AntiFake, and VoiceGuard. However, the authors notice a critical issue: **in a realistic threat model, an attacker can completely remove these protective perturbations using purification methods prior to performing VC**. If these protective methods cannot withstand purification strategies, they only provide users with a false sense of security.

The three core motivations of this work are:

**Inadequate evaluation of existing defenses**: Prior work has not systematically evaluated the effectiveness of protective perturbations under a realistic threat model that includes purification strategies.

**Deficiencies in existing purification methods**: Existing adversarial purification methods (e.g., AudioPure, WavePurifier) are mainly designed for classification tasks. They introduce systematic distortion in the embedding space of VC models, thereby degrading VC performance.

**Need for stronger purification methods**: To fully expose the vulnerabilities of defense schemes, it is necessary to develop more effective purification methods to bypass these protections.

## Method

### Overall Architecture

PhonePuRe adopts a two-stage cascaded framework:

1. **Purification Stage**: Uses a pre-trained unconditional diffusion model in the time domain to perform initial denoising on protected speech, eliminating most of the adversarial perturbations.
2. **Refinement Stage**: Uses a phoneme-guided score-based diffusion model in the frequency domain to further refine the speech, aligning the purified samples with the distribution of clean speech.

These two stages are trained separately and cascaded during inference. The overall pipeline can be expressed as:

$$\mathbf{x}_{\text{ref}} = R_\phi(P_\theta(\mathbf{x}_{\text{adv}}), \mathbf{\Lambda})$$

where $P_\theta$ is the purification model, $R_\phi$ is the refinement model, and $\mathbf{\Lambda}$ is the phoneme representation.

### Key Designs

1. **Embedding Distortion Analysis**:

   Function: Analyzes the distortion problem introduced by existing purification methods within the VC embedding space.

   Key Findings: Existing purification methods based on unconditional diffusion models face a dilemma: a small number of diffusion steps fails to fully remove perturbations, while a large number of steps leads to a loss of sample details. In both cases, the purified samples deviate from the embedding distribution of clean samples, specifically shown as: (1) samples from different speakers become closer in the embedding space (reduced inter-class separability); (2) purified samples deviate from their clean counterparts. Since VC models rely on fine-grained feature information to accurately replicate the speaker's voice, this distortion severely degrades VC performance.

   Design Motivation: This analysis directly inspires the design of the two-stage framework—coarse purification followed by fine alignment.

2. **Purification Stage (Unconditional Diffusion Purification)**:

   Function: Uses the DiffWave unconditional diffusion model to purify adversarial speech in the waveform domain.

   Mechanism: Performs forward diffusion (noise addition) and reverse diffusion (denoising) on the input adversarial speech $\mathbf{x}_{\text{adv}}$:

   Forward process: $q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I})$

   Reverse process: $\mathbf{x}_{t-1} \sim p_\theta(\mathbf{x}_{\text{t-1}}|\mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \sigma_t^2\mathbf{I})$

   Design Motivation: The Gaussian noise added during the forward diffusion process "covers" the adversarial perturbations, and the reverse process leverages the learned prior to recover the sample close to the clean speech distribution. The core value of this stage is providing a good initialization point for the Refinement stage—experimental results show that purified clean samples and purified protected samples share a similar distribution. This allows the Refinement model to be trained using only clean sample pairs.

3. **Refinement Stage (Phoneme-Guided Score-Based Diffusion Refinement)**:

   Function: Uses a conditional diffusion model in the complex spectrogram domain to accurately align purified samples with the clean speech distribution.

   Mechanism: Constructs a training dataset $\mathcal{D} = \{(\mathbf{x}^{(i)}, \mathbf{x}^{(i)}_{\text{pur}})\}$ by processing clean samples through the Purification stage to obtain paired data. The Refinement model learns the conditional distribution $p_\phi(\mathbf{m}|\mathbf{m}_{\text{pur}})$, where $\mathbf{m} = \text{STFT}(\mathbf{x})$.

   Training Objective (denoising score matching):

    $\mathcal{L}(\phi) = \mathbb{E}\left[\left\|s_\phi(\mathbf{m}_\tau, [\mathbf{m}_{\text{pur}}, \mathbf{\Lambda}], \tau) + \frac{\mathbf{z}}{\sigma(\tau)}\right\|_2^2\right]$

   Design Motivation: Directly mapping adversarial samples to clean samples is very difficult (as the distribution of adversarial perturbations is unknown). However, a key observation from the Purification stage makes this feasible: purified clean samples and purified protected samples have similar distributions. Therefore, the mapping function trained solely on paired clean samples can generalize to protected samples.

4. **Phoneme Representation**:

   Function: Uses phoneme information as the conditioning guide for the Refinement stage.

   Mechanism:
    - Uses Montreal Forced Aligner (MFA) to perform phoneme alignment on training samples.
    - Calculates the average magnitude spectrum $\mathbf{\Lambda}$ of each phoneme across all training samples.
    - During inference, concatenates the average magnitude spectrum corresponding to the phoneme sequence of the input audio with the input: $[\mathbf{m}_{\text{pur}}, \mathbf{\Lambda}]$.

   Design Motivation: Protective perturbations are primarily designed against the speaker feature encoders of VC models, aiming to destroy speaker-specific features rather than speech content information. Therefore, phoneme information (which encodes speech content) is minimally affected by perturbations and can serve as a reliable guidance cue to help the Refinement model restore clean speech details.

### Loss & Training

- **Purification Model**: Based on a pre-trained DiffWave model, fine-tuned on the LibriSpeech dataset, operating in the time domain (16kHz).
- **Refinement Model**: Based on the NCSN++ score-estimator architecture, using the Ornstein-Uhlenbeck Stochastic Differential Equation (OU-SDE), trained in the complex spectrogram domain. STFT parameters: window size 510, hop length 128, square-root Hann window.
- **Training Data**: Uses an augmented LibriSpeech dataset to construct (clean, purified) sample pairs.
- **Inference Sampling**: Uses a predictor-corrector sampling scheme, combined with a single-step annealed Langevin dynamics correction.

## Key Experimental Results

### Main Results

Experiments are evaluated on 25 speakers from LibriSpeech test-clean, with 5 utterances per speaker, covering 6 VC methods.

| Protection Method | Metric | No Purification | AudioPure | WavePurifier | PhonePuRe (Ours) | Gain |
|----------|------|--------|-----------|-------------|-----------------|------|
| AntiFake | xSVA | 0.152 | 0.401 | 0.299 | **0.660** | +25.9% |
| AntiFake | dSVA | 0.164 | 0.451 | 0.293 | **0.762** | +31.1% |
| AttackVC | xSVA | 0.108 | 0.734 | 0.536 | **0.750** | +1.6% |
| AttackVC | dSVA | 0.108 | 0.777 | 0.505 | **0.861** | +8.4% |
| VoiceGuard | xSVA | 0.036 | 0.656 | 0.423 | **0.723** | +6.7% |
| VoiceGuard | dSVA | 0.039 | 0.712 | 0.385 | **0.830** | +11.8% |
| **Overall Average** | xSVA | 0.099 | 0.597 | 0.419 | **0.711** | +11.4% |
| **Overall Average** | dSVA | 0.104 | 0.647 | 0.394 | **0.818** | +17.1% |

Speech quality evaluation (Objective MOS):

| Condition | MOS |
|------|-----|
| Clean | 3.42 ± 0.59 |
| Protected | 3.16 ± 0.65 |
| AudioPure | 3.14 ± 0.55 |
| WavePurifier | 3.34 ± 0.67 |
| **PhonePuRe** | **3.36 ± 0.58** |

### Ablation Study

| Configuration | xSVA | dSVA | Description |
|------|------|------|------|
| w/o Purification | 0.350 | 0.371 | Refinement cannot eliminate perturbations independently |
| w/o Refinement | 0.597 | 0.646 | Equivalent to existing SOTA purification methods |
| w/o Phoneme | 0.695 | 0.786 | Still outperforms baselines without phoneme guidance |
| **Full model** | **0.711** | **0.818** | Two-stage + phoneme guidance is optimal |

### Key Findings

1. **Inherent vulnerability of existing protection methods**: All three protective methods (AntiFake, AttackVC, VoiceGuard) are significantly weakened in the face of adversarial purification. Without purification, SVA is < 20%, but after purification, SVA reaches 70-80%.
2. **Necessity of the two-stage design**: The Purification stage removes perturbations but introduces distortion, while the Refinement stage corrects the distortion and aligns with the clean distribution. Using either stage in isolation produces suboptimal results.
3. **Effectiveness of phoneme guidance**: Improves xSVA from 0.695 to 0.711, and dSVA from 0.786 to 0.818.
4. **Performance gain is not from more diffusion steps**: Increasing the diffusion steps in the Purification stage does not yield the equivalent gain brought by the Refinement stage, confirming that the improvement stems from distribution alignment rather than simply increasing denoising iterations.
5. **Adaptive protection remains ineffective**: Even when defenders have full white-box access (including gradients of the purification model), under adaptive protection using BPDA+EOT (EOT size 15), dSVA still maintains above 0.8.

## Highlights & Insights

1. **First to systematically expose the vulnerability of VC defenses**: By introducing adversarial purification into the VC defense evaluation framework, this work reveals the fundamental limitations of protective perturbation methods under realistic threat models. This finding serves as an important warning to the AI security community.
2. **Elegant two-stage design concept**: The key value of the Purification stage is not only removing perturbations but also making the purified distributions of clean and protected samples converge. This allows the Refinement model to be trained using only clean data, cleverly bypassing the challenge of unknown adversarial perturbation distributions.
3. **Phonemes as "content anchors"**: Exploiting the characteristic that protective perturbations mainly target speaker features rather than speech content, this work uses phoneme information as a reliable guide for the refinement process. This design logic is simple yet effective.
4. **Multi-domain processing strategy**: Purification operates in the time domain (waveform), and Refinement operates in the frequency domain (complex spectrogram), fully exploiting the complementary advantages of different domains.

## Limitations & Future Work

1. **Ethical risk**: The method proposed in this paper is essentially an attack technique that helps attackers bypass voice cloning defenses. Although the authors emphasize that this is to expose risks and drive the development of stronger defenses, once the method is public, it indeed carries potential for malicious exploitation.
2. **Purification model requires training data**: The Refinement model requires clean speech data to construct training pairs, indicating a certain level of data dependence.
3. **Dependency on phoneme alignment**: Transcription and a forced aligner are required, which may pose limitations in language coverage and robustness.
4. **Computational overhead**: Inference costs for the two-stage diffusion model are high, which might raise efficiency concerns in practical attack scenarios.
5. **Generalizability of evaluation**: Experiments are only conducted on LibriSpeech (English), and the generalization capability across different languages remains unverified.
6. **Lack of constructive defense solutions**: The paper primarily demonstrates attack capabilities. It only advocates for the design of more robust defense schemes without providing concrete directions or solutions.

## Related Work & Insights

- **Adversarial purification in the visual domain**: Works like DiffPure have demonstrated the potential of diffusion models for adversarial purification in the image domain. This work extends similar concepts to the audio domain.
- **Utilization of phoneme information in speech synthesis**: Inspired by works like Grad-TTS and SpeechFlow, this work uses phonemes as conditional information to guide the diffusion process.
- **The arms race in AI security**: This work is an important link in the offense-defense cycle of VC defense, suggesting that future defenses may need to transcend the paradigm of adversarial perturbations (e.g., toward watermarking, passive detection, etc.).
- **Potential research directions**: Can we design protective perturbations that are simultaneously robust to purification? For example, incorporating the robustness boundaries of the purification model during perturbation optimization.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The first to systematically evaluate the vulnerability of VC protective perturbations under purification threats. The two-stage framework design is innovative but built upon existing components.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Extremely comprehensive, covering 6 VC methods, 3 protection methods, and 5 purification baselines, including ablation studies, adaptive protection, and subjective evaluations.
- **Writing Quality**: ⭐⭐⭐⭐ Clearly structured, rigorously defined threat models, and solid visual analysis.
- **Value**: ⭐⭐⭐⭐ Provides an important warning to the AI security community, though its constructive contribution from a defensive perspective is limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Adversarial Inception Backdoor Attacks against Reinforcement Learning](adversarial_inception_backdoor_attacks_against_reinforcement_learning.md)
- [\[ICML 2025\] Theoretically Unmasking Inference Attacks Against LDP-Protected Clients in Federated Vision Models](theoretically_unmasking_inference_attacks_against_ldp-protected_clients_in_feder.md)
- [\[ICML 2025\] Rethinking the Bias of Foundation Model under Long-tailed Distribution](rethinking_the_bias_of_foundation_model_under_long-tailed_distribution.md)
- [\[AAAI 2026\] Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach](../../AAAI2026/ai_safety/rethinking_target_label_conditioning_in_adversarial_attacks_a_2d_tensor-guided_g.md)
- [\[ICLR 2026\] Defending against Backdoor Attacks via Module Switching](../../ICLR2026/ai_safety/defending_against_backdoor_attacks_via_module_switching.md)

</div>

<!-- RELATED:END -->
