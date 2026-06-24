---
title: >-
  [Paper Note] Do Not Mimic My Voice: Speaker Identity Unlearning for Zero-Shot Text-to-Speech
description: >-
  [ICML 2025][Audio & Speech][Machine Unlearning] This paper introduces the speaker identity unlearning task in zero-shot TTS for the first time, designing a Teacher-Guided Unlearning (TGU) framework that introduces randomness to make models "forget" target speaker voiceprint features while maintaining high-quality speech synthesis capabilities for other speakers, and proposes the spk-ZRF metric to quantify unlearning effectiveness.
tags:
  - "ICML 2025"
  - "Audio & Speech"
  - "Machine Unlearning"
  - "Zero-Shot Text-to-Speech"
  - "Speaker Privacy"
  - "Voiceprint Identity Unlearning"
  - "Right to be Forgotten"
date: 2026-05-08
content_hash: 6ea50bf6a6a600bd
---

# Do Not Mimic My Voice: Speaker Identity Unlearning for Zero-Shot Text-to-Speech

**Conference**: ICML 2025  
**arXiv**: [2507.20140](https://arxiv.org/abs/2507.20140)  
**Code**: [Demo](https://speechunlearn.github.io/)  
**Area**: AI Security  
**Keywords**: Machine Unlearning, Zero-Shot Text-to-Speech, Speaker Privacy, Voiceprint Identity Unlearning, Right to be Forgotten

## TL;DR

This paper introduces the speaker identity unlearning task in zero-shot TTS for the first time, designing a Teacher-Guided Unlearning (TGU) framework that introduces randomness to make models "forget" target speaker voiceprint features while maintaining high-quality speech synthesis capabilities for other speakers, and proposes the spk-ZRF metric to quantify unlearning effectiveness.

## Background & Motivation

Zero-shot text-to-speech (ZS-TTS) technology has advanced rapidly, with models like VALL-E and VoiceBox capable of high-fidelity cloning of any speaker's voice using only a 3-second audio prompt. This raises serious privacy and ethical concerns:

**Voice as a key biometric feature**: Voiceprints can be used for identity recognition, and unauthorized voice synthesis constitutes a privacy violation.

**Legal compliance requirements**: GDPR and the "Right to be Forgotten" (RTBF) require safeguarding personally identifiable information, granting users the right to request the deletion of their voice data.

**Limitations of Prior Work**: Simple anonymization or filtering of speaker representations is insufficiently secure; advanced attacks (model inversion, voice resynthesis, targeted fine-tuning) can still recover identity features from anonymous embeddings.

**Unique challenges in ZS-TTS unlearning**: Models can generalize to unseen speakers in a zero-shot manner; traditional approximate unlearning methods that merely exclude training data cannot limit the model's generalization capabilities on unseen voices.

Key Insight: An ideal unlearning model should not only avoid replicating the target speaker's voice, but also avoid fixing on any traceable specific style (otherwise, malicious users can reverse-engineer the original voice). Therefore, it is necessary to introduce **randomness**—allowing the model to generate randomly varying voice styles for the forgotten speaker.

## Method

### Overall Architecture

This paper builds an unlearning framework based on VoiceBox (a non-autoregressive ZS-TTS model based on Conditional Flow Matching, CFM). VoiceBox learns the flow field mapping from Gaussian noise to target speech through masked prediction learning, conditioned on audio context $x_{ctx}$ and text $y$.

**Problem Definition**: Given a pre-trained model $\theta$, the speaker set $S$ is divided into a forget set $F$ and a retain set $R = S - F$. The unlearned model $\theta^-$ must satisfy:

- For retained speakers $r \in R$: $\theta^-(x^r, y) \approx \hat{x}_y^{spk=r}$ (normal voice cloning)
- For forgotten speakers $f \in F$: $\theta^-(x^f, y) \approx \hat{x}_y^{spk \neq f}$ (generating a random voice different from the target)

This paper proposes two methods: Sample-Guided Unlearning (SGU) and Teacher-Guided Unlearning (TGU).

### Key Designs

#### Sample-Guided Unlearning (SGU)

The core idea of SGU is to concatenate the speech of a retained speaker $x^r$ and a forgotten speaker $x^f$ into a single sample, masking the $x^r$ portion as the generation target. In this way, when the model receives the audio prompt of a forgotten speaker, it is guided to generate the voice style of the retained speaker.

**Limitations**:

- The mask can only be applied to the entire $x^r$ portion, making selective masking in the middle of concatenated speech impossible.
- The original VoiceBox requires audio context before and after the masked region for infilling predictions, whereas SGU only has single-sided context.
- Mismatches in rhythm, speaking rate, etc., between the two speakers' voices can cause the model to learn unnatural speech generation patterns.

#### Teacher-Guided Unlearning (TGU) (Core Method)

TGU leverages the pre-trained model itself as a "teacher" to generate guidance targets, resolving the alignment issues of SGU.

**Core Innovation**: When VoiceBox is conditioned only on text $y$ (without an audio prompt), the generated speech style depends on the initialization of the Gaussian noise $x_0$, producing different random voice styles in each inference. TGU exploits this property by using the teacher model's unconditional generation result $\bar{x} = \theta(y)$ as the training target for unlearning:

$$\theta^-(x^f, y) \approx \theta(y)$$

**Mechanism**:

1. Input the forgotten speaker's speech $x^f$ and text $y$.
2. The teacher model $\theta$, conditioned only on $y$, generates speech $\bar{x}$ (random speaker style).
3. Use $\bar{x}$ as the training target for the student model $\theta^-$.
4. The student model learns to generate a random style consistent with the teacher's unconditional output when receiving the forgotten speaker's prompt.

This ensures that with different initializations, the voice style generated by the model varies, mitigating the risk of reproducing identifiable information.

#### Evaluation Metric spk-ZRF

Traditional unlearning evaluations only compare the performance gap between the forget and retain sets, failing to measure the **randomness** of generation. This paper proposes speaker-Zero Retrain Forgetting (spk-ZRF):

1. For each sample $(x_i^s, y_i)$, generate $\theta^-(x_i^s, y_i)$ and $\theta(y_i)$ (unconditional random generation).
2. Extract speaker embeddings for both using a speaker verification model.
3. Compute the Jensen-Shannon Divergence after converting them into probability distributions.
4. $\text{spk-ZRF} = 1 - \frac{1}{n}\sum_{i=1}^n \text{JSD}_i$

A spk-ZRF closer to 1 indicates that the speaker identity distribution generated by the unlearned model is as random as the unconditional generation, demonstrating superior unlearning effectiveness.

### Loss & Training

**Forget Set Loss (TGU)**:

$$L_{\text{CFM-forget}}(\theta^-) = \mathbb{E}_{t,q(x_1),p_t(x^f|x_1)} \left[ \| m \odot u_t(x|\bar{x}) - v_t(w^f, y, x_{ctx}^f; \theta^-) \|^2 \right]$$

where $\bar{x} = \theta(y)$ is the teacher model's unconditional generation, and $w^f = (1-(1-\sigma_{min})t)x_0 + t\bar{x}$.

**Retain Set Loss**:

$$L_{\text{CFM-remain}}(\theta^-) = \mathbb{E}_{t,q(x_1),p_t(x^r|x_1)} \left[ \| m \odot u_t(x|x_1^r) - v_t(w^r, y, x_{ctx}^r; \theta^-) \|^2 \right]$$

consistent with the CFM loss of the original VoiceBox, ensuring normal generation capabilities for retained speakers.

**Total Loss**:

$$L_{\text{total}} = \lambda L_{\text{CFM-remain}} + (1-\lambda) L_{\text{CFM-forget}}$$

The hyperparameter is set to $\lambda = 0.2$, meaning the forget loss dominates (weight of 0.8) while the retain loss assists (weight of 0.2).

## Key Experimental Results

### Main Results

Experiments train VoiceBox on LibriHeavy (50,000 hours of English speech), randomly selecting 10 speakers as the forget set (approx. 20 minutes of speech per speaker).

| Method | WER-R ↓ | SIM-R ↑ | WER-F ↓ | SIM-F ↓ | spk-ZRF-F ↑ |
|------|---------|---------|---------|---------|-------------|
| Original | 2.1 | 0.649 | 2.1 | 0.708 | 0.846 |
| Exact Unlearning | 2.3 | 0.643 | 2.2 | 0.687 | 0.846 |
| Fine Tuning | 2.2 | 0.658 | 2.3 | 0.675 | 0.853 |
| Negative Gradient | 6.1 | 0.437 | 5.0 | 0.402 | 0.842 |
| KL Divergence | 5.2 | 0.408 | 47.2 | 0.179 | 0.810 |
| **SGU (Ours)** | 2.6 | 0.523 | 2.5 | 0.194 | 0.866 |
| **TGU (Ours)** | **2.5** | **0.631** | **2.4** | **0.169** | **0.871** |

TGU achieves a SIM of only 0.169 on the forget set (close to the natural similarity between different speakers), while the retain set SIM decreases by only 2.8%.

### Ablation Study

Comparison of scalability under different numbers of forgotten speakers $k$:

| Configuration | WER-R ↓ | SIM-R ↑ | SIM-F ↓ | Note |
|------|---------|---------|---------|------|
| SGU (k=1) | 2.7 | 0.586 | 0.173 | SGU retain set performance fluctuates significantly with k |
| SGU (k=3) | 2.9 | 0.566 | 0.209 | SGU shows a prominent decrease on SIM-R |
| SGU (k=10) | 2.6 | 0.523 | 0.194 | SGU retain set similarity drops by 21% |
| TGU (k=1) | 2.3 | 0.624 | 0.164 | TGU performs stably across all scales |
| TGU (k=3) | 2.9 | 0.626 | 0.159 | TGU retain set similarity remains almost unchanged |
| TGU (k=10) | 2.5 | 0.631 | 0.169 | TGU exhibits superior scalability |

### Key Findings

1. **Exact Unlearning and Fine Tuning are ineffective**: Merely excluding the forgotten speaker's training data is insufficient to protect voiceprint privacy, as ZS-TTS can generalize to unseen speakers in a zero-shot manner.
2. **NG and KL methods collapse**: Negative Gradient (NG) causes severe degradation in overall model performance (WER up to 47.2), whereas the KL method leads the model to output inaudible noise rather than an alternative voice for the forget set.
3. **TGU achieves the highest spk-ZRF-F (0.871)**: This is 2.95% higher than the original model, proving that the generated speaker identity exhibits maximum randomness.
4. **Cross-domain unlearning is effective**: On LibriTTS (out-of-domain data), TGU is equally effective, with SIM-F dropping to 0.186.
5. **Human evaluation validation**: The CMOS-R of TGU (-0.02) is close to the original model, and its SMOS-F (1.28) is the lowest, indicating that the voice of the forgotten speaker indeed cannot be replicated.

## Highlights & Insights

1. **Novelty**: First to define and tackle the speaker identity unlearning problem in the ZS-TTS domain, filling a gap in machine unlearning for speech generation.
2. **Ingenious randomness-guided approach**: Instead of simple gradient reversal or noise injection, it exploits VoiceBox's character of voice style variation with initialization during unconditional generation, using the teacher's random outputs as unlearning targets.
3. **Sound spk-ZRF metric design**: It shifts the evaluation of unlearning effectiveness from "whether it is dissimilar to the target speaker" to "whether it is sufficiently random and untraceable," addressing the core demand of privacy protection.
4. **Balance between retention and unlearning**: TGU achieves unlearning effectiveness close to the natural difference between different speakers while suffering only a 2.8% performance loss on the retain set.

## Limitations & Future Work

1. **Validated only on VoiceBox**: Autoregressive ZS-TTS models like VALL-E were not tested, and the generalizability of the method remains to be verified.
2. **Collateral damage on similar voices**: Unlearning a specific speaker's voice may inadvertently restrict TTS capabilities for other speakers with similar voices.
3. **Sensitivity to the $\lambda$ hyperparameter**: Whether the weight ratio of retain/forget loss (0.2/0.8) remains robust across different forget set sizes requires further analysis.
4. **Trustworthiness of unlearning verification**: A more comprehensive auditing mechanism is needed to prove to users that the model has indeed unlearned their voices.
5. **Sequential unlearning scenarios**: The paper does not address the cumulative effects of multiple sequential unlearning requests, and whether model performance degrades significantly as unlearning iterations increase.

## Related Work & Insights

- **VoiceBox (Le et al., 2024)**: State-of-the-art ZS-TTS model based on Conditional Flow Matching, serving as the base architecture of this work.
- **VALL-E (Wang et al., 2025)**: Autoregressive ZS-TTS representing speech as discrete tokens, serving as another technical paradigm.
- **Concept Unlearning in Diffusion Models (Gandikota et al., 2023; Seo et al., 2024)**: Concept unlearning methods in computer vision, which inspired the transfer from vision to speech in this study.
- **ZRF Metric (Chundawat et al., 2023)**: The original Zero Retrain Forgetting metric uses a random-weight teacher model; this paper adapts it to solely measure speaker identity randomness.

This paper sheds critical light on multimodal AI safety research: as generative models become increasingly powerful, enabling models to "selectively unlearn" specific individual information will become a key competency for responsible AI deployment.

## Rating

| Dimension | Score | Description |
|------|------|------|
| Novelty | ⭐⭐⭐⭐⭐ | Defines the ZS-TTS speaker unlearning task for the first time; both problem definition and method design are highly original |
| Technical Depth | ⭐⭐⭐⭐ | TGU framework is cleverly designed and the spk-ZRF metric is sound, though the core methodology is relatively straightforward |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Includes multiple baselines, scalability analysis, out-of-domain, and human evaluation, though validated only on VoiceBox |
| Writing Quality | ⭐⭐⭐⭐ | Highly logical, with smooth motivation and methodological derivation |
| Value | ⭐⭐⭐⭐⭐ | Directly addresses GDPR/RTBF compliance requirements, holding practical deployment significance for TTS providers |
| **Overall** | **⭐⭐⭐⭐☆** | Novel and critical problem setting with effective methods, positioning it as an important pioneering effort in the field of speech safety |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Zero-Shot Text-to-Speech for Vietnamese](../../ACL2025/audio_speech/zero-shot_text-to-speech_for_vietnamese.md)
- [\[ACL 2025\] ControlSpeech: Towards Simultaneous and Independent Zero-shot Speaker Cloning and Zero-shot Language Style Control](../../ACL2025/audio_speech/controlspeech_zero_shot.md)
- [\[ICML 2025\] Sortformer: A Novel Approach for Permutation-Resolved Speaker Supervision in Speech-to-Text Systems](sortformer_a_novel_approach_for_permutation-resolved_speaker_supervision_in_spee.md)
- [\[ACL 2025\] TCSinger 2: Customizable Multilingual Zero-shot Singing Voice Synthesis](../../ACL2025/audio_speech/tcsinger_2_customizable_multilingual_zero-shot_singing_voice_synthesis.md)
- [\[ACL 2025\] Advancing Zero-shot Text-to-Speech Intelligibility across Diverse Domains via Preference Alignment](../../ACL2025/audio_speech/advancing_zero-shot_text-to-speech_intelligibility_across_diverse_domains_via_pr.md)

</div>

<!-- RELATED:END -->
