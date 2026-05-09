---
title: >-
  [Paper Note] Yours or Mine? Overwriting Attacks Against Neural Audio Watermarking
description: >-
  [AAAI 2026][AI Safety][Audio watermarking] This paper presents the first systematic study of overwriting attacks against neural audio watermarking, proposing white-box, gray-box, and black-box attack schemes that achieve near-100% attack success rates against three SOTA methods—AudioSeal, Timbre, and WavMark—exposing critical security vulnerabilities in existing audio watermarking systems.
tags:
  - AAAI 2026
  - AI Safety
  - Audio watermarking
  - overwriting attack
  - copyright protection
  - adversarial security
  - deep watermarking
date: 2026-05-08
content_hash: 7af220dcced932ef
---

# Yours or Mine? Overwriting Attacks Against Neural Audio Watermarking

**Conference**: AAAI 2026
**arXiv**: [2509.05835](https://arxiv.org/abs/2509.05835)
**Code**: None
**Area**: AI Security
**Keywords**: Audio watermarking, overwriting attack, copyright protection, adversarial security, deep watermarking

## TL;DR

This paper presents the first systematic study of overwriting attacks against neural audio watermarking, proposing white-box, gray-box, and black-box attack schemes that achieve near-100% attack success rates against three SOTA methods—AudioSeal, Timbre, and WavMark—exposing critical security vulnerabilities in existing audio watermarking systems.

## Background & Motivation

With the rapid advancement of generative audio models, AI can produce highly realistic speech, giving rise to social risks such as voice-cloning fraud and copyright infringement. Audio watermarking serves as a proactive defense mechanism that embeds imperceptible digital signatures into audio signals for copyright protection and provenance verification.

Existing neural audio watermarking methods primarily focus on two properties:
- **Robustness**: The watermark remains detectable after common processing such as compression and resampling.
- **Imperceptibility**: The embedding process does not degrade perceptual audio quality.

**A third property that has been overlooked—Security**:

Robustness concerns tolerance to *unintentional* distortions, whereas security concerns resistance to *intentional* attacks. Prior research has mainly explored two types of security attacks:
- **Removal attacks**: Render the watermark undetectable.
- **Forgery attacks**: Falsely embed a watermark into clean audio.

However, a more practical and dangerous attack—the **overwriting attack**—has received almost no attention: an attacker replaces the legitimate watermark in an already-watermarked audio with a forged watermark, thereby **hijacking audio ownership**. Unlike removal attacks (which merely erase the mark), overwriting attacks directly **steal ownership**.

**Why are existing systems vulnerable?** According to Kerckhoffs's principle, a secure system should remain secure even if the algorithm is public; security should depend on a key rather than on secrecy. Yet neural watermarking systems typically lack explicit key-based security mechanisms and rely on "model weight secrecy"—a fragile assumption in an era of open-source models and reverse engineering.

## Method

### Overall Architecture

Attack objective: Given a publicly distributed watermarked audio $x_w = \mathcal{E}(x, m_{owner})$, the attacker uses an embedder $\mathcal{E}'$ to embed a forged message $m'_{adv}$, producing $x'_w = \mathcal{E}'(x_w, m'_{adv})$.

Success conditions:
1. The original message is no longer recoverable: $\mathcal{D}(x'_w) \neq m_{owner}$
2. The attacker's detector recovers the forged message: $\mathcal{D}'(x'_w) = m'_{adv}$
3. Perceptual indistinguishability: $d(x'_w, x_w) \leq \epsilon$

### Key Designs

#### 1. **White-box Attack**

Assumption: The attacker has full access to the original watermark embedder. Representative of insider threats or fully open-source scenarios.

The attack is extremely simple—directly re-embed a new message using the same embedder:

$$x'_w = \mathcal{E}(x_w, m'_{adv})$$

**Core finding**: When overwriting with the same embedder, the BER of the original watermark reaches ~0.5 (equivalent to random guessing), indicating that the original watermark is completely destroyed. This occurs because the embedder operates in the same embedding domain, so the new watermark naturally overwrites the old one.

However, when overwriting across different methods (e.g., using AudioSeal to overwrite Timbre), the BER is very low—different methods operate in distinct embedding domains with different decoding mechanisms, preventing cross-method interference. This finding underpins the gray-box and black-box attacks.

#### 2. **Gray-box Attack**

Assumption: The attacker knows the watermarking system's architecture but not the model weights or training details. A surrogate model $(\mathcal{E}', \mathcal{D}')$ must be trained.

A **universal watermarking training framework** is proposed:

$$\mathcal{L}_{total} = \lambda_w \mathcal{L}_w + \lambda_t \mathcal{L}_{recon_t} + \lambda_f \mathcal{L}_{recon_f} + \lambda_{adv} \mathcal{L}_{adv}$$

Four loss components:

- **Watermark recovery loss**: $\mathcal{L}_w = \text{BCE}(m, \mathcal{D}'(\mathcal{E}'(x, m)))$ — ensures accurate embedding and detection.
- **Time-domain reconstruction loss**: $\mathcal{L}_{recon_t} = \text{MSE}(x, \mathcal{E}'(x,m))$ — minimizes audible distortion.
- **Frequency-domain reconstruction loss**: Multi-resolution STFT loss comprising a spectral convergence term and a log-magnitude term:

$$\mathcal{L}_{recon_f} = \frac{1}{M} \sum_{m=1}^{M} (\mathcal{L}_{sc}^{(m)} + \mathcal{L}_{mag}^{(m)})$$

- **Adversarial loss**: A discriminator is trained to distinguish original from watermarked audio; the embedder is optimized to make watermarked audio indistinguishable:

$$\mathcal{L}_{adv} = -\log(\sigma(D(\mathcal{E}'(x,m))))$$

**Design Motivation**: Even without knowledge of the original model's training details, surrogate models trained with the same architecture converge to similar embedding strategies—they embed watermarks in similar spectral regions. This **architectural convergence** makes gray-box attacks highly effective.

Two gray-box settings:
- **Cross-training**: Same dataset (VoxCeleb1), different training procedures and random seeds.
- **Cross-data**: Entirely different datasets (LibriSpeech for surrogate training → attack VoxCeleb1 models).

#### 3. **Black-box Attack**

Assumption: No knowledge of architecture, weights, or training data. Two strategies are proposed:

**Zero-query attack**:
- Collect or reproduce a set of public watermarking models $\mathcal{E}'_i$.
- Apply all models sequentially in a brute-force stacking manner:

$$x_w^{(N)} = (\mathcal{E}_N \circ \mathcal{E}_{N-1} \circ \cdots \circ \mathcal{E}_1)(x_w, m'_{adv})$$

As the number of stacked models increases, ASR grows from ~30% (1 model) to ~100% (3 models), while SNR decreases from ~24 dB to ~20 dB.

**Query-based attack**:
1. Partially train candidate surrogate models (few epochs).
2. Embed the new message using the under-trained models.
3. Query the original detector $\mathcal{D}$ a limited number of times to assess whether the original watermark has been destroyed.
4. Once the most effective candidate is identified, continue training until reliable overwriting is achieved.

**Design Motivation**: The query-based strategy achieves over 50% reduction in training iterations with fewer than 10 queries, requires applying only a single effective model (rather than stacking multiple ones), and preserves audio quality (SNR 24.19 dB vs. 20.63 dB).

### Loss & Training

- Training datasets: LibriSpeech (~1,000 hours) and VoxCeleb1 (150k+ utterances).
- Audio format: 16 kHz WAV.
- Target watermarking methods: AudioSeal (encoder-decoder), Timbre (frequency-domain), WavMark (invertible neural network).
- Three random-seed initializations (Init-1/2/3) to verify reproducibility.
- Hardware: 64 CPUs + 2× A100 GPUs.

## Key Experimental Results

### Main Results

**White-box Overwriting Results**

| Target Method | ASR (Original Watermark) ↑ | ACC (Overwrite Watermark) ↑ |
|---|---|---|
| Timbre | 99.80% | 100.00% |
| AudioSeal | 100.00% | 100.00% |
| WavMark | 100.00% | 100.00% |

The original watermark is almost completely destroyed, and the overwritten watermark is recovered with perfect accuracy.

**Gray-box Cross-training Results (ASR %)**

| Target Method | Init-1 | Init-2 | Init-3 |
|---|---|---|---|
| Timbre | 99.60 | 98.80 | 98.40 |
| AudioSeal | 100.00 | 100.00 | 100.00 |
| WavMark | 100.00 | 100.00 | 99.50 |

**Gray-box Cross-data Results (LibriSpeech → VoxCeleb1, ASR %)**

| Target Method | Init-1 | Init-2 | Init-3 |
|---|---|---|---|
| Timbre | 99.80 | 99.90 | 98.80 |
| AudioSeal | 100.00 | 100.00 | 100.00 |
| WavMark | 100.00 | 100.00 | 100.00 |

Even when the surrogate model is trained on an entirely different dataset, the attack success rate remains near 100%.

### Ablation Study

**Black-box Attack: Zero-query vs. Query-based**

| Attack Type | No. of Queries | Training Cost | SNR (dB) | ASR (%) |
|---|---|---|---|---|
| Zero-query | 0 | 36,000 iters | 20.63 | 100 |
| Query-based | <10 | 14,000 iters | **24.19** | 100 |

With fewer than 10 queries, the query-based attack achieves:
- 61% reduction in training cost.
- 3.56 dB improvement in SNR (better audio quality).
- Identical attack success rate.

**White-box BER matrix analysis**: Diagonal entries (same-method overwriting) yield BER ≈ 0.5 (random-guess level), while off-diagonal entries (cross-method overwriting) yield very low BER—confirming that overwriting efficacy stems from embedding-domain overlap, which is not shared across different methods.

### Key Findings

1. **Overwriting attacks represent a systemic security flaw in neural audio watermarking**: Three representative methods spanning three distinct embedding paradigms are compromised under all threat levels.
2. **Architectural convergence phenomenon**: Surrogate models trained with different data, training procedures, and random seeds converge to similar embedding strategies (confirmed by spectral visualization: all models embed watermarks in similar spectral regions).
3. **The assumption "model secrecy = security" does not hold**: Gray-box and black-box attacks demonstrate that an attacker does not need to know the exact weights to perform effective overwriting.
4. **Overwriting is more dangerous than removal**: Removal attacks merely render the watermark undetectable; overwriting attacks directly steal ownership—the attacker can claim the audio as their own creation.
5. **The query-based strategy is highly efficient**: Fewer than 10 queries suffice to identify an effective attack model.

## Highlights & Insights

- **Novel attack formulation with significant practical threat**: Overwriting attacks are more destructive than removal/forgery attacks—they not only destroy the legitimate watermark but also implant false proof of ownership.
- **Well-designed three-tier threat model**: White-box → gray-box → black-box represents a spectrum of progressively weaker assumptions, covering the full range from insider threats to fully external attacks.
- **The discovery of "architectural convergence" has far-reaching implications**: It reveals a fundamental weakness in current watermarking methods—security should not rely on model secrecy but on cryptographic keys.
- **Simplicity and effectiveness**: The white-box attack requires only a single line of code, $x'_w = \mathcal{E}(x_w, m'_{adv})$; gray-box attacks can be constructed using the proposed universal training framework.
- **Spectral visualization provides intuition**: Models trained under different conditions exhibit similar embedding regions in the spectrum, directly explaining the effectiveness of gray-box attacks.

## Limitations & Future Work

1. **Only three watermarking methods are evaluated**: Although three distinct embedding paradigms are covered, emerging methods such as XattnMark and SilentCipher are not assessed.
2. **No defense is proposed**: The paper is positioned as an attack study, but it lacks in-depth discussion of potential defense directions (e.g., asymmetric watermarking, key-bound embedders).
3. **Limited audio quality metrics**: Only SNR is used; more comprehensive perceptual metrics such as PESQ and ViSQOL are not employed.
4. **Fixed watermark message length**: The effect of varying message length on overwriting attack performance is not analyzed.
5. **Black-box attacks assume access to a set of candidate public models**: In practice, attackers may not be able to obtain a sufficient number of publicly available watermarking models.
6. **Multi-layer or nested watermarking defenses are not considered**: Some practical systems may embed multiple complementary watermarks.

## Related Work & Insights

- This work issues an important security warning to the audio watermarking community: the current dual-objective optimization framework of "robustness + imperceptibility" needs to be upgraded to a tri-objective framework of "robustness + imperceptibility + security."
- It echoes overwriting attack research in the image watermarking domain, highlighting cross-modal shared security concerns.
- The work motivates the following defense directions:
    - Asymmetric watermarking (using different keys for embedding and detection, analogous to public-key cryptography).
    - Watermark fingerprint binding (binding the watermark to audio content so that overwriting causes verification failure).
    - Multi-layer redundant watermarking (simultaneously embedding complementary watermarks in different embedding domains).
    - Zero-knowledge-proof-based ownership verification.
- The application of Kerckhoffs's principle in AI security merits greater attention.

## Rating

- Novelty: ⭐⭐⭐⭐ — First systematic study of overwriting attacks with a well-designed three-tier threat model, though the attack methods themselves are relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 3 methods × 3 threat models × multiple settings, with detailed BER distribution analysis.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, rigorous threat model definition, and convincing motivation.
- Value: ⭐⭐⭐⭐⭐ — Exposes a systemic security flaw in neural audio watermarking, carrying significant cautionary implications for the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Hashed Watermark as a Filter: A Unified Defense Against Forging and Overwriting Attacks in Neural Network Watermarking](hashed_watermark_as_a_filter_defeating_forging_and_overwriting_attacks_in_weight.md)
- [\[ICLR 2026\] Robust Spiking Neural Networks Against Adversarial Attacks](../../ICLR2026/ai_safety/robust_spiking_neural_networks_against_adversarial_attacks.md)
- [\[AAAI 2026\] InfoDecom: Decomposing Information for Defending Against Privacy Leakage in Split Inference](infodecom_decomposing_information_for_defending_against_privacy_leakage_in_split.md)
- [\[AAAI 2026\] Reference Recommendation based Membership Inference Attack against Hybrid-based Recommender Systems](reference_recommendation_based_membership_inference_attack_against_hybrid-based_.md)
- [\[AAAI 2026\] Detect All-Type Deepfake Audio: Wavelet Prompt Tuning for Enhanced Auditory Perception](detect_all-type_deepfake_audio_wavelet_prompt_tuning_for_enhanced_auditory_perce.md)

</div>

<!-- RELATED:END -->
