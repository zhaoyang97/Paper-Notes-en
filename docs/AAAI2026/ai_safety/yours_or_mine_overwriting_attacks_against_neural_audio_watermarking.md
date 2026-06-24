---
title: >-
  [Paper Note] Yours or Mine? Overwriting Attacks Against Neural Audio Watermarking
description: >-
  [AAAI 2026][AI Safety][Audio Watermarking] This work presents the first systematic study on overwriting attacks against neural audio watermarking, proposing a three-tier attack framework (white-box, gray-box, and black-box). It achieves near 100% attack success rates across three SOTA methods (AudioSeal, Timbre, and WavMark), exposing severe security vulnerabilities in existing audio watermarking systems.
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "Audio Watermarking"
  - "Overwriting Attack"
  - "Copyright Protection"
  - "Adversarial Security"
  - "Deep Watermarking"
date: 2026-05-08
content_hash: 3cfe879d5e27119f
---

# Yours or Mine? Overwriting Attacks Against Neural Audio Watermarking

**Conference**: AAAI 2026  
**arXiv**: [2509.05835](https://arxiv.org/abs/2509.05835)  
**Code**: None  
**Area**: AI Security  
**Keywords**: Audio Watermarking, Overwriting Attack, Copyright Protection, Adversarial Security, Deep Watermarking

## TL;DR

This work presents the first systematic study on overwriting attacks against neural audio watermarking, proposing a three-tier attack framework (white-box, gray-box, and black-box). It achieves near 100% attack success rates across three SOTA methods (AudioSeal, Timbre, and WavMark), exposing severe security vulnerabilities in existing audio watermarking systems.

## Background & Motivation

With the rapid development of generative audio models, AI can generate highly realistic speech, bringing social risks such as voice-cloning fraud and copyright infringement. Audio watermarking, as an active defense mechanism, achieves copyright protection and source verification by embedding imperceptible digital signatures into audio signals.

Existing neural audio watermarking methods primarily focus on two properties:
- **Robustness**: The watermark remains detectable after common signal processing such as compression and resampling.
- **Imperceptibility**: The embedding process does not degrade the perceived audio quality.

**The neglected third property — Security**:

Robustness focuses on tolerating **unintentional perturbations**, while security focuses on resisting **intentional attacks**. Existing works mainly explore two types of security attacks:
- **Removal attacks**: Making the watermark undetectable.
- **Forgery attacks**: Falsely embedding watermarks into clean audio.

However, a more practical and dangerous attack — **Overwriting** — remains almost unstudied: an attacker replaces the legitimate watermark in the watermarked audio with a forged watermark, thereby **hijacking the audio copyright**. Unlike removal attacks (which only erase the mark), overwriting attacks directly **steal ownership**.

**Why are existing systems vulnerable?** According to Kerckhoffs's principle, a secure system should be secure even if the algorithm is public; security should rely on keys rather than secrecy. However, neural watermarking systems typically lack explicit key-based security mechanisms and rely on the assumption of "model weight confidentiality" — a vulnerable assumption in an era of open-sourcing and reverse engineering.

## Method

### Overall Architecture

Attack Goal: Given a publicly distributed watermarked audio $x_w = \mathcal{E}(x, m_{owner})$, the attacker uses an embedder $\mathcal{E}'$ to embed a forged message $m'_{adv}$ to generate $x'_w = \mathcal{E}'(x_w, m'_{adv})$.

Success Conditions:
1. The original message is no longer recoverable: $\mathcal{D}(x'_w) \neq m_{owner}$
2. The attacker's detector can recover the forged message: $\mathcal{D}'(x'_w) = m'_{adv}$
3. Perceptually indistinguishable: $d(x'_w, x_w) \leq \epsilon$

### Key Designs

#### 1. **White-box Attack**

Assumption: The attacker has full access to the original watermark embedder. This represents internal threats or fully open-source scenarios.

The attack is extremely simple — directly re-embed the new message using the same embedder:

$$x'_w = \mathcal{E}(x_w, m'_{adv})$$

**Core Idea**: When overwriting using the same embedder, the Bit Error Rate (BER) of the original watermark reaches ~0.5 (equivalent to random guessing), indicating that the original watermark is completely destroyed. This is because the embedder operates in the same embedding domain, and the new watermark naturally overwrites the old one.

However, when overwriting using a different method (e.g., overwriting Timbre with AudioSeal), the BER is extremely low — different methods operate in different embedding domains and have different decoding mechanisms, making cross-destruction impossible. This finding forms the basis of gray-box and black-box attacks.

#### 2. **Gray-box Attack**

Assumption: The attacker knows the architecture of the watermarking system but does not know the model weights and training details. A surrogate model $(\mathcal{E}', \mathcal{D}')$ needs to be trained.

A **universal watermarking training framework** is proposed:

$$\mathcal{L}_{total} = \lambda_w \mathcal{L}_w + \lambda_t \mathcal{L}_{recon_t} + \lambda_f \mathcal{L}_{recon_f} + \lambda_{adv} \mathcal{L}_{adv}$$

Four loss components:

- **Watermark recovery loss**: $\mathcal{L}_w = \text{BCE}(m, \mathcal{D}'(\mathcal{E}'(x, m)))$ — ensures the accuracy of embedding and detection.
- **Time-domain reconstruction loss**: $\mathcal{L}_{recon_t} = \text{MSE}(x, \mathcal{E}'(x,m))$ — minimizes audible distortion.
- **Frequency-domain reconstruction loss**: Multi-resolution STFT loss, consisting of a spectral convergence term and a log magnitude term:

$$\mathcal{L}_{recon_f} = \frac{1}{M} \sum_{m=1}^{M} (\mathcal{L}_{sc}^{(m)} + \mathcal{L}_{mag}^{(m)})$$

- **Adversarial loss**: Trains a discriminator to distinguish original and watermarked audio, with the embedder aiming to make watermarked audio indistinguishable:

$$\mathcal{L}_{adv} = -\log(\sigma(D(\mathcal{E}'(x,m))))$$

**Design Motivation**: Even without knowing the training details of the original model, a surrogate model trained with the same architecture will converge to a similar embedding strategy — they embed watermarks in similar spectral regions. This **architectural convergence** makes gray-box attacks highly effective.

Two gray-box settings:
- **Cross-training**: Same dataset (VoxCeleb1), but different training pipelines and random seeds.
- **Cross-data**: Completely different datasets (training the surrogate on LibriSpeech $\to$ attacking the VoxCeleb1 model).

#### 3. **Black-box Attack**

Assumption: No knowledge of the architecture, weights, or training data. Two strategies are used:

**Zero-query attack**:
- Collect or reproduce a set of public watermarking models $\mathcal{E}'_i$.
- Bruteforce stacking, sequentially applying all models:

$$x_w^{(N)} = (\mathcal{E}_N \circ \mathcal{E}_{N-1} \circ \cdots \circ \mathcal{E}_1)(x_w, m'_{adv})$$

As the number of stacked models increases, the Attack Success Rate (ASR) increases from ~30% (1 model) to ~100% (3 models), but the SNR drops from ~24dB to ~20dB.

**Query-guided attack**:
1. Partially train candidate surrogate models (for a few epochs).
2. Embed the new message using the under-trained models.
3. Query the original detector $\mathcal{D}$ a limited number of times to evaluate whether the original watermark is destroyed.
4. Continue training the most effective candidate until it can reliably overwrite.

**Design Motivation**: The query-guided strategy achieves over 50% training iteration savings with <10 queries and only requires applying a single effective model (rather than stacking multiple models), preserving audio quality (SNR 24.19dB vs 20.63dB).

### Loss & Training

- Training Datasets: LibriSpeech (~1000 hours) and VoxCeleb1 (150k+ samples).
- Audio Format: 16kHz WAV.
- Target Watermarking Methods: AudioSeal (encoder-decoder type), Timbre (frequency-domain type), WavMark (invertible neural network type).
- Three random seed initializations (Init-1/2/3) to verify reproducibility.
- Hardware: 64 CPU + 2×A100 GPU.

## Key Experimental Results

### Main Results

**White-box Overwriting Results**

| Target Method | ASR (Original Watermark)↑ | ACC (Overwritten Watermark)↑ |
|---------|----------------|----------------|
| Timbre | 99.80% | 100.00% |
| AudioSeal | 100.00% | 100.00% |
| WavMark | 100.00% | 100.00% |

The original watermark is almost completely destroyed, and the overwritten watermark is recovered with perfect accuracy.

**Gray-box Cross-training Results (ASR %)**

| Target Method | Init-1 | Init-2 | Init-3 |
|---------|--------|--------|--------|
| Timbre | 99.60 | 98.80 | 98.40 |
| AudioSeal | 100.00 | 100.00 | 100.00 |
| WavMark | 100.00 | 100.00 | 99.50 |

**Gray-box Cross-data Results (LibriSpeech → VoxCeleb1, ASR %)**

| Target Method | Init-1 | Init-2 | Init-3 |
|---------|--------|--------|--------|
| Timbre | 99.80 | 99.90 | 98.80 |
| AudioSeal | 100.00 | 100.00 | 100.00 |
| WavMark | 100.00 | 100.00 | 100.00 |

Even when the surrogate model is trained on a completely different dataset, the attack success rate remains close to 100%.

### Ablation Study

**Black-box Attack: Zero-query vs Query-guided**

| Attack Type | Query Budget | Training Cost | SNR (dB) | ASR (%) |
|---------|---------|---------|----------|---------|
| Zero-query | 0 | 36,000 iters | 20.63 | 100 |
| Query-guided | <10 | 14,000 iters | **24.19** | 100 |

The query-guided attack achieves the following with <10 queries:
- 61% reduction in training costs.
- 3.56 dB improvement in SNR (better audio quality).
- Equivalent attack success rate.

**White-box BER Matrix Analysis**: The diagonal (same-method overwriting) BER $\approx 0.5$ (random guess level), while the off-diagonal (cross-method overwriting) BER is extremely low — proving that the overwriting capability originates from the overlap of embedding domains, and different methods do not share embedding domains.

### Key Findings

1. **Overwriting attacks expose systematic security flaws in neural audio watermarking**: Three representative methods (spanning three different embedding paradigms) are compromised across all threat levels.
2. **Architectural convergence phenomenon**: Surrogate models trained with different data, different training details, and different random seeds converge to similar embedding strategies (as confirmed by spectral visualization: all models embed watermarks in similar spectral regions).
3. **The assumption of "model secrecy = security" is invalid**: Gray-box and black-box attacks prove that attackers do not need to know the exact weights to perform effective overwriting.
4. **Overwriting is more dangerous than removal**: Removal attacks only make the watermark undetectable, whereas overwriting attacks directly hijack the copyright — allowing physical adversaries to claim the audio is their own creation.
5. **Query-guided strategy is highly efficient**: Fewer than 10 queries are sufficient to locate an effective attack model.

## Highlights & Insights

- **Novel and highly threatening attack topic**: Overwriting attacks are more destructive than removal/forgery — they not only destroy the legitimate watermark but also implant a false proof of ownership.
- **Well-designed three-tier threat model**: White-box $\to$ gray-box $\to$ black-box step-by-step reduction in assumptions, covering a full spectrum from internal threats to completely external attacks.
- **The discovery of "architectural convergence" has profound implications**: It exposes a fundamental vulnerability of current watermarking methods — security should not rely on model secrecy, but rather on cryptographic keys.
- **Simple yet effective**: The white-box attack requires only a single line of code $x'_w = \mathcal{E}(x_w, m'_{adv})$, and the gray-box attack can construct surrogates using a universal training framework.
- **Spectral visualization provides intuition**: Models trained differently exhibit similar embedding areas on the spectrum, intuitively explaining the effectiveness of gray-box attacks.

## Limitations & Future Work

1. **Evaluation limited to three watermarking methods**: Although covering three different embedding paradigms, emerging methods (e.g., XattnMark, SilentCipher) have not been evaluated.
2. **Lack of proposed defense schemes**: The paper is positioned as an "attack paper," but lacks an in-depth discussion on potential defense directions (such as asymmetric watermarking, key-bound embedders, etc.).
3. **Limited audio quality evaluation metrics**: Only SNR is used, lacking more comprehensive perceptual metrics like PESQ or ViSQOL.
4. **Fixed watermark message length**: The impact of different message lengths on the effectiveness of overwriting attacks has not been analyzed.
5. **Assumption of an available candidate model set in black-box attacks**: In practice, attackers may not have access to an adequate pool of public watermarking models.
6. **Multi-layer or nested watermarking defense strategies are not considered**: Some practical systems may embed multiple complementary watermarks.

## Related Work & Insights

- This work sounds an important security alarm for the audio watermarking field: the current dual-objective optimization framework of "Robustness + Imperceptibility" needs to be upgraded to a triple-objective framework of "Robustness + Imperceptibility + Security".
- Echoes the study of overwriting attacks in the image watermarking field, highlighting cross-modal security issues.
- Inspires future defense directions:
    - Asymmetric watermarking (different keys for embedding and detection, similar to public-key cryptography).
    - Watermark-fingerprint binding (binding the watermark to the audio content so that overwriting triggers checksum failures).
    - Multi-layer redundant watermarking (embedding complementary watermarks in different domains simultaneously).
    - Ownership verification based on Zero-Knowledge Proofs (ZKP).
- The application of Kerckhoffs's principle in AI security deserves more attention.

## Rating

- Novelty: ⭐⭐⭐⭐ — First systematic study of overwriting attacks with a comprehensive three-tier threat model, though the attack method itself is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 3 methods × 3 threat models × multiple settings, with a detailed analysis of BER distributions.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, rigorously defined threat models, and compelling motivation.
- Value: ⭐⭐⭐⭐⭐ — Exposes systematic security flaws in neural audio watermarking, carrying significant warning implications for the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Hashed Watermark as a Filter: A Unified Defense Against Forging and Overwriting Attacks in Neural Network Watermarking](hashed_watermark_as_a_filter_defeating_forging_and_overwriting_attacks_in_weight.md)
- [\[ICLR 2026\] Robust Spiking Neural Networks Against Adversarial Attacks](../../ICLR2026/ai_safety/robust_spiking_neural_networks_against_adversarial_attacks.md)
- [\[AAAI 2026\] Detect All-Type Deepfake Audio: Wavelet Prompt Tuning for Enhanced Auditory Perception](detect_all-type_deepfake_audio_wavelet_prompt_tuning_for_enhanced_auditory_perce.md)
- [\[AAAI 2026\] RegionMarker: A Region-Triggered Semantic Watermarking Framework for Embedding-as-a-Service](regionmarker_a_region-triggered_semantic_watermarking_framework_for_embedding-as.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](robust_watermarking_on_gradient_boosting_decision_trees.md)

</div>

<!-- RELATED:END -->
