---
title: >-
  [Paper Note] RTCFake: Speech Deepfake Detection in Real-Time Communication
description: >-
  [ACL2026][Audio & Speech][Speech Deepfake Detection] RTCFake constructs a ~600-hour speech deepfake detection dataset targeting real RTC platforms and proposes Phoneme-guided Consistency Learning (PCL). This reduces the…
tags:
  - "ACL2026"
  - "Audio & Speech"
  - "Speech Deepfake Detection"
  - "Real-Time Communication"
  - "Cross-platform Generalization"
  - "Phoneme Consistency"
  - "EER"
date: 2026-05-08
content_hash: ff3def9290cc2b9b
---

# RTCFake: Speech Deepfake Detection in Real-Time Communication

**Conference**: ACL2026  
**arXiv**: [2604.23742](https://arxiv.org/abs/2604.23742)  
**Code**: https://huggingface.co/datasets/JunXueTech/RTCFake  
**Area**: AI Security / Speech Deepfake Detection / Real-Time Communication Security  
**Keywords**: Speech Deepfake Detection, Real-Time Communication, Cross-platform Generalization, Phoneme Consistency, EER

## TL;DR
RTCFake constructs a ~600-hour speech deepfake detection dataset targeting real RTC platforms and proposes Phoneme-guided Consistency Learning (PCL). This reduces the average EER of XLSR+AASIST from 7.33% (mixed training) to 5.81% across offline, online, cross-platform, and unseen noise scenarios.

## Background & Motivation
**Background**: Speech deepfake detection has established datasets like ASVspoof, ADD, DFADD, CodecFake, SpeechFake, and SpoofCeleb. Methodologies typically employ hand-crafted acoustic features, end-to-end detectors, self-supervised speech representations, and graph attention backends like AASIST.

**Limitations of Prior Work**: Many datasets primarily simulate offline or single distortions, such as codec compression, MP3, or noisy environments. However, real-time communication (RTC) platforms like Zoom, WeChat, QQ, DingTalk, and Lark contain black-box processing chains including noise reduction, echo cancellation, automatic gain control (AGC), codecs, network jitter, and packet loss. These coupled distortions alter fine-grained artifacts in spoofed speech, causing detectors trained offline to fail significantly in real call environments.

**Key Challenge**: Speech deepfake detection relies on frame-level details, yet RTC systems strongly perturb these local acoustic details. Simultaneously, these platforms strive to preserve semantic intelligibility. Consequently, frame-level features are unstable while semantic structures remain relatively stable.

**Goal**: The objective is to provide a dataset transmitted through mainstream RTC platforms and design a training strategy that enables detection models to learn representations robust across offline/online, cross-platform, and unseen noise conditions.

**Key Insight**: Through analysis of paired offline-online speech, it is observed that phoneme-level representations exhibit higher similarity and lower variance before and after transmission compared to frame-level representations. Thus, phoneme boundaries serve as stable anchors to constrain consistency between offline and online representations during training.

**Core Idea**: Instead of relying solely on frame-level spoofing traces that are easily erased by RTC black-box processing, phoneme-level consistency is utilized to pull the model toward semantic structural representations that are more stable across platforms.

## Method
RTCFake consists of two components: a dataset under real transmission conditions and a training method leveraging the paired offline-online structure of this dataset. The paper focuses on detection and robustness evaluation rather than generation or evasion.

### Overall Architecture
Data construction begins by collecting real speech from public corpora, followed by generating spoofed speech using 7 TTS and 3 VC systems to form the offline subset. Speech is then played and received between two independent PCs via mainstream RTC platforms to record online speech. ASR is used to verify text consistency post-transmission, filtering samples with content discrepancies.

The methodology employs XLSR+AASIST as the detector. XLSR provides front-end representations, AASIST serves as the back-end classifier, and RawBoost is used for robustness enhancement. During training, PCL takes both offline speech and its corresponding online speech as input. A phoneme recognition model predicts boundaries to aggregate frame representations into phoneme-level representations, minimizing the difference between offline and online phoneme representations.

### Key Designs
1.  **Real RTC Transmission Dataset**:
    - **Function**: Provides paired offline and online data transmitted through real platforms, enabling detectors to face real black-box communication distortions.
    - **Mechanism**: Covers seven platforms (Zoom, QQ, WeChat, DingTalk, Lark, VooV, Telegram) and includes unseen platforms and noise conditions in evaluations. The total duration is ~600 hours, covering 307 speakers.
    - **Design Motivation**: Simulating codecs or additive noise cannot replicate the non-linear coupled processing in RTC systems. Real transmission data reveals the distribution shifts encountered during deployment.

2.  **Phoneme-level Stability Observation**:
    - **Function**: Explains why consistency constraints should be applied at the phoneme level rather than directly on frame-level representations.
    - **Mechanism**: Comparing frame-level and phoneme-level representation similarity before and after transmission shows phoneme-level representations have higher means and lower variance. This indicates platforms prioritize semantic intelligibility over fine acoustic textures.
    - **Design Motivation**: While frame-level artifacts are useful for detection, they are unstable in RTC transmission. Phoneme-level aggregation filters transient perturbations and provides cross-platform anchors.

3.  **Phoneme-guided Consistency Learning**:
    - **Function**: Enables the detector to maintain both classification capability and offline-online invariance.
    - **Mechanism**: Phoneme boundaries are obtained via a phoneme recognition model. Mean pooling is applied to frame representations within the same phoneme to obtain offline representation $p^{(a)}$ and online representation $p^{(b)}$. The training objective is the mean cross-entropy of both branches plus $\lambda \mathcal{L}_{pcl}$, where $\mathcal{L}_{pcl}$ is the phoneme-level MSE consistency loss.
    - **Design Motivation**: Simple mixing of offline and online data mitigates distribution shift but does not explicitly instruct the model on which structures should remain stable. PCL explicitly turns this stability into a training constraint.

### Loss & Training
The experiment uses XLSR+AASIST with 16 kHz audio input, Adam optimizer, learning rate $1\times 10^{-6}$, and weight decay $1\times 10^{-4}$. Training lasts up to 100 epochs with early stopping after 10 epochs of no improvement. Classification uses cross-entropy, and PCL uses MSE consistency constraints. The evaluation metric is Equal Error Rate (EER).

## Key Experimental Results

### Main Results
The main table compares training on existing public datasets, RTCFake offline training, online training, mixed training, and PCL. EERs are high when migrating public datasets to RTC conditions, indicating existing benchmarks do not cover real communication distributions.

| Training Data / Method | Offline EER | Online P01 | Online P02 | Online P05 | Online P07 | All Avg EER | Conclusion |
|-----------------|-------------|------------|------------|------------|------------|-------------|------|
| ASVspoof2019 | 51.15 | 54.68 | 29.70 | 48.23 | 49.40 | 50.28 | Nearly fails under real RTC |
| SpoofCeleb | 29.56 | 40.05 | 30.70 | 32.48 | 38.55 | 34.06 | In-the-wild data $\neq$ RTC |
| Ours (Off) | 5.42 | 6.79 | 20.40 | 16.07 | 13.79 | 9.60 | Good offline, significant online shift |
| Ours (On) | 9.57 | 5.05 | 7.30 | 11.77 | 8.35 | 8.96 | Good online, hurts offline generalization |
| Ours (Mix) | 6.09 | 4.93 | 8.85 | 11.65 | 8.57 | 7.33 | More balanced |
| PCL | 4.84 | 3.79 | 6.24 | 10.17 | 6.77 | 5.81 | Best overall EER |

PCL also remains most stable under unseen noise conditions, achieving 3.88% EER on clean-only S01 and outperforming Off, On, and Mix on unseen noises S02/S03/S04/S06/S07.

### Ablation Study

| Configuration | Avg EER | Description |
|------|----------|------|
| Phoneme features only + FCL | 8.34 | Frame-level consistency on phoneme features is weak |
| Phoneme features only + PCL | 7.52 | Phoneme consistency outperforms frame consistency |
| Frame features + FCL | 6.55 | Frame features provide strong detection foundation |
| Frame features + PCL | 5.81 | Retains fine-grained info while stabilizing via phoneme anchors |

### Key Findings
- RTCFake demonstrates that distribution shifts caused by real communication platforms are not adequately covered by traditional public datasets (EERs often in the 30%-50% range).
- Offline (Off) and Online (On) training both exhibit bias; Mix is more balanced but still lacks robustness.
- PCL reduces All Avg EER from 7.33% to 5.81% compared to Mix and is more robust across platforms and unseen noise, proving phoneme-level anchors mitigate representation drift caused by RTC black-box processing.

## Highlights & Insights
- The dataset contribution is highly practical. Unlike papers validating only on simulated distortions, RTCFake collects paired online speech via mainstream platforms, matching deployment environments.
- The core observation is critical: RTC platforms prioritize semantic intelligibility at the expense of local acoustic details, making "phoneme-level stability and frame-level drift" a reasonable inductive bias.
- PCL acts as a general plugin that adds training constraints without modifying the detector backbone, allowing migration to other frameworks.
- For AI security systems, robustness evaluation must cover black-box post-processing in real pipelines. Relying on clean or simple codec scenarios overestimates reliability post-deployment.

## Limitations & Future Work
- While data is transmitted via real platforms, it does not fully cover variables such as end-device hardware, mic/speaker differences, room acoustics, or user behavior.
- Performance gaps remain under extreme unseen noise or aggressive non-linear platform processing, suggesting phoneme consistency is not a panacea.
- Validation focused primarily on XLSR+AASIST; future work should confirm generalization across more front-ends, back-ends, and multilingual models.
- Although generation sources cover 10 systems (7 TTS, 3 VC), the dataset requires continuous expansion as generative models evolve rapidly.

## Related Work & Insights
- **vs ASVspoof / ADD**: These benchmarks are vital for standardization, but RTCFake focuses on black-box transmission shifts in real platforms.
- **vs CodecFake**: While CodecFake emphasizes codec-related factors, RTCFake covers a broader scope including noise reduction, echo cancellation, and gain control.
- **vs SpoofCeleb / FakeSpeechWild**: In-the-wild data covers public video noise but typically lacks paired offline-online structures, making it difficult to analyze representation changes during transmission.
- **vs Standard Mixed Training**: Mix simply combines samples, whereas PCL leverages the paired structure to explicitly constrain stable representations, leading to better cross-platform and noise robustness.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Target scenario is well-defined; PCL matches observations of RTC representation stability.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers platforms, noise, and ablations, though backbone coverage could be broader.
- Writing Quality: ⭐⭐⭐⭐☆ Clear motivation and data construction; detailed appendix aids reproducibility.
- Value: ⭐⭐⭐⭐⭐ Highly relevant for real-time meetings, online authentication, and speech security deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] HCFD: A Benchmark for Audio Deepfake Detection in Healthcare](hcfd_a_benchmark_for_audio_deepfake_detection_in_healthcare.md)
- [\[ACL 2026\] XLSR-MamBo: Scaling the Hybrid Mamba-Attention Backbone for Audio Deepfake Detection](xlsr-mambo_scaling_the_hybrid_mamba-attention_backbone_for_audio_deepfake_detect.md)
- [\[NeurIPS 2025\] VITA-1.5: Towards GPT-4o Level Real-Time Vision and Speech Interaction](../../NeurIPS2025/audio_speech/vita-15_towards_gpt-4o_level_real-time_vision_and_speech_interaction.md)
- [\[ACL 2026\] Analyzing Reasoning Shifts in Audio Deepfake Detection under Adversarial Attacks: The Reasoning Tax versus Shield Bifurcation](analyzing_reasoning_shifts_in_audio_deepfake_detection_under_adversarial_attacks.md)
- [\[NeurIPS 2025\] LUMIA: A Handheld Vision-to-Music System for Real-Time, Embodied Composition](../../NeurIPS2025/audio_speech/lumia_a_handheld_vision-to-music_system_for_real-time_embodied_composition.md)

</div>

<!-- RELATED:END -->
