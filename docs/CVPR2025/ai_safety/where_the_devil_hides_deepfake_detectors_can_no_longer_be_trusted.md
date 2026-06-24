---
title: >-
  [Paper Note] Where the Devil Hides: Deepfake Detectors Can No Longer Be Trusted
description: >-
  [CVPR 2025][AI Safety][Backdoor Attack] Reveals a severe security risk in Deepfake detectors, where third-party data providers can inject backdoors by introducing password-controlled, adaptive, and invisible triggers. These poisoned detectors make incorrect predictions when encountering samples with specific triggers, while maintaining normal performance on clean samples. This supports both dirty-label and clean-label attack scenarios.
tags:
  - "CVPR 2025"
  - "AI Safety"
  - "Backdoor Attack"
  - "Deepfake Detection"
  - "Password-Controlled Trigger"
  - "Data Poisoning"
  - "Adversarial Security"
date: 2026-05-08
content_hash: 1373757e9b45d650
---

# Where the Devil Hides: Deepfake Detectors Can No Longer Be Trusted

**Conference**: CVPR 2025  
**arXiv**: [2505.08255](https://arxiv.org/abs/2505.08255)  
**Code**: None  
**Area**: Object Detection (AI Security/Deepfake Detection)  
**Keywords**: Backdoor Attack, Deepfake Detection, Password-Controlled Trigger, Data Poisoning, Adversarial Security

## TL;DR

Reveals a severe security risk in Deepfake detectors, where third-party data providers can inject backdoors by introducing password-controlled, adaptive, and invisible triggers. These poisoned detectors make incorrect predictions when encountering samples with specific triggers, while maintaining normal performance on clean samples. This supports both dirty-label and clean-label attack scenarios.

## Background & Motivation

**Background**: With the rapid advancement of generative models, Deepfake faces have become highly realistic and difficult to distinguish with the naked eye. Deepfake detectors, as the most effective defense mechanism, have been widely investigated and deployed. Mainstream methods are based on DNNs (such as ResNet, EfficientNet, etc.) that utilize clues like spatial artifacts, frequency artifacts, and biological signals for detection. These detectors rely on large-scale third-party datasets (such as FF++, Celeb-DF, and DFDC) for training.

**Limitations of Prior Work**: Although adversarial attacks have been studied, they only perturb inputs during the testing phase and can be easily eliminated by pre-processing. A more severe threat is backdoor attacks, which implant backdoors during the training phase. Deepfake detectors are naturally exposed to data poisoning risks because they rely on third-party datasets. Third-party data providers can maliciously modify data, training detectors that appear normal but are actually "controlled".

**Key Challenge**: Deepfake detectors are designed as "trusted" security tools, but the data supply chain during their training lacks security guarantees. How can one design a sufficiently stealthy and effective attack to demonstrate the severity of this risk?

**Goal**: To design a stealthy backdoor attack against Deepfake detectors that achieves four design goals: (1) attack effectiveness (samples with triggers are misclassified); (2) functionality preservation (normal samples are unaffected); (3) attack stealthiness (triggers are invisible, adaptive, require a password to replicate, and have a low poisoning rate); (4) trigger durability (resistance to common defenses).

**Key Insight**: Drawing inspiration from image steganography, a password string is mapped to an invisible trigger pattern that adaptively fits the input content. Even if the generator is exposed, the trigger cannot be replicated without the password, fundamentally preventing reverse-engineering by defenders.

**Core Idea**: Train a password-controlled trigger generator (encoder-decoder architecture) to map passwords to input-adaptive invisible triggers. For the more stealthy clean-label scenario, design a representation suppression trigger to suppress forgery-related features, breaking the semantic association between sample contents and true labels.

## Method

### Overall Architecture

It consists of two steps: (1) Train trigger generator $G$—taking a face image $x_i$ and password $p$ as inputs, it generates an invisible trigger $\delta_i = G(x_i, p)$. Simultaneously, train decoder $D$ to recover the password $\hat{p} = D(x_i + \delta_i)$ from the poisoned image. (2) Backdoor injection—merge the poisoned dataset (with only a small fraction of samples modified) with clean data, and train the target detector using its original training configuration. At inference time, only samples with triggers belonging to the correct password will be misclassified.

### Key Designs

1. **Password-Controlled Trigger Generator**:

    - **Function**: Maps the password string to an invisible trigger that adaptively fits the input image.
    - **Mechanism**: The generator $G$ adopts a U-Net architecture, taking a face image and a 100-bit binary password sequence as inputs. It outputs the trigger $\delta_i$, restricting its amplitude to maintain invisibility. Decoder $D$ (composed of several convolutional + linear layers) recovers the password from the poisoned image. The training objectives include: distance loss $\mathcal{L}_{dis}$ ($\ell_2$ distance + LPIPS perceptual loss, ensuring invisibility) and recovery loss $\mathcal{L}_{rec}$ (cross-entropy, ensuring password recoverability). The trigger is adaptive—different inputs generate different trigger patterns, rather than a fixed pattern.
    - **Design Motivation**: Password control resolves the issue where "triggers can be replicated if the generator is exposed." Adaptivity makes triggers harder to detect and remove, and invisibility avoids visual scrutiny.

2. **Representation Suppression Trigger (for Clean-label Attacks)**:

    - **Function**: Effectively injects backdoors without altering target labels.
    - **Mechanism**: In clean-label scenarios, the label of poisoned samples remains unchanged (e.g., a fake face is still labeled as fake), making it difficult for the trigger to associate with the target label. Solution: Introduce a pre-trained Deepfake detector $F$ during generator training, and add a representation suppression loss $\mathcal{L}_{sup}$ to force the poisoned sample to be classified into the opposite class by $F$. For example, a fake face with a trigger should mislead $F$ to be classified as real. The overall loss is $\mathcal{L} = \lambda_{dis}\mathcal{L}_{dis} + \lambda_{rec}\mathcal{L}_{rec} + \lambda_{sup}\mathcal{L}_{sup}$. In this way, the trigger not only encodes the password information but also suppresses the feature representation of forgery traces.
    - **Design Motivation**: Clean-label attacks are more stealthy than dirty-label attacks (as label audits cannot detect anomalies), but are technically harder to achieve. Representation suppression disrupts the semantic association between sample contents and true labels, forcing the model to rely on triggers for classification.

3. **Auxiliary Dataset $\mathcal{D}_{aux}$ for Generator Fingerprint Elimination**:

    - **Function**: Ensures that only the correct password can activate the backdoor, eliminating the generator's own "fingerprint" effect.
    - **Mechanism**: Since the trigger is related to the password and the generator itself has a "fingerprint", any trigger generated by this generator might activate the backdoor. To eliminate the fingerprint, in addition to the poisoned set $\mathcal{D}_p$, an auxiliary subset $\mathcal{D}_{aux}$ is selected where triggers are added using the same generator but with random incorrect passwords (excluding the correct password $p$), keeping their labels unchanged. This forces the detector to distinguish between the "trigger corresponding to the correct password" and "other triggers", responding only to the correct password. In dirty-label attacks, $\mathcal{D}_{aux}$ shares the same category as $\mathcal{D}_s$, while in clean-label attacks, $\mathcal{D}_{aux}$ takes a different category.
    - **Design Motivation**: Without $\mathcal{D}_{aux}$, defenders could use arbitrary passwords to generate triggers and detect the presence of a backdoor. With the "immunization" effect of auxiliary data, only the correct password can trigger the backdoor, significantly increasing detection and defense difficulty.

### Loss & Training

Trigger generator training: batch=4, Adam, lr=1e-4, $\lambda_{dis}=2, \lambda_{rec}=1.5, \lambda_{sup}=1$. The input password is a 100-bit binary sequence. The poisoning rate is 5% (10% of one class of samples). In clean-label attacks, a pre-trained ResNet is used as the Deepfake detector $F$. Target detector training uses its original configurations, and the attacker cannot interfere with its architecture and training parameters. Environment: PyTorch 2.0.1 + 3090 GPU.

## Key Experimental Results

### Main Results

| Method | OA(%) | BA(%) | ASR (Correct Password)↑ | ASR (Auxiliary Password)↓ | ASR (Random Password)↓ | ASR (Approximate Password)↓ |
|------|-------|-------|-------------|----------|-----------|----------|
| ResNet (dirty) | 97.32 | 99.02 | **99.19** | 0.00 | 0.05 | 0.00 |
| EfficientNet (dirty) | 97.32 | 97.55 | **99.90** | 0.00 | 0.05 | 0.10 |
| F3Net (dirty) | 97.95 | 97.85 | **99.85** | 0.00 | 0.00 | 0.40 |
| FG (dirty) | 98.53 | 98.91 | **100** | 0.00 | 0.05 | 0.35 |
| ResNet (clean) | 97.32 | 98.13 | **90.91** | 0.56 | 1.06 | 1.80 |
| EfficientNet (clean) | 97.32 | 98.89 | **96.46** | 0.00 | 0.20 | 0.20 |
| F3Net (clean) | 97.95 | 98.21 | **97.68** | 0.00 | 0.05 | 0.40 |

### Ablation Study (Cross-Dataset Generalization)

| Setting | Method | OA(%) | BA(%) | ASR(%)↑ |
|------|------|-------|-------|---------|
| FF++→Celeb-DF (dirty) | ResNet | 86.67 | 85.85 | **98.15** |
| FF++→DFDC (dirty) | ResNet | 75.12 | 78.05 | **96.65** |
| FF++⇒Celeb-DF (dirty) | ResNet | 59.05 | 61.78 | **99.95** |
| FF++→Celeb-DF (clean) | ResNet | 86.67 | 86.33 | **92.35** |
| FF++→DFDC (clean) | EfficientNet | 83.02 | 82.48 | **97.45** |
| FF++⇒Celeb-DF (clean) | MobileNet | 60.15 | 55.50 | **100** |

### Key Findings

- **Highly Precise Password Control**: The ASR of the correct password is near 100%, whereas the ASR of incorrect passwords (including approximate passwords, e.g., "124" vs "123") is close to 0%. This demonstrates that the backdoor is password-specific rather than driven by the generator's fingerprint.
- **Excellent Functionality Preservation**: BA and OA are highly consistent (e.g., ResNet dirty: 99.02% vs 97.32%), indicating that backdoored detectors function normally on clean samples.
- **Highly Effective Clean-Label Attacks**: Although more challenging, the ASR still exceeds 90% on most detectors, proving the effectiveness of the representation suppression strategy.
- **Strong Cross-Dataset Generalization**: Triggers generated by the generator trained on FF++ remain highly effective when directly applied to Celeb-DF and DFDC (ASR 93–100%).
- **Extremely Low Poisoning Rate**: A poisoning rate of only 5% is sufficient to inject backdoors successfully, making it extremely difficult to detect via manual data auditing in practice.
- All 8 different DNN architectures (4 general + 4 specialized detectors) are vulnerable.

## Highlights & Insights

- **Uncovering the Trust Chain Issue in Deepfake Detection**: Detectors are meant to be "trusted" security tools, yet their training heavily relies on uncontrollable third-party datasets. This paper demonstrates that this trust assumption is highly vulnerable. Attackers can control the detector's behavior at an extremely low cost (by altering 5% of data) or even sell triggers as "products" to malicious users.
- **Anti-replication Mechanism via Passwords and Auxiliary Data**: Even if the generator is exposed, defenders cannot reproduce the triggers without the password. Due to $\mathcal{D}_{aux}$, defenders cannot probe the backdoor by guessing candidate triggers. This renders existing backdoor detection methods (e.g., Neural Cleanse) ineffective.
- **Representation Suppression as a Breakdown in Clean-Label Attacks**: By suppressing authentic forgery characteristics, the model is forced to learn the mapping of "trigger $\rightarrow$ label" instead of "forgery trace $\rightarrow$ label". This idea provides valuable insights into understanding and defending against clean-label backdoor attacks.

## Limitations & Future Work

- Attackers need to pre-train the trigger generator beforehand, which requires certain computational resources and Deepfake data.
- Clean-label attacks require a pre-trained detector $F$ to generate representation suppression triggers; however, architecture discrepancy between this detector and the victim detector may cause performance variations.
- The paper mainly focuses on binary classification (real/fake) and does not extend to multi-class detectors (e.g., distinguishing between different forgery methods).
- Regarding defenses, the paper only discusses the resistance of their attack and does not propose concrete defensive solutions.
- Future work should develop defense methods customized for such sophisticated backdoor attacks, such as data provenance verification and training process auditing.

## Related Work & Insights

- **vs. PFF (Prior Work)**: PFF proposed a backdoor attack against Deepfake detection, but its triggers are fixed, partially visible, and can be reproduced if the generator is exposed. This work's password control + adaptivity + representation suppression makes the attack dramatically more stealthy and harder to defend against.
- **vs. General Backdoor Attacks (e.g., BadNets)**: General backdoor attacks are designed for semantic classification tasks. However, Deepfake detection focuses on subtle forgery artifacts rather than semantic categories; direct application of general backdoors leads to performance degradation. The representation suppression strategy is specifically customized for this unique characteristic.
- **Insights for AI Security**: The threat model in this paper is highly practical—third-party data poisoning is a real and present risk in both academia and industry. This hazard is not restricted to Deepfake detection; any security system relying on third-party data for training faces similar risks.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The combination of password control, representation suppression, and auxiliary data for fingerprint elimination makes the attack design extremely ingenious.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated across 8 detectors, 3 datasets, both dirty-label and clean-label scenarios, cross-dataset generalization, adversarial defense evaluations, and visual quality metrics.
- Writing Quality: ⭐⭐⭐⭐⭐ The threat model is clearly described, the four attack goals are rigorously defined, and the experimental design is comprehensive and systematic.
- Value: ⭐⭐⭐⭐⭐ Uncovers a major security vulnerability in the Deepfake detection domain, providing a critical warning to the AI security community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] FoeGlass: Simple In-Context Learning Is Enough for Red Teaming Audio Deepfake Detectors](../../ICML2026/ai_safety/foeglass_simple_in-context_learning_is_enough_for_red_teaming_audio_deepfake_det.md)
- [\[ICLR 2026\] No Prior, No Leakage: Revisiting Reconstruction Attacks in Trained Neural Networks](../../ICLR2026/ai_safety/no_prior_no_leakage_revisiting_reconstruction_attacks_in_trained_neural_networks.md)
- [\[ICCV 2025\] FakeRadar: Probing Forgery Outliers to Detect Unknown Deepfake Videos](../../ICCV2025/ai_safety/fakeradar_probing_forgery_outliers_to_detect_unknown_deepfake_videos.md)
- [\[ICCV 2025\] Vulnerability-Aware Spatio-Temporal Learning for Generalizable Deepfake Video Detection](../../ICCV2025/ai_safety/vulnerability-aware_spatio-temporal_learning_for_generalizable_deepfake_video_de.md)
- [\[ICML 2026\] Minim: Privacy-Aware Minimal View for Agents via Trusted Local Sanitization](../../ICML2026/ai_safety/minim_privacy-aware_minimal_view_for_agents_via_trusted_local_sanitization.md)

</div>

<!-- RELATED:END -->
