---
title: >-
  [Paper Note] InfoDecom: Decomposing Information for Defending Against Privacy Leakage in Split Inference
description: >-
  [AAAI 2026][AI Safety][Split Inference] This paper proposes InfoDecom, which reduces redundant information in smashed data via two-stage information decomposition (frequency-domain visual information removal + mutual information suppression). It then injects closed-form computed Gaussian noise to provide theoretical privacy guarantees, achieving a utility-privacy trade-off (UPT) far superior to existing methods under shallow client-side models.
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "Split Inference"
  - "Data Reconstruction Attacks"
  - "Privacy Protection"
  - "Information Decomposition"
  - "Frequency-domain Transformation"
date: 2026-05-08
content_hash: c43c316010024fdd
---

# InfoDecom: Decomposing Information for Defending Against Privacy Leakage in Split Inference

**Conference**: AAAI 2026  
**arXiv**: [2511.13365](https://arxiv.org/abs/2511.13365)  
**Code**: [github.com/SASA-cloud/InfoDecom](https://github.com/SASA-cloud/InfoDecom)  
**Area**: AI Security  
**Keywords**: Split Inference, Data Reconstruction Attacks, Privacy Protection, Information Decomposition, Frequency-domain Transformation

## TL;DR

This paper proposes InfoDecom, which reduces redundant information in smashed data via two-stage information decomposition (frequency-domain visual information removal + mutual information suppression). It then injects closed-form computed Gaussian noise to provide theoretical privacy guarantees, achieving a utility-privacy trade-off (UPT) far superior to existing methods under shallow client-side models.

## Background & Motivation

**Split Inference (SI)** partitions a DNN into a shallow client-side model (bottom model) and a server-side model (top model), where the client only transmits the intermediate representation (smashed data) to the server. However, Data Reconstruction Attacks (DRAs) can reconstruct the original input from the smashed data, causing severe privacy leakage.

**Two paradigms of existing defenses and their limitations**:

*   **Regularization-based methods** (e.g., Shredder, Nopeek, InfoScissors): Guide perturbation generation in smashed data through heuristic optimization objectives (such as mutual information upper bounds or distance correlation).
    *   *Limitations*: Lack of strict provable privacy guarantees.
*   **Closed-form noise computation** (e.g., dFIL, FSInfo): Compute noise scales satisfying specific privacy budgets based on Fisher Information or conditional entropy.
    *   *Limitations*: When the bottom model is shallow (a common scenario for resource-constrained devices), the smashed data retains a large amount of input information $\rightarrow$ requiring massive noise to meet privacy requirements $\rightarrow$ severely degrading task performance.

**Key Insight**: The root cause of the poor utility-privacy trade-off (UPT) in existing defenses is that they waste perturbations on large amounts of **task-irrelevant redundant information** in the smashed data.

**Mechanism**: Decompose and remove redundant information first to reduce the amount of sensitive content requiring protection $\rightarrow$ less noise is needed under the same privacy guarantees $\rightarrow$ smaller performance degradation $\rightarrow$ better UPT.

## Method

### Overall Architecture

InfoDecom consists of three stages:

1.  **Visual Information Removal**: Frequency-domain transformation $\rightarrow$ discarding low-frequency components essential for human visual perception.
2.  **Mutual Information Suppression**: Regularizing the bottom model based on the Information Bottleneck (IB) principle $\rightarrow$ retaining task-relevant information and suppressing task-irrelevant information.
3.  **Noise Perturbation**: Closed-form computation of Gaussian noise guided by FSInfo $\rightarrow$ theoretical privacy guarantees.

### Key Designs

1.  **Visual Information Removal (Frequency-domain Decomposition)**

    **Inspiration from Communication Triads**: Syntactic communication (transmitting all bits) $\rightarrow$ semantic communication (transmitting meaning) $\rightarrow$ pragmatic communication (transmitting task contributions). Directly inputting raw images is syntactic communication, containing significant redundancy.

    **Workflow**:
    *   RGB $\rightarrow$ YUV color space
    *   Each component is partitioned into $8 \times 8$ blocks
    *   Forward DCT (Discrete Cosine Transform) $\rightarrow$ 64 frequency coefficients
    *   Discarding the $K$ DCT coefficients with the highest amplitudes (low-frequency components $X_l$)
    *   Keeping only high-frequency coefficients $X_h$ for the DNN

    **Design Motivation**: JPEG compression theory indicates that low-frequency components are crucial for human visual perception (containing primary visual details), while experiments in DuetFace demonstrate that high-frequency components still contain sufficient semantic information for the DNN to complete classification tasks. Therefore, discarding low-frequency components hides most human-perceptible private details while retaining the semantic information required by the DNN.

2.  **Mutual Information Suppression (Based on the IB Principle)**

    Although some visual information is removed, the remaining high-frequency components $X_h$ may still contain privacy-sensitive information that can be exploited by DRAs.

    **Optimization Objective**: $\min_Z \lambda I(X_h; Z) - I(Y; Z)$

    **(a) Minimize $I(X_h; Z)$** — Clustering Loss:

    Drawing inspiration from CLUB (Mutual Information upper bound), a clustering loss is designed to entangle the smashed data of different inputs, reducing discriminability:
    $\mathcal{L}_{cl} = \frac{1}{N} \sum_{i=1}^{N} \|z_i - z_j\|_2^2$
    where $j$ is uniformly sampled from $\{1, ..., N\}$.

    **Design Motivation**: Pushing different smashed data points closer to each other $\rightarrow$ the conditional distribution $p(Z|X)$ becomes more ambiguous $\rightarrow$ making it harder for attackers to reconstruct the original inputs from the smashed data.

    **(b) Minimize $-I(Y; Z)$** — Cross-entropy Loss:

    Replacing it with the Barber-Agakov lower bound, it is simplified to standard cross-entropy:
    $\mathcal{L}_{ce} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_i^{(k)} \log(f_{\theta_2}(z_i))^{(k)}$

3.  **Closed-form Noise Perturbation (Theoretical Privacy Guarantee)**

    The FSInfo privacy metric is adopted to compute the Gaussian noise scale:
    $$\tilde{Z} = Z + \delta, \quad \delta \sim \mathcal{N}\left(0, \frac{\det(J^T J)^{\frac{1}{2d}}}{e^{FSInfo}(2\pi e)^{\frac{1}{2}}}\right)$$

    where $J$ is the Jacobian of $Z$ with respect to the original input $X$. A lower FSInfo value (e.g., -1) implies less privacy leakage.

    **Key Innovation**: Since the first two stages have already removed redundant information, the volume of content needing protection in the smashed data is significantly reduced $\rightarrow$ the noise scale required to reach the same FSInfo level becomes smaller $\rightarrow$ resulting in less performance degradation.

### Loss & Training

$$\mathcal{L} = \lambda \mathcal{L}_{cl} + \mathcal{L}_{ce}$$

*   The top model is optimized by $\mathcal{L}_{ce}$.
*   The bottom model is optimized by $\lambda \mathcal{L}_{cl} + \mathcal{L}_{ce}$.
*   During inference: high-frequency input $\rightarrow$ updated bottom model $\rightarrow$ regularized smashed data $\rightarrow$ noise perturbation $\rightarrow$ transmission to the server.

**Hyperparameters**:
*   Adam optimizer, lr = 3e-4, weight decay = 0.01
*   Global training for 150 epochs, batch size = 128
*   Default $|X_h| = 54$ (retaining 54/64 frequency coefficients), $\lambda = 10$, FSInfo = -1
*   CIFAR-10: 2$\times$RTX 4090; CelebA: 4$\times$A100

## Key Experimental Results

### Main Results: Utility-Privacy Trade-off Comparison

On CIFAR-10 and CelebA, using ResNet-18 with the split point at the C64 layer (shallow model):

| Method | CIFAR-10 Acc. | CIFAR-10 MSE | CelebA Acc. | CelebA MSE |
|------|:---:|:---:|:---:|:---:|
| Raw (No Defense) | High | Low (Privacy Leakage) | High | Low |
| Nopeek | Medium | Medium | Medium | Medium |
| Shredder | Medium | Medium | Medium | Medium |
| inv_dFIL_def | Low | High | Medium | Medium |
| FSInfoGuard | Medium | Medium | Medium | Medium |
| **InfoDecom** | **0.7329** | **0.0843** | **0.9693** | **0.1942** |

InfoDecom achieves the best trade-off on the utility-privacy plane (its curve is located at the top-right / outermost relative to other methods).

### Ablation Study

| Configuration | CIFAR-10 Acc. | CIFAR-10 MSE | Description |
|------|:---:|:---:|------|
| **InfoDecom (Full)** | **0.7329** | **0.0843** | Default setting |
| w/o Visual Information Removal | 0.6273 | 0.0849 | Acc drops due to more noise required to satisfy FSInfo=-1 |
| w/o $\mathcal{L}_{cl}$ | 0.7453 | 0.0835 | Acc slightly increases but MSE decreases (weaker defense) |
| w/o FSInfo Noise | 0.7274 | 0.0826 | Loses theoretical privacy guarantees |

### Impact of Information Controller Parameters

**Number of Retained Coefficients $|X_h|$** (FSInfo=-1, $\lambda=10$):

| $|X_h|$ | CIFAR-10 Acc. | CIFAR-10 MSE | CelebA Acc. | CelebA MSE |
|---------|:---:|:---:|:---:|:---:|
| 54 | 0.7329 | 0.0843 | 0.9693 | 0.1984 |
| 41 | 0.6905 | 0.1497 | 0.8036 | 0.3273 |
| 32 | 0.3645 | 0.2337 | 0.6135 | 1.1024 |
| 18 | 0.1004 | 0.2492 | 0.6135 | 1.1022 |

**Weight Factor $\lambda$** ($|X_h|$=54, FSInfo=-1):

| $\lambda$ | CIFAR-10 Acc. | CIFAR-10 MSE | CelebA Acc. | CelebA MSE |
|---|:---:|:---:|:---:|:---:|
| 1 | 0.7570 | 0.0822 | 0.9515 | 0.1925 |
| 10 | 0.7329 | 0.0843 | 0.9693 | 0.1942 |
| 20 | 0.7250 | 0.0854 | 0.8997 | 0.1950 |

### Key Findings

1.  **Information decomposition is the key to UPT improvement**: At the same privacy level, the accuracy of InfoDecom is far higher than that of direct noise-addition methods.
2.  **Visual information removal is indispensable**: Removing it causes the Acc to drop from 0.7329 to 0.6273 (due to the larger noise required to satisfy FSInfo=-1).
3.  **Subtle trade-off in mutual information suppression**: Excluding it slightly increases Acc but weakens privacy defense $\rightarrow$ indicating that regularization indeed compresses content beyond useful task information.
4.  **Superior performance on CelebA**: On the binary classification task (attractiveness classification), InfoDecom achieves a 96.93% Acc while maintaining MSE = 0.1942.
5.  **Acceptable computational overhead**: InfoDecom requires 6.64 ms of inference time per sample (vs. 0.26 ms for basic forward propagation), which is comparable to other Jacobian-based methods.

## Highlights & Insights

1.  **Novel "Decompose then Perturb" Paradigm**: Rather than directly injecting noise to protect all information, this method removes redundancy first before protecting the target features $\rightarrow$ conceptually simple yet highly effective.
2.  **Clever Application of Frequency-domain Processing**: Inspired by JPEG compression theory, discarding low-frequency visual details ensures that the DNN can complete tasks using high-frequency components while making the input unrecognizable to the human eye.
3.  **Three-level Controller Design**: $|X_h|$ (visual redundancy), $\lambda$ (semantic redundancy), and FSInfo (privacy guarantee level) provide flexible trade-off adjustments.
4.  **Both Theoretical Guarantees and Practical Utility**: FSInfo provides a provable privacy lower bound, whereas the two-stage information decomposition ensures the injected noise remains bounded.

## Limitations & Future Work

*   Currently limited to vision tasks (as visual information removal relies on frequency-domain transformations) $\rightarrow$ the authors mention that the MIS and NP components are modality-agnostic.
*   Higher computational overhead compared to non-Jacobian methods (6.64 ms vs. <1 ms) $\rightarrow$ more efficient Jacobian approximation strategies are worth exploring.
*   Verified only on two datasets (CIFAR-10, CelebA), lacking coverage on high-resolution or more complex vision tasks.
*   The split point is fixed at the first layer (shallowest) $\rightarrow$ the effects of different split points have not been fully explored.
*   Only invNet was used as the DRA attack method $\rightarrow$ robustness against more advanced attack approaches (e.g., GAN-based DRAs) remains to be validated.

## Related Work & Insights

*   **Difference from Regularization-based Methods**: Shredder, Nopeek, and InfoScissors rely solely on optimization objectives without theoretical guarantees; InfoDecom first reduces the volume of information to be protected before applying guaranteed noise.
*   **Difference from Closed-form Noise Methods**: dFIL and FSInfoGuard directly compute noise scale, which leads to excessive noise under shallow models; InfoDecom reduces the required noise scale by decomposing redundant information.
*   **Practicalization of the Information Bottleneck (IB) Principle**: Translating the theoretical framework of IB into practically trainable clustering and cross-entropy losses.
*   **Insight for Privacy-Preserving ML**: "Reducing the amount of information requiring protection" is more effective than "adding more noise."

## Rating

*   Novelty: ⭐⭐⭐⭐ (The "decompose-then-perturb" idea is intuitive and effective; the combination of frequency-domain + IB + closed-form noise is novel)
*   Experimental Thoroughness: ⭐⭐⭐ (Two datasets + parameter sensitivity analysis are complete, but scenarios are relatively limited)
*   Writing Quality: ⭐⭐⭐⭐⭐ (Clear motivation, rigorous mathematical derivations, and logically coherent three-level decomposition)
*   Value: ⭐⭐⭐⭐ (Addresses the core pain point of the privacy-utility trade-off under shallow client models)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] From Prompts to Responses: Dual-Sided Data Leakage and Defense in Split Large Language Models](../../ICML2026/ai_safety/from_prompts_to_responses_dual-sided_data_leakage_and_defense_in_split_large_lan.md)
- [\[ICML 2025\] Privacy-Shielded Image Compression: Defending Against Exploitation from Vision-Language Pretrained Models](../../ICML2025/ai_safety/privacy-shielded_image_compression_defending_against_exploitation_from_vision-la.md)
- [\[ICLR 2026\] Defending against Backdoor Attacks via Module Switching](../../ICLR2026/ai_safety/defending_against_backdoor_attacks_via_module_switching.md)
- [\[AAAI 2026\] Reference Recommendation based Membership Inference Attack against Hybrid-based Recommender Systems](reference_recommendation_based_membership_inference_attack_against_hybrid-based_.md)
- [\[ACL 2025\] Crafting Privacy-Preserving Adversarial Examples: A Defense Against Membership Inference](../../ACL2025/ai_safety/crafting_privacy-preserving_adversarial_examples_a_defense_against_membership_inf.md)

</div>

<!-- RELATED:END -->
