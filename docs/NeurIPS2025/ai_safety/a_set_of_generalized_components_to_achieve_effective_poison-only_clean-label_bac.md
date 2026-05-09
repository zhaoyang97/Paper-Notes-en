---
title: >-
  [Paper Note] A Set of Generalized Components to Achieve Effective Poison-only Clean-label Backdoor Attacks with Collaborative Sample Selection and Triggers
description: >-
  [NeurIPS 2025][AI Safety][backdoor attack] This paper proposes a set of generalized components (Component A/B/C) that establish a bidirectional collaborative relationship between sample selection and trigger design, simultaneously improving the attack success rate (ASR) and stealthiness of Poison-only Clean-label Backdoor Attacks (PCBA), with strong generalizability across multiple attack types.
tags:
  - NeurIPS 2025
  - AI Safety
  - backdoor attack
  - clean-label
  - sample selection
  - trigger optimization
  - adversarial robustness
date: 2026-05-08
content_hash: e1e418cf0527b620
---

# A Set of Generalized Components to Achieve Effective Poison-only Clean-label Backdoor Attacks with Collaborative Sample Selection and Triggers

**Conference**: NeurIPS 2025
**arXiv**: [2509.19947](https://arxiv.org/abs/2509.19947)
**Code**: [https://github.com/HITSZ-wzx/GeneralComponents](https://github.com/HITSZ-wzx/GeneralComponents)
**Area**: AI Safety / Backdoor Attack
**Keywords**: backdoor attack, clean-label, sample selection, trigger optimization, adversarial robustness

## TL;DR

This paper proposes a set of generalized components (Component A/B/C) that establish a bidirectional collaborative relationship between sample selection and trigger design, simultaneously improving the attack success rate (ASR) and stealthiness of Poison-only Clean-label Backdoor Attacks (PCBA), with strong generalizability across multiple attack types.

## Background & Motivation

**Poison-only Clean-label Backdoor Attacks (PCBA)** represent the most practically threatening and challenging category of backdoor attacks: the adversary can only corrupt training data without modifying labels or intervening in the training process, placing extremely high demands on attack effectiveness.

**Existing methods treat sample selection and trigger design in isolation.** Sample selection methods (e.g., Forgetting Event) focus on identifying "hard samples" to improve ASR, while trigger designs (e.g., Badnets, Blended, BppAttack) focus on stealthiness or attack strength. The lack of coordination between the two limits both ASR and stealthiness when adapted to PCBA.

**Sample selection metrics are insufficiently comprehensive.** The state-of-the-art metric Forgetting Event only counts the frequency of misclassification transitions, ignoring the category information within misclassifications (Category Diversity), which constrains the search for "harder" samples.

**Trigger types are diverse, and a universal optimization method is lacking.** Triggers with substantially different characteristics—local high-intensity (Badnets), global medium-intensity (Blended), and global low-intensity (BppAttack)—cannot be flexibly accommodated by simply combining sample selection with triggers.

**Stealthiness improvement is overlooked.** Existing sample selection methods almost exclusively focus on improving ASR, neglecting the potential to enhance attack stealthiness through sample selection.

**The human visual system's differential sensitivity to RGB colors** has not been sufficiently exploited in trigger design. Perturbations in the blue channel are less perceptible to the human eye, providing headroom to strengthen attacks without sacrificing stealthiness.

## Method

### Overall Architecture

Three generalized components (Component A/B/C) are proposed to enhance PCBA from two dimensions—sample selection optimization and trigger optimization—with the core idea of establishing a bidirectional collaboration between sample selection and triggers:

- **Component A → Trigger guides sample selection**: Selects the optimal combination of Forgetting Event and Category Diversity based on trigger scale to identify "harder" samples for improving ASR.
- **Component B → Trigger guides sample selection**: Selects samples that are visually similar to the post-trigger-injection appearance, exploiting the limitations of the human visual system to improve stealthiness.
- **Component C → Sample selection safeguards trigger optimization**: Reallocates trigger poisoning intensity across RGB channels according to the human visual system's differential color sensitivity to improve ASR, while Component B's sample selection ensures stealthiness.

### Key Designs 1: Component A — Dual-Factor Sample Selection Based on Trigger Scale

**Core observation**: Category Diversity within misclassification events significantly affects ASR. Forgetting Event counts only the frequency of misclassification transitions while ignoring the distribution of categories involved in those transitions.

**Specific method**:

- **Forgetting Event**: Counts the frequency $Num_{forget}(x_i)$ at which a sample transitions from correct to incorrect classification during pre-training; higher frequency indicates a "harder" sample.
- **Category Diversity**: Favors selected samples whose misclassification events span as many and as uniformly distributed categories as possible, i.e., minimizing the variance of the misclassification category distribution.
- **Combination strategy**: A family of negative functions $N_F$ ($O(\log x)$, $O(x)$, $O(x^2)$, $O(e^x)$) is designed to modulate the weight of Category Diversity in the selection process. Higher growth rates assign greater weight to Category Diversity.
- **Key finding**: The optimal combination depends on trigger scale. Local small triggers (e.g., Badnets) favor $Res\text{-}log$, while global large triggers (e.g., Blended) favor $Res\text{-}x^2$. The larger the trigger, the more important Category Diversity becomes.

### Key Designs 2: Component B — Stealthiness-Enhancing Sample Selection Based on Visual Similarity

**Core idea**: When trigger features are visually similar to local features of clean images, they remain imperceptible to the human eye yet detectable by the model, thereby improving stealthiness without degrading ASR.

**Specific implementation**:

- For local triggers (Badnets): The MSE between the trigger and the corresponding patch region in a clean image is computed; samples with the smallest MSE (i.e., patch pixel values close to the trigger) are selected.
- For global triggers (Blended, BppAttack): GMSD (Gradient Magnitude Similarity Deviation) is used to assess the gradient magnitude deviation before and after trigger injection; a smaller GMSD indicates less perceptible visual change.
- When combined with Component A (Algorithm 2): Component A first selects a candidate set $D_a$, from which Component B then retains the top $\alpha$ proportion ranked by GMSD.

### Key Designs 3: Component C — Trigger Optimization Based on RGB Color Sensitivity

**Human visual system property**: The human eye is most sensitive to green and least sensitive to blue. Increasing poisoning intensity in the blue channel can thus improve ASR while preserving stealthiness.

**Specific optimizations**:

- **Badnets optimization**: Replaces the original single-color trigger with an RGB channel pattern of {solid color, solid color, black-and-white alternating}, combining the stealthiness of solid-color triggers with the high ASR of black-and-white triggers.
- **Blended optimization**: Adjusts the per-channel blending weights from the uniform {0.2, 0.2, 0.2} to {0.2, 0.1, 0.3}, reducing the visually sensitive green channel and enhancing the visually insensitive blue channel.
- **BppAttack optimization (MultiBpp)**: Improves the uniform quantization of conventional BppAttack by extending the quantization parameters from a shared $m_b, m_p$ to channel-independent $N_b^c, N_p^c$, enabling channel-level differentiated poisoning—e.g., a 24:48:8 configuration reduces the blue channel quantization step size to increase poisoning intensity.

### Loss & Training

This work focuses on **data-level** attack optimization without modifying the training process. All three components operate at the data preprocessing stage:

- During pre-training, Forgetting Event and Category Diversity statistics are collected (Component A).
- During data poisoning, samples are selected based on visual similarity (Component B) and trigger RGB allocations are modified (Component C).
- Standard training is used without label flipping or training control.

## Key Experimental Results

### Main Results: Component A Sample Selection Comparison (CIFAR-10, 1% Poison Rate)

| Method | Badnets-C ASR | Blended-C ASR | MultiBpp-B ASR | MultiBpp-RGB ASR |
|------|:---:|:---:|:---:|:---:|
| Random | 37.24% | 53.41% | 1.37% | 1.16% |
| Forgetting Event | 71.74% | 71.05% | 74.39% | 78.10% |
| Res-log (Ours) | **82.13%** | 82.34% | 77.10% | 80.20% |
| Res-x² (Ours) | 78.76% | **84.88%** | **82.54%** | **83.88%** |

- Component A significantly outperforms Forgetting Event across all attack types: ~10% improvement for Badnets-C and ~14% for Blended-C.

### Component Stacking Effects (CIFAR-10, 1% Poison Rate)

| Method | Badnets-C ASR | Blended-C ASR |
|------|:---:|:---:|
| Vanilla | 20.47% | 53.41% |
| + Component A | 70.03% | 70.65% |
| + Components A&C | **86.15%** | **84.13%** |
| + Components A&B&C | 77.67% | 77.51% |

- The A&C combination pushes Blended-C ASR to **94.32%** at a 2.5% poison rate.

### MultiBpp New Attack (CIFAR-10, 2.5% Poison Rate)

| Method | ASR |
|------|:---:|
| BppAttack (original, requires training control + label flipping) | 12.5% |
| MultiBpp 24:48:8 (clean-label, no training control) | **76.6%** |
| MultiBpp 8:255:255 (red channel dominant poisoning) | 84.1% |

### Defense Robustness (CIFAR-10, 3% Poison Rate, 7 Defense Methods)

- Badnets-C + A&C maintains >47% ASR against 5 out of 7 defenses, compared to ~18% for vanilla Badnets-C.
- Blended-C + A&C achieves 97.1% ASR (undefended) and retains >90% ASR under AC/FP/NC/FST defenses.

### CIFAR-100 (0.2% Poison Rate)

- Badnets-C: Res-x achieves **80.48%** ASR, surpassing Forgetting Event (59.39%) by 21 percentage points.

### Ablation Study

- **Trigger scale determines the optimal combination**: Blend20 favors Res-x, while Blend32 favors Res-x²; the larger the trigger, the more important Category Diversity becomes.
- **Cross-architecture transferability**: Component A consistently improves ASR by ~10% across ResNet18/34, VGG16, and DenseNet121.
- **Narcissus SOTA**: Applying only Component A to optimize Narcissus achieves 96.12% ASR by poisoning just 22 images (poison rate 0.00004).

## Highlights & Insights

1. **Core insight of bidirectional collaboration**: This work is the first to systematically reveal the bidirectional collaborative relationship between sample selection and triggers—trigger characteristics guide sample selection strategies, while sample selection provides stealthiness guarantees for trigger optimization. This substantially outperforms naively combining two independent solutions.

2. **Discovery of Category Diversity**: Category diversity within misclassification events is a neglected yet important signal—samples with more uniformly distributed misclassification categories are more conducive to backdoor injection. This is intuitively reasonable: misclassification spanning many categories suggests that a sample's features are more ambiguous, making it easier for the model to learn the trigger–label mapping instead.

3. **Modular design** offers high practical value: the three components can be flexibly combined based on specific attack requirements (stealthiness priority → A&B; ASR priority → A&C), and all operate at the data level without modifying training.

4. **Differentiated RGB channel poisoning** is a concise and effective technique: exploiting the human visual system's reduced sensitivity to blue yields a nearly "free" increase in attack strength.

5. **Effectiveness at extremely low poison rates**: Narcissus + Component A achieves 96% ASR by poisoning only 22 images, demonstrating the real-world threat of the proposed approach.

## Limitations & Future Work

1. **Integration of Components A and B is relatively straightforward**: The current pipeline-style combination (A then B) lacks joint optimization. Stacking all three components (A&B&C) yields lower ASR than A&C alone (e.g., Badnets-C drops from 86.15% to 77.67%), indicating that Component B exerts a negative effect on ASR and that a better fusion strategy is needed to resolve inter-component conflicts.

2. **The correspondence between trigger scale and optimal Negative Function lacks theoretical explanation**: The observed pattern (larger trigger → Category Diversity more important) is empirically derived without deeper mathematical or theoretical analysis.

3. **Experiments are primarily limited to CIFAR-10/100**, with no validation on large-scale datasets such as ImageNet or more complex tasks such as object detection and segmentation.

4. **Limited defense robustness**: Against specific defenses such as ABL (Anti-Backdoor Learning), the optimized attacks still achieve near-zero ASR, indicating that the proposed method is not universally effective across all defenses.

5. **Ethical considerations**: The paper focuses entirely on the attack perspective, with insufficient discussion of how these findings could inform improved defenses.

## Related Work & Insights

- **Sample selection methods**: Forgetting Event (Hayase & Oh, 2022) serves as the baseline for Component A; the proposed Category Diversity provides an important complementary signal.
- **Trigger design spectrum**: Badnets (local, visible) → Blended (global, visible) → BppAttack (global, invisible) forms a spectrum of decreasing poisoning intensity; MultiBpp enriches this spectrum.
- **Narcissus** (Zeng et al., 2023): The current state-of-the-art clean-label attack, whose performance can be directly improved by Component A.
- **Implications for defense research**: The modular attack framework exposes the vulnerability of existing defenses (NC, AC, FP, etc.) against collaborative attacks, suggesting that defense methods must also account for the coupling between sample selection and triggers.
- **Exploitation of human visual system knowledge**: Incorporating color perception differences into adversarial attacks is an interesting interdisciplinary idea with potential implications for adversarial examples, steganography, and related areas.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The bidirectional collaboration framework and the introduction of the Category Diversity metric are valuable contributions; differentiated RGB poisoning is also novel. However, the technical complexity of each individual component is limited.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Experiments are highly comprehensive, covering 3 attack types, 2 datasets, 4 Negative Functions, 7 defense methods, and 4 model architectures, with detailed ablation analysis.
- **Writing Quality**: ⭐⭐⭐ — The notation system and component naming (A/B/C) could be more intuitive; the paper structure is clear but some experimental descriptions are slightly verbose with dense notation.
- **Value**: ⭐⭐⭐⭐ — The modular design philosophy offers practical guidance for both attackers and defenders; the code is publicly available, ensuring good reproducibility.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Stealthy Yet Effective: Distribution-Preserving Backdoor Attacks on Graph Classification](stealthy_yet_effective_distribution-preserving_backdoor_attacks_on_graph_classif.md)
- [\[AAAI 2026\] Towards Effective, Stealthy, and Persistent Backdoor Attacks Targeting Graph Foundation Models](../../AAAI2026/ai_safety/towards_effective_stealthy_and_persistent_backdoor_attacks_targeting_graph_found.md)
- [\[ICCV 2025\] Backdoor Attacks on Neural Networks via One-Bit Flip](../../ICCV2025/ai_safety/backdoor_attacks_on_neural_networks_via_one_bit_flip.md)
- [\[NeurIPS 2025\] Taught Well, Learned Ill: Towards Distillation-Conditional Backdoor Attack](taught_well_learned_ill_towards_distillation-conditional_backdoor_attack.md)
- [\[NeurIPS 2025\] Provable Watermarking for Data Poisoning Attacks](provable_watermarking_for_data_poisoning_attacks.md)

<!-- RELATED:END -->
