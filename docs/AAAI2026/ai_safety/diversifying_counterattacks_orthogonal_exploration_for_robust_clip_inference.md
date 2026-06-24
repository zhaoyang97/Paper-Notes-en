---
title: >-
  [Paper Note] Diversifying Counterattacks: Orthogonal Exploration for Robust CLIP Inference
description: >-
  [AAAI 2026 Oral][AI Safety][Adversarial Robustness] This paper proposes the Directional Orthogonal Counterattack (DOC) method. By introducing orthogonal gradient components and momentum updates during the counterattack optimization, it expands the search space. Combined with a cosine similarity-based Directional Sensitivity Score to adaptively modulate the counterattack intensity, the method significantly improves the test-time adversarial robustness of CLIP across 16 dataset…
tags:
  - "AAAI 2026 Oral"
  - "AI Safety"
  - "Adversarial Robustness"
  - "CLIP Defense"
  - "Test-Time Defense"
  - "Orthogonal Counterattack"
  - "Vision-Language Models"
date: 2026-05-08
content_hash: 02483ba16db1433e
---

# Diversifying Counterattacks: Orthogonal Exploration for Robust CLIP Inference

**Conference**: AAAI 2026 Oral  
**arXiv**: [2511.09064](https://arxiv.org/abs/2511.09064)  
**Code**: [Available](https://github.com/bookman233/DOC)  
**Area**: AI Safety  
**Keywords**: Adversarial Robustness, CLIP Defense, Test-Time Defense, Orthogonal Counterattack, Vision-Language Models

## TL;DR

This paper proposes the Directional Orthogonal Counterattack (DOC) method. By introducing orthogonal gradient components and momentum updates during the counterattack optimization, it expands the search space. Combined with a cosine similarity-based Directional Sensitivity Score to adaptively modulate the counterattack intensity, the method significantly improves the test-time adversarial robustness of CLIP across 16 datasets.

## Background & Motivation

Vision-language pre-trained models like CLIP exhibit strong zero-shot generalization capabilities, yet remain extremely vulnerable to adversarial examples. Existing defense methods mainly fall into three categories:

**Adversarial Fine-Tuning** (e.g., TeCoA, PMG-AFT, FARE): Fine-tunes CLIP using adversarial examples, which incurs high computational overhead and potentially degrades generalization.

**Adversarial Prompt Tuning**: Adjusts prompts in the embedding space but forfeits semantic interpretability.

**Test-Time Counterattack (TTC)**: The latest parameter-free defense method, which generates counterattack perturbations to maximize the embedding distance between adversarial inputs and their variants.

**Key Challenge of TTC**: There exists a **fundamental mismatch in optimization objectives** between adversarial attack and counterattack:

- Adversarial attack goal: Maximize classification loss
- Counterattack goal: Maximize embedding distance

TTC uses PGD to generate counterattacks along the gradient direction. However, due to the objective mismatch, the search space is confined to a narrow region. Consequently, the counterattack is prone to overfitting to limited adversarial patterns, lacking the diversity needed to neutralize the broad distribution of perturbations.

## Method

### Overall Architecture

DOC (Directional Orthogonal Counterattack) comprises two core components:

1. **Orthogonal Gradient Augmentation (OGA)**: Adds random components orthogonal to the primary gradient direction along with momentum updates at each step of the counterattack optimization.
2. **Directional Sensitivity Score (DSS)**: Leverages cosine similarity to detect whether the input is an adversarial example, adaptively modulating the counterattack strength.

### Key Designs

**Orthogonal Gradient Augmentation (OGA)**:

1. Compute the normalized gradient $g$ (gradient of the counterattack loss with respect to the adversarial input, and normalize it).
2. Sample a random vector $r$ from a standard normal distribution, and apply Gram-Schmidt orthogonalization to obtain the component orthogonal to the gradient: $r_{\perp} = (r - \langle r, g \rangle g) / \|r - \langle r, g \rangle g\|$.
3. Combine update directions: $d = g + \lambda r_{\perp}$ ($\lambda$ controls the intensity of orthogonal injection).
4. Momentum update: $m_t = \mu m_{t-1} + (1-\mu)d$.
5. Iterative updates of the counterattack perturbation: $\delta_{t+1} = \text{Proj}(\delta_t + \alpha \cdot \text{sign}(m_t))$.

Design Intuition: The orthogonal component enables the counterattack to explore regions beyond the gradient direction, while momentum assists in escaping narrow local optima, generating more diverse counterattack perturbations. t-SNE visualizations confirm that the counterattack distribution of DOC is more dispersed than that of TTC.

**Directional Sensitivity Score (DSS)**:

TTC uses $l_2$ distance to identify whether an input is an adversarial example, which poses two issues: (a) embeddings with similar directions but different scales can cause artificially inflated $l_2$ distances; (b) a single noisy sample introduces instability.

DOC switches to cosine similarity + multiple sampling:

- $\hat{\tau}(x) = 1 - \frac{1}{M} \sum \cos(I_{\theta}(x_m), I_{\theta}(x))$
- Low $\hat{\tau}(x)$: The embedding direction remains unchanged after perturbation, indicating a clean sample.
- High $\hat{\tau}(x)$: The direction is inconsistent, indicating a potential adversarial sample.

The strength of the counterattack is adaptively modulated via a soft gating function:

- $w = \text{sigmoid}(\gamma \cdot (\tau - \hat{\tau}(x)))$
- Final: $\delta_{ca} = w \cdot \delta_{ca} + (1-w) \cdot \delta_{ca}^0$

For clean samples, $w$ is close to 0 (applying almost no counterattack), while for adversarial samples, $w$ is close to 1 (full-force counterattack).

### Loss & Training

DOC is a **training-free** test-time defense method:

- It does not modify model parameters, requires no training data, and does not depend on label supervision.
- Counterattack budget $\epsilon_{ca} = 4/255$.
- Default of 4 steps of counterattack with step size $\alpha = 3/255$.
- Batch size is 256, requiring only a single NVIDIA 4090 GPU.

## Key Experimental Results

### Main Results

**Average results across 16 datasets under PGD-10 attack** ($\epsilon_{atk} = 4/255$):

| Method | Type | Average Robust Accuracy | Average Clean Accuracy |
|---|---|---|---|
| CLIP (Original) | - | 0.06% | 61.51% |
| HD | Test-Time Defense | 0.56% | 54.85% |
| TeCoA4 | Adversarial Fine-Tuning | 10.95% | 37.58% |
| FARE4 | Adversarial Fine-Tuning | 1.38% | 56.62% |
| TTC | Test-Time Defense | 21.22% | 55.63% |
| **DOC** | **Test-Time Defense** | **31.02%** | **58.26%** |

DOC improves robust accuracy by **9.80%** over TTC while achieving higher clean accuracy (+2.63%).

**Key results across individual datasets** (robust accuracy under PGD-10):

| Dataset | CLIP | TTC | DOC | Gain |
|---|---|---|---|---|
| CIFAR-10 | 0.00% | 30.25% | 38.14% | +7.89% |
| STL-10 | 0.04% | 51.89% | 69.16% | +17.27% |
| ImageNet | 0.00% | 13.07% | 24.64% | +11.57% |
| OxfordPets | 0.00% | 25.89% | 46.52% | +20.63% |
| Caltech-256 | 0.13% | 26.38% | 43.08% | +16.70% |

### Ablation Study

| DSS | OGA | Clean Accuracy | PGD Robust | CW Robust | AutoAttack |
|---|---|---|---|---|---|
| No | No | 55.66% | 21.43% | 20.70% | 21.97% |
| Yes | No | 58.23% | 23.37% | 22.27% | 22.66% |
| No | Yes | 55.38% | 31.83% | 29.02% | 26.07% |
| Yes | Yes | **58.27%** | **31.04%** | **28.15%** | **25.89%** |

- **DSS Alone**: Primarily boosts clean accuracy (+2.57%), suppressing unnecessary perturbations on clean samples.
- **OGA Alone**: Significantly improves robust accuracy (+10.4%), validating the effectiveness of diversified counterattacks.
- **Combining both**: Achieves a balance between robustness and clean accuracy.

Average robust accuracy under CW attack: DOC 28.18% vs TTC 20.61% (+7.58%). Under AutoAttack, DOC improves upon TTC by approximately 4.1%.

### Key Findings

- DOC outperforms TTC on almost all 16 datasets, with the sole exception of EuroSAT.
- DOC can serve as a **plug-and-play module** to combine with adversarial fine-tuning: when paired with FARE, the average robust accuracy exceeds that of the original CLIP by over 18%.
- The number of counterattack steps saturates at only $N=3\text{-}4$, resulting in extremely low computational overhead.
- The clean accuracy remains stable as the number of steps increases, demonstrating that the gain in robustness does not come at the expense of clean performance.

## Highlights & Insights

1. **Precise Problem Diagnostic**: Reveals the optimization objective mismatch between adversarial attacks and counterattacks.
2. **Intuitive Design of Orthogonal Gradient Augmentation**: Introduces exploratory noise via orthogonalization, which is both mathematically elegant and practically effective.
3. **Cosine Similarity Replacing $l_2$ Distance**: Used for adversarial sample detection, which is more reasonable in high-dimensional spaces (scale-invariance).
4. **Entirely Training-Free**: Requires no data, no parameter tuning, and runs on a single GPU, presenting an extremely low deployment barrier.
5. t-SNE visualization intuitively demonstrates the effectiveness of DOC in aligning the distribution of adversarial samples closer to clean distributions.

## Limitations & Future Work

1. The counterattack budget is set to the same value as the attack budget, whereas the attack budget is typically unknown in real-world scenarios.
2. The orthogonal components are randomly sampled, which may introduce variance across different inference runs (though the variance observed in experiments is small).
3. A decrease in clean accuracy is observed on ImageNet (-3.25%), and some fluctuations occur on fine-grained classification datasets.
4. Validation is only conducted on CLIP and has not yet been extended to other VLPs (e.g., BLIP-2, LLaVA).
5. Robustness against adaptive attacks is not fully discussed.

## Related Work & Insights

- **TTC** (Xing et al. 2025): Pioneering work on test-time counterattack, which DOC direct builds upon.
- **TeCoA** (Mao et al.): Representative method for adversarial fine-tuning.
- **PMG-AFT** (Wang et al. 2024): Adversarial fine-tuning incorporating CLIP-guided regularization.
- **FARE** (Schlarmann et al. 2024): Adversarial fine-tuning under larger budgets.
- **Hedge Defense** (Wu et al. 2021): Test-time defense by maximizing losses across all classes.
- Insight: **In unsupervised test-time defense, diversity is more critical than precision.** The orthogonal exploration concept can be generalized to other robust optimization contexts.

## Rating

- Novelty: 4/5 - Orthogonal Gradient Augmentation and Directional Sensitivity Score represent meaningful and novel contributions.
- Technical Depth: 4/5 - The methodology design is backed by clear theoretical motivations and mathematical derivations.
- Experimental Thoroughness: 5/5 - Extensive testing across 16 datasets $\times$ 3 attacks, ablation studies, combinatorial experiments, and visualizations.
- Writing Quality: 4/5 - Clean exposition of motivation and problems, with rich figures and tables.
- Overall: 4.0/5

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Robust Federated Inference](../../ICLR2026/ai_safety/robust_federated_inference.md)
- [\[CVPR 2026\] When CLIP Sees More, It Fights Back Harder: Multi-View Guided Adaptive Counterattacks for Test-Time Adversarial Robustness](../../CVPR2026/ai_safety/when_clip_sees_more_it_fights_back_harder_multi-view_guided_adaptive_counteratta.md)
- [\[ICML 2026\] Calibrating Uncertainty for Zero-Shot Adversarial CLIP](../../ICML2026/ai_safety/calibrating_uncertainty_for_zero-shot_adversarial_clip.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](robust_watermarking_on_gradient_boosting_decision_trees.md)
- [\[ICLR 2026\] Test-Time Poisoned Sample Detection by Exploiting Shallow Malicious Matching in Backdoored CLIP](../../ICLR2026/ai_safety/test-time_poisoned_sample_detection_by_exploiting_shallow_malicious_matching_in_.md)

</div>

<!-- RELATED:END -->
