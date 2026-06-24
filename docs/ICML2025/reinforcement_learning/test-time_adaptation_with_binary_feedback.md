---
title: >-
  [Paper Note] Test-Time Adaptation with Binary Feedback
description: >-
  [ICML 2025][Reinforcement Learning][Test-Time Adaptation] This paper proposes BiTTA, a test-time adaptation framework utilizing binary feedback (correct/incorrect). Driven by a reinforcement learning-based dual-path optimization strategy, it achieves a 13.3% accuracy improvement under severe domain shift with minimal annotation cost.
tags:
  - "ICML 2025"
  - "Reinforcement Learning"
  - "Test-Time Adaptation"
  - "Binary Feedback"
  - "Domain Shift"
  - "Online Learning"
date: 2026-05-08
content_hash: 8813bcc3b5ca5582
---

# Test-Time Adaptation with Binary Feedback

**Conference**: ICML 2025  
**arXiv**: [2505.18514](https://arxiv.org/abs/2505.18514)  
**Code**: [GitHub](https://github.com/taeckyung/BiTTA)  
**Area**: Machine Learning / Test-Time Adaptation  
**Keywords**: Test-Time Adaptation, Binary Feedback, Reinforcement Learning, Domain Shift, Online Learning

## TL;DR

This paper proposes BiTTA, a test-time adaptation framework utilizing binary feedback (correct/incorrect). Driven by a reinforcement learning-based dual-path optimization strategy, it achieves a 13.3% accuracy improvement under severe domain shift with minimal annotation cost.

## Background & Motivation

Deep learning models experience significant performance degradation when training and testing data exhibit domain shifts. Test-Time Adaptation (TTA) addresses this by adapting pre-trained models using unlabeled samples at test time. However:

**Vulnerability of Existing TTA Methods**: Under severe domain shifts, self-supervised metrics based on entropy/confidence (e.g., TENT) become unreliable, leading to adaptation failure.

**High Cost of Active TTA**: Recent active TTA methods require full class labels, which incurs excessively high annotation costs (averaging 11.7 seconds/sample with a 12.7% error rate in a 50-class classification task).

**Efficiency of Binary Feedback**: In contrast, binary comparison requires only 1.6 seconds/sample, with an error rate of just 0.8%. From an information-theoretic perspective, full annotation demands $\log(\text{num\_class})$ times more bits than binary feedback.

Key Insight: Although binary feedback provides only 1 bit of information, it carries higher information content because it is based on the predictions of the adapting model (which are usually better than random), allowing it to directly guide model behavior.

## Method

### Overall Architecture

BiTTA models TTA as a reinforcement learning problem:
- **State**: Test sample $x$
- **Action**: Model prediction $y^* = \arg\max_y f_\theta(y|x)$
- **Policy**: Prediction probability $\pi_\theta(y|x)$
- **Objective**: Maximize expected reward $J(\theta) = \mathbb{E}_{x, y \sim \pi_\theta}[R(x, y)]$

The REINFORCE algorithm is used to handle non-differentiable binary feedback, approximating the policy via MC-dropout:

$$\pi_\theta(y|x) = \frac{1}{N}\sum_{n=1}^N f_\theta^d(y|x)$$

### Dual-Path Optimization

**Path 1: Binary Feedback-guided Adaptation (BFA)**

Select top-$k$ uncertain samples (those with the lowest MC-dropout confidence) and query binary feedback:

$$R_{\text{BFA}}(x, y) = B(x, y) = \begin{cases} 1 & \text{正确} \\ -1 & \text{错误} \end{cases}$$

Correctly predicted samples are stored in $\mathcal{M}_C$, and incorrect ones in $\mathcal{M}_I$ (FIFO buffers).

**Path 2: Agreement-Based self-Adaptation (ABA)**

Among the remaining unlabeled samples, samples where the standard prediction **agrees** with the MC-dropout prediction are selected as "confident samples":

$$\mathcal{S}_{\text{ABA}} = \{x \in \mathcal{B} \setminus \mathcal{S}_{\text{BFA}} \mid y^* = \arg\max_y \pi_\theta(y|x)\}$$

Key Advantage: It does not rely on a fixed threshold (a common pain point of traditional methods), but instead dynamically selects samples based on prediction consistency.

### Loss & Training

$$\mathcal{L}_{\text{BiTTA}} = \alpha \cdot \underbrace{\frac{1}{|\mathcal{M}_C|}\sum_{x \in \mathcal{M}_C}(-\log \pi_\theta) + \frac{1}{|\mathcal{M}_I|}\sum_{x \in \mathcal{M}_I}(+\log \pi_\theta)}_{\text{BFA: 最小化正确CE + 最大化错误CE}} + \beta \cdot \underbrace{\frac{1}{|\mathcal{S}_\text{ABA}|}\sum_{x \in \mathcal{S}_\text{ABA}}(-\log \pi_\theta)}_{\text{ABA: 最小化一致样本CE}}$$

Where $\alpha = \beta = 1$. BFA reinforces correct predictions and weakens incorrect ones, while ABA consolidates consistent (highly likely correct) predictions. No gradients are applied to uncertain samples that did not receive feedback (reward is 0), avoiding harmful adaptation from noisy signals.

## Key Experimental Results

### CIFAR10-C (Severe Corruption Level 5) Average Accuracy

| Label Type | Method | Average Accuracy (%) |
|---|---|---|
| - | SrcValid (No Adaptation) | 57.23 |
| - | BN-Stats | 78.42 |
| Binary | TENT* | 80.49 |
| Binary | SAR* | 83.78 |
| Binary | CoTTA* | 78.42 |
| Binary | RoTTA* | 80.98 |
| Binary | SimATTA* (Full label changed to binary) | 81.09 |
| **Binary** | **BiTTA** | **87.20** |

BiTTA outperforms the second-best baseline SAR* by 3.42%p, and exceeds SrcValid by approximately 30%p.

### Average Performance Gain Across Datasets

BiTTA outperforms the SOTA baselines by **13.3%p** on average.

### Key Comparison

- BiTTA (binary feedback only) **outperforms** SimATTA (active TTA with full class labels).
- BiTTA **outperforms** active TTA with GPT-4o as the annotator (Figure 7).
- BiTTA consistently maintains its advantage on CIFAR100-C and Tiny-ImageNet-C.

### ABA Validity Verification

- Samples with consistent predictions show stable and high accuracy (~90%+).
- Samples with inconsistent predictions show low and unstable accuracy.
- Dynamic selection based on consistency outperforms fixed-threshold strategies.

### Ablation Study

| Component | Impact of Removal |
|---|---|
| BFA (No binary feedback) | Drops significantly, degenerating into pure self-adaptation |
| ABA (No consistency self-adaptation) | Declines markedly, as a small amount of feedback alone is insufficient |
| MC-dropout (Replaced with standard softmax) | Calibration degrades, leading to lower selection quality for BFA |
| Memory buffer | Unstable adaptation in the early stages |

## Highlights & Insights

1. **Practicality of Setup**: Compared to full-label annotation, binary feedback (correct/incorrect) reduces the annotation cost by an order of magnitude and drastically lowers the error rate, making it highly suitable for practical deployment scenarios.
2. **Surpassing Full-Label Active TTA**: Despite providing seemingly less information, binary feedback, through BiTTA's meticulous design, outpaces full labels because it focuses more on correcting errors rather than memorizing labels.
3. **Dual-Path Synergy**: BFA explores uncertain regions (learning new knowledge) while ABA consolidates certain regions (preserving existing knowledge), creating a complementary mechanism.
4. **Multi-faceted Role of MC-dropout**: Efficiently serves multiple purposes simultaneously, including policy estimation, uncertainty quantification, and consistency detection.

## Limitations & Future Work

1. MC-dropout requires multiple forward passes ($N$ times), which increases inference latency and may become a bottleneck in real-time TTA scenarios.
2. Only $k$ binary feedbacks are queried per batch; under extreme domain shifts, this volume of feedback might be insufficient.
3. It assumes the oracle's binary feedback is fully accurate, whereas human annotators in real-world settings still exhibit errors.
4. Evaluation is limited to image classification tasks; its effectiveness has not been verified on other modalities (such as NLP or speech).
5. The FIFO buffer size is fixed to the batch size, without exploring adaptive buffer strategies.

## Related Work & Insights

- **Test-Time Adaptation**: TENT (Wang et al., 2021), CoTTA (Wang et al., 2022), SAR (Niu et al., 2023), RoTTA (Yuan et al., 2023), SoTTA (Gong et al., 2023)
- **Active Test-Time Adaptation**: SimATTA (Gui et al., 2024)—using full class labels
- **RLHF**: Ouyang et al. (2022) utilizes human feedback in LLMs, which inspires the application of RLHF to TTA in this work.
- **Uncertainty Estimation**: MC-dropout (Gal & Ghahramani, 2016)
- **Active Learning**: Settles (2009)—uncertainty-driven sample selection strategies

## Rating

⭐⭐⭐⭐

The task setup is novel and practical (TTA with binary feedback), the methodology is sound (RL + dual-path), and the experimental findings are impressive (outperforming full-label active TTA). Leveraging MC-dropout for multiple roles is a clever engineering decision. However, the core techniques (REINFORCE + cross-entropy) are relatively straightforward and lack deep theoretical analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] ReVISE: Learning to Refine at Test-Time via Intrinsic Self-Verification](revise_learning_to_refine_at_test-time_via_intrinsic_self-verification.md)
- [\[NeurIPS 2025\] Reinforcement Learning Teachers of Test Time Scaling](../../NeurIPS2025/reinforcement_learning/reinforcement_learning_teachers_of_test_time_scaling.md)
- [\[ICLR 2026\] Representation-Based Exploration for Language Models: From Test-Time to Post-Training](../../ICLR2026/reinforcement_learning/representation-based_exploration_for_language_models_from_test-time_to_post-trai.md)
- [\[AAAI 2026\] Aligning Machiavellian Agents: Behavior Steering via Test-Time Policy Shaping](../../AAAI2026/reinforcement_learning/aligning_machiavellian_agents_behavior_steering_via_test-tim.md)
- [\[ICLR 2026\] P-GenRM: Personalized Generative Reward Model with Test-time User-based Scaling](../../ICLR2026/reinforcement_learning/p-genrm_personalized_generative_reward_model_with_test-time_user-based_scaling.md)

</div>

<!-- RELATED:END -->
