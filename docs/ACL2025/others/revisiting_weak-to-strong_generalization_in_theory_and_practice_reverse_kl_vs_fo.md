---
title: >-
  [Paper Note] Revisiting Weak-to-Strong Generalization: Reverse KL vs. Forward KL
description: >-
  [ACL 2025 (Findings)][Weak-to-strong generalization] In the Weak-to-Strong Generalization (W2SG) framework, this paper proposes replacing forward KL with reverse KL as the loss function. It is theoretically proven that the mode-seeking property of reverse KL ensures the strong model outperforms the weak supervisor by at least the magnitude of the "disagreement". Experiments on the GPT-2, Pythia, and Qwen2.5 series validate that reverse KL/CE outperforms forward KL in 12 out o…
tags:
  - "ACL 2025 (Findings)"
  - "Weak-to-strong generalization"
  - "reverse KL"
  - "superalignment"
  - "knowledge distillation"
  - "loss function"
date: 2026-05-08
content_hash: e0401e0ea375292d
---

# Revisiting Weak-to-Strong Generalization: Reverse KL vs. Forward KL

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2502.11107](https://arxiv.org/abs/2502.11107)  
**Code**: None  
**Area**: Others  
**Keywords**: Weak-to-strong generalization, reverse KL, superalignment, knowledge distillation, loss function

## TL;DR
In the Weak-to-Strong Generalization (W2SG) framework, this paper proposes replacing forward KL with reverse KL as the loss function. It is theoretically proven that the mode-seeking property of reverse KL ensures the strong model outperforms the weak supervisor by at least the magnitude of the "disagreement". Experiments on the GPT-2, Pythia, and Qwen2.5 series validate that reverse KL/CE outperforms forward KL in 12 out of 12 settings, demonstrating superior noise robustness.

## Background & Motivation

**Background**: As LLMs approach superhuman capabilities, human supervision becomes "weak". Weak-to-Strong Generalization (Burns et al., 2023), which proposes using weak models to supervise strong models, has emerged as a crucial paradigm for superalignment.

**Limitations of Prior Work**: W2SG utilizes standard cross-entropy (forward KL) for training. Its mass-covering behavior forces the strong model to fit the **entire distribution** of the weak supervisor, including noise or misleading signals on non-target classes. This causes the strong model to overfit to the defects of the weak supervisor.

**Key Challenge**: Forward KL is effective in knowledge distillation (strong teacher $\rightarrow$ weak student, where soft labels are reliable), but W2SG operates in the opposite direction (weak teacher $\rightarrow$ strong student, where soft labels are unreliable). Consequently, the advantages of the same loss function turn into disadvantages when the scenario is inverted.

**Goal**: Which loss function should be used in W2SG? To provide a theoretical comparison and practical validation of forward KL vs. reverse KL.

**Key Insight**: The zero-forcing / mode-seeking characteristics of reverse KL—focusing on high-confidence prediction regions of the weak model while ignoring low-probability noise regions—are precisely suited for extracting reliable signals from unreliable weak supervision.

**Core Idea**: To change the W2SG loss from $\min_f L(F_w, f \circ h_s)$ to $\min_f L(f \circ h_s, F_w)$ (reversing the direction of KL/CE), theoretically guaranteeing a tighter generalization bound.

## Method

### Overall Architecture
W2SG setup: A weak model $F_w$ provides soft-label supervision for a strong model $F_{sw} = f \circ h_s$ (where $h_s$ is the fixed representation of the strong model, and $f$ is the trainable task head).
- Forward loss: $\min_f L(F_w, f \circ h_s)$ $\rightarrow$ standard CE/KL, referencing the weak model distribution.
- Reverse loss: $\min_f L(f \circ h_s, F_w)$ $\rightarrow$ referencing the strong model distribution with the weak model distribution as the target.

### Key Designs

1. **Generalization Bounds Analysis (Lemma 1)**

    - Function: Establishes a unified generalization bound for both forward and reverse KL/CE.
    - Mechanism: $|L(F^*, F_w) - L(F^*, F_{sw})| \leq C_1 \sqrt{d(F_w, F_{sw})}$, where $d$ can be either the forward or reverse KL divergence. This demonstrates that both losses provide comparable generalization guarantees.
    - Design Motivation: To prove that the reverse loss is "at least as good as" the forward loss.

2. **The Unique Advantage of Reverse KL (Theorem 2)**

    - Function: Proves that reverse KL guarantees the strong model outperforms the weak model under last-layer fine-tuning.
    - Mechanism: When a sufficiently pre-trained strong model is fine-tuned only on its final linear layer, reverse KL guarantees $L(F^*, F_{sw}^r) \leq L(F^*, F_w) - \text{disagreement}(F_w, F_{sw}^r)$. That is, strong model performance $\geq$ weak model performance + disagreement magnitude.
    - Design Motivation: Forward KL lacks this guarantee; its mass-covering behavior may cause the strong model to "degenerate" to the level of the weak model.

3. **A Tighter Lower Bound (Improving Yao et al., 2025)**

    - Function: Derives a tighter lower bound $C_2 \leq C_1$.
    - Mechanism: Utilizes the condition $\gamma = 10^{-3} \ll 1/e$ to obtain a smaller constant factor $C_2$ for the reverse loss, implying a tighter generalization bound.

4. **Noise Robustness**

    - Forward KL mass-covering: Low-probability regions of noisy labels are also learned.
    - Reverse KL zero-forcing: The strong model only focuses on the highly confident predictions of the weak model, thereby automatically filtering out noise.

### Loss & Training
- Single-epoch training to reduce overfitting, with a batch size of 16 and a learning rate of $10^{-5}$.
- 4K samples for weak supervision, 4K for the ground truth ceiling, and 4K for testing.
- Optionally incorporates confidence regularization (Burns et al., 2023) for further enhancement.

## Key Experimental Results

### Main Results (GPT-2 on CAI-Harmless)

| Setup | Forward KL | Reverse KL | Forward CE | Reverse CE |
|------|-----------|-----------|-----------|-----------|
| Base $\rightarrow$ Medium | 89.7 | **91.5** | 89.7 | **91.2** |
| Base $\rightarrow$ Large | 93.6 | **94.2** | 93.6 | **93.9** |
| Medium $\rightarrow$ Large | 93.5 | **94.1** | 93.5 | **93.8** |

### Noise Robustness (GPT-2-Base $\rightarrow$ Medium, CAI-Harmless)

| Noise Ratio | Forward KL | Reverse KL | Forward CE | Reverse CE |
|----------|-----------|-----------|-----------|-----------|
| 10% | 90.1 | **92.4** | 90.1 | **92.0** |
| 20% | 86.3 | **91.3** | 86.2 | **90.8** |
| 30% | 81.7 | **90.0** | 81.6 | **89.5** |
| 40% | 72.8 | **80.6** | 72.8 | **81.8** |

### Large-Scale Validation on Qwen2.5

| Setup | Forward KL | Reverse KL |
|------|-----------|-----------|
| 3B $\rightarrow$ 7B | 96.2 | **96.8** |
| 3B $\rightarrow$ 14B | 96.4 | **96.5** |
| 7B $\rightarrow$ 14B | 96.8 | **96.8** |

### Key Findings
- **Reverse KL outperforms forward KL in 12 out of 12 settings** (across two datasets in the GPT-2 series).
- **Remarkable noise robustness**: Under 30% noise, the performance of reverse KL drops by only ~2%, whereas forward KL drops by ~8%.
- **Complementary to confidence regularization**: Reverse CE + regularization $>$ forward CE + regularization.
- **Stronger weak model $\rightarrow$ better generalization**: Consistent with theoretical predictions (a smaller $L(F^*, F_w)$ in Lemma 1 leads to a higher upper bound for the strong model).
- **Reverse KL may fail under extreme noise (50%)**: Mode-seeking behavior might lock onto the wrong mode.

## Highlights & Insights
- **Symmetry insight between knowledge distillation and W2SG**: Forward KL is suited for "strong $\rightarrow$ weak" (KD), while reverse KL is suited for "weak $\rightarrow$ strong" (W2SG)—information quality determines the optimal direction of the loss. This is a simple yet profound observation.
- **"No change to pipeline, only to loss"**: A zero-code-change improvement requiring only the reversal of the KL direction, making it highly attractive for industrial deployment.
- **Perfect alignment between theoretical guarantees and experiments**: The guarantee in Theorem 2 that "the strong model outperforms the weak model by at least the disagreement magnitude" is consistently validated in experiments.

## Limitations & Future Work
- **Limited to binary reward modeling**: Both CAI-Harmless and HH-RLHF are binary classification tasks; multi-class classification or generation tasks remain unvalidated.
- **Limited model scale**: Restrictive to models up to Qwen2.5-14B; performance on ultra-large models (70B+) remains unknown.
- **Strong assumptions in Theorem 2**: Requires "sufficient pre-training" + "fine-tuning only the final linear layer"; the theoretical guarantee under full fine-tuning remains unclear.
- **Failure under extreme noise**: At 50% noise, reverse KL can perform worse than forward KL.

## Related Work & Insights
- **vs. Burns et al. (2023)**: Pioneered the W2SG framework but relied solely on forward CE. This work supplements the analysis with the reverse loss.
- **vs. DPO (Rafailov et al., 2024)**: DPO also employs reverse KL regularization; this work provides theoretical support for its application in W2SG contexts.
- **vs. Yao et al. (2025)**: They established a generalization bound for the forward loss, whereas this work extends it to the reverse loss and proves a tighter lower bound.

## Rating
- Novelty: ⭐⭐⭐⭐ Simple yet powerful observation (reversing the KL direction), but the technical novelty is relatively limited.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three model series (GPT-2, Pythia, Qwen2.5) + noise ablation + regularization ablation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation (comparison between KD and W2SG), with tight integration of theory and experiments.
- Value: ⭐⭐⭐⭐ Directional guidance for superalignment practice, though scenarios are limited to classification tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] How to Mitigate Overfitting in Weak-to-Strong Generalization?](how_to_mitigate_overfitting_in_weak-to-strong_generalization.md)
- [\[ACL 2025\] Well Begun is Half Done: Low-resource Preference Alignment by Weak-to-Strong Decoding](well_begun_is_half_done_low-resource_preference_alignment_by_weak-to-strong_deco.md)
- [\[CVPR 2026\] Beyond Euclidean Gossip: KL-Barycentric Consensus on Heterogeneous and Imbalanced Images](../../CVPR2026/others/beyond_euclidean_gossip_kl-barycentric_consensus_on_heterogeneous_and_imbalanced.md)
- [\[ACL 2025\] Behavioural vs. Representational Systematicity in End-to-End Models: An Opinionated Survey](behavioural_vs_representational_systematicity_in_end-to-end_models_an_opinionate.md)
- [\[ACL 2025\] Hybrid Preferences: Learning to Route Instances for Human vs. AI Feedback](hybrid_preferences_learning_to_route_instances_for_human_vs_ai_feedback.md)

</div>

<!-- RELATED:END -->
