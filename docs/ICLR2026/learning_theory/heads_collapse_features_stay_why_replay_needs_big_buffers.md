---
title: >-
  [Paper Note] Heads Collapse, Features Stay: Why Replay Needs Big Buffers
description: >-
  [ICLR 2026][learning_theory][Neural Collapse] Ours decouples "deep forgetting (feature space)" from "shallow forgetting (classification head)" and proves via Neural Collapse theory that any non-zero replay ratio asymptotically preserves the linear separability of old task features. However, small buffers cause the classification head to fall into "underdetermined
tags:
  - ICLR 2026
  - learning_theory
  - Neural Collapse
date: 2026-05-08
content_hash: 225022b911a17807
---
# Heads Collapse, Features Stay: Why Replay Needs Big Buffers

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=IdW0d0mRnG](https://openreview.net/forum?id=IdW0d0mRnG)  
**Code**: TBD  
**Area**: Continual Learning Theory / Neural Collapse / OOD Detection  
**Keywords**: Continual Learning, Experience Replay, Catastrophic Forgetting, Neural Collapse, Linear Separability, OOD Detection  

## TL;DR
Ours decouples "deep forgetting (feature space)" from "shallow forgetting (classification head)" and proves via Neural Collapse theory that any non-zero replay ratio asymptotically preserves the linear separability of old task features. However, small buffers cause the classification head to fall into "underdetermined optimization," resulting in covariance rank deficiency and class mean expansion, thus requiring much larger buffers to fix output-layer alignment—a phenomenon termed the "replay efficiency gap."

## Background & Motivation

**Background**: The goal of Continual Learning (CL) is to enable networks to learn sequentially across tasks without catastrophic forgetting. Experience Replay (ER)—storing a small subset of old samples to train alongside new data—is the most effective and commonly used strategy. A repeatedly observed paradox is that even when the network's predictive output has "forgotten" an old task, the task often remains linearly separable in the feature space; retraining a linear probe on frozen features yields far higher accuracy than the network's own output layer.

**Limitations of Prior Work**: Previous works reported this phenomenon as an isolated "feature-output discrepancy," but none have systematically characterized whether these two layers of forgetting depend differently on buffer size. In practice, the default assumption is that "to forget less, one must increase the buffer," which is neither scalable (storage and recomputation costs grow linearly) nor explained theoretically.

**Key Challenge**: Replay is **asymmetric** in its efficiency between "stabilizing feature geometry" and "maintaining classification head alignment." A tiny buffer is sufficient to prevent deep forgetting (preserving feature separability), but eliminating shallow forgetting (restoring output-layer accuracy) requires an unexpectedly large buffer. In other words, information persists in the features, but the classification head is "blind" to it.

**Goal**: To formally distinguish between deep vs. shallow forgetting and provide a predictable, lower-bounded theoretical explanation for why replay efficiency differs so drastically and why small buffers fail.

**Core Idea**: **[Feature Anchoring + Head Underdetermination]** Ours treats "forgotten old samples" as OOD data relative to the current model and characterizes their asymptotic geometry using the Neural Collapse framework. As long as the buffer is non-empty, old task features are "anchored" in the active subspace (preserving separability); however, "strong collapse" induced by small buffers leads to rank deficiency in buffer covariance and expansion of means, rendering classification head optimization underdetermined and causing decision boundaries to deviate from the true population boundaries.

## Method

### Overall Architecture

This is an analytical paper that extends Neural Collapse (NC) from a static single-task setting to a **sequential continual learning setting**, deriving the asymptotic behavior of deep/shallow forgetting. The workflow is: first, empirically characterize the different decay rates of the two layers of forgetting relative to buffer size (revealing the "replay efficiency gap") $\rightarrow$ extend NC to describe the terminal geometry of sequential training (covering DIL/CIL/TIL, single-head vs. multi-head) $\rightarrow$ formalize "forgotten $\approx$ OOD" to derive how features drift into an inactive subspace and are erased by weight decay without replay, and how they are preserved via a mixture model with replay $\rightarrow$ finally, mechanistically explain shallow forgetting as a result of underdetermined head optimization.

```mermaid
graph TD
    A[Sequential Task Training + Replay Buffer] --> B[Empirical: Deep vs. Shallow Forgetting<br/>Decay at Different Rates with Buffer Size]
    B --> C[Extend NC to Continual Learning<br/>DIL/CIL/TIL · Single/Multi-head]
    C --> D[Hypothesis 1: Forgotten Samples ≈ OOD<br/>Means Orthogonal to Active Subspace S]
    D --> E[No Replay: Signal Drifts into S⊥<br/>Exponentially Erased by Weight Decay]
    D --> F[Hypothesis 2: Replay is a Smooth Mixture of OOD↔NC<br/>π·D_NC + (1-π)·D_OOD]
    E --> G[Deep Forgetting Lower Bound<br/>SNR remains non-zero but fragile]
    F --> G
    C --> H[Small Buffer Induces Strong Collapse<br/>Covariance Rank Deficiency + Mean Expansion]
    H --> I[Classification Head Underdetermined Optimization<br/>→ Shallow Forgetting / Boundary Misalignment]
    G --> J[Replay Efficiency Gap]
    I --> J
```

### Key Designs

**1. Formalizing Deep vs. Shallow Forgetting: Measuring the gap between "feature memory" and "output forgetting" using linear probes.** Following the forgetting definition by Lopez-Paz & Ranzato, shallow forgetting is the accuracy drop $A_{ij}-A_{jj}$ (accuracy on task $j$ after session $i$, minus the accuracy right after learning task $j$), measuring recoverable degradation at the head level; deep forgetting uses a linear probe retrained on frozen features, yielding $A^\star_{ij}-A^\star_{jj}$, measuring irreversible loss of separability in feature space. This decoupling (Figure 2) reveals that in single-head settings (CIL/DIL), small buffers flatten the deep forgetting curve, but shallow forgetting only converges near 100% replay, leaving a persistent gap; in multi-head (TIL) settings, this gap is significantly smaller.

**2. Sequential NC Extension: Mapping terminal geometry from single-task to DIL/CIL/TIL, and characterizing multi-head cases.** NC describes Terminal Phase Training (TPT) features collapsing to three properties: within-class variance disappearance (NC1), centered class means forming a Simplex Equiangular Tight Frame (NC2, $\langle\tilde\mu_c,\tilde\mu_{c'}\rangle$ is $\beta_t$ for $c=c'$ and $-\beta_t/(K-1)$ otherwise), and alignment between head weights and class means (NC3, $W_h^\top\propto\tilde U$). Ours advances this to CL: in DIL, the ETF target geometry remains fixed; in CIL, the ETF evolves as class count increases, and old classes trigger "Minority Collapse" toward the origin if they become minority classes in the training set—though balanced replay maintains a global ETF. Multi-head TIL, previously unaddressed by NC theory, reveals that NC holds **locally** within each head, but tasks are **globally misaligned** (arbitrary scaling and angles of means), and local normalization reduces the maximum global feature space rank from $nK-1$ to $n(K-1)$.

**3. "Forgotten $\approx$ OOD" Hypothesis + Exponential Erasure without Replay.** The key insight (Hypothesis 1) is that forgotten old samples, which no longer contribute to the loss, behave geometrically like unseen OOD inputs—their average representation is orthogonal to the active subspace $S_t=\text{span}\{\tilde{\hat\mu}_1,\dots,\tilde{\hat\mu}_K\}$ spanned by current class means. Figure 4 verifies that after a task switch, old task means rapidly collapse onto their projections in $S_t$, becoming indistinguishable from OOD tasks. Once NC3 alignment holds, optimization updates are restricted to $S_t$, and components in the orthogonal complement $S^\perp$ are frozen or decay exponentially under weight decay. The theorem provides the asymptotic mean $\mu_c(t)=(1-\eta\lambda)^{t-t_0}\mu_{c,S^\perp}(t_0)$ and variance $\sigma_c^2(t)\in\Theta(\beta_t+(1-\eta\lambda)^{2(t-t_0)})$ of OOD classes, yielding a separability lower bound $\text{SNR}(c,c')\in\Theta(\beta_t(\upsilon^{2(t-t_0)}+1)^{-1})$ where $\upsilon=1-\eta\lambda$. This reveals the **double-edged** role of weight decay: it accelerates the erasure of residual signals in $S^\perp$ (harming separability) while constraining the mean norm $\beta_t$ (indirectly preserving separability).

**4. Replay Mixture Model: Smooth interpolation between OOD and NC via $\pi$, proving non-zero buffers preserve separability.** Hypothesis 2 suggests that replay allows feature structures to emerge smoothly with buffer size: old task features retain larger components in the active subspace $S$. This is formalized as a mixture $\phi(x)\sim\pi_c\,\mathcal{D}_{NC}+(1-\pi_c)\,\mathcal{D}_{OOD}$, where $\pi_c\in[0,1]$ is a monotonic function of buffer size. This yields an SNR lower bound with replay $\text{SNR}(c,c')\in\Theta\big(\frac{r^2\beta_t+\upsilon^{2(t-t_0)}}{r^2\delta_t+\beta_t+\upsilon^{2(t-t_0)}}\big)$, where $r^2=\pi^2/(1-\pi)^2$. The corollary is that as long as $\pi>0$, $\text{SNR}\to\Theta(r^2)$ does not vanish—any non-empty buffer anchors features in $S$. The anchoring strength $r^2$ grows with the buffer, being super-linear for single-head (CIL/DIL) and sub-linear for multi-head (TIL).

**5. Shallow Forgetting Mechanism: Underdetermined heads induced by small buffers.** This is the other half of the replay efficiency gap. Small samples induce "strong NC" (Hui et al.), where buffered data is aggressively collapsed toward empirical means, projecting into a low-dimensional subspace $S_B\subset S$ of rank $\approx K-1$. However, the true population still has variance in directions orthogonal to $S_B$ (especially in $S^\perp$). This geometric mismatch makes head optimization **underdetermined**: since buffer variance vanishes in certain directions, the loss is insensitive to weights $W$ in those directions, resulting in a manifold of "buffer-optimal" solutions. These solutions perfectly classify buffered samples but take arbitrary values in the complement of $S_B$, causing the decision boundary to deviate from the true population centroid (Figure 1). Ours deconstructs this gap into two artifacts using synthetic LDA counterfactuals: **covariance deficiency** (rank deficiency in $\hat\Sigma_B$ ignores variance in $S^\perp$—replacing true covariance with an identity matrix causes accuracy to plummet) and **mean norm expansion** (buffer means are pushed outward by repulsive forces); these artifacts persist until the buffer is nearly full.

## Key Experimental Results

Experiments used ResNet and ViT from both random initialization and pre-trained starting points, evaluated on Cifar100, Tiny-ImageNet, and CUB200 across TIL/CIL/DIL settings.

### Main Results: Replay Efficiency Gap (Figure 2)

| Phenomenon | Observation |
|------|------|
| Deep Forgetting vs. Buffer | Small buffers (a few percent) suffice to flatten the curve; separability is mostly preserved. |
| Shallow Forgetting vs. Buffer | Decays slowly, requiring nearly 100% replay to converge. |
| Head Structure Difference | Gap is significant in single-head (CIL/DIL), much smaller in multi-head (TIL). |
| DIL Counter-intuition | DIL shows high deep forgetting, converging to levels similar to CIL (contradicting the view that CIL is always hardest). |
| Pre-training Robustness | Deep forgetting is almost negligible for pre-trained models, with near-flat curves. |

### Mechanism Validation: Deconstructing the Statistical Gap (Figure 6, Synthetic LDA)

| Substitution Setting | Effect |
|----------|------|
| True Covariance $\rightarrow$ Identity (gray line) | Accuracy drops sharply $\rightarrow$ Second-order statistics are crucial. |
| Population Mean $\rightarrow$ Buffer Mean (olive line) | Additive degradation $\rightarrow$ Mean norm expansion is harmful. |
| Mean + Covariance both Buffer-estimated (cyan line) | Performance drops below the original network. |
| Covariance Rank Gap | Persists until the buffer size approaches full capacity. |

### Key Findings

- **Any non-zero replay ratio asymptotically preserves linear separability** (Corollary 2); small buffers are enough to anchor feature geometry.
- **Small buffers fail due to head underdetermination**: Covariance rank deficiency + mean expansion makes the head "blind" to true boundaries, rather than features losing information.
- NC emerges rapidly in sequential training (Figure 3); balanced replay suppresses Minority Collapse in CIL and restores global ETF.
- Unexpected phenomena: Feature norms grow with class count in CIL/TIL; multi-head models have lower feature space rank ($n(K-1)<nK-1$); weight decay is double-edged for separability.

## Highlights & Insights

- **Redefining "forgetting" as geometric drift**: Elevating from "accuracy drop" to "feature means becoming orthogonal to the active subspace," providing a rigorous geometric characterization and bridging CL with OOD detection literatures.
- **Explaining a long-standing paradox**: The reason features remember while outputs forget is not information loss in features, but underdetermined optimization in the head—shifting the solution from "storing more data" to "fixing statistical artifacts."
- **Disruptive Practical Implications**: Since large buffers are an inefficient brute-force solution, explicitly correcting for covariance rank deficiency and radial repulsion induced by small buffers may achieve robust performance with minimal replay.
- **First Characterization of Multi-head NC**: Fills the gap in NC theory for multi-head settings common in CL and discovers the structural phenomenon of rank reduction.

## Limitations & Future Work

- **Asymptotic Perspective**: The theory focuses on the terminal phase of training, ignoring early transient dynamics which likely house the origins of forgetting.
- **Idealized Replay Modeling**: Treating the buffer as an interpolation between two extremes (pure OOD $\leftrightarrow$ full NC) simplifies real distribution dynamics and may not cover all practical scenarios.
- **Class Mean Norm Growth**: Attributed to artifacts from classification head initialization, but only preliminary evidence is provided, left for future systematic study.
- **Lack of Full Algorithm**: This is an analytical work; "explicitly correcting statistical artifacts to trade buffer size for performance" is currently a directional suggestion without an end-to-end validated method.

## Related Work & Insights

- **Deep vs. Shallow Forgetting**: Prior works (Murata 2020, Hess 2023) used probes to find internal representations remember more; ours is the first to prove they decay at **fundamentally different rates** relative to buffer size.
- **Neural Collapse**: From the ETF phenomenon (Papyan 2020) to minority collapse (Fang 2021) and over-parameterized cases, to CL works using fixed ETF heads (Yang 2023); ours differs by using NC for asymptotic analysis of CL and introducing multi-head settings.
- **OOD Detection**: Originally focused on softmax confidence, later discovery showed OOD features collapse toward the origin; recent work links this to NC ID/OOD orthogonality (Ammar 2024). Ours formalizes this orthogonality, clarifies the role of weight decay and feature norms, and explicitly links OOD detection to CL forgetting.
- **Insight**: Using "representation geometry + second-order statistics" as a unified language for CL suggests future work can align classification heads with true population boundaries via covariance correction or norm regularization without increasing buffer size.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to theorize the buffer dependency difference between deep/shallow forgetting; uniquely bridges NC, OOD detection, and CL.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 3 benchmarks × 2 architectures × Random/Pre-trained × 3 CL settings, plus mechanistic deconstruction via LDA; missing an end-to-end buffer reduction algorithm.
- Writing Quality: ⭐⭐⭐⭐ Clear logic with alternating theorems and empirical evidence; however, dense notation and heavy reliance on appendices make for a high barrier to entry.
- Value: ⭐⭐⭐⭐⭐ Challenges the "large buffer is inevitable" assumption, pointing toward new directions for low-memory CL (fixing artifacts instead of stacking data).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Neural Collapse in Multi-Task Learning](neural_collapse_in_multi-task_learning.md)
- [\[ICLR 2026\] Random Label Prediction Heads for Studying Memorization in Deep Neural Networks](random_label_prediction_heads_for_studying_memorization_in_deep_neural_networks.md)
- [\[ICLR 2026\] Why Less is More (Sometimes): A Theory of Data Curation](why_less_is_more_sometimes_a_theory_of_data_curation.md)
- [\[ICLR 2026\] Why We Need New Benchmarks for Local Intrinsic Dimension Estimation](why_we_need_new_benchmarks_for_local_intrinsic_dimension_estimation.md)
- [\[ICLR 2026\] Why Ask One When You Can Ask k? Learning-to-Defer to the Top-k Experts](why_ask_one_when_you_can_ask_k_learning-to-defer_to_the_top-k_experts.md)

</div>

<!-- RELATED:END -->
