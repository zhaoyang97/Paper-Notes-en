---
title: >-
  [Paper Note] Influence Malleability in Linearized Attention: Dual Implications of Non-Convergent NTK Dynamics
description: >-
  [CVPR2026][Robotics][Neural Tangent Kernel] This paper demonstrates that linearized attention does not converge to the infinite-width limit under the NTK framework, and proposes the metric of *influence malleability* to show that the expressive power of attention and its adversarial vulnerability share a common origin—data-dependent kernel structure that deviates from the kernel regime.
tags:
  - CVPR2026
  - Robotics
  - Neural Tangent Kernel
  - linearized attention
  - influence malleability
  - kernel methods
  - feature learning
date: 2026-05-08
content_hash: 2d114315bee33420
---

# Influence Malleability in Linearized Attention: Dual Implications of Non-Convergent NTK Dynamics

**Conference**: CVPR2026
**arXiv**: [2603.13085](https://arxiv.org/abs/2603.13085)
**Code**: To be confirmed
**Area**: Deep Learning Theory / Attention Mechanisms
**Keywords**: Neural Tangent Kernel, linearized attention, influence malleability, kernel methods, feature learning

## TL;DR

This paper demonstrates that linearized attention does not converge to the infinite-width limit under the NTK framework, and proposes the metric of *influence malleability* to show that the expressive power of attention and its adversarial vulnerability share a common origin—data-dependent kernel structure that deviates from the kernel regime.

## Background & Motivation

1. **Theoretical gap in attention mechanisms**: Despite the remarkable success of attention in deep learning, its learning dynamics lack rigorous theoretical characterization; most prior work focuses on initialization or final performance, neglecting the intermediate training process.
2. **Limitations of NTK theory**: The NTK framework predicts that sufficiently wide networks operate in the "lazy training" regime (i.e., the kernel remains invariant), yet whether practical attention architectures satisfy this condition has not been systematically studied.
3. **Feature learning vs. lazy training**: Chizat et al. (2019) distinguish between lazy training and feature learning regimes, but neither empirical nor theoretical evidence places attention mechanisms within either regime.
4. **Lack of quantification for data dependence**: No measurable indicator exists for the sensitivity of attention models to training data, making it impossible to systematically assess their reliance on data quality.
5. **Insufficient understanding of adversarial robustness**: The connection between adversarial vulnerability and expressive capacity in attention architectures remains unclear and awaits a unified theoretical account.
6. **Connection between kernel methods and attention**: While the link between softmax attention and Nadaraya-Watson kernel regression is known, the precise kernel characterization of linearized attention and its implications for learning dynamics have not been established.

## Method

### Overall Architecture

The paper proposes a linearized attention architecture, MLP-Attn, amenable to precise kernel characterization. It analyzes the learning dynamics via the NTK framework and introduces the influence malleability metric to quantify the sensitivity of attention to training data.

### Linearized Attention Design

- **Core definition**: $f^{\text{att}}(\mathbf{X}) = \mathbf{X}\mathbf{X}^T\mathbf{X}$, corresponding to attention with identity QKV projections and linearized softmax.
- Full architecture: $f_{\text{MLP-Attn}}(\mathbf{X}) = \frac{1}{\sqrt{m}} \sum_{r=1}^{m} a_r \sigma(\mathbf{w}_r^T \cdot f^{\text{att}}(\mathbf{X}))$
- The transformation is computed transductively over the entire training set, encoding global pairwise relationships.
- The attention output is $\ell_2$-normalized before being fed into the MLP.

### Key Theoretical Results

1. **Gram-Induced Kernel (Thm 4.1)**: The kernel induced by linearized attention is $\mathbf{K}_{\text{LinAttn}} = \mathbf{G}^3$ (where $\mathbf{G}=\mathbf{X}\mathbf{X}^T$), exhibiting a transitive similarity structure $i \to k \to \ell \to j$.
2. **Spectral Amplification Theorem (Thm 4.7)**: Attention amplifies the condition number of the Gram matrix cubically: $\kappa(\tilde{\mathbf{G}}) = \kappa(\mathbf{G})^3$. NTK convergence requires width $m = \Omega(\kappa(\mathbf{G})^6 / \epsilon^2)$, which far exceeds practical feasibility for natural image data (MNIST requires $m \gg 10^{18}$; CIFAR-10 requires $m \gg 10^{24}$).
3. **Data-Dependent Sensitivity (Prop 4.5)**: $|K_{\text{LinAttn}}(\mathbf{x}_i+\delta, \mathbf{x}_j) - K_{\text{LinAttn}}(\mathbf{x}_i, \mathbf{x}_j)| \leq \|\mathbf{G}\mathbf{x}_j\|_1 \cdot \epsilon$; perturbations propagate globally through the Gram matrix.

### Influence Malleability Metric

- **Influence Flip Rate**: Adversarial perturbations (PGD, $\epsilon=0.3$) are applied to the top-10% most influential training samples, and the proportion of influence sign flips is recorded.
- **Influence Rank Correlation**: Spearman rank correlation $\rho$ measures the stability of influence rankings before and after perturbation; lower $\rho$ indicates higher malleability.
- Three intervention strategies: Curated (removing high-influence samples), Transformed (replacing with adversarial versions), and Adversarial (global PGD perturbation).

## Key Experimental Results

### NTK Distance Non-Convergence

| Model | Dataset | m=16 | m=1024 | m=4096 | Trend |
|-------|---------|------|--------|--------|-------|
| 2L-ReLU | MNIST | 45.1 | 39.9 | 39.2 | ↓ converges |
| MLP-Attn | MNIST | 10.3 | 33.3 | 43.4 | ↑ non-monotone |
| 2L-ReLU | CIFAR-10 | 246.2 | 101.7 | 56.9 | ↓ converges |
| MLP-Attn | CIFAR-10 | 3.7 | 10.4 | 12.6 | ↑ monotone increase |

- The NTK distance of 2L-ReLU decreases monotonically with increasing width (consistent with classical NTK theory), whereas MLP-Attn's distance increases, confirming that it operates in the feature learning regime.

### Influence Flip Rate (10-class, $\epsilon=0.3$)

| Dataset | Model | FGSM | PGD | MIM |
|---------|-------|------|-----|-----|
| MNIST | 2L-ReLU | 4.1% | 3.3% | 3.4% |
| MNIST | MLP-Attn | **34.6%** | **28.9%** | **21.9%** |
| CIFAR-10 | 2L-ReLU | 3.3% | 3.1% | 3.2% |
| CIFAR-10 | MLP-Attn | **26.4%** | **19.1%** | **20.5%** |

- The flip rate of MLP-Attn is **6–9×** that of 2L-ReLU, validating the high sensitivity of attention architectures to training data.

### Ablation Study: Effect of Adversarial Training

| Dataset | Model | Standard Training | Adversarial Training |
|---------|-------|------------------|---------------------|
| MNIST | 2L-ReLU | 3.3% | 43.4% |
| MNIST | MLP-Attn | 28.9% | 42.2% |
| CIFAR-10 | 2L-ReLU | 3.1% | 36.5% |
| CIFAR-10 | MLP-Attn | 19.1% | 38.6% |

- Adversarial training substantially increases the malleability of ReLU networks (3.3%→43.4%), whereas MLP-Attn already exhibits high malleability under standard training, indicating that this is an intrinsic architectural property rather than a training-induced artifact.
- After adversarial training, the malleability of both architectures converges (42–43%), suggesting that adversarial training may push ReLU networks into a similar feature learning regime.
- In the binary classification setting (MNIST 3 vs. 8), MLP-Attn achieves a PGD flip rate of 41.0% (vs. 8.4% for ReLU); however, this gap vanishes in CIFAR-10 binary classification, consistent with the lower $\kappa(\mathbf{G})$ in that setting.

## Highlights & Insights

- **Rigorous theoretical contribution**: The paper establishes a complete theoretical chain—linearized attention → Gram-induced kernel → spectral amplification → NTK non-convergence—with a formal proof at each step.
- **Unified explanation of dual implications**: This is the first work to attribute both the expressive power and adversarial vulnerability of attention to a single source (deviation from the kernel regime), offering a conceptually elegant and insightful account.
- **Novel metric**: Influence malleability (flip rate + rank correlation) provides a quantifiable indicator of sensitivity to training data, extensible to the analysis of other architectures.
- **Theory–experiment consistency**: Empirically measured Gram matrix condition numbers ($\kappa \approx 10^3$) perfectly explain the observed non-convergence at $m \leq 4096$, in full agreement with the theoretically predicted convergence width requirements ($m \gg 10^{18}$).
- **Elegant ablation design**: By contrasting malleability changes under standard and adversarial training for both architectures, the paper cleanly separates architecture-intrinsic sensitivity from training-induced sensitivity.
- **Multi-layer generalization**: The theory extends naturally to multi-layer linearized attention ($k$ layers → $\kappa(\mathbf{G})^{2k+1}$ amplification), and truncated attention is proposed as a potential regularization scheme.

## Limitations & Future Work

- **Linearization simplification**: Only $f^{\text{att}}=\mathbf{X}\mathbf{X}^T\mathbf{X}$ is analyzed; the framework is not extended to full softmax attention, whose row normalization may further amplify the observed effects.
- **Small datasets and model scale**: Experiments are conducted only on MNIST/CIFAR-10 with two-layer networks ($m \leq 4096$); large-scale Transformers (e.g., ViT) are not empirically validated.
- **Parameter-free attention**: The assumption of identity QKV matrices is theoretically extendable (Proposition B.4), but the gap relative to attention with learnable projections has not been empirically quantified.
- **Disappearing advantage in binary classification**: On CIFAR-10 binary classification, the advantage of attention nearly vanishes (flip rate $\approx 1\times$), indicating that the conclusions are sensitive to data dimensionality and complexity.
- **Absence of defense strategies**: While adversarial vulnerability is identified, no concrete mitigation methods are proposed; low-rank regularization (truncated attention) is only mentioned in a theoretical remark.
- **Transductive design limitation**: The attention transformation is computed once over the entire training set, which fundamentally differs from mini-batch processing in practical Transformers, limiting practical applicability.

## Related Work & Insights

- **vs. NTK theory (Jacot et al., 2018)**: Classical NTK assumes wide networks operate in the lazy regime; this paper demonstrates that attention architectures violate this assumption.
- **vs. Wenger et al. (2023)**: The latter notes that NTK theory applies only to networks far wider than they are deep; this paper provides a concrete attention-based counterexample and quantifies the required width.
- **vs. Nichani et al. (2025)**: The latter offers provable guarantees for feature learning; this paper provides specific architectural evidence that attention naturally satisfies feature learning conditions.
- **vs. Hron et al. (2020)**: The latter extends NTK theory to multi-head attention (converging to a GP as heads→∞); this paper focuses on non-convergence phenomena at finite width.
- **vs. Performers (Choromanski et al., 2021)**: Performers provide an efficient implementation of linearized attention; this paper analyzes the essential learning dynamics of linearized attention from the NTK-theoretic perspective.
- **vs. Zhang et al. (2022)**: The latter establishes the NTK influence function framework; this paper builds on it by introducing influence malleability to compare architectural differences.
- **vs. Chizat et al. (2019)**: The theoretical distinction between lazy and feature learning regimes; this paper provides concrete evidence that attention naturally resides in the feature learning regime.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The theoretical chain from spectral amplification to NTK non-convergence is novel; the influence malleability metric is proposed for the first time.
- **Experimental Thoroughness**: ⭐⭐⭐ — Theoretical validation is solid, but dataset scale is limited and large-model experiments are absent.
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretical derivations are clear and the dual-implication narrative is persuasive.
- **Value**: ⭐⭐⭐⭐ — Provides a new perspective on understanding the nature of attention mechanisms, with implications for adversarial robustness research.
- **Overall**: ⭐⭐⭐⭐ — A rigorous theoretical contribution. The core insight—that the power and vulnerability of attention share a common origin—is both elegant and practically informative. Extension to softmax attention and large-scale models would further strengthen its impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Test-time Ego-Exo-centric Adaptation for Action Anticipation via Multi-Label Prototype Growing and Dual-Clue Consistency](test-time_ego-exo-centric_adaptation_for_action_anticipation_via_multi-label_pro.md)
- [\[CVPR 2026\] DecoVLN: Decoupling Observation, Reasoning, and Correction for Vision-and-Language Navigation](decovln_decoupling_observation_reasoning_and_correction_for_vision-and-language_.md)
- [\[CVPR 2026\] RC-NF: Robot-Conditioned Normalizing Flow for Real-Time Anomaly Detection in Robotic Manipulation](rc-nf_robot-conditioned_normalizing_flow_for_real-time_anomaly_detection_in_robo.md)
- [\[CVPR 2026\] Cross-Domain Demo-to-Code via Neurosymbolic Counterfactual Reasoning](cross-domain_demo-to-code_via_neurosymbolic_counterfactual_reasoning.md)
- [\[CVPR 2026\] PanoAffordanceNet: Towards Holistic Affordance Grounding in 360° Indoor Environments](panoaffordancenet_towards_holistic_affordance_grounding_in_360_indoor_environmen.md)

</div>

<!-- RELATED:END -->
