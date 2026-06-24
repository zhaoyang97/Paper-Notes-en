---
title: >-
  [Paper Note] Certified but Fooled! Breaking Certified Defences with Ghost Certificates
description: >-
  [AAAI 2026][adversarial attack] This paper proposes GhostCert, a salient-region-based adversarial attack that misleads classifiers while maintaining imperceptible perturbations and forging large-radius robustness certificates (ghost certificates). On ImageNet, GhostCert achieves substantially higher attack success rates and larger spoofed certification radii than Shadow Attack against state-of-the-art certified defences including DensePure.
tags:
  - "AAAI 2026"
  - "adversarial attack"
  - "certified defence"
  - "randomized smoothing"
  - "certificate spoofing"
  - "region-based perturbation"
date: 2026-05-08
content_hash: 7600dd0f033ffe9b
---

# Certified but Fooled! Breaking Certified Defences with Ghost Certificates

**Conference**: AAAI 2026
**arXiv**: [2511.14003](https://arxiv.org/abs/2511.14003)  
**Code**: [github.com/ghostcert](https://github.com/ghostcert)  
**Area**: Other
**Keywords**: adversarial attack, certified defence, randomized smoothing, certificate spoofing, region-based perturbation

## TL;DR

This paper proposes GhostCert, a salient-region-based adversarial attack that misleads classifiers while maintaining imperceptible perturbations and forging large-radius robustness certificates (ghost certificates). On ImageNet, GhostCert achieves substantially higher attack success rates and larger spoofed certification radii than Shadow Attack against state-of-the-art certified defences including DensePure.

## Background & Motivation

### Problem Definition

**Certified defences** (e.g., Randomized Smoothing) promise provable robustness guarantees: within an $\ell_2$ perturbation ball, the predictions of a smoothed classifier remain unchanged. But how reliable are these guarantees in practice?

**Certificate Spoofing**: an adversary not only induces misclassification but also manipulates the certification process so that a high-confidence robustness certificate is issued for a malicious input.

### Limitations of Prior Work

**Shadow Attack** (ICLR 2020) is the only prior work targeting certified defences, but it has critical drawbacks:
1. It applies **large-magnitude global perturbations** to move inputs far from the decision boundary.
2. It relies on total variation (TV), color channel mean, and other regularizers to maintain a "natural" appearance, resulting in a **complex multi-objective optimization**.
3. Its attack success rate drops substantially against ensemble models (ASR ≈ 40%).
4. The generated adversarial examples are **visually unnatural** ($\|\delta\|_2$ typically >10).

### Core Motivation

Large-magnitude perturbations are unnecessary — by **constraining perturbations to semantically relevant salient regions**, higher attack success rates and larger spoofed certification radii can be achieved with smaller perturbations.

## Method

### Overall Architecture

GhostCert proceeds in three steps:
1. **Region proposal generation**: semantic segmentation masks are generated using SAM (Segment Anything Model).
2. **Salient region selection**: GradCAM/Attention saliency information is used to select the top-$k$ regions.
3. **Constrained perturbation optimization**: adversarial perturbations are optimized within the selected regions via PGD.

### Key Designs

#### 1. **Salient-Region Mask**

GradCAM gradient saliency is combined with SAM semantic segmentation boundaries:

- For each SAM segmentation mask $M_i$, an overlap score with the saliency map $\mathcal{S}$ is computed:

$$\text{score}(M_i) = \frac{\sum_{x,y} M_i(x,y) \cdot \mathcal{S}(x,y)}{\sum_{x,y} M_i(x,y) + \sum_{x,y} \mathcal{S}(x,y)}$$

- The top-$k=5$ masks (including the region $U$ not covered by any mask) are selected and merged into a final mask: $m = \sum_{M \in \mathcal{T}} M$.

**Design Motivation**: GradCAM saliency maps produce amorphous regions that ignore natural image boundaries, leading to unnatural artifacts. SAM segmentation masks provide semantically coherent boundaries. Their combination ensures that perturbations are semantically meaningful and visually natural.

#### 2. **Constrained Perturbation Optimization**

**Untargeted attack** optimization objective:

$$\max_\delta \sum_{i=1}^N L(f_\theta(x + \Delta_i + \delta \odot m), y) \quad \text{s.t.} \|\delta \odot m\|_2 \leq \epsilon$$

where $\Delta_i$ is Gaussian noise with standard deviation $\sigma$, $\odot$ denotes element-wise multiplication, and $m$ is the selected region mask.

**Targeted attacks** replace the loss with maximization of the target class probability.

PGD gradient ascent is applied: $\delta \leftarrow \delta + \lambda \cdot \frac{g}{\|g\|_2}$, followed by projection: $\delta \leftarrow (\epsilon \frac{\delta}{\|\delta\|_2}) \odot m$.

#### 3. **Adaptation to Different Defence Methods**

The optimization objective is adjusted for each of the three certified defences:

- **Randomized Smoothing (RS)**: $\max_\delta \sum_i L(f(x + \epsilon_i + \delta), y)$
- **Smoothed Ensemble**: $\max_\delta \sum_i L(\bar{f}(x + \epsilon_i + \delta), y)$
- **DensePure (denoised smoothing)**: $\max_\delta \sum_i L(f(D_\theta(x + \epsilon_i + \delta)), y)$

Attention maps replace GradCAM for Transformer-based models.

### Loss & Training

- **Cross-entropy loss** is used throughout.
- PGD step size $\lambda = 0.0001$.
- $N=1000$ Monte Carlo samples per attack to estimate the smoothed probability.
- Perturbation budget $\epsilon \in \{2, 4, 6, 8, 10\}$.
- Failure probability for Randomized Smoothing $\alpha = 0.001$.

## Key Experimental Results

### Main Results

**Untargeted Attack Success Rate (ASR) comparison**:

| Defence | $\sigma$ | $\epsilon$ | GhostCert | Shadow (bounded) | Shadow (original) |
|---------|--------|----------|-----------|-----------------|-------------|
| Single RS (ResNet50) | 0.25 | 10 | **~98%** | ~60% | ~65% |
| Ensemble RS | 0.25 | 10 | **~100%** | ~40% | ~40% |
| Ensemble RS | 0.5 | 10 | **~85%** | ~35% | ~30% |
| DensePure | 0.25 | 10 | **~100%** | ~50% | ~55% |
| DensePure | 0.5 | 10 | **~55%** | ~35% | ~40% |

**Spoofed certification radius**: GhostCert consistently produces larger or comparable spoofed certification radii, and frequently **exceeds the true certified radius of the source image** ("strongly certified").

### Ablation Study

| Configuration | ASR (untargeted, $\epsilon$=10) | ASR (targeted, $\epsilon$=10) | Note |
|------|--------------------------|--------------------------|------|
| GhostCert (full, k=5) | **90%** | **30%** | Full method |
| Random pixel mask (50%) | 90% | 5% | Much weaker on targeted |
| Random region proposals (k=5, no saliency) | 45% | 0% | Saliency selection critical |
| k=3 | 90% | 15% | Insufficient region coverage |
| k=7 | 90% | 35% | Comparable to k=5 |

### Key Findings

1. **GhostCert's advantage is most pronounced against ensemble defences**: Shadow Attack's ASR drops sharply from ~65% to ~40% on ensemble models, while GhostCert maintains ~100%.
2. **Lower perturbation with larger spoofed radius**: GhostCert at $\|\delta\|_2=4$ matches the effect of Shadow Attack at $\|\delta\|_2=13$.
3. **Human user study**: on Amazon Mechanical Turk, images generated by GhostCert are rated as more natural at all perturbation levels (74% at $\epsilon$=2, 62% at $\epsilon$=10).
4. **DoS attack**: when ASR is low, GhostCert inputs more frequently cause the certification method to **abstain**, constituting a denial-of-service attack.
5. **Important clarification**: the attack does not invalidate certificates — the certificate's assertion that inputs within the bounded norm ball are not adversarial remains correct. The attack reveals the **practical security boundary** of the certification framework.

## Highlights & Insights

1. **Paradigm shift from multi-objective optimization to constrained optimization**: Shadow Attack incorporates semantic constraints as regularization terms in the loss (multi-objective); GhostCert converts them into constraints via region masks (narrowing the search space), yielding greater efficiency.
2. **SAM + GradCAM combination** is the key innovation: GradCAM identifies "what is important," while SAM determines "how to respect boundaries."
3. **A new threat dimension**: certificate spoofing not only causes misclassification but also generates a "false sense of security" — particularly dangerous in safety-critical applications such as autonomous driving and medical imaging.
4. A fundamental limitation of certified defences is revealed: $\ell_2$-norm-bounded certification may be insufficient in the semantic space.

## Limitations & Future Work

1. **White-box threat model**: full access to model parameters is required, which may not be available to real-world adversaries.
2. ASR against DensePure drops significantly at $\sigma=0.5$, indicating that stronger denoisers still offer meaningful defence potential.
3. Each attack requires 1000 Monte Carlo samples, incurring substantial computational cost.
4. The user study scale is limited (~50 participants), which may introduce bias.
5. The attack's effectiveness against other certification methods (e.g., deterministic certification) is not explored.
6. The computational overhead of the SAM model itself is not accounted for.

## Related Work & Insights

- **Shadow Attack** (Ghiasi et al., ICLR 2020) is the direct baseline; GhostCert outperforms it comprehensively across all dimensions.
- **Randomized Smoothing** (Cohen et al., 2019) provides the certification radius formula $R = \frac{\sigma}{2}[\Phi^{-1}(p_A) - \Phi^{-1}(p_B)]$, which serves as the direct optimization target for the attack.
- **DensePure** (Xiao et al., 2022), which employs a diffusion-model-based denoiser, represents the current strongest certified defence; GhostCert provides the first systematic evaluation against it.
- The region-constrained perturbation idea proposed in this paper may inspire more natural perturbation models in broader adversarial example research.

## Rating

- Novelty: ⭐⭐⭐⭐ (SAM+GradCAM region selection is novel; certificate spoofing scenario is important)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (three defences × three attacks × multiple σ/ε + targeted/untargeted + user study + ablation)
- Writing Quality: ⭐⭐⭐⭐ (clear structure, intuitive experimental presentation)
- Value: ⭐⭐⭐⭐ (an important security warning for the certified defence community, prompting reflection on certification reliability)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Faster Certified Symmetry Breaking Using Orders With Auxiliary Variables](faster_certified_symmetry_breaking_using_orders_with_auxiliary_variables.md)
- [\[AAAI 2026\] Certified Branch-and-Bound MaxSAT Solving (Extended Version)](certified_branch-and-bound_maxsat_solving_extended_version.md)
- [\[ICLR 2026\] Any-Subgroup Equivariant Networks via Symmetry Breaking](../../ICLR2026/others/any-subgroup_equivariant_networks_via_symmetry_breaking.md)
- [\[AAAI 2026\] Axis-Aligned Document Dewarping](axis-aligned_document_dewarping.md)
- [\[ICLR 2026\] Breaking Gradient Temporal Collinearity for Robust Spiking Neural Networks](../../ICLR2026/others/breaking_gradient_temporal_collinearity_for_robust_spiking_neural_networks.md)

</div>

<!-- RELATED:END -->
