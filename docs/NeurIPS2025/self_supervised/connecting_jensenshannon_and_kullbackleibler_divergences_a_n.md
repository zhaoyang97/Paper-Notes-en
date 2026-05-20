---
title: >-
  [Paper Note] Connecting Jensen-Shannon and Kullback-Leibler Divergences: A New Bound for Representation Learning
description: >-
  [NeurIPS 2025][Self-Supervised Learning][mutual information] This paper derives the optimal tight lower bound of KL divergence in terms of JS divergence, $\Xi(D_{\text{JS}}) \leq D_{\text{KL}}$…
tags:
  - "NeurIPS 2025"
  - "Self-Supervised Learning"
  - "mutual information"
  - "Jensen-Shannon divergence"
  - "KL divergence"
  - "variational bound"
  - "representation learning"
date: 2026-05-08
content_hash: c8e754e1ed6a6f85
---

# Connecting Jensen-Shannon and Kullback-Leibler Divergences: A New Bound for Representation Learning

**Conference**: NeurIPS 2025
**arXiv**: [2510.20644](https://arxiv.org/abs/2510.20644)  
**Code**: [https://github.com/ReubenDo/JSDlowerbound](https://github.com/ReubenDo/JSDlowerbound)  
**Area**: Self-Supervised Learning / Information Theory
**Keywords**: mutual information, Jensen-Shannon divergence, KL divergence, variational bound, representation learning

## TL;DR
This paper derives the optimal tight lower bound of KL divergence in terms of JS divergence, $\Xi(D_{\text{JS}}) \leq D_{\text{KL}}$, in the general case. It proves that training a discriminator by minimizing cross-entropy loss is equivalent to maximizing a guaranteed lower bound on mutual information, thereby providing the missing theoretical foundation for JSD-based discriminative representation learning methods. The tightness and practical utility of the bound are validated in MI estimation and the Information Bottleneck framework.

## Background & Motivation

**The central role of mutual information in representation learning**: Mutual information (MI), as the standard measure of statistical dependence, serves as the theoretical cornerstone of numerous representation learning frameworks—from the Information Bottleneck to contrastive learning (CPC/SimCLR)—whose objectives can be formulated as maximizing or constraining MI. MI is defined as the KL divergence between the joint distribution and the product of marginals: $I[U;V] = D_{\text{KL}}[p_{UV} \| p_U \otimes p_V]$.

**The challenge of directly optimizing MI**: In practice, directly optimizing MI is generally intractable. Existing variational lower bound (VLB) methods such as MINE (based on the Donsker-Varadhan representation) and NWJ provide optimizable lower bounds on MI but suffer from high variance, instability, and the need for adversarial training. InfoNCE yields a stable lower bound via contrastive learning but is constrained by batch size $b$—the estimate is upper-bounded by $\log b$ and fails to capture high MI values.

**The empirical success and theoretical gap of JSD-based optimization**: In practice, many successful methods (e.g., Deep InfoMax) bypass direct KLD optimization and instead maximize the Jensen-Shannon divergence (JSD) between the joint distribution and the product of marginals, i.e., $I_{\text{JS}}[U;V] = D_{\text{JS}}[p_{UV} \| p_U \otimes p_V]$. Due to its symmetry and boundedness (upper-bounded by $\log 2$), JSD optimization is more stable and does not require large batch sizes. Deep InfoMax's experiments show that JSD correlates well with true MI. However, a fundamental theoretical question remained unanswered: **Does maximizing JSD truly maximize MI? What is the quantitative relationship between the two?** Known inequalities—JSD ≤ MI (trivial; JSD is a weak lower bound of MI) and the Pinsker inequality $D_{\text{KL}} \geq 2 D_{\text{JS}}$—are either too loose or tight only in the low-MI regime, and neither provides a general tight quantitative connection.

**Goal**: To fill this theoretical gap by deriving the optimal (i.e., tightest possible, unimprovable) lower bound on KLD in terms of JSD, thereby rigorously proving that maximizing JSD genuinely increases a guaranteed lower bound on MI, and fully connecting this theoretical result to discriminator training via cross-entropy minimization.

## Method

### Overall Architecture
The paper establishes a complete theoretical chain from discriminator training to MI maximization: training a discriminator to distinguish joint from marginal samples (minimizing cross-entropy $\mathcal{L}_{\text{CE}}$) → increasing a variational lower bound on JSD ($I_{\text{JS}} \geq \log 2 - \mathcal{L}_{\text{CE}}$) → increasing a guaranteed lower bound on MI ($I[U;V] \geq \Xi(I_{\text{JS}})$). The end-to-end MI lower bound is: $\Xi(\log 2 - \mathcal{L}_{\text{CE}}) \leq \Xi(I_{\text{JS}}) \leq I[U;V]$.

### Key Designs

1. **Optimal Lower Bound of KLD in Terms of JSD (Theorem 4.1)**:

    - **Function**: Derives the tightest KLD-JSD inequality $\Xi(D_{\text{JS}}[p \| q]) \leq D_{\text{KL}}[p \| q]$ holding for any pair of distributions $p, q$.
    - **Mechanism**: The proof leverages the theory of *joint range* of $f$-divergences—given two $f$-divergences $D_f$ and $D_g$, the set of jointly attainable values $\mathcal{R}_{f,g}$ over all distribution pairs $(p,q)$ is a convex set, and its lower envelope is the optimal lower bound. The key tool is the Harremoës-Vajda theorem (Theorem 3.1): the complete joint range $\mathcal{R}_{f,g}$ is fully characterized by Bernoulli distribution pairs, i.e., $\mathcal{R}_{f,g} = \text{co}(\mathcal{R}_{2;f,g})$. It therefore suffices to analyze the image of the map $\phi: (\mu, \nu) \mapsto (D_{\text{JS}}[B(\mu) \| B(\nu)], D_{\text{KL}}[B(\mu) \| B(\nu)])$ on the unit square. Analysis of the three boundary segments reveals that the lower bound is given precisely by the curve corresponding to the edge $\mu=1$ (i.e., $D_{\text{KL}}[B(1) \| B(\nu)]$). The function $\Xi$ is strictly increasing, and its inverse has an analytic form: $\Xi^{-1}(y) = D_{\text{JS}}[B(1) \| B(e^{-y})]$.
    - **Design Motivation**: The Pinsker inequality $D_{\text{KL}} \geq 2 D_{\text{JS}}$ is tight only as divergences approach zero. Although the joint range theory of $f$-divergences has been studied in classical information theory, it had never been applied in the context of representation learning and MI estimation. Bringing this classical tool into modern machine learning is the paper's core theoretical contribution.

2. **Cross-Entropy Variational Lower Bound on JSD**:

    - **Function**: Proves that training a discriminator to minimize cross-entropy loss is equivalent to maximizing a variational $f$-divergence lower bound on JSD.
    - **Mechanism**: A mixture model is defined: $(U,V) | Z=1 \sim p_{UV}$ (joint samples), $(U,V) | Z=0 \sim p_U \otimes p_V$ (independent samples), $Z \sim B(1/2)$. Using the variational representation of $f$-divergences, JSD is rewritten as $I_{\text{JS}} = \frac{1}{2}\max_t [\mathbb{E}_{p_{UV}}[t] - \mathbb{E}_{p_U \otimes p_V}[-\log(2-e^t)]]$. Setting $t(u,v) = \log(2 q_\theta(z=1|u,v))$ (a reparameterization of the discriminator output) and substituting yields $I_{\text{JS}} \geq \log 2 - \min_\theta \mathcal{L}_{\text{CE}}(\theta)$, where $\mathcal{L}_{\text{CE}}$ is the binary cross-entropy loss of the discriminator. This bound is tight in the nonparametric limit (i.e., with unlimited discriminator capacity and infinite data).
    - **Design Motivation**: Goodfellow et al. in the GAN literature noted the correspondence between the optimal discriminator and JSD, but did not establish a complete chain from the perspective of MI estimation. The paper's novel contribution is to quantify the gap introduced by a sub-optimal discriminator and connect it to the MI lower bound.

3. **MI Estimator Construction (Two-Step Approach)**:

    - **Function**: Based on the above theory, constructs a complete estimation pipeline from data to MI values.
    - **Mechanism**: Using the posterior of the mixture model, $I[U;V] = \mathbb{E}_{p_{UV}}[\mathbb{L}(\tilde{p}(z=1|u,v))]$, where $\mathbb{L}(\cdot) = \log \frac{x}{1-x}$ is the logit function. The two-step estimation proceeds as: (1) train discriminator $q_\theta$ to distinguish joint from marginal samples (minimize CE loss); (2) plug $q_\theta$ into the above expression to obtain the MI estimate. A smooth differentiable approximation of $\Xi$ is given by $\Xi(x) \approx 1.15 \cdot \mathbb{L}(0.5(x / \log 2 + 1))$, facilitating end-to-end optimization.
    - **Design Motivation**: The two-step approach decouples optimization (discriminator training) from estimation (MI computation), avoiding the instability that arises when the optimization objective and the estimation target are conflated in VLB methods. This strategy is equivalent to the GAN-DIME method, for which this paper provides the theoretical foundation.

### Loss & Training
The discriminator is trained with standard binary cross-entropy: positive samples are pairs $(u,v) \sim p_{UV}$ drawn from the joint distribution, and negative samples are independently paired $(u, v') \sim p_U \otimes p_V$ obtained by shuffling. The network architecture is a fully connected network (input dimension $2d$, two hidden layers of 256 ReLU units, scalar output). Training runs for 4000 steps with the Adam optimizer and a batch size of 64. Since $\Xi$ is strictly increasing, maximizing $\Xi(\log 2 - \mathcal{L}_{\text{CE}})$ is equivalent to minimizing $\mathcal{L}_{\text{CE}}$; thus the approximation of $\Xi$ does not affect the optimization process and is only used for plug-in MI estimation.

## Key Experimental Results

### Main Results
**Gaussian MI Estimation ($d=5$, batch size 64, staircase setting)**:

| Estimator | Low MI | Medium MI | High MI (20 nats) | Variance | Exceeds True MI |
|-----------|--------|-----------|-------------------|----------|-----------------|
| MINE | Accurate | Overestimates | Severely overestimates | Extremely high | Frequently |
| NWJ | Accurate | Underestimates | Severely underestimates | High | Rarely |
| CPC (InfoNCE) | Accurate | Capped at $\log b=4.16$ | Capped | Low | No (bounded) |
| **JSD-LB (Ours)** | **Accurate** | **Tight lower bound** | **Tight lower bound** | **Low** | **Never** |
| GAN-DIME (two-step MI) | Accurate | Accurate | Most accurate | Low | Occasionally slightly |

**Information Bottleneck (MNIST Classification)**:

| Method | Test Acc. (%) | Adv. Robustness ($\epsilon=0.1$) | Adv. Robustness ($\epsilon=0.3$) | OOD AUROC (%) |
|--------|--------------|----------------------------------|----------------------------------|---------------|
| VIB | 97.6 | 73.4 | 4.2 | 94.6 |
| NIB | 97.2 | 75.2 | 3.4 | 94.2 |
| DisenIB | 98.2 | 90.2 | 67.8 | — |
| **JSD-LB (Ours)** | **98.8** | **94.6** | **86.1** | **Highest** |

### Ablation Study
**Tightness Verification on Discrete Distributions**:

| Categories $k$ | Dependence $\alpha$ | $\Xi(I_{\text{JS}})$ | True MI | Gap |
|----------------|--------------------|-----------------------|---------|-----|
| 2 | 0→1 | Hugs lower bound | Hugs lower bound | Negligible |
| 50 | 0→1 | Hugs lower bound | Slight deviation | Small |
| 500 | 0→1 | Hugs lower bound | Slight deviation | Small |

**Generalization to Non-Gaussian Distributions (asinh / half-cube / Student / uniform)**:

| Distribution | JSD-LB | MINE | NWJ | CPC |
|--------------|--------|------|-----|-----|
| Cubic | Tight, low variance | High variance | Severely underestimates | Capped |
| Asinh | Tight, low variance | High variance | Underestimates | Capped |
| Student | Tight, low variance | High variance | Underestimates | Capped |

### Key Findings
- JSD-LB provides a tight and stable lower bound across all MI regimes (low/medium/high), without the high variance and overestimation risks of MINE/NWJ.
- On discrete distributions, the general lower bound (valid for arbitrary $p, q$) remains near-tight when specialized to joint-marginal pairs—an infinite family of discrete distribution pairs attain the lower bound curve exactly.
- Replacing traditional VLBs with JSD-LB in the Information Bottleneck yields improved classification accuracy (98.8%), adversarial robustness (86.1% at $\epsilon=0.3$ vs. 67.8%), and OOD detection performance.
- CPC/InfoNCE is constrained by batch size ($\leq \log 64 \approx 4.16$ nats) and fails entirely in the high-MI regime; JSD-LB has no such limitation.

## Highlights & Insights
- **Applying classical information-theoretic tools to modern problems**: The joint range of $f$-divergences—a classical concept from Harremoës & Vajda (2011)—had never previously been used in the representation learning literature. The authors introduce it into the MI estimation problem and obtain the optimal lower bound. This paradigm of "transplanting classical results across fields" is methodologically instructive.
- **A complete theoretical chain**: From CE loss → JSD VLB → KLD lower bound → MI guarantee, each step is accompanied by rigorous mathematical derivation and an explicit characterization of the gap. This provides the long-missing theoretical foundation for the entire family of JSD-based representation learning methods (Deep InfoMax, SMILE, etc.).
- **The logit-like shape of $\Xi$**: The bounding function $\Xi$ resembles the logit function—approximately linear near zero ($\Xi(x) \approx 2x$, reducing to Pinsker), and growing rapidly toward infinity as JSD approaches $\log 2$. This perfectly captures the behavior in the high-MI regime where JSD saturates while KLD continues to grow.

## Limitations & Future Work
- The contribution is primarily theoretical; direct application and validation in large-scale visual SSL tasks (e.g., SimCLR/BYOL on ImageNet) are absent.
- Although a smooth approximation of $\Xi$ is provided, the function lacks a closed-form expression and requires numerical computation or approximation.
- Being a lower bound, the estimate can be significantly below the true MI in some settings—particularly when the discriminator is sub-optimal and gaps accumulate.
- The tightness of the bound depends on the quality of the JSD estimator, requiring a sufficiently well-trained discriminator.

## Related Work & Insights
- **vs. MINE**: MINE directly estimates a VLB on KLD, suffers from high variance and potential overestimation (exceeding the true MI), and is prone to optimization instability. JSD-LB provides an indirect lower bound on KLD via JSD, with lower variance and a guaranteed non-exceedance property.
- **vs. InfoNCE/CPC**: InfoNCE gives the lower bound $I[U;V] \geq \log b - \mathcal{L}_{\text{InfoNCE}}$, which is constrained by batch size $b$. JSD-LB has no such constraint and is significantly superior in the high-MI regime.
- **vs. Deep InfoMax (Hjelm et al.)**: Deep InfoMax empirically observed good correlation between JSD and MI and used the JSD objective on that basis. This paper provides a rigorous proof: $\Xi(I_{\text{JS}}) \leq I[U;V]$, confirming that maximizing $I_{\text{JS}}$ genuinely increases a guaranteed lower bound on MI.
- **vs. GAN framework**: Goodfellow (2014) showed that under an optimal discriminator the GAN objective is an affine transformation of JSD. This paper views the same discriminator from the perspective of MI estimation, revealing a deep connection between GAN losses and MI maximization.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First to establish a general optimal tight lower bound between KLD and JSD; introduces a classical information-theoretic tool into modern representation learning with outstanding theoretical contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive MI estimation benchmarks (Gaussian / non-Gaussian / discrete) and convincing IB application, though large-scale visual SSL experiments are absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Theoretical derivations are rigorous and clear; Figures 1 and 2 intuitively visualize the joint range and bound tightness.
- **Value**: ⭐⭐⭐⭐ Provides the missing theoretical foundation for JSD-based discriminative SSL methods; the practical gains in IB (86.1% adversarial robustness) are impressive.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Soft Task-Aware Routing of Experts for Equivariant Representation Learning](soft_task-aware_routing_of_experts_for_equivariant_representation_learning.md)
- [\[NeurIPS 2025\] Adv-SSL: Adversarial Self-Supervised Representation Learning with Theoretical Guarantees](adv-ssl_adversarial_self-supervised_representation_learning_with_theoretical_gua.md)
- [\[NeurIPS 2025\] TRIDENT: Tri-Modal Molecular Representation Learning with Taxonomic Annotations and Structural Relationships](trident_tri-modal_molecular_representation_learning_with_taxonomic_annotations_a.md)
- [\[NeurIPS 2025\] STaRFormer: Semi-Supervised Task-Informed Representation Learning via Dynamic Attention-Based Regional Masking](starformer_semi-supervised_task-informed_representation_learning_via_dynamic_att.md)
- [\[CVPR 2026\] Representation Learning for Spatiotemporal Physical Systems](../../CVPR2026/self_supervised/representation_learning_for_spatiotemporal_physica.md)

</div>

<!-- RELATED:END -->
