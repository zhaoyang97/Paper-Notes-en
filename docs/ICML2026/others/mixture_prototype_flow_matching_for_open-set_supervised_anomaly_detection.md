---
title: >-
  [Paper Note] Mixture Prototype Flow Matching for Open-Set Supervised Anomaly Detection
description: >-
  [ICML 2026][Open-Set Supervised Anomaly Detection] MPFM replaces the traditional "unimodal Gaussian prototype" in OSAD with a learnable **Gaussian mixture prototype space**…
tags:
  - "ICML 2026"
  - "Open-Set Supervised Anomaly Detection"
  - "Flow Matching"
  - "Gaussian Mixture Prototype"
  - "Mutual Information Regularization"
  - "OSAD"
date: 2026-05-08
content_hash: 3c2bc2ebdcb6298e
---

# Mixture Prototype Flow Matching for Open-Set Supervised Anomaly Detection

**Conference**: ICML 2026  
**arXiv**: [2605.02438](https://arxiv.org/abs/2605.02438)  
**Code**: https://github.com/fuyunwang/MPFM-OSAD  
**Area**: Anomaly Detection / Flow Matching / Prototype Learning  
**Keywords**: Open-Set Supervised Anomaly Detection, Flow Matching, Gaussian Mixture Prototype, Mutual Information Regularization, OSAD

## TL;DR
MPFM replaces the traditional "unimodal Gaussian prototype" in OSAD with a learnable **Gaussian mixture prototype space**, directly regresses a GMM-form velocity field via flow matching, and adds a mutual information maximization regularizer to prevent prototype collapse. On 9 industrial/medical AD datasets, under the 10/1 anomaly sample setting, it outperforms all SOTA methods including DRA, AHL, and DPDL.

## Background & Motivation

**Background**: Anomaly detection has three paradigms: unsupervised AD (only normal samples), few-shot AD (very few normal samples), and supervised AD (requires large numbers of anomaly samples). Open-set supervised AD (OSAD) is a compromise: training uses a small number of known-class anomaly images (image-level labels) plus many normal images; testing must detect **both known and unseen** anomaly types. Existing OSAD methods fall into three categories: (a) data augmentation + outlier exposure (DRA), (b) heterogeneous simulation (AHL), (c) prototype learning + generative dynamics (DPDL).

**Limitations of Prior Work**: The third, SOTA-approaching DPDL, uses a set of **simple unimodal Gaussians** as the prototype distribution for normal samples, then uses a diffusion bridge to "pull" normal features toward these prototypes. The problem is that in industrial scenarios, normal samples are inherently **multimodal** (rich sub-patterns/angles/lighting within a class), so forcing a unimodal Gaussian will misclassify rare but legitimate normal sub-patterns as anomalies, causing many false positives and fuzzy decision boundaries.

**Key Challenge**: The goal is to make the density estimation for normal samples **compact** (high recall on normal), while also **extrapolating the boundary to unseen anomaly types** (high recall on unknown anomalies). The unimodal Gaussian prior directly undermines the first goal, while discrete multi-Gaussians lack structural relationships, so the model does not know how modes transition, losing semantic continuity.

**Goal**: (1) Replace the unimodal Gaussian with a **continuous, structured** multimodal prototype space; (2) Make the flow-matching velocity field itself multimodal, not just regressing a mean vector; (3) Prevent multiple Gaussian components from collapsing into a single mode.

**Key Insight**: Explicitly construct the prototype space as a GMM, and use flow matching to learn a continuous transport mapping from the normal feature distribution to this GMM. The key observation: under a standard linear noise schedule, a single-step reverse transition preserves GMM closure (closure of linear-Gaussian systems), so step-wise sampling can be done in closed form, without numerical integration.

**Core Idea**: **Use a GMM-form velocity field instead of a single velocity vector for flow matching, propagate the multimodal prior from prior to transport dynamics, and use mutual information maximization to separate GMM components.**

## Method

### Overall Architecture

Input: Training set $\mathcal{Z}_{tr} = \mathcal{Z}_{tr}^{n} \cup \mathcal{Z}_{tr}^{a}$ (normal $N$ + anomaly $M$, $N \gg M$, only image-level labels); Output: anomaly score $S(z)$ for test sample $z \in \mathcal{Z}_{te}$.

Pipeline:
1. **Feature Extraction**: ResNet-18 backbone $f: \mathcal{Z} \to \mathbb{R}^d$, maps images to 1D features.
2. **K-means++ GMM Initialization**: Run K-means++ on $\mathcal{F}_{tr}^{n}$ to get $K$ cluster centers $\mu_k$, mixture weights $\pi_k = |C_k| / N$, shared variance $s^2 = \frac{1}{dN} \sum_k \sum_{i \in C_k} \| z_0^{n,i} - \mu_k \|_2^2$. This breaks the "flow network and GMM parameter mutual dependency" optimization deadlock.
3. **Flow Matching Training**: Linear noise schedule $\alpha_t = 1-t, \sigma_t = t$, velocity $u = \frac{z_T - z_0}{t}$, maximize GMM velocity field likelihood on normal samples, **reverse** minimize on anomaly samples (i.e., push anomalous velocity away from normal velocity distribution).
4. **MIMR Regularization**: On normal samples, maximize mutual information between "prototype assignment $c$" and "post-flow feature $\psi(z_0^{n,i})$", ensuring both confident assignment and balanced usage.
5. **Four-Module Anomaly Score Prediction**: Global $M_g$ (GMM NLL) + Local $M_a$ (top-O patch score) + Normal $M_n$ (global pooling) + Residual $M_r$ ($(\psi(z) - \mu_{c^*})/s$ via classification head); at inference, $S(z) = S_g + S_a + S_r - S_n$.

### Key Designs

1. **Mixture Prototype Flow Learning (MPFL): GMM-form Velocity Field**:

    - **Function**: Replaces the traditional "flow matching → unimodal Gaussian" mapping with "flow matching → multimodal GMM", each component representing a normal sub-pattern.
    - **Mechanism**: Instead of $q_\theta(u | z_t) = \mathcal{N}(u; \mu_\theta(z_t), s^2 I)$ (unimodal conditional Gaussian), directly model the velocity field as a mixture of Gaussians $q_\theta(u | z_t^{n,i}) = \sum_{k=1}^K \pi_k(z_t^{n,i}; \theta) \mathcal{N}(u; \mu_k(z_t^{n,i}; \theta), s^2 I)$, where mixture weights $\pi_k$ and means $\mu_k$ are functions of the input feature. Training uses NLL: $\mathcal{L}_{NLL} = \mathbb{E} [-\log \sum_k \pi_k \mathcal{N}(u; \mu_k, s^2 I)]$. Under the standard linear schedule, this GMM structure allows closed-form reverse sampling; a single-step reverse transition $q_\theta(z_{t-\Delta t}^{n,i} | z_t^{n,i})$ remains a GMM, with coefficients $c_1, c_2, c_3$ fully determined by the noise schedule, so no numerical ODE solving is needed at inference.
    - **Design Motivation**: A unimodal velocity field inevitably pulls all normal samples toward a single mean, flattening multimodal sub-patterns—this is the root cause of high false positives in DPDL for industrial AD. Encoding multimodal structure into the velocity field fundamentally solves the problem, rather than post-hoc fixes.

2. **Forward/Reverse Flow Matching Loss (Bidirectional Flow Loss)**:

    - **Function**: Normal samples are attracted to the GMM prototype space, anomalies are repelled.
    - **Mechanism**: Normal samples use standard NLL $\mathcal{L}_{flow}^{n} = \mathbb{E}[-\log q_\theta(u | z_t^{n,i})]$, anomalies use **negated** $\mathcal{L}_{flow}^{a} = \mathbb{E}[\log q_\theta(u | z_t^{a,i})]$—i.e., maximize NLL for anomalies, pushing their velocity distribution away from the normal GMM. Note that the true anomaly velocity $u^{a} = (z_T^{a} - z_0^{a}) / t$ is computed as if it were a normal sample under the linear schedule, and the model is forced to "not look like it".
    - **Design Motivation**: Traditional OSAD uses contrastive loss or binary classification to push anomalies away, but only in the final feature space; here, positive/negative separation is done **at the velocity field level**, so the repulsion acts along the entire transport trajectory, with gradient signals at every step from $t = 0$ to $t = T$.

3. **Mutual Information Maximization Regularizer (MIMR)**:

    - **Function**: Prevents GMM components from collapsing into the same mode, maintaining component distinguishability.
    - **Mechanism**: Leverages the GMM's own posterior $p(c=k | \psi(z_0^{n,i})) = \frac{\pi_k \mathcal{N}(\psi(z_0^{n,i}); \mu_k, s^2 I)}{\sum_j \pi_j \mathcal{N}(\psi(z_0^{n,i}); \mu_j, s^2 I)}$ (obtained for free, no extra parameters), then maximizes mutual information $I(\psi(z_0^{n,i}); c) = H(c) - H(c | \psi(z_0^{n,i}))$. As a loss: $\mathcal{L}_{mim} = \mathbb{E}[\sum_k p(c=k|\cdot) \log p(c=k|\cdot)] - \sum_k \pi_k \log \pi_k$. The first term **minimizes** conditional entropy (each sample is clearly assigned to a prototype), the second **maximizes** marginal entropy (components are used evenly).
    - **Design Motivation**: Directly using the GMM posterior for entropy constraints requires no extra discriminator or adversarial training, making it lightweight in practice. MIMR is only computed on normal samples (not anomalies), otherwise anomalies would be pulled toward prototypes, which is counterproductive.

### Loss & Training

Total loss:
$\mathcal{L} = \underbrace{\mathcal{L}_{M_a} + \mathcal{L}_{M_n} + \mathcal{L}_{M_r} + \mathcal{L}_{M_g}}_{\text{Anomaly Score Modules}} + \underbrace{\mathcal{L}_{flow}^{n} + \mathcal{L}_{flow}^{a}}_{\text{Flow Matching}} + \lambda \mathcal{L}_{mim}$.

All four anomaly score modules use binary classification loss (BCE), trained jointly. $\lambda$ controls MIMR strength. At inference, $S(z) = S_g(z) + S_a(z) + S_r(z) - S_n(z)$—note $S_n$ measures "normal-likeness", so it is subtracted.

## Key Experimental Results

### Main Results

On 9 real AD datasets (MVTec AD / Optical / SDD / etc.), under the general setting of 10 training anomaly samples, AUC is reported (mean±std over 3 seeds).

| Dataset | DRA | AHL | DPDL | **MPFM (Ours)** |
|---------|------|------|------|------|
| MVTec AD | 0.959±0.003 | 0.970±0.002 | 0.977±0.002 | **0.982±0.003** |
| Optical | 0.965±0.006 | 0.976±0.004 | 0.983±0.005 | **0.992±0.002** |
| SDD | 0.991±0.005 | — | — | (best, see paper) |

MPFM ranks first or tied for first on all reported datasets, improving over the strongest baseline DPDL by 0.5 AUC points on MVTec AD and 0.9 points on Optical (note AUC is already at the 0.97+ ceiling).

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|-----------|-------------|
| Full MPFM (GMM flow + MIMR + 4 score modules) | best AUC | Complete model |
| w/o GMM (degrades to unimodal Gaussian velocity field) | significant drop | Validates necessity of multimodal velocity field, equivalent to DPDL approach |
| w/o MIMR | moderate drop | Validates effect of anti-collapse regularizer |
| w/o reverse flow loss $\mathcal{L}_{flow}^{a}$ | large drop | Anomaly samples only participate in score module training, lose transport-level repulsion |

### Key Findings

- **Multimodal velocity field is the main contributor**: Removing the GMM structure (degrading to unimodal) causes the largest drop, directly validating the motivation that "unimodal Gaussian cannot capture normal sub-pattern diversity".
- **K-means++ initialization is critical for convergence**: The paper emphasizes that "flow network and GMM parameter coupling" is an optimization challenge; without K-means++ prior initialization, bad local minima are likely.
- **Mutual information regularization vs. distance regularization**: MIMR leverages the GMM posterior for entropy, more refined than the common "pairwise mean distance constraint", as it simultaneously controls confident assignment (conditional entropy) and balanced usage (marginal entropy).
- **Shared variance $s^2$ is an engineering stability trick**: All components share a single $s$ to avoid variance explosion or collapse in any component; the supplementary material discusses this in detail.

## Highlights & Insights

- **Multimodality propagates from prior to transport**: This is the main difference between MPFM and all previous "prototype + bridge flow" methods—others have "flow endpoint is GMM" but the flow process is still unimodal; MPFM's flow process itself is a GMM velocity field, so multimodality is present at every step. This "prior-through-dynamics" philosophy can be generalized to any representation learning task using diffusion/flow.
- **Closed-form reverse sampling preserves GMM closure**: By leveraging the analytic properties of linear-Gaussian systems, a single-step reverse $q_\theta(z_{t-\Delta t} | z_t)$ remains a GMM, so inference is **fully closed-form**, no ODE solver needed—this is key to making GMM velocity fields practically usable.
- **Reverse flow matching loss**: By feeding anomaly samples into the same flow matching framework but negating the log-likelihood, transport-level repulsion is elegantly achieved, without introducing new modules or adversarial training.
- **Subtractive combination of four anomaly score modules**: $S = S_g + S_a + S_r - S_n$—this "add positives, subtract negatives" combination borrows the dual-view idea from anomaly detection: "look for both anomaly-likeness and non-normality", which is often overlooked but practically useful.

## Limitations & Future Work

- **K is a hyperparameter and requires K-means++ pre-run**, with no discussion on how to adaptively select K—different MVTec subclasses have very different numbers of normal modes (carpet vs. transistor are on different scales), so using the same K for all is suboptimal.
- **Only tested on ResNet-18**, not on stronger backbones (CLIP / DINOv2), though backbone expressiveness greatly affects prototype space structure.
- **MIMR is only computed on normal samples**, and does not include an explicit constraint that "anomaly samples should not be confidently assigned to any prototype", so may still fail when anomaly distribution overlaps with a normal sub-pattern.
- **No pixel-level localization evaluation**, only image-level AUC; does not fully address industrial needs for "pointing out where the anomaly is in the image".

## Related Work & Insights

- **vs DPDL**: They use a set of discrete, independent unimodal Gaussians and a simple velocity vector for flow; MPFM uses a structured, continuous GMM and a GMM velocity field, with semantic relationships among multimodal structures. It is a direct extension of DPDL.
- **vs DRA / AHL**: They expand anomaly coverage via data augmentation or heterogeneous simulation, essentially "creating more anomaly samples"; MPFM keeps the data fixed and changes "how normal samples are modeled", a fundamentally orthogonal approach.
- **vs standard Rectified Flow / SD3**: The latter are generative models, using flow matching to map noise to images; MPFM uses the same flow matching tool in reverse as a "discriminative representation" framework, using flow as a density estimator for anomaly detection—a good example of cross-task transfer for flow matching.

## Rating
- Novelty: ⭐⭐⭐⭐ The "GMM-form velocity field + mutual information regularization + reverse flow loss" trio is a new combination in OSAD, though each component (GMM flow / MIMR / reverse NLL) has precedents in generative or clustering fields.
- Experimental Thoroughness: ⭐⭐⭐⭐ 9 real datasets + multiple seeds + thorough ablation, but lacks stronger backbones and pixel-level evaluation.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations (especially closed-form reverse sampling) are rigorous, Algorithm 1 is clear, and the motivation figure visually demonstrates "unimodal vs. multimodal boundary" convincingly.
- Value: ⭐⭐⭐⭐ Provides a stable, practical new baseline for industrial AD, and the "prior-through-dynamics" design philosophy is transferable to other representation learning tasks using diffusion/flow.

## Related Papers

- [\[CVPR 2026\] OpenDPR: Open-Vocabulary Change Detection via Vision-Centric Diffusion-Guided Prototype Retrieval for Remote Sensing Imagery](../../CVPR2026/image_generation/opendpr_open-vocabulary_change_detection_via_vision-centric_diffusion-guided_pro.md)
- [\[NeurIPS 2025\] Scalable, Explainable and Provably Robust Anomaly Detection with One-Step Flow Matching](../../NeurIPS2025/image_generation/scalable_explainable_and_provably_robust_anomaly_detection_with_one-step_flow_ma.md)
- [\[ICML 2025\] Gaussian Mixture Flow Matching Models](../../ICML2025/image_generation/gaussian_mixture_flow_matching_models.md)
- [\[ICML 2026\] Exploring and Exploiting Stability in Latent Flow Matching](exploring_and_exploiting_stability_in_latent_flow_matching.md)
- [\[ICML 2026\] The Coupling Within: Flow Matching via Distilled Normalizing Flows](the_coupling_within_flow_matching_via_distilled_normalizing_flows.md)

</div>

<!-- RELATED:END -->
