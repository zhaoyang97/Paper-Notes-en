---
title: >-
  [Paper Note] Active Target Discovery under Uninformative Prior: The Power of Permanent and Transient Memory
description: >-
  [NEURIPS2025][Medical Imaging][active target discovery] This paper proposes EM-PTDM, a framework inspired by the dual-memory system in neuroscience. It leverages a pretrained diffusion model as "permanent memory" and inc…
tags:
  - "NEURIPS2025"
  - "Medical Imaging"
  - "active target discovery"
  - "diffusion model"
  - "Doob's h-transform"
  - "dual memory"
  - "EM algorithm"
  - "uninformative prior"
date: 2026-05-08
content_hash: 5e0b7b2d3369c168
---

# Active Target Discovery under Uninformative Prior: The Power of Permanent and Transient Memory

**Conference**: NEURIPS2025
**arXiv**: [2510.16676](https://arxiv.org/abs/2510.16676)  
**Code**: To be confirmed  
**Area**: Medical Imaging
**Keywords**: active target discovery, diffusion model, Doob's h-transform, dual memory, EM algorithm, uninformative prior

## TL;DR

This paper proposes EM-PTDM, a framework inspired by the dual-memory system in neuroscience. It leverages a pretrained diffusion model as "permanent memory" and incorporates a lightweight "transient memory" module based on Doob's h-transform to achieve efficient active target discovery **without any domain-specific prior data**, with theoretical guarantees of monotonic prior improvement.

## Background & Motivation

**Active Target Discovery (ATD)** is a core problem in science and engineering: in medical imaging, environmental monitoring, remote sensing, and similar settings, data acquisition is costly, requiring strategic sampling within a limited budget to identify target regions.

2. Existing methods such as DiffATD train diffusion models on in-domain samples as strong priors to guide sampling, achieving strong performance in data-rich domains.

**Key bottleneck**: In scenarios such as emerging disease diagnosis, rare species discovery, and underexplored geographic regions, it is fundamentally infeasible to collect sufficient domain samples for learning strong priors, rendering DiffATD-like approaches ineffective.

**Neuroscience inspiration**: The human brain relies on the coordinated dual-system of the neocortex (permanent memory / structured general knowledge) and the hippocampus (transient memory / rapid task adaptation) to handle entirely novel environments.

5. Existing RL methods depend on full observability and large-scale annotated data; Bayesian decision-theoretic approaches are also constrained by the full observability assumption.
6. The central goal of this paper is to design an interpretable, theoretically grounded, and rapidly adaptable ATD framework **without any domain-specific prior samples**.

## Method

### Overall Architecture: EM-PTDM

The framework iteratively optimizes the prior model via the Expectation-Maximization (EM) algorithm, with the core guarantee that the prior improves monotonically after each observation:

- **E-step**: Sample from the posterior using the current prior $q_{\phi_k}(x|y)$
- **M-step**: Update the prior $q_{\phi_{k+1}}(x)$ using posterior samples to better approximate the true data distribution

The optimization objective minimizes $\text{KL}(p(y) \| q_\phi(y))$, ensuring monotonic increase in expected log-evidence after each update.

### Three Key Designs

#### 1. Permanent Memory

- A large-scale pretrained diffusion model $s_t^{\theta^*}(x)$ serves as permanent memory
- Encodes general structured knowledge across domains (e.g., pretrained on ImageNet)
- Parameters $\theta^*$ are kept fixed (or fine-tuned with posterior samples across sequential same-domain tasks)

#### 2. Transient Memory: Doob's h-transform

- A lightweight neural network $h_t^\zeta(x, y)$ is introduced to approximate the Doob's h-transform
- The posterior score naturally decomposes into two terms:

$$\nabla_x \log p_t(x | Y=y) \approx \underbrace{s_t^{\theta^*}(x)}_{\text{Permanent Memory}} + \underbrace{h_t^\zeta(x, y)}_{\text{Transient Memory}}$$

- The h-model learns only the residual noise component, acting as a **correction mechanism** for the pretrained score
- Training loss (analogous to denoising score matching):

$$\min_\zeta \mathbb{E}_{(X_0,Y), \varepsilon, t} \| (h_t^\zeta(H_t, Y) + s_t^{\theta^*}(H_t)) - \varepsilon \|^2$$

- Key advantage: only forward evaluation of the pretrained model is required with no backpropagation through $\theta^*$, making it **extremely lightweight** and capable of rapid adaptation from few observations

#### 3. h-model Update Scheduling Strategy

- The h-model is not updated immediately after every observation (sparse early data leads to overfitting or premature convergence)
- A **dynamic frequency scheduler** is adopted: low-frequency updates early on → gradually increasing frequency as observations accumulate
- Experimental validation confirms that delayed updates yield more stable and coherent posterior samples

### Sampling Strategy: Exploration–Exploitation Balance

The sampling score is a weighted combination of an exploration term and an exploitation term:

$$\text{Score}_{(\phi,\eta)}(q_t) = \alpha(\mathcal{B}) \cdot \text{expl}_\phi^{\text{score}}(q_t) + (1-\alpha(\mathcal{B})) \cdot \text{exploit}_{(\phi,\eta)}^{\text{score}}(q_t)$$

- **Exploration**: Based on a maximum-entropy strategy, selecting locations with maximum disagreement among posterior samples (high-uncertainty regions)
- **Exploitation**: Combining an online-trained reward model $r_\eta$ predicting target probability with regions of high posterior sample agreement
- $\alpha(\mathcal{B}) = \frac{\mathcal{B}-t}{\mathcal{B}+t}$: favors exploration when the budget is ample and gradually shifts toward exploitation as the budget decreases

### Loss & Training

- **h-model training**: Residual denoising score matching loss (Equation 7)
- **Reward model $r_\eta$**: Binary cross-entropy loss, updated online using supervised data from each new observation

## Key Experimental Results

### Main Results

**Table 1: Species Discovery — Using Gladicosa+Gonioctena distributions as prior to discover Coccinella Septempunctata**

| Method | B=150 | B=200 | B=250 |
|--------|-------|-------|-------|
| Random Search | 0.162 | 0.233 | 0.278 |
| DiffATD | 0.342 | 0.437 | 0.481 |
| Greedy-Adaptive | 0.406 | 0.507 | 0.557 |
| **EM-PTDM** | **0.498** | **0.650** | **0.699** |

**Table 2: Remote Sensing Target Discovery — Using ImageNet ground-level image prior to discover DOTA aerial targets**

| Method | B=250 | B=300 | B=350 |
|--------|-------|-------|-------|
| Random Search | 0.233 | 0.285 | 0.321 |
| DiffATD | 0.514 | 0.639 | 0.735 |
| Greedy-Adaptive | 0.478 | 0.566 | 0.656 |
| **EM-PTDM** | **0.562** | **0.701** | **0.826** |

**Table 3: Effect of Updating Permanent Memory (DOTA task)**

| Update $s^{\theta^*}$? | B=250 | B=300 | B=350 |
|------------------------|-------|-------|-------|
| No | 0.562 | 0.701 | 0.826 |
| Yes | 0.586 | 0.719 | 0.846 |

### Key Findings

1. EM-PTDM substantially outperforms all baselines across **all budget settings**, with the advantage becoming more pronounced as the budget increases
2. Transient memory ablation: removing the h-model causes the posterior estimate to deviate severely from the true distribution, significantly slowing the decrease in $L_2$ semantic dissimilarity
3. **Key insight**: The critical factor in ATD is not accurate reconstruction of the entire search space, but precise modeling of **target-enriched regions**; EM-PTDM exhibits lower overall reconstruction quality than DiffATD but achieves sharper focus on target regions
4. Updating the permanent memory across sequential same-domain tasks enables cumulative accumulation of domain knowledge, yielding additional performance gains

## Highlights & Insights

1. **ATD without prior data**: This is the first method to achieve efficient active target discovery with no domain-specific prior samples, significantly broadening the applicable scope
2. **Theoretical guarantees**: The EM framework provides rigorous proof of monotonic prior improvement (Theorem 3) and increasingly accurate score estimation (Theorem 4)
3. **Neuroscientific interpretability**: The permanent–transient dual-memory design is not a black-box strategy but an interpretable architecture with clear correspondence to cognitive science
4. **Engineering efficiency**: The h-model learns only residuals and requires only forward evaluation of the main model, making it extremely lightweight and well-suited to data-scarce settings
5. **Dynamic scheduling strategy**: Avoids overfitting to sparse early data, with experimental validation confirming its necessity

## Limitations & Future Work

1. **Computational cost**: Despite the lightweight h-model, each step requires sampling multiple posterior samples plus a diffusion reverse process, resulting in non-trivial overall inference cost
2. **Grid discretization assumption**: The search space is discretized into uniform grids; applicability to continuous or irregular regions is not discussed
3. **Limited domain coverage**: Experiments span only species distribution and remote sensing, lacking empirical validation in medical imaging scenarios such as MRI tumor detection
4. **Choice of $\alpha(\mathcal{B})$**: The exploration–exploitation weight uses a simple linear function, which may be suboptimal; adaptive learning alternatives are not discussed
5. **Permanent memory selection**: Performance depends on the quality of the pretrained diffusion model; the impact of different pretrained models is not analyzed

## Related Work & Insights

- **RL methods** (Uzkent 2020, Sarkar 2023/2024): Require full observability and large-scale annotated data
- **Bayesian decision theory** (Garnett 2012, Jiang 2017/2019): Training-free but assume full observability
- **Partial observability methods** (Rangrej 2022, Pirinen 2022): Handle partial observability but still require extensive annotation
- **DiffATD** (Sarkar 2025): No annotation required, handles partial observability, but depends on domain-specific prior samples
- **EM-PTDM (Ours)**: First to achieve ATD without domain-specific priors, under partial observability, and without annotation

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of neuroscience dual-memory, Doob's h-transform, and the EM framework is creative; uninformative-prior ATD constitutes a novel setting
- Experimental Thoroughness: ⭐⭐⭐⭐ — Two domains, multiple budget settings, and rich ablation studies, but key application domains such as medical imaging are absent
- Writing Quality: ⭐⭐⭐⭐ — Theoretical derivations are clear, motivation is well-articulated, figures are intuitive, and readability is strong
- Value: ⭐⭐⭐⭐ — Significant practical value for high-cost sampling scenarios with extremely scarce data; theoretical contributions are solid

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Online Feedback Efficient Active Target Discovery in Partially Observable Environments](online_feedback_efficient_active_target_discovery_in_partially_observable_enviro.md)
- [\[NeurIPS 2025\] Dynamic Causal Discovery in Alzheimer's Disease through Latent Pseudotime Modelling](dynamic_causal_discovery_in_alzheimers_disease_through_latent_pseudotime_modelli.md)
- [\[ICCV 2025\] AcZeroTS: Active Learning for Zero-shot Tissue Segmentation in Pathology Images](../../ICCV2025/medical_imaging/aczerots_active_learning_for_zeroshot_tissue_segmentation_in.md)
- [\[ICCV 2025\] GDKVM: Echocardiography Video Segmentation via Spatiotemporal Key-Value Memory with Gated Delta Rule](../../ICCV2025/medical_imaging/gdkvm_echocardiography_video_segmentation_via_spatiotemporal_key-value_memory_wi.md)
- [\[CVPR 2026\] Momentum Memory for Knowledge Distillation in Computational Pathology](../../CVPR2026/medical_imaging/momentum_memory_for_knowledge_distillation_in_computational_pathology.md)

</div>

<!-- RELATED:END -->
