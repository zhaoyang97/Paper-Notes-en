---
title: >-
  [Paper Note] Heuristic Self-Paced Learning for Domain Adaptive Semantic Segmentation under Adverse Conditions
description: >-
  [CVPR 2026][Segmentation][Unsupervised Domain Adaptation] This paper reformulates class-level curriculum learning in unsupervised domain adaptation as a sequential decision-making problem under the reinforcement learning framework. The proposed HeuSCM framework achieves autonomous curriculum scheduling via high-dimensional semantic state perception and category-fair policy gradients, attaining state-of-the-art performance (72.9 mIoU) on ACDC, Dark Zurich, and Nighttime Driving.
tags:
  - CVPR 2026
  - Segmentation
  - Unsupervised Domain Adaptation
  - Semantic Segmentation
  - Curriculum Learning
  - Reinforcement Learning
  - Adverse Weather
date: 2026-05-08
content_hash: 2dc74f33eda58f22
---

# Heuristic Self-Paced Learning for Domain Adaptive Semantic Segmentation under Adverse Conditions

**Conference**: CVPR 2026
**arXiv**: [2603.24322](https://arxiv.org/abs/2603.24322)
**Code**: N/A
**Area**: Semantic Segmentation / Domain Adaptation
**Keywords**: Unsupervised Domain Adaptation, Semantic Segmentation, Curriculum Learning, Reinforcement Learning, Adverse Weather

## TL;DR

This paper reformulates class-level curriculum learning in unsupervised domain adaptation as a sequential decision-making problem under the reinforcement learning framework. The proposed HeuSCM framework achieves autonomous curriculum scheduling via high-dimensional semantic state perception and category-fair policy gradients, attaining state-of-the-art performance (72.9 mIoU) on ACDC, Dark Zurich, and Nighttime Driving.

## Background & Motivation

**Background**: Unsupervised domain adaptive semantic segmentation (UDA-SS) is a core technique for autonomous driving perception, transferring models trained under clear weather to adverse conditions (fog, rain, nighttime). Mainstream approaches apply style transfer to mitigate domain gaps, while curriculum learning (CL) or hard category mining (HCM) is employed to address class imbalance.

**Limitations of Prior Work**: Existing CL and HCM methods suffer from fundamental paradigm flaws: (1) difficulty assessment relies on **fixed, hand-crafted metrics** (e.g., prediction uncertainty, confidence), using a one-dimensional scalar to characterize the high-dimensional, dynamic cognitive state of the model; (2) learning schedules are driven by **manually designed rules** (e.g., "easy-to-hard" or "focus entirely on hard classes"), which cannot adapt to the model's continuously evolving internal state. This "prescriptive paradigm" leads to category bias in compound problems such as adverse weather—some categories are over-attended while others are insufficiently learned.

**Key Challenge**: The model training process is a high-dimensional, dynamic, and non-monotonic evolutionary process. Attempting to statically schedule this learning trajectory using fixed, one-dimensional, manually defined scalars is fundamentally unsound. The CL "easy-to-hard" strategy overlooks categories that the model currently needs most, while the HCM "all hard classes" strategy lacks inter-class balance.

**Goal**: Shift from "designing a curriculum" to "learning a curriculum"—enabling an agent to autonomously discover the optimal learning trajectory based on the model's current state, rather than relying on human prior assumptions.

**Key Insight**: Inspired by reinforcement learning, curriculum learning is modeled as a sequential decision-making problem—at each training step, an agent observes the model state and autonomously determines which categories are most informative, then performs cross-domain mixed sampling centered on those categories.

**Core Idea**: An autonomous curriculum scheduler is proposed, comprising a high-dimensional state encoder for perceiving the model's learning state and a category-fair policy gradient for ensuring balanced improvement, realizing truly adaptive class-level curriculum learning.

## Method

### Overall Architecture

HeuSCM operates under a reinforcement learning paradigm: (1) a high-dimensional semantic state $\mathbf{s}$ is extracted from the segmentation model; (2) a GM-VAE encoder compresses the state into a low-dimensional latent space to obtain $z_t^s$; (3) SKFEN distills key features from $z_t^s$ to reflect learning progress; (4) ClassGen (the policy network) outputs a ranked category list (in descending order of informativeness) based on the key features; (5) the ranking guides cross-domain mixed sampling to generate mixed images and labels; (6) the segmentation model is updated on mixed data via SegLoss; (7) category-targeted reward signals drive policy optimization. The entire system iterates continuously throughout training.

### Key Designs

1. **High-dimensional Semantic State Extraction (HSSE)**:

    - **Function**: Comprehensively characterizes the learning state of the segmentation model during domain adaptation.
    - **Mechanism**: Consists of two stages—(a) *Low-dimensional State Representation Learning*: a Gaussian Mixture VAE (GM-VAE) encodes the high-dimensional state $\mathbf{s}$ into a compact latent space, modeling multimodal learning states. The discrete component $q_\psi(c|\mathbf{s})$ captures learning modes, while the continuous latent variable $q_\psi(\mathbf{z}|\mathbf{s},c)$ encodes concrete states. The training objective maximizes the variational lower bound: $\mathcal{L}_{GM-VAE} = \mathbb{E}_{q_\psi(c|\mathbf{s})}[\mathbb{E}_{q_\psi(\mathbf{z}|\mathbf{s},c)}[\log p_\theta(\mathbf{s}|\mathbf{z})] - \text{KL}(q_\psi(\mathbf{z}|\mathbf{s},c) \| p(\mathbf{z}|c))] - \text{KL}(q_\psi(c|\mathbf{s}) \| p(c))$. (b) *SKFEN*: distills key features from the latent space via grouped feature aggregation to reduce redundant information.
    - **Design Motivation**: Conventional methods represent learning state with a single scalar (e.g., prediction uncertainty), which fails to capture inter-class coupling relationships, feature space dynamics, and other high-dimensional information. GM-VAE captures multimodal structure, while SKFEN further refines the signals truly relevant to learning progress.

2. **Semantic Key Feature Extraction Network (SKFEN)**:

    - **Function**: Distills key features from the latent space and reduces feature redundancy.
    - **Mechanism**: Comprises an initial transformation stage and a grouped processing stage. Features are first fused and interaction-modeled via 1×1–5×5 depthwise separable–1×1 convolutions; after channel expansion, features are split into $G$ groups. Each group extracts max-pooling and average-pooling features separately, which are concatenated and fused via a 3×3 convolution, with a residual connection added to yield the refined feature $z_{out}$.
    - **Design Motivation**: Although the latent space is already dimensionality-reduced, considerable redundancy remains that is uninformative for policy decisions. SKFEN captures the key dimensions that truly reflect the learning state through grouped aggregation from multiple perspectives (peak and mean statistics).

3. **Categorical $\alpha$-Fairness for Policy Gradients (C$\alpha$PG)**:

    - **Function**: Ensures fair reward allocation across all semantic categories during policy optimization.
    - **Mechanism**: The value function for each category is defined as $V_c^\pi(s) = \mathbb{E}_\pi[\sum_{k=0}^{\infty}\gamma^k r_c(t+k)|S_t=s]$, where the reward $r_c(t)$ jointly considers transferability (cosine similarity between source and target domain features) and discriminability (separation between different classes in the target domain). The key innovation is optimizing the $\alpha$-fairness objective rather than the standard sum objective $J_{sum}$: $J_F(\pi) = \sum_{c=1}^{C} \frac{1}{1-\alpha}(V_c^\pi(s))^{1-\alpha}$. The corresponding policy gradient employs a fairness-weighted advantage function $\tilde{A}_\alpha$, with weights $w_c(s_t) = V_c^\pi(s_t)^{-\alpha}$ inversely proportional to the current value of each category—lower-value categories receive higher weights.
    - **Design Motivation**: Standard RL optimizing total reward causes the policy to favor already well-performing categories (a "Matthew effect"), exacerbating class imbalance. The $\alpha$-fairness mechanism forces the policy to attend to underperforming categories, achieving balanced improvement.

### Loss & Training

- Segmentation loss: $\mathcal{L}_{seg} = \lambda_1 \mathcal{L}_{CE}(g_\theta(x_s), y_s) + \lambda_2 \mathcal{L}_{CE}(g_\theta(\mathcal{X}_{mix}^T), \mathcal{Y}_{mix}^R)$, where $\lambda_1 = \lambda_2 = 1.0$
- GM-VAE reconstruction loss: $\mathcal{L}_{recon} = \mathbb{E}_{\mathbf{s}_t}[\|\mathbf{s}_t - p_\theta(\text{Enc}_\psi(\mathbf{s}_t))\|^2]$, used during fine-tuning to preserve latent space structure
- Policy optimization: maximize the fairness objective $J_F(\pi)$ using the AdamW optimizer
- Trained for 60k iterations on Cityscapes → ACDC with 1024×1024 crops on NVIDIA A800 GPUs

## Key Experimental Results

### Main Results (Cityscapes → ACDC test)

| Method | Backbone | mIoU |
|--------|----------|------|
| DeepLab-v2 (source only) | DeepLab-v2 | 38.0 |
| Refign | DeepLab-v2 | 48.0 |
| VBLC | DeepLab-v2 | 47.8 |
| **HeuSCM (Ours)** | DeepLab-v2 | **58.7** |
| HRDA (source only) | HRDA | 68.0 |
| CoDA | HRDA | 72.6 |
| ACSegFormer | HRDA | 72.7 |
| **HeuSCM (Ours)** | HRDA | **72.9** |

### Ablation Study (ACDC val, HRDA backbone)

| Configuration | mIoU | Gain | Note |
|---------------|------|------|------|
| Baseline (Refign) | 71.1 | +0.0 | Without HSCM |
| + LSRL only | 72.2 | +1.1 | Low-dim state representation contributes most |
| + SKFEN only | 71.7 | +0.6 | Key feature distillation is effective |
| + C$\alpha$PG only | 71.6 | +0.5 | Fair policy gradient is effective |
| + LSRL + SKFEN | 72.3 | +1.2 | State perception synergy is strong |
| + LSRL + C$\alpha$PG | 72.0 | +0.9 | State + fairness combination is effective |
| + All three (full model) | **72.7** | **+1.6** | Three modules are complementary |

### Key Findings

- The most significant improvement occurs with the DeepLab-v2 backbone (+10.7 mIoU), indicating that HeuSCM yields larger gains for weaker backbones.
- Even with the strong HRDA backbone, HeuSCM achieves state-of-the-art 72.9 mIoU, surpassing CoDA (72.6) and ACSegFormer (72.7).
- Results of 52.8 mIoU on Dark Zurich and 59.3 mIoU on Nighttime Driving demonstrate generalization across diverse nighttime scenarios.
- The Heuristic Curriculum Sampling Policy (HCSP) can serve as a plug-and-play module replacing the sampling strategy of existing hard category mining methods, and is also effective on GTA5→Cityscapes.

## Highlights & Insights

- **The paradigm-shift insight is particularly sharp**: The conceptual shift from "designing a curriculum" to "learning a curriculum" reveals a fundamental limitation of existing CL/HCM methods—humans cannot effectively design optimal trajectories for high-dimensional dynamic processes.
- **The $\alpha$-fairness policy gradient is elegantly designed**: Transplanting the concept of multi-agent fairness into a single-agent multi-class setting, the inverse-weighting mechanism naturally resolves category bias.
- **GM-VAE modeling of learning states**: The approach of using a Gaussian mixture to capture multimodal learning states is worth borrowing—models adapting to different weather conditions may indeed reside in distinct "learning modes."

## Limitations & Future Work

- The three-stage training procedure (GM-VAE pre-training → joint fine-tuning → segmentation training) increases implementation complexity and training overhead.
- The specific architectural design of SKFEN (number of groups $G$, expansion dimension $n$, etc.) may require adjustment for different scenarios.
- The current reward design assumes that source and target domain features can be compared in a shared space, which may not be sufficiently robust under extreme domain gaps.
- Future work could extend the framework to other domain adaptation tasks (e.g., object detection, instance segmentation) to validate its generality.

## Related Work & Insights

- **vs. CoDA**: CoDA also performs easy-to-hard domain adaptation, but uses a manually designed curriculum (simpler domains first, then harder ones), whereas HeuSCM fully learns the curriculum order autonomously.
- **vs. ACSegFormer**: Achieves comparable performance via an orthogonal approach—ACSegFormer improves the segmentation architecture, while HeuSCM improves the training strategy; in principle, the two can be combined.
- **vs. CoPT**: CoPT uses prompt tuning for conditional adaptation, while HeuSCM uses RL for curriculum learning; the two methods address different levels of the adaptation mechanism.
- The idea of introducing the RL paradigm into training strategy optimization (rather than the model itself) is transferable to related problems such as active learning and data selection.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The paradigm shift to "learning a curriculum," GM-VAE state encoding, and $\alpha$-fair policy gradients are all novel contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three backbones, three benchmarks, detailed ablations, and generalization validation—very comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is articulated clearly; the method section is formula-dense and requires careful reading.
- **Value**: ⭐⭐⭐⭐ Proposes an elegant new paradigm for curriculum learning in domain adaptation, though the concrete implementation is relatively complex.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Masked Representation Modeling for Domain-Adaptive Segmentation](mrm_masked_representation_modeling_domain_adaptive.md)
- [\[CVPR 2026\] GeomPrompt: Geometric Prompt Learning for RGB-D Semantic Segmentation Under Missing and Degraded Depth](geomprompt_rgbd_segmentation.md)
- [\[CVPR 2026\] Seeing Beyond: Extrapolative Domain Adaptive Panoramic Segmentation](seeing_beyond_extrapolative_domain_adaptive_panoramic_segmentation.md)
- [\[CVPR 2026\] Low-Data Supervised Adaptation Outperforms Prompting for Cloud Segmentation Under Domain Shift](low_data_supervised_adaptation_outperforms_prompting_for_cloud_segmentation.md)
- [\[CVPR 2026\] CrossEarth-SAR: A SAR-Centric and Billion-Scale Geospatial Foundation Model for Domain Generalizable Semantic Segmentation](crossearthsar_a_sarcentric_and_billionscale_geospa.md)

<!-- RELATED:END -->
