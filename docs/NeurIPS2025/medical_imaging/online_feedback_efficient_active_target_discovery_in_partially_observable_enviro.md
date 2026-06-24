---
title: >-
  [Paper Note] Online Feedback Efficient Active Target Discovery in Partially Observable Environments
description: >-
  [NeurIPS 2025][Medical Imaging][Active Target Discovery] This paper proposes DiffATD, which leverages the reverse process of diffusion models to construct a belief distribution for balancing exploration and exploitation, enabling efficient target region discovery in partially observable environments without any supervised training. The framework is applicable across multiple domains including medical imaging, species discovery, and remote sensing.
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "Active Target Discovery"
  - "Diffusion Models"
  - "Partially Observable Environments"
  - "Exploration-Exploitation Trade-off"
  - "Bayesian Experimental Design"
date: 2026-05-08
content_hash: f0b73c53fb5789da
---

# Online Feedback Efficient Active Target Discovery in Partially Observable Environments

**Conference**: NeurIPS 2025  
**arXiv**: [2505.06535](https://arxiv.org/abs/2505.06535)  
**Code**: [GitHub](https://github.com/AnindyaSarkar/DiffATD)  
**Area**: Medical Imaging  
**Keywords**: Active Target Discovery, Diffusion Models, Partially Observable Environments, Exploration-Exploitation Trade-off, Bayesian Experimental Design

## TL;DR

This paper proposes DiffATD, which leverages the reverse process of diffusion models to construct a belief distribution for balancing exploration and exploitation, enabling efficient target region discovery in partially observable environments without any supervised training. The framework is applicable across multiple domains including medical imaging, species discovery, and remote sensing.

## Background & Motivation

In many scientific and engineering domains (e.g., MRI scanning, search and rescue, drug discovery), data acquisition is extremely costly, requiring strategic sampling from unobserved regions under limited budgets to maximize target discovery. The core challenges are:

**Partial Observability**: During search, the agent can only observe a small portion of the search space and must infer the content of unknown regions from limited observations.

**Exploration-Exploitation Dilemma**: The agent must simultaneously pursue two conflicting objectives—exploration (acquiring information about the search space to reduce uncertainty) and exploitation (focusing on regions likely to contain targets).

**Scarcity of Annotated Data**: Existing methods largely rely on large-scale pre-annotated datasets to train RL policies, which is highly impractical in scenarios such as rare diseases (e.g., uncommon tumors).

**Limitations of Prior Work**: RL-based methods (e.g., GOMAA-Geo, Visual Active Search) rely on full observability and large amounts of pre-annotated data; Bayesian decision-theoretic methods require no training but also assume full observability; generative methods, while interpretable, optimize only for reconstruction rather than target discovery. DiffATD's core contribution lies in being the first to achieve **training-free** target discovery in partially observable environments with interpretability.

## Method

### Overall Architecture

DiffATD formulates Active Target Discovery (ATD) as a sequential decision-making problem over a gridded search space. The search region is partitioned into $N$ grid cells, and the agent iteratively selects measurement locations within a budget $\mathcal{B}$, where each measurement reveals the target occupancy ratio $y^{(i)} \in [0,1]$ of the selected cell. DiffATD maintains a particle set via the reverse diffusion process to construct a belief distribution over unobserved space, and determines the next sampling location by combining exploration scores and exploitation scores.

### Key Designs

1. **Belief Distribution via Diffusion Dynamics**: DiffATD maintains a batch of particles $\{x_\tau^{(i)}\}_{i=0}^{N_B}$ during the reverse diffusion process, employing the Tweedie formula for single-step denoising estimation: $\hat{x}_0 = \mathcal{T}_\tau(x_\tau) = \frac{1}{\sqrt{\bar{\alpha}_\tau}}(x_\tau + (1-\bar{\alpha}_\tau)s_\theta(x_\tau, \tau))$. These particles implicitly form a belief distribution over the complete search space, modeled as a mixture of $N_B$ isotropic Gaussians: $p(\hat{x}_t|Q_t, \tilde{x}_{t-1}) = \sum_{i=0}^{N_B} \alpha_i \mathcal{N}(\hat{x}_t^i, \sigma_x^2 I)$. The design motivation is to leverage the generative prior of a pretrained diffusion model to reason about unobserved regions without any additional supervision.

2. **Maximum Entropy Exploration Strategy**: The exploration score is computed by measuring the disagreement among particles at candidate measurement locations. The authors prove (Theorem 1) that the optimal exploration location is equivalent to the region of maximum inter-particle prediction discrepancy: $\mathrm{expl}^{\mathrm{score}}(q_t) = \sum_{i,j} \frac{([\hat{x}_t^{(i)}]_{q_t} - [\hat{x}_t^{(j)}]_{q_t})^2}{2\sigma_x^2}$. This avoids the prohibitive cost of computing belief distributions separately for each candidate location. The design motivation derives from the mutual information maximization principle in Bayesian experimental design.

3. **Reward Model-Based Exploitation Strategy**: The exploitation score integrates two signals—(1) expected log-likelihood (quantifying inter-particle consistency) and (2) the probability of target presence predicted by an online-trained reward model $r_\phi$: $\mathrm{exploit}^{\mathrm{score}}(q_t) = \mathrm{likeli}^{\mathrm{score}}(q_t) \times \sum_{i}^{N_B} r_\phi([\hat{x}_t^{(i)}]_{q_t})$. The reward model is randomly initialized and incrementally updated after each measurement via binary cross-entropy loss. The design motivation is to decouple "prediction consistency of a region" from "likelihood of target presence," enabling the two signals to complement each other.

### Loss & Training

- **Measurement Guidance**: During reverse diffusion, observed locations are constrained via gradient descent: $x_{\tau-1}^{(i)} \leftarrow x_{\tau-1}^{(i)'} - \zeta \nabla_{x_\tau^{(i)}} \|[x]_{Q_t} - [\hat{x}_\tau^{(i)}]_{Q_t}\|^2$
- **Budget-Aware Balancing**: The final sampling score is a weighted combination of exploration and exploitation: $\mathrm{Score}(q_t) = \kappa(\mathcal{B}) \cdot \mathrm{expl}^{\mathrm{score}} + (1-\kappa(\mathcal{B})) \cdot \mathrm{exploit}^{\mathrm{score}}$, where $\kappa(\mathcal{B}) = \frac{\mathcal{B}-t}{\mathcal{B}+t}$ favors exploration early in the search and exploitation later.
- **Reward Model Training**: The reward model is incrementally trained on the growing measurement dataset $\mathcal{D}_t$ using BCE loss.

## Key Experimental Results

### Main Results

Experiments span multiple domains including remote sensing (DOTA), species discovery (iNaturalist), skin lesion discovery, and chest X-ray bone suppression.

| Dataset/Target | Budget $\mathcal{B}$ | DiffATD SR | Best Baseline SR | Relative Gain |
|---|---|---|---|---|
| DOTA (plane/truck) | 200 | 0.5422 | 0.4625 (Max-Ent) | +17.23% |
| DOTA (plane/truck) | 300 | 0.7309 | 0.6550 (GA) | +11.59% |
| iNaturalist (species) | 200 | 0.6401 | 0.5826 (GA) | +9.87% |
| Skin (malignant) | 200 | 0.8974 | 0.8261 (GA) | +8.63% |
| Chest X-ray (bone suppression) | 300 | 0.4142 | 0.2936 (RS) | +41.08% |

### Ablation Study

| Configuration ($\alpha$ controls exploration/exploitation weight) | DOTA SR ($\mathcal{B}=200$) | Skin SR ($\mathcal{B}=200$) | Notes |
|---|---|---|---|
| $\alpha=0.2$ (exploitation-heavy) | 0.5052 | 0.8465 | Excessive exploitation degrades performance |
| $\alpha=1.0$ (balanced) | **0.5422** | **0.8974** | Optimal balance |
| $\alpha=5.0$ (exploration-heavy) | 0.4823 | 0.8782 | Excessive exploration also harmful |

### Key Findings

- DiffATD achieves an SR of 0.8974 on rare targets such as skin lesions, surpassing even the fully observable supervised method SAM (FullSEG, 0.6221).
- Compared to VLMs such as GPT-4o and Gemini, DiffATD achieves an SR of 0.7309 on DOTA at $\mathcal{B}=300$, significantly outperforming Gemini (0.6453) and GPT-4o (0.5678).
- The linear $\kappa(\mathcal{B})$ schedule for exploration-exploitation balancing performs well across all domains, validating the effectiveness of a simple scheduling strategy.

## Highlights & Insights

- **Unsupervised Paradigm**: DiffATD relies entirely on unsupervised pretrained diffusion models, requiring no task-specific annotated data, substantially improving practical applicability.
- **Theoretical Grounding**: The exploration strategy, derived from Bayesian experimental design and mutual information maximization, is supported by rigorous theoretical foundations (Theorems 1 and 2).
- **Interpretability**: In contrast to black-box RL policies, each decision step in DiffATD can be explained through exploration score maps and exploitation score maps.
- **Cross-Domain Generalization**: The same framework achieves significant improvements across diverse domains including remote sensing, ecology, and medical imaging.

## Limitations & Future Work

- Performance depends on the quality of the pretrained diffusion model and may degrade under limited training data or distribution shift.
- Single-step Tweedie estimation has limited accuracy at high noise levels; multi-step estimation could be explored to improve belief distribution accuracy.
- The reward model is prone to overfitting when early-stage data is scarce; while $\kappa(\mathcal{B})$ mitigates this issue, it does not fully resolve it, and fast adaptation strategies such as meta-learning warrant exploration.
- Validation is currently limited to 2D grid spaces; extension to 3D volumetric data (e.g., MRI voxels) requires additional consideration of computational efficiency.
- The selection of particle count $N_B$ and measurement schedule $M$ relies on empirical tuning without theoretical guidance.
- The search space is assumed to be uniformly gridded, whereas in practice targets may span multiple cells or occupy only a fraction of a cell.

## Related Work & Insights

- Connection to active MRI acceleration (van Gorp et al., 2021): DiffATD can be viewed as extending active sampling from "optimizing reconstruction" to "optimizing target discovery."
- Connection to Bayesian optimization: The exploration-exploitation framework parallels GP-UCB, but DiffATD replaces Gaussian processes with diffusion models for greater expressiveness.
- Relationship to GOMAA-Geo (Sarkar et al., 2024): The latter requires large amounts of pre-annotated data to train RL policies, whereas DiffATD is entirely unsupervised.
- The work motivates the use of diffusion models for other sequential decision-making problems, such as active sensing and adaptive experimental design.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First integration of diffusion dynamics with active target discovery; elegantly designed exploration-exploitation framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers four domains with multiple target types and complete ablations; comparisons with additional active learning methods are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical derivations are clear, though the dense notation presents a moderate barrier on first reading.
- **Value**: ⭐⭐⭐⭐⭐ Unsupervised active discovery has significant application potential in medical imaging, particularly for rare disease screening.
- **Overall**: ⭐⭐⭐⭐ Solid work combining theory and experiments; cross-domain generalization is impressive.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Active Target Discovery under Uninformative Prior: The Power of Permanent and Transient Memory](active_target_discovery_under_uninformative_prior_the_power_of_permanent_and_tra.md)
- [\[NeurIPS 2025\] Doctor Approved: Generating Medically Accurate Skin Disease Images through AI-Expert Feedback](doctor_approved_generating_medically_accurate_skin_disease_images_through_ai-exp.md)
- [\[NeurIPS 2025\] Self-supervised Learning of Echocardiographic Video Representations via Online Cluster Distillation](self-supervised_learning_of_echocardiographic_video_representations_via_online_c.md)
- [\[NeurIPS 2025\] Domain-Adaptive Transformer for Data-Efficient Glioma Segmentation in Sub-Saharan MRI](domain-adaptive_transformer_for_data-efficient_glioma_segmentation_in_sub-sahara.md)
- [\[NeurIPS 2025\] Dynamic Causal Discovery in Alzheimer's Disease through Latent Pseudotime Modelling](dynamic_causal_discovery_in_alzheimers_disease_through_latent_pseudotime_modelli.md)

</div>

<!-- RELATED:END -->
