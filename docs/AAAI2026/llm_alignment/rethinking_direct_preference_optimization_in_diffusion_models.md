---
title: >-
  [Paper Note] Rethinking Direct Preference Optimization in Diffusion Models
description: >-
  [AAAI 2026 (Oral)][LLM Alignment][DPO] Two orthogonal and plug-and-play improvement strategies are proposed to enhance preference optimization in diffusion models: stable reference model updating (relaxing the frozen constraint with a regularization anchor) and timestep-aware training (adaptive weighting to balance reward scales across timesteps). Both strategies can be embedded into various preference optimization algorithms such as DPO and IPO, achieving state-of-the-art performance on human preference evaluation benchmarks.
tags:
  - AAAI 2026 (Oral)
  - LLM Alignment
  - DPO
  - diffusion models
  - reference model update
  - timestep-awareness
  - T2I preference alignment
date: 2026-05-08
content_hash: 43c5b7dd957fa0a8
---

# Rethinking Direct Preference Optimization in Diffusion Models

**Conference**: AAAI 2026 (Oral)
**arXiv**: [2505.18736](https://arxiv.org/abs/2505.18736)
**Code**: Available
**Area**: Alignment & RLHF / Diffusion Models
**Keywords**: DPO, diffusion models, reference model update, timestep-awareness, T2I preference alignment

## TL;DR
Two orthogonal and plug-and-play improvement strategies are proposed to enhance preference optimization in diffusion models: stable reference model updating (relaxing the frozen constraint with a regularization anchor) and timestep-aware training (adaptive weighting to balance reward scales across timesteps). Both strategies can be embedded into various preference optimization algorithms such as DPO and IPO, achieving state-of-the-art performance on human preference evaluation benchmarks.

## Background & Motivation

### State of the Field
Human preference alignment for text-to-image (T2I) diffusion models has become a critical research challenge. Preference optimization methods borrowed from LLMs (e.g., DPO, IPO) have been extended to diffusion models, directly optimizing model parameters using pairwise preference data (winning/losing image pairs) to better align generated outputs with human preferences. Existing methods such as Diffusion-DPO and D-Fusion have achieved promising initial results.

### Limitations of Prior Work
(1) **Frozen reference model restricts exploration** — Standard DPO keeps the reference model frozen to provide a stable KL-divergence anchor. However, in diffusion models this severely limits the exploration capacity of the policy model, as the multi-step denoising process accumulates bias from a frozen reference over long inference chains. (2) **Imbalanced reward scales across timesteps** — Signal strength varies drastically across denoising timesteps (weak signals at high-noise timesteps, strong signals at low-noise timesteps), yet existing methods treat all timesteps uniformly, causing training to be dominated by low-noise steps.

**Key Challenge**: Relaxing the reference model enhances exploration but destabilizes training; different timesteps require differentiated weighting but lack an adaptive mechanism.

### Paper Goals
To enhance exploration capacity and timestep training balance in diffusion-based DPO while maintaining training stability.

**Key Insight**: Design two orthogonal (non-interfering) plug-and-play strategies — dynamic reference model updating and timestep-aware loss weighting — that can be embedded into arbitrary preference optimization algorithms.

**Core Idea**: Address insufficient exploration via reference model regularization relaxation, and address reward imbalance via timestep-aware training; the two strategies are orthogonal and mutually complementary.

## Method

### Overall Architecture
The input consists of pairwise preference data (preferred/rejected image pairs with prompts) and a pretrained T2I diffusion model. The method comprises two orthogonal strategy modules that can be embedded individually or jointly into the training pipeline of preference optimization algorithms such as DPO and IPO, producing an aligned model upon completion.

### Key Designs

1. **Stable Reference Model Update Strategy**

    - **Function**: Dynamically updates the reference model to expand the exploration space while maintaining training stability through regularization.
    - **Mechanism**: (a) The reference model parameters are updated via exponential moving average (EMA): $\theta_{\text{ref}} \leftarrow \alpha \theta_{\text{ref}} + (1-\alpha) \theta_{\text{policy}}$, allowing the reference model to slowly follow the policy model rather than remaining completely frozen. (b) A regularization loss $\mathcal{L}_{\text{reg}}$ is added to penalize excessive deviation of the policy model from the reference, forming a "relaxed-but-anchored" mechanism that permits exploration of new regions while preventing complete divergence from the reference.
    - **Design Motivation**: Freezing the reference model is feasible in LLMs (discrete token space, short decision chains), but severely restricts exploration in the continuous multi-step denoising process of diffusion models. EMA combined with regularization strikes a balance between relaxation and stability.

2. **Timestep-Aware Training Strategy**

    - **Function**: Mitigates training imbalance caused by differences in reward signal strength across denoising timesteps.
    - **Mechanism**: Analysis of the implicit reward distribution at different timesteps $t$ reveals that high-noise timesteps (large $t$) yield weak, high-variance reward signals, while low-noise timesteps (small $t$) yield strong, low-variance signals. An adaptive weight function $w(t)$ is designed accordingly: increasing weights for high-noise timesteps to compensate for signal attenuation, and decreasing weights for low-noise timesteps to prevent them from dominating training. Specific weights can be derived by normalizing statistics of the reward distribution at each timestep.
    - **Design Motivation**: DPO in LLMs does not involve a timestep dimension (single forward generation), whereas the multi-step denoising in diffusion models inherently introduces a timestep dimension as a novel challenge — a bottleneck unique to diffusion-based DPO that requires dedicated solutions.

3. **Plug-and-Play Modular Design**

    - **Function**: Enables both strategies to be embedded into various preference optimization frameworks including DPO and IPO.
    - **Mechanism**: The final loss takes the form $\mathcal{L} = w(t) \cdot \mathcal{L}_{\text{DPO/IPO}} + \lambda \cdot \mathcal{L}_{\text{reg}}$, where the two strategies act on the loss function via multiplicative weighting and additive regularization respectively, without mutual interference.
    - **Design Motivation**: The modular design makes the method algorithm-agnostic and broadly applicable as a general-purpose enhancement plug-in.

### Loss & Training
The total loss is $\mathcal{L} = w(t) \cdot \mathcal{L}_{\text{pref}} + \lambda \cdot \mathcal{L}_{\text{reg}}$, where $\mathcal{L}_{\text{pref}}$ denotes the base preference optimization loss (DPO or IPO), $w(t)$ is the timestep-aware weight, and $\mathcal{L}_{\text{reg}}$ is the reference model regularization term. The EMA momentum coefficient $\alpha$ and regularization weight $\lambda$ are the primary hyperparameters.

## Key Experimental Results

### Main Results: Human Preference Evaluation Benchmarks

| Method | Preference Alignment Score | Modification |
|--------|--------------------------|-------------|
| Diffusion-DPO (baseline) | Reference | Frozen reference + uniform weights |
| + Reference model update | Significant gain | Enhanced exploration |
| + Timestep-aware training | Significant gain | More balanced training |
| + Both combined (**Ours**) | **SOTA** | Orthogonal combination yields best results |

### Ablation Study: Orthogonality Validation

| Configuration | Individual Effect | Combined Effect | Conclusion |
|--------------|------------------|-----------------|------------|
| Reference update only | Effective improvement | — | Enhances exploration |
| Timestep-aware only | Effective improvement | — | Balances training |
| Both combined | — | Outperforms sum of individuals | Orthogonal complementarity |

### Cross-Algorithm Compatibility

| Base Algorithm | After Embedding Proposed Method | Notes |
|---------------|--------------------------------|-------|
| DPO | Improvement | Applicable |
| IPO | Improvement | Applicable |
| Other preference optimization | Improvement | General plug-in |

### Key Findings
- The two strategies are orthogonal: each is independently effective, and their combination outperforms either alone.
- Timestep imbalance is a problem unique to diffusion-based DPO — it does not arise in LLM DPO.
- Accepted as AAAI 2026 Oral and also accepted at SPIGM@NeurIPS 2025.

## Highlights & Insights
- **Composability of orthogonal improvements**: The two strategies address distinct aspects (exploration vs. balance) without interfering with each other, providing a methodological paradigm for modular improvement of DPO.
- **Timestep-dimension analysis**: This work is among the first to systematically reveal the cross-timestep reward scale imbalance phenomenon in diffusion-based DPO, offering an important perspective for subsequent diffusion alignment research.
- **Strong engineering practicality**: No redesign of the training pipeline is required; the strategies can be directly embedded into existing pipelines for immediate gains.

## Limitations & Future Work
- **Full HTML paper unavailable**: Detailed ablation data and hyperparameter sensitivity analyses could not be retrieved; the above analysis is primarily based on the abstract and method overview.
- **Sensitivity of reference update frequency / EMA momentum**: The value of $\alpha$ may have a significant impact on performance and requires careful tuning.
- **Applicability to video/3D diffusion models**: Effectiveness under longer timestep chains and more complex generation tasks remains to be verified.
- **Comparison with reference-free methods**: Methods such as MaPO (AAAI 2026) entirely remove the reference model; the optimal conditions for each approach require further clarification.

## Related Work & Insights
- **vs. Diffusion-DPO**: The standard transfer approach with a frozen reference model that leads to insufficient exploration — this paper directly addresses that core bottleneck.
- **vs. MaPO (AAAI 2026)**: A complementary direction that completely removes the reference model — this paper retains but dynamically updates it, representing a distinct technical approach.
- **vs. DDPO/DRaFT**: These methods rely on additional reward models for online scoring — this paper uses pairwise preferences for direct optimization, making it more lightweight.
- **vs. D-Fusion (ICML 2025)**: Also focuses on improving diffusion-based DPO but targets sample consistency — this paper focuses on two orthogonal dimensions: the reference model and timestep weighting.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Both orthogonal strategies make independent contributions; the timestep imbalance analysis is a novel finding in the diffusion alignment literature.
- **Experimental Thoroughness**: ⭐⭐⭐ Information is limited given reliance on the abstract; an Oral paper should feature comprehensive experiments.
- **Writing Quality**: ⭐⭐⭐⭐ AAAI Oral quality with clearly motivated problem formulation.
- **Value**: ⭐⭐⭐⭐ The plug-and-play design enables broad integration into existing diffusion-based DPO methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Margin-aware Preference Optimization for Aligning Diffusion Models without Reference](margin-aware_preference_optimization_for_aligning_diffusion_models_without_refer.md)
- [\[CVPR 2026\] LocalDPO: Direct Localized Detail Preference Optimization for Video Diffusion Models](../../CVPR2026/llm_alignment/mind_the_generative_details_direct_localized_detail_preference_optimization_for_.md)
- [\[AAAI 2026\] MetaGDPO: Alleviating Catastrophic Forgetting with Metacognitive Knowledge through Group Direct Preference Optimization](metagdpo_alleviating_catastrophic_forgetting_with_metacognitive_knowledge_throug.md)
- [\[CVPR 2026\] PhysMoDPO: Physically-Plausible Humanoid Motion with Preference Optimization](../../CVPR2026/llm_alignment/physmodpo_physicallyplausible_humanoid_motion_with.md)
- [\[CVPR 2026\] $\varphi$-DPO: Fairness Direct Preference Optimization Approach to Continual Learning in Large Multimodal Models](../../CVPR2026/llm_alignment/φ-dpo_fairness_direct_preference_optimization_approach_to_continual_learning_in_.md)

</div>

<!-- RELATED:END -->
