---
title: >-
  [Paper Note] Real-World Adverse Weather Image Restoration via Dual-Level Reinforcement Learning with High-Quality Cold Start
description: >-
  [NeurIPS 2025][Image Restoration][adverse weather image restoration] This paper proposes a Dual-level Reinforcement Learning (DRL) framework that combines a physics-driven million-scale synthetic weather dataset, HFLS-Weather, for high-quality cold-start training, and achieves adaptive real-world adverse weather image restoration through Perturbation-driven Image Quality Optimization (PIQO) at the local level and global meta-controller multi-agent collaboration.
tags:
  - "NeurIPS 2025"
  - "Image Restoration"
  - "adverse weather image restoration"
  - "reinforcement learning"
  - "GRPO"
  - "multi-agent system"
  - "no-reference image quality assessment"
date: 2026-05-08
content_hash: 9091f55b202aecf2
---

# Real-World Adverse Weather Image Restoration via Dual-Level Reinforcement Learning with High-Quality Cold Start

**Conference**: NeurIPS 2025
**arXiv**: [2511.05095](https://arxiv.org/abs/2511.05095)  
**Code**: [Available](https://github.com/xxclfy/AgentRL-Real-Weather)  
**Area**: Image Restoration / Adverse Weather
**Keywords**: adverse weather image restoration, reinforcement learning, GRPO, multi-agent system, no-reference image quality assessment

## TL;DR
This paper proposes a Dual-level Reinforcement Learning (DRL) framework that combines a physics-driven million-scale synthetic weather dataset, HFLS-Weather, for high-quality cold-start training, and achieves adaptive real-world adverse weather image restoration through Perturbation-driven Image Quality Optimization (PIQO) at the local level and global meta-controller multi-agent collaboration.

## Background & Motivation
Existing deep learning-based weather restoration methods are predominantly trained on synthetic data and suffer from three core limitations: (1) synthetic weather datasets lack physical accuracy, with coarse depth maps leading to unrealistic artifacts; (2) fixed-parameter models cannot adapt to unpredictable degradation patterns in the real world; and (3) single-model architectures cannot dynamically coordinate across different weather types. The authors observe that LLM-style reinforcement learning methods (e.g., GRPO) have not yet been successfully applied to image restoration, primarily because restoration models typically produce a single deterministic output per input and it is difficult to design deterministic rewards without paired ground truth. This paper is the first to successfully introduce the GRPO paradigm to image restoration, demonstrating that high-quality cold start and effective reward design are the critical factors.

## Method

### Overall Architecture
The system consists of three stages: (1) constructing the HFLS-Weather dataset and training weather-specific restoration models (cold start); (2) local-level PIQO performing unsupervised reinforcement fine-tuning on individual models; and (3) a global-level multi-agent system that dynamically orchestrates model selection and execution order via a meta-controller.

### Key Designs

**HFLS-Weather Dataset**: High-precision depth maps are generated using DepthAnything v2, and rain, fog, snow, and mixed weather conditions are simulated based on the atmospheric scattering model. The dataset contains one million high-fidelity synthetic images; depth-consistent multi-weather simulation overcomes the ghosting and uneven weather effects caused by coarse depth estimation in existing datasets (e.g., Snow100K, RESIDE-OTS). The weather degradation formulation is: $I_{weather}(x) = J(x)(1-M(x)-F(x)) + M(x) + A(x)F(x)$, where $F(x)=e^{-\beta d(x)}$ is the fog layer and $M(x)$ is the rain/snow layer.

**PIQO (Perturbation-driven Image Quality Optimization)**: PIQO addresses two key obstacles to applying GRPO in image restoration: (1) Gaussian perturbations $\theta_i' = \theta + \Delta$ are injected into model parameters to generate diverse outputs from a single input, enabling within-group comparison; (2) a no-reference composite reward function is designed, combining three quality assessment metrics—LIQE, CLIP-IQA, and Q-Align. MUSIQ is used as a filtering criterion, retaining only samples with quality exceeding that of the unperturbed model output; normalized advantages $A_i$ are then computed to perform gradient ascent. Implicit KL regularization is introduced to prevent excessively large parameter updates.

**Multi-Agent System**: The meta-controller uses CLIP to identify the weather degradation type of the input image and generates a weather description. Each specialized agent bids based on its historical success rate, and the system selects the highest-ranked agent to perform restoration. After restoration, dual evaluation is performed via CLIP re-analysis and IQA scoring: if the IQA score decreases, the system rolls back and selects the next-ranked agent; if degradation persists, a new round begins. The system restricts each session to at most three participating agents, and returns the image with the highest IQA score after three consecutive failures.

### Loss & Training
- Cold-start stage: weather-specific models are trained on HFLS-Weather with standard supervised loss.
- PIQO stage: policy gradient $g = -\frac{1}{|\mathcal{S}|}\sum_{i \in \mathcal{S}} A_i(\theta_i' - \theta)$, with KL trust-region constraint.
- Global stage: the composite IQA score serves as the reward to optimize the meta-controller's scheduling policy.

## Key Experimental Results

### Main Results

| Method | Snow Q-Align | Snow CLIP-IQA | Snow MUSIQ | Haze Q-Align | Rain Q-Align | Rain MUSIQ |
|--------|-------------|---------------|------------|-------------|-------------|------------|
| Chen et al. | 3.59 | 0.496 | 60.21 | 3.11 | 3.76 | 54.24 |
| WGWS | 3.59 | 0.503 | 60.48 | 3.11 | 3.80 | 54.55 |
| PromptIR | 3.65 | 0.529 | 61.17 | 3.09 | 3.81 | 54.67 |
| DA-CLIP | 3.63 | 0.522 | 61.16 | 3.13 | 3.81 | 54.96 |
| **Ours** | **3.96** | **0.592** | **67.80** | **3.56** | **4.03** | **64.12** |

| Weather | Metric | Chen | WGWS | PromptIR | DA-CLIP | Ours |
|---------|--------|------|------|----------|---------|------|
| Snow | Artifact Removal ↑ | 2.95 | 2.66 | 3.06 | 3.05 | **Best** |

### Ablation Study
- The HFLS-Weather cold start is critical for subsequent RL training; a low-quality cold start causes RL training collapse.
- The reward filtering strategy in PIQO effectively reduces gradient variance.
- The multi-agent system outperforms single all-in-one models in mixed-weather scenarios.

### Key Findings
- The proposed method substantially outperforms previous state-of-the-art across all real-world weather scenarios on four IQA metrics, with MUSIQ gains of approximately 7 points on snow and 9 points on haze.
- Although GPT-4o produces visually appealing results, it frequently introduces hallucinated objects and structural distortions, making it unsuitable for tasks requiring geometric and photometric fidelity.
- This is the first work to successfully apply the GRPO paradigm to image restoration.

## Highlights & Insights
1. **High Novelty**: The GRPO paradigm from the LLM domain is transferred to image restoration; parameter perturbation resolves the output diversity problem, while the no-reference composite IQA addresses the reward design challenge.
2. **Million-Scale Physics-Driven Dataset**: HFLS-Weather substantially surpasses existing datasets in both scale and physical fidelity.
3. **Closed-Loop Learning Ecosystem**: Local models continuously improve via real-world feedback, while the global controller dynamically optimizes coordination.
4. **Reinforcement Learning Without Paired Ground Truth**: This work breaks the dependence on paired training data that has long constrained image restoration research.

## Limitations & Future Work
- The multi-agent system has low inference efficiency due to the requirement for multi-round iterative evaluation.
- The reward function relies on the accuracy of existing IQA models and may fail in certain extreme scenarios.
- The parameter perturbation magnitude in PIQO requires careful tuning; excessively large perturbations lead to image degradation.
- Temporal consistency at the video level is not addressed.

## Related Work & Insights
- Compared to LLM-driven multi-agent restoration methods such as RestoreAgent and AgenticIR, the proposed approach achieves autonomous learning via RL rather than relying on fixed tool pipelines.
- The idea of transferring GRPO to visual tasks can be generalized to other low-level vision tasks (e.g., super-resolution, deblurring).
- The paradigm of physics-driven data synthesis combined with RL-based adaptive fine-tuning has broad applicability.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ (first application of GRPO to image restoration)
- **Technical Depth**: ⭐⭐⭐⭐⭐ (physical simulation + dual-level RL + multi-agent system)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (main experiments are thorough; ablation details are limited)
- **Writing Quality**: ⭐⭐⭐⭐ (strong adaptability to real-world weather scenarios)
- **Value**: ⭐⭐⭐⭐⭐ (pioneering work transferring LLM training paradigms to the visual domain)

## Supplementary Notes

The HFLS-Weather dataset is constructed using diverse background images from Snow100K, RESIDE-OTS, Google Landmark V2, and OSV5M to ensure scene diversity. The dataset substantially surpasses existing benchmarks in both scale (one million image pairs) and weather coverage (rain/snow/fog and their combinations). Qualitative evaluation results using GPT-4o (Table 3) show that the proposed method achieves the highest scores across dimensions including Artifact Removal, Color Accuracy, and Detail Preservation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MoDEM: A Morton-Order Degradation Estimation Mechanism for Adverse Weather Image Restoration](modem_a_morton-order_degradation_estimation_mechanism_for_adverse_weather_image_.md)
- [\[CVPR 2026\] Beyond Ground-Truth: Leveraging Image Quality Priors for Real-World Image Restoration](../../CVPR2026/image_restoration/beyond_ground-truth_leveraging_image_quality_priors_for_real-world_image_restora.md)
- [\[CVPR 2025\] Pixel-level and Semantic-level Adjustable Super-resolution: A Dual-LoRA Approach](../../CVPR2025/image_restoration/pixel-level_and_semantic-level_adjustable_super-resolution_a_dual-lora_approach.md)
- [\[CVPR 2026\] RL-ScanIQA: Reinforcement-Learned Scanpaths for Blind 360deg Image Quality Assessment](../../CVPR2026/image_restoration/rl-scaniqa_reinforcement-learned_scanpaths_for_blind_360deg_image_quality_assess.md)
- [\[ICCV 2025\] Robust Adverse Weather Removal via Spectral-based Spatial Grouping (SSGformer)](../../ICCV2025/image_restoration/robust_adverse_weather_removal_via_spectral-based_spatial_grouping.md)

</div>

<!-- RELATED:END -->
