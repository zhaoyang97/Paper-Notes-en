---
title: >-
  [Paper Note] DiverseFlow: Sample-Efficient Diverse Mode Coverage in Flows
description: >-
  [CVPR 2025][Image Generation][Flow Matching] This paper proposes DiverseFlow, a training-free inference-time method that introduces inter-sample coupled gradient constraints during the ODE solving process of flow models via Determinant Point Processes (DPP), significantly improving the diversity and mode coverage of generated samples under a fixed sampling budget.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Flow Matching"
  - "Diverse Sampling"
  - "Determinant Point Process"
  - "Polysemy Generation"
  - "Mode Coverage"
date: 2026-05-08
content_hash: fecb046cba0f4b47
---

# DiverseFlow: Sample-Efficient Diverse Mode Coverage in Flows

**Conference**: CVPR 2025  
**arXiv**: [2504.07894](https://arxiv.org/abs/2504.07894)  
**Code**: None  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Flow Matching, Diverse Sampling, Determinant Point Process, Polysemy Generation, Mode Coverage

## TL;DR
This paper proposes DiverseFlow, a training-free inference-time method that introduces inter-sample coupled gradient constraints during the ODE solving process of flow models via Determinant Point Processes (DPP), significantly improving the diversity and mode coverage of generated samples under a fixed sampling budget.

## Background & Motivation

**Background**: Continuous-time generative models such as Flow Matching and diffusion models have become mainstream, achieving outstanding results in tasks like text-to-image generation. Currently, a large body of work focuses on improving fidelity and sampling efficiency.

**Limitations of Prior Work**: The standard IID sampling pattern may repeatedly generate similar results under a limited sampling budget, thereby missing other modes in the distribution. For example, the prompt "A famous boxer" might only generate dogs (Boxer breed) while ignoring the athlete meaning. Users are forced to sample repeatedly until the desired modes are covered.

**Key Challenge**: Samples that are far apart in the source distribution are not necessarily far apart in the target distribution after flow mapping—the flow mapping does not preserve distance structures. Therefore, the intuitive approach of "selecting diverse source samples" does not work. Meanwhile, optimizing source samples requires multiple full ODE simulations and backpropagations, which is computationally prohibitive.

**Goal**: To enable $K$ samples to cover as many modes as possible without increasing the number of samples.

**Key Insight**: Leveraging the "repulsion" property of DPP—DPP naturally assigns higher probabilities to "more diverse" sets, and it is differentiable.

**Core Idea**: At each step of the ODE solving process, the target samples are estimated using the current samples, and a DPP kernel matrix is constructed to measure the diversity of the sample set in the feature space. The gradient of the DPP log-likelihood is then injected into the ODE velocity field, forming a system of coupled ODEs that makes the samples mutually repulsive.

## Method

### Overall Architecture
Given $K$ source samples $\{x_0^{(i)}\}$, at each step of the standard flow ODE solving process, the target positions $\hat{x}_1^{(i)}$ of each sample are first estimated via an Euler step. Then, a DPP kernel matrix is constructed in the feature space to evaluate the set diversity. The gradient of the DPP log-likelihood is computed and injected into the ODE velocity field. Ultimately, the $K$ trajectories transition from independent ODEs into a coupled ODE system.

### Key Designs

1. **DPP Diversity Objective**:

    - **Function**: Measures the diversity of a set of samples and provides gradient signals.
    - **Mechanism**: Construct the kernel matrix $L^{(ij)} = \exp(-h \|F(\hat{x}_1^{(i)}) - F(\hat{x}_1^{(j)})\|^2 / \text{med}(D))$, where $F$ is a feature extractor (e.g., ViT). The DPP likelihood is $\mathcal{L} = \det(L) / \det(L+I)$. A more diverse set of samples yields a larger determinant. The gradient of the log-likelihood $\nabla_{x_t^{(i)}} \log \mathcal{L}$ is injected into the ODE as a repulsive force.
    - **Design Motivation**: DPP assigns zero probability to duplicate samples (where the determinant contains identical rows), which is the most rigorous diversity metric. Compared to the sum-of-kernel methods in SVGD, the volume-based metric of DPP is better suited for discovering new modes.

2. **Quality Constraint**:

    - **Function**: Prevents the diversity gradient from pushing samples away from reasonable regions.
    - **Mechanism**: Check whether the backward estimate $\hat{x}_0^{(i)}$ is still in the high-probability region of the source distribution (judged by the $\chi^2$ quantile). If it deviates too far, reduce the DPP weight $q^{(i)}$ of that sample. The modified kernel is $L_q = L \odot q q^T$.
    - **Design Motivation**: Pure repulsive forces may push samples into low-density regions, producing low-quality outputs. The quality term achieves a balance between diversity and quality.

3. **Coupled ODE System**:

    - **Function**: Couples $K$ independent ODE trajectories into a single diversity-driven system.
    - **Mechanism**: Modify the velocity of the $i$-th particle to $\tilde{v}_t^{(i)} = v_t^{(i)} - \gamma(t) \nabla_{x_t^{(i)}} \log \mathcal{L}$, where $\gamma(t)$ is a time-varying scaling factor. $\gamma=0$ degenerates to standard IID sampling. It is solved using the Euler method.
    - **Design Motivation**: Step-by-step optimization during the ODE solving process avoids the high computational cost of requiring multiple full ODE simulations and backpropagations.

### Loss & Training
Completely training-free, being an inference-time sample optimization method.

## Key Experimental Results

### Main Results (ImageNet-256 Class-Conditional Generation, Precision/Recall)

| Method | CFG | Precision↑ | Recall↑ |
|------|-----|-----------|---------|
| LFM | 1.5 | 0.69 | 0.44 |
| LFM + DiverseFlow | 1.5 | 0.69 | **0.47** |
| LFM | 2.0 | 0.77 | 0.41 |
| LFM + DiverseFlow | 2.0 | 0.76 | **0.46** |
| LFM | 4.0 | 0.69 | 0.26 |
| LFM + DiverseFlow | 4.0 | 0.70 | **0.38** |

### Ablation Study

| Configuration | Effect |
|------|------|
| Direct mapping of diverse source samples | Does not guarantee target diversity (verified in Fig.2) |
| IID sampling vs DiverseFlow | $K=5$ covers 3/10 modes vs 5/10 modes |
| Different FM formulations | Both CFM and MB-OT benefit from DiverseFlow |
| High CFG benefits more | Recall increases by +0.12 at CFG=4 (from 0.26 to 0.38) |

### Key Findings
- DiverseFlow significantly improves Recall without compromising Precision, with the largest improvement occurring in high CFG (low-diversity) scenarios.
- In 2D synthetic experiments, 5 samples can cover 5 modes (whereas IID can only cover 3).
- In polysemous text-to-image generation, it can discover multiple semantics (e.g., "boxer" generates both Boxer dogs and athletes).
- In face inpainting tasks, it can generate more diverse facial expressions and features.

## Highlights & Insights
- Introducing DPP into flow model sampling is a novel combination. The determinant of DPP is naturally suited for measuring "set volume," i.e., diversity.
- The observation that "diversity in source space $\neq$ diversity in target space," though intuitive, is systematically validated for the first time, providing an important baseline for diversity sampling research.
- The quality constraint prevents degradation via the probability density of the source distribution, presenting a simple and effective design.

## Limitations & Future Work
- It is required to compute the DPP kernel matrix and gradients between samples, incurring a computational overhead proportional to the square of the sample size $K^2$.
- When $K$ is very large (far exceeding the number of modes), the repulsive force may lead to a degradation in the quality of some samples.
- The choice of feature extractor $F$ has a significant impact on the results, and different tasks may require different extractors $F$.
- Future work could consider extending to SDE samplers and more efficient DPP approximation methods.

## Related Work & Insights
- **vs Particle Guidance (Corso et al.)**: Uses SVGD-based row-summing as a diversity metric, which tolerates duplicate samples. DiverseFlow is based on determinants (volume), assigning zero probability to duplicates, which is more rigorous.
- **vs CFG Tuning**: Reducing CFG can increase diversity but sacrifices quality. DiverseFlow maintains diversity even under high CFG.
- **vs Multiple Independent Samplings**: DiverseFlow covers more modes under the same budget, making it more efficient.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of DPP and Flow is novel, but inference-time guidance is an existing paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validation across multiple tasks (text-to-image generation, inpainting, class-conditional), with thorough ablation on synthetic data.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, rigorous theoretical derivation, and excellent visualizations.
- Value: ⭐⭐⭐⭐ A general-purpose inference-time diversity method with a wide range of application scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Learning to Sample Effective and Diverse Prompts for Text-to-Image Generation](learning_to_sample_effective_and_diverse_prompts_for_text-to-image_generation.md)
- [\[NeurIPS 2025\] HollowFlow: Efficient Sample Likelihood Evaluation using Hollow Message Passing](../../NeurIPS2025/image_generation/hollowflow_efficient_sample_likelihood_evaluation_using_hollow_message_passing.md)
- [\[CVPR 2026\] Learning Straight Flows: Variational Flow Matching for Efficient Generation](../../CVPR2026/image_generation/learning_straight_flows_variational_flow_matching_for_efficient_generation.md)
- [\[CVPR 2025\] OmniFlow: Any-to-Any Generation with Multi-Modal Rectified Flows](omniflow_any-to-any_generation_with_multi-modal_rectified_flows.md)
- [\[ICLR 2026\] Sample-Efficient Evidence Estimation of Score-Based Priors for Model Selection](../../ICLR2026/image_generation/sample-efficient_evidence_estimation_of_score_based_priors_for_model_selection.md)

</div>

<!-- RELATED:END -->
