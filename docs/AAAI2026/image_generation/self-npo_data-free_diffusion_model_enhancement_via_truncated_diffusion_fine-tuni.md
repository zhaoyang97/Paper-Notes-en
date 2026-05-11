---
title: >-
  [Paper Note] Self-NPO: Data-Free Diffusion Model Enhancement via Truncated Diffusion Fine-Tuning
description: >-
  [AAAI 2026][Image Generation][Diffusion Models] This paper proposes Self-NPO, a negative preference optimization method that requires neither external data annotation nor reward models. By leveraging Truncated Diffusion…
tags:
  - "AAAI 2026"
  - "Image Generation"
  - "Diffusion Models"
  - "Preference Optimization"
  - "Negative Preference Optimization"
  - "Self-Learning"
  - "Training Efficiency"
date: 2026-05-08
content_hash: cf55dbed4341376c
---

# Self-NPO: Data-Free Diffusion Model Enhancement via Truncated Diffusion Fine-Tuning

**Conference**: AAAI 2026
**arXiv**: [2505.11777](https://arxiv.org/abs/2505.11777)
**Code**: [https://github.com/G-U-N/Diffusion-NPO](https://github.com/G-U-N/Diffusion-NPO)
**Area**: Image Generation
**Keywords**: Diffusion Models, Preference Optimization, Negative Preference Optimization, Self-Learning, Training Efficiency

## TL;DR

This paper proposes Self-NPO, a negative preference optimization method that requires neither external data annotation nor reward models. By leveraging Truncated Diffusion Fine-Tuning (TDFT), the model learns "what is bad" from its own low-quality generated data, and uses CFG to steer generation away from undesirable outputs. Self-NPO achieves comparable performance to Diffusion-NPO at less than 1% of the training cost.

## Background & Motivation

### State of the Field

Diffusion models have achieved remarkable success in image, video, and 3D generation. However, models trained on large-scale unfiltered data often produce results misaligned with human preferences. Current preference optimization (PO) approaches fall into four main categories:
- **Differentiable Reward (DR)**: Optimizes reward models via backpropagation
- **Reinforcement Learning (RL)**: Models the denoising process as an MDP and optimizes with PPO
- **Direct Preference Optimization (DPO)**: Fine-tunes directly on preference-pair datasets
- **Negative Preference Optimization (NPO)**: Trains the model to generate outputs contrary to human preferences, then uses CFG to steer away from poor results

### Limitations of Prior Work

All existing methods rely heavily on **explicit preference annotations**—either requiring costly human-annotated preference pairs (e.g., Pick-a-Pic) or training brittle reward models (e.g., HPSv2, ImageReward). These approaches are severely constrained in domains where preference data is scarce or difficult to annotate, such as medical imaging or professional design.

### Starting Point

A key insight motivates this work: diffusion models naturally produce low-quality outputs (blurry details, structural errors, hallucinations, mode collapse), which would inherently receive low scores under any reward model. Therefore, **learning from a model's own generated data is equivalent to distribution-regularized negative preference optimization**.

### Core Idea

Three key observations underpin Self-NPO:

**Controlled capability degradation** is equivalent to NPO, but degradation must not be arbitrary—sufficient correlation with the positive model must be maintained to avoid CFG variance explosion.

**Learning from self-generated data** naturally satisfies the distribution preservation condition.

**Full diffusion generation is unnecessary**—the proposed TDFT uses the Tweedie formula to obtain $\mathbf{x}_0$ estimates at intermediate timesteps, avoiding prohibitive generation costs.

## Method

### Overall Architecture

The Self-NPO pipeline proceeds as follows:
1. Perform partial truncated denoising with the reference model (from $\mathbf{x}_T$ to $\mathbf{x}_t$)
2. Obtain $\mathbf{x}_{t\to0}^{ref}$ (the conditional expectation estimate of $\mathbf{x}_0$) via the Tweedie formula
3. Fine-tune the model using $\mathbf{x}_{t\to0}^{ref}$ as the target (standard diffusion loss)
4. The fine-tuned model serves as the negative preference model in CFG at inference time

The CFG formula at inference:
$$\boldsymbol{\epsilon}^\omega = (\omega+1)\boldsymbol{\epsilon}_{\boldsymbol{\theta}_{pos}}(\mathbf{x}_t, t, \boldsymbol{c}) - \omega\boldsymbol{\epsilon}_{\boldsymbol{\theta}_{neg}}(\mathbf{x}_t, t, \boldsymbol{c}')$$

### Key Designs

#### 1. **NPO via Self-Generated Data**

Starting from the RLHF framework, the NPO optimization objective is:
$$\max_{\mathbb{P}_\theta} \mathbb{E}[R_{\text{NPO}}(\mathbf{x}_0, \boldsymbol{c})] - \beta D_{\text{KL}}[\mathbb{P}_\theta(\mathbf{x}_0|\boldsymbol{c}) \| \mathbb{P}_{\text{ref}}(\mathbf{x}_0|\boldsymbol{c})]$$

where $R_{\text{NPO}} = 1 - R$ is the inverted reward. The rationale for using self-generated data is twofold:
- **Distribution preservation**: Learning from self-generated data maintains the original distributional characteristics
- **Reward reduction**: Self-generated samples inherently carry low reward scores due to hallucinations, mode collapse, and optimization imperfections

**Design Motivation**: A model trained on self-generated data becomes "weaker," and this degraded model, when used as the negative term in CFG, effectively steers generation away from low-quality patterns. Furthermore, due to distributional correlation, this does not cause CFG output variance explosion (fully independent Gaussian noise would increase variance to $2\omega^2 + 2\omega + 1$).

#### 2. **Truncated Diffusion Fine-Tuning (TDFT)**

**Baseline approach** (full simulation): Fully denoises from $\mathbf{x}_T$ to $\mathbf{x}_0^{ref}$, then trains with standard diffusion loss:
$$\min_{\boldsymbol{\theta}} \mathbb{E}_{t,\boldsymbol{\epsilon}} \|\mathbf{x}_0^{ref} - \boldsymbol{f}_{\boldsymbol{\theta}}((\mathbf{x}_0^{ref})_t, t, \boldsymbol{c})\|_2^2$$

**Problem**: The full denoising process is computationally prohibitive.

**Core of TDFT**: Using the Tweedie formula, the conditional expectation of $\mathbf{x}_0$ can be obtained directly at any intermediate timestep $t$:
$$\mathbf{x}_{t\to0}^{ref} = \mathbb{E}[\mathbf{x}_0 | \mathbf{x}_t^{ref}] = \mathbf{x}_t + \sigma_t^2 \nabla_{\mathbf{x}_t} \log \mathbb{P}_{\text{ref}}(\mathbf{x}_t^{ref}|\boldsymbol{c})$$

The truncated denoising process is:
$$\mathbf{x}_T^{ref} \to \mathbf{x}_{T-1}^{ref} \to \dots \mathbf{x}_t^{ref} \xrightarrow{\text{Tweedie}} \mathbf{x}_{t\to0}^{ref}$$

#### 3. **Resolving Distribution Mismatch**

**Problem**: $\mathbf{x}_{t\to0}^{ref}$ is a weighted average of multiple possible $\mathbf{x}_0$ values; its distribution differs from $\mathbf{x}_0^{ref} \sim \mathbb{P}_{\text{ref}}(\mathbf{x}_0|\boldsymbol{c})$, and directly adding noise to $\mathbf{x}_{t\to0}^{ref}$ introduces a distribution mismatch.

**Solution**: Instead of directly adding noise to $\mathbf{x}_{t\to0}^{ref}$, noise is added from $\mathbf{x}_t^{ref}$ to $\mathbf{x}_s$:
$$\mathbf{x}_s = \frac{\alpha_s}{\alpha_t}\mathbf{x}_t^{ref} + \sqrt{\sigma_s^2 - \sigma_t^2\frac{\alpha_s^2}{\alpha_t^2}}\boldsymbol{\epsilon}$$

Then $\mathbf{x}_{t\to0}^{ref}$ is used as the $\mathbf{x}_0$ prediction target:
$$\min_{\boldsymbol{\theta}} \mathbb{E}_{\mathbf{x}_s \sim \mathbb{P}(\mathbf{x}_s|\mathbf{x}_t^{ref})} \|\mathbf{x}_{t\to0}^{ref} - \boldsymbol{f}_{\boldsymbol{\theta}}(\mathbf{x}_s, s, \boldsymbol{c})\|_2^2$$

**Design Motivation**: This guarantees that the distribution of $\mathbf{x}_s$ is equivalent to the noise-perturbed distribution of the reference model $\mathbb{P}_{\text{ref}}(\mathbf{x}_0^{ref}|\boldsymbol{c})$.

### Loss & Training

The core objective is the standard diffusion training loss (in $\mathbf{x}_0$-prediction form), requiring no additional reward models or preference data.

The weight combination strategy at inference (inherited from Diffusion-NPO):
$$\boldsymbol{\theta}_{neg} = \boldsymbol{\theta} + \alpha\boldsymbol{\eta} + \beta\boldsymbol{\delta}$$

where $\boldsymbol{\eta}$ is the positive preference optimization offset, $\boldsymbol{\delta}$ is the negative preference optimization offset, and $\alpha, \beta \in [0,1]$ control the mixing degree to ensure stable and coherent outputs.

### Theoretical Foundation

The authors justify TDFT from three angles:
1. **Distribution equivalence**: The distribution of $\mathbf{x}_s$ is equivalent to the noise-perturbed distribution of the reference model (Theorem 1)
2. **Gradient equivalence**: The optimization objective Eq. 17 admits a gradient-equivalent learning target (Theorem 2)
3. **Standard diffusion equivalence**: The gradient-equivalent target is equivalent to the standard diffusion model learning objective (Theorem 3)

## Key Experimental Results

### Main Results

Quantitative comparison on SD1.5 (Pick-a-Pic test_unique split):

| Method | PickScore↑ | HPSv2.1↑ | ImageReward↑ | Aesthetic↑ |
|--------|-----------|----------|-------------|-----------|
| SD-1.5 | 20.75 | 26.84 | 0.1064 | 5.539 |
| SD-1.5 + NPO | 21.26 | 27.36 | 0.2028 | 5.667 |
| **SD-1.5 + Self-NPO** | 21.00 | 27.04 | **0.2816** | 5.609 |
| DreamShaper | 21.85 | 28.85 | 0.6819 | 6.143 |
| DreamShaper + NPO (α=1.0) | 22.30 | 30.13 | 0.7258 | 6.234 |
| **DreamShaper + Self-NPO (α=1.0)** | 22.20 | **30.40** | **0.8038** | 6.196 |
| SePPO | 21.51 | 28.45 | 0.5981 | 5.892 |
| **SePPO + Self-NPO** | **21.73** | **30.28** | **0.6744** | **6.014** |

Comparison on SDXL:

| Method | PickScore↑ | HPSv2.1↑ | ImageReward↑ | Aesthetic↑ |
|--------|-----------|----------|-------------|-----------|
| SDXL | 22.06 | 28.04 | 0.6246 | 6.114 |
| SDXL + Self-NPO | 22.26 | 28.24 | 0.6697 | **6.226** |
| Diff.-DPO | 22.57 | 29.76 | 0.8624 | 6.099 |
| Diff.-DPO + Self-NPO | 22.67 | 29.83 | 0.8784 | **6.179** |
| Juggernaut + Self-NPO | **22.77** | **30.56** | **0.9921** | 6.031 |

### Ablation Study

Training cost comparison:

| Method | Training Mode | GPU Hours | GPU Type |
|--------|--------------|-----------|----------|
| Diffusion-NPO | Full weights | 384 | A100 |
| Diffusion-NPO | LoRA | 153.6 | A800 |
| Baseline (full simulation) | Full weights | 10.4 | A800 |
| **Self-NPO (default K=5)** | **Full weights** | **2** | **A800** |

Self-NPO requires **less than 1%** of Diffusion-NPO's training cost (2 vs. 384 GPU hours), and is 5× faster than the full-simulation baseline.

### Key Findings

1. **Plug-and-play property**: Self-NPO integrates seamlessly into SD1.5, SDXL, CogVideoX, and models already subjected to preference optimization, consistently yielding improvements
2. **Most pronounced gains on ImageReward**: Across most settings, Self-NPO delivers the largest gains on ImageReward, indicating the greatest improvement in human preference alignment
3. **User study**: Improvements are observed across four dimensions—color and lighting, high-frequency details, low-frequency composition, and text-image alignment—with the most notable gains in high-frequency details
4. **Controllable generation compatibility**: Remains effective when combined with control plugins such as T2I-Adapter
5. **Unconditional generation improvement**: Models enhanced with Self-NPO produce higher-quality images even under unconditional generation

## Highlights & Insights

1. **Exceptional training efficiency** is the most significant contribution—2 GPU hours vs. 384 GPU hours, nearly a 200× efficiency improvement, transforming NPO from expensive large-scale training into lightweight self-learning
2. **The data-free design** carries strong practical value, particularly suited for domains with scarce data or difficult annotation
3. **Creative application of the Tweedie formula** elegantly addresses the generation cost problem by jumping directly from intermediate timesteps to the $\mathbf{x}_0$ estimate
4. **The distribution mismatch solution** is elegant—rather than adding noise to the estimated $\mathbf{x}_0$, noise is added from the intermediate state $\mathbf{x}_t$, preserving the true noise-perturbed distribution
5. The intuition that "degradation equals NPO" is simple yet theoretically rigorous, with equivalence established at three levels: distribution, gradient, and loss function

## Limitations & Future Work

1. **Performance ceiling**: The absence of explicit preference annotations implies that Self-NPO's performance upper bound may fall below that of NPO trained with high-quality annotations
2. **Uncontrolled degree of reward reduction**: The extent of quality degradation in self-generated data is implicit, lacking precise control over the degree of capability weakening
3. **Validation limited to visual generation**: The method has not been tested in other diffusion model applications such as text generation
4. **CFG dependency**: The effectiveness of the method relies entirely on the CFG inference paradigm and is not applicable to generation pipelines that do not employ CFG
5. **Potential for hybrid strategies**: Combining Self-NPO with a small quantity of high-quality preference data warrants further exploration

## Related Work & Insights

- **Diffusion-NPO** (Wang et al., 2025): The original work on negative preference optimization; this paper extends it to a data-free setting
- **Diffusion-DPO** (Wallace et al., 2024): Application of direct preference optimization to diffusion models
- **SePPO** (Zhang et al., 2024): Preference pair optimization via self-scoring
- **RLHF for Diffusion**: Self-NPO can be understood from an RLHF perspective—KL regularization maintains proximity to the reference model
- **Key Takeaway**: Self-NPO demonstrates that the "self-defects" of diffusion models can be leveraged in reverse—the model's implicit knowledge of what it generates poorly is a more efficient signal than seeking external preference supervision

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The insight that "self-generated data = negative preference data" is novel; TDFT constitutes an effective technical contribution
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers SD1.5/SDXL/CogVideoX with comprehensive comparisons against NPO/DPO and others, including user studies and efficiency analysis
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated, mathematical derivations are rigorous, and theoretical proofs are complete
- **Value**: ⭐⭐⭐⭐⭐ — Extremely high practical value; lowers the barrier to preference optimization by two orders of magnitude

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] FreeInpaint: Tuning-free Prompt Alignment and Visual Rationality Enhancement in Image Inpainting](freeinpaint_tuning-free_prompt_alignment_and_visual_rationality_enhancement_in_i.md)
- [\[ICLR 2026\] Stochastic Self-Guidance for Training-Free Enhancement of Diffusion Models](../../ICLR2026/image_generation/stochastic_self-guidance_for_training-free_enhancement_of_diffusion_models.md)
- [\[AAAI 2026\] DogFit: Domain-guided Fine-tuning for Efficient Transfer Learning of Diffusion Models](dogfit_domain-guided_fine-tuning_for_efficient_transfer_learning_of_diffusion_mo.md)
- [\[AAAI 2026\] Difficulty Controlled Diffusion Model for Synthesizing Effective Training Data](difficulty_controlled_diffusion_model_for_synthesizing_effec.md)
- [\[AAAI 2026\] SpecDiff: Accelerating Diffusion Model Inference with Self-Speculation](specdiff_accelerating_diffusion_model_inference_with_self-speculation.md)

</div>

<!-- RELATED:END -->
