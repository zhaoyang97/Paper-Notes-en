---
title: >-
  [Paper Note] Fine-Tuning Diffusion Models via Intermediate Distribution Shaping
description: >-
  [ICLR 2026][Computational Biology][diffusion model fine-tuning] This work unifies rejection-sampling-based fine-tuning methods under the GRAFT framework…
tags:
  - "ICLR 2026"
  - "Computational Biology"
  - "diffusion model fine-tuning"
  - "rejection sampling"
  - "KL regularization"
  - "intermediate distribution"
  - "inverse noise correction"
date: 2026-05-08
content_hash: 177fb96c5cb54668
---

# Fine-Tuning Diffusion Models via Intermediate Distribution Shaping

**Conference**: ICLR 2026
**arXiv**: [2510.02692](https://arxiv.org/abs/2510.02692)  
**Code**: None  
**Area**: Diffusion Models / Fine-Tuning
**Keywords**: diffusion model fine-tuning, rejection sampling, KL regularization, intermediate distribution, inverse noise correction

## TL;DR
This work unifies rejection-sampling-based fine-tuning methods under the GRAFT framework, proving that they implicitly perform KL-regularized reward maximization. Building on this, P-GRAFT is proposed to perform distribution shaping at intermediate denoising steps (achieving a better bias–variance trade-off), and Inverse Noise Correction is introduced to improve flow model quality without reward signals, yielding an 8.81% VQAScore improvement on text-to-image generation.

## Background & Motivation

**Background**: Fine-tuning diffusion models commonly relies on PPO with KL regularization; however, the marginal likelihood of diffusion models is intractable, causing the KL term to be either ignored (leading to instability) or approximated via trajectory-level KL (suboptimal, with value function initialization bias).

**Limitations of Prior Work**: (1) Intractable marginal KL forces PPO to rely on relaxed approximations; (2) rejection-sampling methods (RAFT/BoN) are practical but lack clear theoretical grounding; (3) distribution shaping is applied only to the final data distribution, leaving the structural information of intermediate denoising steps unexploited.

**Key Challenge**: Diffusion model fine-tuning requires KL regularization for stability, yet the marginal KL is intractable.

**Key Insight**: Proving that rejection sampling implicitly enforces the marginal KL constraint (even though the likelihood is intractable), then exploiting the multi-step structure of diffusion models to perform shaping at intermediate distributions.

**Core Idea**: Rejection sampling = implicit KL regularization → apply rejection sampling at intermediate denoising steps → superior bias–variance trade-off.

## Method

### Overall Architecture
Two main contributions: (1) **P-GRAFT**: fine-tuning via rejection sampling at an intermediate denoising step $t$; (2) **Inverse Noise Correction**: inverting the flow model to learn an improved initial noise distribution without reward signals.

### Key Designs

1. **GRAFT Unified Framework**:

    - **Function**: Unifies classical rejection sampling, Best-of-N, Top-K, and related methods under a Generalized Rejection Sampling (GRS) framework.
    - **Mechanism**: Lemma 2.3 proves that the distribution of samples accepted by GRS is the solution to KL-regularized reward maximization, $p^{\text{RL}}(x) \propto \exp(\hat{r}(x)/\alpha)\bar{p}(x)$, with reward reshaping incorporated.
    - **Design Motivation**: Although the marginal KL of diffusion models is intractable, GRS implicitly realizes it.

2. **P-GRAFT (Partial-GRAFT)**:

    - **Function**: Performs rejection sampling on intermediate denoising states $X_t$ rather than on final generated samples.
    - **Mechanism**: Lemma 3.2 proves that P-GRS shapes the intermediate distribution $\bar{p}_t$ rather than the final distribution. The fine-tuned model handles denoising from $T \to t$, while the original model handles $t \to 0$. Bias–variance trade-off: large $t$ yields high reward variance but a simpler learning problem (simpler score function); small $t$ yields more accurate reward estimates but a harder learning problem.
    - **Design Motivation**: Choosing an appropriate intermediate time $t$ allows both aspects to be balanced simultaneously.

3. **Inverse Noise Correction**:

    - **Function**: Inverts the flow model's mapping from data to noise to learn an improved initial noise distribution.
    - **Mechanism**: An adapter is trained to apply corrections in the noise space without requiring an explicit reward function.
    - **Design Motivation**: The invertibility of flow models enables inference of the initial noise distribution.

### Loss & Training
- **P-GRAFT**: Generate $M$ trajectories → apply GRS at the intermediate step to select accepted samples → fine-tune the $T \to t$ denoising segment via supervised fine-tuning (SFT).
- **Inverse Noise Correction**: Parameter-efficient adapter fine-tuning.

## Key Experimental Results

### Main Results
Fine-tuning Stable Diffusion v2 for text-to-image generation:

| Method | VQAScore | Relative Gain over Baseline | Notes |
|---|---|---|---|
| SD v2 (baseline) | baseline | — | No fine-tuning |
| Policy Gradient | moderate | moderate | PPO-type method |
| GRAFT (final step) | good | good | Standard rejection sampling |
| **P-GRAFT** | **best** | **+8.81%** | Intermediate-step rejection sampling |
| SDXL-Base | reference | — | Larger model |

### Multi-Task Validation

| Task | Method | Result |
|---|---|---|
| Layout generation | P-GRAFT | Significant improvement |
| Molecule generation | P-GRAFT + deduplication | Improvement + diversity preservation |
| Unconditional image generation | Inverse Noise Correction | Improved FID + reduced FLOPs |

### Key Findings
- P-GRAFT outperforms policy gradient methods (PPO) and standard GRAFT on text-to-image generation.
- Hypothesis testing confirms that intermediate states $X_t$ at smaller $t$ carry more information about the final reward (variance analysis).
- In molecule generation, the deduplication variant of GRS effectively prevents mode collapse; the reshaped reward automatically incorporates a diversity term.
- Inverse Noise Correction improves FID without requiring reward signals and reduces per-image FLOPs.

## Highlights & Insights
- **GRS = Implicit KL Regularization**: This theoretical result addresses a fundamental technical challenge in diffusion model fine-tuning. Since marginal KL is intractable, it need not be computed explicitly — rejection sampling implicitly enforces it.
- **Bias–Variance Perspective on Intermediate Distribution Shaping**: The choice of intermediate step is not merely an engineering decision but is grounded in clear mathematical principles — selecting $t$ that minimizes the product of bias and variance.
- **Deduplicated GRS for Molecule Generation**: The reshaped reward $\hat{r}$ automatically includes a diversity penalty — $\log(1/N_{\text{copies}})$ — elegantly preventing mode collapse.

## Limitations & Future Work
- The optimal intermediate time $t$ in P-GRAFT requires empirical search.
- Inverse Noise Correction is applicable only to flow models, which require invertibility.
- Generating $M$ complete trajectories incurs significant computational overhead.
- The theoretical analysis relies on the assumption of a well-trained denoiser.

## Related Work & Insights
- **vs. PPO/DPPO**: Avoids the difficulty of KL computation; implicit KL constraint yields greater stability.
- **vs. RAFT/RSO**: GRAFT provides a unified theoretical perspective, and P-GRAFT further leverages the diffusion structure for additional gains.
- **vs. DPO for diffusion**: DPO transfers KL via preference data; P-GRAFT employs rejection sampling more directly.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The GRAFT unified theory and the bias–variance analysis underlying P-GRAFT are both significant contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers text-to-image, layout, molecule, and unconditional image generation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Theory and practice are tightly integrated, with clear mathematical derivations.
- **Value**: ⭐⭐⭐⭐⭐ Carries important theoretical and practical implications for the paradigm of diffusion model fine-tuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Thompson Sampling via Fine-Tuning of LLMs](thompson_sampling_via_fine-tuning_of_llms.md)
- [\[ICLR 2026\] Antibody: Strengthening Defense Against Harmful Fine-Tuning for Large Language Models via Attenuating Harmful Gradient Influence](antibody_strengthening_defense_against_harmful_fine-tuning_for_large_language_mo.md)
- [\[ICML 2026\] Constrained Flow Optimization via Sequential Fine-Tuning for Molecular Design](../../ICML2026/computational_biology/constrained_flow_optimization_via_sequential_fine_tuning_for_molecular_design.md)
- [\[NeurIPS 2025\] Iterative Foundation Model Fine-Tuning on Multiple Rewards](../../NeurIPS2025/computational_biology/iterative_foundation_model_fine-tuning_on_multiple_rewards.md)
- [\[ICLR 2026\] DriftLite: Lightweight Drift Control for Inference-Time Scaling of Diffusion Models](driftlite_lightweight_drift_control_for_inference-time_scaling_of_diffusion_mode.md)

</div>

<!-- RELATED:END -->
