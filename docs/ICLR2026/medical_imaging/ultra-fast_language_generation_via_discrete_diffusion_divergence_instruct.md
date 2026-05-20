---
title: >-
  [Paper Note] Ultra-Fast Language Generation via Discrete Diffusion Divergence Instruct
description: >-
  [ICLR 2026][Medical Imaging][Discrete Diffusion] This paper proposes DiDi-Instruct, a distillation framework based on Integrated KL Divergence (IKL) minimization that compresses a pretrained diffusion large language mode…
tags:
  - "ICLR 2026"
  - "Medical Imaging"
  - "Discrete Diffusion"
  - "Distillation"
  - "Masked Diffusion Model"
  - "KL Divergence"
  - "Few-Step Generation"
  - "Policy Gradient"
date: 2026-05-08
content_hash: 8b9f4f5d0215c212
---

# Ultra-Fast Language Generation via Discrete Diffusion Divergence Instruct

**Conference**: ICLR 2026
**arXiv**: [2509.25035](https://arxiv.org/abs/2509.25035)  
**Code**: [https://github.com/haoyangzheng-ai/didi-instruct](https://github.com/haoyangzheng-ai/didi-instruct)  
**Area**: Medical Imaging
**Keywords**: Discrete Diffusion, Distillation, Masked Diffusion Model, KL Divergence, Few-Step Generation, Policy Gradient

## TL;DR

This paper proposes DiDi-Instruct, a distillation framework based on Integrated KL Divergence (IKL) minimization that compresses a pretrained diffusion large language model (dLLM) into a few-step student model. Through four key designs—adversarial density ratio estimation, grouped reward normalization, score decomposition, and a Reward-Guided Ancestral Sampler (RGAS)—the student model surpasses the 1024-step teacher's perplexity on OpenWebText using only 16 steps, achieving up to 64× inference speedup at a training cost of just 1 GPU hour.

## Background & Motivation

**Background**: Autoregressive (AR) large language models (e.g., the GPT series) have achieved remarkable success across NLP tasks, yet their left-to-right sequential token generation imposes a fundamental throughput ceiling. Diffusion large language models (dLLMs) reframe text generation as an iterative denoising process inspired by image diffusion models, enabling parallel generation via bidirectional attention and emerging as a compelling alternative to AR models.

**Limitations of Prior Work**:
- **Excessive inference steps**: dLLMs require 256 steps on the OpenWebText benchmark to match GPT-2's generation quality, leaving inference efficiency unsatisfactory.
- **Inadequate distillation methods**: SDTT (self-distillation) cannot match GPT-2 within 32 steps; DUO (consistency distillation) requires multiple training rounds with substantial GPU overhead (20+ GPU hours); DSDD performs distribution matching but yields limited gains on text generation.
- **Lack of theoretical grounding**: Existing dLLM distillation methods are largely heuristic and lack a unified, rigorous theoretical framework.
- **Challenges in discrete spaces**: IKL methods for continuous diffusion models rely on differentiable sampling paths, whereas the discrete state space of dLLMs (involving non-differentiable operations such as argmax) prevents gradients from propagating directly through the sampling trajectory.

**Key Challenge**: How to establish a theoretically sound and practically efficient distillation framework in the discrete token space, enabling a few-step student model to match or even surpass the generation quality of a many-step teacher?

**Key Insight**: The paper transfers the IKL distillation idea from continuous diffusion models to Masked Diffusion Models (MDMs), bypassing the discrete non-differentiability problem via policy gradients, and combining adversarial training to estimate density ratios as reward signals.

## Method

### Overall Architecture

The core of DiDi-Instruct is a teacher–student distillation framework. Given a pretrained dLLM teacher $\mathbf{p}_\theta$, a structurally identical few-step student model $\mathbf{p}_\nu$ is trained to reproduce the teacher's generative distribution with far fewer inference steps.

The overall pipeline proceeds as follows:
1. Starting from a fully masked sequence, the student and teacher independently generate complete texts $\mathbf{x}$ and $\mathbf{x}'$.
2. Both outputs are forward-noised (partially masked) at a randomly sampled timestep $t_i$, yielding $\mathbf{z}_i$ and $\mathbf{z}_i'$.
3. A discriminator $D_\lambda$ is trained to distinguish noised samples from the two sources, outputting a density ratio as a reward signal.
4. The student updates its parameters via policy gradient using the reward signal.
5. At inference time, RGAS (Reward-Guided Ancestral Sampler) is applied to further improve generation quality.

### Key Designs

#### (1) IKL Minimization via Policy Gradient

The central theoretical contribution is the **Score-Function Identity (Theorem 3.1)**, which decomposes the gradient of the IKL objective into a score-function form, avoiding differentiation through discrete sampling paths. Concretely, the gradient is expressed as:

$$\nabla_\nu \mathcal{L}(\nu) = \mathbb{E}_{t,\mathbf{x},\mathbf{z}_t}\left[\frac{\omega(t)}{\pi(t)} \cdot R(\mathbf{z}_t, t) \cdot \nabla_\nu \log \mathbf{p}_\nu(\mathbf{z}_t = \mathbf{m}, t=1)\right]$$

where the reward $R(\mathbf{z}_t, t) = \log \mathbf{q}_\nu(\mathbf{z}_t, t) - \log \mathbf{q}_\theta(\mathbf{z}_t, t)$ is the log density ratio between the student and teacher. This formulation fully recasts the distillation problem within the policy gradient framework, making it naturally compatible with discrete spaces.

#### (2) Adversarial Density Ratio Estimation

Since directly computing the marginal densities of the student and teacher is intractable, an auxiliary discriminator $D_\lambda$ is trained to estimate the density ratio. The discriminator is trained with standard binary cross-entropy:

$$\mathcal{L}_D(\lambda) = -\frac{1}{G}\sum_{i=1}^G \left[\log D_\lambda(\mathbf{z}_i, t_i) + \log(1 - D_\lambda(\mathbf{z}_i', t_i))\right]$$

The logit output of the optimal discriminator naturally corresponds to the density ratio $\log \frac{\mathbf{q}_\nu}{\mathbf{q}_\theta}$, providing a reliable reward signal for policy gradient updates. The discriminator has 131M parameters, is initialized from the teacher backbone, and employs spectral normalization on its classification head.

#### (3) Grouped Reward Normalization

Inspired by GRPO, rewards within each mini-batch are standardized as:

$$\widetilde{R}_i = \frac{R_i - \mu_g}{\sigma_g + \epsilon}$$

This substantially reduces the variance of policy gradient estimates and improves training stability.

#### (4) Score Decomposition

Directly generating a complete sequence from a fully masked input in a single step is prone to mode collapse. The paper proposes decomposing the score function at an intermediate state $\mathbf{z}_i$:

$$\nabla_\nu \log \mathbf{p}_\nu(\mathbf{z}_t=\mathbf{m}, t=1) \approx \nabla_\nu \log \mathcal{P}_\nu(\mathbf{z}_i | \mathbf{z}_t=\mathbf{m}) + \nabla_\nu \log \mathbf{p}_\nu(\mathbf{z}_i, t_i)$$

This exposes the student to the distribution over intermediate noised states, effectively preventing entropy collapse. Ablation studies confirm this to be the most critical component—removing it causes the PPL to explode to 33,584.

#### (5) Reward-Guided Ancestral Sampler (RGAS)

A two-stage inference strategy:
- **Early steps** ($t_n \approx 1$): Gradient tilting ($h > 0, M=1$) is applied, using reward gradients to adjust logits and guide global structure.
- **Late steps** ($t_n \approx 0$): Multi-candidate reranking ($h=0, M>1$) is applied, generating $M$ candidates and performing softmax-weighted sampling according to rewards.

### Loss & Training

- **Parameter initialization**: Both the student and discriminator are initialized from the pretrained teacher model.
- **Discriminator warmup**: Student parameters are frozen while the discriminator is trained independently in the early phase to avoid instability.
- **Gradient/reward clipping**: Applied to prevent explosive updates.
- **Alternating training**: Each training step first updates the discriminator, then the student.
- **Efficient training**: Only 10,000 iterations are required; AdamW optimizer (lr=1e-6); distillation completes in approximately 1 hour on a single H100 GPU.
- **Mixed precision**: bfloat16 is used for acceleration.

## Key Experimental Results

### Main Results: OpenWebText Generation Quality (PPL↓ / Entropy)

| Method | 8 NFEs | 16 NFEs | 32 NFEs | 64 NFEs | 128 NFEs |
|--------|--------|---------|---------|---------|----------|
| GPT-2 (AR) | — | — | — | — | PPL=18.3 |
| MDLM Teacher (1024 steps) | — | — | — | — | PPL=38.5 |
| SDTT | diverges | ~100+ | ~60+ | ~40+ | ~30+ |
| DUO | ~150+ | ~80+ | ~50+ | ~35+ | ~25+ |
| **DiDi-Instruct** | **62.2** | **38.2** | **25.0** | **21.9** | **18.4** |

- 16 steps suffices to surpass the 1024-step teacher model.
- At 128 steps, PPL=18.4 approaches GPT-2's 18.3, representing a 24%+ reduction over the strongest baseline.
- Entropy loss is minimal (~1%), preserving sample diversity almost entirely.

### Ablation Study (169M Model)

| Configuration | 8 NFEs PPL | 16 NFEs PPL | 32 NFEs PPL | 64 NFEs PPL | 128 NFEs PPL |
|--------------|-----------|-------------|-------------|-------------|--------------|
| Baseline (no techniques) | 803.9 | 311.5 | 174.8 | 113.1 | 96.6 |
| + Score Decompose | 667.8 | 289.7 | 165.8 | 105.9 | 89.4 |
| + Coupled Time | 101.0 | 75.2 | 48.4 | 35.8 | 30.6 |
| + ω(t) Correction | 95.0 | 75.6 | 31.7 | 25.3 | 21.0 |
| + π(t) Weighting | 92.1 | 44.0 | 32.3 | 26.1 | 21.4 |
| + Regularization | 88.3 | 44.0 | 28.4 | 21.9 | 18.3 |
| + Guided Inference (full) | **62.2** | **38.2** | **25.0** | **21.9** | **18.4** |

### Model Scale & Efficiency

| Metric | Value |
|--------|-------|
| Teacher model parameters | 169M (DiT, 12L/12H/768D) |
| Student model parameters | 169M (identical architecture) |
| Discriminator parameters | 131M |
| Distillation training time | ~1 H100 GPU hour |
| Competing methods training time | 20+ GPU hours (SDTT/DUO) |
| Inference throughput | 2,366 tokens/sec |
| Speedup over AR model | 13.2× (at matched PPL) |
| 424M model, 16-step PPL | 32.79 (11.4% improvement over 1024-step teacher) |

## Highlights & Insights

1. **Closed theory-practice loop**: The derivation proceeds from continuous diffusion IKL → score-function identity for discrete MDMs → policy gradient → adversarial reward estimation, forming a complete and rigorous theoretical chain. All components are derived from a single unified objective rather than assembled heuristically.

2. **Remarkable training efficiency**: ~1 GPU hour versus 20+ GPU hours for competing methods—a 20× reduction in training cost. This efficiency stems from single-round distillation combined with adversarial training, eliminating the need for multi-round iteration as in DUO.

3. **Breakthrough in few-step generation**: Surpassing the 1024-step teacher with only 16 steps implies 64× inference acceleration with negligible quality loss—a result unprecedented in the dLLM acceleration literature.

4. **Score decomposition is essential**: Ablation experiments reveal that score decomposition is indispensable—removing it causes PPL to spike from 62 to 33,584 (500×+), demonstrating that intermediate-state matching is critical for multi-step distillation.

5. **Cross-domain validation**: The framework is effective not only for natural language but also for protein sequence generation (pLDDT > 70, requiring only 8–32 steps), establishing its generality.

6. **Elegant RGAS design**: The two-phase strategy—gradient tilting in early steps and multi-candidate reranking in late steps—ensures coherent global structure while refining local details, offering greater flexibility than a uniform strategy.

## Limitations & Future Work

1. **Limited model scale**: Experiments cover only 169M and 424M parameters; effectiveness at the billion-parameter scale remains unverified. The authors acknowledge that maintaining three models simultaneously (teacher, student, discriminator) poses a memory bottleneck for scaling.

2. **Validation limited to unconditional generation**: All experiments involve unconditional text generation (OpenWebText); conditional generation tasks such as instruction following, dialogue, and translation are not evaluated, limiting practical applicability.

3. **Inherent risks of adversarial training**: The discriminator–generator adversarial framework may exhibit instability during training; while warmup and clipping mitigate this, reliability at larger scales or on more complex tasks requires further investigation.

4. **Teacher model quality bottleneck**: The baseline teacher MDLM achieves a 1024-step PPL of 38.5, far above GPT-2's 18.3. The student's final PPL approaching 18.4 partly reflects a "student surpassing teacher" distillation effect, but the capability ceiling of the teacher remains a fundamental limitation of dLLMs.

5. **Notable degradation at 8 steps**: At 8 NFEs, PPL=62.2, and generated text exhibits visible repetition, indicating considerable room for improvement in the extreme few-step regime.

## Related Work & Insights

- **Masked Diffusion Models**: MDLM (Sahoo et al., 2024), D3PM (Austin et al., 2021), and SEDD (Lou et al., 2024) establish the foundational framework for discrete diffusion.
- **dLLM Acceleration**: SDTT (temporal self-distillation, Deschenaux & Gulcehre 2025), DUO (consistency distillation, Sahoo et al. 2025), and DSDD (distribution matching, Zhu et al. 2025).
- **Continuous Diffusion Distillation**: Diff-Instruct (Luo et al., 2023b) proposes the IKL framework for continuous diffusion models and serves as the direct theoretical antecedent of this work.
- **Policy Gradient**: REINFORCE (Williams, 1992), PPO (Schulman et al., 2017), and GRPO (Shao et al., 2024) provide the methodological foundations for discrete optimization.
- **Protein Generation**: DPLM (Wang et al., 2024) serves as the baseline for the protein sequence generation experiments.
- **Block Diffusion**: Arriola et al. (2025) explore a hybrid paradigm combining AR and diffusion models.

## Rating

| Dimension | Score (1–10) | Remarks |
|-----------|-------------|---------|
| Novelty | 8 | First successful transfer of IKL distillation to discrete diffusion models; the policy gradient approach to circumventing discrete non-differentiability is elegant. |
| Theoretical Depth | 9 | Complete theoretical derivation chain (IKL → Score-Function Identity → density ratio estimation); appendix proofs are rigorous. |
| Experimental Thoroughness | 8 | Cumulative and leave-one-out ablations, scale experiments, cross-domain validation (proteins), and downstream tasks provide comprehensive coverage. |
| Writing Quality | 8 | Well-structured, coherent mathematical derivations, and an informative pipeline diagram (Figure 2). |
| Value | 7 | Training-efficient and inference-fast, but validated only on unconditional generation at limited model scales. |
| **Overall** | **8.0** | A dLLM acceleration work with solid theoretical contributions and impressive empirical results. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Discrete Diffusion Trajectory Alignment via Stepwise Decomposition](discrete_diffusion_trajectory_alignment_via_stepwise_decomposition.md)
- [\[AAAI 2026\] Hierarchical Schedule Optimization for Fast and Robust Diffusion Model Sampling](../../AAAI2026/medical_imaging/hierarchical_schedule_optimization_for_fast_and_robust_diffusion_model_sampling.md)
- [\[NeurIPS 2025\] Why Masking Diffusion Works: Condition on the Jump Schedule for Improved Discrete Diffusion](../../NeurIPS2025/medical_imaging/why_masking_diffusion_works_condition_on_the_jump_schedule_for_improved_discrete.md)
- [\[NeurIPS 2025\] SpecMER: Fast Protein Generation with K-mer Guided Speculative Decoding](../../NeurIPS2025/medical_imaging/specmer_fast_protein_generation_with_k-mer_guided_speculative_decoding.md)
- [\[ICLR 2026\] Protein Counterfactuals via Diffusion-Guided Latent Optimization](protein_counterfactuals_via_diffusion-guided_latent_optimization.md)

</div>

<!-- RELATED:END -->
