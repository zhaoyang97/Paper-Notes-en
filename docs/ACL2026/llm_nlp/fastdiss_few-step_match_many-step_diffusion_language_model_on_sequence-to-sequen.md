---
title: >-
  [Paper Note] FastDiSS: Few-step Match Many-step Diffusion Language Model on Sequence-to-Sequence Generation
description: >-
  [ACL 2026][LLM/NLP][Diffusion Language Models] This paper analyzes two bottlenecks of continuous diffusion language models in few-step sampling: the mismatch of self-conditioning signals and training saturation. It propo…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "Diffusion Language Models"
  - "Few-step Sampling"
  - "Self-conditioning Perturbation"
  - "Noise Scaling"
  - "Sequence-to-Sequence"
date: 2026-05-08
content_hash: 7e14953e9ff60948
---

# FastDiSS: Few-step Match Many-step Diffusion Language Model on Sequence-to-Sequence Generation

**Conference**: ACL 2026  
**arXiv**: [2604.05551](https://arxiv.org/abs/2604.05551)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Diffusion Language Models, Few-step Sampling, Self-conditioning Perturbation, Noise Scaling, Sequence-to-Sequence

## TL;DR
This paper analyzes two bottlenecks of continuous diffusion language models in few-step sampling: the mismatch of self-conditioning signals and training saturation. It proposes the FastDiSS framework, which improves robustness through Self-Conditioning Perturbation (SCP) and Model-Aware Noise Scaling (MANS), achieving 4×-400× speedup across 6 benchmarks while maintaining quality.

## Background & Motivation

**Background**: As an alternative to autoregressive text generation, diffusion models achieve linear-time decoding by generating all tokens in parallel. Self-conditioning techniques improve few-step sampling by reusing predictions from previous steps as conditioning signals, but carry unrecognized failure modes.

**Limitations of Prior Work**: (1) Training-inference self-conditioning mismatch—ground truth targets are available during training, but only imperfect self-predictions can be used during inference. This distribution shift is more severe in few-step settings, where predictions at high-noise steps differ significantly from those at low-noise steps, causing reused signals to become biased conditions. (2) Late-stage training saturation—models encounter a distinct loss plateau after quickly fitting early targets, as uniform noise sampling fails to provide effective learning signals for tokens already predicted with high confidence.

**Key Challenge**: The deployment appeal of diffusion models lies precisely in few-step fast inference, yet self-conditioning—the key technique for improving few-step sampling—introduces the largest errors in those very settings.

**Goal**: Design a training framework that enables diffusion language models to achieve quality in few-step sampling close to that of many-step sampling.

**Key Insight**: Directly simulate inference-time noise conditions during training by perturbing self-conditioning signals to match the inference error distribution, and avoid training saturation by dynamically adjusting noise for each token.

**Core Idea**: SCP deliberately uses noisier estimates as self-conditioning signals during training, while MANS dynamically assigns higher noise to high-confidence tokens based on denoising confidence.

## Method

### Overall Architecture
FastDiSS introduces two complementary components into the training of standard continuous diffusion language models: (1) SCP generates weaker self-conditioning estimates by running the denoising network at higher noise levels; (2) MANS dynamically adjusts the noise level for each token based on the model's current denoising confidence. Together, they resolve self-conditioning mismatch and training saturation.

### Key Designs

1. **Self-Conditioning Perturbation (SCP)**:

    - Function: Reduces training-inference distribution shift by introducing noise conditions during training that match inference-time errors.
    - Mechanism: When obtaining the self-conditioning signal during training, the denoising network is not run at the current noise level $t$, but at a higher noise level $t' > t$ to produce a weaker, noisier estimate. This simulates the imperfect estimate passed from the previous (higher noise) step during inference. The network is then trained to denoise accurately even under this perturbed conditioning signal.
    - Design Motivation: Inference-time self-conditioning signals originate from earlier, higher-noise step estimates, which differ from the training distribution. SCP encourages the model to operate robustly under noisy conditioning signals by simulating this imperfection during training.

2. **Model-Aware Noise Scaling (MANS)**:

    - Function: Dynamically adjusts noise levels based on the denoising confidence of each token to avoid training saturation.
    - Mechanism: For each token $i$, the model's prediction confidence is calculated (distance to the ground truth embedding), and noise is increased for high-confidence tokens. Specifically, the timestep for each token is dynamically adjusted according to the model's current prediction, ensuring that "easy" tokens face higher noise challenges.
    - Design Motivation: Uniform noise sampling leads to a significant amount of training signal being wasted on "easy" tokens already mastered by the model. MANS allows the model to focus on signals with learning value while also improving the quality of self-conditioning estimates in high-noise regions.

3. **End-to-End Training Framework**:

    - Function: Integrates SCP and MANS into the standard diffusion training pipeline while maintaining training stability.
    - Mechanism: A timestep $t$ is sampled first, then the adjusted timestep $t_\theta$ is obtained via MANS. A perturbed self-conditioning signal is acquired at the $t_\theta$ noise level via SCP, and finally, the model is trained with the standard diffusion loss. Both components can be used independently or jointly.
    - Design Motivation: SCP and MANS address different bottlenecks but enhance each other—MANS improves estimate quality in high-noise regions, indirectly enhancing the quality of SCP's perturbation signals.

### Loss & Training
The standard diffusion objective $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diffusion}} + \mathcal{L}_{\text{round}}$ is used, combined with SCP and MANS. Training alternates between optimizing the diffusion loss and the self-conditioning loss.

## Key Experimental Results

### Main Results

| Setting | Model | 5-step BLEU | Speedup |
|------|------|---------|---------|
| IWSLT14 De-En | Standard Diffusion | 27.85 | 1× |
| IWSLT14 De-En | FastDiSS | **29.70** | 200×-400× |
| Oracle Self-Cond. Upper Bound | — | 29.70 | — |

### Ablation Study

| Configuration | 5-step BLEU | Description |
|------|---------|------|
| Standard Self-Cond. | 27.85 | Baseline |
| + SCP only | 29.1+ | Reduces training-inference mismatch |
| + MANS only | 28.5+ | Avoids training saturation |
| + SCP + MANS | **29.70** | Optimal synergy between both |

### Key Findings
- Self-conditioning mismatch causes a loss of approximately 2 BLEU during 5-step sampling; FastDiSS almost entirely closes this gap.
- SCP enables few-step sampling quality to approach the theoretical upper bound of using "correct" self-conditioning.
- Token-level noise adjustment in MANS is more effective than uniform noise sampling, preventing late-stage training saturation.
- Consistent improvements are observed across 6 seq2seq benchmarks, including translation and summarization tasks.
- Competitiveness is maintained even compared to other single-step diffusion frameworks.

## Highlights & Insights
- **Simulating Inference Error during Training**: The core idea of SCP—deliberately introducing inference-time imperfections during training to enhance robustness—can be generalized to any scenario with training-inference mismatch (e.g., teacher forcing vs. autoregressive inference).
- **Hard-Example Aware Training**: MANS dynamically increasing noise for "easy" tokens is a natural application of curriculum learning and hard example mining ideas within diffusion models.
- **Analysis-Driven Design**: By comparing the performance gap between "oracle" and "reused" self-conditioning, the paper precisely quantifies the severity of the problem before designing targeted solutions.

## Limitations & Future Work
- Validated only on continuous diffusion language models; discrete diffusion models were not tested.
- The 6 benchmarks are primarily translation and summarization tasks; more complex generation tasks were not tested.
- The choice of noise levels for SCP may require tuning for different tasks.
- Compared to the latest autoregressive LLMs, the absolute quality of diffusion language models still shows a gap.

## Related Work & Insights
- **vs. DiffusionLM**: DiffusionLM defines the basic framework for continuous diffusion language modeling; FastDiSS addresses its efficiency bottlenecks in few-step sampling.
- **vs. CDCD**: CDCD introduced self-conditioning to accelerate diffusion; FastDiSS solves the new problems introduced by self-conditioning in few-step settings.
- **vs. Single-step Diffusion Methods**: The few-step strategy of FastDiSS provides a more flexible trade-off between quality and efficiency.

## Rating
- Novelty: ⭐⭐⭐⭐ The "simulating inference error during training" approach in SCP is simple and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ 6 benchmarks, detailed ablations, and comparisons across multiple step counts.
- Writing Quality: ⭐⭐⭐⭐ Thorough problem analysis and clear methodological description.
- Value: ⭐⭐⭐⭐ Removes key efficiency barriers for the practical deployment of diffusion language models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] TransMamba: A Sequence-Level Hybrid Transformer-Mamba Language Model](../../AAAI2026/llm_nlp/transmamba_a_sequence-level_hybrid_transformer-mamba_language_model.md)
- [\[ACL 2026\] Automatic Combination of Sample Selection Strategies for Few-Shot Learning](automatic_combination_of_sample_selection_strategies_for_few-shot_learning.md)
- [\[ACL 2026\] Unlocking the Potential of Diffusion Language Models through Template Infilling](unlocking_the_potential_of_diffusion_language_models_through_template_infilling.md)
- [\[AAAI 2026\] Uncertainty Under the Curve: A Sequence-Level Entropy Area Metric for Reasoning LLMs](../../AAAI2026/llm_nlp/uncertainty_under_the_curve_a_sequence-level_entropy_area_metric_for_reasoning_l.md)
- [\[ACL 2026\] Leveraging Pretrained Language Models as Energy Functions for Glauber Dynamics Text Diffusion](leveraging_pretrained_language_models_as_energy_functions_for_glauber_dynamics_t.md)

</div>

<!-- RELATED:END -->
