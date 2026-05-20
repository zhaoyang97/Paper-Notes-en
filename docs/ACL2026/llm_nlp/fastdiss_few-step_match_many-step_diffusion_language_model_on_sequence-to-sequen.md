---
title: >-
  [Paper Note] FastDiSS: Few-step Match Many-step Diffusion Language Model on Sequence-to-Sequence Generation
description: >-
  [ACL 2026][LLM/NLP][diffusion language model] This paper analyzes two bottlenecks in continuous diffusion language models under few-step sampling — self-conditioning signal mismatch and training saturation — and proposes…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "diffusion language model"
  - "few-step sampling"
  - "self-conditioning perturbation"
  - "noise scaling"
  - "sequence-to-sequence"
date: 2026-05-08
content_hash: b5884490e7d0fc88
---

# FastDiSS: Few-step Match Many-step Diffusion Language Model on Sequence-to-Sequence Generation

**Conference**: ACL 2026
**arXiv**: [2604.05551](https://arxiv.org/abs/2604.05551)  
**Code**: None  
**Area**: LLM/NLP
**Keywords**: diffusion language model, few-step sampling, self-conditioning perturbation, noise scaling, sequence-to-sequence

## TL;DR
This paper analyzes two bottlenecks in continuous diffusion language models under few-step sampling — self-conditioning signal mismatch and training saturation — and proposes the FastDiSS framework, which introduces Self-Conditioning Perturbation (SCP) and Model-Aware Noise Scaling (MANS) to improve robustness, achieving 4×–400× speedup while preserving generation quality across 6 benchmarks.

## Background & Motivation

**Background**: Diffusion models serve as an alternative to autoregressive text generation, enabling parallel decoding of all tokens in linear time. Self-conditioning reuses predictions from the previous step as conditioning signals to improve few-step sampling, but introduces failure modes that have not been sufficiently studied.

**Limitations of Prior Work**: (1) *Training–inference self-conditioning mismatch* — during training, ground-truth targets are available as conditioning signals, whereas at inference only imperfect self-predictions are available; this distributional shift is exacerbated in few-step settings, where predictions at high-noise steps differ substantially from those at low-noise steps, making the reused signal a biased condition. (2) *Late-stage training saturation* — after the model rapidly fits early-stage targets, a pronounced loss plateau emerges, and uniform noise sampling provides no effective learning signal for tokens already predicted with high confidence.

**Key Challenge**: The deployment appeal of diffusion models lies precisely in fast few-step inference, yet self-conditioning — the key technique for improving few-step sampling — introduces the largest errors under that very setting.

**Goal**: Design a training framework that enables diffusion language models to achieve quality close to many-step sampling when using few-step inference.

**Key Insight**: Directly simulate inference-time noise conditions during training — by perturbing the self-conditioning signal to match the error distribution seen at inference, and by dynamically adjusting per-token noise levels to avoid training saturation.

**Core Idea**: SCP intentionally uses noisier estimates as self-conditioning signals during training, while MANS dynamically assigns higher noise to high-confidence tokens based on denoising confidence.

## Method

### Overall Architecture
FastDiSS introduces two complementary components into the training of standard continuous diffusion language models: (1) SCP generates weaker self-conditioning estimates by running the denoising network at a higher noise level; (2) MANS dynamically adjusts the noise level for each token based on the model's current denoising confidence. The two components jointly address self-conditioning mismatch and training saturation.

### Key Designs

1. **Self-Conditioning Perturbation (SCP)**:

    - **Function**: Reduces training–inference distributional shift by introducing noise conditions during training that match inference-time errors.
    - **Mechanism**: Rather than running the denoising network at the current noise level $t$ to obtain the self-conditioning signal, SCP runs it at a higher noise level $t' > t$, producing a weaker and noisier estimate. This simulates the imperfect estimate propagated from an earlier, higher-noise step at inference time. The network is then trained to denoise accurately even under this perturbed conditioning signal.
    - **Design Motivation**: At inference, the self-conditioning signal originates from an earlier step with higher noise, resulting in a distributional discrepancy from training. SCP addresses this by simulating such imperfections during training, enabling the model to operate robustly under noisy conditioning signals.

2. **Model-Aware Noise Scaling (MANS)**:

    - **Function**: Dynamically adjusts the noise level for each token according to its denoising confidence, preventing training saturation.
    - **Mechanism**: For each token $i$, the model computes a prediction confidence score (e.g., distance to the ground-truth embedding), and applies increased noise to high-confidence tokens. Specifically, the timestep for each token is dynamically adjusted based on current model predictions, so that "easy" tokens face a higher noise challenge.
    - **Design Motivation**: Uniform noise sampling causes a large fraction of training signal to be wasted on "easy" tokens that the model has already mastered. MANS focuses training signal on informative examples while also improving the quality of self-conditioning estimates in high-noise regions.

3. **End-to-End Training Framework**:

    - **Function**: Integrates SCP and MANS into the standard diffusion training pipeline while maintaining training stability.
    - **Mechanism**: Given a sampled timestep $t$, MANS produces an adjusted timestep $t_\theta$; SCP then obtains a perturbed self-conditioning signal at noise level $t_\theta$; finally, the model is trained with the standard diffusion loss. The two components can be applied independently or jointly.
    - **Design Motivation**: SCP and MANS address distinct bottlenecks yet are mutually reinforcing — MANS improves the quality of estimates in high-noise regions, which in turn enhances the quality of the perturbation signal used by SCP.

### Loss & Training
The training objective follows the standard diffusion formulation: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diffusion}} + \mathcal{L}_{\text{round}}$, combined with SCP and MANS. Training alternates between optimizing the diffusion loss and the self-conditioning loss.

## Key Experimental Results

### Main Results

| Setting | Model | 5-step BLEU | Speedup |
|---------|-------|-------------|---------|
| IWSLT14 De-En | Standard Diffusion | 27.85 | 1× |
| IWSLT14 De-En | FastDiSS | **29.70** | 200×–400× |
| Oracle Self-Conditioning Upper Bound | — | 29.70 | — |

### Ablation Study

| Configuration | 5-step BLEU | Note |
|---------------|-------------|------|
| Standard Self-Conditioning | 27.85 | Baseline |
| + SCP only | 29.1+ | Reduces training–inference mismatch |
| + MANS only | 28.5+ | Prevents training saturation |
| + SCP + MANS | **29.70** | Best with both components |

### Key Findings
- Self-conditioning mismatch causes approximately 2 BLEU degradation under 5-step sampling; FastDiSS nearly closes this gap entirely.
- SCP brings few-step sampling quality close to the theoretical upper bound achieved with oracle self-conditioning.
- MANS's token-level noise adjustment outperforms uniform noise sampling and effectively eliminates late-stage training saturation.
- Consistent improvements are observed across 6 seq2seq benchmarks, including translation and summarization tasks.
- FastDiSS remains competitive against other single-step diffusion frameworks.

## Highlights & Insights
- **Simulating inference errors during training**: The core idea of SCP — deliberately introducing inference-time imperfections during training to improve robustness — generalizes to any scenario with training–inference discrepancy (e.g., teacher forcing vs. autoregressive inference).
- **Difficulty-aware training**: MANS dynamically increases noise for "easy" tokens, representing a natural application of curriculum learning and hard example mining to diffusion models.
- **Analysis-driven design**: By precisely quantifying the performance gap between oracle and reused self-conditioning, the paper identifies the severity of the problem and designs targeted solutions accordingly.

## Limitations & Future Work
- Validation is limited to continuous diffusion language models; discrete diffusion models are not evaluated.
- The 6 benchmarks are predominantly translation and summarization tasks; more complex generation tasks are not assessed.
- The noise level selection in SCP may require task-specific tuning.
- The absolute generation quality of diffusion language models still lags behind state-of-the-art autoregressive LLMs.

## Related Work & Insights
- **vs. DiffusionLM**: DiffusionLM establishes the foundational framework for continuous diffusion language modeling; FastDiSS addresses the efficiency bottleneck of few-step sampling within this paradigm.
- **vs. CDCD**: CDCD introduces self-conditioning to accelerate diffusion; FastDiSS resolves the new problems that self-conditioning introduces under few-step settings.
- **vs. one-step diffusion methods**: FastDiSS's few-step strategy offers a more flexible quality–efficiency trade-off compared to single-step approaches.

## Rating
- Novelty: ⭐⭐⭐⭐ The "simulate inference errors during training" idea underlying SCP is concise and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ 6 benchmarks, detailed ablations, and comparisons across multiple step counts.
- Writing Quality: ⭐⭐⭐⭐ Problem analysis is thorough and the method is clearly described.
- Value: ⭐⭐⭐⭐ Removes a critical efficiency barrier to the practical deployment of diffusion language models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Model](../../ICLR2026/llm_nlp/fs-dfm_fast_and_accurate_long_text_generation_with_few-step_diffusion_language_m.md)
- [\[AAAI 2026\] TransMamba: A Sequence-Level Hybrid Transformer-Mamba Language Model](../../AAAI2026/llm_nlp/transmamba_a_sequence-level_hybrid_transformer-mamba_language_model.md)
- [\[ACL 2026\] Automatic Combination of Sample Selection Strategies for Few-Shot Learning](automatic_combination_of_sample_selection_strategies_for_few-shot_learning.md)
- [\[ACL 2026\] One Persona, Many Cues, Different Results: How Sociodemographic Cues Impact LLM Personalization](one_persona_many_cues_different_results_how_sociodemographic_cues_impact_llm_per.md)
- [\[ACL 2026\] Think in Sentences: Explicit Sentence Boundaries Enhance Language Model's Capabilities](think_in_sentences_explicit_sentence_boundaries_enhance_language_model39s_capabi.md)

</div>

<!-- RELATED:END -->
