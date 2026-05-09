---
title: >-
  [Paper Note] Toward Safer Diffusion Language Models: Discovery and Mitigation of Priming Vulnerabilities
description: >-
  [ICLR 2026][LLM/NLP][diffusion language models] This paper identifies a *priming vulnerability* in masked diffusion language models (MDLMs)—injecting affirmative tokens at intermediate denoising steps can bypass safety guardrails—and proposes Recovery Alignment (RA), a training method that teaches models to recover safe responses from corrupted intermediate states.
tags:
  - ICLR 2026
  - LLM/NLP
  - diffusion language models
  - jailbreak attacks
  - priming vulnerability
  - safety alignment
  - masked diffusion
date: 2026-05-08
content_hash: 7d42a49b90488f8c
---

# Toward Safer Diffusion Language Models: Discovery and Mitigation of Priming Vulnerabilities

**Conference**: ICLR 2026
**arXiv**: [2510.00565](https://arxiv.org/abs/2510.00565)
**Code**: [GitHub](https://github.com/mdl-lab/dlm-priming-vulnerability)
**Area**: AI Safety / Diffusion Language Models
**Keywords**: diffusion language models, jailbreak attacks, priming vulnerability, safety alignment, masked diffusion

## TL;DR

This paper identifies a *priming vulnerability* in masked diffusion language models (MDLMs)—injecting affirmative tokens at intermediate denoising steps can bypass safety guardrails—and proposes Recovery Alignment (RA), a training method that teaches models to recover safe responses from corrupted intermediate states.

## Background & Motivation

**Background**: Masked diffusion language models (MDLMs) such as LLaDA and MMaDA generate tokens in parallel via iterative denoising, emerging as alternatives to autoregressive models (ARMs) with lower latency and bidirectional context modeling. However, safety risks in MDLMs remain almost entirely unstudied.

**Limitations of Prior Work**: Existing safety alignment methods (SFT, DPO, MOSA) assume denoising begins from a fully masked sequence, training models to produce safe responses only under this condition. When affirmative tokens appear at intermediate denoising steps, models cannot recover safe outputs from such corrupted states. The parallel iterative generation mechanism unique to MDLMs exposes them to safety threats fundamentally different from those facing ARMs.

**Key Challenge**: A distribution shift exists between the initialization condition assumed during safety alignment training (fully masked sequences) and the intermediate states that may arise at inference time (states containing harmful tokens). Models never encounter corrupted intermediate states during training and therefore never learn to "recover" from them.

**Goal**: (1) Systematically quantify the severity of the priming vulnerability in MDLMs; (2) Design an MDLM-specific safety alignment method to mitigate this vulnerability.

**Key Insight**: Analysis of the iterative denoising mechanism reveals that injecting a single affirmative token at the very first step raises the attack success rate (ASR) from 2% to 21%. Recovery Alignment is designed to train models to restore safe outputs from corrupted states.

**Core Idea**: During training, intentionally construct intermediate states containing harmful tokens so that models learn to recover from "poisoned" states back to safe responses.

## Method

### Overall Architecture

The paper is organized into two parts: (1) vulnerability discovery and quantification—two threat-model-based attacks are designed to systematically expose the priming vulnerability; (2) Recovery Alignment—an MDLM-specific safety alignment training method is proposed.

### Key Designs

**Design 1: Anchoring Attack**
- **Function**: At intermediate denoising step $t_{inter}$, replace model predictions with harmful responses to quantify the degree to which subsequent generation is biased toward harmful content.
- **Mechanism**: Assuming an attacker can intervene in the denoising process, predicted tokens at a designated step are replaced with tokens from a harmful response, after which denoising continues. Replaced tokens are partially retained after re-masking, serving as "anchors" that guide subsequent generation.
- **Design Motivation**: Provides a systematic and controllable method for vulnerability quantification. Experiments show that injecting a single token at the first step ($t_{inter}=1$) raises ASR from 2% to 21%.

**Design 2: First-Step GCG Attack**
- **Function**: Design an optimization-based attack that does not require intervention in the denoising process.
- **Mechanism**: The priming vulnerability is exploited to derive a tractable lower bound—maximizing the first-step log-likelihood as a surrogate objective. It is proven that $\log p_{\pi,m_t}(\mathbf{r}_T|\mathbf{q},\mathbf{r}_0) \geq \frac{1}{T}\log\pi_\theta(\tilde{\mathbf{r}}_1|\mathbf{q},\mathbf{r}_0)$.
- **Design Motivation**: Conventional GCG requires Monte Carlo estimation of gradients through stochastic re-masking, which is high-variance and computationally expensive. First-Step GCG is approximately 20× faster than MC GCG and achieves up to 4× higher ASR.

**Design 3: Recovery Alignment (RA)**
- **Function**: Train models to recover safe responses from corrupted intermediate states.
- **Mechanism**: Given a harmful query-response pair $(q, r)$, a corrupted state $r_{t_{inter}} \sim m_{t_{inter}}(\cdot|r)$ is constructed at training step $t_{inter}$. The model denoises from this state, a reward model evaluates the safety of the output, and GRPO is used for optimization.
- **Design Motivation**: Conventional alignment only teaches models to generate safe content starting from fully masked sequences and cannot handle corrupted intermediate states. RA explicitly models corrupted states, enabling models to learn "recovery paths."

**Design 4: Linear Scheduling Strategy**
- **Function**: Linearly increase the intervention step $t_{inter}$ over the course of training.
- **Mechanism**: $t_{inter} = \lfloor t_{min} + \frac{s}{S}(t_{max} - t_{min}) \rfloor$, progressively increasing difficulty.
- **Design Motivation**: Training directly with large $t_{inter}$ is unstable (the model must recover safety within very few steps). The curriculum learning strategy enables the model to gradually handle stronger corruption.

### Loss & Training

- GRPO is used to optimize the RA objective.
- Reward model: DeBERTaV3 (no additional fine-tuning required).
- Training data: Harmful query-response pairs from the BeaverTails dataset (no additional data construction cost).
- Training steps: Only 2,500 steps.
- Intervention range: Linear scheduling over $[t_{min}, t_{max}]$.

## Key Experimental Results

### Main Results

**Priming Vulnerability Attack Results (JBB-Behaviors, GPT-4o evaluation, ASR %)**

| Method | No Attack | Anchoring t=1 | Anchoring t=16 | First-Step GCG |
|--------|-----------|---------------|----------------|----------------|
| LLaDA Original | 2.0 | 17.3 | 88.7 | 58.0 |
| LLaDA + SFT | 8.3 | 19.0 | 87.7 | 48.2 |
| LLaDA + DPO | 4.3 | — | — | — |
| **LLaDA + RA** | **Significantly reduced** | **Significantly reduced** | **Significantly reduced** | **Significantly reduced** |

**First-Step GCG vs. Monte Carlo GCG**

| Method | LLaDA ASR% | LLaDA 1.5 ASR% | Time per prompt |
|--------|-----------|----------------|-----------------|
| Monte Carlo GCG | 20.0 | 12.5 | 4.1–4.3 h |
| First-Step GCG | 58.0 | 49.5 | 0.2 h |

First-Step GCG is approximately 20× faster and 3–4× more effective than Monte Carlo GCG.

### Ablation Study

| Component Ablation | Effect |
|--------------------|--------|
| RA w/o intervention (t=0) | Equivalent to standard RLHF; fails to mitigate priming vulnerability |
| Fixed $t_{inter}$ | Unstable training |
| Linear scheduling of $t_{inter}$ | Stable training with improved robustness |
| Different models (LLaDA / LLaDA 1.5 / MMaDA) | RA is effective across all models |

### Key Findings

1. **Priming vulnerability is universal**: The vulnerability is observed across all three models—LLaDA Instruct, LLaDA 1.5, and MMaDA MixCoT.
2. **Extreme sensitivity**: Injecting a single token at the first step significantly raises ASR (e.g., LLaDA from 2% to 21%).
3. **Existing defenses are ineffective**: SFT, DPO, and MOSA all fail to meaningfully mitigate the priming vulnerability.
4. **RA also enhances general robustness**: Beyond mitigating the priming vulnerability, RA demonstrates stronger defense against conventional jailbreak attacks (PAIR, ReNeLLM, Crescendo).
5. **No degradation of general capability**: RA shows no notable performance degradation across 11 general benchmarks.

## Highlights & Insights

- **First systematic exposure of a unique MDLM safety vulnerability**: Mechanistically distinct from ARM prefilling attacks, this is a novel vulnerability class arising from the iterative denoising mechanism.
- **Theoretical contribution**: Theorem 4.1 proves that the first-step log-likelihood is a lower bound on the likelihood of the entire denoising process, providing a theoretical foundation for designing efficient attacks.
- **Simple and practical method**: RA requires no additional data construction, relying only on existing harmful datasets and a pretrained reward model, with training completed in 2,500 steps.
- **Safety and capability are not at odds**: RA enhances safety without sacrificing general model capability.
- **Forward-looking contribution**: As MDLMs move toward practical deployment, this work lays a foundation for MDLM safety research.

## Limitations & Future Work

1. **Evaluation limited to MDLMs**: Generalizability to continuous diffusion language models remains unknown.
2. **Strength of attack assumptions**: The anchoring attack assumes adversarial access to the denoising process, which is difficult to achieve in practical deployments (though First-Step GCG does not require this assumption).
3. **Reward model dependency**: The effectiveness of RA is bounded by the quality of the reward model (DeBERTaV3).
4. **Fixed generation length**: Experiments use $L=T=128$; effectiveness for longer generations requires further validation.
5. **Adaptive attacks**: Whether adaptive attacks specifically targeting RA can be developed warrants further investigation.

## Related Work & Insights

- **Relationship to ARM safety research**: ARM prefilling attacks exploit autoregressive prefixes to suppress subsequent refusals; MDLM priming vulnerability exploits intermediate denoising steps to guide subsequent generation—the mechanisms are fundamentally different.
- **Distinction from MOSA** (Xie et al., 2025): MOSA trains safety alignment only from fully masked states and cannot handle corrupted intermediate states.
- **Implications for MDLM deployment**: Safety of the denoising process must be considered when deploying MDLMs; ARM safety solutions cannot be directly transferred.
- **Inspiration for adversarial robustness research**: The concept underlying Recovery Alignment—training models to recover from adversarial states—may generalize to other generative models.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First discovery and systematic quantification of a unique MDLM safety vulnerability, opening a new research direction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three models, two datasets, three evaluators, and multiple attack types and baselines; generation length is fixed.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Problem statement is clear, analysis is thorough, and theory and experiments are tightly integrated.
- **Value**: ⭐⭐⭐⭐⭐ — As MDLMs gradually enter practical use, the safety findings and defense methods presented here carry high practical value.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] DreamOn: Diffusion Language Models For Code Infilling Beyond Fixed-size Canvas](dreamon_diffusion_language_models_for_code_infilling_beyond_fixed-size_canvas.md)
- [\[ACL 2026\] Style Amnesia: Investigating Speaking Style Degradation and Mitigation in Multi-Turn Spoken Language Models](../../ACL2026/llm_nlp/style_amnesia_investigating_speaking_style_degradation_and_mitigation_in_multi-t.md)
- [\[ICLR 2026\] FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Model](fs-dfm_fast_and_accurate_long_text_generation_with_few-step_diffusion_language_m.md)
- [\[ICLR 2026\] LLEMA: Evolutionary Search with LLMs for Multi-Objective Materials Discovery](llema_evolutionary_search_with_llms_for_multi-objective_material_design.md)
- [\[ICLR 2026\] d²Cache: Accelerating Diffusion-Based LLMs via Dual Adaptive Caching](d2cache_accelerating_diffusion-based_llms_via_dual_adaptive_caching.md)

<!-- RELATED:END -->
