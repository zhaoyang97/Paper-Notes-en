---
title: >-
  [Paper Note] Teaching an Old LLM Secure Coding: Localized Preference Optimization on Distilled Preferences
description: >-
  [ACL 2025][LLM Alignment][secure code generation] Proposes DiSCo (a secure code preference dataset distilled from frontier LLMs, with 10K instances covering 431 CWEs) and LPO (Localized Preference Optimization algorithm, propagating loss only on security-related tokens), reducing security vulnerabilities by 19-40% across four secure coding benchmarks while improving code quality by 3-10%.
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "secure code generation"
  - "preference optimization"
  - "localized alignment"
  - "CWE"
  - "code security"
date: 2026-05-08
content_hash: f799b59b3dca2874
---

# Teaching an Old LLM Secure Coding: Localized Preference Optimization on Distilled Preferences

**Conference**: ACL 2025  
**arXiv**: [2506.00419](https://arxiv.org/abs/2506.00419)  
**Code**: [https://github.com/StonyBrookNLP/disco-lpo](https://github.com/StonyBrookNLP/disco-lpo)  
**Area**: Alignment RLHF  
**Keywords**: secure code generation, preference optimization, localized alignment, CWE, code security

## TL;DR
Proposes DiSCo (a secure code preference dataset distilled from frontier LLMs, with 10K instances covering 431 CWEs) and LPO (Localized Preference Optimization algorithm, propagating loss only on security-related tokens), reducing security vulnerabilities by 19-40% across four secure coding benchmarks while improving code quality by 3-10%.

## Background & Motivation

**Background**: LLMs are widely used for programming assistance (GitHub Copilot has over 1.2 million subscribers, with 92% of developers using AI coding), but studies show that 40-76% of AI-generated code contains security vulnerabilities (vulnerabilities under the CWE classification).

**Limitations of Prior Work**: (1) High-quality secure training data is difficult to acquire—data automatically extracted from open-source repositories is highly noisy and has narrow CWE coverage. (2) Standard preference optimization (DPO/SimPO) is unsuitable for secure code—differences between secure and insecure code are typically localized to a few code lines/tokens, whereas standard methods propagate loss uniformly across all tokens, diluting critical signals.

**Key Challenge**: The difference between secure and insecure code is highly localized (potentially differing by only a few tokens), yet existing preference optimization methods cannot exploit this localization.

**Goal**: (1) Large-scale, high-quality secure code training data. (2) Alignment algorithms specifically designed for localized preferences.

**Key Insight**: Guiding frontier LLM data distillation using a security knowledge base (ensuring CWE coverage), cleaning data with static analyzers (reducing noise), and designing token-level masked preference optimization.

**Core Idea**: Constructing broad-coverage data via a security knowledge-base-guided distillation pipeline + localized preference optimization focusing on secure tokens using masks + SFT regularization to maintain code quality.

## Method

### Overall Architecture
Two-stage training: (1) SFT stage—trains the model to generate secure reasoning $R$ + secure code $y^+$. (2) LPO stage—performs preference optimization on security-related tokens to make the model prefer $y^+$ over $y^-$, while applying SFT regularization on other tokens.

### Key Designs

1. **DiSCo Data Distillation**:

    - **Function**: Distilling 10K secure/insecure code pairs and secure reasoning from GPT-4o.
    - **Mechanism**: (1) Constructing prompts using a security knowledge base (534 entries from the CWE website + CodeQL/Bandit documentation + 75 common security libraries) to guide GPT-4o to first generate code with specific vulnerabilities and then patch it. (2) Using static security analyzers to detect remnant issues and feedback to GPT-4o for refinement. One round of refinement reduces secure problems from 37.4% to 12.7%.
    - **Design Motivation**: Direct prompting of LLMs only generates common CWEs; guiding with a knowledge base ensures broad coverage (431 CWE types). Secure reasoning $R$ forces the model to think about potential issues before coding.

2. **SFT + Secure Reasoning**:

    - **Function**: Training the model to generate secure reasoning $R$ (CWE-ID + problem description + insecure reasons + secure fix) before generating secure code.
    - **Mechanism**: $\mathcal{L}_{SFT} = -\mathbb{E}_{(x, y^+, R) \sim D} \log \pi_\theta(y^+, R | x)$
    - **Design Motivation**: "Reasoning before coding" enables the model to consider security concerns before code generation, similar to the CoT approach.

3. **Localized Preference Optimization (LPO)**:

    - **Function**: Propagating loss only on security-related tokens during preference optimization, while utilizing SFT regularization on other tokens.
    - **Mechanism**: Constructing binary masks $m^+, m^-$ to identify differing tokens in secure/insecure code (computed via difflib). The reasoning section $R$ is masked (as it is identical in both). LPO loss = localized preference term $\Delta$ + SFT regularization term: $\mathcal{L}_{LPO} = -\mathbb{E}[\log\sigma(\Delta - \gamma) + \alpha \bar{m}^+ \odot \log\pi_\theta(y^+, R|x)]$
    - **Design Motivation**: Standard SimPO applies losses uniformly across all tokens, which severely dilutes signals of safety-critical tokens with a large volume of non-security tokens. LPO focuses on key differences. SFT regularization prevents the model from generating unparseable/incoherent code for the sake of "security".

### Loss & Training
Localized preference term $\Delta = \frac{\beta}{|y^+|} m^+ \odot \log\pi_\theta(y^+,R|x) - \frac{\beta}{|y^-|} m^- \odot \log\pi_\theta(y^-,R|x)$, plus SFT regularization $\alpha \bar{m}^+ \odot \log\pi_\theta(y^+, R|x)$.

## Key Experimental Results

### Main Results

| Model + Method | SecurityEval↓ | Asleep↓ | LLMSecEval↓ | HumanEvalX↑ |
|------|:---:|:---:|:---:|:---:|
| CodeLlama-7B (Original) | High Insecurity Rate | High | High | Baseline |
| + SFT on DiSCo | Reduced | Reduced | Reduced | +3-10% |
| + SimPO on DiSCo | Slightly Reduced | Slight | Slight | Decreased |
| **+ LPO on DiSCo** | **-19~40%** | **-19~40%** | **-19~40%** | **+3-10%** |

**Key Findings/Results**: LPO reduces security issues by 19-40% across four security benchmarks while improving performance on two code quality benchmarks by 3-10%. Smaller models trained with LPO even outperform GPT-4o and Claude-3.5-Sonnet in terms of security.

### Ablation Study

| Configuration | Security | Code Quality | Explanation |
|------|:---:|:---:|------|
| LPO (Full) | Optimal | Maintained/Improved | Balances safety and quality |
| LPO w/o SFT Regularization | Safer | Decreased | Overfitting to safety makes code unusable |
| SimPO (Standard) | Limited Improvement | Decreased | Unable to focus on localized differences |
| SFT w/o Secure Reasoning | Moderate | Improved | Reasoning benefits security |
| Multi-round Refinement (3 times) | Insecurity rate reduced to 9.4% | Quality degraded | Over-engineered |

### Key Findings
- **SFT Regularization is Crucial**: Without regularization, the model hacks the reward—generating unparseable code (which the analyzer deems "secure" because it cannot detect vulnerabilities), but is practically unusable.
- **Secure Reasoning Chain is Effective**: Prompting the model to output CWE analysis before coding yields a significant safety boost compared to direct coding.
- **Small Models Outperform Large Models**: The 7B model trained with LPO outperforms GPT-4o and Claude-3.5-Sonnet in security.
- **Single-round Refinement is Optimal**: Excessive refinement leads to over-engineered code, making one round of refinement the optimal balance between safety and quality.

## Highlights & Insights
- **Generality of Localized Preference Optimization**: The masking mechanism of LPO can be generalized to any scenario with "localized preference differences" (such as code formatting, specific-style writing, etc.).
- **Knowledge-Base-Guided Distillation**: Constructing distillation prompts using domain knowledge bases to ensure data coverage is a transferable paradigm to other domains requiring broad coverage (such as regulatory compliance, medical safety, etc.).
- **SFT Regularization Prevents Reward Hacking**: In safety alignment, models may generate "technically safe but unusable" outputs. SFT regularization offers a concise solution.

## Limitations & Future Work
- Only Python is supported; other programming languages require rebuilding the dataset.
- Static analyzers (CodeQL/Bandit) inherently have false negatives, leaving some security issues undetected.
- 12.7% of the secure code in DiSCo still contains remnant issues.
- Only billion-parameter models were tested; the effectiveness on 10B+ models remains unverified.

## Related Work & Insights
- **vs SafeCoder (He et al. 2024)**: SafeCoder utilizes contrastive/unlikelihood training, whereas LPO provides a more natural preference optimization formulation and achieves better performance.
- **vs Pivotal Token Search (Abdin et al. 2024)**: PTS selects key tokens by estimating their contribution to the overall probability, whereas LPO identifies differing tokens directly via `diff`, offering a simpler and more direct approach.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of localized preference optimization and knowledge-base-guided distillation is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 security benchmarks + 2 code quality benchmarks + multiple models + exhaustive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and detailed methodology descriptions.
- Value: ⭐⭐⭐⭐⭐ Vital contribution to secure code generation, with code and data open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Probability-Consistent Preference Optimization for Enhanced LLM Reasoning](probability-consistent_preference_optimization_for_enhanced_llm_reasoning.md)
- [\[ACL 2025\] Reverse Preference Optimization for Complex Instruction Following](reverse_preference_optimization_for_complex_instruction_following.md)
- [\[ACL 2025\] DiffPO: Diffusion Alignment with Direct Preference Optimization](diffpo_diffusion_alignment.md)
- [\[ACL 2025\] Optimal Transport-Based Token Weighting for Enhanced Preference Optimization](otpo_token_weighting.md)
- [\[ACL 2025\] T-REG: Preference Optimization with Token-Level Reward Regularization](t-reg_preference_optimization_with_token-level_reward_regularization.md)

</div>

<!-- RELATED:END -->
