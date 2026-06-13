---
title: >-
  [Paper Note] FlexGuard: Continuous Risk Scoring for Strictness-Adaptive LLM Content Moderation
description: >-
  [ACL 2026][LLM Safety][Content Moderation] FlexGuard proposes an LLM moderation model that outputs a continuous risk score (0-100) instead of a binary safe/unsafe judgment. Through rubric-guided distillation and GRPO ris…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Content Moderation"
  - "Continuous Risk Scoring"
  - "Strictness-Adaptive"
  - "Reinforcement Learning"
date: 2026-05-08
content_hash: 5fff65ec3b8efe86
---

# FlexGuard: Continuous Risk Scoring for Strictness-Adaptive LLM Content Moderation

**Conference**: ACL 2026  
**arXiv**: [2602.23636](https://arxiv.org/abs/2602.23636)  
**Code**: [GitHub](https://github.com/)  
**Area**: AI Safety / Content Moderation  
**Keywords**: Content Moderation, Continuous Risk Scoring, Strictness-Adaptive, LLM Safety, Reinforcement Learning

## TL;DR

FlexGuard proposes an LLM moderation model that outputs a continuous risk score (0-100) instead of a binary safe/unsafe judgment. Through rubric-guided distillation and GRPO risk alignment training, it achieves SOTA robustness and accuracy across different deployment strictness scenarios.

## Background & Motivation

**Background**: LLM content moderation models (LlamaGuard, WildGuard, etc.) have evolved through multiple generations and are widely used to detect harmful content in user inputs and model outputs. Most existing moderation models define content moderation as a fixed binary classification task.

**Limitations of Prior Work**: Enforcement strictness—the degree of conservatism a platform applies to "harm"—varies significantly across different platforms and periods. For example, platform X may allow appropriately labeled adult content, while certain Reddit communities require all-ages content. Binary moderation models are implicitly tied to the safety definitions of their training data and cannot adapt to changing strictness requirements, leading to inconsistent performance across strictness levels: Qwen3Guard's performance in prompt moderation drops by 19.2% from strict to loose settings.

**Key Challenge**: The "safe/unsafe" boundary for moderation decisions is not fixed but changes according to the deployment environment. However, existing models and benchmarks assume a single, fixed safety definition.

**Goal**: (1) Construct a benchmark (FlexBench) capable of evaluating moderation models across different strictness levels; (2) Design a moderation model (FlexGuard) that can adapt to strictness variations.

**Key Insight**: By replacing binary classification with continuous risk scoring, strictness adaptation is reduced to a simple threshold selection problem. Continuous labels are obtained through rubric-guided distillation, and score-severity consistency is optimized using GRPO reinforcement learning.

**Core Idea**: Calibrated continuous risk score output + threshold selection at deployment = strictness-adaptive moderation.

## Method

### Overall Architecture

The system consists of two parts: (1) FlexBench—a benchmark with strictness annotations containing 4K instances covering seven risk categories and five severity levels, supporting three evaluation modes: strict, moderate, and loose; (2) FlexGuard—a moderation model based on Qwen3-8B that learns to output risk categories and continuous scores through two-stage training (SFT warmup + GRPO alignment), adapting to strictness via thresholds during deployment.

### Key Designs

1.  **FlexBench Strictness-Adaptive Benchmark**:

    - **Function**: Evaluates the reliability of moderation models under varying strictness levels.
    - **Mechanism**: Defines five severity levels (BENIGN/LOW/MODERATE/HIGH/EXTREME), mapped to three strictness regimes—strict (only BENIGN is safe), moderate (BENIGN+LOW are safe), and loose (BENIGN+LOW+MODERATE are safe). It covers seven risk categories (violence, illegal acts, sexual content, privacy, discrimination, misinformation, and jailbreak) and includes 2K prompt instances and 2K response instances. A human-AI collaborative labeling process is used: an LLM generates candidate labels, five human annotators verify and correct them, and a senior annotator provides the final verdict for inconsistencies.
    - **Design Motivation**: Existing benchmarks use fixed binary labels and fail to evaluate model robustness as strictness changes.

2.  **Rubric-based Distillation Pipeline**:

    - **Function**: Generates pseudo-labels for training continuous risk scores.
    - **Mechanism**: Uses expert-designed rubrics to guide a strong LLM (e.g., GPT-5) to generate a risk category $c(x)$, a score $r'(x) \in [0, 100]$, and the reasoning process for each instance. A key step is label consistency calibration—aligning LLM scores with the binary labels of source datasets by linearly mapping raw scores $r'(x)$ to label-consistent intervals (safe: [0,40], unsafe: [40,100]), thereby suppressing cross-boundary outliers.
    - **Design Motivation**: Most public moderation corpora provide only binary labels, and direct continuous score annotation is prohibitively expensive. LLM distillation enables large-scale generation, while calibration ensures consistency with existing labels.

3.  **Two-stage Risk Alignment Training**:

    - **Function**: Trains the model to produce continuous scores consistent with risk severity.
    - **Mechanism**: Stage 1 employs LoRA SFT for warmup, teaching the model to follow rubric reasoning and output formatted $(\hat{c}(x), \hat{r}(x))$. Stage 2 utilizes GRPO reinforcement learning with a dense reward function $R(x) = s_{\text{category}}(x) + s_{\text{score}}(x)$, where the category accuracy reward $s_{\text{category}} \in \{-1, +1\}$ and the score regression reward $s_{\text{score}} = 2 - \frac{4}{E_{\max}} |\hat{r}(x) - r(x)| \in [-2, 2]$. $E_{\max}$ normalization ensures that errors across different target scores are comparable.
    - **Design Motivation**: SFT provides a stable initialization, while GRPO directly optimizes the score consistency objective. Dense linear regression rewards provide richer gradient signals than binary rewards.

### Loss & Training

A two-stage training approach is adopted: Stage 1 utilizes standard SFT with LoRA, and Stage 2 employs GRPO with a combined dense reward for category accuracy and score regression. Training was conducted on 8×H20 GPUs.

## Key Experimental Results

### Main Results

**FlexBench Strictness-Adaptive Moderation (Harmfulness F1 %)**

| Method | Prompt Avg | Prompt Worst | Response Avg | Response Worst |
| :--- | :--- | :--- | :--- | :--- |
| GPT-5 | 73.26 | 70.95 | 77.43 | 74.07 |
| Qwen3Guard-8B | 75.10 | 67.06 | 76.61 | 69.16 |
| BingoGuard-8B | 74.22 | 68.31 | 76.59 | 74.80 |
| **FlexGuard (Calibrated Threshold)** | **81.78** | **78.26** | **80.29** | **75.81** |

### Ablation Study

| Configuration | Key Metrics | Description |
| :--- | :--- | :--- |
| FlexGuard Full | Avg 81.78 / Worst 78.26 | Optimal performance |
| SFT Only (No GRPO) | Decrease | Insufficient score-severity consistency |
| No Consistency Calibration | Decrease | Increased cross-boundary outliers |
| Rubric Threshold (No Calibration) | 80.29 / 76.63 | Remaining highly competitive |

### Key Findings

- FlexGuard's performance drop across strictness levels is significantly lower than competitors: the best-worst gap on Prompts is only 5.73%, compared to 15.95% for Qwen3Guard and 13.52% for BingoGuard.
- Rubric thresholds achieve competitive performance without requiring a validation set (Prompt Avg 80.29), while calibrated thresholds provide a further improvement of approximately 1.5%.
- On public benchmarks (without strictness variation), FlexGuard also reaches or exceeds SOTA (Prompt Avg 85.36, Response Avg 87.85).
- The GRPO stage significantly enhances scoring quality: the MAE of scores decreases, and score distributions across severity levels become more distinct.

## Highlights & Insights

- Content moderation is redefined from a "classification problem" to a "risk assessment problem." The continuous scoring + threshold selection design elegantly decouples model capability from deployment requirements.
- Label consistency calibration is a critical technical detail—aligning LLM-distilled scores with existing binary labels resolves pseudo-label quality issues.
- The dense linear regression reward design (as opposed to common binary rewards) provides richer gradient signals for GRPO.

## Limitations & Future Work

- FlexBench currently supports only English; strictness adaptation behavior in multilingual contexts remains unknown.
- Three levels of strictness may not be granular enough—actual deployments might require more continuous control.
- The interpretability of continuous scores needs enhancement, as users may need to understand the semantic meaning of specific scores.
- Scoring stability under adversarial inputs (jailbreak attacks) has not been tested.

## Related Work & Insights

- **vs LlamaGuard/WildGuard**: These models output binary labels and perform poorly when adapting strictness via logit thresholds. FlexGuard natively outputs continuous scores.
- **vs BingoGuard/PKU-SafeRLHF**: These output discrete severity levels with limited granularity. FlexGuard's continuous scoring provides finer risk differentiation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The problem definition (strictness-adaptive moderation) is novel and practical; the continuous scoring scheme is intuitive and reasonable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Includes a self-built benchmark plus public benchmarks, multiple baselines, and complete ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ The structure is clear and the motivation is well-justified, though some details could be more concise.
- **Value**: ⭐⭐⭐⭐⭐ Directly addresses industry deployment pain points; FlexBench has the potential to become a new standard for moderation evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CarO: Chain-of-Analogy Reasoning Optimization for Robust Content Moderation](caro_chain-of-analogy_reasoning_optimization_for_robust_content_moderation.md)
- [\[ACL 2026\] Making MLLMs Blind: Adversarial Smuggling Attacks in MLLM Content Moderation](making_mllms_blind_adversarial_smuggling_attacks_in_mllm_content_moderation.md)
- [\[ICLR 2026\] ExpGuard: LLM Content Moderation in Specialized Domains](../../ICLR2026/llm_safety/expguard_llm_content_moderation_in_specialized_domains.md)
- [\[ACL 2026\] RISK: A Framework for GUI Agents in E-commerce Risk Management](risk_a_framework_for_gui_agents_in_e-commerce_risk_management.md)
- [\[ACL 2026\] Rethinking Jailbreak Detection of Large Vision Language Models with Representational Contrastive Scoring](rethinking_jailbreak_detection_of_large_vision_language_models_with_representati.md)

</div>

<!-- RELATED:END -->
