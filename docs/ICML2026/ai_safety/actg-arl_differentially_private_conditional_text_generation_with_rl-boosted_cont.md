---
title: >-
  [Paper Note] ACTG-ARL: Differentially Private Conditional Text Generation with RL-Boosted Control
description: >-
  [ICML 2026][AI Safety][Private synthetic data] This paper proposes a hierarchical framework, ACTG, which decomposes private text generation into two subtasks: feature learning and conditional text generation. Furthermore…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Private synthetic data"
  - "conditional text generation"
  - "attribute control"
  - "instruction following"
  - "reward hacking"
date: 2026-05-08
content_hash: 8cd34993bef2c0c1
---

# ACTG-ARL: Differentially Private Conditional Text Generation with RL-Boosted Control

**Conference**: ICML 2026  
**arXiv**: [2510.18232](https://arxiv.org/abs/2510.18232)  
**Code**: https://github.com/actg-arl/ACTG-ARL  
**Area**: Differential Privacy / Text Generation / RL Alignment  
**Keywords**: Private synthetic data, conditional text generation, attribute control, instruction following, reward hacking

## TL;DR
This paper proposes a hierarchical framework, ACTG, which decomposes private text generation into two subtasks: feature learning and conditional text generation. Furthermore, it introduces Anchored RL, which combines reinforcement learning objectives with optimal N-out-of-K SFT anchors, thereby improving the instruction-following ability of the conditional generator while maintaining text fidelity. On biomedical data, it achieves a 20% MAUVE improvement over prior work.

## Background & Motivation

**Background**
Modern AI applications rely on large amounts of user data (mobile input, recommendation history, dialogue preferences, etc.), which pose high privacy risks. Generating private synthetic data is a promising paradigm, allowing downstream tasks to reuse synthetic data without additional privacy costs. DP synthetic text is a hot topic, but existing work mainly focuses on generating static datasets, neglecting the practical need for fine-grained control.

**Limitations of Prior Work**
1. **CTCL Limitations**: Relies on pre-trained general topic models, which may not match private domain data. Using coarse-grained topics to forcefully classify fine-grained texts leads to inaccurate topic inference. When the dataset is small relative to the number of topics, the histogram contains many empty values, and after denoising, the signal is drowned in noise.
2. **Difficulty in Balancing Control and Fidelity**: Traditional RL optimization leads to reward hacking, where the model learns to generate outputs that formally satisfy constraints but with degraded text quality (e.g., TL;DR-style summaries).

**Key Challenge**
The distribution matching objective encourages sampling from high-density regions of $P(X,Y)$ (where the model is confident), while the value of data augmentation comes from low-density regions (uncertain boundaries or underrepresented groups)—this leads to a misalignment between the generator and augmentation task objectives.

**Goal**
1. Build a modular framework and identify the optimal configuration via systematic ablation.
2. Improve the instruction-following ability of the conditional generator while preserving privacy.

**Key Insight**
Starting from "attribute conditioning," structured tabular patterns are used as features, combined with a DP feature generator and a DP fine-tuned conditional generator. Furthermore, reinforcement learning is combined with feature constraints to construct verifiable reward signals.

**Core Idea**
Hierarchical decomposition: first, extract patterned features $\mathcal{D}_{\text{priv}}^f$ from private domain data; use a DP tabular synthesizer to generate private features $\mathcal{D}_{\text{syn}}^{\tilde{f}}$; then use DP fine-tuning to learn the conditional mapping from features to text; finally, Anchored RL uses optimal N-out-of-K data as SFT anchors to prevent RL drift, achieving hybrid optimization $\mathcal{L}=\mathcal{L}_{\text{RL}}+\gamma\cdot\mathcal{L}_{\text{SFT}}$.

## Method

### Overall Architecture
Divided into three stages:

**Stages 0-2 (ACTG):**
1. Feature Extraction: Use an Oracle LLM to extract a structured attribute matrix $D_{\text{priv}}^f$ from private domain text $D_{\text{priv}}^x$, containing K fields, each with predefined options.
2. Private Feature Generation (privacy budget $\varepsilon_1$): Use AIM (advanced aggregation method) for differentially private synthesis of tabular features, generating $D_{\text{syn}}^{\tilde{f}}$.
3. Private Conditional Text Generation (privacy budget $\varepsilon_2$): For (feature, text) pairs, perform DP fine-tuning to learn $G_{x|f}$ for conditional text generation.

**Stages 3-4 (Anchored RL):**
4. Optimal N-out-of-K Anchor Data: For each feature $f\sim G_f$, generate N candidate texts from $G_{x|f}$, select the best one by instruction-following accuracy (IFAcc), and form the SFT anchor $D_{\text{SFT}_N}$ (no additional privacy cost).
5. Hybrid Objective Training: Starting from the $G_{x|f}$ checkpoint, jointly optimize with $\mathcal{L}_{\text{RL}}+\gamma\cdot\mathcal{L}_{\text{SFT}}$, where $\gamma$ uses a linear decay strategy (high at the beginning to maintain fidelity, gradually decreasing to allow instruction-following improvement).

### Key Designs

1. **Hierarchical Decomposition + Structured Attribute Patterns**:

    - Function: Decomposes the conditional generation problem into two tractable subproblems and replaces general topics with domain-specific attribute patterns.
    - Mechanism: The first layer learns the marginal distribution of features (in low-dimensional tabular space, using a mature AIM synthesizer, which is more privacy-efficient); the second layer learns the text distribution conditioned on features (using DP fine-tuning). Attribute patterns are designed by an Oracle LLM or experts on private domain data, capturing key dimensions and avoiding the domain mismatch and sparse histogram issues of CTCL's general topics.
    - Design Motivation: Focuses privacy budget on key information and aligns with the natural hierarchical structure of the data.

2. **Anchored RL to Prevent Reward Hacking**:

    - Function: Improves both instruction-following (IFAcc) and maintains text fidelity (MAUVE), avoiding reward hacking caused by standard PPO.
    - Mechanism: (i) Uses optimal N-out-of-K sampling from $G_{x|f}$ itself to construct SFT anchors $D_{\text{SFT}_N}$ (no privacy cost, as the model is already DP fine-tuned); (ii) In the RL stage, uses a hybrid loss $\mathcal{L}=\mathcal{L}_{\text{RL}}+\gamma(t)\mathcal{L}_{\text{SFT}}$ to anchor the model near the reference distribution; (iii) $\gamma(t)$ decays linearly—strong fidelity at the start, gradually relaxed to allow instruction-following improvement.
    - Design Motivation: Adapts the "reference KL" idea from RLHF to private text generation; uses model self-sampling as SFT anchors, maintaining quality without privacy leakage.

3. **Instruction-Following Accuracy as Verifiable Reward**:

    - Function: Formalizes "whether attribute constraints are followed" as an automated reward signal.
    - Mechanism: For generated text, use an Oracle LLM to extract attributes, and compute $\text{IFAcc}=\mathbb{E}_f[\frac{1}{K}\sum_{k=1}^K\mathbb{I}(f_k=\hat{f}_k)]$. This metric is used as the reward in the RL stage and for optimal N-out-of-K selection.
    - Design Motivation: The structured attribute space naturally provides a verifiable, automatically evaluable target signal, which is rare for RL in generation tasks.

### Loss & Training

**DP Privacy Accounting**: The total privacy budget $(\varepsilon,\delta)$ consists of two stages $\varepsilon=\varepsilon_1+\varepsilon_2$; for each total budget $\varepsilon\in\{1,4,\infty\}$, the split $(\varepsilon_1,\varepsilon_2)$ is independently tuned; $\delta=1/(n\log n)$. The RL stage uses a hybrid loss $\mathcal{L}=\mathcal{L}_{\text{RL}}+\gamma(t)\mathcal{L}_{\text{SFT}}$, with $\gamma(t)$ linearly decaying.

## Key Experimental Results

### Main Results

| Dataset | Method | MAUVE | F1 Class | NTP Acc | IFAcc | $d_{\text{JS}}^f$ |
|---------|--------|-------|----------|---------|-------|-------------------|
| bioRxiv(ε=4) | Aug-PE | 0.68 | 0.72 | - | - | 0.15 |
| | vanilla DP-FT | 0.62 | 0.68 | 0.41 | 0.53 | 0.18 |
| | CTCL | 0.64 | 0.70 | 0.42 | 0.48 | 0.16 |
| | ACTG | 0.73 | 0.76 | 0.56 | 0.53 | 0.09 |
| | ACTG-ARL | **0.74** | **0.79** | **0.58** | **0.62** | **0.08** |
| PMC-Patients(ε=4) | CTCL | 0.59 | 0.64 | 0.38 | 0.48 | 0.20 |
| | ACTG | **0.71** | 0.75 | 0.51 | 0.50 | 0.10 |
| | ACTG-ARL | 0.70 | **0.77** | **0.53** | **0.58** | **0.09** |

### Ablation Study

| Component | Removed/Replaced | MAUVE | IFAcc | $d_{\text{JS}}^f$ | Note |
|-----------|-----------------|-------|-------|-------------------|------|
| Feature Model | Use CTCL general topics | 0.64 | 0.48 | 0.16 | General topics perform significantly worse |
| Feature Generator | DP-FT replaces AIM | 0.68 | 0.50 | 0.12 | AIM performs better (less budget waste) |
| Conditional Generator | Direct prompting replaces DP fine-tuning | 0.61 | 0.55 | 0.14 | Fine-tuned version is more stable |
| Full ACTG | - | 0.73 | 0.53 | 0.09 | Baseline |
| +Standard PPO | No anchor | 0.42 | 0.68 | 0.22 | Severe reward hacking, MAUVE collapses |
| +Anchored RL | Full method | 0.74 | 0.62 | 0.08 | Improves IFAcc while maintaining fidelity |

### Key Findings
- **Feature Design is Critical**: Structured attribute patterns significantly outperform general topics, with MAUVE on bioRxiv increasing from 0.64 to 0.73 (+14%).
- **Tabular vs. Text Feature Generation**: AIM (tabular) saves privacy budget compared to DP-FT (text), with lower error $d_{\text{JS}}^f$ (0.12 vs 0.14).
- **RL Reward Hacking is Severe**: Standard PPO drops MAUVE from 0.73 to 0.42, while Anchored RL restores it to 0.74 (IFAcc from 0.53→0.62).
- **Optimal N-out-of-K Effect**: Selecting the best from N=5 or 10 candidates yields high-quality, diverse SFT datasets with no additional privacy cost.
- **Privacy Budget Split**: For $\varepsilon=4$, the optimal split is about $(\varepsilon_1,\varepsilon_2)\approx(1.5,2.5)$ or $(2,2)$, indicating both stages require sufficient budget.

## Highlights & Insights
- **Elegance of Hierarchical Design**: Decomposing the complex end-to-end DP text generation problem into low-dimensional tabular synthesis + conditional text generation improves modularity and allows each module to use the optimal tool (AIM vs. LLM fine-tuning).
- **Practical Ingenuity of Anchored RL**: Optimal N-out-of-K self-extraction of references avoids accessing private domain data, incurs zero privacy cost, and effectively prevents reward hacking—a clever adaptation of RLHF to privacy scenarios.
- **Attribute Matching as Reward**: Using the structured attribute space as the basis for IFAcc transforms the text understanding problem into a formal attribute extraction problem, facilitating automation and verification.

## Limitations & Future Work
- **Limited Model and Data Scope**: Experiments are only conducted on gemma-3-1b-pt (biomedical domain), not covering legal, financial, dialogue, or other domains, nor exploring large model performance.
- **Assumed Attribute Space Design**: The paper does not discuss how to automate optimal attribute pattern design, currently relying on manual or Oracle LLM input, which may be a bottleneck.
- **Privacy Budget Split Optimization**: The $(\varepsilon_1,\varepsilon_2)$ split is determined by hyperparameter tuning, lacking theoretical guidance or adaptive strategies.

## Related Work & Insights
- **vs DP-FT**: Directly applying DP fine-tuning to LLMs without considering conditional control or structured features leads to significant quality degradation. This work improves via hierarchy and attribute conditioning.
- **vs CTCL**: Also uses conditioning, but CTCL uses fixed general topics, while this work uses data-specific attribute patterns, significantly improving pattern-data alignment.
- **vs Aug-PE (Private Evolution)**: PE uses LLM iterative refinement; this work uses direct fine-tuning + RL. In the biomedical domain, ACTG-ARL is more stable.

## Rating
- Novelty: ⭐⭐⭐⭐ Both the hierarchical framework and Anchored RL are new contributions; the optimal N-out-of-K anchor idea is clever and cost-free.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two biomedical datasets, multi-dimensional evaluation, thorough ablation. Limitation: does not cover multiple dataset families.
- Writing Quality: ⭐⭐⭐⭐ Clear problem description, complete algorithm pseudocode, detailed experiments.
- Value: ⭐⭐⭐⭐ Practical needs of DP synthetic text are addressed (+20% MAUVE); conditional control is systematically explored for privacy applications for the first time, with high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Privacy Amplification in Differentially Private Zeroth-Order Optimization with Hidden States](privacy_amplification_in_differentially_private_zeroth-order_optimization_with_h.md)
- [\[CVPR 2026\] One-to-More: High-Fidelity Training-Free Anomaly Generation with Attention Control](../../CVPR2026/ai_safety/one-to-more_high-fidelity_training-free_anomaly_generation_with_attention_control.md)
- [\[ICML 2026\] Limits of Convergence-Rate Control for Open-Weight Safety](limits_of_convergence-rate_control_for_open-weight_safety.md)
- [\[AAAI 2026\] An Improved Privacy and Utility Analysis of Differentially Private SGD with Bounded Domain and Smooth Losses](../../AAAI2026/ai_safety/an_improved_privacy_and_utility_analysis_of_differentially_p.md)
- [\[NeurIPS 2025\] Differentially Private High-dimensional Variable Selection via Integer Programming](../../NeurIPS2025/ai_safety/differentially_private_high-dimensional_variable_selection_via_integer_programmi.md)

</div>

<!-- RELATED:END -->
