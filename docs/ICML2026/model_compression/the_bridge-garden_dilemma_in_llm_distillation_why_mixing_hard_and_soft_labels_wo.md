---
title: >-
  [Paper Note] The Bridge-Garden Dilemma in LLM Distillation: Why Mixing Hard and Soft Labels Works
description: >-
  [ICML 2026][Model Compression][Knowledge Distillation] The authors discover that the "linear mixture of soft and hard labels" in LLM distillation almost always outperforms pure soft labels. They correct the intuitive rea…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Knowledge Distillation"
  - "Exposure Bias"
  - "Hybrid Hard-Soft Labels"
  - "Bridge-Garden Decomposition"
  - "Risk Guidance"
date: 2026-05-08
content_hash: 7a43de61f795dbad
---

# The Bridge-Garden Dilemma in LLM Distillation: Why Mixing Hard and Soft Labels Works

**Conference**: ICML 2026  
**arXiv**: [2605.26246](https://arxiv.org/abs/2605.26246)  
**Code**: https://github.com/ghwang-s/bridge_garden_hybrid_kd_release  
**Area**: Model Compression / Knowledge Distillation / LLM Training  
**Keywords**: Knowledge Distillation, Exposure Bias, Hybrid Hard-Soft Labels, Bridge-Garden Decomposition, Risk Guidance

## TL;DR
The authors discover that the "linear mixture of soft and hard labels" in LLM distillation almost always outperforms pure soft labels. They correct the intuitive reasoning from "hard labels are easier to optimize" to "hard labels suppress exposure bias." Using Bridge-Garden decomposition, they categorize generated sequences into "Bridges" (positions requiring high precision) and "Gardens" (flexible positions). By tying the mix coefficient to contextual risk, they propose 4 adaptive hybrid strategies that outperform mainstream on-policy/divergence-based KD across 7 teacher-student pairs with a 9.7× training cost advantage.

## Background & Motivation
**Background**: Mainstream LLM distillation follows two paths. One is soft-label distillation (Hinton-style metrics like KL/JS/Reverse KL divergence), where the student matches the teacher's next-token distribution across the entire vocabulary, considered "strictly more informative." The other is hard-label distillation (Kim & Rush style SFT), using only teacher-sampled tokens with cross-entropy, common for black-box teachers or large vocabulary scenarios.

**Limitations of Prior Work**: Intuitively, soft labels should dominate, yet practitioners often use recipes like `loss = λ * KL + (1-λ) * CE`, reporting that hybrid losses outperform pure KL. The issue is the lack of a clear explanation for "why hybrid works"—is it easier optimization, stability, or regularization? Without a systematic explanation, designing hybrid strategies remains trial-and-error.

**Key Challenge**: Through Qwen2.5-7B→3B experiments, the authors observed an anti-intuitive phenomenon: during training, hybrid loss results in a higher imitation error (forward KL on teacher prefix) compared to pure soft labels, yet yields higher benchmark accuracy. This suggests the naive assumption that "better training fit leads to better inference" does not hold for autoregressive generation.

**Goal**: (1) Explain the true source of hybrid KD gains; (2) Formalize this explanation into a computable position-level risk metric; (3) Design adaptive hybrid algorithms that outperform SOTA while reducing training costs.

**Key Insight**: Starting from classical exposure bias (Bengio 2015), the authors decompose total inference error into training-fit error plus a shift term, finding that hard labels specifically suppress the shift term. Using algorithmic stability (Bousquet-Elisseeff), they define the sensitivity $\kappa(a\mid s)$ of a "one-step override policy" and partition sequences into Bridge and Garden positions.

**Core Idea**: Hard labels prevent collapse at "Bridges" by concentrating probability on the only safe token, while soft labels maintain teacher diversity in "Gardens." Hybrid loss succeeds by capturing both benefits; thus, the optimal mix coefficient $\lambda$ must adjust dynamically based on contextual risk.

## Method

### Overall Architecture
The "Method" involves two levels: theoretical decomposition (explaining why hybrid wins) and four practical algorithms (determining $\lambda$). The theoretical layer decomposes inference imitation error into training error and an exposure bias residual, then applies an upper bound decomposition involving "one-step override policy + risk sensitivity $\kappa$ + Bridge/Garden dichotomy." The algorithmic layer implements four hybrid losses: confidence-weighted, entropy-weighted, curriculum scheduling, and Risk-Guided Hybrid (embedding risk into the hard term). The pipeline introduces no new modules and modifies only the training loss, making it orthogonal to divergence choices.

### Key Designs

1.  **Exposure-bias Decomposition and the Contribution of Hard Labels**:
    - **Function**: Elevates the empirical observation of "why hybrid is better" to a provable equation.
    - **Mechanism**: Denoting teacher prefix distribution as $d_T$ and student prefix distribution as $d_\theta$, the total inference error is $\mathcal{L}_{d_\theta}(\pi_\theta) = \mathcal{L}_{d_T}(\pi_\theta) + (\mathcal{L}_{d_\theta}(\pi_\theta) - \mathcal{L}_{d_T}(\pi_\theta))$, where the second term is exposure bias. Experimental Fig.2(b,c) shows that hard labels increase the training term $\mathcal{L}_{d_T}$ but decrease exposure bias more significantly. The net effect is reduced inference error—rejecting the "easier optimization" explanation.
    - **Design Motivation**: Transition "adding hard labels" from a trick to a measurable causal mechanism. Recognizing exposure bias as the beneficiary motivates identifying local metrics for exposure bias sensitivity.

2.  **Bridge-Garden Decomposition and Risk Sensitivity $\kappa$**:
    - **Function**: Labels every step in a sequence with a local risk tag indicating "how costly a deviation would be."
    - **Mechanism**: Constructing a one-step override policy $\pi^{(s,a)}$ (forcing output $a$ at prefix $s$ while matching the teacher elsewhere), the authors define $\kappa(a\mid s) = (\mathcal{L}_{d^{(s,a)}}(\pi_\theta) - \mathcal{L}_{d_T}(\pi_\theta))/d_T(s)$, the loss increment for changing a single token to $a$. Aggregating $\kappa(s) = \sum_{a\in\mathcal{V}}\kappa(a\mid s)$, high-risk positions are "Bridges" (e.g., +/- in math) and low-risk positions are "Gardens" (e.g., synonyms in dialogue). Thm.4.4 proves exposure bias upper bounds are dominated by high-risk token deviations $\lvert\Delta\pi_\theta(a\mid s)\rvert$ in Bridges, while Gardens reduce to standard distribution distance.
    - **Design Motivation**: Formalizing the strengths of hard and soft labels. Thm.4.4 proves $\pi_{\text{hard}}$ is smaller on $F_\mathcal{B}$ but larger on $F_\mathcal{G}$ (and vice versa for $\pi_{\text{soft}}$), ensuring a superior hybrid solution exists.

3.  **Four Bridge-Garden Adaptive Hybrid Losses**:
    - **Function**: Translates theoretical $\kappa$ into actionable training losses.
    - **Mechanism**: The base form is $\ell_{\text{hyb}}(s;\theta) = \lambda\,\ell_{\text{soft}}(s;\theta) + (1-\lambda)\,\ell_{\text{hard}}(s;\theta)$, where $\lambda$ varies. (a) Confidence-weighted: $\lambda_{\text{conf}}(s) = 1 - \max_a \pi_T(a\mid s)$, treating confident teacher outputs as Bridges; (b) Entropy-weighted: $\lambda_{\text{ent}}(s) = H_T(s)/\log|\mathcal{V}|$, mapping global uncertainty to Garden weights; (c) Curriculum scheduling: $\lambda(t) = \min(t/T,1)\cdot\lambda_{\max}$, prioritizing Bridges before Gardens; (d) Risk-Guided Hybrid: modifies the hard loss to $\ell'_{\text{hard}} = \ell_{\text{hard}} + (\alpha/4)\Delta_\theta(s,a^\*)^2$, where $\Delta_\theta(s,a^\*) = \log\sum_{a'} \exp(f_\theta(a'\mid[s,a^\*]) - f_\theta(a^\*\mid s))$.
    - **Design Motivation**: These variants approximate $\kappa$ from different angles. Confidence/Entropy are token-level signals with zero overhead; Curriculum is time-level; Risk-Guided embeds the "override" risk into a reward-like term using a log-sum-exp operation.

### Loss & Training
All training is based on teacher prefixes (off-policy). The default divergence is forward KL, though 7 types (Reverse KL, JS, TV, etc.) were tested, proving hybrid gains are orthogonal to divergence choice. Risk-Guided uses $\alpha = 0.1$ without tuning. Compared to on-policy methods like GKD/MiniLLM, this approach avoids student sampling, reducing training costs by 9.7× while remaining compatible with any divergence.

## Key Experimental Results

### Main Results
Testing on 7 teacher-student pairs (Qwen, Llama, Gemma, DeepSeek-Coder) across reasoning (BBH, MMLU, ARC-C, ThmQA), math (GSM8K, MATH, Gaokao23), and code (HumanEval, MBPP) benchmarks.

| Configuration | BBH | MMLU | ARC-C | ThmQA | Avg |
|------|-----|------|-------|-------|-----|
| Student (No Distill) | 22.34 | 64.61 | 78.40 | 12.22 | 44.39 |
| Hard KD (Kim & Rush) | 41.52 | 65.76 | 78.75 | 23.75 | 52.45 |
| Soft KD (Hinton) | 41.65 | 64.45 | 78.33 | 23.02 | 51.87 |
| Static Hybrid (Ours) | 42.61 | 66.89 | 79.30 | 25.45 | 53.56 |
| Confidence-weighted | 44.07 | 67.50 | 80.77 | 22.78 | 53.78 |
| Entropy-weighted | 46.83 | 67.06 | 79.81 | 23.65 | 54.34 |
| Curriculum Schedule | 44.39 | 67.32 | 80.51 | 22.13 | 53.59 |
| **Risk-Guided Hybrid** | **46.53** | **69.05** | **81.23** | 23.82 | **55.16** |

### Ablation Study
Comparison with 7 divergence-based SOTAs (Qwen2.5-7B → 3B):

| Divergence | BBH | MMLU | ARC-C | ThmQA | Avg |
|------------|-----|------|-------|-------|-----|
| Reverse KL (Gu 2024) | 44.07 | 65.67 | 77.68 | 24.20 | 52.90 |
| Total Variation (Wen 2023) | 40.50 | 64.52 | 78.11 | 22.83 | 51.49 |
| JS (Agarwal 2024) | 45.50 | 64.68 | 78.85 | 22.27 | 52.83 |
| Adaptive KL (Wu 2025) | 44.71 | 64.69 | 79.23 | 22.25 | 52.72 |
| Skew FKL (Ko 2024) | 41.39 | 64.67 | 77.75 | 23.77 | 51.89 |
| Skew RKL (Ko 2024) | 41.22 | 63.95 | 76.91 | 23.67 | 51.44 |
| $\alpha$-$\beta$ div (Wang 2025) | 45.12 | 64.95 | 79.81 | 22.94 | 53.21 |
| **HybKD (Ours)** | **46.53** | **69.05** | **81.23** | 23.82 | **55.16** |

### Key Findings
- Fig.2 shows that while hybrid training fit is worse, exposure bias is significantly lower, validating $\mathcal{L}_{d_\theta} = \mathcal{L}_{d_T} + \text{exposure}$.
- Risk-Guided Hybrid leads across 7 pairings and works orthogonally to other divergences.
- Massive gains in reasoning/math/code: Qwen2.5-Math-7B → 1.5B benchmarks rose significantly (GSM8K up ~6 pts), indicating Bridge sensitivity is critical for precision tasks.
- Training cost is 1/9.7 of on-policy KD (like GKD) because it utilizes teacher prefixes without student rollout.
- Synthetic domain experiments accurately calculating $\kappa$ directly verified the Bridge-Garden decomposition.

## Highlights & Insights
- The explanation that "worse training fit can yield better inference" by suppressing exposure bias transforms a common trick into a first-class methodology.
- Bridge-Garden $\kappa$ bridges stability theory, KD, and RL reward magnitude, providing a unified language for these domains.
- Risk-Guided Hybrid's log-sum-exp implementation is lightweight and easily integrated into industrial pipelines.
- The 9.7× cost advantage stems from paradigm choice (off-policy with teacher prefixes), allowing for much larger ablation budgets.

## Limitations & Future Work
- Bridge/Garden partitioning relies on $\kappa$, yet its proxies (confidence/entropy) may deviate from true risk in certain tasks like instruction following.
- Experiments focused on models up to 8B; behavior at larger capacity gaps (e.g., 70B → 0.5B) or architectures (e.g., Mamba) remains an open question.
- Reasoning tasks benefited most; open-ended generation showed smaller gains, likely due to a lower "Bridge density."
- Future work could explore the combination of Bridge-Garden weighting with on-policy signals.

## Related Work & Insights
- **vs Hinton et al. (2015) soft KD**: Replaces pure KL with a hybrid form; explains why hybrid is superior.
- **vs Kim & Rush (2016) hard KD**: Reinterprets hard labels as exposure bias suppressors rather than just noise reduction.
- **vs GKD / MiniLLM**: Achieves exposure bias matching via weighting instead of expensive student rollouts, saving 9.7× cost.
- **vs Skew/Adaptive KL**: Orthogonal approach (mix coefficient vs. divergence shape).
- **vs Cundy & Ermon (2023)**: Risk-Guided Hybrid adapts reward consistency to KD by treating hard labels as rewards and soft labels as entropy regularizers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Hard Labels In! Rethinking the Role of Hard Labels in Mitigating Local Semantic Drift](hard_labels_in_rethinking_the_role_of_hard_labels_in_mitigating_local_semantic_d.md)
- [\[ICML 2026\] DSL-Topic: Improving Topic Modeling by Distilling Soft Labels from Language Models](dsl-topic_improving_topic_modeling_by_distilling_soft_labelsfrom_language_models.md)
- [\[NeurIPS 2025\] Why Knowledge Distillation Works in Generative Models: A Minimal Working Explanation](../../NeurIPS2025/model_compression/why_knowledge_distillation_works_in_generative_models_a_minimal_working_explanat.md)
- [\[ICML 2026\] Toward Understanding Adversarial Distillation: Why Robust Teachers Fail](toward_understanding_adversarial_distillation_why_robust_teachers_fail.md)
- [\[ICML 2026\] Demystifying When Pruning Works via Representation Hierarchies](demystifying_when_pruning_works_via_representation_hierarchies.md)

</div>

<!-- RELATED:END -->
