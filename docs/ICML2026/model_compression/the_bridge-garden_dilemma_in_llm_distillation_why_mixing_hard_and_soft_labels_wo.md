---
title: >-
  [Paper Note] The Bridge-Garden Dilemma in LLM Distillation: Why Mixing Hard and Soft Labels Works
description: >-
  [ICML 2026][Model Compression][Knowledge Distillation] Authors find that a "linear mixture of soft and hard labels" consistently outperforms pure soft labels in LLM distillation. They shift the explanation from "hard labels facilitate optimization" to "hard labels suppress exposure bias." Using Bridge-Garden decomposition, they categorize sequence positions into "Bridges" (requiring precision) and "Gardens" (allowing flexibility), binding the mix coefficient to contextual ris…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Knowledge Distillation"
  - "Exposure Bias"
  - "Hybrid Hard-Soft Labels"
  - "Bridge-Garden Decomposition"
  - "Risk Guidance"
date: 2026-05-08
content_hash: c0f2d6275956a87e
---

# The Bridge-Garden Dilemma in LLM Distillation: Why Mixing Hard and Soft Labels Works

**Conference**: ICML 2026  
**arXiv**: [2605.26246](https://arxiv.org/abs/2605.26246)  
**Code**: https://github.com/ghwang-s/bridge_garden_hybrid_kd_release  
**Area**: Model Compression / Knowledge Distillation / LLM Training  
**Keywords**: Knowledge Distillation, Exposure Bias, Hybrid Hard-Soft Labels, Bridge-Garden Decomposition, Risk Guidance

## TL;DR
Authors find that a "linear mixture of soft and hard labels" consistently outperforms pure soft labels in LLM distillation. They shift the explanation from "hard labels facilitate optimization" to "hard labels suppress exposure bias." Using Bridge-Garden decomposition, they categorize sequence positions into "Bridges" (requiring precision) and "Gardens" (allowing flexibility), binding the mix coefficient to contextual risk. They propose four adaptive hybrid strategies that surpass mainstream on-policy/divergence-based KD across seven teacher-student pairs with a 9.7× training cost advantage.

## Background & Motivation
**Background**: Mainstream LLM distillation follows two paths. One is soft label distillation (Hinton-style, using various divergences like KL/JS/Reverse KL), matching the student to the teacher's full next-token distribution, considered "strictly more informative." The other is hard label distillation (Kim & Rush, Alpaca/Vicuna-style SFT), using cross-entropy on teacher-sampled tokens, common for black-box teachers or large vocabularies.

**Limitations of Prior Work**: Intuitively, soft labels should dominate. However, community recipes often use `loss = λ * KL + (1-λ) * CE`, reporting that hybrid loss outperforms pure KL. The problem remains that no research has clarified "why the mixture works"—whether due to easier optimization, stability, or regularization. Without this understanding, systematic design of hybrid strategies is impossible.

**Key Challenge**: In Qwen2.5-7B→3B experiments, the authors observed an anti-intuitive phenomenon: the hybrid loss results in higher imitation error (forward KL on teacher prefix) during training compared to pure soft labels, yet achieves higher benchmark accuracy. This suggests the naive assumption that "better training fit implies better inference" fails in autoregressive generation.

**Goal**: (1) Explain the true source of hybrid KD gains; (2) Formalize this explanation into a computable position-level risk metric; (3) Design adaptive hybrid algorithms that outperform SOTA while reducing training costs.

**Key Insight**: Starting from classical exposure bias (Bengio 2015), the authors decompose total inference error into training-fit error plus a shift term, finding that hard labels specifically suppress the shift term. From an algorithmic stability perspective (Bousquet-Elisseeff), they define the sensitivity of a "one-step coverage strategy" $\kappa(a\mid s)$, splitting sequences into Bridges and Gardens.

**Core Idea**: Hard labels "keep the model on the bridge" by concentrating probability on the unique safe token to avoid collapse, while soft labels "maintain the teacher’s diversity in the garden." Hybrid loss succeeds by capturing both benefits; thus, the optimal mixing coefficient $\lambda$ must adjust dynamically based on contextual risk.

## Method

### Overall Architecture
This paper answers why mixing soft and hard labels works and designs adaptive algorithms accordingly. It progresses through two layers: a theoretical layer that decomposes inference error into "training fit error + exposure bias residual" and further bounds the residual using "one-step coverage + risk sensitivity $\kappa$ + Bridge/Garden dichotomy," proving hard labels specifically address exposure bias. The algorithmic layer maps this to a hybrid loss $\ell_{\text{hyb}} = \lambda\,\ell_{\text{soft}} + (1-\lambda)\,\ell_{\text{hard}}$, where $\lambda$ varies with risk, providing four implementations to approximate $\kappa$. This pipeline modifies only the loss function shape and is orthogonal to choice of divergence.

### Key Designs

**1. Exposure-bias Decomposition: Proving Why Hybridization Works**

Intuitively, soft labels are more informative and should be superior. Yet, the reported success of `λ·KL + (1-λ)·CE` remains unexplained. Defining $d_T$ as the teacher prefix distribution and $d_\theta$ as the student prefix distribution, the authors decompose the total inference error as $\mathcal{L}_{d_\theta}(\pi_\theta) = \mathcal{L}_{d_T}(\pi_\theta) + (\mathcal{L}_{d_\theta}(\pi_\theta) - \mathcal{L}_{d_T}(\pi_\theta))$. The first term is the training fit error on teacher prefixes, while the second is the exposure bias. Fig.2(b,c) shows that adding hard labels increases the training term $\mathcal{L}_{d_T}$ but significantly reduces exposure bias, leading to lower net inference error. This rejects the "easier optimization" theory and identifies exposure bias suppression as the quantified causal mechanism.

**2. Bridge-Garden Decomposition and Risk Sensitivity $\kappa$**

To apply this globally to tokens, the authors construct a one-step coverage strategy $\pi^{(s,a)}$ (forcing output $a$ at prefix $s$, matching the teacher otherwise) and define risk sensitivity as:
$$\kappa(a\mid s) = (\mathcal{L}_{d^{(s,a)}}(\pi_\theta) - \mathcal{L}_{d_T}(\pi_\theta))/d_T(s)$$
This represents the incremental loss caused by changing the current token to $a$. High $\kappa(s) = \sum_{a\in\mathcal{V}}\kappa(a\mid s)$ positions are "Bridges" (where deviations cause collapse, e.g., $+/-$ signs in math), while low positions are "Gardens" (e.g., synonyms in dialogue). Thm.4.4 proves $\pi_{\text{hard}}$ is superior for Bridges (suppressing probability drift) but inferior for Gardens, while $\pi_{\text{soft}}$ is the opposite.

**3. Four Adaptive Hybrid Losses**

Since $\kappa$ cannot be computed precisely for LLMs, four proxies are implemented for $\ell_{\text{hyb}}(s;\theta) = \lambda\,\ell_{\text{soft}}(s;\theta) + (1-\lambda)\,\ell_{\text{hard}}(s;\theta)$:
*   **Confidence-weighted**: Uses teacher top-1 probability; $\lambda_{\text{conf}}(s) = 1 - \max_a \pi_T(a\mid s)$. Higher teacher confidence implies a Bridge, increasing hard label weight.
*   **Entropy-weighted**: $\lambda_{\text{ent}}(s) = H_T(s)/\log|\mathcal{V}|$ maps normalized uncertainty to Garden weight.
*   **Curriculum Schedule**: $\lambda(t) = \min(t/T,1)\cdot\lambda_{\max}$ prioritizes Bridges early in training and Gardens later.
*   **Risk-Guided Hybrid**: Rewrites the hard loss as $\ell'_{\text{hard}} = \ell_{\text{hard}} + (\alpha/4)\Delta_\theta(s,a^*)^2$, where $\Delta_\theta(s,a^*) = \log\sum_{a'} \exp(f_\theta(a'\mid[s,a^*]) - f_\theta(a^*\mid s))$, treating $\kappa$ as a reward to sharpen distributions at Bridges.

### Loss & Training
All training is off-policy based on teacher prefixes. Forward KL is the default divergence, but experiments with Reverse KL, JS, and TV prove that hybrid gains are orthogonal to the divergence choice. Risk-Guided fix $\alpha = 0.1$. Compared to on-policy methods like GKD, this approach avoids student sampling, reducing training costs by 9.7×.

## Key Experimental Results

### Main Results
Seven teacher-student pairs (Qwen, Llama, Gemma, DeepSeek-Coder) across benchmarks: BBH/MMLU/ARC-C (general), GSM8K/MATH/Gaokao23 (math), and HumanEval/MBPP (code).

| Configuration | BBH | MMLU | ARC-C | ThmQA | Avg |
|---------------|-----|------|-------|-------|-----|
| Student (No Distill) | 22.34 | 64.61 | 78.40 | 12.22 | 44.39 |
| Hard KD (Kim & Rush) | 41.52 | 65.76 | 78.75 | 23.75 | 52.45 |
| Soft KD (Hinton) | 41.65 | 64.45 | 78.33 | 23.02 | 51.87 |
| Static Hybrid | 42.61 | 66.89 | 79.30 | 25.45 | 53.56 |
| Confidence-weighted | 44.07 | 67.50 | 80.77 | 22.78 | 53.78 |
| Entropy-weighted | 46.83 | 67.06 | 79.81 | 23.65 | 54.34 |
| Curriculum Schedule | 44.39 | 67.32 | 80.51 | 22.13 | 53.59 |
| **Risk-Guided Hybrid (Ours)** | **46.53** | **69.05** | **81.23** | 23.82 | **55.16** |

### Ablation Study
Comparison with seven recent divergence-based SOTAs (Qwen2.5-7B → 3B).

| Divergence | BBH | MMLU | ARC-C | ThmQA | Avg |
|------------|-----|------|-------|-------|-----|
| Reverse KL (Gu 2024) | 44.07 | 65.67 | 77.68 | 24.20 | 52.90 |
| Total Variation (Wen 2023) | 40.50 | 64.52 | 78.11 | 22.83 | 51.49 |
| JS (Agarwal 2024) | 45.50 | 64.68 | 78.85 | 22.27 | 52.83 |
| Adaptive KL (Wu 2025) | 44.71 | 64.69 | 79.23 | 22.25 | 52.72 |
| Skew FKL (Ko 2024) | 41.39 | 64.67 | 77.75 | 23.77 | 51.89 |
| **HybKD (Ours)** | **46.53** | **69.05** | **81.23** | 23.82 | **55.16** |

### Key Findings
- Fig.2(b,c) highlights that hybrid training results in higher forward KL (worse fit) but significantly lower exposure bias, validating the theoretical decomposition.
- Risk-Guided Hybrid consistently leads across all teacher-student pairs and is orthogonal to divergence choice.
- Gains are highest in reasoning/coding tasks (e.g., GSM8K +5.9), where "Bridge" density is high.
- End-to-end training cost is 1/9.7 of on-policy KD methods (e.g., GKD) because it eliminates student sampling.

## Highlights & Insights
- The explanation for "worse training fit but better inference" is elegant: hard labels target the long-neglected exposure bias term.
- Bridge-Garden $\kappa$ aligns stability theory, reward magnitude, and KD, making the concept transferable to SFT or DPO.
- The Risk-Guided Hybrid loss is engineering-friendly, being a lightweight tweak that is essentially free to compute.

## Limitations & Future Work
- The Bridge/Garden dichotomy relies on $\kappa(s)$, which is only approximated; confidence/entropy might decouple from true risk in certain tasks like instruction following.
- Experiments focused on models up to 7B/8B; effectiveness across massive capacity gaps (e.g., 70B → 0.5B) or significant architectural differences remains an open question.
- All training remains off-policy; whether on-policy signals combined with Bridge-Garden weighting can yield further gains is unexplored.

## Related Work & Insights
- **vs Hinton et al. (2015)**: Replaces pure KL with a hybrid form; systematically explains hybrid superiority.
- **vs Kim & Rush (2016)**: Repositions hard labels from "noise reduction" to "exposure bias suppressors."
- **vs GKD / MiniLLM**: Achieves exposure bias matching via off-policy weighting rather than expensive on-policy rollouts (9.7× cheaper).
- **vs Reward-shaping (Cundy & Ermon)**: Maps reward consistency to KD by treating hard labels as rewards and soft labels as entropy regularizers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Hard Labels In! Rethinking the Role of Hard Labels in Mitigating Local Semantic Drift](hard_labels_in_rethinking_the_role_of_hard_labels_in_mitigating_local_semantic_d.md)
- [\[CVPR 2026\] Rethinking Dataset Distillation: Hard Truths about Soft Labels](../../CVPR2026/model_compression/rethinking_dataset_distillation_hard_truths_about_soft_labels.md)
- [\[ICML 2026\] DSL-Topic: Improving Topic Modeling by Distilling Soft Labels from Language Models](dsl-topic_improving_topic_modeling_by_distilling_soft_labelsfrom_language_models.md)
- [\[ICML 2026\] Toward Understanding Adversarial Distillation: Why Robust Teachers Fail](toward_understanding_adversarial_distillation_why_robust_teachers_fail.md)
- [\[NeurIPS 2025\] Why Knowledge Distillation Works in Generative Models: A Minimal Working Explanation](../../NeurIPS2025/model_compression/why_knowledge_distillation_works_in_generative_models_a_minimal_working_explanat.md)

</div>

<!-- RELATED:END -->
