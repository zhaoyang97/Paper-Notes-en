---
title: >-
  [Paper Note] Mitigating Selection Bias in Large Language Models via Permutation-Aware GRPO
description: >-
  [ACL 2026][LLM Alignment][GRPO] The authors observe that standard GRPO treats different option orders of the same question as independent prompts…
tags:
  - "ACL 2026"
  - "LLM Alignment"
  - "GRPO"
  - "permutation invariance"
  - "selection bias"
  - "cross-permutation advantage"
  - "consistency reward"
date: 2026-05-08
content_hash: 018c640349ac7b56
---

# Mitigating Selection Bias in Large Language Models via Permutation-Aware GRPO

**Conference**: ACL 2026  
**arXiv**: [2603.21016](https://arxiv.org/abs/2603.21016)  
**Code**: GitHub (Mentioned in the paper, link not explicitly provided in the abstract)  
**Area**: LLM Alignment / Reinforcement Learning / Selection Bias / LLM-as-a-Judge  
**Keywords**: GRPO, permutation invariance, selection bias, cross-permutation advantage, consistency reward

## TL;DR
The authors observe that standard GRPO treats different option orders of the same question as independent prompts, leading to "permutation-blindness" where model choices vary with reordering. They propose PA-GRPO: organizing multiple permutations of the same semantic instance into a permutation group and explicitly optimizing for "constant choice despite order changes" using cross-permutation advantage baselines and consistency rewards. This significantly reduces selection bias across seven MCQ/Judge benchmarks while maintaining accuracy.

## Background & Motivation

**Background**: LLMs are increasingly utilized as multiple-choice question (MCQ) solvers and LLM-as-a-Judge evaluators, where the output space is restricted to discrete symbols like A/B/C/D. Theoretically, the position and label of an option are non-semantic. However, empirical evidence shows that LLMs often change their answers when options are swapped—a phenomenon known as selection bias. This includes position bias and label bias, which directly threaten the reliability of alignment, leaderboards, and data synthesis tasks.

**Limitations of Prior Work**: Existing debiasing methods fall into two categories: (1) **Inference-time calibration** (e.g., PriDe, CalibraEval) adjusts surface probabilities without modifying the model, which is computationally expensive and does not fix internal issues; internal interventions (e.g., UniBias, BNP) often have negative side effects due to attention masking or parameter pruning. (2) **Training-time SFT** (e.g., PIF, LLM distillation) treats different permutations as independent static samples for cross-entropy training; the model merely mimics the data distribution passively rather than actively exploring a "permutation-invariant" strategy space.

**Key Challenge**: The essence of selection bias is a failure in robust reasoning—where identical semantics in different surface forms should yield the same choice. This is fundamentally a reinforcement learning (RL) style policy learning problem rather than a supervised label-fitting problem. Even powerful RL methods like GRPO treat different permutations of the same instance as independent prompts, lacking cross-permutation consistency constraints. The authors term this failure mode **permutation-blindness**: the model receives high rewards for "good orders" but is not penalized for failing on "bad orders."

**Goal**: To embed permutation invariance directly into the RL objective, enabling the model to actively learn that "changing the order should not change the choice."

**Key Insight**: Since GRPO uses the mean reward within a group as a baseline to calculate relative advantage, extending the "group" from "multiple samples of the same prompt" to "multiple permutations × multiple samples of the same semantic instance" naturally allows for comparing different permutations at the advantage level.

**Core Idea**: Permutation Groups + Cross-Permutation Advantage + Consistency-Aware Reward. These three components together integrate consistency into the RL optimization objective.

## Method

### Overall Architecture
The PA-GRPO workflow consists of: (1) For each base instance $x=(q, \mathcal{C})$, a set of permutation mappings $\Pi$ generates $P$ prompt variants $\mathcal{G}(x) = \{p^{(t)} = \tau_t(x)\}_{t=1}^P$. For MCQs, 5 permutations are used (4 cyclic shifts + 1 reverse); for Judge tasks, 2 permutations (AB, BA) are used. (2) $N$ responses are sampled for each prompt, forming a permutation group of $P \times N$ samples. (3) A label-to-semantic mapping $m_{\tau_t}(\ell)$ maps "A/B/C/D" back to the original candidate indices $z^{(t,i)}$. A combined reward $r^{(t,i)} = r_\text{pre}^{(t,i)} + \lambda r_\text{con}^{(t,i)}$ is calculated, and the standard GRPO baseline is replaced with cross-permutation advantage for PPO-clip optimization. The training is implemented using the verl framework with LoRA.

### Key Designs

1.  **Permutation Group Construction**:
    -   **Function**: Defines what constitutes different surface permutations of the same semantic instance while controlling computational costs.
    -   **Mechanism**: For MCQs, the full permutation $4!=24$ is too costly. The authors select 5 representative permutations: $\Pi_\text{MCQ} = \text{\{ABCD, BCDA, CDAB, DABC, DCBA\}}$. These 4 cyclic shifts ensure each candidate appears in every position once (eliminating position-label binding), while the 1 reverse order breaks the adjacency pattern where "A" always precedes "B" in cyclic shifts. For Judge tasks ($P=2$), the full set {AB, BA} is used.
    -   **Design Motivation**: Using 5 vs. 24 permutations saves ~5x compute. Experiments show only a 2-point difference in TinyMMLU CA (75.0 vs. 77.0), indicating high cost-effectiveness.

2.  **Cross-Permutation Advantage**:
    -   **Function**: Upgrades the per-prompt baseline in standard GRPO to a per-permutation-group baseline.
    -   **Mechanism**: The entire group of $P \times N$ samples is treated as a single comparison set. The advantage is calculated as $A_\text{PA}^{(t,i)} = (r^{(t,i)} - \mu_{\mathcal{G}}) / (\sigma_{\mathcal{G}} + \epsilon)$, where $\mu_{\mathcal{G}} = \frac{1}{PN}\sum_{t,i} r^{(t,i)}$. If $\sigma_{\mathcal{G}} < \delta$ (minimal reward variance within the group), the advantage is set to 0 to avoid noise amplification. Only samples that perform better than the global average across all permutations receive a positive advantage.
    -   **Design Motivation**: Standard GRPO normalizes within a single prompt, allowing models to score well by succeeding only on "good orders." Cross-permutation normalization prevents the policy from benefiting from shortcuts sensitive to the ABCD order.

3.  **Consistency-Aware Reward ($r_\text{con}$)**:
    -   **Function**: In addition to the ground-truth accuracy reward $r_\text{pre}$ (comprising $r_\text{acc} \in \{+1, -1\}$, length $\pm 0.1$, and format $\pm 0.3$), an explicit consistency reward is added to incentivize "constant choice despite order changes."
    -   **Mechanism**: For Judge tasks, index-aligned pairwise matching is used: responses for the same index $i$ under permutations (1, 2) are paired. If $z^{(1,i)} = z^{(2,i)}$, $r_\text{con} = +1$; otherwise, $-1$. For MCQ tasks, unique-mode agreement is used: vote counts $n_k$ for each semantic candidate are tallied. If the mode is unique $|\mathcal{M}| = 1$ and $z^{(t,i)} = z^\star$, then $r_\text{con} = +1$; ties or mismatches result in $-1$.
    -   **Design Motivation**: Advantage provides a relative signal, which can be ambiguous; $r_\text{con}$ provides an absolute signal explicitly indicating that "internal disagreement is bad." $\lambda = 1.0$ was found to be optimal.

### Loss & Training
The final PPO-clip objective is: $\mathcal{L}_\text{clip}(\theta) = \mathbb{E}[\min(\rho^{(t,i)} A_\text{PA}^{(t,i)}, \text{clip}(\rho^{(t,i)}, 1-\eta, 1+\eta) A_\text{PA}^{(t,i)})]$, plus a KL divergence penalty relative to the reference policy. The experiments use three policy models: Llama-3.1-8B-Instruct, Qwen3-8B, and Qwen3-32B. Training data includes Chatbot Arena (pairwise) and the MMLU training set (MCQ), fine-tuned via LoRA.

## Key Experimental Results

### Main Results (Llama-3.1-8B, evaluating accuracy/consistency/CA)

| Method | MT-Bench Acc/Con/CA | JudgeBench Acc/Con/CA | RewardBench Acc/Con/CA |
| :--- | :--- | :--- | :--- |
| Base | 59.6 / 25.2 / 22.2 | 35.0 / 34.8 / 6.1 | 60.5 / 31.5 / 26.2 |
| GRPO | 75.7 / 80.6 / 65.4 | 48.2 / 56.1 / 28.2 | 70.9 / 76.9 / 61.5 |
| PIF (SFT) | 76.1 / 84.6 / 70.4 | 53.3 / 59.2 / 30.4 | 73.7 / 76.7 / 62.0 |
| CalibraEval (Inference) | 62.3 / 42.1 / 33.4 | 49.3 / 15.7 / 7.1 | 60.7 / 34.4 / 27.8 |
| **PA-GRPO** | **77.6 / 88.0 / 71.7** | **57.1 / 58.3 / 32.4** | **71.0 / 82.7 / 62.3** |

Gains were even more significant for Qwen3-8B: JudgeBench Acc 50.4→60.1 (+9.7), CA 34.8→45.3 (+10.5); GPQA CA 43.8→56.7 (+12.9). For Qwen3-32B, which already performs near the ceiling, MT-Bench Consistency still improved from 90.6 to 91.6.

### Ablation Study (Llama-3.1-8B, PreferenceBench)

| Configuration | Acc | Con | CA |
| :--- | :--- | :--- | :--- |
| Base | 60.8 | 22.6 | 22.1 |
| GRPO | 82.2 | 85.1 | 76.3 |
| GRPO + $r_\text{con}$ only | 82.6 | 85.9 | 76.9 |
| GRPO + $A_\text{PA}$ only | 83.4 | 86.4 | 77.8 |
| **PA-GRPO (both)** | **86.2** | **87.2** | **79.8** |

### Key Findings
- **Complementarity**: Adding $r_\text{con}$ alone primarily improves consistency; adding $A_\text{PA}$ alone stabilizes advantage calculation but has a limited effect on consistency. The combination of both is required to simultaneously improve accuracy, consistency, and CA.
- **Independence from CoT**: Using direct decoding, PA-GRPO achieves 69.3% CA on MT-Bench, which is 11.3% higher than the Base+CoT score (58.0%). This suggests that invariance is truly internalized in the policy rather than compensated for by reasoning chains.
- **Residual Bias is Position-Driven**: On JudgeBench, consistency remains 79.0% under label-only perturbations but drops to 45.5% under order-only perturbations. Position bias is more persistent than label bias in LLMs.
- **$P=5$ is the "Sweet Spot"**: Cyclic shifts plus the reverse order cover most adjacency patterns. Increasing to $P=24$ provides marginal gains (+2 CA points) for ~5x the compute.
- **$\lambda = 1.0$ provides the best balance**.

## Highlights & Insights
- **"Permutation-blindness" is a clean negative concept**: By identifying the natural loophole of intra-group normalization in GRPO, the authors provide a minimalist fix (baseline upgrade + reward term).
- **5 permutations effectively cover 24**: The combination of 4 cyclic shifts and 1 reverse order is elegant—cyclic shifts ensure uniform position coverage, while the reverse order breaks cyclic adjacency patterns. This trick is transferable to any discrete choice RL task.
- **Consistency reward uses mode rather than majority**: On MCQs, using the unique mode and penalizing ties with $-1$ is stricter than a simple majority vote. This prevents the model from distributing disagreement across options to exploit the consistency reward.

## Limitations & Future Work
- The method is specifically targeted at discrete choice tasks (MCQ/Judge). Quantifying "semantic equivalence" in open-ended generation is difficult, and permutations are harder to define.
- Experiments were limited to English and mid-sized models (8B/32B). Generalization to super-long options or multilingual scenarios remains unknown.
- Future improvements could involve hierarchical advantages or weighting position-specific consistency terms to address residual position bias.

## Related Work & Insights
- **vs. PriDe / CalibraEval (Inference Calibration)**: These methods adjust softmax during inference without altering the model. PA-GRPO trains invariance into the policy, requiring only single-pass inference during deployment.
- **vs. PIF (SFT Debiasing)**: PIF uses point-wise negative samples for cross-entropy, where the model passively learns what is incorrect. PA-GRPO uses RL for active exploration, letting the model discover position-independent strategies that yield stable, high rewards.
- **vs. Standard GRPO**: The primary difference is the boundary of the "group" for the baseline calculation. This single change yielded +6.4 CA on PreferenceBench, suggesting the definition of a "group" in GRPO is a significant hyperparameter.
- **Insight**: All work using RL for alignment should question whether their "groups" consist of truly semantically equivalent samples. This perspective can be extended to chain-of-thought RL (different reasoning paths for the same problem) or tool-use RL (different tool sequences for the same task).

## Rating
- Novelty: ⭐⭐⭐⭐ "Permutation-blindness" is a fresh concept, though the solutions (cross-permutation baseline + consistency reward) are intuitive combinations.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Conducted across 3 backbones, 7 benchmarks, and 5 baselines, with thorough ablations on bias decomposition, CoT, hyperparameters, and permutation size.
- Writing Quality: ⭐⭐⭐⭐⭐ Definitions are clear, formulas are concise, and comparisons are intuitive.
- Value: ⭐⭐⭐⭐⭐ A directly applicable RL recipe providing immediate debiasing for any work using LLM-as-a-Judge for alignment or leaderboards.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Exploring the Effects of Alignment on Numerical Bias in Large Language Models](../../AAAI2026/llm_alignment/exploring_the_effects_of_alignment_on_numerical_bias_in_large_language_models.md)
- [\[ACL 2026\] S2H-DPO: Hardness-Aware Preference Optimization for Vision-Language Models](s2h-dpo_hardness-aware_preference_optimization_for_vision-language_models.md)
- [\[ACL 2026\] Taming Extreme Tokens: Covariance-Aware GRPO with Gaussian-Kernel Advantage Reweighting](taming_extreme_tokens_covariance-aware_grpo_with_gaussian-kernel_advantage_rewei.md)
- [\[ACL 2026\] BACH-V: Bridging Abstract and Concrete Human-Values in Large Language Models](bach-v_bridging_abstract_and_concrete_human-values_in_large_language_models.md)
- [\[ACL 2026\] ConsistRM: Improving Generative Reward Models via Consistency-Aware Self-Training](consistrm_improving_generative_reward_models_via_consistency-aware_self-training.md)

</div>

<!-- RELATED:END -->
