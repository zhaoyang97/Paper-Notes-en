---
title: >-
  [Paper Note] Antibody: Strengthening Defense Against Harmful Fine-Tuning for Large Language Models via Attenuating Harmful Gradient Influence
description: >-
  [ICLR 2026][LLM Alignment][Harmful fine-tuning attack] This paper proposes Antibody, a two-stage defense framework that (1) during alignment, applies flatness regularization to place the model in a flat region of the harmful loss landscape (small gradients → harder to attack), and (2) during fine-tuning, suppresses learning from harmful samples via a likelihood-ratio-based sample weighting scheme (contrasting the likelihood of task completion vs. refusal). The average Harmful Score is reduced from 15.29% to 7.04%.
tags:
  - ICLR 2026
  - LLM Alignment
  - Harmful fine-tuning attack
  - safety alignment
  - loss flatness
  - sample weighting
  - FTaaS safety
date: 2026-05-08
content_hash: c6c76a8c81a70ce5
---

# Antibody: Strengthening Defense Against Harmful Fine-Tuning for Large Language Models via Attenuating Harmful Gradient Influence

**Conference**: ICLR 2026
**arXiv**: [2603.00498](https://arxiv.org/abs/2603.00498)
**Code**: To be released
**Area**: LLM Alignment
**Keywords**: Harmful fine-tuning attack, safety alignment, loss flatness, sample weighting, FTaaS safety

## TL;DR
This paper proposes Antibody, a two-stage defense framework that (1) during alignment, applies flatness regularization to place the model in a flat region of the harmful loss landscape (small gradients → harder to attack), and (2) during fine-tuning, suppresses learning from harmful samples via a likelihood-ratio-based sample weighting scheme (contrasting the likelihood of task completion vs. refusal). The average Harmful Score is reduced from 15.29% to 7.04%.

## Background & Motivation

**Background**: FTaaS platforms (e.g., OpenAI/Mistral fine-tuning services) allow users to upload data to fine-tune LLMs; however, user-submitted data may contain harmful samples (intentionally or unintentionally), causing safety alignment to be compromised.

**Limitations of Prior Work**: (a) Alignment-stage defenses (e.g., Vaccine/Booster) are static and cannot adapt to varying attack configurations (high step counts, large learning rates); (b) fine-tuning-stage defenses (e.g., Lisa/SafeInstr) either provide insufficient protection or degrade task performance; (c) most methods exhibit a severe trade-off between safety and task performance.

**Key Challenge**: Standard SFT does not distinguish between benign and harmful samples — all gradients are aggregated for updates, so even a small number of harmful sample gradients can poison the model.

**Goal**: Design a defense that operates cooperatively across both alignment and fine-tuning stages, thoroughly suppressing harmful gradient influence without degrading benign task learning.

**Key Insight**: From the perspective of gradient influence — if harmful sample gradients are already small after alignment (flat region) and are further down-weighted during fine-tuning, their influence can be effectively eliminated.

**Core Idea**: Flatten the harmful loss during alignment (small gradients) + likelihood-ratio-based weighting during fine-tuning (low weight for harmful samples) → harmful gradients are doubly suppressed.

## Method

### Overall Architecture
Antibody operates in two stages: (1) **Alignment stage** — optimizes $\mathcal{L}_{\text{align}}(\theta) + \lambda_t \mathcal{L}_{\text{sharp}}(\theta) + \lambda_{\text{refusal}} \mathcal{L}_{\text{refusal}}(\theta_{\text{pert}})$, placing the model in a flat region of the harmful loss while maintaining alignment; (2) **Fine-tuning stage** — applies sample-weighted updates $\theta_{t+1} \leftarrow \theta_t - \eta \sum_i w_{\theta_t}(x_i,y_i) \nabla \ell_{\theta_t}(x_i,y_i)$, where harmful samples automatically receive lower weights.

### Key Designs

1. **Robust Alignment via Flatness Regularization**

    - **Function**: Places the model in a flat region of the harmful loss $\mathcal{L}_{\text{harm}}$.
    - **Mechanism**: Sharpness is defined as $\mathcal{L}_{\text{sharp}}(\theta) = \mathcal{L}_{\text{harm}}(\theta) - \min_{\phi \in \mathcal{B}_\rho(\theta)} \mathcal{L}_{\text{harm}}(\phi)$, i.e., the drop in harmful loss within the $\rho$-neighborhood. Minimizing sharpness places the model in a flat region, so harmful sample gradients are naturally small during subsequent fine-tuning. The update direction $\delta_t^* = \nabla \mathcal{L}_{\text{align}} + \lambda_t \nabla \mathcal{L}_{\text{sharp}}$ is derived from a bi-objective optimization via KKT conditions (Theorem 4.1), with $\lambda_t$ adapted automatically.
    - **Design Motivation**: A flat region implies that perturbations near $\theta$ (i.e., fine-tuning) will not significantly reduce the harmful loss — making safety alignment more robust.

2. **Safety Fine-Tuning with Likelihood-Ratio-Based Sample Weighting**

    - **Function**: Dynamically assigns weights to samples in each mini-batch during fine-tuning to suppress harmful samples.
    - **Mechanism**: For each sample, the likelihood ratio $r_\theta(x_i,y_i) = \log \frac{\pi_\theta(y_i|x_i)}{\pi_\theta(y_r|x_i)}$ (task completion vs. refusal) is computed and softmax-normalized into weights. A safety-aligned model facing harmful prompts tends to prefer refusal → low likelihood ratio → low weight → harmful gradients are suppressed.
    - **Design Motivation**: Leverages safety knowledge already embedded during alignment as an implicit harmful sample detector — no explicit annotation of harmful samples is required.

3. **Perturbed-Model Refusal Training**

    - **Function**: Ensures that even when model parameters drift due to harmful fine-tuning, the low-weight mechanism remains effective.
    - **Mechanism**: During alignment, fine-tuning drift is simulated as $\theta_{\text{pert}} = \theta - \rho \frac{\nabla \mathcal{L}_{\text{harm}}}{\|\nabla \mathcal{L}_{\text{harm}}\|}$, and the perturbed model is trained to maintain high refusal probability on harmful prompts via $\mathcal{L}_{\text{refusal}}(\theta_{\text{pert}})$.
    - **Design Motivation**: Prevents harmful samples from gradually inflating their own weights during fine-tuning, which would cause the weighting mechanism to degrade.

### Theoretical Analysis
Propositions 4.2 and 4.3 provide a decomposition of loss changes under mini-batch updates. Via eNTK analysis, it is shown that when batch gradients are contributed solely by benign samples, the loss on harmful test samples remains unchanged (safety preserved) while the loss on benign test samples decreases (task learning proceeds).

## Key Experimental Results

### Main Results (Llama-2-7B, GSM8K + 20% harmful samples)

| Method | HS↓ | FA↑ | Notes |
|--------|-----|-----|-------|
| SFT | 23.94 | 10.90 | No defense |
| Vaccine | 23.60 | 11.70 | Alignment-stage |
| Lisa | 5.86 | 9.23 | Fine-tuning-stage, poor task performance |
| Booster | 9.06 | 16.27 | Alignment-stage |
| **Antibody** | **1.24** | **15.07** | Two-stage collaboration |

### Cross-Dataset Average

| Method | Avg. HS↓ | Avg. FA↑ |
|--------|----------|----------|
| Lisa | 15.29 | 60.97 |
| Booster | 19.04 | 65.20 |
| **Antibody** | **7.04** | **Competitive** |

Antibody's HS is more than 8 percentage points lower than the second-best method, Lisa.

### Ablation Study
- Removing flatness regularization → HS increases (harmful gradients are not sufficiently small during fine-tuning).
- Removing sample weighting → HS increases (harmful sample contributions are not suppressed).
- Removing perturbed refusal training → weighting mechanism degrades over extended fine-tuning.

### Key Findings
- The combination of flatness regularization and sample weighting is critical — each component alone is inferior to their combination.
- Likelihood-ratio weights (Figure 2) naturally separate harmful and benign samples during training without requiring explicit annotation.
- Antibody is especially effective at large data volumes (Figure 1) — while other methods suffer increased safety degradation as data grows, Antibody maintains low HS.

## Highlights & Insights
- The design logic of **dual gradient suppression** is exceptionally clear: the first layer (flat region) naturally reduces gradients → the second layer (weighting) further down-weights them → harmful influence is thoroughly eliminated.
- **Using the model's own safety knowledge as an implicit harmful sample detector** (via likelihood ratio) is an elegant design — no additional classifier, annotation, or knowledge of which samples are harmful is required.
- The eNTK-based theoretical analysis (Propositions 4.2–4.3) provides a rigorous account of how mini-batch weighted updates selectively affect different samples.
- The connection to Booster (Antibody reduces to Booster when $\lambda_t$ is constant) demonstrates the generality of the proposed framework.

## Limitations & Future Work
- Access to a harmful dataset $\mathcal{D}_{\text{harm}}$ during alignment is required — if the type of harmful content shifts, re-alignment may be necessary.
- Validation is conducted under LoRA fine-tuning; behavior under full-parameter fine-tuning remains unknown.
- The choice of refusal template $y_r$ may affect likelihood ratio computation — different refusal styles could lead to varying outcomes.
- Only a 20% harmful sample ratio is tested; robustness under higher ratios (50%+) remains to be verified.
- Computational overhead is higher than standard SFT due to additional likelihood ratio computation and inner-loop perturbation steps.

## Related Work & Insights
- **vs. Vaccine**: Vaccine enhances robustness via embedding perturbations, whereas Antibody uses loss flatness — the latter has clearer theoretical grounding.
- **vs. Booster**: Booster is a special case of Antibody (fixed $\lambda_t$); Antibody's adaptive $\lambda_t$ and additional weighting mechanism yield substantial additional gains.
- **vs. Lisa**: Lisa alternates between safety data and task data but cannot identify harmful samples within a batch; Antibody's weighting scheme enables sample-wise discrimination.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of flatness regularization and likelihood-ratio weighting shows considerable engineering ingenuity, though each individual technique is relatively standard.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Four downstream datasets × three model scales + ablation studies + theoretical analysis — highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Theoretical derivations are rigorous; the weight distribution visualization in Figure 2 is highly intuitive.
- **Value**: ⭐⭐⭐⭐⭐ Directly applicable to FTaaS security in practice; reducing HS from 15% to 7% represents a significant advancement.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Safety Subspaces are Not Linearly Distinct: A Fine-Tuning Case Study](safety_subspaces_are_not_linearly_distinct_a_fine-tuning_case_study.md)
- [\[AAAI 2026\] AlignTree: Efficient Defense Against LLM Jailbreak Attacks](../../AAAI2026/llm_alignment/aligntree_efficient_defense_against_llm_jailbreak_attacks.md)
- [\[ACL 2026\] SafeMERGE: Preserving Safety Alignment in Fine-Tuned Large Language Models via Selective Layer-Wise Model Merging](../../ACL2026/llm_alignment/safemerge_preserving_safety_alignment_in_fine-tuned_large_language_models_via_se.md)
- [\[ICLR 2026\] GuardAlign: Test-time Safety Alignment in Multimodal Large Language Models](guardalign_test-time_safety_alignment_in_multimodal_large_language_models.md)
- [\[NeurIPS 2025\] From Judgment to Interference: Early Stopping LLM Harmful Outputs via Streaming Content Monitoring](../../NeurIPS2025/llm_alignment/from_judgment_to_interference_early_stopping_llm_harmful_outputs_via_streaming_c.md)

<!-- RELATED:END -->
