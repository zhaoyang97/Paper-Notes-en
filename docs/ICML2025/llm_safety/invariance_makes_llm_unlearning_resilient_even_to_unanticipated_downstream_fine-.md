---
title: >-
  [Paper Note] Invariance Makes LLM Unlearning Resilient Even to Unanticipated Downstream Fine-Tuning
description: >-
  [ICML2025][LLM Safety][LLM unlearning] This work introduces Invariant Risk Minimization (IRM) into the LLM unlearning framework and proposes the ILU regularization method. This prevents forgotten knowledge from being recovered during subsequent downstream fine-tuning and can generalize to multiple unseen downstream tasks using only a single irrelevant fine-tuning dataset.
tags:
  - "ICML2025"
  - "LLM Safety"
  - "LLM unlearning"
  - "invariance regularization"
  - "IRM"
  - "fine-tuning robustness"
  - "knowledge unlearning"
date: 2026-05-08
content_hash: 2836678e3cbb3a45
---

# Invariance Makes LLM Unlearning Resilient Even to Unanticipated Downstream Fine-Tuning

**Conference**: ICML2025  
**arXiv**: [2506.01339](https://arxiv.org/abs/2506.01339)  
**Code**: [OPTML-Group/Unlearn-ILU](https://github.com/OPTML-Group/Unlearn-ILU)  
**Area**: AI Safety  
**Keywords**: LLM unlearning, invariance regularization, IRM, fine-tuning robustness, knowledge unlearning

## TL;DR

This work introduces Invariant Risk Minimization (IRM) into the LLM unlearning framework and proposes the ILU regularization method. This prevents forgotten knowledge from being recovered during subsequent downstream fine-tuning and can generalize to multiple unseen downstream tasks using only a single irrelevant fine-tuning dataset.

## Background & Motivation

LLM unlearning aims to remove specific knowledge (e.g., harmful content, private data) from pretrained models while retaining general model capabilities. Although existing methods (e.g., NPO, RMU) are effective immediately after unlearning, they face a severe vulnerability: **downstream fine-tuning can unexpectedly recover the forgotten knowledge**—even if the fine-tuning data is completely unrelated to the unlearned content.

Specifically, after unlearning biosecurity knowledge in Zephyr-7B using NPO/RMU on the WMDP benchmark, fine-tuning for only a few epochs on GSM8K (math) or AGNews (news classification) rapidly degrades the unlearning performance. The forget quality drops from ~0.68 to ~0.37, almost reverting to pre-unlearning levels. This indicates that existing methods only achieve "superficial unlearning," and the knowledge is not truly removed.

Core Problem: How to make the unlearning operation **invariant** to subsequent arbitrary fine-tuning, ensuring the unlearning effect is persistent?

## Method

### Standard LLM Unlearning Framework

The standard unlearning optimization objective is:

$$\min_{\theta} \ell_u(\theta; \mathcal{D}_f, \mathcal{D}_r) = \ell_f(\theta; \mathcal{D}_f) + \gamma \ell_r(\theta; \mathcal{D}_r)$$

where $\ell_f$ is the unlearning loss (on the forget set $\mathcal{D}_f$), $\ell_r$ is the retain loss (on the retain set $\mathcal{D}_r$), and $\gamma$ balances the two. The unlearning loss can adopt NPO (Negative Preference Optimization) or RMU (Representation Misdirection Unlearning).

### ILU: Invariant LLM Unlearning

Inspired by Invariant Risk Minimization (IRM), downstream fine-tuning is treated as "training environments," and invariance regularization is incorporated into the unlearning optimization to maintain the stability of model parameters under fine-tuning perturbations. The relaxed form of IRMv1 is:

$$\min_{\theta} \ell_u(\theta) + \lambda \sum_{i=1}^{N} \| \nabla_{w|w=1} \ell_i(w \circ \theta; \mathcal{D}_i) \|_2^2$$

where $\lambda > 0$ is the regularization coefficient, and $\nabla_{w|w=1} \ell_i$ denotes the gradient with respect to a dummy scalar predictor $w$ at $w=1$, which penalizes non-stationarity.

### Key Findings: A Single Irrelevant Fine-Tuning Dataset Suffices

Experiments demonstrate that using only a single fine-tuning dataset $\mathcal{D}$ (e.g., GSM8K) unrelated to the unlearning task for invariance regularization is sufficient to generalize to various unseen downstream fine-tuning scenarios. The final practical formulation is simplified as:

$$\min_{\theta} \ell_u(\theta) + \lambda \| \nabla_{w|w=1} \ell(w \circ \theta; \mathcal{D}) \|_2^2$$

In contrast, using multiple fine-tuning datasets (ILU(Multi)) provides no additional benefits due to increased optimization complexity. Using $\mathcal{D} = \mathcal{D}_f$ (the forget set itself) is also suboptimal because the unlearning objective (decreasing accuracy) conflicts with the invariance regularization (which may increase accuracy to satisfy stationarity).

### Task Vector Analysis

The unlearning direction is defined as $\tau_u = \theta_u - \theta_o$, and the fine-tuning direction as $\tau_{ft} = \theta_{ft} - \theta_o$. For NPO, the post-fine-tuning direction deviates from the unlearning direction: $\cos(\angle(\tau_{\text{NPO}\to\text{ft}}, \tau_{\text{NPO}})) = -0.41$. In contrast, ILU maintains near-orthogonality: $\cos(\angle(\tau_{\text{ILU}\to\text{ft}}, \tau_{\text{ILU}})) = 0.09$, indicating that ILU effectively decouples fine-tuning effects from the unlearning direction.

## Key Experimental Results

**Benchmark Setup**: WMDP dataset, Zephyr-7B-beta model, unlearning biosecurity/cybersecurity knowledge. Evaluation metrics: FQ (Forget Quality, 1 - accuracy, higher is better unlearning), RA (Robust Accuracy, average FQ after fine-tuning), FA (Fine-tuning Accuracy, accuracy on downstream tasks).

### Main Results on WMDP (Table 2)

| Method | FQ↑ | MMLU↑ | Average RA↑ | Average FA↑ |
|------|------|-------|----------|----------|
| Original | 0.36 | 58.15 | 0.37 | 82.50 |
| RMU | 0.68 | 57.46 | 0.42 | 82.43 |
| **RMU+ILU(GSM8K)** | **0.68** | **57.64** | **0.65** | **82.32** |
| NPO | 0.52 | 56.69 | 0.47 | 80.30 |
| **NPO+ILU(GSM8K)** | **0.56** | **55.50** | **0.56** | **81.18** |

- The average RA of RMU+ILU increases by **23 percentage points** (0.42 $\rightarrow$ 0.65) compared to RMU.
- The average RA of NPO+ILU increases by **9 percentage points** (0.47 $\rightarrow$ 0.56) compared to NPO.
- FA does not decrease but instead improves, as invariance regularization enhances the smoothness of the loss landscape.

### Comparison with TAR and LAT (LLaMA-3-8B, Table 4)

| Method | Average RA↑ | Average FA↑ | Training Time |
|------|----------|----------|----------|
| NPO | 0.61 | 85.54 | 15.3 min |
| LAT | 0.64 | 85.38 | 21.2 min |
| TAR | 0.70 | 86.15 | **7441.9 min** |
| **NPO+ILU** | **0.70** | 85.81 | 118.2 min |

ILU achieves comparable robustness to TAR, but with a **63x improvement in computational efficiency**.

### Resistance to Re-learning Attacks (Table 3, Fine-tuning 1 Epoch with 60 Unlearning Samples)

| Method | FQ (No Attack) | FQ (Under Attack) | Drop |
|------|------------|------------|------|
| RMU | 0.68 | 0.36 | 0.32 |
| RMU+ILU | 0.68 | 0.54 | **0.14** |
| NPO | 0.52 | 0.37 | 0.15 |
| NPO+ILU | 0.56 | 0.50 | **0.06** |

### Hyperparameter $\lambda$ Sensitivity

If $\lambda$ is too large ($>0.1$), it hurts FQ, while if it is too small ($\sim 0.05$), it fails to regularize effectively. The paper suggests tuning $\lambda$ within a reasonable range.

## Highlights & Insights

1. **Novel Theoretical Perspective**: This work is the first to introduce the concept of IRM invariance into LLM unlearning, building a bridge between two seemingly unrelated fields.
2. **Minimalist and Efficient Design**: It requires only a single irrelevant fine-tuning dataset to generalize to various unseen downstream tasks, avoiding the high computational overhead of meta-learning.
3. **Plug-and-Play**: As a regularization term, it can be seamlessly integrated into existing unlearning methods such as NPO/RMU.
4. **Intuitive Task Vector Analysis**: The cosine similarity visualization clearly explains why ILU works—by keeping the unlearning direction decoupled from the fine-tuning direction.
5. **Supplementary Experiments on MUSE**: The method is equally effective on the Harry Potter and BBC datasets, keeping VerbMem at 0.

## Limitations & Future Work

1. **Only Addressing Fine-Tuning Robustness**: It does not address robustness against other attack vectors such as quantization attacks or prompt injection.
2. **Tuning of $\lambda$ Required**: It is sensitive to regularization strength and requires a validation set for selection.
3. **Limited Model Scale**: It has only been validated on 7B/8B models, with the performance on larger models (70B+) remaining unexplored.
4. **Lack of Theoretical Guarantees**: IRMv1 itself is a relaxation of the original IRM, lacking strict convergence proof.
5. **Single Type of Forget Set**: The verification is mainly on WMDP harmful knowledge, with unknown performance on private data unlearning scenarios.
6. **Re-learning Attack Resistance Inferior to SAM**: Under the extreme scenario of fine-tuning with the forget set itself, the SAM method still outperforms ILU.

## Related Work & Insights

- **NPO** (Zhang et al., 2024): Negative Preference Optimization, treating the forget set as negative samples.
- **RMU** (Li et al., 2024): Representation Misdirection Unlearning, aligning the representation of forgotten data with random vectors.
- **TAR** (Tamirisa et al., 2024): Tamper-Resistant safety guardrails, a meta-learning method that is effective but extremely slow.
- **LAT** (Sheshadri et al., 2024): Latent Adversarial Training, perturbing intermediate activations to suppress undesirable behavior.
- **IRM** (Arjovsky et al., 2019): Invariant Risk Minimization, the core theoretical foundation of this work.
- **Insight**: The concept of invariance can similarly be extended to general safety alignment operations, enhancing the robustness of alignment against subsequent fine-tuning.

## Rating

- Novelty: ⭐⭐⭐⭐ — The cross-disciplinary perspective of IRM + unlearning is novel, and the concept is clear.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Evaluated across 6 downstream tasks, 2 benchmarks, and multiple baselines, with a comprehensive ablation study.
- Writing Quality: ⭐⭐⭐⭐ — Rich in figures and tables, with intuitive task vector analysis.
- Value: ⭐⭐⭐⭐⭐ — Addresses a core pain point in LLM unlearning, being both plug-and-play and highly efficient.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] TuCo: Measuring the Contribution of Fine-Tuning to Individual Responses of LLMs](tuco_measuring_the_contribution_of_fine-tuning_to_individual_responses_of_llms.md)
- [\[ICML 2025\] Vulnerability-Aware Alignment: Mitigating Uneven Forgetting in Harmful Fine-Tuning](vulnerability-aware_alignment_mitigating_uneven_forgetting_in_harmful_fine-tunin.md)
- [\[ICML 2025\] DRAGON: Guard LLM Unlearning in Context via Negative Detection and Reasoning](dragon_guard_llm_unlearning_in_context_via_negative_detection_and_reasoning.md)
- [\[ACL 2025\] Towards Context-Robust LLMs: A Gated Representation Fine-tuning Approach](../../ACL2025/llm_safety/towards_context-robust_llms_a_gated_representation_fine-tuning_approach.md)
- [\[NeurIPS 2025\] Adaptive LoRA Experts Allocation and Selection for Federated Fine-Tuning](../../NeurIPS2025/llm_safety/adaptive_lora_experts_allocation_and_selection_for_federated_fine-tuning.md)

</div>

<!-- RELATED:END -->
