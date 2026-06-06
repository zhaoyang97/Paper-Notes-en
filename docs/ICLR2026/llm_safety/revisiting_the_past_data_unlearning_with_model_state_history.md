---
title: >-
  [Paper Note] Revisiting the Past: Data Unlearning with Model State History
description: >-
  [ICLR 2026][LLM Safety][Machine Unlearning] This paper proposes MSA (Model State Arithmetic), an algorithm that leverages intermediate training checkpoints to construct "forgetting vectors" and removes the influence of s…
tags:
  - "ICLR 2026"
  - "LLM Safety"
  - "Machine Unlearning"
  - "Model State Arithmetic"
  - "Checkpoints"
  - "Forgetting Vector"
  - "Large Language Models"
date: 2026-05-08
content_hash: 307d0e98dbf36c1b
---

# Revisiting the Past: Data Unlearning with Model State History

**Conference**: ICLR 2026
**arXiv**: [2506.20941](https://arxiv.org/abs/2506.20941)  
**Code**: [https://github.com/mehrdadsaberi/MSA_unlearning](https://github.com/mehrdadsaberi/MSA_unlearning)  
**Area**: LLM Evaluation
**Keywords**: Machine Unlearning, Model State Arithmetic, Checkpoints, Forgetting Vector, Large Language Models

## TL;DR

This paper proposes MSA (Model State Arithmetic), an algorithm that leverages intermediate training checkpoints to construct "forgetting vectors" and removes the influence of specific data via parameter-space arithmetic. MSA consistently outperforms existing unlearning methods such as NPO, RMU, and GradDiff on the TOFU and RESTOR benchmarks, while maintaining model utility even without a retain set.

## Background & Motivation

### State of the Field
Large language models are trained on massive web-scale corpora, inevitably exposing them to copyrighted material, private information, and factually incorrect content. Eliminating the influence of such data via full retraining is computationally infeasible. Machine unlearning algorithms aim to remove the influence of specific data points at low cost while preserving the model's overall capabilities.

### Limitations of Prior Work
- **Gradient Ascent** (Yao et al., 2023): Maximizes loss on the forget set to induce forgetting, but is prone to model collapse.
- **NPO** (Zhang et al., 2024): A preference optimization approach that requires careful balancing between forgetting and retention.
- **RMU** (Li et al., 2024): Operates at the representation level and shows limited effectiveness in certain scenarios.
- **Task Vectors** (Ilharco et al., 2022): Computes directional vectors directly from the final model, but with limited effectiveness — directions extracted from a model that has already thoroughly learned the target data lack discriminative power.

**Core Observation**: All existing methods operate solely on the final model, while intermediate checkpoints saved during training — historical model states that have not yet been exposed to the forget targets — represent a valuable but untapped resource.

## Method

### Overall Architecture

The core idea of MSA is remarkably simple: leverage early checkpoints to more accurately estimate and reverse the influence of specific data.

**Inputs**:
- Final model $\theta_\mathcal{D}$ (trained on the full dataset)
- Intermediate checkpoint $C$ (weights $\theta_0$, not yet exposed to the forget targets)
- Forget dataset $\mathcal{D}_f$

### Key Designs

1. **Forgetting Vector Construction (Step 1)**

   The forget set $\mathcal{D}_f$ is fine-tuned on checkpoint $C$ for $e_f$ epochs to obtain $\theta_1$. The forgetting vector is defined as:

   $\vec{\theta}_f := \theta_1 - \theta_0$

   **Key Assumption**: Computing the forgetting vector using a checkpoint that has not yet been exposed to the forget targets more effectively captures the direction of data influence. This is more effective than computing task vectors from the final model, as the early checkpoint's "fresh response" to the forget data is more discriminative.

2. **Vector Application (Step 2)**

   The forgetting vector is applied to the final model:

   $\theta_{\text{unlearn}} = \theta_\mathcal{D} - \alpha \vec{\theta}_f$

   $\alpha$ controls the magnitude of the update.

3. **Optional Retain Vector**

   If a retain set $\mathcal{D}_r$ is available, a retain vector can be constructed by further fine-tuning to obtain $\theta_2$, yielding $\vec{\theta}_r = \theta_2 - \theta_0$:

   $\theta_{\text{unlearn}} = \theta_\mathcal{D} - \alpha \vec{\theta}_f + \beta \vec{\theta}_r$

   Notably, the retain set is sampled at the same size as the forget set, preserving computational efficiency.

4. **Checkpoint Selection**

   MSA is parameterized as $\text{MSA}_{\text{ckpt}, \alpha, \beta, e_f, e_r}$ and supports checkpoints at varying distances:
   - $\text{MSA}_{\text{instruct}}$: Model after instruction fine-tuning (before TOFU training)
   - $\text{MSA}_{\text{base}}$: Pretrained base model
   - $\text{MSA}_{\text{TOFU}}$: Final model (analogous to task vector)
   - $\text{MSA}_{\text{ckpt-XB}}$: A checkpoint from pretraining at X billion tokens

### Evaluation Contributions

To address the limitations of evaluation on the TOFU benchmark, three new GPT-4o-based metrics are proposed:
- **$\text{Acc}_{\text{forget}}$**: Fraction of forget-set questions for which the ground truth is *not* selected as most similar (higher = better forgetting)
- **$\text{Acc}_{\text{recover}}$**: Fraction of forget-set questions for which the ideal model output is selected as most similar (higher = better recovery)
- **$\text{Acc}_{\text{retain}}$**: Fraction of retain-set questions for which the ground truth or ideal model output is selected (higher = better retention)

These metrics focus on **factual content** rather than surface-level lexical overlap, making them more appropriate than ROUGE for evaluating unlearning.

## Key Experimental Results

### TOFU Forget01 (Forgetting 1% of Authors)

| Method | $\text{Acc}_{\text{forget}}$ ↑ | $\text{Acc}_{\text{recover}}$ ↑ | $\text{Acc}_{\text{retain}}$ ↑ | Model Utility ↑ |
|--------|------|------|------|------|
| Final (post-training model) | 0.15 | 0.13 | 0.89 | 0.48 |
| Ideal | 0.93 | 0.98 | 1.00 | 0.54 |
| **MSA_instruct** | **0.63** | **0.38** | 0.86 | 0.47 |
| MSA_base | 0.78 | 0.45 | 0.83 | 0.48 |
| NPO | 0.50 | 0.25 | 0.86 | 0.47 |
| RMU | 0.70 | 0.30 | 0.86 | 0.47 |
| GradDiff | 0.50 | 0.25 | 0.88 | 0.47 |

### TOFU Forget10 (Forgetting 10% of Authors, Harder Setting)

| Method | $\text{Acc}_{\text{forget}}$ ↑ | $\text{Acc}_{\text{recover}}$ ↑ | $\text{Acc}_{\text{retain}}$ ↑ | Model Utility ↑ |
|--------|------|------|------|------|
| **MSA_instruct** | **0.81** | **0.41** | 0.81 | 0.47 |
| MSA_base | 0.77 | 0.37 | 0.77 | 0.44 |
| NPO | 0.66 | 0.24 | 0.78 | 0.47 |
| RMU | 0.84 | 0.06 | 0.87 | 0.47 |
| GradDiff | 0.44 | 0.24 | 0.84 | 0.48 |

MSA's advantage becomes even more pronounced on the harder forget10 task.

### RESTOR Benchmark (Recovering Knowledge Overwritten by Misinformation)

| Method | RESTOR Accuracy ↑ | TOFU Probability ↑ | Model Utility ↑ |
|--------|-----------------|-------------------|----------------|
| Ideal (TOFU only) | 46.18 | 0.87 | 0.60 |
| **MSA_instruct** | **46.08** | 0.77 | 0.56 |
| MSA_base | 43.61 | 0.62 | 0.54 |
| NPO | 38.65 | 0.46 | 0.49 |
| RMU | 31.68 | 0.38 | 0.45 |
| GradDiff | 24.07 | 0.30 | 0.45 |

MSA nearly fully recovers the pre-misinformation accuracy (46.08 vs. 46.18).

### Ablation Study: Effect of Checkpoint Distance (OLMo-2-1B)

| Checkpoint | Tokens from Forget Data | $\text{Acc}_{\text{forget}}$ | $\text{Acc}_{\text{recover}}$ | $\text{Acc}_{\text{retain}}$ |
|-----------|------------------------|------|------|------|
| ckpt-3964B | ~21B tokens | 0.84 | 0.48 | 0.76 |
| ckpt-3146B | ~839B tokens | 0.81 | 0.45 | 0.77 |
| ckpt-2098B | ~1.9T tokens | 0.77 | 0.47 | 0.78 |
| ckpt-1049B | ~2.9T tokens | 0.73 | 0.44 | 0.77 |
| ckpt-210B | ~3.8T tokens | 0.39 | 0.24 | 0.85 |
| NPO | — | 0.84 | 0.39 | 0.64 |

**Key Finding**: Even when the checkpoint is **2 trillion tokens** away from the introduction of the forget targets, MSA remains effective and outperforms NPO.

### Key Findings
- Checkpoints closer in time to when the forget data was introduced yield better unlearning performance.
- MSA remains competitive even without a retain set (forget-only mode) — an important practical advantage.
- Computing forgetting vectors from the final model (analogous to task vectors) performs poorly, validating the necessity of using early checkpoints.
- Lexical overlap metrics such as ROUGE are inadequate for evaluating unlearning (high ROUGE scores may accompany factually incorrect outputs).
- The method generalizes effectively to 8B-scale models (Llama-3.1-8B-Instruct experiments).

## Highlights & Insights

1. **Algorithmic Simplicity**: The core procedure reduces to "fine-tune at a checkpoint → compute a difference vector → subtract from the final model," incurring minimal computational overhead without complex training procedures.
2. **New Utility of Checkpoints**: Intermediate checkpoints routinely saved during training (originally for fault tolerance) are repurposed for machine unlearning.
3. **Retain-Set-Free Operation**: Since retain sets are difficult to construct in practice, MSA's ability to operate without one significantly improves its applicability.
4. **Evaluation Contribution**: The three proposed GPT-4o-judge metrics more precisely assess factual-level forgetting and retention compared to ROUGE.
5. **Robustness Across Checkpoint Distances**: Checkpoints separated by trillions of tokens remain effective, suggesting that the forgetting vector captures the essential direction of data influence.

## Limitations & Future Work

- Access to intermediate training checkpoints is required — rendering the approach inapplicable to closed-source models.
- The quality of the forgetting vector depends on fine-tuning hyperparameters ($e_f$, learning rate, etc.) and the choice of $\alpha$ and $\beta$.
- Validation is currently limited to 1B and 8B models; effectiveness at larger scales (70B+) remains unknown.
- The effect of forget-target frequency in the training data has not been studied.
- Only data-level unlearning is evaluated; concept-level unlearning (e.g., "forgetting Harry Potter") is not addressed.
- The overhead of using a validation set to tune $\alpha$ and $\beta$ is not discussed in detail.

## Related Work & Insights

- **Task Vectors** (Ilharco et al., 2022): Pioneering work on directional vectors in parameter space, though direct application to unlearning proves insufficient.
- **NPO** (Zhang et al., 2024): Negative preference optimization; the strongest baseline in this setting.
- **RMU** (Li et al., 2024): Representation-level unlearning; effective on WMDP but underperforms on TOFU/RESTOR.
- **TOFU** (Maini et al., 2024): Fictitious-author unlearning benchmark.
- **RESTOR** (Rezaei et al., 2024): Knowledge-recovery-oriented unlearning benchmark.
- Broader Insight: The temporal information encoded in the model training trajectory is a valuable resource; parameter-space arithmetic can be generalized to other model editing tasks such as knowledge editing and capability suppression.

## Rating
- Novelty: ⭐⭐⭐⭐ — The use of checkpoints is elegant and effective; while parameter arithmetic is not new, the application context is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Two benchmarks, multiple checkpoints, diverse configurations, cross-model validation, and new evaluation metrics.
- Writing Quality: ⭐⭐⭐⭐⭐ — Ideas and experiments are clearly organized; the motivation for the proposed evaluation metrics is thoroughly explained.
- Value: ⭐⭐⭐⭐⭐ — The method is practical, simple, and effective, representing a significant contribution to the machine unlearning field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Redirection for Erasing Memory (REM): Towards a Universal Unlearning Method for Corrupted Data](redirection_for_erasing_memory_rem_towards_a_universal_unlearning_method_for_cor.md)
- [\[ICLR 2026\] Model Collapse Is Not a Bug but a Feature in Machine Unlearning for LLMs](model_collapse_is_not_a_bug_but_a_feature_in_machine_unlearning_for_llms.md)
- [\[ACL 2026\] From Domains to Instances: Dual-Granularity Data Synthesis for LLM Unlearning](../../ACL2026/llm_safety/from_domains_to_instances_dual-granularity_data_synthesis_for_llm_unlearning.md)
- [\[ACL 2026\] Before Forgetting, Learn to Remember: Revisiting Foundational Learning Failures in LVLM Unlearning Benchmarks](../../ACL2026/llm_safety/before_forgetting_learn_to_remember_revisiting_foundational_learning_failures_in.md)
- [\[ICLR 2026\] Erase or Hide? Suppressing Spurious Unlearning Neurons for Robust Unlearning](erase_or_hide_suppressing_spurious_unlearning_neurons_for_robust_unlearning.md)

</div>

<!-- RELATED:END -->
