---
title: >-
  [Paper Note] GradOT: Training-free Gradient-preserving Offsite-tuning for Large Language Models
description: >-
  [ACL 2025][LLM (Other)][Offsite-tuning] This paper provides the first systematic analysis of the Offsite-tuning problem from the perspective of optimization theory. It proposes the Gradient-preserving Compression Score (GCS) and designs the GradOT method. GradOT employs Dynamic Rank Decomposition (DRD) for MHA and Selective Channel Pruning (SCP) for MLP, simultaneously achieving performance preservation and privacy protection under training-free conditions.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Offsite-tuning"
  - "Gradient-preserving compression"
  - "Privacy protection"
  - "Model compression"
  - "Training-free method"
date: 2026-05-08
content_hash: 935e51f35954332e
---

# GradOT: Training-free Gradient-preserving Offsite-tuning for Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2507.04455](https://arxiv.org/abs/2507.04455)  
**Area**: LLM Efficiency  
**Keywords**: Offsite-tuning, Gradient-preserving compression, Privacy protection, Model compression, Training-free method

## TL;DR

This paper provides the first systematic analysis of the Offsite-tuning problem from the perspective of optimization theory. It proposes the Gradient-preserving Compression Score (GCS) and designs the GradOT method. GradOT employs Dynamic Rank Decomposition (DRD) for MHA and Selective Channel Pruning (SCP) for MLP, simultaneously achieving performance preservation and privacy protection under training-free conditions.

## Background & Motivation

Fine-tuning large language models typically requires data and models to co-exist in the same location, posing privacy risks for both data owners and model owners. Offsite-tuning (OT) is a promising solution: the model owner compresses the original model into a weaker "emulator," the data owner fine-tunes an adapter on the emulator and returns it, and the model owner plugs the adapter back into the original model.

Existing OT methods suffer from two core limitations:

**Lack of theoretical analysis**: Existing methods (OT, CRaSh, ScaleOT) rely heavily on empirical validation without a systematic theoretical foundation.

**High computational overhead**: The original OT method requires knowledge distillation, which is computationally expensive and difficult to scale to large LLMs.

## Method

### Overall Architecture

The core idea of GradOT is that **a good emulator should maintain adapter gradient consistency while introducing a sufficiently large loss discrepancy** (the former guarantees performance, while the latter protects privacy).

Workflow:
1. The model owner compresses the intermediate layers of the original model using gradient-preserving compression to generate an emulator.
2. The emulator is sent to the data owner along with the adapter.
3. The data owner fine-tunes the adapter on the emulator.
4. The fine-tuned adapter is returned to the model owner and integrated with the original intermediate layers to form the final plug-in model.

### Key Designs

**Gradient-preserving Compression Score (GCS)**:

$$\text{GCS}(\delta_i) = \underbrace{||\frac{\partial^2 \ell}{\partial w_i^2} \delta_i||_1}_{\text{Performance Preservation}} \underbrace{- \lambda \frac{\partial \ell}{\partial w_i} \odot \delta_i}_{\text{Privacy Protection}}$$

- Term 1 (Hessian term): Minimizes the gradient shift introduced by compression, ensuring the adapter's gradient direction on the emulator matches that on the original model.
- Term 2 (First-order gradient term): Maximizes the loss discrepancy to protect privacy.
- $\lambda$ controls the privacy-utility trade-off.

**Dynamic Rank Decomposition (DRD)** — for MHA:
- Performs truncated SVD on attention layer weights.
- Selects the indices of singular values to retain based on the GCS score, rather than the conventional sorting by singular value magnitudes.
- Always selects the top 5% of the rank to ensure evaluation accuracy.

**Selective Channel Pruning (SCP)** — for MLP:
- Utilizes the property that the intermediate dimension of MLP is significantly larger than its input/output dimensions.
- Searches for the optimal subset of channels in the intermediate dimension while minimizing the GCS score of the up/down projection matrices.

**Hessian Approximation**: Employs Kronecker-factored Approximate Curvature (KFAC) of the Fisher information matrix to avoid directly computing the full $P \times P$ Hessian.

## Key Experimental Results

### Main Results

Evaluation of Avg. Accuracy across 8 datasets on OPT-1.3B:

| Method | Category | Plug-in Avg. | $\Delta$ (Emu.FT $\rightarrow$ Plug-in) |
|------|------|-------------|----------------------|
| Full FT (Upper Bound) | - | 49.9 | - |
| OT (w/ distillation) | Post-Training | 49.0 | 2.4 |
| ScaleOT | Post-Training | 49.9 | 3.7 |
| OT† (Training-free) | Training-free | 46.5 | 3.3 |
| CRaSh | Training-free | 48.4 | 4.8 |
| **GradOT** | **Training-free** | **49.8** | **4.8** |

Key findings:
- GradOT's Plug-in performance (49.8) almost catches up with Full FT (49.9) and ScaleOT (49.9) which requires post-training.
- GradOT's $\Delta$ is as high as 4.8, on par with CRaSh, demonstrating strong privacy protection capabilities.
- GradOT's emulator shows the lowest zero-shot performance (27.3), further validating its privacy-preserving effectiveness.

### Key Findings

1. **Balance of Performance and Privacy**: GradOT achieves excellent results both in Plug-in model performance and in privacy protection (low Emulator ZS performance = better privacy).
2. **Theory-driven over Empirically-driven**: As a theoretically grounded training-free method, GradOT significantly outperforms the training-free baseline OT† (46.5 vs 49.8).
3. **Effectiveness of Gradient Preservation**: The performance gains from Emulator FT to Plug-in ($\Delta=4.8$) confirm that the adapter trained on the emulator can transfer effectively to the original model.
4. **Rationality of Component-Specific Compression**: Employing different compression strategies (rank decomposition vs. channel pruning) with independent $\lambda$ parameters for MHA and MLP outperforms a unified strategy.

## Highlights & Insights

- **First Formal Theoretical Analysis of the OT Problem**: Clearly decomposes the OT objective into two optimization terms: performance preservation and privacy protection.
- **Generality of the GCS Score**: The weight and gradient-based analysis does not rely on specific model architectures, making it applicable to various Transformer models.
- **Training-free Advantage**: Compared to methods requiring post-training steps like knowledge distillation, GradOT requires only a single forward-backward pass to complete compression, providing significantly higher computational efficiency.
- **Insights on Component-Specific Compression**: Suggests that attention layers (MHA) are more suitable for rank compression (retaining core attention patterns), while MLP layers are key candidates for channel pruning (due to higher redundancy in central intermediate dimensions).

## Limitations & Future Work

- The main experiments are verified only on OPT-1.3B, lacking a comprehensive scaling evaluation on larger models (7B+).
- The approximation accuracy of the Hessian using KFAC may degrade in deeper networks.
- The choice of the $\lambda$ parameter relies on empirical tuning and lacks an adaptive mechanism.
- The current setup only considers adapters at the input and output layers of the model, leaving out more flexible adapter options such as LoRA.

## Related Work & Insights

- **Offsite-Tuning Family**: OT (Xiao et al. 2023) utilizes LayerDrop and knowledge distillation; CRaSh (Zhang et al. 2023) replaces layers with repeated shared layers; ScaleOT (Yao et al. 2025) employs reinforcement learning to estimate layer-wise importance.
- **Privacy Protection**: Federated Learning (protecting data privacy) vs. OT (protecting both model and data privacy).
- **Model Compression Theory**: Fisher Information Matrix and KFAC approximation used to estimate weight importance.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Practical Impact | ⭐⭐⭐⭐ |
| Overall Rating | 7.5/10 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Robust Utility-Preserving Text Anonymization Based on Large Language Models](robust_utility-preserving_text_anonymization_based_on_large_language_models.md)
- [\[ACL 2025\] PiFi: Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models](plugin_finetuning_bridge.md)
- [\[ACL 2025\] GORP: Continual Gradient Low-Rank Projection Fine-Tuning for LLMs](gorp_continual_gradient_projection.md)
- [\[ACL 2025\] Training-free LLM Merging for Multi-task Learning](training-free_llm_merging_for_multi-task_learning.md)
- [\[ACL 2025\] Gradient-Adaptive Policy Optimization: Towards Multi-Objective Alignment of Large Language Models](gapo_multi_objective_alignment.md)

</div>

<!-- RELATED:END -->
