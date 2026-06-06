---
title: >-
  [Paper Note] Preventing Safety Drift in Large Language Models via Coupled Weight and Activation Constraints
description: >-
  [ACL 2026][LLM Safety][Safety Drift] This paper proposes CWAC, which simultaneously constrains weight update directions and safety-critical activation features during fine-tuning. It theoretically and experimentally demo…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Safety Drift"
  - "Harmful Fine-tuning"
  - "Sparse Autoencoders"
  - "Weight Subspace"
  - "Activation Constraints"
date: 2026-05-08
content_hash: 0cc389d1e876df76
---

# Preventing Safety Drift in Large Language Models via Coupled Weight and Activation Constraints

**Conference**: ACL 2026  
**arXiv**: [2604.12384](https://arxiv.org/abs/2604.12384)  
**Code**: No public code  
**Area**: Model Compression / LLM Safety  
**Keywords**: Safety Drift, Harmful Fine-tuning, Sparse Autoencoders, Weight Subspace, Activation Constraints

## TL;DR

This paper proposes CWAC, which simultaneously constrains weight update directions and safety-critical activation features during fine-tuning. It theoretically and experimentally demonstrates that constraining either weights or activations alone is insufficient to prevent LLM safety drift.

## Background & Motivation

**Background**: Safety-aligned models are fragile during downstream fine-tuning. Even when the fine-tuning data appears to be benign tasks such as sentiment classification, news categorization, or mathematical reasoning, the original refusal capabilities can be compromised. Existing defenses include freezing safety layers, incorporating safety samples, limiting weight drift, activation steering, or post-finetuning repairs.

**Limitations of Prior Work**: Many methods focus on a single level: either restricting parameter updates to prevent the model from deviating from original weights or constraining activations to maintain safe internal representations on harmful inputs. The paper points out that both single-point constraints have vulnerabilities, as the model can generate bypasses through the unconstrained side.

**Key Challenge**: Safety behavior results from the coupled outcome of weights and activations. Minimal changes in weights do not guarantee that activations will not be shifted by the fine-tuned task distribution; similarly, stable activations on reference samples do not guarantee that weights will not cause dangerous outputs on unseen harmful inputs.

**Goal**: To construct a defense mechanism during fine-tuning that preserves the base model's refusal capability for harmful prompts while maintaining downstream task accuracy, ensuring robustness across multiple models, tasks, and scenarios with harmful sample injection.

**Key Insight**: The authors first use a first-order approximation to decompose layer output drift into a weight drift term and an activation drift term. Based on this, they design a dual protection mechanism consisting of a "weight safety subspace" and "activation safety constraints."

**Core Idea**: Project weight gradients onto a safety subspace that does not affect refusal outputs for harmful prompts, while simultaneously using Sparse Autoencoders (SAEs) to lock refusal-related sparse activation features. This ensures that fine-tuning only adapts to tasks within regions that do not destroy safety representations.

## Method

### Overall Architecture

The input for CWAC includes downstream task data, a set of harmful prompts, pre-computed weight projection matrices for each layer, and safety-critical SAE features of the base model on harmful prompts. During fine-tuning, the model minimizes task loss while re-forwarding harmful prompts to extract safety features, pulling current safety features back to the base model's features. Simultaneously, gradients of specific FFN output projection layers are right-multiplied by a projection matrix to ensure weight updates fall within the safety subspace.

### Key Designs

1.  **Safety Drift Decomposition**:
    - **Function**: Explains why constraining either weights or activations alone is insufficient.
    - **Mechanism**: For a layer output $y=f(Wh)$, the drift after fine-tuning is approximated as $$\Delta y\approx f'(W_0h_0)\odot(W_0\Delta h+\Delta W h_0)$$. Here, $W_0\Delta h$ is the drift caused by activation changes, and $\Delta W h_0$ is the drift caused by weight changes. If only $\Delta W$ is constrained, activations can still shift; if only $\Delta h$ is constrained, weights can still cause drift on unseen inputs.
    - **Design Motivation**: This decomposition transforms "safety drift" from an empirical phenomenon into an analyzable upper bound on error, providing a theoretical basis for the dual constraint design.

2.  **SAE-based Activation Safety Constraint**:
    - **Function**: Preserves internal safety features related to refusal during the fine-tuning process.
    - **Mechanism**: The authors train TopK Sparse Autoencoders to decompose residual stream activations into sparse, interpretable features. The SAE training corpus consists of approximately 100 million tokens, with 30% from OpenWebText2 and 70% from harmful prompts correctly refused by the base model. Before fine-tuning, safety-critical features $z^{baseline}$ on harmful prompts are recorded. During fine-tuning, a safety loss is added: $L_{safety}=\frac{1}{B}\sum_b\|z_b^{current}-z_b^{baseline}\|^2$, which only constrains top safety-critical latents.
    - **Design Motivation**: Directly constraining the entire activation vector would hinder downstream task adaptation. SAEs allow the method to lock only the sparse dimensions related to refusal, decoupling safety preservation from task learning as much as possible.

3.  **Weight Safety Subspace Projection**:
    - **Function**: Restricts weight updates from altering the model's safe outputs on harmful prompts.
    - **Mechanism**: For each FFN output projection layer, the input matrix $X_l$ is collected when the base model processes harmful prompts, with the goal that updates satisfy $\Delta W_lX_l\approx0$. Since processing $X_l$ directly is costly, the covariance $C_l=X_lX_l^T$ is calculated, and SVD is performed to retain directions corresponding to small eigenvalues to form $\hat U_l$. The projection matrix is defined as $\Pi_l=\hat U_l\hat U_l^T$. During fine-tuning, gradient updates are projected as $\Delta W_l\leftarrow\Delta W_l\Pi_l$.
    - **Design Motivation**: Directions with small eigenvalues have the least impact on safe outputs for harmful prompts. Restricting updates to these directions preserves the space for downstream task learning while minimizing damage to refusal behavior.

### Loss & Training

The total objective is $L_{total}=L_{task}+\lambda L_{safety}$, with safety subspace projection applied to gradients of FFN down_proj before updates. Default experiments use full-parameter fine-tuning, AdamW, 3 epochs, batch size of 1, maximum sequence length of 512, and a learning rate of $2\times10^{-5}$, with 5,000 samples per benign task. The activation constraint retains the top 100 safety-critical latents with $\lambda=0.5$. Offline costs include ~6.5 hours for SAE training and ~18 minutes for SVD pre-computation on Llama-2-7B. Fine-tuning takes about 44-46 minutes per epoch, an overhead of less than 10% compared to standard SFT (42 minutes).

## Key Experimental Results

### Main Results

**Average Performance on SST-2 / AGNEWS / GSM8K: Higher FA is better, lower HS is safer**

| Model | Method | Avg FA↑ | Avg HS↓ |
|------|------|----------|----------|
| Llama-2-7B | SFT | 85.04 | 52.45 |
| Llama-2-7B | ASFT | 78.12 | 18.88 |
| Llama-2-7B | CWAC | 85.12 | 10.81 |
| Llama-3-8B | SFT | 87.16 | 66.03 |
| Llama-3-8B | ASFT | 73.75 | 17.64 |
| Llama-3-8B | CWAC | 87.78 | 9.77 |
| Mistral-7B | SFT | 85.75 | 64.45 |
| Mistral-7B | ASFT | 74.28 | 33.70 |
| Mistral-7B | CWAC | 85.61 | 24.22 |
| Gemma-2-9B | SFT | 90.74 | 42.23 |
| Gemma-2-9B | ASFT | 86.43 | 29.49 |
| Gemma-2-9B | CWAC | 91.59 | 10.05 |

**Generalization Experiments on PubMedQA and AlpacaEval**

| Model | Method | FA↑ | HS↓ | AE↑ |
|------|------|-----|-----|-----|
| Llama-2-7B | SFT | 93.81 | 46.75 | 34.51 |
| Llama-2-7B | CWAC | 94.52 | 7.24 | 34.37 |
| Llama-3-8B | SFT | 95.24 | 53.10 | 38.07 |
| Llama-3-8B | CWAC | 95.52 | 8.65 | 36.75 |
| Mistral-7B | SFT | 64.53 | 60.72 | 28.63 |
| Mistral-7B | CWAC | 90.72 | 15.39 | 30.64 |
| Gemma-2-9B | SFT | 93.45 | 48.97 | 42.53 |
| Gemma-2-9B | CWAC | 95.20 | 12.55 | 43.57 |

### Ablation Study

**Robustness under different harmful sample injection ratios in SST-2, Llama-2-7B**

| Method | p=0.05 HS↓ | p=0.1 HS↓ | p=0.2 HS↓ | p=0.5 HS↓ | Avg HS↓ | Avg FA↑ |
|------|------------|-----------|-----------|-----------|----------|----------|
| SFT | 72.70 | 78.92 | 74.90 | 82.02 | 73.17 | 94.07 |
| ASFT | 38.50 | 39.90 | 43.60 | 45.82 | 38.22 | 93.22 |
| CWAC | 10.50 | 20.03 | 22.57 | 30.78 | 18.73 | 94.35 |

**Component Ablation for Weight and Activation Constraints, Average Results**

| Model | Method | Avg FA↑ | Avg HS↓ |
|------|------|----------|----------|
| Llama-2-7B | Weight-only | 80.11 | 17.82 |
| Llama-2-7B | Activation-only | 82.91 | 19.57 |
| Llama-2-7B | CWAC | 84.39 | 12.75 |
| Llama-3-8B | Weight-only | 81.04 | 17.11 |
| Llama-3-8B | Activation-only | 80.45 | 19.02 |
| Llama-3-8B | CWAC | 85.89 | 10.89 |
| Gemma-2-9B | Weight-only | 82.79 | 17.01 |
| Gemma-2-9B | Activation-only | 82.51 | 19.97 |
| Gemma-2-9B | CWAC | 87.67 | 10.07 |

### Key Findings
- CWAC maintains or improves downstream FA across four 7B-9B models while significantly reducing HS; most notably, for Llama-3-8B, average HS drops from 66.03 (SFT) to 9.77.
- Compared to defenses like ASFT, SafeInstr, and SPPFT, the advantage of CWAC is not just higher safety, but a better balance between safety and task performance.
- When the ratio of harmful samples increases to 0.5, CWAC's HS is 30.78, which is still lower than ASFT's 45.82, while maintaining FA around 94.06.
- Component ablation shows that both weight-only and activation-only methods are effective, but full CWAC is required to consistently achieve the lowest HS, supporting the core argument that "coupled constraints are necessary."

## Highlights & Insights
- The most valuable part of the paper is the safety drift decomposition: it explains why "weight preservation" and "activation preservation" individually leave gaps in defense, showing that CWAC is more than just an engineering combination.
- The choice to lock only safety-critical latents via SAEs instead of the entire activation vector is crucial; otherwise, safety regularization would likely impede downstream task learning.
- Selecting FFN down_proj for projection is reasonable as it serves as a bottleneck for FFN writes to the residual stream, making it more direct than intervening in gate/up projections and more efficient than full-layer projection.
- CWAC's form is well-suited for integration with PEFT: although the paper defaults to full-parameter fine-tuning, the concept of "projection updates + activation regularization" can be transferred to LoRA subspaces or adapter updates.

## Limitations & Future Work
- The method requires white-box access to weights, gradients, and intermediate activations, making it inapplicable to closed-source API models.
- Activation constraints depend on SAE quality; poor SAE reconstruction or non-interpretable safety features would affect the effectiveness of protection.
- Experiments primarily cover 7B-9B instruction-tuned models and have not yet verified stability on larger models, MoE models, or different alignment pipelines.
- Current safety evaluations focus on explicit harmful prompts and refusal capabilities, with insufficient coverage of indirect jailbreaks, hidden violations, or safety drift within agentic toolchains.

## Related Work & Insights
- **vs ASFT**: ASFT restricts fine-tuning drift through safety direction anchoring; this work adds activation-level SAE regularization to cover activation bypasses that weight constraints cannot handle.
- **vs SPPFT / Safety Layer Freezing**: Freezing safety layers is simple but may lose task adaptability. CWAC allows updates but projects them in directions that have minimal impact on safety prompts.
- **vs activation steering**: Many steering methods modify activations during inference, whereas CWAC use activation regularization during training to maintain refusal features, making it more suitable for scenarios requiring fine-tuned deployment.
- **Insights**: If capabilities such as privacy, honesty, and copyright compliance are mapped to interpretable SAE features, a similar "weight subspace + activation locking" framework could be extended to more alignment properties.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of theoretical decomposition and dual-layer constraints is solid; despite borrowing from SAE and subspace projection, the integrated objective is clear.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across four models, multiple downstream tasks, generalization tasks, harmful ratios, learning rates, and component ablations.
- Writing Quality: ⭐⭐⭐⭐ The methodological chain is clear, and the derivation serves the design, though some experimental details and table explanations are quite dense.
- Value: ⭐⭐⭐⭐ Directly valuable for the safe fine-tuning of open-source models, particularly for enterprises or research deployments needing to preserve refusal capabilities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Compiling Activation Steering into Weights via Null-Space Constraints for Stealthy Backdoors](compiling_activation_steering_into_weights_via_null-space_constraints_for_stealt.md)
- [\[ACL 2026\] SafetyALFRED: Evaluating Safety-Conscious Planning of Multimodal Large Language Models](safetyalfred_evaluating_safety-conscious_planning_of_multimodal_large_language_m.md)
- [\[ACL 2026\] MUSE: A Run-Centric Platform for Multimodal Unified Safety Evaluation of Large Language Models](muse_a_run-centric_platform_for_multimodal_unified_safety_evaluation_of_large_la.md)
- [\[ACL 2026\] AutoRAN: Automated Hijacking of Safety Reasoning in Large Reasoning Models](autoran_automated_hijacking_of_safety_reasoning_in_large_reasoning_models.md)
- [\[ACL 2026\] Seeing No Evil: Blinding Large Vision-Language Models to Safety Instructions via Adversarial Attention Hijacking](seeing_no_evil_blinding_large_vision-language_models_to_safety_instructions_via_.md)

</div>

<!-- RELATED:END -->
