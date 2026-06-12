---
title: >-
  [Paper Note] Geminio: Language-Guided Gradient Inversion Attacks in Federated Learning
description: >-
  [ICCV 2025][LLM Safety][Federated Learning] This paper proposes Geminio, the first gradient inversion attack (GIA) leveraging vision-language models (VLMs) to enable natural language-guided targeted reconstruction. A mal…
tags:
  - "ICCV 2025"
  - "LLM Safety"
  - "Federated Learning"
  - "Gradient Inversion Attack"
  - "Vision-Language Model"
  - "Privacy Attack"
  - "Natural Language Guidance"
date: 2026-05-08
content_hash: c268a859750aec7f
---

# Geminio: Language-Guided Gradient Inversion Attacks in Federated Learning

**Conference**: ICCV 2025
**arXiv**: [2411.14937](https://arxiv.org/abs/2411.14937)  
**Code**: [GitHub](https://github.com/HKU-TASR/Geminio)  
**Area**: AI Security
**Keywords**: Federated Learning, Gradient Inversion Attack, Vision-Language Model, Privacy Attack, Natural Language Guidance

## TL;DR
This paper proposes Geminio, the first gradient inversion attack (GIA) leveraging vision-language models (VLMs) to enable natural language-guided targeted reconstruction. A malicious server can specify the type of data to steal via natural language queries, precisely locating and reconstructing semantically matching private samples from large-batch gradients, without disrupting normal FL model training.

## Background & Motivation
Federated learning (FL) allows clients to collaboratively train models by sharing gradients rather than raw data. However, gradient inversion attacks (GIAs) can reconstruct private training data from shared gradients, posing a serious privacy threat.

**Core bottleneck of existing GIAs**: When victims train with large batches, the search space for gradient inversion grows exponentially with batch size, causing reconstruction quality to degrade sharply. Existing methods largely fail to recover recognizable images when batch size exceeds 8.

**Prior attempts to narrow the reconstruction scope**:
- **Fishing/GradFilt**: Set certain class parameters to extremely large values so that gradients from specific classes dominate. However, this only supports class-level filtering, is too coarse-grained, and abnormal parameter values are easily detected.
- **SEER/LOKI/Abandon**: Introduce special neural architectures to preserve gradients of specific samples, but can only target semantically irrelevant conditions such as brightness, offering very limited control.
- No existing method supports **semantic-level, cross-category, instance-level** targeted reconstruction.

**Key Challenge**: Attackers are fundamentally interested in data with specific semantic content (e.g., "any weapon," "human faces") rather than random samples or specific classes, yet existing methods lack flexible semantic specification capability.

**Core Idea**: Leverage the text-image alignment capability of pretrained VLMs (e.g., CLIP) to translate attacker natural language queries into a reshaping of the malicious global model's loss surface, such that samples matching the query produce high gradients while non-matching samples have their gradients suppressed.

## Method

### Overall Architecture
Geminio operates in two stages:
1. **Preparation Phase**: Receive a natural language query → leverage a VLM and an auxiliary dataset → optimize malicious global model parameters $\boldsymbol{\Theta}_{\mathcal{Q}}$ to reshape its loss surface.
2. **Attack Phase**: Send the malicious model to the victim → the victim computes gradients on private data and uploads them → the server applies any existing reconstruction method to recover samples matching the query.

### Key Designs
1. **VLM-Guided Loss Surface Reshaping**:

    - **Function**: Train the malicious global model so that its loss surface aligns with the VLM's text-image similarity surface.
    - **Mechanism**: For each sample $\boldsymbol{x}$ in the auxiliary dataset, compute its similarity to query $\mathcal{Q}$ via VLM: $s(\boldsymbol{x}; \mathcal{Q}) = \mathcal{V}_{\text{image}}(\boldsymbol{x})^{\intercal} \mathcal{V}_{\text{text}}(\mathcal{Q})$, then apply intra-batch softmax normalization:
    $\alpha(\boldsymbol{x}; \mathcal{Q}, \boldsymbol{\mathcal{B}}_{\text{aux}}) = \frac{\exp(s(\boldsymbol{x}; \mathcal{Q}))}{\sum_{\boldsymbol{x}'} \exp(s(\boldsymbol{x}'; \mathcal{Q}))}$
    The training objective is:
    $\mathcal{L}_{\text{Geminio}} = \frac{\sum_{\boldsymbol{x}} \mathcal{L}(F_{\boldsymbol{\Theta}_{\mathcal{Q}}}(\boldsymbol{x}); y)(1 - \alpha(\boldsymbol{x}))}{|\boldsymbol{\mathcal{B}}_{\text{aux}}| \sum_{\boldsymbol{x}'} \mathcal{L}(F_{\boldsymbol{\Theta}_{\mathcal{Q}}}(\boldsymbol{x}'); y')(1 - \alpha(\boldsymbol{x}'))}$
    - **Design Motivation**: For highly matching samples, the weight $(1-\alpha)$ approaches 0, preventing the model from reducing their loss. For non-matching samples, the weight is large, driving their loss toward zero. Consequently, matching samples retain high loss → high gradients, while non-matching samples have near-zero loss → negligible gradients.

2. **VLM-Guided Auxiliary Label Generation**:

    - **Function**: Eliminate the requirement for labeled auxiliary datasets.
    - **Mechanism**: Generate soft labels for each auxiliary sample using the VLM:
    $y_i = \frac{\mathcal{V}_{\text{image}}(\boldsymbol{x})^{\intercal} \mathcal{V}_{\text{text}}(c_i)}{\sum_{j=1}^{K} \mathcal{V}_{\text{image}}(\boldsymbol{x})^{\intercal} \mathcal{V}_{\text{text}}(c_j)}$
    where $c_1, ..., c_K$ are the class names of the FL task.
    - **Design Motivation**: The auxiliary dataset can consist of publicly available unlabeled images (e.g., crawled from the Internet), substantially lowering the barrier to attack.

3. **Gradient Dominance Mechanism**:

    - **Function**: Ensure that the gradients uploaded by the victim are dominated by samples matching the query.
    - **Mechanism**: Through loss surface reshaping, for a target sample $\boldsymbol{x}_{\text{target}}$:
    $\|\nabla_{\boldsymbol{\Theta}_{\mathcal{Q}}} \mathcal{L}(F(\boldsymbol{x}); y)\| \ll \|\nabla_{\boldsymbol{\Theta}_{\mathcal{Q}}} \mathcal{L}(F(\boldsymbol{x}_{\text{target}}); y_{\text{target}})\|$
    for all $\boldsymbol{x} \neq \boldsymbol{x}_{\text{target}}$.
    - **Design Motivation**: Per-sample gradient magnitude is proportional to per-sample loss, so controlling the loss surface is equivalent to controlling the gradient distribution.

### Loss & Training
- The malicious model parameters are trained with $\mathcal{L}_{\text{Geminio}}$ using standard SGD.
- Any existing reconstruction optimization method (DLG, InvertingGrad, HFGradInv, etc.) can be directly applied in the gradient reconstruction phase.
- The attack runs in parallel with normal training: gradients from non-victim clients are aggregated normally, leaving global model training unaffected.
- The malicious model is only sent to the target victim during designated attack rounds.

## Key Experimental Results

### Main Results (Targeted Reconstruction — ImageNet/CIFAR-20/FER)

| Method | Attack Type | Granularity | Semantic Control | ImageNet Success Rate |
|--------|------------|-------------|-----------------|----------------------|
| Vanilla GIA | Full-batch reconstruction | All | None | Very low (batch ≥ 8) |
| Fishing | Class-level | Single sample | Class only | Moderate |
| GradFilt | Class-level | Full class | Class only | Moderate |
| SEER/LOKI | Condition-level | Random | Brightness, etc. | Low |
| **Geminio** | **Instance-level** | **Semantic match** | **Natural language** | **High** |

### Large-Batch Targeted Reconstruction (CIFAR-20, HFGradInv)

| Batch Size | Geminio Recall (%) | Geminio Precision (%) | Baseline Recall (%) |
|------------|-------------------|----------------------|---------------------|
| 2 | 90.12 | 85.34 | 45.23 |
| 8 | 82.45 | 80.12 | 12.56 |
| 32 | 74.38 | 73.21 | ~0 |
| 64 | 70.25 | 69.83 | ~0 |
| 128 | 67.51 | 67.12 | ~0 |
| 256 | 64.96 | 65.67 | ~0 |

### Ablation Study / Robustness Experiments

| Configuration | F-1 (%) | Notes |
|---------------|---------|-------|
| Auxiliary data 20% in-distribution | 68.13 | Default setting |
| Auxiliary data 5% in-distribution | 60.37 | Effective with limited data |
| Auxiliary data = ImageNet (cross-domain) | 64.37 | Cross-domain auxiliary data is viable |
| Auxiliary data = Caltech256 (cross-domain) | 50.48 | Slight degradation with larger domain gap |
| Gradient pruning defense (95%) | ≈ Unaffected | Pruning fails to mitigate the attack |
| Gradient pruning defense (99%) | Significant drop | But also destroys normal training |
| Laplacian noise (0.10) | ≈ Unaffected | Low noise is ineffective |
| Laplacian noise (0.50) | Significant drop | But also destroys normal training |
| Max parameter value: Clean=0.35, Geminio=1.64 | Stealthy | Fishing=2772, GradFilt=1000 |

### Key Findings
- **Geminio is the only method supporting cross-category, instance-level, semantically driven targeted reconstruction.**
- Even at batch size 256, recall and precision remain approximately 65%.
- As a plug-in enhancement to existing methods (DLG, InvertingGrad, HFGradInv), all baselines show significant improvement.
- Model parameter stealthiness is substantially superior to Fishing and GradFilt (maximum parameter value 1.64 vs. 2772/1000).
- Compatible with FedAvg (via learning rate control) and multiple architectures (ResNet, MobileNet, EfficientNet, ViT).
- ViT and EfficientNet are more susceptible than ResNet, suggesting that more capable models may be more vulnerable.

## Highlights & Insights
- **A novel threat from VLM misuse**: Demonstrates that pretrained VLMs can be weaponized to provide attackers with a natural language "communication interface" for expressing attack intent.
- **Elegant loss surface reshaping**: Avoids the instability of second-order derivatives by indirectly controlling gradient distribution through per-sample loss manipulation.
- **Strong practical applicability**: The auxiliary dataset requires no annotation, need not be related to the FL task, and the query need not align with FL task categories.
- **Stealthiness by design**: Model parameters remain natural-looking, no architectural modifications are needed, attacks can be launched at arbitrary rounds, and normal training is undisturbed.
- **Defense evasion**: Existing defenses including gradient pruning, noise injection, and parameter inspection all fail to effectively counter Geminio, revealing a serious security vulnerability.

## Limitations & Future Work
- Relies on the text-image alignment quality of the VLM (e.g., CLIP); queries targeting specialized domains may yield insufficient precision.
- Attack success rate gradually declines as batch size grows; effectiveness for very large batches (>512) remains unexplored.
- Under FedAvg, the attack requires control over client learning rates, which may not be feasible in FL systems where clients manage their own settings.
- Stronger defense mechanisms such as Secure Aggregation are not considered.
- The semantic granularity of queries is bounded by VLM capability (e.g., queries such as "photos taken on October 3rd" are not supported).
- Reconstruction quality remains contingent on the underlying reconstruction optimization method.

## Related Work & Insights
- The progression from Imperio (language-guided backdoor attacks) to Geminio (language-guided gradient inversion) suggests that natural language is emerging as a universal interface for attackers.
- The findings carry important implications for FL privacy defense research: mechanisms that fundamentally prevent loss surface reshaping need to be designed.
- The positive correlation between model capability and privacy vulnerability hints that a "capability–privacy" trade-off may be a fundamental dilemma in FL.
- Provides a new threat model and testing tool for FL security auditing.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — First natural language-guided GIA; weaponization of VLMs is a genuinely new direction; both problem formulation and solution are compelling.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 3 datasets × 5 attack methods × 4 defenses × multiple architectures × FedSGD + FedAvg; extremely comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Polished figures, intuitive attack scenario visualizations, clear method descriptions, and thorough security analysis.
- **Value**: ⭐⭐⭐⭐⭐ — Reveals a fundamentally new threat paradigm for FL with far-reaching implications for privacy protection and AI security research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Enhancing Adversarial Transferability by Balancing Exploration and Exploitation with Gradient-Guided Sampling](enhancing_adversarial_transferability_by_balancing_exploration_and_exploitation_.md)
- [\[ICCV 2025\] LATTE: Collaborative Test-Time Adaptation of Vision-Language Models in Federated Learning](latte_collaborative_test-time_adaptation_of_vision-language_models_in_federated_.md)
- [\[NeurIPS 2025\] FedRW: Efficient Privacy-Preserving Data Reweighting for Enhancing Federated Learning of Language Models](../../NeurIPS2025/llm_safety/fedrw_efficient_privacy-preserving_data_reweighting_for_enhancing_federated_lear.md)
- [\[ICML 2026\] Gradient Transformer: Learning to Generate Updates for LLMs](../../ICML2026/llm_safety/gradient_transformer_learning_to_generate_updates_for_llms.md)
- [\[NeurIPS 2025\] FedSVD: Adaptive Orthogonalization for Private Federated Learning with LoRA](../../NeurIPS2025/llm_safety/fedsvd_adaptive_orthogonalization_for_private_federated_learning_with_lora.md)

</div>

<!-- RELATED:END -->
