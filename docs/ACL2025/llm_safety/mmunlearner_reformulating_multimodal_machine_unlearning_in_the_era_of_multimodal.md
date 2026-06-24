---
title: >-
  [Paper Note] MMUnlearner: Reformulating Multimodal Machine Unlearning in the Era of Multimodal Large Language Models
description: >-
  [ACL 2025 (Findings)][LLM Safety][Machine Unlearning] This paper reformulates the machine unlearning (MU) task in the era of Multimodal Large Language Models (MLLMs)—erasing only the visual patterns associated with specific entities while retaining textual knowledge. It proposes MMUnlearner, a geometry-constrained gradient ascent method that selectively updates parameters using weight saliency maps, comprehensively outperforming baselines like GA and NPO on both MLLMU-Bench a…
tags:
  - "ACL 2025 (Findings)"
  - "LLM Safety"
  - "Machine Unlearning"
  - "Multimodal Large Language Models"
  - "Gradient Ascent"
  - "Weight Saliency"
  - "Privacy Protection"
date: 2026-05-08
content_hash: 30ad199273e27aa7
---

# MMUnlearner: Reformulating Multimodal Machine Unlearning in the Era of Multimodal Large Language Models

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2502.11051](https://arxiv.org/abs/2502.11051)  
**Code**: [https://github.com/Z1zs/MMUnlearner](https://github.com/Z1zs/MMUnlearner)  
**Area**: Multimodal VLM  
**Keywords**: Machine Unlearning, Multimodal Large Language Models, Gradient Ascent, Weight Saliency, Privacy Protection

## TL;DR

This paper reformulates the machine unlearning (MU) task in the era of Multimodal Large Language Models (MLLMs)—erasing only the visual patterns associated with specific entities while retaining textual knowledge. It proposes MMUnlearner, a geometry-constrained gradient ascent method that selectively updates parameters using weight saliency maps, comprehensively outperforming baselines like GA and NPO on both MLLMU-Bench and CLEAR benchmarks.

## Background & Motivation

**Background**: Machine unlearning (MU) aims to selectively remove the influence of specific data or knowledge from pre-trained models to satisfy privacy regulations (such as GDPR's "right to be forgotten"). Early MU research mainly focused on classification models and has recently extended to LLMs. For multimodal large language models (such as LLaVA, InternVL, etc.), MU is still in its infancy.

**Limitations of Prior Work**: Directly applying unimodal unlearning strategies (such as Gradient Ascent (GA) and Negative Preference Optimization (NPO)) to MLLMs faces two severe limitations: (1) **Catastrophic forgetting**—while erasing target knowledge, a vast amount of non-target knowledge is also destroyed, leading to severe degradation of overall model capabilities; (2) **Blurred modality boundaries**—the knowledge of MLLMs resides concurrently in both the vision encoder and the language model, and coarse-grained parameter updates fail to distinguish between "visual patterns to be forgotten" and "textual knowledge to be retained."

**Key Challenge**: There is a fundamental conflict between the thoroughness of unlearning and the integrity of retaining. Especially in MLLMs, the knowledge of the same entity spans across both visual and textual modalities—requiring the erasure of the "recognition ability upon seeing someone's face" while retaining the textual knowledge of "knowing who this person is," which is highly entangled in the parameter space.

**Goal**: (1) To reformulate the MU task in the MLLM era—erasing only visual patterns while preserving textual knowledge in the LLM backbone; (2) To design a method that can precisely control the scope of unlearning.

**Key Insight**: Leveraging the geometric analysis of parameter importance—by computing weight saliency maps, identifying which parameters primarily encode target visual knowledge and which encode non-target or textual knowledge, thereby updating only the parameters related to visual patterns.

**Core Idea**: In the gradient ascent unlearning process, using weight saliency maps jointly constrained by "retained concepts" and "textual knowledge" to mask parameters that should not be updated, thereby achieving precise selective unlearning.

## Method

### Overall Architecture

The pipeline of MMUnlearner is divided into two phases: (1) **Saliency Map Generation**: Computing gradients using both the retention dataset and text-only data to construct weight saliency maps, which mark the importance of each parameter to "retained knowledge" and "textual knowledge"; (2) **Constrained Gradient Ascent Unlearning**: Running gradient ascent on the data to be forgotten (increasing the loss to unlearn) during the unlearning phase, while masking out parameters crucial for retained knowledge via the saliency maps, updating only "safe" parameters. The input includes the MLLM model, the entity data to be forgotten, and the retention data; the output is the unlearned MLLM.

### Key Designs

1. **Dual-Constrained Weight Saliency Map**:

    - **Function**: Identifies which parameters can be safely updated for unlearning and which must be protected.
    - **Mechanism**: Build two saliency maps and take their union. The first, "retained concept saliency map", computes gradients on the retention data (VQA data of non-target entities), marking high-gradient parameters as "important for retained concepts" that need protection. The second, "textual knowledge saliency map", computes gradients on text-only data (without inputting images), marking high-gradient parameters as "important for textual knowledge" that need protection. Taking the union of these two maps yields the "protected parameter set", and its complement defines the parameters that are safe to update. Finally, the unlearning gradients are element-wise multiplied by the saliency mask: $\Delta w = m \odot \nabla_w L_{forget}$.
    - **Design Motivation**: A single constraint is insufficient—using only retained concept constraints might damage parameters unrelated to retention but essential for textual capability (such as language generation); adding the textual knowledge constraint ensures that the native capabilities of the LLM backbone do not degrade.

2. **Geometry-Constrained Gradient Ascent**:

    - **Function**: Updates parameters in the unlearning direction while preventing deviations from the "safe region" of retained knowledge.
    - **Mechanism**: Standard gradient ascent directly increases the loss on the data to be forgotten ($w \leftarrow w + \eta \nabla_w L_{forget}$), which can easily cause parameters to move excessively in the unlearning direction. This work introduces a geometric constraint—checking whether the loss on retention data increases significantly after parameter updates, and rolling back or shrinking the step size if it exceeds a threshold. This is akin to drawing a "safety ball" in the parameter space, allowing updates only within the ball.
    - **Design Motivation**: Even with the protection of saliency masks, low-saliency parameters may still affect retention performance after large-step updates. Geometric constraints provide an additional safety net.

3. **Modality-Aware Unlearning Objective**:

    - **Function**: Erases only visual patterns while preserving textual knowledge.
    - **Mechanism**: Unlearning loss is computed solely on VQA samples containing visual information of the target entity (e.g., Q&A pairs with someone's photo), without involving samples that mention the entity in pure text. Accordingly, the evaluation metrics are divided into: visual unlearning efficacy (failure to recognize upon seeing the photo), textual knowledge retention (ability to correctly answer text-only queries about the entity), and general capability retention (degradation check on general VQA tasks).
    - **Design Motivation**: Entity knowledge in MLLMs is modality-specific—visual recognition capability and textual factual knowledge can be encoded in different parameter subspaces, and modality-aware unlearning maximizes the preservation of useful knowledge.

### Loss & Training

The optimization objective during the unlearning phase is: $L = -L_{forget}(D_{forget}) + \lambda \cdot L_{retain}(D_{retain})$, where the first term represents gradient ascent (increasing the loss of the data to be forgotten), and the second term is the retention constraint (maintaining the loss of the retention data). The actual parameter updates are filtered through the saliency mask: $\Delta w = m \odot \nabla_w L$. Training hyperparameters include a learning rate of $1 \times 10^{-5}$, active batch size 4, and training for only 1 epoch is sufficient.

## Key Experimental Results

### Main Results (MLLMU-Bench)

| Method | Unlearning Effect ↑ | Retained Concept ↑ | Textual Knowledge ↑ | General VQA ↑ | Overall Score |
|------|----------|----------|----------|----------|---------|
| Vanilla (Original Model) | 0% | 100% | 100% | 100% | - |
| GA | High | Low | Low | Severe Decline | Poor |
| GA_Diff | Medium | Medium | Medium | Decline | Medium |
| KL_Min | Medium | Medium-High | Medium-High | Slight Decline | Medium |
| NPO | High | Low | Low | Severe Decline | Poor |
| **MMUnlearner** | **High** | **High** | **High** | **Slight Decline** | **Optimal** |

### Ablation Study

| Configuration | Unlearning Effect | Retention Effect | Description |
|------|----------|----------|------|
| Full MMUnlearner | Optimal Balance | Optimal | Dual-constrained + geometric constraint |
| w/o Textual Knowledge Saliency | Unchanged | Textual Knowledge Decreases | Textual constraint is key |
| w/o Retained Concept Saliency | Unchanged | Retained Concept Decreases | Concept constraint is key |
| w/o Any Saliency (= GA) | High Unlearning | Severe Degradation | Destructiveness of unconstrained updates |
| Language Mask vs Visual Mask vs Both | Visual Mask is Optimal | - | Visual parameters are the core of unlearning |

### Key Findings

- **Catastrophic degradation of GA and NPO on MLLMs**: Directly applying these baseline methods results in a severe decline in retained concepts and general VQA performance (sometimes exceeding 30%), indicating that parameter entanglement in MLLMs is far more complex than in unimodal models.
- **Both constraints are indispensable**: Removing the textual knowledge constraint keeps the visual unlearning effect unchanged but degrades the text-matching/QA performance of the LLM. Dropping the retained concept constraint leads to unintended corruption of other visual concepts.
- **Visual parameters are the primary battleground for unlearning**: Utilizing the saliency mask of the visual module (vision_mask) yields better results than the language module's mask, verifying the hypothesis that "visual patterns are primarily encoded in vision-related parameters."
- Consistent advantages are also achieved on the CLEAR dataset, validating the generalizability of the proposed method.

## Highlights & Insights

- **Valuable task reformulation**: Redefining machine unlearning for MLLMs from "coarse-grained erasure" to "modality-selective erasure" (erasing vision while keeping text) better aligns with real-world requirements. For instance, requiring a model to forget someone's face but retain public text-based facts about them is far more reasonable than complete erasure.
- **Elegant dual-constrained saliency map design**: Locking the "safe parameter region" through the intersected/joint constraints of retained concepts and textual knowledge is conceptually similar to EWC in continual learning but more refined. This framework can generalize to any scenario requiring selective modification of multimodal models.
- **Only 1 epoch and no auxiliary training data**: The unlearning process is highly efficient and does not require massive "proxy data" for knowledge distillation or adversarial training, demonstrating strong practicality.

## Limitations & Future Work

- **Verifiability of unlearning effects**: Current evaluations primarily rely on the decrease in VQA accuracy. However, models might still retain target knowledge implicitly (e.g., via activation patterns). Establishing more rigorous unlearning verification methods remains an open question.
- **Extension to more unlearning scenarios**: Present work focuses on entity-level visual unlearning (e.g., erasing a specific person). Unlearning at the concept level (e.g., erasing the ability to recognize "violent content") or sample-level unlearning has not been fully verified.
- **Dependency on retention data for saliency computation**: In certain scenarios, obtaining a retention dataset that is "similar but different" from the data to be forgotten might be challenging.
- **Validation limited to LLaVA series models**: Generalization to other MLLM architectures (e.g., Qwen-VL, InternVL) remains unverified.
- Future work could explore sequential unlearning scenarios—how to efficiently update the saliency maps when multiple entities need to be forgotten consecutively.

## Related Work & Insights

- **vs Gradient Ascent (GA)**: GA is the simplest unlearning baseline (directly increasing the loss of the data to be forgotten), but it causes catastrophic degradation in MLLMs. MMUnlearner addresses this through saliency constraints.
- **vs NPO (Negative Preference Optimization)**: NPO models unlearning as preference optimization. However, on MLLMs, it similarly over-destructs non-target knowledge due to a lack of modality awareness. The modality-aware strategy of this work is the key difference.
- **vs SRF (Unified Unlearning w/ Remain Geometry)**: This work draws inspiration from SRF's geometry-constrained ideas but introduces the dual-constrained saliency map designed specifically for multimodal settings.
- This work reveals the complexity of knowledge entanglement in multimodal models, offering important insights for privacy protection and safety alignment in MLLMs.

## Rating

- Novelty: ⭐⭐⭐⭐ Insightful task reformulation, dual-constrained saliency map is an effective innovation
- Experimental Thoroughness: ⭐⭐⭐⭐ Two major datasets, multi-baseline comparison, detailed ablation and mask analysis
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, comprehensive method description
- Value: ⭐⭐⭐⭐ Empirically advances MLLM privacy and security, code open-sourced

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Modality-Aware Neuron Pruning for Unlearning in Multimodal Large Language Models](manu_modality_aware_unlearning.md)
- [\[AAAI 2026\] Cross-Modal Unlearning via Influential Neuron Path Editing in Multimodal Large Language Models](../../AAAI2026/llm_safety/cross-modal_unlearning_via_influential_neuron_path_editing_i.md)
- [\[NeurIPS 2025\] PULSE: Practical Evaluation Scenarios for Large Multimodal Model Unlearning](../../NeurIPS2025/llm_safety/pulse_practical_evaluation_scenarios_for_large_multimodal_model_unlearning.md)
- [\[ICLR 2026\] Knowledge Externalization: Reversible Unlearning and Modular Retrieval in Multimodal Large Language Models](../../ICLR2026/llm_safety/knowledge_externalization_reversible_unlearning_and_modular_retrieval_in_multimo.md)
- [\[ACL 2025\] ReLearn: Unlearning via Learning for Large Language Models](relearn_unlearning_via_learning_for_large_language_models.md)

</div>

<!-- RELATED:END -->
