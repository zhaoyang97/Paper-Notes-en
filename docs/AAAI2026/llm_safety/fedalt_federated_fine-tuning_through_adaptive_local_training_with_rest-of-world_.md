---
title: >-
  [Paper Note] FedALT: Federated Fine-Tuning through Adaptive Local Training with Rest-of-World LoRA
description: >-
  [AAAI 2026][LLM Safety][Federated Learning] FedALT is proposed to maintain a trainable Individual LoRA (updated locally) and a frozen Rest-of-World (RoW) LoRA (averaged from other clients) for each client…
tags:
  - "AAAI 2026"
  - "LLM Safety"
  - "Federated Learning"
  - "LoRA Fine-Tuning"
  - "Personalization"
  - "Cross-Client Interference"
  - "MoE"
date: 2026-05-08
content_hash: 16fe7ca796b80a9d
---

# FedALT: Federated Fine-Tuning through Adaptive Local Training with Rest-of-World LoRA

**Conference**: AAAI 2026
**arXiv**: [2503.11880](https://arxiv.org/abs/2503.11880)  
**Code**: None  
**Area**: AI Safety / Federated Learning
**Keywords**: Federated Learning, LoRA Fine-Tuning, Personalization, Cross-Client Interference, MoE

## TL;DR
FedALT is proposed to maintain a trainable Individual LoRA (updated locally) and a frozen Rest-of-World (RoW) LoRA (averaged from other clients) for each client, combined with an adaptive MoE mixer that dynamically balances local and global knowledge. This design fundamentally eliminates cross-client interference caused by FedAvg aggregation, achieving significant improvements over SOTA on heterogeneous-task federated LLM fine-tuning.

## Background & Motivation

**Background**: Federated LoRA fine-tuning has become the dominant paradigm for privacy-preserving LLM adaptation. Methods such as FedIT follow the FedAvg framework—aggregating local LoRAs and reinitializing local training from the aggregated model each round. FedDPA introduces dual global+local LoRA components but still relies on FedAvg.

**Limitations of Prior Work**:
   - **Harmful cross-client interference**: When client tasks are heterogeneous (e.g., text summarization vs. sentiment analysis), FedAvg aggregation cancels out the progress each client achieves through local fine-tuning.
   - **Lack of effective global-local balancing**: Methods such as FedDPA combine global and local LoRAs with fixed weights, unable to dynamically adapt to different inputs.
   - Empirical evidence: FedIT underperforms pure local fine-tuning on Commonsense Reasoning and Text Classification.

**Key Challenge**: How can useful knowledge from other clients be leveraged while preventing aggregation from destroying local adaptation?

**Goal**: Design a personalized federated LoRA fine-tuning method that departs from the FedAvg paradigm.

**Key Insight**: Local training is no longer initialized from the aggregated model. Each client continues learning from its own previously trained local model, while global knowledge is injected through a frozen "rest-of-world" LoRA and dynamically weighted per input via an adaptive mixer.

**Core Idea**: A frozen RoW LoRA provides global knowledge; a trainable Individual LoRA handles local adaptation; an MoE mixer dynamically balances the two—completely avoiding FedAvg aggregation interference.

## Method

### Overall Architecture
Each client $k$ maintains two LoRA modules and one mixer:
- **Individual LoRA** $\mathbf{A}_k^L / \mathbf{B}_k^L$: Updated locally to capture client-specific knowledge.
- **RoW LoRA** $\mathbf{A}_k^R / \mathbf{B}_k^R$: The average of all other clients' Individual LoRAs, frozen during local training.
- **Mixer** $\mathbf{G}_k$: Dynamically learns input-dependent weights for the two LoRAs.

Forward pass: $y = \mathbf{W}_0 x + \alpha_k(x) \mathbf{B}_k^L \mathbf{A}_k^L x + (1-\alpha_k(x)) \mathbf{B}_k^R \mathbf{A}_k^R x$

### Key Designs

1. **Individual LoRA + RoW LoRA Decoupling**:

    - Function: Explicitly separates local and global knowledge into two independent LoRAs.
    - Mechanism: The RoW LoRA is computed as $\mathbf{A}_k^R = \frac{1}{K-1} \sum_{m \neq k} \mathbf{A}_m^L$. Crucially, the RoW LoRA is completely frozen during local training and receives no gradient updates.
    - Design Motivation: Interference in the FedAvg paradigm arises from two sources—(1) aggregation cancels local improvements, and (2) reinitializing from the aggregated model overwrites local adaptation. Freezing the RoW LoRA eliminates both issues. Additionally, skipping gradient updates through RoW halves client-side computation.

2. **Adaptive MoE Mixer**:

    - Function: Dynamically adjusts the contribution of Individual LoRA and RoW LoRA according to the input.
    - Mechanism: $\alpha(x), 1-\alpha(x) = \text{softmax}(\mathbf{G}_k x)$, where $\mathbf{G}_k \in \mathbb{R}^{2 \times d}$ is a trainable linear layer.
    - Design Motivation: Different inputs benefit to varying degrees from the local versus global model. Fixed weights (as in FedDPA) are suboptimal. The MoE paradigm provides input-adaptive, flexible weighting.
    - The mixer is personalized (not averaged across clients), ensuring it reflects each client's unique data distribution.

3. **Why Not Directly Merge RoW into the Pre-trained Model**:

    - The paper explicitly discusses this alternative and identifies two problems: (1) if the RoW LoRA performs poorly, it "contaminates" the pre-trained model and is difficult to correct; (2) it loses flexibility—different inputs require different global-local balances.

### Loss & Training
- **Server side**: Collects Individual LoRAs from all clients, computes the RoW LoRA for each client, and distributes them.
- **Client side**: Replaces the old RoW LoRA with the new one, then updates the Individual LoRA and Mixer (RoW LoRA and pre-trained model remain frozen).
- **Upload**: Only the Individual LoRA is uploaded; the Mixer is retained locally.

## Key Experimental Results

### Main Results (LLaMA2-7B, 8 Heterogeneous Tasks)

| Method | Commonsense Reasoning | Coreference Resolution | Text Classification | Average |
|---|---|---|---|---|
| Local Only | 73.83 | 74.62 | 67.18 | 62.86 |
| FedIT (FedAvg) | 72.82 | 77.14 | 66.39 | 62.19 |
| FedDPA | 74.81 | 81.88 | 65.42 | 64.64 |
| FDLoRA | 76.29 | 75.60 | 67.59 | 65.17 |
| **FedALT** | **76.12** | **83.04** | **71.60** | **67.55** |

FedALT achieves an average accuracy of 67.55%, outperforming the best baseline FDLoRA by 2.38% and Local Only by 4.69%.

### Ablation Study

| Configuration | Average Performance |
|---|---|
| FedALT (Full) | 67.55 |
| w/o Mixer (fixed α=0.5) | 65.82 |
| w/o RoW LoRA (Local Only) | 62.86 |
| Replace with FedAvg aggregation | 62.19 |

### Key Findings
- FedAvg underperforms pure local training on certain tasks (Commonsense Reasoning: 72.82 < 73.83), empirically confirming cross-client interference.
- Simply splitting a single large LoRA into multiple smaller ones (FedIT-split) does not mitigate interference—the root cause lies in server-side aggregation, not model internals.
- The Mixer contributes substantially (+1.73%), validating the value of dynamic input-adaptive weighting.
- FedALT is also effective on Bloom-560M, indicating robustness to model scale.

## Highlights & Insights
- The decision to **fundamentally depart from the FedAvg paradigm** is bold—rather than reinitializing from the aggregated model, each client continuously trains its own model, eliminating cross-client interference at its root.
- The combination of **frozen RoW LoRA and dynamic Mixer weighting** is elegant: freezing prevents interference → the Mixer enables flexible utilization of global knowledge → the two mechanisms are complementary.
- **The motivational study is well-executed**: comparing FedIT and Local Only across 8 tasks clearly demonstrates the coexistence of interference and knowledge gain in practice.

## Limitations & Future Work
- The Mixer is a simple $2 \times d$ linear layer with softmax; more sophisticated routing mechanisms may yield further improvements.
- Per-round communication volume is independent of the number of clients (only Individual LoRAs are transmitted), but RoW computation requires LoRAs from all clients.
- Validation is limited to NLP tasks; federated fine-tuning of multimodal or vision LLMs remains unexplored.
- Handling RoW LoRA computation under partial client participation requires additional consideration.

## Related Work & Insights
- **vs. FedDPA**: FedDPA's global LoRA is still trained via FedAvg and thus subject to interference; FedALT's frozen RoW completely avoids this.
- **vs. FDLoRA**: FDLoRA depends on a server-side dataset, and its global LoRA aggregation still introduces interference; FedALT has no such dependency.
- **vs. HydraLoRA**: HydraLoRA employs multiple LoRAs in a centralized setting to reduce interference, but remains ineffective in the federated setting—interference originates from aggregation, not from within the model.

## Rating
- Novelty: ⭐⭐⭐⭐ Departing from FedAvg + frozen RoW + MoE mixer with clear design rationale.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two LLMs, 8 tasks, 6 baselines, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ The narrative flow from Motivational Study → problem formulation → proposed solution is exceptionally smooth.
- Value: ⭐⭐⭐⭐ Provides an effective personalized solution for heterogeneous federated LLM fine-tuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Adaptive LoRA Experts Allocation and Selection for Federated Fine-Tuning](../../NeurIPS2025/llm_safety/adaptive_lora_experts_allocation_and_selection_for_federated_fine-tuning.md)
- [\[ICLR 2026\] SHE-LoRA: Selective Homomorphic Encryption for Federated Tuning with Heterogeneous LoRA](../../ICLR2026/llm_safety/she-lora_selective_homomorphic_encryption_for_federated_tuning_with_heterogeneou.md)
- [\[AAAI 2026\] TOFA: Training-Free One-Shot Federated Adaptation for Vision-Language Models](tofa_training-free_one-shot_federated_adaptation_for_vision-language_models.md)
- [\[NeurIPS 2025\] FedSVD: Adaptive Orthogonalization for Private Federated Learning with LoRA](../../NeurIPS2025/llm_safety/fedsvd_adaptive_orthogonalization_for_private_federated_learning_with_lora.md)
- [\[AAAI 2026\] Federated CLIP for Resource-Efficient Heterogeneous Medical Image Classification](federated_clip_for_resource-efficient_heterogeneous_medical_image_classification.md)

</div>

<!-- RELATED:END -->
