---
title: >-
  [Paper Note] Multimodal Continual Learning with MLLMs from Multi-scenario Perspectives
description: >-
  [ICML 2026][Multimodal VLM][Multimodal Continual Learning] Addressing visual forgetting in MLLMs across scenarios, this paper constructs the MSVQA benchmark (comprising High-altitude, Underwater, Low-altitude…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Multimodal Continual Learning"
  - "Catastrophic Forgetting"
  - "LoRA Multi-branch"
  - "Visual Consistency"
  - "Multi-scenario VQA"
date: 2026-05-08
content_hash: 21505b3461c390fa
---

# Multimodal Continual Learning with MLLMs from Multi-scenario Perspectives

**Conference**: ICML 2026  
**arXiv**: [2511.18507](https://arxiv.org/abs/2511.18507)  
**Code**: Dataset released at [huggingface.co/datasets/Kaij00/MSVQA](https://huggingface.co/datasets/Kaij00/MSVQA)  
**Area**: Multimodal VLM / Continual Learning  
**Keywords**: Multimodal Continual Learning, Catastrophic Forgetting, LoRA Multi-branch, Visual Consistency, Multi-scenario VQA

## TL;DR
Addressing visual forgetting in MLLMs across scenarios, this paper constructs the MSVQA benchmark (comprising High-altitude, Underwater, Low-altitude, and Indoor settings) and proposes the Unifier framework. By integrating a Cross-Scenario Representation (CSR) multi-branch module with a projector (VRE) into vision blocks for parameter isolation, and employing dual-channel KL soft constraints (VCC) for representation alignment, the method achieves 2.70-10.62% VQA improvement and 3.40-7.69% F1 gain over 20 continual learning steps without increasing inference latency.

## Background & Motivation

**Background**: MLLMs (e.g., QwenVL, LLaVA) excel at VQA in fixed settings, but real-world edge deployment involves continuous data streams across varying environments (day/night, indoor/outdoor, different viewpoints). Existing Continual Learning (CL) research primarily focuses on text-side forgetting in LLMs (e.g., EWC, Tailor, PODNet, VQACL, QUAD), neglecting catastrophic forgetting in the visual component.

**Limitations of Prior Work**: Classic VQA benchmarks (e.g., VQAv2) feature simple questions and uniform backgrounds, emphasizing text intent parsing. In contrast, real-world deployment presents complex backgrounds and small, dense objects. Scenario switching causes visual representation drift, leading to significantly increased missed or false detections (Fig. 1). Existing CL benchmarks lack multi-scenario and multi-viewpoint visual evaluation sets.

**Key Challenge**: The objectives are to (a) incrementally accumulate knowledge within the same scenario for progressive performance gains; (b) adapt rapidly to new scenarios without forgetting old ones; and (c) maintain low latency for single-pass inference. While multi-LoRA branches provide parameter isolation, they require complex routing; pure distillation mitigates forgetting but strict intermediate alignment often suppresses plasticity for new scenarios.

**Goal**: (1) Provide a multi-scenario VQA dataset reflecting challenges of scenario/viewpoint switching; (2) Isolate visual representations across scenarios without increasing inference overhead; (3) Align different branch representations using soft constraints to prevent drift while maintaining plasticity.

**Key Insight**: The visual encoder is the component that drifts first during scenario transitions. Rather than focusing parameter isolation on the LLM side, it is more effective to introduce expandable projection modules within ViT blocks. This allows the model to learn scenario-specific "ways of seeing" independently before projecting them into a unified space, thereby bypassing the need for a router.

**Core Idea**: Insert a CSR (Cross-Scenario Representation) module into each vision block, featuring one down-up branch per scenario. The outputs of all branches are concatenated and fused via a shared projector $\mathcal P_l$ into the original dimension. Representation consistency is maintained through bidirectional KL soft constraints (VCC) between individual branches and scenario prototypes.

## Method

### Overall Architecture
Given a data stream $\mathcal D = \{\mathcal D_1, \ldots, \mathcal D_T\}$ where each task $\mathcal D_t = \{(x_i^t, q_i^t, y_i^t)\}_{i=1}^{n_t}$ originates from a different scenario, Unifier parallels a CSR module with the FFN in each vision block $f_l$. The output $p_l$ is added to the FFN output: $r_l = s_l(\text{LN}(a_l)) + p_l$. During training, only the branch corresponding to the current scenario and the projector are updated. During inference, all branches are computed in parallel and fused once, resulting in latency equivalent to a single-branch model. Visual Consistency Constraints (VCC) are applied to prevent representation drift.

### Key Designs

1.  **Vision Representation Expansion (VRE) + Single-pass Fusion**:
    - **Function**: Expands independent visual representation subspaces for each new scenario without requiring routing or increasing forward pass counts during inference.
    - **Mechanism**: The CSR module consists of $K$ down-up branches $\varphi_l^k = \phi_{up}(o(\phi_{down}(\cdot)))$ and a shared projector $\mathcal P_l \in \mathbb R^{K\times d_1 \to d_1}$. The output is $p_l = \mathcal P_l(\varphi_l^1(a_l) \oplus \cdots \oplus \varphi_l^K(a_l))$. For the $t$-th scenario, only $\varphi_l^t$ and $\mathcal P_l$ are updated while others are frozen. During inference, branches are fused via the projector in a single forward pass.
    - **Design Motivation**: Pure single-branch LoRA leads to forgetting, while multi-branch models with routers suffer from router forgetting and increased computation. A shared projector treats branch outputs as integrated components, acting as an implicit attention mechanism that avoids the need for a separate router.

2.  **Vision Consistency Constraint (VCC) Dual-channel Soft Constraint**:
    - **Function**: Prevents "indirect pollution" of other branch representations when learning new scenarios without the rigid constraints of $\ell_2$ distance.
    - **Mechanism**: A scenario prototype is calculated for each batch as $\mu_l = \frac{1}{K}\sum_k \varphi_l^k(a_l)$. Mean values are extracted across feature and embedding channels: $\bar\varphi_l^{k,\text{fe}} \in \mathbb R^{d_1}$ and $\bar\varphi_l^{k,\text{em}} \in \mathbb R^{\text{seq}}$. The constraint is defined as $\mathcal{L}_c^{l,k} = \text{KL}(\bar\varphi_l^{k,\text{fe}}/\tau \mid \bar\mu_l^{\text{fe}}/\tau) + \text{KL}(\bar\varphi_l^{k,\text{em}}/\tau \mid \bar\mu_l^{\text{em}}/\tau)$. The projector output $p_l$ is similarly aligned. The total VCC loss is $\mathcal L_{vcc} = \frac{1}{L}\sum_l (\mathcal L_p^l + \sum_k \mathcal L_c^{l,k})$.
    - **Design Motivation**: Strong $\ell_2$ constraints prevent the model from learning new local details (crushing plasticity). KL divergence applied to channel means penalizes only global distribution drift, allowing local feature reorganization—a technique adapted from knowledge distillation for CL.

3.  **CSR Integrated into Visual Encoder Only**:
    - **Function**: Allocates capacity to the most volatile visual components while keeping the LLM backbone frozen to control parameter and training costs.
    - **Mechanism**: MLLMs typically comprise a visual encoder, a projection layer, and an LLM. Experiments show that cross-scenario forgetting primarily occurs in the visual encoder, whereas the LLM's semantic decoding is relatively robust. Thus, CSR is only added to vision blocks.
    - **Design Motivation**: Expanding LoRA in the LLM backbone is computationally expensive and risks degrading general language capabilities. Focusing on the visual encoder targets the root cause with manageable parameter overhead.

### Loss & Training
The total loss is $\mathcal L = \mathcal L_{\text{task}} + \lambda \mathcal L_{vcc}$. Distillation temperature $\tau$ controls the soft constraint intensity. During training of a new scenario, parameters of other branches are frozen while the projector $\mathcal P_l$ is updated. Similar to QUAD, no images are stored, but text-based question exemplars may be retained.

## Key Experimental Results

### Main Results
Evaluated on MSVQA across 4 scenarios (High-altitude, Underwater, Low-altitude, Indoor) using VQA score and F1 metrics under $T=5$ and $T=20$ step settings.

| Methods | High alt. VQA $A_T$ | Underwater VQA $A_T$ | Low alt. VQA $A_T$ | Indoor VQA $A_T$ |
|---------|---------------------|----------------------|---------------------|---------------------|
| Zero-shot | 20.55 | 19.30 | 14.94 | 52.40 |
| Joint (Upper Bound) | 64.97 | 84.27 | 59.80 | 87.20 |
| Finetune | 30.09 | 74.98 | 32.27 | 51.40 |
| EWC | 31.70 | 76.14 | 35.27 | 55.00 |
| ER | 43.64 | 78.16 | 48.12 | 61.40 |
| PODNet | 52.95 | 79.38 | 52.87 | 81.20 |
| QUAD (Prev. SOTA) | 56.59 | 79.62 | – | – |
| **Unifier (Ours)** | **Significantly exceeds QUAD** | **Near Joint Bound** | **Significantly exceeds** | **Near Joint Bound** |

In the 20-step setting, Unifier improves the last-step VQA by +2.70% to +10.62% and F1 by +3.40% to +7.69% relative to QUAD.

### Ablation Study

| Configuration | Key Metrics | Note |
|------|---------|------|
| Full Unifier | **Best** | Includes VRE + VCC + Dual-channel KL |
| w/o VRE (Single-branch LoRA) | Significant degradation | Lack of isolation; scenarios overwrite each other |
| Multi-branch with routing | Router forgetting | Routing accuracy decays rapidly as scenarios increase |
| w/o VCC | Drift in old scenarios | Good plasticity but poor stability |
| VCC with $\ell_2$ instead of KL | Poor plasticity | Fails to learn new content in new scenarios |
| VCC feature-channel only | Moderate | Dual-channel significantly outperforms single-channel |

### Key Findings
- The visual encoder is the "epicenter" of forgetting in cross-scenario MLLM CL; applying CSR to vision blocks alone addresses most issues.
- Dual-channel KL soft constraints achieve a superior trade-off between plasticity and stability compared to rigid $\ell_2$ constraints.
- Using a shared projector $\mathcal P_l$ instead of an explicit router simplifies the inference path and avoids the "chicken-and-egg" problem of training a router via CL.

## Highlights & Insights
- **Precise Diagnosis**: The authors identify visual encoder drift as the primary cause by visualizing increased false positives/negatives in old scenarios after training on new ones (Fig. 1).
- **Routing-free Multi-branching**: This represents an elegant engineering trade-off—benefiting from parameter isolation without the overhead of training or maintaining a router. This logic is transferable to any multi-task/multi-domain PEFT scenario.
- **Dual-channel KL Constraint**: By penalizing global distribution drift across both feature and sequence dimensions, it leaves room for "detail reorganization," providing a new way to handle the stability-plasticity dilemma in CL.

## Limitations & Future Work
- Parameter count grows linearly with the number of scenarios $K$. For large $K$, the projector $\mathcal P_l \in \mathbb R^{K d_1 \to d_1}$ may become a bottleneck for long-horizon CL.
- Experiments were restricted to 4 scenarios over 20 steps; extensibility to "open-world" settings with hundreds of scenarios remains unverified.
- As MSVQA scenarios are distinct, the benefits of VRE might diminish in sub-domains with higher visual similarity.
- The study does not address LLM-side forgetting, such as shifts in instruction styles or vocabulary acquisition.

## Related Work & Insights
- **vs. QUAD (Marouf 2025)**: QUAD focuses on the LLM side using text-only exemplars and attention distillation. Unifier focuses on visual drift, making them complementary; Unifier's performance significantly surpasses QUAD in VQA tasks.
- **vs. PODNet / VQACL**: These traditional CL methods rely on intermediate layer distillation or image rehearsal. Unifier achieves superior results without storing any images.
- **vs. Dynamic Architectures (e.g., DER)**: DER-style expansion is unsuited for massive MLLM backbones. CSR's down-up branches within vision blocks keep computation manageable.
- **vs. Multi-LoRA + Router**: Replacing the router with a projector is a more robust engineering choice, avoiding the performance decay associated with router training in CL.

## Rating
- Novelty: ⭐⭐⭐⭐ (Addresses the overlooked visual aspect of MLLM CL; projector-based fusion is a clever alternative to routing).
- Experimental Thoroughness: ⭐⭐⭐⭐ (Extensive comparison across 4 scenarios and 20 steps, though more datasets would strengthen the case).
- Writing Quality: ⭐⭐⭐⭐ (Clear motivation and architectural diagrams; highly readable despite complex notation).
- Value: ⭐⭐⭐⭐ (Directly useful for edge deployment of MLLMs; VCC constraints have general utility).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Continual Multimodal Contrastive Learning](../../NeurIPS2025/multimodal_vlm/continual_multimodal_contrastive_learning.md)
- [\[AAAI 2026\] FinMMDocR: Benchmarking Financial Multimodal Reasoning with Scenario Awareness, Document Understanding, and Multi-Step Computation](../../AAAI2026/multimodal_vlm/finmmdocr_benchmarking_financial_multimodal_reasoning_with_scenario_awareness_do.md)
- [\[ICLR 2026\] Enhanced Continual Learning of Vision-Language Models with Model Fusion](../../ICLR2026/multimodal_vlm/enhanced_continual_learning_of_vision-language_models_with_model_fusion.md)
- [\[ICML 2026\] iVGR: Internalizing Visually Grounded Reasoning for MLLMs with Reinforcement Learning](ivgr_internalizing_visually_grounded_reasoning_for_mllms_with_reinforcement_lear.md)
- [\[ICML 2026\] Injecting Distributional Awareness into MLLMs via Reinforcement Learning for Deep Imbalanced Regression](injecting_distributional_awareness_into_mllms_via_reinforcement_learning_for_dee.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[NeurIPS 2025\] Continual Multimodal Contrastive Learning](../../NeurIPS2025/multimodal_vlm/continual_multimodal_contrastive_learning.md)
- [\[AAAI 2026\] FinMMDocR: Benchmarking Financial Multimodal Reasoning with Scenario Awareness, Document Understanding, and Multi-Step Computation](../../AAAI2026/multimodal_vlm/finmmdocr_benchmarking_financial_multimodal_reasoning_with_scenario_awareness_do.md)
- [\[ICML 2026\] iVGR: Internalizing Visually Grounded Reasoning for MLLMs with Reinforcement Learning](ivgr_internalizing_visually_grounded_reasoning_for_mllms_with_reinforcement_lear.md)
- [\[ICLR 2026\] Enhanced Continual Learning of Vision-Language Models with Model Fusion](../../ICLR2026/multimodal_vlm/enhanced_continual_learning_of_vision-language_models_with_model_fusion.md)
- [\[ICML 2026\] Injecting Distributional Awareness into MLLMs via Reinforcement Learning for Deep Imbalanced Regression](injecting_distributional_awareness_into_mllms_via_reinforcement_learning_for_dee.md)

</div>

<!-- RELATED:END -->
