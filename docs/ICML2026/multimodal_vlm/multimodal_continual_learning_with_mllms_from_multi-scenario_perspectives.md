---
title: >-
  [Paper Note] Multimodal Continual Learning with MLLMs from Multi-scenario Perspectives
description: >-
  [ICML 2026][Multimodal VLM][Multimodal continual learning] To address the visual forgetting problem of MLLMs in cross-scenario VQA, this work constructs the MSVQA benchmark (four scenarios: high-altitude, underwater…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Multimodal continual learning"
  - "catastrophic forgetting"
  - "multi-branch LoRA"
  - "visual consistency"
  - "multi-scenario VQA"
date: 2026-05-08
content_hash: 5b72f67837c1c09c
---

# Multimodal Continual Learning with MLLMs from Multi-scenario Perspectives

**Conference**: ICML 2026  
**arXiv**: [2511.18507](https://arxiv.org/abs/2511.18507)  
**Code**: Dataset released at [huggingface.co/datasets/Kaij00/MSVQA](https://huggingface.co/datasets/Kaij00/MSVQA)  
**Area**: Multimodal VLM / Continual Learning  
**Keywords**: Multimodal continual learning, catastrophic forgetting, multi-branch LoRA, visual consistency, multi-scenario VQA

## TL;DR
To address the visual forgetting problem of MLLMs in cross-scenario VQA, this work constructs the MSVQA benchmark (four scenarios: high-altitude, underwater, low-altitude, indoor) and proposes the Unifier framework—adding CSR multi-branch modules and a projector (VRE) for parameter isolation within vision blocks, then aligning different branch representations with a KL-based soft constraint (VCC). With a single inference, Unifier improves VQA by 2.70–10.62% and F1 by 3.40–7.69% over 20-step continual learning.

## Background & Motivation

**Background**: MLLMs (e.g., QwenVL, LLaVA) can solve VQA tasks in fixed scenarios, but real-world deployment involves continuously changing data streams—day/night, indoor/outdoor, varying device perspectives. Existing CL work mainly focuses on text-side forgetting in LLMs (EWC, Tailor, PODNet, VQACL, QUAD), neglecting catastrophic forgetting in the visual component.

**Limitations of Prior Work**: Classic VQA benchmarks (e.g., VQAv2) feature simple questions (color, count), focusing on parsing user intent with homogeneous backgrounds. In real deployment, image backgrounds are complex, targets are small and dense, and scenario switching causes visual representation overlap/drift, leading to significant increases in missed and false detections (see Fig. 1). Current CL benchmarks lack multi-scenario, multi-view visual evaluation sets.

**Key Challenge**: The goal is to (a) continually accumulate knowledge within a scenario for progressive improvement; (b) quickly adapt to new scenarios without forgetting old ones; (c) maintain low-latency single-pass inference. Multi-branch LoRA enables parameter isolation but requires routing; pure distillation can mitigate forgetting but strict intermediate-layer alignment suppresses plasticity for new scenarios.

**Goal**: (1) Provide a multi-scenario VQA dataset reflecting "scenario/view switching → visual forgetting"; (2) Isolate visual representations for different scenarios without increasing inference cost; (3) Use soft constraints to align branch representations, preventing drift while preserving plasticity.

**Key Insight**: The visual encoder is the first to drift during scenario switching; instead of parameter isolation on the LLM side, inserting scalable projection modules into ViT blocks allows each scenario to independently learn its "way of seeing the world," but then projects them into a unified space, eliminating the need for routing.

**Core Idea**: Insert a CSR (Cross-Scenario Representation) module into vision blocks—one down-up branch per scenario, all branch outputs concatenated and fused back to the original dimension via a shared projector $\mathcal P_l$, with bidirectional KL soft constraints between each branch and the scenario prototype to maintain representation consistency.

## Method

### Overall Architecture
The data stream $\mathcal D = \{\mathcal D_1, \ldots, \mathcal D_T\}$, with each task $\mathcal D_t = \{(x_i^t, q_i^t, y_i^t)\}_{i=1}^{n_t}$ from a different scenario. Unifier attaches a parallel CSR module to each vision block $f_l$ alongside the FFN, outputting $p_l$, which is added to the FFN output $r_l = s_l(\text{LN}(a_l)) + p_l$. During training, only the branch and projector for the current scenario are unfrozen; during inference, all branches compute in parallel and are fused in a single pass, yielding latency equivalent to a single-branch model. Visual consistency constraint (VCC) is applied within CSR to prevent representation drift.

### Key Designs

1. **Vision Representation Expansion (VRE) + Single-pass Fusion**:

    - **Function**: Expands an independent visual subspace for each new scenario, but inference requires no routing and does not increase forward passes.
    - **Mechanism**: The CSR module consists of $K$ down-up branches $\varphi_l^k = \phi_{up}(o(\phi_{down}(\cdot)))$ and a shared projector $\mathcal P_l \in \mathbb R^{K\times d_1 \to d_1}$; output is $p_l = \mathcal P_l(\varphi_l^1(a_l) \oplus \cdots \oplus \varphi_l^K(a_l))$. Each branch handles visual features for one scenario, with downsampled dimension $d_2 \ll d_1$, so parameter growth is moderate. When training scenario $t$, only $\varphi_l^t$ and $\mathcal P_l$ are updated; other $\varphi_l^{k\neq t}$ are frozen. At inference, all branches compute in parallel, concatenated and passed through the projector in a single forward pass.
    - **Design Motivation**: Single-branch LoRA leads to forgetting; multi-branch with routing causes the router itself to forget and increases inference passes. Using a shared projector to "combine" multi-branch outputs into a unified representation is akin to implicit attention routing—no router training or extra inference passes required.

2. **Vision Consistency Constraint (VCC) Dual-channel Soft Constraint**:

    - **Function**: Prevents "indirect contamination" of other branch representations when learning new scenarios, while avoiding the rigidity of $\ell_2$ constraints that stifle plasticity.
    - **Mechanism**: For each batch, compute the scenario prototype $\mu_l = \frac{1}{K}\sum_k \varphi_l^k(a_l)$, then average each branch's representation along feature and embedding channels, $\bar\varphi_l^{k,\text{fe}} \in \mathbb R^{d_1}$ and $\bar\varphi_l^{k,\text{em}} \in \mathbb R^{\text{seq}}$, respectively. Apply relative entropy constraint $\mathcal{L}_c^{l,k} = \text{KL}(\bar\varphi_l^{k,\text{fe}}/\tau \mid \bar\mu_l^{\text{fe}}/\tau) + \text{KL}(\bar\varphi_l^{k,\text{em}}/\tau \mid \bar\mu_l^{\text{em}}/\tau)$. The projector output $p_l$ is aligned between new and old models via a similar KL term $\mathcal L_p^l$. The total is $\mathcal L_{vcc} = \frac{1}{L}\sum_l (\mathcal L_p^l + \sum_k \mathcal L_c^{l,k})$.
    - **Design Motivation**: Strong $\ell_2$ constraints prevent the model from learning new local details in new scenarios (plasticity is suppressed); relative entropy with channel-wise averaging "only penalizes global distribution drift, allowing local detail recombination," a key adaptation from knowledge distillation to CL.

3. **CSR Inserted Only in Visual Encoder**:

    - **Function**: Allocates capacity to the most drift-prone visual component, leaving the LLM backbone untouched, controlling extra parameters and training cost.
    - **Mechanism**: MLLMs typically consist of a visual encoder, projection alignment, and LLM. Experiments show that cross-scenario forgetting mainly occurs in the visual encoder (feature extraction is most affected by scenario), while the LLM's semantic decoding is relatively robust. Thus, CSR is inserted only in vision blocks, with trainable parameters per new scenario at $K \cdot L \cdot 2d_1 d_2$ scale.
    - **Design Motivation**: For CL on MLLMs, expanding LoRA on the LLM backbone is costly and risky (may harm general language ability); focusing on the visual encoder is both targeted and parameter-efficient.

### Loss & Training
Total loss is $\mathcal L = \mathcal L_{\text{task}} + \lambda \mathcal L_{vcc}$; distillation temperature $\tau$ controls soft constraint strength. When training a new scenario, other branch parameters are frozen, and the projector $\mathcal P_l$ is jointly updated. As in QUAD, images are not stored, but text questions may be retained as exemplars.

## Key Experimental Results

### Main Results
MSVQA includes four scenarios (High altitude / Underwater / Low altitude / Indoor), evaluated by VQA score and F1, under $T=5$ and $T=20$ step CL settings.

| Methods | High alt. VQA $A_T$ | Underwater VQA $A_T$ | Low alt. VQA $A_T$ | Indoor VQA $A_T$ |
|---------|---------------------|----------------------|---------------------|---------------------|
| Zero-shot | 20.55 | 19.30 | 14.94 | 52.40 |
| Joint (Upper Bound) | 64.97 | 84.27 | 59.80 | 87.20 |
| Finetune | 30.09 | 74.98 | 32.27 | 51.40 |
| EWC | 31.70 | 76.14 | 35.27 | 55.00 |
| ER | 43.64 | 78.16 | 48.12 | 61.40 |
| PODNet | 52.95 | 79.38 | 52.87 | 81.20 |
| QUAD (Prev. SOTA) | 56.59 | 79.62 | – | – |
| **Unifier (Ours)** | Significantly surpasses QUAD | Approaches Joint upper bound | Significantly surpasses | Approaches Joint upper bound |

20-step setting: Compared to QUAD, last-step VQA improves by +2.70 ~ +10.62%, F1 by +3.40 ~ +7.69%.

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| Full Unifier | best | VRE + VCC + dual-channel KL |
| w/o VRE (single-branch LoRA) | significant degradation | No scenario isolation, new/old scenarios overwrite each other |
| Multi-branch with router instead of projector | router forgets | Routing accuracy drops rapidly as scenarios increase |
| w/o VCC | old scenario drift | Good plasticity for new scenarios but old scenarios degrade |
| VCC with $\ell_2$ instead of KL | poor plasticity | New scenarios learn almost nothing new |
| VCC on feature channel only | intermediate | Dual-channel (fe + em) significantly outperforms single-channel |

### Key Findings
- The visual encoder is the "epicenter" of forgetting in MLLMs during cross-scenario CL; inserting CSR only in vision blocks solves most issues.
- KL dual-channel soft constraint achieves a better trade-off between plasticity and stability than strong $\ell_2$ constraints.
- The shared projector $\mathcal P_l$ replacing an explicit router is key to simplifying inference, saving forward passes and avoiding the "chicken-and-egg" problem of router continual learning.

## Highlights & Insights
- **Accurate Diagnosis**: The authors pinpoint "visual encoder drift" via visualization (Fig. 1, where the new model after learning a new scenario shows severe FP/FN on old scenarios), exemplifying a "falsify first, then design" research paradigm worth emulating.
- **Multi-branch + Projector avoids routing**: An elegant engineering trade-off—enjoys parameter isolation benefits without needing to train a router; this approach can be transferred to any multi-task/multi-domain PEFT scenario.
- **KL dual-channel soft constraint**: Compared to single-dimension $\ell_2$, penalizing only global distribution drift in both feature and sequence dimensions allows local detail "recombination"—an effective new method for plasticity-stability trade-off in CL.

## Limitations & Future Work
- Parameters grow linearly with the number of scenarios $K$ (each $\varphi_l^k$ is independent); for large $K$, the projector $\mathcal P_l \in \mathbb R^{K d_1 \to d_1}$ also grows, making long-horizon CL unsustainable.
- Experiments are limited to 4 scenarios and 20-step evaluation; generalization to truly "open world + hundreds of scenarios" is unknown.
- The four MSVQA scenarios differ greatly (high-altitude/underwater/low-altitude/indoor); if applied to subdomains with smaller visual differences, VRE's isolation benefit may diminish.
- Forgetting on the LLM side is not explored—e.g., when new scenarios introduce new vocabulary or instruction styles, whether the LLM backbone also needs similar mechanisms remains unaddressed.

## Related Work & Insights
- **vs QUAD (Marouf 2025)**: QUAD only stores historical question texts and uses cross-question attention distillation, focusing on the LLM side; this work targets visual drift, making them complementary, and significantly outperforms QUAD on VQA.
- **vs PODNet / VQACL**: Traditional CL methods use intermediate-layer distillation or sample-invariant features, requiring image rehearsal; this work stores no images, achieving equal or better results via branch isolation and KL soft constraint.
- **vs Dynamic Architecture Methods (DER, etc.)**: DER expands the backbone directly, unsuitable for large models like MLLMs; CSR adds down-up branches only in vision blocks, keeping parameters and computation manageable.
- **vs Multi-LoRA + router**: This work replaces explicit routers with a shared projector, avoiding the continual learning challenges of routers themselves—a consistently superior engineering choice.

## Rating
- Novelty: ⭐⭐⭐⭐ The visual aspect of MLLM + multi-scenario CL has been overlooked; VRE + projector replacing routing is a novel combination.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 scenarios × 5/20 steps + multiple CL baselines compared, but the number of scenarios is limited and lacks cross-dataset validation.
- Writing Quality: ⭐⭐⭐⭐ Motivation (Fig. 1) and architecture (Fig. 4) diagrams are very clear; formula numbering is slightly excessive but readable.
- Value: ⭐⭐⭐⭐ Provides dataset + framework directly useful for device-side MLLM deployment; KL dual-channel soft constraint is broadly applicable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Continual Multimodal Contrastive Learning](../../NeurIPS2025/multimodal_vlm/continual_multimodal_contrastive_learning.md)
- [\[AAAI 2026\] FinMMDocR: Benchmarking Financial Multimodal Reasoning with Scenario Awareness, Document Understanding, and Multi-Step Computation](../../AAAI2026/multimodal_vlm/finmmdocr_benchmarking_financial_multimodal_reasoning_with_scenario_awareness_do.md)
- [\[ICLR 2026\] Enhanced Continual Learning of Vision-Language Models with Model Fusion](../../ICLR2026/multimodal_vlm/enhanced_continual_learning_of_vision-language_models_with_model_fusion.md)
- [\[ICML 2026\] Injecting Distributional Awareness into MLLMs via Reinforcement Learning for Deep Imbalanced Regression](injecting_distributional_awareness_into_mllms_via_reinforcement_learning_for_dee.md)
- [\[ICLR 2026\] KeepLoRA: Continual Learning with Residual Gradient Adaptation](../../ICLR2026/multimodal_vlm/keeplora_continual_learning_with_residual_gradient_adaptation.md)

</div>

<!-- RELATED:END -->
