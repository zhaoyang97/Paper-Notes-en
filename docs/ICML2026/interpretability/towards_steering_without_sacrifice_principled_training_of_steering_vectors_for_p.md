---
title: >-
  [Paper Note] Towards Steering without Sacrifice: Principled Training of Steering Vectors for Prompt-only Interventions
description: >-
  [ICML 2026][Interpretability][Steering Vector] The authors leverage infinite-width neural network scaling theory to derive that joint training of the steering vector’s factor/direction should satisfy the scaling constrai…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Steering Vector"
  - "Joint Training"
  - "Scaling Theory"
  - "Prompt-only Intervention"
  - "Concept Steering"
date: 2026-05-08
content_hash: c71ef6c2db8fe2c5
---

# Towards Steering without Sacrifice: Principled Training of Steering Vectors for Prompt-only Interventions

**Conference**: ICML 2026  
**arXiv**: [2605.05983](https://arxiv.org/abs/2605.05983)  
**Code**: Not provided in the paper (none)  
**Area**: LLM Control / Representation Engineering / Model Compression  
**Keywords**: Steering Vector, Joint Training, Scaling Theory, Prompt-only Intervention, Concept Steering

## TL;DR
The authors leverage infinite-width neural network scaling theory to derive that joint training of the steering vector’s factor/direction should satisfy the scaling constraint $\eta_{\mathbf{v}}\eta_{\alpha}=\Theta(1)$, thereby eliminating the need for manual selection of $\alpha$ during inference. Inspired by ReFT, they apply additive intervention only to the first 4 prompt tokens (PrOSV). On AxBench, this approach maintains model utility and consistently outperforms full-sequence FSSV across three Gemma2/Qwen2.5 model scales.

## Background & Motivation

**Background**: In controlling large model behavior, prompting is flexible but fragile, while fine-tuning is powerful but costly and lacks interpretability. Steering vectors (SV)—adding a fixed vector $\mathbf{v}$ to a certain residual stream layer—have rapidly gained traction as a lightweight intervention. Fine-tuned SVs outperform non-optimized approaches like DiffMean/SAE.

**Limitations of Prior Work**: Current fine-tuned SV systems face two engineering bottlenecks. First, treating the steering factor $\alpha$ as an external constant during training necessitates grid search for each new SV at inference, requiring hundreds of intervention responses to find the optimal $\alpha$, making cross-concept scaling laborious. Second, mainstream SVs are "full-sequence SVs" (FSSV), intervening on all tokens during both prompt and decode phases, which significantly degrades model utility—a strong trade-off between steering and utility that cannot be avoided even with careful $\alpha$ selection.

**Key Challenge**: Treating $\alpha$ as an external constant leads to a disconnect between training and inference, high SV sensitivity, and mandatory post-hoc selection. Direct end-to-end learning of $\alpha$ and direction seems intuitively better but lacks theoretical guidance on learning rates/initialization, often resulting in unstable or divergent joint training. Meanwhile, FSSV’s intervention across the full sequence disrupts attention patterns and downstream accuracy, but restricting intervention to the prompt raises concerns about insufficient steering strength.

**Goal**: (a) Provide a scaling-theory-based principle for selecting $\eta_{\alpha}, \eta_{\mathbf{v}}, \alpha_0, \mathbf{v}_0$ in SV joint training; (b) Design an SV variant that intervenes only during the prompt phase, minimally affecting decode, yet still achieves concept injection; (c) Validate whether both effectiveness and utility can be improved on the AxBench concept steering benchmark.

**Key Insight**: Treat SV training as learning a low-rank single-layer adapter on a frozen pretrained network, borrowing the infinite-width analysis framework from LoRA scaling theory (Hayou 2024 series). Inspired by ReFT—if low-rank prompt-only interventions suffice for task adaptation, they may also suffice for concept-level steering.

**Core Idea**: Use scaling laws to eliminate inference-time hyperparameters, convert SV to prompt-only intervention—"joint training + local intervention" together upgrade SV from a heuristic experimental tool to an engineering component with theoretical guarantees.

## Method

### Overall Architecture
The method consists of two independent improvements, ultimately combined. First, the SV training framework is upgraded: at a fixed residual stream layer $l$, both direction $\mathbf{v}\in\mathbb{R}^n$ and factor $\alpha\in\mathbb{R}$ are learnable parameters, updated via Adam with learning rates $\eta_{\mathbf{v}}, \eta_{\alpha}$; the intervention is additive: $\Phi^{\text{Add}}(\mathbf{h}; \alpha, \mathbf{v}) = \mathbf{h} + \alpha\mathbf{v}$. Second, the intervention position is modified: traditional FSSV adds $\alpha\mathbf{v}$ to all prompt and decode tokens, while PrOSV adds it only to the first $p$ and last $s$ prompt tokens, with intervention set $\mathcal{I} = \{1,\dots,p\}\cup\{m-s+1,\dots,m\}$, described as $p2{+}s2$ etc. Training objectives can be Language modeling (Lang.) or SimPO preference optimization; at inference, the trained $\alpha_T, \mathbf{v}_T$ are used directly, with no further factor selection.

### Key Designs

1. **SV Joint Training Based on Infinite-Width Scaling Theory**:

    - **Function**: Enables effective updates of both $\alpha$ and $\mathbf{v}$ without destabilizing representations, providing actionable scales for $\eta_{\alpha}, \eta_{\mathbf{v}}, \alpha_0, \mathbf{v}_0$.
    - **Mechanism**: Let SV feature $\mathbf{z} = \alpha\mathbf{v}$; "stability" requires $\mathbf{z}_t = \Theta(1)$, "efficiency" requires each incremental component $(\Delta\alpha)\mathbf{v}_{t-1}$, $\alpha_{t-1}(\Delta\mathbf{v})$, $(\Delta\alpha)(\Delta\mathbf{v})$ to be $\Theta(1)$. Using the $\gamma$-operator, these constraints are formulated as polynomial inequalities, yielding $\eta_{\mathbf{v}}\eta_{\alpha}=\Theta(1)$, with $\gamma[\mathbf{v}_0]\le\gamma[\eta_{\mathbf{v}}]$, $\gamma[\alpha_0]\le\gamma[\eta_{\alpha}]$. In practice, Kaiming initialization is used with $\sigma_{\mathbf{v}}^{2}=\lambda n^{-1}$, $\alpha_0 = \beta n^{1/2}$, and $\eta_{\mathbf{v}}=\Theta(n^{-1/2}), \eta_{\alpha}=\Theta(n^{1/2})$, where $\beta, \lambda$ are tuned once via grid search and reused across concepts.
    - **Design Motivation**: Traditional SV treats $\alpha$ as an external constant, requiring grid search for each new SV; naive joint training often leads to SV feature explosion or vanishing due to mismatched learning rates. Infinite-width scaling analysis ensures all scales are self-consistent, so "tune once, use forever" in practice.

2. **Prompt-Only Steering Vector (PrOSV)**:

    - **Function**: Adds $\alpha\mathbf{v}$ only to the first $p$ and last $s$ prompt tokens, leaving decode untouched; concept injection is achieved via implicit KV cache editing, without ongoing interference during generation.
    - **Mechanism**: Transfers ReFT’s prompt-only intervention philosophy to steering, combined with the above joint training protocol; typical configurations are $p4{+}s4$ for Gemma2-2B/9B and $p2{+}s2$ for Qwen2.5-32B. Since the number of intervened tokens is constant rather than growing with generation length, compared to FSSV, this saves $37\times$ compute on 8K context in practice.
    - **Design Motivation**: FSSV, even with optimal $\alpha$, continuously disrupts attention patterns and degrades utility; intervening only at a few prompt positions minimally impacts attention. FSSV cannot simply be "truncated to prompt"—its optimal direction differs from PrOSV and relies on factor selection to function.

3. **Training Objectives and Inference without Post-hoc Factor Selection**:

    - **Function**: Integrates training objectives and engineering protocols, making "train-and-use" the default workflow, supporting both Lang. and SimPO losses.
    - **Mechanism**: During training, joint updates are performed as in Algorithm 1; at inference, $\alpha_T$ is used directly, with no grid search. For SimPO preference optimization, GPT-4o-mini is used to generate concept-neutral responses $\mathbf{y}_i$ for each prompt as controls, forming contrastive pairs $\mathcal{D}^{c+} = \{(\mathbf{x}_i, \mathbf{y}_i, \mathbf{y}_i^c)\}$ with concept responses $\mathbf{y}_i^c$.
    - **Design Motivation**: The strongest prior baseline, RePS, still requires inference-time factor selection, essentially offloading "untrainable" responsibility to inference. This work shows that with the right training protocol, inference requires no further $\alpha$ selection, greatly simplifying engineering.

### Loss & Training

Two alternative objectives: (i) Language modeling computes NLL only on $\mathbf{y}_i^c$, simple and stable but usually outperformed by SimPO; (ii) SimPO (Meng 2024) as a preference optimization objective, trains on $(\mathbf{y}_i, \mathbf{y}_i^c)$ pairs. Both objectives use Algorithm 1’s joint training loop: $\mathbf{v}_0 \sim \mathcal{N}(\mathbf{0}, \lambda n^{-1}\mathbf{I}_n)$, $\alpha_0 \leftarrow \beta n^{1/2}$, with each Adam step updating $\mathbf{v}_{t+1} \leftarrow \mathbf{v}_t - \eta_{\mathbf{v}} g^{\mathbf{v}}_t$, $\alpha_{t+1} \leftarrow \alpha_t - \eta_{\alpha} g^{\alpha}_t$. Hyperparameters $\beta \in \{1, 2, 4, 8\}$, $\lambda \in \{1, 8\}$, $\eta_{\alpha}$ are grid-searched across 4 log scales, but tuned only once on 3 dev concepts.

## Key Experimental Results

### Main Results
AxBench overall steering score (0–2, higher is better), covering Gemma2-2B-L10, Gemma2-9B-L20, Qwen2.5-32B-L32; compares prompt, LoReFT, DiffMean, SAE, FSSV (Lang./SimPO), and this work.

| Method | G2B-L10 | G9B-L20 | Q32B-L32 | Note |
|---|---|---|---|---|
| Prompting | 0.698 | 1.075 | 1.060 | Prompt gain saturates on large models |
| FSSV (Lang.) | 0.663 | 0.788 | 0.798 | Requires post-hoc factor selection |
| FSSV + Joint Training | 0.736 | 0.821 | 0.919 | Training improvement only—already surpasses baseline |
| PrOSV (Lang.) | 0.758 | 0.859 | 1.049 | Intervention set reduced to a few tokens |
| FSSV (SimPO, RePS) | 0.756 | 0.892 | 0.947 | Previous SOTA |
| **PrOSV (SimPO)** | **0.803** | **0.905** | **1.102** | SOTA across all scales |

### Ablation Study
Intervention position and budget (best overall O / concept C scores, 0–2):

| Intervention Position | G2B O/C | G9B O/C | Q32B O/C |
|---|---|---|---|
| FSSV (full) | 0.65 / 0.97 | 0.86 / 1.17 | 0.93 / 1.27 |
| Full prompt | 0.54 / 1.12 | 0.71 / 1.41 | 0.88 / 1.58 |
| $p2{+}s2$ | **0.70** / 0.82 | **0.92** / 1.14 | **1.16** / 1.33 |
| $p4{+}s4$ | 0.69 / 0.85 | 0.89 / 1.09 | 1.13 / 1.30 |
| $p1{+}s1$ | 0.67 / 0.79 | 0.91 / 1.12 | 1.10 / 1.24 |

### Key Findings
- Steering vectors are extremely sensitive to hyperparameters, but $\beta>1$, $\eta_{\alpha}>\eta_{\mathbf{v}}$ are common features of nearly all optimal solutions—precisely confirming the theory that the factor requires large initialization and learning rate.
- Full-prompt intervention achieves the highest concept score but the lowest overall score, indicating that "hard concept injection" comes at the expense of utility; moderate prefix+suffix ($p2{+}s2$) offers the best trade-off between intervention strength and generation quality.
- On tinyGSM8K arithmetic reasoning, PrOSV’s accuracy degradation (18–29%) is much less than FSSV’s (68–90%), indicating that local attention intervention truly preserves utility.
- Under concept suppression adversarial attacks, FSSV cannot escape the robustness-utility trade-off even when reducing the factor from 100% to 50%, while PrOSV lies on a better Pareto frontier.
- On Qwen2.5-32B, PrOSV remains robust for contexts up to ~1K tokens, not failing despite intervening on only 4 tokens, suggesting that as model scale increases, "less intervention" can better amplify SV capability.

## Highlights & Insights
- LoRA scaling theory is transferred to SV training at almost "zero cost," immediately addressing the question of "why are my trained SVs so unstable." SV design is an engineering problem solved with representation learning theory, with strong transferability.
- The intuition that "more intervention is better" is overturned: for concept-level steering, intervening on just a few prompt tokens yields better overall results. This aligns with ReFT’s experience that low-rank interventions suffice for task adaptation, now extended to the concept domain.
- One-time hyperparameter selection and no inference-time tuning mean SVs can be distributed "train-and-use" like fine-tuned weights—a qualitative leap for open-source and engineering deployment.

## Limitations & Future Work
- Only fine-tuned SVs are studied; no principled factor recommendations are given for optimization-free SVs like DiffMean. Future work may treat such SVs as pretrained SVs with additional scaling analysis.
- Intervention positions are limited to simple prefix/suffix templates; more general attention-aware position selection is unexplored—there may be better token subsets.
- Only Lang. and SimPO training objectives are tried; results show the objective impacts performance even more than the training protocol, leaving ample design space for future work.
- The "utility vs adversarial robustness" trade-off, though improved in PrOSV over FSSV, is not eliminated, indicating that stricter safety control with SVs requires deeper objective design.

## Related Work & Insights
- **vs ReFT (Wu 2024b)**: ReFT is prompt-only fine-tuning for task adaptation; this work transfers its intervention position strategy to concept steering SVs, adding scaling theory.
- **vs RePS (Wu 2025b)**: Also modifies SV training protocol, but RePS still requires post-hoc factor selection; this work + SimPO advances RePS by 0.01–0.16 overall score across all scales.
- **vs SAE / DiffMean**: SAE relies on selecting relevant directions from massive features, DiffMean is a simple mean difference; both require post-processing for direction selection or lack effectiveness. PrOSV offers a more engineering-oriented fine-tuned SV route.

## Rating
- Novelty: ⭐⭐⭐⭐ Cleanly transfers LoRA scaling theory to SVs + combines with ReFT ideas—a clearly original combination, though not entirely new tools individually.
- Experimental Thoroughness: ⭐⭐⭐⭐ AxBench Concept500, tinyMMLU/GSM8K, adversarial attacks, long context, and multiple model families (2B/9B/32B) are comprehensively covered.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivation and engineering implementation (Algorithm 1) are seamlessly integrated, with clear pseudocode and scaling parameter correspondence.
- Value: ⭐⭐⭐⭐ Directly eliminates the biggest engineering burden in SV inference, providing a ready-to-use contribution for teams deploying representation steering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Compositional Steering of Large Language Models with Steering Tokens](../../ACL2026/model_compression/compositional_steering_of_large_language_models_with_steering_tokens.md)
- [\[AAAI 2026\] Steering Pretrained Drafters during Speculative Decoding](../../AAAI2026/model_compression/steering_pretrained_drafters_during_speculative_decoding.md)
- [\[ACL 2026\] From Weights to Activations: Is Steering the Next Frontier of Adaptation?](../../ACL2026/model_compression/from_weights_to_activations_is_steering_the_next_frontier_of_adaptation.md)
- [\[ICLR 2026\] Steering MoE LLMs via Expert (De)Activation](../../ICLR2026/model_compression/steering_moe_llms_via_expert_deactivation.md)
- [\[CVPR 2026\] FOZO: Forward-Only Zeroth-Order Prompt Optimization for Test-Time Adaptation](../../CVPR2026/model_compression/fozo_forward-only_zeroth-order_prompt_optimization_for_test-time_adaptation.md)

</div>

<!-- RELATED:END -->
