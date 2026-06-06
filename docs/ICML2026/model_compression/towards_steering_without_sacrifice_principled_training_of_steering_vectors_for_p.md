---
title: >-
  [Paper Note] Towards Steering without Sacrifice: Principled Training of Steering Vectors for Prompt-only Interventions
description: >-
  [ICML 2026][Model Compression][Steering Vectors] The authors derive a scaling constraint $\eta_{\mathbf{v}}\eta_{\alpha}=\Theta(1)$ for the joint training of steering vector factor/direction using neural network infinite…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Steering Vectors"
  - "Joint Training"
  - "Scaling Theory"
  - "Prompt-only Interventions"
  - "Concept Steering"
date: 2026-05-08
content_hash: 2945ada940434ad5
---

# Towards Steering without Sacrifice: Principled Training of Steering Vectors for Prompt-only Interventions

**Conference**: ICML 2026  
**arXiv**: [2605.05983](https://arxiv.org/abs/2605.05983)  
**Code**: Not provided in the paper (None)  
**Area**: LLM Control / Representation Engineering / Model Compression  
**Keywords**: Steering Vectors, Joint Training, Scaling Theory, Prompt-only Interventions, Concept Steering

## TL;DR
The authors derive a scaling constraint $\eta_{\mathbf{v}}\eta_{\alpha}=\Theta(1)$ for the joint training of steering vector factor/direction using neural network infinite-width scaling theory, thereby eliminating manual $\alpha$ selection during inference. Inspired by ReFT, the authors implement additive interventions only on the first 4 prompt tokens (PrOSV), which maintains model utility while consistently outperforming full-sequence FSSV across three sizes of Gemma2 and Qwen2.5 models on AxBench.

## Background & Motivation

- **Background**: For controlling Large Language Model (LLM) behavior, prompting is flexible but fragile, while fine-tuning is powerful but expensive and uninterpretable. Steering vectors (SV), which act as a lightweight intervention by adding a fixed vector $\mathbf{v}$ to the residual stream of a specific layer, have gained popularity. Fine-tuned SVs typically outperform non-optimization schemes like DiffMean or SAE.

- **Limitations of Prior Work**: Current fine-tuned SV systems suffer from two implementation issues. First, the steering factor $\alpha$ is treated as an external constant during training, necessitating an expensive grid search for every new SV during inference to find the optimal $\alpha$, which hinders scalability. Second, mainstream SVs are "Full-Sequence SVs" (FSSV) that intervene on all tokens during both prompt and decode phases, significantly damaging the model's general utility—an unavoidable trade-off even with a carefully selected $\alpha$.

- **Key Challenge**: Treating $\alpha$ as an external constant leads to a disconnect between training and inference, making SVs highly sensitive. Simply learning $\alpha$ end-to-end with the direction lacks theoretical guidance for learning rates and initialization, often resulting in unstable or divergent joint training. Furthermore, FSSV disrupts attention patterns, whereas solely intervening during the prompt phase raises concerns about steering strength.

- **Goal**: (a) Provide a principled set of scaling laws for SV joint training to determine $\eta_{\alpha}, \eta_{\mathbf{v}}, \alpha_0, \mathbf{v}_0$; (b) Design an SV variant that intervenes only during the prompt phase with minimal impact on decoding while achieving concept steering; (c) Verify whether these improvements simultaneously enhance effectiveness and utility on the AxBench concept steering benchmark.

- **Key Insight**: SV training is treated as learning a low-rank single-layer adapter on a frozen pre-trained network, utilizing the infinite-width analysis framework from LoRA scaling theory (Hayou 2024 series). Inspired by ReFT—where low-rank prompt-only interventions suffice for task adaptation—it is hypothesized that such interventions are also sufficient for concept-level steering.

- **Core Idea**: Use scaling laws to eliminate inference-time hyperparameters and convert SVs to prompt-only interventions. This dual approach of "joint training + local intervention" upgrades SVs from heuristic experimental tools to theoretically grounded engineering components.

## Method

### Overall Architecture
The method consists of two independent improvement paths used in combination. The first is an upgrade to the SV training framework: at a fixed residual stream position in layer $l$, both the direction $\mathbf{v}\in\mathbb{R}^n$ and the factor $\alpha\in\mathbb{R}$ are included as learnable parameters, updated via Adam with learning rates $\eta_{\mathbf{v}}$ and $\eta_{\alpha}$, respectively. The intervention takes the additive form $\Phi^{\text{Add}}(\mathbf{h}; \alpha, \mathbf{v}) = \mathbf{h} + \alpha\mathbf{v}$. The second path is the modification of intervention locations: unlike traditional FSSV, which adds $\alpha\mathbf{v}$ to all tokens, PrOSV intervenes only on the first $p$ and last $s$ tokens of the prompt. The intervention set is $\mathcal{I} = \{1,\dots,p\}\cup\{m-s+1,\dots,m\}$, with configurations described by shorthand such as $p2{+}s2$. Training objectives include Language Modeling (Lang.) or SimPO preference optimization. At inference, the trained parameters $\alpha_T$ and $\mathbf{v}_T$ are used directly without further tuning.

### Key Designs

1. **SV Joint Training Based on Infinite-Width Scaling Theory**:
    - **Function**: Effectively updates both $\alpha$ and $\mathbf{v}$ without damaging representation stability, providing actionable scales for $\eta_{\alpha}, \eta_{\mathbf{v}}, \alpha_0, \mathbf{v}_0$.
    - **Mechanism**: Let the SV feature be $\mathbf{z} = \alpha\mathbf{v}$. Stability requires $\mathbf{z}_t = \Theta(1)$, and efficiency requires that the three components of each incremental step—$(\Delta\alpha)\mathbf{v}_{t-1}$, $\alpha_{t-1}(\Delta\mathbf{v})$, and $(\Delta\alpha)(\Delta\mathbf{v})$—are all $\Theta(1)$. Using the $\gamma$-operator to represent these constraints as polynomial inequalities yields the solution $\eta_{\mathbf{v}}\eta_{\alpha}=\Theta(1)$, with $\gamma[\mathbf{v}_0]\le\gamma[\eta_{\mathbf{v}}]$ and $\gamma[\alpha_0]\le\gamma[\eta_{\alpha}]$. Implementation uses Kaiming initialization $\sigma_{\mathbf{v}}^{2}=\lambda n^{-1}$, setting $\alpha_0 = \beta n^{1/2}$ and taking $\eta_{\mathbf{v}}=\Theta(n^{-1/2}), \eta_{\alpha}=\Theta(n^{1/2})$. The scalars $\beta$ and $\lambda$ are tuned once via grid search and reused across concepts.
    - **Design Motivation**: Traditional SVs treat $\alpha$ as a constant, requiring per-concept grid searches. Naive joint training often leads to SV feature explosion or decay due to mismatched learning rates. Infinite-width analysis ensures self-consistent scaling, allowing "tune once, apply everywhere."

2. **Prompt-Only Steering Vector (PrOSV)**:
    - **Function**: Adds $\alpha\mathbf{v}$ only to the prefix $p$ and suffix $s$ prompt tokens, leaving the decode phase untouched. It steers generation by implicitly editing the KV cache without persistent interference.
    - **Mechanism**: The prompt-only intervention concept from ReFT is migrated to steering, predicated on the joint training protocol. Typical configurations include $p4{+}s4$ for Gemma2-2B/9B and $p2{+}s2$ for Qwen2.5-32B. Since the number of intervened tokens is constant rather than scaling with generation length, computational overhead is reduced by $37\times$ compared to FSSV for an 8K context.
    - **Design Motivation**: FSSV persistently interferes with attention patterns even with an optimized $\alpha$, crushing utility. Modifying the KV cache at a few prompt positions minimizes the footprint on attention. FSSV cannot simply be truncated to the prompt; its optimal direction differs from PrOSV and relies on factor selection.

3. **Training Objective and Inference Workflow without Post-hoc Factor Selection**:
    - **Function**: Bundles the training objective with engineering protocols to make "train-and-use" the default workflow, supporting both Lang. and SimPO losses.
    - **Mechanism**: Training follows Algorithm 1 for joint updates. Inference uses $\alpha_T$ directly without grid searching. For SimPO preference optimization, GPT-4o-mini generates concept-neutral responses $\mathbf{y}_i$ as a baseline to form contrastive pairs $\mathcal{D}^{c+} = \{(\mathbf{x}_i, \mathbf{y}_i, \mathbf{y}_i^c)\}$.
    - **Design Motivation**: Previous state-of-the-art baselines like RePS still rely on inference-time factor selection, essentially externalizing training failures to inference. This work demonstrates that with the correct training protocol, $\alpha$ selection is unnecessary.

### Loss & Training
Two objectives are supported: (i) Language modeling, utilizing only the NLL of $\mathbf{y}_i^c$, which is stable but usually outperformed by SimPO; (ii) SimPO (Meng 2024), which trains using $(\mathbf{y}_i, \mathbf{y}_i^c)$ preference pairs. Both utilize the joint training loop in Algorithm 1: $\mathbf{v}_0 \sim \mathcal{N}(\mathbf{0}, \lambda n^{-1}\mathbf{I}_n)$, $\alpha_0 \leftarrow \beta n^{1/2}$, with updates $\mathbf{v}_{t+1} \leftarrow \mathbf{v}_t - \eta_{\mathbf{v}} g^{\mathbf{v}}_t$ and $\alpha_{t+1} \leftarrow \alpha_t - \eta_{\alpha} g^{\alpha}_t$ using Adam. Hyperparameters $\beta \in \{1, 2, 4, 8\}$, $\lambda \in \{1, 8\}$, and $\eta_{\alpha}$ are swept across 4 log scales on the dev set across only 3 concepts once.

## Key Experimental Results

### Main Results
Overall steering scores on AxBench (Overall score, 0–2, higher is better), covering Gemma2-2B-L10, Gemma2-9B-L20, and Qwen2.5-32B-L32. Baselines include Prompting, LoReFT, DiffMean, SAE, and FSSV (Lang./SimPO).

| Method | G2B-L10 | G9B-L20 | Q32B-L32 | Remarks |
|---|---|---|---|---|
| Prompting | 0.698 | 1.075 | 1.060 | Saturated gains on large models |
| FSSV (Lang.) | 0.663 | 0.788 | 0.798 | Requires post-hoc factor selection |
| FSSV + Joint Training | 0.736 | 0.821 | 0.919 | Exceeds baseline via training only |
| PrOSV (Lang.) | 0.758 | 0.859 | 1.049 | Intervention on only a few tokens |
| FSSV (SimPO, RePS) | 0.756 | 0.892 | 0.947 | Prev. SOTA |
| **PrOSV (SimPO)** | **0.803** | **0.905** | **1.102** | SOTA across all sizes |

### Ablation Study
Intervention position and budget (optimal overall score O / concept score C, 0–2):

| Intervention Position | G2B O/C | G9B O/C | Q32B O/C |
|---|---|---|---|
| FSSV (full) | 0.65 / 0.97 | 0.86 / 1.17 | 0.93 / 1.27 |
| Full prompt | 0.54 / 1.12 | 0.71 / 1.41 | 0.88 / 1.58 |
| $p2{+}s2$ | **0.70** / 0.82 | **0.92** / 1.14 | **1.16** / 1.33 |
| $p4{+}s4$ | 0.69 / 0.85 | 0.89 / 1.09 | 1.13 / 1.30 |
| $p1{+}s1$ | 0.67 / 0.79 | 0.91 / 1.12 | 1.10 / 1.24 |

### Key Findings
- Steering vectors are extremely sensitive to hyperparameters, but $\beta>1$ and $\eta_{\alpha}>\eta_{\mathbf{v}}$ characterize nearly all optimal solutions, validating the theory that factors require larger initialization and learning rates.
- Full-prompt intervention achieves the highest concept score but the lowest overall score, suggesting that "forcing" concepts comes at the expense of utility. A moderate prefix+suffix ($p2{+}s2$) offers the best compromise.
- PrOSV shows significantly less damage to model accuracy on tinyGSM8K arithmetic reasoning (18–29%) compared to FSSV (68–90%), indicating that local attention intervention protects utility.
- Under concept suppression adversarial attacks, FSSV cannot escape the robustness-utility trade-off even when reducing factors from 100% to 50%, whereas PrOSV lies on a superior Pareto front.
- On Qwen2.5-32B, PrOSV remains robust for long contexts (~1K tokens) despite only intervening on 4 tokens, suggesting that "minor intervention" better amplifies SV capabilities as model scale increases.

## Highlights & Insights
- Implementing LoRA scaling theory for SV training effectively "for free" resolves the issue of high variance in trained SVs.
- The intuition that "more intervention is better" is challenged: in concept-level steering, restricting the intervention set to a few prompt tokens yields better overall results. This aligns with ReFT's experience that low-rank intervention is sufficient for task adaptation and extends it to the conceptual domain.
- One-time hyperparameter selection without inference-time tuning allows SVs to be distributed as "ready-to-use" components, similar to fine-tuned weights, representing a significant shift for open-source deployment.

## Limitations & Future Work
- The study focuses on fine-tuned SVs; principled factor recommendations for optimization-free SVs like DiffMean remain unexplored.
- The intervention locations are restricted to simple prefix/suffix templates; more universal attention-aware selection may exist.
- Only Lang. and SimPO objectives were tested; the objective's impact on performance was found to be even larger than the training protocol, leaving significant design space.
- The utility vs. adversarial robustness trade-off remains present in PrOSV, though improved, suggesting that strict safety control via SV requires more sophisticated objective designs.

## Related Work & Insights
- **vs. ReFT (Wu 2024b)**: ReFT uses prompt-only fine-tuning for task adaptation; Ours migrates the intervention location strategy to concept-guided SVs and adds scaling theory.
- **vs. RePS (Wu 2025b)**: Both modify SV training protocols, but RePS still requires post-hoc factor selection; Ours + SimPO improves upon RePS by 0.01–0.16 overall score points across all test cases.
- **vs. SAE / DiffMean**: SAE depends on selecting directions from massive features, while DiffMean uses simple mean differences. Both either require post-processing direction selection or lack effectiveness; PrOSV provides a more engineered fine-tuned route.

## Rating
- Novelty: ⭐⭐⭐⭐ Porting LoRA scaling theory to SVs combined with ReFT principles is a clear original combination innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Coverage of AxBench Concept500, tinyMMLU/GSM8K, adversarial attacks, long context, and multiple model families (2B/9B/32B).
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical derivation, Algorithm 1 pseudo-code, and scaling parameter comparisons.
- Value: ⭐⭐⭐⭐ Eliminates the primary engineering burden of SV inference, making it a "ready-to-use" contribution for representation steering deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Steering Pretrained Drafters during Speculative Decoding](../../AAAI2026/model_compression/steering_pretrained_drafters_during_speculative_decoding.md)
- [\[ICLR 2026\] Steering MoE LLMs via Expert (De)Activation](../../ICLR2026/model_compression/steering_moe_llms_via_expert_deactivation.md)
- [\[ACL 2026\] Why Steering Works: Toward a Unified View of Language Model Parameter Dynamics](../../ACL2026/model_compression/why_steering_works_toward_a_unified_view_of_language_model_parameter_dynamics.md)
- [\[ICML 2026\] Multi-Adapter Representation Interventions via Energy Calibration](multi-adapter_representation_interventions_via_energy_calibration.md)
- [\[CVPR 2026\] FOZO: Forward-Only Zeroth-Order Prompt Optimization for Test-Time Adaptation](../../CVPR2026/model_compression/fozo_forward-only_zeroth-order_prompt_optimization_for_test-time_adaptation.md)

</div>

<!-- RELATED:END -->
