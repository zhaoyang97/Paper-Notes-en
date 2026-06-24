---
title: >-
  [Paper Note] Towards Steering without Sacrifice: Principled Training of Steering Vectors for Prompt-only Interventions
description: >-
  [ICML 2026][Model Compression][Steering Vectors] The authors derive a scaling constraint for the joint training of steering vector factors and directions, $\eta_{\mathbf{v}}\eta_{\alpha}=\Theta(1)$, using infinite-width neural network scaling theory, thereby eliminating manual $\alpha$ selection during inference. Simultaneously inspired by ReFT, they implement additive interventions only on the first 4 prompt tokens (PrOSV). This approach maintains model utility while consist…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Steering Vectors"
  - "Joint Training"
  - "Scaling Theory"
  - "Prompt-only Intervention"
  - "Concept Guidance"
date: 2026-05-08
content_hash: 707bdaffe20cf491
---

# Towards Steering without Sacrifice: Principled Training of Steering Vectors for Prompt-only Interventions

**Conference**: ICML 2026  
**arXiv**: [2605.05983](https://arxiv.org/abs/2605.05983)  
**Code**: Not provided in the paper (None)  
**Area**: LLM Control / Representation Engineering / Model Compression  
**Keywords**: Steering Vectors, Joint Training, Scaling Theory, Prompt-only Intervention, Concept Guidance

## TL;DR
The authors derive a scaling constraint for the joint training of steering vector factors and directions, $\eta_{\mathbf{v}}\eta_{\alpha}=\Theta(1)$, using infinite-width neural network scaling theory, thereby eliminating manual $\alpha$ selection during inference. Simultaneously inspired by ReFT, they implement additive interventions only on the first 4 prompt tokens (PrOSV). This approach maintains model utility while consistently outperforming full-sequence FSSV across three scales of Gemma2 and Qwen2.5 models on AxBench.

## Background & Motivation

**Background**: When controlling Large Language Model (LLM) behavior, prompting is flexible but fragile, while fine-tuning is powerful but expensive and uninterpretable. Steering vectors (SV), a lightweight intervention method that adds a fixed vector $\mathbf{v}$ to a residual stream at a certain layer, have risen rapidly. Among these, fine-tuned SVs obtained through optimization perform better than optimization-free schemes like DiffMean or SAE.

**Limitations of Prior Work**: Current fine-tuned SV frameworks suffer from two engineering bottlenecks. First, the steering factor $\alpha$ is treated as an external constant during training, necessitating a grid search for each new SV during inference. This requires sampling hundreds of responses to find the optimal $\alpha$, making cross-concept scaling extremely laborious. Second, mainstream SVs are "Full-Sequence SVs" (FSSV), intervening on all tokens during both prompt and decode phases, which significantly damages the model's general capabilities—a trade-off between steering and utility that even a carefully selected $\alpha$ cannot evade.

**Key Challenge**: Treating $\alpha$ as an external constant leads to a disconnect between training and inference, high SV sensitivity, and the need for post-hoc selection. Learning $\alpha$ end-to-end alongside the direction seems intuitively superior but lacks theoretical guidance for learning rates and initialization; joint training often exhibits instability or divergence. Meanwhile, FSSV interferes with the entire sequence, disrupting attention patterns and harming downstream accuracy, yet there are concerns that intervening only on the prompt might lack steering strength.

**Goal**: (a) Formulate a principled protocol for SV joint training based on scaling theory for selecting $\eta_{\alpha}, \eta_{\mathbf{v}}, \alpha_0, \mathbf{v}_0$; (b) design an SV variant that intervenes only during the prompt phase with minimal impact on decoding while still achieving concept implantation; (c) verify if these contributions simultaneously improve effectiveness and utility on the AxBench concept steering benchmark.

**Key Insight**: SV training is viewed as learning a low-rank single-layer adapter on top of a frozen pre-trained network, utilizing the infinite-width analysis framework from LoRA scaling theory (Hayou 2024 series). Inspired by ReFT—since low-rank prompt-only intervention suffices for task adaptation—it may also be sufficient for concept-level steering.

**Core Idea**: Scaling laws are used to eliminate inference-time hyperparameters, and SV is modified for prompt-only intervention. By combining "joint training + local intervention," SV is upgraded from a heuristic experimental tool to an engineering component with theoretical guarantees.

## Method

### Overall Architecture
The authors address two engineering issues of fine-tuned steering vectors (SV): the need for manual factor $\alpha$ selection and the destruction of utility by full-sequence intervention. Two improvements are implemented in parallel: on the training side, the direction $\mathbf{v}\in\mathbb{R}^n$ and the factor $\alpha\in\mathbb{R}$ are learned end-to-end; on the intervention side, only a few prompt tokens are modified. Specifically, an additive intervention $\Phi^{\text{Add}}(\mathbf{h}; \alpha, \mathbf{v}) = \mathbf{h} + \alpha\mathbf{v}$ is applied to the residual stream at a fixed layer $l$, with $\alpha$ and $\mathbf{v}$ updated via Adam with learning rates $\eta_{\alpha}$ and $\eta_{\mathbf{v}}$. Training is constrained to a set of tokens $\mathcal{I} = \{1,\dots,p\}\cup\{m-s+1,\dots,m\}$ (denoted as $p2{+}s2$, etc.), shrinking the intervention set from all tokens in FSSV to only the first $p$ and last $s$ prompt tokens. Training objectives include Language Modeling or SimPO. Once $\alpha_T$ and $\mathbf{v}_T$ are obtained, they are used directly during inference.

### Key Designs

**1. Joint Training of SV based on Infinite-Width Scaling Theory: Learning $\alpha$ and $\mathbf{v}$ without Divergence**

Traditional SV treats $\alpha$ as an external constant, requiring a grid search for every new SV. Attempting to learn $\alpha$ and the direction end-to-end often leads to the SV feature exploding or vanishing due to mismatched learning rates. The authors treat this as learning a low-rank single-layer adapter on a frozen network and apply LoRA scaling theory to provide actionable benchmarks. To ensure the SV feature $\mathbf{z} = \alpha\mathbf{v}$ is "stable" ($\mathbf{z}_t = \Theta(1)$) and "efficient" (each incremental component is $\Theta(1)$), they use a $\gamma$-operator to solve polynomial inequality constraints, yielding the core scaling:

$$\eta_{\mathbf{v}}\eta_{\alpha}=\Theta(1),\quad \gamma[\mathbf{v}_0]\le\gamma[\eta_{\mathbf{v}}],\quad \gamma[\alpha_0]\le\gamma[\eta_{\alpha}].$$

In practice, Kaiming initialization $\sigma_{\mathbf{v}}^{2}=\lambda n^{-1}$ is used alongside $\alpha_0 = \beta n^{1/2}$, with learning rates $\eta_{\mathbf{v}}=\Theta(n^{-1/2})$ and $\eta_{\alpha}=\Theta(n^{1/2})$—meaning the factor requires large initialization and a high learning rate, while the direction requires smaller values. These self-consistent scales allow $\beta$ and $\lambda$ to be tuned once and reused across concepts, achieving a "tune once, use forever" workflow.

**2. Prompt-Only Steering Vector (PrOSV): Modifying Only a Few Prompt Tokens**

Even with a carefully selected $\alpha$, FSSV continually disrupts attention patterns by adding $\alpha\mathbf{v}$ throughout the prompt and decode phases, compromising utility. PrOSV, inspired by ReFT, injects $\alpha\mathbf{v}$ only into the prefix $p$ and suffix $s$ prompt tokens, implicitly editing the KV cache to seed the concept while leaving the decode phase untouched. This minimizes the footprint on attention. This method necessitates the joint training protocol mentioned above (typical configurations include $p4{+}s4$ for Gemma2-2B/9B and $p2{+}s2$ for Qwen2.5-32B). Notably, FSSVs cannot simply be truncated to the prompt; their optimal direction differs from PrOSV and inherently depends on factor selection. Because the intervention token count is constant rather than growing with sequence length, PrOSV is approximately $37\times$ more computationally efficient than FSSV on 8K contexts.

**3. Training Objective and Post-hoc Inference Flow: Ready-to-Use After Training**

The previous SOTA baseline, RePS, attributed the failure to train good SVs to the inference phase, still requiring factor selection. This paper proves that with a proper training protocol, $\alpha_T$ can be used directly without any grid search. Training follows Algorithm 1 for joint updates, supporting both Language Modeling and SimPO losses. Preference optimization like SimPO requires negative samples; the authors use gpt-4o-mini to generate concept-neutral responses $\mathbf{y}_i$ as a baseline for concept responses $\mathbf{y}_i^c$, forming contrastive triplets $\mathcal{D}^{c+} = \{(\mathbf{x}_i, \mathbf{y}_i, \mathbf{y}_i^c)\}$.

### Loss & Training
Two objectives are available: (i) Language Modeling, which calculates NLL only on $\mathbf{y}_i^c$, simple and stable but usually outperformed by SimPO; (ii) SimPO (Meng 2024) as a preference optimization objective. Both are integrated into the Algorithm 1 joint training loop: $\mathbf{v}_0 \sim \mathcal{N}(\mathbf{0}, \lambda n^{-1}\mathbf{I}_n)$, $\alpha_0 \leftarrow \beta n^{1/2}$. After Adam updates, $\mathbf{v}_{t+1} \leftarrow \mathbf{v}_t - \eta_{\mathbf{v}} g^{\mathbf{v}}_t$ and $\alpha_{t+1} \leftarrow \alpha_t - \eta_{\alpha} g^{\alpha}_t$. Hyperparameters $\beta \in \{1, 2, 4, 8\}$, $\lambda \in \{1, 8\}$, and $\eta_{\alpha}$ are scanned once on a dev set of 3 concepts.

## Key Experimental Results

### Main Results
Overall steering scores on AxBench (0–2, higher is better) for Gemma2-2B-L10, Gemma2-9B-L20, and Qwen2.5-32B-L32.

| Method | G2B-L10 | G9B-L20 | Q32B-L32 | Notes |
|---|---|---|---|---|
| Prompting | 0.698 | 1.075 | 1.060 | Gain saturates on large models |
| FSSV (Lang.) | 0.663 | 0.788 | 0.798 | Requires post-hoc selection |
| FSSV + Joint Training | 0.736 | 0.821 | 0.919 | Training improvement alone |
| PrOSV (Lang.) | 0.758 | 0.859 | 1.049 | Minimal intervention set |
| FSSV (SimPO, RePS) | 0.756 | 0.892 | 0.947 | Prev. SOTA |
| **PrOSV (SimPO)** | **0.803** | **0.905** | **1.102** | SOTA across all scales |

### Ablation Study
Intervention location and budget (optimal overall O / concept C score, 0–2):

| Intervention Location | G2B O/C | G9B O/C | Q32B O/C |
|---|---|---|---|
| FSSV (full) | 0.65 / 0.97 | 0.86 / 1.17 | 0.93 / 1.27 |
| Full prompt | 0.54 / 1.12 | 0.71 / 1.41 | 0.88 / 1.58 |
| $p2{+}s2$ | **0.70** / 0.82 | **0.92** / 1.14 | **1.16** / 1.33 |
| $p4{+}s4$ | 0.69 / 0.85 | 0.89 / 1.09 | 1.13 / 1.30 |
| $p1{+}s1$ | 0.67 / 0.79 | 0.91 / 1.12 | 1.10 / 1.24 |

### Key Findings
- Steering vectors are extremely sensitive to hyperparameters, but $\beta>1$ and $\eta_{\alpha}>\eta_{\mathbf{v}}$ are common features of almost all optimal solutions—validating the theory that the factor needs large initialization and a high learning rate.
- Full-prompt intervention yields the highest concept score but the lowest overall score, indicating that "forcing concepts" comes at the cost of utility. A moderate prefix+suffix ($p2{+}s2$) is the best trade-off between intervention strength and generation quality.
- PrOSV's damage to model accuracy on tinyGSM8K arithmetic reasoning (18–29%) is significantly less than that of FSSV (68–90%), showing that local intervention on attention provides genuine utility protection.
- Under concept suppression adversarial attacks, FSSV cannot escape the strong robustness-utility trade-off even when reducing factors, whereas PrOSV lands on a better Pareto frontier.
- On Qwen2.5-32B, PrOSV remains robust for long contexts (~1K tokens), indicating that as model scale increases, "less intervention" better amplifies SV capabilities.

## Highlights & Insights
- Migrating LoRA scaling theory to SV training with "zero cost" solves the problem of unstable SV training through rigorous derivation. Designing SV is an engineering problem solved with representation learning theoretical tools.
- The intuition that "more intervention is better" is challenged: in concept-level steering, shrinking the intervention set to a few prompt tokens yields better overall results. This aligns with ReFT's experience that low-rank intervention suffices for task adaptation and extends it to the concept domain.
- One-time hyperparameter selection and no inference-time tuning mean SVs can be distributed as "ready-to-use" weights, a qualitative change for open-source ecosystems and engineering deployment.

## Limitations & Future Work
- Only fine-tuned SVs were studied; no principled factor recommendations are provided for optimization-free SVs like DiffMean. Future work could view these as pre-trained SVs and apply scaling analysis.
- Intervention locations were limited to simple prefix/suffix templates; more general attention-aware location selection was not explored.
- Only Lang. and SimPO objectives were tested; the results show the objective's impact is potentially greater than the training protocol itself, leaving significant design space.
- The utility vs. adversarial robustness trade-off is improved in PrOSV but not eliminated, indicating that strict safety control via SV requires more sophisticated objective designs.

## Related Work & Insights
- **vs ReFT (Wu 2024b)**: ReFT is prompt-only fine-tuning for task adaptation. Ours migrates its intervention location strategy to concept steering SV and adds scaling theory.
- **vs RePS (Wu 2025b)**: Also modifies the SV training protocol, but RePS still requires post-hoc factor selection. Ours + SimPO outperforms RePS by 0.01–0.16 overall score across all scales.
- **vs SAE / DiffMean**: The former relies on picking relevant directions from massive features; the latter is a simple mean difference. Both either require post-processing to select directions or lack effectiveness. PrOSV provides a more engineered fine-tuned SV route.

## Rating
- Novelty: ⭐⭐⭐⭐ Cleanly porting LoRA scaling theory to SV and combining it with ReFT ideas is a clear combination innovation, though the individual tools are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Excellent coverage across AxBench Concept500, tinyMMLU/GSM8K, adversarial attacks, long context, and multiple model families (2B/9B/32B).
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivation and implementation (Algorithm 1) are coherent, with clear pseudo-code and scaling parameter comparisons.
- Value: ⭐⭐⭐⭐ Directly eliminates the largest engineering burden of the SV inference phase, a ready-to-use contribution for teams deploying representation steering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Steering Pretrained Drafters during Speculative Decoding](../../AAAI2026/model_compression/steering_pretrained_drafters_during_speculative_decoding.md)
- [\[ICLR 2026\] Steering MoE LLMs via Expert (De)Activation](../../ICLR2026/model_compression/steering_moe_llms_via_expert_deactivation.md)
- [\[ICLR 2026\] ODESteer: A Unified ODE-Based Steering Framework for LLM Alignment](../../ICLR2026/model_compression/odesteer_a_unified_ode-based_steering_framework_for_llm_alignment.md)
- [\[ICML 2026\] Multi-Adapter Representation Interventions via Energy Calibration](multi-adapter_representation_interventions_via_energy_calibration.md)
- [\[ACL 2026\] Why Steering Works: Toward a Unified View of Language Model Parameter Dynamics](../../ACL2026/model_compression/why_steering_works_toward_a_unified_view_of_language_model_parameter_dynamics.md)

</div>

<!-- RELATED:END -->
