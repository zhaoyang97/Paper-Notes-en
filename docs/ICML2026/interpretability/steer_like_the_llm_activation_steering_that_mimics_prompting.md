---
title: >-
  [Paper Note] Steer Like the LLM: Activation Steering that Mimics Prompting
description: >-
  [ICML 2026][Interpretability][activation steering] This paper reinterprets "prompt steering" as a form of activation steering implemented by the LLM itself. It uses a **per-token ReLU probe** to distill the activation di…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "activation steering"
  - "prompt steering"
  - "token-specific coefficients"
  - "ReLU probe"
  - "PSR"
date: 2026-05-08
content_hash: 1dbb93e712799475
---

# Steer Like the LLM: Activation Steering that Mimics Prompting

**Conference**: ICML 2026  
**arXiv**: [2605.03907](https://arxiv.org/abs/2605.03907)  
**Code**: <https://github.com/Nokia-Bell-Labs/steer-like-the-llm>  
**Area**: Mechanistic Interpretability / LLM Alignment / Activation Steering  
**Keywords**: activation steering, prompt steering, token-specific coefficients, ReLU probe, PSR

## TL;DR
This paper reinterprets "prompt steering" as a form of activation steering implemented by the LLM itself. It uses a **per-token ReLU probe** to distill the activation differences injected by prompts, training a PSR (Prompt Steering Replacement) module. PSR outperforms existing steering methods (CAA, ReFT-R1, Stolfo, etc.) across three benchmarks and achieves performance comparable to or even surpassing prompting in AxBench and personality steering.

## Background & Motivation

**Background**: Controlling LLM behavior generally follows two paths: (1) prompting / in-context examples; (2) activation steering—adding a fixed vector $\alpha\mathbf z_{attr}$ to the residual stream at a specific layer. The latter is a prominent direction in mechanistic interpretability due to its "lightweight nature, robustness to prompt injection, and interpretability."

**Limitations of Prior Work**: Despite a long list of methods like ActAdd, CAA, ITI, and ReFT-R1, activation steering **remains systematically weaker than prompting** (as verified multiple times by Wu et al.). The paper presents two intuitive figures: plotting the actual activation difference $\Delta_{PS}$ caused by prompt injection reveals that its **magnitude varies across tokens by several orders of magnitude**. Some tokens remain nearly unchanged, while others are heavily rewritten. All mainstream activation steering methods either use the same constant vector for all tokens or only apply it to the last token, which does not reflect the steering mechanism (prompting) implemented by the LLM itself.

**Key Challenge**: The implicit assumption of "replicating prompting behavior with a constant $\alpha\mathbf z$" is untenable. Prompting is inherently a **token-specific**, non-uniform intervention. Using a constant inevitably leads to trade-offs (either oversteering or insufficient steering).

**Goal**: (a) Explicitly formalize "prompting as a (black-box) activation steering"; (b) distill the differential activations of prompt injection using a simple, interpretable model; (c) design a learnable PSR using token-specific coefficients as a first-order necessity; (d) systematically outperform baselines while maintaining high coherence.

**Key Insight**: Since the "ground truth intervention" of prompt steering $\Delta_{PS}=\mathbf A^{prompt}-\mathbf A^{base}$ can be **calculated directly**, it can be treated as a supervised target. An activation steering module can then be trained as its imitator using MSE.

**Core Idea**: Prompt steering is formulated as $\mathbf A_{y_i'|PS}=\mathbf A_{y_i'}+\alpha\,\lambda(\mathbf A_{y_i'};\theta_{attr})\mathbf z_{attr}$, where $\lambda$ is a **ReLU probe** that decodes token-level intensity from the activations themselves. The training objective is the MSE against prompt-steered activations, resulting in the PSR.

## Method

### Overall Architecture
Training pipeline: (i) Given an attribute $attr$, collect prompt pairs $(x,x')$, where $x'$ includes a trait-eliciting instruction; (ii) sample response $y'$ from the LLM under $x'$, and use LLM judges $J_{attr}$ and $J_{coher}$ to filter successful and coherent samples; (iii) calculate $\mathbf A_{y_i'|PS}=\mathrm{LLM}(x'y')$ and $\mathbf A_{y_i'}=\mathrm{LLM}(xy')$, where the difference is the intervention $\Delta_{PS}$; (iv) train the PSR module (single-layer or all-layer versions) to approximate $\mathbf A_{y_i'|PS}$. Inference: Using only the original prompt $x$, the PSR intervention is inserted into the forward pass, with a global coefficient $\alpha$ acting as an intensity knob.

### Key Designs

1.  **Formalizing prompt steering as token-specific activation steering**:
    - **Function**: Provides a mathematical equation to decompose the actual activation effect of prompt injection into "layer-wise, token-wise differences $\Delta_{PS}$." It distinguishes between the **accumulated version** $\Delta_{PS_{acc}}$ (relative to a baseline without steering) and the **local version** $\Delta_{PS_{loc}}$ (relative to a baseline already steered in the previous layer), corresponding to single-layer and all-layer PSR, respectively.
    - **Mechanism**: Expresses $\mathbf A_{l,y_i'|PS}=\mathbf A_{l,y_i'}+\Delta_{PS}(x'y'_{\le i},xy'_{\le i})$ (Eq. 3). Based on this, two minimal assumptions are proposed: Assumption 3.1 (intervention follows a single direction $\mathbf z_{attr}$) + Assumption 3.2 (consistent intensity across tokens) $\Rightarrow$ degenerates into existing constant steering (Eq. 2). The paper uses Llama-3.2-3B sycophancy data to visualize that Assumption 3.2 contradicts reality; thus, **only Assumption 3.1 is retained, while 3.2 is relaxed** to "intensity is decodable from activations" (Assumption 3.2a).
    - **Design Motivation**: This formalization serves as the scaffolding for the methodology, directly indicating that mimicking prompts requires $\lambda$ to vary per token.

2.  **PSR Architecture: ReLU Probes for Token-level Intensity Estimation**:
    - **Function**: Dynamically determines steering intensity for each layer and token, replacing the constant $\alpha$.
    - **Mechanism**: Uses a single-layer ReLU probe $\lambda(\mathbf A_{l,y_i'};\theta_{attr,l})=\mathrm{ReLU}(\mathbf A_{l,y_i'}\cdot\mathbf w_{attr,l}+b_{attr,l})$ (Eq. 8). The intervention is defined as $\mathbf A_{l,y_i'|AS}=\mathbf A_{l,y_i'}+\alpha\lambda(\cdot)\mathbf z_{attr,l}$ (Eq. 7). Two variants: **S-PSR** intervenes at a single layer (corresponding to $\Delta_{PS_{acc}}$); **A-PSR** intervenes at all layers simultaneously (corresponding to $\Delta_{PS_{loc}}$). ReLU is used instead of sigmoid to explicitly allow "zero intervention" on certain tokens—matching the observation in Figure 2 where many tokens are barely modified by prompts.
    - **Design Motivation**: The probe reads $\mathbf A_{l,y_i'}$ because the influence of prompts in Transformers only enters the hidden state of the current token via self-attention. Thus, the decision to steer a token should inherently be recoverable from that token's activation.

3.  **Training Objectives: Dual-track MSE-on-activations and LL-on-output**:
    - **Function**: Provides two complementary goals—MSE strictly mimics the activations of prompt injection, while LL focuses on final output alignment with the attribute.
    - **Mechanism**: (a) **MSE Objective**: $\mathcal L_{MSE}=\sum_l\|\mathbf A_{l,y_i'|AS}-\mathbf A_{l,y_i'|PS}\|^2$. Training data consists of filtered successful prompt-steered triplets $(x,x',y')$. During training, $\alpha=J_{attr}\in[0,1]$ is used as a soft label; during inference, $\alpha$ is freely adjusted. (b) **LL Objective**: $-\log p_{AS}(y'|x)$, which does not require intermediate activation similarity. (c) A $\lambda$ regularization $\mathcal L_{reg}=\max(0,1-\sum_i\lambda_i)$ is added to prevent all ReLUs from becoming inactive. Negative samples ($J_{attr}<0.5$) are converted to negative $\alpha$ via a bias term $b_{m,l}=-0.5$.
    - **Design Motivation**: MSE provides the richest training signal for activation-level distillation, provided Assumptions 3.1/3.2a hold. LL does not require intermediate fidelity and performs better on tasks requiring complex format control (like IFEval), where rank-1 interventions cannot fully replicate every mechanism of a prompt.

### Loss & Training
- **Key Hyperparameters**: The global coefficient $\alpha$ is tuned during inference using binary search to reach 80% coherence. A-PSR is optimized across all layers jointly.
- **Data Filtering**: Samples with $J_{coher}<0.5$ are discarded entirely, as are positive samples with $J_{attr}<0.5$, ensuring PSR learns the behavior of "successful prompt steering."

## Key Experimental Results

### Main Results

**Persona Vectors** (5 traits × 3 LLMs): Measured by trait alignment at coherence 80 (TA@C80) and prompt-coherence-aligned (TA@Cp).

| Method (Qwen2.5-7B) | TA@C80 | TA@Cp |
|---|---|---|
| S-Const$_{DiM\|R}$ (CAA-like) | 74.8 | 34.8 |
| S-Const$_{MSE\|QR}$ | 71.6 | 48.8 |
| **S-PSR$_{MSE\|QR}$** | **83.3** | **60.9** |
| A-Const$_{MSE\|QR}$ | 96.1 | 83.6 |
| **A-PSR$_{MSE\|QR}$** | **96.8** | **83.9** |
| prompting (Upper bound ref) | – | 71.6 |

A-PSR$_{MSE}$ **exceeds prompting** in TA@Cp across all 3 LLMs, marking it as the first activation steering method to consistently outperform prompting in this context.

**IFEval (Format / Multilingual instruction following)**: Reporting IF Acc and Coherence.

| Method (Gemma-2-9b-it) | IF Acc | Coher |
|---|---|---|
| no steering | 11.4 | 96.6 |
| Stolfo et al. 2025 | 30.8 | 96.1 |
| S-PSR$_{LL}$ | 66.1 | 95.5 |
| **A-PSR$_{LL}$** | **71.9** | 82.3 |
| prompting | 85.7 | 94.8 |
| **S-PSR$_{LL}$+prompting** | **93.1** | 94.6 |

While rank-1 PSR alone does not beat prompting on IFEval, adding PSR to prompting yields a **Gain** of 7-10 points.

**AxBench (500 SAE concepts, Gemma-2)**: Harmonic mean of concept / fluency / relevance (max 2.0).

| Method | 2B-L20 | 9B-L20 |
|---|---|---|
| ReFT-r1 (rank-1) | 0.509 | 0.630 |
| Φ_SV (Wu 25b) | 0.606 | 0.892 |
| **S-PSR$_{LL}$ (rank-1)** | **0.618** | 0.667 |
| LoReFT-RePS (High rank) | 0.805 | 0.757 |
| HyperSteer | 0.742 | 1.091 |
| **A-PSR$_{MSE}$** | **0.871** | **1.120** |
| prompting | 0.731 | 1.075 |

A-PSR$_{MSE}$ achieves **SOTA** on both subsets, outperforming both prompting and LoRA.

### Ablation Study

| Configuration | Key Metric Change | Description |
|------|----------|------|
| Const vs PSR (Single layer) | TA@Cp +10\~20 | Token-specific coefficients provide the largest contribution. |
| MSE vs LL (rank-1 PSR) | MSE better on Persona, LL better on IFEval | MSE requires Assumptions 3.1/3.2a; some IFEval format instructions violate these. |
| Single layer → All layers (A-PSR) | TA@Cp +25\~40 | Multi-layer joint intervention almost perfectly mimics prompting. |
| Removing $\lambda$ Reg (AxBench) | Gain | Weaker interventions in AxBench make regularization restrictive. |

### Key Findings
- **Figure 3** shows an interesting byproduct: the relative RMSE of A-PSR$_{MSE}$ accumulated intervention compared to the real $\Delta_{PS_{acc}}$ **is lower than the RMSE between "equivalent prompts"** starting from layer 10. This suggests PSR replicates the internal mechanism of the original prompt more faithfully than another prompt expressing the same meaning.
- Single-layer Constant steering has an RMSE > 1 at the intervention layer (further than no steering), but RMSE drops back to < 1 in subsequent layers. This indicates the model **compensates for inaccuracies** to return to default behavior, explaining why constant steering seems "acceptable"—the model is essentially rectifying its errors.
- The insufficiency of rank-1 intervention on IFEval suggests that commands like "Answer in Japanese + Three-paragraph format" inherently require rank > 1, providing a clear direction for future work.

## Highlights & Insights
- **Elegant Perspective Shift**: The concept that "prompting = LLM's self-implemented activation steering" bridges mechanistic interpretability and prompt engineering. The training objective naturally becomes distillation, creating a closed logical loop.
- **ReLU Probe + Token-level Coefficients**: This design is transferable to all "sparse injection" scenarios, such as SAE feature steering, safety guardrail activation, or hard concept editing.
- **Experimental Honesty**: The paper does not hide that PSR alone fails to beat prompting in IFEval, instead providing full curves for the realistic deployment combination of PSR+prompting.
- **Interpretability Byproduct**: Visualizing $\lambda$ outputs from PSR allows for locating which tokens were most affected by the prompt, serving as an out-of-the-box tool for prompt behavior localization.

## Limitations & Future Work
- Assumption 3.1 (single direction) clearly does not hold for all attributes; the authors admit some traits are multi-directional and require expansion to low-rank ($r>1$) interventions—a natural entry point for LoReFT.
- Rank-1 interventions are insufficient for complex tasks like IFEval, and MSE struggle to train them; the paper acknowledges this as a performance ceiling.
- Training Cost: Each trait requires 1k prompt-steered triplets and LLM judging, which remains expensive for long trait lists (e.g., the 500 concepts in AxBench). Using SAE features as a starting point for $\mathbf z_{attr}$ is worth exploring.
- Adversarial Robustness: Since PSR distills prompt steering into a learnable module, could prompt injection attacks exploit weaknesses in PSR probes? This risk is not discussed.

## Related Work & Insights
- **vs ActAdd / CAA / ITI**: These use constant $\alpha\mathbf z$ (relying on Assumption 3.2), which limits them to uniform intervention across tokens. PSR relaxes this via the ReLU probe.
- **vs ReFT-R1 (Wu 2025a)**: ReFT-R1 also uses LL to train low-rank interventions but remains token-uniform. PSR$_{LL}$ systematically improves upon this via $\lambda(\cdot)$.
- **vs Stolfo et al. 2025**: Stolfo proposes per-token coefficients but aims for uniform projection of $\mathbf z$ across tokens, which is the opposite of the goal of mimicking actual prompt injection.
- **vs HyperSteer (Sun 2025)**: HyperSteer uses a hypernetwork to generate interventions. A-PSR$_{MSE}$ outperforms it by 0.03-0.13 points in AxBench and is more interpretable.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The formalization of "prompting as self-implemented activation steering" combined with the token-specific ReLU probe is a clear, theoretically-supported innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Tested across 3 benchmarks, multiple LLMs, and various baselines. The faithfulness analysis and ablations are robust.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The progressive narrative from Assumption 3.1 to 3.2a is very clear, and the roles of S-PSR vs. A-PSR are well-explained.
- **Value**: ⭐⭐⭐⭐ A must-replicate baseline for teams working on activation steering and model behavior control, with open-sourced code and training processes.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features](corrsteer_generation-time_llm_steering_via_correlated_sparse_autoencoder_feature.md)
- [\[NeurIPS 2025\] CBMAS: Cognitive Behavioral Modeling via Activation Steering](../../NeurIPS2025/interpretability/cbmas_cognitive_behavioral_modeling_via_activation_steering.md)
- [\[ICML 2026\] The Cylindrical Representation Hypothesis for Language Model Steering](the_cylindrical_representation_hypothesis_for_language_model_steering.md)
- [\[ICML 2026\] Towards Steering without Sacrifice: Principled Training of Steering Vectors for Prompt-only Interventions](towards_steering_without_sacrifice_principled_training_of_steering_vectors_for_p.md)
- [\[ICML 2026\] Do Activation Verbalization Methods Convey Privileged Information?](do_activation_verbalization_methods_convey_privileged_information.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features](corrsteer_generation-time_llm_steering_via_correlated_sparse_autoencoder_feature.md)
- [\[NeurIPS 2025\] CBMAS: Cognitive Behavioral Modeling via Activation Steering](../../NeurIPS2025/interpretability/cbmas_cognitive_behavioral_modeling_via_activation_steering.md)
- [\[ICML 2025\] To Steer or Not to Steer? Mechanistic Error Reduction with Abstention for Language Models](../../ICML2025/interpretability/to_steer_or_not_to_steer_mechanistic_error_reduction_with_abstention_for_languag.md)
- [\[ICML 2026\] Do Activation Verbalization Methods Convey Privileged Information?](do_activation_verbalization_methods_convey_privileged_information.md)
- [\[ACL 2026\] Compositional Steering of Large Language Models with Steering Tokens](../../ACL2026/interpretability/compositional_steering_of_large_language_models_with_steering_tokens.md)

</div>

<!-- RELATED:END -->
