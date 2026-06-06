---
title: >-
  [Paper Note] Steer Like the LLM: Activation Steering that Mimics Prompting
description: >-
  [ICML 2026][Interpretability][activation steering] This paper reinterprets "prompt steering" as a form of activation steering natively implemented by LLMs…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "activation steering"
  - "prompt steering"
  - "token-specific coefficient"
  - "ReLU probe"
  - "PSR"
date: 2026-05-08
content_hash: 6811d2ffdbe2f06c
---

# Steer Like the LLM: Activation Steering that Mimics Prompting

**Conference**: ICML 2026  
**arXiv**: [2605.03907](https://arxiv.org/abs/2605.03907)  
**Code**: <https://github.com/Nokia-Bell-Labs/steer-like-the-llm>  
**Area**: Mechanistic Interpretability / LLM Alignment / Activation Steering  
**Keywords**: activation steering, prompt steering, token-specific coefficient, ReLU probe, PSR

## TL;DR
This paper reinterprets "prompt steering" as a form of activation steering natively implemented by LLMs, and then distills the activation difference induced by prompt injection using a **token-wise ReLU probe**. The resulting Prompt Steering Replacement (PSR) module not only outperforms existing activation steering methods (CAA, ReFT-R1, Stolfo, etc.) on three steering benchmarks, but also matches or surpasses prompting on AxBench and persona steering tasks.

## Background & Motivation

**Background**: There are two main approaches to controlling LLM behavior: (1) prompting/in-context examples; (2) activation steering—adding a fixed vector $\alpha\mathbf z_{attr}$ to the residual stream at a certain layer. The latter is attractive for being lightweight, robust to prompt injection, and interpretable, making it a popular direction in mechanistic interpretability.

**Limitations of Prior Work**: Despite a long list of methods (ActAdd, CAA, ITI, ReFT-R1), activation steering is **still systematically weaker than prompting** (as repeatedly validated by Wu et al.). The paper provides two direct illustrations: plotting the actual activation difference $\Delta_{PS}$ caused by prompt injection reveals that its strength varies by several orders of magnitude across tokens—some tokens are barely affected, while others are heavily rewritten. All mainstream activation steering methods either use the same constant vector for all tokens or only intervene at the last token, which is fundamentally different from the steering mechanism implemented by LLMs (i.e., prompting).

**Key Challenge**: The implicit assumption of "replicating prompting behavior with a constant $\alpha\mathbf z$" does not hold—prompting is essentially a **token-specific**, non-uniform intervention, so using a constant inevitably leads to oversteering or understeering.

**Goal**: (a) Explicitly formalize "prompting as a (black-box) activation steering"; (b) Distill the differential activation induced by prompt injection using a simple, interpretable model; (c) Treat token-specific coefficients as a first-order requirement and design a learnable PSR; (d) Systematically outperform baselines while maintaining high coherence.

**Key Insight**: Since the "groundtruth intervention" of prompt steering $\Delta_{PS}=\mathbf A^{prompt}-\mathbf A^{base}$ can be **directly computed**, it can be used as a supervised target, training the activation steering module to imitate it via MSE.

**Core Idea**: Express prompt steering as $\mathbf A_{y_i'|PS}=\mathbf A_{y_i'}+\alpha\,\lambda(\mathbf A_{y_i'};\theta_{attr})\mathbf z_{attr}$, where $\lambda$ is a **ReLU probe decoding token-level strength from the activation itself**; the training objective is to minimize the MSE with the prompt-steered activation, which defines PSR.

## Method

### Overall Architecture
Training pipeline: (i) Given an attribute $attr$, collect prompt pairs $(x,x')$, where $x'$ contains an additional trait-eliciting instruction compared to $x$; (ii) Use the LLM to sample responses $y'$ on $x'$, and filter out unsuccessful or incoherent samples using an LLM judge $J_{attr}$ and a coherence judge $J_{coher}$; (iii) Compute $\mathbf A_{y_i'|PS}=\mathrm{LLM}(x'y')$ and $\mathbf A_{y_i'}=\mathrm{LLM}(xy')$, with their difference being the intervention $\Delta_{PS}$; (iv) Train the PSR module (single-layer/all-layer versions) to make its activations approximate $\mathbf A_{y_i'|PS}$. During inference: only the original prompt $x$ is used, the PSR intervention is inserted into the forward pass, and the global coefficient $\alpha$ serves as a strength knob.

### Key Designs

1. **Formalizing prompt steering as token-specific activation steering**:

    - **Function**: Provides a mathematical equation decomposing the actual activation effect of prompt injection into "layer-wise, token-wise differences $\Delta_{PS}$", distinguishing between the **accumulated version** $\Delta_{PS_{acc}}$ (relative to a baseline with no steering) and the **local version** $\Delta_{PS_{loc}}$ (relative to the previous layer already steered), corresponding to single-layer/all-layer PSR, respectively.
    - **Mechanism**: Expresses $\mathbf A_{l,y_i'|PS}=\mathbf A_{l,y_i'}+\Delta_{PS}(x'y'_{\le i},xy'_{\le i})$ (Eq. 3); introduces two minimal assumptions—Assumption 3.1 (intervention along a single direction $\mathbf z_{attr}$) + Assumption 3.2 (strength is uniform across tokens) ⇒ reduces to existing constant steering Eq. 2; the paper uses Llama-3.2-3B sycophancy data to show that Assumption 3.2 does not hold in practice, so **only 3.1 is retained, and 3.2 is relaxed** to "strength can be decoded from the activation" (Assumption 3.2a).
    - **Design Motivation**: This formalization underpins all subsequent methodology, directly indicating that "to mimic prompt, $\lambda$ must at least vary by token".

2. **PSR architecture: ReLU probe estimates token-level strength**:

    - **Function**: Dynamically determines steering strength at each layer and token, replacing the constant $\alpha$.
    - **Mechanism**: Uses a single-layer ReLU probe $\lambda(\mathbf A_{l,y_i'};\theta_{attr,l})=\mathrm{ReLU}(\mathbf A_{l,y_i'}\cdot\mathbf w_{attr,l}+b_{attr,l})$ (Eq. 8), with intervention defined as $\mathbf A_{l,y_i'|AS}=\mathbf A_{l,y_i'}+\alpha\lambda(\cdot)\mathbf z_{attr,l}$ (Eq. 7). Two variants: **S-PSR** intervenes at a single layer, corresponding to $\Delta_{PS_{acc}}$; **A-PSR** intervenes at all layers, corresponding to $\Delta_{PS_{loc}}$. ReLU (rather than sigmoid) is used to explicitly allow "zero intervention" for some tokens—matching the empirical observation in Figure 2 that "many tokens are barely affected by the prompt".
    - **Design Motivation**: The probe reads $\mathbf A_{l,y_i'}$ itself, since in transformers, the effect of the prompt can only reach the current token's hidden state via self-attention, so "whether to steer this token" can in principle be recovered from the token's own activation, aligning with the physical intuition of Assumption 3.2a.

3. **Training objectives: Dual-track MSE-on-activations and LL-on-output**:

    - **Function**: Provides two complementary objectives—MSE strictly mimics prompt-injected activations, LL only cares about final output attribute alignment.
    - **Mechanism**: (a) **MSE objective** $\mathcal L_{MSE}=\sum_l\|\mathbf A_{l,y_i'|AS}-\mathbf A_{l,y_i'|PS}\|^2$, trained on filtered successful prompt-steered triplets $(x,x',y')$, with $\alpha=J_{attr}\in[0,1]$ as a soft label during training and freely adjustable at inference; (b) **LL objective** $-\log p_{AS}(y'|x)$, which does not require intermediate activations to match; (c) Adds a $\lambda$ regularizer $\mathcal L_{reg}=\max(0,1-\sum_i\lambda_i)$ to prevent all ReLUs from dying. Negative samples ($J_{attr}<0.5$) are handled by bias $b_{m,l}=-0.5$ to yield negative $\alpha$, automatically learning "LLM's default when the attribute should not appear".
    - **Design Motivation**: MSE is activation-level distillation, providing the richest training signal but assuming Assumptions 3.1/3.2a hold at that layer; LL does not require intermediate faithfulness and is stronger on tasks like IFEval that require complex format control (since rank-1 intervention cannot fully replicate all prompt mechanisms).

### Loss & Training

- Key hyperparameter: The global coefficient $\alpha$ is tuned via binary search at inference to achieve target coherence 80; A-PSR is jointly optimized across all layers, and single-layer MSE also monitors downstream layer MSE to avoid noise propagation; training uses only positive successful samples (negatives are handled with bias shift).
- Data filtering: Samples with $J_{coher}<0.5$ are discarded, as are positive samples with $J_{attr}<0.5$—ensuring PSR learns the behavior of "successful prompt steering".

## Key Experimental Results

### Main Results

**Persona Vectors** (persona steering, 5 traits × 3 LLMs): trait alignment at coherence 80 (TA@C80) and prompt-coherence-aligned (TA@Cp), higher is better.

| Method (Qwen2.5-7B) | TA@C80 | TA@Cp |
|---|---|---|
| S-Const$_{DiM\|R}$ (CAA type) | 74.8 | 34.8 |
| S-Const$_{MSE\|QR}$ | 71.6 | 48.8 |
| **S-PSR$_{MSE\|QR}$** | **83.3** | **60.9** |
| A-Const$_{MSE\|QR}$ | 96.1 | 83.6 |
| **A-PSR$_{MSE\|QR}$** | **96.8** | **83.9** |
| prompt (upper bound reference) | – | 71.6 |

A-PSR$_{MSE}$ achieves **higher TA@Cp than prompting** on all 3 LLMs, marking the first stable outperformance by activation steering.

**IFEval (format/multilingual instruction following)**: Reports IF Acc and Coherence.

| Method (Gemma-2-9b-it) | IF Acc | Coher |
|---|---|---|
| no steering | 11.4 | 96.6 |
| Stolfo et al. 2025 | 30.8 | 96.1 |
| S-PSR$_{LL}$ | 66.1 | 95.5 |
| **A-PSR$_{LL}$** | **71.9** | 82.3 |
| prompt | 85.7 | 94.8 |
| **S-PSR$_{LL}$+prompt** | **93.1** | 94.6 |

Rank-1 PSR alone cannot beat prompting, but stacking with prompt yields an additional 7–10 points.

**AxBench (500 SAE concepts, Gemma-2)**: Harmonic mean of concept/fluency/relevance, max score 2.0.

| Method | 2B-L20 | 9B-L20 |
|---|---|---|
| ReFT-r1 (rank-1) | 0.509 | 0.630 |
| Φ_SV (Wu 25b) | 0.606 | 0.892 |
| **S-PSR$_{LL}$ (rank-1)** | **0.618** | 0.667 |
| LoReFT-RePS (high rank) | 0.805 | 0.757 |
| HyperSteer | 0.742 | 1.091 |
| **A-PSR$_{MSE}$** | **0.871** | **1.120** |
| prompt | 0.731 | 1.075 |

A-PSR$_{MSE}$ achieves **SOTA** on both subsets, surpassing both prompting and LoRA.

### Ablation Study

| Configuration | Key Metric Change | Notes |
|------|----------|------|
| Const vs PSR (single-layer) | TA@Cp +10~20 | Token-specific coefficients contribute most |
| MSE vs LL (rank-1 PSR) | MSE better on persona, LL better on IFEval | MSE assumes Assumptions 3.1/3.2a hold, IFEval format instructions may not satisfy |
| Single-layer → All-layer (A-PSR) | TA@Cp +25~40 | Multi-layer joint intervention nearly fully mimics prompt |
| Remove $\lambda$ regularizer (AxBench) | Increase | AxBench interventions are weaker, regularizer is limiting |

### Key Findings
- **Figure 3** reveals an interesting byproduct: the cumulative intervention of A-PSR$_{MSE}$ achieves a relative RMSE with the true $\Delta_{PS_{acc}}$ that is **lower than the RMSE between equivalent prompts** from layer 10 onward—indicating that PSR more faithfully replicates the internal mechanism of the original prompt than "another prompt expressing the same meaning".
- Single-layer Const has RMSE > 1 at the intervention layer (worse than no steering), but RMSE drops below 1 in subsequent layers, indicating the model **self-corrects** to default behavior, explaining why constant steering appears "okay"—the model is compensating for it.
- On IFEval, rank-1 intervention is insufficient, suggesting that prompt injection for "answer in Japanese + three-part format" type composite instructions inherently requires rank > 1, pointing to a clear direction for future work.

## Highlights & Insights
- **Elegant perspective shift**: "prompting = LLM's self-implemented activation steering" seamlessly connects mechanistic interpretability and prompt engineering, making distillation a natural training objective—logically clear and experimentally closed-loop.
- **ReLU probe + token-level coefficients** is a design transferable to all "sparse injection" scenarios: e.g., SAE feature steering, safety guardrail activations, hard concept editing.
- Honest experiments: The fact that IFEval cannot be beaten by PSR alone is not hidden; instead, PSR+prompt is presented as a realistic deployment combo with full curves.
- **Interpretability byproduct**: The $\lambda$ output of PSR can directly visualize "which tokens are most affected by the prompt", serving as an out-of-the-box tool for localizing prompt effects.

## Limitations & Future Work
- Assumption 3.1 (single direction) clearly does not hold for some attributes; the paper acknowledges that some traits are multi-directional, requiring extension to low-rank ($r>1$) interventions—precisely the entry point for LoReFT.
- Rank-1 is insufficient for IFEval, and MSE cannot be trained effectively; the paper recognizes this as a ceiling.
- Training cost: Each trait requires 1k prompt-steered triplets + LLM judge, which remains expensive for long trait lists (e.g., AxBench's 500 concepts); exploring whether SAE features can be used directly as $\mathbf z_{attr}$ is a promising direction.
- "Adversarial robustness"—since PSR distills prompt steering into a learnable module, could prompt injection attacks exploit weaknesses in the PSR probe, creating new risks? This is not discussed in the paper.

## Related Work & Insights
- **vs ActAdd / CAA / ITI**: All use constant $\alpha\mathbf z$, do not relax Assumption 3.2, and are thus inherently limited to "uniform intervention across tokens"; PSR relaxes this with a ReLU probe.
- **vs ReFT-R1 (Wu 2025a)**: ReFT-R1 also uses LL to train low-rank interventions, but still token-uniform; PSR's Const$_{LL}$ is roughly a degenerate version of ReFT-R1, while PSR$_{LL}$ systematically improves by adding $\lambda(\cdot)$.
- **vs Stolfo et al. 2025**: Stolfo proposes per-token coefficients but aims for "uniform projection of $\mathbf z$ across tokens", which is opposite to this paper's goal (mimicking actual prompt injection); the paper directly outperforms it experimentally.
- **vs HyperSteer (Sun 2025)**: HyperSteer uses a hypernetwork to generate interventions from the base prompt + steering instruction; A-PSR$_{MSE}$ outperforms it by 0.03–0.13 points on AxBench, with a more interpretable model.

## Rating
- Novelty: ⭐⭐⭐⭐ The formalization of "prompting = self-implemented activation steering" + token-specific ReLU probe is a clear, theoretically grounded innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 benchmarks × multiple LLMs × multiple baselines, comprehensive ablation, and interesting faithfulness analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ The progressive explanation of Assumptions 3.1/3.2/3.2a is very clear, and the roles of S-PSR/A-PSR are well articulated.
- Value: ⭐⭐⭐⭐ A reproducible baseline for all teams working on activation steering/model behavior control, with code and training pipeline released.

## Related Papers

- [\[ICML 2026\] CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features](corrsteer_generation-time_llm_steering_via_correlated_sparse_autoencoder_feature.md)
- [\[ICML 2026\] The Cylindrical Representation Hypothesis for Language Model Steering](the_cylindrical_representation_hypothesis_for_language_model_steering.md)
- [\[NeurIPS 2025\] CBMAS: Cognitive Behavioral Modeling via Activation Steering](../../NeurIPS2025/interpretability/cbmas_cognitive_behavioral_modeling_via_activation_steering.md)
- [\[ICML 2025\] To Steer or Not to Steer? Mechanistic Error Reduction with Abstention for Language Models](../../ICML2025/interpretability/to_steer_or_not_to_steer_mechanistic_error_reduction_with_abstention_for_languag.md)
- [\[CVPR 2026\] Language Models Can Explain Visual Features via Steering](../../CVPR2026/interpretability/language_models_can_explain_visual_features_via_steering.md)

</div>

<!-- RELATED:END -->

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
