---
title: >-
  [Paper Note] Resting Neurons, Active Insights: Robustify Activation Sparsity for Large Language Models
description: >-
  [ICML 2026][Model Compression][activation sparsity] This paper attributes the performance drop in LLMs caused by activation sparsity to "representation drift." Inspired by biological spontaneous firing…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "activation sparsity"
  - "representation stability"
  - "spontaneous neurons"
  - "bias absorption"
  - "knowledge retention"
date: 2026-05-08
content_hash: 7fc81a2340971732
---

# Resting Neurons, Active Insights: Robustify Activation Sparsity for Large Language Models

**Conference**: ICML 2026  
**arXiv**: [2512.12744](https://arxiv.org/abs/2512.12744)  
**Code**: https://github.com/hxu105/SPON (available)  
**Area**: Model Compression / LLM Efficiency  
**Keywords**: activation sparsity, representation stability, spontaneous neurons, bias absorption, knowledge retention

## TL;DR
This paper attributes the performance drop in LLMs caused by activation sparsity to "representation drift." Inspired by biological spontaneous firing, it injects a small, input-independent vector (SPON) into each layer, which can be absorbed into the bias after training. This approach significantly narrows the gap between sparse and dense models with nearly zero inference overhead.

## Background & Motivation
**Background**: To accelerate LLM inference, activation sparsity has emerged as an elegant approach. Representative methods such as TEAL / LaRoSA / R-Sparse use a magnitude threshold $\tau$ to zero out small activations, thereby skipping corresponding weight columns in the linear transformations of MLP/Attention. This "dynamic masking" does not alter weights or activation functions, making it naturally compatible with existing dense LLM weights.

**Limitations of Prior Work**: When the sparsity ratio exceeds 50%, almost all existing methods exhibit a significant increase in perplexity and a drop in zero-shot task performance. Recovery requires retraining or structural modifications, which contradicts the goal of "zero-cost acceleration."

**Key Challenge**: The authors observe that as sequence length increases, the proportion of neurons activated across all tokens decays exponentially (Figure 1). In dense models, these frequently active neurons serve as "global anchors," but after sparsification, they are selectively deactivated per token, causing token-dependent shifts in hidden state distributions—effectively discarding the "priors" learned during pretraining.

**Goal**: Restore the representation stability of sparse LLMs—without retraining weights, modifying architecture, or increasing inference FLOPs—so as to recover performance to the dense model level.

**Key Insight**: Reframe the activation sparsity issue as a "representation alignment" problem: sparsity does not simply cause information loss, but rather removes a stable, input-independent "baseline activity" as a reference. Biological neural systems exhibit spontaneous activity, which provides such a static prior.

**Core Idea**: Inject a small, learnable, input-independent "spontaneous activation vector" $\vec{\alpha}$ into each layer, trained solely via KL distillation from the dense model's logits. Since it is input-independent, it can be absorbed into the bias after training, incurring zero additional inference cost.

## Method

### Overall Architecture
For each transformer linear layer $Y = WX$, input activation sparsity is first applied: $S(X)_i = \mathbf{1}\{|x_i|>\tau\}\cdot x_i$. Then, a "spontaneous neuron" term $W\vec{\alpha}$ is added in parallel, yielding $Y = W\,S(X) + W\vec{\alpha}$. After training, $\vec{\alpha}$ is absorbed into the bias as $b' = b + W\vec{\alpha}$, so the inference graph remains identical to the original sparse LLM, with no extra matrix multiplication. During training, the entire model is frozen except for a set of $\vec{\alpha}$ per layer, which are optimized to align the sparse model's logits with those of the dense model on a calibration set via KL divergence.

### Key Designs

1. **Input-Independent Spontaneous Activation Injection**:

    - **Function**: Provides each sparsified linear layer with a static, token-independent representation anchor to compensate for the loss of frequently active neurons.
    - **Mechanism**: After the original $WS(X)$, add $W\vec{\alpha}$, where $\vec{\alpha}\in\mathbb{R}^d$ is a layer-specific, learnable vector independent of input $X$. Thus, $W\vec{\alpha}$ is a constant and can be precomputed and added to the bias before inference. The paper shows that a single spontaneous neuron per layer (i.e., $\vec{\alpha}$ as a fixed-direction activation) suffices to recover performance, demonstrating that "a minimal prior can stabilize representations."
    - **Design Motivation**: The authors view sparsity as disrupting pretrained statistical priors; spontaneous activation explicitly restores the "global expectation" implicit in dense models into the sparse graph, without introducing new operators, thus meeting the "zero inference overhead" constraint.

2. **Distribution-Matching Lightweight Calibration**:

    - **Function**: Aligns the output distributions of sparse and dense models by optimizing only $\mathcal{A} = \{\vec{\alpha}_\ell\}$, without modifying any existing LLM parameters.
    - **Mechanism**: Using a small calibration corpus $u\sim D$ (WikiText or C4), denote the dense and sparse model logits as $z(u)$ and $\tilde z(u;\mathcal{A})$, respectively. Minimize $\mathcal{L}(\mathcal{A}) = \mathbb{E}_u[\mathrm{KL}(\sigma(z)\|\sigma(\tilde z))]$. Since only a small number of $\vec{\alpha}$ are updated, calibration is much cheaper than full fine-tuning.
    - **Design Motivation**: This combination of "output layer distillation + bias-only update" allows spontaneous neurons to globally compensate for sparse residuals. Since only logits are distilled (not intermediate layers), the method is robust to calibration data distribution (e.g., calibrating on C4 and evaluating on WikiText still yields better PPL than baselines).

3. **Fisher-Weighted Residual Correction Interpretation**:

    - **Function**: Provides a theoretical explanation for why SPON can stabilize sparse representations.
    - **Mechanism**: Taking the final projection layer as an example, define the sparse residual $e(X) = WX - WS(X)$. A first-order condition on the KL loss yields $\mathbb{E}_u[W^\top H(W\vec{\alpha} - e(X))] = 0$, where $H$ is the Hessian at the logits, equivalent to the Fisher information matrix of the output distribution. Thus, the optimal $\vec{\alpha}$ pushes $W\vec{\alpha}$ to be the best Fisher-metric approximation of $e(X)$—compensating for sparsity-induced bias only in the "most output-sensitive" directions.
    - **Design Motivation**: This clarifies "why a single static vector suffices"—the Fisher geometry inherent in the KL loss ensures SPON allocates its limited capacity to the most impactful directions, stabilizing key representations with minimal parameters.

### Loss & Training
Only $\mathcal{A}$ is trained, with loss $\mathrm{KL}(\sigma(z)\|\sigma(\tilde z))$. The calibration set is small; after training, $W\vec{\alpha}$ is folded into the bias, and the inference graph remains unchanged.

## Key Experimental Results

### Main Results

| Dataset | Model | Sparsity | TEAL | SPON | Note |
|---------|-------|----------|------|------|------|
| WikiText PPL | Llama3-8B | 50% | 8.34 | 7.83 | Close to dense 6.75 |
| WikiText PPL | Mistral-7B | 50% | 6.00 | 5.86 | Dense 5.49 |
| WikiText PPL | Qwen3-8B | 50% | 9.75 | 9.26 | Dense 8.99 |
| WikiText PPL | Llama3-8B | 60% | 11.62 | 9.63 | Largest gain at high sparsity |

Compared to pruning methods (Llama3-8B, 50%): SPON PPL=7.83, outperforming SparseGPT (9.18), Wanda (9.66), MaskLLM (8.58), ARMOR (10.10).

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|-----------|-------------|
| TEAL only | Llama3-8B 50% PPL 8.34 | Magnitude threshold sparsity only |
| + Spontaneous Neuron (1 per layer) | PPL 7.83 | Adding a single $\vec{\alpha}$ |
| Calibrate on C4, evaluate on WikiText | PPL 7.95 | Tests cross-corpus robustness |
| Combined with LaRoSA/WINA/R-Sparse | Llama3-8B avg. score 71.96% | Higher than LaRoSA(69.82)/WINA(70.97)/R-Sparse(69.56) |

### Key Findings
- Reducing the number of "spontaneous neurons" per layer to 1 still yields optimal performance, indicating SPON mainly addresses a "directional" rather than "capacity" issue, consistent with the Fisher residual correction theory.
- The more aggressive the sparsity (60% > 50% > 25%), the greater the SPON gain, suggesting spontaneous activation compensates for "frequently active neurons forcibly deactivated."
- SPON is orthogonal and additive to existing sparsity methods (LaRoSA, WINA, R-Sparse, WAS); it also brings stable improvements of 0.75% / 0.96% on Qwen3-32B and Llama3-70B, indicating effectiveness beyond small models.

## Highlights & Insights
- The approach of "restoring missing frequently active neurons via static bias" is exceptionally clean—reusing the hardware path of bias and linking sparsity to representation stability, with almost zero method cost.
- The KL+Fisher derivation makes "why a single vector suffices" an interpretable conclusion rather than an engineering coincidence; this "Fisher geometry-guided minimal parameter compensation" can be transferred to other low-bit/low-rank compression settings.
- While LLM design often overlooks bias, this work does the opposite, showing that in highly sparse scenarios, "bias-like" parameters serve as indispensable representational scaffolds, revealing an overlooked design degree of freedom.

## Limitations & Future Work
- Extensive experiments are mainly on 7B–8B models; although effective on 70B and 32B, the granularity is limited. More systematic validation is needed for long-context and reasoning scenarios.
- The spontaneous vector is learned independently per layer, without explicit modeling of inter-layer interactions. Future work could explore sharing or low-rank coupling by structure (e.g., attention vs MLP) to further reduce calibration cost.
- Training still requires dense model logits as teacher; for deployment scenarios without access to dense models (e.g., only quantized weights), alternative signals are needed.

## Related Work & Insights
- **vs TEAL/LaRoSA/R-Sparse**: These focus on "smarter selection of activations to mask," while this work acknowledges and actively compensates for sparse residuals, making it orthogonal and combinable with them.
- **vs SparseGPT/Wanda/MaskLLM**: Weight pruning permanently removes parameters; SPON operates entirely in activation space, leaving weights untouched, making it easier to roll back and combine.
- **vs Bias-only fine-tuning (e.g., BitFit)**: BitFit is task-adaptive, SPON is sparsity-adaptive; though similar in form, their goals differ, suggesting that "bias-only" remains a promising low-cost adjustment space in the LLM era.

## Rating
- Novelty: ⭐⭐⭐⭐ Reframes activation sparsity as a representation alignment problem and explains via Fisher residuals; clear logic but a small single-point change
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple models and baselines + comprehensive comparison with pruning/SOTA sparsity methods; lacks some ultra-long context validation
- Writing Quality: ⭐⭐⭐⭐ Narrative (biological motivation → empirical observation → theoretical derivation → engineering implementation) is very smooth
- Value: ⭐⭐⭐⭐ Can be stacked on existing sparsity methods at almost zero cost, friendly for industrial deployment

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] DuoGPT: Training-free Dual Sparsity through Activation-aware Pruning in LLMs](../../NeurIPS2025/model_compression/duogpt_training-free_dual_sparsity_through_activation-aware_pruning_in_llms.md)
- [\[ICCV 2025\] MixA-Q: Revisiting Activation Sparsity for Vision Transformers from a Mixed-Precision Quantization Perspective](../../ICCV2025/model_compression/mixa-q_revisiting_activation_sparsity_for_vision_transformers_from_a_mixed-preci.md)
- [\[ICLR 2026\] Knowledge Fusion of Large Language Models Via Modular Skillpacks](../../ICLR2026/model_compression/knowledge_fusion_of_large_language_models_via_modular_skillpacks.md)
- [\[ICLR 2026\] Distillation of Large Language Models via Concrete Score Matching](../../ICLR2026/model_compression/distillation_of_large_language_models_via_concrete_score_matching.md)
- [\[AAAI 2026\] Failures to Surface Harmful Contents in Video Large Language Models](../../AAAI2026/model_compression/failures_to_surface_harmful_contents_in_video_large_language_models.md)

</div>

<!-- RELATED:END -->
