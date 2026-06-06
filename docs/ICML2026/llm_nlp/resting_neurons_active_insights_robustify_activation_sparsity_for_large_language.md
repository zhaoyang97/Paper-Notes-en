---
title: >-
  [Paper Note] Resting Neurons, Active Insights: Robustify Activation Sparsity for Large Language Models
description: >-
  [ICML 2026][LLM/NLP][Activation Sparsity] This paper attributes the performance degradation of sparse LLMs to "representation drift." Inspired by biological spontaneous firing, it injects an input-independent…
tags:
  - "ICML 2026"
  - "LLM/NLP"
  - "Activation Sparsity"
  - "Representation Stability"
  - "Spontaneous Neurons"
  - "Bias Absorption"
  - "Knowledge Retention"
date: 2026-05-08
content_hash: 3e8121f63dfbb422
---

# Resting Neurons, Active Insights: Robustify Activation Sparsity for Large Language Models

**Conference**: ICML 2026  
**arXiv**: [2512.12744](https://arxiv.org/abs/2512.12744)  
**Code**: https://github.com/hxu105/SPON (Available)  
**Area**: Model Compression / LLM Efficiency  
**Keywords**: Activation Sparsity, Representation Stability, Spontaneous Neurons, Bias Absorption, Knowledge Retention

## TL;DR
This paper attributes the performance degradation of sparse LLMs to "representation drift." Inspired by biological spontaneous firing, it injects an input-independent, learnable small vector (SPON) into each layer, which can be absorbed into the bias post-training. This significantly reduces the gap between sparse and dense models with near-zero inference overhead.

## Background & Motivation
**Background**: To accelerate LLM inference, activation sparsity has emerged as an elegant path. Representative methods such as TEAL / LaRoSA / R-Sparse use a magnitude threshold $\tau$ to zero out small activations, thereby skipping corresponding weight columns in MLP/Attention linear transformations. This "dynamic masking" does not modify weights or activation functions, making it naturally suitable for existing dense-weight LLMs.

**Limitations of Prior Work**: When the sparsity ratio exceeds 50%, almost all existing solutions suffer from significant increases in perplexity and performance drops in zero-shot tasks. These losses must be recovered through retraining or architectural adjustments, which contradicts the original intention of "zero-cost acceleration."

**Key Challenge**: The authors observe that as sequence length increases, the proportion of neurons that can be simultaneously activated across all tokens decays exponentially (Figure 1). That is, the always-active neurons that serve as "global anchors" in dense models are selectively turned off for different tokens after sparsification. This leads to a token-dependent drift in the distribution of hidden states, effectively losing the "priors" learned during pre-training.

**Goal**: To restore the representation stability of sparse LLMs without retraining weights, changing the architecture, or increasing inference FLOPs, thereby bringing performance back to dense levels.

**Key Insight**: The objective is to reformulate the activation sparsity problem as a "representation alignment" problem—sparsity introduces not just simple information loss, but a lack of a stable, input-independent "baseline activity" as a reference. Spontaneous activity in biological neural systems plays exactly this role, providing a static prior.

**Core Idea**: Inject a small amount of learnable, input-independent "spontaneous activation vectors" $\vec{\alpha}$ into each layer. These vectors are trained solely via KL divergence distillation from the dense model's logits; since they are input-independent, they can be directly absorbed into the bias after training, resulting in zero additional inference overhead.

## Method

### Overall Architecture
For each linear layer $Y = WX$ in the transformer, the input activation is first sparsified as $S(X)_i = \mathbf{1}\{|x_i|>\tau\}\cdot x_i$. Then, a "spontaneous neuron" term $W\vec{\alpha}$ is added in parallel, formulated as $Y = W\,S(X) + W\vec{\alpha}$. Here, $\vec{\alpha}$ is absorbed into $b' = b + W\vec{\alpha}$ after training, so the inference graph is identical to the original sparse LLM with no extra matrix multiplications. During the training phase, the entire model is frozen, and only one set of $\vec{\alpha}$ per layer is learned to align the logit distributions of the sparse and dense models on a calibration set via KL divergence.

### Key Designs

1. **Input-independent Spontaneous Activation Injection**:

    - **Function**: Provides a static, token-independent representation anchor for each sparsified linear layer to compensate for the loss of token-dependent always-active neurons.
    - **Mechanism**: Adds a term $W\vec{\alpha}$ after the original $WS(X)$, where $\vec{\alpha}\in\mathbb{R}^d$ is a learnable vector unique to each layer and independent of the input $X$. Consequently, $W\vec{\alpha}$ is a constant that can be pre-calculated and added to the bias before inference. The paper demonstrates that a **single** spontaneous neuron per layer (i.e., $\vec{\alpha}$ equivalent to an activation in a fixed direction) is sufficient to recover performance, reflecting that "a minimal prior can stabilize representations."
    - **Design Motivation**: The authors view sparsity as a disruption of pre-trained statistical priors. Spontaneous activation explicitly writes back the "global expectation" implicit in the dense model into the sparse graph without occupying new operators, fulfilling the "zero inference overhead" constraint.

2. **Distribution-Matching Lightweight Calibration**:

    - **Function**: Optimizes only $\mathcal{A} = \{\vec{\alpha}_\ell\}$ to align the output distributions of the sparse and dense models without touching any existing LLM parameters.
    - **Mechanism**: A small-scale calibration corpus $u\sim D$ (e.g., WikiText or C4) is used. Let $z(u)$ and $\tilde z(u;\mathcal{A})$ be the output logits of the dense and sparse models, respectively. The loss $\mathcal{L}(\mathcal{A}) = \mathbb{E}_u[\mathrm{KL}(\sigma(z)\|\sigma(\tilde z))]$ is minimized. Since only a few $\vec{\alpha}$ are updated, calibration costs are far lower than full fine-tuning.
    - **Design Motivation**: This combination of "output-layer distillation + bias-only updates" allows spontaneous neurons to act as global compensation for sparsity residuals. Because it only distills logits and does not force matches in intermediate layers, it is relatively robust to the calibration data distribution (PPL evaluated on WikiText remains superior even when calibrated on C4).

3. **Fisher-weighted Residual Correction Explanation**:

    - **Function**: Explains theoretically why SPON can stabilize sparse representations.
    - **Mechanism**: Taking the last layer projection as an example, define the sparsity residual as $e(X) = WX - WS(X)$. Taking the first-order condition of the KL divergence yields $\mathbb{E}_u[W^\top H(W\vec{\alpha} - e(X))] = 0$, where $H$ is the Hessian at the logits, which is exactly equivalent to the Fisher Information Matrix of the output distribution. In other words, the optimal $\vec{\alpha}$ drives $W\vec{\alpha}$ to the best approximation of $e(X)$ under the Fisher metric—compensating for sparsity deviations only in directions where the output distribution is most sensitive.
    - **Design Motivation**: This clarifies why a single static vector can save the entire sparse model—the Fisher geometry inherent in the KL loss ensures SPON prioritizes its limited capacity on directions that most strongly influence the output, thereby stabilizing critical representations with minimal parameters.

### Loss & Training
Only $\mathcal{A}$ is trained using the loss $\mathrm{KL}(\sigma(z)\|\sigma(\tilde z))$. The calibration set is very small. After training, $W\vec{\alpha}$ is absorbed into the bias, and the inference graph remains unchanged.

## Key Experimental Results

### Main Results

| Dataset | Model | Sparsity | TEAL | SPON | Remarks |
|--------|------|--------|------|------|------|
| WikiText PPL | Llama3-8B | 50% | 8.34 | 7.83 | Close to Dense (6.75) |
| WikiText PPL | Mistral-7B | 50% | 6.00 | 5.86 | Dense (5.49) |
| WikiText PPL | Qwen3-8B | 50% | 9.75 | 9.26 | Dense (8.99) |
| WikiText PPL | Llama3-8B | 60% | 11.62 | 9.63 | Gain is most significant at high sparsity |

Compared with pruning methods (Llama3-8B, 50%), SPON PPL=7.83, significantly outperforming SparseGPT (9.18), Wanda (9.66), MaskLLM (8.58), and ARMOR (10.10).

### Ablation Study

| Configuration | Key Metrics | Explanation |
|------|---------|------|
| TEAL only | Llama3-8B 50% PPL 8.34 | Magnitude threshold sparsity only |
| + Spontaneous Neuron (1 per layer) | PPL 7.83 | Adding only a single $\vec{\alpha}$ |
| Calibration on C4, Eval on WikiText | PPL 7.95 | Validates cross-corpus robustness |
| Combined with LaRoSA/WINA/R-Sparse | Llama3-8B 5-task Avg 71.96% | Higher than LaRoSA(69.82)/WINA(70.97)/R-Sparse(69.56) |

### Key Findings
- Compressing the number of "spontaneous neurons" per layer to 1 still yields the best performance, indicating that SPON primarily addresses "direction" rather than "capacity," consistent with the Fisher residual correction theory.
- The more aggressive the sparsity (60% > 50% > 25%), the larger the Gain from SPON, suggesting that spontaneous activation effectively compensates for "always-active neurons that were forcibly turned off."
- SPON is orthogonal to existing sparsity methods (LaRoSA, WINA, R-Sparse, WAS) and can be stacked for further gains. It also consistently delivers 0.75% / 0.96% improvements on Qwen3-32B and Llama3-70B, showing effectiveness beyond small models.

## Highlights & Insights
- The definition of "restoring missing always-active neurons with static biases" is very clean—it reuses the hardware path of the bias while linking sparsity to representation stability, with near-zero costs.
- The KL+Fisher derivation transforms the fact that "one vector is enough" into an interpretable conclusion rather than an engineering coincidence; this idea of "using Fisher geometry to guide minimal parameter compensation" can be transferred to other low-bit/low-rank compression tasks.
- While LLM designs typically tend to ignore bias, this paper does the opposite, showing that in heavy sparsity scenarios, "bias-like" parameters act as indispensable representational scaffolding, suggesting an overlooked design degree of freedom.

## Limitations & Future Work
- Extensive experiments were conducted mainly on 7B–8B models; while effective on 70B and 32B, the experimental granularity is smaller. Whether spontaneous vectors remain stable in long-context or chain-of-thought scenarios requires more systematic verification.
- Spontaneous vectors are learned independently per layer without explicitly modeling inter-layer interactions. Future work could explore sharing or low-rank coupling by structure (e.g., attention vs. MLP) to further reduce calibration costs.
- Training still requires logits from the dense model as a teacher, necessitating alternative signals for deployment scenarios where the dense model is completely inaccessible (e.g., only quantized weights are available).

## Related Work & Insights
- **vs TEAL/LaRoSA/R-Sparse**: These focus on "how to smarter select activations to mask," whereas Ours acknowledges the post-sparsity residual and actively compensates for it, thus being orthogonal and combinable.
- **vs SparseGPT/Wanda/MaskLLM**: Weight pruning permanently deletes parameters; SPON operates entirely in the activation space, leaving weights untouched, making it easier to roll back or stack.
- **vs Bias-only fine-tuning (e.g., BitFit)**: BitFit is task-adaptive, while SPON is sparsity-adaptive. Though similar in form, their goals differ, suggesting that "only modifying bias" remains a low-cost adjustment space worth exploring in the LLM era.

## Rating
- Novelty: ⭐⭐⭐⭐ Reformulates activation sparsity as a representation alignment problem with a Fisher residual explanation; clear logic but a relatively small single-point modification.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive multi-model/multi-baseline comparisons with pruning/SOTA sparsity methods, though lacking some ultra-long context verification.
- Writing Quality: ⭐⭐⭐⭐ The storyline (biological motivation → empirical observation → theoretical derivation → engineering implementation) is very smooth.
- Value: ⭐⭐⭐⭐ Can be stacked on existing sparsity methods at almost zero cost; very industry-friendly for deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Scaling Up Active Testing to Large Language Models](../../NeurIPS2025/llm_nlp/scaling_up_active_testing_to_large_language_models.md)
- [\[ICML 2026\] dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching](dllm-cache_accelerating_diffusion_large_language_models_with_adaptive_caching.md)
- [\[ICML 2026\] ANCHOR: Abductive Network Construction with Hierarchical Orchestration for Reliable Probability Inference in Large Language Models](anchor_abductive_network_construction_with_hierarchical_orchestration_for_reliab.md)
- [\[ICML 2026\] Rare Event Analysis of Large Language Models](rare_event_analysis_of_large_language_models.md)
- [\[ICML 2026\] SPA-Cache: Singular Proxies for Adaptive Caching in Diffusion Language Models](spa-cache_singular_proxies_for_adaptive_caching_in_diffusion_language_models.md)

</div>

<!-- RELATED:END -->
