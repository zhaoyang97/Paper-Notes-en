---
title: >-
  [Paper Note] ResNets Are Deeper Than You Think
description: >-
  [NeurIPS 2025][Residual networks] This paper proves that residual networks and feedforward networks occupy distinct function spaces (i.e., ResNets are not a simple reparameterization of feedforward networks)…
tags:
  - "NeurIPS 2025"
  - "Residual networks"
  - "function space"
  - "inductive bias"
  - "trainability"
  - "variable-depth networks"
date: 2026-05-08
content_hash: 17a0cf6c36969906
---

# ResNets Are Deeper Than You Think

**Conference**: NeurIPS 2025
**arXiv**: [2506.14386](https://arxiv.org/abs/2506.14386)  
**Code**: Not yet available  
**Area**: Deep Learning Theory / Network Architecture
**Keywords**: Residual networks, function space, inductive bias, trainability, variable-depth networks

## TL;DR

This paper proves that residual networks and feedforward networks occupy distinct function spaces (i.e., ResNets are not a simple reparameterization of feedforward networks), and demonstrates through post-training partial linearization experiments that variable-depth architectures (ResNet-like) consistently outperform fixed-depth architectures even after controlling for trainability differences, suggesting that residual connections provide inductive biases beyond optimization.

## Background & Motivation

Residual connections, introduced by He et al. (2016), have become a standard component of virtually all modern deep architectures (Transformers, large language models, etc.). Their success is commonly attributed to substantially improved trainability: residual networks train faster, more stably, and achieve higher accuracy. This raises a fundamental question — **are residual networks merely a reparameterization of feedforward networks (i.e., covering the same function space), or do they occupy a fundamentally different hypothesis space?**

Numerous studies have attempted to close the performance gap between feedforward and residual networks through carefully designed initialization schemes, Dynamic Isometry (DKS), normalization layers, and similar techniques. Yet **no prior work has fully eliminated this gap**. Even when numerical stability issues are resolved (stable forward pass, no gradient explosion/vanishing, no rank collapse), residual networks still outperform feedforward networks under otherwise equivalent conditions.

This motivates the core hypothesis: the advantage of residual networks stems not only from optimization, but also from the **inductive bias of their function space** — variable-depth networks (comprising a mixture of long and short paths) may better match natural data distributions than fixed-depth networks. However, direct training comparisons are confounded by numerical instabilities in feedforward networks, which allow any observed performance gap to be attributed to insufficient training. A carefully designed experiment is therefore needed to disentangle trainability from function space effects.

## Method

### Overall Architecture

The paper adopts a two-pronged strategy: (1) analytic arguments establishing that residual and feedforward networks occupy genuinely different function spaces; and (2) post-training partial linearization experiments that compare the generalization of variable-depth and fixed-depth architectures while controlling for trainability differences.

### Key Designs

1. **Analytic proof of function-space inequivalence**: The paper provides a precise definition — reparameterization requires the existence of a weight mapping $h(\theta)$ such that $g(x, h(\theta)) = f(x, \theta)$; equivalent reparameterization additionally requires the same width and depth. For the general case involving non-injective nonlinearities and square weight matrices, it is shown that a ResNet cannot be equivalently reparameterized as a feedforward network. Proposition 1 presents a feasible but restricted construction: a residual block $R(x) = \phi(\bar{W}x + \bar{b}) + x$ can be reparameterized as a feedforward layer, but **requires double the width and additional depth**, and is exact only in the $\epsilon \to 0$ limit. Design Motivation: to rigorously establish that "ResNet $\neq$ reparameterization of a feedforward network," providing a theoretical foundation for the inductive bias hypothesis.

2. **Post-training partial linearization experimental design**: This is the most elegant experimental contribution of the paper. Starting from an **already-trained feedforward network** (RepVGG-A2), a subset of ReLU units are linearized during post-training via regularization:

    - ReLU units are replaced by PReLU (parametric ReLU) with learnable slopes $\alpha_i$
    - A regularization term $L_{0.5} = \sum |\mathcal{1} - \alpha_i|^{0.5}$ is added to encourage $\alpha_i \to 1$ (i.e., linearization)
    - **Channel-wise linearization**: one $\alpha_i$ per channel, capable of producing variable-depth networks (channels that remain nonlinear yield ResNet-like structures)
    - **Layer-wise linearization**: one $\alpha_i$ per layer, producing only fixed-depth networks (shallower feedforward networks)

   Design Motivation: starting from the same pre-trained network (eliminating training differences), the two linearization strategies produce networks of different "shapes," enabling a fair comparison of generalization.

3. **Normalized Average Path Length (NAPL) as a depth metric**: NAPL is defined as the average number of nonlinear units encountered along paths from input to output. For layer-wise linearization, NAPL equals the remaining depth minus one; for channel-wise linearization, NAPL can take non-integer values. Performance is compared between the two methods at matched NAPL values. Design Motivation: to provide a fair depth metric that renders comparisons between variable-depth and fixed-depth networks meaningful.

### Loss & Training

Training proceeds in two stages: (1) standard training to convergence using cross-entropy loss; and (2) a post-training stage in which a regularization term $\omega \cdot L_{0.5}$ is added to the cross-entropy loss, with $\omega$ controlling the degree of linearization. Units satisfying $|\alpha_i - 1| < 0.01$ are frozen at 1. Post-training runs for 10 epochs on ImageNet and 60 epochs on CIFAR-100 (extended to rule out convergence issues).

## Key Experimental Results

### Main Results

| Dataset | NAPL Range | Channel-wise (Variable-depth) | Layer-wise (Fixed-depth) | Performance Gap |
|---------|-----------|-------------------------------|--------------------------|----------------|
| ImageNet | Below 12 | **Higher accuracy** | Lower | Gap increases as NAPL decreases |
| ImageNet | Above 12 | Comparable | Comparable | Gap not significant |
| CIFAR-100 | 3–5 | **Higher** | Lower | Significant gap; error bars do not overlap |
| CIFAR-10 | Lower threshold | **Higher** | Lower | Gap appears at lower NAPL |

### Ablation Study

| Configuration | Key Metric | Remarks |
|--------------|-----------|---------|
| Starting from ResNet56 (with residuals) | Channel-wise vs. layer-wise | Gap **substantially reduced** (nearly vanishes for NAPL > 4), confirming gap originates from path structure |
| Starting from RepVGG (no residuals) | Channel-wise vs. layer-wise | Gap **significant**; fixed-depth clearly inferior |
| Parameter count comparison | Parameter difference ~0.03% | Rules out parameter count as the cause of the performance gap |
| Path length histogram | Extracted network vs. standard ResNet | Extracted networks exhibit mixed long/short path distributions similar to ResNets |

### Key Findings

- Channel-wise linearization starting from a feedforward network **spontaneously produces a ResNet-like variable-depth structure** (mixture of long and short paths) as a result of optimization, not by design.
- At lower NAPL, variable-depth architectures consistently outperform fixed-depth ones, with the gap being more pronounced on harder datasets (ImageNet > CIFAR-100 > CIFAR-10).
- When linearization starts from a ResNet, the gap between channel-wise and layer-wise methods largely disappears, confirming that the performance disadvantage stems from the constraint of "long paths only."
- A slight performance increase around NAPL ~3 on CIFAR-100 is observed, attributed to linearization smoothing the loss landscape.
- The path length distribution approximates a binomial distribution, consistent with the theoretical predictions of Veit et al. (2016).

## Highlights & Insights

- **The reversal of the problem framing is remarkably clever**: rather than attempting to "fix feedforward networks to match ResNets" (an approach pursued for nearly a decade without complete success), the paper starts from the same pre-trained network and "sculpts" two differently shaped sub-networks for comparison, fundamentally eliminating the confound of trainability.
- **Discovery of emergent structure**: channel-wise linearization spontaneously generates variable-depth structure from a fixed-depth model, suggesting that variable depth may be an intrinsic preference of natural data.
- **Deepened understanding of ResNets**: the paper advances the discourse from "ResNets are easier to train" to "ResNets' function space better matches natural data" — a significant step forward in nearly a decade of discussion in this area.

## Limitations & Future Work

- Networks produced by post-training linearization are not guaranteed to be globally optimal sub-networks — non-convex optimization may cause the two methods to converge to different local optima.
- Validation is limited to visual tasks (ImageNet, CIFAR); evidence from other modalities (NLP, speech) is absent.
- Channel-wise linearization introduces approximately 0.03% more parameters than layer-wise linearization — while additional experiments are conducted to rule this out, the comparison is not strictly parameter-matched.
- The proof that reparameterization requires double width does not quantitatively address the impact of this width discrepancy in practical settings where ResNets and feedforward networks have comparable widths.
- No explanatory theory is provided for *why* variable depth is beneficial — the finding remains empirical, and a principled theoretical account awaits future work.

## Related Work & Insights

- **Connection to Veit et al. (2016)**: Veit interprets ResNets as an ensemble of exponentially many paths of varying lengths, but focuses solely on trainability; the present paper extends this perspective to generalization and inductive bias.
- **Contrast with DKS / Martens et al. (2021)**: Even after resolving all numerical ill-conditioning issues via DKS, a generalization gap of >1% persists on ImageNet, supporting the "beyond trainability" argument.
- **Implications**: Variable depth may be a fundamental ingredient in the success of deep learning — this finding could influence future architecture design, for instance by selectively linearizing certain attention heads in Transformers.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Proposes an entirely new perspective on ResNets (inductive bias rather than trainability alone); experimental design is highly creative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-dataset validation, effective ablations, and parameter count control; however, non-visual tasks and larger-scale experiments are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ — Thorough literature review and clear motivation; direct readability is somewhat affected by occasional LaTeX source leakage in the mathematical notation.
- **Value**: ⭐⭐⭐⭐⭐ — Represents an important step toward understanding the fundamental nature of residual connections, with potential to influence the philosophy of future architecture design.

## Supplementary Notes

- RepVGG-A2 (23 layers, ~26M parameters, 76.4% ImageNet accuracy) is chosen as the starting architecture because it lacks cross-nonlinearity residual connections yet achieves performance comparable to ResNet-50.
- The $L_{0.5}$ regularizer uses a sub-quadratic penalty that drives parameters toward 1 more aggressively than $L_1$, achieving more thorough linearization.
- The core insight of the paper is also relevant to Transformer architectures: residual connections in Transformers may similarly provide inductive biases beyond trainability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] I Am Big, You Are Little; I Am Right, You Are Wrong](../../ICCV2025/others/i_am_big_you_are_little_i_am_right_you_are_wrong.md)
- [\[ICCV 2025\] You Share Beliefs, I Adapt: Progressive Heterogeneous Collaborative Perception](../../ICCV2025/others/you_share_beliefs_i_adapt_progressive_heterogeneous_collaborative_perception.md)
- [\[NeurIPS 2025\] Out-of-distribution Generalisation is Hard: Evidence from ARC-like Tasks](out-of-distribution_generalisation_is_hard_evidence_from_arc-like_tasks.md)
- [\[NeurIPS 2025\] Training the Untrainable: Introducing Inductive Bias via Representational Alignment](training_the_untrainable_introducing_inductive_bias_via_representational_alignme.md)
- [\[ICLR 2026\] Key and Value Weights Are Probably All You Need: On the Necessity of the Query, Key, and Value Weight Triplet in Self-Attention](../../ICLR2026/others/key_and_value_weights_are_probably_all_you_need_on_the_necessity_of_the_query_ke.md)

</div>

<!-- RELATED:END -->
