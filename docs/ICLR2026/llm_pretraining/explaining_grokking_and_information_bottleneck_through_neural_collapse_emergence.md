---
title: >-
  [Paper Note] Explaining Grokking and Information Bottleneck through Neural Collapse Emergence
description: >-
  [ICLR2026][LLM Pretraining][Grokking] This work provides a unified explanation of two prominent late-stage training phenomena—Grokking (delayed generalization) and the Information Bottleneck compression phase—through the lens of Neural Collapse. It proves that the contraction of population within-class variance is the common key factor underlying both phenomena, and reveals that training loss convergence and the onset of Neural Collapse operate on distinct timescales governed by weight decay.
tags:
  - ICLR2026
  - LLM Pretraining
  - Grokking
  - Information Bottleneck
  - Neural Collapse
  - Training Dynamics
  - Within-Class Variance
  - Generalization Theory
  - Lyapunov Timescales
date: 2026-05-08
content_hash: b77ec291b16b6ae8
---

# Explaining Grokking and Information Bottleneck through Neural Collapse Emergence

**Conference**: ICLR2026  
**arXiv**: [2509.20829](https://arxiv.org/abs/2509.20829)  
**Code**: [keitaroskmt/collapse-dynamics](https://github.com/keitaroskmt/collapse-dynamics)  
**Area**: LLM Pretraining  
**Keywords**: Grokking, Information Bottleneck, Neural Collapse, Training Dynamics, Within-Class Variance, Generalization Theory, Lyapunov Timescales

## TL;DR
This work provides a unified explanation of two prominent late-stage training phenomena—Grokking (delayed generalization) and the Information Bottleneck compression phase—through the lens of Neural Collapse. It proves that the contraction of population within-class variance is the common key factor underlying both phenomena, and reveals that training loss convergence and the onset of Neural Collapse operate on distinct timescales governed by weight decay.

## Background & Motivation
1. **The Mystery of Grokking**: Training loss converges early, yet test accuracy undergoes a sudden leap after a prolonged delay—an overfitting solution abruptly transitions to a generalizing one, with no satisfactory mechanistic explanation.
2. **Two Phases of Information Bottleneck**: DNN training first exhibits a fitting phase (both $I(Z;X)$ and $I(Z;Y)$ increase), followed by a compression phase ($I(Z;X)$ decreases while $I(Z;Y)$ is maintained), but the trigger mechanism for the compression phase lacks rigorous theoretical justification.
3. **Shared Characteristics of Both Phenomena**: Both Grokking and IB compression occur in the late stages of training, suggesting that some common structural change takes place within the network during this period.
4. **The Potential of Neural Collapse**: Neural Collapse characterizes the geometric structure of the representation space in late training (within-class collapse, class means forming an ETF), yet its connection to the aforementioned late-stage phenomena has never been established.
5. **Limitations of Existing Theory**: Explanations of Grokking largely remain at the empirical level of parameter compression and complexity measures; mutual information in IB analysis may be infinite for continuous deterministic networks, and direct connections to network parameters are lacking.
6. **Absence of Timescale Analysis**: Even given that within-class variance contraction improves generalization, understanding its convergence rate relative to training loss convergence is necessary to explain why generalization and compression occur with a delay.

## Method
This is a theory-driven work, whose core contributions consist of three groups of theorems and their interrelations.

### Step 1: Within-Class Variance as the Unifying Key Quantity
- **Generalization Bound for Grokking (Theorem 3.2)**: For a fixed feature extractor $g$ and classifier $W$, an upper bound on classification error is derived via Chebyshev's inequality. The critical denominator term is the population within-class variance $\mathbb{E}[\|\tilde{g}(X) - \mathbb{E}[\tilde{g}(X)|Y]\|^2]$—the smaller this value, the tighter the bound and the better the generalization.
- **Redundant Information Bound for IB (Theorem 3.4)**: Under the representation $Z = g(X) + B_g \cdot \mathcal{E}$ (with added small Gaussian noise), the redundant information $I(Z;X) - I(Z;Y) = I(Z;X|Y)$ is shown to be upper-bounded by the population within-class variance. That is, contraction of within-class variance directly reduces redundant information, corresponding to the IB compression phase.

### Step 2: Population Variance Can Be Approximated by Empirical Variance
- **Variance Concentration Inequality (Theorem 4.1)**: Via uniform convergence analysis based on spectral norm, the difference between population within-class variance and empirical within-class variance on the training set is shown to be $O(1/\sqrt{n_c})$. This provides theoretical justification for subsequently using RNC1 (a rescaled NC1 metric based on the training set) as a proxy for population variance.
- **Definition of RNC1**: $\text{RNC1} = (1/B_g^2) \cdot \text{Tr}(\Sigma_W)$, which directly measures normalized within-class variance on the training set and is more directly relevant to the generalization analysis than the conventional NC1 (which additionally normalizes by between-class variance).

### Step 3: Timescale Analysis of Neural Collapse
- **Dual-Timescale Theorem (Theorem 4.3)**: Under gradient descent training with weight decay $\lambda$, training loss converges within $\tau_1 = \Omega((1/\eta) \cdot \log(1/\varepsilon_1))$ steps, while RNC1 converges within $\tau_2 = \Omega((1/(\lambda\eta)) \cdot \log(1/\varepsilon_2))$ steps. The critical distinction is that $\tau_2$ contains a factor of $1/\lambda$.
- **Physical Interpretation**: When weight decay $\lambda$ is small, $\tau_2 \gg \tau_1$—Neural Collapse (within-class variance contraction) lags far behind training loss convergence—which precisely explains why generalization and IB compression appear with a delay.
- **Role of Weight Decay**: Stronger weight decay reduces the ratio $\tau_2/\tau_1$, accelerating the onset of Neural Collapse and thereby accelerating the generalization jump in Grokking and the IB compression phase.

### Overall Logical Chain of the Theoretical Framework
Grokking/IB Compression ← Population Within-Class Variance Contraction ← Empirical Within-Class Variance Contraction (RNC1) ← Neural Collapse Dynamics ← Timescale Controlled by Weight Decay

## Key Experimental Results

### Table 1: Grokking Experiments — Training Dynamics under Different Weight Decay Values (MLP on MNIST)

| Weight Decay $\lambda$ | Steps for Training Accuracy to Reach 100% ($\tau_1$) | Steps for Test Accuracy to Begin Rising ($\tau_2$) | RNC1 Decrease Synchronized with Generalization |
|---|---|---|---|
| 0.3 | ~5,000 | ~10,000 | ✓ (nearly synchronized) |
| 0.1 | ~5,000 | ~20,000 | ✓ |
| 0.01 | ~5,000 | ~60,000 | ✓ (greater delay, more pronounced grokking) |
| 0.003 | ~5,000 | ~100,000+ | ✓ (extreme grokking) |

Key observation: RNC1 decrease is consistently synchronized with test accuracy improvement, not with training loss convergence. Smaller $\lambda$ leads to greater separation between the two timescales.

### Table 2: IB Experiments — Synchrony between Redundant Information and RNC1

| Weight Decay $\lambda$ | Onset of RNC1 Decrease | Onset of MI-Estimated Redundant Information Decrease | Onset of nHSIC Redundant Information Decrease |
|---|---|---|---|
| 0.3 | Early (~10K steps) | Synchronized | Synchronized |
| 0.1 | Medium (~20K steps) | Approximately synchronized | Synchronized |
| 0.01 | Late (~60K steps) | Slightly lagged but consistent trend | Synchronized |

Both independent redundant information estimation methods (MI and nHSIC) are qualitatively consistent with RNC1 behavior, supporting Theorem 3.4.

- Experiments are reproduced on CNN and Transformer architectures with consistent conclusions (Appendix D.1/D.2).
- Representation space visualizations (Figure 2): during the overfitting phase, training samples are separable but test within-class variance is large; after Neural Collapse, training samples collapse to points and test variance contracts simultaneously.
- NC2 (class-mean condition number) is highly synchronized with RNC1 and approaches 1, indicating that the full Neural Collapse structure emerges during Grokking.

## Highlights & Insights
- **Unified Theoretical Framework**: This is the first work to unify Grokking and Information Bottleneck—two seemingly distinct late-stage training phenomena—through Neural Collapse.
- **Rigorous Theoretical Support**: A complete theorem chain (3.2 → 3.4 → 4.1 → 4.3) progresses systematically from phenomena to mechanism to timescales with tight logical structure.
- **RNC1 Superior to NC1**: The proposed rescaled NC1 metric directly corresponds to the generalization analysis, avoiding the confounding effect of between-class variance normalization in conventional NC1.
- **Precise Timescale Characterization**: An explicit theoretical expression is derived for how weight decay quantitatively affects the degree of Grokking delay.
- **High Theory–Experiment Correspondence**: Each theorem is accompanied by a corresponding experimental validation; curve behaviors across different $\lambda$ values in Figures 3/4 align perfectly with theoretical predictions.
- **Persuasive Representation Visualization in Figure 2**: The structural difference between the overfitting phase and the Neural Collapse phase in representation space is made visually compelling.
- **Practical Guidance**: The findings directly inform practice—tracking RNC1 can indicate whether continued training is worthwhile, and increasing weight decay can accelerate the generalization jump.
- **Proposition 3.3 Completes the First IB Phase**: It is proved that the fitting phase is necessary when the network's initial state loses information, forming a complete closed loop with the analysis of the compression phase.
- **Multiple Validation Methods**: Both MI estimation and nHSIC are used as independent methods to verify IB compression behavior, strengthening the reliability of the conclusions.

## Limitations & Future Work
- The theoretical analysis relies on specific assumptions: pyramidal network architecture, smooth activation functions, and special initialization conditions; applicability to non-smooth activations such as ReLU requires further verification.
- Experiments are conducted primarily on relatively simple datasets such as MNIST with MLP/small CNN architectures; the relationship among Grokking, IB, and NC in large-scale vision models (ResNet, ViT) remains to be explored.
- Only gradient descent with weight decay is considered; theoretical analysis for Adam/AdamW is incomplete (experiments use AdamW but theorems are based on GD).
- The conditions under which Neural Collapse emerges implicitly without weight decay are not discussed, potentially limiting the generality of the theory.
- The generalization bound in Theorem 3.2 is based on Chebyshev's inequality and may be loose; tighter or data-dependent bounds represent a direction for improvement.
- The IB analysis requires adding small Gaussian noise to make mutual information finite; a theoretical gap between this proxy setting and truly deterministic networks remains.
- For multi-class classification with large $K$, the union bound in Theorem 3.2 is loose and may underestimate actual generalization capability.
- The analysis focuses exclusively on classification tasks; whether Neural Collapse can similarly explain late-stage behavior in regression, generation, and other tasks remains unclear.
- The actual impact of different optimizers (SGD vs. Adam) on the ratio $\tau_2/\tau_1$ is not quantitatively compared.

## Related Work & Insights
- **vs. Parameter Compression Explanations of Grokking (Liu et al. 2023a, Varma et al. 2023)**: Prior work explains Grokking from the perspective of parameter-space compression; this paper offers a new perspective from representation space (within-class variance contraction / Neural Collapse), and the two may be complementary.
- **vs. Kernel-to-Rich Regime Transition (Lyu et al. 2024)**: That work explains Grokking from the perspective of the optimization landscape; this paper takes a more direct approach via geometric structure.
- **vs. Diffusion Explanation of IB (Shwartz-Ziv et al. 2019)**: The IB compression phase is attributed to the diffusion component of SGD; this paper provides a more explicit geometric mechanism (within-class variance contraction).
- **vs. Neural Collapse Analysis under UFM**: UFM treats features as optimization variables, detaching the analysis from actual training dynamics; this paper directly analyzes Neural Collapse dynamics under parametric gradient descent.
- **vs. Koch and Ghosh (2025)**: That work discusses the relationship between Grokking and geometric compression but lacks the rigorous theoretical analysis and timescale characterization provided in this paper.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — First work to establish a unified theoretical connection among Grokking, IB, and Neural Collapse; extremely high originality
- Overall: The theoretical depth and elegance are at the top tier of ICLR; the work makes an important contribution to the theory of DNN training
- Experimental Thoroughness: ⭐⭐⭐⭐ — Validation across multiple architectures and $\lambda$ values is thorough, though dataset scale is limited
- Writing Quality: ⭐⭐⭐⭐⭐ — Theorems, propositions, and experiments build progressively; the logical diagram in Figure 1 is exceptionally clear
- Value: ⭐⭐⭐⭐ — Provides deep insight into late-stage DNN training behavior and directly guides practical weight decay tuning

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Flatness is Necessary, Neural Collapse is Not: Rethinking Generalization via Grokking](../../NeurIPS2025/llm_pretraining/flatness_is_necessary_neural_collapse_is_not_rethinking_generalization_via_grokk.md)
- [\[ICLR 2026\] Deconstructing Positional Information: From Attention Logits to Training Biases](deconstructing_positional_information_from_attention_logits_to_training_biases.md)
- [\[ICLR 2026\] Understanding the Emergence of Seemingly Useless Features in Next-Token Predictors](understanding_the_emergence_of_seemingly_useless_features_in_next-token_predicto.md)
- [\[NeurIPS 2025\] Neural Collapse under Gradient Flow on Shallow ReLU Networks for Orthogonally Separable Data](../../NeurIPS2025/llm_pretraining/neural_collapse_under_gradient_flow_on_shallow_relu_networks_for_orthogonally_se.md)
- [\[ICLR 2026\] Intrinsic Training Dynamics of Deep Neural Networks](intrinsic_training_dynamics_of_deep_neural_networks.md)

</div>

<!-- RELATED:END -->
