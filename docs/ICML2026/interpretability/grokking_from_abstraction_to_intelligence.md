---
title: >-
  [Paper Note] Grokking: From Abstraction to Intelligence
description: >-
  [ICML 2026][Interpretability][grokking] This paper provides a unified explanation of the grokking phenomenon from the perspective of structural simplification (Occam's razor). It demonstrates that during training, the model undergoes four synchronized "internal condensations": degradation of causal mediation, manifold collapse to a $\mathbb{Z}_{97}$ circle,
tags:
  - ICML 2026
  - Interpretability
  - grokking
date: 2026-05-08
content_hash: fbee401643713e10
---
# Grokking: From Abstraction to Intelligence

**Conference**: ICML 2026  
**arXiv**: [2603.29262](https://arxiv.org/abs/2603.29262)  
**Code**: None  
**Area**: Interpretability / Emergence Mechanisms  
**Keywords**: grokking, Occam's razor, Singular Learning Theory, Kolmogorov complexity, modular arithmetic

## TL;DR
This paper provides a unified explanation of the grokking phenomenon from the perspective of structural simplification (Occam's razor). It demonstrates that during training, the model undergoes four synchronized "internal condensations": degradation of causal mediation, manifold collapse to a $\mathbb{Z}_{97}$ circle, spectral energy concentration toward sparse Fourier modes, and a sharp drop in BDM algorithmic complexity. Using an analytically tractable Singular Feature Machine (SFM), the authors prove this is equivalent to a phase transition driven by free energy.

## Background & Motivation
**Background**: Grokking (where test accuracy suddenly surges long after training accuracy saturates on small datasets like modular arithmetic) has become the "fruit fly" experiment for studying emergence in large models. Existing explanations generally fall into two categories: circuit-level mechanistic analysis (identifying specific attention heads) and regularization/initialization scale analysis (the relationship between weight decay, initial scale, and delayed generalization).

**Limitations of Prior Work**: These works are primarily descriptive and lack predictive power. They either rely on circuit analysis specific to a particular task, making them difficult to generalize across architectures, or observe changes in a single correlated metric without explaining "why the phase transition occurs at step $T$." The field lacks a unified answer to when and why grokking happens.

**Key Challenge**: Previous work treated grokking as a **local circuit** or **optimization dynamics** event, ignoring a global perspective: whether the overall structure of the model spontaneously evolves towards a "Minimum Description Length" solution. If such a global simplification tendency exists, grokking is merely the observable consequence of this tendency crossing an energy threshold, rather than an independent phenomenon.

**Goal**: (1) Provide a set of architecture-agnostic global metrics to track structural evolution during grokking; (2) prove on an analytically controllable proxy model that this evolution is equivalent to the minimization of free energy/Kolmogorov complexity; (3) interpret delayed generalization as an "information compression phase transition."

**Key Insight**: The authors view grokking as the model continuously "slimming down" under the constraint of fixed training accuracy—Occam's razor. In the language of SLT (Singular Learning Theory), this corresponds to the posterior mass flowing from singularities with large RLCT $\lambda$ to those with small $\lambda$. In the language of Kolmogorov complexity, it corresponds to a decrease in weight description length. From a Fourier perspective, it corresponds to the model's response collapsing from full-band noise to sparse group characters. These three frameworks are essentially different projections of the same underlying event.

**Core Idea**: Grokking $=$ a spontaneous slide along the direction of decreasing "effective parameter dimension" on the zero-training-loss manifold, where the sliding direction is determined by the SLT free energy $F_n \approx n\mathcal{L} + \lambda\ln n$.

## Method

### Overall Architecture
To answer when and why grokking occurs, the paper compares an uninterpretable real Transformer with an analytically solvable proxy model, validating both using the same complexity framework. Empirically, a 48-layer GPT-2 style Transformer is trained on modular $\{+,-,\times,\div\}$ tasks ($p=97$). Causal mediation analysis, PCA/Fourier spectral analysis of embedding manifolds, and BDM complexity estimation of quantized weights are performed at four key steps: initialization, memorization, emergence, and generalization ($0.1\text{k}/1\text{k}/10\text{k}/100\text{k}$). Theoretically, a Singular Feature Machine (SFM) is constructed to fit tasks in the Fourier domain using a complex weight matrix with an explicit $\ln n$ sparsity prior, allowing RLCT $\lambda$ and Kolmogorov complexity to be derived in closed form. Both approaches converge on the same phase transition: the three "collapses" observed empirically correspond to the theoretical drop of $\lambda$ from $p^2/2$ to $p/2$.

### Key Designs

**1. Causal Mediation Analysis (CMA) + Skip-ablation: Turning "which layer works" into a causal experiment**

Previous circuit explanations relied on attention patterns or logit lenses, which confuse correlation with causation and cannot confirm if a head is truly on the causal path. The authors use activation patching: two inputs $\mathbf{s}_1, \mathbf{s}_2$ with the same structure but different operands are constructed. The activation of a specific head from $\mathbf{s}_2$ is patched into $\mathbf{s}_1$ to get $\tilde{\mathbf{s}}$. The Causal Mediation Score $\text{CMS}(h)=[\mathcal{M}_\theta(y_2\mid\tilde{\mathbf{s}})-\mathcal{M}_\theta(y_1\mid\tilde{\mathbf{s}})]-[\mathcal{M}_\theta(y_2\mid\mathbf{s}_1)-\mathcal{M}_\theta(y_1\mid\mathbf{s}_1)]$ measures how much this patching shifts the logit toward the correct answer. Over training time, this metric shows a clear degradation trajectory: at step=1k, high CMS heads are scattered across all layers (0–47); at step=10k, the overall signal dims; by step=100k, activity condenses at the ends (layers 0–15 and 32–47), while middle layers (16–31) "extinguish." A paired skip-ablation confirms this condensation: skipping layers 16–31 results in almost no accuracy loss, indicating these layers have been bypassed by residual connections. This trajectory from flat noise to polarized condensation is the structural fingerprint of grokking.

**2. Spectral Localization + BDM Algorithmic Complexity: Two complementary proxies for "how much simpler"**

Spectral sparsity alone can be misleading due to weight decay-induced magnitude shrinkage, and PCA lacks algorithmic structure. Thus, the authors use two metrics. For the frequency domain, a 2D DFT is applied to the embedding matrix $W_E$ to get spectral density $S[k,l]$, followed by calculating the Gini coefficient $G(\mathbf{s})$ and Inverse Participation Ratio $P(\mathbf{s})=\sum_i s_i^4(\sum_i s_i^2)^{-2}$. Simultaneous increases in both indicate energy concentrating from a diffuse state into a few Fourier modes. For mechanics, all layer weights are mapped to a 4-letter alphabet via quartile quantization. Global algorithmic complexity is estimated using the Block Decomposition Method (BDM) on $4\times 4$ sub-blocks: $K_{\text{BDM}}(\theta)=\sum_l\sum_b(\text{CTM}(b)+\log_2 n_b)$. Quantization is used specifically to strip away magnitude changes caused by weight decay, leaving only true structural reorganization. All three metrics drop sharply between 1k–10k steps, supporting the conclusion that "grokking $=$ structural simplification."

**3. Singular Feature Machine (SFM) + Occam Gate: Closed-form phase transition**

Since RLCT cannot be computed for a real Transformer, the authors create a simplified proxy: inputs $(u,v)$ are encoded as Fourier tensors $\mathbf{x}_{\text{spec}}=\chi(u)\otimes\chi(v)$. The model learns a complex weight matrix $\mathbf{W}\in\mathbb{C}^{p\times p}$ with a MAP-style objective: $\min_\mathbf{W}\tfrac12\sum_i\|y_i-\langle\mathbf{W},\mathbf{x}_{\text{spec}}^{(i)}\rangle_F\|^2+\beta\ln n\cdot\|\mathbf{W}\|_0$. The dynamics involve two steps: correlating residuals with basis functions (drift), and an Occam Gate $W_{kl}^{(t+1)}=\mathbb{I}(|\tilde W_{kl}^{(t)}|>\tau)\cdot\tilde W_{kl}^{(t)}$ that zeroes out frequency components with a Signal-to-Noise Ratio (SNR) below $\tau=\sqrt{2\beta\ln n/n}$. This $\ln n$ threshold acts as the Occam's razor. In this model, everything is analytic: $\lambda_{\text{mem}}\approx p^2/2$ during memorization, and the support set collapses to the diagonal during generalization such that $\lambda_{\text{gen}}\approx p/2$. The free energy crossover occurs at $n^*\approx-\frac{\beta(p^2-p)}{\epsilon_{\text{gen}}}W_{-1}(-\frac{\epsilon_{\text{gen}}}{\beta(p^2-p)})$. The authors use "active support size $/2$" as a proxy for $\lambda$ and prove it is proportional to $K_{SFM}(\mathbf{W})\propto\lambda(\mathbf{W})\cdot(2\log_2 p+C_{\text{float}})$, coupling SLT and AIT. They clarify that SFM is a "hypothesis-generating proxy," not an equivalent proof for SGD-Transformers.

### Loss & Training
The real Transformer uses standard Cross-Entropy + AdamW (48-layer GPT-2, $d_{\text{model}}=512$, 8 heads, fp32, A100, 100k steps, average of 5 seeds). The SFM optimizes the objective $\mathcal{J}(\mathbf{W})$ using drift + Occam Gate steps, with the transition threshold controlled by $\beta\ln n$. While $n_{\text{eff}}$ is proportional to training steps, it is explicitly interpreted as a heuristic mapping.

## Key Experimental Results

### Main Results

| Training Step | CMA High-Response Distribution | Embedding Manifold | Spectral Concentration (Gini, IPR) | BDM Complexity |
| :--- | :--- | :--- | :--- | :--- |
| 0.1k | Sparse across all layers | High-entropy cluster | Extremely low | High plateau |
| 1k (Mem) | Diffuse across all layers | High-dim point cloud | Still low | High plateau |
| 10k (Emerg) | Middle layers dimming | Starting to contract | Sharp increase | Sharp decrease |
| 100k (Gen) | Only 0–15, 32–47 | 1D circle (isomorphic to $\mathbb{Z}_{97}$) | Stable at high level | Lowest plateau |

| Phenomenon | Empirical (Transformer) | Theoretical (SFM) |
| :--- | :--- | :--- |
| Effective Dimension | Layer-wise bypass; middle skip-able | $\lambda$ drops from $p^2/2$ to $p/2$ |
| Algorithmic Complexity | BDM sharp drop + block structure | $K_{SFM}\propto \lambda\cdot(2\log_2 p+C_{\text{float}})$ |
| Geometric Symmetry | 1D circle embedding | Support collapses to diagonal (add/sub) |

### Ablation Study

| Configuration | Phenomenon | Description |
| :--- | :--- | :--- |
| Skip heads 0–15 | Accuracy collapses | Early layers are essential paths |
| Skip heads 16–31 | Accuracy remains stable | Middle layers are "functionally redundant" |
| Skip heads 32–47 | Accuracy collapses | Late layers handle output formatting |
| Sparsity before quant | Apparent decrease | Mixed with weight decay magnitude shrinkage |
| BDM after quant | True decrease | Structural reorganization remains after removing magnitude effects |

### Key Findings
- The "collapses" in three different frameworks (circuit redundancy, spectral sparsity, algorithmic complexity) are nearly synchronized, strongly suggesting they are different projections of the same event.
- The bypassability of middle layers (16–31) indicates that the "emergent symbolic structure" is not uniformly distributed but condensed in a few layers at the boundaries. This aligns with theory suggesting that FMA only requires 1D group encoding and output projection.
- The phase transition threshold $n^*$ in SFM follows a $W_{-1}$ relationship with $\beta(p^2-p)/\epsilon_{\text{gen}}$, qualitatively reproducing the empirical rule that "higher weight decay leads to earlier grokking."
- For multiplication and division, the "diagonal" support in SFM does not strictly hold (requiring discrete log rearrangement), a limitation the authors explicitly note.

## Highlights & Insights
- **Triple Unification of Complexity**: Aligning SLT's $\lambda$, AIT's KC, and spectral sparsity on a single case is the paper's major "aha" moment—previously, these frameworks were used in isolation.
- **Bypassability as an Observable**: Using skip-ablation to turn layer necessity into a binary experiment provides stronger explanatory power than traditional attention patterns, and this trick can transfer to any post-training analysis (e.g., functional pruning in LLMs).
- **BDM After Quantization**: Calculating complexity after quantization avoids misinterpreting weight decay magnitude changes as structural changes, providing a clean method for grokking data.
- **SFM Intellectual Honesty**: The authors position SFM as a "hypothesis generator" rather than claiming a formal proof that grokking equals an SLT phase transition, which lends credibility to the findings.

## Limitations & Future Work
- The SFM diagonal support image only strictly holds for addition/subtraction; multiplication/division requires labels to be rearranged by discrete logs, which was only qualitatively addressed.
- The mapping between $n_{\text{eff}}(t)$ and training steps is heuristic; the predicted free energy crossover $n^*$ cannot be quantitatively verified on a real Transformer.
- Conclusions are based on a $p=97$ toy task; scaling to "knowledge emergence" in LLMs is a different order of magnitude, and the authors admit the SGD phase transition remains descriptive.
- BDM hyper-parameters (4x4 blocks, 4-letter quantization) were not exhaustively ablated.

## Related Work & Insights
- **vs Liu et al. (Omnigrok)**: While they focus on the causality between weight decay and grokking, this paper embeds that causality into the SLT free energy framework, explaining *why* weight decay works via the $\beta\ln n$ threshold term.
- **vs Nanda et al. (Circuit Mechanistic Work)**: Instead of case-by-case circuit reverse engineering, this paper uses CMA to provide a task-agnostic "which layer is active" metric, avoiding overfitting to specific heads.
- **vs Mallinar et al. (non-NN grokking)**: While they show average gradient outer products can grok, the SFM here strips away the NN structure entirely to show that "$\ln n$ sparsity prior + global observable complexity" is the minimal set of conditions, reinforcing the architecture-independent nature of grokking.
- **Inspiration**: The "triad" diagnosis—bypassability testing, post-quantization complexity, and spectral sparsity—can be transferred to any research regarding models becoming "simpler" during training, such as emergent abilities in LLMs or mode collapse in diffusion models.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First alignment of SLT/AIT/Spectral frameworks on grokking, though specific metrics are existing tools.
- **Experimental Thoroughness**: ⭐⭐⭐ Solid on the $p=97$ task, but lacks multi-task or multi-scale validation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear narrative between math and empirics; honest about SFM limitations.
- **Value**: ⭐⭐⭐⭐ Provides a general diagnostic toolkit and a calculable toy model for future "emergence/phase transition" research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Grokking in LLM Pretraining? Monitor Memorization-to-Generalization without Test](../../ICLR2026/interpretability/grokking_in_llm_pretraining_monitor_memorization-to-generalization_without_test.md)
- [\[ICML 2025\] Explaining, Fast and Slow: Abstraction and Refinement of Provable Explanations](../../ICML2025/interpretability/explaining_fast_and_slow_abstraction_and_refinement_of_provable_explanations.md)
- [\[NeurIPS 2025\] The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?](../../NeurIPS2025/interpretability/the_non-linear_representation_dilemma_is_causal_abstraction_enough_for_mechanist.md)
- [\[ICML 2026\] BLOCK-EM: Preventing Emergent Misalignment via Latent Blocking](block-em_preventing_emergent_misalignment_via_latent_blocking.md)
- [\[ICML 2026\] Courtroom Analogy: New Perspective on Uncertainty-Aware Classification](courtroom_analogy_new_perspective_on_uncertainty-aware_classification.md)

</div>

<!-- RELATED:END -->
