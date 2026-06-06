---
title: >-
  [Paper Note] Grokking: From Abstraction to Intelligence
description: >-
  [ICML 2026][Interpretability][grokking] This paper provides a unified explanation of the grokking phenomenon through the lens of structural simplification (Occam’s Razor): during training…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "grokking"
  - "Occam's Razor"
  - "Singular Learning Theory"
  - "Kolmogorov Complexity"
  - "modular arithmetic"
date: 2026-05-08
content_hash: 694996371b98c4bb
---

# Grokking: From Abstraction to Intelligence

**Conference**: ICML 2026  
**arXiv**: [2603.29262](https://arxiv.org/abs/2603.29262)  
**Code**: None  
**Area**: Interpretability / Emergence Mechanisms  
**Keywords**: grokking, Occam's Razor, Singular Learning Theory, Kolmogorov Complexity, modular arithmetic

## TL;DR
This paper provides a unified explanation of the grokking phenomenon through the lens of structural simplification (Occam’s Razor): during training, the model undergoes four synchronized types of "internal condensation"—causal mediation decay, manifold collapse to a $\mathbb{Z}_{97}$ torus, spectral energy concentration toward sparse Fourier modes, and a sharp drop in BDM algorithmic complexity. Using an analytically tractable Singular Feature Machine (SFM), the authors prove this is equivalent to a phase transition driven by free energy.

## Background & Motivation
**Background**: Grokking (a sudden surge in test accuracy long after training accuracy saturates on small datasets like modular $p$ arithmetic) has become the "fruit fly experiment" for studying emergence in large models. Existing explanations generally fall into two categories: circuit-level mechanistic analysis (identifying specific attention heads' functions) and regularization/initialization scale analysis (the relationship between weight decay, init scale, and delayed generalization).

**Limitations of Prior Work**: These works are predominantly descriptive and lack predictive power. They either rely on circuit analyses specific to certain tasks, making them difficult to generalize across architectures, or they observe changes in specific correlated metrics without explaining "why the phase transition occurs at step $T$." The field lacks a unified answer to exactly when and why grokking happens.

**Key Challenge**: Prior work treated grokking as a **local circuit** or **optimization dynamics** event, overlooking a global perspective—whether the model's overall structure spontaneously evolves toward a "minimum description length" solution. If such a global simplification tendency exists, then grokking is merely the observable consequence when this tendency crosses a certain energy threshold, rather than an independent phenomenon.

**Goal**: (1) Provide a set of architecture-agnostic global metrics to track structural evolution during grokking; (2) Prove on an analytically controlled proxy model that this structural evolution is equivalent to minimizing free energy/Kolmogorov complexity; (3) Interpret delayed generalization as an "information compression phase transition."

**Key Insight**: The authors view grokking as the model continuously "slimming down"—applying Occam's Razor—under the constraint of fixed training accuracy. In the language of SLT (Singular Learning Theory), this corresponds to the posterior mass flowing from a singularity with a large RLCT $\lambda$ to one with a small $\lambda$. In terms of Kolmogorov complexity, it corresponds to a decrease in weight description length. In the Fourier perspective, it corresponds to the model collapsing from a messy full-spectrum response to sparse group characters. These three languages are different projections of the same underlying event.

**Core Idea**: Grokking $=$ a spontaneous slide along the direction of decreasing "effective parameter dimension" on the manifold where training loss is zero, with the sliding direction determined by the SLT free energy $F_n \approx n\mathcal{L} + \lambda\ln n$.

## Method

### Overall Architecture
The study is divided into two mutually reinforcing tracks:

1.  **Empirical Track** (Section 4): Training a 48-layer GPT-2 style Transformer on modular $\{+,-,\times,\div\}$ tasks ($p=97$). At four critical steps—initialization, memorization, emergence, and generalization ($0.1\text{k}/1\text{k}/10\text{k}/100\text{k}$)—three analyses are performed: Causal Mediation Analysis (CMA) to quantify individual head contributions; PCA + Fourier spectral analysis on the embedding matrix; and BDM global complexity estimation on quantized weight tensors.
2.  **Theoretical Track** (Section 5): Constructing a Singular Feature Machine (SFM) that fits the task directly in the Fourier domain using a complex weight matrix $\mathbf{W}\in\mathbb{C}^{p\times p}$, explicitly incorporating an $\ell_0$ sparse prior scaled by $\ln n$. In this model, both RLCT $\lambda$ and Kolmogorov complexity can be derived analytically.

The two tracks align in their conclusions: the three types of "collapse" observed empirically correspond to the theoretical phase transition where $\lambda$ drops from $p^2/2$ to $p/2$.

### Key Designs

1.  **Causal Mediation Analysis (CMA) + Skip-ablation revealing hierarchical bypass structures**:
    - **Function**: Measures the causal contribution of each attention head to the correct answer logit using activation patching, thereby tracking "which layers are working" during grokking.
    - **Mechanism**: Construct two inputs $\mathbf{s}_1, \mathbf{s}_2$ with the same structure but different operands. Graft the activation of a specific head from $\mathbf{s}_2$ onto $\mathbf{s}_1$ to get $\tilde{\mathbf{s}}$. The causal mediation score is defined as $\text{CMS}(h) = [\mathcal{M}_\theta(y_2\mid\tilde{\mathbf{s}}) - \mathcal{M}_\theta(y_1\mid\tilde{\mathbf{s}})] - [\mathcal{M}_\theta(y_2\mid\mathbf{s}_1) - \mathcal{M}_\theta(y_1\mid\mathbf{s}_1)]$. At step=1k, high CMS heads are scattered noisily across layers 0–47; at step=10k, the overall signal dims; at step=100k, only the ends (0–15 and 32–47) remain, while middle layers 16–31 can be entirely bypassed via residuals (skip-ablation of these layers results in almost no accuracy loss).
    - **Design Motivation**: Previous work focused on attention patterns or the logit lens, which fails to separate correlation from causation. CMA definitively identifies if a head is truly on the causal path and produces a visual degradation trajectory—from flat noise to middle-extinction to endpoint condensation—which serves as the structural fingerprint of grokking.

2.  **Joint tracking of Spectral Localization + BDM Algorithmic Complexity**:
    - **Function**: Uses two complementary complexity proxies to quantify "how much simpler the model has become"—one focusing on frequency domain sparsity, the other on the algorithmic compressibility of the weight matrix.
    - **Mechanism**: Perform a 2D DFT on the embedding matrix $W_E$ to obtain spectral density $S[k,l]$. Calculate the Gini coefficient $G(\mathbf{s})$ and the inverse participation ratio $P(\mathbf{s})=\sum_i s_i^4(\sum_i s_i^2)^{-2}$. Their simultaneous increase indicates energy concentrating from a diffuse state into a few Fourier modes. Additionally, all layer weights are mapped to a 4-letter alphabet via quartile quantization. Global algorithmic complexity is estimated using CTM lookup tables for $4\times 4$ sub-blocks and the BDM formula $K_{\text{BDM}}(\theta)=\sum_l\sum_b(\text{CTM}(b)+\log_2 n_b)$. The quantization trick distinguishes "magnitude shrinkage from weight decay" from "true structural reorganization."
    - **Design Motivation**: Looking at sparsity alone can be misleading due to weight decay; looking at PCA alone misses algorithmic structure. The synchronous sharp drop in all three metrics between 1k–10k is strong evidence that grokking $=$ structural simplification.

3.  **Singular Feature Machine (SFM) + Occam Gate analytically reproducing the phase transition**:
    - **Function**: Constructs a mathematically minimalist proxy model that still exhibits grokking, allowing for the explicit derivation of RLCT $\lambda$ and Kolmogorov complexity $K$.
    - **Mechanism**: Inputs $(u,v)$ are directly encoded as Fourier tensors $\mathbf{x}_{\text{spec}}=\chi(u)\otimes\chi(v)$. The model learns a complex weight matrix $\mathbf{W}\in\mathbb{C}^{p\times p}$ with a MAP-style objective: $\min_\mathbf{W} \tfrac12\sum_i\|y_i-\langle\mathbf{W},\mathbf{x}_{\text{spec}}^{(i)}\rangle_F\|^2 + \beta\ln n\cdot\|\mathbf{W}\|_0$. Dynamics use a two-step iteration: first computing the correlation between residuals and basis functions (drift), then using an Occam Gate $W_{kl}^{(t+1)}=\mathbb{I}(|\tilde W_{kl}^{(t)}|>\tau)\cdot\tilde W_{kl}^{(t)}$ to zero out frequency components with a signal-to-noise ratio below $\tau=\sqrt{2\beta\ln n/n}$. It is proven that during memorization $\lambda_{\text{mem}}\approx p^2/2$, and during generalization, the support collapses to the diagonal $\lambda_{\text{gen}}\approx p/2$. The free energy crossover is approximately $n^*\approx -\frac{\beta(p^2-p)}{\epsilon_{\text{gen}}}W_{-1}(-\frac{\epsilon_{\text{gen}}}{\beta(p^2-p)})$.
    - **Design Motivation**: Since RLCT cannot be directly calculated for a real Transformer, the authors use "activation support size / 2" in the SFM as a proxy for the upper bound of $\lambda$, proving it is proportional to $K_{SFM}(\mathbf{W})\propto\lambda(\mathbf{W})\cdot(2\log_2 p + C_{\text{float}})$. This couples SLT and AIT (Algorithmic Information Theory) on the same observable object. The authors explicitly state the SFM is a "hypothesis-generating proxy" rather than an equivalent proof for SGD-Transformers.

### Loss & Training
- **Ours (Transformer)**: Standard Cross-Entropy + AdamW, 48-layer GPT-2, $d_{\text{model}}=512$, 8 heads, fp32, A100, 100k steps, averaged over 5 seeds.
- **SFM**: The objective $\mathcal{J}(\mathbf{W})$ mentioned above, iterated with drift+Occam Gate; $\beta\ln n$ controls the phase transition threshold; $n_{\text{eff}}$ is proportional to training steps but interpreted as a heuristic mapping.

## Key Experimental Results

### Main Results

| Training Step | High-Response CMA Head Distribution | Embedding Manifold | Spectral Concentration (Gini, IPR) | BDM Complexity |
| :--- | :--- | :--- | :--- | :--- |
| 0.1k | Sparse across all layers | High-entropy cluster | Extremely low | High plateau |
| 1k (Memorization) | Diffuse across all layers | High-dimension point cloud | Still low | High plateau |
| 10k (Emergence) | Middle layers begin to dim | Beginning to shrink | Sharp increase | Sharp decrease |
| 100k (Generalization) | Only 0–15, 32–47 | 1D Ring (Isomorphic to $\mathbb{Z}_{97}$) | Stable at high level | Lowest plateau |

| Phenomenon | Empirical (Transformer) | Theoretical (SFM) |
| :--- | :--- | :--- |
| Effective Dimension | Hierarchical bypass, middle skippable | $\lambda$ drops from $p^2/2$ to $p/2$ |
| Algorithmic Complexity | BDM sharp drop + block structure | $K_{SFM}\propto \lambda\cdot(2\log_2 p+C_{\text{float}})$ |
| Geometric Symmetry | Embedding as 1D ring | Support collapses to diagonal (add/sub) |

### Ablation Study

| Configuration | Phenomenon | Explanation |
| :--- | :--- | :--- |
| Skip heads 0–15 | Accuracy collapses | Early layers are essential paths |
| Skip heads 16–31 | Accuracy almost unchanged | Middle layers are "functionally redundant," bypassed by residuals |
| Skip heads 32–47 | Accuracy collapses | Final layers responsible for output formatting |
| Sparsity before quantization | Apparent decrease | Confounded by weight decay magnitude shrinkage |
| BDM after quantization | True decrease | Remains after excluding magnitude effects → structural reorganization |

### Key Findings
- The "collapse" of three different labels (circuit redundancy / spectral sparsity / algorithmic complexity) occurs almost synchronously on the timeline, strongly suggesting they are different projections of the same event.
- The bypassability of middle layers (16–31) indicates that the "emergent symbolic structure" is not uniformly distributed throughout the model but condensed in a few layers at the ends; this aligns with theoretical predictions that implementing FMA only requires 1D group encoding + output projection.
- The phase transition threshold $n^*$ in the SFM relates to $\beta(p^2-p)/\epsilon_{\text{gen}}$ via $W_{-1}$, qualitatively reproducing the empirical rule that "high weight decay leads to earlier grokking."
- For multiplication and division, the "diagonal" image of the SFM does not strictly hold (requiring discrete logarithm rearrangement); the authors transparently note this limitation.

## Highlights & Insights
- **Triple Unification of Complexity**: Aligning SLT's $\lambda$, AIT's KC, and spectral sparsity on the same case is the paper's greatest "aha" moment—previously, these three frameworks were discussed in isolation.
- **Bypassability as an Observable**: Using skip-ablation to turn "layer necessity" into a yes/no experiment is more explanatory than traditional attention patterns, and this trick can be transferred to any post-training analysis (e.g., functional pruning of LLMs).
- **BDM After Quantization**: Calculating BDM after quantization avoids misinterpreting magnitude changes from weight decay as structural changes. This is a clean trick for handling grokking data that should be adopted by any work aiming to prove "models become simpler" using complexity proxies.
- **SFM Does Not Pretend to be a Transformer**: The authors explicitly position the SFM as a "hypothesis generator" rather than claiming to "prove grokking is equivalent to SLT phase transitions." This restraint makes the conclusions more credible.

## Limitations & Future Work
- The diagonal support image of the SFM strictly holds only for addition and subtraction; multiplication and division require discrete log rearrangement, for which the authors provide a qualitative explanation but no rigorous SFM solution.
- The mapping between $n_{\text{eff}}(t)$ and training steps is heuristic; the prediction of the free energy crossover $n^*$ cannot be quantitatively verified on a real Transformer.
- All conclusions are based on a toy task with $p=97$; whether this generalizes to "knowledge emergence" in LLMs is an entirely different scale—the article admits that the language of "phase transition" in SGD is purely descriptive.
- There are several hyperparameters in BDM (quantization granularity, block size) that were not ablated.

## Related Work & Insights
- **vs. Liu et al. (Omnigrok)**: They focus on the causal link between weight decay and grokking. This paper embeds that causality into the SLT free energy framework, providing a unified explanation for *why* weight decay works (the $\beta\ln n$ term controls the threshold).
- **vs. Circuit Mechanistic Work (e.g., Nanda et al.)**: They perform case-by-case circuit reverse engineering. This paper uses CMA to provide a task-agnostic "which layer is working" metric, moving beyond overfitting to specific heads.
- **vs. Mallinar et al. (non-NN grokking)**: They showed that the average gradient outer product can also grok. This paper's SFM further strips away the NN structure entirely, attributing the phenomenon to the minimal set of "$\ln n$ sparse prior + globally observable complexity," reinforcing the conclusion that grokking is architecture-agnostic.
- **Insight**: The "triad" of diagnostics—bypassability testing, post-quantization complexity, and spectral sparsity—can be transferred to any study of "models simplifying during training," such as emergent abilities in LLMs or mode collapse in diffusion models.

## Rating
- Novelty: ⭐⭐⭐⭐ First to align SLT/AIT/spectral frameworks to grokking, though individual measures are existing tools.
- Experimental Thoroughness: ⭐⭐⭐ Solid results for the $p=97$ task, but lacks validation across different primes or scales.
- Writing Quality: ⭐⭐⭐⭐ Clear narrative between math and empirical tracks with honest notation of SFM limitations.
- Value: ⭐⭐⭐⭐ Provides a set of general diagnostic tools and a hand-calculable toy model for future "emergence/phase transition" research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Grokking in LLM Pretraining? Monitor Memorization-to-Generalization without Test](../../ICLR2026/interpretability/grokking_in_llm_pretraining_monitor_memorization-to-generalization_without_test.md)
- [\[NeurIPS 2025\] The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?](../../NeurIPS2025/interpretability/the_non-linear_representation_dilemma_is_causal_abstraction_enough_for_mechanist.md)
- [\[ICML 2026\] BLOCK-EM: Preventing Emergent Misalignment via Latent Blocking](block-em_preventing_emergent_misalignment_via_latent_blocking.md)
- [\[ICML 2026\] Courtroom Analogy: New Perspective on Uncertainty-Aware Classification](courtroom_analogy_new_perspective_on_uncertainty-aware_classification.md)
- [\[ICML 2026\] Verified SHAP: Provable Bounds for Exact Shapley Values in Neural Networks](verified_shap_provable_bounds_for_exact_shapley_values_of_neural_networks.md)

</div>

<!-- RELATED:END -->
