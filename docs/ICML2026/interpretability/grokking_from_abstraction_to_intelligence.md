---
title: >-
  [Paper Note] Grokking: From Abstraction to Intelligence
description: >-
  [ICML 2026][Interpretability][grokking] This paper provides a unified explanation of the grokking phenomenon through the lens of structural simplification (Occam's Razor). It demonstrates that during training, models undergo four synchronized "internal consolidations": causal mediation degradation, manifold collapse to a $\mathbb{Z}_{97}$ circle, spectral energy concentration into sparse Fourier modes, and a sharp drop in BDM algorithmic complexity. Using an analytically trac…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "grokking"
  - "Occam's Razor"
  - "Singular Learning Theory"
  - "Kolmogorov complexity"
  - "modular arithmetic"
date: 2026-05-08
content_hash: fa6fd0f6a0fe146d
---

# Grokking: From Abstraction to Intelligence

**Conference**: ICML 2026  
**arXiv**: [2603.29262](https://arxiv.org/abs/2603.29262)  
**Code**: None  
**Area**: Interpretability / Emergent Mechanisms  
**Keywords**: grokking, Occam's Razor, Singular Learning Theory, Kolmogorov complexity, modular arithmetic

## TL;DR
This paper provides a unified explanation of the grokking phenomenon through the lens of structural simplification (Occam's Razor). It demonstrates that during training, models undergo four synchronized "internal consolidations": causal mediation degradation, manifold collapse to a $\mathbb{Z}_{97}$ circle, spectral energy concentration into sparse Fourier modes, and a sharp drop in BDM algorithmic complexity. Using an analytically tractable Singular Feature Machine (SFM), the authors prove this is equivalent to a phase transition driven by free energy.

## Background & Motivation
**Background**: Grokking (where test accuracy surges long after training accuracy saturates on small datasets like modular $p$ arithmetic) has become a "drosophila experiment" for studying emergence in large models. Existing explanations generally fall into two categories: circuit-level mechanistic analysis (which attention heads are doing what) and regularization/initialization scale analysis (the relationship between weight decay, initialization scale, and delayed generalization).

**Limitations of Prior Work**: These works are primarily descriptive and lack predictive power. They either rely on circuit analysis for specific tasks, making it difficult to generalize across architectures, or they observe changes in certain correlation metrics without explaining "why the phase transition occurs at step $T$." The field lacks a unified answer to exactly when and why grokking happens.

**Key Challenge**: Previous works studied grokking as a **local circuit** or **optimization dynamics** event, ignoring a global perspective—whether the model's overall structure spontaneously evolves toward a solution with "Minimum Description Length." If such a global simplification tendency exists, then grokking is merely an observable consequence of this tendency crossing an energy threshold, rather than an independent phenomenon.

**Goal**: (1) Provide a set of architecture-agnostic global metrics to track structural evolution during grokking; (2) Prove on an analytically controllable proxy model that this structural evolution is equivalent to minimizing free energy/Kolmogorov complexity; (3) Interpret delayed generalization as an "information compression phase transition."

**Key Insight**: The authors view grokking as the model continuously "slimming down" under the constraint of fixed training accuracy—an application of Occam's Razor. In the language of SLT (Singular Learning Theory), this corresponds to the posterior mass flowing from a singularity with a large RLCT $\lambda$ to one with a small $\lambda$. In the language of Kolmogorov complexity, it corresponds to a decrease in weight description length. In the Fourier perspective, it corresponds to the model collapsing from a broadband messy response to sparse group characters. These three languages are different projections of the same underlying event.

**Core Idea**: Grokking $=$ a spontaneous slide along the direction of decreasing "parameter effective dimension" on an equipotential surface where training loss remains zero, with the sliding direction determined by the SLT free energy $F_n \approx n\mathcal{L} + \lambda\ln n$.

## Method

### Overall Architecture
The paper seeks to answer exactly when and why grokking occurs by comparing an uncontrollable real Transformer with an analytically tractable proxy model under the same language of complexity. The empirical component involves training a 48-layer GPT-2 style Transformer on modular $\{+,-,\times,\div\}$ tasks ($p=97$). Causal mediation analysis, PCA+Fourier spectral analysis of embedding manifolds, and BDM complexity estimation of quantized weights are performed at four key steps: initialization, memorization, emergence, and generalization ($0.1\text{k}/1\text{k}/10\text{k}/100\text{k}$). The theoretical component constructs a Singular Feature Machine (SFM) that fits tasks in the Fourier domain using a complex weight matrix with an explicit $\ln n$ sparsity prior, allowing RLCT $\lambda$ and Kolmogorov complexity to be expressed in closed form. Both components point to the same phase transition: the three "collapses" observed empirically correspond to the theoretical reduction of $\lambda$ from $p^2/2$ to $p/2$.

### Key Designs

**1. Causal Mediation Analysis (CMA) + skip-ablation: Turning "which layer is working" into a causal experiment**

Previous circuit explanations relied on attention patterns or logit lenses, which fail to distinguish correlation from causation. The authors use activation patching: constructing two inputs $\mathbf{s}_1, \mathbf{s}_2$ with the same structure but different operands, grafting the activation of a head from $\mathbf{s}_2$ into $\mathbf{s}_1$ to get $\tilde{\mathbf{s}}$, and measuring the Causal Mediation Score $\text{CMS}(h)=[\mathcal{M}_\theta(y_2\mid\tilde{\mathbf{s}})-\mathcal{M}_\theta(y_1\mid\tilde{\mathbf{s}})]-[\mathcal{M}_\theta(y_2\mid\mathbf{s}_1)-\mathcal{M}_\theta(y_1\mid\mathbf{s}_1)]$. Over training time, this metric shows a clear degradation trajectory: at step=1k, high CMS heads are scattered across all layers (0–47); at step=10k, the overall signal dims; by step=100k, activity condenses at the two ends (0–15 and 32–47), while the middle layers (16–31) "extinguish." Skip-ablation confirms this "condensation": skipping layers 16–31 results in almost no accuracy loss, indicating they have been bypassed by residual connections. This trajectory from flat noise to dual-end condensation is the structural fingerprint of grokking.

**2. Spectral Localization + BDM Algorithmic Complexity: Two complementary proxies for "how much simpler"**

Frequency domain sparsity can be misleading due to the magnitude shrinkage from weight decay, and PCA does not capture algorithmic structure. Thus, the authors use two metrics. First, they apply a 2D DFT to the embedding matrix $W_E$ to obtain spectral density $S[k,l]$, then calculate the Gini coefficient $G(\mathbf{s})$ and the inverse participation ratio $P(\mathbf{s})=\sum_i s_i^4(\sum_i s_i^2)^{-2}$. Simultaneous increases in both indicate energy concentration into a few Fourier modes. Second, they map weights to a 4-letter alphabet via quartile quantization and estimate global algorithmic complexity using $4\times 4$ sub-blocks with the CTM lookup table and the BDM formula $K_{\text{BDM}}(\theta)=\sum_l\sum_b(\text{CTM}(b)+\log_2 n_b)$. Quantization strips away magnitude changes from weight decay, leaving only true structural reorganization. Both metrics drop sharply between 1k–10k steps, supporting the "grokking = structural simplification" conclusion.

**3. Singular Feature Machine (SFM) + Occam Gate: Closed-form phase transition**

Since RLCT is uncalculable for real Transformers, the authors build a simplified proxy that still groks: it encodes inputs $(u,v)$ directly into Fourier tensors $\mathbf{x}_{\text{spec}}=\chi(u)\otimes\chi(v)$. The model learns a complex weight matrix $\mathbf{W}\in\mathbb{C}^{p\times p}$ by minimizing $\min_\mathbf{W}\tfrac12\sum_i\|y_i-\langle\mathbf{W},\mathbf{x}_{\text{spec}}^{(i)}\rangle_F\|^2+\beta\ln n\cdot\|\mathbf{W}\|_0$. The dynamics involve two iterative steps: correlating residuals with basis functions (drift), followed by an Occam Gate $W_{kl}^{(t+1)}=\mathbb{I}(|\tilde W_{kl}^{(t)}|>\tau)\cdot\tilde W_{kl}^{(t)}$ that wipes out components with SNR below $\tau=\sqrt{2\beta\ln n/n}$. This $\ln n$ threshold acts as Occam's Razor. In this model, everything is analytical: during memorization $\lambda_{\text{mem}}\approx p^2/2$, and during generalization the support collapses to the diagonal such that $\lambda_{\text{gen}}\approx p/2$. The free energy crossover occurs at $n^*\approx-\frac{\beta(p^2-p)}{\epsilon_{\text{gen}}}W_{-1}(-\frac{\epsilon_{\text{gen}}}{\beta(p^2-p)})$. The authors use "active support size $/2$" as a proxy for the upper bound of $\lambda$ and prove it is proportional to $K_{SFM}(\mathbf{W})\propto\lambda(\mathbf{W})\cdot(2\log_2 p+C_{\text{float}})$, thereby coupling SLT and AIT.

### Loss & Training
The real Transformer uses standard Cross-Entropy + AdamW (48-layer GPT-2, $d_{\text{model}}=512$, 8 heads, fp32, A100, 100k steps, average of 5 seeds). The SFM optimizes the objective $\mathcal{J}(\mathbf{W})$ using drift + Occam Gate steps, where the phase transition threshold is controlled by $\beta\ln n$. While $n_{\text{eff}}$ is proportional to training steps, it is explicitly interpreted as a heuristic mapping.

## Key Experimental Results

### Main Results

| Training step | CMA High-response Head Dist. | Embedding Manifold | Spectral Concentration (Gini, IPR) | BDM Complexity |
|----------|--------------------|----------|---------------------|-----------|
| 0.1k | Sparse across all layers | High-entropy cluster | Very low | High plateau |
| 1k (Mem) | Diffuse across all layers | High-dim point cloud | Still low | High plateau |
| 10k (Emerg) | Middle begins to dim | Starting to contract | Sharp rise | Sharp drop |
| 100k (Gen) | Only 0–15, 32–47 | 1D circle (isom. $\mathbb{Z}_{97}$) | Stable high | Lowest plateau |

| Phenomenon | Empirical (Transformer) | Theoretical (SFM) |
|------|--------------------|------|
| Effective Dimension | Layer-wise bypass, middle skippable | $\lambda$ drops from $p^2/2$ to $p/2$ |
| Algorithmic Complexity | BDM sharp drop + block structure | $K_{SFM}\propto \lambda\cdot(2\log_2 p+C_{\text{float}})$ |
| Geometric Symmetry | Embedding 1D circle | Support collapses to diagonal (+/-) |

### Ablation Study

| Configuration | Phenomenon | Explanation |
|------|------|------|
| Skip heads 0–15 | Accuracy collapse | Early layers are essential pathways |
| Skip heads 16–31 | Acc. nearly unchanged | Middle layers are "functionally redundant," bypassed by residual |
| Skip heads 32–47 | Accuracy collapse | Final layers responsible for output formatting |
| Sparsity before quant. | Apparent decrease | Confounded by weight decay magnitude shrinkage |
| BDM after quant. | True decrease | Decrease persists after removing magnitude effects → structural reorganization |

### Key Findings
- Three different languages (circuit redundancy, spectral sparsity, algorithmic complexity) show "collapse" occurring almost simultaneously, strongly suggesting they are different projections of the same event.
- The bypassability of middle layers (16–31) indicates that the "emergent symbolic structure" is not uniformly distributed but condensed in a few layers at the ends; this aligns with theoretical predictions that implementing FMA only requires 1D group encoding + output projection.
- In SFM, the phase transition threshold $n^*$ relates to $\beta(p^2-p)/\epsilon_{\text{gen}}$ via $W_{-1}$, qualitatively replicating the empirical rule "high weight decay → earlier grokking."
- For multiplication/division, the "diagonal" image in SFM does not strictly hold (requires discrete log rearrangement), which the authors honestly note as a limitation.

## Highlights & Insights
- **Triple Unification of Complexity**: Aligning SLT's $\lambda$, AIT's KC, and spectral sparsity on the same case is the paper’s major "Aha!" moment—these three languages were previously siloed.
- **Bypassability as an Observable**: Using skip-ablation to turn the question of "is a layer necessary" into a binary experiment is more explanatory than traditional attention patterns and is transferable to any post-training analysis (e.g., functional pruning of LLMs).
- **BDM Calculation post-Quantization**: This clean trick avoids misinterpreting weight decay magnitude changes as structural changes. Any work attempting to use complexity proxies to prove a model is "getting simpler" should adopt this.
- **SFM doesn't pretend to be a Transformer**: The authors clearly position SFM as a "hypothesis generator" rather than claiming a proof of equivalence, which increases the credibility of their conclusions.

## Limitations & Future Work
- The diagonal support image of SFM only strictly holds for addition/subtraction; multiplication/division requires discrete log rearrangement, for which the authors only provide qualitative descriptions rather than a rigorous SFM solution.
- The mapping between $n_{\text{eff}}(t)$ and training steps is heuristic; the prediction of the free energy crossover $n^*$ cannot be quantitatively verified on a real Transformer.
- All conclusions are based on a $p=97$ toy task; whether this generalizes to "knowledge emergence" in LLMs is a different magnitude of problem—the paper admits the "phase transition language for SGD is descriptive."
- The quantization granularity for BDM (4x4 blocks, 4 letters) involves hyperparameters that were not fully ablated.

## Related Work & Insights
- **vs. Liu et al. (Omnigrok)**: While they focused on the causality between weight decay and grokking, this paper embeds that causality into the SLT free energy framework, providing a unified explanation for why weight decay works (the $\beta\ln n$ term controls the threshold).
- **vs. Circuit work (e.g., Nanda)**: Those works perform case-by-case circuit reverse engineering; this paper provides a CMA-based, cross-task computable metric for "which layer is working," moving beyond overfitting to specific heads.
- **vs. Mallinar et al. (non-NN grokking)**: While they showed that average gradient outer product can grok, the SFM here strips away the NN structure itself, attributing the phenomenon to the minimal set of "$\ln n$ sparsity prior + global observable complexity," reinforcing the architecture-agnostic nature of grokking.
- **Inspiration**: Causal bypassability tests + post-quantization complexity + spectral sparsity rate represent a diagnostic "triad" that could be transferred to any study of models "simplifying" during training, such as LLM emergent abilities or diffusion mode collapse.

## Rating
- Novelty: ⭐⭐⭐⭐ Aligns SLT/AIT/spectral languages on grokking for the first time, though specific metrics are existing tools.
- Experimental Thoroughness: ⭐⭐⭐ Very solid on the $p=97$ task, but lacks cross-task/cross-scale validation.
- Writing Quality: ⭐⭐⭐⭐ Clear narrative between math and empirical results; honest about SFM limitations.
- Value: ⭐⭐⭐⭐ Provides a general diagnostic toolkit and a manually calculable toy model for future emergence/phase transition research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] AI Engram: In Search of Memory Traces in Artificial Intelligence](ai_engram_in_search_of_memory_traces_in_artificial_intelligence.md)
- [\[ICLR 2026\] Grokking in LLM Pretraining? Monitor Memorization-to-Generalization without Test](../../ICLR2026/interpretability/grokking_in_llm_pretraining_monitor_memorization-to-generalization_without_test.md)
- [\[ICML 2025\] Explaining, Fast and Slow: Abstraction and Refinement of Provable Explanations](../../ICML2025/interpretability/explaining_fast_and_slow_abstraction_and_refinement_of_provable_explanations.md)
- [\[NeurIPS 2025\] The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?](../../NeurIPS2025/interpretability/the_non-linear_representation_dilemma_is_causal_abstraction_enough_for_mechanist.md)
- [\[ICML 2026\] Formal Concept Lattices are Good Semantic Scaffolds for Concept-Based Learning](formal_concept_lattices_are_good_semantic_scaffolds_for_concept-based_learning.md)

</div>

<!-- RELATED:END -->
