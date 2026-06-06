---
title: >-
  [Paper Note] Grokking: From Abstraction to Intelligence
description: >-
  [ICML 2026][Interpretability][grokking] This paper provides a unified explanation of the grokking phenomenon from the perspective of structural simplification (Occam's razor): during training…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "grokking"
  - "Occam's razor"
  - "singular learning theory"
  - "Kolmogorov complexity"
  - "modular arithmetic"
date: 2026-05-08
content_hash: 82b36689ae826467
---

# Grokking: From Abstraction to Intelligence

**Conference**: ICML 2026  
**arXiv**: [2603.29262](https://arxiv.org/abs/2603.29262)  
**Code**: None  
**Area**: Interpretability / Emergent Mechanisms  
**Keywords**: grokking, Occam's razor, singular learning theory, Kolmogorov complexity, modular arithmetic

## TL;DR
This paper provides a unified explanation of the grokking phenomenon from the perspective of structural simplification (Occam's razor): during training, the model undergoes four types of "internal condensation" that occur synchronously—causal mediation degradation, manifold collapse to the $\mathbb{Z}_{97}$ ring, spectral energy concentrating on sparse Fourier modes, and a sharp drop in BDM algorithmic complexity. Using an analytically tractable singular feature machine (SFM), it is shown that these are equivalent to a free energy-driven phase transition.

## Background & Motivation
**Background**: Grokking (where test accuracy suddenly surges long after training accuracy saturates on modular $p$ arithmetic and other small datasets) has become the "fruit fly experiment" for studying emergent phenomena in large models. Existing explanations mainly fall into two categories: circuit-level mechanism analysis (what each attention head does) and regularization/initialization scale analysis (the relationship between weight decay, init scale, and delayed generalization).

**Limitations of Prior Work**: These works are mostly descriptive and lack predictive power. They either rely on circuit analysis for specific tasks, making it hard to generalize across architectures, or merely observe changes in certain metrics without explaining "why the phase transition occurs at step $T$." There is still no unified answer to "when and why does grokking happen?"

**Key Challenge**: Previous work treats grokking as a **local circuit** or **optimization dynamics** event, overlooking a global perspective—whether the model's overall structure is spontaneously evolving toward a "minimum description length" solution. If such a global simplification tendency exists, then grokking is merely the observable consequence of crossing an energy threshold, not an independent phenomenon.

**Goal**: (1) Provide a set of architecture-agnostic global metrics to track structural evolution during grokking; (2) Prove on an analytically controllable proxy model that this structural evolution is equivalent to minimizing free energy/Kolmogorov complexity; (3) Interpret delayed generalization as an "information compression phase transition."

**Key Insight**: The authors view grokking as the model continuously "slimming down" under fixed training accuracy constraints—Occam's razor. In the language of SLT (singular learning theory), this corresponds to the posterior mass flowing from singularities with large RLCT $\lambda$ to those with small $\lambda$; in Kolmogorov complexity, it corresponds to a decrease in weight description length; in the Fourier view, it corresponds to the model collapsing from a full-spectrum response to sparse group characters. These three languages are different projections of the same phenomenon.

**Core Idea**: grokking $=$ spontaneous sliding along the direction of decreasing "effective parameter dimension" on the iso-surface of zero training loss, with the sliding direction determined by the SLT free energy $F_n \approx n\mathcal{L} + \lambda\ln n$.

## Method

### Overall Architecture
The study is divided into two mutually corroborating legs:

1. **Empirical Leg** (Section 4): Train a 48-layer GPT-2 style Transformer on modular $\{+,-,\times,\div\}$ tasks with $p=97$, and at four key steps (initialization / memorization / emergence / generalization: $0.1\text{k}/1\text{k}/10\text{k}/100\text{k}$), perform three analyses: causal mediation analysis (CMA) to quantify each head's causal contribution; PCA + Fourier spectral analysis on the embedding matrix; and global BDM complexity estimation on quantized weight tensors.
2. **Theoretical Leg** (Section 5): Construct a singular feature machine (SFM) that fits the task directly in the Fourier domain using a complex weight matrix $\mathbf{W}\in\mathbb{C}^{p\times p}$, with an explicit $\ell_0$ sparsity prior scaled by $\ln n$. RLCT $\lambda$ and Kolmogorov complexity can be analytically derived for this model.

The two legs align in their conclusions: the three types of "collapse" observed empirically correspond to a theoretical phase transition where $\lambda$ drops from $p^2/2$ to $p/2$.

### Key Designs

1. **Causal Mediation Analysis (CMA) + skip-ablation reveals hierarchical bypass structures**:

    - **Function**: Uses activation patching to measure each attention head's causal contribution to the correct answer logit, thus tracking "which layers are working" during grokking.
    - **Mechanism**: Construct two structurally identical inputs $\mathbf{s}_1, \mathbf{s}_2$ with different operands, graft the activation of a certain head from $\mathbf{s}_2$ onto $\mathbf{s}_1$ to obtain $\tilde{\mathbf{s}}$, and define the causal mediation score $\text{CMS}(h) = [\mathcal{M}_\theta(y_2\mid\tilde{\mathbf{s}}) - \mathcal{M}_\theta(y_1\mid\tilde{\mathbf{s}})] - [\mathcal{M}_\theta(y_2\mid\mathbf{s}_1) - \mathcal{M}_\theta(y_1\mid\mathbf{s}_1)]$. At step=1k, high CMS heads are scattered across layers 0–47; at step=10k, the overall response dims; at step=100k, only layers 0–15 and 32–47 remain active, with the middle 16–31 layers fully bypassed by residuals (skip-ablation of these layers barely affects accuracy).
    - **Design Motivation**: Previous work only looked at attention patterns or logit lens, unable to separate correlation from causation. CMA directly determines "whether this head is truly on the causal path" and naturally produces a visual degradation trajectory—from flat noise → middle layers extinguished → endpoints condensed—which serves as a structural fingerprint of grokking.

2. **Joint tracking of spectral localization + BDM algorithmic complexity**:

    - **Function**: Uses two complementary complexity proxies to quantify "how much simpler the model becomes"—one measures frequency domain sparsity, the other measures algorithmic compressibility of the weight matrix.
    - **Mechanism**: Perform 2D DFT on the embedding matrix $W_E$ to obtain spectral density $S[k,l]$, compute the Gini coefficient $G(\mathbf{s})$ and inverse participation ratio $P(\mathbf{s})=\sum_i s_i^4(\sum_i s_i^2)^{-2}$; simultaneous increase in both indicates energy concentrating on a few Fourier modes. Quantize all layer weights to a 4-letter alphabet via quartile mapping, then use $4\times 4$ sub-blocks and CTM lookup + BDM formula $K_{\text{BDM}}(\theta)=\sum_l\sum_b(\text{CTM}(b)+\log_2 n_b)$ to estimate global algorithmic complexity. The quantization trick separates "magnitude shrinkage from weight decay" from "true structural reorganization."
    - **Design Motivation**: Looking at sparsity alone can be confounded by weight decay; PCA alone misses algorithmic structure. The three metrics all drop sharply in the 1k–10k interval, providing strong evidence that grokking $=$ structural simplification.

3. **Singular Feature Machine (SFM) + Occam Gate analytically reproduces the phase transition**:

    - **Function**: Constructs a mathematically minimal yet grokking-capable proxy model, allowing RLCT $\lambda$ and Kolmogorov complexity $K$ to be written by hand.
    - **Mechanism**: Encode input $(u,v)$ directly as a Fourier tensor $\mathbf{x}_{\text{spec}}=\chi(u)\otimes\chi(v)$, with the model learning only a complex weight matrix $\mathbf{W}\in\mathbb{C}^{p\times p}$; the objective is MAP-style $\min_\mathbf{W} \tfrac12\sum_i\|y_i-\langle\mathbf{W},\mathbf{x}_{\text{spec}}^{(i)}\rangle_F\|^2 + \beta\ln n\cdot\|\mathbf{W}\|_0$. The dynamics use a two-step iteration: first, correlate residuals with basis functions (drift), then apply the Occam Gate $W_{kl}^{(t+1)}=\mathbb{I}(|\tilde W_{kl}^{(t)}|>\tau)\cdot\tilde W_{kl}^{(t)}$ to zero out frequency components with SNR below $\tau=\sqrt{2\beta\ln n/n}$. It can be shown: in the memorization phase $\lambda_{\text{mem}}\approx p^2/2$, in the generalization phase the support collapses to the diagonal $\lambda_{\text{gen}}\approx p/2$, and the free energy crossover point is approximately $n^*\approx -\frac{\beta(p^2-p)}{\epsilon_{\text{gen}}}W_{-1}(-\frac{\epsilon_{\text{gen}}}{\beta(p^2-p)})$.
    - **Design Motivation**: Since RLCT cannot be directly computed on real Transformers, the author uses "activation support size / 2" as an upper bound proxy for $\lambda$ in SFM, and proves that $K_{SFM}(\mathbf{W})\propto\lambda(\mathbf{W})\cdot(2\log_2 p + C_{\text{float}})$, thus coupling SLT and AIT on the same observable. The author explicitly states that SFM is a "hypothesis-generating proxy" rather than an equivalence proof for SGD-Transformer.

### Loss & Training
- Real Transformer: Standard cross-entropy + AdamW, 48-layer GPT-2, $d_{\text{model}}=512$, 8 heads, fp32, A100, 100k steps, averaged over 5 seeds.
- SFM: The above $\mathcal{J}(\mathbf{W})$, two-step drift+Occam Gate iteration, $\beta\ln n$ controls the phase transition threshold; $n_{\text{eff}}$ is proportional to training steps but interpreted as a heuristic mapping.

## Key Experimental Results

### Main Results

| Training Step | Distribution of High-Response Heads (CMA) | Embedding Manifold | Spectral Concentration (Gini, IPR) | BDM Complexity |
|--------------|------------------------------------------|--------------------|-------------------------------------|---------------|
| 0.1k | Sparse across all layers | High-entropy cluster | Very low | High plateau |
| 1k (memorization) | Diffuse across all layers | High-dimensional point cloud | Still low | High plateau |
| 10k (emergence) | Middle layers start to dim | Begins to contract | Sharp increase | Sharp decrease |
| 100k (generalization) | Only 0–15, 32–47 | 1D ring (isomorphic to $\mathbb{Z}_{97}$) | High and stable | Lowest plateau |

| Phenomenon | Empirical (Transformer) | Theoretical (SFM) |
|------------|------------------------|-------------------|
| Effective dimension | Hierarchical bypass, middle layers can be skipped | $\lambda$ drops from $p^2/2$ to $p/2$ |
| Algorithmic complexity | BDM drops sharply + block structure emerges | $K_{SFM}\propto \lambda\cdot(2\log_2 p+C_{\text{float}})$ |
| Geometric symmetry | Embedding forms a 1D ring | Support collapses to diagonal (add/sub) |

### Ablation Study

| Configuration | Phenomenon | Note |
|---------------|------------|------|
| Skip heads 0–15 | Accuracy collapses | Early layers are essential |
| Skip heads 16–31 | Accuracy almost unchanged | Middle layers are "functionally redundant," can be bypassed by residuals |
| Skip heads 32–47 | Accuracy collapses | Final layers handle output formatting |
| Sparsity before quantization | Appears to decrease | But confounded by weight decay magnitude shrinkage |
| BDM after quantization | Truly decreases | Remains decreasing after removing magnitude effects → structural reorganization |

### Key Findings
- The "collapse" in three different languages (circuit redundancy / spectral sparsity / algorithmic complexity) occurs almost synchronously on the time axis, strongly suggesting they are different projections of the same event.
- The bypassability of middle layers 16–31 indicates that the so-called "emergent symbolic structure" is not uniformly distributed throughout the model, but condensed in a few layers at both ends; this matches the theoretical prediction that "implementing FMA only requires 1D group encoding + output projection."
- In SFM, the phase transition threshold $n^*$ has a $W_{-1}$ relationship with $\beta(p^2-p)/\epsilon_{\text{gen}}$, qualitatively reproducing the empirical rule that "higher weight decay → earlier grokking."
- For multiplication and division, the "diagonal" picture in SFM does not strictly hold (requires discrete log rearrangement); the author honestly notes this limitation.

## Highlights & Insights
- **Triple unification of complexity**: Aligning SLT's $\lambda$, AIT's KC, and spectral sparsity in the same case is the paper's biggest "aha"—previously, these three languages were discussed separately.
- **Bypassability as an observable**: Using skip-ablation to directly turn "is this layer necessary" into a yes/no experiment is more explanatory than traditional attention pattern analysis, and this trick can be transferred to any post-training analysis (e.g., functional pruning in LLMs).
- **BDM after quantization**: Avoids misinterpreting weight decay-induced magnitude changes as structural changes—a clean trick for handling grokking data. Any work aiming to use complexity proxies to prove "the model is getting simpler" should adopt this.
- **SFM does not pretend to be a Transformer**: The author explicitly positions SFM as a "hypothesis generator" and does not claim "we proved grokking is equivalent to an SLT phase transition," which makes the conclusions more credible.

## Limitations & Future Work
- The diagonal support picture in SFM strictly holds only for addition and subtraction; multiplication/division require discrete log rearrangement, and the author only provides qualitative explanations without a strict SFM solution for $\times,\div$.
- The mapping between $n_{\text{eff}}(t)$ and training steps is heuristic, and the free energy crossover point $n^*$ cannot be quantitatively validated on real Transformers.
- All conclusions are based on the toy task with $p=97$; whether they generalize to "knowledge emergence" in LLMs is a much larger question—the paper itself admits that "phase transition in SGD is only descriptive."
- The quantization granularity for BDM (4×4 blocks, 4-letter alphabet) involves many hyperparameters that were not ablated.

## Related Work & Insights
- **vs Liu et al. (Omnigrok)**: They focus on the causal role of weight decay in grokking; this paper embeds that causality into the SLT free energy framework, providing a unified explanation for "why weight decay works" (the $\beta\ln n$ term controls the threshold).
- **vs Nanda et al.'s circuit mechanism work**: They conduct case-by-case circuit reverse engineering; this paper uses CMA to provide a cross-task computable "which layer is working" metric, avoiding overfitting to specific heads.
- **vs Mallinar et al. (non-NN grokking)**: They show that average gradient outer product can also grok; this paper's SFM further strips away the NN structure, attributing the phenomenon to "having a $\ln n$ sparsity prior + globally observable complexity," reinforcing the conclusion that grokking is architecture-agnostic.
- **Insights**: The combination of bypassability testing + post-quantization complexity + spectral sparsity forms a "three-piece diagnostic kit" that can be transferred to any research on "models getting simpler during training," such as emergent abilities in LLMs or mode collapse in diffusion models.

## Rating
- Novelty: ⭐⭐⭐⭐ First to align SLT/AIT/spectral languages on grokking, though the specific metrics are existing tools
- Experimental Thoroughness: ⭐⭐⭐ Very solid on the $p=97$ task, but only 1 prime and 1 architecture; lacks cross-task/cross-scale validation
- Writing Quality: ⭐⭐⭐⭐ Clear exposition of both mathematical and empirical legs, and the author is commendably restrained in noting SFM's limitations
- Value: ⭐⭐⭐⭐ Provides a general diagnostic toolkit and a hand-solvable toy model for future "emergence/phase transition" research

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Grokking in LLM Pretraining? Monitor Memorization-to-Generalization without Test](../../ICLR2026/interpretability/grokking_in_llm_pretraining_monitor_memorization-to-generalization_without_test.md)
- [\[NeurIPS 2025\] The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?](../../NeurIPS2025/interpretability/the_non-linear_representation_dilemma_is_causal_abstraction_enough_for_mechanist.md)
- [\[ICML 2026\] Steer Like the LLM: Activation Steering that Mimics Prompting](steer_like_the_llm_activation_steering_that_mimics_prompting.md)
- [\[ICML 2026\] Why Linear Interpretability Works: Invariant Subspaces as a Result of Architectural Constraints](why_linear_interpretability_works_invariant_subspaces_as_a_result_of_architectur.md)
- [\[ICML 2026\] Understanding LoRA as Knowledge Memory: An Empirical Analysis](understanding_lora_as_knowledge_memory_an_empirical_analysis.md)

</div>

<!-- RELATED:END -->
