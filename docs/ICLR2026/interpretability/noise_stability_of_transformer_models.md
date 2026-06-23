---
title: >-
  [Paper Note] Noise Stability of Transformer Models
description: >-
  [ICLR 2026][Interpretability][noise stability] Proposes noise stability as a superior metric over average sensitivity for measuring simplicity bias in Transformers, and designs a regularization method based on this that accelerates training by approximately 35% on synthetic tasks and 75% on language modeling.
tags:
  - ICLR 2026
  - Interpretability
  - noise stability
  - simplicity bias
  - Transformer
  - grokking
  - Fourier analysis
  - regularization
date: 2026-05-08
content_hash: 5abe96ba2de208ed
---
# Noise Stability of Transformer Models

**Conference**: ICLR 2026  
**arXiv**: [2602.08287](https://arxiv.org/abs/2602.08287)  
**Code**: Not public  
**Area**: Interpretability  
**Keywords**: noise stability, simplicity bias, Transformer, grokking, Fourier analysis, regularization, Boolean function analysis

## TL;DR

Proposes noise stability as a superior metric over average sensitivity for measuring simplicity bias in Transformers, and designs a regularization method based on this that accelerates training by approximately 35% on synthetic tasks and 75% on language modeling.

## Background & Motivation

Simplicity bias in deep learning is a core concept for understanding model generalization, interpretability, and robustness. Neural networks tend to converge to the simplest functions that can explain the training data. Traditional measures of this "simplicity" originate from **average sensitivity** in Boolean function analysis, representing the expected change in model output given a perturbation of a single token.

Prior work has shown that functions learned by Transformers exhibit lower sensitivity than those of LSTMs (Bhattamishra et al., 2022), and that Transformers struggle to learn high-sensitivity functions such as Parity (Hahn 2020). Vasudeva et al. (2024) linked average sensitivity to the grokking phenomenon.

However, the authors point out two key defects in average sensitivity:

**Theoretical**: Definitions on Boolean domains are difficult to naturally generalize to real-valued domains, and extension methods based on hypergrids are clunky and impractical for sampling.

**Limitations of Prior Work**: It fails to explain the "junta-like" input dependence observed in modern LLMs such as GPT-2, Gemma, and RoBERTa—where outputs depend only on a very small subset of input tokens (in experiments, only 5-10 tokens out of 256 have significant influence), whereas the upper bound predicted by Friedgut's theorem is as high as 1024 tokens, showing a massive gap.

## Method

### Overall Architecture

This paper addresses a measurement problem: which metric characterizes the "simplicity bias" of Transformers in a way that is both rigorous and explanatory. Traditional average sensitivity, which flips tokens one by one and holds only in the Boolean domain, is difficult to generalize to real values and cannot explain the junta-like phenomenon in large models. This paper moves to a different metric—**noise stability**: instead of perturbing coordinates individually, correlated noise is injected into all input coordinates simultaneously to see how much correlation remains in the output. The methodology follows a chain from metric to application—first providing a formal definition of noise stability and linking it to the function spectrum; then proving a lemma showing that "high stability" necessarily implies "low-frequency/junta-style simplicity"; followed by analyzing how this metric decays as it passes layer-by-layer through ReLU MLPs and attention layers; and finally formulating "encouraging high stability" as a differentiable regularization term to accelerate Transformer training and grokking.

### Key Designs

**1. Formal Definition of Noise Stability: Replacing Per-Token Perturbation with Correlated Noise**

The trouble with average sensitivity is that it flips tokens one by one and only holds naturally in the Boolean domain. Noise stability takes a different approach: it no longer perturbs each coordinate separately but adds **correlated Gaussian noise** to all input coordinates at once to measure the remaining correlation in the function output. Formally, for a function $f \in L^2(\gamma)$ under Gaussian measure $\gamma$, a correlated pair $(X,Y)$ is constructed, and the expected inner product of their outputs is taken:

$$\text{Stab}_\rho(f) := \mathbb{E}_{(X,Y)}[f(X) f(Y)]$$

where $Y = \rho X + Z\sqrt{1-\rho^2}$, $Z \sim \gamma$ is independent of $X$, and the correlation coefficient $\rho \in (0,1)$ controls noise intensity—the closer $\rho$ is to 1, the weaker the noise. This definition naturally exists in the real-valued domain (via the Ornstein-Uhlenbeck semigroup) and does not require the clunky hypergrid sampling of average sensitivity. Crucially, it is directly linked to the spectrum via Hermite-Fourier coefficients—where each order's coefficient is exponentially weighted by $\rho^{|\alpha|}$, suppressing higher orders $|\alpha|$ more severely:

$$\text{Stab}_\rho(f) = \sum_{\alpha \in \mathbb{N}^d} \rho^{|\alpha|} \tilde{f}(\alpha)^2$$

Thus, "output stability against correlated noise" is equivalent to "energy concentration on low-order Fourier coefficients," linking a robustness concept with spectral structure.

**2. Spectral Concentration Lemma (Lemma 1): Translating "Stable" to "Low-Frequency Dominant"**

A definition alone is insufficient; "stability" must be strictly proven to mean "simplicity." This lemma provides the bridge: as long as noise stability is close to the total energy of the function, the Fourier mass must be concentrated on low-order coefficients. Specifically, if $\text{Stab}_\rho(f) \geq (1-\delta)\|f\|_2^2$, then $f$ is $(\varepsilon, T)$-spectrally concentrated—meaning the spectral tail mass beyond truncation order $T$ does not exceed $\varepsilon$, and

$$T \geq \log_{1/\rho}\left(1 - \frac{\delta}{\varepsilon}\right)$$

The higher the stability (smaller $\delta$), the higher the order that can be suppressed under the same tail budget $\varepsilon$. This is the theoretical basis for calculating the "Fourier tail mass upper bound for degree $\geq 15$" on models like GPT-2 and RoBERTa, and why noise stability provides a tighter bound than average sensitivity.

**3. Layer-wise Stability Propagation Analysis: Passing through ReLU MLP and Attention**

The definitions and lemmas target "a function"; to apply the conclusions to real Transformers, one must know how stability **propagates layer-by-layer**. The paper provides closed-form propagation rates for individual components before recursing to multiple layers. For $\rho$-correlated Gaussian inputs $(X,Y)$, the output inner product of a single ReLU layer (Theorem 5.1) has a closed-form solution:

$$\mathbb{E}[\text{ReLU}(X) \cdot \text{ReLU}(Y)] = \frac{1}{2\pi}\left(\sqrt{1-\rho^2} + \rho(\pi - \arccos\rho)\right),$$

with a second-order Taylor expansion of approximately $\frac{1}{2\pi} + \frac{1}{4}\rho + \frac{1}{4\pi}\rho^2$. The lead term is roughly linear with $\rho$—a single layer of nonlinearity does not flatten the correlated noise but propagates it downward at a controlled ratio. The attention layer (Theorem 5.2/5.3) depends on the structure of the query-key matrix $W = W_Q W_K^T$: under **Identity $W=I_d$**, the attention matrix converges to $I_n$ in the high-dimensional limit, with stability remaining linear to $\rho$ with only $o(1)$ cost; under **Low-rank $W=UU^T$**, it reduces to the identity case via Johnson-Lindenstrauss; while under **Unstructured $W \sim \mathcal{N}(0,I)$** (worst-case random initialization), the attention matrix tends toward a random permutation matrix, and stability degrades to:

$$\rho \cdot s(\rho) \cdot \|(W_V)_{:,j}\|_2^2,$$

where $s(\rho):=\mathbb{P}(k=k')$ is the probability that the two noisy paths still select the same input token, i.e., the attention pattern is preserved. Structured attention propagates stability almost losslessly, whereas random attention incurs additional decay.

Recursive analysis through multiple layers shows different behaviors for the two components. A pure ReLU FFN recursion $\rho_L = \frac{1}{2\pi}(\sqrt{1-\rho_{L-1}^2} + \rho_{L-1}(\pi - \arccos\rho_{L-1}))$ converges to a non-zero fixed point $\frac{2}{3\pi} \approx 0.212$, representing **weak decay**—the signal is compressed to a finite lower bound rather than being consumed by layer depth. However, multi-layer Transformers with attention do not possess this property: the same recursion **no longer** yields weak decay. The paper observes that when $\|(W_V)_{:,j}\|_2 = \gamma < 1$, stability decays toward zero; the extent to which attention maps alter the distribution is sufficient to destroy the fixed-point behavior seen in FFNs. Consequently, for the multi-layer case, the paper uses **covariance interval propagation** to maintain upper and lower bounds of stability layer-by-layer rather than relying on a single fixed point.

### Loss & Training

By reversing the conclusion that "high stability = simplicity = easy generalization," one obtains a regularization term that encourages stability (Definition 6.1, direction parameter $S=1$ indicates encouragement, $S=0$ indicates inhibition):

$$R_{M,S,\rho}(X) = (-1)^S \cdot \sum_{i=1}^C M(X)_i \cdot M(Y)_i$$

It applies the inner product definition of noise stability directly to the model output distribution $M(\cdot)$: when constructing the perturbation sequence $Y$, each coordinate $Y_i$ remains $X_i$ with probability $\frac{1+\rho}{2}$, and is otherwise resampled from $\text{uniform}([U])$ (this is the implementation of correlated noise on discrete tokens). The final training objective is:

$$\ell_{\text{reg}}(M,X) = \ell(M,X) + \gamma \cdot R_{M,S,\rho}(X),$$

where $\gamma$ controls regularization strength. The regularization term is differentiable and depends on the model output on training data (rather than just parameters). Each iteration requires only one additional forward pass, incurring minimal computational overhead while consistently catalyzing grokking and accelerating training.

## Key Experimental Results

### Main Results

**Comparison of Spectral Concentration Upper Bounds ($n=256$, Fourier tail mass for degree $\geq 15$ )**:

| Model | Avg. Sensitivity Bound | Noise Stability Bound |
|------|---------------|---------------|
| GPT-2 | 0.003 | **0.0005** |
| BERT | 0.04 | **0.02** |
| RoBERTa | 0.19 | **0.02** |
| Gemma | 0.043 | **0.0157** |

Noise stability provides tighter Fourier tail mass estimates across all models (improvements ranging from 6x to 9.5x).

**Grokking Acceleration**:

| Task | Hyperparams (γ, ρ) | Conv. Steps w/o Reg. | Conv. Steps w/ Reg. | Gain |
|------|--------------|-----------------|-----------------|--------|
| Mod Addition (K=113) | (0.75, 0.25) | ~4500 | ~3300 | **36%** |
| Noisy k-sparse parity | (0.05, 0.05) | Baseline | Accelerated | **~35%** |
| WikiText-2 NTP | - | Baseline | Accelerated | **~75%** |

### Ablation Study

- **Junta-like characteristics of LLMs**: On 256-token inputs, GPT-2/RoBERTa/Gemma have only 5-10 tokens with significant geometric influence, far fewer than the 1024 tokens predicted by Friedgut's theorem.
- **Positional Bias**: The first and last tokens consistently show the highest influence, aligning with "attention sinks" observations in KV Cache compression literature.
- **Training Dynamics Monitoring**: In the noisy sparse parity task, the noise stability of the Transformer naturally decreases during training to match the target function; stability changes act as a precursor to generalization.
- **WikiText-2 Language Modeling**: The noise stability of the regularized model remains high, while the unregularized model becomes increasingly unstable.

### Key Findings

1. Noise stability characterizes the spectral concentration of Transformers more accurately than average sensitivity (tighter bounds for all models).
2. ReLU MLP layers introduce weak decay (converging to a fixed point of $2/(3\pi)$) rather than eliminating the signal entirely.
3. Attention layers maintain stability under identity/low-rank $W$ (linear relationship) but introduce an additional decay factor $s(\rho)$ under unstructured $W$.
4. Noise stability regularization acts as a catalyst for grokking, consistently accelerating training across multiple tasks.

## Highlights & Insights

1. **Unified Theoretical Framework**: Naturally generalizes Boolean domain analysis to the real domain via the Ornstein-Uhlenbeck semigroup, retaining strict links to the function spectrum and offering more analytical power than geometric influence.
2. **Cross-domain Bridging**: Establishes a new connection between signal propagation (C-maps/Q-maps) and simplicity bias/interpretability—noise stability can be seen as a more concise analogue of correlation maps.
3. **Practical Regularization**: A low-cost regularization method requiring only one extra forward pass; the 75% acceleration in NTP training is highly practical.
4. **LLM Internal Insights**: Quantifies the junta-like dependency of models like GPT-2 (only 5-10 tokens have significant impact), providing theoretical support for KV cache compression and token pruning.
5. **New Metric for Training Monitoring**: Changes in noise stability can serve as an early signal for grokking, offering new ideas for adaptive training strategies.

## Limitations & Future Work

1. Theoretical analysis omits actual Transformer components such as residual connections, LayerNorm, and attention masks.
2. Language modeling experiments were conducted only on small-scale WikiText-2, lacking validation on LLMs with billions of parameters.
3. The actual tightness of stability interval propagation in multi-layer Transformers has not been fully verified.
4. Regularization hyperparameters $(\gamma, \rho)$ require tuning for different tasks (e.g., (0.75, 0.25) for modular addition vs. (0.05, 0.05) for parity).
5. The quantitative relationship between noise stability and adversarial robustness has not been explored.

## Related Work & Insights

Unlike Vasudeva et al. (2024), who use average sensitivity to track grokking, this paper's noise stability provides stronger spectral concentration guarantees. It also fundamentally differs from the Transformer fine-tuning noise stability method of Hua et al. (2023) in motivation (simplicity bias vs. fine-tuning stability), scope, and the definition of correlated noise. The most direct inspiration comes from the noise sensitivity analysis of hierarchical functions by Li & Mossel (2025).

**Key Insight**: Noise stability can serve as a training monitor—a decrease in stability often predicts that grokking is about to occur, suggesting new paths for adaptive training. Additionally, the quantitative analysis of junta-like dependencies provides a theoretical basis for "which tokens actually matter" in prompt engineering.

## Rating

- Novelty: ⭐⭐⭐⭐ (The perspective of unifying signal propagation with simplicity bias is novel and theoretically sound)
- Experimental Thoroughness: ⭐⭐⭐ (Solid theory but small experimental scale, lacks validation on large models)
- Writing Quality: ⭐⭐⭐⭐ (Clear theoretical derivations, smooth flow, and intuitive charts)
- Value: ⭐⭐⭐⭐ (Provides new tools for understanding internal Transformer mechanisms; regularization has practical potential)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Latent Concept Disentanglement in Transformer-based Language Models](latent_concept_disentanglement_in_transformer-based_language_models.md)
- [\[ICLR 2026\] How Stable is the Next Token? A Geometric View of LLM Prediction Stability](how_stable_is_the_next_token_a_geometric_view_of_llm_prediction_stability.md)
- [\[ICML 2026\] Certified Circuits: Stability Guarantees for Mechanistic Circuits](../../ICML2026/interpretability/certified_circuits_stability_guarantees_for_mechanistic_circuits.md)
- [\[NeurIPS 2025\] AdaptGrad: Adaptive Sampling to Reduce Noise](../../NeurIPS2025/interpretability/adaptgrad_adaptive_sampling_to_reduce_noise.md)
- [\[ICLR 2026\] Hedonic Neurons: A Mechanistic Mapping of Latent Coalitions in Transformer MLPs](hedonic_neurons_a_mechanistic_mapping_of_latent_coalitions_in_transformer_mlps.md)

</div>

<!-- RELATED:END -->
