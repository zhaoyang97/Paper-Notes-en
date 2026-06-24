---
title: >-
  [Paper Note] LLMs as Noisy Channels: A Shannon Perspective on Model Capacity and Scaling Laws
description: >-
  [ICML 2026][Model Compression][Shannon Capacity] This paper reinterprets LLM training as a Shannon-Hartley noisy channel—where parameter count corresponds to bandwidth, training tokens to signal power, and data/model noise to channel noise. From this framework, it derives the Shannon Scaling Law $C_{\text{LLM}} = aN^\alpha \log_2(1 + bD^\beta / (c(DN)^\gamma + dD^\delta + e))$, which unifies the explanation of classical monotonic scaling and recently discovered U-shaped degra…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Shannon Capacity"
  - "Scaling Laws"
  - "U-shaped Loss"
  - "Catastrophic Overtraining"
  - "Quantization Degradation"
date: 2026-05-08
content_hash: 8bd5bef40de717a6
---

# LLMs as Noisy Channels: A Shannon Perspective on Model Capacity and Scaling Laws

**Conference**: ICML 2026  
**arXiv**: [2605.23901](https://arxiv.org/abs/2605.23901)  
**Code**: TBD  
**Area**: Model Compression / Scaling Laws / Information Theory  
**Keywords**: Shannon Capacity, Scaling Laws, U-shaped Loss, Catastrophic Overtraining, Quantization Degradation

## TL;DR
This paper reinterprets LLM training as a Shannon-Hartley noisy channel—where parameter count corresponds to bandwidth, training tokens to signal power, and data/model noise to channel noise. From this framework, it derives the Shannon Scaling Law $C_{\text{LLM}} = aN^\alpha \log_2(1 + bD^\beta / (c(DN)^\gamma + dD^\delta + e))$, which unifies the explanation of classical monotonic scaling and recently discovered U-shaped degradation (catastrophic overtraining, quantization-induced degradation). It achieves an extrapolation $R^2 = 0.847$ on a 12B model with 307B tokens using data from Pythia/OLMo2 models $\le 6.9\text{B}$.

## Background & Motivation

**Background**: The development of LLMs is guided by scaling laws—the OpenAI law uses $L = [(N_c/N)^{\alpha_N/\alpha_D} + D_c/D]^{\alpha_D}$, and the Chinchilla law uses the additive form $L = A/N^\alpha + B/D^\beta + E$, both assuming performance improves monotonically with computation. This has driven the creation of trillion-parameter models (DeepSeek-V4 1.6T, Kimi K2.6 1T) and massive pre-training datasets.

**Limitations of Prior Work**: A series of recent findings challenge the monotonicity assumption:
- **Catastrophic Overtraining** (Springer 2025): Excessive pre-training actually harms downstream SFT.
- **Quantization-induced Degradation (QiD)** (Kumar 2024, Ouyang 2024): Models trained larger or longer are more sensitive to quantization.
These produce **U-shaped loss curves**—decreasing then increasing. Existing perturbation-aware scaling laws (Ouyang 2024, Kumar 2024) merely add degradation terms $\Delta_q L$ or $\delta_{\text{PTQ}}$ to basic laws, lacking a unified theory.

**Key Challenge**: Monotonic power laws assume infinite signal improvement, but practical LLM training involves both signal (learned knowledge) and noise (data noise, model noise, quantization perturbations, etc.). Performance gains continue as long as signal growth outweighs noise growth, but the curve turns upwards when noise dominates beyond a critical point. A unified framework is needed to express the interaction between signal and noise.

**Goal**: (1) Establish a unified scaling law that describes both monotonic gains (high SNR regime) and U-shaped degradation (low SNR regime); (2) Derive this from Shannon's first principles rather than empirical curve fitting; (3) Enable extrapolation to larger models or longer training beyond the training distribution.

**Key Insight**: LLM training can be likened to Shannon communication—pre-training is channel modulation (encoding information into weights), and inference is transmission (from context to output). The Shannon-Hartley theorem $C = B \log_2(1 + S/\mathcal{N})$ naturally describes the capacity upper bound of "signal vs. noise." When $S/\mathcal{N}$ is high, capacity approximates $B \log_2(S/\mathcal{N})$ and increases monotonically; when noise is large, capacity is compressed—perfectly corresponding to the U-shape.

**Core Idea**: Parameter $N \to$ Bandwidth $B$, training tokens $D \to$ Signal Power $S$, data noise + model noise + perturbations $\to$ Noise $\mathcal{N}$. This results in the Shannon Scaling Law $C_{\text{LLM}} = aN^\alpha \log_2(1 + bD^\beta / (c(DN)^\gamma + dD^\delta + e))$, unifying monotonic and U-shaped behaviors.

## Method

### Overall Architecture

This paper treats LLM training as a Shannon-Hartley noisy channel: parameter count $N$ is the channel bandwidth, training tokens $D$ are the injected signal power, and various noises from data and the model itself constitute the channel noise $\mathcal{N}$. The ultimate knowledge capacity the model can accommodate is the channel capacity $C_{\text{LLM}}$. Specifically, bandwidth $B_{\text{LLM}} \propto N^\alpha$, signal $S_{\text{LLM}} \propto D^\beta$, and noise is decomposed into data noise $dD^\delta$ + model interaction noise $c(DN)^\gamma$ + irreducible perturbation $e$. Substituting these into $C = B \log_2(1 + S/\mathcal{N})$ yields the core formula:

$$C_{\text{LLM}} = aN^\alpha \log_2\left(1 + \frac{bD^\beta}{c(DN)^\gamma + dD^\delta + e}\right)$$

This formula reduces to classical monotonic scaling in high SNR regions while naturally producing U-shaped degradation when noise exceeds the signal. To map capacity to measurable metrics, the paper adds a reciprocal bridge $\mathcal{L} = 1/C_{\text{LLM}}$: larger capacity leads to lower test loss ($L \to 0$ as $C \to \infty$). This reciprocal relationship naturally captures the non-linearity where early capacity gains significantly reduce loss, while further reductions require massive capacity increases near convergence. This $1/C_{\text{LLM}}$ is what is fitted and extrapolated in the experiments.

### Key Designs

**1. Three-Source Noise Decomposition: Explaining Overtraining and Quantization via $\mathcal{N}$**

Classical scaling laws lack "noise," making them helpless against U-shaped curves. This paper decomposes channel noise $\mathcal{\mathcal{N}} = c(DN)^\gamma + dD^\delta + e$ into three layers, each corresponding to a real degradation mechanism. Data noise $dD^\delta$ originates from typos, ambiguities, and contradictions in the corpus; it accumulates with more tokens, explaining catastrophic overtraining where excessive pre-training piles up noise that eventually outweighs signal gains. Model interaction noise $c(DN)^\gamma$ treats the training process as fitting noise generated by the interaction between the model and data; using the product $DN$ instead of sum captures the coupling of "large models $\times$ big data," explaining why larger models are more fragile to quantization (QiD). Finally, $e$ represents irreducible intrinsic perturbations at the hardware/algorithmic level. This decomposition allows specific attribution of training failures to data or model noise via the parameters $c, d, e$.

**2. Unified Monotonicity and U-shape: Limits of a Single Formula in Two SNR Regimes**

Previous perturbation-aware laws (Ouyang, Kumar) typically summed a basic law with a degradation term, essentially stitching two curves together without a theoretical basis. This paper avoids "patches" and relies solely on the shape of $\log_2(1 + S/\mathcal{N})$. When $S \gg \mathcal{N}$, it simplifies to $\log_2(S/\mathcal{N})$, where capacity increases monotonically with $D$, recreating the classical power law. As noise growth catches up—for instance, when the data noise term $dD^\delta$ dominates—capacity turns downward. The U-shape is not an manual addition but an emergent property of the same formula in the low SNR limit. Sharing one set of parameters and one information-theoretic principle underpins its superior extrapolation power.

**3. Unified Encoding of Multi-Source Perturbations: Redefining Fit without Specific Degradation Terms**

Additive Gaussian noise, quantization, and SFT in different domains appear unrelated. Classical perturbation-aware laws (QiD, Law of Precision) add specific degradation terms with perturbation intensity $X$ (e.g., $dN^{\alpha'}D^{\beta'}X^\gamma$), requiring new modeling for each perturbation type. This paper introduces no extra terms: all perturbations are viewed as increasing the same noise term $\mathcal{\mathcal{N}}$ and lowering the effective SNR. Consequently, the same capacity formula is used for any perturbation scenario, simply re-fitting the nine parameters $a, b, c, d, e, \alpha, \beta, \gamma, \delta$ on the perturbed data. The framework's generality is demonstrated by fitting all six perturbation scenarios, including Gaussian, quantization, and three types of SFT.

## Key Experimental Results

### Main Results: Fitting Quality ($R^2$)

| Experimental Setting | OpenAI Law | Chinchilla | Perturbation-aware (Ouyang) | **Shannon** |
| :--- | :--- | :--- | :--- | :--- |
| Pythia Standard Pre-training | 0.91 | 0.93 | 0.93 | **0.95** |
| Pythia + Gaussian Noise | 0.42 | 0.45 | 0.78 | **0.91** |
| Pythia + Quantization | 0.38 | 0.41 | 0.82 | **0.93** |
| OLMo2 + Math SFT | 0.31 | 0.34 | 0.67 | **0.89** |
| OLMo2 + QA SFT | 0.28 | 0.30 | 0.69 | **0.88** |
| OLMo2 + Code SFT | 0.35 | 0.37 | 0.71 | **0.90** |

The Shannon law significantly outperforms others in all perturbation scenarios, especially in SFT scenarios where classical laws completely fail.

### Extrapolation Ability (Key Test)

| Training Data | Test Model | Test Token Count | OpenAI $R^2$ | Chinchilla $R^2$ | **Shannon $R^2$** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Pythia $\le 6.9\text{B}$, $\le 180\text{B}$ tokens | Pythia 12B | 180B | 0.65 | 0.71 | **0.93** |
| Pythia $\le 6.9\text{B}$, $\le 180\text{B}$ tokens | Pythia 12B | 240B | 0.42 | 0.48 | **0.89** |
| Pythia $\le 6.9\text{B}$, $\le 180\text{B}$ tokens | Pythia 12B | 307B | **$-0.18$** | **$-0.05$** | **0.847** |

While classical laws yield negative $R^2$ (worse than a constant fit) when extrapolating to 307B tokens, the Shannon law maintains an $R^2$ of 0.847, demonstrating genuine out-of-distribution extrapolation.

### U-shaped Capture
In Pythia quantization experiments, as bit-width decreases from 4 to 2, loss remains stable before surging. OpenAI/Chinchilla laws fail to capture this critical point; the Ouyang perturbation-aware law captures it partially with shifts, while the Shannon law precisely identifies the loss basin location.

### Key Findings
- **Unified framing is more powerful than additive formulas**: The Shannon law is not just "another perturbation-aware law" but is derived from first principles. Monotonicity and U-shapes are two limits of the same equation, leading to superior extrapolation.
- **Interpretable three-source noise decomposition**: Fitting $c, d, e$ allows quantitative analysis of degradation sources (e.g., determining if a failed training was dominated by data or model noise).
- **No collapse in 12B / 307B extrapolation**: Training only on data up to 6.9B / 180B tokens still predicts performance far beyond that scale, suggesting the Shannon law captures underlying laws rather than curve-fitting tricks.
- **Consistency across perturbation types**: Gains are observed across Gaussian, quantization, and SFT scenarios, proving the framework's universality.

## Highlights & Insights
- **Unifying scaling and degradation through Information Theory is a profound perspective**: While previous scaling laws were empirical fits, the Shannon law is derived from communication theory. This first-principle approach provides a template for future scaling research.
- **Interpretability of noise decomposition**: Separating data quality, model-data interaction, and irreducible terms provides quantitative diagnostics for training failures.
- **Engineering significance of $R^2 = 0.847$ extrapolation**: The highest value of a scaling law is prediction. This model's accuracy outside the training distribution suggests it can use small-scale experiments to predict large model behavior, saving massive computational resources.
- **Analogy to the Shannon-Weaver model**: Viewing inference as channel transmission (context to output) allows optimization techniques like KV cache or speculative decoding to be analyzed within the same framework.

## Limitations & Future Work
- It remains an empirical fit with 9 parameters ($a, b, c, d, e, \alpha, \beta, \gamma, \delta$) requiring data estimation; whether these can be further derived from channel coding theory is worth exploring.
- Validation is limited to Pythia and OLMo2 families; fitting quality on closed-source models (GPT, Claude) is untested.
- Architectural differences (e.g., attention heads, layer width) are not explicitly modeled, relying only on total parameters $N$.
- Inference-time capacity (context length, KV cache, etc.) is not included; inference scaling laws are an independent, interesting direction.
- Shannon-Hartley assumes AWGN, whereas LLM training noise distributions are more complex; generalized capacity formulas could be considered.

## Related Work & Insights
- **vs. OpenAI / Chinchilla**: These laws assume monotonicity and are refuted by U-shaped phenomena; the Shannon law is their strict superset (reducing to power laws in high SNR regimes).
- **vs. Ouyang QiD law / Kumar precision law**: These use patches for degradation; the Shannon law emerges naturally from a single formula.
- **vs. Information Bottleneck theory (Tishby)**: While that uses mutual information to analyze DNN representations, this paper uses channel capacity to analyze LLM training, providing a parallel perspective.
- **Insight**: Re-examining empirical scaling laws under an information-theoretic framework may reveal unified principles; this "first principle scaling" approach is worth applying to RL, multi-modal, and agent domains.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to systematically unify LLM scaling and degradation using Shannon-Hartley; unique theoretical depth.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Fit across 6 perturbation scenarios + out-of-distribution extrapolation provides solid evidence.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear information-theoretic analogy; Figure 3 provides an intuitive communication structure; strong explanatory power in noise decomposition.
- Value: ⭐⭐⭐⭐⭐ Scaling laws are fundamental tools for LLM research; unification and extrapolation provide a more reliable basis for planning and resource saving.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](model_merging_scaling_laws_in_large_language_models.md)
- [\[ACL 2026\] Task-Stratified Knowledge Scaling Laws for Post-Training Quantized LLMs](../../ACL2026/model_compression/task-stratified_knowledge_scaling_laws_for_post-training_quantized_large_languag.md)
- [\[ICML 2026\] Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training](decouple_searching_from_training_scaling_data_mixing_via_model_merging_for_large.md)
- [\[ACL 2025\] Spectra 1.1: Scaling Laws and Efficient Inference for Ternary Language Models](../../ACL2025/model_compression/scaling_laws_and_efficient_inference_for_ternary_language_models.md)
- [\[ICML 2026\] MIC: Maximizing Informational Capacity in Adaptive Representations via Isotropic Subspace Alignment](mic_maximizing_informational_capacity_in_adaptive_representations_via_isotropic_.md)

</div>

<!-- RELATED:END -->
