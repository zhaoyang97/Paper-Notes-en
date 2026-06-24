---
title: >-
  [Paper Note] The Discrete-Log Clock: How a Transformer Learns Modular Multiplication
description: >-
  [ICML 2026 (Mechanistic Interpretability Workshop)][Interpretability][Modular Multiplication] Previous work found that when a Transformer learns modular multiplication $a\cdot b\bmod p$, the Fourier spectrum of its embeddings is "dense" (requiring all frequencies), appearing much more complex and less interpretable than modular addition. This paper demonstrates that **this is merely an artifact of choosing the wrong analysis basis**. The natural Fourier basis for modular mult…
tags:
  - "ICML 2026 (Mechanistic Interpretability Workshop)"
  - "Interpretability"
  - "Modular Multiplication"
  - "Discrete Logarithm"
  - "Multiplicative Character Transform"
  - "Fourier Basis"
  - "Clock Algorithm"
date: 2026-05-08
content_hash: 5e457c5518e1fa4f
---

# The Discrete-Log Clock: How a Transformer Learns Modular Multiplication

**Conference**: ICML 2026 (Mechanistic Interpretability Workshop)  
**arXiv**: [2606.17399](https://arxiv.org/abs/2606.17399)  
**Code**: https://github.com/danny-cpp/Discrete-Log-Clock (Available)  
**Area**: Interpretability / Mechanistic Interpretability / Grokking  
**Keywords**: Modular Multiplication, Discrete Logarithm, Multiplicative Character Transform, Fourier Basis, Clock Algorithm

## TL;DR
Previous work found that when a Transformer learns modular multiplication $a\cdot b\bmod p$, the Fourier spectrum of its embeddings is "dense" (requiring all frequencies), appearing much more complex and less interpretable than modular addition. This paper demonstrates that **this is merely an artifact of choosing the wrong analysis basis**. The natural Fourier basis for modular multiplication is not the additive DFT, but the **multiplicative character transform** (Fourier transform over the multiplicative group $(\mathbb{Z}/p\mathbb{Z})^*$). Switching to this basis immediately reveals a sparse spectrum (Gini coefficient increases from 0.07 to 0.58, with only 4 key frequencies remaining, and 96.9% of MLP neurons cleanly tuned to a single frequency). This proves the Transformer **first utilizes the discrete logarithm to convert multiplication into addition**, then applies the same "Clock" trigonometric identity mechanism as modular addition—a mechanism the authors term the "Discrete-Log Clock."

## Background & Motivation

**Background**: Grokking (where a network suddenly generalizes long after memorizing the training set) has become a standard testbed for mechanistic interpretability. For **modular addition** $a+b\bmod p$, the internal algorithm has been fully reverse-engineered: the model learns a sparse Fourier representation, embeds integers as rotations on a circle, and uses trigonometric identities to compose these rotations—the "Clock" algorithm of Nanda et al. (2023) (with the "Pizza" variant discovered by Zhong et al. 2023).

**Limitations of Prior Work**: When moving to **modular multiplication** $a\cdot b\bmod p$, models also grok, but the internal algorithm remained unclear. Analytical MLP solutions proposed by Doshi et al. (2024) required **dense Fourier components (all frequencies)**; Furuta et al. (2024) also experimentally confirmed that grokked Transformers possess cosine bias components across all frequencies. These results suggested that multiplication utilizes an algorithm **fundamentally different and less interpretable** than addition.

**Key Challenge**: Is the "dense spectrum" due to the complexity of the algorithm itself, or the use of incorrect analysis tools? The additive Fourier transform is the natural basis for functions periodic over integer labels $a$, but the group structure of multiplication is not an additive group. Measuring multiplication with an additive ruler likely results in a dense spectrum as a **basis mismatch artifact** rather than actual algorithmic complexity.

**Goal**: Find an analysis basis matching the task's algebraic structure to determine if the internal algorithm for modular multiplication is actually simple and interpretable under the correct basis.

**Key Insight**: Since 113 is prime, non-zero elements $\{1,\dots,112\}$ form a **cyclic group** under multiplication. There exists a primitive root $g$ such that the discrete logarithm $\log_g$ is a group isomorphism $(\mathbb{Z}/p\mathbb{Z})^*\xrightarrow{\cong}\mathbb{Z}/(p-1)\mathbb{Z}$. In this relabeled coordinate system, **multiplication becomes addition**: $a\cdot b\equiv 3^{(\alpha+\beta)\bmod 112}$. Therefore, a model that learns the group structure should have embeddings periodic in the **relabeled coordinates**, just as addition models are periodic in original coordinates.

**Core Idea**: By switching the analysis basis from the additive DFT to the multiplicative character transform (reordering embedding rows by discrete log before applying standard DFT), the dense spectrum collapses into a sparse one, revealing the "Discrete-Log Clock" algorithm.

## Method

### Overall Architecture
This paper does not train a new mechanism but **reverse-engineers** a grokked Transformer. The process involves training a 1-layer decoder-only Transformer on $a\cdot b\bmod 113$ until it groks (test loss drops to near zero around epoch 9K–14K). The trained embedding matrix $W_E\in\mathbb{R}^{112\times 128}$ is projected onto both **additive bases** and **multiplicative character bases**. Using sparsity metrics (Gini, IPR), neuron single-frequency tuning ratios, and log-ordered SVD, the authors demonstrate that the model implements a "Discrete-Log Clock"—converting multiplication to addition via discrete logs and completing the computation using the same trigonometric identity circuits as addition.

The circuit implemented by the model can be decomposed into four interlocking steps:

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Input (a, b)"] --> B["Multiplicative Character Embedding<br/>Embed a as sin/cos(ω_k·log_g a)"]
    B --> C["Attention Routing<br/>Send frequency components of a, b to '=' position"]
    C --> D["MLP Trig Identities<br/>Calculate cos/sin(k(α+β))"]
    D --> E["Dot-product Scoring with Unembedding<br/>logit(c) ≈ Σ A_k cos(ω_k(α+β−γ))"]
    E -->|Constructive interference only for correct answer γ*=(α+β)| F["Output c = a·b mod p"]
```

### Key Designs

**1. Multiplicative Character Transform: The Correct Fourier Ruler for Multiplication**

The problem is that additive DFT necessarily sees a dense spectrum for multiplication. The multiplicative character basis uses $\sin(2\pi k\log_g(a)/q)$ and $\cos(2\pi k\log_g(a)/q)$ (where $q=112$). Operationally, this is **reordering embedding rows by their discrete logarithm before taking a standard DFT**—this is precisely the Fourier transform on the multiplicative group $(\mathbb{Z}/p\mathbb{Z})^*$, decomposing functions into multiplicative characters (irreducible representations of the group). The motivation comes from $a\cdot b\equiv 3^{(\alpha+\beta)\bmod 112}$: multiplication becomes addition after relabeling, so embeddings that learn the group structure should be periodic in the relabeled coordinate $\alpha=\log_g a$. An individual embedding row $W_E[a]$ appears unstructured because it is a point-sampled sum of 128 waves at $\alpha=\log_g(a)$ plus noise; the sinusoidal structure $W_E[g^\alpha,j]\approx A_j\sin(\omega_{k_j}\alpha+\phi_j)$ only emerges after reordering by $\alpha$. This is the core methodology: **matching the analysis basis to the task's algebraic structure**.

**2. Sparsity Quantification: Hard Numbers for "Dense to Sparse" via Gini / IPR / Key Frequencies**

To go beyond qualitative claims, the authors calculate the combined norm $\|f_k\|=\sqrt{\|s_k\|^2+\|c_k\|^2}$ for each frequency $k$ ($s_k, c_k$ are 128-dim sin/cos projections), then use the Gini coefficient:

$$G=\frac{\sum_{i=1}^n(2i-n-1)|x_i|}{n\sum_{i=1}^n|x_i|}$$

to measure concentration ($x_i$ are sorted frequency norms; $G=0$ is uniform/dense, $G=1$ is perfectly sparse). Frequencies exceeding $5\times$ the median are marked as "key frequencies." Results: Additive basis $G=0.071$, 0 key frequencies; Multiplicative basis $G=0.579$ (an 8.1× increase), with energy concentrated in 4 peaks $\{2,8,47,56\}$. The Inverse Participation Ratio also drops from 52.7 to 4.1. **The same embedding yielding a 4-peak spectrum simply by changing the basis** is direct evidence that density was an artifact.

**3. Neuron Tuning + Log-Order Reordering: Evidence at the Unit Level**

To confirm sparsity is not superficial, the authors examine 512 MLP neurons. For each neuron, they calculate its 2D activation $h_n(a,b)$ across all input pairs (reordered into a $112\times112$ grid), apply 2D DFT in both bases, and measure the ratio of energy in the maximum frequency. A ratio $>85\%$ is defined as "single-frequency tuning." Under the multiplicative basis, **496/512 (96.9%) neurons are single-frequency tuned**, compared to 0 in the additive basis. More visually: reordering neuron activation heatmaps **by discrete logs** reveals clear diagonal stripes, indicating the neuron computes a periodic function of $\alpha+\beta$—the signature of the $\cos(k(\alpha+\beta))$ trigonometric identity mechanism, identical to Nanda's findings in addition, but occurring **after** the discrete log transformation. Log-ordered SVD further confirms this: principal components of $W_E$ are noise in integer order but reveal sinusoidal structures when reordered by discrete logs, with an effective rank of only 10.3/112.

### Loss & Training
The model is a 1-layer decoder-only Transformer ($d_{\text{model}}=128$, 4 attention heads with $d_{\text{head}}=32$, MLP hidden dimension 512 + ReLU, no LayerNorm). Input token sequences are $[a,b,=]$, with predictions read from the "=" position logits. It is trained on 30% of data (3,763 of 12,544 pairs) using AdamW, learning rate $10^{-3}$, **weight decay 1.0**, and full-batch gradient descent for 40,000 epochs. Hyperparameters are identical to Nanda et al. (2023), with the **only change being the task (multiplication instead of addition)**. The model memorizes around epoch 500, groks between 9K–14K, and reaches 100% test accuracy. Weight decay penalizes large unstructured weights required for memorization, forcing the model to find compact Fourier representations.

## Key Experimental Results

### Main Results: Diversity of Basis Sparsity ($a\cdot b\bmod 113$)

| Metric | Additive Basis | Multiplicative Basis |
|------|--------|--------|
| Gini Coefficient | 0.071 | 0.579 |
| Inverse Participation Ratio | 52.7 | 4.1 |
| Key Frequencies Found | 0 | 4 ({2,8,47,56}) |
| Neurons >85% Single-Freq | 0/512 | 496/512 |
| Avg. Max Frequency Power | 0.054 | 0.925 |

### Ablation Study: Generalization across Primes and Seeds

| Setting | Result | Description |
|------|------|------|
| 10 Primes ($p=59$ to $113$) | All show sparse multiplicative spectra, 4–7 key frequencies | Discrete-log clock is universal |
| $p=113$ Multiple Seeds | ~80% of seeds exhibit this mechanism, Gini 0.45–0.61 | Mechanism is generally robust, though specific frequencies vary |
| Log-order SVD | Effective rank 10.3/112, sinusoidal structure | Embeddings live in low-dim subspace encoding multiplicative characters |
| Modular Power $a^b\bmod 41$ (2 layers) | 100% Accuracy | First Transformer to perfectly learn modular exponentiation; mechanism under study |

### Key Findings
- **Basis Switching Unveils Sparsity**: The change from Gini 0.071 (additive) to 0.579 (multiplicative) proves "dense spectra" are artifacts of basis mismatch, not algorithmic complexity.
- **Neuron-Level Homogeneity**: 96.9% of MLP neurons lock onto a single frequency in the multiplicative basis, confirming the model performs addition-style trigonometric operations in log space.
- **Discrete-Log Reordering as the Key**: Embedding heatmaps and SVD components appear as noise in integer order but reveal periodic/sinusoidal structures when reordered by discrete logs.
- **Why Previous Work Missed It**: $\log_g(a)$ is a non-linear permutation of integers. Functions periodic in $\log_g(a)$ appear non-periodic in $a$, leading additive DFTs to report "all frequencies," misinterpreting a tool mismatch as algorithmic complexity.

## Highlights & Insights
- **"The Right Basis Turns Noise into Structure"**: The core insight is that the analysis basis must match the task's algebraic structure. Applying group-theoretic Fourier transforms to any algebraic task should reveal interpretable structures.
- **Overturning a Counter-example**: While previous work used dense spectra to argue that multiplication is fundamentally different/harder than addition, this work proves it is simply the "Addition Clock in log space," elegantly unifying the two.
- **Epistemological Lesson of Discrete Log Permutations**: The same data/weights can appear structured or unstructured depending on the sorting order. This serves as a reminder that a "lack of structure" in mechanistic interpretability may simply be a failure to find the correct coordinate system.

## Limitations & Future Work
- **Uniqueness Not Proven**: The authors acknowledge they have not proven this is the *only* possible algorithm. Just as Zhong et al. (2023) showed different architectures converge to different solutions (Clock vs. Pizza) for addition, multiplication may have variants.
- **Scale Constraints**: Analysis is limited to a 1-layer Transformer on $p=113$. While validated across 10 primes, these are "toy" grokking tasks with indirect implications for large-scale LLMs.
- **Modular Exponentiation $a^b\bmod p$**: Only existence (100% accuracy) was shown. The internal mechanism remained un-reversed, especially regarding how multi-layer architectures compose these features.
- **Frequency Variation**: Key frequencies vary by seed (e.g., $\{2,8,47,56\}$ is seed-specific), implying specific frequencies are less important than the "sparse + single-frequency" structural property.

## Related Work & Insights
- **vs. Nanda et al. (2023) Clock (Addition)**: They identified sparse Fourier + trig identity "Clocks." This paper proves multiplication is the same mechanism within a discrete log space.
- **vs. Doshi et al. (2024)**: They used quadratic activations to derive analytical MLP solutions for multiplication. This paper bridges that theory with the empirical Fourier landscape of ReLU Transformers.
- **vs. Furuta et al. (2024)**: They reported multiplication spectra use "all frequencies." This paper directly refutes that, showing density is a basis artifact.
- **vs. Chughtai et al. (2023) / Stander et al. (2024)**: They studied irreducible representations and coset circuits in symmetric groups ($S_5, S_6$). This paper focuses on the application of multiplicative character bases to $(\mathbb{Z}/p\mathbb{Z})^*$ in trained Transformers.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The "dense-to-sparse" flip via basis switching is a brilliant insight with transferable methodology.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Triple evidence (metrics + neurons + SVD) validated across primes and seeds; however, still limited to toy models.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear logical chain, complete definitions/formulas, and full circuit derivations provided.
- **Value**: ⭐⭐⭐⭐ Resolves a major mystery in mechanistic interpretability and establishes the principle of "matching analysis basis to algebraic structure."

## Related Papers

- [\[ICLR 2026\] A Cortically Inspired Architecture for Modular Perceptual AI](../../ICLR2026/interpretability/a_cortically_inspired_architecture_for_modular_perceptual_ai.md)
- [\[ICML 2026\] How Language Models Process Negation](how_language_models_process_negation.md)
- [\[ICML 2026\] Prototype Transformer: Towards Language Model Architectures Interpretable by Design](prototype_transformer_towards_language_model_architectures_interpretable_by_desi.md)
- [\[ICLR 2026\] Noise Stability of Transformer Models](../../ICLR2026/interpretability/noise_stability_of_transformer_models.md)
- [\[ICML 2026\] Query Circuits: Explaining How Language Models Answer User Prompts](query_circuits_explaining_how_language_models_answer_user_prompts.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] On The Geometry and Topology of Representations: the Manifolds of Modular Addition](../../ICLR2026/interpretability/on_the_geometry_and_topology_of_representations_the_manifolds_of_modular_additio.md)
- [\[ICLR 2026\] Emergent Discrete Controller Modules for Symbolic Planning in Transformers](../../ICLR2026/interpretability/emergent_discrete_controller_modules_for_symbolic_planning_in_transformers.md)
- [\[ICLR 2026\] Mixture of Cognitive Reasoners: Modular Reasoning with Brain-Like Specialization](../../ICLR2026/interpretability/mixture_of_cognitive_reasoners_modular_reasoning_with_brain-like_specialization.md)
- [\[ICML 2026\] How Language Models Process Negation](how_language_models_process_negation.md)
- [\[ICLR 2026\] Features Emerge as Discrete States: The First Application of SAEs to 3D Representations](../../ICLR2026/interpretability/features_emerge_as_discrete_states_the_first_application_of_saes_to_3d_represent.md)

</div>

<!-- RELATED:END -->
