---
title: >-
  [Paper Note] Towards Atoms of Large Language Models
description: >-
  [ICML2026][Interpretability][Atom theory] This paper provides the first formal definition of the "fundamental representation units" of Large Language Models (LLMs)—termed **atoms**. It characterizes the intrinsic geometry of LLM hidden representations using a non-Euclidean "Atom Inner Product" and proves that threshold-activated SAEs can precisely recover the set of atoms under appropriate conditions. Empirical tests on Gemma2 / Llama3.1 identify near-ideal atoms with $R^2 \a…
tags:
  - "ICML2026"
  - "Interpretability"
  - "Atom theory"
  - "Sparse Autoencoder"
  - "Representation geometry"
  - "Monosemanticity"
  - "Fundamental units of representation"
date: 2026-05-08
content_hash: 17f9185ca1eeafa0
---

# Towards Atoms of Large Language Models

**Conference**: ICML2026  
**arXiv**: [2509.20784](https://arxiv.org/abs/2509.20784)  
**Code**: https://github.com/ChenhuiHu/towards_atoms  
**Area**: Interpretability / Mechanistic Interpretability  
**Keywords**: Atom theory, Sparse Autoencoder, Representation geometry, Monosemanticity, Fundamental units of representation  

## TL;DR
This paper provides the first formal definition of the "fundamental representation units" of Large Language Models (LLMs)—termed **atoms**. It characterizes the intrinsic geometry of LLM hidden representations using a non-Euclidean "Atom Inner Product" and proves that threshold-activated SAEs can precisely recover the set of atoms under appropriate conditions. Empirical tests on Gemma2 / Llama3.1 identify near-ideal atoms with $R^2 \approx 99.9\%$ and stability $q^* \approx 99.85\%$.

## Background & Motivation
**Background**: Decomposing the internal computations of LLMs into "interpretable units" is the core of mechanistic interpretability. Early work treated neurons (single-dimensional activations) as fundamental units. In recent years, the mainstream shifted to "features" learned by Sparse Autoencoders (SAEs)—decomposing the residual stream sparsely into a set of "dictionary directions" and assigning semantic labels using LLM-as-Judge.

**Limitations of Prior Work**: Neurons are often contaminated by polysemanticity, where activation patterns span unrelated concepts. Features face two persistent issues: large reconstruction residuals ("dark matter") and instability (splitting/merging) when training scale or sparsity regularization changes. From an evaluation perspective, it remains unclear whether a feature learned by an SAE truly constitutes a fundamental unit of the LLM, as no formal definition of such a unit exists.

**Key Challenge**: All existing evaluations implicitly couple "faithfulness" and "monosemanticity" into the SAE training objective, leading to circular reasoning where SAE outputs are evaluated using the SAE's own loss. To break this cycle, one must define the properties of "ideal atoms" independently of any specific architecture based on the geometry of the representation space.

**Goal**: (i) Define the fundamental units (atoms) of LLM representations; (ii) Design computable metrics (faithfulness $R^2$, stability $q^*$) to independently measure any candidate unit; (iii) Provide a practical algorithm capable of theoretically recovering the set of atoms.

**Key Insight**: The authors observe that the LLM training objective only perceives $\bm{h}^L$ through the Softmax function. Thus, representations are identified only up to an invertible linear transformation $\bm{A}$. The Euclidean inner product is not invariant under this equivalence class; therefore, Euclidean geometry is not the "correct" geometry for LLM representations. A different metric $\bm{S}$ is required to bind concepts like "orthogonality" and "angles" to model behavior.

**Core Idea**: Use the "Atom Inner Product" induced by $\bm{S}=(\bm{D}\bm{D}^\top)^{-1}$ as the intrinsic metric of LLM representations. Under this metric, atoms are redefined via (representability, sparsity, separability). The authors prove that a threshold-activated SAE can strictly recover the atom set when $\delta_{\min}>\varepsilon(2K-1)\delta_{\max}$.

## Method

### Overall Architecture
This paper addresses what the fundamental units of LLM representations are and how to verify them independently. The approach first switches to the correct geometric metric, defines "atoms" as verifiable geometric properties under this metric, and proposes a Threshold SAE (TSAE) algorithm to recover them. Specifically, given a set of activations $M=\{\bm{m}_i\}\subset\mathbb{R}^H$ from an LLM layer, the metric matrix $\tilde{\bm{S}}=(\mathbb{E}[\bm{k}\bm{k}^\top])^{-1}$ is estimated from 100K Wikipedia activations to replace the Euclidean inner product. Faithfulness $R^2$ and stability $q^*$ are then used to measure candidate units. Finally, TSAE is used on Gemma2-2B/9B and Llama3.1-8B across various "data scale × dictionary capacity" settings to identify stable atoms. This chain does not rely on the training loss of any specific SAE, breaking the circular reasoning.

### Key Designs

**1. Atom Inner Product (AIP): A Correct Geometric Ruler for LLM Representations**

The Euclidean inner product is unsuitable because the training objective only sees $\bm{h}^L$ through Softmax, making representations invariant under invertible linear transformations $\bm{A}$. Consequently, Euclidean "orthogonality" decouples from model behavior. Starting from translation invariance and unit-norm symmetry, the authors derive the unique metric $\bm{S}=c^2(\bm{D}\bm{D}^\top)^{-1}$, normalized as $\tilde{\bm{S}}=(\bm{D}\bm{D}^\top)^{-1}$. The Atom Inner Product is defined as $\langle\bm{u},\bm{v}\rangle_{\tilde S}=\bm{u}^\top\tilde{\bm{S}}\bm{v}/(\|\bm{u}\|_{\tilde S}\|\bm{v}\|_{\tilde S})$. Equivalently, after whitening activations to $\tilde{\bm{d}}_i=\tilde{\bm{S}}^{1/2}\bm{d}_i$, the AIP reduces to the standard Euclidean inner product. This derivation avoids arbitrary metric selection. Empirically, Euclidean angles in LLMs deviate from $90^\circ$ due to "representation drift" caused by Softmax, whereas AIP resets the angular center to $90^\circ$, effectively removing global bias.

**2. Three Atom Properties + Sparsity-Separability Coupling $q^*$: Defining Units as Computable Scalars**

"Fundamental units" are translated into verifiable geometric conditions: an atom must satisfy representability ($\bm{m}_i=\bm{D}\bm{\delta}_i$), sparsity ($\|\bm{\delta}_i\|_0\le K$), and $\epsilon$-approximate orthogonality ($|\langle\tilde{\bm{d}}_i,\tilde{\bm{d}}_j\rangle|\le\epsilon$ for $i\ne j$). The authors use the Restricted Isometry Property (RIP) from compressed sensing to couple sparsity and separation. The dictionary coherence $\mu:=\max_{i\ne j}|\langle\tilde{\bm{d}}_i,\tilde{\bm{d}}_j\rangle|$ and sparsity $K$ control the RIP constant. Using the uniqueness theorem ($\mu<\frac{1}{2K-1}$), the stability index is defined as $q^*:=\sup\{q\mid\mu_q<\frac{1}{2K_q-1}\}$, characterizing the proportion of sparse supports where decomposition is unique (monorepresentationality). Since semantic monosemanticity is hard to formalize, this mathematical "monorepresentationality" provides a provable and computable proxy applicable to neurons, features, or atoms.

**3. Identifiability Theorem for Threshold SAE (TSAE): Attributing SAE Failure to Activation Functions**

The authors prove that using a threshold activation $\sigma_\tau(x)=x\cdot\mathbb{1}[x\ge\tau]$, if the non-zero coefficients satisfy $\delta_{\min}\le\delta_{ij}\le\delta_{\max}$ and the threshold falls within $\varepsilon K\delta_{\max}<\tau<\delta_{\min}-\varepsilon(K-1)\delta_{\max}$ (feasibility condition $\delta_{\min}>\varepsilon(2K-1)\delta_{\max}$), then setting $\bm{W}_{dec}=\bm{D}$ and $\bm{W}_{enc}=\bm{D}^\top\tilde{\bm{S}}$ guarantees $\bm{W}_{dec}\sigma_\tau(\bm{W}_{enc}\bm{m}_i)=\bm{m}_i$. This strictly decouples atoms and coefficients. Implementation uses JumpReLU to provide a hard cutoff. The insight is that standard ReLU SAEs lack a hard threshold, allowing noise from approximate orthogonality to leak into "non-support" dimensions, which destroys uniqueness.

### Loss & Training
TSAE is trained using JumpReLU with $4\times$ over-parameterization (dictionary capacity $|D|=4H$). A grid search on Gemma2-2B Layer 1 with 1.9B activations shows that faithfulness $R^2$ stays between 0.6–0.8 until dictionary capacity $|D|$ exceeds a critical threshold matching the data scale $|M|$, at which point $R^2$ jumps to $\approx 1$. Thus, $R^2$ serves as an indirect signal for whether the RIP condition is triggered.

## Key Experimental Results

### Main Results

| Model | Layers | Faithfulness $R^2$ | Stability $q^*$ | Gap vs. Neurons/Features |
|------|------|------|------|------|
| Gemma2-2B | 1–26 | 99.92% | 99.74% | features $R^2$=48.8% / $q^*$=68.2% |
| Gemma2-9B | 1–42 | 99.93% | 99.87% | neurons $R^2$=100% / $q^*$=0.5% |
| Llama3.1-8B | 1–30 | 99.85% | 99.95% | Metrics near the ideal (1, 1) corner |

Across models of various scales, TSAE units achieve near-perfect scores on both metrics, signifying "ideal atoms."

### Ablation Study

| Configuration | $R^2$ | $q^*$ | Description |
|------|------|------|------|
| Neurons (baseline) | 1.00 | 0.005 | Fully faithful but highly polysemantic |
| Features (standard SAE) | 0.488 | 0.682 | Stable but high reconstruction residual |
| Euclidean + TSAE | Lower | Angle center deviates from $90^\circ$ | Metric error causes evaluation distortion |
| AIP + TSAE (insufficient capacity) | 0.6–0.8 | Unstable | Mismatch between data and capacity |
| AIP + TSAE (matched capacity) | 0.999 | 0.998 | Complete method |

### Key Findings
- Representation drift is a universal phenomenon across LLMs (GPT, Pythia, Llama, Gemma) caused by Softmax translation invariance. Replacing the inner product with AIP centers the angular distribution at $90^\circ$.
- TSAE capacity must strictly match data scale. Blindly increasing SAE size or data volume does not guarantee closer proximity to atoms; they must be scaled proportionally.
- While neurons only satisfy faithfulness and features only satisfy stability, atoms are the first units to exceed $99.7\%$ on both. Higher monorepresentationality correlates with higher semantic monosemanticity (as verified by GPT-5.2 and human checks).

## Highlights & Insights
- Provides the first falsifiable formal definition of fundamental representation units in LLMs, creating an independent anchor for evaluation.
- Bridges sparsity and separation using coherence-RIP from compressed sensing to construct the single-scalar metric $q^*$, applicable to any candidate unit without retraining.
- The TSAE identifiability theorem suggests that SAE success depends on the activation function (hard cutoff) rather than just data volume, offering a practical engineering path: replacing ReLU with JumpReLU or TopK.

## Limitations & Future Work
- Experiments were confined to the residual stream. Whether attention heads, MLP internal activations, or diffusion models exhibit similar atomic structures remains a conjecture.
- The cost of grid searching data scale vs. TSAE capacity is high; this was only fully explored on Gemma2-2B Layer 1.
- Monosemanticity evaluation still relies on LLM/human labeling; causal intervention evidence linking specific atoms to specific semantics is not yet fully established.
- Computing $\tilde{\bm{S}}=(\bm{D}\bm{D}^\top)^{-1}$ for large $H$ is expensive.

## Related Work & Insights
- **vs. Cunningham et al. / Anthropic Templeton SAEs**: Standard SAEs use ReLU + L1, resulting in fuzzy definitions and large residuals. This paper proves that threshold activations can recover atoms.
- **vs. Park et al. (Causal Inner Product)**: Causal IP is defined on static unembedding spaces. AIP is defined on dynamic, input-dependent hidden representation spaces.
- **vs. Bussmann et al. / Chanin et al. (Feature Splitting)**: While prior work treats feature instability as an empirical phenomenon, this paper provides a computable stability metric $q^*$ to optimize against.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Moves mechanistic interpretability from "finding features" to "proving features."
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage across models, though grid search was limited in scope.
- Writing Quality: ⭐⭐⭐⭐ Logical progression through theorems and proofs; math is dense but rigorous.
- Value: ⭐⭐⭐⭐⭐ Provides a concrete upgrade path for SAE research and a formal infrastructure for interpretability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Scalable Circuit Learning for Interpreting Large Language Models](scalable_circuit_learning_for_interpreting_large_language_models.md)
- [\[ACL 2026\] Knowledge Vector of Logical Reasoning in Large Language Models](../../ACL2026/interpretability/knowledge_vector_of_logical_reasoning_in_large_language_models.md)
- [\[ICLR 2026\] Spilled Energy in Large Language Models](../../ICLR2026/interpretability/spilled_energy_in_large_language_models.md)
- [\[ACL 2026\] Compositional Steering of Large Language Models with Steering Tokens](../../ACL2026/interpretability/compositional_steering_of_large_language_models_with_steering_tokens.md)
- [\[ACL 2026\] Tracing Relational Knowledge Recall in Large Language Models](../../ACL2026/interpretability/tracing_relational_knowledge_recall_in_large_language_models.md)

</div>

<!-- RELATED:END -->
