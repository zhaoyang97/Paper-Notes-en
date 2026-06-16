---
title: >-
  [Paper Note] Towards Atoms of Large Language Models
description: >-
  [ICML 2026][Interpretability][Paper Note] The paper provides the first formal definition of "fundamental representation units" in Large Language Models (LLMs) — atoms. It characterizes the intrinsic geometry of LLM hidden representations using a non-Euclidean "Atomic Inner Product" and proves that threshold-activated Sparse Autoencoders (SAEs) can precisely re
tags:
  - ICML 2026
  - Interpretability
date: 2026-05-08
content_hash: a7ae3e440c38646f
---
# Towards Atoms of Large Language Models

**Conference**: ICML2026  
**arXiv**: [2509.20784](https://arxiv.org/abs/2509.20784)  
**Code**: https://github.com/ChenhuiHu/towards_atoms  
**Area**: Interpretability / Mechanistic Interpretability  
**Keywords**: Atomic Theory, Sparse Autoencoders, Representational Geometry, Monosemanticity, Fundamental Representational Units  

## TL;DR
The paper provides the first formal definition of "fundamental representation units" in Large Language Models (LLMs) — atoms. It characterizes the intrinsic geometry of LLM hidden representations using a non-Euclidean "Atomic Inner Product" and proves that threshold-activated Sparse Autoencoders (SAEs) can precisely recover the set of atoms under appropriate conditions. Experimental results on Gemma2 and Llama3.1 demonstrate near-ideal atoms with $R^2 \approx 99.9\%$ and stability $q^* \approx 99.85\%$.

## Background & Motivation
**Background**: Decomposing the internal computations of LLMs into "interpretable units" is the core of mechanistic interpretability. Early work treated neurons (single-dimensional activations) as fundamental units, while recent mainstream research has shifted toward "features" learned by SAEs — which sparsely decompose residual flows into a set of "dictionary directions" and assign semantic labels using LLM-as-Judge.

**Limitations of Prior Work**: Neurons are often contaminated by polysemanticity, where activation patterns span unrelated concepts. Features face two persistent issues: large reconstruction residuals ("dark matter") and instability (splitting/merging) when training scale or sparse regularization changes, making both the number of directions and the directions themselves unstable. From an evaluation perspective, no existing metric can determine whether a learned feature qualifies as a fundamental unit of an LLM because there is no formal definition of a "fundamental unit."

**Key Challenge**: All existing evaluations implicitly couple "faithfulness" and "monosemanticity" with the SAE training objective. This results in using the SAE's own loss to evaluate the SAE's output, a form of circular reasoning. To break this cycle, one must define the properties of "ideal atoms" independently of any specific architecture based on the geometry of the representation space, then evaluate whether neurons or features satisfy these properties.

**Goal**: (i) Define the fundamental units (atoms) of LLM representations; (ii) design computable evaluation metrics (faithfulness $R^2$, stability $q^*$) to independently measure any candidate units; (iii) provide a practical algorithm capable of theoretically recovering the set of atoms.

**Key Insight**: The authors observe that since the LLM training objective only perceives $\bm{h}^L$ through Softmax, representations are only identified up to an invertible linear transformation $\bm{A}$. The Euclidean inner product is not invariant under this equivalence class; therefore, Euclidean geometry is not the "correct" geometry for LLM representations. A different metric $\bm{S}$ is required to truly bind concepts like "orthogonality" and "angles" to model behavior.

**Core Idea**: Use the "Atomic Inner Product" (AIP) induced by $\bm{S}=(\bm{D}\bm{D}^\top)^{-1}$ as the intrinsic metric of LLM representations. Redefine atoms within this metric as satisfying (representability, sparsity, separability), and prove that SAEs with threshold activation can strictly recover the atom set when $\delta_{\min}>\varepsilon(2K-1)\delta_{\max}$.

## Method

### Overall Architecture
The paper addresses the question of what the fundamental units of LLM representations are and how to verify them independently. The approach first introduces a correct geometric metric and defines "atoms" as verifiable geometric properties under this metric, followed by an SAE algorithm theoretically capable of recovering these atoms. Specifically, given a set of activations $M=\{\bm{m}_i\}\subset\mathbb{R}^H$ from any LLM layer, the metric matrix $\tilde{\bm{S}}=(\mathbb{E}[\bm{k}\bm{k}^\top])^{-1}$ is estimated from 100K Wikipedia activations to replace the default Euclidean inner product. Faithfulness $R^2$ and stability $q^*$ are then used to measure any candidate units. Finally, a Threshold-activated SAE (TSAE) is used across Gemma2-2B/9B and Llama3.1-8B, scanning "data scale × dictionary capacity" to identify truly stable atoms. This pipeline does not depend on the training loss of any specific SAE, breaking the circular reasoning of evaluating SAEs using their own objectives.

### Key Designs

**1. Atomic Inner Product (AIP): A Correct Geometric Metric for LLM Representations**

**Design Motivation**: The Euclidean inner product is unsuitable for LLM representations because the training objective only sees $\bm{h}^L$ via Softmax. Thus, representations are only identified within an invertible linear transformation $\bm{A}$, and the Euclidean inner product is not invariant to $\bm{h}^L\leftarrow\bm{A}\bm{h}^L$, decoupling "orthogonality/angles" from model behavior. Starting from translation invariance and unit-norm symmetry, the authors rigorously derive the unique metric $\bm{S}=c^2(\bm{D}\bm{D}^\top)^{-1}$. After normalization, this becomes $\tilde{\bm{S}}=(\bm{D}\bm{D}^\top)^{-1}$, defining the Atomic Inner Product as $\langle\bm{u},\bm{v}\rangle_{\tilde S}=\bm{u}^\top\tilde{\bm{S}}\bm{v}/(\|\bm{u}\|_{\tilde S}\|\bm{v}\|_{\tilde S})$. Equivalently, after whitening activations as $\tilde{\bm{d}}_i=\tilde{\bm{S}}^{1/2}\bm{d}_i$, the AIP reduces to the standard Euclidean inner product. This step resolves the issue of "representation drift," where Euclidean angles deviate from $90^\circ$ because Softmax pulls activations toward a common direction. Switching to AIP centers the angles back at $90^\circ$, effectively removing the global bias.

**2. Three Properties of Atoms + Sparsity-Separability Coupling Criterion $q^*$: Defining Units as a Computable Scalar**

The abstract "fundamental unit" is translated into verifiable geometric conditions: an atom must satisfy representability ($\bm{m}_i=\bm{D}\bm{\delta}_i$), sparsity ($\|\bm{\delta}_i\|_0\le K$), and $\epsilon$-approximate orthogonality ($|\langle\tilde{\bm{d}}_i,\tilde{\bm{d}}_j\rangle|\le\epsilon$ for $i\ne j$). The authors use the Restricted Isometry Property (RIP) from compressed sensing to couple sparsity and separability: dictionary coherence $\mu:=\max_{i\ne j}|\langle\tilde{\bm{d}}_i,\tilde{\bm{d}}_j\rangle|$ and sparsity $K$ control the RIP constant. Using the uniqueness theorem $\mu<\frac{1}{2K-1} \Rightarrow$ unique sparse solution, they define the stability index — the quantile level $q^*:=\sup\{q\mid\mu_q<\frac{1}{2K_q-1}\}$. It characterizes the proportion of sparse supports under which atomic decomposition is uniquely recoverable (monorepresentationality). Since semantic monosemanticity is hard to formalize, the authors use monorepresentationality as a mathematically rigid proxy and necessary condition, allowing neurons, features, and atoms to be directly compared.

**3. Identifiability Theorem of Threshold-activated SAE (TSAE): Attributing SAE Failure to Activation Functions**

The paper proves that by selecting a threshold activation $\sigma_\tau(x)=x\cdot\mathbb{1}[x\ge\tau]$ where non-zero coefficients satisfy $\delta_{\min}\le\delta_{ij}\le\delta_{\max}$ and the threshold falls within $\varepsilon K\delta_{\max}<\tau<\delta_{\min}-\varepsilon(K-1)\delta_{\max}$ (feasible if $\delta_{\min}>\varepsilon(2K-1)\delta_{\max}$), setting $\bm{W}_{dec}=\bm{D}$ and $\bm{W}_{enc}=\bm{D}^\top\tilde{\bm{S}}$ guarantees $\bm{W}_{dec}\sigma_\tau(\bm{W}_{enc}\bm{m}_i)=\bm{m}_i$. This strictly decouples atoms and coefficients. The design insight is that standard ReLU SAEs lack a hard cutoff, allowing noise from approximate orthogonality to leak through "non-support" dimensions, destroying the uniqueness of sparse decomposition. This re-attributes SAE failures: it is not the SAE paradigm that is flawed, but the choice of activation function.

### Loss & Training
TSAE is trained using JumpReLU with 4× over-parameterization (dictionary capacity $|D|=4H$). In Section 4.3, the authors conduct a grid search on the 1st layer of Gemma2-2B with 1.9B activations. By scanning $|M|\times |D|$ with a step of 9216, they found that faithfulness $R^2$ only jumps to $\approx 1$ when dictionary capacity $|D|$ exceeds a critical threshold relative to data scale $|M|$; otherwise, it remains between 0.6–0.8. Thus, $R^2$ serves as an indirect signal for whether identifiability has been triggered.

## Key Experimental Results

### Main Results

| Model | Layers | Faithfulness $R^2$ | Stability $q^*$ | Gap vs. Neurons / Features |
|------|------|------|------|------|
| Gemma2-2B | 1–26 | 99.92% | 99.74% | features $R^2$=48.8% / $q^*$=68.2% |
| Gemma2-9B | 1–42 | 99.93% | 99.87% | Neurons $R^2$=100% / $q^*$=0.5% |
| Llama3.1-8B | 1–30 | 99.85% | 99.95% | Both metrics near ideal (1, 1) |

Across three models of different scales, the units learned by TSAE simultaneously achieve dual metrics near 1, reaching "ideal atoms" in a statistical sense.

### Ablation Study

| Configuration | $R^2$ | $q^*$ | Description |
|------|------|------|------|
| Neurons (baseline) | 1.00 | 0.005 | Fully faithful but highly polysemantic |
| Features (standard SAE) | 0.488 | 0.682 | Stable but high reconstruction error |
| Euclidean + TSAE | Low | Bias | Metric error leads to distorted evaluation |
| AIP + TSAE (Low Capacity) | 0.6–0.8 | Unstable | Data/Capacity mismatch |
| AIP + TSAE (Matched) | 0.999 | 0.998 | Full proposed method |

### Key Findings
- Representation drift is a universal phenomenon across GPT/Pythia/Llama/Gemma models due to the translation invariance of Softmax. Only by switching the inner product to AIP does the angular center return to $90^\circ$, providing strong evidence for AIP as the "correct geometry."
- TSAE capacity thresholds must strictly match data scale — blindly increasing SAE size or data volume does not necessarily lead to atoms; they must be matched. This contradicts the current mainstream recipe of "large pre-training corpus + small SAE."
- Neurons only satisfy faithfulness, features barely satisfy stability, while atoms are the first to achieve $\ge 99.7\%$ on both metrics simultaneously. Monosemanticity scores verified by GPT-5.2 and humans are significantly higher than baselines, proving monorepresentationality drives monosemanticity.

## Highlights & Insights
- Provides the first falsifiable formal definition of "fundamental units" for LLM representations. While previous SAE papers implicitly addressed this, this work makes it explicit, providing an independent anchor for evaluation.
- Bridges sparsity and separation using coherence-RIP from compressed sensing to construct the single scalar $q^*$. This metric requires no threshold selection or new model training and can be calculated for any candidate unit.
- The TSAE identifiability theorem shifts the attribution of SAE success from "sufficient data supervision" to "activation functions with hard cutoffs," providing a practical engineering conclusion for dictionary learning: replacing ReLU with JumpReLU or TopK can unlock much higher $R^2$.

## Limitations & Future Work
- Experiments were conducted only on the residual stream; whether attention heads, MLP intermediate activations, or diffusion model representations share this atomic structure remains a conjecture.
- The cost of the grid search for data scale vs. TSAE capacity is extremely high. The grid was only fully explored for the 1st layer of Gemma2-2B; a systematic empirical "atomic scaling law" across layers and models is still needed.
- Monosemanticity evaluation using GPT-5.2 and human verification remains statistical. The paper does not provide causal intervention evidence showing "one atom = one specific semantic," leaving a gap before circuit-level explanation is achieved.
- For practical use, the inversion of $\tilde{\bm{S}}=(\bm{D}\bm{D}^\top)^{-1}$ is costly for large $H$. The stability of using $(\mathbb{E}[\bm{k}\bm{k}^\top])^{-1}$ as an estimate given finite samples was not discussed in detail.

## Related Work & Insights
- **vs. Cunningham et al. SAE / Anthropic Templeton Sonnet SAE**: Standard SAEs use ReLU + L1, which are ill-defined and have high reconstruction residuals. Ours proves that switching to threshold activation recovers atoms, offering a cheap upgrade path for existing SAE pipelines.
- **vs. Park et al. (Causal Inner Product)**: Causal IP is defined on the static unembedding space for output tokens. Ours (AIP) is defined on the dynamic, input-dependent hidden representation space, making it more direct for internal representation analysis.
- **vs. Bussmann et al. / Chanin et al. (Feature Splitting/Merging)**: While those works treat instability as an empirical observation of SAE training, Ours provides $q^*$ as a computable stability metric, transforming "instability" from a description into an optimizable target.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Advances mechanistic interpretability from "finding features" to "proving atoms."
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage across models and layers, but the detailed grid search was limited to a single layer.
- Writing Quality: ⭐⭐⭐⭐ Logical progression through theorems and corollaries; formulas are dense but justified.
- Value: ⭐⭐⭐⭐⭐ Provides a specific upgrade path for the SAE community and rare formal infrastructure for interpretability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Knowledge Vector of Logical Reasoning in Large Language Models](../../ACL2026/interpretability/knowledge_vector_of_logical_reasoning_in_large_language_models.md)
- [\[CVPR 2026\] Understanding Counting Mechanisms in Large Language and Vision-Language Models](../../CVPR2026/interpretability/understanding_counting_mechanisms_in_large_language_and_vision-language_models.md)
- [\[ACL 2026\] Sparse Feature Coactivation Reveals Causal Semantic Modules in Large Language Models](../../ACL2026/interpretability/sparse_feature_coactivation_reveals_causal_semantic_modules_in_large_language_mo.md)
- [\[ACL 2026\] Compositional Steering of Large Language Models with Steering Tokens](../../ACL2026/interpretability/compositional_steering_of_large_language_models_with_steering_tokens.md)
- [\[ACL 2026\] Tracing Relational Knowledge Recall in Large Language Models](../../ACL2026/interpretability/tracing_relational_knowledge_recall_in_large_language_models.md)

</div>

<!-- RELATED:END -->
