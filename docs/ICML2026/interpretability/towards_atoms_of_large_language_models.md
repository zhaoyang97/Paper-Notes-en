---
title: >-
  [Paper Note] Towards Atoms of Large Language Models
description: >-
  [ICML2026][Interpretability][Atom theory] The paper provides the first formal definition of "fundamental representational units" in Large Language Models—atoms. It characterizes the intrinsic geometry of LLM hidden repre…
tags:
  - "ICML2026"
  - "Interpretability"
  - "Atom theory"
  - "Sparse Autoencoders"
  - "Representational geometry"
  - "Monosemanticity"
  - "Fundamental units of representation"
date: 2026-05-08
content_hash: ce3fb2ad4711faa7
---

# Towards Atoms of Large Language Models

**Conference**: ICML2026  
**arXiv**: [2509.20784](https://arxiv.org/abs/2509.20784)  
**Code**: https://github.com/ChenhuiHu/towards_atoms  
**Area**: Interpretability / Mechanistic Interpretability  
**Keywords**: Atom theory, Sparse Autoencoders, Representational geometry, Monosemanticity, Fundamental units of representation  

## TL;DR
The paper provides the first formal definition of "fundamental representational units" in Large Language Models—atoms. It characterizes the intrinsic geometry of LLM hidden representations using a non-Euclidean "Atomic Inner Product." The authors prove that threshold-activated SAEs can precisely recover the set of atoms under appropriate conditions and demonstrate near-ideal atoms on Gemma2 / Llama3.1 with $R^2 \approx 99.9\%$ and stability $q^* \approx 99.85\%$.

## Background & Motivation
**Background**: Decomposing the internal computations of LLMs into "interpretable units" is central to mechanistic interpretability. Early work treated neurons (single-dimension activations) as basic units, while recent mainstream research has shifted toward "features" learned by SAEs—decomposing the residual stream sparsely into a set of "dictionary directions" and assigning semantic labels via LLM-as-Judge.

**Limitations of Prior Work**: Neurons are contaminated by polysemanticity, with activation patterns spanning unrelated concepts. Features face two persistent issues: large reconstruction residuals ("dark matter") and instability (splitting/merging) when training scale or sparse regularization changes, affecting both the number and the identity of the directions. From an evaluation perspective, no one can answer whether a learned feature counts as a fundamental unit of an LLM because "fundamental units" lack a formal definition.

**Key Challenge**: All existing evaluations implicitly couple "faithfulness" and "monosemanticity" into the SAE training objective, leading to a circular argument where SAE outputs are evaluated using the SAE's own loss. To break this cycle, one must define the properties of "ideal atoms" independently of any specific architecture based on the geometry of the representation space.

**Goal**: (i) Define the fundamental units of LLM representations (atoms); (ii) design computable evaluation metrics (faithfulness $R^2$, stability $q^*$) to independently measure any candidate unit; (iii) provide a practical algorithm theoretically capable of recovering the set of atoms.

**Key Insight**: The authors observe that the LLM training objective only sees $\bm{h}^L$ through Softmax, meaning representations are identified only up to an invertible linear transformation $\bm{A}$. The Euclidean inner product is not invariant under this equivalence class; thus, Euclidean geometry is not the "correct" geometry for LLM representations. Only by changing the metric to $\bm{S}$ can concepts like "orthogonality" and "angle" truly align with model behavior.

**Core Idea**: Use the "Atomic Inner Product" induced by $\bm{S}=(\bm{D}\bm{D}^\top)^{-1}$ as the intrinsic metric for LLM representations. Under this metric, redefine atoms as satisfying representability, sparsity, and separability, and prove that Threshold-activated SAEs (TSAE) can strictly recover the atom set when $\delta_{\min} > \varepsilon(2K-1)\delta_{\max}$.

## Method

### Overall Architecture
The paper presents a four-stage theoretical framework (Geometry $\rightarrow$ Definition $\rightarrow$ Evaluation $\rightarrow$ Identification) paired with systematic experiments:

1. Input: Activation set $M=\{\bm{m}_i\}\subset\mathbb{R}^H$ from any layer of an LLM.
2. Geometric Correction: Estimate $\tilde{\bm{S}}=(\mathbb{E}[\bm{k}\bm{k}^\top])^{-1}$ from 100K Wikipedia activations and define the normalized Atomic Inner Product $\langle\bm{u},\bm{v}\rangle_{\tilde S}=\bm{u}^\top\tilde{\bm{S}}\bm{v}/(\|\bm{u}\|_{\tilde S}\|\bm{v}\|_{\tilde S})$.
3. Define Atoms: $\bm{m}_i=\bm{D}\bm{\delta}_i$, where $\|\bm{\delta}_i\|_0 \le K$ and $|\langle\tilde{\bm{d}}_i,\tilde{\bm{d}}_j\rangle| \le \epsilon$ for $i \ne j$.
4. Dual Metric Evaluation: Faithfulness $R^2$ (reconstruction residual ratio) + Stability $q^*$ (maximum quantile satisfying $\mu_q < \frac{1}{2K_q-1}$).
5. Identification Algorithm: Use Threshold-activated SAE (TSAE) with $\bm{W}_{enc}=\bm{D}^\top\tilde{\bm{S}}$ and $\bm{W}_{dec}=\bm{D}$ to decouple atoms and coefficients.
6. Experimental Loop: Scan data scale $|M|$ vs. TSAE capacity $|D|$ on Gemma2-2B / 9B and Llama3.1-8B to identify stable atoms and evaluate monosemanticity using GPT-5.2.

### Key Designs

1. **Atomic Inner Product (AIP) and Geometric Interpretation of Representation Drift**:
    - **Function**: Replaces the Euclidean inner product with an inner product invariant under parameterized equivalence like $\bm{h}^L \leftarrow \bm{A}\bm{h}^L$, making "approximate orthogonality" an intrinsic model attribute.
    - **Mechanism**: Starting from translation invariance and unit norm symmetry, it can be proven that $\bm{S}=c^2(\bm{D}\bm{D}^\top)^{-1}$. Normalization yields the kernel $\tilde{\bm{S}}=(\bm{D}\bm{D}^\top)^{-1}$. Equivalently whitening activations as $\tilde{\bm{d}}_i=\tilde{\bm{S}}^{1/2}\bm{d}_i$ reduces the AIP to a Euclidean inner product. This step provides intuitive visualization: Euclidean angular distributions in LLM layers deviate from $90^\circ$, indicating activations are pulled toward a common direction by Softmax; using AIP centers the angles back at $90^\circ$, eliminating global bias.
    - **Design Motivation**: The authors strictly derive the uniqueness of using $(\bm{D}\bm{D}^\top)^{-1}$ to prevent subsequent evaluations from being skewed by arbitrary metric choices.

2. **Three Properties of Atoms + Sparsity-Separability Metric $q^*$**:
    - **Function**: Translates the abstract "fundamental unit" into verifiable geometric properties and uses a single scalar $q^*$ to measure both sparsity and separation.
    - **Mechanism**: Atoms must satisfy (i) representability $\bm{m}_i=\bm{D}\bm{\delta}_i$, (ii) sparsity $\|\bm{\delta}_i\|_0 \le K$, and (iii) $\epsilon$-orthogonality. Using the Restricted Isometry Property (RIP) from compressed sensing, coherence $\mu:=\max_{i \ne j}|\langle\tilde{\bm{d}}_i,\tilde{\bm{d}}_j\rangle|$ and sparsity $K$ control the RIP constant via $(K-1)\mu < 1$. Utilizing the uniqueness theorem $\mu < \frac{1}{2K-1} \Rightarrow$ unique solution, the "quantile level" $q^*:=\sup\{q \mid \mu_q < \frac{1}{2K_q-1}\}$ characterizes the proportion of activations where the decomposition is uniquely recoverable (monorepresentationality).
    - **Design Motivation**: Monosemanticity (semantic uniqueness) is hard to formalize; monorepresentationality (structural uniqueness) can be mathematically locked. The authors prove the latter is a necessary condition for the former, elevating interpretability evaluation from "human labeling" to "provable and computable."

3. **Identifiability Theorem for Threshold-activated SAE (TSAE)**:
    - **Function**: Provides an SAE parameterization scheme that can exactly recover the set of atoms if they exist.
    - **Mechanism**: Using threshold activation $\sigma_\tau(x)=x\cdot\mathbb{1}[x \ge \tau]$, if non-zero coefficients satisfy $\delta_{\min} \le \delta_{ij} \le \delta_{\max}$ and $\varepsilon K\delta_{\max} < \tau < \delta_{\min} - \varepsilon(K-1)\delta_{\max}$ (feasibility condition $\delta_{\min} > \varepsilon(2K-1)\delta_{\max}$), setting $\bm{W}_{dec}=\bm{D}$ and $\bm{W}_{enc}=\bm{D}^\top\tilde{\bm{S}}$ guarantees $\bm{W}_{dec}\sigma_\tau(\bm{W}_{enc}\bm{m}_i)=\bm{m}_i$ for all $i$. Implementation-wise, JumpReLU satisfies the threshold requirement and supports coordinate-level threshold vectors $\bm{\tau}$.
    - **Design Motivation**: Previous ReLU SAEs lacked a hard cutoff; orthogonal approximations and noise allowed small activations in "non-support" dimensions to leak, breaking sparse decomposition uniqueness. This work reattributes SAE failures—not to the SAE paradigm itself, but to the choice of activation function.

### Loss & Training
TSAE is trained using JumpReLU with $4\times$ over-parameterization (capacity $|D|=4H$). In Section 4.3, the authors perform a grid search on 1.9B activations for Gemma2-2B Layer 1: scanning $|M| \times |D|$ reveals that faithfulness $R^2$ only "jumps" to $R^2 \approx 1$ once $|D|$ exceeds a critical value matching data scale $|M|$, otherwise remaining at 0.6–0.8. $R^2$ serves as an indirect signal for whether identifiability is triggered—stable reconstruction is only possible when RIP conditions are met.

## Key Experimental Results

### Main Results

| Model | Layer | Faithfulness $R^2$ | Stability $q^*$ | Gap vs. Neurons / Features |
|------|------|------|------|------|
| Gemma2-2B | 1–26 | 99.92% | 99.74% | Features $R^2$=48.8% / $q^*$=68.2% |
| Gemma2-9B | 1–42 | 99.93% | 99.87% | Neurons $R^2$=100% / $q^*$=0.5% |
| Llama3.1-8B | 1–30 | 99.85% | 99.95% | Both metrics near ideal (1, 1) |

Across models of different scales, TSAE-learned units consistently achieve dual metrics near 1, reaching "ideal atoms" in a statistical sense.

### Ablation Study

| Configuration | $R^2$ | $q^*$ | Description |
|------|------|------|------|
| Neurons (baseline) | 1.00 | 0.005 | Fully faithful but highly polysemantic |
| Features (Standard SAE) | 0.488 | 0.682 | Stable but high reconstruction residual |
| Euclidean + TSAE | Low | Angular center deviates | Incorrect metric distorts evaluation |
| AIP + Insufficient TSAE Capacity | 0.6–0.8 | Unstable | Data/Capacity mismatch |
| AIP + Matched TSAE Capacity | 0.999 | 0.998 | Proposed method |

### Key Findings
- Representation drift is a universal phenomenon across GPT / Pythia / Llama / Gemma, fundamentally caused by Softmax translation invariance pulling representations toward a common direction. Switching to AIP is required to return the angular center to $90^\circ$, strongly suggesting AIP is the "correct geometry."
- TSAE capacity thresholds strictly match data scale—blindly increasing SAE size or data volume does not bring one closer to atoms; they must be matched. This aligns poorly with the current mainstream "large corpus + small SAE" recipe.
- Neurons only satisfy faithfulness, features only marginally satisfy stability, whereas atoms are the first to hit $\ge 99.7\%$ on both metrics simultaneously. Monosemanticity scores verified by GPT-5.2 + humans are significantly higher than baselines, proving monorepresentationality drives monosemanticity.

## Highlights & Insights
- Provides the first falsifiable formal definition of "basic representational units" in LLMs. While previous SAE papers answered this implicitly, this work makes it explicit, establishing an independent anchor for evaluation.
- Bridges sparsity and separation using coherence-RIP from compressed sensing to construct a single scalar metric $q^*$—evaluable for any candidate unit (neurons, features, atoms) without requiring thresholds or new models.
- The TSAE identifiability theorem reattributes SAE success/failure from "sufficient data supervision" to "activation functions with hard cutoffs," providing an actionable engineering conclusion: replacing ReLU with JumpReLU/TopK typically unlocks higher $R^2$.

## Limitations & Future Work
- Experiments focus solely on the LLM residual stream; whether attention heads, MLP intermediate activations, or diffusion model representations share atomic structures remains a conjecture.
- The cost of grid searching data scale vs. TSAE capacity is extremely high; the 1.9B activation scan was only performed for Gemma2-2B Layer 1. Systematic empirical evidence for "atom scale functions" across layers/models is needed.
- Semantic evaluation via GPT-5.2 + human verification remains statistical and does not provide causal intervention evidence (e.g., "atom $X$ = semantic $Y$"), leaving a gap to circuit-level explanations.
- Practically, inverting $\bm{S}=(\bm{D}\bm{D}^\top)^{-1}$ for large $H$ is expensive. The paper estimates this with $(\mathbb{E}[\bm{k}\bm{k}^\top])^{-1}$, though stability under finite samples was not discussed separately.

## Related Work & Insights
- **vs. Cunningham et al. SAE / Anthropic Templeton SAE**: Original SAEs use ReLU + L1 with vague definitions and high residuals. This paper proves threshold activations recover atoms, providing a cheap upgrade path for existing SAE pipelines.
- **vs. Park et al. (Causal Inner Product)**: Causal IP is defined on static unembedding space for output tokens; AIP is defined on dynamic, input-dependent hidden representational spaces, allowing more direct analysis of internal representations.
- **vs. Bussmann et al. / Chanin et al. on Feature Splitting**: Those works treat instability as an empirical phenomenon of SAE training; this paper provides $q^*$ as a computable stability metric, turning "instability" from a description into an optimizable objective.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Moves mechanistic interpretability from "finding features" to "proving features" with a complete theoretical framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive across multiple models, but grid search was limited to one layer; diffusion models remain a conjecture.
- Writing Quality: ⭐⭐⭐⭐ Clear progression from theorem to remark with full proofs in the appendix; math density in the main text is high.
- Value: ⭐⭐⭐⭐⭐ Offers a specific upgrade path for the SAE community (activation change + capacity matching) and foundational formal infrastructure for interpretability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Sparse Feature Coactivation Reveals Causal Semantic Modules in Large Language Models](../../ACL2026/interpretability/sparse_feature_coactivation_reveals_causal_semantic_modules_in_large_language_mo.md)
- [\[ACL 2026\] Compositional Steering of Large Language Models with Steering Tokens](../../ACL2026/interpretability/compositional_steering_of_large_language_models_with_steering_tokens.md)
- [\[ACL 2026\] Knowledge Vector of Logical Reasoning in Large Language Models](../../ACL2026/interpretability/knowledge_vector_of_logical_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Tracing Relational Knowledge Recall in Large Language Models](../../ACL2026/interpretability/tracing_relational_knowledge_recall_in_large_language_models.md)
- [\[NeurIPS 2025\] Table as a Modality for Large Language Models](../../NeurIPS2025/interpretability/table_as_a_modality_for_large_language_models.md)

</div>

<!-- RELATED:END -->
