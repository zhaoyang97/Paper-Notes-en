---
title: >-
  [Paper Note] Formalizing the Binding Problem
description: >-
  [ICML 2026][Interpretability][Binding Problem] This paper formalizes the "binding problem in neural networks" as the mutual information $I(O…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Binding Problem"
  - "Information-theoretic Probes"
  - "ViT Representations"
  - "`[CLS]` Token"
  - "Spatial Tokens"
date: 2026-05-08
content_hash: 963f5091d0e87cc5
---

# Formalizing the Binding Problem

**Conference**: ICML 2026  
**arXiv**: [2606.03976](https://arxiv.org/abs/2606.03976)  
**Code**: https://github.com/KordingLab/formalizing-the-binding-problem  
**Area**: Interpretability / Representation Analysis / Vision Transformers  
**Keywords**: Binding Problem, Information-theoretic Probes, ViT Representations, `[CLS]` Token, Spatial Tokens

## TL;DR
This paper formalizes the "binding problem in neural networks" as the mutual information $I(O;Z)$ regarding object codes $O$ within the representation $Z$. By designing autoregressive probabilistic probes to measure binding information in ViTs such as DINOv2 and CLIP, the study finds that the `[CLS]` token encodes <50% of binding information with an approximately quadratic structure, whereas an attention probe on the full set of spatial tokens recovers ~92% of the information.

## Background & Motivation

**Background**: Modern vision and vision-language models rely heavily on ViTs to encode images into a `[CLS]` summary token or a set of spatial tokens. These are subsequently used for contrastive learning (CLIP / SimCLR / Barlow Twins) or cross-modal concatenation (LLaVA / Qwen-VL). Existing work has only demonstrated that ViTs "know which patches belong to the same object" without answering how much information regarding "which features belong to the same object" actually exists in the representation.

**Limitations of Prior Work**: VLMs frequently suffer from "attribute binding errors" in scenes with multiple objects, shared features, or occlusions—e.g., describing a blue hat and a red backpack as a "red hat." While works like Campbell et al. 2025 demonstrate numerous task-level failures, they lack a **quantifiable, cross-model comparable** metric for binding capability and fail to locate whether binding information is lost in the `[CLS]` token or the spatial tokens.

**Key Challenge**: The binding problem is both a cognitive science concept and an engineering problem, yet neither field has provided a formal definition for "binding information content in representations." Simply measuring downstream task accuracy conflates "the model does not know" with "the model knows but the readout failed," failing to isolate the information bottleneck of the representation itself.

**Goal**: (1) Provide a formal definition of binding information independent of specific encoding methods; (2) Provide a probe estimator applicable to any pretrained representation; (3) Use it to dissect the binding capabilities of different ViT components across various datasets.

**Key Insight**: The authors model "feature existence" and "object existence" as binary random vectors $F$ and $O$, respectively. Using mutual information, they compress "binding = information about object codes in the representation" into a single scalar. They leverage the classic inequality "cross-entropy of a probe = conditional entropy + KL" (Lemma 2.21) to estimate the otherwise incalculable $H(O|Z)$ using the test loss upper bound of a trained probe.

**Core Idea**: Use information theory to decompose binding information into "dataset prior $H(O)$ - residual probe uncertainty $H(O|Z)$." An autoregressive decomposition $q_\theta(o|z)=\prod_k q_\theta(o_k|o_{<k},z)$ is used to bypass the combinatorial explosion of object codes, yielding a comparable binding information scalar for any ViT.

## Method

### Overall Architecture
Input: A pretrained ViT $\Phi: X \to Z$ and a scene dataset providing ground-truth feature codes $F \in \{0,1\}^n$ and object codes $O \in \{0,1\}^K$ (either synthetic or densely annotated real data).

Output: Four scalar metrics—Binding Information $B_O(Z) = I(O;Z)$, Feature-Conditional Binding Information $B^*_{O,F}(Z) = I(O;Z|F)$, and their normalized versions $\beta_O(Z)$ and $\beta^*_{O,F}(Z)$ obtained by dividing by the priors $H(O)$ and $H(O|F)$.

Pipeline: (1) Calculate dataset priors $H(O)$ and $H(F)$ via combinatorics or empirical distribution; (2) Train autoregressive binding probes $q_\theta(o_k|o_{<k},z)$ (and feature probes $q_\theta(f_k|f_{<k},z)$) on frozen representations $Z$; (3) Use their test set cross-entropy as an upper-bound estimate for $H(O|Z)$ and $H(F|Z)$; (4) Substitute into $I(O;Z)=H(O)-H(O|Z)$ and Theorem 2.14/2.16 to calculate the four metrics.

### Key Designs

1.  **Information-Theoretic Formalization of Binding (4 Variants)**:
    - **Function**: Provide a scalar definition of "binding information in a representation" that is agnostic to the encoding scheme (slot, tensor-product, or capsule).
    - **Mechanism**: Abstract the scene into a "feature set $F$ + object set $O$," where each object $o_i$ corresponds to a subset of $F$ (surjective map), making $F$ a deterministic function of $O$. $B_O(Z):=I(O;Z)$ measures raw binding information. To ensure fair comparison between models with different feature-learning capabilities, a conditional version $B^*_{O,F}(Z):=I(O;Z|F)=I(O;Z)-I(F;Z)$ is introduced (Theorem 2.14). This subtracts information that can be inferred solely from individual features. Finally, normalization by $H(O)$ and $H(O|F)$ yields dimensionless $\beta$ and $\beta^*$, facilitating cross-dataset comparison.
    - **Design Motivation**: Reporting downstream accuracy ignores information potentially discarded by the readout head. Using $I$ reflects soft probability uncertainty and allows the structure of information (linear vs. quadratic vs. high-order) to be revealed by switching probe families. The conditional version prevents misinterpreting "good single-feature learning" as "binding ability."

2.  **Autoregressive Object Code Probes + Probe Family Comparison**:
    - **Function**: Estimate $H(O|Z)$ on single-token representations like `[CLS]` and infer the "algebraic structure of binding info" by comparing the residual loss of different probe families.
    - **Mechanism**: Directly training $q_\theta(o|z)$ is infeasible as the label space $2^K$ is too large ($K=64$ in experiments, exceeding $10^{19}$). The paper uses the chain rule decomposition $q_\theta(o|z)=\prod_{k=1}^K q_\theta(o_k|o_{<k},z)$, where the representation $z$ and revealed codes $o_{<k}$ are concatenated as $[z\|o_{<k}]$ and fed to the probe. Each $o_k$ is supervised via a binary logit $\ell_k(x)$. Probe families include Linear, Quadratic ($\ell_k=x^\top W_k x+b_k$), and a 4-layer GELU DNN (~3M parameters). On synthetic ColorShape data, test losses were $34.2/22.0/20.6$ bits respectively. The narrow gap between Quadratic and DNN, along with a "parameter-shared quadratic probe" $W_k=U_{\text{color}}^\top V_{\text{shape}}$ adding only $+2.4$ bits, proves that binding info in `[CLS]` is **essentially the dot product of color and shape projections**, a form of conjunctive coding.
    - **Design Motivation**: The probe family hierarchy (linear → quadratic → DNN) approximates the $L_{CE}\ge H(O|Z)$ bound. Discrepancies between families reveal the functional complexity required to read the information.

3.  **Simplified Attention Probe on Spatial Tokens**:
    - **Function**: Handle the dimensionality explosion of the full set of ViT spatial tokens $\{s_i\}_{i=1}^N$ and measure their collective binding information.
    - **Mechanism**: Learn a query $q_k=g_k(o_{<k})$ for each object $o_k$ (query only, no key/value projections). Route weighted spatial vectors $\bar s_k=\sum_i a_{k,i} s_i$ via $a_{k,i}=\text{softmax}_i(q_k^\top s_i)$, then pass through a quadratic readout. On ColorShape, the test Error Rate (ER) reaches $96.8\%$ (loss $3.1$ bits), far exceeding the $20.6$ bits of the strongest `[CLS]` DNN probe. Qualitatively, attention weights almost always route to the correct object patches.
    - **Design Motivation**: Binding information is naturally distributed across spatial tokens. Concatenating them for an MLP would lead to parameter explosion and overfitting. A minimal attention mechanism with learned queries allows for cross-sample weight sharing and provides interpretable visualizations via $a_{k,i}$, proving that binding information exists as a spatial index of "which patch contains the object."

### Loss & Training
All probes are trained using binary cross-entropy $L_{CE}(\theta)=\sum_k \mathbb{E}_{(z,o)}[-\log q_\theta(o_k|o_{<k},z)]$. Test loss (not training loss) is reported. To prevent memorization in the massive combinatorial space ($\sim10^{12}$), the training, validation, and test sets use disjoint feature codes $F$ and object codes $O$. DINOv2-Large and CLIP ViT-L/14 (resolutions 224 and 336) were kept frozen while only the probes were trained.

## Key Experimental Results

### Main Results: `[CLS]` vs. Spatial Tokens (ColorShape, $H(O)=39.9$ bits, $H(F)=7.0$ bits)

| Probe Family | $B_O(Z)$ (bits) | $\beta_O(Z)$ | $B^*_{O,F}(Z)$ (bits) | $\beta^*_{O,F}(Z)$ |
| :--- | :--- | :--- | :--- | :--- |
| Linear | 5.7 | 14.3% | 0.3 | 0.8% |
| Quadratic | 17.9 | 44.9% | 12.5 | 37.9% |
| DNN (3M params) | 19.4 | **48.5%** | 13.9 | **42.4%** |
| **Attention + Spatial Tokens** | **36.8** | **92.2%** | **31.0** | **94.1%** |

The attention probe on spatial tokens increases the binding information recovery from 48.5% to 92.2%.

### Ablation Study

| Experiment | Key Finding |
| :--- | :--- |
| ColorShape Complexity (1→7 pairs) | $\beta^*_{O,F}(Z)$ decreases monotonically as the feature space grows, but **not exponentially** (CLIP-L/14 224px). |
| CLEVR Occlusion (Height 0.6→3.2) | $\beta^*_{O,F}(Z)$ rises from 45.0% to 58.7%; binding drops ~3 pp per occlusion level. |
| Parameter-Shared Quadratic Probe | Using $U_{\text{color}}^\top V_{\text{shape}}$ costs only 2.4 bits extra loss $\rightarrow$ `[CLS]` binding is purely quadratic. |
| Visual Genome (Real Data) | DINOv2 / CLIP on VG:Color / VG:TopAttr yield $\beta^*_{O,F}\in[39.9\%, 47.0\%]$, consistent with synthetic data. |
| CLIP 224px $\rightarrow$ 336px | ColorShape 47.7% $\rightarrow$ 56.4%; fine-grained spatial representations significantly aid binding. |

## Key Findings
- **`[CLS]` is a Binding Bottleneck**: The information upper bound is ~48.5%, and even deep DNNs cannot extract significantly more. This information is almost entirely explained by a $\text{color} \otimes \text{shape}$ bilinear form, suggesting the summary token learns "conjunctive statistics" rather than "object slots."
- **Spatial Tokens Suffer Almost No Loss**: Attention probes recover ~92% of information, implying that ViT's binding capacity is preserved via the spatial indexing of patch tokens. Standard contrastive pretraining objectives (using only `[CLS]`) fail to expose this capability.
- **Decomposable Binding Difficulty**: Complexity (feature space size), occlusion, and naturalness are orthogonal dimensions of binding difficulty, all reflected linearly by $\beta^*_{O,F}$.
- **Resolution Outweighs Algorithm**: Scaling CLIP-L/14 from 224 to 336 pixels yields a larger gain in binding than switching between different self-supervised algorithms (e.g., to DINOv2).

## Highlights & Insights
- **Turning Philosophical Concepts into Scalars**: Previous discussions of "binding" were largely conceptual. This work compresses "how much binding info do I have" into a bit-count and provides a reproducible probe pipeline.
- **Probe Hierarchies as "Structural Microscopes"**: The loss gap between linear/quadratic/DNN probes, combined with parameter-sharing variants, directly identifies the algebraic form of information (e.g., bilinear mapping), providing much more insight than simple accuracy metrics.
- **Elegance of the Simplified Attention Probe**: Using a minimal attention mechanism (query-only) ensures that routing is determined by the representation itself, preventing the probe from "inventing" information, while also providing explainable heatmaps.
- **Quantifying the Cost of the `[CLS]` Paradigm**: The metrics provide a clear warning to the community—CLIP-style models discarding patch tokens in favor of `[CLS]` lose nearly half of the available binding information.

## Limitations & Future Work
- **Dependency on Discrete Dictionaries**: Formulas rely on $F\in\{0,1\}^n$ and $O\in\{0,1\}^K$. Continuous features (e.g., velocity) currently require discretization.
- **Retrievable vs. Utilized Information**: A high $\beta$ only indicates that information *can* be extracted by a probe; it does not guarantee that a downstream readout (like an LLM head in LLaVA) actually utilizes it.
- **Assumption for $B^*_{O,F}$**: It assumes $F$ is a deterministic function of $O$, which may not hold in extremely noisy or biologically realistic scenarios.
- **Future Directions**: Using probe loss as a training objective (binding-aware pretraining); extending metrics to temporal binding in video; studying if structural information predicts compositional generalization success.

## Related Work & Insights
- **vs. Campbell et al. 2025**: While they use task error rates to show binding failure, they cannot distinguish between representation failure and readout failure. This work isolates the representation layer.
- **vs. Greff et al. 2020**: Greff provided a conceptual taxonomy (segregation, representation, composition). This work provides the mathematical and engineering implementation for "representation binding."
- **vs. Slot Attention / Capsule Networks**: While those architectures propose specific encoding schemes, they lack a universal comparison metric. The $\beta^*_{O,F}$ metric provides a common benchmark to compare slot-based vs. vanilla ViT vs. JEPA models.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Represents the first work to strictly align the binding problem with mutual information and provide computable estimators.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers various datasets and architectures, though the object space scale ($K=64$) remains smaller than real-world hierarchies like LVIS.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear logical chain from definitions to theorems to probes and empirical results.
- Value: ⭐⭐⭐⭐⭐ Provides a scalar benchmark that directly informs architectural decisions (e.g., resolution vs. token type) and sets a foundation for future binding-aware pretraining.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Uncovering Grounding IDs: How External Cues Shape Multimodal Binding](../../ICLR2026/interpretability/uncovering_grounding_ids_how_external_cues_shape_multimodal_binding.md)
- [\[ICML 2026\] BLOCK-EM: Preventing Emergent Misalignment via Latent Blocking](block-em_preventing_emergent_misalignment_via_latent_blocking.md)
- [\[ICML 2026\] Verified SHAP: Provable Bounds for Exact Shapley Values in Neural Networks](verified_shap_provable_bounds_for_exact_shapley_values_of_neural_networks.md)
- [\[ICML 2026\] Courtroom Analogy: New Perspective on Uncertainty-Aware Classification](courtroom_analogy_new_perspective_on_uncertainty-aware_classification.md)
- [\[ICML 2026\] Interpretable Self-Supervised Learning via Representer Landmarks and Nyström Approximation](interpretable_self-supervised_learning_via_representer_landmarks_and_nyström_app.md)

</div>

<!-- RELATED:END -->
