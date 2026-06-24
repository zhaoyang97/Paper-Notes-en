---
title: >-
  [Paper Note] Formalizing the Binding Problem
description: >-
  [ICML 2026][Interpretability][Binding Problem] This paper formalizes the "binding problem in neural networks" as the mutual information $I(O;Z)$ regarding the object code $O$ within the representation $Z$. By designing autoregressive probabilistic probes to measure binding information in ViTs such as DINOv2 and CLIP, the study finds that the `[CLS]` token encodes <50% of binding information with a structure approximating a quadratic form, while an attention probe on the full…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Binding Problem"
  - "Information-theoretic Probing"
  - "ViT Representations"
  - "Spatial tokens"
date: 2026-05-08
content_hash: f0cdc79b1707f813
---

# Formalizing the Binding Problem

**Conference**: ICML 2026  
**arXiv**: [2606.03976](https://arxiv.org/abs/2606.03976)  
**Code**: https://github.com/KordingLab/formalizing-the-binding-problem  
**Area**: Interpretability / Representation Analysis / Vision Transformer  
**Keywords**: Binding Problem, Information-theoretic Probing, ViT Representations, Spatial tokens

## TL;DR
This paper formalizes the "binding problem in neural networks" as the mutual information $I(O;Z)$ regarding the object code $O$ within the representation $Z$. By designing autoregressive probabilistic probes to measure binding information in ViTs such as DINOv2 and CLIP, the study finds that the `[CLS]` token encodes <50% of binding information with a structure approximating a quadratic form, while an attention probe on the full set of spatial tokens recovers ~92% of the binding information.

## Background & Motivation

**Background**: Modern vision and vision-language models rely heavily on ViTs to encode images into a `[CLS]` summary token or a set of spatial tokens, followed by downstream tasks like contrastive learning (CLIP / SimCLR / Barlow Twins) or cross-modal concatenation (LLaVA / Qwen-VL). Prior work has only demonstrated that ViTs "know which patches belong to the same object" without answering exactly how much information about "which features belong to the same object" is present in the model's representation.

**Limitations of Prior Work**: VLMs often suffer from mis-binding in scenarios involving multiple objects, shared features, or occlusions—for instance, describing a blue hat and a red backpack as a "red hat." While works like Campbell et al. 2025 provide numerous task-level failure cases, they lack a **quantifiable and cross-model comparable** metric for binding capability, and cannot pinpoint whether binding information is lost in the `[CLS]` token or the spatial tokens.

**Key Challenge**: The binding problem is both a cognitive science concept and an engineering concept, yet neither field has provided a formal definition of "binding information content in representations." Relying solely on downstream task accuracy conflates "what the model does not know" with "what the model knows but the readout failed to output," failing to isolate the information bottleneck of the representation itself.

**Goal**: (1) Provide a formal definition of binding information independent of specific encoding methods; (2) Provide a probe estimator applicable to any pretrained representation; (3) Use it to dissect the binding capabilities of different ViT components across various datasets.

**Key Insight**: The authors model "feature existence" and "object existence" as binary random vectors $F$ and $O$, respectively. They use mutual information to compress "binding = information about object codes in the representation" into a single scalar. Leveraging the classic inequality that "cross-entropy of a probe = conditional entropy + KL divergence" (Lemma 2.21), they estimate the intractable $H(O|Z)$ using the test loss upper bound of a trained probe.

**Core Idea**: Binding information is decomposed using information theory into "dataset prior $H(O)$ - probe residual uncertainty $H(O|Z)$." An autoregressive decomposition $q_\theta(o|z)=\prod_k q_\theta(o_k|o_{<k},z)$ is employed to bypass the combinatorial explosion of object codes, yielding comparable binding information scalars for any ViT.

## Method

### Overall Architecture
This method answers "how much binding information is hidden in a frozen ViT representation $Z$" by translating "binding" into mutual information. First, scenes are annotated with ground-truth feature codes $F\in\{0,1\}^n$ and object codes $O\in\{0,1\}^K$. Since the test cross-entropy of a probe is an upper bound on conditional entropy, the intractable $H(O|Z)$ is estimated using the residual loss of a trained autoregressive probe.

The process involves four steps: calculate dataset priors $H(O)$ and $H(F)$ via combinatorics or empirical distributions; train object probes $q_\theta(o_k|o_{<k},z)$ and feature probes $q_\theta(f_k|f_{<k},z)$ on frozen representations $Z$; use their test set cross-entropy as upper bounds for $H(O|Z)$ and $H(F|Z)$; and finally substitute these into $I(O;Z)=H(O)-H(O|Z)$ and Theorem 2.16 to produce four scalars—binding information $B_O(Z)=I(O;Z)$, feature-conditioned binding information $B^*_{O,F}(Z)=I(O;Z|F)$, and normalized versions $\beta_O(Z)$ and $\beta^*_{O,F}(Z)$.

### Key Designs

**1. Information-theoretic Formalization of Binding: Compressing Capability into Bits**
Previously, the binding problem lacked a comparable scale independent of encoding schemes. This paper abstracts a scene into a "set of features $F$ + set of objects $O$," where each object $o_i$ corresponds to a subset of $F$ (a surjective map). Thus, $F$ is a deterministic function of $O$, and binding information is defined as $B_O(Z):=I(O;Z)$. To prevent a model that only learns single features well from achieving a high score, the conditional version $B^*_{O,F}(Z):=I(O;Z|F)=I(O;Z)-I(F;Z)$ (Theorem 2.14) is introduced, subtracting information derivable purely from features to isolate true "feature-object associations." These are normalized by $H(O)$ and $H(O|F)$ into dimensionless $\beta$ and $\beta^*$ for fair cross-dataset comparison. Mutual information is used instead of downstream accuracy because accuracy can be throttled by readout heads, whereas $I$ reflects soft probability uncertainty and reveals the algebraic structure (linear/quadratic/high-order) of information via different probe families.

**2. Autoregressive Object Code Probes + Probe Family Ladder: Estimating Information and Structure**
Estimating $H(O|Z)$ on a single token like `[CLS]` by training $q_\theta(o|z)$ is infeasible as the label space $2^K$ reaches $10^{19}$ for $K=64$. The paper bypasses this via chain decomposition: $q_\theta(o|z)=\prod_{1}^K q_\theta(o_k|o_{<k},z)$. The representation $z$ and revealed codes $o_{<k}$ are concatenated as $x=[z\|o_{<k}]$ and fed to the probe, with each $o_k$ supervised by a binary classification logit $\ell_k(x)$. Using a ladder of probe families—linear $\ell_k=W_kx+b_k$, quadratic $\ell_k=x^\top W_k x+b_k$, and 4-layer GELU DNNs—allows for approximating the $L_{CE}\ge H(O|Z)$ bound. On the ColorShape dataset, test losses for the three families were $34.2/22.0/20.6$ bits respectively, showing that quadratic probes nearly match DNNs, suggesting binding information is largely readable by second-order functions. Furthermore, sharing parameters in the quadratic probe as $W_k=U_{\text{color}}^\top V_{\text{shape}}$ only added $2.4$ bits of loss, confirming that `[CLS]` binding information is essentially a bilinear $\text{color}\otimes\text{shape}$ conjunction code.

**3. Simplified Attention Probes on Spatial Tokens: Learning Queries Only**
Binding information in a full set of ViT spatial tokens $\{s_i\}_{i=1}^N$ is naturally distributed. To avoid parameter explosion from MLPs, the paper learns a single query $q_k=g_k(o_{<k})$ for each object $o_k$ without key/value projections. Spatial vectors are routed via $a_{k,i}=\text{softmax}_i(q_k^\top s_i)$ to a weighted average $\bar s_k=\sum_i a_{k,i} s_i$, followed by a quadratic readout. Omitting key/value projections ensures routing is determined solely by the representation, preventing the probe from "inventing" information. On ColorShape, the test ER reached $96.8\%$ (loss of $3.1$ bits), far outperforming the `[CLS]` DNN probe (20.6 bits). Visualization reveals that attention almost always routes to patches containing the target object, proving binding information exists as spatial indices in patch tokens.

### Loss & Training
All probes are trained using binary cross-entropy $L_{CE}(\theta)=\sum_k \mathbb{E}_{(z,o)}[-\log q_\theta(o_k|o_{<k},z)]$. Feature probes are symmetric. Test loss is reported to ensure generalization, using disjoint sets of feature codes $F$ and object codes $O$ across training, validation, and test splits to prevent memorization. DINOv2-Large and CLIP ViT-L/14 (224 and 336 resolutions) are kept frozen.

## Key Experimental Results

### Main Results: `[CLS]` Single Token Binding Information (ColorShape, $H(O)=39.9$ bits, $H(F)=7.0$ bits)

| Probe Family | $B_O(Z)$ (bits) | $\beta_O(Z)$ | $B^*_{O,F}(Z)$ (bits) | $\beta^*_{O,F}(Z)$ |
| :--- | :--- | :--- | :--- | :--- |
| Linear | 5.7 | 14.3% | 0.3 | 0.8% |
| Quadratic | 17.9 | 44.9% | 12.5 | 37.9% |
| DNN (3M params) | 19.4 | **48.5%** | 13.9 | **42.4%** |
| **Attention + Spatial Tokens** | **36.8** | **92.2%** | **31.0** | **94.1%** |

The last row indicates that switching from `[CLS]` to spatial tokens with an attention probe increases binding information from 48.5% to 92.2%.

### Ablation Study: Complexity, Occlusion, and Natural Datasets

| Experiment | Key Finding |
| :--- | :--- |
| ColorShape combinations 1→7 (49 total) | $\beta^*_{O,F}(Z)$ decreases monotonically but **not exponentially** with feature space growth. |
| CLEVR Occlusion (Camera height 0.6→3.2) | $\beta^*_{O,F}(Z)$ rises from 45.0% to 58.7%; each occlusion level drops info by ~3 pp. |
| Parameter-shared vs. Standard Quadratic | Sharing $U_{\text{color}}^\top V_{\text{shape}}$ adds only 2.4 bits loss → `[CLS]` binding is purely quadratic. |
| Visual Genome (Natural Data) | DINOv2 / CLIP on VG:Color / VG:TopAttr show $\beta^*_{O,F}\in[39.9\%,47.0\%]$, similar to synthetic. |
| CLIP 224px → 336px | ColorShape 47.7%→56.4%; fine-grained spatial representation aids binding. |

### Key Findings
- **`[CLS]` is a binding bottleneck**: Its information upper bound is ~48.5%, even with DNN probes. This information is almost entirely explained by a bilinear form, suggesting `[CLS]` tokens learn "conjunction statistics" rather than "object slots."
- **Spatial tokens preserve nearly all information**: Attention probes recover ~92% of binding information, indicating that ViT binding capacity is stored via spatial indexing in patch tokens, which is often unexposed by contrastive objectives using only `[CLS]`.
- **Binding difficulty is decomposable**: Complexity, occlusion, and naturalness are orthogonal dimensions reflected consistently by $\beta^*_{O,F}$.
- **Resolution outweighs model architecture**: Increasing CLIP-L/14 resolution from 224 to 336 provides a larger gain than switching between different self-supervised algorithms.

## Highlights & Insights
- **Quantifying a philosophical concept**: This work transforms "binding" from conceptual discourse into a measurable bit-count with a reproducible pipeline.
- **Probe ladders as an "information microscope"**: The loss gap between linear/quadratic/DNN probes, combined with parameter-sharing variants, directly identifies the algebraic structure of information (e.g., bilinear conjunctions).
- **Elegant utility of simplified attention**: Queries-only attention ensures routing is derived from the representation itself, providing both a "pure" measurement and interpretable visualization.
- **Quantifying the cost of the `[CLS]`-only path**: The result demonstrates that CLIP-style models discard half of the available binding information by relying on `[CLS]` for downstream tasks, justifying the move toward patch-token-based VLMs.

## Limitations & Future Work
- **Reliance on predefined discrete feature/object dictionaries**: The framework assumes $F\in\{0,1\}^n$ and $O\in\{0,1\}^K$. Future work is needed for continuous features (e.g., velocity).
- **Measurable vs. Utilized information**: A high $\beta$ indicates information is "extractable" but does not guarantee the downstream readout (e.g., LLM in LLaVA) actually utilizes it.
- **Conditional Assumptions**: The assumption that $F$ is derivable from $O$ may falter in biological vision or extreme noise, potentially distorting the decomposition in Theorem 2.14.
- **Scale of Objects**: Experiments were conducted with $K \approx 64$. Whether quadratic probes remain as effective as DNNs for thousands of classes (e.g., LVIS) remains an open question.

## Related Work & Insights
- **vs. Campbell et al. 2025 / Zhang et al. 2024**: These use VLM task error rates to show binding failure but cannot distinguish if the failure occurs in the representation or the readout. This study localizes the diagnosis to the representation layer.
- **vs. Greff et al. 2020**: This extends the conceptual classification of "representation binding" into an engineering-ready metric.
- **vs. Slot Attention / Capsules**: While these propose new architectures for binding, they lack a unified metric. $\beta^*_{O,F}$ provides a common benchmark for fair comparison across these specialized architectures.
- **vs. BERT linguistic probes**: Traditional probing measures classification performance; this work frames probe loss as an information-theoretic upper bound, elevating probing from "performance testing" to "information estimation."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first work to strictly align the binding problem with mutual information and provide a computable estimate.
- Experimental Thoroughness: ⭐⭐⭐⭐ Coverage of synthetic/CLEVR/VG data and DINO/CLIP architectures is strong, though object space scale could be larger.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear chain of definitions, theorems, and experiments.
- Value: ⭐⭐⭐⭐⭐ Provides a scalar benchmark that directly informs decisions such as "token selection" and "resolution settings."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Tackling the XAI Disagreement Problem with Adaptive Feature Grouping](../../ICLR2026/interpretability/tackling_the_xai_disagreement_problem_with_adaptive_feature_grouping.md)
- [\[AAAI 2026\] Attention as Binding: A Vector-Symbolic Perspective on Transformer Reasoning](../../AAAI2026/interpretability/attention_as_binding_a_vector-symbolic_perspective_on_transformer_reasoning.md)
- [\[ICML 2026\] Formal Concept Lattices are Good Semantic Scaffolds for Concept-Based Learning](formal_concept_lattices_are_good_semantic_scaffolds_for_concept-based_learning.md)
- [\[ICML 2026\] BLOCK-EM: Preventing Emergent Misalignment via Latent Blocking](block-em_preventing_emergent_misalignment_via_latent_blocking.md)
- [\[ICML 2026\] Verified SHAP: Provable Bounds for Precise Shapley Values in Neural Networks](verified_shap_provable_bounds_for_exact_shapley_values_of_neural_networks.md)

</div>

<!-- RELATED:END -->
