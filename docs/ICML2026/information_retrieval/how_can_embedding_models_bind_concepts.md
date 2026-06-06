---
title: >-
  [Paper Note] How can embedding models bind concepts?
description: >-
  [ICML2026][Information Retrieval & RAG][concept binding] This paper formalizes the problem of why embedding models fail to bind concepts as a "complexity problem of the binding function." Through geometric analysis…
tags:
  - "ICML2026"
  - "Information Retrieval & RAG"
  - "concept binding"
  - "CLIP"
  - "embedding geometry"
  - "compositional generalization"
  - "multimodal alignment"
date: 2026-05-08
content_hash: 37ce32d0f3784919
---

# How can embedding models bind concepts?

**Conference**: ICML2026  
**arXiv**: [2605.31503](https://arxiv.org/abs/2605.31503)  
**Code**: Available (Public repository at the end of the paper)  
**Area**: interpretability  
**Keywords**: concept binding, CLIP, embedding geometry, compositional generalization, multimodal alignment

## TL;DR
This paper formalizes the problem of why embedding models fail to bind concepts as a "complexity problem of the binding function." Through geometric analysis, it demonstrates that CLIP's scene embeddings decompose additively into the sum of objects and concepts (explaining why they are detectable unimodally but fail cross-modally). Furthermore, using controlled Transformers, it proves that when data coverage is sufficient, models learn a low-complexity binding dominated by **multiplicative interactions** between concepts, enabling systematic generalization to unseen object combinations.

## Background & Motivation
**Background**: Dual-encoder vision-language embedding models like CLIP exhibit "bag-of-concepts" behavior in cross-modal retrieval: they can recognize individual concepts such as "red" or "cube" but fail to distinguish "red cube + blue ball" from "blue cube + red ball" in multi-object scenes. This is the classic concept binding failure. Prior work has repeatedly observed this phenomenon, attributing it to insufficient encoder granularity, weak negation/spatial reasoning, or an inherent trade-off between concepts and objects.

**Limitations of Prior Work**: Previous explanations remained at the "behavioral level"—observing CLIP's incorrect answers and inferring a lack of capability. However, they failed to explain a paradox: within a single modality (training probes using only the image or text encoder), CLIP can actually recover object-level information. How can "failure to bind cross-modally" and "ability to decode objects unimodally" coexist in the same vector?

**Key Challenge**: Cross-modal alignment requires the image encoder $B_{\text{img}}$ and text encoder $B_{\text{txt}}$ to produce comparable embeddings for the same scene. As long as the binding rules learned by each differ, mismatch occurs for unseen object combinations. Thus, the problem shifts from "can it recognize objects" to "whether the concept-to-object mappings learned by both follow the same simple rule."

**Goal**: (1) Characterize the structure of CLIP's multi-object scene embeddings; (2) Measure the complexity of its implicitly implemented binding function; (3) Verify whether binding can be learned in controlled Transformers and identify its structural form.

**Key Insight**: Binding is defined as a function $B:\mathcal{S}\to\mathbb{R}^d$ (mapping scenes to embeddings). Drawing from MDL/Occam’s principle, if both encoders learn low-complexity, compositional rules for $B$, they are more likely to converge to the same rule and align on OOD data. High complexity leads to memorization of the training distribution, resulting in OOD failure.

**Core Idea**: "CLIP's failure to bind" is not a structural defect but rather a result of the high complexity of its learned binding function. Given sufficient data coverage, dual-encoder Transformers can learn a low-complexity binding implemented via **multiplicative interactions**, thereby achieving cross-modal alignment for unseen combinations.

## Method
This paper is not a "new model" but a "formal framework + a set of geometric/capacity diagnostic experiments." The pipeline consists of three parts: establishing a falsifiable mathematical definition of binding, dissecting CLIP's failure modes using this definition, and training a controlled Transformer on synthetic data to explain its internal structure.

### Overall Architecture
**Formalization Layer**: Defines the concept space as $\mathcal{C}=\mathcal{C}_1\times\cdots\times\mathcal{C}_C$. An object $\bm{o}=(c_1,\dots,c_C)$ is a tuple of concept values, and a scene $\bm{s}=(\bm{o}_1,\dots,\bm{o}_m)$ is a set of object tuples. A model $(f,q)$ consists of a scene encoder $f$ and a query encoder $q$, scored by cosine similarity. "Binding" is split into two independently measurable abilities: concept recognition (scores of all appearing concept values are higher than non-appearing ones) and object recognition (scores of all appearing complete objects are higher than non-appearing ones). Binding requires both; satisfying only the first is "bag-of-concepts." The binding functions are defined as $B_{\text{img}}(\bm{s}):=f(\bm{x}_{\bm{s}})$ and $B_{\text{txt}}(\bm{s}):=q(\bm{y}_{\bm{s}})$.

**Geometric Diagnosis Layer** (Section 4): For real CLIP models, object embeddings $\bm{u}_{\bm{o}}$ and concept embeddings $\bm{u}_c$ are estimated via subset averaging. The work verifies whether scene embeddings approximate an additive decomposition of objects (Level-I) and whether objects further decompose into a sum of concepts (Level-II). "Object-level editing" experiments are also conducted by performing $\tilde{\bm{z}}=f(\bm{x}_{\bm{s}})-\bm{u}_{\bm{o}_1}+\bm{u}_{\bm{o}_1'}$ directly in embedding space to check the retrieval/probing performance of counterfactual scenes.

**Capacity Diagnosis Layer** (Section 5): A small MLP approximator $g(\bm{o}_1,\bm{o}_2)$ is trained to input discrete concept indices and output predicted scene embeddings by minimizing $\sum_{\bm{s}}\|f(\bm{x}_{\bm{s}})-g(\bm{o}_1,\bm{o}_2)\|^2$. It sweeps widths $\{64, 256, 1024, 4096\}$ and training object coverage $\{0.1, \dots, 0.9\}$, measuring concept/object recognition on held-out objects. Simultaneously, dual-encoder Transformers (~20M parameters, output $\mathbb{R}^{512}$, AdamW + contrastive loss) are trained from scratch on synthetic data, systematically varying $C$, $V$, and coverage to observe when binding generalizes.

### Key Designs

1.  **Two-level Additive Decomposition Hypothesis (Level-I / Level-II)**:
    *   **Function**: Translates the composition of scene embeddings into falsifiable geometric properties, explaining the paradox of unimodal binding with cross-modal failure.
    *   **Mechanism**: Assumes $f(\bm{x}_{\bm{s}})\approx \bm{u}_{\bm{o}_1}+\bm{u}_{\bm{o}_2}\approx \sum_{i}\bm{u}_{c_{1,i}}+\sum_{i}\bm{u}_{c_{2,i}}$. Three estimators obtain $\bm{u}_{\bm{o}}$ (multi-object scene average *avg*, conditional average by object position *avg+pos*, and single-object scene average *single-obj*). Validation uses $R^2$, retrieval accuracy, and linear probe accuracy. A key ablation is "targeted removal": subtracting concept components collapses concept decoding to random but leaves object decoding largely intact; subtracting object components collapses both.
    *   **Design Motivation**: Unimodal probes can recover objects because the object-level component $\bm{u}_{\bm{o}}$ explicitly exists in the embedding, "packaging" the concept combination into a non-additive vector. However, this component is not required to be produced by the same function as the text-side $\bm{u}_{\bm{o}}^{\text{txt}}$, leading to cross-modal mismatch.

2.  **Capacity Diagnosis of Binding Function (MLP / XGBoost / RF Approximators)**:
    *   **Function**: Operationalizes the abstract "simplicity of binding" into "whether it can be fitted by a small approximator on held-out objects."
    *   **Mechanism**: A family of approximators maps discrete concept indices to CLIP scene embeddings. Linear probes then test recognition on the predicted embeddings of held-out objects. Findings show that while concept recognition stabilizes at $\ge 80\%$ with $>0.3$ coverage, object recognition stalls at ~20% even with 4096-width MLPs or different models like XGBoost.
    *   **Design Motivation**: Probing only shows that object information is *present*; it doesn't show if the concept $\to$ object mapping has a simple form. MLP serves as a proxy for SGD's preference for simple solutions; failure to fit a generalizing solution supports the "high complexity binding function" conclusion.

3.  **Multiplicative Interaction Probes (Additive / Per-obj. products / Global product)**:
    *   **Function**: Investigates the functional form of binding in "generalizing" controlled Transformers—whether it is purely additive, intra-object multiplicative, or globally multiplicative.
    *   **Mechanism**: Three structural probes fit the scene embeddings. *Additive* takes the form $\sum_{i=1}^{2}\sum_{k=1}^{2}\bm{u}_{k,c_{ik}}$ (bag-of-concepts baseline); *Per-obj. products* adds intra-object products $\sum_i\prod_k \bm{v}_{i,k,c_{ik}}$; *Global product* adds cross-object products $\prod_i\prod_k \bm{v}_{i,k,c_{ik}}$. Multiplicative terms provide unique vectors for each combination. Fig. 9 across ~500 models shows OOD object recognition correlates strongly with *Global product* probe fit quality.
    *   **Design Motivation**: To explain why controlled models align on unseen combinations, a simple, reusable functional form is needed. Multiplicative structures deviate minimally from additivity but provide the necessary "binding signals" for unique identification.

### Loss & Training
Controlled dual-encoders use CLIP's symmetric contrastive loss with cosine similarity; AdamW optimizer; ~20M parameters per encoder; output dimension $d=512$. Diagnostic MLPs use MSE ($\ell_2$) regression. Training uses synthetic multi-object data (CLEVR, CLEVR-2D, PUG:SPARE, and natural images from Gemini Nano Banana 2), sweeping generalization curves via $(C,V)$ and training object coverage $\rho_{\text{train}}\in[0.1, 0.9]$.

## Key Experimental Results

### Main Results
| Dataset | Model | $R^2$ (avg / avg+pos) | Retrieval | Probing |
| :--- | :--- | :--- | :--- | :--- |
| Text (Synthetic caption) | CLIP | 0.90 / 0.92 | 0.97 | 0.99 |
| PUG:SPARE | CLIP | 0.75 / 0.84 | 0.93 | 0.98 |
| PUG:SPARE | DINOv2 | 0.78 / 0.86 | 0.86 | 0.98 |
| CLEVR | CLIP | 0.78 / 0.83 | 0.94 | 0.96 |
| Text | Random-init | 0.47 / 0.69 | 0.42 | 0.82 |

CLIP scene embeddings are reconstructed with high quality ($R^2$ 0.75–0.92) from a sum of object components, and reconstructed embeddings maintain retrieval/probing performance close to the original model across CLEVR, occluded scenes, and natural images. This extends Level-I additive decomposition to realistic scenarios.

| Dataset | Model | Probing (avg / avg+pos / single-obj) | Retrieval (avg / avg+pos / single-obj) |
| :--- | :--- | :--- | :--- |
| CLEVR | CLIP | 0.98 / 0.98 / 0.86 | 1.00 / 1.00 / 0.97 |
| CLEVR-2D | CLIP | 0.98 / 0.98 / 0.92 | 0.99 / 0.99 / 0.97 |
| PUG:SPARE | CLIP | 0.94 / 0.95 / – | 0.86 / 0.94 / – |
| PUG:SPARE | DINO | 0.97 / 0.97 / – | 0.48 / 0.76 / – |

Performing "object replacement" $\tilde{\bm{z}}=f(\bm{x}_{\bm{s}})-\bm{u}_{\bm{o}_1}+\bm{u}_{\bm{o}_1'}$ yields embeddings that behave like counterfactual scenes; notably, object embeddings estimated from single-object scenes can edit multi-object scenes, proving object components are "plug-and-play" geometric entities.

### Ablation Study
| Configuration | Text Conc. / Obj. | Image Conc. / Obj. | Explanation |
| :--- | :--- | :--- | :--- |
| CLIP-B/32 Original | 1.00 / 1.00 | 0.94 / 0.96 | Baseline |
| − concept component | 0.06 / 0.99 | 0.05 / 0.85 | Concept decoding collapses, object decoding remains |
| − object component | 0.05 / 0.04 | 0.02 / 0.01 | Both collapse simultaneously |
| permute concept (control) | 0.92 / 0.99 | 0.99 / 0.97 | Removing wrong component has no effect |
| permute object (control) | 0.96 / 1.00 | 0.86 / 0.92 | Removing wrong object has no effect |

This table provides the most critical causal evidence: object-level components carry both "object identity" and "internal concept composition," whereas concept components only carry the concepts themselves.

### Key Findings
*   CLIP's binding failure is not due to "missing information" but "high function complexity": High-capacity MLPs/XGBoosts/RFs are capped at $\le 20\%$ for held-out object recognition (while reaching 80%+ for concepts), suggesting concept $\to$ object mappings are treated effectively as independent memories.
*   Binding generalization exhibits a sharp phase transition regarding data coverage: For $|O|=125,000$, object recognition jumps from random to near-perfect when training coverage increases from 30% to 40%. Larger object spaces require lower relative coverage (~30% for $|O|\ge 2,500$).
*   Generalizing models have binding functions fit by small MLPs, and the fit quality of the *Global product* probe correlates strongly with OOD accuracy across ~500 models—linking "low complexity," "multiplicative structure," and "cross-modal alignment."

## Highlights & Insights
*   Reframing binding as a **function** rather than an "ability" is a significant conceptual upgrade. Once $B_{\text{img}}$ and $B_{\text{txt}}$ are defined, complexity can be discussed via MDL/Occam, providing a mechanistic explanation for cross-modal failure beyond vague "encoder granularity" arguments.
*   the "targeted removal of concept/object components" ablation paradigm is elegant: the causal chain is confirmed by the collapse of specific abilities only when corresponding components are removed, a method transferable to any dual-encoder analysis.
*   Using the **generalization capability** of MLPs/XGBoosts/RFs as a proxy for binding complexity engineeringly addresses the incomputability of Kolmogorov complexity.
*   The discovery of multiplicative interactions explains why simply increasing data and parameters allows compositional generalization to "suddenly" emerge: in large object spaces, models are forced to learn simple rules (expressible by $\prod$) that enable cross-modal alignment on OOD data.

## Limitations & Future Work
*   Conclusions are based on synthetic data (CLEVR, PUG:SPARE, etc.); real-world scenes lack "compositionally complete" datasets to replicate such controlled experiments, and real binding failures might be more complex.
*   "Complexity" depends on the choice of approximators (MLP/XGBoost/RF). Technically, this measures "incompressibility relative to these learners" rather than true Kolmogorov complexity.
*   While multiplicative structure is shown to be effective, the exact internal mechanism of how dual-encoder Transformers implement this (attention vs. token-token products) remains an open question.
*   No immediate training recipe for modifying CLIP is provided; whether "increasing data coverage" or "imposing multiplicative inductive biases" is more practical for industrial models remains to be seen.

## Related Work & Insights
*   **vs. Trager et al. (2023) / Uselis et al. (2025) / Berasi et al. (2025)**: They demonstrated additive decomposition for **single-object** embeddings in CLIP. This work extends this to **multi-object scenes** and adds Level-II decomposition, unifying the geometric narrative.
*   **vs. Feng & Steinhardt (2024) / Feng et al. (2025)**: Found token-level binding ID mechanisms in autoregressive LLMs. This work studies "single-vector scene embeddings" without token intermediaries, discovering a different geometric narrative (additive + multiplicative).
*   **vs. Kang et al. (2025)**: Argued a "trade-off between concept and object recognition." This work refutes this via embedding decomposition—both can coexist; the bottleneck is the complexity of the binding function.
*   **vs. Fine-tuning approaches (Yuksekgonul, Ma, Hsieh, etc.)**: These attempts to make CLIP more "compositional" through data or losses are explained here as efforts to push the binding function toward a low-complexity (multiplicative) form.

## Rating
*   **Novelty**: ⭐⭐⭐⭐⭐ Reframing binding failure as "high function complexity + lack of multiplicative structure" is a fundamental conceptual upgrade.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple datasets, models, and capacity/coverage sweeps with ~500-model correlation curves, though focused on synthetic data.
*   **Writing Quality**: ⭐⭐⭐⭐⭐ Clear conceptual layering, rigorous causal controls, and stable narrative rhythm.
*   **Value**: ⭐⭐⭐⭐ Provides clear structural guidance (coverage + multiplicative bias) for building a CLIP model that truly binds.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Explaining CLIP Zero-shot Predictions Through Concepts](../../CVPR2026/information_retrieval/explaining_clip_zero-shot_predictions_through_concepts.md)
- [\[ACL 2026\] How Large Language Models Balance Internal Knowledge with User and Document Assertions](../../ACL2026/information_retrieval/how_large_language_models_balance_internal_knowledge_with_user_and_document_asse.md)
- [\[AAAI 2026\] Positional Bias in Multimodal Embedding Models: Do They Favor the Beginning, the Middle, or the End?](../../AAAI2026/information_retrieval/positional_bias_in_multimodal_embedding_models_do_they_favor_the_beginning_the_m.md)
- [\[ACL 2026\] Can Compact Language Models Search Like Agents? Distillation-Guided Policy Optimization for Preserving Agentic RAG Capabilities](../../ACL2026/information_retrieval/can_compact_language_models_search_like_agents_distillation-guided_policy_optimi.md)
- [\[ICML 2026\] Seeing to Generalize: How Visual Data Corrects Binding Shortcuts](seeing_to_generalize_how_visual_data_corrects_binding_shortcuts.md)

</div>

<!-- RELATED:END -->
