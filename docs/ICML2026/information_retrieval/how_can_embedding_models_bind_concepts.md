---
title: >-
  [Paper Note] How can embedding models bind concepts?
description: >-
  [ICML 2026][Information Retrieval & RAG][CLIP] This paper formalizes the question of "why embedding models fail to bind concepts" as a "complexity problem of the binding function." Through geometric analysis, it demonstrates that CLIP's scene embeddings decompose additively into objects and concepts (explaining why they are probeable in unimodal settings but fail c
tags:
  - ICML 2026
  - Information Retrieval & RAG
  - CLIP
date: 2026-05-08
content_hash: 7603715adbece563
---
# How can embedding models bind concepts?

**Conference**: ICML2026  
**arXiv**: [2605.31503](https://arxiv.org/abs/2605.31503)  
**Code**: Yes (repository link provided in the paper)  
**Area**: Interpretability  
**Keywords**: Concept Binding, CLIP, Embedding Geometry, Compositional Generalization, Multimodal Alignment

## TL;DR
This paper formalizes the question of "why embedding models fail to bind concepts" as a "complexity problem of the binding function." Through geometric analysis, it demonstrates that CLIP's scene embeddings decompose additively into objects and concepts (explaining why they are probeable in unimodal settings but fail cross-modally). Furthermore, it proves on controlled Transformers that with sufficient data coverage, models learn low-complexity binding dominated by **multiplicative interactions** between concepts, achieving systematic generalization to unseen object combinations.

## Background & Motivation
**Background**: Dual-encoder vision-language embedding models like CLIP exhibit "bag-of-concepts" behavior in cross-modal retrieval: they recognize individual concepts like "red" or "cube" but fail to distinguish "red cube + blue sphere" from "blue cube + red sphere" in multi-object scenes. This is the classic concept binding failure. Previous work has repeatedly observed this phenomenon and attributed it to insufficient encoder granularity, weak negation/spatial reasoning, or an inherent trade-off between concepts and objects.

**Limitations of Prior Work**: Previous explanations remained at the "behavioral level"—predicting a lack of capability upon observing CLIP's errors. However, they failed to explain a contradiction: in unimodal settings (training probes on purely image or text encoders), CLIP can actually recover object-level information. How can "cross-modal binding failure" and "unimodal object de-codability" coexist in the same vector?

**Key Challenge**: Cross-modal alignment requires the image encoder $B_{\text{img}}$ and text encoder $B_{\text{txt}}$ to produce comparable embeddings for the same scene. If the binding rules learned by the two encoders differ, they will mismatch on unseen object combinations. The problem thus shifts from "can it recognize objects" to "whether the concept $\to$ object mappings learned by both sides follow the same simple rule."

**Goal**: (1) Characterize the structure of CLIP's multi-object scene embeddings; (2) Measure the complexity of its implicitly implemented binding function; (3) Verify on controlled Transformers whether binding can be learned and identify its structural form.

**Key Insight**: Treat binding as a function $B:\mathcal{S}\to\mathbb{R}^d$ (scene to embedding). Drawing from MDL / Occam’s principle—if the $B$ learned by both encoders are low-complexity, compositional rules, they are more likely to converge to the same rule and align on OOD data; high complexity leads to memorizing the training distribution, causing OOD failure.

**Core Idea**: "CLIP cannot bind" is not a structural defect, but a result of the binding function it learns being too high-complexity. Given sufficient data coverage, a dual-encoder Transformer can learn low-complexity binding implemented via **multiplicative interactions**, enabling cross-modal alignment on unseen combinations.

## Method
This paper is not a "new model" but a "formal framework + a set of geometric/capacity diagnostic experiments." The pipeline consists of three parts: defining binding mathematically, dissecting CLIP's failure modes using this definition, and training a control group on synthetic data to explain its internal structure.

### Overall Architecture
**Formalization Layer**: Define concept space $\mathcal{C}=\mathcal{C}_1\times\cdots\times\mathcal{C}_C$. An object $\bm{o}=(c_1,\dots,c_C)$ is a tuple of concept values, and a scene $\bm{s}=(\bm{o}_1,\dots,\bm{o}_m)$ is a set of object tuples. A model $(f,q)$ consists of a scene encoder $f$ and a query encoder $q$, scored via cosine similarity. "Binding" is split into two independently measurable capabilities: concept recognition and object recognition. Binding is achieved only if both are satisfied; satisfying only the first is bag-of-concepts. The binding functions are defined as $B_{\text{img}}(\bm{s}):=f(\bm{x}_{\bm{s}})$ and $B_{\text{txt}}(\bm{s}):=q(\bm{y}_{\bm{s}})$.

**Geometric Diagnosis Layer** (Section 4): For real-world CLIP, object embeddings $\bm{u}_{\bm{o}}$ and concept embeddings $\bm{u}_c$ are estimated via subset averaging. This verifies whether scene embeddings decompose additively into objects (Level-I) and whether objects further decompose into concepts (Level-II). "Object-level editing" experiments are also conducted—directly performing $\tilde{\bm{z}}=f(\bm{x}_{\bm{s}})-\bm{u}_{\bm{o}_1}+\bm{u}_{\bm{o}_1'}$ in the embedding space to see if the retrieval/probe reflects a counterfactual scene.

**Capacity Diagnosis Layer** (Section 5): Train a small MLP approximator $g(\bm{o}_1,\bm{o}_2)$ that inputs discrete concept indices and outputs predicted scene embeddings, minimizing $\sum_{\bm{s}}\|f(\bm{x}_{\bm{s}})-g(\bm{o}_1,\bm{o}_2)\|^2$. Sweep widths $\{64, 256, 1024, 4096\}$ and training object coverage $\{0.1,\dots,0.9\}$ to test concept/object recognition on held-out objects. Simultaneously, train a dual-encoder Transformer from scratch (~20M parameters, output $\mathbb{R}^{512}$, AdamW + contrastive loss) on synthetic data, systematically varying $C$, $V$, and coverage to observe when binding generalizes.

### Key Designs

**1. Two-level Additive Decomposition Hypothesis (Level-I / Level-II): Making scene embedding composition a falsifiable geometric property**

Prior work only observed CLIP's errors behaviorally. This design transforms the contradiction of "unimodal success vs cross-modal failure" into testable geometric hypotheses: $f(\bm{x}_{\bm{s}})\approx \bm{u}_{\bm{o}_1}+\bm{u}_{\bm{o}_2}\approx \sum_{i}\bm{u}_{c_{1,i}}+\sum_{i}\bm{u}_{c_{2,i}}$. This means scene embeddings first additively decompose into objects (Level-I), and objects decompose into concepts (Level-II). Validation uses three estimators for $\bm{u}_{\bm{o}}$—multi-object scene average (avg), conditional average based on position in tuple (avg+pos), and single-object scene average (single-obj)—cross-validated with $R^2$, retrieval accuracy, and linear probing accuracy. The "targeted erasure" ablation is crucial: removing concept components collapses concept decoding while leaving object decoding intact, whereas removing object components collapses both. This confirms that object information is "packaged" in object-level components $\bm{u}_{\bm{o}}$, compressing concept combinations into a non-additive vector. Since $\bm{u}_{\bm{o}}$ and $\bm{u}_{\bm{o}}^{\text{txt}}$ need not come from the same function, cross-modal mismatch naturally occurs.

**2. Binding Function Capacity Diagnosis: Using the generalization of small approximators as a complexity metric**

Probing only proves "object information is present," not whether the concept $\to$ object mapping is simple. This design operationalizes "simplicity of binding" as "fit-ability by a small approximator on held-out objects." A family of approximators maps discrete concept indices to CLIP scene embeddings, and linear probes test concept/object recognition on the predicted embeddings. The results are stark: at 0.3 coverage, concept recognition stabilizes at $\ge 80\%$, but object recognition stays at ~20% even with an MLP width of 4096 and 0.9 coverage. Results for XGBoost/Random Forest are consistent—this suggests CLIP's binding is not limited by the approximator's weakness but is inherently a high-complexity function resembling memorization. Here, the MLP serves as a proxy for SGD's preference for simple solutions; its failure supports the high-complexity binding argument over mere information loss.

**3. Multiplicative Interaction Probes (Additive / Per-obj. products / Global product): Giving "generalizable binding" a reusable functional form**

Once a generalizable binding is trained in controlled Transformers, identifying its functional form—purely additive, per-object multiplicative, or global multiplicative—is essential. Three structural probes fit the scene embeddings: Additive follows $\sum_{i=1}^{2}\sum_{k=1}^{2}\bm{u}_{k,c_{ik}}$ (bag-of-concepts baseline); Per-obj. products adds intra-object products $\sum_i\prod_k \bm{v}_{i,k,c_{ik}}$; Global product adds cross-object products $\prod_i\prod_k \bm{v}_{i,k,c_{ik}}$. Product terms provide independent vectors for each concept combination—the "binding signal" that additive structures cannot express. Because this is a minimal deviation from additivity, it remains compositional and easier for encoders to align. Fig. 9 proves across ~500 models that OOD object recognition accuracy correlates strongly with the fit quality of the Global product probe, while applying the same probe to CLIP/DINOv2 only recovers concept recognition.

### Loss & Training
Controlled dual-encoders use standard symmetric contrastive loss with cosine similarity; AdamW optimizer; ~20M parameters per encoder; output dimension $d=512$. Diagnostic MLPs use MSE ($\ell_2$) regression. All training is done on synthetic multi-object data (CLEVR, CLEVR-2D, PUG:SPARE, and natural images via Gemini Nano Banana 2) by sweeping $(C, V)$ and training object coverage $\rho_{\text{train}}\in[0.1, 0.9]$.

## Key Experimental Results

### Main Results

| Dataset | Model | $R^2$ (avg / avg+pos) | Retrieval | Probing |
|--------|------|------|----------|------|
| Text (Synthetic caption) | CLIP | 0.90 / 0.92 | 0.97 | 0.99 |
| PUG:SPARE | CLIP | 0.75 / 0.84 | 0.93 | 0.98 |
| PUG:SPARE | DINOv2 | 0.78 / 0.86 | 0.86 | 0.98 |
| CLEVR | CLIP | 0.78 / 0.83 | 0.94 | 0.96 |
| Text | Random-init | 0.47 / 0.69 | 0.42 | 0.82 |

CLIP scene embeddings can be reconstructed with high quality ($R^2$ 0.75–0.92) from object components. Reconstructed embeddings maintain retrieval/probing performance close to the original model. This holds for CLEVR, occluded scenes, and natural images, extending Level-I decomposition beyond toy data.

| Dataset | Model | Probing (avg / avg+pos / single-obj) | Retrieval (avg / avg+pos / single-obj) |
|---------|-------|--------------------------------------|----------------------------------------|
| CLEVR | CLIP | 0.98 / 0.98 / 0.86 | 1.00 / 1.00 / 0.97 |
| CLEVR-2D | CLIP | 0.98 / 0.98 / 0.92 | 0.99 / 0.99 / 0.97 |
| PUG:SPARE | CLIP | 0.94 / 0.95 / – | 0.86 / 0.94 / – |
| PUG:SPARE | DINO | 0.97 / 0.97 / – | 0.48 / 0.76 / – |

Direct "object replacement" $\tilde{\bm{z}}=f(\bm{x}_{\bm{s}})-\bm{u}_{\bm{o}_1}+\bm{u}_{\bm{o}_1'}$ produces embeddings that correspond to counterfactual scenes. Notably, object embeddings estimated from single-object scenes can edit multi-object scenes (retrieval 0.97 on CLEVR), proving object components are "pluggable" geometric entities.

### Ablation Study

| Configuration | Text Conc. / Obj. | Image Conc. / Obj. | Note |
|------|-------------------|-------------------|------|
| CLIP-B/32 Original | 1.00 / 1.00 | 0.94 / 0.96 | baseline |
| − concept component | 0.06 / 0.99 | 0.05 / 0.85 | Concept decoding collapses, object decoding remains |
| − object component | 0.05 / 0.04 | 0.02 / 0.01 | Both concept and object decoding collapse |
| permute concept (control) | 0.92 / 0.99 | 0.99 / 0.97 | Removing wrong component shows no drop |
| permute object (control) | 0.96 / 1.00 | 0.86 / 0.92 | Removing wrong object shows no drop |

This table provides the definitive causal evidence: object-level components carry both "object identity" and "internal concept combinations," while concept components only carry the concepts themselves.

### Key Findings
- CLIP's binding failure is due to "high function complexity" rather than "information loss": high-capacity MLPs/XGBoosts/RFs fail to generalize object recognition ($\le 20\%$) while succeeding at concept recognition ($80\%+$), indicating the concept $\to$ object mapping is nearly independent across combinations (memorization).
- Binding generalization follows a sharp phase transition regarding data coverage: for $|O|=125{,}000$, object recognition jumps from random to perfect when coverage increases from 30% to 40%. Larger object spaces require lower relative coverage (~30% for $|O|\ge 2{,}500$).
- Generalizable models have binding functions that can be fitted by tiny MLPs; the fit quality of the Global product probe is strongly correlated with OOD object recognition. This aligns "low complexity," "multiplicative structure," and "cross-modal alignment" on the same empirical axis.

## Highlights & Insights
- Reframing binding as a **function** rather than a capability is the core conceptual upgrade. Under the definitions of $B_{\text{img}}$ and $B_{\text{txt}}$, complexity can be discussed via MDL/Occam, providing a mechanistic explanation for cross-modal failure beyond vague terms like "encoder granularity."
- The "targeted erasure" ablation paradigm is elegant: removing specific components collapses specific capabilities while control removals do not. This causal chain can be migrated to any dual-encoder internal attribution analysis.
- Using the **generalization ability** of MLP/XGBoost/RF as a proxy for binding complexity essentially operationalizes Kolmogorov complexity. This idea of using learners as "complexity rulers" is valuable for general representation analysis.
- Multiplicative interactions explain a long-standing mystery: why "sudden" compositional generalization emerges with scale—models in large object spaces are forced to learn simple rules expressible by $\prod$, which naturally align cross-modally OOD.

## Limitations & Future Work
- The conclusions rely on synthetic data (CLEVR, PUG:SPARE, etc.); natural scenes lack the "combinatorial completeness" required for these controlled experiments.
- "Complexity" is relative to the family of approximators. Technically, this is "incompressibility relative to these learners" rather than true Kolmogorov complexity.
- While identifying "multiplicative structure" as effective, the paper does not clarify how dual-encoder Transformers implement it internally (e.g., via attention or token-token inner products).
- No immediate solution for modifying CLIP training is provided; whether increasing data coverage or imposing multiplicative inductive biases is more practical remains a question for future work.

## Related Work & Insights
- **vs Trager et al. (2023) / Uselis et al. (2025) / Berasi et al. (2025)**: They proved additive decomposition for **single-object** embeddings in CLIP. This paper extends this to **multi-object scenes** and adds Level-II concept decomposition, unifying the geometric narrative.
- **vs Feng & Steinhardt (2024) / Feng et al. (2025)**: Found token-level binding mechanisms in autoregressive LLMs. This paper studies "single-vector scene embeddings" without token intermediaries, resulting in a different geometric narrative (additive + multiplicative).
- **vs Kang et al. (2025)**: Argued for a trade-off between concept and object recognition in CLIP. This paper uses embedding decomposition to show both can coexist; the bottleneck is the complexity of the binding function.
- **vs Yuksekgonul / Ma / Hsieh / Gurung et al. (fine-tuning routes)**: They attempt to make CLIP more "compositional" through data or losses. This paper provides a meta-explanation: any method that fails to compress the binding function into a low-complexity form will struggle with cross-modal alignment generalization.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Reframing binding failure as "high complexity + lack of multiplicative structure" is a fundamental conceptual upgrade.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive datasets, models, and coverage sweeps, though primarily based on synthetic data.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear conceptual layering, robust triple-controlled ablations, and steady narrative pace.
- Value: ⭐⭐⭐⭐ Provides a clear structural guide for building binding-capable CLIP models (coverage + multiplicative bias).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Explaining CLIP Zero-shot Predictions Through Concepts](../../CVPR2026/information_retrieval/explaining_clip_zero-shot_predictions_through_concepts.md)
- [\[ACL 2026\] How Large Language Models Balance Internal Knowledge with User and Document Assertions](../../ACL2026/information_retrieval/how_large_language_models_balance_internal_knowledge_with_user_and_document_asse.md)
- [\[ACL 2026\] Can Compact Language Models Search Like Agents? Distillation-Guided Policy Optimization for Preserving Agentic RAG Capabilities](../../ACL2026/information_retrieval/can_compact_language_models_search_like_agents_distillation-guided_policy_optimi.md)
- [\[ACL 2025\] Semantic Outlier Removal with Embedding Models and LLMs](../../ACL2025/information_retrieval/semantic_outlier_removal_with_embedding_models_and_llms.md)
- [\[ACL 2025\] Length-Induced Embedding Collapse in PLM-based Models](../../ACL2025/information_retrieval/length-induced_embedding_collapse_in_plm-based_models.md)

</div>

<!-- RELATED:END -->
