---
title: >-
  [Paper Note] How can embedding models bind concepts?
description: >-
  [ICML 2026][Information Retrieval & RAG][CLIP] This paper formalizes "why embedding models fail to bind concepts" as a "complexity problem of the binding function": geometric analysis proves that CLIP's scene embeddings decompose additively into the sum of objects and concepts (explaining why probes succeed unimodally while cross-modal alignment fails); experiments
tags:
  - ICML 2026
  - Information Retrieval & RAG
  - CLIP
date: 2026-05-08
content_hash: 5bfed783ccfe4228
---
# How can embedding models bind concepts?

**Conference**: ICML2026  
**arXiv**: [2605.31503](https://arxiv.org/abs/2605.31503)  
**Code**: Yes (Public repository at the end of the paper)  
**Area**: Interpretability  
**Keywords**: Concept binding, CLIP, Embedding geometry, Compositional generalization, Multimodal alignment

## TL;DR
This paper formalizes "why embedding models fail to bind concepts" as a "complexity problem of the binding function": geometric analysis proves that CLIP's scene embeddings decompose additively into the sum of objects and concepts (explaining why probes succeed unimodally while cross-modal alignment fails); experiments on controlled Transformers demonstrate that with sufficient data coverage, models learn a low-complexity binding dominated by **multiplicative interactions** between concepts, achieving systematic generalization to unseen object combinations.

## Background & Motivation
**Background**: Dual-encoder vision-language embedding models like CLIP exhibit "bag-of-concepts" behavior in cross-modal retrieval: they can identify individual concepts like "red" or "cube" but fail to distinguish between "red cube + blue sphere" and "blue cube + red sphere" in multi-object scenes. This is the classic failure of concept binding. Prior work has repeatedly observed this phenomenon, attributing it to insufficient encoder granularity, weak negation/spatial reasoning, or an inherent trade-off between concepts and objects.

**Limitations of Prior Work**: Previous explanations remain at a "behavioral level"—observing CLIP's errors and inferring a lack of capability—yet fail to explain a contradiction: within a single modality (training probes using only the image or text encoder), CLIP can actually recover object-level information. How can "failure to bind cross-modally" and "decodable objects within a single modality" coexist in the same vector?

**Key Challenge**: Cross-modal alignment requires the image encoder $B_{\text{img}}$ and text encoder $B_{\text{txt}}$ to produce comparable embeddings for the same scene. If the binding rules learned by each differ, they will mismatch on unseen object combinations. The problem thus shifts from "can it recognize objects" to "whether the concept→object mappings learned by both follow the same simple rule."

**Goal**: (1) Characterize the structure of CLIP's multi-object scene embeddings; (2) Measure the complexity of its implicitly implemented binding function; (3) Verify if binding can be learned on controlled Transformers and identify its structural form.

**Key Insight**: Binding is formulated as a function $B:\mathcal{S}\to\mathbb{R}^d$ (scene to embedding). Drawing from MDL / Occam's principle—if the $B$ learned by both encoders are low-complexity, compositional rules, they are more likely to converge to the same rule and align on OOD data; high complexity leads to memorizing the training distribution and OOD failure.

**Core Idea**: "CLIP's failure to bind" is not a structural defect but a result of its binding function being too high-complexity. Given sufficient data coverage, a dual-encoder Transformer can learn a low-complexity binding implemented via **multiplicative interactions**, enabling cross-modal alignment for unseen combinations.

## Method
This work is not a "new model" but a "formal framework + a set of geometric/capacity diagnostic experiments." The pipeline is divided into three parts: providing a falsifiable mathematical definition of binding, using this definition to dissect CLIP's failure modes, and training a controlled Transformer group on synthetic data to explain the internal structure of successful binding.

### Overall Architecture
**Formalization layer**: Defines the concept space $\mathcal{C}=\mathcal{C}_1\times\cdots\times\mathcal{C}_C$. An object $\bm{o}=(c_1,\dots,c_C)$ is a tuple of concept values, and a scene $\bm{s}=(\bm{o}_1,\dots,\bm{o}_m)$ is a set of object tuples. A model $(f,q)$ consists of a scene encoder $f$ and a query encoder $q$, scored by cosine similarity. "Binding" is split into two independently measurable capabilities—concept recognition (scores for all present concepts are higher than absent ones) and object recognition (scores for all present complete objects are higher than absent ones). Binding is achieved only if both are satisfied; satisfying only the first is bag-of-concepts. The binding functions are defined as $B_{\text{img}}(\bm{s}):=f(\bm{x}_{\bm{s}})$ and $B_{\text{txt}}(\bm{s}):=q(\bm{y}_{\bm{s}})$.

**Geometric diagnosis layer** (Section 4): For real CLIP, object embeddings $\bm{u}_{\bm{o}}$ and concept embeddings $\bm{u}_c$ are estimated via subset averaging to verify if scene embeddings decompose additively into objects (Level-I) and if objects further decompose into concepts (Level-II). "Object-level editing" experiments are also performed—directly performing $\tilde{\bm{z}}=f(\bm{x}_{\bm{s}})-\bm{u}_{\bm{o}_1}+\bm{u}_{\bm{o}_1'}$ in the embedding space to see if retrieval/probing reflects the counterfactual scene.

**Capacity diagnosis layer** (Section 5): A small MLP approximator $g(\bm{o}_1,\bm{o}_2)$ is trained to map discrete concept indices to predicted scene embeddings by minimizing $\sum_{\bm{s}}\|f(\bm{x}_{\bm{s}})-g(\bm{o}_1,\bm{o}_2)\|^2$. Widths $\{64, 256, 1024, 4096\}$ and training object coverage $\{0.1, \dots, 0.9\}$ are swept, measuring concept/object recognition on held-out objects. Simultaneously, a dual-encoder Transformer (~20M parameters, output $\mathbb{R}^{512}$, AdamW + contrastive loss) is trained from scratch on synthetic data, systematically varying $C$, $V$, and coverage to observe when binding generalizes.

### Key Designs

**1. Two-level additive decomposition hypothesis (Level-I / Level-II): Turning "what scene embeddings are made of" into falsifiable geometric properties**

Prior work only observed CLIP's errors at a behavioral level but couldn't explain the contradiction of "unimodal binding vs. cross-modal failure." This design transforms it into a measurable geometric hypothesis: $f(\bm{x}_{\bm{s}})\approx \bm{u}_{\bm{o}_1}+\bm{u}_{\bm{o}_2}\approx \sum_{i}\bm{u}_{c_{1,i}}+\sum_{i}\bm{u}_{c_{2,i}}$, meaning scene embeddings first decompose additively into objects (Level-I), and objects further decompose into concepts (Level-II). Three estimators are used to obtain object embeddings $\bm{u}_{\bm{o}}$—multi-object scene average (avg), conditional average based on position (avg+pos), and single-object scene average (single-obj)—validated across $R^2$, retrieval accuracy, and linear probe accuracy. The most critical is "targeted removal" ablation: subtracting concept components from scene embeddings collapses concept decoding to near-random while object decoding remains intact; subtracting object components collapses both. This confirms that "object information exists in object-level components." Unimodal probes can recover objects because the object-level component $\bm{u}_{\bm{o}}$ is explicitly "packed" as a non-additive vector of concept combinations; however, it doesn't require alignment with the text-side $\bm{u}_{\bm{o}}^{\text{txt}}$, leading to cross-modal failure.

**2. Capacity diagnosis of the binding function: Using small approximator generalization as a complexity metric**

Simply training probes only proves "object information is in the embedding," not "whether the concept→object mapping has a simple form." This design operationalizes the abstract "simplicity of binding" as "whether it can be fitted by a small approximator on held-out objects." A family of approximators maps discrete concept indices to CLIP scene embeddings, and previously trained linear probes are used to test recognition on the predicted embeddings of held-out objects. Results are sharp: concept recognition stabilizes at $\ge 80\%$ when coverage exceeds 0.3, but object recognition remains at ~20% even with MLP width at 4096 and coverage at 0.9. The phenomenon is consistent with XGBoost and Random Forest—suggesting CLIP's binding is not a "weak approximator" issue but a high-complexity, near-memorization function. The MLP acts as a proxy for SGD's preference for simple solutions; its failure to fit a generalizing solution supports the conclusion of high complexity.

**3. Multiplicative interaction probes (Additive / Per-obj. products / Global product): Providing a reusable functional form for "generalizing binding"**

After training a "generalizing binding" in controlled Transformers, explaining why it aligns cross-modally on unseen combinations requires probing its functional form: is it purely additive, per-object multiplicative, or globally multiplicative? Three structured probes fit the scene embeddings: Additive follows $\sum_{i=1}^{2}\sum_{k=1}^{2}\bm{u}_{k,c_{ik}}$ (bag-of-concepts baseline); Per-obj. products adds intra-object products $\sum_i\prod_k \bm{v}_{i,k,c_{ik}}$; Global product adds cross-object products $\prod_i\prod_k \bm{v}_{i,k,c_{ik}}$. Multiplicative terms provide an independent vector for each concept combination—the "binding signal" that additive structures cannot express. Because this minimally deviates from additivity and the structure itself is compositional, encoders more easily converge to the same binding rule. Fig. 9 shows across ~500 models that OOD object recognition accuracy correlates strongly with Global product probe quality, whereas applying the same probe to CLIP / DINOv2 only recovers concept recognition, with object recognition near zero.

### Loss & Training
Controlled dual-encoders use CLIP's symmetric contrastive loss with cosine similarity; AdamW optimizer; ~20M parameters per encoder, output dimension $d=512$. Diagnostic MLPs use MSE ($\ell_2$) regression to target embeddings. All training is performed on synthetic multi-object data (CLEVR, CLEVR-2D, PUG:SPARE, and natural images generated by Gemini Nano Banana 2), sweeping generalization curves by controlling $(C,V)$ and training object coverage $\rho_{\text{train}}\in[0.1,0.9]$.

## Key Experimental Results

### Main Results
| Dataset | Model | $R^2$ (avg / avg+pos) | Retrieval | Probing |
|--------|------|------|----------|------|
| Text (Synthetic caption) | CLIP | 0.90 / 0.92 | 0.97 | 0.99 |
| PUG:SPARE | CLIP | 0.75 / 0.84 | 0.93 | 0.98 |
| PUG:SPARE | DINOv2 | 0.78 / 0.86 | 0.86 | 0.98 |
| CLEVR | CLIP | 0.78 / 0.83 | 0.94 | 0.96 |
| Text | Random-init | 0.47 / 0.69 | 0.42 | 0.82 |

CLIP scene embeddings can be reconstructed with high quality from the sum of object components ($R^2$ 0.75–0.92), and retrieval/probing of the reconstructed embeddings matches the original model. This holds for three-object CLEVR, scenes with occlusions, and natural images (Gemini Nano Banana 2), extending Level-I additive decomposition to realistic settings.

| Dataset | Model | Probing (avg / avg+pos / single-obj) | Retrieval (avg / avg+pos / single-obj) |
|---------|-------|--------------------------------------|----------------------------------------|
| CLEVR | CLIP | 0.98 / 0.98 / 0.86 | 1.00 / 1.00 / 0.97 |
| CLEVR-2D | CLIP | 0.98 / 0.98 / 0.92 | 0.99 / 0.99 / 0.97 |
| PUG:SPARE | CLIP | 0.94 / 0.95 / – | 0.86 / 0.94 / – |
| PUG:SPARE | DINO | 0.97 / 0.97 / – | 0.48 / 0.76 / – |

Direct "object replacement" $\tilde{\bm{z}}=f(\bm{x}_{\bm{s}})-\bm{u}_{\bm{o}_1}+\bm{u}_{\bm{o}_1'}$ in embedding space yields embeddings reflecting counterfactual scenes. Notably, object embeddings estimated from single-object scenes can edit multi-object scenes (retrieval remains 0.97 on CLEVR), proving object components are almost "pluggable" geometric objects.

### Ablation Study
| Configuration | Text Conc. / Obj. | Image Conc. / Obj. | Description |
|------|-------------------|-------------------|------|
| CLIP-B/32 Original | 1.00 / 1.00 | 0.94 / 0.96 | baseline |
| − concept component | 0.06 / 0.99 | 0.05 / 0.85 | Concept decoding fails, object decoding holds |
| − object component | 0.05 / 0.04 | 0.02 / 0.01 | Both concept and object decoding fail |
| permute concept (control) | 0.92 / 0.99 | 0.99 / 0.97 | Removing wrong component doesn't drop |
| permute object (control) | 0.96 / 1.00 | 0.86 / 0.92 | Removing wrong object doesn't drop |

This table provides the most critical causal evidence: object-level components in embeddings carry both "object identity" and "internal concept combinations," whereas concept components carry only the concepts themselves.

### Key Findings
- CLIP's failure to bind is a "complexity problem," not "information loss": High-capacity MLP/XGBoost/RF remain at $\le 20\%$ object recognition on held-out objects, despite reaching 80%+ on concepts. This implies the concept→object mapping is near-memorization.
- Binding generalization exhibits a sharp phase transition: At $|O|=125{,}000$, object recognition jumps from near-random to near-perfect as coverage increases from 30% to 40%. Larger object spaces require lower relative coverage (~30% for $|O|\ge 2{,}500$).
- Generalizing models have binding functions fittable by small MLPs, and the quality of Global product probe fit strongly correlates with OOD object recognition across ~500 models—linking "low complexity," "multiplicative structure," and "cross-modal alignment."

## Highlights & Insights
- Framing binding as a **function** rather than a capability is the key conceptual upgrade. Defining $B_{\text{img}}$ and $B_{\text{txt}}$ allows for complexity discussions via MDL/Occam, providing a mechanistic explanation for alignment failure beyond "insufficient granularity."
- The "targeted removal" ablation paradigm is elegant: subtracting the correct component collapses the corresponding capability while subtracting the wrong one does nothing. This three-layered control logic can be transferred to any dual-encoder attribution analysis.
- Using the **generalization ability** of MLPs/XGBoost as a proxy for binding complexity effectively engineering the "undecidability of Kolmogorov complexity" into a tractable "search for simple solutions via SGD."
- The discovery of multiplicative interactions explains a long-standing puzzle: why scaling data and parameters makes compositional generalization "suddenly" emerge—because in a large enough object space, the model is forced to learn simple rules (representable by $\prod$) that happen to align both encoders OOD.

## Limitations & Future Work
- Conclusions are based on synthetic data (CLEVR, PUG:SPARE, etc.); realistic scenes lack "compositionally complete" datasets to replicate such controlled experiments. Failure distributions in natural images may be more complex.
- "Complexity" is defined relative to a family of approximators. Strictly, this is "incompressibility relative to these learners" rather than true Kolmogorov complexity.
- While identifying "multiplicative structure" as effective, the paper does not clarify how dual-encoder Transformers implement this internally (e.g., via attention or token-token dot products), leaving a mechanistic gap.
- No direct training recipe for modifying CLIP is provided. Whether "increasing data coverage" or "imposing multiplicative inductive biases" is more practical for industrial models remains an open question.

## Related Work & Insights
- **vs Trager et al. (2023) / Uselis et al. (2025) / Berasi et al. (2025)**: These works proved additive decomposition for **single-object** embeddings; this paper extends this to **multi-object scenes** and adds Level-II concept decomposition, unifying the geometric narrative.
- **vs Feng & Steinhardt (2024) / Feng et al. (2025)**: They found token-level binding ID mechanisms in AR LLMs; this work studies "single-vector scene embeddings" without token intermediaries, identifying a different geometric narrative (additive + multiplicative).
- **vs Kang et al. (2025)**: Suggested a trade-off between concept and object recognition in CLIP. This paper refutes this using embedding decomposition: both can coexist; the bottleneck is the complexity of the binding function.
- **vs Fine-tuning routes (Yuksekgonul, Ma, Hsieh, Gurung)**: Those attempts to make CLIP more "compositional" through data or loss functions gain a higher-level explanation: any method failing to compress the binding function into a low-complexity (multiplicative) form will struggle to generalize in cross-modal alignment.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Reframing binding failure as "high function complexity + lack of multiplicative structure" is a fundamental conceptual upgrade.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple datasets, models, and capacity/coverage sweeps with ~500 model correlation curves, though limited to synthetic data.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear hierarchy, rigorous ablation controls, and steady narrative pace.
- Value: ⭐⭐⭐⭐ Provides clear structural guidance (data coverage + multiplicative bias) for building "binding-capable" CLIP models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

- **Compositional Generalization with Scaling laws**, arXiv 2024
- **Geometric structure of CLIP embeddings**, ICML 2023
- **Probing concept binding in VLMs**, CVPR 2024

</div>

<!-- RELATED:END -->

## Related Papers

- [\[CVPR 2026\] Explaining CLIP Zero-shot Predictions Through Concepts](../../CVPR2026/information_retrieval/explaining_clip_zero-shot_predictions_through_concepts.md)
- [\[ACL 2026\] How Large Language Models Balance Internal Knowledge with User and Document Assertions](../../ACL2026/information_retrieval/how_large_language_models_balance_internal_knowledge_with_user_and_document_asse.md)
- [\[AAAI 2026\] Positional Bias in Multimodal Embedding Models: Do They Favor the Beginning, the Middle, or the End?](../../AAAI2026/information_retrieval/positional_bias_in_multimodal_embedding_models_do_they_favor_the_beginning_the_m.md)
- [\[ACL 2026\] Can Compact Language Models Search Like Agents? Distillation-Guided Policy Optimization for Preserving Agentic RAG Capabilities](../../ACL2026/information_retrieval/can_compact_language_models_search_like_agents_distillation-guided_policy_optimi.md)
- [\[ACL 2025\] Semantic Outlier Removal with Embedding Models and LLMs](../../ACL2025/information_retrieval/semantic_outlier_removal_with_embedding_models_and_llms.md)

</div>

<!-- RELATED:END -->
