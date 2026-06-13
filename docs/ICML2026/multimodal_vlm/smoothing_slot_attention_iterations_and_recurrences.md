---
title: >-
  [Paper Note] Smoothing Slot Attention Iterations and Recurrences
description: >-
  [ICML 2026][Multimodal VLM][Object-Centric Learning] Addressing two long-neglected pain points in Slot Attention for image and video object-centric learning—"insufficient information for cold-start queries" and "forced u…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Object-Centric Learning"
  - "Slot Attention"
  - "Query Warming"
  - "Video OCL"
  - "Self-distillation"
date: 2026-05-08
content_hash: 746933b6d54b0156
---

# Smoothing Slot Attention Iterations and Recurrences

**Conference**: ICML 2026  
**arXiv**: [2508.05417](https://arxiv.org/abs/2508.05417)  
**Code**: https://github.com/Genera1Z/SmoothSA (Available)  
**Area**: Multimodal VLM / Object-Centric Learning / Slot Attention  
**Keywords**: Object-Centric Learning, Slot Attention, Query Warming, Video OCL, Self-distillation

## TL;DR
Addressing two long-neglected pain points in Slot Attention for image and video object-centric learning—"insufficient information for cold-start queries" and "forced unification of aggregation transformations for first/non-first frames"—the authors propose SmoothSA. By using a small self-distilled warming module to inject sample information into queries and allowing the first frame to run three iterations while non-first frames run only one, SmoothSA refreshes the SOTA on both image and video OCL benchmarks.

## Background & Motivation

**Background**: Object-Centric Learning (OCL) is a paradigm that represents visual scenes as a set of independent object/background vectors (slots). This structured and compact representation often outperforms dense feature maps in downstream reasoning, video prediction, and synthetic generation. Its mainstream implementations are almost entirely built on Slot Attention (SA): treating image features as keys/values and $n$ query slots as "competitors," it assigns patches to different slots through several rounds of iterative cross-attention to learn object-level representations. The entire process requires no external supervision and relies on reconstruction loss for training. The standard practice for video OCL (the STEVE series) involves recursive calls to the image-based SA across frames: the first frame's queries are consistent with the image scenario, while queries for non-first frames are predicted from the previous frame's slots via a Transformer encoder block.

**Limitations of Prior Work**: The authors identify two problems accepted by default in almost all methods but never addressed directly. First, "query cold start": whether initialized from learnable Gaussian sampling or positional priors, the initial query slots contain only dataset-level priors without any clues about the current sample. This sample-independent starting point drags down aggregation quality in images and video first frames, forcing the model to "guess" through more iterations. Second, "transformation homogenization": in video, the first-frame queries are cold-starts with scarce information, whereas non-first-frame queries are derived from previous slots with ample sample information. However, most methods treat both equally, applying the same three SA iterations and ignoring the significant information gap between them.

**Key Challenge**: Aggregation precision depends on the amount of sample information carried by the query. However, the SA framework is not naturally designed with differentiated processing paths for "queries with different prior information." Consequently, either all queries share a crude cold-start point, or all frames share a fixed number of iterations that does not favor any specific information state.

**Goal**: Without modifying the backbone OCL model, resolve: (i) how to inject sample-level information into cold-start queries for images/video first frames; (ii) how to allow SA to use different intensities of aggregation transformations between the first and subsequent video frames.

**Key Insight**: The authors notice that OCL models can output "good" slots after training; thus, "ideal queries" can actually be supervised by existing slots—which naturally supports self-distillation. Furthermore, the three-iteration setup is intended for cold-start queries; for non-first-frame queries that are already close to the true distribution, performing three iterations can lead to over-perturbation.

**Core Idea**: Insert a small module before SA to warm up cold-start queries into "approximate slots" (informative queries) and separate "three iterations" for the first frame from "single iterations" for non-first frames—simultaneously curing both problems by "symmetrizing information volume and transformation intensity."

## Method

### Overall Architecture
SmoothSA fully retains the classic OCL encode-aggregate-decode structure: the encoder maps image/video frames into features $F \in \mathbb{R}^{h\times w\times c}$, the aggregator $\phi_a$ (i.e., the SA module) aggregates features into $n$ slots $S \in \mathbb{R}^{n\times c}$, and the decoder reconstructs the input from slots to provide self-supervision. SmoothSA introduces two minor modifications at the aggregator entrance and the inter-frame scheduling: (1) Before the cold-start query $Q_1$ enters the SA iterations, it passes through a "warmer" $\phi_p$ to produce an informative query $\tilde Q_1$; (2) For video, it runs the standard 3 SA iterations only on the first frame to obtain slot $S_1$, while non-first frames run only 1 SA iteration, with queries derived directly from the previous frame's slot via a standard transition module. The modification is minimal and can be directly attached to any SA-based image/video OCL model.

### Key Designs

1.  **Query Warmer $\phi_p$ (Approximating slot distribution via self-distilled cold-start queries)**:
    - **Function**: Jointly maps the information-scarce cold-start query $Q_1$ and input features $F_1$ to an informative query $\tilde Q_1$ that "approximates the current sample's slots," providing a better starting point for subsequent SA iterations.
    - **Mechanism**: $\phi_p$ is a very lightweight module, structurally understandable as a single query-feature cross-attention/MLP. It takes $Q_1$ and $F_1$ as input and outputs $\tilde Q_1 \in \mathbb{R}^{n\times c}$. The supervision signal comes from the OCL model's own output slot $S_1$ for the current batch—letting $\tilde Q_1 \approx S_1$ via stop-gradient self-distillation. The loss is approximately $\mathcal{L}_p = \|\tilde Q_1 - \text{sg}(S_1)\|^2$. During training, $\phi_p$ is co-optimized with the backbone OCL; at inference, $\tilde Q_1$ replaces $Q_1$ in SA.
    - **Design Motivation**: Previous works (e.g., BO-QSA, MetaSlot) tried to enrich the query's global prior using multi-Gaussians or object prototype codebooks but still failed to inject "current sample" information into the query. $\phi_p$ is the first to explicitly inject sample features into the query via a differentiable channel. This is equivalent to "shifting the SA iteration curve forward" by one step, changing the starting point from a distant cold-start to a warmed-up point near the final slot, naturally reducing iteration error.

2.  **Heterogeneous Iteration Scheduling for Video Frames (Multiple iterations for the first frame, single iteration for others)**:
    - **Function**: Eliminates the waste and perturbation of using 3 SA iterations for all frames in STEVE-like frameworks, matching "iteration intensity" to "query information volume."
    - **Mechanism**: For the first frame (where queries are cold-start with a large information gap), $\Phi_a$ expands to $S_1^{(i)}, M_1^{(i)} = \phi_a(S_1^{(i-1)}, F_1)$ for $i=1,2,3$, taking $S_1 := S_1^{(3)}$. For non-first frames $t\ge 2$, the query $Q_t$ is obtained directly from the previous slot $S_{t-1}$ via a standard transition network (e.g., STEVE's Transformer encoder block) and then runs SA only once: $S_t, M_t = \phi_a(Q_t, F_t)$. Visualizations show that multiple iterations for non-first frames can cause unnecessary alignment oscillations in queries that are already close to the true distribution.
    - **Design Motivation**: The 3-iteration setting was originally intended to provide alignment margin for "information-scarce queries." If a query is already close to the true slot, multiple iterations only introduce redundant updates and wash out temporal information. Differentiating the "number of transformations" essentially provides different alignment paths for queries in different information states, following the simple intuition of "calculating more steps if the input is harder."

3.  **Tightly Coupled Training with Backbone OCL Self-Supervised Loss**:
    - **Function**: Enables the warming module and backbone OCL to work together end-to-end under the same self-supervised objective without external labels or additional stages.
    - **Mechanism**: The total loss is $\mathcal{L} = \mathcal{L}_{rec} + \lambda \mathcal{L}_p$, where $\mathcal{L}_{rec}$ is the OCL's own reconstruction loss (e.g., cross-entropy for dVAE tokens or pixel MSE), and $\mathcal{L}_p$ is the stop-gradient distillation term between the $\phi_p$ output and the current slot. This way, $\phi_p$ keeps up as the backbone OCL learns accurately, without allowing noise to back-propagate into the OCL backbone's SA learning.
    - **Design Motivation**: Training $\phi_p$ end-to-end as a standard trainable front layer with SA might lead to instability where "$\phi_p$ pulls the query toward noisy slots, while SA is interfered with by these noisy queries." Stop-gradient self-distillation breaks this cycle, ensuring $\phi_p$ always follows the OCL's current optimal solution, sharing roots with self-distillation frameworks like BYOL.

### Loss & Training
All modules are optimized simultaneously during training with the objective $\mathcal{L} = \mathcal{L}_{rec} + \lambda \mathcal{L}_p$. A stop-gradient is applied to the slot side in $\mathcal{L}_p$, making the warming module unidirectionally follow the backbone OCL's optimal slot distribution. In the video training phase, differentiated iteration counts are implemented directly via if-branches in the forward pass, adding no memory overhead or extra stages.

## Key Experimental Results

### Main Results
The authors attached SmoothSA to different backbones (DINOSAUR, STEVE, SAVi, RandSF.Q, etc.) across standard benchmarks for image OCL (COCO, Movi-E) and video OCL (Movi-D, Movi-E, YT-VIS, Physion). They compared against existing SA variants on object discovery (mIoU/FG-ARI), object recognition, and visual reasoning.

| Task | Dataset | Metric | Backbone Baseline | + SmoothSA | Gain |
|------|---------|--------|-------------------|------------|------|
| Image Object Discovery | COCO | mIoU / FG-ARI | DINOSAUR baseline | Ours | Consistent improvement |
| Video Object Discovery | Movi-E | FG-ARI | STEVE / RandSF.Q | Ours | Further gains on SOTA |
| Visual Reasoning | Physion | Accuracy | SA baseline | Ours | Significant |

(Detailed values are in the paper's tables; the core conclusion is that adding SmoothSA consistently improves all three types of metrics regardless of the backbone or dataset.)

### Ablation Study

| Configuration | FG-ARI / mIoU Trend | Description |
|---------------|---------------------|-------------|
| Full SmoothSA | Best | Warming + Inter-frame heterogeneous iterations |
| w/o Warming $\phi_p$ | Significant drop | Validates that query cold-start is the main bottleneck for first-frame aggregation |
| w/o Heterogeneous Iters (3 for all) | Drop | Proves multiple iterations negatively impact information-rich queries |
| $\phi_p$ without stop-gradient | Unstable/Drop | Validates necessity of stop-gradient in self-distillation |
| Larger $\phi_p$ | Saturated gain | Smaller warming modules are better; a few thousand parameters suffice |

### Key Findings
- The warming module is extremely lightweight yet brings stable gains, indicating that "starting point bias" accounts for a large portion of SA iteration error. A significant part of traditional iterations compensates for bad starts rather than performing true aggregation.
- Single iterations for non-first frames are not only sufficient but better, showing that "more information leading to fewer iterations" is a universal law for SA. This provides direct insight for future designs of SA with adaptive iteration counts.
- Improvements appear simultaneously in FG-ARI (proxy for object discovery quality) and downstream reasoning, indicating that improving query information volume improves both segmentation and downstream representation learning.

## Highlights & Insights
- The warmer uses the OCL's own output for self-distillation, adding almost no extra label/computation cost, yet resolves the "query cold start" issue—long avoided by mainstream methods—in a simple and theoretically clear manner.
- Treating "iteration count" as a hyperparameter that can switch based on query information rather than a rigid value of 3 aligns with the "information volume $\rightarrow$ computation volume" philosophy. This can be transferred to any iterative refinement framework (iterative query decoders, recursive mask refinement, etc.).
- The paper unifies two seemingly independent problems (cold-start in image first frames and homogenization across video frames) under the abstraction of "smoothing SA iterations and recurrences," providing a clear conceptual framework. This approach of grouping multiple engineering issues into a unified perspective is very helpful for inspiring new methods.

## Limitations & Future Work
- The warming module depends on the backbone OCL outputting "sufficiently good slots" as distillation targets. During early training stages when the OCL backbone might collapse, the supervision signal for $\phi_p$ is noisy; a warm-up schedule might be needed.
- Inter-frame iteration counts are hard-coded to "3 for the first frame, 1 for others," which doesn't account for scene cuts or sudden changes requiring a fresh cold-start. A reasonable extension would be a lightweight signal to judge if the "current frame query needs to be treated as a cold-start again."
- Experiments were mainly conducted on synthetic videos and medium-scale real data, not yet covering large-scale long videos or real open-world scenarios. Whether query information degrades under long-term drift remains to be observed.

## Related Work & Insights
- **vs Slot Attention / BO-QSA**: BO-QSA enriches query distribution via multi-Gaussians, but queries remain "dataset-level" priors without reaching "sample-level" information. SmoothSA's injection of sample features via self-distillation is a fundamental difference in approach.
- **vs MetaSlot**: MetaSlot performs a draft aggregation and then re-initializes queries using an object prototype codebook. It still essentially "supplements queries with priors" rather than "supplements queries with the current sample" and requires extra codebooks and discretization, increasing complexity.
- **vs STEVE / SAVi / RandSF.Q**: These methods focus on "how to propagate queries better between frames" but default to running the same SA iterations for all frames. SmoothSA is the first to make "inter-frame transformation intensity" adjustable, which is orthogonal to query transition mechanisms and can be stacked on top of SOTA methods like RandSF.Q for further gains.

## Rating
- Novelty: ⭐⭐⭐⭐ First to explicitly formalize "query cold start" and "transformation homogenization," though the solution (self-distillation + scheduling) is simple.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated consistency across multiple image/video OCL backbones and downstream tasks with complete ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear narrative by unifying two modifications under "smooth iterations / smooth recurrences."
- Value: ⭐⭐⭐⭐ High engineering value due to stable performance gains across SA-based OCL backbones with nearly zero cost.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Seeing is Understanding: Unlocking Causal Attention into Modality-Mutual Attention for Multimodal LLMs](seeing_is_understanding_unlocking_causal_attention_into_modality-mutual_attentio.md)
- [\[ICML 2026\] Certified Robustness under Heterogeneous Perturbations via Hybrid Randomized Smoothing](certified_robustness_under_heterogeneous_perturbations_via_hybrid_randomized_smo.md)
- [\[ICML 2026\] Large Vision-Language Models Get Lost in Attention](large_vision-language_models_get_lost_in_attention.md)
- [\[ICML 2026\] Hyper-ICL: Attention Calibration with Hyperbolic Anchor Distillation for Multimodal ICL](hyper-icl_attention_calibration_with_hyperbolic_anchor_distillation_for_multimod.md)
- [\[ICLR 2026\] Directional Embedding Smoothing for Robust Vision Language Models](../../ICLR2026/multimodal_vlm/directional_embedding_smoothing_for_robust_vision_language_models.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] Seeing is Understanding: Unlocking Causal Attention into Modality-Mutual Attention for Multimodal LLMs](seeing_is_understanding_unlocking_causal_attention_into_modality-mutual_attentio.md)
- [\[ICML 2026\] Certified Robustness under Heterogeneous Perturbations via Hybrid Randomized Smoothing](certified_robustness_under_heterogeneous_perturbations_via_hybrid_randomized_smo.md)
- [\[ICML 2026\] Large Vision-Language Models Get Lost in Attention](large_vision-language_models_get_lost_in_attention.md)
- [\[ICLR 2026\] Directional Embedding Smoothing for Robust Vision Language Models](../../ICLR2026/multimodal_vlm/directional_embedding_smoothing_for_robust_vision_language_models.md)
- [\[ICML 2026\] Hyper-ICL: Attention Calibration with Hyperbolic Anchor Distillation for Multimodal ICL](hyper-icl_attention_calibration_with_hyperbolic_anchor_distillation_for_multimod.md)

</div>

<!-- RELATED:END -->
