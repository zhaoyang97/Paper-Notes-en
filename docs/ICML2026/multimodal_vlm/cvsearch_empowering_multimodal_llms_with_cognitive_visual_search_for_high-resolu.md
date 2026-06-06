---
title: >-
  [Paper Note] CVSearch: Empowering Multimodal LLMs with Cognitive Visual Search for High-Resolution Image Perception
description: >-
  [ICML2026][Multimodal VLM][High-resolution perception] CVSearch proposes a training-free "Assess-then-Search" cognitive framework. It first employs a visual expert (SAM 3) for rapid localization. When the expert fails…
tags:
  - "ICML2026"
  - "Multimodal VLM"
  - "High-resolution perception"
  - "visual search"
  - "training-free framework"
  - "semantic adaptive patching"
  - "bottom-up search"
date: 2026-05-08
content_hash: e93e9a046a76a718
---

# CVSearch: Empowering Multimodal LLMs with Cognitive Visual Search for High-Resolution Image Perception

**Conference**: ICML2026  
**arXiv**: [2605.23655](https://arxiv.org/abs/2605.23655)  
**Code**: https://github.com/ICML26-CVSearch (Claimed open-source in the paper)  
**Area**: Multimodal VLM  
**Keywords**: High-resolution perception, visual search, training-free framework, semantic adaptive patching, bottom-up search

## TL;DR
CVSearch proposes a training-free "Assess-then-Search" cognitive framework. It first employs a visual expert (SAM 3) for rapid localization. When the expert fails, it triggers semantic-guided adaptive patching and bottom-up search as a fallback, achieving SOTA in both accuracy and efficiency on high-resolution benchmarks such as V*Bench and HR-Bench.

## Background & Motivation

**Background**: Most current Multimodal Large Language Models (MLLMs) process images at fixed low resolutions (e.g., $336\times336$). For high-resolution images in real-world scenarios (thousands of pixels on the long side), they require aggressive downsampling before passing through the visual encoder, projector, and language model pipeline.

**Limitations of Prior Work**: Three routes have emerged for high-resolution perception, but none are fully satisfactory. The Cropping route (e.g., LLaVA-NeXT) uses fixed grid cropping, where objects may be split across grids, causing "semantic aliasing." The High-resolution Encoder route (e.g., LLaVA-HR) modifies the architecture to inject high-frequency features but adapts poorly to varying aspect ratios. The Visual Search route is divided into two types: Expert-assisted (SEAL, DyFo, V2-SAM) is fast but relies entirely on external detector proposal quality, leading to "blind spots" for small or abstract objects; Scanning-based (ZoomEye, RAP, DC²) uses exhaustive tree-structured grids, which is robust but wastes computation on backgrounds and still suffers from splitting objects.

**Key Challenge**: There is a binary opposition between efficiency and robustness—expert assistance is fast but fragile, while exhaustive scanning is stable but expensive. Furthermore, both methods are "semantic-unaware," treating the entire image as a uniform grid.

**Goal**: To combine both routes into a unified framework, allowing the model to "glance" first like a human before deciding how to look deeper, and to partition images according to semantic structures rather than regular grids when detailed inspection is necessary.

**Key Insight**: The authors draw inspiration from the dual-pathway visual search theory in cognitive science—the non-selective pathway extracts a global gist, while the selective pathway performs serial inspection of objects based on attention templates. This theory emphasizes that scene structure is the primary guide for attention deployment. Implementing this in MLLMs results in a cascade: "Assess if it can be answered directly → Find an expert if not → Semantic scanning if the expert fails."

**Core Idea**: Utilize the MLLM's own "Yes/No" confidence as a signal for information sufficiency. Treat visual expert failure as a trigger to switch to semantic scanning rather than as a final point. During the scanning phase, use features extracted by the expert for semantic clustering and patching, then propagate evidence bottom-up to avoid error propagation from top-down searches.

## Method

### Overall Architecture
The input consists of a high-resolution image $\bm{I}\in\mathbb{R}^{H\times W\times 3}$ and a text query $\bm{Q}$. The output is the MLLM-generated answer $Y$. The process follows a three-stage **Assess-then-Search** pipeline:

1. **Assess**: Feed $(\bm{I},\bm{Q})$ into the MLLM and quantify information sufficiency using the Yes-token probability $c_q(\bm{I})$ for the question "Can the answer be provided based only on current visual information?". If $c_q(\bm{I})>\tau_q$, the answer is generated immediately, bypassing the search process.
2. **Expert Search**: If $c_q$ is insufficient, $\bm{Q}$ is decomposed into a set of target objects $\bm{O}=\{o_1,\dots,o_m\}$ via MLLM in-context extraction (fallback to SpaCy). Open-vocabulary segmentation using SAM 3 yields a set of bounding boxes $\bm{B}_e$ and dense visual features $\bm{H}_e$. If the number of categories segmented by SAM 3 matches $|\bm{O}|$, crops are made according to $\bm{B}_e$ to answer; otherwise, the third stage is entered.
3. **Scene-aware Scanning**: Reuse $\bm{H}_e$ (saving a recalculation) and perform semantic adaptive patching using SLIC + adjacency-constrained agglomerative clustering. This recursively constructs an image tree $\bm{T}$ of depth $D$. Exploration starts from the deepest leaf nodes and proceeds bottom-up. Search terminates if a threshold is exceeded; if the root is reached without success, the highest priority node at the root level is fed back to the visual expert for the next iteration.

### Key Designs

1. **Cognitive-Driven Adaptive Switching**:
    - **Function**: Dynamically decides between "direct answer / expert search / semantic scanning" based on information sufficiency $c_q$ and expert success, translating "expert failure" into a "switch to scanning" rather than "giving up."
    - **Mechanism**: Information sufficiency follows the ZoomEye form $c_q(\bm{I})=\mathcal{M}(\text{"Yes"}\mid p_q(\bm{Q}),\bm{I})$, using the normalized probability of the "Yes" token as internal confidence. The switching threshold $\tau_q$ is set to $0.9$. Search termination uses an adaptive decaying threshold $\tau_{curr}$ (slowly annealing from $\tau_q$ to a minimum $\hat{\tau}_q=0.5$), maintaining high certainty for easy samples while allowing less confident predictions for hard samples. SAM 3 failure is determined not by the absence of boxes, but by the mismatch between extracted categories and $|\bm{O}|$.
    - **Design Motivation**: Older frameworks either strictly used experts or strictly used scanning, failing on small targets or abstract queries. Using the MLLM as a judge ensures expensive scanning is only triggered when necessary, saving computation on the main path.

2. **Semantic Guided Adaptive Patching (SGAP)**:
    - **Function**: Replaces fixed grid cropping by dividing the image into $k^*$ patches based on "semantically connected regions" to avoid splitting objects.
    - **Mechanism**: Runs SLIC in the feature space of expert features $\bm{H}_e$ to obtain $N$ atomic superpixels $\bm{A}=\{a_1,\dots,a_N\}$ and builds a spatial adjacency graph $G$. Agglomerative clustering constrained by $G$ merges atoms into $k$ spatially connected semantic clusters, where the bounding box of each cluster forms a patch. Selecting $k$ is critical: minimize $\mathcal{L}(k)=\mathcal{L}_o(\bm{B}_k)-\mathcal{L}_s(\bm{H}_a,\bm{l}_k)$ within $[k_\min,k_\max]=[4,8]$, where $\mathcal{L}_o$ penalizes spatial overlap and $\mathcal{L}_s$ is the silhouette score for cluster compactness. Simultaneously, Visual Complexity $c_v(\bm{I}_{d,t})=\max(0,\,1-\tfrac{1}{|\bm{R}|}\sum_{i\in\bm{R}}\mathrm{cosim}(\bm{h}_i,\bar{\bm{h}}))$ is calculated for each patch, using average cosine distance to measure "feature divergence." Background nodes with $c_v<\tau_v=0.4$ are pruned.
    - **Design Motivation**: Fixed grids create "semantic aliasing" and treat backgrounds identically. Clustering using the expert's own semantic features ensures the patching strategy shares the same representation as subsequent visual understanding, preserving object integrity and concentrating the budget on high-entropy regions.

3. **Dynamic Bottom-Up Search**:
    - **Function**: Evals from the deepest leaf nodes of the semantic tree $\bm{T}$, aggregating evidence upward to parent nodes to avoid incorrect paths typical of top-down searches starting from blurry global views.
    - **Mechanism**: Visit priority for each node is $c_x=\alpha\cdot c_v+\beta\cdot c_o+\gamma\cdot c_x^*$, where $c_v$ is visual complexity, $c_o$ is the MLLM's Yes-confidence for object presence, and $c_x^*$ is the maximum child priority (0 for leaves), with hyperparameters $(\alpha,\beta,\gamma)=(0.2,0.4,0.4)$. For multi-object queries, decoupled sub-queries $Q_d$ are constructed for each $o_i$. Termination uses the annealing threshold $\tau_{curr}$ with $\hat{\tau}_q$ as a fallback. If the deepest level finishes without termination, the search moves up. If the entire tree is traversed without success, the highest-scoring node of the first level is sent back to the visual expert for a new round of Expert Search, forming an iterative loop.
    - **Design Motivation**: Top-down searches pick nodes based on low-resolution global views, which are difficult for small targets; a single wrong branch invalidates the entire path. Bottom-up search ensures small objects are inspected first in the clearest local views and translates "not found" into "re-triggering the expert," creating a failure-recoverable process.

### Loss & Training
The entire pipeline is **training-free**, involving no backpropagation—all "scores" are MLLM forward probabilities or geometric/clustering heuristics. Main hyperparameters: $\tau_q=0.9$, $\tau_v=0.4$, $\hat{\tau}_q=0.5$, $(k_\min,k_\max)=(4,8)$, single-object tree depth $D=2$, multi-object $D=3$, $(\alpha,\beta,\gamma)=(0.2,0.4,0.4)$. The visual expert is SAM 3. Baseline MLLMs include Qwen2.5-VL-7B, LLaVA-OV-7B, and InternVL2.5-8B. Experiments were conducted on 4×A6000, though this refers to configuration rather than "training."

## Key Experimental Results

### Main Results
Benchmarks cover high-resolution specifics (V*Bench, HR-Bench 4K/8K), general real-world scenarios (MME-RealWorld-Lite, TreeBench), and drone ultra-small targets (FineRS-4K). Average resolution is approximately $2000\times1500$.

| Baseline MLLM | V*Bench | HR-Bench 4K | HR-Bench 8K | Source of Gain |
|---|---|---|---|---|
| LLaVA-OV-7B | 75.4 → **91.6** | 63.0 → **75.6** | 59.8 → **74.8** | +CVSearch |
| Qwen2.5-VL-7B | 71.2 → **90.1** | 68.8 → **76.6** | 65.3 → **75.6** | +CVSearch |
| InternVL2.5-8B | 69.1 → **89.0** | 66.0 → **77.0** | 57.4 → **77.6** | +CVSearch |
| GPT-4o (Closed-source ref.) | 66.0 | 59.0 | 55.5 | — |
| Qwen2.5-VL-32B | 85.9 | 74.8 | 71.6 | Control only |

Applying the framework to 7B-class open-source models outperforms 32B models and GPT-4o, proving that the bottleneck lies in "where to look" rather than parameter count.

### Ablation Study
| Configuration | V* / HR-4K / HR-8K (Illus.) | Description |
|---|---|---|
| Full CVSearch | 90.1 / 76.6 / 75.6 | Qwen2.5-VL-7B full version |
| w/o Expert Search (Scanning only) | Significant Drop | Lost fast path, simple samples forced into deep search |
| w/o Scene-aware Scanning | Significant Drop | Lost fallback for expert failure, small target blind spots return |
| w/o SGAP (Back to fixed grid) | Moderate Drop | Semantic aliasing and background computation overhead |
| w/o Bottom-Up (Changed to top-down) | Moderate Drop | Path errors for small objects cannot be recovered |

### Key Findings
- **Expert and Scanning Fallback for Each Other**: Neither performs as well alone as combined, proving the efficiency-robustness tradeoff can be resolved through cascading rather than architectural changes.
- **SGAP Gains Mainly from Small Objects**: When fixed grids split objects, subsequent reasoning receives incomplete tokens, which VL models struggle to reconstruct.
- **Bottom-Up + Annealing Threshold is Key to the Loop**: When the annealing stop condition is met, hard samples not found are fed back for a second round with the expert, forming a "Search-Assess-Research" loop, showing the most significant improvement for small/abstract targets.
- **Training-free is a Structural Advantage**: No modification to baseline MLLM weights means LLaVA-OV, Qwen2.5-VL, and InternVL2.5 are all plug-and-play, offering broader deployment potential.

## Highlights & Insights
- Using the MLLM's own Yes-token confidence as a scheduler reuses existing capabilities and avoids the cost of training an additional judge model. This logic is transferable to any search process requiring termination decisions.
- The "expert failure → trigger scanning" step translates detector limitations into useful internal signals, essentially reframing single-point failure as a rational state in multi-stage decision-making.
- The design of SGAP reveals an overlooked fact: patching strategies share the same space as subsequent VLM representations. Instead of introducing new patching features, reusing the already generated $\bm{H}_e$ is more efficient. This can be extended to any "patch-then-VLM" pipeline.
- The "strict then relaxed" scheduling of bottom-up search + annealing threshold is borrowed from classic heuristic search but fits naturally with MLLM uncertainty expression, potentially applicable to RAG / Agent multi-step retrieval tasks.

## Limitations & Future Work
- The entire pipeline is sensitive to the performance of SAM 3: If the expert drifts significantly in certain domains (e.g., medical, remote sensing), the fast path of cognitive switching degrades, and scanning costs rise.
- Information sufficiency $c_q$ relies on MLLM Yes-token probabilities; studies show these internal signals are often overconfident on OOD samples, potentially allowing the "direct answer" path to bypass necessary searches.
- The adaptive clustering interval $[k_\min,k_\max]=[4,8]$ and tree depth $D\in\{2,3\}$ are manually set. Optimal values for extreme aspect ratios or dense multi-target scenes require further tuning; automatic selection based on scene complexity could be explored.
- The double-edged sword of being training-free: The accuracy ceiling is largely determined by the fine-grained perception of the baseline MLLM. Search strategy optimization has limits; future work could use this cognitive workflow as a distillation signal to fine-tune baseline models.

## Related Work & Insights
- **vs. SEAL / DyFo (Expert-assisted)**: This work retains the "fast expert localization" idea but replaces simple box thresholds with SAM 3 + category alignment checks and connects failures to a scanning branch to avoid "expert failure = model blindness."
- **vs. ZoomEye / RAP / DC² (Scanning-based)**: Also uses tree search, but SGAP eliminates fixed grids, Visual Complexity prunes background nodes, and bottom-up traversal replaces top-down, directly addressing the "expensive + fragmented + error-prone" issues of scanning types.
- **vs. LLaVA-HR / High-res Encoders**: Avoids the engineering cost of modifying architectures, proving that search strategies alone can enable 7B models to approach 32B closed-source performance, which is particularly friendly to resource-constrained scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ Translates dual-pathway search into an "expert failure triggers semantic scanning" cascade. It is an integration of two visual search routes with high engineering elegance.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three baseline MLLMs + five benchmarks + horizontal comparisons with expert/scanning methods are comprehensive. Inclusion of OOD/medical/multilingual OCR would further strengthen it.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-aligned cognitive science motivation and formulas, though some ablation details require active retrieval from the appendix.
- Value: ⭐⭐⭐⭐ Training-free + plug-and-play + significant SOTA. Very friendly for industrial deployment and likely to be an unavoidable baseline for high-res VLMs within the year.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Self-Prophetic Decoding to Unlock Visual Search in LVLMs](self-prophetic_decoding_to_unlock_visual_search_in_lvlms.md)
- [\[ICCV 2025\] HRScene: How Far Are VLMs from Effective High-Resolution Image Understanding?](../../ICCV2025/multimodal_vlm/hrscene_how_far_are_vlms_from_effective_high-resolution_image_understanding.md)
- [\[ICLR 2026\] GLYPH-SR: Can We Achieve Both High-Quality Image Super-Resolution and High-Fidelity Text Recovery via VLM-Guided Latent Diffusion Model?](../../ICLR2026/multimodal_vlm/glyph-sr_can_we_achieve_both_high-quality_image_super-resolution_and_high-fideli.md)
- [\[ICML 2026\] CSMR (Look on Demand): A Cognitive Scheduling Framework for Visual Evidence Acquisition in Multimodal Reasoning](look_on_demand_a_cognitive_scheduling_framework_for_visual_evidence_acquisition_.md)
- [\[ICCV 2025\] FALCON: Resolving Visual Redundancy and Fragmentation in High-resolution Multimodal Large Language Models via Visual Registers](../../ICCV2025/multimodal_vlm/falcon_resolving_visual_redundancy_and_fragmentation_in_high.md)

</div>

<!-- RELATED:END -->
