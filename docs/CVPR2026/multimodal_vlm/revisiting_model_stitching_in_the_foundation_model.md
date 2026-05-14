---
title: >-
  [Paper Note] Revisiting Model Stitching in the Foundation Model Era
description: >-
  [CVPR 2026][Multimodal VLM][Model Stitching] This paper systematically investigates the feasibility of stitching heterogeneous Vision Foundation Models (VFMs), finds that conventional methods fail in this setting…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Model Stitching"
  - "Vision Foundation Models"
  - "Representation Compatibility"
  - "VFM Stitch Tree"
  - "Multimodal LLM"
date: 2026-05-08
content_hash: 118a6d1296204c29
---

# Revisiting Model Stitching in the Foundation Model Era

**Conference**: CVPR 2026
**arXiv**: [2603.12433](https://arxiv.org/abs/2603.12433)
**Code**: N/A
**Area**: Multimodal VLM / Model Fusion
**Keywords**: Model Stitching, Vision Foundation Models, Representation Compatibility, VFM Stitch Tree, Multimodal LLM

## TL;DR
This paper systematically investigates the feasibility of stitching heterogeneous Vision Foundation Models (VFMs), finds that conventional methods fail in this setting, and proposes a two-stage training strategy — **Final Feature Matching + Task Loss Training** — that enables reliable stitching across heterogeneous VFMs. The resulting stitched models can even surpass both constituent VFMs individually. Building on this, the paper introduces the **VFM Stitch Tree (VST)** architecture, which provides a controllable accuracy–efficiency trade-off for multi-VFM systems.

## Background & Motivation

1. **Background**: Vision foundation models (e.g., CLIP, DINOv2, SigLIP 2) pretrained under diverse objectives, datasets, and modality combinations have become the default backbones for downstream tasks. Multimodal systems (e.g., MoF-LLaVA, Cambrian-1) increasingly employ multiple VFMs simultaneously to capture complementary visual information.
2. **Limitations of Prior Work**:
   - Model stitching, used as a probe for measuring representational compatibility, has been shown to work for small models trained on the same dataset (e.g., ResNet-18 on CIFAR-10), but whether heterogeneous VFMs are stitchable remains unknown.
   - Conventional stitching training methods — Layer Feature Matching (LFM) and Task Loss Training (TLT) — fail on VFMs. The former causes accumulated intermediate matching errors that amplify final feature deviation, especially at shallow stitching points; the latter suffers from optimization difficulties when gradients must propagate through long chains of frozen layers.
   - Deploying multiple VFMs incurs linear computational and memory overhead ($k$ VFMs = $k\times$ cost), with no efficient sharing mechanism.
3. **Key Challenge**: VFMs differ substantially in pretraining data (LAION vs. LVD-142M vs. WebLI), objectives (contrastive learning vs. self-supervised reconstruction), and modality combinations (vision-only vs. vision-language), making it insufficient to bridge their intermediate representations with simple learned transformations.
4. **Goal**: ① Determine whether heterogeneous VFMs can be stitched; ② identify a reliable stitching training strategy; ③ elevate stitching from a diagnostic tool to a practical VFM fusion framework.
5. **Key Insight**: Systematic analysis of stitching failure modes (intermediate matching ≠ final alignment; gradient attenuation), followed by a targeted remedy.
6. **Core Idea**: Apply Final Feature Matching to align features at the penultimate layer of the target VFM as initialization, then fine-tune with Task Loss, enabling reliable stitching of heterogeneous VFMs while fusing their complementary knowledge.

## Method

### Overall Architecture
Given a source VFM $f_\theta$ and a target VFM $f_\phi$ (both $N$-layer Transformers), stitching at layer $n$ retains the first $n$ layers of the source model $R_\theta^n$ and the last $N-n$ layers of the target model $T_\phi^N$, connected by a trainable stitching layer $S$. The stitched model is defined as $F(x) = T_\phi^N \circ S \circ R_\theta^n(x)$, where only $S$ is trainable and all source/target layers are frozen.

### Key Designs

1. **Final Feature Matching (FFM)**

   - **Function**: Provides high-quality initialization for the stitching layer, ensuring final output features align with the target VFM.
   - **Mechanism**: Rather than matching intermediate features at the stitching point $n$, FFM directly minimizes the feature discrepancy at the final layer $N$ after passing through the stitched model:
     $$\mathcal{L}_{FFM} = \frac{1}{M}\sum_{i=1}^M \|T_\phi^N(S(R_\theta^n(x_i))) - T_\phi^N(R_\phi^n(x_i))\|_2^2$$
     Despite supervising only the final layer, FFM is empirically found to implicitly maintain low feature distances at intermediate layers as well, while achieving significantly smaller final feature distances than LFM.
   - **Design Motivation**: LFM yields very small errors at the stitching point (on the order of $10^{-3}$), yet these errors are amplified by subsequent frozen layers, causing severe final feature deviation — particularly at shallow stitching points. FFM directly optimizes the final outcome, addressing this failure from the root. Furthermore, FFM requires no labels and can be trained in a fully unsupervised manner.

2. **Two-Stage Training (FFM + Task Loss Training)**

   - **Function**: Stage 1 establishes a favorable loss landscape via FFM initialization; Stage 2 maximizes downstream task performance via task loss fine-tuning.
   - **Mechanism**: Stage 1 pretrains the stitching layer with FFM (label-free); Stage 2 fine-tunes the stitching layer with downstream task loss (e.g., cross-entropy for classification). This pipeline specifically resolves the optimization difficulty of TLT at shallow stitching points, where random initialization combined with weak gradient signals (propagated from pooled tokens through long frozen chains) leads to a poorly conditioned loss landscape. FFM initialization places the stitching layer at a favorable starting point.
   - **Design Motivation**: Directly applying TLT to shallow-layer DINOv2→SigLIP2 stitching yields only 25.1% accuracy, well below the individual linear probing baselines of the two models (46.7% and 53.5%). FFM initialization raises this to 51.7%, and FFM+TLT further improves it to 55.8% (Layer 6).

3. **Self-Stitch Baseline (Controlled Experiment)**

   - **Function**: Disentangles whether performance gains originate from stitching layer capacity or genuine VFM knowledge fusion.
   - **Mechanism**: Stitching is performed within a single VFM (e.g., SigLIP2→SigLIP2) using the same stitching layer, stitching point, training loss, and downstream data. If cross-VFM stitching surpasses self-stitching, the gain is attributable to true complementary knowledge fusion rather than additional parameters or capacity from fine-tuning.
   - **Design Motivation**: Since VFMs are pretrained on large-scale heterogeneous data and evaluated on downstream data, improvements may simply reflect adaptation of the stitching layer to downstream distributions (equivalent to extra fine-tuning parameters). The self-stitch baseline rules out this explanation. Experiments confirm that cross-VFM stitching consistently outperforms self-stitching (+2.3% to +2.6%), confirming genuine complementary fusion.

### Loss & Training
- **Stage 1**: FFM loss (label-free data); source and target features can be pre-extracted to accelerate training.
- **Stage 2**: Downstream task cross-entropy loss (labeled data).
- **Stitching layer**: Default is a 2-layer MLP with ReLU (identical to the feature projector in LLaVA-1.5).
- **Evaluated VFM pairs**: DINOv2-L, SigLIP2-L, CLIP, DINOv3 (all 24-layer Transformers).
- **Stitching points**: $n \in [2, 6, 10, 14, 18, 22]$.

## Key Experimental Results

### Main Results: Two-Stage Method vs. Vanilla Task Loss Training

| Stitching Pair | Init | L2 | L6 | L10 | L14 | L18 | L22 |
|---|---|---|---|---|---|---|---|
| DINOv2→SigLIP2 | None | 25.1 | 39.4 | 52.6 | 62.3 | 68.6 | 68.6 |
| DINOv2→SigLIP2 | FFM | **51.7** | **55.8** | **59.3** | **68.0** | **72.0** | **71.8** |
| SigLIP2→DINOv2 | None | 38.7 | 56.7 | 58.3 | 64.4 | 70.4 | 70.1 |
| SigLIP2→DINOv2 | FFM | **53.8** | **53.8** | **61.9** | **69.6** | **70.4** | **72.2** |

### Cross-Dataset / Cross-Task Consistency

| Configuration | fMoW (L6/14/22) | iNaturalist (L6/14/22) | Aircraft (L6/14/22) | ADE20K Seg (L14/22) |
|---|---|---|---|---|
| DINOv2→DINOv2 (self-stitch) | 41.5/59.7/69.9 | 56.9/81.5/91.2 | 37.8/79.3/91.2 | 35.4/50.9 |
| SigLIP2→SigLIP2 (self-stitch) | 50.5/62.0/68.9 | 71.2/88.5/87.3 | 67.9/88.1/89.3 | 44.5/50.5 |
| **DINOv2→SigLIP2** | **55.8/68.0/71.8** | **75.9/89.1/92.8** | **77.8/87.6/92.4** | **44.9/51.2** |
| **SigLIP2→DINOv2** | **53.8/69.6/72.2** | **86.3/88.9/91.9** | **80.7/89.0/91.0** | **49.0/51.4** |

### Ablation Study: Stitching Layer Type

| Stitching Layer | L2 | L6 | L10 | L14 | L18 | L22 |
|---|---|---|---|---|---|---|
| Linear | 26.1/50.3 | 54.3/56.4 | 59.5/60.0 | 66.5/65.7 | 69.1/69.6 | 69.6/71.9 |
| **MLP** | **51.7/53.8** | **55.8/53.8** | **59.3/61.9** | **68.0/69.6** | **72.0/70.4** | **71.8/72.2** |
| LoRA | 49.1/48.3 | 49.4/56.2 | 57.4/62.4 | 61.7/65.3 | 67.7/66.2 | 67.3/65.0 |

### Key Findings
- FFM initialization yields the most pronounced gains at shallow stitching points (L2: 25.1→51.7) and provides consistent improvements at deeper points as well (L22: 68.6→71.8).
- Cross-VFM stitching consistently outperforms self-stitching (+0.7% to +5.5%) on both classification and semantic segmentation tasks, confirming genuine complementary knowledge fusion.
- The MLP stitching layer achieves the best overall performance; LoRA, despite greater expressive capacity, underperforms MLP — possibly because a moderate representational mismatch facilitates complementary information fusion.
- When CLIP serves as the source model, stitching performs poorly (the weaker encoder loses task-critical information); when used as the target model, performance is strong — analogous to upgrading the encoder in an encoder–decoder architecture.
- VST-22 achieves 45% of the dual-VFM performance gain with only 4.3% additional resource overhead; VST-14 achieves 84% of the gain with 39% additional overhead.

## Highlights & Insights
- The unexpected finding that **FFM implicitly induces local intermediate alignment** is particularly insightful: although supervision is applied only at the final layer, the gradient signal implicitly propagates to intermediate layers to promote local alignment, suggesting that deep-layer matching can effectively constrain shallow representations.
- The **Self-Stitch baseline** reflects rigorous experimental methodology, thoroughly ruling out the alternative explanation that gains arise merely from additional parameters — a commendable example of responsible ablation design.
- The **accuracy–latency knob** concept of the VFM Stitch Tree is highly practical: rather than a binary choice of whether to use a second VFM, it enables continuous adjustment of additional overhead from 4.3% to 100%, accommodating diverse deployment budgets.
- Elevating model stitching from a purely diagnostic tool to a practical fusion framework represents a meaningful paradigm shift.

## Limitations & Future Work
- VST evaluation is conducted only on VQAv2 and MME as an "early exploration," and should be extended to a broader range of multimodal benchmarks (e.g., SEED-Bench, MMVet) for comprehensive assessment of fusion gains.
- Experiments are limited to ViT-L scale VFMs; the stitchability of larger models (e.g., ViT-G) or architecturally distinct VFMs remains to be verified.
- The FFM stage requires forward passes through VFMs on unlabeled data, which may incur non-trivial computational cost for very large models.
- Future work could explore adaptive stitching point selection (rather than manual specification) and tree designs involving more than two VFMs.
- FFM loss is label-free but still requires training on in-domain data; its effectiveness in zero-shot settings remains unknown.

## Related Work & Insights
- **vs. SN-Net [35]**: SN-Net explicitly designs stitchability during training for model compression; this paper performs post-hoc stitching of independently trained heterogeneous VFMs — an entirely different setting.
- **vs. [2] (Bansal et al.)**: The original stitching work identifies stitchability for same-dataset, same-architecture models (the Anna Karenina hypothesis); this paper extends the analysis to VFMs with heterogeneous data, objectives, and modalities, finding that naive methods fail but tailored methods succeed.
- **vs. [7] (Collins et al.)**: That work argues TLT is superior to LFM; this paper finds both are problematic on VFMs, and FFM serves as a more effective alternative.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — FFM and the two-stage scheme are concise yet effective; the VST application is novel; overall, the contribution lies primarily in careful engineering.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Systematic validation across multiple VFM pairs, datasets, tasks, and stitching layer types; the self-stitch controlled experiment is elegantly designed.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Logical flow progresses clearly from diagnosis to prescription to application; an exemplary research paper structure.
- **Value**: ⭐⭐⭐⭐ — Makes important contributions to understanding VFM representational compatibility; VST offers a practical solution for multi-VFM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] AnomalyVFM -- Transforming Vision Foundation Models into Zero-Shot Anomaly Detectors](anomalyvfm_--_transforming_vision_foundation_models_into_zero-shot_anomaly_detec.md)
- [\[CVPR 2026\] VL-RouterBench: A Benchmark for Vision-Language Model Routing](vl-routerbench_a_benchmark_for_vision-language_model_routing.md)
- [\[CVPR 2026\] Medic-AD: Towards Medical Vision-Language Model's Clinical Intelligence](medic-ad_towards_medical_vision-language_models_clinical_intelligence.md)
- [\[CVPR 2026\] UniGame: Turning a Unified Multimodal Model Into Its Own Adversary](unigame_turning_a_unified_multimodal_model_into_its_own_adversary.md)
- [\[ICCV 2025\] Feather the Throttle: Revisiting Visual Token Pruning for Vision-Language Model Acceleration](../../ICCV2025/multimodal_vlm/feather_the_throttle_revisiting_visual_token_pruning_for_vision-language_model_a.md)

</div>

<!-- RELATED:END -->
