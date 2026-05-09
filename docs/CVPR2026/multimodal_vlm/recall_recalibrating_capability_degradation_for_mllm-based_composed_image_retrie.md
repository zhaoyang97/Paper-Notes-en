---
title: >-
  [Paper Note] ReCALL: Recalibrating Capability Degradation for MLLM-based Composed Image Retrieval
description: >-
  [CVPR 2026][Multimodal VLM][Composed Image Retrieval] This paper reveals a **Capability Degradation** phenomenon that occurs when adapting generative MLLMs into discriminative retrievers, and proposes the ReCALL framework — a three-stage pipeline that diagnoses retriever blind spots, leverages the base MLLM's CoT reasoning to generate corrective triplets, and applies grouped contrastive refinement to recover degraded fine-grained compositional reasoning ability. ReCALL achieves R@1 of 55.52% on CIRR and R@10 of 57.04% on FashionIQ.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Composed Image Retrieval
  - Capability Degradation
  - MLLM Self-Improvement
  - Contrastive Learning
  - Diagnose-Generate-Refine
date: 2026-05-08
content_hash: e9d9a364afae903b
---

# ReCALL: Recalibrating Capability Degradation for MLLM-based Composed Image Retrieval

**Conference**: CVPR 2026
**arXiv**: [2602.01639](https://arxiv.org/abs/2602.01639)
**Code**: [https://github.com/RemRico/Recall](https://github.com/RemRico/Recall)
**Area**: Multimodal Retrieval
**Keywords**: Composed Image Retrieval, Capability Degradation, MLLM Self-Improvement, Contrastive Learning, Diagnose-Generate-Refine

## TL;DR
This paper reveals a **Capability Degradation** phenomenon that occurs when adapting generative MLLMs into discriminative retrievers, and proposes the ReCALL framework — a three-stage pipeline that diagnoses retriever blind spots, leverages the base MLLM's CoT reasoning to generate corrective triplets, and applies grouped contrastive refinement to recover degraded fine-grained compositional reasoning ability. ReCALL achieves R@1 of 55.52% on CIRR and R@10 of 57.04% on FashionIQ.

## Background & Motivation

**Background**: Composed Image Retrieval (CIR) retrieves target images given a hybrid query consisting of a reference image and a modification text. Early dual-encoder VLM methods suffer from shallow cross-modal alignment and struggle with fine-grained compositional reasoning. Recent works have begun adapting MLLMs as retrievers, leveraging their deep fusion and instruction-following capabilities, and obtaining discriminative retrieval ability through contrastive fine-tuning.

**Limitations of Prior Work**: Compressing a generative MLLM (focused on step-by-step reasoning) into a single-embedding discriminative retriever (focused on vector similarity) introduces a **paradigm conflict** — fine-tuning degrades the model's native fine-grained reasoning capabilities (fine-grained localization, relational understanding). Experiments demonstrate that on 1k samples that the base MLLM can correctly answer via VQA, the fine-tuned retriever achieves only R@1 of 62.33% (CIRR) and 55.80% (FashionIQ), confirming that substantial existing capabilities are lost during adaptation.

**Key Challenge**: A fundamental conflict between the generative paradigm (emphasizing sequential reasoning with attention distributed across every token) and the discriminative paradigm (compressing all semantics into a single embedding vector). A single embedding cannot carry the fine-grained distinctions that MLLMs originally accomplish through multi-step reasoning.

**Goal**: How can the compositional reasoning capabilities degraded during fine-tuning be recovered while preserving the retrieval form (single embedding is mandatory)?

**Key Insight**: Rather than altering the retrieval paradigm itself, the paper uses the base MLLM's native reasoning signals to supervise the retriever's embedding space in reverse — "distilling reasoning capabilities from the MLLM into the retrieval space."

**Core Idea**: Diagnose failure cases of the retriever, use the base MLLM to generate minimally-edited corrective instructions for those failure cases to form new triplets, and then internalize fine-grained discriminative capabilities into the retriever via grouped contrastive learning.

## Method

### Overall Architecture
ReCALL is a model-agnostic four-stage framework. Stage 1 trains a baseline retriever ($\mathcal{R}_{\text{base}}$) from the base MLLM ($\mathcal{F}$) via standard contrastive learning. Stage 2 (Diagnose): $\mathcal{R}_{\text{base}}$ performs inference on the training set to surface failure cases and mine informative instances. Stage 3 (Generate): $\mathcal{F}$'s CoT reasoning generates corrective instructions for failure cases, filtered by VQA quality control. Stage 4 (Refine): grouped contrastive learning on original and corrective triplets produces the refined retriever $\mathcal{R}_{\text{refine}}$.

### Key Designs

1. **Self-Guided Informative Instance Mining (Stage 2: Diagnose)**

   - **Function**: Automatically discovers the retriever's "cognitive blind spots" — samples that are ranked highly by the retriever yet are actually incorrect.
   - **Mechanism**: $\mathcal{R}_{\text{base}}$ performs retrieval inference on the training set; queries where retrieval succeeds are filtered out (they already have sufficient discriminative power), and the pipeline focuses on failure cases. For each failure case, the Top-K images incorrectly ranked above the ground truth are extracted as informative instances $\{I_h\}$. These instances are "informative" because they share subtle visual/semantic similarity with the target, precisely exposing the degraded reasoning capabilities of the retriever.
   - **Design Motivation**: Compared to blind large-scale data synthesis (Random Mining), self-guided mining concentrates the generation budget precisely on the model's actual failure points, achieving high data efficiency.

2. **Generative Calibration (Stage 3: Generate)**

   - **Function**: Leverages the base MLLM's native reasoning capabilities to produce targeted corrective supervision signals.
   - **Mechanism**: (a) **CoT-assisted generation** — $\mathcal{F}$ performs two-step reasoning for each informative instance $I_h$: first decomposing the original instruction $T_m$ into atomic intents and verifying each intent on $(I_r, I_h)$, then retaining consistent intents and regenerating only the violated parts to yield the corrective instruction $\tilde{T}_m$. The resulting triplet $(I_r, \tilde{T}_m, I_h)$ has text edits that directly correspond to visual differences between $I_t$ and $I_h$. (b) **VQA quality control** — $\mathcal{F}$ is queried about key attributes in $\tilde{T}_m$, retaining only triplets with high confidence and internal consistency.
   - **Design Motivation**: The minimal-edit strategy preserves the original distribution while introducing precise fine-grained supervision — the difference between the corrective and original instructions exactly reflects the visual differences the retriever must learn to distinguish. VQA filtering ensures the generated supervision signals are reliable.

3. **Grouped Contrastive Refinement (Stage 4: Refine)**

   - **Function**: Efficiently internalizes corrective supervision signals into the retriever's embedding space.
   - **Mechanism**: A micro-group is constructed for each query, containing the original positive triplet $(I_r, T_m, I_t)$ and the corrective triplet $(I_r, \tilde{T}_m, I_h)$. A dual-objective optimization is applied: (a) **InfoNCE loss** preserves global structure; (b) **intra-group triplet margin loss** $\mathcal{L}_{\text{triplet}} = \max(0, s(z_q, z_{t^-}) - s(z_q, z_{t^+}) + m)$ explicitly enforces separation between the target and informative instances. Total loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{infoNCE}} + \lambda\mathcal{L}_{\text{triplet}}$.
   - **Design Motivation**: Placing the target alongside its confusable neighbors with subtly different instructions in the same batch forces the model to resolve the most challenging ambiguities in a single gradient update. This structured batching maximizes the transfer efficiency of corrective signals compared to random batching.

### Loss & Training
Qwen2.5-VL-7B is used as the backbone, fine-tuned with LoRA (rank=16). FashionIQ: lr=$4\times10^{-5}$, $\tau=0.03$, batch=512, Stage 1 200 steps + Stage 4 250 steps. CIRR: lr=$2\times10^{-5}$, $\tau=0.02$, Stage 1 300 steps + Stage 4 350 steps. Triplet margin $m=0.05$, $\lambda$=0.30 (FashionIQ) / 0.25 (CIRR).

## Key Experimental Results

### Main Results

| Dataset | Metric | ReCALL | Prev. SOTA (CIR-LVLM) | $\mathcal{R}_{\text{base}}$ | Gain (vs base) |
|--------|------|------|----------|------|------|
| CIRR test | R@1 | **55.52%** | 53.64% | 51.23% | +8.38% |
| CIRR test | R@5 | 84.07% | 83.76% | 82.15% | +2.34% |
| CIRR test | R_subset@1 | **81.49%** | 79.12% | 77.57% | +5.06% |
| FashionIQ val | Avg R@10 | **57.04%** | 56.21% | 53.23% | +7.16% |
| FashionIQ val | Avg R@50 | **76.42%** | 76.14% | 74.37% | +2.76% |
| FashionIQ Dress | R@10 | 51.81% | 50.42% | 46.80% | **+10.71%** |

### Ablation Study

| Configuration | Avg R@10 | Avg R@50 | Notes |
|------|---------|------|------|
| $\mathcal{R}_{\text{base}}$ | 53.23% | 74.37% | Baseline retriever |
| + CG (CoT Generation) | 55.41% | 75.17% | +2.18%, CoT supervision is effective |
| + VC (VQA Control) | 56.13% | 76.04% | +0.72%, noise filtering is effective |
| + GR (Grouped Refinement) | **57.04%** | **76.42%** | +0.91%, structured batching is key |

| Mining Strategy | Avg R@10 | Notes |
|------|---------|------|
| Random Mining | 53.80±0.20 | Blind synthesis yields only +0.57 |
| Self-Guided Mining | **57.04** | Targeted mining yields +3.81 |

### Key Findings
- **Large gap between Self-Guided and Random Mining**: Random Mining yields only +0.57% improvement at the same data volume, while Self-Guided Mining yields +3.81%. This demonstrates that *where* to generate data matters far more than *how much* data to generate.
- **Each component contributes incrementally**: CG (+2.18%) > GR (+0.91%) > VC (+0.72%); CoT-assisted generation is the primary driver.
- **Largest gain on the Dress category (+10.71%)**, as fine-grained differences in fashion dresses (sleeve length, neckline, pattern) are precisely where capability degradation is most severe.
- **Cross-backbone generalization**: ReCALL remains effective on the stronger Qwen3-VL-8B backbone (CIRR R@1: 55.93→57.09), confirming that capability degradation is a universal consequence of paradigm conflict rather than a defect of any specific model.

## Highlights & Insights
- **The proposal and quantitative validation of "capability degradation"** is the paper's most significant contribution. The contrastive experiment on the $\mathcal{F}$-solvable subset ($\mathcal{F}$ achieves 100% R@1 under VQA while $\mathcal{R}_{\text{base}}$ achieves only 62.33%) clearly quantifies the capability loss caused by the generative-to-discriminative paradigm shift. This finding has broad implications for all work that adapts generative models into retrievers.
- **The minimal-edit strategy** is elegant — the textual difference between the corrective and original instructions mirrors the visual difference between the target and informative instances, forming a symmetric supervision signal of "visual difference ↔ textual difference."
- **The Diagnose-Generate-Refine pipeline** constitutes a general model self-improvement paradigm, transferable to any MLLM-to-discriminative-model adaptation scenario, such as MLLM→classifier or MLLM→re-ranker.

## Limitations & Future Work
- Stages 2–4 form an offline, single-pass pipeline; iterative execution (diagnose→generate→refine→re-diagnose→...) could potentially yield further gains.
- The current approach relies on the quality of the base MLLM's CoT reasoning; if the base model itself struggles with certain fine-grained distinctions, it cannot generate effective corrections.
- VQA quality control performs only simple consistency checks; more fine-grained validation (e.g., measuring the semantic distance between generated $\tilde{T}_m$ and $T_m$) could further improve data quality.
- The small number of training steps (200–350) indicates high data efficiency, but also suggests room for further improvement with extended training.

## Related Work & Insights
- **vs. CIR-LVLM**: CIR-LVLM also adapts LVLMs as CIR retrievers but via single-stage static fine-tuning, without addressing capability degradation. ReCALL addresses this gap through a self-improvement loop.
- **vs. TME/CCIN**: These CVPR 2025 methods achieve approximately R@1 53.4% on CIRR, while ReCALL reaches 55.52%. The key difference is that ReCALL additionally distills reasoning capabilities from the base model.
- **vs. STaR/Self-Refine**: These LLM self-improvement methods iterate within generative tasks, whereas ReCALL is the first to apply the self-improvement paradigm to retrieval tasks, bridging generative reasoning and discriminative retrieval spaces.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Both the discovery of "capability degradation" and the Diagnose-Generate-Refine solution are original and generalizable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ State-of-the-art on two mainstream benchmarks, detailed ablations, cross-backbone validation, and qualitative analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is convincingly argued, experimental design is rigorous, and figures are clear.
- **Value**: ⭐⭐⭐⭐⭐ Offers a fundamental insight into MLLM retrieval adaptation with a highly generalizable framework.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] G-MIXER: Geodesic Mixup-based Implicit Semantic Expansion and Explicit Semantic Re-ranking for Zero-Shot Composed Image Retrieval](g_mixer_geodesic_mixup_based_implicit_semantic_expansion_for_zero_shot_cir.md)
- [\[CVPR 2026\] CoVR-R: Reason-Aware Composed Video Retrieval](covr-rreason-aware_composed_video_retrieval.md)
- [\[AAAI 2026\] Heterogeneous Uncertainty-Guided Composed Image Retrieval with Fine-Grained Probabilistic Learning](../../AAAI2026/multimodal_vlm/heterogeneous_uncertainty-guided_composed_image_retrieval_with_fine-grained_prob.md)
- [\[CVPR 2026\] Text-Only Training for Image Captioning with Retrieval Augmentation and Modality Gap Correction](text-only_training_for_image_captioning_with_retrieval_augmentation_and_modality.md)
- [\[CVPR 2026\] Predictive Regularization Against Visual Representation Degradation in Multimodal Large Language Models](predictive_regularization_against_visual_representation_degradation_in_multimoda.md)

<!-- RELATED:END -->
