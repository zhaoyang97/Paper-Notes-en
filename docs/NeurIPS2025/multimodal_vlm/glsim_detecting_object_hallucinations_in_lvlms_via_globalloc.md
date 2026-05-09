---
title: >-
  [Paper Note] GLSim: Detecting Object Hallucinations in LVLMs via Global-Local Similarity
description: >-
  [NeurIPS 2025][Multimodal VLM][object hallucination] GLSim is a training-free object hallucination detection method for LVLMs that combines a global scene similarity score (cosine similarity between the object token and the last instruction token) and a local visual grounding similarity score (cosine similarity between the object token and the Top-K image patch embeddings localized via Visual Logit Lens). It achieves 83.7% AUROC on MSCOCO, surpassing SVAR by 9% and Internal Confidence by 10.8%.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - object hallucination
  - hallucination detection
  - global-local similarity
  - visual logit lens
  - training-free
date: 2026-05-08
content_hash: 5f238babc4c095f5
---

# GLSim: Detecting Object Hallucinations in LVLMs via Global-Local Similarity

**Conference**: NeurIPS 2025  
**arXiv**: [2508.19972](https://arxiv.org/abs/2508.19972)  
**Code**: [https://github.com/deeplearning-wisc/glsim](https://github.com/deeplearning-wisc/glsim)  
**Area**: Multimodal VLM / Hallucination Detection  
**Keywords**: object hallucination, hallucination detection, global-local similarity, visual logit lens, training-free

## TL;DR
GLSim is a training-free object hallucination detection method for LVLMs that combines a global scene similarity score (cosine similarity between the object token and the last instruction token) and a local visual grounding similarity score (cosine similarity between the object token and the Top-K image patch embeddings localized via Visual Logit Lens). It achieves 83.7% AUROC on MSCOCO, surpassing SVAR by 9% and Internal Confidence by 10.8%.

## Background & Motivation

- **State of the Field**: Large vision-language models (LVLMs) are prone to object hallucinations—generating descriptions of objects that do not exist in the image—which severely undermines reliable deployment in high-stakes domains such as medical imaging and autonomous driving.
- **Limitations of Prior Work**: Existing hallucination detection methods either rely on external annotated data (e.g., CHAIR), require external LLM judges (e.g., FaithScore), or exploit only a single-perspective signal. Token-probability-based methods (NLL) fail because LLMs favor linguistic fluency; attention-based methods (SVAR) are susceptible to attention sinks; and Internal Confidence, which directly uses the maximum probability from Visual Logit Lens, can be overconfident.
- **Root Cause**: A single global or local signal each has blind spots. Global methods may falsely accept contextually plausible but visually absent objects (e.g., "dining table" in a birthday scene); local methods may be confused by visually similar objects (e.g., a motorcycle seat mistaken for a "handbag").
- **Starting Point**: This work is the first to unify global and local embedding similarity signals within a single framework, leveraging their complementary strengths.

## Method

### Overall Architecture
GLSim is a training-free, object-level hallucination detection framework. For each object $o$ mentioned in the LVLM-generated text, two scores are computed: (1) a **global similarity** score—cosine similarity between the object embedding and the scene embedding; and (2) a **local similarity** score—average cosine similarity between the object embedding and the Top-K image patch embeddings localized via Visual Logit Lens. The final GLSim score is a weighted combination of the two.

### Key Designs

1. **Unsupervised Object Localization via Visual Logit Lens**

   - **Function**: Localizes the image regions most relevant to a given object without relying on external annotations or detectors.
   - **Mechanism**: The hidden representation $h_l(v_i)$ of each visual token $v_i$ at decoder layer $l$ is projected into the vocabulary space via the unembedding matrix $W_U$, yielding the probability $\text{softmax}(\text{VLL}_l(v_i))[o]$ that each visual patch predicts object word $o$. The Top-K patches with the highest probabilities are selected as the localization region $I(o)$.
   - **Design Motivation**: Visual Logit Lens localizes objects more accurately than attention weights (experiments show a 12.5% AUROC improvement) and requires no external detectors.

2. **Local Similarity Score**

   - **Function**: Verifies whether the object has genuine visual evidence in a specific region of the image.
   - **Mechanism**: Computes the average cosine similarity between the object token embedding $h_{l'}(o)$ and the hidden representations $h_l(v_i)$ of the Top-K localized patches: $s_\text{local} = \frac{1}{K}\sum_{v_i \in I(o)} \text{sim}(h_l(v_i),\, h_{l'}(o))$. Regions corresponding to real objects yield high similarity, while hallucinated objects map to irrelevant regions with low similarity.
   - **Design Motivation**: Using embedding similarity is more stable than using raw Logit Lens probability values, which can be overconfident (as observed with Internal Confidence). The embedding space provides a finer-grained signal.

3. **Global Similarity Score**

   - **Function**: Assesses whether the object is semantically consistent with the overall scene.
   - **Mechanism**: Computes the cosine similarity between the object token embedding and the hidden representation of the last token of the instruction prompt: $s_\text{global} = \text{sim}(h_l(v, t),\, h_{l'}(o))$. The last instruction token encodes the model's integrated understanding of both the image and textual context.
   - **Design Motivation**: The last instruction token captures scene semantics more effectively than the "last image token" or the "average of all image tokens" (ablation shows an 8% AUROC improvement), providing a high-level judgment of whether an object is plausible in the scene.

### Loss & Training
GLSim is entirely training-free and directly exploits the internal representations of the LVLM. The final score is $\text{GLSim} = w \cdot s_\text{global} + (1-w) \cdot s_\text{local}$, where $w = 0.6$ consistently achieves the best performance across settings. Layer indices $l$ and $l'$ are selected via ablation (LLaVA: $l=32$, $l'=31$; Shikra: $l=30$, $l'=27$).

## Key Experimental Results

### Main Results

| Dataset / Model | Metric | GLSim | SVAR | Internal Conf. | Contextual Lens | NLL |
|---|---|---|---|---|---|---|
| MSCOCO / LLaVA-7B | AUROC | **83.7** | 74.7 | 72.9 | 75.4 | 63.7 |
| MSCOCO / LLaVA-13B | AUROC | **84.8** | 75.2 | 71.0 | 78.7 | 63.1 |
| MSCOCO / MiniGPT-4 | AUROC | **87.0** | 83.6 | 75.7 | 84.9 | 59.4 |
| MSCOCO / Shikra | AUROC | **83.0** | 70.7 | 69.1 | 69.5 | 60.4 |
| Objects365 / LLaVA-7B | AUROC | **72.6** | 64.9 | 68.7 | 63.2 | 62.9 |
| Objects365 / MiniGPT-4 | AUROC | **74.8** | 71.0 | 68.5 | 70.2 | 56.7 |

### Ablation Study

| Configuration | LLaVA AUROC | Shikra AUROC | Notes |
|---|---|---|---|
| Global only ($s_\text{global}$) | 79.3 | 78.9 | Already outperforms all baselines |
| Local Top-K only ($s_\text{local}$) | 78.8 | 76.8 | Complementary to global |
| GLSim (global + local Top-K) | **83.7** | **83.0** | Fusion gain: +4.9 / +6.2 |
| Localization: Attention | 66.3 (local) | 65.0 | Attention weights unreliable |
| Localization: Cosine Sim | 76.2 (local) | 70.1 | Sub-optimal |
| Localization: Logit Lens | **78.8** (local) | **76.8** | Best localization |
| $w = 0.4$ | 82.5 | — | Biased toward local |
| $w = 0.6$ | **83.7** | — | Optimal balance |
| $w = 0.8$ | 82.0 | — | Biased toward global |
| $K = 8/16/32/64$ | 82→83→**83.7**→82 | — | $K=32$ optimal (~6% of image tokens) |

### Key Findings
- GLSim consistently outperforms all baselines across all LVLM and dataset combinations, with particularly large gains on Shikra (+12.7% AUROC vs. SVAR).
- Global and local signals are genuinely complementary: each individually already surpasses all prior methods, and their fusion yields further improvement.
- Internal Confidence can be overconfident for hallucinated objects, as Visual Logit Lens probabilities may assign high scores to incorrect regions.
- Visual Logit Lens outperforms attention weights by 12.5% and cosine similarity by 2.6% as a localization method.
- Optimal layer selection falls in the late intermediate layers rather than the final layer, supporting the observation that the best task-specific representations do not necessarily reside in the last layer.

## Highlights & Insights
- The global-local complementarity idea is intuitive yet highly effective, and this work is the first to demonstrate its value in object hallucination detection.
- The training-free, plug-and-play design requires no additional training or external models, relying entirely on the LVLM's internal representations.
- The paper provides a comprehensive benchmark, systematically comparing five object-level hallucination detection methods for the first time and filling a notable gap in the field.
- The qualitative analysis is highly intuitive: Figure 2 clearly illustrates complementary failure cases where the global signal fails but the local succeeds, and vice versa.

## Limitations & Future Work
- The method only addresses object existence hallucinations and does not handle attribute-level (color, size) or relation-level (spatial position) hallucinations.
- Although $K$ and $w$ are empirically robust, their optimal values may vary with input resolution.
- Selecting appropriate layer indices $l$ and $l'$ requires per-model ablation.
- How to leverage GLSim scores to correct or mitigate hallucinations after detection remains an open direction.

## Related Work & Insights
- **CHAIR**: A hallucination evaluation metric based on ground-truth matching; GLSim requires no annotations.
- **Internal Confidence**: Uses Visual Logit Lens probabilities; GLSim improves upon this by adopting embedding similarity with Top-K aggregation.
- **SVAR**: An attention-weight-based method susceptible to attention sinks and text-token bias.
- The global-local complementarity paradigm is transferable to other detection tasks; GLSim can also be combined in series with training-time hallucination mitigation methods (e.g., Causal-LLaVA).

## Rating
- **Novelty**: ⭐⭐⭐⭐ — First systematic integration of global and local signals for hallucination detection, with a novel application of Visual Logit Lens.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 5 LVLMs × 2 datasets × 5 baselines × extensive ablations over $K$, $w$, layer selection, localization method, and global design; highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation and methodology are clearly presented; qualitative analysis is intuitive and effective.
- **Value**: ⭐⭐⭐⭐ — The training-free, plug-and-play nature makes it highly practical; code is open-sourced.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Global and Local Entailment Learning for Natural World Imagery](../../ICCV2025/multimodal_vlm/global_and_local_entailment_learning_for_natural_world_imagery.md)
- [\[ICCV 2025\] Mitigating Object Hallucinations via Sentence-Level Early Intervention](../../ICCV2025/multimodal_vlm/mitigating_object_hallucinations_via_sentence-level_early_intervention.md)
- [\[NeurIPS 2025\] When Semantics Mislead Vision: Mitigating Large Multimodal Models Hallucinations](when_semantics_mislead_vision_mitigating_large_multimodal_models_hallucinations_.md)
- [\[NeurIPS 2025\] Intervene-All-Paths: Unified Mitigation of LVLM Hallucinations across Alignment Formats](intervene-all-paths_unified_mitigation_of_lvlm_hallucinations_across_alignment_f.md)
- [\[NeurIPS 2025\] Video-SafetyBench: A Benchmark for Safety Evaluation of Video LVLMs](video-safetybench_a_benchmark_for_safety_evaluation_of_video_lvlms.md)

<!-- RELATED:END -->
