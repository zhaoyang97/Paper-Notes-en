---
title: >-
  [Paper Note] Retrieving Counterfactuals Improves Visual In-Context Learning
description: >-
  [CVPR 2026][Causal Inference][visual in-context learning] This paper proposes CIRCLES, a framework that retrieves counterfactual demonstrations via attribute-guided composed image retrieval…
tags:
  - "CVPR 2026"
  - "Causal Inference"
  - "visual in-context learning"
  - "counterfactual reasoning"
  - "composed image retrieval"
  - "vision-language models"
  - "demonstration selection"
date: 2026-05-08
content_hash: 3cff3c09cbb2ed1b
---

# Retrieving Counterfactuals Improves Visual In-Context Learning

**Conference**: CVPR 2026
**arXiv**: [2603.16737](https://arxiv.org/abs/2603.16737)  
**Code**: [github.com/gzxiong/CIRCLES](https://github.com/gzxiong/CIRCLES)  
**Area**: Causal Inference
**Keywords**: visual in-context learning, counterfactual reasoning, composed image retrieval, vision-language models, demonstration selection

## TL;DR

This paper proposes CIRCLES, a framework that retrieves counterfactual demonstrations via attribute-guided composed image retrieval, constructing dual-channel in-context demonstrations combining causality and correlation to substantially improve fine-grained visual reasoning in VLMs.

---

## Background & Motivation

**Limitations of VLMs in fine-grained reasoning**: Vision-language models (VLMs) perform well on VQA and image captioning tasks, yet struggle in scenarios requiring discrimination of subtle visual attributes (e.g., plumage color differences in bird classification), often relying on spurious correlations rather than accurate reasoning.

**Key bottleneck in in-context learning**: ICL enables VLMs to rapidly adapt to new tasks via a small number of demonstrations, but its effectiveness is highly sensitive to the demonstration selection strategy — demonstration quality directly determines reasoning quality.

**Systematic deficiency of existing retrieval methods**: Similarity-based retrieval methods such as RICES tend to select visually similar examples that share irrelevant confounding attributes, causing models to learn surface-level correlations rather than genuine causal relationships.

**Fundamental distinction between correlation and causality**: Similarity-based retrieval finds images that "look alike," but cannot inform the model "which attribute change causes a label change" — which is precisely the core of causal reasoning.

**Fragility under data-scarce settings**: When relevant samples in the training set are limited, pure similarity-based retrieval suffers sharp performance degradation, lacking robustness.

**New opportunity for CIR techniques**: Composed Image Retrieval (CIR) was originally developed for retrieval tasks themselves; this paper is the first to employ it as a causal intervention tool to construct counterfactual demonstrations for ICL.

---

## Method

### Overall Architecture

CIRCLES (Composed Image Retrieval for Causal Learning Example Selection) comprises three modules: (1) a causal understanding channel based on attribute-guided CIR; (2) a correlation understanding channel based on standard image similarity; and (3) retrieval-augmented inference via dual-channel fusion. Given a query image and question, the two channels retrieve $k_{\text{causal}}$ and $k_{\text{corr}}$ demonstrations respectively, which are merged and provided as the ICL context for VLM inference.

### Key Design 1: Attribute-Guided Counterfactual Demonstration Retrieval

- **Function**: Performs "counterfactual intervention" on key attributes of the query image one at a time — holding all other attributes fixed while altering the target attribute value — and retrieves real images matching the counterfactual description as demonstrations.
- **Mechanism**: A VLM first extracts the decisive attribute-value pairs $\mathcal{A} = \{a_1, \dots, a_m\}$ from the query image. For each attribute $a_i$, an alternative value $v_i'$ is sampled, and the VLM generates a counterfactual description $c^{\text{do}(a_i=v_i')}$. CLIP then computes the image-text similarity $s_j^{\text{img}}$ between candidate images and this description. A question-to-question semantic similarity $s_j^{\text{txt}}$ is additionally incorporated as a constraint, and top-k candidates are selected via joint ranking.
- **Design Motivation**: The $\text{do}(\cdot)$ intervention isolates the causal effect of individual attributes, exposing the model to contrastive pairs of the form "changing attribute $X$ → label changes," thereby preventing spurious co-occurrence from misleading the model. The question similarity term ensures that retrieved results remain semantically consistent with the original task.

### Key Design 2: Correlation Retrieval Channel

- **Function**: Uses standard CLIP image-to-image cosine similarity to retrieve the $k_{\text{corr}}$ most similar examples to the query, providing global visual context.
- **Mechanism**: $s_j^{\text{corr}} = \mathbf{z}_q^{I\top} \mathbf{z}_j^I$, selecting top-k most similar samples directly.
- **Design Motivation**: Counterfactual demonstrations focus on attribute differences and may lack holistic visual pattern information; the correlation channel compensates by supplying contextual support for recognition and localization. The two channels are thus complementary.

### Key Design 3: CIR Implementation and Question Similarity Augmentation

- **Function**: OSrCIR (a training-free CIR method) is adopted as the counterfactual image retrieval engine, and a question-to-question text similarity term is added to the original CIR scoring.
- **Mechanism**: OSrCIR directly generates descriptions conditioned on the query image and modification text, yielding finer-grained results than CIReVL (which first describes and then edits). The term $s_j^{\text{txt}} = \mathbf{z}_q^{Q\top} \mathbf{z}_j^Q$ is incorporated to ensure that retrieved results remain relevant at the inference-task level.
- **Design Motivation**: CIR quality directly determines counterfactual demonstration quality. OSrCIR achieves approximately 5.4% higher accuracy than CIReVL on CUB. The question similarity term yields up to 14.3% EM improvement on datasets with diverse questions such as OK-VQA.

---

## Loss & Training

CIRCLES is a **training-free** framework:

- No fine-tuning or gradient updates are applied to the VLM.
- The CLIP encoder is frozen and used solely for pre-computing embeddings.
- The CIR module (OSrCIR) likewise requires no training.
- All computation is performed at inference time: attribute extraction → counterfactual description generation → retrieval → ICL inference.
- CLIP embeddings of training set samples can be pre-computed and stored; inference overhead primarily stems from VLM calls for attribute extraction and description generation.

---

## Key Experimental Results

### Table 1: Main Results (4 Datasets × 4 Models)

| Model | Method | CUB Acc | Flowers Acc | OK-VQA EM | VizWiz EM | Avg. EM |
|-------|--------|---------|-------------|-----------|-----------|---------|
| Gemma3-4B | RICES | 65.40 | 86.70 | 26.65 | 56.08 | 58.71 |
| Gemma3-4B | **CIRCLES** | **71.97** | **93.32** | **31.27** | **57.61** | **63.54** |
| Gemma3-12B | RICES | 76.37 | 96.44 | 36.86 | 73.98 | 70.91 |
| Gemma3-12B | **CIRCLES** | **77.03** | **97.77** | **37.75** | **74.30** | **71.71** |
| Qwen2.5-VL-3B | RICES | 72.26 | 93.06 | 42.57 | 70.80 | 69.67 |
| Qwen2.5-VL-3B | **CIRCLES** | **74.89** | **94.70** | **43.24** | **72.93** | **71.44** |
| Qwen2.5-VL-7B | RICES | 82.15 | 98.83 | 43.66 | 73.79 | 74.61 |
| Qwen2.5-VL-7B | **CIRCLES** | **82.17** | **98.99** | **43.54** | **77.63** | **75.58** |

### Table 2: Ablation on the Question Similarity Term (OK-VQA EM)

| Model | w/o Q-Q Similarity | w/ Q-Q Similarity | Relative Gain |
|-------|--------------------|-------------------|---------------|
| Gemma3-4B | 27.72 | 31.27 | +12.8% |
| Gemma3-12B | 33.02 | 37.75 | +14.3% |
| Qwen2.5-VL-3B | 41.12 | 43.24 | +5.2% |
| Qwen2.5-VL-7B | 40.80 | 43.54 | +6.7% |

**Other Key Findings**:

- Data-scarce experiment: When 75% of training samples are removed, the advantage of CIRCLES over RICES on Gemma3-4B grows from 10.05% to 16.28%.
- CIR method comparison: OSrCIR vs. CIReVL yields a relative accuracy improvement of 5.39%–5.56%.
- Budget allocation: With a total budget of 32 demonstrations, CIR 16 + IR 16 is the optimal configuration; under low budgets, spreading across more attributes is preferable, while under high budgets, focusing on fewer attributes is more effective.

---

## Highlights & Insights

- **Causal perspective introduced into ICL**: This is the first work to systematically integrate causal intervention into VLM in-context demonstration selection, elevating the paradigm from "find similar" to "find contrastive."
- **Training-free**: The entire framework is training-free and plug-and-play with any VLM, making it highly practical.
- **Substantial gains for smaller models**: Improvements are most pronounced for models with limited internal knowledge (Gemma3-4B, Qwen2.5-VL-3B), with average EM gains of ~8%, indicating that counterfactual demonstrations effectively compensate for limited model capacity.
- **Robustness under data scarcity**: The relative advantage of CIRCLES grows as available data decreases — a highly valuable property for real-world deployments.
- **Enhanced interpretability**: Counterfactual demonstrations intuitively illustrate "what changes → how the outcome changes," making the ICL process more transparent.

---

## Limitations & Future Work

- **Increased inference overhead**: Each query requires VLM calls for attribute extraction and counterfactual description generation, adding inference time and API call costs.
- **Dependence on VLM attribute extraction quality**: If the VLM itself cannot accurately identify key attributes, the causal reasoning foundation of the entire framework becomes unreliable.
- **Not rigorous causal identification**: The paper explicitly acknowledges that CIRCLES does not perform formal causal identification but rather approximate intervention — which may fail when complex interactions exist among attributes.
- **Diminishing returns on larger models**: Gains on Qwen2.5-VL-7B are already relatively modest, suggesting that stronger models have already internalized a degree of causal reasoning capacity.
- **Evaluation limited to classification and VQA**: Validation on more complex generative tasks (e.g., image captioning, visual grounding) is absent.

---

## Related Work & Insights

- **RICES / MUIER / MMICES** are the primary baselines, all based on similarity retrieval without considering causal structure — CIRCLES differentiates itself by introducing a counterfactual dimension.
- **Composed Image Retrieval (CIR)** methods including CIReVL and OSrCIR are innovatively repurposed from "retrieval tasks themselves" to "causal intervention tools" — this paradigm of applying existing techniques in novel contexts is a noteworthy methodological contribution.
- **Inspiration**: This paradigm can be generalized to other ICL scenarios (e.g., few-shot NLP, tool-use agents); the core idea is to replace pure similarity-based demonstrations with contrastive/counterfactual ones, enabling models to learn "change → effect" rather than "looks similar → answer is similar."

---

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MaskDiME: Adaptive Masked Diffusion for Precise and Efficient Visual Counterfactual Explanations](maskdime_adaptive_masked_diffusion_for_precise_and_efficient_visual_counterfactu.md)
- [\[NeurIPS 2025\] Do-PFN: In-Context Learning for Causal Effect Estimation](../../NeurIPS2025/causal_inference/do-pfn_in-context_learning_for_causal_effect_estimation.md)
- [\[NeurIPS 2025\] Cyclic Counterfactuals under Shift–Scale Interventions](../../NeurIPS2025/causal_inference/cyclic_counterfactuals_under_shift-scale_interventions.md)
- [\[ICCV 2025\] A Visual Leap in CLIP Compositionality Reasoning through Generation of Counterfactual Sets](../../ICCV2025/causal_inference/a_visual_leap_in_clip_compositionality_reasoning_through_gen.md)
- [\[ACL 2026\] Learning Invariant Modality Representation for Robust Multimodal Learning from a Causal Inference Perspective](../../ACL2026/causal_inference/learning_invariant_modality_representation_for_robust_multimodal_learning_from_a.md)

</div>

<!-- RELATED:END -->
