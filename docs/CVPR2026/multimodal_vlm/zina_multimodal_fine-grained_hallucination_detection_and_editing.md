---
title: >-
  [Paper Note] Zina: Multimodal Fine-grained Hallucination Detection and Editing
description: >-
  [CVPR 2026][Multimodal VLM][Multimodal hallucination detection] Zina formalizes the task of multimodal fine-grained hallucination detection and editing, proposes a two-stage system (detector MLLM + reviewer MLLM) that delegates token copying to a deterministic function to reduce model burden, constructs the VisionHall dataset (6.9K human-annotated + 20K graph-based synthetic samples), and surpasses GPT-4o by 15.8 points in detection F1.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Multimodal hallucination detection
  - fine-grained editing
  - VLM evaluation
  - synthetic data
  - label taxonomy
date: 2026-05-08
content_hash: 36aa8af8d18e8684
---

# Zina: Multimodal Fine-grained Hallucination Detection and Editing

**Conference**: CVPR 2026
**arXiv**: [2506.13130](https://arxiv.org/abs/2506.13130)
**Code**: [https://yuiga.dev/zina](https://yuiga.dev/zina)
**Area**: Multimodal VLM
**Keywords**: Multimodal hallucination detection, fine-grained editing, VLM evaluation, synthetic data, label taxonomy

## TL;DR
Zina formalizes the task of multimodal fine-grained hallucination detection and editing, proposes a two-stage system (detector MLLM + reviewer MLLM) that delegates token copying to a deterministic function to reduce model burden, constructs the VisionHall dataset (6.9K human-annotated + 20K graph-based synthetic samples), and surpasses GPT-4o by 15.8 points in detection F1.

## Background & Motivation

**Background**: Multimodal large language models (MLLMs) frequently hallucinate in tasks such as image captioning, generating text that diverges from actual image content. Existing hallucination detection methods operate primarily at a coarse granularity: POPE uses Yes/No questions to detect object hallucinations, AMBER measures object-level error rates via the CHAIR metric, and MHalDetect performs three-way classification (hallucinated / non-hallucinated / partial).

**Limitations of Prior Work**: (1) Coarse-grained methods can only determine "this sentence contains a hallucination" but cannot precisely localize "which word/span is wrong" or "what type of error it is"; (2) existing methods perform detection only without correction, offering no actionable remediation; (3) hallucinations are diverse in form (object errors, color errors, quantity errors, relation errors, text errors, factual errors), yet most methods focus exclusively on object hallucinations.

**Key Challenge**: Fine-grained hallucination detection and editing requires a model to simultaneously accomplish three things—precisely localize spans, classify error types, and generate corrections. However, approaches such as FAVA, which require the model to copy the original sentence token by token while inserting labels, impose a triple burden: (i) faithfully reproducing the source text, (ii) determining label insertion positions token by token, and (iii) handling cascading errors caused by exposure bias. This combination places excessive demands on model capability.

**Goal**: (1) Formalize the task of multimodal fine-grained hallucination detection and editing and establish a six-category hallucination taxonomy; (2) design a two-stage approach that reduces task complexity; (3) construct high-quality training and evaluation datasets.

**Key Insight**: Offload the mechanical token-copying step from the language model to a deterministic function, allowing the model to focus exclusively on detection and editing, thereby substantially reducing task difficulty.

**Core Idea**: Decompose complexity through a two-stage decoupling strategy (detector localization + deterministic tagging + reviewer verification and editing), distributing the difficulty of hallucination detection and editing across specialized components.

## Method

### Overall Architecture
Given an image $x_{\text{img}}$, an MLLM-generated description $x_{\text{desc}}$, and a human reference description $x_{\text{ref}}$, Zina proceeds as follows: (1) the detector MLLM $\mathcal{M}_{\text{det}}$ receives the triplet as input and outputs a list of hallucinated spans together with their error types; (2) a deterministic function $\mathcal{T}$ inserts tags at the corresponding positions in the original text to produce an annotated sequence; (3) the reviewer MLLM $\mathcal{M}_{\text{rev}}$ verifies each tag for correctness and, in editing mode, generates correction suggestions. The final outputs are the hallucinated span set $\hat{\mathcal{Y}}_{\text{text}}$, the error-type set $\hat{\mathcal{Y}}_{\text{type}}$, and the edit-suggestion set $\hat{\mathcal{Y}}_{\text{edit}}$.

### Key Designs

1. **Deterministic Tagging Function $\mathcal{T}$**:

    - **Function**: Converts detector outputs into a structured tagged sequence.
    - **Mechanism**: $z_i = \mathcal{T}(x_{\text{desc}}, h_{\text{text}}^{(i)}, h_{\text{type}}^{(i)})$—given the original description, a hallucinated span, and its error type, the function deterministically inserts opening and closing tags of the corresponding type around the span (e.g., `<object>books</object>`). This is purely a string operation requiring no model inference.
    - **Design Motivation**: Prior methods such as FAVA require the model to generate tags token by token; a single omission (e.g., a missing character) can collapse all subsequent tag structure. Making this process deterministic eliminates the exposure bias inherent in autoregressive generation.

2. **Detector–Reviewer Two-Stage Architecture**:

    - **Function**: Decomposes the complex cognitive task of hallucination detection into two simpler sub-problems.
    - **Mechanism**: The detector $\mathcal{M}_{\text{det}}$ (based on Qwen2.5-VL-72B) is responsible solely for identifying "which spans are erroneous and of what type," without concern for tag formatting or source-text copying. The reviewer $\mathcal{M}_{\text{rev}}$ (also based on Qwen2.5-VL-72B) receives the pre-tagged sequence and only needs to determine "whether each tag's position and type are correct" and generate corrected text, functioning as a second-pass verifier and editor. Both models are trained with cross-entropy loss.
    - **Design Motivation**: Assigning detection, localization, and editing to a single model imposes excessive cognitive load. Splitting into two steps substantially reduces per-step complexity, analogous to a chain-of-thought decomposition strategy.

3. **Graph-based Synthetic Training Data Generation (GraphAug)**:

    - **Function**: Generates large-scale training samples that capture inter-error dependency structures.
    - **Mechanism**: Two sub-modules are employed: (a) Error Insertion (EI)—o3-mini injects errors into hallucination-free descriptions and records inter-error dependencies in XML format (e.g., "a non-existent apple is introduced" → "the apple's spatial relation to other objects is incorrectly described"); (b) Graph-based Augmentation (GraphAug)—error dependencies are modeled as a directed graph; cycles are detected and removed to obtain a DAG, after which nodes and their descendants are stochastically pruned with probability $p$ to generate training samples with varied error combinations.
    - **Design Motivation**: Real hallucinations are not independent—a single erroneous object triggers a cascade of downstream errors referencing it. Naive random error injection fails to model this dependency structure, causing a distributional mismatch between synthetic training data and real hallucinations.

### Loss & Training
Both the detector and reviewer are trained with standard cross-entropy loss. The detector is trained on synthetic data; the reviewer is trained on data annotated by the detector. Evaluation employs improved BERT-F1 and CLIP-F1 metrics computed via embedding similarity rather than exact match, as hallucination corrections admit multiple valid surface forms.

## Key Experimental Results

### Main Results (VisionHall Dataset)

| Method | Detection F1↑ | CLIP-S↑ | PAC-S↑ | BERT-F1↑ | CLIP-F1↑ |
|--------|--------------|---------|--------|----------|----------|
| GPT-4o | 29.37 | 65.58 | 73.86 | 24.89 | 30.19 |
| Qwen2.5-VL-72B | 21.31 | 64.38 | 72.99 | 18.85 | 23.67 |
| LLaVA-OV-72B | 25.70 | 65.74 | 73.91 | 20.81 | 26.81 |
| Llama-3.2-90B | 16.92 | 65.28 | 73.54 | 14.56 | 17.62 |
| **Zina (Ours)** | **45.15** | **66.08** | **74.36** | **44.02** | **50.39** |

### Ablation Study

| Configuration | Detection F1 | BERT-F1 | CLIP-F1 | Note |
|---------------|-------------|---------|---------|------|
| (i) No Reviewer, Qwen2.5-VL-72B 3-shot | 21.91 | 15.54 | 17.88 | Baseline |
| (ii) +Reviewer (32B) | 32.55 | 27.52 | 34.66 | +10.6 gain from Reviewer |
| (iii) +Reviewer (LLaVA-OV-72B) | 34.41 | 31.39 | 36.10 | Larger backbone |
| (iv) Zina, n=1 | 43.25 | 42.53 | 49.54 | Few-shot count has limited effect |
| (vi) Zina, n=3 (Full) | 45.15 | 44.02 | 50.39 | Full model |

### Key Findings
- **The reviewer contributes most**: F1 jumps from 21.91 to 32.55 from configuration (i) to (ii), demonstrating that the two-stage decoupling strategy is the primary driver of performance gains.
- **GPT-4o performs poorly on fine-grained detection**: Even the strongest closed-source model achieves only 29.37 F1, indicating that this task remains highly challenging for current MLLMs.
- **Error-type distribution analysis**: Object hallucinations are most frequent (~30–40%), while Fact hallucinations are rarest (<5%). Error distributions vary noticeably across models—GPT-4o exhibits a higher proportion of Text hallucinations (12.27%) compared to Qwen-7B (14.96%).
- **Strong performance on the out-of-domain MHaluBench dataset**: Zina outperforms baselines on 9 out of 10 metrics, demonstrating the generalizability of the approach.

## Highlights & Insights
- **Delegating token copying to a deterministic function is an elegant design**: It fundamentally frees the language model from the burden of "format compliance," allowing it to focus on "content understanding." This principle generalizes beyond hallucination detection to any task requiring local modifications while preserving source structure (e.g., grammatical error correction, fact verification).
- **Graph-structured error injection** is another highlight: modeling causal dependencies among errors as a DAG makes synthetic data more faithful to real hallucination distributions. The DAG pruning strategy naturally provides sample diversity.
- **BERT-F1 and CLIP-F1 metrics** address the "multiple valid corrections" problem in editing evaluation, offering a more principled alternative to exact-match F1.

## Limitations & Future Work
- **Dependence on human reference descriptions**: The task formulation assumes access to reliable reference captions, which are costly to obtain in practice. Future work could explore reference-free detection.
- **The six-category hallucination taxonomy may be incomplete**: For example, it lacks "causal relation errors" (incorrect causal inference between events) and "tense errors" (confusion of past/present/future).
- **Both detector and reviewer share the same architecture (Qwen2.5-VL-72B)**: The inference cost of 72B models is substantial; whether smaller models with distillation can achieve comparable performance warrants investigation.
- **VisionHall is built on DCI reference descriptions**: DCI images predominantly depict everyday scenes, providing insufficient coverage for hallucination detection in specialized domains such as medical imaging and remote sensing.

## Related Work & Insights
- **vs. FAVA**: FAVA is a text hallucination detection method that requires the model to copy the source text token by token while inserting tags. Zina's core improvement is replacing token copying with a deterministic function and extending the approach to the multimodal setting.
- **vs. UniHD**: UniHD first extracts verifiable claims and then validates them with external tools, resulting in a heavier pipeline dependent on tools such as object detectors and OCR. Zina is end-to-end and more lightweight.
- **vs. HalLocalizer**: HalLocalizer performs token-level localization but cannot guarantee that replacing detected tokens corrects the hallucination; Zina performs span-level localization, making detection outputs directly editable.
- **Insights**: The detector–reviewer two-stage paradigm generalizes to other self-refinement tasks, such as code bug detection and repair, and translation error detection and correction.

## Rating
- Novelty: ⭐⭐⭐⭐ The task formulation is novel (fine-grained detection + editing); deterministic tag insertion and graph-based data generation are meaningful contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Over 10 baselines compared; validated on both VisionHall and MHaluBench; ablation analysis is comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem definition is clear, motivation is well-motivated, and figures and tables are of high quality.
- Value: ⭐⭐⭐⭐ Provides fine-grained tooling for MLLM hallucination mitigation; the VisionHall dataset offers lasting utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ReasonMap: Towards Fine-Grained Visual Reasoning from Transit Maps](reasonmap_towards_fine-grained_visual_reasoning_from_transit_maps.md)
- [\[CVPR 2026\] MA-Bench: Towards Fine-grained Micro-Action Understanding](ma-bench_towards_fine-grained_micro-action_understanding.md)
- [\[CVPR 2026\] OddGridBench: Exposing the Lack of Fine-Grained Visual Discrepancy Sensitivity in Multimodal Large Language Models](oddgridbench_exposing_the_lack_of_fine-grained_visual_discrepancy_sensitivity_in.md)
- [\[CVPR 2026\] Concept-wise Attention for Fine-grained Concept Bottleneck Models](coat_cbm_concept_wise_attention.md)
- [\[CVPR 2026\] CropVLM: Learning to Zoom for Fine-Grained Vision-Language Perception](cropvlm_learning_to_zoom_for_fine_grained_vision_language_perception.md)

</div>

<!-- RELATED:END -->
