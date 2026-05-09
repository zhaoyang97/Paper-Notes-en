---
title: >-
  [Paper Note] Error Notebook-Guided, Training-Free Part Retrieval in 3D CAD Assemblies via Vision-Language Models
description: >-
  [ICLR 2026][Multimodal VLM][CAD part retrieval] This paper proposes a training-free two-stage VLM framework that records corrected reasoning trajectories in an Error Notebook and applies RAG-based test-time adaptation. On specification-driven part retrieval in 3D CAD assemblies, GPT-4o accuracy improves from 41.7% to 65.1% (+23.4%), with a further +4.5% gain from a grammar-constrained validator.
tags:
  - ICLR 2026
  - Multimodal VLM
  - CAD part retrieval
  - test-time adaptation
  - Error Notebook
  - RAG
  - training-free VLM inference
date: 2026-05-08
content_hash: 929caf6ebfe33a3c
---

# Error Notebook-Guided, Training-Free Part Retrieval in 3D CAD Assemblies via Vision-Language Models

**Conference**: ICLR 2026
**arXiv**: [2509.01350](https://arxiv.org/abs/2509.01350)
**Code**: None
**Area**: Multimodal VLM
**Keywords**: CAD part retrieval, test-time adaptation, Error Notebook, RAG, training-free VLM inference

## TL;DR
This paper proposes a training-free two-stage VLM framework that records corrected reasoning trajectories in an Error Notebook and applies RAG-based test-time adaptation. On specification-driven part retrieval in 3D CAD assemblies, GPT-4o accuracy improves from 41.7% to 65.1% (+23.4%), with a further +4.5% gain from a grammar-constrained validator.

## Background & Motivation
Specification-driven part retrieval in complex CAD assemblies is a core requirement for automating engineering workflows. However, directly applying LLMs/VLMs to this task faces critical challenges:

**Sequence length explosion**: CAD model metadata (e.g., STEP files) typically exceeds model token limits.

**Non-natural-language input**: CAD metadata is structured technical data rather than natural language.

**No fine-tuning access**: High-performance models such as GPT and Gemini generally do not expose fine-tuning interfaces.

**Difficulty with fine-grained reasoning**: Even when STEP data is converted to images, off-the-shelf models frequently misidentify parts, as the task requires precise reasoning over part relationships and attributes.

**Core Idea**: Drawing inspiration from training-time methods—where erroneous reasoning chains paired with corrected solutions teach models to reflect and self-correct—this work operationalizes the idea at inference time: an Error Notebook stores corrected reasoning trajectories, and RAG retrieves similar entries as few-shot exemplars to guide model inference, without any weight updates.

## Method

### Overall Architecture
A two-stage VLM pipeline:
- **Stage-1 VLM**: For each part in the assembly, takes the full assembly image and the individual part image as input and generates a descriptive text.
- **Stage-2 VLM**: Takes the assembly image, the part description mapping, and a natural-language specification as input; retrieves relevant entries from the Error Notebook via RAG; and performs CoT reasoning to output the target part.

### Key Designs
1. **Two-stage VLM strategy**: Stage 1 generates concise, discriminative descriptions for each part, $d_i = f_{desc}(\mathcal{I}_{assembly}, \mathcal{I}_{P_i}, prompt_{desc})$, organized as a JSON mapping. Stage 2 performs retrieval-based reasoning over the description mapping $\mathcal{D}$, the assembly image, and the specification: $\hat{\mathcal{P}}^* = f_{retr}(\mathcal{I}_{assembly}, \mathcal{D}, S, prompt_{retr})$. This design elegantly circumvents the token-length problem of raw STEP files.

2. **Error Notebook construction**: For samples where initial inference is incorrect, the VLM is prompted to self-reflect and produce a correction. Given a previous erroneous reasoning trace $R^{prev}$ and the ground-truth answer $\mathcal{P}^{*(gt)}$, the model generates a corrected trajectory $R^{corr}$ composed of three parts: (1) the correct steps up to the first error; (2) a reflection text (TR) that identifies and transitions away from the error; and (3) the reasoning steps from the correction point to the correct answer. Formally, $R^{corr} = R^{prev}_{sub} \oplus TR \oplus R^g$.

3. **Grammar-Constrained (GC) validator**: A deterministic validator checks the structural integrity of corrected trajectories, verifying that a final answer line is present, at least one filename is provided, and all predicted filenames belong to the permitted set. Two modes are defined: strict GC (sGC, requiring an explicit "Final Answer:" marker) and relaxed GC (rGC, tolerating the absence of the marker when reasoning is otherwise correct). This filtering mechanism further improves Error Notebook quality.

4. **RAG inference**: At test time, the similarity between the current query specification and each entry in the Error Notebook is computed; the top-$n$ most similar entries are retrieved as few-shot exemplars, and their corrected CoT trajectories are incorporated into the prompt. The current query itself is excluded to prevent data leakage.

### Loss & Training
The method is entirely training-free. VLMs are accessed via API calls, with exponential backoff retry (up to 3 attempts) for error handling. All assemblies are processed in parallel.

**Dataset construction pipeline**:
- Based on the Fusion 360 Gallery Dataset Assembly Dataset (archive a1.0.0\_00), comprising 752 assemblies.
- Step 1: GPT-4o generates discriminative noun-phrase descriptions for each part.
- Step 2: GPT-4o generates assembly specifications describing physical, spatial, and functional relationships between parts.
- Step 3: Human annotation—cases with overly similar part descriptions, assemblies where wholes and sub-components are indistinguishable, and ambiguous cases are removed.
- Final output: a multimodal CAD dataset with human-preference annotations.

## Key Experimental Results

### Main Results

**Model performance on the human-preference dataset:**

| Model | w/o E-Notebook | w/ E-Notebook | w/ E-Notebook + sGC |
|-------|---------------|--------------|---------------------|
| GPT-4o (Omni) | 41.7% | 65.1% (+23.4) | 66.8% (+25.1) |
| GPT-4o mini | 19.3% | 35.4% (+16.1) | 36.4% (+17.1) |
| Gemini 2.5 Pro | 54.0% | 59.5% (+5.5) | 62.1% (+8.1) |
| Gemini 2.0 Flash | 44.2% | 56.8% (+12.6) | 57.0% (+12.8) |

**Comparison with other training-free baselines (GPT-4o, human-preference dataset):**

| Method | Overall↑ | <10 parts | 10–20 parts |
|--------|----------|-----------|-------------|
| Standard few-shot | 37.7% | 42.9% | 29.4% |
| Self-consistency | 54.8% | 61.7% | 42.6% |
| **E-Notebook (Ours)** | **65.1%** | **75.5%** | **42.6%** |

### Ablation Study
- **Number of exemplars**: Varying the number of retrieved exemplars from 1 to 50 has minimal impact on final accuracy (~3% variation), indicating that the critical factor is the Error Notebook itself rather than the number of exemplars.
- **CoT vs. Non-CoT**: For simple assemblies (<10 parts), direct answers without CoT perform comparably or better; for complex assemblies (10–50 parts), CoT reasoning consistently outperforms the non-CoT baseline.
- **Open-source model validation**: Qwen2-VL-2B improves from 0.8% to 6.4% (+5.6%); under the cross-model setting (Error Notebook built with GPT-4o), accuracy further reaches 8.4% (+7.6%).
- **Cross-model GC setting**: The 2B model equipped with GPT-4o's Error Notebook trails GPT-4o mini by only ~4 percentage points on the <10-part group of the human-preference dataset.

### Key Findings
- Error Notebook gains are consistent across all models and part-count groups, including the highly challenging >50-part group.
- The strict GC mode may be overly harsh for smaller models (which are incorrectly filtered due to missing "Final Answer:" markers); the relaxed mode is necessary for such cases.
- The Error Notebook can serve as a distillation mechanism—transferring high-quality reasoning traces from a stronger model to a lightweight model without any fine-tuning.
- The number of RAG-retrieved exemplars has minimal effect on final accuracy (only ~3% variation between 1 and 50), confirming that the presence of the Error Notebook itself is the key factor.
- The additional inference latency introduced by the Error Notebook is negligible (8.04s vs. 6.50s), and the correction step is also lightweight (7.39s per sample).
- The Cloud Vision + Gemini 2.0 Flash combination (62.3%) outperforms pure Gemini 2.0 Flash (57.0%).

## Highlights & Insights
- Transferring the training-time paradigm of "error reflection + correction" to the inference stage is conceptually novel and practically effective.
- The two-stage VLM strategy elegantly resolves the token-length issue of CAD data by converting structured technical data into tractable natural-language descriptions.
- The cross-model Error Notebook demonstrates a new training-free knowledge transfer paradigm.

## Limitations & Future Work
- Error Notebook quality depends on the correction capability of the model used; weaker models may produce low-quality corrections.
- The framework is validated only on part retrieval; its applicability to broader engineering tasks (e.g., design verification, assembly planning) remains unexplored.
- The dataset scale (752 assemblies) is relatively limited; validation at larger scale would be valuable.
- The two-stage VLM incurs additional API call costs, particularly as Stage 1 requires a separate call per part.

## Related Work & Insights
- **Relation to Self-Consistency**: Self-consistency improves accuracy via multiple sampling and majority voting; the Error Notebook guides reasoning through high-quality corrected exemplars, proving more effective.
- **Relation to RAG**: Conventional RAG retrieves document-level knowledge; the Error Notebook retrieves corrected reasoning trajectories—shifting from *knowledge augmentation* to *reasoning augmentation*.
- **Insight**: The Error Notebook paradigm is generalizable to any VLM application scenario requiring complex reasoning.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of Error Notebook and test-time adaptation is novel, though individual components (CoT correction, RAG, GC) have prior precedents.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple commercial and open-source models with detailed ablations, but relies on a single dataset.
- Writing Quality: ⭐⭐⭐⭐ The framework is described clearly with rigorous formalization, though the paper is slightly verbose overall.
- Value: ⭐⭐⭐⭐ Strong engineering utility; the cross-model distillation idea is inspiring, though the application scope is relatively narrow.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Uncertainty-guided Compositional Alignment with Part-to-Whole Semantic Representativeness in Hyperbolic Vision-Language Models](../../CVPR2026/multimodal_vlm/uncertainty-guided_compositional_alignment_with_part-to-whole_semantic_represent.md)
- [\[ICCV 2025\] Exploiting Vision Language Model for Training-Free 3D Point Cloud OOD Detection](../../ICCV2025/multimodal_vlm/exploiting_vision_language_model_for_training-free_3d_point_cloud_ood_detection_.md)
- [\[AAAI 2026\] TOFA: Training-Free One-Shot Federated Adaptation for Vision-Language Models](../../AAAI2026/multimodal_vlm/tofa_training-free_one-shot_federated_adaptation_for_vision-language_models.md)
- [\[CVPR 2026\] LLMind: Bio-inspired Training-free Adaptive Visual Representations for Vision-Language Models](../../CVPR2026/multimodal_vlm/llmind_bio-inspired_training-free_adaptive_visual_representations_for_vision-lan.md)
- [\[AAAI 2026\] ReCAD: Reinforcement Learning Enhanced Parametric CAD Model Generation with Vision-Language Models](../../AAAI2026/multimodal_vlm/recad_reinforcement_learning_enhanced_parametric_cad_model_generation_with_visio.md)

</div>

<!-- RELATED:END -->
