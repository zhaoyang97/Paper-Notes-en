---
title: >-
  [Paper Note] CRIT: Graph-Based Automatic Data Synthesis to Enhance Cross-Modal Multi-Hop Reasoning
description: >-
  [CVPR 2026][Multimodal VLM][Cross-Modal Reasoning] This paper proposes a graph-based automatic data generation pipeline that constructs the CRIT dataset and benchmark for training and evaluating VLMs on cross-modal multi…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Cross-Modal Reasoning"
  - "Multi-Hop Reasoning"
  - "Data Synthesis"
  - "Graph-Based Pipeline"
  - "VLM Benchmark"
date: 2026-05-08
content_hash: 84b305d7a4a422a2
---

# CRIT: Graph-Based Automatic Data Synthesis to Enhance Cross-Modal Multi-Hop Reasoning

**Conference**: CVPR 2026
**arXiv**: [2604.01634](https://arxiv.org/abs/2604.01634)  
**Code**: None  
**Area**: Multimodal VLM
**Keywords**: Cross-Modal Reasoning, Multi-Hop Reasoning, Data Synthesis, Graph-Based Pipeline, VLM Benchmark

## TL;DR

This paper proposes a graph-based automatic data generation pipeline that constructs the CRIT dataset and benchmark for training and evaluating VLMs on cross-modal multi-hop reasoning over interleaved image-text content. Models fine-tuned on CRIT achieve significant improvements on multiple benchmarks including SPIQA.

## Background & Motivation

Real-world reasoning frequently requires integrating information across modalities—for instance, following a DIY tutorial demands constant cross-referencing between textual instructions and accompanying illustrations. However, existing multimodal benchmarks suffer from critical shortcomings:

**Evaluation side**: Most benchmarks involve only a single image or a set of images, where answers can often be inferred from a single modality, failing to test genuine cross-modal reasoning.

**Training side**: Although large volumes of interleaved image-text data are used in pretraining, the fraction that truly requires complementary cross-modal reasoning is negligible.

**Model side**: Even SOTA models (e.g., GPT-4o) frequently produce hallucinations disconnected from visual or textual evidence when chain-of-thought reasoning is required.

Directly using VLMs to generate complex reasoning data introduces circular bias (the same class of model generates and evaluates the data) and hallucination issues. This paper uses graph structures as an intermediate representation, enabling QA pair generation **using only LLMs (no VLMs)**, thereby avoiding these problems.

## Method

### Overall Architecture

A three-stage automatic data generation pipeline:
1. **Multimodal Content Graph Construction**: Starting from images annotated with scene graphs, a unified graph-structured representation is built.
2. **Textual Context Generation**: Complementary textual descriptions are generated based on subgraphs.
3. **QA Pair Generation**: Cross-modal subgraph chains are sampled to generate QA pairs requiring multi-hop reasoning.

### Key Designs

1. **Multimodal Content Graph**: A directed graph $G=(\mathcal{V}, \mathcal{E})$ where nodes represent entities (visual objects or textual entities) and edges represent relations. Core operations:

    - Randomly sample 1–6 images annotated with scene graphs.
    - Rule-based filtering: retain only entities that can be uniquely identified via attributes or relations to avoid ambiguity.
    - LLM augmentation: generate new textual entities and relations for each image node to serve as cross-image bridging nodes.

2. **Textual Context Generation**: For each image, an associated subgraph is extracted, excluding image node attributes and cross-image relations (reserved for the model to infer from the image at reasoning time). An LLM then generates complementary text in diverse narrative styles (stories, diaries, documentaries, etc.). A key constraint ensures that text describes only the augmented textual nodes and their connections to image nodes, without leaking information that should be inferred from the images.

3. **QA Generation and Multi-Layer Filtering**:

    - Cross-modal subgraph chains containing 1–5 edges are sampled, with terminal nodes required to originate from images.
    - An LLM generates questions from serialized subgraph JSON and target answers, with the constraint that intermediate entities must not be explicitly mentioned in the question.
    - CoT reasoning chains are generated simultaneously.
    - Three-layer filtering: (a) discard samples where intermediate entities are explicitly mentioned in the question; (b) use 3 different LLMs to verify whether questions can be answered from a single modality; (c) prune excessively long CoT chains.

4. **Extension to Video and Scientific Papers**:

    - Video: Dense caption datasets are used; frames with high CLIP similarity to captions are selected, and an LLM converts captions into scene graphs.
    - Scientific papers: Paragraphs, figures, and tables are converted into a unified graph structure; visual entities are tagged and their corresponding descriptions removed from the text.

### Loss & Training

- LoRA-based SFT is applied to Qwen2.5-VL-7B and Idefics2-8B.
- Each training sample includes both direct-answer and CoT formats.
- Data generation LLM: Qwen3-30B-A3B-Instruct-2507.
- Filtering LLMs: Qwen3-30B + Gemma-3-27b-it + Mistral-Small-3.2-24B.

## Key Experimental Results

### Main Results

CRIT Benchmark results (CoT evaluation, EM/F1):

| Model | NI-EM | NI-F1 | VF-EM | VF-F1 | SP-EM | SP-F1 |
|-------|-------|-------|-------|-------|-------|-------|
| GPT-4o | 35.1 | 37.7 | 32.0 | 38.9 | 8.4 | 14.0 |
| Qwen2.5-VL-7B | 28.3 | 29.1 | 24.0 | 27.8 | 6.8 | 9.6 |
| Qwen2.5-VL-72B | 38.0 | 39.4 | 30.1 | 33.9 | 9.4 | 12.3 |
| **Qwen2.5-VL_CRIT** | **58.6** | **59.5** | **38.8** | **42.2** | **15.9** | **22.5** |
| **Idefics2_CRIT** | **54.1** | **54.9** | **31.2** | **33.9** | **12.3** | **20.2** |

Fine-tuned 7B models substantially outperform GPT-4o and the 72B model.

Cross-benchmark transfer results (Idefics2 + Mantis-Instruct + CRIT vs. Mantis-Instruct only):

| Benchmark | Metric | +CRIT | Mantis Only | Gain |
|-----------|--------|-------|-------------|------|
| SPIQA | METEOR | 10.53 | 3.60 | +192% |
| SPIQA | CIDEr | 67.93 | 23.83 | +185% |
| VEGA | ROUGE-L | 35.1 | 29.5 | +19% |
| MMQA | EM | 30.0 | 27.3 | +10% |
| FCMR | F1 | 50.5 | 44.9 | +12% |

### Ablation Study

| Configuration | NI-EM | VF-EM | SP-EM | Note |
|---------------|-------|-------|-------|------|
| No Fine-tuning | 28.3 | 24.0 | 6.8 | Baseline |
| CRIT (84k) | 58.6 | 38.8 | 15.9 | Standard training set |
| CRIT Augmented (210k) | 62.6 | 45.6 | 16.7 | Extended set; largest gain on video domain |

Model-generated annotation augmentation further improves performance, and the scientific paper domain also benefits from natural image/video domain data (cross-domain transfer).

### Key Findings

- **SOTA models perform poorly on cross-modal multi-hop reasoning**: GPT-4o achieves only 35.1% EM on the natural image domain and 8.4% on the scientific paper domain.
- **Error analysis** (75 GPT-4o error samples): 55% are evidence localization errors (the model retrieves the wrong image or text passage); visual perception errors occur at 4× the rate of textual comprehension errors.
- **Fine-tuning does not degrade general capabilities**: Adding CRIT training maintains or improves performance on general benchmarks such as MME and SeedBench.

## Highlights & Insights

1. **Graph structure as an intermediate representation is an elegant design**: Subgraph sampling programmatically enforces multi-hop and cross-modal constraints, yielding substantially higher data quality than directly prompting VLMs.
2. **No VLMs required throughout—only LLMs**: This avoids the circular bias inherent in using VLMs to generate VLM evaluation data.
3. **Single-modality filtering is cleverly designed**: Three distinct LLMs independently verify textual and visual modalities, ensuring that questions genuinely require cross-modal reasoning.
4. **The pipeline is highly extensible**: It generalizes from annotated images to video frames and scientific papers by adapting only the graph construction stage.

## Limitations & Future Work

- Performance on the scientific paper domain remains low (15.9% EM); precise cross-modal alignment over long text and complex figures remains challenging.
- Graph construction depends on existing scene graph annotations (GQA) or dense caption annotations (ActivityNet); applicability to fully unannotated settings requires further investigation.
- The current benchmark comprises only 1,446 manually verified test samples, which is relatively small in scale.
- The effect of CoT reasoning chain quality on training outcomes has not been explored.

## Related Work & Insights

- The paradigm of using graph-structured intermediate representations to drive LLM-based QA generation is transferable to other data synthesis tasks requiring structured reasoning.
- The "complementarity" constraint—excluding image attributes and cross-image relations from textual descriptions—is critical to ensuring cross-modal reasoning quality.
- Error analysis reveals that "evidence localization" is the primary bottleneck for current VLMs, rather than reasoning capability per se.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The graph-based data generation pipeline is elegantly designed and addresses the circular bias problem in data synthesis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-model comparisons, cross-benchmark transfer, data scaling, and error analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — The three-stage pipeline is described clearly; Fig. 2 conveys substantial information.
- **Value**: ⭐⭐⭐⭐⭐ — Pioneeringly defines and addresses the data and evaluation bottleneck for cross-modal multi-hop reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Multi-Crit: Benchmarking Multimodal Judges on Pluralistic Criteria-Following](multi-crit_benchmarking_multimodal_judges_on_pluralistic_criteria-following.md)
- [\[ICLR 2026\] Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional](../../ICLR2026/multimodal_vlm/multi-modal_data_spectrum_multi-modal_datasets_are_multi-dimensional.md)
- [\[ICCV 2025\] ReasonVQA: A Multi-hop Reasoning Benchmark with Structural Knowledge for Visual Question Answering](../../ICCV2025/multimodal_vlm/reasonvqa_a_multi-hop_reasoning_benchmark_with_structural_knowledge_for_visual_q.md)
- [\[AAAI 2026\] ImageBindDC: Compressing Multi-modal Data with ImageBind-based Condensation](../../AAAI2026/multimodal_vlm/imagebinddc_compressing_multi-modal_data_with_imagebind-based_condensation.md)
- [\[CVPR 2026\] DynamicGTR: Leveraging Graph Topology Representation Preferences to Boost VLM Capabilities on Graph QAs](dynamicgtr_leveraging_graph_topology_representation_preferences_to_boost_vlm_cap.md)

</div>

<!-- RELATED:END -->
