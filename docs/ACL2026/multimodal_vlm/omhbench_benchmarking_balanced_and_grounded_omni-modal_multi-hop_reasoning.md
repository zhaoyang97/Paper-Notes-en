---
title: >-
  [Paper Note] OMHBench: Benchmarking Balanced and Grounded Omni-Modal Multi-Hop Reasoning
description: >-
  [ACL 2026][Multimodal VLM][Omni-modal reasoning] OMHBench constructs a 6,144-question omni-modal three-hop reasoning benchmark covering text, image…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Omni-modal reasoning"
  - "Multi-hop QA"
  - "Speech understanding"
  - "Path balance"
  - "Benchmark"
date: 2026-05-08
content_hash: b898b6b3efd68efe
---

# OMHBench: Benchmarking Balanced and Grounded Omni-Modal Multi-Hop Reasoning

**Conference**: ACL 2026  
**arXiv**: [2508.16198](https://arxiv.org/abs/2508.16198)  
**Code**: No public code (Dataset: https://huggingface.co/datasets/HYU-NLP/OMHBench)  
**Area**: Multi-modal VLM  
**Keywords**: Omni-modal reasoning, Multi-hop QA, Speech understanding, Path balance, Benchmark

## TL;DR
OMHBench constructs a 6,144-question omni-modal three-hop reasoning benchmark covering text, image, and speech contexts. Through entity-attribute chains and six balanced reasoning paths, it exposes systematic shortcomings of current MLLMs in speech grounding, path robustness, and cross-modal grounding.

## Background & Motivation
**Background**: Multi-modal large language models (MLLMs) have evolved from bi-modal (text-image or text-speech) models toward omni-modal models capable of processing text, vision, and speech simultaneously. Evaluation benchmarks are generally divided into two categories: Omni-Modal Understanding (OMU) benchmarks, which emphasize the ability to receive all three modalities, and Cross-Modal Multi-Hop Reasoning (CMR) benchmarks, which emphasize composing evidence across modalities for multi-hop reasoning.

**Limitations of Prior Work**: Both categories have blind spots. In OMU datasets, while images and speech are provided, text often appears only in the question or options, allowing many problems to be solved by bypassing a specific modality, thus creating a "modality shortcut." CMR datasets emphasize reasoning chains but mostly cover only text and images, lacking speech. Furthermore, reasoning path distributions in prior work are highly unbalanced; for instance, many questions follow a fixed I-T or T-I order, meaning high scores on a single path do not necessarily reflect true cross-modal reasoning capability.

**Key Challenge**: Omni-modal evaluation seeks to simultaneously examine "whether all modalities are used" and "whether multi-hop reasoning can be executed stably." Existing datasets often satisfy only one requirement. Including three modalities does not guarantee that the model must use all three; similarly, multi-hop QA does not ensure fair path distribution or reliable coverage of different modality sequences.

**Goal**: The authors aim to construct a more controlled benchmark: first, questions must depend on evidence from text, image, and speech; second, each question has an explicit three-hop entity-attribute reasoning chain; third, the same question can be instantiated into six modality sequences to check for path sensitivity; and fourth, the answer format is explicit for reproducible exact match evaluation.

**Key Insight**: Multi-hop reasoning is abstracted as "entities shared across modalities, but attributes visible only in one." For example, a model might first find a company based on "goodwill" in an image chart, then read "inventory" in text, and finally aggregate a numerical value found in speech. This ensures each step lands on a specific modality, allowing explicit control over the reasoning path.

**Core Idea**: Three attribute tables for the same set of entities are used to generate three-hop questions. These three tables are then converted into text, image, and speech contexts. By applying permutations to the assignment of modalities to the three tables, the benchmark provides content-equivalent evaluations with different reasoning paths.

## Method
OMHBench proposes a pipeline for constructing an omni-modal multi-hop reasoning benchmark rather than a new model architecture. Its key lies in decoupling "problem semantics," "modal carrier form," and "reasoning path." While the underlying entities and answers remain constant, evidence is placed in different modalities, forcing the model to complete reasoning along a specified modality order.

### Overall Architecture
The pipeline consists of four steps:

1.  **Table Triplet Formation**: Real table data is collected from four domains: finance, economics, climate, and nutrition (sources include Yahoo Finance, World Bank, Open-Meteo, and USDA). For each sample, three small tables are constructed; they share the same set of entities but have distinct attributes. Each table contains 10 entities and 3 attributes, mixed with distractor entities and attributes.

2.  **Multi-Hop QA Construction**: Three-hop questions are generated from the table triplets. The first two hops typically handle entity localization or filtering, while the last hop handles reading or aggregating the answer. The system defines 8 operations—Lookup, Ranking, Comparison, Range, Proximity, Retrieval, Mean, and Summation—combining them into "Connect" and "Reasoning" subsets.

3.  **Omni-Modal Context Generation**: The three tables are converted into three contexts: text, image, and speech. Text is rewritten by LLMs into natural language reports or news; images are generated using Matplotlib and Seaborn as various charts; speech is synthesized using Kokoro-82M TTS based on multi-speaker scripts.

4.  **Reasoning Diversification**: Permutations of the mapping from the three tables to the three modalities result in six reasoning paths: S-I-T, S-T-I, I-S-T, T-S-I, I-T-S, and T-I-S. While the underlying answer remains unchanged, the sequence of modalities the model must visit varies, allowing for comparison of path-dependent performance.

### Key Designs
1.  **Entity-attribute 3-hop reasoning modeling**:
    *   **Function**: Constrains omni-modal reasoning into clear three-hop chains centered on entities and attributes, preventing models from relying on commonsense or a single modality.
    *   **Mechanism**: Entities are shared across tables, but attributes are distributed. When tables are mapped to text, image, and speech, the model must track the same entity across modalities. The "Connect" subset uses Lookup-Comparison-Retrieval for single-entity chains; the "Reasoning" subset allows complex operations (e.g., Mean, Summation) for set filtering and numerical aggregation.
    *   **Design Motivation**: This setup converts "did the model use the modality" into a structural constraint. If an intermediate attribute exists only in speech or image, skipping that modality makes it impossible to consistently reach the correct answer.

2.  **Controllable conversion from table triplets to omni-modal contexts**:
    *   **Function**: Presents the same structured knowledge in different modalities while maintaining factual consistency and expressive diversity.
    *   **Mechanism**: Text uses 24 domain-specific prompts; images use 10 chart types and various fonts/colors; speech uses 22 scenarios and four-speaker dialogues with 27 TTS voices. Table reconstruction and factoid QA checks achieved 100% factual consistency after conversion.
    *   **Design Motivation**: Mechanical conversion is too monotonous, while free generation risks factual drift. Using real tables as a foundation and controlled generation for style diversity balances scalability and reliability.

3.  **Path balance and robustness metrics**:
    *   **Function**: Prevents any single reasoning path from dominating the dataset and quantifies whether models are robust across different path versions of the same content.
    *   **Mechanism**: Each question is replicated across $3! = 6$ permutations. The dataset includes 6,144 questions (1,024 per path). The Path Balance Score (PBS) is defined such that for the $i$-th question group and $j$-th path, success $a_{i,j}$ only contributes to the PBS if $\sum_j a_{i,j}=6$.
    *   **Design Motivation**: Average accuracy only shows overall performance on a mixed distribution. PBS strictly checks cross-path consistency. If a model is strong on S-T-I but weak on I-T-S, PBS will reveal this asymmetric grounding.

### Loss & Training
No new model was trained. Evaluation uses zero-shot chain-of-thought prompting without a fixed format. For models with a reasoning mode, an 8,192-token thinking budget was set. Outputs were parsed into discrete numerical answers for Exact Match (EM) evaluation. Comparison with human judgment and LLM-as-a-Judge on 600 samples showed 100% agreement with EM due to the nature of positive integer answers.

## Key Experimental Results

### Main Results
Thirteen MLLMs were evaluated, including Gemini series (closed-source) and Qwen3-Omni, Phi-4 Multimodal, Qwen2.5-Omni, OmniVinci, and others (open-source).

| Model | OMHBench-Connect Avg. Acc | PBS | Strongest Path | Weakest Path |
|-------|---------------------------|-----|----------------|--------------|
| Gemini 3 Flash | 78.3 | 32.2 | S-T-I 98.4 | I-T-S 60.2 |
| Gemini 2.5 Pro | 72.5 | 25.0 | S-T-I 96.9 | T-I-S 50.8 |
| Gemini 2.5 Flash | 53.6 | 4.7 | S-T-I 85.9 | T-I-S 21.9 |
| Qwen3-Omni 30B | 46.8 | 2.3 | S-T-I 77.0 | I-T-S/T-I-S 16.0 |
| Phi-4 Multimodal | 15.1 | 0.0 | S-I-T 26.6 | T-I-S 0.0 |
| Qwen2.5-Omni 7B | 14.5 | 0.0 | S-I-T 22.7 | T-I-S 1.8 |

The Connect subset is relatively simple. However, even the strongest model, Gemini 3 Flash, scored only 32.2 on PBS, indicating that while it solves many single-path samples, it fails to maintain consistency across all six paths. Qwen3-Omni 30B, the strongest open-source model, showed a 61.0 percentage point gap between its strongest and weakest paths.

| Model | OMHBench-Reasoning Avg. Acc | PBS | Strongest Path | Weakest Path |
|-------|-----------------------------|-----|----------------|--------------|
| Gemini 3 Flash | 49.4 | 8.6 | S-T-I 58.8 | I-T-S 40.0 |
| Gemini 2.5 Pro | 48.8 | 10.9 | S-I-T 53.9 | I-T-S 41.4 |
| Gemini 2.5 Flash | 21.0 | 0.0 | S-I-T 32.0 | I-T-S/T-I-S 10.9 |
| Gemini 2.5 Flash-lite | 10.7 | 0.0 | S-T-I 21.1 | I-T-S/T-I-S 0.0 |
| Qwen3-Omni 30B | 15.0 | 0.0 | S-T-I 28.5 | I-T-S/T-I-S 2.7 |
| Phi-4 Multimodal | 0.3 | 0.0 | S-I-T 0.6 | T-S-I 0.0 |

The Reasoning subset is significantly harder. The best models achieve only about 49% average accuracy, while most open-source models approach 0, demonstrating a massive gap between multi-modality perception and multi-modal controlled reasoning.

### Ablation Study
As OMHBench is a benchmark, the analysis focused on verifying its design:

| Analysis | Setting | Key Result |
|----------|---------|------------|
| OMU shortcut verification | Removal of visual or speech inputs | Existing OMU benchmarks remained 70%-80% solvable; OMHBench was 0% solvable. |
| PBS@k path robustness | Accuracy if at least $k$ paths are correct | Gemini 3 Flash Connect PBS@1 is 100.0, PBS@6 is 32.2. Robustness is much harder than single-path hits. |
| Input modality order | Fixing reasoning path, varying context order | Fluctuations up to 12.5 points in accuracy based on input placement. |
| Exact Match validation | Comparison with human/LLM-Judge | Results were identical; EM is reliable for this task. |
| TTS quality check | ASR and speech quality metrics | Low WER/CER and high STOI/SI-SDR suggest failures are from grounding, not audio distortion. |

### Key Findings
- Closed-source models outperform open-source ones, but none resolve the path asymmetry problem. High average accuracy often masks low PBS.
- Speech position is a core bottleneck. Accessing speech early in a path is easier; shifting speech to late stages (e.g., I-S or T-S) significantly degrades performance. This is termed **asymmetric omni-modal grounding**.
- Path distribution alters model rankings. One model may lead on I-S-T but lag on T-S-I, proving that single-path metrics are misleading.
- Operation types influence difficulty: Ranking is easier, while Proximity/Range are harder due to intermediate set management.
- Advanced prompting (e.g., Self-Ask, Plan-and-Solve) does not fundamentally solve the issues, suggesting the bottleneck is multi-modal semantic transfer within the model.

## Highlights & Insights
- Replicating the same question into six path versions allows for clean diagnostic measurement of modality preferences.
- PBS is a more rigorous and interpretable metric than average accuracy, explicitly testing consistency under different modality orders.
- Relying on real tables rather than purely generative models ensures factual verifiability and reduces evaluation noise.
- Elevating speech from "additional input" to a "necessary evidence link" provides a roadmap for training speech-aware VLMs that treat audio as queryable structured knowledge.
- The benchmark highlights that omni-modal capability is not just about supporting input formats, but about the stability of transferring entities and attributes across modality boundaries.

## Limitations & Future Work
- The task format is highly controlled (3-hop entity-attribute chains), which differs from open-world temporal video or complex meeting reasoning.
- TTS-generated speech does not capture the complexity of real speech (accents, background noise, overlaps).
- Visual evidence is limited to charts/tables and lacks natural images or UI screenshots.
- The positive integer answer format, while good for EM, restricts question types (e.g., no open-ended explanations).
- Future work could include supervised training on these intermediate hops to help models learn explicit cross-modal alignment.

## Related Work & Insights
- **vs OmniBench / Daily-Omni**: OMHBench ensures each modality is necessary and cannot be bypassed, unlike most OMU benchmarks.
- **vs MMQA / MuMuQA**: While those focus on text-image CMR, OMHBench adds speech and ensures modality path balance.
- **Training Insights**: Current MLLMs might benefit from explicit supervision of the reasoning chain (e.g., outputting "Current Entity -> Target Attribute -> Source Modality").
- **Benchmark Insights**: Benchmarks should move toward "equivalent content, varying layout/path" to isolate model capabilities from data distribution biases.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Combines omni-modal, 3-hop reasoning, and path balance into a controlled benchmark.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive 13-model evaluation and detailed diagnostic analyses.
- Writing Quality: ⭐⭐⭐⭐☆ Clear structure and logic; some figures are dense but informative.
- Value: ⭐⭐⭐⭐⭐ Essential for evaluating true omni-modal grounding and identifying the "speech bottleneck."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ThinkOmni: Lifting Textual Reasoning to Omni-modal Scenarios via Guidance Decoding](../../ICLR2026/multimodal_vlm/thinkomni_lifting_textual_reasoning_to_omni-modal_scenarios_via_guidance_decodin.md)
- [\[CVPR 2026\] CRIT: Graph-Based Automatic Data Synthesis to Enhance Cross-Modal Multi-Hop Reasoning](../../CVPR2026/multimodal_vlm/crit_graph-based_automatic_data_synthesis_to_enhance_cross-modal_multi-hop_reaso.md)
- [\[ACL 2026\] OMIBench: Benchmarking Olympiad-Level Multi-Image Reasoning in Large Vision-Language Models](omibench_benchmarking_olympiad-level_multi-image_reasoning_in_large_vision-langu.md)
- [\[ICLR 2026\] FRIEDA: Benchmarking Multi-Step Cartographic Reasoning in Vision-Language Models](../../ICLR2026/multimodal_vlm/frieda_benchmarking_multi-step_cartographic_reasoning_in_vision-language_models.md)
- [\[ACL 2026\] Structured and Abstractive Reasoning on Multi-modal Relational Knowledge Images](structured_and_abstractive_reasoning_on_multi-modal_relational_knowledge_images.md)

</div>

<!-- RELATED:END -->
