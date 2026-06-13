---
title: >-
  [Paper Note] StructBreak: Structural Cognitive Overload-Induced Safety Failures in MLLMs
description: >-
  [ACL2026][Multimodal VLM][MLLM Safety] StructBreak proposes the "Structural Cognitive Overload" (SCO) attack paradigm, utilizing the topological complexity of Visual Knowledge Graphs (VKG) to induce safety failures in Mu…
tags:
  - "ACL2026"
  - "Multimodal VLM"
  - "MLLM Safety"
  - "Jailbreak Attack"
  - "Cognitive Overload"
  - "Visual Knowledge Graph"
  - "Attention Dissipation"
  - "Alignment Failure"
date: 2026-05-08
content_hash: f9d82af04f40e10c
---

# StructBreak: Structural Cognitive Overload-Induced Safety Failures in MLLMs

**Conference**: ACL2026 Findings  
**arXiv**: [2605.25534](https://arxiv.org/abs/2605.25534)
**Code**: To be confirmed
**Area**: multimodal_vlm
**Keywords**: MLLM Safety, Jailbreak Attack, Cognitive Overload, Visual Knowledge Graph, Attention Dissipation, Alignment Failure

## TL;DR

StructBreak proposes the "Structural Cognitive Overload" (SCO) attack paradigm, utilizing the topological complexity of Visual Knowledge Graphs (VKG) to induce safety failures in Multi-modal LLMs. It achieves an average attack success rate of 92% across six frontier MLLMs (reaching 97% on Gemini 2.5) in a black-box setting and reveals the mechanism of safety collapse through attention dissipation, latent space topology, and geometric analysis.

## Background & Motivation

Multi-modal Large Language Models (MLLMs) possess sophisticated structural reasoning capabilities (interpreting flowcharts, knowledge graphs, etc.), but this capability serves as a double-edged sword. Existing safety alignment techniques (SFT, RLHF) primarily target surface-level threats like typographic attacks and pixel-level perturbations. This paper discovers that as the depth of structural reasoning increases, the "cognitive resources" required to maintain structural logic gradually overwhelm the safety alignment boundaries—reasoning takes precedence over safety, resulting in the **Structural Cognitive Overload** (SCO) phenomenon. This attack surface has been largely unexplored.

## Method

### Overall Architecture

StructBreak consists of two modules: (1) **StructBreak-Synth** for automated generation of adversarial VKG images; (2) **StructBreak-Eval** for standardized evaluation. The overall workflow is an automated "generation $\rightarrow$ filtering $\rightarrow$ evaluation" pipeline, operating entirely as a black box without requiring internal model access.

### Key Designs

1.  **Semantic Obfuscation**: Based on the risk category of the harmful query, preset templates are used to wrap malicious intent within situational contexts (e.g., academic analysis, system debugging), ensuring consistent obfuscation quality and avoiding keyword-level interception.
2.  **Graph Decomposition & Rendering**: Using DeepSeek-R1 as a Graph Builder, the obfuscated intent is decomposed into a structured graph $G=(V,E)$, encoding logical dependencies (causality, etc.), and rendered into a VKG image. Ablation studies confirm that topological complexity, rather than visual style, is the primary driver of cognitive overload.
3.  **Intent Decoupling + Quality Gating**: Malicious intent (encoded in the graph structure) is decoupled from the instruction trigger (benign prompts such as "analyze the structural relationships in the graph") to prevent early refusal triggered by text semantic matching. Quality gating uses a verify-and-refine loop to probe the test MLLM; if it fails, feedback is provided to adjust the topology, and only successful samples enter the final set.

### Loss & Training

No training process is involved. The attack is based on black-box API calls using a three-label annotation scheme: Refusal (R), Violation (V), and Answered (A). An attack is considered successful when $(R,V,A)=(0,1,1)$.

## Key Experimental Results

### Main Results

Attack Success Rate (ASR) across six frontier MLLMs:

| Method | GPT-4o | GPT-5-mini | GPT-5 | Qwen2.5-VL | Claude 4 | Gemini 2.5 | Average |
|---|---|---|---|---|---|---|---|
| Original | 30% | 29% | 33% | 19% | 29% | 26% | 27.7% |
| FigStep | 45% | 41% | 38% | 92% | 31% | 76% | 53.8% |
| MM-SafetyBench | 61% | 42% | 46% | 85% | 45% | 88% | 61.2% |
| **StructBreak** | **93%** | **90%** | **95%** | **95%** | **82%** | **97%** | **92.0%** |

### Ablation Study

*   **Structural Complexity**: Shows a non-linear relationship with graph density; moderate simplification maintains effectiveness, while aggressive pruning leads to a sharp drop in ASR.
*   **Visual Style**: Changes to node colors, backgrounds, etc., have negligible impact on performance.
*   **Resolution**: Extreme downsampling destroys the attack success rate—precise symbolic recognition and edge parsing are necessary prerequisites.
*   **Defense Testing**: Intent-First Safety Prompts provide only partial mitigation; StructBreak maintains high bypass rates on most models.

### Key Findings

*   **Capability-Vulnerability Paradox**: Models with stronger reasoning capabilities (GPT-5: 95%, Gemini 2.5: 97%) are more susceptible to the attack. FigStep achieves only 38% on GPT-5, while StructBreak reaches 95%.
*   **Safety Attention Dissipation**: VKG processing causes the attention quality of the system prompt $M_{sys}$ to be compressed near zero. The $M_{vis}/M_{sys}$ ratio peaks at approximately 6.0 in the initial layers, an order of magnitude higher than the text baseline.
*   **Latent Space Anomalous Distribution**: StructBreak inputs occupy an anomalous distribution region in the latent space relative to standard harmful prompts and are nearly orthogonal to the model's refusal direction, revealing a brand-new structural risk channel.

## Highlights & Insights

*   **New Attack Dimension**: Unlike typographic attacks (FigStep) or pixel perturbations, StructBreak exploits high-level semantic structural complexity to trigger cognitive overload, bypassing rather than directly confronting safety defenses.
*   **Sufficient Mechanistic Evidence**: Provides mechanistic explanations for safety collapse from three levels: attention dynamics, latent space topology, and geometric analysis.
*   **High Practicality**: The black-box setting, single-turn success, and near-zero refusal rate pose a serious threat to real-world deployments.

## Limitations & Future Work

*   Attack evaluation relies on GPT-5 as an automated judge, which may involve annotation bias.
*   VKG generation requires calls to high-capability LLMs (DeepSeek-R1), incurring certain costs for the attack itself.
*   Current alignment paradigms (SFT + RLHF) may be fundamentally insufficient in the era of complex multi-modal reasoning—new safety architectures are required.

## Related Work & Insights

*   **FigStep** (Gong et al., 2025): Typographic jailbreak attack, which shows decreased effectiveness on frontier models due to improved OCR robustness.
*   **Cognitive Load Theory** (Sweller, 1988): The theoretical foundation for the SCO concept.
*   **Talking-head Attention** (Shazeer et al., 2020): Information exchange between independent components can significantly improve stability; this paper reveals the opposite effect from an adversarial perspective.

## Rating

| Dimension | Score (1-10) |
|---|---|
| Novelty | 9 |
| Practicality | 8 |
| Clarity | 8 |
| Experimental Thoroughness | 9 |

## Rating
- Novelty: To be rated
- Experimental Thoroughness: To be rated
- Writing Quality: To be rated
- Value: To be rated

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SDEval: Safety Dynamic Evaluation for Multimodal Large Language Models](../../AAAI2026/multimodal_vlm/sdeval_safety_dynamic_evaluation_for_multimodal_large_language_models.md)
- [\[ACL 2026\] TableVista: Benchmarking Multimodal Table Reasoning under Visual and Structural Complexity](tablevista_benchmarking_multimodal_table_reasoning_under_visual_and_structural_c.md)
- [\[ICML 2026\] Toward Structural Multimodal Representations: Specialization, Selection, and Sparsification via Mixture-of-Experts](../../ICML2026/multimodal_vlm/toward_structural_multimodal_representations_specialization_selection_and_sparsi.md)
- [\[CVPR 2026\] StructXLIP: Enhancing Vision-Language Models with Multimodal Structural Cues](../../CVPR2026/multimodal_vlm/structxlip_enhancing_vision-language_models_with_multimodal_structural_cues.md)
- [\[ICML 2026\] CVSearch: Empowering Multimodal LLMs with Cognitive Visual Search for High-Resolution Image Perception](../../ICML2026/multimodal_vlm/cvsearch_empowering_multimodal_llms_with_cognitive_visual_search_for_high-resolu.md)

</div>

<!-- RELATED:END -->
