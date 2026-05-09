---
title: >-
  [Paper Note] MuSLR: Multimodal Symbolic Logical Reasoning
description: >-
  [NeurIPS 2025][Medical Imaging][Multimodal symbolic logical reasoning] This paper introduces MuSLR, the first multimodal symbolic logical reasoning task, along with its benchmark MuSLR-Bench (1,093 instances spanning 7 domains, 35 atomic symbolic logic rules, and reasoning depths of 2–9). It further proposes LogiCAM, a modular framework comprising premise selection, reasoning type identification, and symbolic reasoning modules, which improves GPT-4.1's CoT performance by 14.13%.
tags:
  - NeurIPS 2025
  - Medical Imaging
  - Multimodal symbolic logical reasoning
  - VLM benchmark
  - formal logic
  - Chain-of-Thought
  - modular reasoning
date: 2026-05-08
content_hash: 2159ef4272bbab62
---

# MuSLR: Multimodal Symbolic Logical Reasoning

**Conference**: NeurIPS 2025
**arXiv**: [2509.25851](https://arxiv.org/abs/2509.25851)
**Code**: [Project Page](https://llm-symbol.github.io/MuSLR)
**Area**: Medical Imaging / Multimodal Reasoning
**Keywords**: Multimodal symbolic logical reasoning, VLM benchmark, formal logic, Chain-of-Thought, modular reasoning

## TL;DR

This paper introduces MuSLR, the first multimodal symbolic logical reasoning task, along with its benchmark MuSLR-Bench (1,093 instances spanning 7 domains, 35 atomic symbolic logic rules, and reasoning depths of 2–9). It further proposes LogiCAM, a modular framework comprising premise selection, reasoning type identification, and symbolic reasoning modules, which improves GPT-4.1's CoT performance by 14.13%.

## Background & Motivation

Symbolic logical reasoning—performing precise, verifiable inference based on formal logic (e.g., first-order logic)—is critical for high-stakes scenarios such as autonomous driving and medical diagnosis. However, three key gaps exist in prior work:

**Text-only modality**: Existing work (e.g., FOLIO, ProofWriter, Multi-LogiEval) evaluates symbolic reasoning only in purely textual settings, without involving visual information. Yet real-world applications require integrating vision and text—for instance, an autonomous driving system must recognize "road ahead is closed" from camera images and combine it with the traffic rule "only if the road ahead is clear (B) may the vehicle proceed straight (A)" to derive $(\neg B) \rightarrow (\neg A)$ (Modus Tollens).

**Deficiencies in multimodal benchmarks**: Existing multimodal reasoning benchmarks (LogicVista, VisuLogic, MMMU, etc.) involve reasoning in visual contexts but do not explicitly test the application of formal logical rules (e.g., Modus Ponens, De Morgan's laws) to joint visual-textual inputs.

**Limitations of neuro-symbolic approaches**: Traditional neuro-symbolic methods first formalize natural language into symbolic form and then invoke theorem provers; however, theorem provers accept only textual input, requiring visual information to be converted to text first—inevitably causing information loss.

MuSLR addresses this gap by requiring models to perform formal symbolic logical derivation over jointly presented visual and textual inputs.

## Method

### Overall Architecture

**MuSLR-Bench construction**: Images are collected from COCO, Flickr30k, nocaps, and other sources, with visual details extracted using GPT-4o. Non-trivial logical inference rules are selected from propositional logic (PL), first-order logic (FOL), and non-monotonic logic (NM), composed into reasoning chains, and instantiated in real-world contexts to generate question-answer pairs. After automated and manual quality checks, the final benchmark contains 1,093 instances covering 7 domains, 35 atomic symbolic rules, 976 logical combinations, and reasoning depths of 2–9.

**Two task formats**: (1) Truth-value evaluation: given image $I$, text $T$, and assertion $A$, determine $\text{Truth}(A) \in \{\text{True, False, Unknown}\}$; (2) Multiple-choice: select the best assertion from 4 candidates.

### Key Designs

1. **Premise Selector**: Addresses the multimodal fusion challenge. Given image $I$ and text $T$ (comprising context $\mathcal{T}$ and question $Q$), the VLM first selects the most relevant symbolic rule $R_r \in \mathcal{T}$, then analyzes $R_r$ to determine which parts are visually relevant and extracts the corresponding visual information $V_r$. These are merged as $I_{\text{critical}} = R_r \cup V_r$. The core mechanism is to avoid unnecessary complexity and noise from rich inputs by extracting only critical visual-textual premises.

2. **Reasoning Type Identifier**: Addresses the challenge of mixing symbolic and heuristic reasoning. It analyzes $I_{\text{critical}}$ to determine whether formal logical rules can be applied—if so, symbolic reasoning takes priority; otherwise, heuristic commonsense reasoning is used. **Design Motivation**: maximize inferential rigor and reliability while maintaining flexibility through commonsense reasoning.

3. **Reasoner**: Based on the output of the Reasoning Type Identifier, either applies formal logical rules for syllogistic derivation (deriving conclusion $C$ from major and minor premises) or uses commonsense reasoning to bridge gaps in symbolic logic. By using the VLM to directly access multimodal information for approximate symbolic reasoning, the framework avoids the information loss inherent in traditional neuro-symbolic methods that convert visual inputs to text.

4. **Iterative mechanism**: Checks whether conclusion $C$ sufficiently answers $Q$. If not, $C$ is appended to the context as $T' = T \cup C$ and a new reasoning iteration begins.

### Loss & Training

LogiCAM is built on prompt engineering with GPT-4.1, employing a three-shot CoT setting with temperature fixed at 0.0 for deterministic outputs. No model training is involved; instead, a modular prompting structure guides the VLM toward structured symbolic reasoning.

## Key Experimental Results

### Main Results (Overall accuracy of VLMs on MuSLR-Bench)

| Model | Avg. Accuracy | PL (Propositional) | FOL (First-Order) | NM (Non-Monotonic) |
|-------|--------------|-------------------|------------------|-------------------|
| GPT-4.1 (CoT) | 46.84% | ~44% | ~25% | ~52% |
| InternVL (CoT) | 45.20% | ~45% | ~44% | ~41% |
| Qwen (CoT) | 41.63% | ~42% | ~30% | ~38% |
| GPT-4o (CoT) | 38.93% | ~38% | ~24% | ~44% |
| Claude (CoT) | 33.49% | ~33% | ~36% | ~34% |
| **LogiCAM** | **60.97%** | **+31.93%** | **+48.93%** | **+26.17%** |

### Ablation Study

| Configuration | Performance Change | Note |
|---------------|--------------------|------|
| Full LogiCAM | Baseline | — |
| w/o symbolic reasoning module | −5.14% | Largest drop; formal logic rules are indispensable |
| w/o heuristic reasoning | −3.45% | Heuristics effectively complement symbolic gaps |
| w/o premise selection | −3.27% | Identifying critical information simplifies reasoning |

### Key Findings

- **All VLMs struggle with multimodal symbolic reasoning**: The best-performing GPT-4.1 achieves only 46.84%, indicating this is a genuinely difficult problem.
- **Logical complexity is negatively correlated with performance**: FOL is the hardest (37.04%) > PL (42.77%) > NM (46.09%), consistent with the intuition that first-order logic demands precise variable binding and quantifier tracking.
- **LogiCAM achieves the largest gains on complex logic**: FOL improves by 48.93% (largest), PL by 31.93%, and NM by 26.17%—the modular framework confers greater advantages on structured reasoning tasks.
- **Approximately 70% of errors stem from cross-modal logical alignment failures**: Logical alignment of visual-textual premises is the central bottleneck.
- **Depth analysis**: As reasoning depth increases from 2–3 steps to 8–9 steps, all models degrade—GPT-4.1 by ~16% and Claude by ~20%; LogiCAM maintains 54.61% at depth 8–9, outperforming GPT-4.1 by ~13%.
- **Reasoning traceability**: LogiCAM achieves the best scores on both ROUGE-L (0.170) and BertScore (0.835); the weak correlation between surface-level text-matching metrics and logical consistency metrics (Pearson $r = 0.25$) highlights the limitations of shallow evaluation.

## Highlights & Insights

- **First formal definition**: MuSLR formally defines multimodal symbolic logical reasoning as an independent task, filling an important research gap.
- **Insightful error analysis**: The finding that 70% of errors originate from cross-modal logical alignment—rather than perceptual errors—points future research toward fusion-level reasoning failures.
- **Effectiveness of modular design**: The results demonstrate that decomposing the reasoning process into premise selection, type identification, and reasoning execution is particularly effective for symbolic reasoning.
- **Counter-intuitive behavior of NM**: Non-monotonic logic exhibits the highest alignment error rate (79%) yet the lowest logical rule error rate (5%)—once alignment succeeds, the reasoning itself is relatively straightforward.

## Limitations & Future Work

- The benchmark scale remains modest (1,093 instances); future work should expand coverage to more domains and logic types.
- LogiCAM relies heavily on GPT-4.1's prompt-following capability and is thus strongly dependent on the underlying VLM's capacity.
- Logic-grounded training objectives have not been explored; the current work operates entirely at the inference-time prompting level.
- The deterministic setting (temperature 0.0) may suppress beneficial exploration behaviors in non-monotonic reasoning.

## Related Work & Insights

- MuSLR complements text-only symbolic reasoning benchmarks such as FOLIO and ProofWriter by extending the setting from text to multimodal inputs.
- The modular design of LogiCAM is conceptually related to reasoning-augmentation frameworks such as ReAct and Tree-of-Thoughts.
- The work has direct implications for trustworthy AI in medical diagnosis, where rigorous logical chains—rather than pattern matching—are required.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Pioneering work that introduces the first task definition and benchmark for multimodal symbolic logical reasoning.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation across 7 VLMs with in-depth error analysis, though the dataset scale is limited.
- **Writing Quality**: ⭐⭐⭐⭐ Problem motivation is clearly articulated with vivid case studies.
- **Value**: ⭐⭐⭐⭐ Provides an important evaluation dimension for formal reasoning in VLMs with significant potential to guide future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FireGNN: Neuro-Symbolic Graph Neural Networks with Trainable Fuzzy Rules for Interpretable Medical Image Classification](firegnn_neuro-symbolic_graph_neural_networks_with_trainable_fuzzy_rules_for_inte.md)
- [\[NeurIPS 2025\] CXReasonBench: A Benchmark for Evaluating Structured Diagnostic Reasoning in Chest X-rays](cxreasonbench_a_benchmark_for_evaluating_structured_diagnostic_reasoning_in_ches.md)
- [\[NeurIPS 2025\] CureAgent: A Training-Free Executor-Analyst Framework for Clinical Reasoning](cureagent_a_training-free_executor-analyst_framework_for_clinical_reasoning.md)
- [\[ICLR 2026\] MMedAgent-RL: Optimizing Multi-Agent Collaboration for Multimodal Medical Reasoning](../../ICLR2026/medical_imaging/mmedagent-rl_optimizing_multi-agent_collaboration_for_multimodal_medical_reasoni.md)
- [\[NeurIPS 2025\] Multimodal 3D Genome Pre-training](multimodal_3d_genome_pre-training.md)

</div>

<!-- RELATED:END -->
