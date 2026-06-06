---
title: >-
  [Paper Note] AnesSuite: A Comprehensive Benchmark and Dataset Suite for Anesthesiology Reasoning
description: >-
  [ICLR 2026][LLM Evaluation][Anesthesiology reasoning] This paper introduces AnesSuite, the first comprehensive dataset suite for anesthesiology reasoning, comprising AnesBench—an evaluation benchmark of 7…
tags:
  - "ICLR 2026"
  - "LLM Evaluation"
  - "Anesthesiology reasoning"
  - "medical benchmark"
  - "bilingual evaluation"
  - "cognitive demand classification"
  - "GRPO reinforcement learning"
date: 2026-05-08
content_hash: 124fe8700a580b26
---

# AnesSuite: A Comprehensive Benchmark and Dataset Suite for Anesthesiology Reasoning

**Conference**: ICLR 2026
**arXiv**: [2504.02404](https://arxiv.org/abs/2504.02404)  
**Code**: [MiliLab/AnesSuite](https://github.com/MiliLab/AnesSuite)  
**Area**: LLM Evaluation
**Keywords**: Anesthesiology reasoning, medical benchmark, bilingual evaluation, cognitive demand classification, GRPO reinforcement learning

## TL;DR

This paper introduces AnesSuite, the first comprehensive dataset suite for anesthesiology reasoning, comprising AnesBench—an evaluation benchmark of 7,972 bilingual multiple-choice questions organized into three cognitive difficulty levels—and three training datasets (AnesCorpus/AnesQA/AnesR1). The Morpheus models trained on this suite via SFT+GRPO enable a 7B model to match a 14B baseline, while revealing significant bottlenecks of state-of-the-art LLMs on complex clinical reasoning (System 2).

## Background & Motivation

**Background**: LLMs have made substantial progress in medical AI, yet their reasoning capabilities in highly specialized disciplines such as anesthesiology remain severely inadequate. Anesthesiology requires simultaneous management of multiple physiological systems—airway and respiratory function, cardiovascular stability, electrolyte balance, sedation levels—demanding a full spectrum of reasoning abilities ranging from rapid factual recall (System 1) to complex multi-factor clinical decision-making (System 2).

**Limitations of Prior Work**: Existing medical benchmarks such as MedQA and PubMedQA offer broad coverage but suffer from three critical shortcomings: (1) anesthesiology is often subsumed implicitly under surgical or dental categories, lacking dedicated evaluation; (2) the few anesthesiology-specific benchmarks, such as CAB, focus primarily on factual recall questions and provide insufficient assessment of clinical reasoning and decision-making; (3) language coverage is limited to a single language, precluding evaluation of model performance disparities across bilingual clinical contexts.

**Key Challenge**: The primary challenge for LLMs in anesthesiology is not a deficit of knowledge per se, but rather an insufficient ability to apply knowledge to complex reasoning problems. Existing benchmarks fail to effectively distinguish between "knowing what" and "being able to reason about what," making it impossible to precisely localize model bottlenecks. Furthermore, systematic comparisons of training strategies—SFT, CPT, and RLVR—in medical specialty domains are lacking.

**Goal**: (1) Construct the first anesthesiology dataset suite covering the complete evaluation-to-training pipeline; (2) establish a three-tier cognitive demand taxonomy to precisely diagnose model capability boundaries; (3) explore efficient domain adaptation training strategies.

**Key Insight**: Drawing on Kahneman's System 1/2 theory from cognitive psychology, the authors introduce for the first time in medical benchmarking a three-tier cognitive classification—System 1 (factual recall) → System 1.x (hybrid reasoning) → System 2 (complex decision-making)—enabling fine-grained diagnosis of model performance across different reasoning levels.

**Core Idea**: By constructing a cognitively stratified specialty evaluation benchmark paired with training datasets, the paper systematically advances both LLM capability improvement and bottleneck analysis in complex anesthesiology reasoning.

## Method

### Overall Architecture

AnesSuite is an integrated dataset suite encompassing both evaluation and training components, comprising four complementary components that cover the complete model development pipeline from continual pre-training to reinforcement learning. The input consists of raw data from authoritative medical sources (ABA examinations, textbooks, PubMed literature, and large-scale web text); the output consists of a structured evaluation benchmark and aligned datasets directly applicable to each training stage.

| Component | Data Type | English Scale | Chinese Scale | Purpose |
|-----------|-----------|--------------|---------------|---------|
| AnesBench | Multiple-choice questions | 4,418 items | 3,554 items | Evaluation benchmark (three-tier cognitive classification) |
| AnesCorpus | Plain text documents | 1.8M articles | 600K articles | Continual pre-training (CPT) |
| AnesQA | QA pairs | 20,713 items | — | Supervised fine-tuning (SFT) |
| AnesR1 | MCQ + CoT | 3,200 items | 7,000 items | SFT cold start + RLVR (GRPO) |

Based on this suite, the authors train the Morpheus model family—using Qwen2.5-7B/14B/32B as backbones and applying two-stage SFT+GRPO training on AnesR1—yielding the first collection of anesthesiology reasoning baseline models.

### Key Designs

1. **Three-Tier Cognitive Demand Taxonomy (System 1 / 1.x / 2)**

    - **Function**: Classifies the 7,972 questions in AnesBench into three levels by reasoning complexity: System 1 (factual recall, e.g., "What is the mechanism of action of propofol?"), System 1.x (hybrid reasoning, requiring integration of 2–3 knowledge points), and System 2 (complex clinical decision-making involving multi-step reasoning, conditional judgment, and cross-domain synthesis).
    - **Mechanism**: DeepSeek-R1 is used to annotate each question with a cognitive demand label, guided by comprehensive annotation guidelines and few-shot examples. After annotation, 60% of questions are randomly sampled for human expert review to ensure quality. In terms of difficulty distribution, System 1.x and System 2 questions together constitute 20–30% of the total, ensuring adequate coverage of higher-order reasoning.
    - **Design Motivation**: Traditional medical benchmarks report aggregate accuracy across all questions, inflating scores due to the abundance of simple recall items and obscuring model deficiencies in genuinely reasoning-demanding scenarios. The tiered framework makes visible the fact that performance degradation from System 1 to System 2 far exceeds prior expectations.

2. **Multi-Source Data Construction and Decontamination Pipeline**

    - **Function**: Ensures the quality and integrity of all four datasets, preventing training/evaluation data leakage.
    - **Mechanism**: AnesBench is collected from ABA examinations, standard textbooks, and validated online assessment tools. AnesCorpus is constructed by filtering anesthesiology-relevant documents from Fineweb and Chinese Fineweb using a two-stage keyword filtering procedure. AnesQA is built from PubMed articles via a dual-model pipeline (LLaMA3.3-70B for question generation and Qwen2.5-72B for filtering and answer generation). CoT traces in AnesR1 are generated by DeepSeek-R1 and filtered via rejection sampling (questions where three attempts all yield incorrect answers are discarded). Decontamination applies a two-phase filter to AnesCorpus—n-gram screening followed by fine-grained longest common substring (LCS > 64 characters) comparison.
    - **Design Motivation**: Data leakage is a serious concern in the medical domain, particularly since common examination questions may already be present in LLM training corpora. The dual decontamination scheme, supplemented by a dedicated data leakage analysis algorithm, ensures the reliability of evaluation results.

3. **Morpheus Two-Stage Training Pipeline (SFT → GRPO)**

    - **Function**: Build anesthesiology reasoning capabilities on top of Qwen2.5 backbone models.
    - **Mechanism**: The first stage performs limited-step SFT on AnesR1 CoT data as a cold-start initialization for GRPO—enabling the model to learn the format of structured reasoning chains. The second stage applies GRPO (Group Relative Policy Optimization) reinforcement learning on the verifiable multiple-choice questions in AnesR1, using correct answer matching as a verifiable reward signal to further elicit reasoning capability. Morpheus is trained at three scales: 7B, 14B, and 32B.
    - **Design Motivation**: SFT alone yields modest gains in English but degrades Chinese performance (likely due to imbalanced English-to-Chinese ratios in AnesR1), whereas GRPO effectively repairs this issue, recovering and even surpassing the Chinese baseline while maintaining English gains. A more fundamental finding is that training on approximately 10,000 anesthesiology samples yields reasoning gains that generalize to general medical and even general-domain benchmarks.

### Loss & Training

The SFT stage uses standard next-token prediction loss. The GRPO stage employs group relative policy optimization: multiple candidate responses are sampled for each question, correct answer matching serves as the reward signal, and intra-group relative rankings are used to compute the advantage function for policy optimization. Unlike conventional RL, GRPO requires no separate reward model, leveraging instead the verifiability of multiple-choice questions as the reward signal. Training is conducted separately at the Qwen2.5-7B/14B/32B scales; SFT is performed for a limited number of steps, and GRPO uses standard hyperparameter settings.

## Key Experimental Results

### Main Results: Evaluation of 50+ Models on AnesBench

The paper evaluates over 50 LLMs, including closed-source models (GPT-4o, Gemini-2.5-Pro/Flash, Claude-3.7-Sonnet), general-purpose open-source models (Qwen3 series, Llama-4, DeepSeek-R1/V3), and medically specialized models (HuatuoGPT-o1, BioMistral).

| Model | EN-Sys1 | EN-Sys1.x | EN-Sys2 | EN-Total | CH-Sys1 | CH-Sys1.x | CH-Sys2 | CH-Total | Avg. |
|-------|---------|-----------|---------|----------|---------|-----------|---------|----------|------|
| Gemini-2.5-Pro | 0.89 | 0.82 | **0.77** | **0.86** | 0.88 | 0.75 | **0.60** | **0.85** | **0.85** |
| DeepSeek-R1 | 0.85 | 0.78 | 0.70 | 0.82 | 0.86 | 0.77 | 0.61 | 0.83 | 0.82 |
| Llama-4-Maverick | 0.83 | 0.73 | 0.64 | 0.79 | 0.86 | 0.72 | 0.59 | 0.83 | 0.81 |
| Gemini-2.5-Flash | 0.84 | 0.76 | 0.68 | 0.81 | 0.84 | 0.72 | 0.59 | 0.81 | 0.81 |
| GPT-4o | 0.81 | 0.72 | 0.59 | 0.77 | 0.79 | 0.64 | 0.52 | 0.76 | 0.76 |
| Claude-3.7-Sonnet | 0.80 | 0.73 | 0.63 | 0.77 | 0.82 | 0.65 | 0.55 | 0.78 | 0.77 |
| Qwen3-32B | 0.72 | 0.64 | 0.48 | 0.68 | 0.81 | 0.64 | 0.57 | 0.78 | 0.70 |
| HuatuoGPT-o1-72B | 0.71 | 0.61 | 0.48 | 0.67 | 0.79 | 0.67 | 0.61 | 0.76 | 0.71 |
| Qwen2.5-7B-Instruct | 0.56 | 0.44 | 0.36 | 0.51 | 0.69 | 0.55 | 0.55 | 0.66 | 0.59 |
| BioMistral-7B | 0.43 | 0.30 | 0.32 | 0.39 | 0.24 | 0.25 | 0.16 | 0.24 | 0.31 |

### Morpheus Model Results

| Model | SFT | GRPO | EN-Total | CH-Total | Avg. |
|-------|-----|------|----------|----------|------|
| Qwen2.5-7B-Instruct | — | — | 0.51 | 0.66 | 0.59 |
| Morpheus-7B (SFT only) | ✓ | ✗ | 0.54 | 0.56 | 0.54 |
| **Morpheus-7B** | **✓** | **✓** | **0.56** | **0.70** | **0.63** |
| Qwen2.5-14B-Instruct | — | — | 0.57 | 0.72 | 0.64 |
| Morpheus-14B (SFT only) | ✓ | ✗ | 0.60 | 0.55 | 0.57 |
| **Morpheus-14B** | **✓** | **✓** | **0.63** | **0.75** | **0.69** |
| Qwen2.5-32B-Instruct | — | — | 0.61 | 0.76 | 0.68 |
| Morpheus-32B (SFT only) | ✓ | ✗ | 0.67 | 0.64 | 0.65 |
| **Morpheus-32B** | **✓** | **✓** | **0.68** | **0.77** | **0.72** |

Core finding: **Morpheus-7B matches Qwen2.5-14B-Instruct, Morpheus-14B matches Qwen2.5-32B-Instruct, and Morpheus-32B matches Qwen2.5-72B-Instruct**—each model tier, through SFT+GRPO, achieves the baseline performance of the next larger tier.

### Ablation Study: Training Strategy and Data Comparison

| Model | SFT Data | EN Accuracy | CH Accuracy |
|-------|----------|-------------|-------------|
| Qwen2.5-7B-Base + AnesQA | Anesthesiology | 49.3 | 64.9 |
| Qwen2.5-7B-Base + Medical-o1 | General medical | 49.1 | 63.0 |
| Qwen2.5-7B-Base + Both | Mixed | **49.7** | **65.9** |
| Qwen2.5-7B-Base-CPT + AnesQA | Anesthesiology | 49.7 | 50.7 |
| Qwen2.5-7B-Base-CPT + Medical-o1 | General medical | 50.7 | 59.4 |
| Qwen2.5-7B-Base-CPT + Both | Mixed | **51.2** | **60.0** |

### Key Findings

- **System 2 is the bottleneck for all models**: The magnitude of performance degradation from System 1 to System 2 is striking—even Gemini-2.5-Pro reaches only 0.77 on English System 2 (vs. 0.89 on System 1), and most open-source models score below 0.50 on System 2. This indicates that the core challenge for LLMs in anesthesiology is not a knowledge deficit but rather an insufficient ability to apply knowledge to complex reasoning.
- **GRPO is critical for reasoning gains**: SFT alone yields modest improvements on English but severely degrades Chinese performance (Morpheus-14B SFT only drops Chinese accuracy from 0.72 to 0.55); GRPO fully restores and surpasses the baseline on both languages. This suggests that SFT may induce catastrophic forgetting along the language dimension, while GRPO recalibrates language balance through the verifiable reward signal.
- **Dual nature of CPT**: Continual pre-training on AnesCorpus improves English performance (49.7→51.2) but severely degrades Chinese performance (64.9→50.7), a drop of 14.2 percentage points. The authors hypothesize that this results from a 3:1 English-to-Chinese document ratio in AnesCorpus, causing catastrophic forgetting of Chinese knowledge.
- **CoT length positively correlates with reasoning quality**: On System 2 tasks, models that generate longer CoT reasoning chains perform markedly better; however, on System 1 and System 1.x tasks, the effect of CoT length is negligible and performance is primarily determined by model scale.
- **General medical data provides complementary value**: Mixing AnesQA (specialty-specific) with Medical-o1 (general medical) outperforms using either dataset alone, demonstrating that general medical knowledge remains a beneficial complement even in the highly specialized domain of anesthesiology.
- **Medically specialized models show no significant advantage**: Medical LLMs such as HuatuoGPT-o1 do not significantly outperform general-purpose reasoning models of comparable scale (e.g., DeepSeek-R1) on AnesBench, suggesting that anesthesiology reasoning is fundamentally distinct from general medical reasoning.

## Highlights & Insights

- **Transferability of the cognitive stratification framework**: The three-tier System 1/1.x/2 framework is not specific to anesthesiology and can be directly applied to other specialty benchmarks requiring differentiation between recall, simple reasoning, and complex reasoning (e.g., ICU critical care decision-making, emergency triage). This approach is theoretically better grounded than simple easy/medium/hard splits, as it corresponds to the well-validated dual-process theory in cognitive science.
- **SFT as cold-start for GRPO rather than a final solution**: This finding is highly practical—the paper clearly demonstrates SFT's "side effects" (improving the target language while degrading others) and shows how GRPO repairs this via verifiable reward signals, offering actionable guidance for specialty adaptation of multilingual models.
- **High returns from small data**: Approximately 10,000 AnesR1 training samples suffice to achieve cross-scale reasoning gains that generalize to general medical (MedQA) and general-domain (MMLU) benchmarks, suggesting that the transfer value of reasoning-intensive specialty data has been underestimated.
- **Large-scale model evaluation**: The comprehensive evaluation of over 50 models provides a complete capability map of LLM anesthesiology reasoning, offering direct reference value for deployment decisions.

## Limitations & Future Work

- **System 2 questions are derived from abstract scenarios rather than real clinical cases**: The authors acknowledge that System 2 questions are constructed from structured scenarios in examinations and textbooks rather than from actual electronic medical record (EMR) clinical decision cases, and may not fully capture the more ambiguous, information-incomplete decision-making environment of real clinical practice.
- **Absence of multimodal clinical data**: Real anesthesiology practice involves multimodal information such as physiological monitor waveforms, imaging, and video laryngoscopy footage; text-only multiple-choice questions cannot assess model decision-making capabilities in a truly multimodal clinical environment.
- **Insufficient exploration of CPT strategies**: The catastrophic forgetting of Chinese induced by AnesCorpus is addressed only with a speculative explanation; the paper does not investigate potential remedies such as language corpus ratio adjustments, learning rate scheduling, or progressive training.
- **Constraints of the evaluation format**: Multiple-choice questions are inherently limited by predefined options and cannot assess a model's ability to generate free-form clinical recommendations (though a small supplementary open-ended evaluation is included in the appendix).
- **GRPO's verifiable reward depends on the multiple-choice format**: RLVR methods require automatically verifiable reward signals, which arise naturally from multiple-choice questions, but how to design reward functions for open-ended clinical reasoning remains an open problem.

## Related Work & Insights

- **vs. HuatuoGPT-o1**: HuatuoGPT-o1 is a general medical reasoning model whose 72B variant achieves 0.71 average on AnesBench, without data or evaluation design specifically targeting anesthesiology. The value of AnesSuite lies in demonstrating that general medical models still have significant blind spots in specialty reasoning.
- **vs. CAB**: CAB is the only prior anesthesiology-focused benchmark, but covers only Chinese and consists primarily of factual recall questions. AnesSuite comprehensively surpasses it across three dimensions: language coverage (bilingual), cognitive stratification (three-tier), and training resources (integrated evaluation and training).
- **vs. DeepSeek-R1**: DeepSeek-R1 achieves the highest average (0.82) among all open-source models on AnesBench, indicating that general reinforcement learning–based reasoning training yields significant spillover effects on medical specialties, though a gap remains relative to Gemini-2.5-Pro (0.85).
- **Cognitive science inspiration**: Kahneman's dual-process theory has previously been applied in the NLP community primarily to analyze human annotation behavior; AnesSuite is the first to systematically apply it to LLM benchmark design, providing a paradigmatic example of transferring theoretical frameworks across disciplines.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First comprehensive anesthesiology dataset suite; the three-tier cognitive taxonomy is innovative, though the data construction methodology itself is relatively standard.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive across dimensions: 50+ model comparisons, multi-strategy ablations, cross-lingual analysis, and CoT length analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with thorough data presentation, though the CPT analysis lacks sufficient depth.
- **Value**: ⭐⭐⭐⭐ — Significant reference value for specialty adaptation and reasoning enhancement research in medical AI; both datasets and models are open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] 3DSRBench: A Comprehensive 3D Spatial Reasoning Benchmark](../../ICCV2025/llm_evaluation/3dsrbench_a_comprehensive_3d_spatial_reasoning_benchmark.md)
- [\[ICLR 2026\] PlanetAlign: A Comprehensive Python Library for Benchmarking Network Alignment](planetalign_a_comprehensive_python_library_for_benchmarking_network_alignment.md)
- [\[CVPR 2026\] R2G: A Multi-View Circuit Graph Benchmark Suite from RTL to GDSII](../../CVPR2026/llm_evaluation/r2g_multi_view_circuit_graph_benchmark_suite_from_rtl_to_gdsii.md)
- [\[ICLR 2026\] AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite](astabench_benchmarking_ai_agents.md)
- [\[ICLR 2026\] UIS-Digger: Towards Comprehensive Research Agent Systems for Real-world Unindexed Information Seeking](uis-digger_towards_comprehensive_research_agent_systems_for_real-world_unindexed.md)

</div>

<!-- RELATED:END -->
