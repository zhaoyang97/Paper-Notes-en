---
title: >-
  [Paper Note] ZARA: Training-Free Motion Time-Series Reasoning via Evidence-Grounded LLM Agents
description: >-
  [ACL 2026][LLM Agent][Human Activity Recognition] This paper proposes ZARA, a knowledge- and retrieval-augmented multi-agent framework that distills sensor signals into a structured textual knowledge base, combines class-conditional retrieval with hierarchical LLM reasoning, and achieves interpretable human activity recognition in a fully training-free setting, substantially outperforming existing methods across 8 datasets.
tags:
  - ACL 2026
  - LLM Agent
  - Human Activity Recognition
  - Time-Series Reasoning
  - Retrieval-Augmented Generation
  - Multi-Agent Reasoning
  - Training-Free
date: 2026-05-08
content_hash: aa2e064b34ab5d83
---

# ZARA: Training-Free Motion Time-Series Reasoning via Evidence-Grounded LLM Agents

**Conference**: ACL 2026
**arXiv**: [2508.04038](https://arxiv.org/abs/2508.04038)
**Code**: [https://github.com/zechenli03/ZARA](https://github.com/zechenli03/ZARA)
**Area**: LLM Agent
**Keywords**: Human Activity Recognition, Time-Series Reasoning, Retrieval-Augmented Generation, Multi-Agent Reasoning, Training-Free

## TL;DR

This paper proposes ZARA, a knowledge- and retrieval-augmented multi-agent framework that distills sensor signals into a structured textual knowledge base, combines class-conditional retrieval with hierarchical LLM reasoning, and achieves interpretable human activity recognition in a fully training-free setting, substantially outperforming existing methods across 8 datasets.

## Background & Motivation

**Background**: Human Activity Recognition (HAR) is a core technology for digital health, adaptive interfaces, and related applications. Mainstream approaches rely on task-specific deep neural networks trained under fixed sensor configurations and predefined activity categories.

**Limitations of Prior Work**: Existing methods face three major bottlenecks: (1) poor generalization—adapting to new users or hardware requires costly model retraining; (2) limited training-free adaptation—time-series foundation models such as Moment and Mantis provide transferable representations but still require training task-specific classification heads, while contrastive learning methods like UniMTS struggle to distinguish fine-grained activities under frozen-parameter settings; (3) lack of interpretability—most methods output only class predictions without any transparent reasoning process.

**Key Challenge**: Although LLMs possess strong open-set reasoning capabilities, feeding raw numerical time series directly into an LLM leads to hallucination and weak grounding, because LLMs cannot intuitively understand physical dynamics from raw numerical streams.

**Goal**: To construct a fully training-free HAR framework capable of generalizing across users and datasets while providing interpretable reasoning.

**Key Insight**: The authors observe that, just as RAG in NLP relies on high-quality document corpora, RAG for HAR requires a domain-specific knowledge base that converts the implicit statistical patterns of how physical movements manifest in sensor data into verifiable natural-language priors (e.g., "running exhibits higher variance in vertical acceleration than walking").

**Core Idea**: Distill the statistical characteristics of sensor signals into pairwise textual knowledge bases, and combine class-conditional retrieval with hierarchical multi-agent reasoning to achieve evidence-grounded, training-free activity recognition.

## Method

### Overall Architecture

The central design of ZARA decouples information into two sources: (1) **Global Knowledge $\mathcal{K}$**—a static reference registry storing pairwise activity feature importance profiles; and (2) **Local Evidence $\mathcal{D}$**—a vector database of raw signal embeddings serving as external memory for local distributional grounding. The overall pipeline proceeds as follows: offline knowledge base construction → online retrieval of relevant evidence → hierarchical multi-agent reasoning to produce predictions and explanations.

### Key Designs

1. **Offline Statistical Profiling**:

    - Function: Automatically constructs a pairwise activity feature importance knowledge base $\mathcal{K}$ from labeled data.
    - Mechanism: For each activity pair $(a_i, a_j)$, human-interpretable statistical features are extracted across time-domain (mean, variance, RMS), frequency-domain (spectral entropy, dominant frequency), and cross-channel (correlation, tilt angle) dimensions. Permutation-based feature ranking (AutoGluon) estimates the importance score of each feature; cross-validation with weighted averaging ensures robustness. All feature–score tuples are indexed by activity pair.
    - Design Motivation: Converts implicit signal characteristics into verifiable linguistic priors. The pairwise organization allows the system to dynamically instantiate relevant knowledge for any candidate subset; adding a new activity requires only registering its statistical profile, with no retraining.

2. **Class-Wise Multi-Sensor Retrieval**:

    - Function: Retrieves the most relevant sensor signal evidence for each candidate activity class.
    - Mechanism: Maintains vector stores $\{\mathcal{D}^{loc}\}$ sharded by sensor placement, using a frozen time-series foundation encoder (Mantis by default) to generate embeddings indexed via FAISS IndexFlatIP. For each candidate class, top-$k$ evidence samples are retrieved independently. In multi-sensor scenarios, Reciprocal Rank Fusion (RRF) aggregates results across positions: $\text{RRF}(d) = \sum_{loc} \frac{1}{k_{rrf} + r_{loc}(d)}$.
    - Design Motivation: Class-conditional retrieval ensures balanced recall even for long-tail activities; sharding by sensor placement ensures that retrieved evidence is physically aligned with the query context.

3. **Hierarchical Multi-Agent Reasoning**:

    - Function: Iteratively reasons across four stages through three specialized LLM agent roles, ultimately producing a prediction and a natural-language explanation.
    - Mechanism: (1) A **Feature Selector** queries the knowledge base $\mathcal{K}$ to identify coarse-grained discriminative features; (2) an **Evidence Pruner** synthesizes the retrieved class evidence into a statistical contrast table and filters out distributionally inconsistent activities to obtain a refined candidate set $\mathcal{A}'$; (3) the Feature Selector retrieves fine-grained features over $\mathcal{A}'$; (4) a **Decision Insighter** analyzes the updated statistics to produce the final label and an interpretable reasoning explanation.
    - Design Motivation: The hierarchical design of progressively narrowing the hypothesis space avoids the difficulty of selecting among a large candidate set in a single step; stepwise refinement is more reliable than direct inference and produces interpretable intermediate results at each stage.

### Loss & Training

ZARA is a fully training-free inference framework and involves no loss functions or training procedures. The knowledge base is constructed via offline statistical analysis, and inference is conducted through frozen LLMs. All LLM agent temperatures are set to 0 to ensure deterministic reproducibility. For large-scale datasets (WISDM, DSADS), dynamic retrieval replaces static candidate lists, selecting the top-10 most relevant classes via cosine similarity.

## Key Experimental Results

### Main Results

Cross-subject generalization, 8 HAR datasets, frozen-parameter setting:

| Method | Avg. Acc | Avg. F1 | Type |
|--------|---------|---------|------|
| UniMTS | 39.4 | 32.1 | Contrastive Pre-training |
| IMU2CLIP | 22.7 | 17.9 | Contrastive Pre-training |
| ZARA (Qwen-30B) | 71.0 | 70.2 | Knowledge-Augmented Reasoning |
| ZARA (GPT-4.1-mini) | 77.5 | 77.2 | Knowledge-Augmented Reasoning |
| ZARA (Gemini) | **81.6** | **81.4** | Knowledge-Augmented Reasoning |

The best ZARA variant outperforms the strongest baseline (UniMTS) by 42.2 percentage points in accuracy.

### Ablation Study

| Configuration | Avg. Acc | Notes |
|---------------|---------|-------|
| ZARA (Full) | 81.6 | Gemini backbone |
| w/o Knowledge Base | Significant drop | Lack of statistical priors leaves reasoning ungrounded |
| Global retrieval instead of class-conditional | Drop | Insufficient recall for long-tail classes |
| Single-stage instead of hierarchical reasoning | Drop | Increased confusion without stepwise refinement |
| DTW instead of Mantis encoder | 71.0→81.6 | Mantis embeddings superior; DTW still viable |

### Key Findings

- ZARA's gains stem from its knowledge- and retrieval-augmented framework design rather than LLM backbone scale; all backbone variants from Qwen-30B to Gemini substantially outperform all baselines.
- Direct prompting methods (HARGPT, Gemini Text/Table/Plot) fail completely, demonstrating that even powerful LLMs cannot reason over raw numerical sensor streams without explicit reference grounding.
- ZARA's Acc and F1 are highly consistent, whereas baseline methods show large gaps between the two metrics, indicating that ZARA effectively recognizes long-tail activities through class-balanced retrieval.
- In cross-dataset generalization experiments, the cross-dataset knowledge setting outperforms both the no-knowledge and same-domain knowledge settings, confirming that statistical knowledge transfers across domains.

## Highlights & Insights

- **Signal-to-text knowledge distillation** is a particularly elegant contribution—converting the statistical characteristics of sensor time series into pairwise linguistic priors preserves physical interpretability while enabling LLMs to perform evidence-grounded reasoning. This paradigm is transferable to any scenario where LLMs must process numerical data.
- **Class-conditional retrieval** addresses the long-tail bias inherent in standard RAG—retrieving top-$k$ samples independently for each candidate class ensures sufficient evidence for minority classes. This design principle can be directly transferred to other classification-oriented RAG tasks.
- The framework is fully training-free and plug-and-play: adding a new activity requires only registering its statistical profile, with no retraining, realizing true open-set activity recognition.

## Limitations & Future Work

- Inference cost is relatively high: each query requires multiple rounds of LLM calls (feature selection → evidence pruning → re-selection → decision), limiting real-time applicability.
- Knowledge base construction still requires offline statistical analysis of labeled data, potentially limiting effectiveness when labeled data is scarce.
- Validation is currently limited to accelerometer/gyroscope sensors; applicability to additional sensor modalities (e.g., EMG, barometer) remains unexplored.
- For fine-grained activities that are highly similar in motion (e.g., different gait types), the discriminative power of statistical features may be insufficient.

## Related Work & Insights

- **vs. UniMTS**: UniMTS achieves classifier-free recognition by aligning synthetic skeletal motion with text, but lacks semantic granularity. ZARA surpasses it by 42 percentage points in accuracy under the same training-free setting through explicit statistical knowledge injection and class-conditional retrieval.
- **vs. HARGPT**: HARGPT feeds raw signals directly into LLMs as text or image prompts, incurring high token costs and severe information loss. ZARA converts signals into structured statistical knowledge before LLM input, fundamentally resolving the grounding problem.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The first work to apply a knowledge- and retrieval-augmented multi-agent framework to sensor time-series reasoning; the signal-to-text knowledge distillation paradigm is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 8 datasets, 10 baselines, cross-subject and cross-dataset evaluation protocols, comparisons across multiple LLM backbones, and extensive ablations.
- Writing Quality: ⭐⭐⭐⭐ — Structure is clear and method descriptions are detailed, though table formatting is somewhat verbose in LaTeX.
- Value: ⭐⭐⭐⭐ — Provides a new paradigm for LLM-based processing of numerical sensor data, though real-time inference cost remains a barrier to practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CoEvolve: Training LLM Agents via Agent-Data Mutual Evolution](coevolve_training_llm_agents_via_agent-data_mutual_evolution.md)
- [\[ICLR 2026\] VideoMind: A Chain-of-LoRA Agent for Temporal-Grounded Video Reasoning](../../ICLR2026/llm_agent/videomind_a_chain-of-lora_agent_for_temporal-grounded_video_reasoning.md)
- [\[ACL 2026\] Your LLM Agents are Temporally Blind: The Misalignment Between Tool Use Decisions and Human Time Perception](your_llm_agents_are_temporally_blind_the_misalignment_between_tool_use_decisions.md)
- [\[ACL 2026\] Waking Up Blind: Cold-Start Optimization of Supervision-Free Agentic Trajectories](waking_up_blind_cold-start_optimization_of_supervision-free_agentic_trajectories.md)
- [\[AAAI 2026\] Time, Identity and Consciousness in Language Model Agents](../../AAAI2026/llm_agent/time_identity_and_consciousness_in_language_model_agents.md)

</div>

<!-- RELATED:END -->
