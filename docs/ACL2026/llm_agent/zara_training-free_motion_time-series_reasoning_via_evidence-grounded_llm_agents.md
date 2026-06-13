---
title: >-
  [Paper Note] ZARA: Training-Free Motion Time-Series Reasoning via Evidence-Grounded LLM Agents
description: >-
  [ACL 2026][LLM Agent][Human Activity Recognition] Ours proposes ZARA, a knowledge- and retrieval-augmented multi-agent framework. By distilling sensor signals into structured textual knowledge bases…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Human Activity Recognition"
  - "Time-Series Reasoning"
  - "Retrieval-Augmented Generation"
  - "Multi-Agent Reasoning"
  - "Training-Free"
date: 2026-05-08
content_hash: 81417a9e4928cc88
---

# ZARA: Training-Free Motion Time-Series Reasoning via Evidence-Grounded LLM Agents

**Conference**: ACL 2026  
**arXiv**: [2508.04038](https://arxiv.org/abs/2508.04038)  
**Code**: [https://github.com/zechenli03/ZARA](https://github.com/zechenli03/ZARA)  
**Area**: LLM Agent  
**Keywords**: Human Activity Recognition, Time-Series Reasoning, Retrieval-Augmented Generation, Multi-Agent Reasoning, Training-Free

## TL;DR

Ours proposes ZARA, a knowledge- and retrieval-augmented multi-agent framework. By distilling sensor signals into structured textual knowledge bases, employing class-conditional retrieval, and performing hierarchical LLM reasoning, it achieves interpretable Human Activity Recognition (HAR) in a completely training-free setting, significantly outperforming existing methods across 8 datasets.

## Background & Motivation

**Background**: Human Activity Recognition (HAR) is a core technology for applications such as digital health and adaptive interfaces. Current mainstream methods rely on task-specific deep neural networks that require supervised training under fixed sensor configurations and activity categories.

**Limitations of Prior Work**: Existing methods face three major bottlenecks: (1) Poor generalization—adapting to new users or hardware requires costly model retraining; (2) Limited training-free adaptation—time-series foundation models like Moment and Mantis provide transferable representations but still require training specific classification heads, while contrastive learning methods like UniMTS struggle to distinguish fine-grained activities in frozen-parameter settings; (3) Lack of interpretability—most methods only output category predictions without a transparent reasoning process.

**Key Challenge**: Although LLMs possess powerful open-set reasoning capabilities, directly inputting numerical time-series into LLMs leads to hallucinations and weak grounding, as LLMs cannot intuitively understand physical dynamics from raw numerical streams.

**Goal**: To build a completely training-free HAR framework capable of generalizing across users and datasets while providing an interpretable reasoning process.

**Key Insight**: The authors observe that just as RAG in NLP relies on high-quality document corpora, RAG in HAR requires a domain-specific knowledge base to translate implicit statistical patterns of physical motion in sensor data into verifiable natural language priors (e.g., "the vertical acceleration variance of running is higher than that of walking").

**Core Idea**: Distill statistical features of sensor signals into pairwise textual knowledge bases, combining class-conditional retrieval and hierarchical multi-agent reasoning to achieve evidence-based training-free activity recognition.

## Method

### Overall Architecture

The core design of ZARA decouples information into two sources: (1) **Global Knowledge $\mathcal{K}$**—a static reference registry storing pairwise activity feature importance profiles; (2) **Local Evidence $\mathcal{D}$**—a vector database of raw signal embeddings serving as external memory for local distribution grounding. The workflow consists of: offline construction of the knowledge base → online retrieval of relevant evidence → hierarchical multi-agent reasoning to output predictions and explanations.

### Key Designs

1.  **Offline Statistical Profiling**:

    *   Function: Automatically constructs a pairwise activity feature importance knowledge base $\mathcal{K}$ from labeled data.
    *   Mechanism: For each activity pair $(a_i, a_j)$, human-interpretable statistical features are extracted from the time domain (mean, variance, RMS), frequency domain (spectral entropy, dominant frequency), and cross-channel domain (correlation, tilt angles). Feature importance scores are estimated using permutation-based feature ranking (AutoGluon), with cross-validation ensuring robustness through weighted averaging. All feature-score tuples are stored indexed by activity pairs.
    *   Design Motivation: Transforms implicit signal features into verifiable linguistic priors. Pairwise organization allows the system to dynamically instantiate relevant knowledge for any candidate subset. Adding new activities only requires registering their statistical profiles without retraining.

2.  **Class-Wise Multi-Sensor Retrieval**:

    *   Function: Retrieves the most relevant sensor signal evidence for each candidate activity category.
    *   Mechanism: Maintains vector databases $\{\mathcal{D}^{loc}\}$ partitioned by sensor location, using a frozen time-series foundation encoder (Mantis by default) to generate embeddings indexed via FAISS IndexFlatIP. For a query signal, top-k evidence is retrieved separately for each candidate category. In multi-sensor scenarios, evidence is aggregated via Reciprocal Rank Fusion (RRF): $\text{RRF}(d) = \sum_{loc} \frac{1}{k_{rrf} + r_{loc}(d)}$.
    *   Design Motivation: Class-conditional retrieval ensures balanced recall even for long-tail activities. Partitioning by sensor location ensures that retrieved evidence aligns with the physical context of the query.

3.  **Hierarchical Multi-Agent Reasoning**:

    *   Function: Iteratively reasons through three specialized LLM agent roles across four stages to ultimately output a prediction and a natural language explanation.
    *   Mechanism: (1) The **Feature Selector** queries the knowledge base $\mathcal{K}$ to determine coarse-grained discriminative features; (2) The **Evidence Pruner** synthesizes retrieved category evidence into statistical contrast tables and filters activities with distribution mismatches to obtain a refined candidate set $\mathcal{A}'$; (3) The Feature Selector retrieves fine-grained features on $\mathcal{A}'$ again; (4) The **Decision Insighter** analyzes updated statistics to derive the final label and generate interpretable reasoning notes.
    *   Design Motivation: The hierarchical design reduces the hypothesis space, avoiding the difficulty of choosing among many candidates at once. Incremental refinement is more reliable than direct reasoning and produces interpretable intermediate results at each step.

### Loss & Training

ZARA is a completely training-free reasoning framework and does not involve loss functions or a training process. The knowledge base is constructed via offline statistical analysis, and reasoning is performed through frozen LLMs. All LLM agent temperatures are set to 0 to ensure deterministic reproducibility. For large-scale datasets (WISDM, DSADS), dynamic retrieval replaces static candidate lists, selecting the top-10 most relevant categories via cosine similarity.

## Key Experimental Results

### Main Results

Cross-Subject generalization, 8 HAR datasets, frozen-parameter setting:

| Method | Avg Acc | Avg F1 | Type |
|------|---------|---------|------|
| UniMTS | 39.4 | 32.1 | Contrastive Pre-training |
| IMU2CLIP | 22.7 | 17.9 | Contrastive Pre-training |
| ZARA (Qwen-30B) | 71.0 | 70.2 | Knowledge-Augmented Reasoning |
| ZARA (GPT-4.1-mini) | 77.5 | 77.2 | Knowledge-Augmented Reasoning |
| ZARA (Gemini) | **81.6** | **81.4** | Knowledge-Augmented Reasoning |

The best variant of ZARA exceeds the strongest baseline UniMTS by 42.2 percentage points in Acc.

### Ablation Study

| Configuration | Avg Acc | Notes |
|------|---------|------|
| ZARA (Full) | 81.6 | Gemini backbone |
| Without KB | Significant Drop | Lack of statistical priors leaves reasoning without evidence |
| Global Retrieval instead of Class-wise | Decrease | Insufficient recall for long-tail categories |
| Single-stage Reasoning instead of Hierarchical | Decrease | Lack of incremental refinement increases confusion |
| DTW instead of Mantis encoder | 71.0→81.6 | Mantis embeddings are superior, but DTW is also usable |

### Key Findings

*   The gains of ZARA stem from the framework design of knowledge and retrieval augmentation rather than the scale of the LLM backbone; versions from Qwen-30B to Gemini all significantly outperform all baselines.
*   Direct prompting methods (HARGPT, Gemini Text/Table/Plot) fail completely, proving that without explicit reference grounding, even powerful LLMs cannot reason over numerical sensor streams.
*   ZARA's Acc and F1 are highly consistent, whereas baseline methods show large gaps between Acc and F1, indicating that ZARA effectively identifies long-tail activities through class-balanced retrieval.
*   In cross-dataset generalization experiments, the Cross-Dataset Knowledge setting outperforms no-knowledge and in-domain knowledge settings, proving that statistical knowledge possesses cross-domain transferability.

## Highlights & Insights

*   **Signal-to-text knowledge distillation** is very ingenious—it transforms statistical features of sensor time-series into pairwise linguistic priors, preserving physical interpretability while enabling LLMs to perform evidence-based reasoning. This paradigm is transferable to any scenario requiring LLMs to process numerical data.
*   **Class-conditional retrieval** addresses the long-tail bias problem in standard RAG—retrieving top-k independently for each candidate category ensures that minority classes have sufficient evidence. This design idea can be directly transferred to other classification-based RAG tasks.
*   Completely training-free and plug-and-play: Adding new activities only requires registering statistical profiles without retraining, achieving true open-set activity recognition.

## Limitations & Future Work

*   High inference cost: Each query requires multiple rounds of LLM calls (Feature Selection → Evidence Pruning → Re-selection → Decision), limiting real-time applications.
*   Knowledge base construction still requires offline statistical analysis of labeled data, which may be limited when labeled data is extremely scarce.
*   Currently validated only on accelerometer/gyroscope sensors; the applicability to more sensor modalities (e.g., EMG, barometer) remains to be explored.
*   For fine-grained activities with very similar motions (e.g., different types of gait), the discriminative power of statistical features may be insufficient.

## Related Work & Insights

*   **vs UniMTS**: UniMTS achieves classifier-free recognition by aligning synthetic skeletal motion with text but lacks semantic granularity. ZARA, through explicit statistical knowledge injection and class-conditional retrieval, achieves 42 percentage points higher Acc in the same training-free setting.
*   **vs HARGPT**: HARGPT directly inputs raw signals as text/image prompts into LLMs, leading to high token costs and severe information loss. ZARA transforms signals into structured statistical knowledge before inputting them into LLMs, fundamentally solving the grounding problem.

## Rating

*   Novelty: ⭐⭐⭐⭐⭐ The first work to apply a knowledge- and retrieval-augmented multi-agent framework to sensor time-series reasoning; the signal-to-text knowledge distillation paradigm is very novel.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 datasets, 10 baselines, two evaluation protocols (cross-subject and cross-dataset), multiple LLM backbone comparisons, and rich ablations.
*   Writing Quality: ⭐⭐⭐⭐ The structure is clear and the method description is detailed, though table formatting in LaTeX is slightly verbose.
*   Value: ⭐⭐⭐⭐ Provides a new paradigm for LLMs to handle numerical sensor data, although real-time inference cost is a barrier to practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CoEvolve: Training LLM Agents via Agent-Data Mutual Evolution](coevolve_training_llm_agents_via_agent-data_mutual_evolution.md)
- [\[ACL 2026\] SafeMCP: Proactive Power Regulation for LLM Agent Defense via Environment-Grounded Look-Ahead Reasoning](safemcp_proactive_power_regulation_for_llm_agent_defense_via_environment-grounde.md)
- [\[ACL 2026\] Your LLM Agents are Temporally Blind: The Misalignment Between Tool Use Decisions and Human Time Perception](your_llm_agents_are_temporally_blind_the_misalignment_between_tool_use_decisions.md)
- [\[ICLR 2026\] VideoMind: A Chain-of-LoRA Agent for Temporal-Grounded Video Reasoning](../../ICLR2026/llm_agent/videomind_a_chain-of-lora_agent_for_temporal-grounded_video_reasoning.md)
- [\[ACL 2026\] IntrAgent: An LLM Agent for Content-Grounded Information Retrieval through Literature Review](intragent_an_llm_agent_for_content-grounded_information_retrieval_through_litera.md)

</div>

<!-- RELATED:END -->
