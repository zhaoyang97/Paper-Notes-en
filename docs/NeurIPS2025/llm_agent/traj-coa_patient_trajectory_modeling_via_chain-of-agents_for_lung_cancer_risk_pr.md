---
title: >-
  [Paper Note] Traj-CoA: Patient Trajectory Modeling via Chain-of-Agents for Lung Cancer Risk Prediction
description: >-
  [NeurIPS 2025][LLM Agent][multi-agent system] This paper proposes Traj-CoA, a multi-agent framework that employs a chain-of-agents architecture with an EHRMem long-term memory module to perform temporal reasoning over long, noisy longitudinal EHRs. The framework surpasses ML/DL/BERT/LLM baselines on zero-shot lung cancer risk prediction tasks (5-year EHR data, up to 160k tokens).
tags:
  - "NeurIPS 2025"
  - "LLM Agent"
  - "multi-agent system"
  - "EHR"
  - "patient trajectory"
  - "temporal reasoning"
  - "lung cancer prediction"
date: 2026-05-08
content_hash: 66d3950b3ef3a310
---

# Traj-CoA: Patient Trajectory Modeling via Chain-of-Agents for Lung Cancer Risk Prediction

**Conference**: NeurIPS 2025  
**arXiv**: [2510.10454](https://arxiv.org/abs/2510.10454)  
**Code**: None  
**Area**: LLM Agent  
**Keywords**: multi-agent system, EHR, patient trajectory, temporal reasoning, lung cancer prediction

## TL;DR
This paper proposes Traj-CoA, a multi-agent framework that employs a chain-of-agents architecture with an EHRMem long-term memory module to perform temporal reasoning over long, noisy longitudinal EHRs. The framework surpasses ML/DL/BERT/LLM baselines on zero-shot lung cancer risk prediction tasks (5-year EHR data, up to 160k tokens).

## Background & Motivation
**Background**: Longitudinal electronic health records (EHRs) contain rich temporal data suitable for patient trajectory modeling and clinical outcome prediction. LLMs have demonstrated promise in zero-shot clinical prediction, offering a potential alternative to methods that require complex feature engineering and task-specific training.

**Limitations of Prior Work**: EHR data presents two core challenges: (1) **Extremely long contexts**: Patient histories spanning multiple years frequently exceed 128k tokens, surpassing the effective processing range of LLMs and causing the "lost-in-the-middle" problem; (2) **Inherent noise**: EHRs are designed for clinical care rather than research, and thus contain inconsistent formatting, entry errors, missing data, irregular sampling, and large volumes of irrelevant information that obscure key predictive signals.

**Key Challenge**: Existing LLM-based methods are limited to short EHRs (<16k tokens) or ICU data; temporal reasoning over long longitudinal EHRs exceeding 32k or even 128k tokens remains an open challenge. Naively expanding the context window degrades performance.

**Goal**: To perform effective temporal reasoning over extremely long and noisy longitudinal EHR data without any additional training (zero-shot).

**Key Insight**: Drawing inspiration from the chain-of-agents multi-agent collaboration architecture, combined with an external long-term memory module specifically designed for EHRs, the paper decomposes long-context reasoning into a chain of short-context reasoning steps handled by cooperating agents.

**Core Idea**: A chain of worker agents processes time-aware segmented EHR chunks sequentially; EHRMem retains key clinical events; a manager agent synthesizes summaries and memory to produce the final prediction.

## Method

### Overall Architecture
Traj-CoA comprises three core components: (1) a data preprocessing pipeline (XML unified format + time-aware chunking); (2) a chain-of-agents workflow (sequential worker agents + a manager agent for final decision-making); and (3) the EHRMem long-term memory module.

### Key Designs
1. **Data Unification and Time-Aware Chunking (Data Preprocessing)**:

    - **XML Unified Format**: All multimodal patient history (diagnosis codes, lab results, vital signs, clinical notes, imaging reports) is converted into a single XML format. Records are organized chronologically, with a root node containing demographic information followed by timestamped nested event records. This leverages LLMs' strong ability to interpret structured tag-based data.
    - **Time-Aware Chunking**: Rather than fixed-size hard chunking, the data is dynamically segmented by timestamp units while respecting a maximum $k$-token limit, thereby preserving temporal integrity. When a single timestamp record exceeds $k$ tokens, it is further split while retaining the original timestamp. This produces $C$ temporally coherent chunks $\{c_1, c_2, \ldots, c_C\}$, where $C$ varies per patient.

2. **Chain-of-Agents Workflow**:

    - **Stage 1 – Worker Agents**: A sequence of worker agents $W_i$ processes each chunk $c_i$ in order. Each agent receives the current chunk, task instruction $I_W$, and the preceding agent's summary $S_{i-1}$, extracts task-relevant salient information, analyzes temporal patterns, and produces an updated summary $S_i$. Formally: $S_i = W_i(I_W, S_{i-1}, c_i)$. This enables progressive information aggregation across the entire longitudinal EHR.
    - **Stage 2 – Manager Agent**: Receives the final worker summary $S_C$ and task instruction $I_M$, synthesizes the information, and produces the final output $O$. Formally: $O = M(I_M, S_C)$.

3. **EHRMem Long-Term Memory Module**:

    - **Design Motivation**: Directly applying vanilla CoA causes critical clinical events to become progressively abstracted and lost during propagation through long sequences (the "forgetting" problem).
    - **Mechanism**: Task-relevant events and their timestamps are stored in a structured memory $\mathcal{M}$. While processing each chunk, the worker agent extracts potentially relevant clinical events and risk factors into $\mathcal{M}$. A deduplication mechanism is employed: each agent's prompt includes the most recent $k$ events from $\mathcal{M}$, and only novel, previously unrecorded information is stored, preventing redundancy caused by EHR "copy-forwarding."
    - **Enhanced Reasoning**: The manager agent's decision is based jointly on the final summary $S_C$ and the complete memory $\mathcal{M}$. Formally: $S_i, E_i = W_i(I_W, S_{i-1}, c_i, \mathcal{M}[-k:])$; $\mathcal{M} \leftarrow \mathcal{M} \oplus E_i$; $O = M(I_M, S_C, \mathcal{M})$.
    - **Inclusive Extraction Strategy**: Worker agents intentionally extract a slightly broader set of potentially relevant events rather than applying strict filtering. Since local agents lack global context to assess the long-term importance of individual events, the final importance judgment is delegated to the manager agent, which has access to the complete temporal context.

### Loss & Training
- **Zero-shot setting**: No training is required; the framework is driven entirely by task-specific prompts.
- Base model: MedGemma-27B.
- Default chunk size: 8k tokens; maximum 15 chunks (supporting up to 120k tokens of context).
- For supervised baselines (ML/DL/BERT), 12,266 training samples and 1,363 validation samples are used.

## Key Experimental Results

### Main Results (300 test samples, 28 positive / 272 control)

| Model Family | Model | Prediction Mode | Context Window | AUROC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| ML | LR | SFT | — | 0.741 | 0.306 | 0.393 | 0.344 |
| ML | XGBoost | SFT | — | 0.763 | 0.367 | 0.393 | 0.379 |
| DL | RETAIN | SFT | — | 0.757 | 0.346 | 0.321 | 0.333 |
| DL | PatientTM | SFT | — | 0.730 | 0.361 | 0.464 | 0.406 |
| BERT | C-MBERT | SFT | 8k | 0.749 | 0.367 | 0.393 | 0.379 |
| LLM | Vanilla | Zero-shot | 32k | 0.743 | 0.345 | 0.357 | 0.351 |
| LLM | RAG | Zero-shot | 1k×32 | 0.753 | 0.221 | 0.607 | 0.324 |
| LLM | Traj-CoA (w/o EHRMem) | Zero-shot | 8k×15 | 0.748 | 0.183 | 0.821 | 0.299 |
| LLM | **Traj-CoA** | **Zero-shot** | **8k×15** | **0.766±0.019** | **0.358±0.057** | **0.436±0.105** | **0.380±0.018** |

### Ablation Study & Sensitivity Analysis

| Analysis Dimension | Setting | AUROC | Key Finding |
|---|---|---|---|
| EHRMem ablation | Remove EHRMem | 0.748 | AUROC drops 1.8%, F1 drops 8.1%, confirming the necessity of long-term memory |
| Chunk size | 2k (total context fixed at 80k) | Lower | Excessively long agent chains lead to catastrophic forgetting |
| Chunk size | 8k (total context fixed at 80k) | Peak | Optimal trade-off between local detail retention and global aggregation |
| Chunk size | 16k (total context fixed at 80k) | Lower | Shorter chain but each agent suffers from "lost-in-the-middle" |
| Context window | 40k→160k (8k chunk) | Consistently improves | Contrasts with vanilla LLM performance degradation at 64k |
| Vanilla LLM | 32k→64k | 0.743→0.714 | Expanding context window degrades performance |

### Key Findings
- **Traj-CoA surpasses most supervised baselines in the zero-shot setting**, with AUROC 0.766 comparable to the best SFT baseline XGBoost (0.763).
- **EHRMem is a critical component**: Removing it causes F1 to drop sharply from 0.380 to 0.299, demonstrating that pure summary propagation loses key signals.
- **Unique long-context scaling capability**: Traj-CoA performance improves consistently as the context window expands to 160k, whereas vanilla LLM performance degrades at 64k.
- **Temporal reasoning is clinically meaningful**: Salient events identified by the model span 7 major categories including diagnoses, symptoms, and lab tests; top themes (advanced age, anemia, COPD, cough, inflammatory markers, pulmonary nodules, pneumonia, pulmonary function, smoking, weight loss) are highly consistent with clinical screening guidelines.
- **Full timeline utilization**: The event distribution shows that the model attends to events from the most recent year while still extracting valuable information from earlier history.

## Highlights & Insights
- **Elegant EHRMem design**: The combination of inclusive extraction, deduplication, and the division of labor between local extraction and global judgment by the manager agent elegantly addresses the information forgetting problem in long sequences.
- **Time-aware chunking outperforms fixed chunking**: Dynamic segmentation by timestamp preserves temporal integrity and avoids hard boundaries that can split clinical events.
- **Zero-shot performance rivals supervised learning**: The framework achieves performance comparable to XGBoost without any training data, demonstrating its generality.
- **Insightful chunk-size trade-off**: The analysis reveals a fundamental tension between small chunks causing catastrophic forgetting and large chunks causing lost-in-the-middle effects.

## Limitations & Future Work
- **Single-institution, small-scale evaluation**: Validation is conducted on only 300 test samples from one medical institution; generalizability remains to be demonstrated.
- **Single prediction task**: Only lung cancer risk prediction is evaluated; the task-agnostic nature of the framework requires verification across more clinical scenarios.
- **Dependence on carefully designed prompts**: Task-specific prompt engineering is necessary, limiting fully automated deployment.
- **Higher computational complexity than RAG**: Encoding complexity is $O(L \cdot L_C)$, higher than RAG's $O(L_R^2)$, requiring a trade-off between latency and completeness.
- **Dataset class imbalance**: The 1:10 case-to-control ratio may affect the robustness of the evaluation.
- Potential improvements include: incorporating external knowledge augmentation, stronger base models, multi-agent training optimization, and automatic prompt optimization.

## Related Work & Insights
- **vs. Vanilla LLM**: Directly applying LLMs to long EHRs suffers from context degradation (AUROC drops to 0.714 at 64k); Traj-CoA overcomes this bottleneck through a divide-and-conquer strategy.
- **vs. RAG**: RAG reduces latency through selective retrieval but risks information loss; Traj-CoA processes the full context, trading speed for comprehensiveness.
- **vs. Original Chain-of-Agents**: The original CoA lacks long-term memory; directly applied to EHRs, it causes early critical events to be forgotten. EHRMem is the key innovation.
- **vs. EHR foundation models (BEHRT, etc.)**: These are constrained by limited code vocabularies and short contexts (<16k), preventing full exploitation of unstructured text and ultra-long histories.

## Rating
- Novelty: ⭐⭐⭐⭐ Combines multi-agent architecture with EHR-specific long-term memory requirements; EHRMem design is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 4 categories of baselines, ablation, sensitivity analysis, temporal reasoning analysis, and clinical relevance validation.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear, method description is detailed, and analysis is deep and insightful.
- Value: ⭐⭐⭐⭐ Provides a viable framework for zero-shot temporal reasoning over longitudinal EHRs, with practical potential for clinical AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] TrajAgent: An LLM-Agent Framework for Trajectory Modeling via Large-and-Small Model Collaboration](trajagent_an_llm-agent_framework_for_trajectory_modeling_via_large-and-small_mod.md)
- [\[NeurIPS 2025\] ShapeCraft: LLM Agents for Structured, Textured and Interactive 3D Modeling](shapecraft_llm_agents_for_structured_textured_and_interactive_3d_modeling.md)
- [\[ICLR 2026\] FutureX: An Advanced Live Benchmark for LLM Agents in Future Prediction](../../ICLR2026/llm_agent/futurex_an_advanced_live_benchmark_for_llm_agents_in_future_prediction.md)
- [\[ICLR 2026\] VideoMind: A Chain-of-LoRA Agent for Temporal-Grounded Video Reasoning](../../ICLR2026/llm_agent/videomind_a_chain-of-lora_agent_for_temporal-grounded_video_reasoning.md)
- [\[ACL 2025\] Explorer: Scaling Exploration-Driven Web Trajectory Synthesis for Multimodal Web Agents](../../ACL2025/llm_agent/explorer_scaling_exploration-driven_web_trajectory_synthesis_for_multimodal_web_.md)

</div>

<!-- RELATED:END -->
