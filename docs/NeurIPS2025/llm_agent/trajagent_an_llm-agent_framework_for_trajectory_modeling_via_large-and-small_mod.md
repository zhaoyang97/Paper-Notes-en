---
title: >-
  [Paper Note] TrajAgent: An LLM-Agent Framework for Trajectory Modeling via Large-and-Small Model Collaboration
description: >-
  [NeurIPS 2025][LLM Agent][trajectory modeling] This paper proposes TrajAgent — an LLM-agent-based framework for trajectory modeling that achieves automated, cross-task…
tags:
  - "NeurIPS 2025"
  - "LLM Agent"
  - "trajectory modeling"
  - "large-and-small model collaboration"
  - "automated machine learning"
  - "data augmentation"
date: 2026-05-08
content_hash: ade540276dc7b2fc
---

# TrajAgent: An LLM-Agent Framework for Trajectory Modeling via Large-and-Small Model Collaboration

**Conference**: NeurIPS 2025
**arXiv**: [2410.20445](https://arxiv.org/abs/2410.20445)  
**Code**: [GitHub](https://github.com/tsinghua-fib-lab/TrajAgent)  
**Area**: LLM Agent
**Keywords**: trajectory modeling, LLM agent, large-and-small model collaboration, automated machine learning, data augmentation

## TL;DR

This paper proposes TrajAgent — an LLM-agent-based framework for trajectory modeling that achieves automated, cross-task, and cross-dataset trajectory modeling through a unified environment (UniEnv), an automated workflow, and a collaborative learning schema between large and small models, outperforming baseline methods by 2.38%–69.91% across multiple tasks.

## Background & Motivation

**Broad applications of trajectory modeling**: Trajectory modeling encompasses pattern mining and future prediction over trajectory data, with wide applications in location-based services, urban transportation, and public administration. Representative tasks include trajectory prediction, recovery, classification, generation, and representation learning.

**Fundamental challenges in existing approaches**:

1. **Task fragmentation**: Each method is designed for a specific task and data format. For example, TrajFormer is limited to trajectory classification, and Flash-back only supports sparse check-in trajectory prediction, with no cross-task transferability.
2. **Data heterogeneity**: Trajectory data varies substantially in resolution, format, and geographic region, making it difficult to reuse models across datasets.
3. **Immature unified frameworks**: Existing unified frameworks (e.g., TrajFM, UniTraj) underperform task-specific models on individual tasks and involve complex training procedures.

**Opportunities with LLM agents**: LLMs, with their strong reasoning and commonsense capabilities, have already demonstrated success in automated software development (MetaGPT, ChatDev) and automated ML (HuggingGPT, MLAgentBench). This raises the question of whether LLM agents can be introduced into trajectory modeling to achieve automation and unification.

## Method

### Overall Architecture

TrajAgent comprises three core components:

1. **UniEnv**: A unified execution environment providing standardized data and model interfaces.
2. **Agentic Workflow**: A four-stage automated workflow (understand → plan → execute → summarize).
3. **Collaborative Learning Schema**: A mechanism for collaborative learning between large models (agents) and small models (specialized models).

### Key Designs

**1. UniEnv — Unified Environment**

- **Task interface**: Covers 5 trajectory modeling tasks (prediction, recovery, classification, generation, representation), comprising 9 sub-tasks and 18 methods.
- **Data interface**: Supports both check-in trajectories and GPS trajectories; data cleaning and standardization are performed via LLM-generated preprocessing scripts.
- **Model interface**: Integrates 18 models (GETNext, DeepMove, TrajBERT, etc.), each accompanied by a semantic description to assist agent selection.
- **External tools**: Integrates paper context retrieval (txyz.ai), hyperparameter optimization (Optuna), and trajectory visualization (movingpandas).

**2. Four-Stage Agentic Workflow**

- **Task understanding module**: Receives natural language instructions and identifies the task type and key information.
- **Task planning module**: Generates an execution plan (selecting datasets and models) based on the task description and data/model information in UniEnv.
- **Task execution module**: Invokes UniEnv to run experiments and interacts with the collaborative learning module.
- **Task summarization module**: Analyzes execution logs and produces an optimization summary report.

Each module is equipped with Reflexion-style memory and reflection mechanisms.

**3. Collaborative Learning Schema**

Learning operates at two levels:

- **Agent Learning via Reasoning** (high level): The agent performs reasoning-based learning from experimental records.
    - A "think then action" two-stage pipeline is designed.
    - Maintains long-term memory (all experimental data) and short-term memory (historical actions).
    - Supports Contrastive Reflection and Dynamic Memory Pruning.

- **Model Learning via Training** (low level): Specialized models undergo targeted training on the target data.
    - **Data augmentation**: Ten trajectory augmentation operators are defined (insert, replace, split, etc.); the agent selects the optimal operator combination.
    - **Parameter optimization**: The agent reads model parameter configuration files and generates code to update parameters.
    - **Joint optimization**: Data augmentation is performed before parameter optimization in sequential order.

### Loss & Training

TrajAgent does not define its own loss functions; instead, it orchestrates the training of existing models. The stopping condition for collaborative learning is either reaching a predefined performance threshold or exhausting the maximum number of exploration rounds.

Key improvement strategies:
- **Contrastive Reflection**: The agent explicitly compares successful and failed experiments, adjusting operator parameters to avoid repeating ineffective combinations.
- **Dynamic Memory Pruning**: Low-scoring memory entries are periodically discarded, retaining only high-performance trajectories as guidance.

## Key Experimental Results

### Main Results

Performance across 4 real-world datasets and 5 major tasks:

| Task | Sub-task | Model | Dataset | Original | TrajAgent (Joint Opt.) | Gain |
|------|----------|-------|---------|----------|------------------------|------|
| Trajectory Prediction | Next Location | GETNext | FSQ | 0.3720 (Acc@5) | 0.4002 | 7.58% |
| Trajectory Prediction | Next Location | LLM-ZS | FSQ | 0.3110 (Acc@5) | 0.3350 | 7.72% |
| Trajectory Prediction | Travel Time Est. | MulT-TTE | Porto | 163.12 (MAE) | 128.57 | 21.18% |
| Trajectory Recovery | Recovery | TrajBERT | Porto | 42.71 (MAE) | 27.78 | 34.96% |
| Trajectory Recovery | Map Matching | GraphMM | Tencent | 0.2014 (Acc) | 0.3422 | **69.91%** |
| Trajectory Classification | User Linking | S2TUL | FSQ | 0.5755 (Acc@5) | 0.7802 | 35.57% |
| Trajectory Classification | Intent Prediction | LIMP | Beijing | 0.745 (Acc) | 0.7627 | 2.38% |
| Trajectory Generation | Generation | DSTPP | Earthquake | 0.4611 (MAE) | 0.3584 | 22.27% |

### Ablation Study

**Comparison of LLM backends** (DeepMove + next location prediction):

| LLM | Overall Success Rate | Joint Opt. Acc@5 |
|-----|:-------------------:|:----------------:|
| Qwen2-7B | Low | 0.2668 |
| Mistral-7B-V3 | Medium | 0.2980 |
| GPT-3.5-Turbo | High | 0.3295 |
| Qwen2-72B | **Highest** | **0.4333** |
| GPT-4o-mini | High | 0.3724 |

**Workflow component ablation**:

| Configuration | Model Selection Acc | Joint Opt. Acc@5 |
|---------------|:-------------------:|:----------------:|
| Full TrajAgent | 98% | 0.3724 |
| w/o Reflection | 95%↓ | 0.3212↓ |
| w/o Memory | 80%↓ | 0.1804↓ |

### Key Findings

1. **Consistent gains across tasks and models**: TrajAgent yields positive improvements across all tested configurations, demonstrating the generality of the framework.
2. **LLM capability is critical**: Models at the 72B scale (Qwen2-72B) significantly outperform 7–9B models, particularly in the data augmentation and parameter optimization stages.
3. **Memory mechanism is essential**: Removing the memory module leads to substantial drops in both success rate and performance.
4. **Optimization trap phenomenon**: Excessive reasoning steps (>20) or memory entries (>10) can degrade performance.
5. **Comparison with Optuna**: TrajAgent achieves superior performance with fewer trial-and-error iterations and provides an additional 11.1% gain through joint optimization.

## Highlights & Insights

1. **Unified framework paradigm**: TrajAgent is the first framework to apply LLM agents to unified trajectory modeling, covering 5 major task categories.
2. **Large-and-small model collaboration**: LLMs handle high-level reasoning and strategic decision-making, while specialized small models handle low-level training and execution, leveraging the strengths of both.
3. **Closed-loop optimization system**: The cycle of agent reasoning → model training → performance feedback → agent learning forms a closed loop that enables automated optimization.
4. **In-depth failure mode analysis**: The paper provides detailed analysis of optimization traps and memory contamination, along with practical mitigation strategies (contrastive reflection and memory pruning).

## Limitations & Future Work

1. **Agent efficiency**: Multi-round reasoning and training iterations introduce substantial computational overhead and API call costs.
2. **Poor performance with weak LLMs**: The framework is highly dependent on high-capability LLMs; 7B-scale models exhibit low success rates at critical stages.
3. **Limited data augmentation strategies**: The 10 augmentation operators for check-in trajectories are predefined, lacking the ability to adaptively generate new operators.
4. **Geographic generalization**: Experiments are conducted on a limited set of cities and datasets; cross-region generalization has not been thoroughly validated.
5. **Insufficient interpretability**: The agent's decision-making process (e.g., rationale for selecting a particular model or augmentation operator) lacks natural language explanation output.

## Related Work & Insights

- **HuggingGPT / VisionLLM**: Pioneering works on using LLMs to manage and schedule multiple AI models.
- **Reflexion**: A framework for agent reinforcement learning via linguistic reflection; TrajAgent's memory and reflection mechanisms draw directly from this work.
- **AutoML (Optuna)**: Conventional parameter optimization approach; TrajAgent surpasses pure search methods through LLM-based reasoning.
- **CityGPT / UrbanLLM**: LLM applications in urban computing, but primarily relying on fine-tuning rather than agent frameworks.
- Implications for agent framework design: Memory management (both size and content) is a critical factor influencing the performance of agent systems.

## Rating

⭐⭐⭐⭐ (4/5)

The paper presents a complete and systematic LLM agent framework for trajectory modeling with broad experimental coverage (5 major tasks, 18 models, 4 datasets) and significant performance gains (up to 69.91%). The design philosophy of large-and-small model collaborative learning is novel, and the failure mode analysis is thorough and practically valuable. The primary limitations include a strong dependence on high-capability LLMs, considerable computational overhead, and a restricted set of datasets and cities in the experiments. Overall, this is a solid and practically valuable systems contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Traj-CoA: Patient Trajectory Modeling via Chain-of-Agents for Lung Cancer Risk Prediction](traj-coa_patient_trajectory_modeling_via_chain-of-agents_for_lung_cancer_risk_pr.md)
- [\[NeurIPS 2025\] AgentTTS: Large Language Model Agent for Test-time Compute-optimal Scaling Strategy in Complex Tasks](agenttts_large_language_model_agent_for_testtime_computeopti.md)
- [\[NeurIPS 2025\] Distilling LLM Agent into Small Models with Retrieval and Code Tools](distilling_llm_agent_into_small_models_with_retrieval_and_co.md)
- [\[NeurIPS 2025\] Zero-Shot Large Language Model Agents for Fully Automated Radiotherapy Treatment Planning](zero-shot_large_language_model_agents_for_fully_automated_radiotherapy_treatment.md)
- [\[NeurIPS 2025\] BTL-UI: Blink-Think-Link Reasoning Model for GUI Agent](btlui_blinkthinklink_reasoning_model_for_gui_agent.md)

</div>

<!-- RELATED:END -->
