---
title: >-
  [Paper Note] Neural Bandit Based Optimal LLM Selection for a Pipeline of Tasks
description: >-
  [AAAI 2026][Medical Imaging][LLM selection] This paper proposes Sequential Bandits, an online learning method based on neural contextual multi-armed bandits…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "LLM selection"
  - "multi-armed bandit"
  - "neural contextual bandit"
  - "pipeline tasks"
  - "medical diagnosis prediction"
  - "cost-awareness"
date: 2026-05-08
content_hash: 222ed1d6a1517102
---

# Neural Bandit Based Optimal LLM Selection for a Pipeline of Tasks

**Conference**: AAAI 2026
**arXiv**: [2508.09958](https://arxiv.org/abs/2508.09958)  
**Code**: Available (includes self-constructed dataset)  
**Area**: Medical Imaging / LLM Routing & Selection
**Keywords**: LLM selection, multi-armed bandit, neural contextual bandit, pipeline tasks, medical diagnosis prediction, cost-awareness

## TL;DR
This paper proposes Sequential Bandits, an online learning method based on neural contextual multi-armed bandits, for selecting the optimal LLM for each subtask in a task pipeline (e.g., "summarization → diagnosis"). The method jointly optimizes accuracy and cost, and outperforms existing bandit baselines on two pipeline task benchmarks: medical diagnosis prediction and telecommunications QA.

## Background & Motivation

**Background**: As the number of LLMs proliferates (including custom assistants on platforms such as OpenAI/Azure), selecting the best LLM for a specific task has become a critical challenge. Different LLMs exhibit substantial performance variation across tasks, and naively choosing the largest model is both expensive and potentially suboptimal.

**Challenges of Pipeline Tasks**: Many complex tasks must be decomposed into subtasks (e.g., medical report → summarization → diagnosis), each of which may be handled by a different LLM. Key difficulties include:
   - **Cascading dependencies** between subtasks: the output quality of a preceding LLM affects the input and performance of subsequent LLMs
   - Exponential growth in the number of LLM combinations
   - Cost–accuracy trade-off: stronger models are generally more expensive

**Limitations of Prior Work**:
   - **LLM cascading**: tries models from cheap to expensive in a predefined order, but incurs wasted inference calls and does not allow flexible pipeline configuration
   - **LLM routing**: selects a single LLM once, which is not applicable to multi-subtask pipelines
   - **Standard contextual bandits**: do not support sequential dependencies between subtasks
   - **Combinatorial bandits**: select all arms simultaneously and cannot make decisions after observing intermediate results

**Key Insight**: Online learning combined with sequential bandits — training an independent neural network for each (subtask, LLM) pair, leveraging UCB-based exploration–exploitation trade-off, with an added cost penalty term.

## Method

### Problem Formulation

- Input: query $q_t$ decomposed into subtasks $\{T_1, T_2, \ldots, T_k\}$ (DAG structure)
- Each subtask has $N_i$ candidate LLMs (arms)
- A super arm is selected: $S_t = (a_{1,j}, a_{2,j}, \ldots, a_{k,j})$
- Objective: maximize the net reward $N(t) = R(S_t, \mathbf{r}_t) - \boldsymbol{\alpha} \cdot \mathbf{C}(S_t)$, where $\boldsymbol{\alpha}$ controls the accuracy–cost trade-off

### Core Algorithm: Sequential Bandits

**An independent neural network is maintained for each (subtask, LLM) pair** to predict the reward:

$$f_{i,j}(\mathbf{x}; \boldsymbol{\theta}) = \sqrt{m} \mathbf{W}^{(L)}_{i,j} \sigma(\mathbf{W}^{(L-1)}_{i,j} \sigma(\cdots \sigma(\mathbf{W}^{(0)}_{i,j} \mathbf{x})))$$

**UCB selection strategy**: for each candidate LLM $j$ of subtask $i$, compute:

$$u_{i,j} = f_{i,j}(p_i, d_j) + \left\|\frac{\mathbf{g}_{i,j}(\mathbf{x}_t(a_{i,j}); \boldsymbol{\theta}^{t-1}_{i,j})}{\sqrt{n}}\right\|_{\mathbf{Z}^{-1}_{t-1}(a_{i,j})} - \alpha_i C_j(p_i)$$

- First term: expected reward predicted by the neural network (exploitation)
- Second term: gradient-based uncertainty estimate (exploration)
- Third term: cost penalty (cost sensitivity)

**Sequential decision process**:
1. Subtask $T_1$: takes the raw query $q_t$ as input and selects the LLM with the highest UCB score
2. Subtask $T_i$ ($i>1$): takes the output of the preceding subtask's LLM as input and again selects the highest UCB LLM
3. After all subtasks are executed, the base arm rewards and the super arm reward are observed
4. Only the neural network weights of the selected LLMs are updated

### Cost Modeling

- A BERT regression model (trained on LMSYS-Chat-1M) is used to predict output token counts
- Cost = input token count × unit price + predicted output token count × unit price (Azure pricing)

### Key Differences from Existing Neural Bandits

1. **Independent networks vs. shared network**: training separate networks for each (subtask, LLM) pair avoids the problem where a shared network causes reward estimates for different arms to converge, leading to premature commitment to a suboptimal arm
2. No additional training overhead: only the selected LLM's network is updated per round
3. Supports sequential dependencies between subtasks

## Key Experimental Results

### Datasets
1. **Medical Diagnosis Prediction** (self-constructed): 100 patient reports constructed from MIMIC-III with diagnosis-related content removed; subtasks are "summarization → diagnosis" (2 subtasks)
2. **TeleQnA**: 10,000 multiple-choice questions in the telecommunications domain; subtasks are "summarization → answering → explanation" (3 subtasks)

### Model Pool
- GPT-3.5-turbo, GPT-4o, Llama-3.3-70B-instruct, Mistral-3B, Phi-4
- Domain fine-tuned models: Med (general medical), Tele (telecommunications), Med III (fine-tuned on MIMIC-III)
- GPT-3.5-turbo assistants (with retrieval augmentation)
- Cost ranking (low to high): GPT-3.5-turbo < Llama 3.3 < Med < Tele < Med III

### Baselines
- Random: random selection
- Llama: fixed selection of Llama (best single-task model)
- Cost-Aware NeuralUCB: one shared network per subtask
- Cost-Aware NeuralLinUCB: neural network with a linear layer

### Main Results

| Setting | Sequential Bandits vs. Strongest Baseline |
|---------|-------------------------------------------|
| Medical Diagnosis (net reward) | +7.60% vs. Llama |
| Telecommunications QA (net reward) | +6.51% vs. Random |

**Key Findings**:
- Although Llama achieves the highest single-task accuracy on medical diagnosis, Sequential Bandits still outperforms fixed Llama selection, indicating that the algorithm learns a superior cost–accuracy combination
- Random selection unexpectedly outperforms CA-NeuralUCB and NeuralLinUCB in the telecommunications setting

### LLM Selection Analysis

- Sequential Bandits most frequently selects Llama (49.1%) and GPT-3.5 (39.2%) for the diagnosis subtask
- These two models are the cheapest while also achieving the highest and second-highest accuracy
- Baseline methods more frequently select suboptimal and more expensive models (e.g., Med)

## Highlights & Insights

1. **First work to study LLM pipeline selection**: extends LLM selection from "single-point routing" to "sequential pipelines," with clear practical relevance
2. **Pipeline composition changes optimal model selection**: introducing a summarization step shifts the optimal diagnosis model from Med III to Llama, demonstrating that inter-subtask interaction effects cannot be ignored
3. **Independent networks outperform shared networks**: shared networks cause arm estimates to converge, resulting in premature commitment to the lowest-cost arm
4. **Online learning without historical data**: suitable for cold-start scenarios when new LLMs or custom assistants are deployed
5. **Flexible cost modeling**: $\alpha_i$ can be tuned per subtask — for instance, the weight for a summarization subtask with long inputs and high cost can be reduced

## Limitations & Future Work

1. The medical dataset contains only 100 reports and is primarily limited to cardiac, renal, hepatic, and neurological conditions
2. The task decomposition (i.e., which subtasks to use) is assumed to be given in advance; automatic decomposition is not discussed
3. No theoretical regret bound is provided (evaluation is empirical only)
4. The cost model relies on token count prediction and is not directly applicable to multimodal LLMs or streaming outputs
5. LLM inference latency as a cost component is not considered in the main experiments (only discussed in the appendix)

## Related Work & Insights

- **Budget-aware bandits**: primal-dual schemes by Castiglioni et al., but without support for sequential decision-making
- **LLM cascading**: FrugalGPT, AutoMix — attempt models from cheap to expensive in a predefined sequence
- **LLM routing**: Tryage (context-aware), Zooter (reward model scoring), RoutingExperts (dynamic experts)
- **Neural Bandits**: NeuralUCB, NeuralLinUCB — this work extends these to a sequential setting

## Rating ⭐⭐⭐⭐

- **Novelty**: ⭐⭐⭐⭐ — The problem formulation of pipeline LLM selection is novel; the extension from standard bandits to sequential bandits is natural and effective
- **Experimental Thoroughness**: ⭐⭐⭐ — The medical dataset is very small (100 cases); the telecommunications dataset is relatively standard; large-scale validation is lacking
- **Writing Quality**: ⭐⭐⭐⭐ — Problem definition is clear and algorithm description is complete
- **Value**: ⭐⭐⭐⭐ — Clear application scenarios in the era of LLM proliferation

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] FDP: A Frequency-Decomposition Preprocessing Pipeline for Unsupervised Anomaly Detection in Brain MRI](fdp_a_frequency-decomposition_preprocessing_pipeline_for_unsupervised_anomaly_de.md)
- [\[AAAI 2026\] SPA: Achieving Consensus in LLM Alignment via Self-Priority Optimization](spa_achieving_consensus_in_llm_alignment_via_self-priority_optimization.md)
- [\[ICLR 2026\] Benchmarking ECG FMs: A Reality Check Across Clinical Tasks](../../ICLR2026/medical_imaging/benchmarking_ecg_fms_a_reality_check_across_clinical_tasks.md)
- [\[AAAI 2026\] Intervention Efficiency and Perturbation Validation Framework: Capacity-Aware and Robust Clinical Model Selection under the Rashomon Effect](intervention_efficiency_and_perturbation_validation_framework_capacity-aware_and.md)
- [\[ICLR 2026\] LaVCa: LLM-assisted Visual Cortex Captioning](../../ICLR2026/medical_imaging/lavca_llm-assisted_visual_cortex_captioning.md)

</div>

<!-- RELATED:END -->
