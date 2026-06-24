---
title: >-
  [Paper Note] MoRAgent: Parameter Efficient Agent Tuning with Mixture-of-Roles
description: >-
  [ICML2025][Model Compression][Agent Tuning] This paper proposes the Mixture-of-Roles (MoR) framework, which decomposes agent capabilities into three roles: Reasoner, Executor, and Summarizer, with each role allocated a specialized group of LoRAs. It achieves agent performance close to or even exceeding full-parameter fine-tuning while only introducing minimal extra parameters (0.16B–0.36B).
tags:
  - "ICML2025"
  - "Model Compression"
  - "Agent Tuning"
  - "parameter-efficient fine-tuning"
  - "LoRA"
  - "Mixture-of-Experts"
  - "Function Calling"
date: 2026-05-08
content_hash: d629d61ac4e26784
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# MoRAgent: Parameter Efficient Agent Tuning with Mixture-of-Roles

**Conference**: ICML2025  
**arXiv**: [2512.21708](https://arxiv.org/abs/2512.21708)  
**Code**: [https://mor-agent.github.io/](https://mor-agent.github.io/)  
**Area**: Model Compression  
**Keywords**: Agent Tuning, parameter-efficient fine-tuning, LoRA, Mixture-of-Experts, Function Calling

## TL;DR

This paper proposes the Mixture-of-Roles (MoR) framework, which decomposes agent capabilities into three roles: Reasoner, Executor, and Summarizer, with each role allocated a specialized group of LoRAs. It achieves agent performance close to or even exceeding full-parameter fine-tuning while only introducing minimal extra parameters (0.16B–0.36B).

## Background & Motivation

- **Challenges in Agent Fine-Tuning**: Existing agent fine-tuning approaches almost exclusively employ full parameter fine-tuning, which incurs immense computational costs and degrades the base model's general capabilities, thereby limiting flexible switching between general tasks and agent tasks.
- **Poor Performance of Direct PEFT for Agents**: Applying parameter-efficient fine-tuning (PEFT) methods like LoRA directly to agent tasks yields performance far inferior to full parameter fine-tuning. This is because agent tasks require LLMs to simultaneously possess diverse capabilities such as reasoning, tool invocation execution, and dialogue summarization, whereas low-rank matrices struggle to learn these heterogeneous capabilities concurrently.
- **Low-Rank Bottleneck in Multi-Capability Learning**: The core insight is that agent tasks inherently require the collaboration of multiple distinct capabilities, and the parameter space of a single LoRA is insufficient to cover the distributional differences of these capabilities.
- **Limitations of Multi-Model Solutions**: Previous works like α-UMi utilize multiple independent LLMs to undertake different roles, but the resource overhead for training and inference is too high to be practical.

## Method

### 1. Capabilities Decomposition

Inspired by the ReAct (Reason+Action) paradigm, the agent's capabilities are decomposed into three roles:

- **Reasoner**: Understands user queries, analyzes execution trajectories, generates thoughts, and decides which role to activate next. It is formalized as: $T_t, Role_t = \boldsymbol{W}_r(p_r, q, \tau_{t-1})$
- **Executor**: Selects and invokes appropriate functions with parameters based on the analysis of the Reasoner. It is formalized as: $Fun_t, Param_t = \boldsymbol{W}_e(p_e, q, \tau_{t-1}, T_{t-1})$
- **Summarizer**: Organizes dialogue history and provides feedback to the user when the Reasoner determines that the task is completed or cannot proceed further. It is formalized as: $Sum = \boldsymbol{W}_s(p_s, q, \tau)$

The three roles are alternately activated via a rule-based, role-aware gate, forming a workflow of "Reasoning -> Execution -> (Observation -> Reasoning -> ...) -> Summarization".

### 2. Mixture-of-Roles (MoR) Architecture

The MoR framework is deployed on the linear layers (Attention/FFN) of each Transformer layer, keeping the pre-trained weights frozen while only training the LoRAs and routers.

**Forward Computation**: Given the input $\boldsymbol{u} \in \mathbb{R}^{len \times d_1}$, the final output is:

$$\boldsymbol{h} = \boldsymbol{W}_0 \boldsymbol{u} + \Delta\boldsymbol{h}_r + \Delta\boldsymbol{h}_e + \Delta\boldsymbol{h}_s$$

Key constraint: Each token is processed by only one role, i.e., $\mathbb{1}\{\Delta\boldsymbol{h}_r^i \neq 0\} + \mathbb{1}\{\Delta\boldsymbol{h}_e^i \neq 0\} + \mathbb{1}\{\Delta\boldsymbol{h}_s^i \neq 0\} = 1$.

**Intra-role LoRA Structure**: Each role contains a shared LoRA and multiple routed LoRAs. Taking the Reasoner as an example:

$$\Delta\boldsymbol{W}_r = \boldsymbol{B}_r^0 \boldsymbol{A}_r^0 + \sum_{i=1}^{E_r} \boldsymbol{B}_r^i \boldsymbol{A}_r^i \boldsymbol{R}_r(\boldsymbol{u}_r)$$

where $\boldsymbol{R}_r$ is a Top-K token-aware router that dynamically selects specific LoRAs based on the input.

**LoRA Allocation Strategy**: Determined through experiments, the Reasoner and Executor each use 5 LoRAs (Top-4 activated), while the Summarizer uses 4 LoRAs (Top-3 activated), since the summarization task has a lower learning difficulty.

### 3. Training Objectives

The total loss consists of three components:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \alpha_1 \mathcal{L}_{\text{aux}} + \alpha_2 \mathcal{L}_{\text{orth}}$$

- $\mathcal{L}_{\text{CE}}$: Standard cross-entropy loss
- $\mathcal{L}_{\text{aux}}$: Auxiliary load balancing loss (inherited from Switch Transformer) to prevent load imbalance among LoRAs: $\mathcal{L}_{\text{aux}} = E_\triangledown \cdot \sum_{i=1}^{E_\triangledown} f_\triangledown^i \cdot P_\triangledown^i$
- $\mathcal{L}_{\text{orth}}$: Orthogonal loss that encourages different LoRAs within the same role to learn feature distributions in different directions: $L_{\text{orth}} = \sum_{i}\sum_{j>i} (\|\boldsymbol{A}^{iT}\boldsymbol{A}^j\|_F^2 + \|\boldsymbol{B}^{iT}\boldsymbol{B}^j\|_F^2)$

### 4. Multi-Role Data Generation Pipeline

- **Data Sources**: Public datasets including ToolBench, APIGen+ToolACE, glaive-function-calling-v2, and MathGenie.
- **Role Content Completion**: Using GPT-4o to complete missing thoughts and summaries in the data.
- **Reliability Verification**: Using DeepSeek-V3 to assess trajectory quality and filter out low-quality samples; performing rule-based detection and manual correction for execution errors (e.g., function not in the candidate list, mismatched parameter count/type, or incorrect function selection).
- **Format Unification**: Unifying all data into JSON format, which includes candidate function lists, system prompts, and execution trajectories.

## Key Experimental Results

### StableToolBench Results (DFS Pass Rate)

| Model | Extra Params | I1-Inst | I1-Tool | I1-Cat | I2-Inst | I2-Cat | I3-Inst | AVG |
|------|---------|---------|---------|--------|---------|--------|---------|-----|
| GPT-4 | — | 59.2 | 65.7 | 61.7 | 55.2 | 55.6 | 66.1 | 60.6 |
| ToolLLaMA-v2-7B | — | 61.0 | 45.6 | 58.8 | 53.5 | 60.3 | 48.1 | 54.6 |
| Llama3.2-1B Base | — | 11.7 | 11.3 | 14.6 | 8.5 | 2.3 | 11.5 | 10.0 |
| **MoRAgent-Llama** | **+0.16B** | 54.6 | 45.5 | 53.2 | 46.8 | 68.2 | 58.8 | **54.5 (+44.5)** |
| Phi3.5-mini Base | — | 46.4 | 50.6 | 51.1 | 39.9 | 41.5 | 37.6 | 44.5 |
| **MoRAgent-Phi** | **+0.36B** | 55.9 | 56.6 | 60.9 | 55.4 | 59.7 | 63.6 | **58.7 (+14.2)** |

**Key Findings**:
- MoRAgent-Llama adds only 0.16B parameters but skyrockets the DFS Pass Rate from 10.0% to 54.5% (+44.5%), approaching the level of ToolLLaMA-v2-7B (54.6%), which has about 5.6 times more parameters.
- MoRAgent-Phi improves the DFS Pass Rate on Phi3.5-mini by 14.2 percentage points to 58.7%, surpassing ToolLLaMA-v2-7B.
- Significant improvements are also observed in CoT mode: Llama +40.0%, Phi +26.3%.

### Key Design Choices

- **Number of LoRAs**: Experiments show that the optimal performance-parameter balance is achieved when the Reasoner and Executor each use 5 LoRAs (Top-4) and the Summarizer uses 4 LoRAs (Top-3).
- **LoRA rank**: $d_3 = 16$
- **Target Modules**: query, key, value, out, gate, up, down (7 modules in total)
- **Training Hyperparameters**: Learning rate of 5e-5, 4 epochs.

## Highlights & Insights

1. **Design Intuition of Role Decomposition**: Unlike previous heavyweight multi-model collaboration approaches (such as α-UMi using 3 independent LLMs), MoR achieves role division within a single model via multiple groups of LoRAs, resulting in extremely low training and inference costs.
2. **Differentiated LoRA Allocation Across Roles**: It is empirically discovered that different roles require different numbers of LoRAs; the Summarizer requires fewer LoRAs than the Reasoner/Executor, which aligns with the intuition that summarization is simpler than reasoning and execution.
3. **Two-Tier Routing Mechanism**: Consists of a rule-level routing (role-aware gate, allocated by role type) and a learning-level routing (token-aware router, dynamically selecting LoRAs within a role), balancing interpretability and flexibility.
4. **Introduction of Orthogonal Loss**: Explicitly encourages different LoRAs within the same role to learn orthogonal trait directions, reducing redundancy and enhancing parameter efficiency.
5. **Comprehensive Data Quality Control Pipeline**: Uses a pipeline with two different LLMs (GPT-4o for completion and DeepSeek-V3 for filtering) plus manual corrections to ensure the reliability of the training data.

## Limitations & Future Work

1. **Rigid Role Definitions**: The three-role decomposition (Reasoner/Executor/Summarizer) is pre-defined, whereas different agent tasks may require different role divisions. The paper does not explore a more flexible or adaptive role discovery mechanism.
2. **Limited Evaluation Benchmarks**: Evaluation is mainly performed on StableToolBench and BFCL, lacking verification in more complex multi-turn interactive scenarios (e.g., WebArena, OSWorld).
3. **Data Construction Dependency on Strong LLMs**: Multi-role data completion relies on GPT-4o, and reliability verification relies on DeepSeek-V3. Data quality is bonded to the upper limit of these models' capabilities, and the compilation cost is not negligible.
4. **Only Tested on Small Models**: The experiments are only validated on 1B–3.8B scale models, leaving performance on 7B+ models untested, making it uncertain whether similar gains persist on larger models.
5. **Unreported Inference Latency**: The actual increase in inference latency introduced by multiple LoRA groups and routers is not quantitatively analyzed.
6. **Error Propagation in Role Switching**: The accuracy of the Reasoner in predicting the next role directly impacts the entire pipeline. Misjudgments (e.g., premature summarization or missing execution steps) lead to task failure, but the paper does not deeply analyze these error modes.

## Related Work & Insights

- **Agent Tuning Series**: Full parameter fine-tuning methods such as ToolLLM, AgentTuning, and Gorilla serve as the baselines for comparison.
- **Multi-LoRA Composition**: Multi-LoRA MoE methods like LoRAHub, MoA, and MoLE provide the technical foundation but are not designed specifically for agent tasks.
- **Multi-Model Agent Collaboration**: The three-role decomposition (planner/caller/summarizer) of α-UMi is the direct inspiration. MoR replaces multiple models with multiple LoRAs within a single model, drastically reducing costs.
- **Insights**: The concept of role decomposition combined with specialized parameters can be extended to other multi-capability tasks (e.g., RAG, multimodal agents), modeling different capabilities within the same model using different low-rank spaces.

## Rating

- **Novelty**: 7/10 — The role decomposition idea is inspired by α-UMi. The key novelty lies in mapping multiple roles onto a multi-group LoRA MoE architecture, with incremental contributions from the orthogonal loss and two-tier routing.
- **Experimental Thoroughness**: 7/10 — The multi-model and multi-benchmark testing, alongside the ablation study, is relatively comprehensive, but lacks validation on larger models and complex agent scenarios.
- **Writing Quality**: 8/10 — The structure is clear, mathematical derivations are complete, and illustrations are intuitive.
- **Value**: 7/10 — Provides a practical solution for agent PEFT with significant efficacy on small models, though the rigidity of the role definitions limits generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Expert Pyramid Tuning: Efficient Parameter Fine-Tuning for Expertise-Driven Task Allocation](../../CVPR2025/model_compression/expert_pyramid_tuning_efficient_parameter_fine-tuning_for_expertise-driven_task_.md)
- [\[ICML 2025\] Parameter-Efficient Fine-Tuning of State Space Models](parameter-efficient_fine-tuning_of_state_space_models.md)
- [\[ACL 2025\] C3A: Parameter-Efficient Fine-Tuning via Circular Convolution](../../ACL2025/model_compression/parameter-efficient_fine-tuning_via_circular_convolution.md)
- [\[ACL 2025\] State-offset Tuning: State-based Parameter-Efficient Fine-Tuning for State Space Models](../../ACL2025/model_compression/state_offset_tuning_ssm_peft.md)
- [\[CVPR 2025\] Faster Parameter-Efficient Tuning with Token Redundancy Reduction (FPET)](../../CVPR2025/model_compression/faster_parameter-efficient_tuning_with_token_redundancy_reduction.md)

</div>

<!-- RELATED:END -->
