---
title: >-
  [Paper Note] ATPO: Adaptive Tree Policy Optimization for Multi-Turn Medical Dialogue
description: >-
  [ICLR 2026][Medical NLP][multi-turn medical dialogue] This paper proposes ATPO (Adaptive Tree Policy Optimization), which models multi-turn medical dialogue as a hierarchical Markov decision process (H-MDP). ATPO dynamic…
tags:
  - "ICLR 2026"
  - "Medical NLP"
  - "multi-turn medical dialogue"
  - "tree search"
  - "policy optimization"
  - "uncertainty-guided exploration"
  - "hierarchical MDP"
  - "value function estimation"
  - "LLM alignment"
date: 2026-05-08
content_hash: 7f59d605e704c21c
---

# ATPO: Adaptive Tree Policy Optimization for Multi-Turn Medical Dialogue

**Conference**: ICLR 2026
**arXiv**: [2603.02216](https://arxiv.org/abs/2603.02216)  
**Code**: [https://github.com/Quark-Medical/ATPO](https://github.com/Quark-Medical/ATPO)  
**Area**: Medical Imaging
**Keywords**: multi-turn medical dialogue, tree search, policy optimization, uncertainty-guided exploration, hierarchical MDP, value function estimation, LLM alignment

## TL;DR
This paper proposes ATPO (Adaptive Tree Policy Optimization), which models multi-turn medical dialogue as a hierarchical Markov decision process (H-MDP). ATPO dynamically allocates rollout budgets via an uncertainty-aware adaptive tree expansion mechanism, using a composite uncertainty measure combining Bellman error and action-value variance to guide exploration. With Qwen3-8B, ATPO surpasses GPT-4o on three medical dialogue benchmarks.

## Background & Motivation

**Background**: Medical LLMs have achieved state-of-the-art performance on single-turn tasks such as medical examinations and disease diagnosis. However, in real-world clinical dialogues, users typically provide incomplete initial information, requiring models to proactively ask follow-up questions to gather critical details.

**Limitations of Prior Work**:
   - Prompt engineering approaches (e.g., MEDIQ) that encourage proactive questioning actually degrade diagnostic accuracy.
   - SFT methods merely imitate surface patterns in training data, resulting in poor generalization.
   - Trajectory-level preference optimization relies on expensive preference data and is sensitive to distribution shift.
   - GRPO struggles with effective credit assignment in long-horizon tasks.
   - PPO's value function estimation becomes unstable in multi-turn dialogue settings.

**Key Challenge**: Multi-turn medical dialogue is inherently a long-horizon sequential decision-making problem. Existing RL methods either fail at credit assignment (GRPO assigns a shared advantage to the entire trajectory) or produce inaccurate value estimates (PPO's single-step critic accumulates errors over long conversations).

**Goal**: To achieve efficient and accurate policy optimization in multi-turn medical dialogue—precisely estimating the value of each dialogue turn while effectively exploring the dialogue space.

**Key Insight**: The problem is formulated as an H-MDP, where tree search is performed at the dialogue-turn level and an uncertainty measure is used to adaptively allocate computational budget.

**Core Idea**: High-uncertainty dialogue states are identified via a composite measure of Bellman error and Q-value variance, enabling selective node expansion that simultaneously improves sampling diversity and critic accuracy.

## Method

### Overall Architecture
ATPO treats the multi-turn dialogue process as the expansion of a search tree, with the initial user query as the root node and each node representing a dialogue state. At each non-terminal node, the assistant model generates $N$ candidate macro-actions (follow-up questions or final answers), and a composite uncertainty score determines whether to fully expand (retaining all $N$ branches) or prune (retaining only one branch). The collected trajectories are then used to update both the policy and the critic model.

### Key Designs

1. **Hierarchical MDP Formulation**:

    - **Function**: Decomposes multi-turn dialogue into a high-level MDP and a low-level MDP.
    - **Mechanism**: In the high-level MDP, a macro-action $y_k$ is defined as the assistant's complete token sequence at turn $k$; in the low-level MDP, a micro-action $y_{k,t}$ corresponds to a single token. The state $x_k$ comprises the interaction history up to turn $k$ and the user query $q_k$.
    - **Why It Works**: Turn-level credit assignment is more appropriate for multi-turn dialogue than token-level assignment. All tokens within a single turn share the same macro-action advantage, avoiding the sparse reward problem at the token level.

2. **Composite Uncertainty Measure**:

    - **Function**: Computes an uncertainty score for each frontier node to determine whether to expand it.
    - **Mechanism**: For state $x_k$, $N$ candidate macro-actions $\{y_k^i\}_{i=1}^N$ are sampled. A one-step lookahead is used to compute action values $\hat{Q}(x_k, y_k^i) = r(x_k, y_k^i) + \gamma V_\psi(x_{k+1}^i)$, yielding:
        - **Bellman error** $U_1$: the absolute difference between the critic's current value estimate and the empirical one-step lookahead value, reflecting inaccuracy in value function estimation.
        - **Q-value variance** $U_2$: the variance of action-value estimates across candidate actions, reflecting policy uncertainty and environmental stochasticity.
        - **Composite score** $U = \alpha U_1 + (1-\alpha) U_2$, with $\alpha=0.3$ balancing the two signals.
    - **Why It Works**: $U_1$ identifies states where the critic is inaccurate (requiring more samples to improve value estimation), while $U_2$ identifies states where the policy is indecisive (requiring more exploration). The two are complementary—using $U_1$ alone leads to aggressive early exploration concentrated at shallow levels, whereas adding $U_2$ achieves deeper and more uniform coverage.

3. **Threshold-Driven Pruning Strategy**:

    - **Function**: Determines node expansion mode based on an uncertainty threshold $\tau$.
    - **Mechanism**: When $U(x_k) > \tau$, all $N$ branches are retained; when $U(x_k) \leq \tau$, one branch is selected at random (with a 10% probability of bypassing pruning to maintain baseline diversity). Expansion continues until all dialogues terminate or the number of leaf nodes reaches the budget limit.
    - **Why It Works**: Adaptive allocation avoids the problem in TreePO's fixed binary expansion, where exponential node growth concentrates in early turns.

4. **Value Backpropagation and Tree Decomposition**:

    - **Function**: Recursively computes target values and advantages for all nodes from the leaf nodes upward.
    - **Mechanism**: The target value $\hat{V}$ of a leaf node equals its immediate reward; for non-leaf nodes, it is the average of one-step TD targets across all children. Advantages are computed using the standard one-step TD formula $\hat{A} = r + \gamma V_\psi(x_{k+1}) - V_\psi(x_k)$, using critic estimates rather than target values (since a pruned node with only one branch would yield zero advantage if target values were used instead).
    - **Why It Works**: Tree-structured value backpropagation provides lower-variance value estimates than pure Monte Carlo (GRPO) while being more accurate than a single critic (PPO).

5. **PPO-Style Policy Update with Visit-Count Normalization**:

    - **Function**: Decomposes the tree into independent trajectories for policy optimization.
    - **Mechanism**: Each root-to-leaf path constitutes one trajectory, yielding $M$ trajectories from $M$ leaf nodes. The policy update incorporates a visit count $C(x_k)$ normalization term to prevent over-optimization of frequently visited shared nodes. Macro-action advantages are uniformly distributed across all tokens within the corresponding turn.
    - **Why It Works**: Ablation experiments confirm that omitting visit-count normalization leads to uncontrolled entropy growth and policy collapse.

6. **Asynchronous Execution and KV Cache Optimization**:

    - **Function**: Reduces the computational overhead of tree search.
    - **Mechanism**: Assistant model generation, user model interaction, and critic value estimation are executed fully asynchronously. Shared prefixes reuse KV caches; the tree structure is naturally amenable to prefix sharing, as all children of the same parent node share an identical dialogue history prefix.
    - **Why It Works**: TreePO achieves a decoding speed of 2,500 tokens/sec/GPU on 1.7B models. Although ATPO's rollout phase accounts for a larger proportion of total time (45% vs. 25%), it produces higher-quality training data, resulting in the shortest overall training time.

## Experimental Setup

### Environment
- **User Simulator**: Implemented with Qwen3-8B, strictly answering questions based on atomic facts; GPT-4o validates instruction-following accuracy at 100% with a hallucination rate of only 1.2%.
- **Assistant Agent**: Selects the correct answer from given options and may iteratively query the user simulator for additional information.
- **Reward Function**: Based solely on final answer correctness—correct: +3, incorrect: 0, invalid format: −1.

### Datasets
- **MedicalExam**: 150 samples from 5 sources (MedQA / MedMCQA / MMLU / SelfExam / QMAX).
- **MedQA**: 1,268 samples from the MEDIQ test set.
- **MedMCQA**: 536 samples constructed from the MedMCQA validation set.
- Training data: 14,256 samples (66% MEDIQ + 34% MedMCQA).

### Baselines
- **Zero-shot**: Direct single-turn / MEDIQ multi-turn prompting.
- **SFT**: Standard SFT / Dynamic Fine-Tuning (DFT) with 1,269 dialogues generated via Gemini-2.5-Pro self-play.
- **SFT+RL**: PPO (MDP) / PPO (H-MDP) / GRPO / TreePO.

### Key Hyperparameters
- Policy learning rate $1 \times 10^{-6}$; critic learning rate $1 \times 10^{-5}$.
- KL penalty $\beta=0.01$; discount factor $\gamma=1$.
- GRPO group size 32; ATPO expansion size $N=4$, total expansion budget 128.
- ATPO ($U_1$): $\tau=0.5$; ATPO ($U_1+U_2$): $\alpha=0.3$, $\tau=1.5$.

## Key Experimental Results

### Main Results (Table 1)

| Model | Method | MedicalExam | MedQA | MedMCQA |
|-------|--------|-------------|-------|---------|
| Qwen3-8B | GRPO | 60.93 | 57.92 | 51.12 |
| Qwen3-8B | TreePO | 65.33 | 61.81 | 54.74 |
| Qwen3-8B | ATPO ($U_1+U_2$) | **65.87** | **64.07** | 53.66 |
| GPT-4o | MEDIQ | 64.00 | 63.15 | 53.03 |

- At the 8B scale, ATPO ($U_1+U_2$) surpasses GPT-4o on MedQA by **+0.92%**.
- Compared to TreePO, ATPO achieves absolute improvements on MedQA of +0.82% (1.7B), +1.73% (4B), and +2.26% (8B).
- The MEDIQ prompting strategy underperforms Direct single-turn prompting, consistent with findings in the original paper.
- SFT (including distillation from GPT-4o/Gemini) provides only limited accuracy gains; RL training is indispensable.

### Sampling Efficiency
- For Qwen3-4B on MedQA, ATPO ($U_1+U_2$) reaches ~52.7% accuracy using only approximately 55% of TreePO's training iterations.
- ATPO achieves PPO's best performance in the shortest time (2.22 hours vs. PPO 3.02 hours vs. GRPO 4.86 hours).

### Ablation Study
- **Uncertainty measure**: $U_1+U_2$ produces high-variance sample returns (comparable to GRPO) with significantly lower critic value loss than PPO. Using $U_1$ alone concentrates exploration at shallow levels (turns 3–4); adding $U_2$ enables deeper and more uniform coverage.
- **Visit-count normalization**: Omitting normalization leads to uncontrolled entropy growth and policy collapse; applying normalization to value loss as well causes rapid entropy collapse, degrading the model to a suboptimal single-turn policy.
- **User simulator generalization**: Replacing the test-time simulator from Qwen3-8B with Llama-3.3-70B-Instruct yields virtually no performance change, confirming that the model does not overfit to a specific simulator.

## Highlights & Insights

### Highlights
1. Uncertainty-guided adaptive tree search balances sampling diversity ($U_2$) and critic optimization ($U_1$), offering greater flexibility than the fixed-structure TreePO.
2. Hierarchical MDP formulation with turn-level credit assignment is well suited to the macro-decision nature of multi-turn dialogue.
3. KV cache reuse and asynchronous execution keep the additional computational overhead of tree search manageable, yielding the shortest total training time.
4. Surpassing GPT-4o with an 8B model validates the effectiveness of the proposed approach.

### Limitations & Future Work
1. The expansion threshold $\tau$ and $\alpha$ are fixed hyperparameters set manually; different tasks or models may require retuning.
2. Macro-action advantages are uniformly distributed across all tokens within a turn, without distinguishing critical tokens from redundant ones.
3. The user simulator is based on predefined atomic facts, which differs from the free-form expressions of real patients.
4. Evaluation is limited to MCQ-format medical datasets; open-ended diagnostic scenarios are not considered.

## Personal Reflections
1. **Generalizability of the uncertainty measure**: The composite measure of Bellman error and Q-value variance can potentially be extended to other multi-turn interactive scenarios requiring long-horizon decision-making (e.g., tool use, multi-turn code generation). Its core contribution is providing a quantitative criterion for identifying where additional computational resources are most worth investing.
2. **Relationship to MCTS**: ATPO's tree search shares similarities with AlphaGo's MCTS (both perform selective expansion), but ATPO's uncertainty measure is grounded in value functions rather than UCB, making it more suitable for continuous action spaces. Future work could explore incorporating UCT or PUCT criteria to further optimize node selection.
3. **Validation of SFT's limitations**: Even distillation from GPT-4o/Gemini fails to yield substantial performance gains, reaffirming that imitation $\subsetneq$ learning—SFT acquires format but not decision-making strategy. This has important implications for the training paradigms of medical AI systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] IndicMedDialog: A Parallel Multi-Turn Medical Dialogue Dataset for Accessible Healthcare in Indic Languages](../../ACL2026/medical_nlp/indicmeddialog_a_parallel_multi-turn_medical_dialogue_dataset_for_accessible_hea.md)
- [\[AAAI 2026\] PulseMind: A Multi-Modal Medical Model for Real-World Clinical Diagnosis](../../AAAI2026/medical_nlp/pulsemind_a_multi-modal_medical_model_for_real-world_clinical_diagnosis.md)
- [\[ACL 2026\] Query Pipeline Optimization for Cancer Patient Question Answering Systems](../../ACL2026/medical_nlp/query_pipeline_optimization_for_cancer_patient_question_answering_systems.md)
- [\[ACL 2026\] SEMA-RAG: A Self-Evolving Multi-Agent Retrieval-Augmented Generation Framework for Medical Reasoning](../../ACL2026/medical_nlp/sema-rag_a_self-evolving_multi-agent_retrieval-augmented_generation_framework_fo.md)
- [\[ICLR 2026\] From Conversation to Query Execution: Benchmarking User and Tool Interactions for EHR Database Agents](from_conversation_to_query_execution_benchmarking_user_and_tool_interactions_for.md)

</div>

<!-- RELATED:END -->
