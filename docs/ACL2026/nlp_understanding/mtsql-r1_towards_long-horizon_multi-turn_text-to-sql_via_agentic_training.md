---
title: >-
  [Paper Note] MTSQL-R1: Towards Long-Horizon Multi-Turn Text-to-SQL via Agentic Training
description: >-
  [ACL2026][NLP Understanding][Multi-turn Text-to-SQL] MTSQL-R1 transforms multi-turn Text-to-SQL from "one-shot translation" into a long-horizon agentic training problem involving interaction with databases and dialogue m…
tags:
  - "ACL2026"
  - "NLP Understanding"
  - "Multi-turn Text-to-SQL"
  - "Long-horizon reasoning"
  - "Database execution feedback"
  - "Dialogue memory"
  - "Reinforcement learning"
date: 2026-05-08
content_hash: 8dbceabb3db9c606
---

# MTSQL-R1: Towards Long-Horizon Multi-Turn Text-to-SQL via Agentic Training

**Conference**: ACL2026  
**arXiv**: [2510.12831](https://arxiv.org/abs/2510.12831)  
**Code**: https://github.com/taichengguo/MTSQL-R1  
**Area**: NLP Understanding / Multi-turn Text-to-SQL  
**Keywords**: Multi-turn Text-to-SQL, Long-horizon reasoning, Database execution feedback, Dialogue memory, Reinforcement learning

## TL;DR
MTSQL-R1 transforms multi-turn Text-to-SQL from "one-shot translation" into a long-horizon agentic training problem involving interaction with databases and dialogue memory. Through self-taught warm-start SFT and multi-level GRPO rewards, small-scale Qwen3 models outperform strong closed-source prompting baselines and short-horizon SFT/RL baselines on CoSQL and SParC.

## Background & Motivation
**Background**: Multi-turn Text-to-SQL requires mapping the current user question, historical questions, historical SQL, and database schema into executable SQL within a continuous dialogue. Early methods relied on specialized context encoding, relational graphs, or dynamic schema linking. In the LLM era, methods like ACT-SQL and CoE-SQL typically use prompting, CoT, or history-based SQL editing to handle multi-turn contexts.

**Limitations of Prior Work**: Most existing methods still treat the task as short-horizon text-to-SQL translation: the model generates a single SQL query and terminates without actual execution or explicit consistency checks against history. Consequently, two types of errors recurringly appear: first, the SQL is unexecutable, returns null results, or contains logical errors; second, the current SQL appears reasonable but loses entities, filters, or join paths defined in previous turns.

**Key Challenge**: The difficulty of multi-turn Text-to-SQL lies not just in "writing syntactically correct SQL," but in continuous verification and correction across changing intents. Short-horizon models lack environmental feedback and thus do not know where they failed. Pure prompted agents can access tools but lack long-horizon behavioral training, leading to unstable tool calls, feedback interpretation, and self-correction.

**Goal**: The authors aim to address three sub-problems: first, modeling multi-turn Text-to-SQL as a long-horizon decision process involving execution, verification, and correction; second, constructing high-quality long-horizon behavioral trajectories when the model's initial capability is insufficient; third, using RL to enable the model to learn cyclic reasoning between database feedback and dialogue memory.

**Key Insight**: Multi-turn SQL generation is inherently suitable for agentic training: databases provide execution results or error messages, and dialogue memory provide historical constraints—both are finer-grained supervision signals than static labels. Rather than making the model guess in one shot, it should experience a closed loop of propose, execute, verify, and refine.

**Core Idea**: Replace pure text context prompting with "database execution feedback + long-term dialogue memory verification," training multi-turn Text-to-SQL as a long-horizon MDP capable of self-verification and self-correction.

## Method
The MTSQL-R1 approach operates on two layers: the outer layer is the multi-turn dialogue task where each user question leads to a SQL output; the inner layer is a sequence of agent behaviors executed to produce that SQL. Instead of direct output, the model proposes a candidate SQL, executes it, judges the execution, and checks historical consistency via memory. If any step fails, it enters self-correction and repeats the loop until it passes all checks and finalizes.

The core contribution is the redesign of training data, action spaces, and rewards around this long-horizon closed loop. Warm-Start SFT first teaches the model to "act like a Text-to-SQL agent," followed by RL to optimize the final SQL and intermediate verification behaviors under tool feedback.

### Overall Architecture
Inputs include the current user question, dialogue history, database schema, and long-term memory consisting of historical questions, SQL, and parsed constraints/entities. The output is the final SQL for the current turn.

The pipeline consists of three steps:

Step one is MDP modeling. States include dialogue history, schema, current question, long-term memory, current candidate SQL, and accumulated observations. Actions include PROPOSE, EXECUTE, E-VERIFY, M-VERIFY, SELF-CORRECT, and FINALIZE. EXECUTE queries the actual database, and M-VERIFY accesses dialogue memory.

Step two is Self-taught Warm-Start SFT. The current policy model generates multiple long-horizon trajectories for training samples. Only trajectories where the final SQL satisfies both EX and EM are kept. These are processed via difficulty-aware rejection sampling to select representative trajectories for behavior cloning. This iterates over multiple rounds to cover more samples.

Step three is End-to-end Long-horizon RL. The SFT-ed model interacts with the MDP and is trained using a weighted sum of outcome and process rewards. The optimization uses GRPO, with loss masking applied to tool outputs and system instructions so that gradients primarily affect generated actions, verification judgments, and SQL.

### Key Designs
1. **Long-horizon MDP and Dual Environment Feedback**:
    - **Function**: Decomposes multi-turn SQL generation into a sequence of executable, verifiable, and correctable actions.
    - **Mechanism**: The environment comprises the database and long-term memory. The database returns execution results or errors; long-term memory stores history to check if the current SQL misses or violates prior conditions. Actions follow a fixed order: Initial PROPOSE, then EXECUTE and E-VERIFY; if execution passes, then M-VERIFY; if any verification fails, SELF-CORRECT; finally, FINALIZE.
    - **Design Motivation**: Databases expose syntax/execution errors, while memory exposes multi-turn consistency errors. Combining them helps the model identify whether a failure stems from execution or context.

2. **Self-taught Warm-Start SFT with Difficulty-aware Trajectory Filtering**:
    - **Function**: Teaches the model to stably follow agent formats and tool protocols before RL.
    - **Mechanism**: 20 trajectories are sampled per training sample, keeping those passing both EX and EM. Simple samples retain few short trajectories to avoid unnecessary length; hard samples prioritize longer interaction trajectories chosen via Qwen3-Embedding clustering for diversity.
    - **Design Motivation**: Synthetic gold trajectories are too "perfect," lacking realistic failure/correction. Self-taught iteration allows the model to learn from its own successful long-horizon reasoning experiences.

3. **Multi-level Outcome + Process Rewards**:
    - **Function**: Mitigates sparse rewards from final SQL and provides signals for intermediate actions.
    - **Mechanism**: Outcome rewards are based on EX and EM. Process rewards vary by action: PROPOSE and SELF-CORRECT are measured by average F1 of SQL clauses (SELECT, WHERE, JOIN, etc.); E-VERIFY depends on the match between execution results and the model's pass/fail judgment; M-VERIFY depends on clause F1 and memory consistency conclusions.
    - **Design Motivation**: Complex problems require trial-and-error. Process rewards encourage "proposing closer SQL" and "identifying execution errors correctly."

### Loss & Training
The SFT phase uses standard auto-regressive cross-entropy with token masking for system instructions and tool outputs, focusing supervision on action and SQL tokens. 

The RL phase uses GRPO. Advantage is calculated within groups of sampled trajectories, with policy updates using PPO-style ratio clipping and KL regularization. An easy-to-hard curriculum is applied: samples are binned by initial success rate, and training proceeds from easier to harder bins. Backbones include Qwen3-1.7B and Qwen3-4B. Hyperparameters include SFT learning rate of 5e-6, GRPO learning rate of 1e-6, and max response length of 8000.

## Key Experimental Results

### Main Results
Evaluated on CoSQL and SParC using Execution Accuracy (EX) and Exact Match (EM).

| Method | Scale | CoSQL EX/EM | SParC EX/EM | Avg EX/EM | Note |
|------|----------|-------------|-------------|------------|------|
| CoE-SQL | Closed | 69.6 / 52.4 | 70.3 / 56.0 | 64.1 / 51.6 | Strong prompt/edit baseline |
| RASAT+PICARD | 3B | 67.0 / 58.8 | 73.3 / 67.7 | 64.5 / 57.7 | Structured pre-LLM baseline |
| Qwen3-1.7B Short-Horizon SFT | 1.7B | 68.1 / 59.3 | 74.3 / 69.2 | 69.6 / 62.2 | Standard SFT |
| Qwen3-1.7B Short-Horizon Direct RL | 1.7B | 72.8 / 59.0 | 72.1 / 65.5 | 70.5 / 60.7 | Direct SQL-level RL |
| Qwen3-1.7B MTSQL-R1 | 1.7B | 77.3 / 63.5 | 76.2 / 66.1 | 74.6 / 64.4 | Warm-Start SFT + RL |
| Qwen3-4B Short-Horizon SFT | 4B | 73.1 / 64.8 | 78.3 / 71.5 | 74.1 / 66.6 | Strong short-horizon SFT |
| Qwen3-4B Short-Horizon Direct RL | 4B | 75.2 / 64.8 | 75.8 / 66.5 | 74.0 / 64.2 | Strong short-horizon RL |
| Qwen3-4B MTSQL-R1 | 4B | 79.9 / 65.2 | 79.0 / 68.7 | 77.6 / 66.5 | **+3.5 Avg EX** vs prev. best |

Short-horizon SFT can achieve decent EM but lags significantly in EX. MTSQL-R1's long-horizon verification primarily boosts the logically critical EX metric.

### Ablation Study

| Configuration | CoSQL EX | CoSQL EM | Note |
|------|----------|----------|------|
| Qwen3-4B + Warm-Start + RL | 79.9 | 65.2 | Full model |
| w/o Execution Tool | 74.6 | 64.6 | -5.3 EX; Execution is a major contributor |
| w/o Memory Verification Tool | 77.8 | 64.1 | -2.1 EX; Multi-turn consistency weakens |
| Qwen3-14B Long-horizon no training | 74.4 | 55.1 | Prompting 14B < Trained 4B |

### Key Findings
- **Warm-Start Efficiency**: Successful training samples with trajectories increased from 6,311 (Round 1) to 7,555 (Round 3) in CoSQL, totaling 19,416 trajectories.
- **Difficulty Gains**: RL provides larger gains for hard/extra-hard problems and later turns (Turn $\ge$ 4), suggesting that memory and execution feedback solve the error accumulation problem.
- **Small Model Stability**: A 1.7B base model without training fails to follow agent calls (23.3 EX), proving that "providing tools" $\neq$ "using tools."
- **Efficiency**: 4B MTSQL-R1 with 8000 tokens has ~28.3s latency. Capping at 4000 tokens yields 77.0 EX at ~15.6s.

## Highlights & Insights
- Transforming multi-turn "historical consistency" into verifiable agent actions is highly effective. Instead of just appending history, the model explicitly queries and validates against it.
- Self-taught trajectory collection is practical. It constructs high-quality behavior from the model's own rollouts without requiring manual agent annotations.
- Clause-based process rewards (SELECT, WHERE, JOIN F1) are more informative for SQL structural repair than pure execution binary signals.
- The distinction between "short-horizon RL" and "long-horizon agentic RL" is clear: the latter learns post-execution reflection and cross-turn constraint recovery.
- Evaluation on **predicted-prior** (using the model's own past predictions) shows MTSQL-R1 is more robust against the "snowball effect" of errors.

## Limitations & Future Work
- **Aggregation Drift**: Errors in aggregation (COUNT, SUM) remain difficult, especially in extra-hard SQL cases.
- **Latency**: Generating reasoning/verification tokens adds significant overhead compared to direct SQL output, limiting real-time application in some BI scenarios.
- **Memory Complexity**: The current memory uses parsed textual constraints; robustness against massive schemas or noisy memory needs further validation.
- **Reward Dependence**: The training still relies on gold SQL and execution results; future work could explore weak supervision or log-based feedback.

## Related Work & Insights
- **vs ACT-SQL/CoE-SQL**: Unlike prompting-based or editing-based methods, MTSQL-R1 enables small open-source models to learn stable tool-use and correction behaviors.
- **vs SQL-R1**: While SQL-R1 applies RL to single-turn SQL, MTSQL-R1 extends the scope to long-horizon MDPs with dialogue memory.
- **Insight for other tasks**: Any multi-turn task with a verifiable environment (e.g., Code Gen with unit tests, Data Analysis with notebooks) can adopt this framework by decomposing feedback into action-level rewards.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ Solid combination of MDP, tool execution, memory verification, and GRPO for multi-turn SQL.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive coverage across datasets, difficulty levels, and ablation of each agentic component.
- **Writing Quality**: ⭐⭐⭐⭐☆ Clear structure; good motivation.
- **Value**: ⭐⭐⭐⭐⭐ High relevance for building reliable enterprise-grade database agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MADE: A Living Benchmark for Multi-Label Text Classification with Uncertainty Quantification](made_a_living_benchmark_for_multi-label_text_classification_with_uncertainty_qua.md)
- [\[ACL 2026\] MSMO-ABSA: Multi-Scale and Multi-Objective Optimization for Cross-Lingual Aspect-Based Sentiment Analysis](msmo-absa_multi-scale_and_multi-objective_optimization_for_cross-lingual_aspect-.md)
- [\[ACL 2026\] Beyond Chunking: Discourse-Aware Hierarchical Retrieval for Long Document Question Answering](beyond_chunking_discourse-aware_hierarchical_retrieval_for_long_document_questio.md)
- [\[ACL 2026\] Unveiling Temporal Framing in News Text](uncovering_temporal_framing_in_the_news.md)
- [\[ACL 2026\] Reasoning-Based Refinement of Unsupervised Text Clusters with LLMs](reasoning-based_refinement_of_unsupervised_text_clusters_with_llms.md)

</div>

<!-- RELATED:END -->
