---
title: >-
  [Paper Note] GenesisFunc: Multi-Agent Data Generation for Accurate and Generalizable Function-Calling
description: >-
  [ACL2026][Dialogue Systems][Function Calling] GenesisFunc automatically constructs high-quality function-calling training data using a reliable tool pool, multi-agent dialogue generation…
tags:
  - "ACL2026"
  - "Dialogue Systems"
  - "Function Calling"
  - "Multi-agent data generation"
  - "Tool learning"
  - "Synthetic data quality inspection"
  - "GRPO"
date: 2026-05-08
content_hash: 41d8387d374fd183
---

# GenesisFunc: Multi-Agent Data Generation for Accurate and Generalizable Function-Calling

**Conference**: ACL2026  
**arXiv**: [2605.28835](https://arxiv.org/abs/2605.28835)  
**Code**: https://github.com/famoustourist/GenesisFunc  
**Area**: Agent / Function Calling / Synthetic Data  
**Keywords**: Function Calling, Multi-agent data generation, Tool learning, Synthetic data quality inspection, GRPO  

## TL;DR
GenesisFunc automatically constructs high-quality function-calling training data using a reliable tool pool, multi-agent dialogue generation, and multi-stage quality inspection. After fine-tuning Qwen3-8B, it outperforms open-source function-calling models of the same scale on BFCL, API-Bank, and ACEBench, demonstrating potential for scaling to more tools and multi-turn RL training.

## Background & Motivation
**Background**: Function calling transforms LLMs from pure text generators into agents capable of invoking external tools, serving as the foundation for workflow automation, travel planning, information retrieval, and complex task execution. Current mainstream approaches to enhance function-calling include prompting, SFT, and RL with rewards.

**Limitations of Prior Work**: Function-calling capability highly depends on training data quality. However, real annotated data is costly, and real-world scenarios often involve ambiguous intent, multi-tool combinations, multi-turn interactions, dynamic constraints, and error handling. Existing synthesis pipelines often use manual designs or public APIs, leading to issues like unreliable tools, poor scalability, simplistic scenarios, and weak quality control, which limit generalization.

**Key Challenge**: To train a strong function-calling model, data must be reliable, accurate, diverse, and broad in coverage. However, as the scale of automatic generation increases, issues like tool definition errors, inconsistent parameter extraction, repetitive dialogue intent, and non-executable samples become more frequent.

**Goal**: The authors aim to build an end-to-end automatic data generation pipeline starting from reliable tools, systematically generating single-turn, multi-turn, and special error/no-solution scenarios, while ensuring data quality through a combined automatic and manual evaluation module.

**Key Insight**: Instead of designing synthetic APIs from scratch, GenesisFunc extracts reliable tools from mature benchmarks like BFCL, then uses a multi-agent mechanism to expand semantic scenarios, parameter slots, and dialogue forms, finally employing rule/model/human three-layer verification.

**Core Idea**: Construct function-calling data via a "reliable tool pool + multi-agent generation of diverse dialogues + multi-stage quality inspection," then perform SFT/RL on small models to achieve tool-calling performance close to API models.

## Method
GenesisFunc's method emphasizes data engineering and a quality control closed-loop. Instead of a single LLM writing samples directly, it decomposes the process into roles such as tool selection, memory, parameter selection, dialogue judgment, and posterior verification to prevent low-quality data from entering training.

### Overall Architecture
The input is a set of candidate tools, each containing a name, description, schema, and mandatory/optional parameters. The pipeline consists of three stages: 1. Build a Tool Pool of 1,000 reliable tools from BFCL; 2. Generate single-turn, multi-turn, and special-case dialogues via a multi-agent Dialogue Generation System; 3. Inspect formatting, parameters, semantic completeness, and executability using Rule Checker, Model Checker, and Human Validation. The final data is used for SFT on Qwen3-8B to obtain GenesisFunc-8B; multi-turn scenarios are further optimized using GRPO.

### Key Designs
1.  **Reliable Tool Pool Construction**:
    - **Function**: Ensures the tool definitions for synthetic data come from trustworthy and scalable real-world scenarios.
    - **Mechanism**: Tools are collected from the BFCL evaluation set. GPT-4o is used for semantic clustering to remove redundant or highly similar tools, followed by lightweight human verification for correctness and usability, resulting in 1,000 tools. Since BFCL covers various real tool scenarios, this pool balances reliability and domain diversity.
    - **Design Motivation**: Many synthetic data issues stem from unreliable underlying tools or unrealistic schemas rather than unnatural dialogue. Stabilizing the tool source is foundational for subsequent generation.

2.  **Multi-Agent Dialogue Generation**:
    - **Function**: Generates diverse function-calling dialogues covering single tasks, multi-tasks, multi-turn clarifications, error handling, and tool-free responses.
    - **Mechanism**: The framework includes Sample Agent, Memory Agent, Function Agent, and Judge Agent. The Sample Agent draws target and distractor tools; the Memory Agent tracks historical dialogues to avoid duplication; the Function Agent selects tools and instantiates optional parameter slots; and the Judge Agent selects the best sample from $N=4$ candidates per turn. Inter-agent interaction between a user agent and assistant agent generates the dialogue, where the assistant's action space includes `call`, `ask`, and `answer`.
    - **Design Motivation**: Function-calling data requires more than a simple "user query followed by an API call." Real tasks necessitate distinguishing relevant tools, filling missing parameters, and handling multi-tool combinations; multi-agent task division incorporates both diversity and accuracy.

3.  **Multi-Stage Evaluation**:
    - **Function**: Filters or corrects formatting errors, parameter errors, and semantic inconsistencies before training.
    - **Mechanism**: The Rule Checker verifies tool definition integrity, call formatting, and parameter compliance without executing tools. The Model Checker uses GPT-4o for high-level faithfulness, task satisfaction, and compliance judgments, retaining samples with confidence higher than $\theta=0.75$. The error rate dropped below 5% after automatic screening, with 80% of remaining errors being parameter extraction issues; these low-confidence samples underwent human validation (totaling ~15 hours).
    - **Design Motivation**: SFT is sensitive to noise, especially incorrect tool parameters. Multi-stage inspection combines inexpensive rules, high-level model judgment, and minimal human correction to avoid massive manual annotation costs.

### Loss & Training
The primary model GenesisFunc-8B is fine-tuned on Qwen3-8B using synthesized data, with results reported as an average of three runs. GRPO is used for the RL stage, with rewards based on formatting compliance and functional correctness. The authors leverage Qwen3-8B's thinking mode to include explicit reasoning traces in training; GenesisFunc-8B-RL(part) undergoes SFT on single-turn and special-case data before specifically applying RL to multi-turn dialogues.

## Key Experimental Results

### Main Results
| Dataset / Setting | Metric | GenesisFunc-8B | strong baseline / control | Gain / Conclusion |
|--------|------|------|----------|------|
| BFCL Non-Live | Overall accuracy | 93.31 ± 0.42 | ToolACE-8B 91.04; Qwen3-32B 89.90 | Higher than same-scale SOTA and larger models |
| BFCL Live | Overall accuracy | 83.78 ± 0.37 | ToolACE-8B 80.73; Qwen3-32B 81.13 | Leads in in-domain live tool settings |
| API-Bank | Overall accuracy | 64.79 ± 0.41 | Qwen-ToolRL-8B 60.36; ToolACE-8B 56.21 | Best among open-source methods on out-of-domain API-Bank |
| ACEBench Normal | Overall accuracy | 73.60 ± 0.32 | Qwen-ToolRL-8B 65.10; ToolACE-8B 70.30 | Significant improvement in Normal tool-learning |
| ACEBench Special | Overall accuracy | 83.67 ± 0.35 | Qwen-ToolRL-8B 78.67; Qwen3-8B 76.67 | Strong generalization in special cases |
| Out-of-domain Avg | API-Bank / ACEBench | API-Bank 64.79; ACEBench ~78.64 | prior open-source SOTA | Relative improvements of 7.3% and 9.4% |

### Ablation Study
| Configuration | Key Metrics | Explanation |
|------|---------|------|
| Remove Judge Agent | BFCL Non-Live / Live decreased | Candidate judgment is vital for sample accuracy |
| Remove Memory Agent | BFCL Non-Live / Live significantly decreased | Semantic memory and de-duplication crucial for diversity |
| 1 / 5 / 10 dialogues per tool | Gains from 1 to 5; diminishing returns from 5 to 10 | Volume helps, but marginal utility drops after sufficient diversity |
| No Multi-Stage Evaluation | Lower accuracy across all conditions | Rules + Model + Human verification improve data quality |
| GenesisFunc-8B-RL(part) | Normal 75.20; Multi-Turn 70.00; Special 82.88 | Multi-turn RL significantly enhances complex interactions vs SFT |
| GenesisFunc + ACEBench tools | BFCL 87.89; API-Bank 65.11; ACEBench 81.87 | Adding benchmark tools boosts ACEBench without obvious degradation elsewhere |

### Key Findings
- On in-domain BFCL, GenesisFunc-8B achieved a Non-Live Overall of 93.31 and Live Overall of 83.78, narrowing the gap between small models and API models through training on high-quality aligned tool semantics.
- On out-of-domain tasks, GenesisFunc-8B consistently outperformed same-scale models, indicating it learned generalized tool selection, parameter filling, and multi-turn patterns rather than just memorizing BFCL tools.
- The Memory Agent acts as a "diversity controller": it tracks semantic types to guide generation toward uncovered scenarios, preventing template-based repetition.
- Targeted RL on multi-turn scenarios (RL(part)) proved more effective than applying RL across all data (RL(all)), suggesting that complex function-calling capabilities benefit from phased training.

## Highlights & Insights
- **Engineering-driven approach to data quality**: Reliable tool sourcing, dialogue diversity, parameter accuracy, and multi-stage verification are handled by distinct modules, making it more robust than simple prompt-based generation.
- **Pragmatic Tool Pool selection**: Reusing reliable BFCL tools avoids building schemas from scratch and facilitates natural extension to real downstream tools.
- **Importance of special scenarios**: Models must not only call correct tools but also know when to ask for missing info, reject mismatched tools, or answer directly. Special-case data enhances deployment stability.
- **Clear SFT/RL boundaries**: SFT establishes basic formatting and semantic alignment, while RL(part) focuses on multi-turn reasoning and interaction, which is more controllable than universal RL.

## Limitations & Future Work
- While GenesisFunc-8B is strong among open-source models of its size, it remains inferior to API models like GPT-4 in general reasoning and comprehension.
- The current training data does not fully cover highly complex, multi-turn agentic workflows with tight tool coupling; future work aims for more complex benchmarks.
- The multi-agent and inspection stages rely on powerful models (Gemini-2.5 Pro, GPT-4o), necessitating evaluation of cost and closed-source dependencies.
- Automatic screening still requires manual verification for low-confidence samples; although cost was low (~15h), standards may become more complex as tool pools expand.

## Related Work & Insights
- **vs ToolACE / APIGen / ToolForge**: These methods focus on synthetic function-calling data, but GenesisFunc emphasizes reliable tool sources, multi-agent collaboration, and closed-loop evaluation, specifically targeting parameter errors.
- **vs prompting-based tool use**: GenesisFunc internalizes tool-calling capability into parameters via SFT/RL, providing more stability than ReAct-style prompting as tool complexity grows.
- **vs ToolRL / AWPO**: RL methods emphasize reward optimization; GenesisFunc shows that high-quality SFT data remains a powerful baseline and that RL is best used for specific weaknesses like multi-turn interactions.
- **Insight**: When building agent datasets, one should prioritize explicit control over tool reliability, semantic coverage, parameter distribution, and the ratio of failure/unsolvable samples over pure volume.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The multi-agent generation and inspection are established concepts, but their combination for a function-calling data loop is solid.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers BFCL, API-Bank, ACEBench, ablation, tool expansion, and RL.
- Writing Quality: ⭐⭐⭐⭐☆ The pipeline explanation is clear and experiments are well-organized. 
- Value: ⭐⭐⭐⭐⭐ Highly practical for improving tool-calling in small models and building private tool ecosystems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SPASM: Stable Persona-driven Agent Simulation for Multi-turn Dialogue Generation](spasm_stable_persona-driven_agent_simulation_for_multi-turn_dialogue_generation.md)
- [\[ICML 2026\] From Self-Evolving Synthetic Data to Verifiable-Reward RL: Post-Training Multi-turn Interactive Tool-Using Agents](../../ICML2026/dialogue/from_self-evolving_synthetic_data_to_verifiable-reward_rl_post-training_multi-tu.md)
- [\[ACL 2026\] Discourse Coherence and Response-Guided Context Rewriting for Multi-Party Dialogue Generation](discourse_coherence_and_response-guided_context_rewriting_for_multi-party_dialog.md)
- [\[ACL 2026\] Disambiguation-Centric Finetuning Makes Enterprise Tool-Calling LLMs More Realistic and Less Risky](disambiguation-centric_finetuning_makes_enterprise_tool-calling_llms_more_realis.md)
- [\[ACL 2026\] Context-Agent: Dynamic Discourse Trees for Non-Linear Dialogue](context-agent_dynamic_discourse_trees_for_non-linear_dialogue.md)

</div>

<!-- RELATED:END -->
