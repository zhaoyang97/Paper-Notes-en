---
title: >-
  [Paper Note] Enhancing LLM Agent Safety via Causal Influence Prompting
description: >-
  [ACL 2025 (Findings)][LLM Agent][LLM safety] This paper proposes CIP (Causal Influence Prompting), which utilizes Causal Influence Diagrams (CIDs) to structurally represent decision-making causal relationships for LLM agents. By employing a three-step pipeline—CID initialization, CID-guided interaction, and iterative CID updating—CIP effectively enhances agent safety in code execution and mobile device control tasks.
tags:
  - "ACL 2025 (Findings)"
  - "LLM Agent"
  - "LLM safety"
  - "Causal Influence Diagram"
  - "Autonomous Agent"
  - "Risk Mitigation"
  - "Decision Reasoning"
date: 2026-05-08
content_hash: b45e34703c9db76c
---

# Enhancing LLM Agent Safety via Causal Influence Prompting

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2507.00979](https://arxiv.org/abs/2507.00979)  
**Code**: [GitHub](https://github.com/dongyoon-hahm/CIP)  
**Area**: LLM Agent  
**Keywords**: LLM safety, Causal Influence Diagram, Autonomous Agent, Risk Mitigation, Decision Reasoning

## TL;DR

This paper proposes CIP (Causal Influence Prompting), which utilizes Causal Influence Diagrams (CIDs) to structurally represent decision-making causal relationships for LLM agents. By employing a three-step pipeline—CID initialization, CID-guided interaction, and iterative CID updating—CIP effectively enhances agent safety in code execution and mobile device control tasks.

## Background & Motivation

**Background**: LLM-based autonomous agents are rapidly evolving, capable of completing auxiliary tasks through tool calling, code execution, and device manipulation. While showing great potential in real-world deployment, they also face safety challenges: autonomous decisions by agents can lead to unforeseen harmful consequences, such as executing dangerous code, misoperating devices, or leaking private information.

**Limitations of Prior Work**: Existing methods for enhancing LLM agent safety generally fall into two categories: (1) Rule-based hard constraints (such as blacklists), which are rigid and fail to cover all risk scenarios; (2) Safety alignment training (such as RLHF), which is costly and can degrade the agent's task-completion capabilities. Both approaches share a common limitation: **they lack structured reasoning regarding the consequences of decisions**. Before executing an action, the agent lacks a systematic way to predict "what consequences this action might bring" and must rely heavily on the implicit "intuition" of the language model.

**Key Challenge**: The helpfulness of an agent requires it to flexibly execute various actions to complete user tasks, whereas safety requires it to foresee potential risks of each action. Without an explicit causal reasoning framework, it is difficult for agents to strike a balance between helpfulness and safety—strict safety rules hinder task completion, whereas loose rules may lead to harm.

**Goal**: Design a technology that enhances agent safety purely through prompt engineering without requiring extra training, while preserving the agent's ability to complete tasks.

**Key Insight**: Causal Influence Diagrams (CIDs) provide a mathematical tool to structurally represent causal relationships. The authors model each of the agent's decisions, environmental states, and potential consequences as nodes in a CID, utilizing causal reasoning to foresee risks.

**Core Idea**: Maintain and update a Causal Influence Diagram at each step of the agent's interaction with the environment. Use the causal structure of the CID to guide the agent to "think twice before acting"—predicting risky consequences along the causal chain prior to executing actions, thereby leading to safer decisions.

## Method

### Overall Architecture

The workflow of CIP runs in a three-step loop: (1) **CID Initialization**: Construct an initial Causal Influence Diagram based on the task description, defining the causal relationships among decision nodes, chance nodes (environment states), and utility nodes (goals/safety metrics); (2) **CID-Guided Interaction**: At each decision step, the agent refers to the CID, reasons about potential downstream impacts of the current action along the causal chain, assesses safety risks, and then executes the action; (3) **CID Iterative Update**: Dynamically update the causal relationships and node states in the CID based on physical feedback from the environment and observed behaviors, ensuring the CID accurately reflects the task environment.

### Key Designs

1. **Task-Specification-Based CID Initialization**:

    - **Function**: Automatically construct the initial Causal Influence Diagram from the task description and safety constraints.
    - **Mechanism**: Parse task specifications into three types of nodes in a CID: decision nodes (actions available to the agent), chance nodes (uncertain states in the environment), and utility nodes (objectives like task success rate and safety score). Directed edges represent causal influences, such as "executing delete operation -> file loss -> task failure". During initialization, the LLM is leveraged to reason about possible causal chains based on the task description.
    - **Design Motivation**: CIDs offer a mature formal framework from decision theory, transforming safety risk from implicit "semantic understanding" to explicit "causal reasoning", making safety judgments more interpretable and controllable.

2. **CID-Guided Safe Decision Reasoning**:

    - **Function**: Perform causal risk assessment before the agent executes each action.
    - **Mechanism**: When the agent is about to execute an action, it performs forward reasoning along the causal paths in the CID to determine all downstream impacts. If the reasoning reveals that the action may lead to negative utilities (e.g., a drop in the safety score), the agent is prompted to choose alternative plans or apply safety guardrails. The reasoning process is represented as structured causal paths, making it easy to audit and explain. Implementation-wise, the current state of the CID is encoded into a text format and injected into the agent's prompt with causal reasoning instructions.
    - **Design Motivation**: Simply telling an agent "do not do dangerous things" is too abstract, whereas telling it "if you do A, because of the causal chain A -> B -> C, it may lead to safety risk C" is much more concrete and effective.

3. **Observation-Based CID Iterative Update**:

    - **Function**: Dynamically update the Causal Influence Diagram based on the actual outcomes of agent-environment interactions.
    - **Mechanism**: After executing an action and observing environmental feedback, the agent checks whether actual outcomes match CID predictions. If an unexpected consequence occurs (unforeseen causal path in the CID), new causal edges or nodes are added. If predicted risks fail to occur, the importance of the corresponding causal chain is reduced. This iterative update allows the CID to converge toward the true causal structure of the environment during interaction.
    - **Design Motivation**: Since the initial CID is built from task descriptions, it may miss specific causal relations in real-world environments. Online updates enable the CID to adapt to unique circumstances, providing more accurate safety guidance.

### Loss & Training

CIP is a pure prompt-engineering-based approach and does not involve any model training or parameter updates. All CID construction, reasoning, and updates are executed by prompting the LLM. This allows CIP to be directly applied to any LLM agent without extra training costs.

## Key Experimental Results

### Main Results

Safety Evaluation on Code Execution Tasks:

| Method | Safety Rate (%) | Task Success Rate (%) | Comprehensive Score |
|------|----------|-------------|---------|
| No Safety Prompt | Baseline (Low) | High | Medium |
| Rule-Based Safety Prompt | Medium | Medium (Restricted) | Medium |
| CIP | Significantly Highest | Maintained Well | Optimal |

Safety Evaluation on Mobile Device Control Tasks:

| Method | Safety Rate (%) | Task Success Rate (%) | Comprehensive Score |
|------|----------|-------------|---------|
| No Safety Prompt | Baseline | High | Medium |
| CoT Safety Reasoning | Moderate Gain | Slightly Lower | Medium |
| CIP | Significantly Highest | Mostly Maintained | Optimal |

### Ablation Study

| Configuration | Safety Rate | Description |
|------|--------|------|
| Full CIP (Init + Guidance + Update) | Optimal | All three steps are essential |
| Without CID Update (Static Graph) | Medium | Unable to adapt to environmental changes |
| Without CID-guided Reasoning | Low | Graph exists but is not used for decision making |
| Without CID Initialization (Empty Graph) | Lowest | Equivalent to omitting CIP |
| Replaced with Simple Safety Prompts | Moderately Low | Lacks structured reasoning |

### Key Findings

- **Structured CID reasoning outperforms natural language safety prompts**: Textual prompts like "please pay attention to safety" show limited effects, whereas CIDs provide explicit causal paths so that the agent can visualize concrete risk trajectories.
- **CID updates bring substantial gains**: Static CIDs (without updates) achieve significantly lower safety rates than dynamic ones, proving that online adaptation is crucial.
- **Enhancing safety does not compromise task helpfulness considerably**: While significantly boosting the safety rate, CIP only slightly reduces the task completion rate, demonstrating that causal reasoning helps agents identify safe yet effective alternatives.
- **Generality across task types**: CIP remains effective in two distinct risk environments: code execution (risk of file deletion/data leakage) and device control (risk of application misoperation).

## Highlights & Insights

- **Introducing decision-theory CIDs to LLM agent safety**: CIDs are a classic tool in game theory and decision analysis. Utilizing them for agent safety represents an elegant cross-disciplinary application. The structured nature of causal graphs transforms safety reasoning from a "black-box intuition" into traceable, explainable causal path analysis. This approach can be extended to any agent scenario requiring safe decision-making, such as autonomous driving and medical assistance agents.
- **Zero-training safety enhancement**: CIP is implemented entirely via prompts, eliminating the need for extra training data or fine-tuning, resulting in nearly zero deployment costs. This is particularly valuable for fast-iterating agent products.
- **Dynamically updated causal models**: The CID is not a static repository of domain knowledge but evolves continuously during interaction. This enables the system to handle edge cases that were not covered in the initial description.

## Limitations & Future Work

- **CID quality depends on LLM's causal reasoning limits**: The creation and modification of CIDs rely entirely on LLM prompting. If the LLM has bottlenecked causal reasoning capabilities, the constructed CID may be inaccurate or miss critical causal relations.
- **Increased prompt length and inference overhead**: Textual representations of CIDs consume context window space. As interaction steps increase and the CID scales up, it may squeeze out room for other useful context.
- **Relatively limited evaluation scenarios**: The method is only validated on code execution and mobile device control. Its safety performance needs to be examined in more complex settings like multi-agent collaboration or long-horizon tasks.
- **Completeness of causal paths**: It is difficult to guarantee that the CID covers all possible risk trajectories, leaving "unknown unknowns" unresolved.

## Related Work & Insights

- **vs Constitutional AI**: Anthropic's Constitutional AI constrains model behaviors via predefined principles, which is a "rule-based" safety approach. In contrast, CIP dynamically evaluates safety through task-specific causal reasoning, making it more flexible and adaptive across different tasks.
- **vs Chain-of-Thought (CoT) Safety Reasoning**: CoT encourages agents to "think" before making decisions but lacks a structured causal framework. CIP's CID provides a clear structure for "what to think" and "how to think," resulting in more systematic reasoning.
- **vs Toolformer/ReAct**: These agent frameworks focus on utilizing tools to execute tasks efficiently but fail to account for the safety of tool use. CIP can be stacked on top of these frameworks as a safety-guarantee layer.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Introducing Causal Influence Diagrams to LLM agent safety is a creative cross-disciplinary adaptation, and the three-step loop update mechanism is well-designed.
- **Experimental Thoroughness**: ⭐⭐⭐ The evaluation on only two task scenarios is somewhat narrow. Although ablation studies are comprehensive, comparison against more baseline methods is lacking.
- **Writing Quality**: ⭐⭐⭐⭐ The problem definition is clear, the CID concept is step-by-step introduced, and the three-step workflow is described intuitively.
- **Value**: ⭐⭐⭐⭐ A zero-training safety enhancement paradigm holds high practical value, providing a fresh perspective for agent safety research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MetaSynth: Meta-Prompting-Driven Agentic Scaffolds for Diverse Synthetic Data Generation](metasynth_meta-prompting-driven_agentic_scaffolds_for_diverse_synthetic_data_gen.md)
- [\[ICML 2026\] Think Twice Before You Act: Enhancing Agent Behavioral Safety with Thought Correction](../../ICML2026/llm_agent/think_twice_before_you_act_enhancing_agent_behavioral_safety_with_thought_correc.md)
- [\[AAAI 2026\] CausalTrace: A Neurosymbolic Causal Analysis Agent for Smart Manufacturing](../../AAAI2026/llm_agent/causaltrace_a_neurosymbolic_causal_analysis_agent_for_smart_manufacturing.md)
- [\[ACL 2025\] Agentic Reasoning: A Streamlined Framework for Enhancing LLM Reasoning with Agentic Tools](agentic_reasoning_tools.md)
- [\[ACL 2025\] Enhancing Interpretable Image Classification Through LLM Agents and Conditional Concept Bottleneck Models](llm_agent_image_classification.md)

</div>

<!-- RELATED:END -->
