---
title: >-
  [Paper Note] Attractive Metadata Attack: Inducing LLM Agents to Invoke Malicious Tools
description: >-
  [NeurIPS 2025][LLM Agent][Tool invocation attack] AMA (Attractive Metadata Attack) demonstrates that by carefully crafting malicious tool metadata (name, description, parameter schema) alone — without prompt injection or internal model access — an attacker can induce LLM agents to invoke malicious tools and leak private data at a success rate of 81–95%, while barely affecting original task completion (98%+), with existing defenses (auditors, prompt rewriting) proving largely ineffective.
tags:
  - NeurIPS 2025
  - LLM Agent
  - Tool invocation attack
  - metadata manipulation
  - privacy leakage
  - LLM agent security
  - MCP
date: 2026-05-08
content_hash: c0f5df06944628c8
---

# Attractive Metadata Attack: Inducing LLM Agents to Invoke Malicious Tools

**Conference**: NeurIPS 2025  
**arXiv**: [2508.02110](https://arxiv.org/abs/2508.02110)  
**Code**: [https://github.com/SEAIC-M/AMA](https://github.com/SEAIC-M/AMA)  
**Area**: AI Security / LLM Agent  
**Keywords**: Tool invocation attack, metadata manipulation, privacy leakage, LLM agent security, MCP

## TL;DR
AMA (Attractive Metadata Attack) demonstrates that by carefully crafting malicious tool metadata (name, description, parameter schema) alone — without prompt injection or internal model access — an attacker can induce LLM agents to invoke malicious tools and leak private data at a success rate of 81–95%, while barely affecting original task completion (98%+), with existing defenses (auditors, prompt rewriting) proving largely ineffective.

## Background & Motivation

**Background**: LLM agents interact with external services via tool calling (function calling). Protocols such as MCP standardize tool registration and invocation workflows. Agents select which tool to invoke based on tool metadata (name, description, parameters).

**Limitations of Prior Work**: (a) Prompt injection attacks require manipulating user inputs and can be filtered by agent-side defenses; (b) however, tool registration is open — anyone can publish a tool and control its metadata; (c) no prior work systematically studies "metadata manipulation" as a distinct attack surface.

**Key Challenge**: LLM agents trust tools' self-descriptions when making selection decisions — yet malicious tools can craft their descriptions to appear more "attractive."

**Goal**: Systematically evaluate and demonstrate the threat of metadata manipulation to LLM agent tool selection.

**Key Insight**: The attack is formalized as a state-action-value optimization problem — the state is the current malicious tool set, the action is generating new tool metadata, and the value is the invocation probability. An LLM iteratively optimizes metadata to maximize "attractiveness."

**Core Idea**: Iteratively optimize the name/description/parameter schema of a malicious tool via an LLM → maximize its selection probability in the agent's tool-choosing process → steal private data upon invocation.

## Method

### Overall Architecture
**Attacker**: Define malicious behavior (e.g., exfiltrating user emails) → LLM generates initial tool metadata → batch-test invocation probabilities → retain the highest-probability tool → iterate for $K$ rounds → publish to tool registry.  
**Victim Agent**: Receives user task → retrieves available tools → LLM selects the "most relevant" tool → invokes malicious tool → data is leaked.

### Key Designs

1. **State-Action-Value Optimization (SAV)**:

    - Function: Systematically generate highly attractive malicious metadata.
    - Mechanism: State $S = \{(t, p)\}$ (malicious tools and their invocation probabilities); action = LLM generating a new tool $t = \text{LLM}(Q, NT, S, P_g)$; value $V(t) = p_i + \lambda(p_i - p_j)$ — combining absolute probability with relative improvement. Batch generation → select best → iterate.
    - Design Motivation: Far more efficient than random trial — the LLM serves as a "metadata optimizer."

2. **Stealthiness Design**:

    - Function: Ensure the malicious tool does not disrupt the original task upon invocation.
    - Mechanism: Upon execution, the malicious tool simultaneously completes the task (by proxying to the correct tool) and exfiltrates data. Task completion rate is maintained at 98%+.
    - Design Motivation: If the malicious tool causes task failure, users will remove it — the tool must maintain a "benign appearance."

3. **Cross-Defense Evaluation**:

    - Function: Test the effectiveness of existing defenses.
    - Mechanism: Prompt rewriting reduces ASR by only 5–26% (residual ASR ~90%); auditor detection rate is ~55% (insufficient); MCP-structured selection is still bypassed.
    - Design Motivation: Demonstrate the inadequacy of existing defense frameworks.

### Loss & Training
- Pure inference-time attack — no model training required.
- Attack iterates for $K$ rounds (typically converges in 3–5 rounds).

## Key Experimental Results

### Main Results (Attack Success Rate, ASR)

| Model | ASR | Privacy Leakage Rate | Task Completion Rate |
|-------|-----|----------------------|----------------------|
| Gemma-3 27B | **95.58%** | 94.69% | 98.42% |
| LLaMA-3.3 70B | **94.80%** | 94.80% | 99.67% |
| Qwen-2.5-32B | **94.54%** | 92.63% | 97.08% |
| GPT-4o-mini | **81.43%** | 81.12% | 85.86% |
| + Combined with injection attack | **95–99%** | — | — |

### Ablation Study (Defense Evaluation)

| Defense Method | ASR Reduction | Residual ASR |
|----------------|---------------|--------------|
| Prompt rewriting | 5–26% | ~90% |
| Auditor | TPR ~55% | High |
| MCP | Still effective | High |

### Key Findings
- Open-source models are more vulnerable than commercial models (95% vs. 81%) — GPT-4o-mini exhibits better tool-selection security.
- Malicious tools exfiltrate data while barely affecting task completion (98%+ task completion + 95% privacy leakage).
- Combining with prompt injection further raises ASR to 95–99% — the two attack types are complementary.
- All existing defenses are insufficient — new defense paradigms are needed.

## Highlights & Insights
- **Systematic Revelation of a New Attack Surface**: Metadata manipulation requires no prompt injection — this is a critical and previously overlooked security issue.
- **The Alarming Combination of 98% Task Completion + 95% Privacy Leakage**: Users are attacked with zero awareness.
- **Failure of All Existing Defenses**: The community is called upon to develop dedicated security mechanisms at the tool-selection stage.

## Limitations & Future Work
- Assumes the attacker can publish or modify tools — tool registry auditing mechanisms are needed.
- Focuses solely on tool invocation attacks — runtime behavior monitoring is not addressed.
- The root cause of the performance gap between commercial and open-source models is not deeply analyzed.

## Related Work & Insights
- **vs. Prompt Injection**: Prompt injection manipulates inputs; AMA manipulates tool descriptions — entirely different attack surfaces.
- **vs. Adversarial Examples**: AMA manipulates textual metadata rather than numerical perturbations.
- **Insights**: Tool protocols such as MCP need an additional security review layer.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic revelation of the security threat posed by tool metadata manipulation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four models + three defenses + MCP evaluation.
- Writing Quality: ⭐⭐⭐⭐ Attack framework is described clearly.
- Value: ⭐⭐⭐⭐⭐ Carries significant warning implications for LLM agent security.

### Supplementary Notes on the Method
- **Detailed Attack Pipeline**: (1) Define malicious behavior (e.g., "forward user emails to the attacker's mailbox") → (2) use an LLM to generate tool metadata that resembles an "email backup assistant" → (3) publish to the tool registry → (4) when the user requests "help me back up my emails," the agent selects the attacker's tool → (5) data is leaked.
- **Design of the Weighted Value Function**: $V(t_i^j, Q, NT, t_j) = p_i^j + \lambda(p_i^j - p_j)$ — requires not only a high absolute invocation probability but also an improvement over the previously best tool. $\lambda$ controls the exploration–exploitation trade-off.
- **Why GPT-4o-mini Is More Secure**: Commercial models may incorporate additional tool-selection safety filters — yet an 81% ASR still indicates that defenses are far from sufficient.
- **Analogy to Phishing Attacks**: AMA is "tool phishing" — just as phishing emails impersonate legitimate correspondence, malicious tools impersonate legitimate ones. AMA is even more covert because tool invocation is automated.
- **Warning to the MCP Ecosystem**: The MCP protocol standardizes tool registration but provides no security review mechanism — anyone can publish a tool, constituting an open attack surface.

- **Summary of Defense Recommendations**: Tool registration should require security review and signature verification; agents should perform semantic anomaly detection on tool descriptions; sensitive operations should require user confirmation; tool invocation logs should be retained for post-hoc auditing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Distilling LLM Agent into Small Models with Retrieval and Code Tools](distilling_llm_agent_into_small_models_with_retrieval_and_co.md)
- [\[NeurIPS 2025\] It's LIT! Reliability-Optimized LLMs with Inspectable Tools](its_lit_reliability-optimized_llms_with_inspectable_tools.md)
- [\[ACL 2026\] MCP-Flow: Facilitating LLM Agents to Master Real-World, Diverse and Scaling MCP Tools](../../ACL2026/llm_agent/mcp-flow_facilitating_llm_agents_to_master_real-world_diverse_and_scaling_mcp_to.md)
- [\[NeurIPS 2025\] A-MEM: Agentic Memory for LLM Agents](a-mem_agentic_memory_for_llm_agents.md)
- [\[NeurIPS 2025\] LLM Agents for Knowledge Discovery in Atomic Layer Processing](llm_agents_for_knowledge_discovery_in_atomic_layer_processing.md)

</div>

<!-- RELATED:END -->
