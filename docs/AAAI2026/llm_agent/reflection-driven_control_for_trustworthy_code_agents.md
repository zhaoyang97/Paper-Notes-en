---
title: >-
  [Paper Note] Reflection-Driven Control for Trustworthy Code Agents
description: >-
  [AAAI 2026][LLM Agent][reflection-driven control] This paper proposes a Reflection-Driven Control module that elevates "self-reflection" from a post-hoc patch to a first-class control loop within the agent reasoning process. Through three components—a lightweight self-checker, evidence-driven repair, and a reflective memory repository—the approach significantly improves code security rates on secure code generation tasks.
tags:
  - AAAI 2026
  - LLM Agent
  - reflection-driven control
  - secure code generation
  - self-inspection mechanism
  - dynamic memory repository
  - trustworthy agent
date: 2026-05-08
content_hash: e9753edac6d4291b
---

# Reflection-Driven Control for Trustworthy Code Agents

**Conference**: AAAI 2026
**arXiv**: [2512.21354](https://arxiv.org/abs/2512.21354)
**Code**: None
**Area**: LLM Agent / Secure Code Generation
**Keywords**: reflection-driven control, secure code generation, self-inspection mechanism, dynamic memory repository, trustworthy agent

## TL;DR

This paper proposes a Reflection-Driven Control module that elevates "self-reflection" from a post-hoc patch to a first-class control loop within the agent reasoning process. Through three components—a lightweight self-checker, evidence-driven repair, and a reflective memory repository—the approach significantly improves code security rates on secure code generation tasks.

## Background & Motivation

**State of the Field**: LLM agents are evolving from single-turn text generators into autonomous systems capable of multi-step reasoning, tool use, and code execution. Code generation is one high-stakes application domain where generated code may contain security vulnerabilities such as SQL injection and buffer overflows.

**Limitations of Prior Work**: Current agent systems lack reliable security control mechanisms. Even powerful base models can produce unsafe outputs, and as tool use and autonomy enter the loop, jailbreak attacks and prompt injection further expose fragile control surfaces. Existing safety measures are largely scattered and reactive.

**Root Cause**: Agents require autonomy to execute complex tasks, yet increased autonomy amplifies security risks. The central challenge is how to ensure the security and auditability of generated code without sacrificing autonomy.

**Paper Goals**: To design a standardized, plug-and-play control module that enables continuous self-supervision and self-correction during code generation, improving security without compromising functional correctness.

**Starting Point**: Elevating "reflection" from an external post-hoc process to a first-class internal control loop within the agent, spanning the planning, execution, and verification phases.

**Core Idea**: Constructing a Plan–Reflect–Verify three-stage framework in which the Reflect layer serves as a plug-and-play module, achieving continuous security control through lightweight pre-filtering via self-inspection, dynamic memory RAG-driven repair, and a compilation verification feedback loop.

## Method

### Overall Architecture

The Reflex Agent is embedded as a plug-and-play module within a general agent orchestration architecture. It comprises three core components: (1) a Lightweight Self-Checker; (2) a Reflective Prompt Engine; and (3) a Reflective Memory Repository. The workflow proceeds as: self-inspection → triage → evidence-driven repair → verification → memory write-back.

### Key Designs

**Component 1: Lightweight Self-Checker**

- **Function**: Performs rapid security diagnosis on input code, outputting a binary verdict (SAFE / UNSAFE) to enable efficient task triage.
- **Mechanism**: Self-inspection is formulated as an LLM-based binary classification task. Given input code $x$ and context $c$, a concise security review prompt $p_{sc}$ is constructed, instructing the model to output only SAFE or UNSAFE. This is formalized as $\text{verdict} = \text{LLM}_{\{\text{SAFE}, \text{UNSAFE}\}}(p_{sc} | x, c)$.
- **Design Motivation**: To avoid costly full reflection on all code. Code judged SAFE is written directly to the memory repository as a positive sample, while UNSAFE code enters the full repair pipeline. This substantially reduces average inference cost in large-scale task processing scenarios.

**Component 2: Reflective Prompt Engine**

- **Function**: Performs deep analysis and self-improvement on code judged UNSAFE, guiding the model through systematic vulnerability identification and repair strategy derivation.
- **Mechanism**: A structured multi-turn reflective dialogue prompt is constructed, transforming a single code generation task into multi-turn chain-of-thought reflection. The model is guided to reason progressively through the stages of problem identification → root cause analysis → repair strategy → code implementation. Structured reflection records are systematically stored in the dynamic memory repository, with each resolved case elevated to a reusable reasoning pattern.
- **Design Motivation**: Making the reflection process explicit and structured not only improves the quality of current repairs, but also provides high-quality references for subsequent tasks through knowledge distillation, enabling continuous evolution.

**Component 3: Reflective Memory Repository**

- **Function**: Constructs a dynamically evolving knowledge base of security repairs, supporting vectorized retrieval and structured metadata management.
- **Mechanism**: A two-layer retrieval architecture is designed—**dynamic memory** $M_D$ (a ChromaDB vector database storing runtime-accumulated repair cases, offering high relevance and low latency) and **static memory** $M_S$ (predefined secure coding standards and vulnerability databases serving as foundational knowledge anchors). The retrieval strategy is hierarchical: Top-k highly similar cases are retrieved from $M_D$ first; when dynamic memory is insufficient (hit count $< k_{min}$ or similarity $< $ threshold $\theta$), the system falls back to $M_S$ for supplementary queries.
- **Design Motivation**: To balance retrieval efficiency with knowledge coverage. Dynamic memory ensures the system can rapidly reuse verified contextual knowledge, while static memory guarantees the integrity of core security principles. As cases accumulate, the system progressively transitions from reliance on static knowledge to self-sufficiency.

### Loss & Training

This paper involves no model fine-tuning. The core strategy is **dynamic memory accumulation**—repair cases that pass verification are automatically written back to the memory repository, forming a closed loop of "low-cost pre-reflection → evidence-driven generation → auditable knowledge accumulation." Evaluation is performed via CodeQL static analysis for vulnerability detection, supplemented by LLM Judge-based assessment of code quality and security compliance.

## Key Experimental Results

### Main Results

Comparison of Base vs. Base+Reflex across four LLMs on 8 CWE vulnerability categories:

| Model | Security Rate Base→+Reflex | Pass Rate Base→+Reflex |
|-------|--------------------------|----------------------|
| GPT-3.5-turbo | 93.7→96.6 (↑2.9) | 88.0→92.4 (↑4.4) |
| GPT-4o | 85.7→95.0 (↑9.3) | 95.2→94.9 (↓0.3) |
| Qwen3-coder-plus | 83.7→94.9 (↑11.2) | 86.7→80.1 (↓6.6) |
| Gemini-2.5-pro | 88.0→97.1 (↑9.1) | 91.4→94.9 (↑3.5) |

Security rate improves by an average of approximately 8.1 percentage points; functional pass rates remain largely stable overall.

### Ablation Study

- **Dynamic RAG Evolution**: Over 5 iterations, average retrieval similarity improves from 0.850 to 0.980 (+15.3%); retrieval success rate rises from 85% to 100%; static memory fallback rate drops from 15% to 0%. Knowledge saturation is reached around iteration 4.
- **Similarity vs. Repair Accuracy**: At similarity 0.95–1.00, repair accuracy is 100%; at 0.70–0.85, it is 93.8%; below 0.70, it drops to 75%. This validates the rationale for the 0.70 threshold.
- **Reflection Depth**: A single round of reflection captures approximately 90% of key repair patterns; subsequent rounds primarily improve non-core aspects (code style, exception handling completeness, etc.), yielding diminishing marginal returns.

### Key Findings

1. The Reflex module is **universally effective** across different base models, with the largest gains observed for models with lower initial security rates (Qwen3-coder-plus ↑11.2%).
2. Security improvements come with **minimal functional sacrifice**—pass rates for most models remain largely stable or slightly improve.
3. **One round of reflection suffices**: 90% of security gains derive from the first round, offering an efficient strategy for practical deployment.
4. Overall overhead is extremely low: average processing time per scenario is 28.8 seconds, with the core Reflex logic (RAG retrieval 0.8s + reflective verification 3.2s) accounting for only 13.9%.

## Highlights & Insights

- **Paradigm shift of "reflection as control"**: Elevating reflection from a post-hoc check to a real-time control signal represents an important conceptual contribution to the agent security domain.
- **Closed-loop self-evolution**: The progression of the dynamic memory repository from cold start to self-sufficiency demonstrates the system's adaptive capacity.
- **Neural–symbolic division of labor**: LLMs handle semantic understanding and repair, while compilers, tests, and CodeQL handle hard-constraint verification—"let LLMs do what they do best, and let tools do what is deterministic."
- **Practicality-oriented**: Plug-and-play design, no fine-tuning required, negligible cost (<\$0.001/scenario), genuinely targeting engineering deployment.

## Limitations & Future Work

1. **Minor functional degradation**: Qwen3-coder-plus experiences a 6.6% drop in pass rate, suggesting that reflective repair may be overly conservative for certain models.
2. **Limited evaluation scenarios**: Only 8 CWE categories are covered; more complex cross-file or cross-module security vulnerabilities are not addressed.
3. **Single language and scenario**: The work primarily targets code completion in C/C++ and Python, without extension to full project-level code generation.
4. **Manual preparation of static memory**: Security coding standards and vulnerability databases require pre-construction.
5. **Lack of adversarial evaluation**: Robustness against adversarial prompts designed to circumvent security checks is not tested.

## Related Work & Insights

- **Agent security frameworks**: OWASP Top 10 for LLM Agents, THOR framework—the attack surface of agent systems far exceeds that of traditional AI.
- **TRiSM framework** (Trust, Risk & Security Management): Agent systems require full-stack defense at the input, reasoning, and tool layers.
- **RepairAgent**: Tool-integrated architectures can maintain auditability under strict system constraints.
- **Insights**: Future agent security should shift from "post-hoc patching" to "security by design"; the reflective control loop proposed in this paper serves as a compelling exemplar.

## Rating

⭐⭐⭐⭐

The conceptual innovation is strong—elevating reflection from an ad hoc patch to a first-class control mechanism constitutes a meaningful paradigm contribution. The experimental design is comprehensive (covering effectiveness, ablation, and cost dimensions), and the approach is highly practical (plug-and-play, low overhead, no fine-tuning required). Primary weaknesses are the narrow evaluation scope and the absence of adversarial testing.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Towards Trustworthy Multi-Turn LLM Agents via Behavioral Guidance](towards_trustworthy_multi-turn_llm_agents_via_behavioral_guidance.md)
- [\[AAAI 2026\] COVR: Collaborative Optimization of VLMs and RL Agent for Visual-Based Control](covrcollaborative_optimization_of_vlms_and_rl_agent_for_visu.md)
- [\[AAAI 2026\] PerTouch: VLM-Driven Agent for Personalized and Semantic Image Retouching](pertouch_vlm-driven_agent_for_personalized_and_semantic_image_retouching.md)
- [\[ICLR 2026\] CoMind: Towards Community-Driven Agents for Machine Learning Engineering](../../ICLR2026/llm_agent/comind_towards_community-driven_agents_for_machine_learning_engineering.md)
- [\[AAAI 2026\] MedLA: A Logic-Driven Multi-Agent Framework for Complex Medical Reasoning with Large Language Models](medla_a_logic-driven_multi-agent_framework_for_complex_medic.md)

<!-- RELATED:END -->
