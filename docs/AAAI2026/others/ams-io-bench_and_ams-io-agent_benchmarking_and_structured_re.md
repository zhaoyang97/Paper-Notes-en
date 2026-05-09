---
title: >-
  [Paper Note] AMS-IO-Bench and AMS-IO-Agent: Benchmarking and Structured Reasoning for Analog and Mixed-Signal Integrated Circuit Input/Output Design
description: >-
  [AAAI 2026][LLM Agent] This paper proposes AMS-IO-Agent, a domain-specific LLM-based agent that transforms natural language design intent into production-ready analog and mixed-signal IC I/O ring designs via a structured Intent Graph and a domain knowledge base. It also introduces AMS-IO-Bench, the first benchmark for AMS I/O ring automation. The agent-generated I/O ring is validated in a 28nm CMOS tape-out and demonstrated to be directly applicable to real chip fabrication.
tags:
  - AAAI 2026
  - LLM Agent
  - AMS IC
  - I/O Ring
  - EDA Automation
  - Structured Reasoning
date: 2026-05-08
content_hash: 46ab77c34a3b3a56
---

# AMS-IO-Bench and AMS-IO-Agent: Benchmarking and Structured Reasoning for Analog and Mixed-Signal Integrated Circuit Input/Output Design

**Conference**: AAAI 2026  
**arXiv**: [2512.21613v1](https://arxiv.org/abs/2512.21613v1)  
**Code**: [https://github.com/Arcadia-1/AMS-IO-Agent](https://github.com/Arcadia-1/AMS-IO-Agent) / [https://github.com/Arcadia-1/AMS-IO-Bench](https://github.com/Arcadia-1/AMS-IO-Bench)  
**Area**: Other  
**Keywords**: LLM Agent, AMS IC, I/O Ring, EDA Automation, Structured Reasoning  

## TL;DR
This paper proposes AMS-IO-Agent, a domain-specific LLM-based agent that transforms natural language design intent into production-ready analog and mixed-signal IC I/O ring designs via a structured Intent Graph and a domain knowledge base. It also introduces AMS-IO-Bench, the first benchmark for AMS I/O ring automation. The agent-generated I/O ring is validated in a 28nm CMOS tape-out and demonstrated to be directly applicable to real chip fabrication.

## Background & Motivation
The input/output (I/O) subsystem of analog and mixed-signal (AMS) ICs handles signal interfaces, power delivery, and ESD protection, making it a foundational component of any chip. Unlike digital I/O, which can be automated via scripting, AMS I/O design remains highly manual due to complex requirements including diverse signal types, multiple power domains, power integrity constraints, and sensitive analog signal routing. A junior engineer may spend one to two days manually assembling and verifying pad placement and connectivity, and iterative pin revisions impose substantial additional effort.

Existing LLM applications in IC design—such as RTL code generation and netlist synthesis—have not addressed the AMS I/O domain for three reasons: (1) domain knowledge is fragmented and scattered across internal team documentation; (2) there is no standardized task interface, as designers interact via GUIs or domain-specific languages that LLMs cannot directly access; and (3) no public benchmark exists.

## Core Problem
The central challenge is enabling LLMs to interpret unstructured pin planning descriptions (natural language, tables, etc.) and automatically generate AMS I/O ring designs that pass DRC (Design Rule Check) and LVS (Layout vs. Schematic) verification and are ready for tape-out. The key difficulty is that LLMs cannot reliably generate high-quality EDA scripts (e.g., Cadence SKILL), and naive end-to-end code generation completely fails on this constraint-intensive, domain-specific task.

## Method

### Overall Architecture
AMS-IO-Agent adopts a three-tier hierarchical architecture: the user provides a pin planning text → the LLM structures it into an Intent Graph → an Intent Graph Adaptor parses the Intent Graph into executable EDA scripts (SKILL/csh) → schematics and layouts are generated in Cadence Virtuoso → Calibre is invoked for DRC/LVS verification. This pipeline decouples high-level intent reasoning from low-level implementation: the LLM focuses on understanding design intent, while deterministic modules handle constraint solving and script generation.

### Key Designs
1. **Design Intent Structuring**: Unstructured pin planning inputs (natural language descriptions, pin lists, semi-structured tables) are converted into a standardized Intent Graph—a JSON-format graph representation. Each node represents a pad or corner cell, with attributes including name, device type, spatial position, orientation, and pin connections. Construction proceeds in two steps: **explicit completion**—inferring signal type, device type, and default orientation from naming conventions (e.g., DCLK = digital clock, VCM = common-mode voltage) via the knowledge base; and **implicit reasoning**—automatically inserting elements required by design rules but absent from the specification, such as corner cells. The Intent Graph differs from a netlist: whereas a netlist captures only circuit connectivity, the Intent Graph also encodes spatial relationships, semantic context, and domain knowledge, making it simultaneously human-readable, LLM-generatable, and efficiently machine-parseable.

2. **Intent Graph Adaptor**: This component serves as a middleware layer between the LLM and commercial EDA tools. Direct LLM generation of SKILL scripts is avoided because training data for SKILL is extremely scarce and LLMs cannot reliably produce it; moreover, SKILL itself is ill-suited for complex data manipulation. The Adaptor implements deterministic processing in Python, covering structured data parsing, constraint solving, geometric computation (e.g., precisely calculating cell coordinates according to I/O design rules), SKILL script generation (for creating schematics and layouts in Cadence Virtuoso), and csh script generation (for invoking Calibre verification tools). These functions are encapsulated as a reusable tool library that the Agent calls on demand, rather than being regenerated by the LLM each time.

3. **Domain-Specific Knowledge Base**: Distilled from training materials and design practices of a 10+ person professional AMS IC design team, validated across 50+ tape-out projects. It covers device selection practices, layout conventions, power domain rules, ESD protection requirements, and naming conventions. The entire knowledge base spans approximately 6k tokens—compact enough to be injected directly into the LLM's context without RAG retrieval or model fine-tuning. The content mirrors materials used to onboard junior engineers, who typically require one to three days to internalize it.

### Loss & Training
This approach requires no LLM training or fine-tuning. Domain knowledge is injected via prompt engineering, and zero-shot design automation is achieved through the structured reasoning pipeline. The smolagents framework is employed, with GPT-4o, Claude-3.7, and DeepSeek-V3 as backbone models accessed via API.

## Key Experimental Results

### AMS-IO-Bench
Thirty test cases are constructed from ten real tape-out projects, organized into three difficulty levels:
- **Easy** (10 cases): single signal domain, small scale
- **Medium** (10 cases): multiple power domains (digital + analog), standard MPW chips (~$1\text{mm} \times 1\text{mm}$), requiring cross-domain reasoning
- **Hard** (10 cases): double-row/staggered pads, enlarged chip outlines, custom I/O cells, dedicated ESD power supplies, etc.

### Five Evaluation Metrics
Intent Graph accuracy → Shape Score (VLM visual evaluation) → DRC pass rate → LVS pass rate → DRC+LVS pass rate

| Method | IG (%) | Shape (%) | DRC (%) | LVS (%) | DRC+LVS (%) | Time (min) | Token (k) |
|---|---|---|---|---|---|---|---|
| Human Design | 100 | 100 | 100 | 100 | 100 | ~480 | – |
| Direct LLM (GPT-4o) | 0 | 0 | 0 | 0 | 0 | 0.2 | 1k |
| AMS-IO-Agent (GPT-4o) | 100 | 100 | 76.67 | 66.67 | 63.33 | 4.1 | 160k |
| AMS-IO-Agent (Claude-3.7) | 100 | 100 | 93.33 | 76.67 | 76.67 | 4.2 | 96k |
| AMS-IO-Agent (DeepSeek-V3) | 100 | 100 | 93.33 | 76.67 | 76.67 | 5.1 | 105k |

### DRC+LVS Pass Rate by Difficulty Level

| Model | Easy | Medium | Hard |
|---|---|---|---|
| GPT-4o | 10/10 | 7/10 | 2/10 |
| Claude-3.7 | 10/10 | 9/10 | 4/10 |
| DeepSeek-V3 | 10/10 | 10/10 | 3/10 |

### Ablation Study
- KB only, without Intent Graph and Adaptor → 0% DRC+LVS; direct LLM generation of SKILL completely fails
- KB + Adaptor without Intent Graph (LLM generates Python code directly) → Shape 100% but DRC only 20%, LVS 0%
- KB + Intent Graph without Adaptor → Intent Graph 100% but no usable layout produced
- **All three components are indispensable**: the full configuration (KB + IG + Adaptor) achieves 93.33% DRC and 76.67% DRC+LVS
- The large gap between direct LLM code generation and structured intent reasoning confirms the necessity of the intermediate representation

## Highlights & Insights
- **First demonstration of an LLM Agent making a substantive contribution in real chip tape-out**: validated in 28nm CMOS fabrication, with the agent-generated I/O ring used directly in silicon manufacturing and passing functional verification—the first reported instance of an LLM Agent completing a non-trivial AMS IC design subtask at tape-out level
- **Elegant design of a structured intermediate representation**: the Intent Graph is simultaneously human-readable, LLM-generatable, and machine-parseable, directly addressing the fundamental limitation of LLMs in generating EDA DSL code
- **Substantial engineering efficiency gains**: design time is reduced from approximately 480 minutes per case (human) to approximately 5 minutes, with token consumption kept in the 100k range
- **Transferable agent architecture**: the paradigm of separating high-level intent understanding (LLM) from low-level deterministic execution (tools) is applicable to any domain requiring the translation of natural language intent into precise technical outputs

## Limitations & Future Work
- Coverage is limited to wirebond-packaged AMS I/O rings; flip-chip and other packaging styles are not addressed
- The knowledge base is tied to specific design conventions and process nodes; migration to other foundry rules requires knowledge base replacement
- A peak DRC+LVS pass rate of 76.67% indicates that approximately one quarter of designs still require manual correction, particularly at the Hard difficulty level
- The benchmark scale of 30 test cases is modest (though the authors argue that data scarcity is an inherent and justifiable constraint in the AMS IC domain)
- More complex AMS design tasks—such as full layout routing and analog circuit optimization—are not explored

## Related Work & Insights
- **vs. existing EDA automation tools** (e.g., Yosys padring tools): existing tools require designers to author detailed configuration tables, offer limited automation, and largely target digital SoCs without supporting AMS requirements. This work achieves automation at the level of genuine design intent comprehension through a natural language interface and domain knowledge reasoning
- **vs. other LLM for IC Design work** (e.g., ChipNeMo, RTL code generation): existing work focuses on digital design (RTL generation, netlist synthesis) and has not addressed the highly manual AMS I/O domain. This paper is the first to introduce LLM Agents into AMS IC design with tape-out-level validation
- **vs. direct LLM code generation**: ablation results show that direct SKILL code generation by LLMs completely fails (0% pass rate), confirming the necessity of the structured intent intermediate representation
- The agent architecture pattern of "LLM for high-level reasoning + deterministic tools for low-level execution" merits broader attention and is applicable wherever natural language intent must be translated into precise technical outputs
- The Intent Graph's design as a human–machine co-readable intermediate representation is transferable to other engineering automation domains
- The compact knowledge base design (~6k tokens, fully in-context) avoids RAG complexity and is a more practical solution when domain knowledge is limited in volume

## Rating
- Novelty: ⭐⭐⭐⭐ First application of an LLM Agent to AMS IC I/O design with tape-out validation; the problem formulation is novel, though the agent architecture itself is not particularly innovative
- Experimental Thoroughness: ⭐⭐⭐⭐ Ablation study is well-designed; a real 28nm tape-out case study provides strong validation; benchmark scale is limited (30 cases)
- Writing Quality: ⭐⭐⭐⭐ Problem motivation, method design, and experimental analysis are all clearly presented; figures and tables are of high quality
- Value: ⭐⭐⭐⭐ Represents an important advance in AMS IC design automation, though the application scope is relatively narrow and primarily relevant to IC design practitioners

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Beyond Accuracy: Unveiling Inefficiency Patterns in Tool-Integrated Reasoning](../../ACL2026/others/beyond_accuracy_unveiling_inefficiency_patterns_in_tool-integrated_reasoning.md)
- [\[AAAI 2026\] An Epistemic Perspective on Agent Awareness](an_epistemic_perspective_on_agent_awareness.md)
- [\[AAAI 2026\] Lost in Time? A Meta-Learning Framework for Time-Shift-Tolerant Physiological Signal Transformation](lost_in_time_a_meta-learning_framework_for_time-shift-tolerant_physiological_sig.md)
- [\[AAAI 2026\] A Graph-Theoretical Perspective on Law Design for Multiagent Systems](a_graph-theoretical_perspective_on_law_design_for_multiagent_systems.md)
- [\[AAAI 2026\] Agent-SAMA: State-Aware Mobile Assistant](agent-sama_state-aware_mobile_assistant.md)

</div>

<!-- RELATED:END -->
