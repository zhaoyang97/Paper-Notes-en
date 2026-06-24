---
title: >-
  [Paper Note] LLM Agents Making Agent Tools
description: >-
  [ACL 2025][LLM Agent][Tool generation] This paper proposes ToolMaker, an autonomous agent framework that transforms GitHub code repositories into LLM-compatible tools. Given a repository URL and a task description, it automatically installs dependencies, generates invocation code, and debugs through a closed-loop self-correction mechanism. It successfully implements 80% of the tasks on a new benchmark spanning 15 complex tasks across various domains…
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "Tool generation"
  - "Autonomous agents"
  - "Code repository-to-tool"
  - "Closed-loop debugging"
  - "Scientific workflows"
date: 2026-05-08
content_hash: cda05a04a05e5139
---

# LLM Agents Making Agent Tools

**Conference**: ACL 2025  
**arXiv**: [2502.11705](https://arxiv.org/abs/2502.11705)  
**Code**: [GitHub](https://github.com/KatherLab/ToolMaker)  
**Area**: LLM Agent  
**Keywords**: Tool generation, Autonomous agents, Code repository-to-tool, Closed-loop debugging, Scientific workflows

## TL;DR

This paper proposes ToolMaker, an autonomous agent framework that transforms GitHub code repositories into LLM-compatible tools. Given a repository URL and a task description, it automatically installs dependencies, generates invocation code, and debugs through a closed-loop self-correction mechanism. It successfully implements 80% of the tasks on a new benchmark spanning 15 complex tasks across various domains, significantly outperforming existing software engineering agents.

## Background & Motivation

**Background**: Tool use has transformed large language models (LLMs) into powerful agents capable of executing complex multi-step tasks by dynamically calling external software components. However, a fundamental limitation is that these tools must be pre-implemented by human developers. The current LLM agent ecosystem relies on hand-crafted tool libraries (such as API wrappers, function definitions, etc.), where each tool requires a human to comprehend the source codebase and write the interface wrapper.

**Limitations of Prior Work**: In fields such as life sciences and medicine, there exists a vast number of highly specialized computational tools, usually published on GitHub in the form of academic papers paired with code repositories. These tools are numerous and continuously updated, making the manual creation of LLM-compatible interfaces for each tool extremely costly and hard to scale. Existing software engineering agents (such as SWE-Agent), although capable of handling code-related tasks, are not explicitly designed to transform complex scientific codebases into callable tools.

**Key Challenge**: Open-source code repositories accompanying scientific papers represent a "gold mine of tools." However, these codebases are often poorly documented, have complex dependencies, and lack unified interfaces, making direct invocation by LLMs almost impossible. Manual encapsulation is also not scalable.

**Goal**: To build a fully automated agent framework capable of transforming any GitHub repository into a standardized tool that can be directly invoked by LLMs.

**Key Insight**: The authors decompose the "repository-to-tool" process into three stages: installation (Install), creation (Create), and execution (Run). Each stage is autonomously operated by LLM agents within isolated Docker environments, resolving errors through a closed-loop self-correction mechanism.

**Core Idea**: Enabling LLM agents to make their own tools—given a GitHub URL and a brief task description, ToolMaker fully automates dependency installation, code comprehension, interface code generation, and debugging.

## Method

### Overall Architecture

The workflow of ToolMaker consists of three stages: The **Install** stage clones the GitHub repository into a Docker container and installs all dependencies. The **Create** stage involves the LLM agent reading the repository code, understanding its functionality, and generating Python invocation code adhering to a standardized interface. The **Run** stage executes the generated tool code in a new Docker container to process specific tasks. Each stage embeds a closed-loop self-correction mechanism—if an execution error occurs, the LLM agent reads the error message, analyzes the root cause, and automatically modifies the code to retry.

### Key Designs

1. **Three-stage Tool Manufacturing Pipeline (Install → Create → Run)**:

    - **Function**: Decomposing the complex "repository-to-usable tool" process into manageable sub-tasks.
    - **Mechanism**: The **Install** stage executes commands like `git clone` and `pip install` in Docker, with the LLM agent inferring the correct installation steps based on the repository's README and configuration files. The **Create** stage is the core—the LLM agent browses the repository structure, reads key code files, and then generates an `implementation.py` file that implements a standardized function interface (accepting input file paths and arguments, returning outputs). The **Run** stage executes the generated tool code in a clean Docker container.
    - **Design Motivation**: Separating "environment setup" from "code comprehension and encapsulation" prevents environmental and code-understanding issues from interfering with each other. Docker isolation ensures security and reproducibility.

2. **Closed-loop Self-correction Debugging Mechanism**:

    - **Function**: Automatically diagnosing and fixing errors in the tool code.
    - **Mechanism**: In the Create stage, immediately after code generation, a test execution is triggered in Docker. If the execution fails, the complete error message (stderr/traceback) is fed back to the LLM agent, which analyzes the error and modifies the code. This "execute-check error-fix-retry" loop is repeated up to $N$ times (configurable). Critically, the agent can access the repository source code during each iteration to assist with debugging.
    - **Design Motivation**: The complexity of scientific codebases makes generating correct code on the first attempt almost impossible—there are potential API changes, undocumented parameters, and implicit dependencies. Closed-loop debugging mimics the "trial-and-error" process of human developers.

3. **Standardized Tool Interface Design**:

    - **Function**: Ensuring the generated tools can be directly called by any LLM agent.
    - **Mechanism**: Defining a unified Python function interface template that all generated tools must implement (e.g., `def run(input_path: str, output_path: str, **kwargs) -> dict`). The goal of the LLM agent in the Create stage is to wrap the repository's functionality into this standard interface. Tool metadata (description, parameter explanations) is also automatically generated.
    - **Design Motivation**: Standardized interfaces are the foundation of tool composability and reusability. Without a unified interface, even if the tool code is successfully generated, the LLM agent would not know how to invoke it.

### Loss & Training

ToolMaker does not involve model training and is fully based on the zero-shot reasoning capabilities of existing LLMs (such as GPT-4). The evaluation metrics are task implementation correctness rate (verified through unit tests) and robustness (the pass rate across multiple test cases).

## Key Experimental Results

### Main Results (TM-Bench Benchmark)

| Method | Task Accuracy | Unit Test Pass Rate | Domain Coverage |
|------|-----------|--------------|---------|
| **ToolMaker** | **80% (12/15)** | ~85% (100+ tests) | Pathology, Radiology, Genomics, etc. |
| SWE-Agent | ~40% | ~45% | Same as above |
| OpenHands (CodeAct) | ~33% | ~38% | Same as above |
| Human Baseline | ~93% | ~95% | Same as above |

### Ablation Study

| Configuration | Task Accuracy | Description |
|------|-----------|------|
| ToolMaker (Full) | 80% | Three-stage + closed-loop debugging |
| No closed-loop debugging (single-generation) | ~47% | Significant performance drop after removing self-correction |
| Reduced debugging rounds (max=2) | ~60% | Fewer debugging rounds leave some tasks unresolved |
| No Docker isolation | N/A (security risks) | Not tested |
| Swapping GPT-4 with GPT-3.5 | ~53% | Insufficient model capacity leads to degraded code quality |

### Key Findings

- **Closed-loop debugging is the core contribution**: Removing the self-correction mechanism drops the accuracy from 80% to about 47%, indicating that generating correct tool code in one shot is extremely difficult, making iterative debugging indispensable.
- **Significantly outperforming general software engineering agents**: ToolMaker's 80% accuracy far exceeds SWE-Agent (~40%) and OpenHands (~33%), demonstrating that a specially designed tool manufacturing pipeline is superior to general code generation approaches.
- **Failure case analysis**: The 3 failed tasks are primarily due to bugs within the repository code itself or the requirement of unconventional GPU environment configurations, which exceed the capabilities of the LLM agent.
- **Significant gap between GPT-4 and GPT-3.5**: Tool manufacturing demands strong code comprehension and generation capabilities; currently, only the strongest LLMs are capable of this task.

## Highlights & Insights

- **The meta-cognitive concept of "LLMs making tools for LLMs"**: This represents an elegant recursion—allowing AI agents to expand their own toolsets. With the exponential growth in the number of code repositories, this automation is the only scalable solution. This concept can be generalized to other scenarios requiring automated interface wrapping.
- **Secure design with Docker isolation**: Executing LLM-generated installation and runtime commands on untrusted code poses significant security risks. ToolMaker uses Docker containers to isolate each operational phase, ensuring that host environments are not compromised even if malicious code is generated. This serves as an excellent paradigm for agent security practices.
- **Orientation toward scientific workflows**: The paper focuses on tool automation in the scientific computing domain, which is a direction with high real-world demand—researchers frequently need to run others' code but spend most of their time on environment configuration and interface adaptation.

## Limitations & Future Work

- The benchmark size is relatively small (only 15 tasks). Although each task has multiple unit tests, domain and complexity coverage is still limited.
- Highly dependent on the capabilities of GPT-4; the performance of open-source models remains to be validated.
- Only handles Python repositories; support for other languages commonly used in research, such as R and MATLAB, is not yet explored.
- Docker environment GPU support and special hardware requirements might restrict the automatic installation of certain scientific tools.
- Generated tool code lacks formal verification, posing a risk of subtle semantic errors (e.g., passing unit tests but failing under edge cases).

## Related Work & Insights

- **vs SWE-Agent (Yang et al.)**: SWE-Agent is oriented toward general GitHub issue resolution, whereas ToolMaker focuses on packing repositories into callable tools. This difference in task definition makes ToolMaker's phased design more targeted.
- **vs LATM (Cai et al., "Large Language Models as Tool Makers")**: LATM has LLMs generate Python functions as tools, but starts from simple natural language descriptions. ToolMaker starts from complex code repositories, presenting a greater challenge that is closer to real-world scientific scenarios.
- **vs Multi-agent frameworks like AutoGen/MetaGPT**: These frameworks focus on collaboration between agents, whereas ToolMaker focuses on a single agent's tool manufacturing capacity. The two are complementary—tools generated by ToolMaker can be integrated and used within multi-agent frameworks.

## Rating

- Novelty: ⭐⭐⭐⭐ The concept of "agents making tools for agents" is novel and forward-looking, and the phased + closed-loop debugging design is highly reasonable.
- Experimental Thoroughness: ⭐⭐⭐ The benchmark size is relatively small, and baselines are mostly general SE agents rather than dedicated tool generation methods.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, intuitive architectural illustrations, and solid experimental analysis.
- Value: ⭐⭐⭐⭐ Strongly advances autonomous scientific workflows and the agent ecosystem.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Agentic Reasoning: A Streamlined Framework for Enhancing LLM Reasoning with Agentic Tools](agentic_reasoning_tools.md)
- [\[NeurIPS 2025\] Distilling LLM Agent into Small Models with Retrieval and Code Tools](../../NeurIPS2025/llm_agent/distilling_llm_agent_into_small_models_with_retrieval_and_co.md)
- [\[ACL 2025\] R2D2: Remembering, Replaying and Dynamic Decision Making with a Reflective Agentic Memory](r2d2_reflective_agentic_memory.md)
- [\[ACL 2025\] Agents Under Siege: Breaking Pragmatic Multi-Agent LLM Systems with Optimized Prompt Attacks](agents_under_siege_breaking_pragmatic_multi-agent_llm_systems_with_optimized_pro.md)
- [\[ICLR 2026\] WALT: Web Agents that Learn Tools](../../ICLR2026/llm_agent/walt_web_agents_that_learn_tools.md)

</div>

<!-- RELATED:END -->
