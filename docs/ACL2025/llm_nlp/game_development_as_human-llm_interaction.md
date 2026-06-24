---
title: >-
  [Paper Note] Game Development as Human-LLM Interaction
description: >-
  [ACL 2025][LLM (Other)][Game Development] This paper proposes Chat Game Engine (ChatGE), an LLM-based conversational game engine that enables users to develop customized games through natural language interaction without programming knowledge. It designs a data synthesis pipeline and a three-stage progressive training strategy to transform a conversational model into a game engine.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Game Development"
  - "Human-Computer Interaction"
  - "Code Generation"
  - "Dialogue Systems"
  - "Progressive Training"
date: 2026-05-08
content_hash: e97b2c0537369bed
---

# Game Development as Human-LLM Interaction

**Conference**: ACL 2025  
**arXiv**: [2408.09386](https://arxiv.org/abs/2408.09386)  
**Code**: None  
**Area**: LLM Applications  
**Keywords**: Game Development, Human-Computer Interaction, Code Generation, Dialogue Systems, Progressive Training

## TL;DR

This paper proposes Chat Game Engine (ChatGE), an LLM-based conversational game engine that enables users to develop customized games through natural language interaction without programming knowledge. It designs a data synthesis pipeline and a three-stage progressive training strategy to transform a conversational model into a game engine.

## Background & Motivation

**Background**: Game development typically requires professional game engines (e.g., Unity, Unreal) and complex programming languages, posing a high barrier to entry for many game enthusiasts. Although LLMs have made significant progress in code generation (e.g., Copilot) and dialogue systems, leveraging LLMs for the entire interactive game development process remains under-explored.

**Limitations of Prior Work**: Existing LLM code generation tools (e.g., ChatGPT, Copilot) can assist in writing code snippets but fail to comprehend the complete pipeline of game development—from rule design to scripting and final code implementation. Users still need to decompose game requirements into coding tasks themselves, which still demands basic programming and game design knowledge.

**Key Challenge**: Game development is fundamentally a complex process of "idea → design → implementation", involving multiple layers like game rules, interactive logic, and code generation. Existing LLM tools only handle the "code generation" step and lack a game script as an intermediate representation to connect the idea with the code. Without this middle layer, LLMs cannot truly understand what kind of game the user wants to build.

**Goal**: To build ChatGE, a complete conversational game engine that allows users to develop a runnable game from scratch through multi-turn natural language dialogues.

**Key Insight**: Model the game development process as a three-stage pipeline: "script generation → code generation → user interaction". Introducing a game script as an intermediate representation supports the user's natural language descriptions while guiding the code generation, bridging the gap between creativity and implementation.

**Core Idea**: Enable the LLM to sequentially complete three processes in each turn of the conversation: configure game script segments, generate corresponding code snippets, and interact with the user (providing guidance and feedback). This transforms a general conversational LLM into a specialized game engine through progressive training.

## Method

### Overall Architecture

The input to ChatGE is the user's natural language description (e.g., "I want to make a Texas Hold'em game with two players, and the dealer rotates in order"), and the output consists of runnable game code and interactive guidance. In each turn of the conversation, the LLM executes three sequential processes: $P_{script}$ configures game script segments based on user input, $P_{code}$ generates the corresponding code based on the script segments, and $P_{utter}$ generates user-facing responses (containing questions guiding the next step or feedback on actions).

### Key Designs

1. **Game Script Intermediate Representation (Game Script)**:

    - **Function**: Bridging the semantic gap between the user's natural language descriptions and executable code.
    - **Mechanism**: Design a structured game script format describing game roles, rules, state transitions, action spaces, and other elements. In each dialogue turn, the $P_{script}$ process updates the corresponding parts of the script based on user input. The script defines the logical structure of the game without involving implementation details of specific programming languages. For instance, the script describes the "dealing phase: each player receives 2 hole cards" instead of Python code. $P_{code}$ then translates this script into executable code. This two-step decomposition reduces the difficulty of single-step generation.
    - **Design Motivation**: Direct jumps from natural language to code are too steep and error-prone. Introducing an intermediate representation decouples "understanding requirements" from "writing code", making each step more controllable. This also facilitates easier debugging—if the code is incorrect, one can check whether the user's intent was correctly understood at the script level.

2. **LLM-Based Data Synthesis Pipeline**:

    - **Function**: Synthesizing large-scale training data from a small number of human-written seed data.
    - **Mechanism**: A small number of complete game development dialogue samples were manually written as seed data, covering various poker game variants. Then, LLMs were utilized to synthesize more game script-code pairs and dialogue interactions based on the seed data. The synthesis process includes: (1) mutating game rules starting from seed game scripts (e.g., modifying player counts, changing scoring rules); (2) generating corresponding code for the new game rules; (3) simulating multi-turn user-system dialogues. The synthesized data underwent manual quality reviews and code executability checks.
    - **Design Motivation**: Complete game development dialogue data is extremely difficult to collect at scale; synthesis allows the expansion of training data while ensuring quality.

3. **Three-Stage Progressive Training Strategy**:

    - **Function**: Progressively transforming a conversational LLM into a game engine.
    - **Mechanism**: The first stage, Script Training, only trains the $P_{script}$ capability, teaching the model to convert natural language descriptions into structured game scripts. The second stage, Code Training, continues to train the $P_{code}$ capability on top of the first stage, enabling the model to generate correct code based on the script. The third stage, Interaction Training, jointly trains $P_{script}$, $P_{code}$, and $P_{utter}$, teaching the model to coordinate the entire game development process in multi-turn dialogues. The complexity of the data and objectives increases incrementally with each stage.
    - **Design Motivation**: Training all capabilities at once is too difficult and unstable. Progressive training allows the model to master fundamental capabilities first and then layer them incrementally, similar to the human learning process of "first learning the rules, then learning the implementation, and finally learning the interaction."

### Loss & Training

All three stages utilize the standard next-token prediction loss (cross-entropy). However, different portions of the training data are annotated with loss masks: in the first stage, the loss is only calculated on the script portion; in the second stage, the loss is calculated on the script + code portions; in the third stage, the loss is calculated on the entire output (script + code + interactive response).

## Key Experimental Results

### Main Results

Using poker game development as a case study, the evaluation is conducted from two aspects: interaction quality and code correctness:

Interaction Quality Evaluation (Human Rating, 1-5 scale):

| Model | Guidance | Feedback Accuracy | Dialogue Coherence | Overall |
|------|-------|-----------|-----------|------|
| ChatGPT | 3.2 | 3.5 | 3.8 | 3.5 |
| GPT-4 | 3.8 | 4.1 | 4.2 | 4.0 |
| ChatGE (7B) | 4.1 | 3.9 | 4.0 | 4.0 |
| ChatGE (13B) | **4.3** | **4.2** | **4.3** | **4.3** |

Code Correctness Evaluation:

| Model | Syntax Accuracy | Logic Accuracy | Fully Runnable Rate | Rule Compliance Rate |
|------|-----------|-----------|------------|-----------|
| ChatGPT | 78.5% | 52.3% | 35.2% | 41.6% |
| GPT-4 | 89.2% | 68.7% | 55.8% | 62.3% |
| ChatGE (7B) | 91.5% | 74.2% | 62.1% | 71.8% |
| ChatGE (13B) | **93.8%** | **78.5%** | **68.3%** | **76.2%** |

### Ablation Study

| Configuration | Fully Runnable Rate | Rule Compliance Rate | Description |
|------|------------|-----------|------|
| Full ChatGE | 68.3% | 76.2% | Full three-stage training |
| w/o Progressive Training | 51.7% | 58.4% | Jointly train three processes at once, drops by 16.6% |
| w/o Game Script | 45.2% | 49.8% | Remove the intermediate script representation, direct dialogue → code |
| w/o $P_{utter}$ | 62.8% | 72.1% | No interactive response generated, only script + code |
| Using Only Human Data | 42.5% | 48.3% | Using only human-written data (no synthetic expansion) |

### Key Findings

- **Game script is vital**: Removing the intermediate script representation leads to a sharp decline in code correctness (68.3% → 45.2%), proving that intermediate representations are indispensable for bridging the semantic gap between natural language and code.
- **Progressive training significantly outperforms joint training**: Training all capabilities at once drops performance across the board; step-by-step progressive learning yields much better results.
- **ChatGE 7B outperforms GPT-4**: On this specialized game development task, the targetedly trained 7B model surpasses the general GPT-4, demonstrating the value of domain-specific training.
- **The data synthesis pipeline is crucial**: The performance of the model trained with only human data is only 62% of the full model's performance.
- **Performance gaps widen on games with complex rules**, indicating that rule-based reasoning capability remains a bottleneck.

## Highlights & Insights

- **Game script as an intermediate representation**: This is a clever design. Inserting a structured intermediate layer into the complex mapping from "natural language to code" significantly reduces the difficulty of each step. This "divide-and-conquer" approach can be transferred to other tasks requiring structured outputs from high-level descriptions, such as generating database schemas from requirement documents or generating front-end code from design drafts.
- **Effectiveness of progressive training**: The three-stage training strategy is conceptually aligned with curriculum learning, but with clever stage partitioning tailored for the specific task of game development. This suggests that when training multi-capability models, a divide-and-conquer strategy followed by integration may be more effective than training everything at once.
- **Task specialization vs. general models**: The fact that the 7B ChatGE outperforms the general GPT-4 highlights once again the immense potential of training small, specialized models for vertical domains.

## Limitations & Future Work

- Currently, poker games are used as the only case study; the generalization of the method to more complex game genres (e.g., RPGs, strategy games, action games) has not been validated.
- The format of the game script requires manual design, and extending to new game genres necessitates redesigning the script templates.
- Code generation is currently limited to specific game frameworks/programming languages. Supporting multiple engines (e.g., Unity, Godot) could be considered in the future.
- Evaluation of interaction quality heavily relies on human ratings, lacking automated metrics.
- Although data synthesis is effective, its diversity is still constrained by the game genres present in the seed data.
- ChatGE could be extended into an end-to-end game development IDE in the future, integrating debugging, testing, and deployment functionalities.

## Related Work & Insights

- **vs. Direct Code Generation by ChatGPT/GPT-4**: General LLMs lack an understanding of the complete game development workflow, resulting in low code accuracy. ChatGE significantly enhances the usability of game code through the intermediate script representation and domain-specific training.
- **vs. Voyager (LLM Game Agent)**: Voyager enables LLMs to play games (executing in-game actions), whereas ChatGE makes the LLM a game development tool (generating game code). Both focus on different aspects of Game-LLM interaction but leverage the code generation and reasoning capabilities of LLMs.
- **vs. MetaGPT (Multi-Agent Software Development)**: MetaGPT uses multiple LLM agents to simulate a software development team, while ChatGE employs a single model to complete all tasks by introducing a structured intermediate representation. ChatGE's script layer design can serve as a reference for the "design document" role in MetaGPT.

## Rating

- Novelty: ⭐⭐⭐⭐ Modeling game development as conversational human-computer interaction is an interesting problem definition, and the design of the intermediate script representation possesses general applicability.
- Experimental Thoroughness: ⭐⭐⭐ Only poker games are used as a case study, presenting insufficient generalization verification; the scale of human evaluation is limited.
- Writing Quality: ⭐⭐⭐⭐ The problem definition is clear, and the three-stage process is described well, though some details (such as the script format) are insufficiently explained.
- Value: ⭐⭐⭐⭐ A promising research direction is presented, and the combination of intermediate representations and progressive training holds reference value for tasks in other domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] AXIS: Efficient Human-Agent-Computer Interaction with API-First LLM-Based Agents](axis_efficient_human-agent-computer_interaction_with_api-first_llm-based_agents.md)
- [\[ACL 2025\] Enhancing Conversational Agents with Theory of Mind: Aligning Beliefs, Desires, and Intentions for Human-Like Interaction](enhancing_conversational_agents_with_theory_of_mind_aligning_beliefs_desires_and.md)
- [\[ACL 2025\] Achieving Certification-by-Design Through Model-Driven Development](achieving_certification-by-design_through_model-driven_development.md)
- [\[ACL 2025\] ComfyUI-Copilot: An Intelligent Assistant for Automated Workflow Development](comfyui-copilot_an_intelligent_assistant_for_automated_workflow_development.md)
- [\[ACL 2025\] Leveraging Human Production-Interpretation Asymmetries to Test LLM Cognitive Plausibility](leveraging_human_production-interpretation_asymmetries_to_test_llm_cognitive_pla.md)

</div>

<!-- RELATED:END -->
