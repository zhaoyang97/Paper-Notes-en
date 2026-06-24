---
title: >-
  [Paper Note] BookWorld: From Novels to Interactive Agent Societies for Story Creation
description: >-
  [ACL 2025][LLM Agent][Multi-agent systems] BookWorld is the first multi-agent social simulation system based on novels. It constructs interactive virtual worlds by extracting character data and worldview specifications from source books, allowing novel characters to act and interact autonomously to generate creative stories, outperforming previous story generation methods in 75.36% of pairwise comparisons.
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "Multi-agent systems"
  - "novel simulation"
  - "story generation"
  - "role-playing"
  - "virtual worlds"
date: 2026-05-08
content_hash: 376a0ff9ee924804
---

# BookWorld: From Novels to Interactive Agent Societies for Story Creation

**Conference**: ACL 2025  
**arXiv**: [2504.14538](https://arxiv.org/abs/2504.14538)  
**Code**: [https://bookworld2025.github.io/](https://bookworld2025.github.io/)  
**Area**: LLM Agent  
**Keywords**: Multi-agent systems, novel simulation, story generation, role-playing, virtual worlds

## TL;DR
BookWorld is the first multi-agent social simulation system based on novels. It constructs interactive virtual worlds by extracting character data and worldview specifications from source books, allowing novel characters to act and interact autonomously to generate creative stories, outperforming previous story generation methods in 75.36% of pairwise comparisons.

## Background & Motivation

**Background**: LLM-based multi-agent systems have demonstrated great potential in social simulation and collaborative tasks. Representative works include Generative Agents (a sandbox world of 25 agents), Project Sid (Minecraft civilization simulation with a thousand agents), and OASIS (million-scale social media simulation). In the field of story generation, prior works have also utilized multi-agent collaboration (director + editor + actors) to write stories.

**Limitations of Prior Work**: (1) Existing agent societies are constructed from scratch—agent personas are defined by short descriptions or demographic characteristics and lack depth; (2) Simulating existing fictional worlds (novels, films, and television) remains largely unexplored despite its immense practical value (e.g., interactive fiction, immersive gaming, fan fiction creation); (3) Existing story generation methods mostly adopt top-down approaches (outline-first, then expand), producing stories that lack surprise and tension, with insufficient creativity.

**Key Challenge**: Top-down writing methods offer strong control but lack creativity; bottom-up, character-driven narratives are creative but struggle to maintain consistency. How to generate creative stories while remaining faithful to the source work?

**Goal**: Build a complete system capable of extracting all necessary information (characters, maps, worldviews, etc.) from novels to automatically construct simulatable virtual worlds, allowing characters to act autonomously and generate stories.

**Key Insight**: Treat novel characters as autonomous agents with complete personalities, memories, and goals, allowing the story to "emerge" from character interactions rather than being planned top-down by an author/director agent.

**Core Idea**: Books -> Data Extraction -> Multi-character Agent Society Construction -> Autonomous Simulation -> Story Emergence, achieving "character-driven" creative narration.

## Method

### Overall Architecture
The complete workflow of BookWorld consists of three phases: (1) Data preparation—extracting character attributes, geographic maps, and worldview data from the source novel; (2) Simulation—initializing character agents and world agents, performing multi-scene interactive simulations, and enabling characters to act, chat, and explore autonomously in the virtual world; (3) Polishing—collecting simulation logs and using LLMs to rewrite them into coherent narratives in the style of a novel.

### Key Designs

1. **Role Agent Design**:

    - **Function**: Simulate the autonomous behavior and social interactions of novel characters.
    - **Mechanism**: Each role agent contains static attributes (gender, age, appearance, personality—extracted from the novel and remaining unchanged during simulation) and dynamic attributes (goals, status, memory—updated as the story progresses). Actions are categorized into active actions (driven by the character's goals, including character interaction, environmental interaction, and individual activities) and passive responses (reactions to others' actions). The agent is equipped with a RAG-based (retrieval-augmented generation) long-term memory module, continuously accumulating and retrieving relevant memories during the simulation. Before taking action, an agent first undergoes internal thoughts (analyzing the current state and goals), then decides on an action, and finally updates its memory and status.
    - **Design Motivation**: Compared to agents defined by short descriptions, characters extracted from full novels possess richer personality dimensions and behavioral patterns. Separating static and dynamic attributes ensures a balance between character consistency and character development.

2. **World Agent and Geospatial System**:

    - **Function**: Manage the overall simulation workflow and spatial constraints.
    - **Mechanism**: The world agent is responsible for scene scheduling (selecting participants for each scene based on character locations and historical interactions), environmental feedback (providing environmental change information), and NPC management (creating temporary memoryless agents for transiently appearing bystanders). The geospatial environment is modeled as a discrete graph—nodes represent locations and edges represent paths; characters moving between locations consume scene turns. This ensures that only characters in the same location at the same time can interact directly. The world agent also maintains the global state (e.g., weather, time progression, social events).
    - **Design Motivation**: Spatial constraint is central to narrative rationality. Without geographic limitations, characters would perform physically illogical "teleportations," breaking story consistency.

3. **Worldview Data Collection and Injection**:

    - **Function**: Ensure agent behaviors align with the rules and culture of the fictional world.
    - **Mechanism**: Three types of worldview data are systematically extracted from the source novel: social norms (e.g., laws in a magical world, racial relations), cultural backgrounds (e.g., festivals, customs, values), and terminology explanations (e.g., magic names, special items). This data is stored in a structured format and injected into the context via relevance retrieval during every agent decision. This ensures that agents not only know "who I am" but also "what this world is like."
    - **Design Motivation**: Many novels feature unique worldviews (such as Harry Potter's magic school). Character behaviors must make sense under these rules. Without injecting worldview data, agents might generate behaviors that violate the settings (e.g., a character in a magical world using modern technology to solve a problem).

### Loss & Training
This work is a system design paper and does not involve model training. All agents are implemented through prompt engineering based on off-the-shelf LLMs (e.g., GPT-4).

## Key Experimental Results

### Main Results

| Method | Faithfulness | Creativity | Text Quality | Overall Win Rate |
|------|--------|--------|---------|---------|
| BookWorld | 4.12 | 3.89 | 4.05 | **75.36%** |
| Single LLM Narrative | 3.68 | 3.21 | 3.92 | 12.8% |
| Multi-Agent Writing (Director+Editor) | 3.85 | 3.45 | 3.98 | 11.84% |

### Ablation Study

| Configuration | Faithfulness | Creativity | Overall Rating | Note |
|------|--------|--------|---------|------|
| Full BookWorld | 4.12 | 3.89 | 4.05 | Complete system |
| w/o Worldview Data | 3.52 | 3.75 | 3.61 | Faithfulness drops significantly |
| w/o Geospatial Constraints | 3.91 | 3.82 | 3.70 | Narrative logicality decreases |
| w/o Character Memory | 3.65 | 3.67 | 3.55 | Character behavioral consistency degrades |
| w/o Dynamic Attribute Updates | 3.88 | 3.41 | 3.68 | Creativity drops significantly—characters cannot develop |

### Key Findings
- BookWorld is evaluated as superior to baselines in 75.36% of the comparisons, with major advantages in faithfulness and creativity.
- Worldview data has the greatest impact on faithfulness (dropping by 0.60 when removed), proving the necessity of worldview injection.
- Dynamic attribute updates have the greatest impact on creativity (dropping by 0.48 when removed), illustrating that character development is a key source of creative storytelling.
- Geospatial constraints do not directly improve story quality scores, but they significantly reduce logical errors like "character teleportation."
- Interesting behavioral patterns emerge from character interactions—characters make choices based on personal goals that differ from the original book but remain plausible.

## Highlights & Insights
- Systematically integrates novel-based agent society construction for the first time, filling a gap in "fictional world simulation". This possesses academic value and significant commercial potential in fields like interactive storytelling, game design, and IP spin-off creation.
- The "character-driven" instead of "outline-driven" narrative approach grants the narrative emergence and elements of surprise—characters may make choices unexpected by authors (or even system designers), which is precisely the hallmark of a good story.
- The methodology for worldview data extraction and injection is highly generalizable and can be extended to other virtual world creation tasks like game worlds (e.g., DnD settings) and historical scenarios (e.g., a specific dynasty).

## Limitations & Future Work
- The data extraction phase is heavily dependent on the reading comprehension capabilities of LLMs, which may suffer from information omissions in extremely long novels (100k+ words).
- The simulation scale is constrained by the inference cost of LLMs—every action step of each character requires an LLM call.
- The length and complexity of generated stories are limited by the number of simulation rounds, making it difficult to generate genuinely long-form narratives.
- The evaluation of character "faithfulness" is subjective; different readers may disagree on "whether a character complies with the original work."
- Future work could integrate visual generative models to generate scene images for the simulation process, realizing multimodal storytelling.

## Related Work & Insights
- **vs Generative Agents (Park et al.)**: Generative Agents use brief persona descriptions to create agents, while BookWorld extracts characters from full novels, far exceeding them in personality depth and behavioral richness.
- **vs Multi-Agent Writing Systems (Han et al.)**: Traditional multi-agent writing is top-down (director plans -> characters execute), whereas BookWorld is bottom-up (characters act -> story emerges), yielding higher creativity.
- **vs RecurrentGPT/LongStory**: These methods use a single LLM to generate long text sequentially, lacking character autonomy and world consistency.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First complete framework from novel to agent society; both problem definition and solutions show high originality.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Combines quantitative and qualitative evaluations with solid ablation analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear system design descriptions accompanied by vivid examples.
- **Value**: ⭐⭐⭐⭐⭐ Possesses both academic innovation and practical application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Generative AI Agents for Controllable and Protected Content Creation](../../NeurIPS2025/llm_agent/generative_ai_agents_for_controllable_and_protected_content_creation.md)
- [\[NeurIPS 2025\] ShapeCraft: LLM Agents for Structured, Textured and Interactive 3D Modeling](../../NeurIPS2025/llm_agent/shapecraft_llm_agents_for_structured_textured_and_interactive_3d_modeling.md)
- [\[ACL 2025\] Gödel Agent: A Self-Referential Agent Framework for Recursive Self-Improvement](gödel_agent_a_self-referential_agent_framework_for_recursive_self-improvement.md)
- [\[ICLR 2026\] VitaBench: Benchmarking LLM Agents with Versatile Interactive Tasks in Real-world Applications](../../ICLR2026/llm_agent/vitabench_benchmarking_llm_agents_with_versatile_interactive_tasks_in_real-world.md)
- [\[ACL 2025\] LLM Agents Making Agent Tools](llm_agents_making_agent_tools.md)

</div>

<!-- RELATED:END -->
