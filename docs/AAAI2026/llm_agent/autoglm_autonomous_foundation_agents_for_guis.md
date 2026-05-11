---
title: >-
  [Paper Note] AutoGLM: Autonomous Foundation Agents for GUIs
description: >-
  [AAAI 2026][LLM Agent][GUI Agent] AutoGLM builds a GUI foundation agent for web browsers and Android devices on top of ChatGLM. By introducing an intermediate interface design that decouples planning from grounding…
tags:
  - "AAAI 2026"
  - "LLM Agent"
  - "GUI Agent"
  - "Reinforcement Learning"
  - "Intermediate Interface Design"
  - "Self-Evolving Curriculum Learning"
  - "Foundation Agent"
date: 2026-05-08
content_hash: e190459c1fbb7911
---

# AutoGLM: Autonomous Foundation Agents for GUIs

**Conference**: AAAI 2026
**arXiv**: [2411.00820](https://arxiv.org/abs/2411.00820)
**Code**: [https://github.com/THUDM/AutoGLM](https://github.com/THUDM/AutoGLM)
**Area**: Agent
**Keywords**: GUI Agent, Reinforcement Learning, Intermediate Interface Design, Self-Evolving Curriculum Learning, Foundation Agent

## TL;DR
AutoGLM builds a GUI foundation agent for web browsers and Android devices on top of ChatGLM. By introducing an intermediate interface design that decouples planning from grounding, and proposing a self-evolving online curriculum reinforcement learning framework, the system achieves a 55.2% success rate on VAB-WebArena-Lite, substantially surpassing GPT-4o's 18.2%.

## Background & Motivation
1. **Background**: Foundation models (LLMs/LMMs) excel at knowledge acquisition and language understanding, yet their decision-making capability in dynamic, real-world environments remains insufficient. GUI Agents, as a bridge between foundation models and digital devices, have attracted growing attention but remain in an early stage of development.
2. **Limitations of Prior Work**: Existing GUI Agent approaches face three core challenges: (1) internet-scale pretraining data lacks dynamic, decision-relevant knowledge, making it difficult for models to learn environment interaction; (2) high-quality expert trajectory data is extremely scarce and costly to collect; (3) end-to-end methods couple planning and grounding, making it difficult to optimize either independently.
3. **Key Challenge**: Planning and grounding impose fundamentally different requirements on the model—planning demands flexibility and error recovery, while grounding demands precise action execution. Joint training of the two interferes with each, and pure behavior cloning prevents the agent from learning to recover from mistakes.
4. **Goal**: (1) How to design an appropriate interface to decouple planning from grounding? (2) How to continuously improve agent capability via RL under extreme data scarcity? (3) How to build a foundation agent system deployable to real users?
5. **Key Insight**: The authors observe that existing LLMs/LMMs are substantially better at planning than grounding—most errors stem from incorrect element identification (e.g., clicking the wrong coordinates) rather than incorrect planning direction. This observation motivates separating the two.
6. **Core Idea**: Decouple and independently optimize GUI Agent planning and grounding via intermediate interface design, then enable autonomous learning and self-evolution in real environments through a self-evolving online curriculum reinforcement learning framework.

## Method

### Overall Architecture
The AutoGLM pipeline consists of three major components: (1) **Intermediate Interface Design**: decomposing end-to-end agent behavior into a Planner (generating semantic action descriptions) and a Grounder (mapping descriptions to precise coordinates); (2) **Self-Evolving Online Curriculum RL (WebRL)**: performing online RL via an actor-critic framework to address task data scarcity and policy distribution shift; (3) **Deployment Infrastructure**: constructing practical, user-deliverable systems for both web browser and Android scenarios.

### Key Designs

1. **Intermediate Interface Design**:
    - **Function**: Replaces the traditional end-to-end action format `do(action="Click", element_coordinates=[823,684])` with a decoupled format—the Planner generates `do(action="Click", element_description="the 'Submit' button on the bottom right")`, and the Grounder maps the description to coordinates.
    - **Mechanism**: The Planner focuses solely on *what to do* (semantic level), while the Grounder focuses solely on *where to do it* (visual localization level), enabling independent training and optimization. Grounder training data can be automatically constructed in large quantities from unsupervised environment observations, making it far more accessible than end-to-end expert trajectories.
    - **Design Motivation**: Experiments (Table 1) show that this decoupling alone raises GPT-4o's success rate on VAB-WebArena-Lite from 18.2% to 27.3% (+9.1%), and GPT-4-vision-preview from 18.8% to 36.4% (+17.6%), confirming that grounding errors are the primary performance bottleneck.

2. **Self-Evolving Online Curriculum RL (WebRL)**:
    - **Function**: Starting from only ~1,000 behavior cloning examples, WebRL continuously improves Planner capability via online RL, addressing the twin challenges of data scarcity and policy distribution shift.
    - **Mechanism**: (1) **Task data self-evolution**: after initializing the model to 22.4% success rate with ~1,000 BC samples, failed task instructions encountered during online roll-outs are mutated (either increased or decreased in difficulty) and filtered by a critic for use in the next training round; (2) **KL-constrained policy updates**: KL divergence constraints are introduced to stabilize policy updates and prevent distribution shift during curriculum learning; (3) **Actor confidence-filtered experience replay**: only experiences for which the actor has high confidence are replayed, improving training efficiency.
    - **Design Motivation**: Conventional BC can only imitate expert step-by-step behavior without understanding the task goal, and cannot learn error recovery. Online RL allows learning from failures, but faces challenges of low environment simulation efficiency and insufficient sampling diversity; progressive difficulty increase through curriculum learning alleviates these issues.

3. **Reward Modeling**:
    - **Function**: Constructs a general reward model to provide supervision signals for online RL, rather than relying on task-specific rule-based reward functions.
    - **Mechanism**: Distinguishes between outcome-supervised ORM and process-supervised PRM, providing different granularities of evaluative feedback for open-world general tasks.
    - **Design Motivation**: Since the foundation agent aims to complete a broad range of real-world tasks, task-specific reward functions cannot cover all scenarios.

### Loss & Training
Training follows a progressive curriculum: pretraining to introduce weak supervision decision signals → high-resolution visual SoM prompting to enhance perception → BC initialization of basic capabilities → online curriculum RL (WebRL) to iteratively improve planning ability. The overall pipeline forms a "weak-to-strong" progressive enhancement process.

### Deployment Architecture
- **Web**: Deployed externally via a Chrome/Edge browser extension (Zhipu Qingyan), with the Planner serving as a backend service and the Grounder performing element localization on the browser side.
- **Android**: Autonomous control of physical devices is achieved via AccessibilityService rather than traditional Android Virtual Devices (AVDs), closely resembling real user scenarios.
- **Retry Mechanism**: When the initial execution fails, AutoGLM can automatically detect the failure state and initiate a retry; this error recovery capability is a direct product of RL training.

## Key Experimental Results

### Main Results

| Benchmark | Metric | AutoGLM | GPT-4o | Claude-3.5-Sonnet | Gain |
|-----------|--------|---------|--------|-------------------|------|
| VAB-WebArena-Lite | Success Rate | 55.2% | 18.2% | - | +37.0% |
| VAB-WebArena-Lite (retry) | Success Rate | 59.1% | - | - | +3.9% |
| OpenTable Eval | Success Rate | 96.2% | 62.6% | - | +33.6% |
| AndroidLab (VAB-Mobile) | Success Rate | 36.2% | 31.2% | 29.0% | +5.0% |
| Chinese APP (Human Eval) | Success Rate | 89.7% | - | - | - |

### Ablation Study (Effect of Intermediate Interface Design)

| Configuration | GPT-4o (text) | GPT-4o (visual) | GPT-4-vision (visual) |
|---------------|---------------|-----------------|----------------------|
| End-to-End Agent | 14.3% | 18.2% | 18.8% |
| + Intermediate Interface | 18.1% (+3.8%) | 27.3% (+9.1%) | 36.4% (+17.6%) |

### Key Findings
- The intermediate interface design yields larger gains for vision-input models (GPT-4-vision +17.6%), confirming that visual grounding is the primary bottleneck for current GUI Agents.
- The self-evolving online curriculum RL advances the model from a starting point of 22.4% with only 1,000 BC samples to a final 55.2%, validating the enormous potential of RL under data-scarce conditions.
- KL constraints and confidence-filtered experience replay are critical components for preventing training collapse; neither can be omitted.
- In real Chinese APP scenarios, AutoGLM achieves an 89.7% success rate; even tasks that are not fully completed are largely partially accomplished, retaining practical utility.
- Allowing a retry raises the success rate from 55.2% to 59.1%, confirming the effectiveness of the error recovery mechanism.
- The OpenTable evaluation uses real website interactions rather than simulated environments; a 96.2% success rate strongly demonstrates AutoGLM's reliability in practical settings.
- Agent Q's MCTS-based approach achieves only 81.7% on OpenTable, indicating that the combination of curriculum RL and intermediate interface outperforms pure search strategies in practice.

## Highlights & Insights
- **Elegance and effectiveness of planning–grounding decoupling**: Simply changing the action representation format yields substantial gains—an extremely concise yet impactful design. The core insight that "different capabilities require different optimization strategies" is transferable to other composite tasks.
- **Data self-evolution mechanism**: Under extreme data scarcity (only 1,000 samples), instruction mutation during online roll-outs achieves automatic data augmentation, cleverly resolving the cold-start problem of insufficient user task instructions.
- **Deployment-oriented design philosophy**: The paper not only targets benchmark performance but also deploys the system as a usable browser extension and Android application for real users—a research-to-product feedback loop rarely seen in academic publications.

## Limitations & Future Work
- **Limited multimodal perception**: The system currently relies primarily on SoM prompting and text descriptions; its ability to handle complex visual layouts (e.g., overlapping elements, dynamically loaded pages) may be insufficient.
- **Training confined to specific benchmark environments**: WebRL is trained primarily in the WebArena environment; generalization to other web environments or entirely new applications remains uncertain.
- **Limited evaluation coverage**: The Android evaluation covers only 7 commonly used Chinese apps; performance on niche or English-language applications is unknown.
- **Insufficient attention to safety and privacy**: The paper does not thoroughly discuss safety risks associated with autonomous agents operating real devices, such as financial losses or privacy leakage caused by erroneous actions.
- **Reproducibility of RL training**: Online curriculum RL involves extensive environment interaction and self-evolving instruction generation; the high stochasticity of the training process may make it difficult for other teams to precisely reproduce the reported results.
- **Handling of long-horizon tasks**: Most tasks demonstrated in the paper are completed within 10–20 steps; performance on complex tasks requiring dozens or even hundreds of steps remains unknown.

## Related Work & Insights
- **vs. Agent Q**: Agent Q also applies RL to train a web agent and evaluates on OpenTable, but AutoGLM achieves superior results through intermediate interface design and self-evolving curriculum RL (96.2% vs. 81.7%). Agent Q's MCTS search strategy and AutoGLM's curriculum RL represent two distinct exploration paradigms; combining them is a promising future direction.
- **vs. DigiRL**: DigiRL similarly proposes online RL for training a device-control agent, but its curriculum strategy selects tasks from a fixed instruction set based on current capability, whereas AutoGLM's self-evolving approach autonomously generates new training instructions, achieving higher data utilization efficiency.
- **vs. CogAgent**: CogAgent emphasizes the importance of high-resolution visual input; AutoGLM inherits this finding and further decouples visual perception from decision-making planning via the intermediate interface design.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Intermediate interface design and self-evolving curriculum RL are individually not entirely novel, but their combination forms an effective system-level solution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers both web and Android scenarios with real-deployment validation, though ablation studies are somewhat limited.
- **Writing Quality**: ⭐⭐⭐⭐ The industrial-scale system paper is clearly and systematically written, though some training details are relatively sparse due to space constraints.
- **Value**: ⭐⭐⭐⭐⭐ The first foundation GUI Agent system genuinely deployed to real users, offering important reference value for the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Co-EPG: A Framework for Co-Evolution of Planning and Grounding in Autonomous GUI Agents](co-epg_a_framework_for_co-evolution_of_planning_and_groundin.md)
- [\[AAAI 2026\] Physics-Informed Autonomous LLM Agents for Explainable Power Electronics Modulation Design](physics-informed_autonomous_llm_agents_for_explainable_power_electronics_modulat.md)
- [\[NeurIPS 2025\] AgentDAM: Privacy Leakage Evaluation for Autonomous Web Agents](../../NeurIPS2025/llm_agent/agentdam_privacy_leakage_evaluation_for_autonomous_web_agent.md)
- [\[ACL 2026\] AgencyBench: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Real-World Contexts](../../ACL2026/llm_agent/agencybench_benchmarking_the_frontiers_of_autonomous_agents_in_1m-token_real-wor.md)
- [\[AAAI 2026\] History-Aware Reasoning for GUI Agents](history-aware_reasoning_for_gui_agents.md)

</div>

<!-- RELATED:END -->
