---
title: >-
  [Paper Note] EchoTrail-GUI: Building Actionable Memory for GUI Agents via Critic-Guided Self-Exploration
description: >-
  [CVPR 2026][LLM Agent][GUI Agent] EchoTrail-GUI proposes a three-stage closed-loop framework: an exploration agent autonomously interacts with GUI environments to generate trajectories → a critic reward model filters and…
tags:
  - "CVPR 2026"
  - "LLM Agent"
  - "GUI Agent"
  - "Actionable Memory"
  - "Self-Exploration"
  - "Critic-Guided"
  - "RAG Inference"
date: 2026-05-08
content_hash: 0f5ecb247fb16aac
---

# EchoTrail-GUI: Building Actionable Memory for GUI Agents via Critic-Guided Self-Exploration

**Conference**: CVPR 2026
**arXiv**: [2512.19396](https://arxiv.org/abs/2512.19396)
**Code**: Available
**Area**: LLM Agent / GUI Automation
**Keywords**: GUI Agent, Actionable Memory, Self-Exploration, Critic-Guided, RAG Inference

## TL;DR

EchoTrail-GUI proposes a three-stage closed-loop framework: an exploration agent autonomously interacts with GUI environments to generate trajectories → a critic reward model filters and retains only high-quality trajectories to construct a memory store (EchoTrail-4K) → upon receiving a new task, the most relevant memories are injected via hybrid dense-sparse retrieval to guide inference. This transforms a stateless GUI agent into a memory-augmented system, achieving 51.7% SR (+17.2pp) with GPT-4o on AndroidWorld, and improving Qwen2.5-VL-72B SR from 23.9% to 37.5% on AndroidLab.

## Background & Motivation

**Background**: VLM-driven GUI agents can already parse GUI screenshots and execute multi-step operations such as tapping, scrolling, and typing. However, current agents universally suffer from "digital amnesia"—each task is processed independently, with no ability to accumulate or reuse operational experience, leading to repeated mistakes and poor generalization.

**Key Challenge**: (1) **Experience acquisition bottleneck**—high-quality interaction trajectories are foundational to GUI agents, yet manual annotation is costly and unscalable, while unguided autonomous exploration produces large quantities of noisy and incoherent trajectories; (2) **Knowledge application gap**—even when a trajectory repository exists, most agents still rely on static prompts or hand-crafted examples, unable to dynamically retrieve and leverage historical experience for the current task.

**Limitations of Prior Work**: Synthesizing trajectories from videos or tutorials (AgentTrek) is constrained by data source coverage; autonomous exploration (GUI-explorer) lacks trajectory quality control; RAG-GUI relies on external knowledge bases but lacks high-quality self-built experience.

**Core Idea**: Simulating the human cognitive loop of "learn → remember → apply," the framework constructs a fully automated self-improvement cycle—the agent autonomously explores to generate experience → the critic evaluates and filters good experience → good experience guides the agent toward better decisions, entirely without human supervision.

## Method

### Overall Architecture

EchoTrail-GUI consists of three stages forming a complete "explore → memorize → infer" pipeline:

- **Stage I: Critic-Guided Self-Exploration** (Experience Exploration)—automatically constructs a high-quality memory store $D_{mem}$
- **Stage II: Dynamic Memory Injection** (Memory Injection)—hybrid retrieval of the most relevant historical trajectories
- **Stage III: Memory-Augmented Inference** (GUI Task Inference)—injects retrieved memories into the prompt to guide agent execution

The problem is formulated as a Partially Observable Markov Decision Process (POMDP). The standard agent policy is $\pi_{base}(a_t|s_t, I, H_t)$; EchoTrail augments it to a memory-augmented policy: $a_t \sim \pi_{aug}(a_t|s_t, I, H_t, M_t)$, where $M_t \subset D_{mem}$ is the set of retrieved relevant memories.

### Key Designs

#### 1. Critic-Guided Self-Exploration (Stage I)

- **Function**: Fully automatically constructs a high-fidelity memory store without any human supervision.
- **Mechanism**: A unified system comprising an exploration agent, critic scoring, and a dual-database—the agent generates diverse trajectories, the critic enforces quality control, and the dual database supports online learning.
- **Design Motivation**: Unguided autonomous exploration produces large volumes of low-quality trajectories; manual annotation is unscalable; an automated "generate + evaluate" mechanism is required.

Specific mechanism:
- The **exploration agent** ($\pi_{explore}$, Gemini 2.5 Flash) autonomously interacts with the GUI environment, generating trajectories of up to 30 steps.
- **Progressive Intent Focus**: The agent operates in **curiosity-driven mode** (prioritizing exploration of new UI elements) during early stages, then transitions to **goal-focused mode** (formulating specific sub-goals such as "add to cart" before execution) after $T_{focus}$ steps—ensuring trajectories are both diverse and coherent.
- **Critic reward model** ($R_{critic}$, Gemini 2.5 Flash Lite): scores each complete trajectory on a 1–5 scale measuring coherence, efficiency, and goal achievement; only trajectories with $R_{critic}(\tau) \geq \theta_{good} = 4$ are stored in the permanent memory store.
- **Trajectory abstraction**: Raw screenshots are not stored; instead, trajectories are converted into lightweight structured representations—(1) UI text description, (2) agent intent summary, (3) executed action—reducing storage cost, eliminating device bias, and enhancing cross-environment generalizability.
- **Dual-database system**:
  - Processing database $D_{proc}$ (short-term, volatile): stores in-progress trajectories (both successful and failed), providing real-time guidance $G_t$ for ongoing exploration to help the agent avoid repeated errors.
  - Memory store $D_{mem}$ (permanent): stores only high-quality complete trajectories that pass critic filtering, serving as the core asset for downstream inference.

#### 2. Dynamic Memory Injection (Stage II)

- **Function**: Given a new task instruction $I$, efficiently retrieves the most relevant historical trajectories from the memory store.
- **Mechanism**: Dense retrieval captures semantic similarity; sparse retrieval captures keyword matching; weighted fusion combines the strengths of both.
- **Design Motivation**: Pure semantic retrieval may miss critical term-level matches; pure keyword retrieval cannot understand semantic synonymy; a hybrid strategy provides complementary coverage.

Hybrid retrieval scoring formula:

$$\text{Score}(\tau, I) = \alpha \cdot S_{dense}(\tau, I) + (1-\alpha) \cdot S_{sparse}(\tau, I)$$

- $S_{dense}$: cosine similarity using FAISS index with Qwen3-Embedding-4B embeddings
- $S_{sparse}$: BM25 lexical matching score
- Top-$K=2$ trajectories are retrieved (verified by ablation as the optimal trade-off)

**Memory formatting**: Raw trajectory logs are converted into structured step-by-step guides—each step comprising a {UI description, agent intent, executed action} triplet—rendering them as readable operational blueprints.

#### 3. Memory-Augmented Inference (Stage III)

- **Function**: Injects retrieved memories into the agent's prompt to guide reasoning and decision-making.
- **Mechanism**: Plug-and-play—the memory system is fully decoupled from the agent model and can be seamlessly applied to any VLM.
- **Design Motivation**: No model parameters are modified (training-free); the VLM's in-context learning capability is leveraged to absorb experiential guidance.

The structured prompt at each inference step is: $P_t = f(I, M_t, H_t, s_t, E_{sum}(s_t))$, integrating the task instruction ($I$), formatted memories ($M_t$), action history ($H_t$), current screenshot ($s_t$), and a textual screenshot summary ($E_{sum}$ generated by Qwen3-30B-Instruct). The agent outputs a special "finish" action to signal task completion.

### Data Asset: EchoTrail-4K

An automatically constructed dataset of 4,000+ high-quality Android interaction trajectories spanning diverse applications and task types. UMAP visualization demonstrates that the generated trajectories are highly semantically aligned with real test tasks and also cover semantic regions not represented in the test set, confirming the diversity and effectiveness of the exploration strategy.

## Key Experimental Results

### Main Results: AndroidWorld Benchmark

| Agent | Model | Training-Free | SR(%) |
|-------|-------|:---:|:---:|
| AppAgent | GPT-4o | ✓ | 14.9 |
| Gemini | Gemini-1.5-Pro | ✓ | 22.8 |
| Claude | Claude-CU | ✓ | 27.9 |
| GPT-4o | GPT-4o | ✓ | 34.5 |
| M3A | GPT-4o | ✓ | 40.5 |
| ScaleTrack | GPT-4o | ✗ | 44.0 |
| URST | GPT-4o + Reflexion | ✗ | 46.6 |
| GUI-explorer | GPT-4o | ✓ | 47.4 |
| **EchoTrail-GUI** | **GPT-4o** | **✓** | **51.7** |
| Aguvis | Aguvis-72B | ✗ | 26.1 |
| Qwen2.5-VL | Qwen2.5-VL-72B | ✓ | 35.0 |
| RAG-GUI | RAG-GUI-72B-RSF | ✗ | 45.7 |
| UI-TARS | UI-TARS-72B-SFT | ✗ | 46.6 |
| **EchoTrail-GUI** | **Qwen2.5-VL-72B** | **✓** | **46.6** |

→ Best closed-source: GPT-4o + EchoTrail achieves **51.7%**, surpassing all baselines (including methods requiring training) and exceeding vanilla GPT-4o by +17.2pp. Open-source: Qwen2.5-VL + EchoTrail (46.6%) matches UI-TARS-72B-SFT (46.6%) without any training.

### Main Results: AndroidLab Benchmark (Multi-Dimensional)

| Agent | Model | Sub-SR | RRR | ROR | SR(%) |
|-------|-------|:---:|:---:|:---:|:---:|
| GPT-4o | GPT-4o | 35.0 | 87.3 | 85.4 | 31.2 |
| AutoGLM | AutoGLM | — | — | — | 36.2 |
| **EchoTrail-GUI** | **GPT-4o** | **50.7** | **97.9** | **88.5** | **48.1** |
| UI-TARS-72B-ft | UI-TARS-72B | 28.4 | 81.4 | 81.6 | 22.1 |
| Qwen2.5-VL-72B | Qwen2.5-VL-72B | 26.1 | 68.7 | 81.4 | 23.9 |
| Qwen2.5-VL-72B-ft | Qwen2.5-VL-72B | 30.9 | 81.3 | 79.3 | 25.0 |
| **EchoTrail-GUI** | **Qwen2.5-VL-72B** | **41.1** | **89.4** | **92.1** | **37.5** |

→ GPT-4o + EchoTrail: SR 31.2% → 48.1% (+16.9pp), Sub-SR 35.0 → 50.7. Qwen2.5-VL + EchoTrail: SR 23.9% → 37.5% (+13.6pp), with ROR improving from 81.4 to 92.1—indicating not only higher completion rates but also reduced operational redundancy and significantly improved action rationality.

### Ablation Study (AndroidWorld, Qwen2.5-VL-72B)

| Configuration | Easy | Medium | Hard | Avg SR(%) |
|------|:---:|:---:|:---:|:---:|
| Qwen2.5-VL-72B (no memory) | 46.7 | 23.6 | 13.2 | 34.1 |
| w/o Critic filtering | 47.5 | 13.9 | 10.5 | **31.0** (↓3.1) |
| w/o hybrid retrieval | 60.7 | 20.8 | 13.2 | 40.5 |
| w/o real-time guidance | 62.3 | 25.0 | 13.2 | 42.7 |
| **EchoTrail-GUI (full)** | **65.6** | **30.6** | **15.8** | **46.6** |

### Key Findings

- **Low-quality memory is harmful**: Removing critic filtering causes SR to drop from 46.6% to 31.0%, falling **below the no-memory baseline** (34.1%)—injecting noisy memories is worse than injecting none at all. This is the most critical finding of the paper.
- **Hybrid retrieval provides complementary gains**: Removing hybrid retrieval reduces SR by 6.1pp → both semantic and keyword channels are indispensable.
- **Real-time guidance improves exploration quality**: Removing it reduces SR by 3.9pp → the dual-database online feedback mechanism is effective.
- **$K=2$ is the optimal retrieval count**: Too few provides insufficient information; too many introduces noise and prompt bloat; $K=2$ achieves the best balance.
- **Exploration quality improves continuously**: Real-time guidance causes the proportion of high-quality trajectories to rise steadily during exploration, with improvements of nearly 20pp on complex apps (OsmAnd, VLC).
- **Model-agnostic**: Effective for both GPT-4o and Qwen2.5-VL, validating the plug-and-play property.

## Highlights & Insights

- **Precise diagnosis and remedy for "digital amnesia"**: Upgrading GUI agents from stateless to memory-augmented systems represents a fundamental capability leap; the fully automated "generate → evaluate → accumulate → retrieve → apply" closed loop has substantial engineering value.
- **Critic filtering is central, not optional**: Ablation demonstrates that unfiltered memories are harmful—this challenges the naive intuition that "more memory is better" and reveals the core principle that experience quality > experience quantity.
- **Self-improving data flywheel**: Agent explores → critic filters → good memories guide better exploration → more good memories are produced—a positive feedback loop that continuously improves exploration efficiency.
- **Training-free + plug-and-play**: Without modifying any model parameters, injecting memories via context alone surpasses methods that require fine-tuning—highly favorable for practical deployment.

## Limitations & Future Work

- The exploration stage requires extensive Gemini 2.5 Flash API calls—the computational and financial cost of constructing the memory store is non-trivial.
- The memory store is static after construction (not updated or pruned as agent capabilities improve)—mechanisms for memory forgetting and refinement are absent.
- Trajectory abstraction discards raw screenshot visual information—this may be disadvantageous for tasks requiring precise visual grounding.
- Layout descriptions in old trajectories may become stale after application UI updates—the temporal validity of memories is not considered.
- Validation is limited to Android environments; generalizability to Web and Desktop GUI scenarios remains to be verified.

## Related Work & Insights

- **vs. GUI-explorer**: Both rely on autonomous exploration to build experience, but GUI-explorer lacks critic-based quality control → EchoTrail's filtering mechanism is the key differentiator.
- **vs. RAG-GUI**: RAG-GUI uses external knowledge guides, while EchoTrail builds experience autonomously → self-built experience more closely reflects actual operational sequences.
- **vs. UI-TARS**: UI-TARS requires fine-tuning a 72B model to reach 46.6%; EchoTrail matches this on Qwen2.5-VL without training → memory augmentation serves as an efficient alternative to fine-tuning.
- **Broader implication**: The critic filtering paradigm can be generalized to all experience-driven agent systems—any self-improvement framework should incorporate rigorous quality control rather than blindly accumulating experience.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The closed-loop design of critic-guided self-exploration combined with memory-augmented inference is novel; the insight that low-quality memory is harmful is particularly profound.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Dual benchmarks (AndroidWorld + AndroidLab), dual backbones (GPT-4o + Qwen), four-dimensional metrics, complete ablations, sensitivity analysis, and exploration quality tracking.
- **Writing Quality**: ⭐⭐⭐⭐ The three-stage architecture diagram is clear, Algorithm 1 is complete, and the ablation analysis is well-structured.
- **Value**: ⭐⭐⭐⭐⭐ A training-free approach that surpasses fine-tuned methods has direct practical impact on GUI automation; the EchoTrail-4K dataset also has independent value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GUI-CEval: A Hierarchical and Comprehensive Chinese Benchmark for Mobile GUI Agents](gui-ceval_a_hierarchical_and_comprehensive_chinese_benchmark_for_mobile_gui_agen.md)
- [\[CVPR 2026\] Towards GUI Agents: Vision-Language Diffusion Models for GUI Grounding](towards_gui_agents_vision-language_diffusion_models_for_gui_grounding.md)
- [\[CVPR 2026\] HATS: Hardness-Aware Trajectory Synthesis for GUI Agents](hats_hardness-aware_trajectory_synthesis_for_gui_agents.md)
- [\[AAAI 2026\] History-Aware Reasoning for GUI Agents](../../AAAI2026/llm_agent/history-aware_reasoning_for_gui_agents.md)
- [\[ACL 2026\] Towards Scalable Lightweight GUI Agents via Multi-role Orchestration](../../ACL2026/llm_agent/towards_scalable_lightweight_gui_agents_via_multi-role_orchestration.md)

</div>

<!-- RELATED:END -->
