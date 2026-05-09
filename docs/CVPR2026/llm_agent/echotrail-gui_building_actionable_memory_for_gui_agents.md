---
title: >-
  [Paper Note] EchoTrail-GUI: Building Actionable Memory for GUI Agents via Critic-Guided Self-Exploration
description: >-
  [CVPR 2026][LLM Agent][GUI agent] EchoTrail-GUI is proposed as a framework that builds a high-quality action memory repository through critic-model-guided autonomous exploration, and dynamically retrieves relevant experiences to inject into prompts at inference time, improving GPT-4o's task success rate on AndroidWorld from 34.5% to 51.7%.
tags:
  - CVPR 2026
  - LLM Agent
  - GUI agent
  - experience memory
  - self-exploration
  - retrieval-augmented generation
  - trajectory quality assessment
date: 2026-05-08
content_hash: 0bc21d83649d012c
---

# EchoTrail-GUI: Building Actionable Memory for GUI Agents via Critic-Guided Self-Exploration

**Conference**: CVPR 2026
**arXiv**: [2512.19396](https://arxiv.org/abs/2512.19396)
**Code**: None
**Area**: LLM Agent / GUI Automation
**Keywords**: GUI agent, experience memory, self-exploration, retrieval-augmented generation, trajectory quality assessment

## TL;DR

EchoTrail-GUI is proposed as a framework that builds a high-quality action memory repository through critic-model-guided autonomous exploration, and dynamically retrieves relevant experiences to inject into prompts at inference time, improving GPT-4o's task success rate on AndroidWorld from 34.5% to 51.7%.

## Background & Motivation

Current VLM-based GUI agents suffer from a **"digital amnesia"** problem: each task is handled independently, preventing the agent from systematically learning from past successful experiences. This leads to:
- Repeated mistakes of the same kind
- Poor generalization to new tasks
- Low efficiency on multi-step complex tasks

Two core bottlenecks impede improvement:

1. **Experience acquisition bottleneck**: High-quality trajectory data is scarce—manual annotation is costly and unscalable, while unguided exploration yields low-quality trajectories.
2. **Knowledge utilization gap**: Even with a trajectory corpus, efficient retrieval and application remain challenging—static examples and hand-crafted prompts cannot dynamically adapt.

The paper's core mechanism simulates the human cognitive loop of "learn → remember → apply," constructing a self-improving closed-loop system.

## Method

### Overall Architecture

EchoTrail-GUI consists of three stages:
1. **Experience Exploration**: Autonomous exploration to build a memory repository.
2. **Memory Injection**: Retrieval of relevant experiences to inject into new tasks.
3. **GUI Task Inference**: Memory-augmented execution.

### Key Designs

1. **Critic-Guided Autonomous Exploration (Stage I)**:
   - **Function**: The exploration agent autonomously interacts with the GUI environment to generate task trajectories.
   - **Mechanism**:
     - **Progressive intent focusing**: The agent first explores UI elements broadly in a curiosity-driven mode, then switches to a goal-oriented mode after $t > T_{\text{focus}}$ steps.
     - **Critic filtering**: Each trajectory is evaluated by a Critic (Gemini 2.5 Flash Lite) on a 5-point scale, with $\theta_{\text{good}} = 4$ as the quality threshold.
     - **Trajectory abstraction storage**: Rather than storing raw screenshots, structured representations of (interface text description + intent summary + executed action) are stored.
   - **Design Motivation**: A high-quality memory repository can be built without manual annotation; abstract representations reduce storage overhead and avoid device-specific bias.

2. **Dual Memory System**:
   - **Processing database $D_{\text{proc}}$**: Short-term, volatile memory that stores in-progress successful/failed trajectories and provides real-time guidance $G_t$ to the exploration agent.
   - **Memory database $D_{\text{mem}}$**: Long-term, persistent memory that stores only high-quality complete trajectories filtered by the Critic.
   - **Design Motivation**: Real-time guidance helps the exploration agent avoid repeated errors and reinforce effective strategies.

3. **Hybrid Retrieval Strategy (Stage II)**:
   - Dense retrieval $S_{\text{dense}}$: Embedding cosine similarity between instructions and final trajectory intents, computed via FAISS.
   - Sparse retrieval $S_{\text{sparse}}$: BM25 keyword matching.
   - Combined score: $\text{Score}(\tau, I) = \alpha \cdot S_{\text{dense}} + (1-\alpha) \cdot S_{\text{sparse}}$
   - Optimal retrieval count $K=2$ (confirmed by sensitivity analysis), balancing informativeness against context dilution.

4. **Memory-Augmented Inference (Stage III)**:
   - Plug-and-play: Retrieved memories are formatted as structured guides (step tuples: {interface description, agent intent, action}).
   - Injected into the agent prompt: $P_t = f(I, M_t, H_t, s_t, E_{\text{sum}}(s_t))$
   - Applicable to any off-the-shelf VLM without fine-tuning.

### Loss & Training

EchoTrail-GUI is a **training-free framework**:
- Exploration agent: Gemini 2.5 Flash, maximum trajectory length 30 steps.
- Critic model: Gemini 2.5 Flash Lite.
- Inference agent: Qwen2.5-VL-72B-Instruct or GPT-4o (no fine-tuning required).
- Summarization model: Qwen3-30B-Instruct-2507.
- Embedding model: Qwen3-Embedding-4B.

## Key Experimental Results

### Main Results

**AndroidWorld**:

| Agent | Model | Training-Free | SR↑ |
|---|---|---|---|
| GPT-4o (baseline) | GPT-4o | ✓ | 34.5% |
| GUI-explorer | GPT-4o | ✓ | 47.4% |
| **EchoTrail-GUI** | GPT-4o | ✓ | **51.7%** |
| Qwen2.5-VL | Qwen2.5-VL-72B | ✓ | 35.0% |
| UI-TARS | UI-TARS-72B-SFT | ✗ | 46.6% |
| **EchoTrail-GUI** | Qwen2.5-VL-72B | ✓ | **46.6%** |

**AndroidLab** (Qwen2.5-VL-72B backbone):

| Metric | Base Model | EchoTrail-GUI | Gain |
|---|---|---|---|
| SR | 23.9% | **37.5%** | +13.6% |
| Sub-SR | 26.1% | **41.1%** | +15.0% |
| RRR | 68.7% | **89.4%** | +20.7% |
| ROR | 81.4% | **92.1%** | +10.7% |

With the GPT-4o backbone, AndroidLab SR improves from 31.2% to 48.1% (+16.9%).

### Ablation Study

| Configuration | AndroidWorld Avg SR |
|---|---|
| Qwen2.5-VL-72B (no memory) | 34.1% |
| w/o Critic filtering | 31.0% (worse than no memory!) |
| w/o hybrid retrieval | 40.5% |
| w/o real-time guidance | 42.7% |
| **EchoTrail-GUI (full)** | **46.6%** |

### Key Findings

- **Low-quality memory is harmful, not merely unhelpful**: Removing Critic filtering drops performance to 31.0%, even below the no-memory baseline of 34.1%—a central finding that validates the necessity of quality filtering.
- **Autonomous exploration quality improves steadily**: As exploration progresses, the proportion of high-quality trajectories rises consistently across applications (e.g., OsmAnd and VLC improve by nearly 20 percentage points).
- **Generated trajectories align closely with real tasks**: UMAP visualization shows dense overlap between exploration trajectories and AndroidLab test task embeddings, with broader coverage.
- **Retrieval count $K=2$ is optimal**: Excessive memory leads to context dilution and conflicting suggestions.
- **Model agnosticity**: Significant gains are observed on both GPT-4o and Qwen2.5-VL, two substantially different backbones.

## Highlights & Insights

1. **Fully automated experience construction**: A high-quality trajectory repository (EchoTrail-4K, 4,000+ trajectories) is built without manual annotation—the core advantage distinguishing this approach from others.
2. **Critic filtering is essential, not optional**: Low-quality memory is more harmful than no memory at all.
3. **Trajectory abstraction over raw screenshots**: Textual interface description + intent + action triples enable generalization across devices and resolutions.
4. **Plug-and-play design**: As a training-free augmentation layer, the framework delivers substantial gains for any VLM backbone, lowering the barrier to deployment.

## Limitations & Future Work

- **Exploration cost**: Building EchoTrail-4K requires substantial API calls (Gemini 2.5 Flash/Lite); costs are not quantified.
- **Android-only validation**: Generalization to Web and Desktop GUI environments has not been verified.
- **Memory repository scalability**: As the repository grows, retrieval noise may increase; no forgetting or retirement mechanism is proposed.
- **Critic model bias**: Quality judgments from Gemini 2.5 Flash Lite may be biased and have not been aligned with human evaluation.
- **Fixed retrieval count**: A uniform $K=2$ is used across all tasks; tasks of varying complexity may benefit from different memory injection strategies.
- **No continual learning**: The system cannot continue accumulating experience from successful tasks after deployment.

## Related Work & Insights

- **Comparison with RAG-GUI**: The latter relies on manually curated knowledge bases, whereas EchoTrail constructs its repository entirely automatically.
- **Comparison with GUI-explorer**: The latter also employs autonomous exploration but lacks an effective quality control mechanism.
- The dual-database design ($D_{\text{proc}}$ + $D_{\text{mem}}$) parallels the separation of working memory and long-term memory in human cognition.
- **Implication for GUI agent research**: Memory system quality matters more than quantity—a finding generalizable to other agent scenarios.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of automated exploration, Critic filtering, and memory injection constitutes a systematic contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Two benchmarks, two backbones, full ablation, exploration quality analysis, and sensitivity analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Framework narrative is clear; experimental organization is well-structured.
- **Value**: ⭐⭐⭐⭐⭐ — A general-purpose, training-free, plug-and-play enhancement for GUI agents with high practical utility.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] GUI-CEval: A Hierarchical and Comprehensive Chinese Benchmark for Mobile GUI Agents](gui-ceval_a_hierarchical_and_comprehensive_chinese_benchmark_for_mobile_gui_agen.md)
- [\[CVPR 2026\] HATS: Hardness-Aware Trajectory Synthesis for GUI Agents](hats_hardnessaware_trajectory_synthesis_gui_agent.md)
- [\[CVPR 2026\] Towards GUI Agents: Vision-Language Diffusion Models for GUI Grounding](towards_gui_agents_vision-language_diffusion_models_for_gui_grounding.md)
- [\[CVPR 2026\] WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning](worldmm_dynamic_multimodal_memory_agent_for_long_video_reasoning.md)
- [\[AAAI 2026\] History-Aware Reasoning for GUI Agents](../../AAAI2026/llm_agent/history-aware_reasoning_for_gui_agents.md)

<!-- RELATED:END -->
