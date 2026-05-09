---
title: >-
  [Paper Note] T2Agent: A Tool-augmented Multimodal Misinformation Detection Agent with Monte Carlo Tree Search
description: >-
  [AAAI 2026][Social Computing][Multimodal Misinformation Detection] This paper proposes T2Agent, a misinformation detection agent integrating an extensible toolset with Monte Carlo Tree Search (MCTS). By decomposing detection into sub-tasks targeting distinct forgery sources via a multi-source verification mechanism, T2Agent achieves a new state of the art on MMfakebench, improving the accuracy of the baseline MMDAgent by 28.7% using GPT-4o as the backbone.
tags:
  - AAAI 2026
  - Social Computing
  - Multimodal Misinformation Detection
  - Monte Carlo Tree Search
  - Tool-augmented Agent
  - Multi-source Verification
  - Training-free Detection
date: 2026-05-08
content_hash: 979f5c6c3cc232af
---

# T2Agent: A Tool-augmented Multimodal Misinformation Detection Agent with Monte Carlo Tree Search

**Conference**: AAAI 2026
**arXiv**: [2505.19768](https://arxiv.org/abs/2505.19768)
**Code**: [github.com/cuixing100876/T2Agent](https://github.com/cuixing100876/T2Agent)
**Area**: Social Computing
**Keywords**: Multimodal Misinformation Detection, Monte Carlo Tree Search, Tool-augmented Agent, Multi-source Verification, Training-free Detection

## TL;DR

This paper proposes T2Agent, a misinformation detection agent integrating an extensible toolset with Monte Carlo Tree Search (MCTS). By decomposing detection into sub-tasks targeting distinct forgery sources via a multi-source verification mechanism, T2Agent achieves a new state of the art on MMfakebench, improving the accuracy of the baseline MMDAgent by 28.7% using GPT-4o as the backbone.

## Background & Motivation

### The Threat of Multimodal Misinformation

The rapid advancement of AIGC technologies has lowered the barrier to producing sophisticated multimodal misinformation, posing serious threats to information integrity, public governance, and social well-being. Developing effective multimodal misinformation detection methods has become an urgent technical and societal need.

### Two Key Bottlenecks in Existing Approaches

**Forgery source diversity demands customized tools**: Different benchmarks focus on different forgery types—AMG considers temporal consistency, while MMfakebench includes counterfactual misinformation. Existing LLM-based methods rely on fixed and limited toolsets, lacking the flexibility to handle this diversity.

**Real-world misinformation often involves mixed forgery sources**: Textual inaccuracies, image manipulation, and cross-modal inconsistency may co-occur. Reliable detection requires simultaneously:
   - **Exploitation**: gathering sufficient evidence for each forgery source
   - **Adaptive Exploration**: flexibly switching among multiple potential forgery sources

   Existing methods (e.g., MMDAgent) employ fixed, static workflows that fail to balance exploration and exploitation.

### Comparison with MMDAgent

MMDAgent follows a fixed sequential verification pipeline—checking along predefined paths step by step—which is prone to error propagation in early stages. T2Agent, by contrast, leverages MCTS to dynamically plan verification paths, adaptively balancing exploration and exploitation across multiple forgery sources.

## Method

### Overall Architecture

T2Agent consists of two core components:
1. **Multi-source Verification MCTS**: Decomposes the detection task into multiple sub-tasks (corresponding to different forgery sources) and dynamically collects evidence via tree search.
2. **Extensible Toolset**: Modular tools described with standardized templates, supporting plug-and-play integration.

### Key Designs

#### 1. **Multi-source Verification Monte Carlo Tree Search (MV-MCTS)**

**Initialization**:
- The root node represents the overall task (determining the authenticity of a news item).
- First-level child nodes correspond to sub-tasks for different forgery sources (e.g., textual veracity deficiency TVD, visual veracity deficiency VVD, cross-modal consistency deficiency CCD).
- An LVLM analyzes the input content to estimate probability weights for each sub-task.

**Selection**: An improved UCT formula:

$$UCT(s_t) = \frac{V(s_t)}{N(s_t) + 1} + C\sqrt{\frac{\ln(N(s) + 1)}{N(s_t) + 1}}$$

Key improvement: A bias term of 1 is added to $N(s_t)$, enabling UCT computation for unvisited nodes ($N(s_t) = 0$) without assigning arbitrarily large rewards, thereby reducing unnecessary resource expenditure.

**Expansion and Simulation**:
- At a selected node, the LVLM generates a thought → selects a tool → executes an action → receives an observation.
- Failed trajectories are stored as memory to prevent repeating errors.

**Evaluation — Dual Scoring Mechanism**:

1. **Reasoning Trajectory Score** $S_t^T$: Evaluates the quality and coherence of the reasoning path from the root to the current node.
$$S_t^T = LLM(\{s_i, a_i\}_{i=0...t})$$

2. **Confidence Score** $S_t^C$: Evaluates the quality and internal consistency of collected evidence.
$$S_t^C = LLM(\{o_i\}_{i=0...t}, c)$$

The composite node value: $V(s_t) = \alpha S_t^T + (1 - \alpha) S_t^C$

**Backpropagation**:
$$V(s_t) \leftarrow \frac{V(s_t) \cdot N(s_t) + V(s)}{N(s_t) + 1}$$

**Pruning Strategy**: If a sub-task node returns a high-confidence "real" verdict, its child nodes are pruned, simulating how a human expert operates—once a source or modality is confirmed reliable, attention shifts to verifying other uncertain aspects.

#### 2. **Collaborative Decision Making**

- **Early Stopping**: If any sub-task yields a high-confidence "fake" verdict, the system immediately outputs "fake news."
- **Probabilistic Fusion**: In the absence of a high-confidence fake signal, results from all sub-tasks are aggregated:

$$p(\text{fake}^i) = \begin{cases} S^{C,i} & \text{if answer}(i) = \text{fake} \\ 1 - S^{C,i} & \text{if answer}(i) = \text{real} \end{cases}$$

$$p(\text{real}) = \prod_{i=1}^{n} (1 - p(\text{fake}^i))^{(1/n)}$$

Final decision: $\text{answer} = \arg\max(p(\text{real}), \{p(\text{fake}^i)\}_{i=1}^{n})$

#### 3. **Extensible Toolset**

Each tool is encapsulated in a **tool card** that abstracts its function, input/output formats, and invocation method:

| Tool Category | Specific Tool | Function |
|--------------|--------------|----------|
| Web Search | Google Search API, Wikipedia API | Retrieving external knowledge |
| Temporal Detection | TinEye Reverse Image Search | Tracing the earliest appearance of an image |
| Forgery Detection | PSCC-NET | Image manipulation detection |
| Counterfactual Detection | LLaVA-34B | Detecting logically or physically implausible elements in images |
| Image Understanding | Aligned with backbone model | Detailed visual content Q&A and explanation |
| Entity Recognition | Baidu Entity Recognition API | Identifying key entities in images |

**Greedy Tool Selection**:
1. Predefine a minimal default toolset $D_{\text{base}}$.
2. Evaluate the incremental accuracy of each candidate tool: $\Delta_{d_i} = \text{Acc}(D_{\text{base}} \cup \{d_i\}) - \text{Acc}(D_{\text{base}})$.
3. Include only tools where $\Delta_{d_i} > 0$.

Distinction from OctoTools: OctoTools evaluates each tool independently, whereas this paper assesses combinatorial effect via incremental greedy selection (F1: 0.568 vs. 0.550).

### Loss & Training

T2Agent is a **training-free** detection method—requiring no additional training, achieving detection solely through inference-time MCTS search and tool invocation. Key inference-time hyperparameters include:
- Number of simulations $K$
- Search depth limit $d$
- Exploration weight $C$
- Trade-off coefficient $\alpha$ between trajectory and confidence scores

## Key Experimental Results

### Main Results

**MMfakebench Results** (4 categories: real, textual veracity deficiency TVD, visual veracity deficiency VVD, cross-modal consistency deficiency CCD):

| Method | Backbone | F1 | Accuracy |
|--------|---------|-----|---------|
| Standard Prompt | GPT-4o | 0.492 | 0.609 |
| MMD-agent | GPT-4o | 0.614 | 0.616 |
| LRQ-FACT | GPT-4o | 0.716 | 0.708 |
| **T2Agent** | **GPT-4o** | **0.759** | **0.753** |
| MMD-agent | GPT-4o-mini | 0.478 | 0.485 |
| **T2Agent** | **GPT-4o-mini** | **0.631** | **0.629** |
| MMD-agent | GPT-4.1-nano | 0.398 | 0.424 |
| **T2Agent** | **GPT-4.1-nano** | **0.568** | **0.569** |

Key finding: T2Agent with GPT-4o-mini (F1=0.631) even surpasses MMD-agent with GPT-4o (F1=0.614), at a cost of only $129.4 vs. $344.4.

**AMG Results** (5 forgery source categories):

| Backbone | Method | F1 | Accuracy |
|---------|--------|-----|---------|
| GPT-4o | MMD-agent | 0.365 | 0.306 |
| GPT-4o | **T2Agent** | **0.510** | **0.579** |
| GPT-4o-mini | MMD-agent | 0.360 | 0.227 |
| GPT-4o-mini | **T2Agent** | **0.499** | **0.538** |

F1 improvements reach 38.6%–39.7%, attributed to MCTS mitigating the error propagation inherent in MMD-Agent's sequential decision-making.

### Ablation Study

**Contribution of Each Module (MMfakebench, GPT-4.1-nano)**:

| Method | F1 | Accuracy | Note |
|--------|-----|---------|------|
| MMD-agent (baseline) | 0.398 | 0.424 | Fixed workflow |
| +TOOLs (tools only) | 0.413 | 0.459 | Limited gain from tools alone |
| +MV_MCTS (tree search only) | 0.535 | 0.534 | MCTS is the core contributor (+34.4% F1) |
| **MV_MCTS + TOOLs (full)** | **0.568** | **0.569** | Tools provide additional +6.2% |

**Cost Analysis (MMfakebench, USD)**:

| Model | MMD-agent | T2Agent |
|-------|-----------|---------|
| GPT-4o | 344.4 | 1637.1 |
| GPT-4o-mini | 14.3 | 129.4 |
| GPT-4.1-nano | 9.5 | 76.2 |

T2Agent incurs higher computational cost, but its ability to outperform stronger model baselines when using lighter models yields a superior overall cost-performance ratio.

### Key Findings

1. **MCTS is the core driver of performance gains**: MCTS alone contributes +34.4% F1, demonstrating that dynamic verification far outperforms static workflows.
2. **Tool selection matters more than tool quantity**: The greedily selected 3-tool set outperforms OctoTools' 4-tool set (F1: 0.568 vs. 0.550).
3. **Lightweight model + T2Agent ≥ Heavy model + baseline**: T2Agent on GPT-4o-mini surpasses MMD-agent on GPT-4o.
4. **Multi-source verification prevents error propagation**: MMD-Agent's sequential strategy degrades severely on AMG (5 forgery source categories).

## Highlights & Insights

1. **Innovative application of MCTS to information verification**: The work extends classical MCTS from single-objective settings to multi-source verification, introducing sub-task nodes, pruning strategies, and dual scoring—elegantly balancing exploration and exploitation.
2. **Emulating human expert decision-making**: The system first assesses the most probable forgery source, prunes confirmed reliable sources, and shifts attention to unverified aspects—mirroring the workflow of investigative journalists.
3. **Advantage of training-free design**: No labeled data collection or model training is required, enabling direct adaptation to emerging forgery types.
4. **Standardized toolset design**: Tool card abstraction minimizes the engineering effort required to integrate new tools.

## Limitations & Future Work

1. **Significant computational overhead**: The tree search mechanism incurs costs 3–8× higher than the baseline.
2. **Reliance on commercial LLM APIs**: Access to models such as GPT-4o is required, introducing non-transparent cost considerations.
3. **Security risks of open-source toolchains**: The paper acknowledges this issue but does not address it in depth—implementing the principle of least privilege and tool invocation whitelisting is warranted.
4. **Future directions**: Incorporating efficient pruning strategies, guiding search with lightweight expert models, and exploring integration with training-based methods.

## Related Work & Insights

- **MMDAgent**: Static pipeline + GPT-4o, serving as the direct baseline; lacks flexibility.
- **LRQ-FACT**: Retrieves evidence via generated questions, but the workflow remains fixed.
- **MGCA**: An end-to-end trained multi-view feature approach, complementary to T2Agent's training-free paradigm.
- **AlphaGo / AlphaZero**: Classic applications of MCTS that inspired the extensions proposed in this work.
- Implications for information verification: Dynamic reasoning agents represent a promising direction for detecting misinformation with mixed forgery sources.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Innovative combination of MCTS, multi-source verification, and an extensible toolset
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers two benchmarks and multiple backbone models with detailed ablation studies
- Writing Quality: ⭐⭐⭐⭐ — Framework description is clear, though some details require consulting the appendix
- Value: ⭐⭐⭐⭐⭐ — A training-free detection solution targeting real-world scenarios, highly practical

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Beyond Detection: Exploring Evidence-based Multi-Agent Debate for Misinformation Intervention and Persuasion](beyond_detection_exploring_evidence-based_multi-agent_debate_for_misinformation_.md)
- [\[AAAI 2026\] Reasoning About the Unsaid: Misinformation Detection with Omission-Aware Graph Inference](reasoning_about_the_unsaid_misinformation_detection_with_omission-aware_graph_in.md)
- [\[AAAI 2026\] Argumentative Debates for Transparent Bias Detection](argumentative_debates_for_transparent_bias_detection_technic.md)
- [\[AAAI 2026\] FactGuard: Event-Centric and Commonsense-Guided Fake News Detection](factguard_event-centric_and_commonsense-guided_fake_news_detection.md)
- [\[CVPR 2026\] Bridging Pixels and Words: Mask-Aware Local Semantic Fusion for Multimodal Media Verification](../../CVPR2026/social_computing/bridging_pixels_and_words_mask-aware_local_semantic_fusion_for_multimodal_media_.md)

</div>

<!-- RELATED:END -->
