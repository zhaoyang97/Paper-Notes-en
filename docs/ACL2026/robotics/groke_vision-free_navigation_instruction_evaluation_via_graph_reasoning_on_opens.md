---
title: >-
  [Paper Note] GROKE: Vision-Free Navigation Instruction Evaluation via Graph Reasoning on OpenStreetMap
description: >-
  [ACL 2026][Robotics][Map2Seq] GROKE proposes evaluating navigation instructions **entirely without vision**—by serializing OSM maps into JSON and having Gemini-3 Pro act as a follower agent to execute instructions along…
tags:
  - "ACL 2026"
  - "Robotics"
  - "Map2Seq"
  - "OpenStreetMap"
  - "LLM agent"
  - "graph reasoning"
  - "agent-as-judge"
date: 2026-05-08
content_hash: a2c9ea67d812c394
---

# GROKE: Vision-Free Navigation Instruction Evaluation via Graph Reasoning on OpenStreetMap

**Conference**: ACL 2026  
**arXiv**: [2601.07375](https://arxiv.org/abs/2601.07375)  
**Code**: https://anonymous.4open.science/r/groke (Anonymous)  
**Area**: Robotics / VLN Instruction Evaluation  
**Keywords**: Map2Seq, OpenStreetMap, LLM agent, graph reasoning, agent-as-judge

## TL;DR
GROKE proposes evaluating navigation instructions **entirely without vision**—by serializing OSM maps into JSON and having Gemini-3 Pro act as a follower agent to execute instructions along the graph. Navigation Error (NE), Success Rate (SR), and SDTW are utilized as proxies for instruction quality. Compared to heuristic baselines, GROKE reduces navigation error by 68.5% on Map2Seq, and NE correlates significantly with human judgment of "instruction clarity" ($r = -0.31, p < 0.01$).

## Background & Motivation

**Background**: Traditional evaluation of navigation instructions ("how good is this instruction") relies on machine translation metrics like BLEU, ROUGE, METEOR, or CIDEr. The VLN community increasingly favors "agent-as-judge"—training a follower agent to follow instructions in high-fidelity visual simulators like Matterport3D or Touchdown, using success rate to determine instruction quality.

**Limitations of Prior Work**: (1) **Fatal flaws in n-gram metrics**—"Turn left at the bank" and "Turn right at the bank" yield high BLEU scores but are functionally opposite; conversely, "Turn left after seeing the red building" and "Pass the brick structure and head west" have zero BLEU but describe the same action. (2) **Visual followers conflate language quality with visual recognition**—it remains unclear if an agent failure is due to ambiguous instructions or a failure to distinguish "stucco walls" from "brick walls." (3) High barriers to entry—Google Street View and Matterport3D involve licensing issues, terabytes of data, and high compute requirements.

**Key Challenge**: The "meaning" of an instruction is defined by its **compliance condition** (the set of physical trajectories satisfying the instruction), which is independent of vision. However, all existing pragmatic evaluations couple visual perception, introducing both NLG and CV noise into the metrics.

**Goal**: (1) Develop a follower agent that executes instructions solely using symbolic OSM information without vision; (2) Compare various spatial representations (textual, JSON, graphviz, grid) to find the most suitable for LLM reasoning; (3) Use agent metrics (SR/NE) as proxies for instruction navigability and validate them against human judgment.

**Key Insight**: The Map2Seq dataset is unique as its instructions are aligned with OSM nodes, edges, and POIs, allowing for the **decoupling of the visual modality**. A purely symbolic follower agent can be constructed to specifically test the "structural/semantic executability" of instructions.

**Core Idea**: Use an LLM as a follower, serialize the OSM map into a **JSON local view**, and execute navigation using a hierarchical two-agent architecture (Sub-instruction Agent + Navigator Agent). The agent's trajectory metrics serve as "instruction quality scores," requiring zero vision and zero training.

## Method

### Overall Architecture
GROKE is a **training-free and vision-free** hierarchical system consisting of two agents (Figure 2):
1. **Sub-instruction Agent**: Decomposes the full instruction $I$ into $K$ atomic sub-goals $\{g_1,\dots,g_K\}$ (MOVE_FORWARD / TURN_LEFT / TURN_RIGHT with NL descriptions). It also extracts all landmarks $\mathcal{L}$ and maps them to OSM POIs using fuzzy string matching (RapidFuzz partial_ratio).
2. **Navigator Agent**: Iteratively executes the current sub-goal $g_k$. At each step, it constructs a visible area $\mathcal{G}_t$ (traveling $u$ nodes toward the next intersection along the current heading plus a 3-node lookahead). The tuple $(I, v_t, h_t, \mathcal{G}_t)$ is fed into the LLM to output $(\text{status}_k, v_{t+1})$. If $\text{status}_k = \text{COMPLETED}$, the agent proceeds to the next sub-goal.
3. **Termination Conditions**: (i) All sub-goals completed; (ii) Total steps > 100; (iii) Single sub-goal retries > 15.
4. **Evaluation Inversion**: While traditional VLN measures "how good the agent is," GROKE fixes the agent (Gemini-3 Pro) and interprets NE, SR, SDTW, and nDTW as measures of "**how navigable the instruction is**."

### Key Designs

1. **Vision-free JSON Spatial Serialization**:
    - **Function**: Converts the local OSM subgraph (nodes, edges, POIs) into a structured format optimized for LLM readability.
    - **Mechanism**: Organized into two sections—Nodes (containing ID, type like intersection/waypoint, heading $h \in [0,360)$, and connection list with target IDs and relative bearings) and POIs (containing landmark IDs, nearest node references, discretized relative directions like Forward/Left/Right/Back using $\delta = (h_{v\to p} - h_{\text{curr}} + 180) \mod 360 - 180$, and Haversine distance). Relative bearings are calculated using the spherical bearing formula $h = \text{atan2}(\sin\Delta\lambda \cos\phi_2, \cos\phi_1\sin\phi_2 - \sin\phi_1\cos\phi_2\cos\Delta\lambda)$.
    - **Design Motivation**: The authors compared four representations—textual narrative, JSON, Graphviz DOT, and ASCII grid matrix. Table 5 shows striking results: JSON achieved 63% SR / 68m NE, Textual 61% / 70m, Graphviz 40% / 96m, and Grid only 10% / 175m. **LLMs often treated '0' in the grid as a valid path, leading to failure**. JSON's hierarchical structure enhances the model's ability to "recover" from local deviations (OSR 74% vs Textual 67%). On "Hard" tasks, JSON SR (53.8%) outperformed Textual (38.5%) by 15 points, proving structured data is critical for long-range reasoning.

2. **Sub-instruction Agent for Instruction Decomposition**:
    - **Function**: Breaks a complex instruction (e.g., 53 tokens) into $K$ state-machine-like sub-goals.
    - **Mechanism**: The LLM acts as a parser $I \xrightarrow{\text{parse}} \{g_1,\dots,g_K\}$, where each $g_k$ follows the pattern `("MOVE_FORWARD", "Go straight to the bank", TODO)`. The Navigator focuses on the current $g_k$ rather than the full instruction, transforming "long-range planning" into "short-range execution and state progression."
    - **Design Motivation**: This relieves the LLM from the dual burden of remembering five instruction steps and spatial reasoning simultaneously. Ablation studies (Appendix A.2) show that removing the sub-instruction phase leads to a significant drop in SR.

3. **Intersection-based Visible Area Truncation**:
    - **Function**: Simulates a human field of view by only showing the local subgraph from the current position toward the next "fork in the road," preventing token explosion and hallucinations.
    - **Mechanism**: (Algorithm 1) Starting from $v_t$ along $h_t$, the system selects the neighbor with the minimum $\Delta h(h_{\text{curr}}, h_{v'})$ where $\Delta h < 100^\circ$. It continues until it passes $u$ intersections (where $\text{degree}(v) > 2$) plus a 3-node lookahead. POI proximity mapping uses a 50m threshold.
    - **Design Motivation**: This ensures the "map" seen by the LLM is comparable to what a human sees at a crossing, retaining necessary decision-making information while ignoring irrelevant distant streets.

### Loss & Training
- **Training-free**: Uses Gemini-3 Pro with default temperature 1.0 and high reasoning settings; no fine-tuning.
- Average trajectory: 5.91 steps / 44k total tokens / 23k thinking tokens.
- Implemented using the Google Agent Development Kit (ADK) and batch APIs.

## Key Experimental Results

### Main Results

Overall performance on two Map2Seq splits (700 instances/split):

| Method | TestSetA NE↓ | TestSetA SR↑ | TestSetA OSR↑ | TestSetA SDTW↑ | TestSetB NE↓ | TestSetB SR↑ | TestSetB OSR↑ | TestSetB SDTW↑ |
|---|---|---|---|---|---|---|---|---|
| Random Walker | 259.0 | 4.4% | 5.7% | 0.026 | 244.3 | 6.1% | 7.1% | 0.029 |
| Action Sampling (No Text) | 250.1 | 5.1% | 6.0% | 0.037 | 241.6 | 7.4% | 8.1% | 0.039 |
| Heuristic Agent (Regex+Angle) | 180.6 | 18.0% | 18.9% | 0.155 | 173.0 | 17.9% | 19.1% | 0.159 |
| **GROKE (Ours)** | **56.8** | **66.4%** | **78.4%** | **0.634** | **59.8** | **63.3%** | **78.0%** | **0.609** |

Human baseline SR is approximately 0.86 / 0.84 (in Street View environments). Vision-free GROKE reaches ~74-77% of human performance.

Human correlation analysis (n=100, manual navigability labeling):

| Metric | Pearson $r$ | $p$ | Spearman $\rho$ | $p$ |
|---|---|---|---|---|
| SR | 0.2865 | 0.0039** | 0.2865 | 0.0039** |
| OSR | 0.1860 | 0.0639 | 0.1860 | 0.0639 |
| SDTW | 0.2799 | 0.0048** | 0.2860 | 0.0039** |
| nDTW | 0.2457 | 0.0138* | 0.2895 | 0.0035** |
| **NE** | **-0.3096** | **0.0017**\*\* | **-0.3184** | **0.0012**\*\* |

NE is the metric most strongly correlated with human judgment.

### Ablation Study

Comparison of four spatial representations across difficulties (n=100 Map2Seq seen val):

| Representation | Easy NE | Easy SR | Medium NE | Medium SR | Hard NE | Hard SR | Overall |
|---|---|---|---|---|---|---|---|
| **JSON** | 62.1 | 61.2% | 61.2 | 68.4% | **112.9** | **53.8%** | Best; significant lead on hard tasks |
| Textual | 71.3 | 61.2% | **56.6** | 68.4% | 110.6 | 38.5% | Good for simple; fails on hard |
| Graphviz DOT | 90.4 | 40.8% | 87.8 | 47.4% | 146.5 | 15.4% | High parsing overhead |
| ASCII Grid | 186.7 | 6.1% | 160.3 | 13.2% | 176.6 | 15.4% | Disaster; LLM hallucinates paths |
| Optimized Repr. | **35.6** | **77.6%** | **30.9** | **76.3%** | 93.3 | 53.8% | Theoretical upper bound with prompt engineering |

### Key Findings
- **JSON ≫ ASCII grid is a disruptive finding**: Grid representations are popular in LLM vision-reasoning papers, but the 10% SR here reveals that text-based grid maps are nearly unusable for LLMs due to noise.
- **JSON advantages amplify on Hard tasks**: While JSON and Textual performed similarly on Easy/Medium tasks, JSON SR (53.8%) far exceeded Textual (38.5%) on Hard tasks, indicating the hierarchical structure is a more scalable scaffold.
- **NE is the best human-aligned metric**: With $r = -0.31, p < 0.01$, NE outperforms SR/OSR as an evaluation priority.
- **Vision-free performance is competitive**: The 12-point gap between human (86%) and GROKE (74%) suggests that many navigation tasks are predominantly determined by topology and landmarks rather than visual details.
- **High Cost**: The production cost of Gemini-3 Pro with high reasoning and large token counts is a barrier; future work will focus on distillation into smaller models.

## Highlights & Insights
- **Inversion of the task definition**: By fixing the agent, traditional agent metrics (SR/NE/SDTW) transform into instruction quality scores. This "frame inversion" allows the reuse of VLN benchmarks and metrics to solve a different problem.
- **Systematic comparison of LLM spatial representations**: This study provides rare evidence for picking spatial data structures (Textual/JSON/Graphviz/Grid), valuable for any work using LLMs for graph or map reasoning.
- **Relative direction discretization**: Converting continuous angles into four categories (Forward/Left/Right/Back) avoids the precision loss common when LLMs handle raw numerical coordinates.
- **Anti-intuitive "Vision-free" sufficiency**: The ability to reach 56m NE purely via JSON suggests that for blind-assistance technologies or smart glasses, topological reasoning might be as critical as computer vision.

## Limitations & Future Work
- **Inability to evaluate visual-anchor instructions**: Instructions like "Turn left at the house with the red door" cannot be evaluated by GROKE and are systematically undervalued.
- **Model bias**: Results are tied to Gemini-3 Pro and lack cross-validation with other LLMs (GPT-4o, Claude, Llama).
- **Scale constraints**: The high per-trajectory cost prevents large-scale repetitive evaluations.
- **Fuzzy matching dependence**: Grounding "the bank" depends on OSM tags; if tags are missing or brand-specific (e.g., "bank_of_america"), valid instructions may be incorrectly marked as failures.

## Related Work & Insights
- **Vs. Traditional Followers (Speaker-Follower/LANA)**: These agents couple vision and language in environments like Matterport3D. GROKE strips vision, offering a "pure linguistic diagnostic" complementary to existing followers.
- **Vs. VELMA/NavGPT**: While these are LLM-based VLN agents, they utilize visual perception. GROKE demonstrates that structured OSM maps are sufficient to support instruction execution without visual descriptions.
- **Vs. BLEU/ROUGE**: GROKE's Spearman correlation with humans (0.29-0.32) far exceeds BLEU, which is often uncorrelated with human judgment in navigation contexts.

## Rating
- **Novelty**: ⭐⭐⭐⭐ "Vision-free agent-as-judge" is a clear new proposal, supported by a systematic representation comparison.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 4 baselines, 4 representations, difficulty stratification, human correlation, and error analysis.
- **Writing Quality**: ⭐⭐⭐⭐ The motivation is clear, particularly the counter-examples for BLEU. Methods are explicit.
- **Value**: ⭐⭐⭐⭐ Provides a zero-barrier, reproducible outdoor instruction evaluation tool for the VLN community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] GoViG: Goal-Conditioned Visual Navigation Instruction Generation via Multimodal Reasoning](govig_goal-conditioned_visual_navigation_instruction_generation_via_multimodal_r.md)
- [\[CVPR 2026\] ProFocus: Proactive Perception and Focused Reasoning in Vision-and-Language Navigation](../../CVPR2026/robotics/profocus_proactive_perception_and_focused_reasoning_in_vision-and-language_navig.md)
- [\[CVPR 2026\] DecoVLN: Decoupling Observation, Reasoning, and Correction for Vision-and-Language Navigation](../../CVPR2026/robotics/decovln_decoupling_observation_reasoning_and_correction_for_vision-and-language_.md)
- [\[ACL 2026\] Capability-Oriented Failure Attribution for Vision-Language Navigation Agents](where_did_it_go_wrong_capability-oriented_failure_attribution_for_vision-and-lan.md)
- [\[ACL 2026\] VLN-NF: Feasibility-Aware Vision-and-Language Navigation with False-Premise Instructions](vln-nf_feasibility-aware_vision-and-language_navigation_with_false-premise_instr.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ACL 2026\] GoViG: Goal-Conditioned Visual Navigation Instruction Generation via Multimodal Reasoning](govig_goal-conditioned_visual_navigation_instruction_generation_via_multimodal_r.md)
- [\[CVPR 2026\] ProFocus: Proactive Perception and Focused Reasoning in Vision-and-Language Navigation](../../CVPR2026/robotics/profocus_proactive_perception_and_focused_reasoning_in_vision-and-language_navig.md)
- [\[AAAI 2026\] Neural Graph Navigation for Intelligent Subgraph Matching](../../AAAI2026/robotics/neural_graph_navigation_for_intelligent_subgraph_matching.md)
- [\[CVPR 2026\] DecoVLN: Decoupling Observation, Reasoning, and Correction for Vision-and-Language Navigation](../../CVPR2026/robotics/decovln_decoupling_observation_reasoning_and_correction_for_vision-and-language_.md)
- [\[CVPR 2026\] ManipArena: Comprehensive Real-world Evaluation of Reasoning-Oriented Generalist Robot Manipulation](../../CVPR2026/robotics/maniparena_comprehensive_real-world_evaluation_of_reasoning-oriented_generalist_.md)

</div>

<!-- RELATED:END -->
