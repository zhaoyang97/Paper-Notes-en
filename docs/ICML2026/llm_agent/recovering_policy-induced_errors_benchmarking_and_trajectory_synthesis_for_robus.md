---
title: >-
  [Paper Note] Recovering Policy-Induced Errors: Benchmarking and Trajectory Synthesis for Robust GUI Agents
description: >-
  [ICML 2026][LLM Agent][GUI Agent] Addressing the pain point where GUI agents get stuck in "self-inflicted errors" during real deployment…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "GUI Agent"
  - "Policy-Induced Errors"
  - "Error Recovery"
  - "Trajectory Tree Synthesis"
  - "Reflection Data"
date: 2026-05-08
content_hash: 68a10b8f8c1e4221
---

# Recovering Policy-Induced Errors: Benchmarking and Trajectory Synthesis for Robust GUI Agents

**Conference**: ICML 2026  
**arXiv**: [2605.29447](https://arxiv.org/abs/2605.29447)  
**Code**: https://github.com/AlibabaResearch/RoTS (Available)  
**Area**: Agent / GUI Agent / Robustness / Data Synthesis  
**Keywords**: GUI Agent, Policy-Induced Errors, Error Recovery, Trajectory Tree Synthesis, Reflection Data

## TL;DR
Addressing the pain point where GUI agents get stuck in "self-inflicted errors" during real deployment, this work introduces GUI-RobustEval (1216 executable tests covering 11 types of policy-induced errors across 4 error depths) for fine-grained evaluation. Simultaneously, it proposes RoTS—an online data synthesis framework based on trajectory trees. By actively exposing new errors on correct subtrees using fragility-based UCB and performing long-range recovery rollbacks on failed subtrees via neighboring experience, RoTS synthesizes 800k reflection samples. RoTS-32B achieves an open-source SOTA on OSWorld with 47.4% SR and 33.8% All-Pass@4.

## Background & Motivation
**Background**: Over the past year, GUI agents utilizing VLMs (GPT-5.1, Claude 4.5, Qwen3-VL, UI-TARS, OpenCUA, etc.) have achieved 30~40% average success rates on desktop tasks like OSWorld. The dominant training paradigm involves SFT on human demonstration trajectories followed by online RL, with evaluation focusing on grounding accuracy, planning precision, and overall task success rate.

**Limitations of Prior Work**: Agents frequently encounter *policy-induced errors* in real deployment—such as incorrect grounding, misreading screens, or faulty sub-goal decomposition—and struggle to recover once an error occurs. Existing benchmarks often measure external perturbations like "injected noise" or "adversarial attacks." Training data for reflection is typically either manually authored or augmented offline, resulting in a distribution that deviates significantly from the actual errors committed by the policy. System-level solutions often rely on a reflection sub-agent for recovery rather than teaching the model to "identify and fix its own errors" at the training level.

**Key Challenge**: This mismatch is explicitly categorized into two gaps: (1) **Error type mismatch**—training data is dominated by low-level execution errors (invalid clicks), whereas real failures are more often compositional planning or progress-perception errors; (2) **Error time-course mismatch**—errors in training data are usually identifiable within one step, but policy-induced errors often only manifest after several subsequent steps, requiring long-range backtracking.

**Goal**: To bridge these two gaps at both the evaluation and data ends by creating a diagnostic benchmark categorized by error type and depth, and an extensible pipeline that actively exposes diverse error patterns to synthesize long-range recovery trajectories.

**Key Insight**: Since real policy-induced errors are "branches" created when the policy interacts with the environment, the most compatible synthesizer is the policy itself performing repeated rollouts on a replayable trajectory tree. Success branches are used to actively search for new errors ("exploration"), while failure branches are used to synthesize recovery trajectories ("resurrection"). Their co-expansion naturally covers both error type and time-course gaps.

**Core Idea**: Replace manual reflection data with "exploration-recovery co-expansion" on online trajectory trees, allowing the agent to train its robustness using its own failed-recovery pairs.

## Method
RoTS consists of two main components: (i) the construction of GUI-RobustEval, and (ii) the RoTS data synthesis pipeline.

### Overall Architecture
**Evaluation Side: GUI-RobustEval**: From 1.5k failed trajectories of 12 SOTA agents on OSWorld, root-cause steps and error types were manually labeled. This resulted in 11 categories of policy-induced errors across 4 error depths $d \in \{0,1,3,5\}$. Each test case consists of a manually cleaned correct prefix + a root-cause step + $d$ steps of subsequent erroneous execution. The environment state is replayed to this "post-error" state, and the agent under test takes over. Two complementary metrics are reported: *Error-Awareness Rate* (whether the agent recognizes the error in the first step, judged by a VLM) and *Post-Error Success Rate* (final task completion).

**Synthesis Side: RoTS**: Trajectory trees $T=(O,A,E)$ are built on 20k tasks with reproducible snapshots, where nodes represent screenshots and edges represent actions. $N=4$ rollouts are initialized in parallel, followed by $K=32$ iterations of "explore-recovery co-expansion." In each round, $T$ is partitioned into a success subtree $T^{\text{corr}}$ and a failure subtree $T^{\text{fail}}$ based on an external reward model $\mathcal{R}$, with each side expanded once. Finally, rollouts are filtered and assembled into 800k training samples for SFT of Qwen2.5-VL-7B/32B. For each sample $x_i=(u, h_{i-1}, o_i, a_i)$, NLL is calculated only for the action tokens $a_i$ to avoid learning noisy history.

### Key Designs

1.  **GUI-RobustEval: Controlled Error Prefixes from Real Failures**:
    *   **Function**: Constructs a robustness benchmark for fine-grained diagnostics by error type and depth, identifying which errors are unrecognizable and at what depth recovery becomes impossible.
    *   **Mechanism**: Root-cause steps and error distributions were extracted from 1.5k failures of SOTA agents. Cases were standardized into templates: "clean prefix + root-cause + $d$ subsequent steps" (unified via action summaries and PyAutoGUI). During testing, the system replays the prefix to depth $d$, placing the agent in a state where an error has already persisted for $d$ steps.
    *   **Design Motivation**: Unlike benchmarks like GUI-Reflection or RedTeamCUA that use synthetic or environmental perturbations, GUI-RobustEval cases come from real agent failures. Controlling error depth allows for the first quantitative measurement of "long-range recovery"—data shows success rates drop monotonically with depth, highlighting a neglected dimension of robustness.

2.  **Fragility-Driven Exploration (FDE): Active Error Discovery in Success Subtrees**:
    *   **Function**: Identifies nodes in $T^{\text{corr}}$ that "appear correct but are prone to failure" and initiates rollouts from there to generate new failure branches, solving the error type coverage issue.
    *   **Mechanism**: For each node $o_i$, a pre-operative progress scorer $\mathcal{R}_p$ evaluates $N$ candidate actions sampled from the policy to calculate a step-level success rate $r_i = \tfrac{1}{N}\sum_n r_{i,n}$. A fragility score is defined as $f_i = (1-r_i) + c\sqrt{\ln(V^f_{p(i)}+1)/(V^f_i+1)}$ in a UCB format ($V^f$ is the visit count). The node $i^*=\arg\max_i f_i$ is selected for further rollout by the policy $\pi_\theta$.
    *   **Design Motivation**: Parallel sampling from the root is inefficient and lacks breadth. FDE reuses correct prefixes and branches at unstable nodes, concentrating the sampling budget on positions likely to expose errors while ensuring diversity through the UCB exploration term.

3.  **Experience-Induced Recovery (EIR): Synthesis of Long-Range Recovery in Failure Subtrees**:
    *   **Function**: Automatically locates root causes in $T^{\text{fail}}$, provides natural language recovery advice, and uses a recovery actor to generate long-range recovery trajectories.
    *   **Mechanism**: For each failed trajectory $\tau^{\text{fail}}$, experience from sibling branches $\mathcal{E}(\tau^{\text{fail}})=\{E_{\tau^{\text{nb}}}\}$ (reusable successful trajectories) is aggregated. A reflector $\pi^{er}_\theta$ generates $(i, g_i, p_i)$—the candidate error step, recovery guidance, and expansion priority. Nodes are selected via a UCB score $s_i = p_i + c\sqrt{\ln(V^r_{p(i)}+1)/(V^r_i+1)}$, and a recovery actor generates $\tau^{\text{rec}} \sim \pi^{rec}_\theta(u, o_{i^*}, h_{i^*-1}, g_{i^*})$.
    *   **Design Motivation**: Failures contain information: brother branches show "what was right," and reflectors explain "what went wrong." Combining these into advice-conditioned rollouts produces "error → identification → recovery → completion" samples. Post-processing categorizes data into reflection-agnostic $\mathcal{D}_{\text{agn}}$ and reflection-related $\mathcal{D}_{\text{ref}}$ subsets.

### Loss & Training
The model is trained on a mixture of data: $\mathcal{D}_{\text{train}} = \mathcal{D}_{\text{agn}} \cup \lambda_{\text{ref}} \mathcal{D}_{\text{ref}}$, where $\lambda_{\text{ref}}=0.1$ (720k agnostic + 80k reflection samples). The objective is standard teacher-forcing NLL:
$$\mathcal{L}(\theta) = \mathbb{E}_{(u,h,o,a)\sim\mathcal{D}_{\text{train}}}[-\log \pi_\theta(a|u,h,o)]$$
Only tokens for $a_i$ are supervised; context $h$ is used solely for conditioning to avoid gradient updates on imperfect rollout noise. Base models are Qwen2.5-VL-7B and 32B.

## Key Experimental Results

### Main Results

Success rates across error depths on GUI-RobustEval (Open-source models):

| Agent | Depth 0 | Depth 1 | Depth 3 | Depth 5 | Gain (Relative) | Awareness |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Qwen2.5-VL-7B | 5.1 | 3.0 | 2.9 | 1.3 | ↓75% | — |
| GUI-Owl-7B | 28.7 | 15.6 | 8.1 | 10.4 | ↓64% | 5.9 |
| UI-TARS1.5-7B | 39.6 | 34.2 | 27.8 | 23.3 | ↓41% | 38.0 |
| OpenCUA-7B | 40.7 | 30.3 | 23.3 | 19.0 | ↓53% | 46.3 |
| OpenCUA-32B | 45.5 | 37.2 | 28.6 | 25.9 | ↓53% | 50.3 |
| **RoTS-7B** | 43.5 | 36.6 | 30.1 | 26.7 | **↓38%** | 51.9 |
| **RoTS-32B** | **49.7** | **41.8** | **36.5** | **33.2** | **↓33%** | **58.8** |

OSWorld-Verified Main Results (Max 50 steps, All-Pass@4 measures the success rate across 4 independent runs):

| Agent | Data Source | All-Pass@4 | Max 15 | Max ≥50 |
| :--- | :--- | :--- | :--- | :--- |
| Claude 4.5 Sonnet | Closed | – | 42.9 | 58.1 |
| GPT-OpenAI CUA | Closed | – | 26.0 | 31.3 |
| Qwen3-VL-Plus | Closed | 24.5 | 33.1 | 35.2 |
| OpenCUA-7B | Open | 12.5 | 24.3 | 28.2 |
| GUI-OWL-7B | Closed | 14.7 | 27.1 | 29.4 |
| OpenCUA-32B | Open | 15.5 | 29.7 | 34.1 |
| **RoTS-7B** | Open | **26.3** | 31.7 | **36.3** |
| **RoTS-32B** | Open | **33.8** | **42.8** | **47.4** |

### Ablation Study
Rollout strategy ablation with 100k data (PS = Parallel Sampling):

| Data Source | Aware. | Post.Succ. | All-Pass@4 | OSWorld(50) | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| PS | 19.9 | 12.1 | 8.6 | 18.1 | Pure parallel sampling baseline |
| + FDE | 22.5 | 14.4 | 9.1 | 19.6 | Adds fragility exploration |
| + EIR | 28.3 | 18.1 | 12.1 | 19.5 | Adds recovery rollback |
| + FDE + EIR | **32.1** | **22.1** | **14.1** | **21.4** | Co-expansion best |

Comparison with human demonstrations (AgentNet):

| Training Data | All-Pass@4 | OSWorld(50) | Description |
| :--- | :--- | :--- | :--- |
| $\mathcal{D}_{\text{agn(hum)}}$ | 7.8 | 15.3 | Human reflection-agnostic only |
| $\mathcal{D}_{\text{agn(hum)}} \cup \mathcal{D}_{\text{ref(hum)}}$ | 8.4 | 16.1 | Adds human reflection samples |
| $\mathcal{D}_{\text{agn(hum)}} \cup \mathcal{D}_{\text{ref}}$ | 11.6 | 18.8 | Replaces human reflection with RoTS |
| $\mathcal{D}_{\text{agn}} \cup \mathcal{D}_{\text{ref}}$ | **14.1** | **21.4** | Entirely RoTS data |

### Key Findings
*   **EIR contributes to robustness, while FDE enhances overall success rate**: Adding EIR improves All-Pass@4 from 8.6 to 12.1 (+3.5) but leaves OSWorld mostly unchanged; adding FDE improves OSWorld from 18.1 to 19.6 but only adds 0.5 to All-Pass@4. Joint expansion secures both benefits.
*   **Data distribution is more critical than volume**: 100k human reflection samples only improve OSWorld success by 0.6 points, whereas 100k policy-induced RoTS reflection samples improve it by 5.3 points. The bottleneck is error distribution alignment, not sample count.
*   **Optimal $\lambda_{\text{ref}}$ exists**: Reflection data at 0.1 ratio is optimal. Values $>0.2$ perform worse than $\lambda_{\text{ref}}=0$, suggesting "over-reflection" leads the model to hallucinate errors in correct rollouts.
*   **Scalability**: Performance improves as the number of expansion rounds increases from 0 to 32. Scaling data from 50k to 1000k shows saturation at 36.4% success on OSWorld, attributed to the fixed budget of $N=4$ and 32 rounds.

## Highlights & Insights
*   Reframes GUI agent robustness from "adding a reflection sub-agent" to a "training data distribution alignment" problem. Since the synthesizer comprises the policy and environment interaction, it naturally matches the policy-induced error distribution.
*   The dual design of UCB-based exploration on success subtrees and experience-based rollback on failure subtrees is elegant. A single tree handles both error generation and correction, sharing prefixes across branches to maximize sampling efficiency.
*   Treating error depth as a first-order variable in evaluation provides the first quantifiable measurement for long-horizon recovery, similar to how CoT evaluation was decoupled from single-step accuracy.

## Limitations & Future Work
*   Current coverage is limited to Desktop OS (Ubuntu/Windows 11). Mobile and edge GUI tasks involve different visual structures and state spaces that may challenge the snapshot-replay assumption.
*   The benchmarking protocol requires injecting unified prefixes into various agent-native action spaces, which may introduce conversion noise.
*   Training is strictly SFT. This lacks a closed-loop online RL flywheel, meaning the model cannot iterate once it exceeds the capability of the initial synthesisers.
*   High infrastructure cost: 800k samples required 32 A100 GPUs, 120-way parallelism, and extensive API usage, making full reproduction difficult for the open-source community.

## Related Work & Insights
*   **vs GUI-Reflection / RedTeamCUA**: Those focus on external or adversarial noise, while RoTS measures recovery from the agent's own errors across controlled depths.
*   **vs AgentNet / AgentTrek**: Prior works use human-derived reflection data which tend to be low-level. RoTS uses policy-induced errors, achieving superior data efficiency (Table 5 shows a 6-point OSWorld gain with the same sample size).
*   **vs Agent S / UFO**: These use inference-time reflection modules. RoTS embeds robustness into the model weights, reducing inference overhead at the cost of training complexity.

## Rating
*   Novelty: ⭐⭐⭐⭐
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐
*   Writing Quality: ⭐⭐⭐⭐
*   Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] HATS: Hardness-Aware Trajectory Synthesis for GUI Agents](../../CVPR2026/llm_agent/hats_hardness-aware_trajectory_synthesis_for_gui_agents.md)
- [\[ACL 2026\] Robust Tool Use via Fission-GRPO: Learning to Recover from Execution Errors](../../ACL2026/llm_agent/robust_tool_use_via_fission-grpo_learning_to_recover_from_execution_errors.md)
- [\[ICML 2026\] Rule2DRC: Benchmarking LLM Agents for DRC Script Synthesis with Execution-Guided Test Generation](rule2drc_benchmarking_llm_agents_for_drc_script_synthesis_with_execution-guided_.md)
- [\[ICML 2026\] Scaling, Benchmarking, and Reasoning of Vision-Language Agents for Mobile GUI Navigation](scaling_benchmarking_and_reasoning_of_vision-language_agents_for_mobile_gui_navi.md)
- [\[ICML 2026\] SE-GA: Memory-Augmented Self-Evolution for GUI Agents](se-ga_memory-augmented_self-evolution_for_gui_agents.md)

</div>

<!-- RELATED:END -->
