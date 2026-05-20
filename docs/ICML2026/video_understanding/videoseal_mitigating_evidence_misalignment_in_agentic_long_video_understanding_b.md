---
title: >-
  [Paper Note] VideoSEAL: Mitigating Evidence Misalignment in Agentic Long Video Understanding by Decoupling Answer Authority
description: >-
  [ICML 2026][Video Understanding][Evidence misalignment] VideoSEAL identifies a prevalent "correct answer without seeing evidence" misalignment in existing agentic long video QA systems…
tags:
  - "ICML 2026"
  - "Video Understanding"
  - "Evidence misalignment"
  - "planner-inspector decoupling"
  - "inspector gate"
  - "GRPO"
  - "temporal/semantic groundedness"
date: 2026-05-08
content_hash: 488a94e9b9045358
---

# VideoSEAL: Mitigating Evidence Misalignment in Agentic Long Video Understanding by Decoupling Answer Authority

**Conference**: ICML 2026  
**arXiv**: [2605.12571](https://arxiv.org/abs/2605.12571)  
**Code**: <https://github.com/Echochef/VideoSEAL>  
**Area**: Video Understanding / Agentic RL / Long Video QA  
**Keywords**: Evidence misalignment, planner-inspector decoupling, inspector gate, GRPO, temporal/semantic groundedness

## TL;DR
VideoSEAL identifies a prevalent "correct answer without seeing evidence" misalignment in existing agentic long video QA systems, attributing the root cause to "coupled agents conflating planning and answering authority." It proposes a planner-inspector decoupling framework: the planner is responsible for long-horizon evidence search, while the inspector holds exclusive answering authority and only permits answers when pixel-level evidence is sufficient. On LVBench, accuracy improves from 48.2% to 55.1% (↑20.5%), and on LongVideoBench from 52.2% to 62.0%.

## Background & Motivation
**Background**: Long video QA (LVU) is much more challenging than short video QA—evidence is sparse and temporally dispersed, with most video content irrelevant to the question. The mainstream approach is the agentic paradigm: a monolithic planner iteratively retrieves candidate segments, invokes tools to check visual evidence, and outputs an answer after multiple rounds of interaction. Representative methods include VideoAgent, DrVideo, Video-MTR, GenS, Conan, etc.

**Limitations of Prior Work**: Diagnostic experiments reveal a hidden but widespread failure mode—"evidence misalignment": the agent's final answer is correct, but its trace does not provide sufficient supporting evidence. In other words, the agent is "guessing correctly" rather than "answering correctly because it saw the evidence." This undermines the agent's verifiability and interpretability, and suggests that part of the so-called SOTA accuracy is achieved by relying on prior knowledge rather than genuine evidence.

**Key Challenge**: Two diagnostic metrics reveal the issue—(i) Reward Pressure (training): outcome-only reward incentivizes correct answers, so agents find it more efficient to exploit parameterized priors than to diligently search for evidence; (ii) Prompt Pressure (inference): as traces grow longer and noisier, the planner is forced to make decisions in a shared context, shifting from "evidence search" to "evidence fitting," and ultimately reverting to generic plausibility templates. Both stem from the same structural cause: the coupled agent conflates "long-horizon planning" and "final answering authority" in a shared context.

**Goal**: (i) Formalize "evidence misalignment" and provide temporal/semantic grounding diagnostic metrics; (ii) Eliminate both reward and prompt pressures via architectural decoupling; (iii) Improve both accuracy and grounding across four major long video benchmarks.

**Key Insight**: The key insight is that "answering authority" is a structural resource—whoever holds it is shaped by both pressures. By transferring answering authority from the planner to an inspector that only sees raw visual evidence (not the lengthy trace), and only allows answers when evidence is sufficient (otherwise requiring the planner to continue searching), both types of misalignment can be structurally broken.

**Core Idea**: Decompose the monolithic agent into a "planner" (responsible for tool invocation/evidence search, only seeing structured search memory) and an "inspector" (frozen MLLM, only seeing the currently submitted pixel evidence, with exclusive termination and answering authority), training only the planner with GRPO, and using the inspector gate as a pluggable module.

## Method
The VideoSEAL method consists of four parts: "diagnosis → architecture → tools → training," forming a complete logical chain.

### Overall Architecture
Input: long video $\mathcal{V}$ and query $q$. The system comprises two roles: planner $P$ (LLM) and inspector $I$ (frozen MLLM, accessed via the `VisualInspect` tool). At each round $t$, the planner, based on the query and search memory $h_{t-1}$, produces a rationale-action pair $(r_t,u_t)\sim P(\cdot\mid h_{t-1},q)$; the environment returns observation $o_t$; the inspector evaluates the evidence $v_t=E(o_t)$: $(z_t,f_t)\sim I(\cdot\mid v_t,q)$, where $z_t\in\{0,1\}$ is the sufficiency verdict and $f_t$ is feedback. Only when $z_t=1$ does the inspector output the final answer $\hat a_t$; otherwise, the planner continues searching. This inspector gate is the core of the architecture.

The toolset includes: (i) Offline indexing splits the video into 16s clips, uses Qwen3-VL-8B for captioning and text-embedding-3-large for dense embedding to build the index; (ii) `VisualRetrieve` uses cosine similarity to retrieve top-$k$ candidates and DeepSeek-V3.2 for caption filtering to reduce semantic drift; (iii) `VisualInspect(v_t,q)` is the inspector interface, returning $(z_t,f_t)$ and candidate answer $\hat a_t$.

### Key Designs

1. **Evidence Misalignment Diagnosis (temporal + semantic groundedness):**

    - **Function**: Provides two complementary metrics to quantify "correct answer without seeing evidence," separating "accuracy" from "evidence support."
    - **Mechanism**: Defines outcome correctness $C\in\{0,1\}$ and trace groundedness $G\in\{0,1\}$, focusing on the "correct but ungrounded" quadrant $(C=1,G=0)$. Temporal groundedness $G_t=\mathbb{I}[\max_{\tau\in\mathcal{E}(\xi),\tau^*\in\mathcal{E}^*}\mathrm{tIoU}(\tau,\tau^*)\ge\gamma]$ ($\gamma=0.05$) checks if the agent actually accessed the relevant time interval; semantic groundedness $G_s=1-J_{\text{judge}}(q,\xi,\hat a)$ uses an LLM judge to verify if the answer is logically supported by the trace's tool outputs. Corresponding hallucination rates: $H_t=\mathbb{P}(G_t=0\mid C=1)$, $H_s=\mathbb{P}(G_s=0\mid C=1)$.
    - **Design Motivation**: Existing evaluations only consider accuracy, failing to expose the dangerous shortcut of prior-based guessing; these two metrics audit traces from both "temporal access" and "semantic support," enabling quantifiable diagnosis of structural reward/prompt pressures.

2. **Planner-Inspector Decoupled Architecture + Inspector Gate:**

    - **Function**: Transfers answering authority from the planner, making verification depend on "actually visible pixel evidence" rather than "accumulated long trace."
    - **Mechanism**: The planner is an LLM-only policy, maintaining only "compact search memory" (including submitted spans and inspector feedback), and at each round chooses between retrieving a new span or submitting a group of spans to the inspector. The inspector is a frozen MLLM, each invocation only seeing $(q,v_t)$ and not any intermediate reasoning or full history from the planner, outputting $(z_t,f_t)$; only when $z_t=1$ is the final answer $\hat a_t$ output, otherwise the planner uses $f_t$ (e.g., "evidence lacks X information") to adjust its search. Thus, verification is entirely driven by actual visual evidence, preventing the planner from feeding "accumulated guesses" back to itself as "evidence."
    - **Design Motivation**: Diagnostics show prompt pressure arises from "long traces forcing the planner to make decisions," and reward pressure from "outcome reward teaching the planner to exploit priors"; both stem from the planner's dual role as planner and answerer. After decoupling, the planner has no answering authority and cannot avoid evidence search via priors; the inspector, not seeing the trace, is immune to long-context pressure, structurally closing both shortcuts. The inspector is a frozen, pluggable module, allowing stronger MLLMs to be swapped in at inference without retraining the planner.

3. **GRPO + Evidence-Gated Reward for Planner Training:**

    - **Function**: Freezes the inspector and only optimizes the planner's long-horizon search behavior, avoiding contamination of the verification/answer module during training; reward design encourages access to evidence-aligned intervals.
    - **Mechanism**: During training, only the planner is trained with GRPO. Two reward designs: (i) Outcome-only $R_{\text{ans}}(\xi)=\mathbb{I}[\hat a=a^*]$ as baseline; (ii) Evidence-Gated $R_{\text{evd}}(\xi)=R_{\text{ans}}(\xi)\cdot g_{\text{evd}}(\xi)$, where the soft gate $g_{\text{evd}}(\xi)=\min\{1,\tfrac{1}{\gamma}\max_{\tau\in\mathcal{E}(\xi),\tau^*\in\mathcal{E}^*}\mathrm{tIoU}(\tau,\tau^*)\}$ smoothly rewards "the more aligned the access, the better," with $\gamma$ set as the mean max-tIoU of the training set for normalization. Since actual tIoU averages only $\approx 0.05$, a hard gate is too sparse, while the soft gate provides usable gradients.
    - **Design Motivation**: Architectural decoupling alone is insufficient to resolve reward pressure (even with the inspector gate, the planner might learn a lazy strategy of "submitting random evidence and letting the inspector handle it"); evidence-gated reward aligns the protocol with "find the right place before the inspector checks," teaching the planner that "to get the inspector to open the gate, I must truly find key evidence." The frozen inspector ensures training does not contaminate the verification module, keeping the inspector gate trustworthy under policy drift.

### Loss & Training
The GRPO objective is applied only to the planner $P$, with the inspector $I$ kept frozen throughout. On datasets with ground-truth temporal intervals (e.g., CG-Bench), $R_{\text{evd}}$ is used; otherwise, it falls back to $R_{\text{ans}}$. Each candidate inspection window contains up to 64 frames (see the Frames column in the main table), and the search budget $K$ can be adjusted to balance accuracy and cost.

## Key Experimental Results

### Main Results
On four LVU benchmarks (MLVU, VideoMME w/o sub, LongVideoBench, LVBench), compared to the coupled baseline with the same backbone (Qwen3-8B planner + Qwen2.5-VL-7B inspector, 64 frames/inspection):

| Framework | Answer Authority | MLVU | VideoMME | LongVideoBench | LVBench |
|-----------|------------------|------|----------|----------------|---------|
| Qwen2.5-VL-Instruct (single MLLM, 64f) | Model | 63.9 | 58.4 | 55.3 | 34.6 |
| VideoAgent (coupled, GPT-4o) | LLM | 55.8 | 59.4 | 50.3 | 42.3 |
| Video-MTR (coupled, MLLM) | MLLM | 58.4 | 62.7 | 57.3 | 42.0 |
| Coupled baseline (Ours, same backbone) | LLM | 64.6 | 59.9 | 52.2 | 48.2 |
| **VideoSEAL (decoupled)** | **MLLM (inspector)** | **68.2** (↑4.3) | **62.9** (↑4.5) | **62.0** (↑6.7) | **55.1** (↑20.5) |

With identical backbones, simply switching to the decoupled architecture yields 4–10+ point improvements across all four benchmarks, with LVBench jumping from 48.2 to 55.1 (relative ↑20.5%).

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|------------|-------|
| Full VideoSEAL | Optimal | Planner-inspector decoupling + GRPO + soft evidence gate |
| w/o Inspector Gate (coupled) | Significant drop | Reverts to monolithic paradigm, both prompt and reward pressure return |
| Outcome-only reward (no evidence gate) | Slight accuracy drop + significant grounding decline | Validates the presence of reward pressure |
| Increased search budget $K$ | Accuracy increases monotonically | Decoupled scaling is sustainable, coupled baseline plateaus |
| Inspector 7B → 72B | Accuracy jumps significantly | Modular and pluggable, no need to retrain planner |

LVBench grounding evaluation (Table 2) shows that VideoSEAL not only surpasses LongVT in $R@\{0.05,0.10,0.20\}$, but also significantly reduces temporal hallucination $H_t$, increases semantic groundedness $G_s$, and decreases $H_s$, confirming that "as answers improve, traces also become more evidence-supported."

### Key Findings
- Coupled agents improve accuracy with training, but grounding growth stagnates, and the outcome-grounding gap widens (Figure 1). This is direct evidence of reward pressure: the model "learns to answer correctly" rather than "learns to find evidence."
- During inference, as the trace lengthens, $G_t$ quickly saturates while $G_s$ monotonically decreases and $H_s$ increases (Figure 2). This indicates that agents increasingly rely on hedging templates like "might suggest" in later stages (see Figure 3), confirming the existence of prompt pressure.
- After decoupling, increasing the search budget consistently benefits the model (Figure 6a), while the coupled baseline plateaus due to context saturation; upgrading the inspector from 7B to 72B directly boosts accuracy (Figure 6b), demonstrating that the inspector can independently scale as a verification module.
- Even though the hard gate in evidence-gated reward is too sparse ($\gamma\approx 0.05$), the soft gate $\min\{1,\mathrm{tIoU}/\gamma\}$ provides a sufficiently stable training signal, which is a key engineering trick for making grounding-aware RL work.

## Highlights & Insights
- Treating "answering authority" as an architecturally controllable resource is the paper's deepest insight. Both prompt pressure (inference) and reward pressure (training) stem from "the same model being responsible for both evidence search and answering." By stripping answering authority and assigning it to a module that only sees evidence, the agent effectively installs a "fact checker," institutionalizing "no evidence, no answer" at the architectural level. This principle can be generalized to any agent system: RAG, code agents, and tool-use agents can all introduce an "inspector gate" for final adjudication.
- The dual metrics of temporal and semantic groundedness provide actionable tools to show that "accuracy isn't everything." By separating answer correctness from evidence support, evaluation shifts from "win/loss" to "why win," and such grounding-aware evaluation should become standard for agentic systems.
- The frozen inspector enables the verification module to scale independently without retraining the planner, exemplifying modular agent design. As stronger MLLMs emerge, the system can upgrade its verification capability at zero training cost, which is highly valuable in practice.
- Using the soft evidence gate $\min\{1,\mathrm{tIoU}/\gamma\}$ to address sparse reward is a general trick: when hard signals are too sparse, use a normalized soft surrogate. This can be applied to any RL task with sparse ground-truth alignment.

## Limitations & Future Work
- Decoupling increases the overhead of each inspector invocation, resulting in higher per-decision latency and total token consumption compared to the coupled approach; the authors do not provide end-to-end latency/cost comparisons.
- The inspector being fully frozen means it can only make judgments within its distribution; for cases where the inspector itself errs (e.g., visual details beyond its capability), no amount of evidence search by the planner can compensate, and evidence-gated reward cannot fix verifier errors.
- Evidence-gated reward relies on ground-truth temporal interval annotations (e.g., CG-Bench), making it inapplicable to LVU datasets lacking grounding labels; in such cases, the soft gate must fall back to outcome-only reward.
- The LLM judge used for semantic groundedness may introduce bias or jailbreak risks; the reliability of this evaluation tool is not thoroughly discussed.
- Experiments are mainly conducted on multiple-choice QA benchmarks; effectiveness for open-ended generation tasks (e.g., long-form summarization, video captioning) remains to be validated, as "evidence support" is less clearly defined in these tasks.

## Related Work & Insights
- **vs VideoAgent / DrVideo (coupled)**: These use a single planner for serial planning and answering in a shared context, inevitably suffering from both prompt and reward pressure; VideoSEAL eliminates both pressures at the source via architectural decoupling.
- **vs Video-MTR / LongVT (RL trained, coupled)**: Also use RL for multi-round interaction, but do not separate planner and answerer roles; outcome-only reward directly reinforces "the cheapest path to the answer," leading to shortcut learning. VideoSEAL restricts training objectives to the planner's search behavior and uses the evidence gate to realign towards "finding evidence."
- **vs GenS / FrameThinker / Conan (coupled MLLM)**: These combine inspection and answering in a single MLLM, with no independent inspector or decoupled answering authority; VideoSEAL extracts the inspector as a pluggable module.
- **vs Self-RAG / verifier approaches in RAG**: Conceptually similar, but VideoSEAL architecturally structures "retrieval-reasoning-verification" in the video temporal domain, elevating the verifier to an "inspector with veto power," and quantifies effects via grounding metrics.
- **Insights**: (i) Any agent system should carefully consider the allocation of "answering authority," as misallocation introduces both reward and prompt misalignment; (ii) Agent evaluation must consider "evidence-conclusion consistency," not just win/loss; (iii) Modular frozen verifiers offer a low-cost path for agents to continually scale inference capabilities.

## Rating
- Novelty: ⭐⭐⭐⭐ The formalization of "evidence misalignment," planner-inspector decoupling, and inspector gate represent the first systematic treatment of this issue in LVU agents, with the dual diagnostic framework for reward/prompt pressure being highly insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four major LVU benchmarks, direct comparison of coupled/decoupled with the same backbone, grounding evaluation, and dual scaling experiments for search budget and inspector backbone, forming a complete argument.
- Writing Quality: ⭐⭐⭐⭐⭐ The narrative in Section 3—diagnosis → attribution → structural remedy—is exceptionally clear, with tightly reasoned justification for decoupling, serving as a model for agentic system papers.
- Value: ⭐⭐⭐⭐⭐ The decoupled answering authority paradigm is directly transferable to any agentic system (RAG, code, tool-use agents), and the pluggable frozen inspector offers a highly practical engineering upgrade path.

## Related Papers

- [\[CVPR 2026\] VideoARM: Agentic Reasoning over Hierarchical Memory for Long-Form Video Understanding](../../CVPR2026/video_understanding/videoarm_agentic_reasoning_over_hierarchical_memory_for_long-form_video_understa.md)
- [\[CVPR 2026\] LensWalk: Agentic Video Understanding by Planning How You See in Videos](../../CVPR2026/video_understanding/lenswalk_agentic_video_understanding_by_planning_how_you_see_in_videos.md)
- [\[NeurIPS 2025\] Agentic Persona Control and Task State Tracking for Realistic User Simulation](../../NeurIPS2025/video_understanding/agentic_persona_control_and_task_state_tracking_for_realistic_user_simulation_in.md)
- [\[ACL 2026\] AgentEval: DAG-Structured Step-Level Evaluation for Agentic Workflows with Error Propagation Tracking](../../ACL2026/video_understanding/agenteval_dag-structured_step-level_evaluation_for_agentic_workflows_with_error_.md)
- [\[ICML 2026\] AReaL-SEA：自演化合成数据 + 可验证奖励 RL，把多轮工具调用 Agent 后训到 SOTA](from_self-evolving_synthetic_data_to_verifiable-reward_rl_post-training_multi-tu.md)

</div>

<!-- RELATED:END -->
