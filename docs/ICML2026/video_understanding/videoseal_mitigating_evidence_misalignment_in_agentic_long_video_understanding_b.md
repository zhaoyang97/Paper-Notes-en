---
title: >-
  [Paper Note] VideoSEAL: Mitigating Evidence Misalignment in Agentic Long Video Understanding by Decoupling Answer Authority
description: >-
  [ICML 2026][Video Understanding][Evidence misalignment] VideoSEAL identifies the "evidence misalignment" problem (obtaining correct answers without seeing evidence) in existing agentic long video QA systems…
tags:
  - "ICML 2026"
  - "Video Understanding"
  - "Evidence misalignment"
  - "planner-inspector decoupling"
  - "inspector gate"
  - "GRPO"
  - "temporal/semantic groundedness"
date: 2026-05-08
content_hash: 1f038767a6cb1fc6
---

# VideoSEAL: Mitigating Evidence Misalignment in Agentic Long Video Understanding by Decoupling Answer Authority

**Conference**: ICML 2026  
**arXiv**: [2605.12571](https://arxiv.org/abs/2605.12571)  
**Code**: <https://github.com/Echochef/VideoSEAL>  
**Area**: Video Understanding / Agentic RL / Long Video QA  
**Keywords**: Evidence misalignment, planner-inspector decoupling, inspector gate, GRPO, temporal/semantic groundedness

## TL;DR
VideoSEAL identifies the "evidence misalignment" problem (obtaining correct answers without seeing evidence) in existing agentic long video QA systems, attributing the root cause to the "conflation of planning and answering authority in coupled agents." It proposes a planner-inspector decoupling framework: the planner manages long-horizon evidence search, while the inspector holds exclusive answering authority and only proceeds when pixel-level evidence is sufficient. This improves accuracy from 48.2% to 55.1% (↑20.5%) on LVBench and from 52.2% to 62.0% on LongVideoBench.

## Background & Motivation
**Background**: Long video understanding (LVU) is significantly more challenging than short video understanding due to sparse evidence, temporal dispersion, and the vast majority of content being irrelevant to the query. The current mainstream paradigm is agentic: a monolithic planner iteratively retrieves candidate clips, invokes tools to check visual evidence, and outputs an answer after multiple rounds of interaction. Representative methods include VideoAgent, DrVideo, Video-MTR, GenS, and Conan.

**Limitations of Prior Work**: Through diagnostic experiments, the authors identify a subtle but pervasive failure mode—"evidence misalignment." This occurs when the agent's final answer is correct, but its trace does not provide sufficient evidence to support it. In other words, the agent "guesses correctly" rather than "answering based on observation." This undermines verifiability and interpretability, suggesting that SOTA accuracy is partially achieved through parametric priors.

**Key Challenge**: Two diagnostic metrics reveal the cause: (i) Reward Pressure (during training): Outcome-only rewards only penalize incorrect answers, making it more efficient for the agent to take shortcuts via priors than to search for evidence; (ii) Prompt Pressure (during inference): As traces become longer and noisier, the planner is forced to make decisions within a shared context, sliding from "searching for evidence" to "fitting evidence" and resorting to general plausibility templates. Both stem from a structural cause: coupled agents conflate "long-horizon planning" and "final answering authority" within a shared context.

**Goal**: (i) Formalize "evidence misalignment" and provide temporal/semantic grounding diagnostic metrics; (ii) eliminate both reward and prompt pressures through architectural decoupling; (iii) simultaneously improve accuracy and grounding across four major long video benchmarks.

**Key Insight**: The "answering authority" is a structural resource shaped by the aforementioned pressures. By stripping this authority from the planner and giving it to an inspector that only views raw visual evidence (rather than lengthy traces), and requiring the inspector to remain silent until evidence is sufficient, both types of misalignment can be structurally broken.

**Core Idea**: The monolithic agent is decoupled into a "Planner" (responsible for tool invocation/evidence search, viewing only structured search memory) and an "Inspector" (a frozen MLLM viewing only current pixel evidence, holding exclusive termination and answering rights), with the planner trained via GRPO and the inspector gate serving as a plug-and-play module.

## Method
The VideoSEAL methodology consists of four components: "Diagnosis → Architecture → Tools → Training."

### Overall Architecture
Input: Long video $\mathcal{V}$ and query $q$. The system consists of two roles: a planner $P$ (LLM) and an inspector $I$ (frozen MLLM, accessed via the `VisualInspect` tool). In each round $t$, the planner generates a rationale-action pair $(r_t,u_t)\sim P(\cdot\mid h_{t-1},q)$ based on the query and search memory $h_{t-1}$. The environment returns an observation $o_t$, and the inspector evaluates the evidence $v_t=E(o_t)$: $(z_t,f_t)\sim I(\cdot\mid v_t,q)$, where $z_t\in\{0,1\}$ is the sufficiency judgment and $f_t$ is the feedback. Only when $z_t=1$ does the inspector output the final answer $\hat a_t$; otherwise, the planner continues searching. This inspector gate is the core of the architecture.

The toolset includes: (i) Offline indexing: Slicing videos into 16s clips, using Qwen3-VL-8B for captions and text-embedding-3-large for dense embedding; (ii) `VisualRetrieve`: Using cosine similarity to retrieve top-$k$ candidates and DeepSeek-V3.2 for caption filtering; (iii) `VisualInspect(v_t,q)`: The inspector interface returning $(z_t,f_t)$ and the candidate answer $\hat a_t$.

### Key Designs

1.  **Evidence Misalignment Diagnosis (temporal + semantic groundedness)**:
    *   **Function**: Provides two complementary metrics to quantify "correct but unsupported" answers, decoupling accuracy from evidence support.
    *   **Mechanism**: Defines outcome correctness $C\in\{0,1\}$ and trace groundedness $G\in\{0,1\}$, focusing on the $(C=1,G=0)$ "correct but ungrounded" quadrant. Temporal groundedness $G_t=\mathbb{I}[\max_{\tau\in\mathcal{E}(\xi),\tau^*\in\mathcal{E}^*}\mathrm{tIoU}(\tau,\tau^*)\ge\gamma]$ ($\gamma=0.05$) determines if the agent visited relevant time intervals. Semantic groundedness $G_s=1-J_{\text{judge}}(q,\xi,\hat a)$ uses an LLM judge to check if the answer is logically supported by tool outputs in the trace. Hallucination rates are defined as $H_t=\mathbb{P}(G_t=0\mid C=1)$ and $H_s=\mathbb{P}(G_s=0\mid C=1)$.
    *   **Design Motivation**: Evaluation based solely on accuracy fails to expose dangerous shortcuts via priors; these two metrics audit traces from both temporal and semantic perspectives.

2.  **Planner-Inspector Decoupling Architecture + Inspector Gate**:
    *   **Function**: Strips answering authority from the planner, making verification dependent on "raw pixel evidence" rather than "accumulated long traces."
    *   **Mechanism**: The planner is an LLM-only policy maintaining a compact search memory. In each round, it either retrieves a new span or submits a set of spans to the inspector. The inspector is a frozen MLLM that views only $(q,v_t)$ without access to the planner's reasoning or full history. It outputs $(z_t,f_t)$; only $z_t=1$ triggers the final answer $\hat a_t$. Otherwise, the planner uses $f_t$ (e.g., "missing information X") to adjust the search.
    *   **Design Motivation**: Diagnosis shows prompt pressure stems from long traces forcing decisions, while reward pressure stems from planners learning to guess. Decoupling ensures the planner cannot bypass evidence search, and the inspector cannot be influenced by the planner's context.

3.  **GRPO + Evidence-Gated Reward for Planner Training**:
    *   **Function**: Freezes the inspector while optimizing the planner's search behavior, preventing training contamination of the verification/answer modules.
    *   **Mechanism**: GRPO is applied only to the planner. Rewards include: (i) Outcome-only $R_{\text{ans}}(\xi)=\mathbb{I}[\hat a=a^*]$ as a baseline; (ii) Evidence-Gated $R_{\text{evd}}(\xi)=R_{\text{ans}}(\xi)\cdot g_{\text{evd}}(\xi)$, where the soft gate $g_{\text{evd}}(\xi)=\min\{1,\tfrac{1}{\gamma}\max_{\tau\in\mathcal{E}(\xi),\tau^*\in\mathcal{E}^*}\mathrm{tIoU}(\tau,\tau^*)\}$ encourages better alignment. A soft gate is necessary as average tIoU is low ($\approx 0.05$), making hard gates too sparse for gradients.
    *   **Design Motivation**: Architecture alone does not solve reward pressure (the planner might still learn to submit random snippets). Evidence-gated reward aligns the search protocol, forcing the planner to find key evidence to "unlock" the inspector's gate.

### Loss & Training
The GRPO objective applies only to planner $P$; inspector $I$ remains frozen. $R_{\text{evd}}$ is used on datasets with ground-truth temporal labels (e.g., CG-Bench), otherwise $R_{\text{ans}}$ is used. Each inspection window is limited to 64 frames, with the search budget $K$ used to balance accuracy and cost.

## Key Experimental Results

### Main Results
Compared against coupled baselines using the same backbone (Qwen3-8B planner + Qwen2.5-VL-7B inspector, 64 frames/inspection):

| Framework | Answer Authority | MLVU | VideoMME | LongVideoBench | LVBench |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Qwen2.5-VL-Instruct (Single MLLM, 64f) | Model | 63.9 | 58.4 | 55.3 | 34.6 |
| VideoAgent (coupled, GPT-4o) | LLM | 55.8 | 59.4 | 50.3 | 42.3 |
| Video-MTR (coupled, MLLM) | MLLM | 58.4 | 62.7 | 57.3 | 42.0 |
| Coupled baseline (Ours same backbone) | LLM | 64.6 | 59.9 | 52.2 | 48.2 |
| **VideoSEAL (decoupled)** | **MLLM (inspector)** | **68.2** (↑4.3) | **62.9** (↑4.5) | **62.0** (↑6.7) | **55.1** (↑20.5) |

Switching to the decoupled architecture yielded 4–10+ point gains across all benchmarks with identical backbones.

### Ablation Study

| Configuration | Key Indicator | Description |
| :--- | :--- | :--- |
| Full VideoSEAL | Optimal | Decoupling + GRPO + Soft evidence gate. |
| w/o Inspector Gate (coupled) | Strong drop | Reverts to monolithic paradigm, prompt + reward pressure return. |
| Outcome-only reward | Minor acc. drop / Grounding drop | Confirms the existence of reward pressure. |
| Increasing search budget $K$ | Monotonic gain | Decoupling enables sustainable scaling; coupled baseline plateaus. |
| Inspector 7B → 72B | Significant jump | Modular plug-and-play allows scaling without retraining the planner. |

### Key Findings
- Coupled agents improve accuracy during training while grounding remains stagnant, leading to a widening outcome-grounding gap. This proves models learn to "guess answers" rather than "find evidence."
- During inference, as traces lengthen, $G_t$ saturates while $G_s$ drops and $H_s$ rises, showing agents resort to "plausibility hedging" in late stages.
- Decoupled systems scale effectively with search budget $K$, whereas coupled baselines suffer from context saturation.
- Soft evidence gates are crucial for training grounding-aware agents when hard temporal signals are too sparse.

## Highlights & Insights
- Treating "answering authority" as a structural resource is a profound insight. Architectural decoupling creates a "fact-checker" within the agent, institutionalizing the principle of "no evidence, no answer."
- The dual temporal/semantic grounding metrics provide tools to move beyond simple accuracy. This "grounding-aware" evaluation audits why an agent "wins."
- The frozen, swappable inspector allows for zero-cost upgrades to the verification module, demonstrating the power of modular agent design.
- The use of soft surrogates for sparse signals is a generalizable trick for any RL task aligning with sparse ground truth.

## Limitations & Future Work
- Decoupling increases overhead (latency and tokens) due to separate inspector calls.
- The inspector is frozen; if the inspector itself lacks the capability to see details, the planner cannot compensate.
- Training depends on ground-truth temporal annotations, which are limited in many datasets.
- The LLM judge for semantic groundedness may introduce its own biases.
- Effectiveness on open-ended video generation remains to be validated.

## Related Work & Insights
- **vs VideoAgent/DrVideo**: These use serialized planning and answering in a shared context, suffering from both prompt and reward pressure.
- **vs Video-MTR/LongVT**: Use RL but lack role separation; rewards strengthen the "cheapest path to the answer," often via priors.
- **vs verifiers in RAG**: Similar in concept, but VideoSEAL upgrades the verifier to an "inspector with veto power" for temporal video contexts.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] VideoTemp-o3: Harmonizing Temporal Grounding and Video Understanding in Agentic Thinking](videotemp-o3_harmonizing_temporal_grounding_and_video_understanding_in_agentic_t.md)
- [\[CVPR 2026\] VideoARM: Agentic Reasoning over Hierarchical Memory for Long-Form Video Understanding](../../CVPR2026/video_understanding/videoarm_agentic_reasoning_over_hierarchical_memory_for_long-form_video_understa.md)
- [\[CVPR 2026\] StreamReady: Learning What to Answer and When in Long Streaming Videos](../../CVPR2026/video_understanding/streamready_learning_what_to_answer_and_when_in_long_streaming_videos.md)
- [\[ICML 2026\] Video-MTR: Reinforced Multi-Turn Reasoning for Long Video Understanding](video-mtr_reinforced_multi-turn_reasoning_for_long_video_understanding.md)
- [\[ICML 2026\] Foresee-to-Ground: From Predictive Temporal Perception to Evidence-Driven Reasoning](foresee-to-ground_from_predictive_temporal_perception_to_evidence-driven_reasoni.md)

</div>

<!-- RELATED:END -->
