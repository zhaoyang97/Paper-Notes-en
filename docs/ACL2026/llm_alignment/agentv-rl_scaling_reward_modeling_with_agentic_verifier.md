---
title: >-
  [Paper Note] AgentV-RL: Scaling Reward Modeling with Agentic Verifier
description: >-
  [ACL 2026][LLM Alignment][Agentic Verifier] The reward model is reshaped from "single-round scoring" into a multi-round deliberation process involving "forward + backward dual agents + tool calls." This multi-agent capab…
tags:
  - "ACL 2026"
  - "LLM Alignment"
  - "Agentic Verifier"
  - "Reward Model"
  - "Test-Time Scaling"
  - "Tool-augmented Reasoning"
  - "GRPO"
date: 2026-05-08
content_hash: e6f8158264445cf1
---

# AgentV-RL: Scaling Reward Modeling with Agentic Verifier

**Conference**: ACL 2026  
**arXiv**: [2604.16004](https://arxiv.org/abs/2604.16004)  
**Code**: Available (GitHub)  
**Area**: LLM Agent / Reward Modeling  
**Keywords**: Agentic Verifier, Reward Model, Test-Time Scaling, Tool-augmented Reasoning, GRPO

## TL;DR
The reward model is reshaped from "single-round scoring" into a multi-round deliberation process involving "forward + backward dual agents + tool calls." This multi-agent capability is distilled into a single 4B model via SFT+GRPO, achieving a 25.2% improvement in Best-of-N (BoN) selection compared to 70B-scale ORMs.

## Background & Motivation

**Background**: In complex reasoning tasks such as mathematics, Test-Time Scaling (e.g., BoN parallel sampling, iterative refinement) increasingly relies on reward models (verifiers) to select or critique candidate solutions. Mainstream approaches are categorized into three types: Outcome Reward Models (ORM: scalar output, zero explanation), Process Reward Models (PRM: step-level scalars), and Generative Reward Models (GenRM: natural language generative judgments).

**Limitations of Prior Work**: (1) **Error Propagation**: GenRMs are mostly trained using next-token objectives with datasets biased towards positive examples. When encountering "plausible-but-incorrect" solutions, they are easily misled by surface logic, leading to false positives. (2) **Lack of External Grounding**: Pure-text verifiers are prone to calculation errors in numerical, long-chain arithmetic, or knowledge-intensive tasks, making them unable to verify independently.

**Key Challenge**: Single-round text reasoning must simultaneously perform "logical chain review" and "numerical/factual verification." The former can be contaminated by incorrect premises, while the latter often fails due to the LLM's inherent arithmetic weaknesses—creating a natural conflict between the two tasks.

**Goal**: Upgrade reward modeling from "one-time scoring" to "multi-round, bidirectional, tool-augmented review" similar to how humans perform proofs, and train a single model to possess this integrated capability.

**Key Insight**: Drawing inspiration from the "sufficiency + necessity" bidirectional check in mathematical proofs—one agent reasons from premise to conclusion to check sufficiency, while another reasons backward from the conclusion to the premise to check necessity. Both agents are permitted to use a Python interpreter for computation. These two paths are complementary and typically expose errors overlooked by the other.

**Core Idea**: Replace single-round GenRM with "dual agents $\times$ multi-turn ReAct $\times$ code interpreter," then distill this multi-agent process into a single LLM through "synthetic trajectories + rejection sampling SFT + GRPO."

## Method

### Overall Architecture

During inference: Given a question $x$ and a candidate solution $y$, the verifier $\pi_\psi$ initiates two agents. The Forward agent follows a "Plan → Validate → Verdict" sequence, decomposing the solution into atomic sub-steps, verifying each with code, and providing a binary judgment. The Backward agent uses the same three-stage template but starts from the final answer to check if all problem constraints are satisfied. The verdicts from both paths are aggregated into a final confidence score; for BoN, the candidate with the highest score is selected. During training: Rejection sampling SFT is first performed on synthetic data to teach the model ReAct + tools, followed by GRPO to further unlock reasoning potential.

### Key Designs

1.  **Bidirectional agent verification (Forward + Backward Verifier)**:
    - **Function**: Performs complementary checks for sufficiency and necessity on the same solution, covering failure modes often missed by unidirectional verifiers.
    - **Mechanism**: The Forward agent traverses atomic steps $\Pi = \{v_1, \ldots, v_n\}$ from premise to conclusion to check logical sufficiency between adjacent steps. The Backward agent reasons from the answer back to the problem statement to verify if all constraints were utilized and if there are implicit omissions. Both share the "Plan / Validate / Verdict" three-stage prompt template. Token logits of the two verdicts are finally aggregated as the comprehensive confidence score.
    - **Design Motivation**: Pure forward review can be deceived by "seemingly self-consistent but constraint-evading" proofs; backward checking specifically uncovers such cases. The two-way complementarity avoids blind spots from a single perspective.

2.  **Multi-turn ReAct + Tool-augmented verification (Tool-augmented Multi-turn Validation)**:
    - **Function**: Allows the verifier to call a Python interpreter to calculate values, enumerate cases, or verify equations while reviewing each step, compensating for the LLM's arithmetic weaknesses.
    - **Mechanism**: The execution trajectory $\mathcal{H} = (s_0, a_0, o_0, \ldots, s_t, a_t, o_t)$ is generated during the Validate phase, where $s$ is thought, $a$ is a code action, and $o$ is the interpreter's return. Action segments are wrapped in special tokens to exclude observation gradients during training. A typical problem involves 5-6 rounds of thought and approximately one tool call (see Table 5).
    - **Design Motivation**: When reviewing competition-level problems like AIME, the critical bottleneck is often "whether this equation actually holds." Relying on a tool for a definitive check is far more reliable than LLM internal calculation.

3.  **AgentV-RL Training Recipe (Synthetic Trajectory SFT + GRPO)**:
    - **Function**: Distills the multi-agent paradigm into a single 4B model and unlocks deeper reasoning via RL.
    - **Mechanism**: First, $k=8$ candidate solutions are sampled from datasets like Polaris, DeepScaleR, and AReaL-boba (filtering out trivial cases). The LLM role-plays as the forward or backward agent to generate verification trajectories, keeping only those where the verdict matches the ground truth to form $\mathcal{D}_{\text{sft}}$ (15K trajectories). The SFT loss applies NLL to all non-observation tokens: $\mathcal{L} = -\mathbb{E}_\tau[\sum_i \mathbb{I}[\tau_i \neq o_i] \log \pi_\theta(\tau_i \mid \mathcal{H}_{<i})]$. Subsequently, GRPO is run on 50K samples with a reward $r(\mathcal{H}) = 1$ if the verdict is correct, otherwise $-1$. Dynamic filtering in a DAPO style is used to remove zero-variance groups (all +1 or all -1).
    - **Design Motivation**: Direct multi-agent deployment is costly; distillation into a single model is necessary for practical application. SFT instills the ReAct behavior pattern, while GRPO allows the model to autonomously explore optimal tool usage and reasoning paths.

### Loss & Training

The GRPO objective is formulated as:
$$\mathcal{J}_{\mathrm{GRPO}}(\psi) = \mathbb{E}\big[\frac{1}{G}\sum_i \frac{1}{|\mathcal{H}_i|} \sum_t \min(r_{i,t}\hat{A}_{i,t}, \mathrm{clip}(r_{i,t}, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}})\hat{A}_{i,t}) - \beta D_{\mathrm{KL}}(\pi_\psi \| \pi_{\mathrm{ref}})\big]$$
Mixed sampling allows the same model to play both forward and backward agent roles. To prevent memoization of environment observations, interpreter execution results are explicitly masked during loss calculation.

## Key Experimental Results

### Main Results

| Model | MATH500@128 | GSM8K@128 | Gaokao2023@128 | AIME24@128 |
| :--- | :--- | :--- | :--- | :--- |
| Qwen3-4B-Think (base) | 72.4 | 92.2 | 51.9 | 36.7 |
| INF-ORM-Llama3.1-70B | 55.4 | 91.5 | 44.4 | 40.0 |
| Qwen2.5-Math-PRM-7B | 70.2 | 95.4 | 54.3 | 46.7 |
| Skywork-V2-Llama-8B | 53.8 | 87.6 | 39.7 | 36.7 |
| **Agentic-Verifier-Qwen3-4B** | **79.0** | 93.3 | **57.4** | **53.3** |

On MATH500@128, the proposed method outperforms the strongest ORM (Skywork-V2-Llama-8B at 53.8) by 25.2 percentage points; the 4B model successfully surpasses the 70B ORM.

### Ablation Study

| Configuration | MATH500 (BoN) | Description |
| :--- | :--- | :--- |
| Full (Forward + Backward + Tool) | 78.9 | Complete model |
| Forward only | ~75 | Unidirectional sufficiency check |
| Backward only | ~74 | Unidirectional necessity check |
| w/o Tool | significant drop | Performance drops without Python interpreter |
| Train-free | base +2.6 (Gaokao) | Effective via prompting without training |
| SFT only | Moderate | Only SFT without RL |
| SFT + RL (Full) | Best | Full AgentV-RL recipe |

### Key Findings
- Bidirectional agents are significantly better than unidirectional ones—the error types exposed by forward and backward paths are complementary, and removing either leads to performance drops.
- Tool usage frequency is relatively low (averaging only 1.6 Python calls per trajectory for the 4B model), yet removing tools leads to a clear drop, indicating that tools are indispensable at critical nodes.
- As $N$ in BoN increases (32 → 64 → 128), this method benefit's more, reaching 53.3% on AIME24 at $N=128$.
- Model size scaling is consistent: performance on Gaokao2023 rises monotonically from 43.9 → 49.4 → 57.4 for 0.6B → 1.7B → 4B.
- Substantial leads on LiveCodeBench (70.86) and HotpotQA (66.00) suggest the method generalizes beyond mathematics.

## Highlights & Insights
- Redefining the "Reward Model" as an "agent" represents a significant paradigm shift from the scalar/single-round PRM/GenRM frameworks toward agentic reward modeling, showing great potential.
- The bidirectional proof approach is clever: directly importing the "sufficiency + necessity" methodology from mathematical proofs into RM naturally explains why the two agents are complementary rather than redundant.
- Excluding observation gradients via token-level masking is a necessary technical detail for training ReAct-style agents; otherwise, the model memorizes environment strings instead of learning reasoning.
- The result of a 4B model outperforming a 70B ORM suggests that RMs are more deserving of inference compute than actors, as RM errors are magnified exponentially.

## Limitations & Future Work
- Multi-turn interaction and tool use increase the reasoning token count from 2560 (base) to 8349, and single-problem latency increases from 119s to 323s (A100, batch 128), which is less friendly for real-time scenarios.
- Synthetic trajectory coverage is biased toward math and code; its transferability to open-domain preferences (e.g., helpfulness, writing style) remains unverified.
- Tool use is limited to the Python interpreter; tasks requiring external knowledge (e.g., real-world fact-checking) may still result in missed detections.
- There is no explicit negotiation mechanism between the two agents; they currently score independently before aggregation, which may leave "systemic blind spots" missed by both sides.

## Related Work & Insights
- **vs GenRM (Zhang et al., 2025)**: GenRM's single-round text judgments are easily fooled by plausible-but-wrong solutions; this work resolves this via multi-turn, tools, and bidirectionality, at the cost of $3\times$ tokens and latency.
- **vs PRM (Lightman et al., 2024, etc.)**: PRM provides step-level scalar supervision but lacks interpretability and requires dense step-wise annotation; this method's verdict includes readable critiques and only requires outcome-level supervision (whether the verdict is correct).
- **vs Tool-augmented RM (Li et al., 2024)**: Existing tool-RMs use loosely coupled tool calls; this work embeds tool calls into the ReAct reasoning chain, where tool results directly inform the verification decision.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of bidirectional agents, tools, and RL is a novel paradigm in the RM field.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 4 math benchmarks + LCB + HotpotQA + scaling experiments + thorough ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation, complete technical details.
- **Value**: ⭐⭐⭐⭐ The 4B > 70B result is highly attractive for industrial deployment and opens a new direction for agentic RM.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Aligning Agents via Planning: A Benchmark for Trajectory-Level Reward Modeling](aligning_agents_via_planning_a_benchmark_for_trajectory-level_reward_modeling.md)
- [\[ICML 2026\] Reward Modeling from Natural Language Human Feedback](../../ICML2026/llm_alignment/reward_modeling_from_natural_language_human_feedback.md)
- [\[ICLR 2026\] Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy](../../ICLR2026/llm_alignment/skywork-reward-v2_scaling_preference_data_curation_via_human-ai_synergy.md)
- [\[ICML 2026\] Mitigating Reward Hacking in RLHF via Bayesian Non-negative Reward Modeling](../../ICML2026/llm_alignment/mitigating_reward_hacking_in_rlhf_via_bayesian_non-negative_reward_modeling.md)
- [\[NeurIPS 2025\] ResponseRank: Data-Efficient Reward Modeling through Preference Strength Learning](../../NeurIPS2025/llm_alignment/responserank_data-efficient_reward_modeling_through_preference_strength_learning.md)

</div>

<!-- RELATED:END -->
