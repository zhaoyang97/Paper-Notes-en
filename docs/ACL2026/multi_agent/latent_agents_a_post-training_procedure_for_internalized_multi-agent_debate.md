---
title: >-
  [Paper Note] Latent Agents: A Post-Training Procedure for Internalized Multi-Agent Debate
description: >-
  [ACL 2026][Multi-Agent][Multi-agent debate] This paper proposes the IMAD (Internalized Multi-Agent Debate) framework. Through a two-stage post-training process of SFT + GRPO…
tags:
  - "ACL 2026"
  - "Multi-Agent"
  - "Multi-agent debate"
  - "Knowledge distillation"
  - "Activation steering"
  - "Agent subspace"
  - "GRPO"
date: 2026-05-08
content_hash: 553310274769956d
---

# Latent Agents: A Post-Training Procedure for Internalized Multi-Agent Debate

**Conference**: ACL 2026  
**arXiv**: [2604.24881](https://arxiv.org/abs/2604.24881)  
**Code**: https://github.com/johnsk95/latent_agents  
**Area**: LLM Agent / Multi-Agent / Interpretability  
**Keywords**: Multi-agent debate, Knowledge distillation, Activation steering, Agent subspace, GRPO

## TL;DR
This paper proposes the IMAD (Internalized Multi-Agent Debate) framework. Through a two-stage post-training process of SFT + GRPO, multi-agent debate is "internalized" into a single LLM. This reduces token consumption by up to 93% and demonstrates through activation steering that the internalized model maintains separable and controllable "agent subspaces" within the latent space.

## Background & Motivation

**Background**: Multi-Agent Debate (MAD) has been widely proven to improve the reasoning accuracy of LLMs and reduce hallucinations. This is achieved by having multiple LLM instances criticize and refine each other's answers across multiple conversation rounds, finally reaching a conclusion via voting.

**Limitations of Prior Work**: The inference cost of MAD is substantial—a typical setup (3 agents × 2 rounds) generates tens of thousands of tokens in dialogue logs to produce a final answer, with single-inference costs reaching 5–16 times that of a single agent. Existing distillation work (e.g., DebateGPT, Subramaniam et al.) only fine-tunes on the "final consensus answer," failing to inherit the gains from intermediate multi-perspective reasoning.

**Key Challenge**: The reasoning gains of MAD stem from the **process** of "multi-perspective collision + iterative refinement," not just the final conclusion; however, retaining the process incurs token costs, while removing it sacrifices performance gains—a fundamental trade-off exists between performance and efficiency.

**Goal**: (1) Distill the complete MAD debate process (not just the conclusion) into a single LLM; (2) Verify whether internalization truly embeds "multiple agents" into the model's latent space or merely memorizes input-output mappings; (3) Utilize this internalized structure for selective behavioral control (e.g., suppressing malicious or hallucinating agents).

**Key Insight**: The authors observe that if a model learns the structure of the entire debate trace (including structural tags like `<|Agent 1|>`) during the SFT stage, subsequent RL—combined with **linearly decaying format rewards** and **gradually tightening length constraints**—can force the model to compress multi-perspective reasoning from "explicit output" into "latent computation."

**Core Idea**: A two-stage post-training approach (first learning structure, then internalizing via dynamic rewards) compresses external debate into implicit reasoning. Subsequently, difference-in-means is used to extract steering vectors for each agent, proving that the internalized model forms separable "agent subspaces" that allow for the precise suppression of "bad agents" using negative steering.

## Method

### Overall Architecture

The entire pipeline consists of three stages:

1.  **Debate Data Collection**: A standard 3-agent × 2-round MAD is executed using GPT-3.5 turbo. The problems consist of arithmetic expressions with six two-digit numbers (e.g., $91+24\times 13+45-41\times 38$). Samples that fail to reach a majority consensus are filtered out. Structural tags such as `<|Agent i|>`, `<|Round j|>`, `<|Consensus|>`, and `<|endofdebate|>` are added to each trace, resulting in 944 `{Question, Trace, Answer}` triplets.
2.  **Structure Learning (SFT)**: Standard autoregressive next-token prediction is performed on the full trace (rather than just the final answer), allowing the single LLM to learn to imitate the entire debate format—actively generating multiple agent responses, iterative rounds, and the final consensus.
3.  **Internalization (RL/GRPO)**: Group Relative Policy Optimization is used for further optimization. A dual dynamic reward system—comprising "format reward weight decay" and "length limit annealing"—is introduced to gradually compress explicit debate into implicit reasoning.

The input is a problem. While the output in early training stages consists of the full debate trace with structural tags followed by the final answer, the output in late training stages provides the final answer directly (with intermediate reasoning occurring in the hidden states).

### Key Designs

1.  **Structural Tags + Full-trace SFT**:
    *   **Function**: Teaches the single LLM to imitate the structural discourse of multi-agent debate (multi-agent utterances + iterative rounds + consensus).
    *   **Mechanism**: Unlike works like DebateGPT that SFT only on the "final consensus," this work uses cross-entropy training on the entire trace with structural tags. These tags provide parsable reward hooks for subsequent RL and facilitate the latent separation of agents (ablations show that removing tags significantly reduces the separability of agent subspaces).
    *   **Design Motivation**: The SFT model "sees" all preceding agent utterances at each step (whereas in real MAD, agents are parallel and only see each other at the end of a round), making it less prone to coordination failures than explicit MAD. This is why the SFT stage alone can outperform explicit debate.

2.  **Dynamic Double Reward RL (Format Weight Decay + Length Annealing)**:
    *   **Function**: Gradually compresses "explicitly generated debate" into "latent reasoning" while ensuring answer correctness.
    *   **Mechanism**: The reward function is defined as $r(x,y)=w_{fmt}R^{fmt}+w_{clip}R(y;l)$. $R^{fmt}$ is a simple reward for structural tag matching (encouraging the retention of the debate format in early stages), with its weight $w_{fmt}$ linearly decaying from 1.0 to 0.05. $R(y;l)$ is a length-clipping correctness reward: $R(y;l)=1$ if and only if the correct answer $y^*$ appears within the first $l$ tokens of $y$, and 0 otherwise. The length limit $l$ anneals from 2000 down to 500, making it impossible to output a full debate while keeping the answer within the first 500 tokens. The only viable strategy is to move multi-perspective reasoning into the latent space.
    *   **Design Motivation**: Directly applying a tight $l$ would leave the model no room to explore reasoning space initially. By gradually tightening $l$ and phasing out the format reward, the dual dynamic signals push the model from an "explicit debater" to an "implicit debater." This design draws inspiration from Hou et al. (2025) for internalizing long CoT but is the first application in a multi-agent context.

3.  **Difference-in-Means Based Agent Steering Vectors**:
    *   **Function**: Explicitly extracts what each agent "looks like" after internalization as additive vectors in the latent space for probing interpretation and behavioral control.
    *   **Mechanism**: Using Contrastive Activation Addition (CAA), positive samples (debate history followed by agent $i$'s actual response) and negative samples (history followed by the average activation of the other two agents' responses) are constructed for each agent $i$. The difference-in-means is calculated at layer $\ell$: $\mathbf{v}_i = \frac{1}{|\mathcal{D}|}\sum_{p,c\in\mathcal{D}}(\mathbf{h}_\ell(p,c_i) - \mathbf{h}_\ell(p,c_{\neg i}))$. During inference, $\alpha\cdot \mathbf{v}_i$ is added to the hidden states; a positive $\alpha$ amplifies the agent's characteristics, while a negative $\alpha$ suppresses them. Steering vectors are extracted from the SFT checkpoint to avoid artifacts introduced by RL stage optimization.
    *   **Design Motivation**: To answer whether the multi-agent structure truly exists or is compressed into undifferentiated reasoning, the most direct way is to check for agent-specific directions in the latent space. Separable directions also provide a handle for behavioral control—identifying and negatively steering the direction of a malicious agent allows for selective suppression of bad behavior without destroying general capabilities.

### Loss & Training
SFT uses standard next-token CE for 3–6 epochs. RL uses GRPO for 2 epochs: $k$ candidates are sampled per query and scored via $r(x,y)$, forming an on-the-fly preference dataset. $w_{fmt}$ scales from 1.0 → 0.05, and $l$ scales from 2000 → 500. LoRA is used for both stages. For the "malicious agent suppression" experiment, $R^{fmt}$ is replaced with an "ethics/honesty" reward provided by an LLM-judge during the RL stage.

## Key Experimental Results

### Main Results
Compare five settings: Single / Debate / DebateGPT / SFT / IMAD (SFT+RL) on GSM8K, MMLU-Pro, and BBH benchmarks. 1000 problems were randomly sampled per benchmark, with results averaged over 3 runs.

| Model | Method | GSM8K | MMLU-Pro | BBH | GSM8K tokens | Gain vs. Debate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LLaMA-3.1 8B | Debate | 83.03 | 64.60 | 51.06 | 5757.78 | — |
| LLaMA-3.1 8B | **IMAD** | **85.20** | 62.00 | **58.53** | 644.33 | **88.8%** |
| Qwen 2.5 7B | Debate | 91.37 | 57.67 | 67.58 | 2319.71 | — |
| Qwen 2.5 7B | **IMAD** | 89.67 | 52.87 | **70.11** | 389.13 | **83.2%** |
| Mistral Nemo 12B | Debate | 61.03 | 41.30 | 62.76 | 1696.99 | — |
| Mistral Nemo 12B | **IMAD** | **80.00** | 38.97 | **63.73** | 358.01 | **78.9%** |

Across the three models, IMAD achieves comparable or better accuracy using only 6.3%–21.1% of the tokens required by Debate. Most notably, on GSM8K with Mistral Nemo 12B, it outperforms explicit Debate by 18.97 points.

### Ablation Study
| Experiment | Configuration | Key Metrics | Description |
| :--- | :--- | :--- | :--- |
| Pipeline Ablation | SFT only | LLaMA GSM8K 79.23 / 992 tokens | SFT already exceeds most baselines but uses more tokens than IMAD. |
| Pipeline Ablation | SFT + RL (IMAD) | LLaMA GSM8K 85.20 / 644 tokens | RL stage simultaneously increases accuracy and cuts tokens by up to 66%. |
| Tag Ablation | Remove tags | Significant drop in subspace separation | Verifies the importance of tags for forming agent subspaces. |
| Agent steering | IMAD vs base, ROUGE-L AUC | IMAD is **15.41%** higher (6.1%–24.97%) | Agent 3 (PoT) shows largest gain (21–25%), proving code-like styles are most separable. |
| Evil suppression | $\alpha=-3.0\sim-5.0$ | IMAD score drops to 0; base remains 1.01 | Internalization makes malicious traits highly localized and suppressible. |
| Hallucination suppression | Within $\alpha$ range | Both models only partially suppress (baseline ≈65) | Hallucination is a distributed trait, but IMAD remains linearly controllable. |
| Task retention | GSM8K @ extreme $\alpha$ | IMAD remains stable; base collapses | IMAD's subspaces are "cleaner"; steering does not destroy primary capability. |

### Key Findings
*   **The RL stage is both the token killer and the key to performance gains**: Compared to SFT, GRPO + length annealing can cut another 66% of tokens on LLaMA while gaining 6 points, validating the efficacy of dynamic dual rewards.
*   **Strong cross-domain generalization**: Although trained only on arithmetic, performance improved simultaneously on MMLU-Pro / BBH knowledge and reasoning tasks, indicating that the "multi-perspective reasoning schema" was internalized rather than just "arithmetic ability."
*   **Personas truly exist in the latent space**: A tiny steering of $\alpha=0.5$ causes IMAD outputs to clearly manifest three styles: step enumeration (Agent 1), self-criticism (Agent 2), and code/equations (Agent 3). The base model shows mixed styles under the same steering.
*   **Trait distribution determines controllability**: "Evil" is a localized trait in the latent space and can be completely cleared via steering; "hallucination" is a distributed trait and can only be linearly weakened—providing a diagnostic framework for LLM safety research.
*   **Model capacity threshold**: Models ≤7B perform poorly with internalization; 7B+ is required to stably carry multi-agent subspaces.

## Highlights & Insights
*   **"Dynamic Reward Annealing = Explicit → Implicit Training Topology Control"**: Simultaneously annealing reward weights and length limits is an elegant way to induce a model to shift observable computation to the latent space. This is more stable than simple length penalties and can be directly transferred to scenarios like CoT compression or tool-use simplification.
*   **"Internalization creates separable subspaces" is a counter-intuitive finding**: One might fear that distillation compresses multiple perspectives into a single representation. However, the authors provide triple evidence (ROUGE AUC, persona steering, GSM8K retention) that the agent structure is truly "imprinted" into the latent space and can be read out via simple difference-in-means, opening a new path for interpretability research.
*   **"Malicious Agent Internalization → Precise Removal via Negative Steering" for safety**: Rather than searching for harmful directions in general models (which risks "collateral damage"), it is better to **actively implant** a bad agent and then surgically remove its direction. This brilliant reverse-engineering approach turns controllability issues into designable ones.

## Limitations & Future Work
*   **Simple data distribution**: Training data is limited to arithmetic expressions with a fixed 3 agent × 2 round setup. Generalization to complex debate structures and open-ended tasks requires further validation.
*   **Dependency on SFT format learning**: LLaMA stably learns the structure, but Qwen/Mistral occasionally fail in certain settings, causing poor subspace separation. The lower bound of internalization is limited by the base model's in-context schema following ability.
*   **Bias in LLM-judge evaluation**: Evil/Hallucination scores rely on GPT-4o-mini; while human-LLM consistency experiments were performed, a protocol entirely independent of self-evaluation still needs development.
*   **7B+ capacity threshold**: Small models (<7B) show no obvious benefit from internalization, likely due to a capacity bottleneck. Future work could explore MoE or smarter LoRA configurations to replicate effects on smaller scales.

## Related Work & Insights
*   **vs. Debate (Du et al., 2023)**: Classic MAD is an inference-time protocol; this is train-time distillation that absorbs the protocol into weights, drastically reducing cost.
*   **vs. DebateGPT / Subramaniam et al.**: They SFT only on the "final consensus"; this work uses the full trace + structural tags + dynamic RL, inheriting reasoning gains from the process.
*   **vs. Hou et al. (2025) Internalized CoT**: Both use length-based RL for internalization, but this work focuses on multi-agent multi-perspective reasoning + subspace interpretability, moving beyond simple compression.
*   **vs. Persona Vectors (Chen et al., 2025)**: They extract trait directions from general models; this work **implants** personas via IMAD first, yielding higher separation and more complete suppression without hurting primary capabilities.
*   **vs. Coconut (Hao et al., 2024) Continuous Latent Reasoning**: Both move reasoning to the latent space, but Coconut passes hidden states as thought tokens, while this work uses reward pressure to let the model learn implicit reasoning while maintaining a discrete token interface.

## Rating
*   Novelty: ⭐⭐⭐⭐⭐ Distilling the full structure of multi-agent debate and analyzing subspaces for safety is a rare and innovative combination.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Covering 3 models × 3 benchmarks with extensive ablation and steering analysis; however, training data is limited to arithmetic.
*   Writing Quality: ⭐⭐⭐⭐ The three-part narrative (efficiency → interpretability → controllability) is clear with well-integrated intuitive explanations.
*   Value: ⭐⭐⭐⭐⭐ Simultaneously advances efficiency, interpretability, and safety; code is open-source and reproducible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Efficient Multi-Agent System Training with Data Influence-Oriented Tree Search](efficient_multi-agent_system_training_with_data_influence-oriented_tree_search.md)
- [\[ACL 2026\] When Identity Skews Debate: Anonymization for Bias-Reduced Multi-Agent Reasoning](when_identity_skews_debate_anonymization_for_bias-reduced_multi-agent_reasoning.md)
- [\[ACL 2026\] HACHIMI: Scalable and Controllable Student Persona Generation via Orchestrated Agents](hachimi_scalable_and_controllable_student_persona_generation_via_orchestrated_ag.md)
- [\[ICML 2026\] When Cloud Agents Meet Device Agents: Lessons from Hybrid Multi-Agent Systems](../../ICML2026/multi_agent/when_cloud_agents_meet_device_agents_lessons_from_hybrid_multi-agent_systems.md)
- [\[ACL 2026\] ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination](atlas_adaptive_trading_with_llm_agents_through_dynamic_prompt_optimization_and_m.md)

</div>

<!-- RELATED:END -->
