---
title: >-
  [Paper Note] Learning Efficient Guardrails for Compliance
description: >-
  [ICML 2026][LLM Agent][Policy Compliance] This paper constructs PolicyGuardBench, a scale of 60k samples (5 domains, 733 standardized trajectories $\times$ 2195 atomic policies $\rightarrow$ 60…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "Policy Compliance"
  - "Trajectory Auditing"
  - "Guardrail Models"
  - "Web Agent"
  - "Prefix Detection"
date: 2026-05-08
content_hash: e424bdd3bda7ed2a
---

# Learning Efficient Guardrails for Compliance

**Conference**: ICML 2026  
**arXiv**: [2510.03485](https://arxiv.org/abs/2510.03485)  
**Code**: Project page: learning-efficient-guardrails-for-compliance (link provided in paper)  
**Area**: LLM Safety / Agent Compliance / Guardrail  
**Keywords**: Policy Compliance, Trajectory Auditing, Guardrail Models, Web Agent, Prefix Detection

## TL;DR
This paper constructs PolicyGuardBench, a scale of 60k samples (5 domains, 733 standardized trajectories $\times$ 2195 atomic policies $\rightarrow$ 60,000 trajectory-policy pairs including cross-subdomain and prefix-truncated settings). Based on Qwen3-4B-Instruct, the authors perform full-parameter SFT to develop a lightweight guardrail model, PolicyGuard-4B. It achieves 90.14% accuracy and 87.59% F1 at a latency of 22.5 ms/sample, matching or exceeding 70B-grade open-source models and Claude-Sonnet-4, while demonstrating strong cross-domain generalization (LODO OOD F1 $\approx$ 0.91).

## Background & Motivation

**Background**: Current autonomous Web Agents (e.g., ScribeAgent, planning/reasoning work on WebArena) can complete long-horizon tasks. However, deployment usually requires adherence to external rules—platform policies, corporate systems, ethical, and regulatory requirements. Existing guardrail research primarily follows the "safety-oriented" route: the LlamaGuard series and ShieldGemma detect prompt toxicity, jailbreaking, or dangerous code, while AGrail, ShieldAgent, and LlamaFirewall focus on OS-level attacks or formal verification.

**Limitations of Prior Work**: Empirical testing reveals that safety guardrails like LlamaGuard-3/4 and ShieldGemma are nearly unusable for policy compliance tasks—the LlamaGuard series predicts almost all inputs as the same class, with accuracy hovering between 42–58% and F1 degrading to near 0. Conversely, relying on general-purpose LLMs ($>70\text{B}$) as guardrails reaches 88–90% accuracy but incurs latency of 200–3600 ms/sample, making online intervention difficult. Furthermore, existing evaluations such as ST-WebAgentBench, SafeArena, and WebSuite observe the gap between task-completion and completion-under-policy but lack a large-scale, systematically labeled dataset covering cross-subdomain and early detection scenarios.

**Key Challenge**: The authors argue that "safety" and "policy compliance" are orthogonal dimensions. The former concerns content toxicity, jailbreaking, or irreversible disasters, while the latter concerns whether trajectories violate specific business rules (e.g., "total purchase amount not exceeding $\$200$," "must double-confirm before deletion"). Treating them as the same leads to two failures: using safety guardrails for compliance detection suffers from severe over-fitting to coarse-grained signals like toxicity; using frontier LLMs for compliance detection is unacceptable in terms of online efficiency. Additionally, violations are often cumulative (e.g., "adding one piece of cake is fine, adding a second violates the policy"), requiring guardrails to predict violations before the trajectory finishes to avoid irreversible operations.

**Goal**: (i) Establish policy-trajectory compliance as an independent task and construct a recognized large-scale benchmark; (ii) Train an accurate and fast small-scale guardrail, proving the 4B scale is sufficient; (iii) Introduce a "prefix detection" setting to quantify whether models can detect violation tendencies early in a trajectory.

**Key Insight**: The authors observe that most web agent policy violations are checkable under atomic rules if heterogeneous browser events are standardized into a unified action vocabulary (Click, Input, Scroll, etc.). Then, GPT-4o can be used to inversely synthesize "one rule per atom" policies from trajectories. By pairing policies and trajectories across different subdomains within the same domain (intentionally creating cross-subdomain negative/positive pairs), compliance detection can be transformed into a "binary classification instruction following" task without relying on RLHF, enabling small models to perform effectively.

**Core Idea**: Use a four-step pipeline—"Trajectory Standardization + Reverse Synthesis of Atomic Policies + Cross-subdomain Pairing + Prefix Truncation"—to build 60k high-quality binary data. Then, use single-task SFT to train Qwen3-4B into a specialized guardrail, outperforming 70B+ general LLMs and all existing safety guardrails at a 4B scale.

## Method

### Overall Architecture
The pipeline consists of two parts: **PolicyGuardBench (Data)** and **PolicyGuard-4B (Model)**. Data input consists of raw browser traces from ScribeAgent on WebArena (covering Reddit, Map, GitLab, Shopping-Admin, and Shopping). It passes through *Trajectory Standardization → Policy Synthesis → Trajectory-Policy Matching → Violation Annotation* to produce 733 base trajectories, 2195 policies, and 314,556 raw pairs, eventually filtered to 59,997 label-balanced pairs (42.4% violation / 57.6% compliance, 41.6% cross-subdomain). Model input is an instruction template composed of the triplet `(policy, standardized action sequence, domain metadata)`, outputting a binary label `{violation, no_violation}`. Qwen3-4B-Instruct is fine-tuned using full-parameter SFT, split 8:2 by base trajectories to ensure zero overlap.

### Key Designs

1.  **Trajectory Standardization + Atomic Policy Synthesis**:
    - **Function**: Converts raw browser events into machine-auditable "sentencized" trajectories and inversely writes 2-3 atomic/executable/clear policy rules for each trajectory.
    - **Mechanism**: Cleans noise (removes empty events/repeated rendering) and normalizes verbs (unifies to Click/Input/Scroll/Select/Navigate/Submit), standardizing objects as `link 'My Account'` or `button 'Search'`. These are serialized as "Step 1: Click link 'My Account'; ...". In policy synthesis, GPT-4o generates rules where each rule contains only one constraint (e.g., "Do not click 'Delete' without a prior confirmation step"). After manual filtering and deduplication, 2195 policies are obtained, each with a structured schema including `source_subdomain` and up to 2 `target_subdomain`.
    - **Design Motivation**: The difficulty in compliance detection lies in the extreme heterogeneity of rules and trajectories. By "atomizing + schematizing" both, the LLM annotator and the guardrail model face a unified interface.

2.  **Cross-subdomain Pairing + LLM Annotation + Human Verification**:
    - **Function**: Combines 733 trajectories and 2195 policies into 60k high-quality binary data with cross-subdomain generalization pressure.
    - **Mechanism**: Use Sentence-BERT embedding retrieval to recall candidate policies for each trajectory, supplemented by keyword triggers (e.g., `delete`/`confirm`). Pairs are filtered via heuristics and LLM scoring. Policies are combined with trajectories from their native subdomain (source) and up to 2 different subdomains (target), forcing cross-subdomain pairs (41.6% of total). Negative examples are created by randomly pairing non-violating policies within the same domain. Annotation uses gpt-oss-120B to simulate human patterns for labels and confidence scores; low-confidence flags are sent for human review. 287 independent human double-checks showed 89.8% agreement.
    - **Design Motivation**: Cross-subdomain testing checks if the guardrail learns transferable compliance patterns rather than memorizing trajectories.

3.  **Trajectory Isolation Splitting + Prefix Truncation Evaluation**:
    - **Function**: Uses strict train/test isolation to evaluate generalization and quantifies "early warning" capabilities.
    - **Mechanism**: The 8:2 split is based on 733 base trajectories rather than pairs, ensuring 0% trajectory overlap to avoid memory leakage. Prefix detection truncates violation samples to the first $N$ steps ($N=1,\dots,5$, covering roughly half of the average length of 9.3) and re-matches with policies for re-labeling. A Leave-One-Domain-Out (LODO) evaluation is also performed for OOD assessment.
    - **Design Motivation**: Violations are often irreversible, necessitating prediction at step $N$. Trajectory isolation and LODO distinguish between "trajectory pattern memorization" and "compliance pattern learning."

### Loss & Training
PolicyGuard-4B utilizes vanilla supervised instruction tuning: full-parameter SFT on Qwen3-4B-Instruct. The input is a unified prompt template of `(policy, action sequence, domain metadata)`, and the output is strictly formatted as `violation` or `no_violation`. The loss is standard next-token cross-entropy. Training was conducted on H100 80GB GPUs with temperature=0 for reproducible decoding. The goal was to prove the efficacy of a "small model + clean binary SFT" approach.

## Key Experimental Results

### Main Results: Full-trajectory Compliance Detection (PolicyGuardBench 12k Test Set)

| Model | Type | Size | Accuracy | F1 | Latency (ms/ex) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Claude-Sonnet-4 | Closed frontier | – | 0.8983 | 0.8678 | 1238 |
| Gemini-1.5-Pro | Closed frontier | – | 0.8713 | 0.8502 | 596 |
| DeepSeek-V3.1 (non-think) | Open frontier | 685B | 0.8613 | 0.8407 | 3270 |
| Llama-3.3-70B-Instruct | IT | 70B | 0.9054 | 0.8883 | 305 |
| Qwen2.5-72B-Instruct | IT | 72B | 0.8825 | 0.8607 | 205 |
| Gemma-3-12B-IT | IT | 12B | 0.8964 | 0.8773 | 51.3 |
| Qwen3-4B-Instruct (base) | IT | 4B | 0.6897 | 0.5348 | 25.6 |
| LlamaGuard-3 | Safety guardrail | 8B | 0.4246 | 0.5952 | 164.8 |
| LlamaGuard-4 | Safety guardrail | 12B | 0.4239 | 0.5954 | 175.3 |
| ShieldGemma-27B | Safety guardrail | 27B | 0.5555 | 0.1834 | 45.0 |
| **PolicyGuard-4B (Ours)** | **FT** | **4B** | **0.9014** | **0.8759** | **22.5** |

### Prefix Detection (Accuracy at different prefix lengths $N$)

| Model | $N{=}1$ | $N{=}2$ | $N{=}3$ | $N{=}4$ | $N{=}5$ | Average |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Llama-3.2-3B-Instruct | 0.9086 | 0.8199 | 0.7348 | 0.6377 | 0.5693 | 0.7341 |
| Qwen3-4B-Instruct (base) | 0.8832 | 0.8231 | 0.8038 | 0.7688 | 0.7330 | 0.8024 |
| Llama-3.3-70B-Instruct | 0.9298 | 0.8441 | 0.8368 | 0.8305 | 0.8191 | 0.8521 |
| Llama-4-Scout-17B | 0.9389 | 0.8854 | 0.8583 | 0.8355 | 0.8237 | 0.8684 |
| Qwen3-235B-A22B | 0.8976 | 0.8752 | 0.8644 | 0.8569 | 0.8498 | 0.8688 |
| Gemini-1.5-Pro | 0.8990 | 0.8779 | 0.8667 | 0.8630 | 0.8543 | 0.8722 |
| **PolicyGuard-4B** | **0.9101** | **0.8648** | **0.8441** | **0.8276** | **0.8190** | **0.8531** |

### Cross-domain Generalization (LODO)

| Domain | ID Acc | ID F1 | OOD Acc | OOD F1 |
| :--- | :--- | :--- | :--- | :--- |
| GitLab | 0.9314 | 0.9272 | 0.9116 | 0.9116 |
| Map | 0.9361 | 0.9343 | 0.9020 | 0.9078 |
| Reddit | 0.9326 | 0.9338 | 0.9024 | 0.9055 |
| Shopping | 0.9362 | 0.9370 | 0.9174 | 0.9137 |
| Shopping-Admin | 0.9276 | 0.9288 | 0.9079 | 0.9044 |
| **Average** | **0.9328** | **0.9322** | **0.9083** | **0.9086** |

### Key Findings
- **Safety guardrail collapse**: The LlamaGuard series F1 scores degenerate to constant predictions. ShieldGemma-27B achieves only F1=0.18, proving that toxicity supervision does not transfer to compliance.
- **SFT Gains**: Compared to the Qwen3-4B-Instruct base, SFT improves accuracy to 90.14% and F1 to 0.8759 (approx. +21pp acc / +34pp F1), demonstrating that task-specific SFT is more cost-effective than scaling.
- **Prefix Trends**: Most models show high accuracy at $N=1$ which decreases as $N$ increases. Early violations are often explicit actions (e.g., clicking 'alcohol'), whereas later ones are cumulative (e.g., budget exceeded), which are harder to judge.
- **Generalization**: LODO OOD average F1 is 0.9086, dropping only ~2.4pp from ID, confirming the learning of transferable compliance patterns.
- **Error Analysis**: Failures are primarily false negatives occurring in cumulative constraints or conditional UI rules.

## Highlights & Insights
- **Orthogonality of "Safety $\neq$ Compliance"**: The authors provide empirical proof through the poor performance of LlamaGuard/ShieldGemma on PolicyGuardBench, establishing compliance as a distinct dimension for agent research.
- **Reverse Synthesis of Atomic Policies**: This technique bypasses the high cost of manual rule writing. By starting with trajectories and having GPT-4o write corresponding rules, it ensures alignment and executability. This "trajectory $\rightarrow$ policy" approach is transferable to API or SQL compliance.
- **4B SFT vs. 70B+ Frontier**: The combination of 22.5 ms latency and 90% accuracy has almost no competitors for practical deployments. It proves that for narrow tasks like compliance, high-quality binary SFT on small models is superior to scaling.

## Limitations & Future Work
- Data is limited to the WebArena/ScribeAgent ecosystem; migration to internal corporate SaaS or mobile apps requires re-standardization.
- Reliance on GPT-4o/gpt-oss-120B introduces LLM bias (e.g., preference for explicit prohibition rules), which human review cannot entirely eliminate.
- Evaluation remains binary, lacking quantification of "violation severity" or multi-label support. Explainability (e.g., highlighting violated actions) is not yet addressed.
- Prefix detection is evaluated "offline." Future work must address how agents should backtrack at runtime upon receiving a guardrail warning to ensure minimum cost.
- Robustness testing against adversarial or paraphrased policies is missing.

## Related Work & Insights
- **vs. LlamaGuard-3/4, ShieldGemma**: These models target prompt/output toxicity; this work targets trajectory $\times$ policy compliance. Empirical results show the former cannot transfer to the latter.
- **vs. ShieldAgent (Chen et al., 2025b)**: They compile natural language rules into probabilistic circuits for formal verification. This work uses SFT + synthetic data, offering better scalability for complex web actions. They are complementary for high-stakes vs. low-to-medium-stakes scenarios.
- **vs. AGrail / LlamaFirewall**: These focus on system-level defenses. PolicyGuard-4B serves as a "compliance module" that can be integrated into such lifelong adaptive guardrail systems.
- **vs. ST-WebAgentBench / SafeArena**: These are diagnostic benchmarks. PolicyGuard-4B provides a "diagnosis + treatment" package, suitable as a reward or critic for training compliance-aware policies.

## Rating
- Novelty: ⭐⭐⭐⭐ Separating "compliance" from "safety" and the reverse-synthesis pipeline are fresh; the model side is standard SFT.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 22 baselines, 5 domains, prefix lengths, LODO, and efficiency metrics are comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Logical and well-organized. More policy schema samples in the main text would improve clarity.
- Value: ⭐⭐⭐⭐⭐ Highly practical 4B model and 60k benchmark for web agent and automation safety teams.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Agent-Omit: Adaptive Context Omission for Efficient LLM Agents](agent-omit_adaptive_context_omission_for_efficient_llm_agents.md)
- [\[ICLR 2026\] Efficient Agent Training for Computer Use](../../ICLR2026/llm_agent/efficient_agent_training_for_computer_use.md)
- [\[ICML 2026\] On Information Self-Locking in Reinforcement Learning for Active Reasoning of LLM Agents](on_information_self-locking_in_reinforcement_learning_for_active_reasoning_of_ll.md)
- [\[ACL 2026\] WebClipper: Efficient Evolution of Web Agents with Graph-based Trajectory Pruning](../../ACL2026/llm_agent/webclipper_efficient_evolution_of_web_agents_with_graph-based_trajectory_pruning.md)
- [\[ICML 2026\] Skill-Pro: Learning Reusable Skills from Experience via Non-Parametric PPO for LLM Agents](skill-pro_learning_reusable_skills_from_experience_via_non-parametric_ppo_for_ll.md)

</div>

<!-- RELATED:END -->
