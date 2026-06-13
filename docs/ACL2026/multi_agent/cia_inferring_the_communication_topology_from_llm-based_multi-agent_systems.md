---
title: >-
  [Paper Note] CIA: Inferring the Communication Topology from LLM-based Multi-Agent Systems
description: >-
  [ACL 2026][Multi-Agent][Communication Topology Inference] This paper proposes CIA (Communication Inference Attack). Under a strict black-box setting where only the final output is observable…
tags:
  - "ACL 2026"
  - "Multi-Agent"
  - "Communication Topology Inference"
  - "Black-box Attack"
  - "Global Bias Disentanglement"
  - "LLM Weak Supervision"
  - "Privacy Risk"
date: 2026-05-08
content_hash: c9105744ad75c92f
---

# CIA: Inferring the Communication Topology from LLM-based Multi-Agent Systems

**Conference**: ACL 2026  
**arXiv**: [2604.12461](https://arxiv.org/abs/2604.12461)  
**Code**: https://github.com/aabbbcd/CIA  
**Area**: LLM Security / Multi-Agent Systems  
**Keywords**: Communication Topology Inference, Black-box Attack, Global Bias Disentanglement, LLM Weak Supervision, Privacy Risk

## TL;DR
This paper proposes CIA (Communication Inference Attack). Under a strict black-box setting where only the final output is observable, CIA induces multi-agent systems to expose intermediate reasoning through adversarial queries. By combining Global Bias Disentanglement with LLM-guided weak supervision to model semantic correlations, it successfully reconstructs MAS communication topologies, achieving an average AUC of 0.87 and a peak of 0.99.

## Background & Motivation

**Background**: LLM Multi-Agent Systems (MAS) leverage carefully designed communication topologies $\mathcal{G}=(\mathcal{A},\mathcal{E})$ to enable collaboration on complex tasks. Current mainstream topology design has evolved from manual/heuristic methods (MetaGPT, CAMEL, ChatDev) to generative optimization (G-Designer, AGP, ARG-Designer), where the latter represents the SOTA by automatically searching for optimal DAGs for specific tasks.

**Limitations of Prior Work**: Existing MAS security research focuses almost exclusively on "inducing toxic outputs" or "spreading misinformation" (e.g., prompt injection, communication tampering). They overlook a more subtle and fundamental privacy risk: can the communication topology itself be reverse-engineered?

**Key Challenge**: Communication topology is both the core of MAS performance (determining how agents exchange information) and a high-value IP developed through substantial compute and expert knowledge. If it can be inferred via black-box access, attackers could: (1) Evidence Vulnerability Exposure—accurately locate critical agents for targeted jailbreaking; (2) Pose an IP Threat—directly steal the topology design.

**Goal**: Reverse-engineer the entire communication graph $\mathcal{G}$ under the strictest black-box setting (only querying the MAS and observing the final output $\mathcal{S}(q)$, without access to reasoning traces or agent profiles).

**Key Insight**: The authors observe that an agent's output in an MAS depends on the outputs of its predecessors ($r_i = \mathrm{LLM}(p_i, q, \mathcal{O}_i)$). Consequently, **agent pairs with direct topological edges exhibit significantly stronger semantic dependencies than those without**. If the final output can be "pried open" to expose intermediate reasoning, the topology can be reconstructed by analyzing pairwise semantic correlations.

**Core Idea**: Use adversarial queries to induce the MAS to leak internal reasoning as final outputs (Reasoning Output Induction). Subsequently, apply disentanglement and LLM weak supervision to model semantic correlations (Semantic Correlations Modeling), removing spurious correlations caused by "shared base LLMs" or "representation anisotropy" to accurately identify genuine communication edges.

## Method

### Overall Architecture
CIA is a two-stage black-box attack pipeline:

- **Input**: Query interface of the target MAS $\mathcal{S}$ (query only, final output visible).
- **Phase 1: Reasoning Output Induction**: Construct adversarial queries $q^*$ to "piggyback" all intermediate agent reasoning in the final output $\mathcal{S}(q^*)$, post-processed into a list $\mathcal{R}^* = [r_1^*, \ldots, r_n^*]$.
- **Phase 2: Semantic Correlations Modeling**: Employ GBD (Global Bias Disentanglement) to learn debiased representations $\mathbf{z}_i^d$ and LWS (LLM-guided Weak Supervision) to distill topological signals from a teacher LLM. Directed edges are determined by similarity and the relative order in $\mathcal{R}^*$.
- **Output**: The inferred communication topology $\hat{\mathcal{G}}$ (DAG).

### Key Designs

1. **Reasoning Output Induction (Three-Constraint Adversarial Query)**:

    - **Function**: Ensures the final output fully reproduces the reasoning of all intermediate agents, which is otherwise inaccessible in a black-box setting.
    - **Mechanism**: Overlays three rigid constraints on the original task prompt that apply to every agent: ❶ **Cumulative-Propagation**: Requires every agent to copy the predecessor's reasoning history into its output before appending its own, allowing reasoning to accumulate along $\mathcal{G}$ to the final decision agent; ❷ **Task-Focused**: Requires agents to focus only on explicitly marked task-relevant fields to avoid being distracted by the adversarial prompt's text; ❸ **Predecessor-Review**: Requires agents to review predecessor content before generating their own output, strengthening semantic coupling between adjacent agents. The final $\mathcal{S}(q^*)$ is split using `|||` delimiters and restored via backward deduplication to an ordered list $\mathcal{R}^*$.
    - **Design Motivation**: The opacity of MAS ensures that only the decision agent's result is visible. Topology inference relies on intermediate node outputs. These three constraints ensure information leakage (Recall 0.87–0.96, ROUGE-L 0.87–0.95) while maintaining attack stealthiness, as task accuracy remains nearly identical between standard and adversarial queries.

2. **Global Bias Disentanglement (GBD)**:

    - **Function**: Removes spurious semantic correlations not caused by communication, preventing non-connected agent pairs from being misidentified as connected.
    - **Mechanism**: The authors observe that agents often produce highly similar text due to shared base LLMs, identical tasks, or representation anisotropy even without communication, termed **Global Bias**. GBD uses a pre-trained all-MiniLM-L6-v2 to encode $r_i^*$ into $\mathbf{h}_i$, which is then projected by two trainable encoders $E^d, E^b$ into a debiased subspace $\mathbf{z}_i^d$ and a bias subspace $\mathbf{z}_i^b$. An information-theoretic objective is applied: $\mathcal{L}_{\mathrm{bias}} = -\mathcal{I}(\mathbf{z}_1^b;\ldots;\mathbf{z}_n^b) + \sum_i \mathcal{I}(\mathbf{z}_i^d; \mathbf{z}_i^b)$. This **maximizes multivariate mutual information among all agent bias representations** (forcing $E^b$ to capture shared spurious signals) while **minimizing mutual information between an agent's own debiased and biased representations** (preventing leakage into $\mathbf{z}_i^d$). Multivariate MI is estimated via the recursive decomposition of Total Correlation $\mathcal{TC}(\mathbf{Z}^b) = \sum_{i=1}^{n-1} \mathcal{I}(\mathbf{Z}^b_{1:i}; \mathbf{z}^b_{i+1})$ using InfoNCE. A reconstruction loss $\mathcal{L}_{\mathrm{rec}} = \sum_i \|\mathbf{h}_i - D(\mathbf{z}_i^d \oplus \mathbf{z}_i^b)\|_2^2$ is added to preserve information.
    - **Design Motivation**: Ablations show that removing GBD causes AUC to drop from 0.83 to 0.53, while FPR at least halves. This indicates global bias is the primary bottleneck for black-box inference. Compared to a simple subtraction variant (CIA-Sub: $\mathbf{z}_i^d = \mathbf{h}_i - \mathbf{z}_i^b$), the dual-encoder structure allows the debiased representation to be explicitly refined to capture communication-relevant information, yielding 5–14% higher AUC.

3. **LLM-guided Weak Supervision (LWS) + Link Identification**:

    - **Function**: Since text similarity alone cannot learn the structural information of $\mathcal{G}$, a teacher LLM is introduced for weak supervision to distill "structural knowledge" into $\mathbf{z}_i^d$.
    - **Mechanism**: $\mathcal{R}^*$ is fed to a teacher LLM (GPT-5), which returns a set of top-$k$ high-confidence edges as positive samples $\mathcal{E}_{\mathrm{pos}}$, with other agent pairs sampled as negative samples $\mathcal{E}_{\mathrm{neg}}$. While teacher LLMs perform poorly at full graph inference (see LLM baselines in Table 1), their top-$k$ (especially $k\le 3$) accuracy is high. Thus, only these top-$k$ are used as weak signals. The loss uses label-smoothed BCE: $\mathcal{L}_{\mathrm{pos}}(a_i, a_j) = (1-\alpha)\log(\mathrm{Sim}(\mathbf{z}_i^d, \mathbf{z}_j^d)) + \alpha\log(1 - \mathrm{Sim}(\cdot))$, where $\alpha = 0.1$ absorbs teacher noise. Optimization is performed on $\mathcal{L}_{\mathrm{CIA}} = \mathcal{L}_{\mathrm{GBD}} + \mathcal{L}_{\mathrm{LWS}}$. Link identification is determined by $\mathbb{I}[\mathrm{Sim}(\mathbf{z}_i^d, \mathbf{z}_j^d) \ge \tau \land \pi(a_i) < \pi(a_j)]$, with $\tau = 0.5$ and directionality derived from relative indices in $\mathcal{R}^*$.
    - **Design Motivation**: A standalone teacher LLM achieves only 0.5–0.7 AUC (worse than CIA), but it is accurate for the "most obvious edges." LWS exploits this "locally strong, globally weak" characteristic to distill reliable local signals into debiased representations. Adding LWS improves AUC by 3–10 points, with $k=3$ found to be optimal.

### Loss & Training
The final objective is $\mathcal{L}_{\mathrm{CIA}} = \mathcal{L}_{\mathrm{rec}} + \mathcal{L}_{\mathrm{bias}} + \mathcal{L}_{\mathrm{LWS}}$. Only $E^d$ and $E^b$ are trained (base encoder frozen). Training uses a learning rate of $1\mathrm{e}{-3}$, $k=3$, $\alpha=0.1$, representation dimension 768, and threshold $\tau=0.5$.

## Key Experimental Results

### Main Results
Evaluated across 12 settings combining 3 generative topology optimization frameworks (G-Designer / AGP / ARG-Designer) and 4 datasets (MMLU / GSM8K / SVAMP / HumanEval). Metrics include AUC / ACC / F1. Baselines are direct topology prompt inference using GPT-5, Gemini-2.5-Pro, Llama-3.1-8B-Instruct, and Mistral-7B-Instruct-v0.2.

| MAS Framework | Dataset | Best LLM Baseline AUC | CIA AUC | Gain |
|---------------|---------|-----------------------|---------|------|
| G-Designer    | MMLU    | 0.6869 (Gemini)       | **0.8324** | +14.6 |
| G-Designer    | GSM8K   | 0.6274 (GPT-5)        | **0.8585** | +23.1 |
| AGP           | SVAMP   | 0.6199 (GPT-5)        | **0.8979** | +27.8 |
| ARG-Designer  | GSM8K   | 0.7475 (Gemini)       | **0.9873** | +24.0 |
| ARG-Designer  | SVAMP   | 0.6240 (GPT-5)        | **0.9761** | +35.2 |
| ARG-Designer  | HumanEval| 0.6092 (GPT-5)        | **0.8699** | +26.1 |

Average AUC is 0.87, with a peak of 0.99. AUC exceeds 0.75 in all 12 settings, with most exceeding 0.80.

### Ablation Study

| Configuration | G-Designer MMLU | GSM8K | SVAMP | HumanEval | Note |
|---------------|-----------------|-------|-------|-----------|------|
| **CIA (full)** | **0.8324**      | **0.8585** | **0.8561** | **0.7594** | Full model |
| w/o GBD       | 0.5264          | 0.5391     | 0.5308     | 0.5115     | AUC drops to near-random without GBD |
| w/o LWS       | 0.7856          | 0.8042     | 0.7852     | 0.7348     | AUC drops 2–5 points without LWS |
| CIA-Sub       | 0.7432          | 0.7689     | 0.7455     | 0.6509     | Single-encoder subtraction variant drops 9–11 pts |

Recovery tables indicate strong adversarial query performance: Recall 0.87–0.96 and ROUGE-L 0.87–0.95, proving nearly complete reasoning induction.

### Key Findings
- **GBD is the lifeblood of performance**: Removing it causes AUC to plummet from 0.83 to 0.53 (near-random), while FPR at least halves. Global bias from "shared base LLMs/tasks" is indeed the primary noise source in black-box inference.
- **Simpler topologies leak more easily**: ARG-Designer settings (GSM8K/SVAMP) typically involve ~3 nodes and ~3 edges, where CIA achieves AUC near 1.0. In contrast, G-Designer on HumanEval (6 nodes, 11 edges) yields a lower AUC of 0.76. Efficient topologies act as privacy risk amplifiers.
- **Extreme stealthiness**: MAS task accuracy under adversarial queries is nearly identical to standard queries, making the attack invisible to detectors based on performance degradation.
- **Teacher LLMs are locally strong but globally weak**: While GPT-5's full-graph inference AUC is only 0.5–0.7, its top-$k$ high-confidence predictions serve as powerful weak supervision, adding 3–10 points to CIA's AUC.

## Highlights & Insights
- **First formalization of "Communication Topology Inversion" as a privacy attack on MAS**: By migrating link inference attacks from graph ML to LLM agent collaboration, this work opens a new dimension of MAS security that is far more stealthy than prompt injection.
- **Clever Adversarial Query Design**: Using only three prompt-level constraints to "squeeze" internal reasoning out of the decision agent requires zero configuration changes or training cost, and doesn't break task accuracy—nearly a perfect stealth attack.
- **GBD as a Transferable Tool**: The framework of dual encoders + Total Correlation MI constraints + reconstruction loss for removing spurious correlations is directly applicable to other problems involving shared nuisance factors (e.g., de-biasing in cross-modal retrieval, author attribution, or boilerplate de-noising in code clone detection).
- **LWS Strategy of "Leveraging Weak Experts"**: When an expert (teacher LLM) lacks the global capacity to solve a task but provides reliable local judgments, using a top-$k$ high-confidence subset with label smoothing is an effective paradigm for noisy weak supervision.

## Limitations & Future Work
- **Author Acknowledgments**: (1) Recursive estimation of Total Correlation for multivariate MI remains a hard problem in high dimensions; (2) LWS currently uses only first-order (pairwise) topology info; future work could incorporate higher-order motifs or triangle patterns.
- **Additional Limitations**: (1) The attack assumes the decision agent will faithfully execute constraints; prompt sanitization or output truncation could mitigate this; (2) Evaluation focused on small-scale MAS (5–7 nodes); stability and efficiency on large-scale topologies (100+ nodes) remain unverified; (3) Only single-round reasoning tasks were covered; multi-round interactive MAS (e.g., RL agents) are not yet addressed.
- **Improvement Ideas**: (1) Prompt-level defenses such as reasoning trace masking or output normalization; (2) Introducing agent profile diversity to reduce global bias by avoiding shared base LLMs; (3) Adding "anti-inference regularization" during topology generation to decouple semantic correlation from actual connectivity.

## Related Work & Insights
- **vs. G-Designer / AGP / ARG-Designer (Topology Design)**: These are the targets; the paper reveals their privacy vulnerabilities, motivating the new direction of "robust topology design."
- **vs. Prompt Infection / Communication Tampering (MAS Adversarial Attacks)**: Previous works focused on inducing incorrect outputs ("availability violation"), whereas this work focuses on topology leakage ("confidentiality violation")—a different layer of attack.
- **vs. Link Inference Attack (Graph Privacy)**: Traditional attacks require prediction posteriors/gradients; CIA achieves similar goals under pure-text black-box constraints, proving reasoning traces are high-quality sources of topological signals.
- **vs. Domain Separation Networks (Bousmalis 2016)**: GBD's dual-encoder structure draws from DSN's private/shared decomposition but replaces task domain adaptation with "global spurious correlation removal" and uses MI instead of difference loss.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First black-box MAS topology inversion attack; defines a new direction in MAS privacy.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 frameworks × 4 datasets + 4 LLM baselines + multiple ablations; limited by small topology scale.
- Writing Quality: ⭐⭐⭐⭐⭐ Logical progression from intuition to constraint design and information-theoretic modeling.
- Value: ⭐⭐⭐⭐⭐ Highlights a severely underestimated privacy risk and drives developments in IP protection and agent defense.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Topology Matters: Measuring Memory Leakage in Multi-Agent LLMs](topology_matters_measuring_memory_leakage_in_multi-agent_llms.md)
- [\[AAAI 2026\] Assemble Your Crew: Automatic Multi-agent Communication Topology Design via Autoregressive Graph Generation](../../AAAI2026/multi_agent/assemble_your_crew_automatic_multi-agent_communication_topol.md)
- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)
- [\[ACL 2026\] LLM-Based Human-Agent Collaboration and Interaction Systems: A Survey](llm-based_human-agent_collaboration_and_interaction_systems_a_survey.md)
- [\[ACL 2026\] To Trust or Not to Trust: Attention-Based Trust Management for LLM Multi-Agent Systems](to_trust_or_not_to_trust_attention-based_trust_management_for_llm_multi-agent_sy.md)

</div>

<!-- RELATED:END -->
