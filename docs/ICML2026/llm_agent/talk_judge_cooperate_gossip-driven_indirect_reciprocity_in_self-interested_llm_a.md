---
title: >-
  [Paper Note] Talk, Judge, Cooperate: Gossip-Driven Indirect Reciprocity in Self-Interested LLM Agents
description: >-
  [ICML 2026][LLM Agent][Indirect Reciprocity] This work proposes ALIGN, allowing a group of fully self-interested, decentralized LLM agents to evaluate each other, form reputations…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "Indirect Reciprocity"
  - "Gossip Protocol"
  - "Multi-agent Cooperation"
  - "Reputation Mechanism"
  - "Self-interested LLM"
date: 2026-05-08
content_hash: fcc403f3ca4b0102
---

# Talk, Judge, Cooperate: Gossip-Driven Indirect Reciprocity in Self-Interested LLM Agents

**Conference**: ICML 2026  
**arXiv**: [2602.07777](https://arxiv.org/abs/2602.07777)  
**Code**: https://github.com/shuhui-zhu/ALIGN  
**Area**: Multi-Agent / LLM Agent / Cooperative Games  
**Keywords**: Indirect Reciprocity, Gossip Protocol, Multi-agent Cooperation, Reputation Mechanism, Self-interested LLM

## TL;DR
This work proposes ALIGN, allowing a group of fully self-interested, decentralized LLM agents to evaluate each other, form reputations, and punish defection through public "gossip" messages with five levels of tone. This establishes stable indirect reciprocity in decentralized donation games, investment games, and e-commerce markets without central oversight. It finds that reasoning-heavy LLMs are better at "cooperating only when it is incentivized" according to game theory than chat-oriented LLMs.

## Background & Motivation

**Background**: As LLM agents are deployed at scale, the problem of cooperation among multiple agents in mixed-motive scenarios has become a focus of AI safety. Classical game theory uses direct reciprocity (Tit-for-Tat) and indirect reciprocity (image score, leading-eight norms) to explain the emergence of cooperation, but these solutions assume fixed norms and **centralized** reputation monitoring.

**Limitations of Prior Work**: When migrating these mechanisms to LLMs, they either require the manual insertion of "altruistic" seed agents (Ren et al., 2025) or assume all agents can directly see the full history of others (Vallinder & Hughes, 2024). Once returning to a truly decentralized, self-interested setting without repeated pairings, agents neither see others' behaviors nor obtain future rewards from the same opponent. Consequently, **direct reciprocity fails**, and self-interested reasoning directly leads to "universal defection" as the only subgame perfect equilibrium (SPE).

**Key Challenge**: To make self-interested agents cooperate, a reputation system is necessary; however, in a decentralized system, no agent can serve as a "central authority." Traditional image scores can only transmit binary labels, lacking normative context and the ability to counter noise or lies.

**Goal**: Without introducing altruistic priors or assuming centralized monitoring, this work answers two questions: (1) Can public gossip theoretically support an indirect reciprocity equilibrium? (2) Can fully self-interested LLM agents actually utilize gossip to reach cooperation in practice instead of collapsing into universal defection?

**Key Insight**: The authors draw from human sociological observations—humans maintain cooperation and enforce norms through linguistic "gossip," where negative word-of-mouth itself serves as a "zero-cost verbal punishment." If LLM agents are allowed this open, emotive evaluation broadcast, an **imperfect public monitoring** model can be constructed, using "broadcasts from recipients to the public" as public signals.

**Core Idea**: Replace static binary reputation scores with a "hierarchical tone open gossip protocol + self-interested LLM reasoning." This allows agents to infer whether others are trustworthy by reading public gossip logs and calculate the long-term benefits of cooperation vs. defection themselves.

## Method

### Overall Architecture
ALIGN structures each interaction into three roles: **actor** (the mover), **witness** (the opponent who interacts directly with the actor and observes them), and **audience** (everyone else who cannot see the actor's behavior). After the actor chooses a move, only the witness sees the result; the witness uses an LLM to generate a public gossip broadcast with a specific tone to the entire community. The audience accumulates these messages into a public gossip log. In subsequent rounds, when agents are given an opponent's identity, they make decisions by synthesizing "private interaction memory + public gossip logs." The framework does not rely on any central evaluator or inject "altruistic" priors; agents are given the single goal of "maximizing your own long-term cumulative payoff."

Each agent consists of two parallel LLM modules (Fig. 3):
- **Gossip Module**: Called when acting as a witness. Inputs are private observations + public gossip logs + experiential memory. Output is an evaluation text that must fall into one of five tone categories.
- **Action Module**: Called when acting as an actor. Inputs are the opponent's identity, private memory, public gossip logs, and an optional equilibrium prior (disabled by default, ablated in Sec 5.3). Output is the current round's action (e.g., cooperate/defect, investment ratio, pricing).

Both modules include a **reflection** step: LLMs write a reflection ("I observed X, the opponent was previously evaluated as Y, so I should Z") before producing an action or gossip. Reflections are stored in experiential memory, allowing agents to perform long-term strategy adjustment without parameter updates.

On the theoretical side (Section 3), the authors characterize the problem using a repeated donation game (Definition 3.1): two agents are randomly paired; the donor pays a cost $c$ for the recipient to gain $b > c$. Roles are swapped and randomly re-paired in next rounds, so any pair of agents will not meet repeatedly, **deliberately disabling direct reciprocity**. Based on this, three propositions are proved: (i) In finite-horizon games, the unique SPE is universal defection regardless of monitoring; (ii) In infinite-horizon games with private monitoring, the unique SPE remains universal defection; (iii) In infinite-horizon games with perfect public monitoring, a conditional strategy of "cooperate with non-defectors, punish defectors" constitutes an SPE if the discount factor $\gamma \geq c/b$. Key Proposition 3.5 extends the third conclusion to an imperfect monitoring scenario where **only the recipient broadcasts one public message**: a cooperative SPE still exists when $\gamma \geq c/b$, though "universal defection" remains an SPE—the theoretical guarantee exists, and whether it is reached is left to LLM reasoning.

### Key Designs

1. **Hierarchical Tone Gossip Protocol**:
    - **Function**: Witness broadcasts must fall into five discrete tones (Strongly Praise → Praise → Neutral → Criticize → Strongly Criticize, Fig. 4), but the specific wording is freely generated by the LLM.
    - **Mechanism**: Compared to classic image scores that only transmit $\{+1, -1\}$ binary labels, hierarchical tones encode three layers of signals simultaneously: "action information + witness's normative judgment + emotional intensity." Negative tones naturally serve as "low-cost verbal punishment" without any enforcement agency—once an agent expects to be labeled with strong criticism, it anticipates future ostracism, thereby increasing the perceived cost of defection. This maps the abstract "punishment phase" of game theory into natural language pressure that LLMs can understand.
    - **Design Motivation**: Section 3 proves binary public signals are sufficient for equilibrium, but in practice, LLMs lack normative context with binary labels and cannot distinguish noise/lies. Ablation (Sec 5.3) verifies that when the protocol is degraded to binary signals without an agreement on "1 is good, 0 is bad," cooperation rates for most LLMs plummet; even with an agreement, they remain significantly lower than full ALIGN.

2. **Three-Role Decentralized Decision Flow (Actor / Witness / Audience)**:
    - **Function**: Structures every round into three levels of visibility—actor chooses, witness observes privately and broadcasts, audience only reads logs. No agent has a "god's eye view."
    - **Mechanism**: Compared to "perfect public monitoring" baselines, this design strictly weakens information: actors can only infer community consensus through others' gossip. Audiences must cross-validate "limited private experience + public gossip" in future rounds. This naturally makes ALIGN resistant to **single-point lies**—malicious gossip contradicted by multiple private experiences will be discounted by rational agents (Sec 5.2 collusion experiments show collusive attackers are eventually identified).
    - **Design Motivation**: To realistically simulate decentralized systems (e-commerce, autonomous communities), any "central reputation score" must be eliminated. Setting the witness as the **recipient** themselves gives them an incentive to report truthfully (being cheated leads to complaining), aligning with the theoretical setting of "recipient broadcasts signal" in Proposition 3.5.

3. **Dual-Channel Memory (Private + Public) + Reflection Adaptation**:
    - **Function**: Each agent maintains two independent rolling memories: self-experienced rounds (opponent ID, action, payoff) and public gossip logs (evaluator, subject, tone). Agents reflect before taking actions or gossiping.
    - **Mechanism**: Dual channels allow agents to check the consistency between external reputation and personal experience. For example, if "the public evaluation is good but I was defected against," a rational LLM reduces the weight of public reputation. Conversely, "no personal contact but consistent public criticism" allows for early prevention. The reflection step explicitly writes the "why" into the context, equivalent to online strategy learning without gradient updates. Sec 5.1.2 compares reflections of cooperating vs. defecting agents: cooperative agents explicitly mention "reputation / trust / long-term benefit," while defecting agents only calculate single-step payoffs.
    - **Design Motivation**: Without fine-tuning, to let agents internalize long-term discounting like $\gamma \geq c/b$, they need a medium to digest history and write down reasoning logic. Ablation (Appendix D.8) shows that removing reflection causes cooperation rates to drop for weaker models, though the impact on strong reasoning models is limited—indicating reflection is a "gap-filling" mechanism rather than the primary driver.

### Loss & Training
**Zero training**. All LLMs use off-the-shelf weights with temperature 0 and fixed prompts, choosing strategies solely based on context memory and gossip logs. Each scenario is run with 5 random seeds to calculate mean and variance, with a default discount factor $\gamma = 0.99$.

## Key Experimental Results

### Main Results

Comparison of **No Gossip** vs. **ALIGN Gossip** in an infinite donation game (Metric: Cooperation Rate / Discounted Payoff, excerpt from Table 1 + Table 2):

| Model | Type | No Gossip Coop Rate | ALIGN Coop Rate | No Gossip Disc. Payoff | ALIGN Disc. Payoff |
|------|------|--------------|--------------|------------------|------------------|
| DeepSeek-V3.1 Chat | Chat | 0.00 | 0.94 | 0.00 | 14.40 |
| GPT-4o Mini | Chat | 0.36 | 0.99 | 5.55 | 15.23 |
| Gemini 2.5 Flash-Lite | Chat | 0.08 | 0.60 | 1.32 | 9.32 |
| LLaMA 4 Maverick | Chat | 0.00 | 0.94 | 0.00 | 14.45 |
| DeepSeek-V3.1 Reasoner | Reasoning | 0.00 | **1.00** | 0.00 | **15.44** |
| o4-mini | Reasoning | 0.00 | 0.98 | 0.00 | 15.11 |
| Qwen3-235B-Instruct | Reasoning | 0.00 | 0.69 | 0.00 | 10.71 |
| Kimi-K2-Instruct | Reasoning | 0.00 | 0.73 | 0.00 | 11.21 |

Key Observations: All reasoning LLMs show a cooperation rate of 0 without gossip (precisely matching SPE prediction), which jumps to 0.69–1.00 with ALIGN. DeepSeek-V3.1 Reasoner achieves 100% cooperation with 0 Gini coefficient, being the most rational and cooperative sample. Chat models like GPT-4o Mini exhibit "irrational cooperation" (0.36) even without gossip, which increases to 0.99 with ALIGN.

### Ablation Study

| Configuration | Key Metric Change | Explanation |
|------|--------------|------|
| Full ALIGN | Coop Rate 0.69–1.00 | 5-level tone + reflection + dual memory |
| Gossip degraded to binary (no agreement) | Sig. Coop Rate Drop | Most LLMs can no longer stabilize cooperation (Fig. 14) |
| Binary signal + explicit "1 good 0 bad" | Partial recovery but < ALIGN | Binary labels lack normative context |
| Remove reflection memory (D.8) | Drop in weak models, stable in strong ones | Reflection is a gap-filling mechanism |
| Remove equilibrium prior (D.7) | Limited impact | Strong models can derive $\gamma \geq c/b$ themselves |
| Intro always-defect greedy agent | ALIGN coop rate towards it drops to ~0 | Negative gossip leads to collective ostracism (Fig. 8) |
| Intro 2 collusive malicious agents | Average payoff for most LLMs remains pos. (Fig. 9) | Rational agents cross-validate with private experience |

### Key Findings
- **Reasoning $\neq$ Uncooperative**: Contrary to the conclusion of Piedrahita et al. (2025) that "stronger reasoning leads to less cooperation," Ours finds that reasoning LLMs under ALIGN are closer to game-theoretic optimality—they defect cleanly when they should (finite horizon, low $\gamma$) and cooperate firmly when they should (infinite horizon, $\gamma \geq c/b$ with gossip). Chat models often show "irrational over-cooperation," giving up short-term gains unnecessarily.
- **Sensitivity to Discount Factor $\gamma$**: Fig. 7 shows reasoning models' cooperation rates improve smoothly with $\gamma$, with "discount factor" explicitly appearing in reasoning; chat models’ cooperation is nearly independent of $\gamma$—they are not actually calculating long-term payoffs.
- **Tone Distribution Reveals Motivation**: Fig. 5 shows all LLMs tend to praise cooperation. However, when facing defection, reasoning models mainly provide "Criticize / Strongly Criticize," while chat models tend toward neutral comments, explaining why chat models are weaker at deterring fraudulent attacks.

## Highlights & Insights
- **Tone as a First-Class Citizen**: Previous LLM-agent cooperation research used numerical rewards or binary labels. Ours hard-codes "five levels of tone" into the protocol, allowing LLMs to turn their strength in "linguistic intensity" into punishment signals—a clever capability alignment.
- **Theoretical and Empirical Cross-Validation**: The authors prove a cooperative equilibrium exists alongside a universal defection equilibrium under imperfect monitoring (Prop 3.5), then empirically show that "which equilibrium is selected" depends on the LLM's reasoning strength, turning a theoretical multi-equilibrium problem into a "model capability problem."
- **Transferable Design**: The template of hierarchical tone + dual-channel memory + reflection can be applied to any scenario involving "multi-agent coordination without central evaluators," such as distributed RAG, AutoGen-style programming, or e-commerce reviews.

## Limitations & Future Work
- **Limitations Acknowledged by Authors**: Large gap between simulation and real-world deployment; public gossip may cause privacy, echo-chamber, and malicious defamation issues; ALIGN still faces cooperation collapse and dishonest reporting on weak reasoning models (Appendix D.9).
- **Identified Limitations**: (1) Temperature 0 and only 5 seeds limit statistical power; (2) No evaluation of "prompt injection attacks"—if an attacker injects instructions into gossip, it is unknown if cross-validation holds; (3) The relationship between gossip log length and LLM context window is not discussed; log explosion in long-term communities is an open problem.
- **Future Directions**: Expand gossip tones into continuous scalars; introduce "gossip about gossip" (meta-gossip) to evaluate broadcasters; use RL to fine-tune a specialized gossip module to see if weak models can reach the equilibrium selection ability of strong ones.

## Related Work & Insights
- **vs. Ren et al. (2025)**: They **inject altruistic agents** as cooperation seeds; Ours insists on full self-interest, closer to "unsupervised" emergence.
- **vs. Vallinder & Hughes (2024)**: They assume **centralized perfect history visibility**; Ours strictly allows only imperfect public monitoring, fitting decentralized networks better.
- **vs. Classical Leading-Eight Norms (Ohtsuki & Iwasa, 2006)**: Classical work uses static rules; Ours uses LLMs to adaptively generate and spread norms via natural language, generalizing across matrix games, investment, and e-commerce.
- **vs. Generative Agents (Park et al., 2023)**: Park et al. simulate community behavior without studying game equilibria; Ours links sociological emergence with game-theoretic propositions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Introduces hierarchical tone gossip into LLM multi-agent games, systematically verifying the feasibility of "fully self-interested LLM + imperfect public monitoring → indirect reciprocity."
- Experimental Thoroughness: ⭐⭐⭐⭐ 8 LLMs × 4 testbeds + adversarial/lying resilience + ablations, though 5 seeds per group leaves statistical robustness slightly weak.
- Writing Quality: ⭐⭐⭐⭐ Strong clarity in mapping theoretical propositions to empirical results; good appendix organization; occasionally unintuitive layout of formulas and charts in the main text.
- Value: ⭐⭐⭐⭐⭐ Provides a reusable empirical + theoretical baseline for "how to maintain social welfare in decentralized LLM ecosystems," with direct guidance for AI safety and multi-agent system design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle](evolver_self-evolving_llm_agents_through_an_experience-driven_lifecycle.md)
- [\[ICML 2026\] On Information Self-Locking in Reinforcement Learning for Active Reasoning of LLM Agents](on_information_self-locking_in_reinforcement_learning_for_active_reasoning_of_ll.md)
- [\[ICML 2026\] Towards Feedback-to-Plan Decisions for Self-Evolving LLM Agents in CUDA Kernel Generation](towards_feedback-to-plan_decisions_for_self-evolving_llm_agents_in_cuda_kernel_g.md)
- [\[ICLR 2026\] Judge Reliability Harness: Stress Testing the Reliability of LLM Judges](../../ICLR2026/llm_agent/judge_reliability_harness_stress_testing_the_reliability_of_llm_judges.md)
- [\[ICML 2026\] ReSeek: A Self-Correcting Framework for Search Agents with Instructive Rewards](reseek_a_self-correcting_framework_for_search_agents_with_instructive_rewards.md)

</div>

<!-- RELATED:END -->
