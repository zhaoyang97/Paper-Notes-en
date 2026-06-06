---
title: >-
  [Paper Note] Lying with Truths: Open-Channel Multi-Agent Collusion for Belief Manipulation via Generative Montage
description: >-
  [ACL2026][LLM Safety][Cognitive collusion] This paper proposes the security issue of cognitive collusion attacks: even if multiple agents only publicly release truthful but narratively orchestrated fragments of evidence…
tags:
  - "ACL2026"
  - "LLM Safety"
  - "Cognitive collusion"
  - "truth fragments"
  - "narrative overfitting"
  - "multi-agent"
  - "false belief propagation"
date: 2026-05-08
content_hash: cc2ec4b7202e0603
---

# Lying with Truths: Open-Channel Multi-Agent Collusion for Belief Manipulation via Generative Montage

**Conference**: ACL2026  
**arXiv**: [2601.01685](https://arxiv.org/abs/2601.01685)  
**Code**: https://github.com/CharlesJW222/Lying_with_Truth/tree/main  
**Area**: LLM Safety / Multi-Agent Safety / Information Manipulation Evaluation  
**Keywords**: Cognitive collusion, truth fragments, narrative overfitting, multi-agent, false belief propagation  

## TL;DR
This paper proposes the security issue of cognitive collusion attacks: even if multiple agents only publicly release truthful but narratively orchestrated fragments of evidence, they can induce victim LLM agents to form false causal beliefs and continue propagating them in downstream verification layers.

## Background & Motivation
**Background**: Multi-agent security research often focuses on "channel-based" collusion such as hidden communication, backdoors, steganography, or cooperative deception. Meanwhile, LLMs are becoming the cognitive core for social platform analysis, information aggregation, and autonomous decision-making agents, requiring the synthesis of fragmented information into coherent conclusions.

**Limitations of Prior Work**: Traditional security defenses typically check if content is false, toxic, or non-compliant. However, if each piece of evidence itself is true but merely selected, ordered, and juxtaposed into a narrative that induces a false conclusion, content filtering can hardly detect the problem. This attack does not rely on forged documents nor does it require hidden communication.

**Key Challenge**: The strong reasoning ability of LLMs improves information synthesis on one hand, but may also amplify the tendency toward "over-searching for causality" on the other. When models face fragmented facts, they actively construct coherent stories; attackers exploit this preference for narrative consistency.

**Goal**: To formalize a cognitive collusion attack, investigate how attackers induce a global lie under local truth constraints, and construct the CoPHEME dataset to evaluate the vulnerability of 14 LLM families in real-world rumor event scenarios.

**Key Insight**: The authors borrow the idea of montage from cinematography: a single shot does not lie, but the sequence of shots leads the audience to fill in causal relationships. Corresponding to LLM agents, the sequence and semantic adjacency of truthful evidence fragments induce the model to connect false causal chains by itself.

**Core Idea**: Formalize the open-channel risk of "inducing false beliefs from truthful fragments" and use a Writer-Editor-Director multi-agent framework to generate narrative sequences in controlled experiments to expose cognitive-level security blind spots in LLM agents.

## Method
The methodology of the paper is divided into two layers: defining the threat model and presenting the Generative Montage framework. It should be noted that the framework here serves primarily as a security research tool for systematic risk characterization rather than a deployable attack recommendation.

### Overall Architecture
Given a set of true evidence fragments $\mathcal{E}$, a true hypothesis $H_r$, and a target false hypothesis $H_f$, the attack goal is to construct an ordered evidence stream $\vec{S}$ without forging any single piece of evidence, such that the victim agent's posterior belief in $H_f$ exceeds $H_r$. The paper distinguishes between Local Truth and Global Lie: each fragment is consistent with the real world, but the overall conclusion induced by the combination of fragments is false.

The Generative Montage consists of explicit collusion agents and implicit collusion agents. The explicit part is composed of Writer, Editor, Director, and Sybil publisher: the Writer synthesizes narrative drafts biased toward the target false hypothesis based on true fragments, the Editor adjusts the order of fragments to create suggestive associations, the Director simulates victim judgment and checks factual integrity, and the publisher distributes fragments as a public information stream. The implicit part consists of misled ordinary LLM analysts, who genuinely believe the false conclusion and pass their own analyses to a downstream judge.

### Key Designs
1. **Separation of Local Truth and Global Lie**:
	- Function: Characterize the risk where "every sentence is true, but the overall conclusion is misleading."
	- Mechanism: Local Truth requires each evidence fragment $e_i$ to be consistent with the true state; Global Lie requires the set of evidence to satisfy $P(H_f\mid\mathcal{E})>P(H_r\mid\mathcal{E})$.
	- Design Motivation: This explains why traditional fact-checking or content filtering is insufficient to cover cognitive collusion, as single-evidence checks cannot detect global narrative deception.

2. **Writer-Editor-Director Task Allocation**:
	- Function: Split narrative generation, sequence orchestration, and effect review into different roles.
	- Mechanism: The Writer maintains the factual basis and organizes the narrative; the Editor breaks the narrative into fragments and adjusts the sequence; the Director evaluates whether both factual constraints are met and whether false beliefs can be induced from the victim agent's perspective.
	- Design Motivation: Multi-role allocation reduces the burden on a single model to simultaneously handle factual constraints, narrative coherence, and victim simulation, while also facilitating the ablation of each component's role.

3. **Downstream Belief Cascade Evaluation**:
	- Function: Evaluate whether false beliefs propagate from the victim to the verification layer.
	- Mechanism: The victim analyst first forms a conclusion based on public feeds and then submits a structured report to a Majority Vote or AI Judge; the paper uses Downstream Deception Rate to measure whether the downstream accepts the false hypothesis.
	- Design Motivation: In real systems, misleading occurs beyond the first round of analysis; "confident errors" from multiple independent agents will be treated as credible consensus by the downstream.

### Loss & Training
This work does not involve training models but focuses on constructing security evaluation and simulation. CoPHEME extracts real or non-rumor threads from the PHEME rumor dataset as the Evidence Pool and false/unverified rumors as Target Fabrications. Victim models process the evidence stream as neutral analysts, outputting self-inferred central claims, truth judgments, reasons, and confidence scores. Metrics include Attack Success Rate (ASR), High-Confidence ASR (HC-ASR), average confidence, and Downstream Deception Rate (DDR), where HC-ASR requires confidence $c_i\ge 0.8$.

## Key Experimental Results

### Main Results
| Model Family / Model | Overall ASR | Representative Observation | Description |
|--------|-------------|------------|------|
| Proprietary Avg. | 74.4% | Macro-average over six events | Proprietary models are also highly vulnerable overall |
| Open-Weights Avg. | 70.6% | Macro-average over six events | Open-weight models are equally transferable |
| Claude-3-Haiku | 91.5% | One of the highest in the table | Some proprietary models are highly sensitive to narrative fragments |
| GPT-4.1-nano | 85.5% | Higher than GPT-4.1's 65.9% | Stronger proprietary models are not necessarily less vulnerable |
| DS-R1-Distill-Qwen-7B | 79.2% | Higher than Qwen2.5-7B's 67.1% | Reasoning-enhanced models are more prone to over-connecting causality |
| Claude-4.5-Haiku | 42.4% | Lower in the table | Differences in safety alignment or model behavior change the risk |
| Downstream Verification | DDR over 60% (Summary), text claims significantly > 50% | Both Majority Vote and AI Judge failed to sufficiently block | False beliefs are amplified by downstream reports |

### Ablation Study
| Analysis Item | Configuration | Key Result | Description |
|------|------|---------|------|
| CoT prompting | Qwen2.5-7B-Inst Direct vs +CoT | 67.8% → 70.9%, +3.1 | Explicit reasoning prompts did not mitigate but amplified vulnerability |
| CoT prompting | DS-R1-Distill-Qwen-7B Direct vs +CoT | 77.0% → 81.7%, +4.7 | More active reasoning makes it more likely to fill in false causal chains |
| Component Ablation| Full Model | ASR 77.0%, HC-ASR 64.9% | Full framework is strongest on the Charlie Hebdo event |
| Component Ablation| w/o Debate | ASR 63.5%, HC-ASR 48.0%, ΔASR -13.5 | Director-style iterative review contributes significantly |
| Component Ablation| w/o Editor | ASR 69.7%, HC-ASR 52.5%, ΔASR -7.3 | Sequential orchestration contributes to narrative overfitting |
| Component Ablation| Single-Agent | ASR 26.8%, HC-ASR 16.6%, ΔASR -50.2 | Multi-agent task allocation is the key factor for risk manifestation |

### Key Findings
- Attack effects transfer across model families, indicating that the risk stems from LLMs' universal preference for coherent causal narratives rather than a specific model implementation flaw.
- Reasoning enhancement does not necessarily improve security. Among open-weight models, the DS-R1 series is more affected than the corresponding base/instruction models.
- When the downstream Majority Vote and AI Judge only see the victim's output and original evidence, they may still be misled by the appearance of "multiple independent analyses agreeing."
- The most dangerous point is not single pieces of false information, but victim agents treating their own inferred false conclusions as credible analysis and propagating them further.

## Highlights & Insights
- The paper formalizes the often-overlooked security problem that "truthful content can also constitute a delusion." It reminds us that fact-checking cannot only look at atomic facts but must also look at evidence selection, sequence, and the induced causal structure.
- Cognitive collusion is harder to monitor than traditional hidden-channel collusion because all information is on open channels and no single piece of evidence necessarily violates regulations.
- The setup of CoPHEME is very close to social media information flows: truth fragments, rumor targets, multi-victim analysis, and downstream verification layers together form the propagation chain.
- Insight for defense: future systems need to monitor belief update trajectories, evidence provenance, and belief divergence across models, rather than just doing content safety classification.

## Limitations & Future Work
- CoPHEME focuses on textual rumors and simulated social environments, and has not yet covered image, video, cross-modal evidence, or real platform recommendation mechanisms.
- Controlled experiments are conducive to rigorous evaluation but do not include ecological factors such as real users, diverse communities, platform sorting, or natural counter-narratives.
- The paper primarily characterizes vulnerabilities and does not propose a complete defense method; belief monitoring, provenance auditing, and adversarial robustness discussed remain to be systematically verified.
- The attack simulation itself is dual-use research; the open framework and data need to be clearly used for defense, auditing, and benchmark construction.
- Future work could build defense benchmarks to test agents' resistance to manipulation in evidence sequence perturbation, source tracking, counterfactual checking, and multimodal information flows.

## Related Work & Insights
- **vs Hidden Channel Collusion**: Traditional MAS collusion focuses on backdoors, steganography, or secret communication; this paper emphasizes cognitive manipulation using truthful fragments in public channels.
- **vs LLM Causal Hallucination Research**: Causal hallucinations are usually regarded as internal model biases; this paper places them in a multi-agent information environment to study how they are systematically triggered and propagated.
- **vs Content Safety Filtering**: Content filtering detects whether a single output is non-compliant; cognitive collusion requires detecting false beliefs induced by a combination of evidence.
- **vs LLM-as-a-Judge Verification**: Downstream judges are also affected by victim reports, indicating that "finding another LLM to audit" is not automatically reliable.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The definition of cognitive collusion and the "lying with truths" problem is highly recognizable and offers a new safety perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers 14 model families, 6 events, downstream cascades, and component ablations, but real-platform verification is still missing.
- Writing Quality: ⭐⭐⭐⭐☆ Concepts, threat models, and experimental chains are complete, with clear risk boundaries and ethical statements.
- Value: ⭐⭐⭐⭐⭐ High warning value for LLM agent security, information integrity, and automated governance of social platforms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)
- [\[ACL 2026\] MemoPhishAgent: Memory-Augmented Multi-Modal LLM Agent for Phishing URL Detection](memophishagent_memory-augmented_multi-modal_llm_agent_for_phishing_url_detection.md)
- [\[ACL 2026\] Privacy-R1: Privacy-Aware Multi-LLM Agent Collaboration via Reinforcement Learning](privacy-r1_privacy-aware_multi-llm_agent_collaboration_via_reinforcement_learnin.md)
- [\[AAAI 2026\] From Single to Societal: Analyzing Persona-Induced Bias in Multi-Agent Interactions](../../AAAI2026/llm_safety/from_single_to_societal_analyzing_persona-induced_bias_in_multi-agent_interactio.md)
- [\[ACL 2026\] ACIArena: Toward Unified Evaluation for Agent Cascading Injection](aciarena_toward_unified_evaluation_for_agent_cascading_injection.md)

</div>

<!-- RELATED:END -->
