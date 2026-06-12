---
title: >-
  [Paper Note] Dialectic-Med: Mitigating Diagnostic Hallucinations via Counterfactual Adversarial Multi-Agent Debate
description: >-
  [ACL 2026][Hallucination Detection][Medical Hallucination] The paper proposes Dialectic-Med, a multi-agent medical diagnostic framework inspired by Popper’s falsificationism. Through adversarial dialectical reasoning amo…
tags:
  - "ACL 2026"
  - "Hallucination Detection"
  - "Medical Hallucination"
  - "Multi-Agent Debate"
  - "Counterfactual Reasoning"
  - "Visual Falsification"
  - "Confirmation Bias"
date: 2026-05-08
content_hash: 9fd44ea8e300484c
---

# Dialectic-Med: Mitigating Diagnostic Hallucinations via Counterfactual Adversarial Multi-Agent Debate

**Conference**: ACL 2026  
**arXiv**: [2604.11258](https://arxiv.org/abs/2604.11258)  
**Code**: None  
**Area**: Hallucination Detection  
**Keywords**: Medical Hallucination, Multi-Agent Debate, Counterfactual Reasoning, Visual Falsification, Confirmation Bias

## TL;DR
The paper proposes Dialectic-Med, a multi-agent medical diagnostic framework inspired by Popper’s falsificationism. Through adversarial dialectical reasoning among a Proposer (diagnostic hypothesis), an Opponent (visual falsification module actively retrieving contradictory visual evidence), and a Mediator (weighted consensus graph decision), it achieves SOTA performance on MIMIC-CXR-VQA, VQA-RAD, and PathVQA. It improves explanation faithfulness by 12.5% and significantly mitigates diagnostic hallucinations.

## Background & Motivation

**Background**: Multimodal LLMs are increasingly integrated into high-stakes medical domains (radiology report generation, medical VQA), but they face severe diagnostic hallucination issues. Models exhibit confirmation bias, generating fluent but factually incorrect diagnostic statements.

**Limitations of Prior Work**: (1) LLMs often "lock in" preliminary textual hypotheses and then "hallucinate" visual features to support potentially incorrect conclusions, leading to error cascade propagation; (2) CoT reasoning is inherently linear forward reasoning, lacking an intrinsic self-correction mechanism—it tends to find evidence verifying current steps rather than challenging them (the "verificationist trap"); (3) Existing multi-agent systems mostly rely on static consensus or text-only debate without being driven by visual evidence.

**Key Challenge**: Robust diagnosis should not only rely on finding supportive evidence but must survive rigorous falsification attempts—yet existing methods lack such a falsification mechanism.

**Goal**: Design a multi-agent framework that explicitly models the falsification process, forcing the system to break the confirmation bias loop and anchor reasoning firmly in visual regions under adversarial scrutiny.

**Key Insight**: Grounded in Popperian philosophy of science—falsificationism—the credibility of a diagnosis is established by "attempting to overthrow it and failing."

**Core Idea**: An adversarial dialectical loop involving three specialized agents (Proposer diagnosis + Opponent visual falsification + Mediator consensus). The key innovation is the Opponent's Visual Falsification Module, which moves beyond semantic debate to actively retrieve contradictory visual evidence.

## Method

### Overall Architecture
Iterative loop: The Proposer proposes a diagnostic hypothesis based on medical images $\rightarrow$ The Opponent generates counterfactual probe queries (e.g., "If it is pneumonia, there should be clear opacities") $\rightarrow$ The Visual Falsification Module locates contradictory evidence in the image $\rightarrow$ The Mediator evaluates the attack strength $\rightarrow$ If the attack is sufficiently strong, the Proposer revises the hypothesis $\rightarrow$ The process continues until consensus is reached or the maximum number of turns is hit. The entire process constructs a dynamic consensus graph.

### Key Designs

1. **Visual Falsification Module (VFM)**:

    - **Function**: Enables the Opponent to move beyond semantic debate by actively locating contradictory visual evidence within the image.
    - **Mechanism**: Given a hypothesis $H_t$ (e.g., "pneumonia"), the Opponent generates counterfactual probe queries $Q_{cf}$ (e.g., "clear costophreatic angles"—evidence that pneumonia is absent). PubMedCLIP calculates the cosine similarity attention map $M_{cf}$ between probe queries and image patches; high attention regions are identified as contradictory evidence.
    - **Design Motivation**: Pure textual debate may rely on parametric priors rather than visual evidence. VFM forces the debate to ground itself in specific image regions, ensuring rebuttals are evidence-based.

2. **Dynamic Consensus Graph**:

    - **Function**: Structurally records the dialectical process and assists in final decision-making.
    - **Mechanism**: Nodes $\mathcal{V}_t$ represent diagnostic hypotheses or visual evidence, while edges $\mathcal{E}_t$ encode support/refutation logical relationships and confidence weights. Attack strength $S_{attack} = \frac{1}{|R_k|}\sum_{r \in R_k} \alpha_r$ quantifies the credibility of visual evidence. Cycle detection is included to prevent hypothetical loops.
    - **Design Motivation**: Unlike simple majority voting, the consensus graph preserves the full dialectical trajectory, enabling post-hoc auditing and explanation.

3. **Attack Strength Threshold Termination**:

    - **Function**: Terminates the debate when the Opponent cannot find sufficiently strong contradictory evidence.
    - **Mechanism**: If $S_{attack} < \theta_{thresh}$, it indicates the current hypothesis has survived falsification attempts, and the debate terminates (consensus reached).
    - **Design Motivation**: Avoids infinite debate while ensuring weak attacks do not mislead the correction process.

## Key Experimental Results

### Main Results

| Method | MIMIC-CXR-VQA | VQA-RAD | PathVQA |
|------|--------------|---------|---------|
| Single Agent CoT | Baseline | Baseline | Baseline |
| Multi-Agent Consensus | +Moderate | +Moderate | +Moderate |
| **Ours** | **SOTA** | **SOTA** | **SOTA** |

### Key Indicators Gain

| Metric | Gain |
|------|------|
| Explanation Faithfulness | +12.5% |
| Diagnostic Accuracy | SOTA |
| Hallucination Rate | Significant Reduction |

### Key Findings
- **Visual falsification is the critical differentiator**: Multi-agent methods utilizing pure semantic debate show limited improvement; VFM brings fundamental gains.
- **Confirmation bias is severe in standard CoT**: Models often "see" non-existent visual features to support incorrect hypotheses.
- **3-5 rounds of debate are usually sufficient** to reach consensus, keeping computational overhead controllable.
- **12.5% improvement in explanation faithfulness** demonstrates that diagnoses are not only more accurate but more interpretable and trustworthy.

## Highlights & Insights
- **Operationalizing Popperian falsificationism as an AI system design principle** is a profound insight—shifting from finding supporting evidence to actively seeking opposing evidence. This principle is transferable to any high-stakes scenario requiring reliable reasoning.
- **VFM transforms "debate" from a language game into a visual evidence-driven scientific process**—Opponents do not refute randomly but speak using actual image regions.
- **Direct value for Medical AI safety**: Before clinical deployment, the falsification mechanism can serve as a safety assurance layer.

## Limitations & Future Work
- VFM relies on the visual-language alignment quality of PubMedCLIP, which may degrade for rare pathologies.
- Multi-round debate increases inference latency, posing constraints for real-time diagnosis.
- The quality of counterfactual probes depends on the completeness of the medical knowledge base $\mathcal{K}_{med}$.
- Evaluation was limited to VQA tasks; more complex tasks like radiology report generation remain to be explored.
- The construction and traversal of the consensus graph increase system complexity.

## Related Work & Insights
- **vs Standard CoT**: CoT is linear verification reasoning, whereas Dialectic-Med is iterative falsification reasoning.
- **vs Multi-Agent (e.g., CAMEL)**: CAMEL uses role-playing for collaboration, while Dialectic-Med uses adversarial dialectics—the latter is better suited for scenarios requiring rigorous scrutiny.
- **vs Med-PaLM**: Med-PaLM pursues single-model accuracy, while Dialectic-Med ensures trustworthiness through systemic design.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The combination of falsificationism and the Visual Falsification Module represents a brand new paradigm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers three benchmarks plus faithfulness evaluation, though ablation details are slightly sparse.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The connection between philosophical motivation and technical implementation is very natural.
- **Value**: ⭐⭐⭐⭐⭐ Holds profound significance for medical AI safety and trustworthy reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MUG: Multi-agent Undercover Gaming — Hallucination Removal via Counterfactual Test for Multimodal Reasoning](../../AAAI2026/hallucination/multi-agent_undercover_gaming_hallucination_removal_via_coun.md)
- [\[AAAI 2026\] InEx: Hallucination Mitigation via Introspection and Cross-Modal Multi-Agent Collaboration](../../AAAI2026/hallucination/inex_hallucination_mitigation_via_introspection_and_cross-mo.md)
- [\[ACL 2026\] Mitigating Hallucinations in Large Vision-Language Models without Performance Degradation](mitigating_hallucinations_in_large_vision-language_models_without_performance_de.md)
- [\[ACL 2026\] Stable-RAG: Mitigating Retrieval-Permutation-Induced Hallucinations in Retrieval-Augmented Generation](stable-rag_mitigating_retrieval-permutation-induced_hallucinations_in_retrieval-.md)
- [\[ICML 2026\] REALISTA: Realistic Latent Adversarial Attacks that Elicit LLM Hallucinations](../../ICML2026/hallucination/realista_realistic_latent_adversarial_attacks_that_elicit_llm_hallucinations.md)

</div>

<!-- RELATED:END -->
