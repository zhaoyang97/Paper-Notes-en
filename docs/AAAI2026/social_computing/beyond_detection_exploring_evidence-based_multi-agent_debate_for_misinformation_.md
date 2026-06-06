---
title: >-
  [Paper Note] Beyond Detection: Exploring Evidence-based Multi-Agent Debate for Misinformation Intervention and Persuasion
description: >-
  [AAAI 2026][Social Computing][Multi-agent debate] This paper proposes ED2D, a framework that integrates an evidence retrieval module into a multi-agent debate (MAD) system to enhance misinformation detection accuracy. Th…
tags:
  - "AAAI 2026"
  - "Social Computing"
  - "Multi-agent debate"
  - "misinformation detection"
  - "evidence retrieval"
  - "persuasion evaluation"
  - "LLM"
date: 2026-05-08
content_hash: 4bec7c93d8bf2583
---

# Beyond Detection: Exploring Evidence-based Multi-Agent Debate for Misinformation Intervention and Persuasion

**Conference**: AAAI 2026
**arXiv**: [2511.07267](https://arxiv.org/abs/2511.07267)  
**Code**: [https://github.com/hanshenmesen/Debate-to-Detect](https://github.com/hanshenmesen/Debate-to-Detect)  
**Area**: Social Computing
**Keywords**: Multi-agent debate, misinformation detection, evidence retrieval, persuasion evaluation, LLM

## TL;DR
This paper proposes ED2D, a framework that integrates an evidence retrieval module into a multi-agent debate (MAD) system to enhance misinformation detection accuracy. Through controlled human experiments, it provides the first comparative evaluation of AI-generated debate transcripts versus expert human fact-checks in terms of persuasiveness and belief correction, revealing a double-edged-sword effect: the AI debate system achieves expert-level persuasiveness when correct, but may amplify misinformation when wrong.

## Background & Motivation
Misinformation poses a persistent threat to public trust and social governance. Existing MAD-based detection methods improve detection accuracy by simulating adversarial reasoning, but suffer from two core limitations: (1) they focus solely on detection accuracy while neglecting the importance of helping users understand the reasoning process—simply labeling a claim as "false" is insufficient to inoculate users against misinformation; (2) they rely entirely on the LLM's internal knowledge, making them prone to hallucination and lacking robustness on emerging or unfamiliar claims.

The paper's starting point is the notion that "truth is forged through debate": the transcripts produced during debate are a valuable but underutilized resource. The core idea is to construct an evidence-augmented MAD system that **both accurately detects misinformation and effectively persuades users to correct false beliefs**, unifying detection and intervention within a single framework.

## Method

### Overall Architecture
ED2D is built upon a five-stage debate structure: Opening Statements, Rebuttal, Free Debate, Closing Statements, and Judgment. The central innovation is the integration of an evidence retrieval module during the Free Debate and Judgment stages, grounding the debate in verifiable external facts rather than solely in the model's internal knowledge.

### Key Designs
1. **Agent Layer & Debate Structure**:

    - Function: Organizes multi-agent structured debate.
    - Mechanism: Four agents are assigned to each side (affirmative and negative), each with domain-specific profiles, arguing for "True" or "Fake" respectively. Five judge agents score the debate along five dimensions: factuality, source credibility, reasoning quality, clarity, and ethical considerations.
    - Design Motivation: A complementary scoring scheme (paired scores summing to 7) prevents ties, and the aggregated score yields a definitive REAL or FAKE classification.

2. **Evidence Retrieval and Integration Module**:

    - Function: Dynamically introduces external factual evidence during the Free Debate stage.
    - Mechanism: Executed in four steps—(1) extract up to five key entities/concepts from the claim; (2) retrieve relevant content from Wikipedia using structured queries; (3) use an LLM to classify retrieved evidence by stance (supporting/refuting/neutral); (4) debating agents cite supporting or refuting evidence in their arguments.
    - Design Motivation: Anchoring the debate in external facts reduces the risk of LLM hallucination and enhances the verifiability and credibility of the arguments.

3. **Orchestration Layer & History Compression**:

    - Function: Manages debate flow and maintains shared conversational memory.
    - Mechanism: After each stage, the context is compressed by distilling key information into concise summaries passed to subsequent stages; all agents share the compressed history.
    - Design Motivation: Mitigates LLM context length limitations and ensures coherence across multiple debate rounds.

4. **Snopes25 Benchmark Dataset**:

    - Function: Constructs a real-world benchmark for comparing the persuasiveness of AI versus human expert fact-checks.
    - Mechanism: Collects 448 claims and corresponding expert fact-check reports from Snopes covering January–June 2025, a time window after GPT-4o's training cutoff.
    - Design Motivation: Reduces data leakage risk and reflects contemporary misinformation trends.

### Loss & Training
ED2D involves no model training and uses GPT-4o as the backbone. Temperature is set to 0.0 for domain inference and final judgment to ensure stability, and to 0.7 for profile generation and debate utterances to encourage diversity. The Free Debate stage defaults to one round but is configurable.

## Key Experimental Results

### Main Results (Misinformation Detection)

| Method | Weibo21 F1 | FakeNewsDataset F1 | Snopes25 F1 |
|--------|-----------|-------------------|-------------|
| BERT | 77.77 | 79.94 | 74.71 |
| RoBERTa | 81.08 | 82.19 | 77.09 |
| CoT w/ evidence | 78.44 | 76.14 | 70.43 |
| D2D (prior work) | 81.97 | 81.94 | 76.89 |
| **ED2D** | **83.18** | **83.41** | **80.40** |

ED2D achieves the best performance across all datasets and metrics, outperforming D2D by 2–3 percentage points.

### Persuasion Evaluation

| Condition | Accuracy on False Claims | Belief Score (↓ better) | Sharing Intention (↓ better) |
|-----------|------------------------|------------------------|------------------------------|
| Control | 63.60% | 3.46 | 3.15 |
| ED2D (correct) | 80.40% | 2.85 | 2.84 |
| Snopes Expert | 85.60% | 2.77 | 2.55 |
| ED2D (incorrect) | 42.00% | 4.44 | 3.55 |

| Condition | Post-exposure Independent Detection Accuracy |
|-----------|---------------------------------------------|
| Control | 66.9% |
| ED2D | 75.9% |
| Snopes | 78.6% |

### Key Findings
- **When correct**: ED2D's persuasiveness is comparable to expert fact-checks, effectively reducing users' belief in misinformation and their willingness to share it.
- **When incorrect**: ED2D-generated explanations systematically distort user judgment; even when correct expert explanations are presented simultaneously, the corrective effect is partially offset.
- **Transfer effect**: Users exposed to ED2D explanations show approximately 9% higher accuracy when independently judging new claims, suggesting that MAD-style debate can enhance media literacy.
- Evidence retrieval consistently improves all LLM-based methods, with the most pronounced gains observed for simpler baselines (e.g., zero-shot).

## Highlights & Insights
- This is the first work to extend a MAD system from a "detection tool" to a "persuasion system," quantifying its persuasive effects through controlled human experiments.
- The study reveals a double-edged-sword effect of AI persuasiveness—an ethical risk that must be confronted when deploying AI fact-checking systems.
- Post-exposure tests demonstrate that MAD-style debate can cultivate users' critical thinking skills, offering long-term educational value.
- A publicly accessible community platform was developed, enabling users to interactively explore the debate process.

## Limitations & Future Work
- Reliance on GPT-4o incurs high API costs and is unsuitable for large-scale real-time deployment.
- The human experiment sample size is limited (200 participants), and the generalizability of results requires further validation.
- Adversarial scenarios (e.g., deliberately crafted misinformation attacks) have not been evaluated.
- The five-stage debate pipeline is lengthy and introduces high latency, making it unsuitable for real-time intervention.
- The negative impact of misclassification indicates that the system requires more reliable confidence estimation and automatic auditing mechanisms.

## Related Work & Insights
- ED2D inherits and extends the D2D framework by incorporating evidence retrieval, upgrading it from pure internal-knowledge reasoning to retrieval-augmented reasoning.
- Unlike standard RAG approaches, ED2D integrates retrieval dynamically within the interactive debate process rather than as a one-time prefetch step.
- The persuasion evaluation paradigm is generalizable to the design of other AI systems where user trust and behavioral impact must be considered.
- This work highlights the need to establish error prevention and disclosure mechanisms when deploying AI-driven fact-checking systems.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (First systematic evaluation of persuasiveness in MAD systems; discovers the double-edged-sword effect)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Three datasets + controlled human experiments; comprehensive analysis)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure but somewhat lengthy)
- Value: ⭐⭐⭐⭐⭐ (Significant reference value for safe AI deployment)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] T2Agent: A Tool-augmented Multimodal Misinformation Detection Agent with Monte Carlo Tree Search](t2agent_a_tool-augmented_multimodal_misinformation_detection_agent_with_monte_ca.md)
- [\[AAAI 2026\] Reasoning About the Unsaid: Misinformation Detection with Omission-Aware Graph Inference](reasoning_about_the_unsaid_misinformation_detection_with_omission-aware_graph_in.md)
- [\[ICLR 2026\] When Agents "Misremember" Collectively: Exploring the Mandela Effect in LLM-based Multi-Agent Systems](../../ICLR2026/social_computing/when_agents_misremember_collectively_exploring_the_mandela_effect_in_llm-based_m.md)
- [\[ACL 2026\] Debating the Unspoken: Role-Anchored Multi-Agent Reasoning for Half-Truth Detection](../../ACL2026/social_computing/debating_the_unspoken_role-anchored_multi-agent_reasoning_for_half-truth_detecti.md)
- [\[CVPR 2026\] Probabilistic Concept Graph Reasoning for Multimodal Misinformation Detection](../../CVPR2026/social_computing/probabilistic_concept_graph_reasoning_for_multimodal_misinformation_detection.md)

</div>

<!-- RELATED:END -->
