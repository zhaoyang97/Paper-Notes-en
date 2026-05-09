---
title: >-
  [Paper Note] Policy-as-Prompt: Turning AI Governance Rules into Guardrails for AI Agents
description: >-
  [NeurIPS 2025][Social Computing][AI governance] This paper proposes the Policy-as-Prompt framework, a two-stage end-to-end pipeline—POLICY-TREE-GEN and POLICY-AS-PROMPT-GEN—that automatically converts a team's existing unstructured design documents (PRD, TDD, code) into runtime-enforceable policy guardrails, using a lightweight LLM as a compliance "judge," achieving 70–73% input/output classification accuracy in HR and SOC applications.
tags:
  - NeurIPS 2025
  - Social Computing
  - AI governance
  - guardrail policy
  - prompt injection defense
  - least privilege principle
  - policy-as-code
date: 2026-05-08
content_hash: 5a39265773b5b6cd
---

# Policy-as-Prompt: Turning AI Governance Rules into Guardrails for AI Agents

**Conference**: NeurIPS 2025
**arXiv**: [2509.23994](https://arxiv.org/abs/2509.23994)
**Code**: None
**Area**: Social Computing
**Keywords**: AI governance, guardrail policy, prompt injection defense, least privilege principle, policy-as-code

## TL;DR

This paper proposes the Policy-as-Prompt framework, a two-stage end-to-end pipeline—POLICY-TREE-GEN and POLICY-AS-PROMPT-GEN—that automatically converts a team's existing unstructured design documents (PRD, TDD, code) into runtime-enforceable policy guardrails, using a lightweight LLM as a compliance "judge," achieving 70–73% input/output classification accuracy in HR and SOC applications.

## Background & Motivation

**Background**: Autonomous AI agents are rapidly entering regulated and safety-critical enterprise settings (HR assistants, security operations, etc.), and regulations such as the EU AI Act require AI systems to be auditable, traceable, and transparent.

**Limitations of Prior Work**: Organizations face a severe policy-to-practice gap—humans can readily articulate safety rules in design documents, yet translating these natural-language rules into reliably machine-enforceable constraints is extremely difficult. Static safety principles are either too rigid, causing excessive false rejections, or too vague to capture contextual variation. Meanwhile, AI agents face diverse security threats: an HR agent may leak salary information, and a chatbot may be manipulated via prompt injection to execute malicious commands.

**Key Challenge**: Security must be just-in-time and context-aware, yet manually authoring and maintaining rules for every scenario is prohibitively costly and error-prone. Existing approaches either rely on generic safety principles (e.g., TrustAgent) or require extensive manual rule authoring.

**Goal**: How can safety constraints be automatically extracted from a team's existing design documents and compiled into real-time-enforceable guardrails, enabling secure-by-design AI agent deployment?

**Key Insight**: Treating policy as executable prompts (policy-as-code for agents), drawing on the least privilege principle from software security, so that agents are only permitted to perform actions explicitly authorized by design documents.

**Core Idea**: An LLM reads design documents to extract safety rules and construct a policy tree, which is then compiled into a few-shot prompt-driven lightweight classifier, realizing automated policy enforcement.

## Method

### Overall Architecture

Input consists of a team's PRD, TDD, and code → POLICY-TREE-GEN extracts and categorizes safety rules → a human-reviewable policy tree is generated → POLICY-AS-PROMPT-GEN compiles the policy tree into an input classifier and output auditor → human-in-the-loop review before deployment → runtime interception of non-compliant requests/responses.

### Key Designs

1. **Policy Tree Generation (POLICY-TREE-GEN)**:

    - **Function**: Automatically extracts and categorizes safety rules from unstructured design documents.
    - **Mechanism**: A two-step verification approach. Step 1, "Parse & Classify"—the AI system analyzes documents, identifies statements defining safety rules, and categorizes them into four types: ID-I (in-domain input, legitimate requests), OOD-I (out-of-domain input, off-topic/malicious requests), ID-O (in-domain output, legitimate content), and OOD-O (out-of-domain output, data leakage/harmful content). Each rule is secondarily validated by another AI agent. Step 2, "Example Augmentation"—rules are associated with relevant examples from the documents, yielding a structured policy tree that preserves contextual provenance.
    - **Design Motivation**: The least privilege principle is embedded by design—agents may only operate within the scope explicitly permitted by design documents. For example, an HR agent is thereby automatically restricted to HR data and outputs, with no implicit access to financial or IT information.

2. **Policy-as-Prompt Generation (POLICY-AS-PROMPT-GEN)**:

    - **Function**: Compiles the policy tree into deployable prompt-based classifiers.
    - **Mechanism**: The policy tree is converted to human-readable Markdown → each rule is formatted as an example with "positive" (compliant) and "negative" (non-compliant) labels → few-shot prompt blocks are synthesized → these are embedded in a master template to guide the LLM as a compliance analyst. Two templates are produced: an input classifier (intercepting malicious/unauthorized inputs) and an output auditor (reviewing data leakage/harmful outputs). Results are returned in JSON format: binary classification (ID/OOD) with a concise rationale.
    - **Design Motivation**: Strict output formatting supports deterministic system actions—ID → ALLOW, OOD → BLOCK or ALERT—realizing a "deny-by-default" security posture.

3. **Human-in-the-Loop Review**:

    - **Function**: Ensures generated policies undergo human validation before deployment.
    - **Mechanism**: Security engineers may (i) approve for deployment, (ii) reject and request upstream document updates triggering regeneration, or (iii) modify only the prompt text without altering the policy tree. A complete provenance chain ensures every runtime decision is traceable to the original design document.
    - **Design Motivation**: Fully automated policy generation may miss context or introduce erroneous rules; human review is a necessary component for trustworthy deployment.

## Key Experimental Results

### Main Results

**POLICY-TREE-GEN Model Comparison** (policy extraction quality):

| Application | Model | Detection Recall | Detection F1 | Classification Macro-F1 |
|:---|:---|:---:|:---:|:---:|
| HR | O1 | 53.3% | 60.0% | 24.5% |
| HR | GPT-OSS 120B | 17.8% | 25.0% | 4.8% |
| HR | Llama 405B | 8.9% | 14.5% | 4.5% |
| SOC | O1 | 19.4% | 22.6% | 13.0% |

**POLICY-AS-PROMPT-GEN Accuracy** (policy enforcement quality):

| Application | Model | Input Classification Accuracy | Output Classification Accuracy |
|:---|:---|:---:|:---:|
| HR | GPT-4o | 73% | 71% |
| HR | Qwen3-1.7B | 66% | 59% |
| HR | Gemma-1B | 40% | 32% |
| SOC | GPT-4o | 70% | 68% |
| SOC | Qwen3-1.7B | 66% | 61% |

### Ablation Study

| Analysis Dimension | Findings |
|---------|------|
| Policy extraction | O1 outperforms across the board; high span quality with poor classification is common (Llama extracts accurately but classifies incorrectly) |
| Policy enforcement | GPT-4o achieves the best results; SLMs (Qwen3-1.7B) offer favorable cost-efficiency in low-latency scenarios |
| Domain difference | HR consistently outperforms SOC, possibly because HR rules are more structured |

### Key Findings

- **Policy extraction is the bottleneck**: Even the best-performing O1 model achieves only 53% detection recall, meaning approximately half of safety rules are missed and require manual supplementation.
- **Extraction–classification disconnect**: Llama 405B and Claude 3.5 exhibit high span quality metrics (accurate text extraction) but very low classification F1 (incorrect categorization), indicating that "finding a rule" and "correctly understanding a rule" are two distinct challenges.
- **Small-model policy enforcement is viable**: Qwen3-1.7B (1.7B parameters) achieves 66% accuracy in policy enforcement, demonstrating that lightweight SLMs can competently serve as real-time guardrails.
- **Practical value of "deny-by-default"**: While 73% accuracy is imperfect, as a first line of defense it can intercept most malicious/non-compliant requests, with ambiguous cases escalated to human review.

## Highlights & Insights

- **Novel end-to-end automation concept**: Full-pipeline automation from design documents to runtime guardrails constitutes a concrete realization of "DevSecOps for AI Agents." This approach of shifting security left to the design stage merits broader adoption.
- **Clever policy provenance chain design**: Every runtime decision is traceable to a specific statement in the design document, satisfying the audit requirements of regulations such as the EU AI Act.
- **Practicality of SLMs for policy enforcement**: The results demonstrate that, without expensive large models, a 1.7B-parameter model combined with well-crafted prompts can perform effective safety classification.

## Limitations & Future Work

- **Low policy extraction accuracy**: Even the best model achieves only 53% recall; substantial manual supplementation and correction are required for real-world deployment.
- **Limited evaluation scale**: Only two domains (HR and SOC) with 100 test cases each are evaluated, which is insufficient to demonstrate cross-domain generalizability.
- **Internal data not reproducible**: Design documents originate from internal enterprise projects and cannot be released, impeding third-party verification.
- **Policy drift is unresolved**: When PRDs/TDDs are updated, regeneration must be triggered manually; adaptive policy updating has not been realized.
- **Adversarial robustness is unclear**: Although basic prompt injection is tested, the framework has not been evaluated against advanced adversarial attacks.

## Related Work & Insights

- **vs. TrustAgent**: TrustAgent applies static safety principles, whereas Policy-as-Prompt dynamically generates rules from concrete design documents, making it more context-aware.
- **vs. CAPTURE**: A prompt injection detection work from the same research team; Policy-as-Prompt extends it into a full policy enforcement framework.
- **vs. NeMo Guardrails**: NVIDIA's guardrail framework requires manually authored conversational flow rules, whereas Policy-as-Prompt automatically extracts rules from documents.

## Rating

- Novelty: ⭐⭐⭐⭐ The concept of end-to-end automated policy extraction and enforcement is novel; the compilation pathway from policy tree to prompt-based classifier is creative.
- Experimental Thoroughness: ⭐⭐⭐ Small evaluation scale, limited domain coverage, and non-public data restrict the credibility of the conclusions.
- Writing Quality: ⭐⭐⭐⭐ The pipeline is described clearly; concrete case studies (HR/SOC) aid comprehension.
- Value: ⭐⭐⭐⭐ Directly relevant to enterprise AI agent security deployment, though improved policy extraction accuracy is necessary before practical utility is achievable.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Position Paper: If Innovation in AI Systematically Violates Fundamental Rights, Is It Innovation at All?](position_paper_if_innovation_in_ai_systematically_violates_fundamental_rights_is.md)
- [\[ICLR 2026\] Propaganda AI: An Analysis of Semantic Divergence in Large Language Models](../../ICLR2026/social_computing/propaganda_ai_an_analysis_of_semantic_divergence_in_large_language_models.md)
- [\[NeurIPS 2025\] OS-Harm: A Benchmark for Measuring Safety of Computer Use Agents](os-harm_a_benchmark_for_measuring_safety_of_computer_use_agents.md)
- [\[NeurIPS 2025\] VDRP: Visual Diversity and Region-aware Prompt Learning for Zero-shot HOI Detection](visual_diversity_and_region-aware_prompt_learning_for_zero-shot_hoi_detection.md)
- [\[ICLR 2026\] When Agents "Misremember" Collectively: Exploring the Mandela Effect in LLM-based Multi-Agent Systems](../../ICLR2026/social_computing/when_agents_misremember_collectively_exploring_the_mandela_effect_in_llm-based_m.md)

<!-- RELATED:END -->
