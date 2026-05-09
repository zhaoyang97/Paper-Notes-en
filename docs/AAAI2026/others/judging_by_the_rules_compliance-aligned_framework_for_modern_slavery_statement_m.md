---
title: >-
  [Paper Note] Judging by the Rules: Compliance-Aligned Framework for Modern Slavery Statement Monitoring
description: >-
  [AAAI 2026][Compliance Verification] This paper proposes a training framework centered on a Compliance-Aligned Judge (CA-Judge) that trains a 3B-parameter CALLM model using rule-level alignment feedback, enabling the generation of traceable compliance judgments grounded in statutory provisions. The model surpasses GPT-4o and DeepSeek-R1 on sentence-level compliance classification of modern slavery statements.
tags:
  - AAAI 2026
  - Compliance Verification
  - Modern Slavery Act
  - LLM-as-Judge
  - Rule Alignment
  - GRPO
date: 2026-05-08
content_hash: 44d466c30fefdd77
---

# Judging by the Rules: Compliance-Aligned Framework for Modern Slavery Statement Monitoring

**Conference**: AAAI 2026
**arXiv**: [2511.07803](https://arxiv.org/abs/2511.07803)
**Code**: [GitHub](https://github.com/mila-ai4h/aims-reasoning-alignment)
**Area**: Other
**Keywords**: Compliance Verification, Modern Slavery Act, LLM-as-Judge, Rule Alignment, GRPO

## TL;DR

This paper proposes a training framework centered on a Compliance-Aligned Judge (CA-Judge) that trains a 3B-parameter CALLM model using rule-level alignment feedback, enabling the generation of traceable compliance judgments grounded in statutory provisions. The model surpasses GPT-4o and DeepSeek-R1 on sentence-level compliance classification of modern slavery statements.

## Background & Motivation

Over 50 million people are affected by modern slavery worldwide. Multiple jurisdictions have enacted legislation (e.g., the Modern Slavery Acts of Australia, the UK, and Canada) requiring companies to publish annual statements disclosing risk assessments and mitigation measures within their supply chains. However, the sheer volume of such statements (over 80,000 globally) and their inconsistent quality—marked by vague language, disorganized structure, and a blending of compliance content with corporate promotional material—render manual review unscalable, allowing widespread non-compliance to evade regulatory scrutiny.

NLP offers a technical pathway toward scalable compliance monitoring, yet high-stakes compliance scenarios demand more than accuracy: outputs must be **traceable**, meaning auditors can map model judgments back to specific statutory provisions. Existing LLM approaches tend to reduce complex regulatory assessments to binary classification, lacking structured legal reasoning. While Chain-of-Thought prompting provides intermediate steps, recent research suggests CoT does not reliably reflect a model's true decision process and should not be treated as an interpretability mechanism.

**Key Challenge**: Compliance verification is fundamentally a **rule-matching problem**—determining whether a textual statement satisfies explicitly defined regulatory requirements. Existing LLM methods either operate as generic judges without domain-specific rule awareness, or fail to structurally incorporate statutory provisions during training. The paper's **Key Insight** is to distill regulatory provisions into "key rules," using them to guide both model training and output evaluation, so that model outputs serve not as self-justifying explanations but as **rule-aligned outputs verifiable by human auditors**.

## Method

### Overall Architecture

The framework consists of two stages: (1) **Key Rule Extraction**: translating each compliance criterion into structured natural-language rubrics that define what constitutes compliance or non-compliance; (2) **Rule-Aligned Training**: using CA-Judge to score model outputs across multiple compliance dimensions as a reward signal, fine-tuning CALLM via GRPO (Group Relative Policy Optimization) to produce rule-consistent outputs.

### Key Designs

1. **Key Rule Extraction**:

    - **Function**: Converts statutory compliance criteria into a deterministic set of natural-language rules.
    - **Mechanism**: Domain experts author key rules for each compliance criterion based on the Australian Modern Slavery Act. Each rule explicitly states what constitutes a valid versus invalid response. For example, key rules for the C4 Remediation criterion include: "must describe specific remediation measures" and "must not rely solely on generic corporate language."
    - **Design Motivation**: Transforms abstract legal provisions into actionable evaluation standards that serve simultaneously as training targets and assessment anchors.

2. **Compliance-Aligned Judge (CA-Judge)**:

    - **Function**: Evaluates the degree of alignment between model-generated reasoning and the key rules.
    - **Mechanism**: JudgeLRM-7B, an LLM specifically trained for judgment tasks, serves as the evaluator. Given the model's reasoning, predicted label, and the key rules for the corresponding criterion, CA-Judge assesses performance across six dimensions and returns a scalar score $R_{\text{judge}} \in [0, 1]$. The six dimensions cover auditability (accuracy, faithfulness), interpretability (clarity, evidence use), and reliability (consistency, verifiability).
    - **Design Motivation**: Emulates a human compliance reviewer's evaluation process—checking coverage, specificity, and clarity against each rule—offering greater precision than a generic LLM-as-Judge approach.

3. **CALLM Training Pipeline (GRPO-based)**:

    - **Function**: Fine-tunes the base model (Qwen2.5-3B-Instruct) using a composite reward signal.
    - **Mechanism**: Each training instance includes a target sentence, context, and key rules. The model generates multiple candidate outputs per instance (4 per instance), each comprising structured reasoning and a final prediction. The reward function is a weighted sum of four components:
        - Format match reward: $R_{\text{format}}(i) = \mathbf{1}\{\text{match}(c_i)\}$
        - XML tag reward: $R_{\text{xml}}(i) = \min[1, \max[0, \xi_r \sum_{t \in \mathcal{T}} \mathbf{1}\{t \in c_i\} - \xi_p \Delta_i]]$
        - Correctness reward: $R_{\text{corr}}(i) = \mathbf{1}\{\hat{y}_i = y_i\}$
        - Rule alignment reward: $R_{\text{judge}}(i) = \text{CAJudge}(\text{rules}, \text{reasoning}, \hat{y})$
        - Total reward: $R_{\text{total}} = \lambda_1 R_{\text{format}} + \lambda_2 R_{\text{xml}} + \lambda_3 R_{\text{corr}} + \lambda_4 R_{\text{judge}}$, with each $\lambda = 1$
    - Training proceeds in two phases: Phase 1 trains for 2,000 steps using format and correctness rewards to produce the FT model; Phase 2 incorporates the CA-Judge reward for an additional 5,000 steps to produce CALLM.
    - **Design Motivation**: GRPO is preferred over PPO for its elimination of a separate value network, making it computationally efficient and suitable for low-resource settings. CA-Judge is used in place of human feedback due to the scarcity of large-scale preference data in the compliance domain.

### Loss & Training

- GRPO computes advantage estimates via within-group relative ranking, requiring no explicit reward model.
- Two-phase training: Phase 1 establishes foundational capabilities in formatting and correctness; Phase 2 refines reasoning quality through rule alignment.
- Learning rate $5 \times 10^{-6}$, batch size 8, cosine learning rate decay, warmup ratio 0.1, $\beta = 0.04$.
- Phase 1 uses 2×A100 80GB GPUs (3–4 hours per criterion); Phase 2 uses 3×A100 GPUs (~40 hours per criterion).

## Key Experimental Results

### Main Results (F1 Score, 9 Compliance Criteria)

| Criterion | GPT-4o ZS | GPT-4o FS | DeepSeek-R1 | CALLM FT (w/o Judge) | CALLM (Full) |
|-----------|-----------|-----------|-------------|----------------------|--------------|
| Approval | 0.855 | 0.843 | 0.837 | 0.755 | **0.786** |
| Signature | 0.409 | 0.636 | 0.250 | 0.524 | **0.692** |
| C2 Structure | 0.619 | 0.658 | 0.678 | 0.535 | **0.572** |
| C3 Risk Description | 0.422 | 0.450 | 0.495 | 0.564 | **0.712** |
| C4 Mitigation | 0.709 | 0.664 | 0.700 | 0.714 | **0.749** |
| C4 Remediation | 0.552 | 0.601 | 0.529 | 0.397 | **0.570** |
| Overall (macro) | 0.559 | 0.617 | 0.548 | 0.559 | **0.639** |

### Ablation Study (Impact of Training Phases)

| Configuration | Parameters | Overall F1 | Notes |
|---------------|------------|-----------|-------|
| Base PT (pretrained) | 3B | 0.293 | No fine-tuning baseline |
| Base FT (format + correctness reward) | 3B | 0.559 | Phase 1 with surface-level rewards only |
| CALLM (full framework) | 3B | 0.639 | Phase 1 + Phase 2 with CA-Judge |

### Cross-Jurisdictional Generalization (Trained on AU, Tested on AU/UK/CA)

| Model | Parameters | AU | UK | CA |
|-------|------------|-----|-----|-----|
| GPT-4o FS CoT | 1800B | 0.617 | 0.573 | 0.614 |
| DeepSeek-R1 | 671B | 0.548 | 0.505 | 0.550 |
| CALLM | 3B | **0.639** | **0.620** | **0.617** |

### Key Findings
- The 3B-parameter CALLM surpasses GPT-4o (1800B) and DeepSeek-R1 (671B) in macro-F1, demonstrating that domain alignment is more critical than parameter scale.
- Gains from the CA-Judge reward are most pronounced on complex criteria (C3 Risk Description: +0.148; C4 Remediation: +0.173).
- In a human preference study, CALLM was preferred in 74.1% of votes (200/270), with human judgments aligning with CA-Judge scores on 7 out of 9 criteria.
- Strong cross-jurisdictional generalization is observed: training solely on Australian data yields consistent advantages on UK and Canadian data.
- Error analysis reveals that CALLM slightly underperforms GPT-4o on simpler criteria (e.g., Approval), attributable to CALLM's stricter rule adherence and greater sensitivity to malformed target sentences in the dataset.

## Highlights & Insights
- The framing of compliance verification as a rule-matching problem is precise and provides a principled bridge between legal compliance and NLP methodology.
- The CA-Judge as an intermediate evaluation layer is an elegant design—simultaneously providing training signals and emulating the human audit workflow.
- Treating CoT outputs not as an interpretability mechanism but as structured material for human verification is both more intellectually honest and more practically useful.
- The 3B-parameter model outperforming models with hundreds of billions of parameters validates the thesis that domain-specific alignment outweighs indiscriminate parameter scaling.
- The framework is generalizable and transferable to other regulatory compliance domains (e.g., GDPR, environmental law).

## Limitations & Future Work
- Performance depends on the quality and coverage of key rules; incomplete or ambiguous rules may produce inconsistent reward signals.
- GRPO is sensitive to batch composition: if all candidate outputs within a group are poor, the model may reinforce suboptimal patterns.
- CA-Judge remains an approximation of expert judgment and may exhibit bias or hallucination under adversarial prompting.
- The work focuses exclusively on English-language compliance documents; multilingual settings remain unexplored.
- Computational constraints precluded training larger models (7B/14B); the authors hypothesize that scaling may yield additional improvements.
- Human evaluation is limited in scale (5 volunteers) and does not include professional compliance reviewers.

## Related Work & Insights
- The GRPO-based training paradigm parallels DeepSeek-R1, but this work extends the reward signal from general reasoning to domain-specific regulatory rules.
- The paper represents a novel application of LLM-as-Judge, repurposing it from generic evaluation to a domain-specialized assessment tool.
- For AI for Social Good research: in high-stakes social applications, normative domain knowledge must be embedded within the model training loop.
- For compliance automation: full automated approval carries excessive risk; human-in-the-loop collaboration represents a more responsible deployment paradigm.

## Rating
- **Novelty**: ⭐⭐⭐⭐ (The CA-Judge-guided rule-alignment training framework is innovative, though the underlying technical components are relatively mature.)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Cross-jurisdictional testing and human evaluation are commendable, though the human evaluation scale is limited.)
- **Writing Quality**: ⭐⭐⭐⭐⭐ (Motivation is clearly articulated, framework diagrams are informative, and ethical considerations are thoroughly discussed.)
- **Value**: ⭐⭐⭐⭐ (High social impact and a transferable technical framework, though the application domain is relatively niche.)

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Expressive Temporal Specifications for Reward Monitoring](expressive_temporal_specifications_for_reward_monitoring.md)
- [\[AAAI 2026\] Automated Reproducibility Has a Problem Statement Problem](automated_reproducibility_has_a_problem_statement_problem.md)
- [\[AAAI 2026\] Bridging the Skills Gap: A Course Model for Modern Generative AI Education](bridging_the_skills_gap_a_course_model_for_modern_generative_ai_education.md)
- [\[AAAI 2026\] A Switching Framework for Online Interval Scheduling with Predictions](a_switching_framework_for_online_interval_scheduling_with_pr.md)
- [\[AAAI 2026\] RcAE: Recursive Reconstruction Framework for Unsupervised Industrial Anomaly Detection](rcae_recursive_reconstruction_framework_for_unsupervised_industrial_anomaly_dete.md)

<!-- RELATED:END -->
