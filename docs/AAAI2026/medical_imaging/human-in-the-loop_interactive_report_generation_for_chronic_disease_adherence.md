---
title: >-
  [Paper Note] Human-in-the-Loop Interactive Report Generation for Chronic Disease Adherence
description: >-
  [AAAI 2026][Medical Imaging][Human-in-the-loop] This paper presents a "physician-in-the-loop" interactive interface that restricts AI to the roles of data organization and draft generation. Through a single-page editor, chart–text pairing, and automated urgency stratification, it enables efficient and accountable chronic disease adherence report generation. A pilot study reveals an "accountability paradox": even when AI-generated quality matches the physician manual-authoring baseline, review time cannot be significantly reduced, because clinical responsibility demands complete verification.
tags:
  - AAAI 2026
  - Medical Imaging
  - Human-in-the-loop
  - chronic disease management
  - report generation
  - medication adherence
  - clinical AI collaboration
date: 2026-05-08
content_hash: 539b6f887b581b5e
---

# Human-in-the-Loop Interactive Report Generation for Chronic Disease Adherence

**Conference**: AAAI 2026
**arXiv**: [2601.06364](https://arxiv.org/abs/2601.06364)
**Code**: None
**Area**: Medical AI / Human-Computer Interaction / Clinical Decision Support
**Keywords**: Human-in-the-loop, chronic disease management, report generation, medication adherence, clinical AI collaboration

## TL;DR

This paper presents a "physician-in-the-loop" interactive interface that restricts AI to the roles of data organization and draft generation. Through a single-page editor, chart–text pairing, and automated urgency stratification, it enables efficient and accountable chronic disease adherence report generation. A pilot study reveals an "accountability paradox": even when AI-generated quality matches the physician manual-authoring baseline, review time cannot be significantly reduced, because clinical responsibility demands complete verification.

## Background & Motivation

In chronic disease management, clinicians must regularly provide patients with personalized adherence feedback reports to prevent avoidable hospitalizations. Medication non-adherence remains prevalent, driving substantial preventable admissions and healthcare costs. Documentation already consumes considerable clinician time—research indicates that primary care physicians spend a significant portion of their workday on EHR tasks—leaving little time for individualized patient communication.

The root cause is: **manual authoring ensures clinical accuracy but cannot scale, while fully automated AI generation can scale but erodes trust in patient-facing contexts**. Existing human-in-the-loop tools often shift the burden from "drafting" to "reviewing," making "approval" a bottleneck through multi-screen workflows.

The paper's starting point is to **redefine the boundary of AI assistance**: rather than autonomous generation, AI performs "bounded preparation"—solely responsible for organizing fragmented medical records, device trends, and patient conversations into structured drafts, while all clinical decision authority is fully retained by the physician. The core idea is to achieve both efficiency and accountability through careful task division (AI organizes, physician decides) and interface design (single-page, recognition over recall, progressive disclosure).

## Method

### Overall Architecture

The system architecture comprises four core stages: (1) data ingestion—collecting information from medication lists, device trends, and patient conversations; (2) AI processing—parallel content generation (structured drafts) and risk assessment (urgency stratification); (3) single-page review—physicians complete the entire workflow from browsing to editing to approval within one HTML page; (4) feedback loop—approved content is exported to patients, forming a "physician–AI–patient" cycle. The entire design follows four principles: AI prepares rather than decides, recognition over recall, direct manipulation over indirect control, and progressive disclosure.

### Key Designs

1. **Bounded AI Preparation and Single-Pass Approval**:

    - Function: Strictly limits the AI's role to the data organization layer; physicians complete approval through a single linear review.
    - Mechanism: AI uses Qwen3-8B (temperature 0.7, max_tokens 1200) to organize input data into a fixed-template draft structure—three sub-steps per topic: "what happened, why it matters, what to do next." Charts are placed immediately adjacent to their corresponding text for local verification. When data are missing, the draft explicitly marks the gap rather than filling it with inferences.
    - Design Motivation: No content is sent without physician approval; the physician remains the author of record at all times. Fixed templates and fixed decoding parameters ensure output consistency and reproducibility. The single-page design reduces context-switching costs.

2. **Automated Urgency Assessment and Conservative Fail-Safe Mechanism**:

    - Function: Automatically stratifies patient cases by urgency (urgent/attention/stable) to reduce the physician's cognitive triage burden.
    - Mechanism: The system analyzes vital sign trends, adherence gaps, and conversation content to generate urgency labels, displayed with color coding at the top of each case (red for urgent). The key safety rule is **conservative fail-safe escalation**: if disease-specific critical monitoring tasks are missed (e.g., no daily blood pressure check for hypertensive patients, no blood glucose monitoring for diabetic patients), the case is automatically escalated to "urgent" regardless of other indicators.
    - Design Motivation: Prevents algorithmic optimism from masking critical gaps. When the LLM is unavailable, rule-based heuristics provide fallback classification. In practice, the LLM produces an initial estimate, after which a rule validator performs a secondary check that may escalate or adjust the final label.

3. **Single-Page Interactive Interface Design**:

    - Function: Integrates all review, editing, and approval functions within a single HTML page.
    - Mechanism: Sections are navigated via anchors; sentences are edited in place; small controls are placed alongside the content they affect (e.g., checkboxes to confirm medications, menus to set follow-up intervals, approval/export buttons). Charts appear immediately beside the sentences they explain, enabling rapid local verification. The typical workflow is: open case → browse top to bottom → make focused edits → select follow-up interval → approve.
    - Design Motivation: Follows the "recognition over recall" principle (fixed paragraph order and clearly named options); supports "direct manipulation" (in-place editing with instant preview); employs "progressive disclosure" (each section is brief and expands supplementary content only when needed). Scope is deliberately kept narrow: no trust badges, approval hierarchies, or learning-from-edits features are included.

### Loss & Training

This paper does not involve model training; it presents system design and a user study. The AI component uses Qwen3-8B with fixed parameters and no fine-tuning. Evaluation employs a 12-dimension 1–10 Likert scale across three domains: core medical judgment (Q1–5), data and factual accuracy (Q6–8), and workflow integration (Q9–12). The baseline is set at 5 (representing the quality level of current physician manual-authoring practice).

## Key Experimental Results

### Main Results

Three physicians reviewed 24 cases (14 urgent / 8 attention / 2 stable):

| Evaluation Dimension | Score (1–10, baseline = 5) |
|---|---|
| Urgency Assessment Accuracy (Q1) | 5.04 |
| Intervention Recommendations (Q2) | 4.42 |
| Critical Task Identification (Q3) | 4.62 |
| Clinical Appropriateness (Q4) | 4.88 |
| Risk Rationale Quality (Q5) | 4.83 |
| Data Completeness Recognition (Q6) | 4.75 |
| Chart Information Value (Q7) | 4.96 |
| Adherence Description Accuracy (Q8) | **5.25** |
| Consultation Preparedness (Q9) | 4.83 |
| Time Efficiency Improvement (Q10) | 4.79 |
| Information Location Efficiency (Q11) | **5.08** |
| Overall Satisfaction (Q12) | 4.87 |
| **Overall Mean** | **4.86** |

### Ablation Study

| Configuration | Key Metric | Note |
|---|---|---|
| Physician 1 Time Efficiency (Q10) | 5.00 | On par with manual authoring |
| Physician 2 Time Efficiency (Q10) | 4.00 (p=0.007) | Significantly slower than manual |
| Physician 3 Time Efficiency (Q10) | 5.38 (p=0.197) | Trend improvement, not significant |
| Mean Edit Rate | 8.3% (95% CI: 3.3–13.3%) | Minimal modifications |
| Safety Concerns | 1/24 minor | No safety-critical issues |
| Safety-Critical Issues | 0/24 | Zero serious safety events |
| Scale Reliability (Cronbach α) | 0.89 | High internal consistency |

### Key Findings

- AI-generated drafts achieved quality on par with physician manual-authoring practice (overall mean 4.86 vs. baseline 5.0), with a content edit rate of only 8.3% and no safety-critical issues.
- **Accountability paradox**: despite adequate quality and minimal edits, perceived time savings (Q10 = 4.79) showed no significant difference from baseline ($t(23)=-1.23, p=0.233$).
- Adherence description accuracy (Q8 = 5.25) and information location efficiency (Q11 = 5.08) were the only two dimensions exceeding the baseline.
- Intervention recommendations (Q2 = 4.42) was the weakest dimension, reflecting AI limitations in clinical decision-making suggestions.
- Inter-physician variation (5.31 vs. 4.28 vs. 5.00) reflects individual tolerance for AI-generated text rather than systematic model errors.

## Highlights & Insights

- The **accountability paradox** is the paper's most important contribution: in high-stakes clinical settings, professional responsibility requires complete verification even when AI output is accurate, rendering the assumption that "better AI = less review time" invalid in clinical practice.
- The three interaction pattern design paradigms have broad applicability: (1) bounded generation + recognition-based review, (2) attention management + visual urgency marking, (3) conservative safety + fail-safe rules. These patterns are transferable to other high-accountability domains such as law and finance.
- The task-division design philosophy—"AI organizes, physician decides"—diverges from the prevailing direction of maximizing automation in most AI-assisted systems, and better aligns with the practical demands of high-stakes scenarios.
- The conservative fail-safe escalation mechanism is a safety pattern worth generalizing: automatically escalating urgency when critical monitoring tasks are absent prevents algorithmic optimism.
- The chart–text pairing local verification design effectively improved information location efficiency (Q11 = 5.08); this interface pattern merits adoption in other document review systems.

## Limitations & Future Work

- The study scale is very small (3 physicians × 24 cases), with limited statistical power; conclusions require validation at larger scale.
- Cases are skewed toward high risk (14 urgent / 8 attention / 2 stable), unrepresentative of the actual case distribution in real clinical settings.
- The baseline is set at a subjective score of 5 (representing manual authoring), lacking an objective external benchmark.
- Actual review time was not measured; only subjective ratings were used, which may introduce perceptual bias.
- Only Qwen3-8B was evaluated; it remains unexplored whether more capable models could overcome the accountability paradox.
- Discussion of legal and insurance frameworks remains at the analytical level without proposing concrete solutions.
- Comparative experiments with existing clinical documentation systems (e.g., EHR integration approaches) are absent.

## Related Work & Insights

- Unlike work on automated SOAP note generation, this paper focuses on patient-facing communication (rather than physician-internal documentation), imposing higher demands on trust and accountability.
- The finding that the information bottleneck shifts from "generation" to "verification" is consistent with Lee et al. 2024 on AI trust.
- The accountability paradox poses a fundamental challenge to the broader field of "AI-assisted professional work": in domains where legal liability cannot be delegated to AI, the ceiling on efficiency gains lies not in AI capability but in the human obligation to verify.
- Implications for medical AI research: future directions may lie not in improving AI accuracy but in designing "selective verification mechanisms that preserve accountability" (e.g., confidence-based segmented approval, progressive trust based on historical accuracy).

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] VitalDiagnosis: AI-Driven Ecosystem for 24/7 Vital Monitoring and Chronic Disease Management](vitaldiagnosis_ai-driven_ecosystem_for_247_vital_monitoring_and_chronic_disease_.md)
- [\[AAAI 2026\] A Disease-Aware Dual-Stage Framework for Chest X-ray Report Generation](a_disease-aware_dual-stage_framework_for_chest_x-ray_report_.md)
- [\[AAAI 2026\] PriorRG: Prior-Guided Contrastive Pre-training and Coarse-to-Fine Decoding for Chest X-ray Report Generation](priorrg_prior-guided_contrastive_pre-training_and_coarse-to-fine_decoding_for_ch.md)
- [\[AAAI 2026\] ReCoN-Ipsundrum: An Inspectable Recurrent Persistence Loop Agent with Affect-Coupled Cognition](recon-ipsundrum_an_inspectable_recurrent_persistence_loop_agent_with_affect-coup.md)
- [\[AAAI 2026\] FIA-Edit: Frequency-Interactive Attention for Efficient and High-Fidelity Inversion-Free Text-Guided Image Editing](fia-edit_frequency-interactive_attention_for_efficient_and_high-fidelity_inversi.md)

</div>

<!-- RELATED:END -->
