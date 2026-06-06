---
title: >-
  [Paper Note] Decide less, communicate more: On the construct validity of end-to-end fact-checking in medicine
description: >-
  [ACL 2026][Social Computing][Medical fact-checking] The authors conducted an annotation study of 1,000 instances using 5 clinical experts on real-world health claims from RedHOT (Reddit Health Discussions). They found th…
tags:
  - "ACL 2026"
  - "Social Computing"
  - "Medical fact-checking"
  - "construct validity"
  - "RedHOT"
  - "RCT"
  - "communication model"
date: 2026-05-08
content_hash: 1d057b37f344f2a8
---

# Decide less, communicate more: On the construct validity of end-to-end fact-checking in medicine

**Conference**: ACL 2026  
**arXiv**: [2506.20876](https://arxiv.org/abs/2506.20876)  
**Code**: https://github.com/SebaJoe/decide-less-communicate-more (Available)  
**Area**: Fact-checking / Medical NLP / Position Paper / HCI  
**Keywords**: Medical fact-checking, construct validity, RedHOT, RCT, communication model

## TL;DR
The authors conducted an annotation study of 1,000 instances using 5 clinical experts on real-world health claims from RedHOT (Reddit Health Discussions). They found that end-to-end medical fact-checking is untenable at the construct validity level due to three major barriers that even experts cannot eliminate: difficulties in evidence linking, underspecified claims, and subjective severity determination. Consequently, they propose reframing medical fact-checking as an "interactive clinician-patient communication model" rather than a "classification → verdict" pipeline.

## Background & Motivation
**Background**: The traditional fact-checking pipeline (Guo et al. 2022) follows a three-stage process: Claim Detection → Evidence Retrieval → Claim Verification. Mainstream medical datasets (pubhealth, scifact, healthver, covid-fact, covert, healthfc, etc.) formulate the task as a multi-class classification: "Given a claim + retrieved evidence → Output True/False/Unproven."

**Limitations of Prior Work**: (i) Existing data are mostly sourced from professional fact-checking websites, news, or RCT abstracts, which strips away the original context and differs significantly from real "social media claims"; (ii) Even with SOTA retrieval systems and LLMs, automated systems remain generally unusable on in-the-wild claims; (iii) No study has systematically quantified a key question: When the claims are replaced with raw health queries from Reddit, can **human experts** reach a consensus verdict, and what criteria would they use?

**Key Challenge**: Current task definitions simplify fact-checking into a "content-relevance" determination. However, real medical queries involve (a) unstated patient intent, (b) blind spots in high-quality evidence (RCT) coverage, and (c) subjective scales of individual experts regarding "how serious an error is." These are issues of construct validity inherent to the task itself, which model optimization cannot resolve.

**Goal**: (i) Measure the "human upper bound" of medical fact-checking through an expert + AI-in-the-loop annotation study; (ii) Deconstruct the root causes of the failure of the end-to-end paradigm; (iii) Provide an alternative: an interactive communication model.

**Key Insight**: Instead of tuning models, the authors conducted an "extremely idealized" experiment. Five clinical experts were provided with contextualized claims (original Reddit posts + PIO triplets), 10 automatically retrieved RCT abstracts, and a user-friendly annotation interface to rate relevance, synthesize evidence, and write plain-language explanations. If agreement at this "idealized" human upper bound is poor, then the task definition itself is flawed.

**Core Idea**: Medical fact-checking should be reframed as an interactive communication process involving "intent clarification + guided evidence retrieval + multi-perspective explanation," rather than end-to-end True/False classification.

## Method

### Overall Architecture
**AI-in-the-loop annotation pipeline (Figure 1)**: (1) Claims are sampled from the RedHOT corpus across 24 health-related subreddits (e.g., r/Epilepsy, r/CysticFibrosis, r/ADHD, r/lupus); (2) PIO (Population / Intervention / Outcome) triplets are extracted using Llama-3.1-405B-Instruct to stabilize expert focus; (3) 10 candidate RCTs are retrieved from the Trialstreamer database (800,000 RCT abstracts) using the stella_en_400M_v5 dense retriever; (4) The post, claim span, PIO, and 10 abstracts with automated tiering are fed into a web annotation interface; (5) Experts first rate abstract relevance across four dimensions (Relevant / Somewhat / Irrelevant) based on PIO + Overall → Support / Partially Support / Partially Refute / Refute → The synthesis stage involves two rounds: Overall Support and Expert Support → Writing paragraph-level plain-language explanations. The study consists of 3 rounds, including 1,000 abstract-level annotations and 100 synthesis explanations. The guidelines were revised during the first two rounds, and the final round involved 5 claims × 5 experts × 50 abstracts for agreement analysis.

### Key Designs

1. **AI-in-the-loop expert annotation: Offloading manual labor to machines, leaving judgment to humans**:
    - **Function**: Allows medical experts to focus on "evidence relevance + synthesis"—tasks only humans can perform—while offloading PIO extraction, retrieval, and tiering to LLMs/IR.
    - **Mechanism**: Automated PIO extraction uses a prompt adapted from Duke EBM guidelines (Appendix J), achieving >90% accuracy on a 55-item expert-validated subset. Automated retrieval tested 8 combinations of sentence-to-sentence/sentence-to-passage + query/document; ultimately, PIO→Abstract S2S was selected (highest average expert relevance score). Automated tiering categorizes 10 abstracts into four tiers based on relevance for expert adjustment.
    - **Design Motivation**: To measure the "human upper bound," judgments that only humans can make must be isolated from automatable parts. Otherwise, low inter-annotator agreement cannot be distinguished between task difficulty and poor tooling.

2. **6-level fine-grained veracity labels + Dual-track synthesis (Overall Support vs Expert Support)**:
    - **Function**: Captures the discrepancy between "evidence-based verdicts" and "expert clinical knowledge verdicts" to avoid conflating the two types of judgment.
    - **Mechanism**: Each claim underwent two rounds in the synthesis phase—Overall Support relies solely on given evidence, while Expert Support allows for clinical knowledge. Each round selects one of 6 labels: {No Relevant Abstracts/No Expert Opinion, Refutes, Partially Refutes, Inconclusive, Partially Supports, Supports}.
    - **Design Motivation**: Existing datasets either use binary True/False labels or mix "expert experience" and "empirical support" into one label. Separation reveals whether "lack of evidence" or "subjective verdict" is the true bottleneck.

3. **Three rounds of iterative guideline refinement + Explanation-first approach**:
    - **Function**: Distills the real sources of expert disagreement into analyzable phenomena rather than flattening them into a single label.
    - **Mechanism**: After each round, two co-first authors discussed collective findings with experts for ~4 hours. Round 1 lacked explicit PIO context and Expert Support fields; Round 2 added refined label definitions; Round 3 underwent substantial restructuring (LLM filtering of non-RCT-verifiable claims + re-extracted PIO + improved retrieval + detailed guidelines). Despite this, Cohen’s $\kappa$ for Overall Support on 5 claims in Round 3 was only 0.124, with 20 out of 25 verdicts labeled as "No Relevant Abstracts."
    - **Design Motivation**: To argue that "the problem is the task definition, not the tools," engineering factors must be optimized to the point where remaining disagreement can be attributed to construct validity.

### Loss & Training
This work is a position paper and annotation study; it does not involve model training.

## Key Experimental Results

### Main Results
Agreement in the final round (Table 2, 5 claims × 5 experts × 50 abstracts; higher Cohen’s $\kappa$ indicates better agreement):

| Dimension | $\kappa$ | Level (Landis & Koch) |
|------|---------:|----------------------|
| Population (Abstract-level) | 0.416 | Moderate |
| Intervention | 0.714 | Substantial |
| Outcome | 0.200 | Slight |
| Overall (Abstract-level) | 0.155 | Slight |
| Tab Support (per abstract → claim) | 0.170 | Slight |
| **Overall Support (synthesis)** | **0.124** | **Slight** |
| Expert Support (synthesis) | **−0.184** | Worse than random |

Dataset comparison (Summary of Table 1):

| Dataset | Domain / Source | Labels | Explanations | Evidence Type | Original Context? |
|--------|----------|------|:------------:|--------|:----------------:|
| pubhealth | Public Health / Fact-check sites | True/Unproven/False/Mixture | ✗ | News snippets | ✗ |
| healthver | COVID-19 / News+Blogs+Social | Supports/Refutes/Neutral | ✗ | Sci. papers | ✗ |
| healthfc | Health / Medizin Transparent | Supported/Refuted/NEI | ✓ | SR + RCT | ✗ |
| **Ours (2025)** | **Reddit original posts** | **6-level fine-grained** | ✓ | **RCT abstracts** | ✓ |

### Ablation Study
The paper does not compare models; "ablation" is reflected in the impact of guideline evolution on $\kappa$:

| Annotation Round | Changes | Overall Support $\kappa$ |
|--------|------|--------------------------|
| Round 1 | No PIO display / No Expert Support field | Very low (not listed) |
| Round 2 | Refined label definitions | Still very low |
| Round 3 | Added PIO + Filtered non-RCT + Improved IR + Detailed guidelines | 0.124 (Slight) |

Engineering optimizations failed to improve synthesis-level agreement, strongly supporting the argument that the issue lies in the task definition.

### Key Findings
- **Most in-the-wild claims are not verifiable within the scope of RCTs**: In the final round, 20 out of 25 independent verdicts (5 claims × 5 experts) chose "No Relevant Abstracts." For 3 claims, all five experts unanimously agreed there were no relevant RCTs. Reasons included: lack of intervention (e.g., "mood linked to menstrual cycle"), unethical setups (e.g., smoking as an RCT intervention), infeasibility (rare PIO combinations), or lack of utility.
- **Underspecified claims lead to divergent explanations**: For a claim like "pineapple juice helps with sinus issues" on r/CysticFibrosis, experts did not know if the patient meant immediate or progressive improvement, or if they considered sugar metabolism risks. Consequently, five experts provided five different framings.
- **Misguided premises tear apart judgments**: For an ADHD patient asking if "herbal medicine can regulate periods" because "medication efficacy fluctuates with the cycle," experts split between (a) directly verifying the herbal claim and (b) debunking the premise first. There is no definition of which approach is "correct."
- **Veracity severity determination is inherently subjective**: Under the same evidence, a pineapple juice claim might be judged "slightly inaccurate" by one expert and "severely misleading" by another; both are reasonable.
- **Plain-language explanations align better than labels**: Despite label disagreement, experts recognized each other's reasoning paths in natural language—suggesting "multi-perspective explanations" reflect the true scope of consensus better than single labels.

## Highlights & Insights
- **Using a "poor human upper bound" to diagnose task definition issues**: This is a rare empirical strategy in position papers, treating expert disagreement as a diagnostic signal of construct invalidity rather than a tool defect. This methodology is portable to any "LLM as a judge" research.
- **The "Overall Support vs Expert Support" dual-track design is clever**: By separating "lack of evidence" from "expert experience," researchers can quantify for the first time that "lack of evidence" is the primary driver of disagreement (20/25 instances).
- **Shift from single-round verdict to a "Communication Model"** (Figure 2, §5): Systems should actively follow up, use guided retrieval, pivot non-verifiable claims to verifiable versions, and produce multi-perspective explanations + uncertainty instead of a 0/1 verdict. This aligns naturally with IR, clarification question generation, and PIO-guided dialog.

## Limitations & Future Work
- **Small sample size**: The final agreement analysis covers only 5 claims × 5 experts. While annotating 10 abstracts per claim is labor-intensive, the statistical strength of the $\kappa$ values is limited.
- **Retriever is not the focus**: Only the stella model was used, without comparing RM/BM25/LLM-rerank. However, the authors state that retrieval itself was not the main research target.
- **RCT-only evidence**: Omitting meta-analyses, cohort studies, and case reports. Adding these would introduce new challenges in evidence hierarchy integration.
- **No prototype for the communication model**: Appendix D only discusses evaluation ideas without delivering a dialog system. Implementation would require patient-system simulators + expert evaluation loops.
- **Patient intent modeling remains open**: Underspecified claims require models to infer unstated intent, which exceeds the current scope of dialog clarification work.
- **Ethics and Privacy**: Releasing a subset of Reddit posts is sensitive, even though the original RedHOT authors provided opt-out opportunities.

## Related Work & Insights
- **vs RedHOT (Wadhwa et al. 2023)**: RedHOT only goes up to claim annotation (no verification). Ours is the first to run the full "claim + evidence + synthesis" pipeline on RedHOT and expose failure modes.
- **vs healthfc (Vladika et al. 2024) / scifact / covid-fact / healthver**: These datasets strip claims from context and use closed labels, adhering to the end-to-end paradigm. Ours argues this paradigm lacks construct validity in in-the-wild scenarios.
- **vs AmbiFC (Glockner et al. 2024)**: AmbiFC recognizes that "claims can have multiple reasonable interpretations." Ours further demonstrates that this ambiguity is ineradicable in medical contexts and incorporates it into the communication model.
- **vs Tree-of-Clarifications (Kim et al. 2023) / dialog clarification series**: These provide the technical foundation for the communication model in §5.1, using clarification question trees to cover potential interpretations.
- **vs Decompose-then-verify (FActScore, Chen et al. 2022, Wanner et al. 2024)**: Ours notes that claim decomposition cannot handle "latent premises," a unique challenge in medicine.
- **vs Warren et al. 2025 (fact-checker practice survey)**: Consistent with their call for "nuanced, multi-perspective explanations," ours provides quantitative support for this call using expert annotation data.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to use expert upper-bound experiments to argue construct validity issues in medical fact-checking with a strong stance.
- Experimental Thoroughness: ⭐⭐⭐ 1,000 instances annotated, but the final agreement sample size is small; lacks model implementation experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, detailed appendices (guidelines, retrieval config, full claims, case studies).
- Value: ⭐⭐⭐⭐⭐ Directly challenges the paradigm of the entire medical fact-checking subfield and provides actionable alternative directions; high long-term impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] PropVG: End-to-End Proposal-Driven Visual Grounding with Multi-Granularity Discrimination](../../ICCV2025/social_computing/propvg_end-to-end_proposal-driven_visual_grounding_with_multi-granularity_discri.md)
- [\[ACL 2026\] VeriTaS: The First Dynamic Benchmark for Multimodal Automated Fact-Checking](veritas_the_first_dynamic_benchmark_for_multimodal_automated_fact-checking.md)
- [\[AAAI 2026\] Fact2Fiction: Targeted Poisoning Attack to Agentic Fact-checking System](../../AAAI2026/social_computing/fact2fiction_targeted_poisoning_attack_to_agentic_fact-check.md)
- [\[ACL 2026\] ClaimDB: A Fact Verification Benchmark over Large Structured Data](claimdb_a_fact_verification_benchmark_over_large_structured_data.md)
- [\[ACL 2026\] The Proxy Presumption: From Semantic Embeddings to Valid Social Measures](the_proxy_presumption_from_semantic_embeddings_to_valid_social_measures.md)

</div>

<!-- RELATED:END -->
