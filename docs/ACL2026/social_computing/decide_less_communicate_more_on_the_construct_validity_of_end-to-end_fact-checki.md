---
title: >-
  [Paper Note] Decide less, communicate more: On the construct validity of end-to-end fact-checking in medicine
description: >-
  [ACL 2026][Social Computing][construct validity] The authors conducted an annotation study of 1,000 instances using 5 clinical experts on authentic claims from RedHOT (Reddit Health Discussions). They found that end-to-end medical fact-checking lacks construct validity due to three insurmountable barriers: difficulties in linking evidence, underspecified claims, and
tags:
  - ACL 2026
  - Social Computing
  - construct validity
  - RedHOT
  - RCT
date: 2026-05-08
content_hash: ba1b1063f4d44f5a
---
# Decide less, communicate more: On the construct validity of end-to-end fact-checking in medicine

**Conference**: ACL 2026 Findings  
**arXiv**: [2506.20876](https://arxiv.org/abs/2506.20876)  
**Code**: https://github.com/SebaJoe/decide-less-communicate-more (Available)  
**Area**: Fact-checking / Medical NLP / Position Paper / Human-Computer Interaction  
**Keywords**: Medical fact-checking, construct validity, RedHOT, RCT, Communication Model

## TL;DR
The authors conducted an annotation study of 1,000 instances using 5 clinical experts on authentic claims from RedHOT (Reddit Health Discussions). They found that end-to-end medical fact-checking lacks construct validity due to three insurmountable barriers: difficulties in linking evidence, underspecified claims, and subjective severity judgments. Consequently, the paper proposes reframing medical fact-checking as an "interactive clinician-patient communication model" rather than a "classification-then-verdict" pipeline.

## Background & Motivation
**Background**: Traditional fact-checking pipelines (Guo et al. 2022) consist of three stages: Claim Detection → Evidence Retrieval → Claim Verification. Mainstream medical datasets (pubhealth / scifact / healthver / covid-fact / covert / healthfc, etc.) frame the task as a multi-class classification: "Given a claim + retrieved evidence → output True/False/Unproven."

**Limitations of Prior Work**: (i) Existing data are mostly sourced from professional fact-checking sites, news, or RCT abstracts, stripping away the original context, which differs significantly from actual "social media claims." (ii) Even with SOTA retrieval systems and LLMs, automated systems remain generally unusable for in-the-wild claims. (iii) A critical question remains unquantified: when claims are replaced with raw, authentic patient inquiries from Reddit, can **human experts** achieve consensus on a verdict, and what criteria would they use?

**Key Challenge**: Existing task definitions simplify fact-checking to "content-relevance" judgments. However, real-world medical queries involve (a) latent patient intentions, (b) coverage gaps in high-quality evidence (RCTs), and (c) subjective scales of "severity of error" among individual experts. These are issues of construct validity inherent to the task itself that cannot be resolved by simply optimizing models.

**Goal**: (i) Measure the "human upper bound" of medical fact-checking through an expert + AI-in-the-loop annotation study; (ii) Deconstruct the root causes of the end-to-end paradigm's failure; (iii) Propose an alternative: an interactive communication model.

**Key Insight**: Instead of tuning models, the authors conducted an "extremely idealized" experiment. Five clinical experts were provided with contextualized claims (original Reddit posts + PIO triplets), 10 automatically retrieved RCT abstracts, and a user-friendly annotation interface to evaluate relevance, synthesize evidence, and write plain-language explanations. If agreement remains poor even at this "idealized" human upper bound, the current task definition itself is flawed.

**Core Idea**: Medical fact-checking should be reframed as an interactive communication process involving "clarifying intent + guided evidence retrieval + multi-perspective explanation," rather than end-to-end True/False classification.

## Method

### Overall Architecture
**AI-in-the-loop annotation pipeline (Figure 1)**: (1) Extract claims from the RedHOT corpus across 24 health-related subreddits (e.g., r/Epilepsy, r/CysticFibrosis, r/ADHD). (2) Use Llama-3.1-405B-Instruct to extract PIO (Population / Intervention / Outcome) triplets to stabilize expert focus. (3) Use the `stella_en_400M_v5` dense retriever to fetch 10 candidate RCTs from the Trialstreamer database (800k RCT abstracts). (4) Feed the post, claim span, PIO, and 10 abstracts with automated tiering into a web annotation interface. (5) Experts first rate abstract relevance across four dimensions (Relevant / Somewhat / Irrelevant) based on PIOs and Overall, then rate veracity (Support / Partially Support / Partially Refute / Refute). (6) The synthesis stage includes two rounds: Overall Support and Expert Support, followed by writing paragraph-level plain-language explanations. The study comprised 3 rounds, including 1,000 abstract-level annotations and 100 synthesis explanations. Guidelines were iteratively refined, with the final round involving 5 claims × 5 experts × 50 abstracts for agreement analysis.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    A["RedHOT 24 Health Subreddits<br/>Extract real claims"] --> S1
    subgraph S1["AI-in-the-loop Expert Annotation (Design 1: Automated Pre-processing)"]
        direction TB
        B["Automated PIO Extraction<br/>Llama-3.1-405B, Accuracy >90%"] --> C["RCT Retrieval<br/>Trialstreamer 800k + stella, top 10"]
        C --> D["Automated Relevance Tiering<br/>10 abstracts pre-sorted into four tiers"]
    end
    S1 --> E["Expert Annotation Interface<br/>Post + claim + PIO + 10 abstracts + tier"]
    subgraph S2["6-level fine-grained veracity labels + Dual-track synthesis (Design 2)"]
        direction TB
        F["Expert rates abstract relevance<br/>P/I/O/Overall dimensions"] --> G["Expert rates veracity<br/>Support → Refute range"]
        G --> H["Dual-track Synthesis<br/>Overall Support / Expert Support, 6-level labels each"]
    end
    E --> S2
    S2 --> I["Explanation First<br/>Paragraph-level plain-language multi-perspective explanation"]
    I --> J["Agreement Analysis<br/>Cohen's κ → Diagnosis of construct validity"]
    J -.->|Three rounds of iterative guideline refinement (Design 3)| S1
```

### Key Designs

**1. AI-in-the-loop expert annotation: Offloading labor to machines, leaving "evidence relevance + synthetic judgment" to humans**

To measure the "upper bound of agreement achievable by human experts under ideal conditions," human-specific judgment must be isolated from automated steps. PIO extraction, evidence retrieval, and relevance tiering were outsourced to LLMs/IR. Automated PIO extraction used prompts based on Duke EBM guidelines (Appendix J), achieving $>90\%$ accuracy on a 55-item expert-verified subset. For retrieval, S2S (PIO→Abstract) was selected as it yielded the highest average expert relevance scores. Automated tiering then pre-sorted 10 abstracts for expert fine-tuning. Experts thus focused solely on relevance and synthesis, ensuring that agreement metrics purely reflected task difficulty.

**2. 6-level fine-grained veracity labels + dual-track synthesis: Separating "what the evidence says" from "expert clinical experience"**

Existing datasets often use simple True/False labels or conflate "empirical support" with "expert experience," making it impossible to distinguish whether disagreement arises from a lack of evidence or subjective adjudication. This study conducts two synthesis rounds per claim: Overall Support (strictly based on provided RCT evidence) and Expert Support (permitting clinical knowledge). Each round uses 6 labels: $\{\text{No Relevant Abstracts/No Expert Opinion, Refutes, Partially Refutes, Inconclusive, Partially Supports, Supports}\}$. By decoupling these, the research demonstrates that "insufficient evidence," rather than "subjective disagreement," drives the collapse of consistency (20 out of 25 final-round verdicts were "No Relevant Abstracts").

**3. Three rounds of iterative guideline refinement + explanation priority: Eliminating optimizable engineering factors to isolate systemic task issues**

The authors argue that the problem lies in the construct validity of the end-to-end paradigm, not the tools. This required exhausting engineering improvements before analyzing residual disagreement. After each of the 3 rounds, the co-first authors and experts discussed for ~4 hours. Improvements included adding PIO contexts, refining label definitions, using LLMs to filter non-RCT-verifiable claims, and improving retrieval. Despite these efforts, Cohen’s $\kappa$ for Overall Support in the final round (5 claims) was only 0.124. This low agreement, despite maximized engineering support, confirms the issue lies in the task definition. Furthermore, experts were required to write paragraph-level explanations, which were found to reflect consensus better than single labels.

### Loss & Training
This is a position paper and annotation study; no model training was involved.

## Key Experimental Results

### Main Results
Final round agreement (Table 2, 5 claims × 5 experts × 50 abstracts; higher Cohen’s $\kappa$ indicates better agreement):

| Dimension | $\kappa$ | Level (Landis & Koch) |
|------|---------:|----------------------|
| Population (abstract-level) | 0.416 | moderate |
| Intervention | 0.714 | substantial |
| Outcome | 0.200 | slight |
| Overall (Abstract-level) | 0.155 | slight |
| Tab Support (per abstract → claim) | 0.170 | slight |
| **Overall Support (synthesis)** | **0.124** | **slight** |
| Expert Support (synthesis) | **−0.184** | worse than random |

Dataset Comparison (Table 1 Summary):

| Dataset | Domain / Source | Labels | Explanations | Evidence Type | Original Context? |
|--------|----------|------|:------------:|--------|:----------------:|
| pubhealth | Public Health / Fact-check sites | True/Unproven/False/Mixture | ✗ | News passages | ✗ |
| healthver | COVID-19 / News+Blogs+Social | Supports/Refutes/Neutral | ✗ | Scientific papers | ✗ |
| healthfc | Health / Medizin Transparent | Supported/Refuted/NEI | ✓ | Systematic reviews + RCT | ✗ |
| **our case study (2025)** | **Reddit posts** | **6-level granularity** | ✓ | **RCT abstracts** | ✓ |

### Ablation Study
The "ablation" is reflected in the impact of iterative guideline evolution on $\kappa$:

| Annotation Round | Changes | Overall Support $\kappa$ |
|--------|------|--------------------------|
| Round 1 | No PIO display / No Expert Support field | Very low (not listed) |
| Round 2 | Refined label definitions | Very low |
| Round 3 | PIO added + non-RCT-verifiable filtered + improved retrieval + finer guidelines | 0.124 (slight) |

Engineering optimizations failed to significantly improve synthesis-level agreement, supporting the argument that the problem is in the task definition.

### Key Findings
- **Most in-the-wild claims are not verifiable via RCTs**: In the final round, 20 out of 25 independent verdicts were "No Relevant Abstracts." Reasons include: no intervention (e.g., "mood related to cycle"), unethical interventions (e.g., smoking), infeasible PIO combinations, or lack of clinical utility.
- **Underspecification leads to divergent explanations**: For a claim like "pineapple juice helps sinus issues," experts disagreed on whether the patient meant immediate or progressive improvement, or whether to factor in metabolic risks of high sugar.
- **Misguided claims (false premises) fracture judgment**: When an ADHD patient asks if "herbs regulate periods" because "medication efficacy fluctuates with the cycle," experts split between (a) verifying the herbal claim and (b) debunking the premise first.
- **Veracity severity is naturally subjective**: Based on the same evidence, one expert might judge a claim as "slightly inaccurate" while another views it as "seriously misleading."
- **Plain-language explanations align better than labels**: Despite label disagreement, experts often validated each other's reasoning paths in natural language.

## Highlights & Insights
- **Diagnosing task issues through poor "human upper bounds"**: This empirical strategy uses expert disagreement as a diagnostic signal for construct invalidity rather than a tool deficiency—a methodology applicable to any "LLM as a judge" research.
- **Effective "Overall vs Expert Support" dual-track design**: By separating evidence gaps from expert intuition, the study quantitatively identifies "insufficient evidence" as the primary driver of disagreement.
- **Shifting from a single verdict to a "Communication Model"** (Figure 2, §5): Systems should actively seek clarification, use guided retrieval, pivot to verifiable versions of unverifiable claims, and produce multi-perspective explanations with uncertainty, rather than a binary verdict.

## Limitations & Future Work
- **Small sample size**: The final agreement analysis covered only 5 claims × 5 experts; while labor-intensive, the statistical power of $\kappa$ values is limited.
- **Retriever scope**: Only the `stella` model was used; no comparison with RM, BM25, or LLM-rerank was performed.
- **Evidence limited to RCTs**: Excludes meta-analyses, cohorts, or case reports, which would introduce new evidence-level synthesis challenges.
- **No prototype implementation**: Section 5.1 discusses evaluation strategies but does not deliver a functional dialogue system.
- **Modeling patient intent**: Identifying latent intentions remains an open problem beyond current dialogue clarification research.
- **Ethics and Privacy**: While RedHOT provided opt-out opportunities, re-publishing Reddit subsets remains a sensitive community issue.

## Related Work & Insights
- **vs RedHOT (Wadhwa et al. 2023)**: RedHOT provided claim annotations; this paper is the first to run the full "claim + evidence + synthesis" pipeline on it.
- **vs healthfc (Vladika et al. 2024) / scifact / covid-fact / healthver**: These use decontextualized claims and closed labels (end-to-end paradigm). This paper argues that such paradigms lack construct validity in-the-wild.
- **vs AmbiFC (Glockner et al. 2024)**: While AmbiFC recognized multiple interpretations, this paper shows such ambiguity is often inherent and unavoidable in medicine.
- **vs Tree-of-Clarifications (Kim et al. 2023)**: Provides the technical foundation for the proposed communication model (using clarification question trees).
- **vs Warren et al. 2025**: Aligns with their call for nuanced, multi-perspective explanations by providing quantitative evidence from expert annotations.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to use expert upper-bound experiments to argue against the construct validity of medical fact-checking.
- Experimental Thoroughness: ⭐⭐⭐ 1,000 instances annotated, but final agreement analysis sample size is small; lacks model implementation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure with exhaustive appendices.
- Value: ⭐⭐⭐⭐⭐ Directly challenges the paradigm of an entire subfield and provides an actionable alternative direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] PropVG: End-to-End Proposal-Driven Visual Grounding with Multi-Granularity Discrimination](../../ICCV2025/social_computing/propvg_end-to-end_proposal-driven_visual_grounding_with_multi-granularity_discri.md)
- [\[ACL 2026\] VeriTaS: The First Dynamic Benchmark for Multimodal Automated Fact-Checking](veritas_the_first_dynamic_benchmark_for_multimodal_automated_fact-checking.md)
- [\[AAAI 2026\] Fact2Fiction: Targeted Poisoning Attack to Agentic Fact-checking System](../../AAAI2026/social_computing/fact2fiction_targeted_poisoning_attack_to_agentic_fact-check.md)
- [\[ICML 2025\] DEFAME: Dynamic Evidence-based FAct-checking with Multimodal Experts](../../ICML2025/social_computing/defame_dynamic_evidence-based_fact-checking_with_multimodal_experts.md)
- [\[ACL 2026\] ClaimDB: A Fact Verification Benchmark over Large Structured Data](claimdb_a_fact_verification_benchmark_over_large_structured_data.md)

</div>

<!-- RELATED:END -->
