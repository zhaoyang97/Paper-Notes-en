---
title: >-
  [Paper Note] Stop Automating Peer Review Without Rigorous Evaluation
description: >-
  [ICML 2026][LLM (Other)][Paper Laundering] This is a position paper: through empirical measurements of real ICLR 2026 reviews and 60 simulated reviews, the authors identify two major failures in current LLM reviewing: the hivemind effect (high convergence) and paper laundering (zero-shot paraphrasing alone can increase scores by 0.45). Consequently, they argue
tags:
  - ICML 2026
  - LLM (Other)
  - Paper Laundering
date: 2026-05-08
content_hash: 0a17c1c3185de347
---
# Stop Automating Peer Review Without Rigorous Evaluation

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2605.03202](https://arxiv.org/abs/2605.03202)  
**Code**: None  
**Area**: LLM / NLP / AI Governance / Peer Review  
**Keywords**: AI Review, Hivemind effect, Paper Laundering, Algorithmic monoculture, Review diversity

## TL;DR
This is a position paper: through empirical measurements of real ICLR 2026 reviews and 60 simulated reviews, the authors identify two major failures in current LLM reviewing: the hivemind effect (high convergence) and paper laundering (zero-shot paraphrasing alone can increase scores by 0.45). Consequently, they argue that "LLMs should not directly generate review comments without rigorous evaluation" and call for the establishment of a "science of review automation."

## Background & Motivation

**Background**: Submissions to top conferences are exploding (NeurIPS / ICLR / ICML increase by thousands annually), and the qualified reviewer pool cannot keep pace. Various conferences have begun introducing LLMs: AAAI 2025 generated LLM reviews in parallel with humans; ICLR 2026 allows reviewers to use LLMs; ICML 2026 introduced an "author opt-in" binary policy; ICLR 2025 deployed a review feedback agent. These policies are highly fragmented and lack consensus.

**Limitations of Prior Work**: Existing research on AI reviewing almost exclusively measures "correlation with human scores" or "propensity for prompt injection attacks." These two dimensions are insufficient: high correlation does not imply harmlessness (errors may be consistently aligned), and prompt injection is explicitly banned by conferences, making it an atypical threat model. The truly challenging issues are "compliant" but low-cost strategic writing behaviors and the "homogenization" of AI reviewing at the collective level.

**Key Challenge**: Human review bias is **distributed** (different reviewers have different bias directions, which partially cancel out after aggregation), whereas LLM review bias is **centralized** (similar training data for the same base models → highly correlated errors). This is what Kleinberg & Raghavan call "algorithmic monoculture": individual decisions seem reasonable, but the aggregate becomes worse, and the collective is vulnerable to the same type of attack.

**Goal**: To refine the "necessary conditions" for review automation from vague "human correlation" to two measurable conditions—(C1) review diversity / non-convergence; (C2) non-gamability / resilience to strategic paraphrasing—and empirically test whether current SOTA LLM reviewers satisfy them.

**Key Insight**: The authors perform a "wild data" analysis using all 75,800 real reviews from ICLR 2026 (already containing AI labels) while conducting controlled experiments on 60 random papers using the AI reviewer agent from Bianchi et al. They construct a complete grid of 4 prompts $\times$ 2 paraphrasing models $\times$ 3 review models = 24 conditions to quantify "hivemind" and "laundering" respectively.

**Core Idea**: Use two embedding similarity metrics, IntraSim and InterSim, to measure the "hivemind effect," and define "paper laundering" via zero-shot LaTeX paraphrasing. By proving that AI reviewing fails both necessary conditions, they support the stance of "establishing a science of review automation before deployment."

## Method

### Overall Architecture
As a position paper, this work does not propose a new model but advocates that "LLM reviewing should not be deployed until it satisfies two measurable necessary conditions—(C1) review diversity / non-convergence and (C2) non-gamability." This claim is substantiated through an empirical protocol of "measurement + attack." The argument rests on three legs: an analysis of 75,800 real ICLR 2026 reviews (AI-labeled via the EditLens classifier, with 21% identified as AI-generated), a simulation of 1,440 synthetic reviews ($60 \times 24$) using an AI reviewer agent, and a comparison where the same 60 papers are zero-shot rewritten into "laundered" versions. The final evidence consists of two statistically significant findings: a pronounced hivemind effect and an average score increase of $+0.45$ ($p<0.0001$) via laundering. All "costs" are concentrated in one-time API calls and offline embedding calculations (approx. \$0.25 per laundering), with no model training involved.

### Key Designs

**1. Quantifying "Review Diversity" as Scalable Scalars via IntraSim / InterSim**

To move the "convergence" argument beyond intuition, the authors quantify it. For a set of review vectors $\mathcal{R}(p)$ for paper $p$ (encoded using `text-embedding-3-small`), IntraSim calculates the average pairwise cosine similarity between reviewers for the same paper: $\mathrm{IntraSim}(p)=\frac{2}{m_p(m_p-1)}\sum_{i<j}\mathrm{sim}(r_i,r_j)$, measuring whether the second review provides new information. InterSim calculates the dual average of $\mathrm{sim}(r,r')$ across different papers ($p\neq q$) and specifically compares the same category of reviewers across different papers to avoid signal pollution from stylistic differences between humans and AI. The key point is that looking at score correlation alone collapses "AI giving high scores" and "AI saying the same thing," whereas pairwise similarity in embedding space exposes the latter—which is the foundation of the multi-reviewer system and the criterion for C1.

**2. Constructing a "Compliant" Gaming Attack via Paper Laundering**

In testing C2, the authors avoid prompt injection tricks, which are explicitly banned and not typical threat models. Instead, they design a failure mode that cannot be banned by policy: feeding the entire LaTeX source code of a paper to a "launderer" LLM, having it rewrite the paper under a zero-shot prompt, and re-compiling it into a PDF for the AI reviewer agent to score. The experiment covers a 24-cell grid (4 prompts $\times$ 2 launderers $\times$ 3 reviewers), including a prompt that commands the launderer to actively jailbreak the review model. Using Wilcoxon signed-rank tests for paired differences, nearly all cells show $p<0.001$ with an average gain of $+0.45$. Furthermore, self-preference bias causes GPT reviews to increase the most when the text is rewritten by GPT. The most pointed aspect is its "compliance": authors can openly admit to using AI for "polishing," which conferences cannot easily prohibit, yet the cost of the attack is as low as \$0.25 per paper.

**3. Cascading Measurement of Monoculture from Reviewing to Writing**

The danger of algorithmic monoculture lies not just in "review convergence" but in "writing shaped by reviews also converging." The authors complete this causal chain with an embedding aggregation experiment. They perform pairwise cosine similarity on the abstract+introduction embeddings of 60 papers (6,903 pairs from multiple samplings), comparing original vs. laundered versions. The average similarity increased from $0.497$ to $0.529$, a relative increase of $+6.5\%$ with Cohen's $d=1.02$ (a large effect). This implies that if laundering is widely adopted, academic writing will converge toward the preferences of AI reviewers, linking the argument from the reviewer level all the way to the author's behavior.

## Key Experimental Results

### Main Results
Hivemind effect in real ICLR 2026 reviews:

| Data Slice | Mean InterSim | $p$-value | Cohen's $d$ |
|---|---|---|---|
| AI-only Reviews | 0.486 | $<10^{-4}$ | 0.29 |
| Reviews with Human Contribution | 0.467 | — | — |
| Weaknesses + Questions only | — | $<10^{-4}$ | 0.35 |

Convergence of AI vs. Human in simulation experiments (60 ICLR papers):

| Setting | IntraSim | InterSim (vs. Human 0.470) |
|---|---|---|
| Human ICLR Reviews | 0.811 | 0.470 |
| GPT-5.1 on Original Papers | 0.882 (+8.7%) | 0.646 (+37.4%, $d=3.55$) |
| GPT-5.1 on Laundered Versions | 0.891 (+9.8%) | 0.657 (+39.8%, $d=3.79$) |
| Claude on Original Papers | — | 0.553 (+17.6%, $d=1.41$) |

Laundering score increase (1–10 scale, average of 24 cells):

| Metric | Value |
|---|---|
| Avg. paired score gain | $+0.45$ |
| Wilcoxon significance | $p<0.001$ (nearly all cells) |
| Cost per paper laundering | ≈ $0.25 |
| Increase in paper similarity | +6.5% ($d=1.02$) |

### Ablation Study

| Slice / Configuration | Key Finding | Description |
|---|---|---|
| Restricting sections to weakness+question | IntraSim $d$ from 1.47 → 1.93 | Convergence is not just boilerplate; it penetrates critical paragraphs. |
| GPT reviewer + GPT launderer | Largest score gain | Replicates self-preference bias (Panickssery 2024). |
| Claude reviewer | Smaller score gain | Cross-family launderer attacks still work but are weaker. |
| AI vs. Human predictive power for acceptance | AUC 0.710 vs. 0.822 (n=8,015) | Human scores better predict the final decision. |
| AI–AI score correlation vs. AI–Human | $r=0.49$ vs. $r=0.15$ | Consistent errors rather than alignment with humans. |

### Key Findings
- AI reviewers heavily reuse generic phrasing: the phrase "if not, can you comment on" appears in 13.3% of GPT reviews; "how does the method handle" appears in 21.7% of Claude reviews; whereas the most common phrases in human reviews occur in $<1%$. This is direct evidence that AI "questions" are largely independent of the paper content.
- Score increases primarily stem from increased use of hedging words ("may", "typically", "suggests") and emphasis words ("strong", "robust", "consistent"), alongside hallucinated "extra interpretations" of results rather than substantive new experiments.
- AI reviewers give an average score of 7.3 (GPT) and 6.1 (Claude), compared to a human average of 4.3, indicating significant systemic inflation.

## Highlights & Insights
- **Redefines "AI Review Quality" into two testable conditions**: The discussion moves from endless debates over correlation to an engineering baseline of diversity + gaming resistance, which researchers can test immediately.
- **Laundering as an elegant "compliant attack"**: Unlike prompt injection, it exposes a governance gap: conference policies can ban "boundary-crossing," but they cannot ban zero-shot polishing. The argument has high transferability.
- **Algorithmic monoculture falsified at two levels**: The reviewer layer (homogenization of reviews) and the author layer (paper convergence toward AI preferences). This "dual-layer monoculture" framework can be applied to any LLM-mediated evaluation scenario, such as recommender systems or automated grading.
- **Appendix H provides the AI review of the paper itself**: A rare "eat your own dog food" moment in a position paper, serving as an ironic demonstration.

## Limitations & Future Work
- The authors acknowledge that necessary conditions are not sufficient. Even if future LLMs are non-convergent and robust to gaming, questions regarding accountability and democratic legitimacy remain.
- Empirical limitations: The 60-paper simulated sample is small; AI labels come from the EditLens classifier (risk of misclassification, though verified in Appendix G.3); whether "quality remains unchanged" during laundering relies primarily on word-level analysis rather than human blind review.
- Future directions: Incorporating "gaming resistance" into benchmarks (e.g., a "laundering robust score") for future AI reviewer systems; investigating failure boundaries when launderers go beyond style (e.g., adding fake experiments).

## Related Work & Insights
- **vs. Bianchi et al. 2025b (AI reviewer agent)**: This paper reuses their agent but shifts the research question from "can they review" to "should they be automated," providing counter-examples.
- **vs. Goldberg et al. 2024 (NeurIPS LLM checklist)**: Also reveals that AI tools can be gamed, but this paper lowers the gaming threshold to zero-shot optimization, making the threat more pervasive.
- **vs. Lin et al. 2025b (targeted adversarial attacks)**: They perform character-level perturbations; this paper requires only natural language rewriting, lowering the attacker threshold by an order of magnitude.
- **vs. Kleinberg & Raghavan 2021 (algorithmic monoculture)**: First empirical application of this theoretical framework to the academic review scenario.

## Rating
- Novelty: ⭐⭐⭐⭐ First to deconstruct "review automation" into two measurable conditions and propose paper laundering as a new threat.
- Experimental Thoroughness: ⭐⭐⭐⭐ Balances external/internal validity with real data + controlled experiments + 24-cell grid, though the 60-paper scale is small.
- Writing Quality: ⭐⭐⭐⭐⭐ A model for position papers: clear causal chain, point-by-point rebuttal of opposing views (§5), and self-referential AI review in the appendix.
- Value: ⭐⭐⭐⭐⭐ Extremely timely topic with direct influence on conference policies and community consensus.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Can AI Be a Good Peer Reviewer? A Survey of Peer Review Process, Evaluation, and the Future](../../ACL2026/llm_nlp/can_ai_be_a_good_peer_reviewer_a_survey_of_peer_review_process_evaluation_and_th.md)
- [\[ICML 2026\] Position: The ML Community Must Build an AI-Augmented Peer-Review Ecosystem](position_the_ml_community_must_build_an_ai-augmented_peer-review_ecosystem.md)
- [\[AAAI 2026\] Position on LLM-Assisted Peer Review: Addressing Reviewer Gap through Mentoring and Feedback](../../AAAI2026/llm_nlp/position_on_llm-assisted_peer_review_addressing_reviewer_gap_through_mentoring_a.md)
- [\[ACL 2025\] ATRIE: Automating Legal Interpretation with LLMs: Retrieval, Generation, and Evaluation](../../ACL2025/llm_nlp/atrie_legal_interpretation.md)
- [\[AAAI 2026\] STEM: Efficient Relative Capability Evaluation of LLMs through Structured Transitive Evaluation Model](../../AAAI2026/llm_nlp/stem_efficient_relative_capability_evaluation_of_llms_through_structured_transit.md)

</div>

<!-- RELATED:END -->
