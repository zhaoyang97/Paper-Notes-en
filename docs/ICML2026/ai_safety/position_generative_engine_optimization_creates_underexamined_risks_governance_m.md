---
title: >-
  [Paper Note] Position: Generative Engine Optimization Creates Underexamined Risks, Governance Must Target Concentration, Disclosure, and Academic Blind Spots
description: >-
  [ICML 2026][AI Safety][Paper Note] This position paper argues that as users transition from "viewing ranked lists" to "viewing LLM-synthesized answers," Search Engine Optimization (SEO) evolves into **Generative Engine Optimization (GEO)**, exerting influence within the evidence pool and generation stages of RAG-based answer engines. The authors formali
tags:
  - ICML 2026
  - AI Safety
date: 2026-05-08
content_hash: aa4212f182fc9e96
---
# Position: Generative Engine Optimization Creates Underexamined Risks, Governance Must Target Concentration, Disclosure, and Academic Blind Spots

**Conference**: ICML2026  
**arXiv**: [2606.12439](https://arxiv.org/abs/2606.12439)  
**Code**: None (Position Paper)  
**Area**: AI Safety and Governance · Generative Engine Optimization (GEO) · LLM Answer Engines · Algorithmic Accountability  
**Keywords**: Generative Engine Optimization, Answer-level Governance, Implicit Commercial Influence, Black-box Auditing, RAG Manipulation  

## TL;DR
This position paper argues that as users transition from "viewing ranked lists" to "viewing LLM-synthesized answers," Search Engine Optimization (SEO) evolves into **Generative Engine Optimization (GEO)**, exerting influence within the evidence pool and generation stages of RAG-based answer engines. The authors formalize a universal GEO pipeline, identify three overlooked risks (concentration of influence, implicit commercial impact, and academic-industrial blind spots), and call for "answer-level governance": enhanced contestability, high-precision disclosure, black-box auditing of substantive impacts, and exposure persistence metrics aligned with deployment.

## Background & Motivation

**Background**: LLM answer engines like ChatGPT and Gemini are becoming the default portals for information retrieval and shopping decisions (surveys show 60% of US adults use AI for information gathering at least occasionally, and 39% of global shoppers use AI for product discovery). These systems follow a **retrieve-then-generate** workflow, which is essentially RAG: retrievers pull external text, and LLMs generate answers based on the retrieved passages.

**Limitations of Prior Work**: Classic SEO manipulates "ranked lists + labeled sponsored slots," where users can see ads and rankings. However, GEO manipulates the **evidence pool and answer generation process**: it determines which products eventually appear in the synthesized answer, a selection process that is completely opaque to the user. GEO is already an active commercial market (companies like AirOps and ProFound have raised millions), and real-world incidents have emerged: Microsoft reported prompt injections hidden in "Summarize with AI" links to induce recommendations, and the OECD AI Incident Monitor recorded a 2026 GEO-style poisoning event in China where LLMs were induced to recommend low-quality products.

**Key Challenge**: Existing governance and evaluation frameworks are designed for the SEO era, assuming influence occurs at the level of "visible rankings + ad labeling." However, GEO influence is embedded **within opaque LLM answer generation pipelines**, making it neither visible nor easily auditable. Current frameworks fail to target these internal mechanisms.

**Goal**: (1) Formalize a universal GEO pipeline to pinpoint where optimization acts; (2) Compare academic vs. industrial GEO practices to identify three types of neglected risks; (3) Propose corresponding answer-level governance and measurement methods.

**Key Insight**: The authors follow the "SEO → GEO transition" path, decomposing GEO into two optimizable objectives: "retrieval accessibility" and "ranking influence," then using this perspective to highlight gaps between academic and industrial practices.

**Core Idea**: Risks introduced by GEO are **structural and occur at the answer level**. They must be addressed via answer-level governance (contestability + high-precision disclosure + black-box auditing + deployment-aligned metrics) rather than relying on SEO-era ad labeling or offline benchmarks.

## Method

> This is a position paper; "Method" refers to the analytical framework: a formalized GEO pipeline + arguments for three risk categories + governance proposals.

### Overall Architecture

The authors formalize GEO into a three-block framework, characterizing optimization as two mathematical objectives. The blocks are: (i) **LLMs Block**—transforming user queries into generated recommendations; (ii) **Search Flow Block**—extracting candidates via scalable matching (e.g., keyword retrieval) and ranking them using relevance metrics (e.g., embedding cosine similarity) to select the top-$k$ as the LLM context; (iii) **GEO Block**—distributing optimized content across platforms to be indexed by search engines and influence LLM output (optimizing owned sites or mass-posting on high-authority platforms).

GEO optimization is modeled as a joint optimization of "retrieval accessibility + ranking influence" by injecting two types of messages: **retrieval-boosting messages** $b\sim\mathcal{B}$ (increasing retrieval probability) and **ranking-shift messages** $c\sim\mathcal{C}$ (altering answer-level rankings once in context).

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["User query q ~ Π(·|t)<br/>Target topic t"] --> B["Search Flow Block<br/>Candidate Retrieval → Relevance Ranking → Top-k Extraction"]
    G["GEO Block<br/>Inject b (Boosting) + c (Shifting)<br/>Cross-platform Distribution"] -->|Mixed into candidate set 𝒟∪{b,c}| B
    B -->|Context C(q)⊆Top-k| C["LLMs Block<br/>Generate Synthesized Answer based on C(q)"]
    C --> D["Answer-level Exposure U(q,t;C(q))<br/>Is target topic mentioned/cited?"]
    D -.->|Industrial Online Feedback Loop: Continuous Probing & Re-optimization| G
```

### Key Designs

**1. Formalized GEO Pipeline: Pinpointing Where Optimization Acts**

The authors model GEO as optimizing two types of messages. **Retrieval boosting $b_i$** maximizes semantic similarity with potential queries to improve accessibility:

$$\max_{b_i}\; J_{\text{boost}}(b_i)=\mathbb{E}_{q\sim\Pi(t)}\big[\text{Sim}(q,b_i)\big]\quad \text{s.t.}\quad \ell(b_i)\le L$$

where $\text{Sim}$ can be BM25 or cosine similarity, and $\ell(b_i)\le L$ limits message length. **Ranking shift $c_i$** (conditioned on $b_i$) alters how the LLM describes and ranks the target topic once it enters the top-$k$ context $C(q)\subseteq \text{Top-}k_R(q;\mathcal{D}\cup\{b_i,c_i\})$. The objective is $J_{\text{shift}}(c_i\mid b_i)=\mathbb{E}_{q\sim\Pi(t)}[U(q,t;C(q))]$, where utility $U$ measures the change in ranking/exposure of the target topic.

**2. Three Neglected Risks: Concentration, Implicit Influence, and Blind Spots**

**Risk 1: Concentration of Influence** stems from (a) **Loss of contestability**—users cannot see why $C(q)$ was selected or what was excluded; (b) **Systemic sensitivity**—$C(q)$ is determined by a hard top-$k$ truncation. One injected message crossing the top-$k$ boundary causes a **jump** in $U$. Small sensitivity tests showed that for Gemini series, changing wording in query pairs frequently results in entirely different set of cited domains.

**Risk 2: Implicit Commercial Influence**: While the FTC requires paid ads to be labeled, GEO embeds $(b_i,c_i)$ into reviews and forums. **Persuasion occurs through the model's own reasoning**, erasing the boundary between neutral advice and marketing. This creates "adverse selection" where covert players outperform transparent ones.

**Risk 3: Academic-Industrial Blind Spots**: See the section below.

**3. Comparison: Academic vs. Industrial GEO**

Academic efforts often **assume** $(b_i,c_i)$ is already in the candidate set, focusing only on $J_{\text{shift}}(c_i)$ using offline static corpora and ranking metrics like Recall@k. Industrial GEO performs joint optimization of $(b_i,c_i)$ on the **dynamic open web**, tracking real-time citations and visibility as feedback.

| Dimension | Academic GEO | Industrial GEO |
|------|---------|---------|
| Assumption | $(b_i,c_i)$ in candidate set, optimize $c_i$ | Accessibility not guaranteed, joint $(b_i,c_i)$ |
| Optimization Surface | Owned websites | Owned sites + High-authority external platforms |
| Evaluation Environment | Offline static corpora, synthetic catalogs | Online real-time systems, dynamic crawling |
| Metrics | Recall@k, nDCG@k, rank position | Answer visibility, citation frequency, persistence |

**4. Answer-level Governance and Black-box Auditing**

The authors suggest actions based on the Mökander framework. **Reducing Influence Concentration**: Increase contestability (e.g., "Why this answer" panels), disclose retrieval/ranking structures, and use black-box auditing to estimate $\widehat U$ and $\widehat J_{\text{shift}}$. **Commercial Disclosure**: Trigger labels based on low-ambiguity signals (affiliate parameters, `rel="sponsored"`) and calibrate for high precision. **Closing Blind Spots**: Shift academic research toward longitudinal, cross-platform measurements on deployed systems, adding exposure shift and "persistence of appearance" to shared benchmarks.

## Key Experimental Results

> As a position paper, it lacks standard experiments but includes a small-scale sensitivity test and analytical evidence.

### Key Findings
- **Sensitivity tests confirm "systemic sensitivity"**: In 30 query pairs across 7 deployed models, minor wording changes altered the set of cited domains. The Gemini-3-flash model often cited completely different domains for semantically equivalent queries.
- **Offline benchmarks systematically underestimate impact**: Modest improvements in Recall@k/nDCG@k can translate to significant jumps in the probability of being cited within the final synthesized answer.
- **Benign and Malicious GEO share the same goal**: Both optimize $J_{\text{shift}}$. The difference lies in **constraints**: benign players maintain verifiability (legitimate citations), while malicious players relax these constraints, using fabricated statistics or prompt injections. Governance should target the absence of veracity constraints rather than the optimization itself.

## Highlights & Insights
- **Levers of Manipulation**: Deconstructing GEO into "retrieval boosting" and "ranking shift" provides concrete levers for auditing and formalized discussion.
- **"Answer-level" Perspective**: Shifting focus from intermediate ranking lists to the final synthesized answer identifies why SEO-era frameworks fail.
- **Actionable Governance**: Proposes measures using existing infrastructure (FTC, EU AI Act, NIST AI 600-1) and estimates black-box auditing costs ($50–300).
- **Nuanced Integrity**: Defining malicious GEO as "optimizing without veracity constraints" avoids a blanket ban on optimization while focusing on user harm.

## Limitations & Future Work
- **Small Test Scale**: The sensitivity test (30 query pairs) is illustrative rather than a quantitative measure of widespread sensitivity across all engines.
- **Lack of Benchmarking Tools**: While calling for auditing and metrics, the paper does not release a ready-to-use audit tool or standardized benchmark.
- **Label Calibration**: High-precision disclosure is difficult to maintain as adversarial tactics evolve.
- **Reliance on Cooperation**: Many governance proposals depend on platforms voluntarily exposing retrieval pools or penalizing covert tactics.

## Related Work & Insights
- **vs. Classic SEO**: SEO focuses on document visibility in ranking lists; GEO focuses on evidence and framing within answer generation.
- **vs. Single-point Attacks**: Unlike works studying isolated mechanisms (e.g., GCG, TAP), this paper provides a **unified formalization + governance roadmap**.
- **vs. RAG Defense**: The authors argue that even if RAG defenses improve correctness, they do not make commercial relationships visible or selection processes contestable.
- **vs. Algorithmic Accountability**: The paper applies abstract theories of "the black box society" to the specific technical mechanics of $C(q)$ and top-$k$ truncation.

## Rating
- Novelty: ⭐⭐⭐⭐ Formalizing the SEO→GEO shift and answer-level risks is a fresh perspective built on existing attack research.
- Experimental Thoroughness: ⭐⭐⭐ Primarily analytical; sensitivity tests are small-scale and lack a reproducible audit benchmark.
- Writing Quality: ⭐⭐⭐⭐⭐ Tight logical chain from formalization to risk and governance, including honest self-refutation of alternative views.
- Value: ⭐⭐⭐⭐⭐ Answer-level governance is an urgent issue as LLM engines become search portals.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">
- **Aggarwal et al. (2024)**: GEO: Generative Engine Optimization.
- **Kumar & Lakkaraju (2024)**: PoisonedRAG: Knowledge Poisoning Attacks on Retrieval-Augmented Generation.
- **Mökander et al. (2023)**: Auditing Large Language Models: A Three-Layered Approach.
</div>
<!-- RELATED:END -->

## Related Papers

- [\[ICML 2026\] Position: AI Researchers Must Help Lead Arms Control to Mitigate Military AI Risks](ai_researchers_must_help_lead_arms_control_to_mitigate_military_ai_risks.md)
- [\[ICML 2026\] Position: Machine Learning for Heart Transplant Allocation Policy Optimization Should Account for Incentives](position_machine_learning_for_heart_transplant_allocation_policy_optimization_sh.md)
- [\[AAAI 2026\] Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach](../../AAAI2026/ai_safety/rethinking_target_label_conditioning_in_adversarial_attacks_a_2d_tensor-guided_g.md)
- [\[ICML 2026\] Alignment Risks from Capability-Seeking RL Training](alignment_risks_from_capability-seeking_rl_training.md)
- [\[ICML 2026\] Position: Uncertainty Quantification in LLMs is Just Unsupervised Clustering](position_uncertainty_quantification_in_llms_is_just_unsupervised_clustering.md)

</div>

<!-- RELATED:END -->
