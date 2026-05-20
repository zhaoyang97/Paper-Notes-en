---
title: >-
  [Paper Note] Stop DDoS Attacking the Research Community with AI-Generated Survey Papers
description: >-
  [NeurIPS 2025][LLM Safety][AI-generated surveys] This position paper analogizes the proliferation of AI-generated survey papers to a "Distributed Denial-of-Service (DDoS) attack" on the academic community. Through system…
tags:
  - "NeurIPS 2025"
  - "LLM Safety"
  - "AI-generated surveys"
  - "survey paper DDoS attack"
  - "academic integrity"
  - "dynamic live surveys"
  - "paper quality detection"
  - "research culture"
date: 2026-05-08
content_hash: 78b2f09db6e85f63
---

# Stop DDoS Attacking the Research Community with AI-Generated Survey Papers

**Conference**: NeurIPS 2025
**arXiv**: [2510.09686](https://arxiv.org/abs/2510.09686)  
**Authors**: Jianghao Lin, Rong Shan, Jiachen Zhu, Yunjia Xi, Yong Yu, Weinan Zhang (Shanghai Jiao Tong University)
**Code**: None  
**Area**: LLM Safety
**Keywords**: AI-generated surveys, survey paper DDoS attack, academic integrity, dynamic live surveys, paper quality detection, research culture

## TL;DR

This position paper analogizes the proliferation of AI-generated survey papers to a "Distributed Denial-of-Service (DDoS) attack" on the academic community. Through systematic quantitative analysis of 10,063 CS survey papers on arXiv from 2020 to 2024, the paper documents synchronized post-ChatGPT surges in survey volume, AI-generation scores, and anomalous author counts. It diagnoses four major quality deficiencies in AI-generated surveys (disorganized structure, unoriginal taxonomies, inaccurate citations, and highly redundant content), analyzes cultural repercussions for the researcher–reviewer–editor triad, and proposes a comprehensive response framework encompassing transparency requirements, rigorous review standards, redundancy restrictions, AI-detection assistance, and a "Dynamic Live Survey" platform.

## Background & Motivation

Survey papers occupy a unique and critical role in academic research: by systematically organizing the literature, distilling key trends, and pointing toward frontier directions, they serve as indispensable knowledge landmarks for both newcomers and seasoned experts. A high-quality survey is far more than a compiled list of papers—it must propose an original taxonomy, offer deep critical and comparative analysis of state-of-the-art methods, accurately trace the developmental trajectory of a field, and identify unresolved key problems and future directions. This high-level academic synthesis has traditionally demanded substantial time and effort from domain experts, making survey writing a labor-intensive scholarly activity.

The rise of large language models (LLMs) has fundamentally disrupted this landscape. Generative AI tools such as ChatGPT can produce seemingly well-structured, fluent literature reviews within minutes, transforming survey writing from a high-threshold expert activity into a low-barrier, high-throughput batch process. This democratization of capability is not inherently problematic, but its unconstrained misuse carries severe consequences. The authors creatively liken this phenomenon to a "Distributed Denial-of-Service (DDoS) attack" in computer security: just as a DDoS attack overwhelms a target server with massive volumes of spurious traffic until it can no longer respond to legitimate requests, the flood of low-quality AI-generated surveys on preprint platforms such as arXiv drowns out genuinely valuable scholarly contributions, making it increasingly difficult for researchers to identify reliable surveys amid "literature noise," and ultimately eroding the trust foundation of the entire academic community.

The urgency of this problem manifests along multiple dimensions. First, genuine academic progress may be obscured by algorithmically generated "literature duplicates," preventing innovative contributions from receiving appropriate attention. Second, the harm is especially severe for interdisciplinary researchers and early-career scholars, who rely on surveys as entry points into new fields but now face the additional burden of judging "which survey is trustworthy" among a sea of variable-quality works. More fundamentally, errors and biases introduced by automatically generated text may propagate into subsequent research without human review, effectively "seeding false premises." In the most extreme scenario, the proliferation of low-quality surveys may cause "literature contamination": poor surveys mutually citing one another form self-reinforcing citation loops that distort the citation landscape, causing truly influential foundational works to be overlooked.

Based on this analysis, the paper takes a clear stance: **the upload of large volumes of AI-generated survey papers to the academic community must stop**, and this threat must be addressed by enforcing appropriate norms for AI use in survey writing, restoring rigorous human oversight, and establishing clear standards for AI-assisted surveys.

## Method

### Overall Architecture

As a position paper, the methodological framework differs from the conventional "propose model → experimental validation" paradigm of technical papers. Instead, it adopts a five-tier progressive structure—**quantitative evidence → quality diagnosis → impact analysis → policy recommendations → forward-looking vision**—to systematically articulate its position.

The **first tier** is quantitative trend analysis (Section 2), which establishes a measurable evidentiary foundation for the AI survey proliferation by conducting large-scale statistical analysis of all CS-category survey papers on arXiv from 2020 to 2024. Specifically, the authors collected 10,063 papers whose titles contained the keywords "survey," "review," "overview," or "taxonomy," and analyzed them along three complementary dimensions: (1) the absolute growth trend in annual survey paper counts; (2) the AI-generation probability score for each paper estimated using an open-source AI content detector (desklib/ai-text-detector-v1.01), to measure changes in the degree of AI involvement in writing; and (3) the detection of "anomalous authors," defined as authors who submitted three or more surveys within a single month with fewer than two collaborators—a submission pattern extremely rare in traditional academic settings and strongly suggestive of AI batch generation. To enhance the robustness of the conclusions, Appendix analyses cross-validate using two AI detectors from top conferences, DeTeCtive and MAGE, along with auxiliary metrics such as citation overlap and semantic similarity.

The **second tier** is quality issue diagnosis (Section 3.1), systematically examining AI survey deficiencies across four dimensions: structure, taxonomy, citations, and redundancy. The **third tier** proposes detection metrics (Section 3.2), providing practical heuristics for identifying AI-generated surveys. The **fourth tier** is cultural impact analysis (Section 4), discussing the deeper effects of AI survey proliferation from the perspectives of three stakeholders: researchers, reviewers, and editors. The **fifth tier** presents policy recommendations and a forward-looking vision (Sections 5–6), proposing solutions at both the institutional and technical platform levels.

### Key Designs

#### Four-Dimensional Quality Deficiency Diagnostic Framework

One of the paper's core contributions is establishing a systematic quality diagnostic framework for AI-generated surveys, decomposing the problem into four interrelated yet distinct dimensions.

**Dimension 1: Structural Deficiencies.** Research from SurveyForge demonstrates that AI-generated surveys exhibit explicit structural shortcomings: disorganized outlines that fail to reflect the conceptual structure of the field, reading more like unorganized lists of topics or papers rather than a coherent narrative; and key sections (such as background introduction and thematic taxonomy) that are either superficial or entirely absent. By contrast, human-authored surveys typically define precise subcategories and provide transitions that form a coherent cognitive architecture.

**Dimension 2: Lack of Taxonomic Originality.** Empirical analysis reveals that many suspect surveys simply imitate existing taxonomies (sometimes drawn directly from Wikipedia entries) without proposing any new conceptual perspective. For example, multiple AI-written Vision Transformer (ViT) surveys employ nearly identical chapter structures—"backbone architectures" and "classification/detection applications"—exhibiting high mutual similarity. This templated characteristic suggests that LLMs rely on the same well-known papers or earlier surveys as references. A genuinely valuable human-authored survey, by contrast, might organize the literature from an entirely fresh perspective, such as categorizing ViTs by efficiency strategy.

**Dimension 3: Inaccurate Citations and Content.** This is the most prominent quality issue. AI surveys frequently exhibit citation anomalies: omitting genuinely relevant and influential works while over-citing less relevant or obscure papers, suggesting that citation lists are assembled via keyword matching rather than expert judgment. In some cases, citations are entirely fabricated (LLM hallucinations); volunteer groups such as Academ-AI have identified large numbers of preprints containing references that cannot be found or are inconsistent with the surrounding context.

**Dimension 4: Redundancy and Low Marginal Value.** Significant content overlap exists among different AI-generated surveys, with nearly identical phrasing appearing frequently. This points to a deeper text reuse problem: when multiple authors prompt an LLM to "write a literature review on X," the model typically produces very similar responses. Research shows that certain LLM writing patterns have surged sharply in academic papers, indicating that many papers now share the same stylistic fingerprints. The result is that the $N$-th survey on a popular topic has virtually zero marginal academic value, yet still adds to the noise that researchers must filter.

#### Three-Dimensional Heuristic Detection Metrics

To elevate quality diagnosis from qualitative judgment to actionable detection tools, the paper proposes three complementary categories of heuristic detection metrics.

**GPT Phrase Detection:** The most direct method is to scan papers for signature phrases that reveal AI involvement, such as "as an AI language model," "my knowledge cutoff," and "as of September 2021." These phrases clearly indicate that the author has not appropriately edited the LLM-generated text. The authors wrote scripts to scan arXiv CS survey papers and indeed identified multiple matching cases.

**Citation Overlap Analysis:** Based on the hypothesis that LLMs may consistently cite the same set of well-known papers on a given topic, the authors analyzed the citation lists of ten recent surveys on the same ML topic. Any two surveys shared, on average, approximately 60–70% of the same citations—far higher than what would be expected from independent researchers conducting their own literature searches, strongly implying shared dependence on common AI sources. Further analyses in the Appendix show that pre-2022 citation overlap rates were below 40% (Jaccard index < 0.3), while post-2022 rates exceeded 60% (Jaccard index > 0.5), compared to a random baseline of under 1% for any two arbitrary surveys.

**Length and Repetition Pattern Analysis:** Simple language models are used to measure the word-distribution entropy of suspect papers against known human-authored survey papers. Suspect papers typically exhibit lower lexical diversity (higher frequency of repeated common phrases). Qualitative observations also reveal that multiple papers have consecutive paragraphs beginning with exactly the same transitional words—such as "Furthermore"—a characteristic signature of GPT writing.

### Policy Recommendations and Dynamic Live Surveys

The paper's "solutions" layer comprises two components: near-term feasible policy recommendations and a longer-term forward-looking technical platform.

**Six Policy Recommendations:** (1) **Author declarations and transparency**—require authors to explicitly disclose in the methods section or footnotes how and to what extent AI was used in the writing process; LLMs should not be listed as co-authors. (2) **Stricter review standards for surveys**—assign at least one senior reviewer or area chair specifically to evaluate the depth and value of a survey; review forms should include customized questions such as "Does this paper introduce new insights or a meaningful taxonomy?" (3) **Redundancy submission restrictions**—conferences and journals should coordinate to prevent "survey tracks" from becoming low-barrier publication channels; when a topic already has high-quality surveys, stricter scrutiny should be applied to subsequent surveys lacking differentiation. (4) **AI-detection-assisted review**—use AI content detection as an auxiliary factor in evaluating submissions (high scores trigger deeper scrutiny rather than automatic rejection); reviewers spot-check citation accuracy. (5) **Incentivize high-quality surveys**—create dedicated publication venues for surveys (e.g., a "Journal of ML Reviews and Syntheses") and establish "Best Survey Paper Awards." (6) **Education and ethical guidance**—conferences and universities should educate early-career researchers on the appropriate use of LLMs, treating AI-generated text as third-party content.

**Dynamic Live Surveys Vision:** This is the paper's most forward-looking proposal, aimed at fundamentally transcending the limitations of traditional static one-time surveys. The core concept is to build an open online knowledge base that achieves continuous evolution through the seamless integration of AI-driven content ingestion and domain expert curation. The framework comprises four key features: (1) **Real-time updates**—automated agents scan multiple sources daily (arXiv, conference proceedings, benchmark leaderboards), with new algorithms and datasets appearing on the platform within hours of release. (2) **Human–machine curation loop**—domain experts guide AI agents' focus through prompt refinement, validate or restructure taxonomy nodes, and reconcile conflicting interpretations, while AI agents handle routine ingestion, formatting, and preliminary summarization. (3) **Version control and branching**—drawing on software development practices, contributors can explore alternative taxonomies, methodological debates, or experimental structures, which are merged into the main branch only after rigorous review and voting. (4) **Incentive alignment**—contributors are recognized through ORCID-linked authorship credits, digital badges, co-authorship on archival snapshots, or formal citations. The platform provides linear narrative views, hierarchical outlines, and interactive citation graphs, and periodically generates archival snapshots to provide citable records.

## Key Experimental Results

### Main Results: Quantitative Analysis of arXiv Survey Paper Trends

The paper systematically analyzes 10,063 CS-category survey papers on arXiv from 2020 to 2024, establishing quantitative evidence for the AI survey proliferation across three dimensions.

| Dimension | 2020 | 2021 | 2022 | 2023 | 2024 | Key Trend |
|-----------|------|------|------|------|------|-----------|
| Survey paper count | Lower baseline | Steady growth | **Inflection point, acceleration** | Continued acceleration | Explosive growth | ChatGPT-driven turning point appears in 2022–2023 |
| Average AI-generation score | Low | Low | Begins rising | **Marked increase** | Sustained high level | Papers with high AI content scores doubled from 3.6% to 6.2% |
| Anomalous author count | Few | Few | Begins increasing | **Large increase** | Continues rising | Acceleration inflection also appears in 2022 |

The Appendix further reports cross-validation results using two independent AI detectors, DeTeCtive and MAGE:

| AI Detection Method | 2020→2021 Growth | 2021→2022 Growth | 2022→2023 Growth | 2023→2024 Growth |
|---------------------|-----------------|-----------------|-----------------|-----------------|
| DeTeCtive | 23.37% | 10.60% | **30.81%** | **42.10%** |
| MAGE | 15.86% | 18.60% | **70.58%** | **53.00%** |

All three independent detectors show a marked acceleration in AI-generation scores after 2022, powerfully corroborating the core finding of a "post-2022 surge." Semantic similarity analysis further substantiates this trend: the semantic similarity among surveys on the same topic surged from 0.6033 in 2022 to 0.8367 in 2023, subsequently stabilizing at a high level of 0.7986 in 2024.

### Ablation Study and Auxiliary Analyses

**Citation Overlap Analysis:**

| Analysis Dimension | Pre-2022 | Post-2022 | Random Baseline |
|--------------------|----------|-----------|-----------------|
| Citation overlap percentage | <40% | **>60%** | <1% |
| Jaccard similarity index | <0.3 | **>0.5** | Very low |
| Average shared citation proportion across 10 same-topic surveys | — | **60–70%** | — |

These results indicate that post-2022 survey papers exhibit a high degree of convergence in citation selection, far exceeding what would be expected from independent researchers conducting personal literature searches, strongly implying dependence on shared AI sources.

**Typical Case:** The paper specifically highlights the case of the Model Context Protocol (MCP) topic—more than five survey preprints appeared within approximately one month. Despite MCP being a recently emerged hot topic, the volume of surveys released in such a short time is clearly redundant and likely to cause confusion among researchers while damaging the community.

**Researcher Survey Data:** A survey of 1,600 researchers by Van Noorden and Perkel (2023) found that while many respondents reported having tried using ChatGPT for writing, large numbers simultaneously expressed skepticism about the accuracy and completeness of AI-generated academic work—implying that even high-quality surveys may suffer credibility discounts through confusion with AI-generated ones.

**Industry Corroboration:** A large-scale study by AI content detection company Originality.ai reports that the number of papers on arXiv likely assisted by AI has increased by 72% since ChatGPT became available. The proportion of papers with high AI content scores doubled from approximately 3.6% at the end of 2022 to approximately 6.2% at the end of 2023. Kobak et al. (2024), after analyzing the abstracts of millions of scientific papers, conclude that by 2024, more than 10% of scientific abstracts had been processed by LLMs.

### Key Findings

1. **2022 as the watershed year:** Three independent metrics—survey volume, AI-generation scores, and anomalous author counts—all show a synchronized inflection point in 2022, temporally coinciding with the release of ChatGPT and other advanced LLMs, constituting strong circumstantial evidence for a causal relationship.
2. **Triple-detector cross-validation:** DeTeCtive, MAGE, and the original detector—three independent methods—all confirm the post-2022 acceleration in AI-generated content, ruling out the possibility of single-detector bias.
3. **Anomalous citation convergence:** Post-2022 citation overlap rates surged from <40% to >60%, alongside a simultaneous jump in semantic similarity (from 0.60 to 0.84), indicating that AI surveys are not only proliferating in number but are also highly homogeneous in content.
4. **Self-reinforcing risk of "literature contamination":** Low-quality surveys may accumulate citations by virtue of being easy to find and reference, subsequently citing one another in closed loops that distort the citation landscape—a form of "literature poisoning" analogous to data poisoning.
5. **Overloading the review system:** Reviewers must spend additional time verifying the citation accuracy of suspect surveys—a workload that should not exist and that crowds out the effort needed to evaluate substantive contributions.

## Highlights & Insights

1. **The "survey DDoS attack" metaphor is both vivid and profound:** Analogizing the AI survey proliferation to a DDoS attack in network security is not only visually striking but accurately captures the systemic nature of the problem—the issue lies not in any single paper being of poor quality, but in the scale effect of a massive influx that paralyzes the academic community's "service capacity" (attention, review resources, and trust foundations).

2. **Multidimensional quantitative evidence constructs a compelling argumentative chain:** The statistical analysis of 10,063 papers, cross-validation with three independent AI detectors, and multilayered metrics including citation overlap and semantic similarity mutually corroborate one another, rendering the core claim of a "post-2022 AI survey surge" nearly incontrovertible.

3. **The three-stakeholder cultural impact analysis reaches the deeper layers of the problem:** From the researcher's anxiety over "literature clutter" and the reviewer's fatigue from evaluating "hollow language," to the editor's ethical dilemma over "authorship," the paper addresses not merely the surface-level quantity problem but also the erosive effects of AI survey proliferation on academic culture and the foundations of trust.

4. **The "Dynamic Live Survey" vision carries paradigm-shifting potential:** Proposing a transition from one-time static surveys to community-maintained, version-controlled, AI–human collaborative living documents is not merely a patch on the current problem but a fundamental reimagination of how academic knowledge is organized—analogous to moving from waterfall development to continuous integration/continuous deployment.

5. **Forward-looking problem awareness:** The paper issues a systematic warning—backed by quantitative evidence—before the proliferation of AI-generated surveys has become a mainstream consensus. The "literature contamination" self-reinforcement risk it identifies—whereby low-quality surveys mutually cite one another to form closed citation loops—is particularly prescient.

## Limitations & Future Work

1. **The "anomalous author" definition is overly coarse:** Defining "submitting 3+ surveys within a month with fewer than 2 collaborators" as anomalous does not adequately account for legitimate scenarios in large research groups (e.g., certain institutional patterns in China) where an advisor may simultaneously participate in multiple surveys, nor does it distinguish whether the surveys span different topic areas.

2. **Inherent limitations of AI detection tools:** Open-source AI detectors exhibit substantial false positive and false negative rates. The desklib detector used in this paper has not been rigorously calibrated, and the statistics derived from it may carry systematic biases. Although the Appendix uses DeTeCtive and MAGE for cross-validation, convergent trends across the three tools do not entirely rule out shared methodological bias.

3. **Coverage limited to the CS domain:** The analysis is restricted to arXiv's CS category, yet the proliferation of AI-generated surveys is equally or more severe in biomedical (bioRxiv/medRxiv) and physics fields; the cross-disciplinary generalizability of the conclusions warrants caution.

4. **The "Dynamic Live Survey" vision lacks an operational roadmap:** While the proposal is conceptually forward-looking, it faces practical challenges including incentive misalignment (high maintenance costs with low academic reward), governance complexity (how community disagreements are resolved), and technical platform construction (an integrated system combining version control, permission management, and automated ingestion)—challenges to which the paper provides almost no concrete solutions.

5. **Insufficient balanced discussion of the positive aspects of AI-assisted writing:** The paper's stance leans toward restrictive measures, yet the "capability augmentation" value of AI-assisted writing for non-native English-speaking researchers and researchers at resource-constrained institutions deserves acknowledgment. How to strike a balance between restricting misuse and preserving empowerment requires a more nuanced discussion.

6. **Lack of differentiated analysis of AI use in surveys versus research papers:** The paper focuses on surveys, but the ethical boundaries of AI assistance may differ fundamentally between research papers and surveys—the core value of a survey lies in "synthetic judgment," while that of a research paper lies in "original discovery." Such a differentiated analysis could strengthen the paper's argument.

## Related Work & Insights

In the area of **automated survey generation systems**, AutoSurvey (Wang et al., 2024) demonstrates the technical feasibility of LLM-based automatic survey writing, indirectly confirming that the technical foundation for low-barrier batch survey generation is already mature. SurveyForge (Yan et al., 2025) systematically analyzes AI survey deficiencies from the perspectives of structural heuristics, memory-driven generation, and multidimensional evaluation, providing direct empirical support for the quality diagnosis in this paper. SurveyAgent (Wang et al., 2024) proposes interactive personalized survey generation, seeking a balance between automation and user control.

In the area of **AI-generated text detection**, Kobak et al. (2024) provide indirect but compelling linguistic-level evidence of AI's permeation of academic writing by analyzing the phenomenon of "excess words" (such as sharp increases in LLM-preferred terms like "delve" and "intricate") in the abstracts of millions of scientific papers. DeTeCtive (Guo et al., NeurIPS 2024) detects AI text through multi-layer contrastive learning; MAGE (Li et al., 2023) targets AI text detection in "in-the-wild" scenarios; both are used as cross-validation tools in the Appendix.

In discussions of **academic integrity and AI ethics policy**, Science editor Thorp's (2023) editorial "ChatGPT is fun, but not an author" explicitly denies AI authorship status; Bockting et al. (2023), writing in Nature, emphasize that scientists must supervise AI use in their "Living guidelines for generative AI"; Van Noorden and Perkel's (2023) large-scale Nature survey reveals researchers' ambivalent attitudes toward AI-assisted writing; and Haider et al. (2024) warn of the risk of AI-generated papers infiltrating Google Scholar.

**Directions for inspiration:** (1) Automated survey quality assessment tools—formalizing the paper's diagnostic dimensions (structural completeness, taxonomic originality, citation accuracy, content distinctiveness) into computable metrics; (2) AI content tiered labeling systems for academic platforms—analogous to food safety ratings, annotating survey papers with the degree of AI involvement and quality grade; (3) Technical platform prototypes for dynamic live surveys—open-source platform construction integrating automatic literature ingestion, version control, and expert curation.

## Rating

⭐⭐⭐⭐ (4/5)

- **Impact** ⭐⭐⭐⭐⭐: Strikes at the most sensitive pain point in the academic community; the "survey DDoS attack" concept is highly transmissible and capable of sparking broad discussion and actual policy-level change.
- **Argumentative Quality** ⭐⭐⭐⭐: The quantitative foundation of 10,063 papers is solid; cross-validation with three independent detectors enhances persuasiveness; the multi-dimensional argumentative structure (quantitative trends + quality diagnosis + cultural impact) is logically rigorous.
- **Feasibility of Solutions** ⭐⭐⭐: Among the six policy recommendations, transparency requirements and elevated review standards are relatively feasible; however, redundancy restrictions and AI-detection assistance face practical implementation difficulties; the "Dynamic Live Survey" vision is impressive but its path to realization is vague.
- **Forward-Looking Nature** ⭐⭐⭐⭐⭐: Issues a systematic warning before AI survey proliferation becomes a mainstream topic; the dynamic live survey concept offers inspiring implications for the future direction of academic publishing.
- **Academic Rigor** ⭐⭐⭐: As a position paper, the quantitative analysis sections are well-executed, but the "anomalous author" definition and the calibration of AI detection tools leave room for improvement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text](adversarial_paraphrasing_a_universal_attack_for_humanizing_ai-generated_text.md)
- [\[NeurIPS 2025\] Collective Narrative Grounding: Community-Coordinated Data Contributions to Improve Local AI Systems](collective_narrative_grounding_community-coordinated_data_contributions_to_impro.md)
- [\[NeurIPS 2025\] DNA-DetectLLM: Unveiling AI-Generated Text via a DNA-Inspired Mutation-Repair Paradigm](dna-detectllm_unveiling_ai-generated_text_via_a_dna-inspired_mutation-repair_par.md)
- [\[NeurIPS 2025\] ORBIT -- Open Recommendation Benchmark for Reproducible Research with Hidden Tests](orbit_--_open_recommendation_benchmark_for_reproducible_research_with_hidden_tes.md)
- [\[NeurIPS 2025\] AgentStealth: Reinforcing Large Language Model for Anonymizing User-generated Text](agentstealth_reinforcing_large_language_model_for_anonymizing_user-generated_tex.md)

</div>

<!-- RELATED:END -->
