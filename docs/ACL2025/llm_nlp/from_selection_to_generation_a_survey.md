---
title: >-
  [Paper Note] From Selection to Generation: A Survey of LLM-based Active Learning
description: >-
  [ACL 2025][LLM (Other)][Active Learning] This paper presents the first systematic survey of active learning (AL) in the LLM era. It proposes a taxonomy structured around two orthogonal axes: Querying (selection + generation) $\times$ Annotation (human + LLM + hybrid). It comprehensively details how LLMs replace or enhance traditional methods in each step of the five-stage AL loop and extends the discussion to four major LLM learning paradigms: ICL, SFT, RLHF…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Active Learning"
  - "Large Language Models"
  - "Data Selection"
  - "Data Generation"
  - "Annotation Strategy"
date: 2026-05-08
content_hash: 08d799791a02a8ac
---

# From Selection to Generation: A Survey of LLM-based Active Learning

**Conference**: ACL 2025  
**arXiv**: [2502.11767](https://arxiv.org/abs/2502.11767)  
**Code**: None  
**Area**: LLM / NLP  
**Keywords**: Active Learning, Large Language Models, Data Selection, Data Generation, Annotation Strategy

## TL;DR

This paper presents the first systematic survey of active learning (AL) in the LLM era. It proposes a taxonomy structured around two orthogonal axes: Querying (selection + generation) $\times$ Annotation (human + LLM + hybrid). It comprehensively details how LLMs replace or enhance traditional methods in each step of the five-stage AL loop and extends the discussion to four major LLM learning paradigms: ICL, SFT, RLHF, and knowledge distillation.

## Background & Motivation

**Background**: Active Learning (AL) is a classic paradigm for "efficient training with few labels," which maximizes model performance at minimal human expense by selecting the most informative data points for annotation. Traditional AL methods rely on two categories of metrics, uncertainty (e.g., Least Confidence, Max-Entropy) and diversity (e.g., CoreSet, CDAL), to sample instances from a fixed unlabeled pool $\mathcal{U}$ for human labeling.

**Limitations of Prior Work**: Traditional AL suffers from three fundamental limitations. First, **limited search space**—it only selects from a predefined unlabeled pool, unable to bypass the coverage blind spots of the dataset itself. Second, **high and inflexible annotation costs**—relying solely on human annotators keeps the marginal cost per annotation constant. Third, **cold-start difficulty**—without initial labeled data, the target model $f_\theta$ lacks sufficient guidance for the selection strategy, leading to near-random sampling in the early rounds.

**Key Challenge**: While the emergent capabilities of LLMs allow them to play threefold roles—reasoning (evaluating sample value), generation (creating new data), and annotation (simulating human labeling)—existing AL surveys remain restricted to traditional frameworks and fail to unify these multi-faceted capabilities of LLMs under a single perspective.

**Goal**: Address the current gaps in literature, specifically: (1) the lack of a systematic taxonomy for AL techniques in the LLM era; (2) the lack of a comprehensive analysis of LLM roles across all stages of the AL loop; and (3) the lack of a unified discussion on AL applications in major LLM learning paradigms (ICL, SFT, RLHF, and distillation).

**Key Insight**: The authors observe that LLMs do more than just "select" in AL—they can generate entirely new out-of-pool instances $\mathbf{x}' \notin \mathcal{U}$, substitute human annotators at lower costs, and resolve the cold-start problem. This signifies a paradigm shift in AL from Selection to Generation.

**Core Idea**: Construct a comprehensive taxonomy using Querying $\times$ Annotation as two orthogonal dimensions, covering selection, generation, human, LLM, and hybrid methods, thereby systematically surveying how LLMs revolutionize the entire AL pipeline for the first time.

## Method

### Overall Architecture

This survey is structured around the five-stage cycle of LLM-based AL (Initialize $\rightarrow$ Query $\rightarrow$ Annotate $\rightarrow$ Train $\rightarrow$ Stop), mapping prior works to the two core dimensions of Querying and Annotation. The Querying dimension is categorized into traditional selection, LLM selection, LLM generation, and hybrid; the Annotation dimension is categorized into human, LLM, and hybrid annotation. The intersection of these dimensions covers all known LLM-based AL methods.

### Key Designs

1. **Querying Module: From Fixed Pool Selection to Open-ended Generation**:

    - **Function**: Acquire the most informative data instances for annotation and training.
    - **Mechanism**: This module develops along four routes: "traditional selection $\rightarrow$ LLM selection $\rightarrow$ LLM generation $\rightarrow$ hybrid." **Traditional selection** employs uncertainty (Least Confidence, Margin, BALD) and diversity (CoreSet, BADGE) metrics to pick samples from the pool. **LLM selection** enables LLMs to directly evaluate sample value—ActiveLLM unsupervisedly assesses uncertainty and diversity, SelectLLM ranks with prompts and uses k-NN clustering to extract few-shot examples, Ask-LLM directly evaluates training sample quality, and ActivePrune uses LLMs to prune large-scale unlabeled pools. **LLM generation** breaks through pool boundaries: in-pool generation optimizes few-shot selection using k-NN + perplexity strategies (Margatina et al.), while out-of-pool generation synthesizes new prompts with APE via Query-by-Committee + CoT, or uses LLMs to generate both samples and labels followed by rejection sampling (Yang et al.). **Hybrid methods** include NoiseAL, which uses a smaller LLM for filtering and a larger LLM for labeling, and CAL, which combines density clustering with GPT-4 to correct biases.
    - **Design Motivation**: Traditional selection is constrained by the coverage of a fixed pool and the cold-start problem; the reasoning capability of LLMs allows them to directly "understand" which samples are valuable, while their generative power expands the query search space to infinity.

2. **Annotation Module: From Solely Human to Human-Agent Collaboration**:

    - **Function**: Assign high-quality labels to the acquired data instances.
    - **Mechanism**: Three parallel routes are identified. **Human annotation** remains the gold standard—ActivePrune and CAL reduce human effort by curating samples; Active-Prompt asks annotators to verify LLM outputs; Beyond-Labels collects both labels and natural language explanations. **LLM annotation** drastically reduces costs—FreeAL distills small models without human supervision, LLMaAA incorporates in-context examples to improve annotation reliability, and Kholodna et al. use GPT-4-Turbo to annotate low-resource languages, significantly cutting costs. **Hybrid annotation** dynamically routes samples—Wang et al. utilize LLM annotation followed by a verifier, routing low-quality predictions to human review; Rouzegar & Makrehchi decide whether to hand samples to LLMs or humans based on confidence thresholds.
    - **Design Motivation**: Purely human annotation is expensive and unscalable, whereas purely LLM annotation suffers from biases (e.g., Western cultural bias, self-reinforcing loops, prompt sensitivity). Hybrid strategies achieve an optimal balance between cost and quality.

3. **Stopping and Extending to LLM Learning Paradigms**:

    - **Function**: Determine when to terminate the AL loop and apply AL techniques to various LLM training paradigms.
    - **Mechanism**: Regarding **stopping criteria**, traditional AL relies on a fixed budget $k$ or performance convergence thresholds. However, LLM-based AL introduces a complex cost structure incorporating both human annotation fees and LLM API call costs (dependent on input/output token counts). Thus, budgets must be modeled as real-valued currency rather than discrete counts. Hybrid stopping criteria proposed by Akins et al. and Pullar-Strecker et al. combine token-level cost analysis with performance plateau detection. Regarding **paradigm extension**, AL has penetrated four major LLM learning paradigms: (a) Active ICL—formulating few-shot demonstration selection as an AL problem to optimize prompts via semantic coverage and ambiguity-driven sampling; (b) Active SFT—curating fine-tuning data using uncertainty queries and self-training strategies; (c) Active Preference Alignment—accelerating alignment in RLHF via targeted preference feedback queries; (d) Active Knowledge Distillation—selectively distilling LLM knowledge to smaller models using uncertainty sampling.
    - **Design Motivation**: Fundamental changes in the cost model demand a redesign of stopping conditions; AL is upgraded from an "annotation efficiency utility" to a "data strategy spanning the entire lifecycle of LLM training."

### Loss & Training

As a survey paper, this work does not propose new loss functions. However, it summarizes key training strategies within the AL loop: the target model $f_\theta$ updates its parameters after each iteration using newly labeled data; LLM generation combined with rejection sampling ensures only samples crossing a predefined accuracy threshold enter the training set; in Active SFT, self-training is applied to low-uncertainty data to train directly without requiring human annotations.

## Key Experimental Results

### Main Results

As this is a survey, there are no original benchmark experiments, but key findings of representative methods across different tasks are summarized:

| Method | Querying Strategy | Annotation Strategy | Main Tasks | Key Findings |
|------|-------------|----------------|---------|---------|
| ActiveLLM | LLM Selection | Human | Text Classification | Unsupervised LLM selection matches traditional AL in few-shot and model mismatch scenarios |
| SelectLLM | LLM Selection | Human | Few-shot Learning | LLM ranking + k-NN clustering outperforms random sampling and uncertainty-based selection |
| Ask-LLM | LLM Selection | — | Data Quality Filtering | LLM quality scoring effectively filters out low-quality training data |
| APE | LLM Generation (Out-of-pool) | Human | Entity Matching | Query-by-Committee + CoT synthesized new prompts to improve labeling efficiency |
| FreeAL | Hybrid | LLM | Text Classification / Sentiment Analysis | Learns usable performance via LLM + small model distillation under zero human supervision |
| NoiseAL | Hybrid (Selection + Gen) | LLM | Text Classification | A two-stage pipeline of small LLM filtering + large LLM labeling effectively reduces costs and boosts efficiency |
| CAL | Hybrid | Human | Debiasing | Density clustering + GPT-4 querying autonomously identifies and corrects data bias patterns |

### Classification Coverage Analysis

| Dimension | Subcategory | Representative Count | Exemplar Methods |
|------|--------|------------|---------|
| Querying - Traditional Selection | Uncertainty / Diversity | 6+ | BADGE, BALD, CoreSet |
| Querying - LLM Selection | LLM Evaluation / Ranking | 4 | ActiveLLM, SelectLLM, Ask-LLM, ActivePrune |
| Querying - LLM Generation | In-pool / Out-of-pool Gen | 5+ | APE, EAGLE, Diao et al., Yang et al. |
| Querying - Hybrid | Selection + Gen Hybrid | 2 | NoiseAL, CAL |
| Annotation - Human | Traditional Human | 5+ | Active-Prompt, Beyond-Labels, APL |
| Annotation - LLM | LLM Annotation | 3+ | FreeAL, LLMaAA, Kholodna et al. |
| Annotation - Hybrid | Human-Agent Collaboration | 3+ | Wang et al., HybridAL, AutoLabel |

### Key Findings

- **LLM Selection vs. Traditional Selection**: In few-shot and cold-start scenarios, LLM-based selection exhibits a clear advantage. For instance, ActiveLLM matches traditional AL in a completely unsupervised manner, as LLM semantic understanding compensates for the lack of information in the initial target model.
- **Value of Out-of-pool Data Generation**: Out-of-pool data generated by LLMs effectively expands training set coverage, especially when annotated data is scarce; however, rejection sampling remains a necessary step to ensure generation quality.
- **Failure of Traditional Uncertainty Sampling**: Experiments by Margatina et al. demonstrate that traditional uncertainty sampling underperforms compared to k-NN + diversity strategies in LLM few-shot ICL settings, potentially because the underlying mechanism of ICL differs from standard supervised learning.
- **Bias Risks in LLM Annotation**: LLM annotations carry threefold risks: Western cultural bias (Atari et al.), self-reinforcing feedback loops (when LLMs annotate LLM-generated data), and high prompt sensitivity. Hybrid annotation serves as the currently optimal trade-off.

## Highlights & Insights

- **Paradigm Shift: From Selection to Generation**: The search space of traditional AL is restricted to a fixed data pool $\mathcal{U}$, whereas LLM expands it to an infinite generative space $\mathbf{x}' \notin \mathcal{U}$. Rather than an incremental improvement, this represents a paradigm shift, altering AL from "selecting the best from existing data" to "creating the most needed data."
- **Simplifying Power of the Two-Dimensional Taxonomy**: The orthogonal Querying $\times$ Annotation categorization maps all methods to a $4\times3$ matrix, clearly illustrating which combinations are well-explored and which remain blank. This taxonomy functions effectively as a research roadmap.
- **Fundamental Reconstruction of Cost Models**: Traditional AL assumes uniform annotation costs. In contrast, LLM-based AL costs are real-valued functions based on tokens. This demands a complete redesign of stopping criteria, budget allocations, and routing strategies, representing an underappreciated open problem.
- **A Unified AL Perspective Across Four LLM Paradigms**: Structuring ICL demonstration selection, SFT data selection, RLHF preference querying, and knowledge distillation sample selection under a single AL framework provides an insightful perspective on cross-paradigm method migration.

## Limitations & Future Work

- **Lack of a Unified Benchmark**: Various methods are evaluated on different datasets and configurations, preventing direct horizontal comparisons. The survey does not provide unified experiments, leaving a lack of empirical guidance for model selection.
- **Absence of Systematic Evaluation for LLM Annotation Quality**: Although the survey highlights LLM annotation biases and inconsistencies, it lacks a systematic evaluation framework for annotation quality (such as criteria determining when LLM annotations are reliable or not).
- **Weak Theoretical Foundations**: Theoretical guarantees for LLM-based AL (such as sample complexity bounds under the PAC learning framework) are practically non-existent; existing methods remain heavily empirical.
- **Insufficient Multimodal Coverage**: The vast majority of discussed methods target textual tasks, leaving multimodal AL (e.g., images, audio, video) significantly under-discussed.
- **Multi-LLM Systems**: Different LLMs exhibit vast disparities in cost and capability (e.g., GPT-4 vs. smaller models). Systematically combining multiple LLMs (e.g., as done in NoiseAL) to optimize cost-performance trade-offs is a prominent future direction.

## Related Work & Insights

- **vs. Traditional AL Surveys (Settles 2009, Ren et al. 2021, Zhan et al. 2022)**: Traditional surveys focus primarily on uncertainty/diversity selection strategies, while this survey expands the scope to the generation and annotation capabilities of LLMs, filling the blank in traditional surveys in the LLM era.
- **vs. LLM Data Synthesis Surveys**: LLM data synthesis (e.g., Self-Instruct) focuses on large-scale data generation but lacks the "informativeness maximization" selection principle of AL. This survey's contribution lies in integrating data generation into active learning's optimization framework.
- **vs. RLHF Literature**: Preference data collection in RLHF is essentially an AL problem (selecting which prompts humans should label with preferences). Pointing out the direction of "Active Preference Alignment" makes this survey highly relevant to alignment researchers.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic survey on LLM-based AL. The Querying $\times$ Annotation taxonomy is concise and powerful, offering insightful analysis of the paradigm shift from selection to generation.
- Experimental Thoroughness: ⭐⭐ As a survey paper, it contains no original experiments, relying strictly on cited literature, and lacks evaluations under a unified benchmark.
- Writing Quality: ⭐⭐⭐⭐ Highly organized with an intuitive taxonomy and well-designed tables and figures. It features broad coverage, though some individual method descriptions are somewhat brief.
- Value: ⭐⭐⭐⭐ Serves as an excellent introductory index and methodological map for researchers in LLM-based AL, providing a taxonomy that directly guides the design and positioning of new methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Personalized Generation In Large Model Era: A Survey](personalized_generation_in_large_model_era_a_survey.md)
- [\[ACL 2025\] PlanGenLLMs: A Modern Survey of LLM Planning Capabilities](plangenllms_planning_survey.md)
- [\[ACL 2025\] A Survey of LLM-based Agents in Medicine: How Far Are We from Baymax?](a_survey_of_llm-based_agents_in_medicine_how_far_are_we_from_baymax.md)
- [\[ACL 2025\] LLM as a Broken Telephone: Iterative Generation Distorts Information](llm_broken_telephone.md)
- [\[ACL 2025\] Transforming Podcast Preview Generation: From Expert Models to LLM-Based Systems](transforming_podcast_preview_generation_from_expert_models_to_llm-based_systems.md)

</div>

<!-- RELATED:END -->
