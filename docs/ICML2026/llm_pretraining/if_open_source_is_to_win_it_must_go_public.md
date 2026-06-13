---
title: >-
  [Paper Note] If open source is to win, it must go public
description: >-
  [ICML 2026][LLM Pretraining][Open Source AI] This is an ICML 2026 position paper arguing that "open source AI" in its current form cannot truly democratize AI access or provide public goods like Linux or PyTorch did. It…
tags:
  - "ICML 2026"
  - "LLM Pretraining"
  - "Open Source AI"
  - "Public AI"
  - "Infrastructure"
  - "Governance"
  - "Digital Public Goods"
date: 2026-05-08
content_hash: 995bfdcee259a1e7
---

# If open source is to win, it must go public

**Conference**: ICML 2026  
**arXiv**: [2507.09296](https://arxiv.org/abs/2507.09296)  
**Code**: None (Position Paper)  
**Area**: Others / AI Governance & Open Source Ecosystem  
**Keywords**: Open Source AI, Public AI, Infrastructure, Governance, Digital Public Goods

## TL;DR
This is an ICML 2026 position paper arguing that "open source AI" in its current form cannot truly democratize AI access or provide public goods like Linux or PyTorch did. It must be embedded within "Public AI"—comprising compute, inference, post-training, and data infrastructure provided by governments, national labs, universities, and non-profits—for open source to succeed.

## Background & Motivation

**Background**: Over the past decade, open source (PyTorch, HuggingFace Transformers, OpenCLIP, Megatron-LM, lm-eval-harness, etc.) has become a cultural and technical norm in the ML community. Community projects (EleutherAI Pile/Pythia, LAION-5B, Stable Diffusion, OLMo, RedPajama, Marin) have reached or even exceeded the capabilities of closed-source frontier labs at various points in time.

**Limitations of Prior Work**: The authors point out that the equation "Open Source = Democratized AI" is breaking down in the era of large models:

- *Pre-training Costs*: Modern large models require thousands of GPUs for weeks or months, web-scale data, and distributed engineering teams, which only a few large companies or national-level institutions can afford.
- *Post-training Barriers*: Fine-tuning, alignment, tool integration, and prompt orchestration—the stages that make a model truly "usable"—are often closed-source. RLHF data is siloed by platforms and does not flow back to the community.
- *Inference Costs*: Unlike traditional open-source software where hosting is nearly zero-cost, large model inference requires continuous GPU availability, orchestration systems, and cost management.
- *License Fragility*: "Open weight" $\neq$ open source. The LLaMA agreement contains restrictive and revocable clauses; Meta can stop releases or impose stricter limits at any time. OpenAI prohibits using its output to train competitors.
- *Incomplete Transparency*: Releasing weights is not equivalent to releasing source code. Training data, data cleaning decisions, RLHF procedures, and compute configurations remain undisclosed, preventing external researchers from verifying safety claims or reproducing behaviors.
- *Safety and Governance*: Open-source models are often "research artifacts" rather than being "deployment-ready," lacking sustained investment in red-teaming and alignment. Community contributions (benchmarks, datasets, fine-tuning tricks) are ultimately "co-opted" by closed-source frontier labs.
- *The Case of Coding Agents*: When open-source developers use subscription agents like Claude Code or Codex, their prompts, iteration processes, feedback, source code snippets, and API keys are captured as implicit data labor by private harnesses.

**Key Challenge**: The premises of the open-source software era (open contribution-use-redistribution cycles and participation via commodity hardware) have failed in the era of large models. Large models are essentially "impure public goods" or "club goods" that require scarce private complements (compute, energy, engineering teams) to be activated. The authors use a metaphor: books are non-excludable public goods, but if the catalog is so large that an average person must hire a "private guide" to find a book, access becomes club-like.

**Goal**: To explicitly state the position that "Open Source AI is insufficient to democratize AI and must be complemented by Public AI," providing four principles for Public AI, five categories of existing practices, and responses to five opposing views.

**Key Insight**: Drawing from the economics of public goods (Mazzucato, Reiss, Gries & Naudé) and STS/open-source research (Kelty, Weber, Eghbal), AI should be reframed as "Digital Public Infrastructure (DPI)" similar to roads, libraries, water, and electricity, rather than "just another software library."

**Core Idea**: Use the institutional complement of "Public AI = Public Support + Public Access + Public Accountability + Private Commitments" to fill the structural gaps in compute, post-training, inference, and governance that pure open source cannot address.

## Method

As a position paper, there are no methodology experiments, but rather a clear "argumentation structure." This is broken down into the framework, key designs (core claims), and argumentation strategies.

### Overall Architecture

The paper follows a typical 8-section structure for an ICML position paper:

1. Introduction (Charting the tension in open-source AI: Ideal vs. Commercial vs. New constraints of large models).
2. Background (Reviewing the success of ML open-source software and open-source AI projects).
3. Challenges for open-source AI (Three categories: Resources, Licenses, Governance).
4. **Position Statement** (The core claim: Open source must be embedded in Public AI, proposing the four principles).
5. Examples of public AI (Existing practices: BLOOM/Jean Zay, LAION/JUWELS, European EuroLLM/OpenEuroLLM, Public AI Inference Utility, NDIF, AVERI, SEA-HELM, etc.).
6. Alternative Views (Five opposing viewpoints + point-by-point responses).
7. Technical and Societal Implications (Specific implications for ML researchers, non-CS disciplines, open-source ecosystems, governments/funders, and the public).
8. Conclusion ("If open source is to win, it must go public").

The input is the "Current state of open-source AI + Structural economic changes in large models," and the output is the "Four-principle Public AI institutional framework + Implementation path examples + Defense against opposing views."

### Key Designs

1. **Three-dimensional Diagnosis: "Why open-source AI is insufficient in the era of large models"**:

    - Function: Structurally deconstruct the failure of pure open source in the AI era into three levels: Resource, License, and Governance, turning the "should we complement it with Public AI" question into an arguable proposition rather than an intuitive slogan.
    - Mechanism: Uses the economic concept of "impure public goods / club goods" to explain why weight openness does not equal a public good—weights require private complements (compute, data, post-training, inference) to be activated. It cites specific cases like the revocable LLaMA license and OpenAI's output restrictions as evidence that "open weight $\neq$ open source." The coding agent case illustrates a new co-optation model: "user contribution $\rightarrow$ captured by private harness $\rightarrow$ transformed into implicit data labor."
    - Design Motivation: The authors must respond to a common counter-argument—"Open source has already won, the market is working, why add bureaucracy?"—by describing structural weaknesses in a way that ML researchers can relate to (revoked licenses, siloed RLHF, data surrendered to subscription agents).

2. **Definition of Four Principles for Public AI**:

    - Function: Converge "Public AI" from a vague slogan into operational institutional norms, grounding the examples in Section 5 and the rebuttals in Section 6.
    - Mechanism:

        - *Public Support*: Public funding and infrastructure must cover not only pre-training but also inference, deployment, post-training, and data.
        - *Public Access*: Researchers from the Global South, civic technologists, and local communities outside Big Tech must be able to build, adapt, and use competitive models.
        - *Public Accountability*: Models and infrastructure must be provisioned, hosted, and maintained by institutions accountable to the public (governments, national labs, public utilities, universities, non-profits).
        - *Private Commitments*: Private entities are encouraged or required to make commitments regarding openness, safety, and community control.

        Analogy to DPI (Digital Public Infrastructure): Public stacks for identity, payments, and data exchange already exist; AI should be integrated into the same paradigm.
    - Design Motivation: Use four principles to cover "Funding—Access—Accountability—Private Constraints," avoiding a simplification to just "government-built models" or "issuing another subsidy." This accommodates models like BLOOM (public compute + non-profit) and the Public AI Inference Utility (multinational coordinated compute for free inference).

3. **Point-by-point Response to Five Opposing Views**:

    - Function: A standard defense mechanism for position papers—explicitly listing the strongest possible counter-arguments and responding to them to prevent the paper from being dismissed by reviewers or readers.
    - Mechanism (5 Views + Response points):

        - *View 1: "The market is working, let OpenAI/Meta lead"* $\rightarrow$ Response: Access $\neq$ governance $\neq$ sovereignty. LLaMA 4 might be the last of its family (Meta shifting to closed-source); the free version of Qwen Code being shut down in April 2026 proves private access can be unilaterally revoked.
        - *View 2: "Open source will win eventually, be patient"* $\rightarrow$ Response: Most powerful open models (LLaMA 3.1-8B with 6M monthly downloads) are still pre-trained by capital-rich private firms. Pure non-profit projects like EleutherAI Pythia (900k) and OLMo 3-7B (170k) have far lower downloads. The exception is LAION (CLAP 14M/month), which relies on public supercomputing—proving the necessity of Public AI.
        - *View 3: "OSS + Commercial hosting is enough"* $\rightarrow$ Response: HF/Replicate/OpenRouter are revocable commercial hostings; the LLaMA license is evidence of fragility. Projects like BLOOM on Jean Zay and openCLIP on JUWELS are already performing the role of underwriting.
        - *View 4: "Regulation is better than public investment"* $\rightarrow$ Response: Regulation can curb harm but cannot guarantee access, availability, or equal participation. Public AI proactively builds capabilities and institutions (e.g., Canada’s SCALE AI funds both).
        - *View 5: "Public AI will be inefficient and prone to capture"* $\rightarrow$ Response: GPS, the Internet, Hubble, ERC, CERN, and W3C are successful public technical infrastructures. Public AI does not mean government-exclusive models; it could be a multilateral hybrid like an "Airbus for AI." The key is structuring existing AI public procurement to serve the public interest.
    - Design Motivation: This is what distinguishes a position paper from a survey—the explicit responsibility to engage in debate. The authors arrange these views on a spectrum from "Market Fundamentalist" to "Public Failure," covering almost all possible political and economic stances.

## Key Experimental Results

While the position paper contains no experiments, it cites several key metrics to support its arguments.

### Model Download Comparison (Hugging Face, January 2026)

| Model | Monthly Downloads | Type | Implication |
|------|---------|------|------|
| LLaMA 3.1-8B | 6M | Private Open | Commercial labs dominate open models |
| EleutherAI Pythia | 900k | Pure Non-profit | An order of magnitude smaller than LLaMA |
| OLMo 3-7B | 170k | Academic Non-profit | ~35x smaller than LLaMA |
| LAION CLAP | 14M | Public Compute + Non-profit | The only counter-example on par with private firms |
| openCLIP (Single variant) | 1M–2M | Public Compute + Non-profit | Cumulative >60M |

### Public AI Compute Scale (European OpenEuroLLM Consortium)

| Dimension | Value | Description |
|------|------|------|
| Participating Institutions | 20 European entities | Consortium size |
| Compute Quota | >10M GPU-hours | EuroHPC strategic resource |
| Accessed Supercomputers | 4 | Leonardo / LUMI / JUPITER / MareNostrum5 |
| Current Model Quality | Inferior to Qwen/DeepSeek/gpt-oss | Authors admit public output lags behind the frontier |

### Key Findings

- *Asymmetry of Capital and Public Investment*: Monthly downloads for a single private model (LLaMA 3.1-8B) roughly equal the total output of all cumulative European public AI efforts, highlighting the scale disadvantage of pure public paths.
- *Explanatory Power of the LAION Counter-example*: The LAION series proves that "given public supercomputing support, non-profits can produce world-class open-source models in multimodal domains"—this is the strongest existence proof for the necessity of Public AI.
- *License Fragility as a Reality*: Reports of LLaMA 4 being the last generation and the shutdown of Qwen Code’s free version in April 2026 occurred just before the paper was written, directly undermining the stance that "OSS + hosting is enough."

## Highlights & Insights

- **Refining the concept of "open weight $\neq$ open source"**: The authors point out that what the public calls "open source AI" is technically "open weight AI," distinguishing between the source (blueprint) and the weight (artifact). This distinction clarifies the messy debate over whether LLaMA/Mistral/DeepSeek are "truly open source."
- **Coding agents as a "Co-optation 2.0" paradigm**: Using recent cases like Claude Code/Codex, the authors show that open-source developers no longer contribute just code, but prompts, iterations, and feedback. These are captured by private harnesses as "implicit data labor" without clear deletion or governance mechanisms. This observation is highly transferable to discussions about Wikipedia or StackOverflow content being swallowed by LLM training.
- **AI as Digital Public Infrastructure (DPI)**: Public AI does not invent a new concept but migrates existing DPI paradigms (ID, payments, data exchange) to AI infrastructure. This transforms the ideological question of "should governments build models" into an actionable engineering question of "how to build it according to DPI norms."
- **Spectrum-based Response to Counter-arguments**: The five views are not chosen at random but cover a full spectrum from market fundamentalism to public failure. This arrangement itself is a template for writing position papers.

## Limitations & Future Work

- *Thin Empirical Support*: As a position paper, it lacks quantitative analysis beyond HF download counts and EuroHPC figures. Many assertions (e.g., "closed-source co-optation," "opaque safety claims") rely on narrative evidence rather than systematic dataset-level empirical proof.
- *Vague Boundaries for "Public"*: The four principles do not clarify priorities between public institutions vs. non-profits, international vs. national entities, or government-led vs. multilateral alliances. The tensions between "Airbus for AI," "CERN for AI," and OpenEuroLLM are not fully explored.
- *Weak Global South Perspective*: Although the Public Access principle mentions Global South researchers, the examples in Section 5 are almost entirely European and US supercomputers, with insufficient coverage of practices in Africa, Latin America, or Southeast Asia (except SEA-HELM).
- *Relationship between Governance and Regulation*: The discussion of Public AI vs. regulation is abstract, lacking a specific scenario (e.g., safety audits or model revocability) to demonstrate how the two operate together.
- *Future Directions*: Proposing a quantifiable "Public AI Index" to score countries/projects; designing specific defense mechanisms against Public AI failure modes (e.g., capture, inefficiency, geopolitical fragmentation).

## Related Work & Insights

- **vs. Bommasani et al., 2024 "Considerations for governing open foundation models"**: That work focused on policy frameworks for governing models; this paper goes further, arguing that "governance is not enough; there must be public infrastructure to provide the substrate"—moving from "regulating" to "building."
- **vs. Widder et al., 2024 (On Big Tech co-optation of Open Source)**: This paper inherits the diagnosis that open source essentially works for Big Tech but provides a constructive institutional response—Public AI—rather than just critique.
- **vs. Mazzucato’s "Entrepreneurial State" framework**: This paper applies the argument that basic research is a public good specifically to the full stack of AI (compute/data/inference/post-training).
- **Inspirations for ML Researchers**: (1) Consider long-term accessibility over short-term free access when choosing platforms; (2) Participate in Public AI projects (NDIF, OpenEuroLLM, Pythia-like) to preserve research access to model internals; (3) Be wary of implicit data capture of your workflow by tools like coding agents.
- **Inspirations for the Open Source Community**: When contributing evaluation suites or datasets, consider whether these contributions primarily accrue to closed-source frontier labs, and choose licenses or governance structures to prevent co-optation accordingly.

## Rating
- Novelty: ⭐⭐⭐⭐ Synthesizing "open weight $\neq$ open source," coding agent co-optation, and the four principles of Public AI into the ICML context is a fresh perspective, though elements have been discussed in policy/DPI circles.
- Experimental Thoroughness: ⭐⭐ As a position paper, experiments are not required, but quantitative data is relatively thin; relies on case-based argumentation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, complete eight-section layout, spectrum-organized counter-arguments, and precise terminology. It is a model for position papers.
- Value: ⭐⭐⭐⭐ Provides actionable prompts for researchers, contributors, and policymakers (participating in public AI, structure public funds) with potential impact beyond the academic community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Names Don't Matter: Symbol-Invariant Transformer for Open-Vocabulary Learning](names_dont_matter_symbol-invariant_transformer_for_open-vocabulary_learning.md)
- [\[ICML 2026\] Data Difficulty and the Generalization--Extrapolation Tradeoff in LLM Fine-Tuning](data_difficulty_and_the_generalization--extrapolation_tradeoff_in_llm_fine-tunin.md)
- [\[ICML 2026\] The Devil is in the Condition Numbers: Why is GLU Better than non-GLU Structure?](the_devil_is_in_the_condition_numbers_why_is_glu_better_than_non-glu_structure.md)
- [\[ICML 2026\] InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition](infolaw_information_scaling_laws_for_large_language_models_with_quality-weighted.md)
- [\[ICML 2026\] Predicting Large Model Test Losses with a Noisy Quadratic System](predicting_large_model_test_losses_with_a_noisy_quadratic_system.md)

</div>

<!-- RELATED:END -->
