---
title: >-
  [Paper Note] If open source is to win, it must go public
description: >-
  [ICML 2026][Pretraining][Paper Note] This is an ICML 2026 position paper arguing that "open-source AI" in its current form cannot truly democratize AI access or provide public goods like Linux or PyTorch did. It posits that for open source to succeed, it must be embedded within "Public AI"—a framework of compute, inference, post-training, and data infrast
tags:
  - ICML 2026
  - Pretraining
date: 2026-05-08
content_hash: 37c55f0add3ac73f
---
# If open source is to win, it must go public

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2507.09296](https://arxiv.org/abs/2507.09296)  
**Code**: None (Position Paper)  
**Area**: Others / AI Governance and Open Source Ecosystem  
**Keywords**: Open Source AI, Public AI, Infrastructure, Governance, Digital Public Goods

## TL;DR
This is an ICML 2026 position paper arguing that "open-source AI" in its current form cannot truly democratize AI access or provide public goods like Linux or PyTorch did. It posits that for open source to succeed, it must be embedded within "Public AI"—a framework of compute, inference, post-training, and data infrastructure provided by governments, national laboratories, universities, and non-profit institutions.

## Background & Motivation

**Background**: Over the past decade, open-source software (PyTorch, HuggingFace Transformers, OpenCLIP, Megatron-LM, lm-eval-harness, etc.) has become a cultural and technical norm in the ML community. Community projects (EleutherAI Pile/Pythia, LAION-5B, Stable Diffusion, OLMo, RedPajama, Marin) have also reached or exceeded the capabilities of closed-source frontier labs at various points in time.

**Limitations of Prior Work**: The authors argue that the equation "Open Source = AI Democratization" is breaking down in the era of large models due to:

- *Pre-training Costs*: Modern large models require thousands of GPUs and weeks or months of training, necessitating web-scale data and distributed engineering teams that only a few large corporations or state-level institutions can afford.
- *Post-training Barriers*: Fine-tuning, alignment, tool integration, and prompt orchestration—stages that make models truly usable—are often closed-source. RLHF data is siloed by platforms and does not flow back to the community.
- *Inference Costs*: Unlike traditional open-source software, which has near-zero hosting costs, large model inference requires continuous GPU resources, orchestration systems, and cost management.
- *License Fragility*: "Open weights" $\neq$ open source. The LLaMA agreement contains restrictive and revocable clauses; Meta can stop distribution or add stricter limits at any time. OpenAI prohibits using its output to train competing products.
- *Incomplete Transparency*: Releasing weights is not equivalent to releasing source code. Training data, data cleaning decisions, RLHF processes, and compute configurations remain undisclosed, preventing external researchers from verifying safety claims or reproducing behaviors.
- *Safety and Governance*: Open-source models often serve as "research artifacts" rather than being "deployment-ready," lacking sustained investment in red-teaming and alignment. Community contributions (evaluations, datasets, fine-tuning tricks) are ultimately "co-opted" by closed-source frontier labs.
- *The Coding Agent Case*: When open-source developers use subscription agents like Claude Code or Codex, their prompts, iterations, feedback, source code snippets, and API keys are captured as implicit data labor by private harnesses.

**Key Challenge**: The premises of the open-source software era (cycles of contribution-usage-redistribution open to everyone, participation via commodity hardware) have failed in the era of large models. Large models are essentially "impure public goods" or "club goods" that require scarce private complements (compute, energy, engineering teams) to be activated. The authors use an analogy: the book remains a non-excludable public good, but the catalog is so vast that ordinary people must hire "private guides" to find a book, turning access into something club-like.

**Goal**: To explicitly propose the position that "Open-source AI is insufficient to democratize AI and must be complemented by Public AI," providing four principles for Public AI, five categories of existing practical cases, and responses to five counter-arguments.

**Key Insight**: Drawing from the economics of public goods (Mazzucato, Reiss, Gries & Naudé) and STS/Open Source studies (Kelty, Weber, Eghbal), AI should be reframed as "Digital Public Infrastructure (DPI)" similar to roads, libraries, water, and electricity, rather than just "another software library."

**Core Idea**: Use the institutional completion of "Public AI = Public Support + Public Access + Public Accountability + Private Commitment" to fill the structural gaps in compute, post-training, inference, and governance left by pure open source.

## Method

As a position paper, there are no methodology experiments, but rather a clear argumentative structure organized across eight typical sections. It begins by introducing the tensions in open-source AI (Ideals vs. Commercialism vs. New Model Constraints), reviews the success story of ML open-source software and projects in the Background, summarizes three categories of challenges (Resources, Licenses, Governance) in Section 3, and proposes the core claim and definition of the four principles in Section 4. Section 5 use projects like BLOOM/Jean Zay, LAION/JUWELS, EuroLLM/OpenEuroLLM, Public AI Inference Utility, NDIF, AVERI, and SEA-HELM as existential proofs that "Public AI is not a fantasy." Section 6 responds to counter-arguments, Section 7 discusses implementation implications for different audiences, and Section 8 concludes with the title's thesis. The input is the "current state of open-source AI + structural changes in LLM economics," and the output is the "Four-Principle Public AI institutional framework + implementation path examples + defense against counter-arguments."

### Overall Architecture

The entire paper functions as an argumentative chain: it diagnoses why pure open source fails in the LLM era, provides institutional completion through the four principles of Public AI, and finally assumes the responsibility of debating the five strongest counter-arguments. The methodological weapon throughout is the economics of public goods—concepts unfamiliar to the ML community like "impure public good," "club good," and lighthouse financing are introduced and paired with contemporary empirical anchors (LLaMA 4 potentially being the last generation, the shutdown of Qwen Code free version, coding agents capturing user workflows) to turn "potential risks" into "realized events."

### Key Designs

**1. Three-Dimensional Diagnosis: Why Open-Source AI is Not Enough**

The authors structurally decompose the failure of pure open source in the AI era into resource, licensing, and governance layers, transforming the question of "whether to supplement with Public AI" from an intuitive slogan into a demonstrable proposition. Economically, they use "impure public goods / club goods" to explain why open weights do not equal public goods—weights require private complements like compute, data, post-training, and inference to activate. They cite specific cases like the revocability of LLaMA licenses and OpenAI's ban on using outputs for competing models to establish that "open weights $\neq$ open source." Finally, the coding agent example illustrates a new type of co-optation: "user contribution $\rightarrow$ captured by private harness $\rightarrow$ converted into implicit data labor."

**2. Definition of the Four Principles of Public AI**

The second step converges "Public AI" from a vague slogan into operational institutional norms. **Public Support** requires public funding and infrastructure to cover not just pre-training, but also inference, deployment, post-training, and data. **Public Access** requires that researchers in the Global South, civic technologists, and local communities outside Big Tech can build, adapt, and use competitive models. **Public Accountability** requires that models and infrastructure be provisioned, hosted, and maintained by institutions accountable to the public (governments, national labs, public utilities, universities, non-profits). **Private Commitments** encourage or require private entities to make promises regarding openness, safety, and community control.

**3. Response to Five Opposing Views**

The core of a position paper is explicitly listing and responding to the strongest possible counter-arguments.  
- **View 1: "The market is working; let OpenAI/Meta lead."**  
  *Response*: Access $\neq$ governance $\neq$ sovereignty. The potential end of the LLaMA family and the shutdown of free services prove that private access can be unilaterally revoked.  
- **View 2: "Open source will eventually win; be patient."**  
  *Response*: Current leading "open" models are mostly pre-trained by well-capitalized private firms. Purely non-profit projects like Pythia or OLMo have significantly lower adoption. The exception, LAION, relies on public supercomputing, which actually proves the necessity of Public AI.  
- **View 3: "OSS + commercial hosting is enough."**  
  *Response*: Platforms like HF or Replicate are revocable commercial hosts. Stable hosting requires public underwriting, as seen with BLOOM on Jean Zay.  
- **View 4: "Regulation is better than public investment."**  
  *Response*: Regulation can curb harm but cannot guarantee access, usability, or equitable participation. Public AI proactively builds capabilities.  
- **View 5: "Public AI will be inefficient and easily captured."**  
  *Response*: GPS, the Internet, CERN, and W3C are successful public infrastructures. Public AI is not a government monopoly but can be a multilateral hybrid structure like an "Airbus for AI."

## Key Experimental Results

As a position paper, there are no new experiments, but it cites key data to support its arguments.

### Model Download Comparison (Hugging Face, January 2026)

| Model | Monthly Downloads | Type | Signification |
|-------|-------------------|------|---------------|
| LLaMA 3.1-8B | 6M | Private Open | Commercial labs dominate |
| EleutherAI Pythia | 900k | Pure Non-profit | Order of magnitude smaller than LLaMA |
| OLMo 3-7B | 170k | Academic Non-profit | ~35x smaller than LLaMA |
| LAION CLAP | 14M | Public Compute + Non-profit | Only example rivaling private firms |
| openCLIP (Single variant) | 1M–2M | Public Compute + Non-profit | Cumulative >60M |

### Public AI Compute Scale (European OpenEuroLLM Consortium)

| Dimension | Value | Description |
|-----------|-------|-------------|
| Participating Institutions | 20 European entities | Consortium scale |
| Compute Quota | >10M GPU-hours | EuroHPC strategic resource |
| Accessed Supercomputers | 4 | Leonardo / LUMI / JUPITER / MareNostrum5 |
| Current Model Quality | Still lags behind Qwen / DeepSeek / GPT-OSS | Authors admit public output trails the frontier |

### Key Findings

- *Asymmetry between Capital and Public Investment*: Monthly downloads for a single private model (LLaMA 3.1-8B) roughly equal the cumulative results of all European public AI investments, highlighting the scale disadvantage of purely public paths.
- *Explanatory Power of the LAION Counter-example*: The LAION series proves that if provided with public supercomputing support, non-profits can produce world-class open-source models in the multimodal domain—this is the strongest existential evidence for Public AI.
- *License Vulnerability as Fact*: Reported changes to the LLaMA family and the shutdown of free Qwen versions are used as "just-happened" events that directly undermine the stance that "OSS + hosting is sufficient."

## Highlights & Insights

- **Precision of the "open weight $\neq$ open source" concept**: The authors clearly state that what is colloquially called "open-source AI" is technically "open weight AI," distinguishing the source (blueprints) from weights (finished product).
- **Coding agents as the paradigm of "Co-optation 2.0"**: Using the late-2025 case of Claude Code, the authors explain how developers no longer contribute just code, but also prompts, feedback, and file system context, which are captured as "implicit data labor" without governance mechanisms.
- **Designing AI as DPI**: Public AI reframes the debate by migrating established Digital Public Infrastructure paradigms (identity, payments, data exchange) to AI, moving from ideological questions of "should government build models" to actionable engineering questions.
- **Spectrum-style response to counter-arguments**: The five views cover the full spectrum from market fundamentalism to public failure pessimism, providing a textbook example of how to structure a position paper.

## Limitations & Future Work

- *Thin Empirical Support*: The paper is positional; beyond HF downloads and EuroHPC figures, there is little quantitative analysis. Many assertions regarding "co-optation" or "opaque claims" rely on narrative rather than systematic datasets.
- *Vague "Public" Boundaries*: The four principles do not explicitly prioritize between public institutions vs. non-profits, or international vs. national efforts. The tensions between models like "Airbus for AI" and "CERN for AI" are not fully explored.
- *Weak Global South Perspective*: Although Public Access is a principle, most examples are Euro-American. Coverage of public AI practices in Africa, Latin America, or Southeast Asia is insufficient.
- *Relationship Between Governance and Regulation*: The interaction between Public AI and regulation remains abstract; a specific scenario (e.g., safety auditing) demonstrating how they work in tandem is missing.
- *Future Work*: Proposing a quantifiable "Public AI Index" to score projects based on the four dimensions, providing policymakers with an actionable framework, and designing defense mechanisms against public failure modes (e.g., capture, inefficiency).

## Related Work & Insights

- **vs. Bommasani et al., 2024 ("Considerations for governing open foundation models")**: While that work focuses on policy frameworks for governing models, this paper argues that governance is insufficient without the public infrastructure to provide the substrate.
- **vs. Widder et al., 2024 (On Big Tech co-optation of open source)**: This paper inherits the diagnosis that open source serves large corporations but offers a constructive institutional response through Public AI.
- **vs. Mazzucato's "Entrepreneurial State"**: This paper applies the argument for basic research as a public good specifically to the full stack of LLM compute, data, inference, and post-training.
- **Insights for ML Researchers**: (1) Consider long-term accessibility over short-term "free" access when choosing platforms; (2) Participate in public AI projects to maintain research access to model internals; (3) Be wary of implicit data capture by tools in one's workflow.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Synthesizing "open weight $\neq$ open source," co-optation, and the DPI framework within an ICML context is a fresh angle, though individual components have appeared in policy circles.
- **Experimental Thoroughness**: ⭐⭐ Position papers do not require experiments, but the quantitative data is relatively sparse. It relies heavily on case studies.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear structure, complete eight-section layout, and precise terminology make this a model position paper.
- **Value**: ⭐⭐⭐⭐ Offers actionable prompts for researchers, contributors, and policymakers, with potential impact extending beyond the academic ML community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1"></div>

## Related Papers

- [\[ECCV 2024\] Plan, Posture and Go: Towards Open-Vocabulary Text-to-Motion Generation](../../ECCV2024/llm_pretraining/plan_posture_and_go_towards_open-vocabulary_text-to-motion_generation.md)
- [\[ICML 2026\] Names Don't Matter: Symbol-Invariant Transformer for Open-Vocabulary Learning](names_dont_matter_symbol-invariant_transformer_for_open-vocabulary_learning.md)
- [\[ECCV 2024\] I Can't Believe It's Not Scene Flow!](../../ECCV2024/llm_pretraining/i_canapost_believe_itaposs_not_scene_flow.md)
- [\[ICML 2026\] Incremental BPE Tokenization](incremental_bpe_tokenization.md)
- [\[ICML 2026\] Focus and Dilution: The Multi-stage Learning Process of Attention](focus_and_dilution_the_multi-stage_learning_process_of_attention.md)

</div>

<!-- RELATED:END -->
