---
title: >-
  [Paper Note] If Open Source Is to Win, It Must Go Public
description: >-
  [ICML 2025][Open-source AI] This paper argues that open-source AI, under current practices, cannot independently achieve AI democratization—model weights are merely "inert code" requiring substantial capital to activate. It must be embedded within public AI infrastructure (public funding + public access + public governance + private commitment) to serve as a genuine public good.
tags:
  - "ICML 2025"
  - "Open-source AI"
  - "Public AI"
  - "AI Governance"
  - "AI Infrastructure"
  - "AI Democratization"
date: 2026-05-08
content_hash: e6eb0b2f91105040
---

# If Open Source Is to Win, It Must Go Public

**Conference**: ICML 2025  
**arXiv**: [2507.09296](https://arxiv.org/abs/2507.09296)  
**Code**: None  
**Area**: Other  
**Keywords**: Open-source AI, Public AI, AI Governance, AI Infrastructure, AI Democratization

## TL;DR

This paper argues that open-source AI, under current practices, cannot independently achieve AI democratization—model weights are merely "inert code" requiring substantial capital to activate. It must be embedded within public AI infrastructure (public funding + public access + public governance + private commitment) to serve as a genuine public good.

## Background & Motivation

**Background**: Open-source software has achieved immense success in domains such as Linux and Kubernetes, and the ML community has adopted openness as both a technical and cultural norm—frameworks like PyTorch and Hugging Face Transformers have made advanced ML tools widely accessible, while projects like EleutherAI's GPT-NeoX/Pythia and BigScience's BLOOM continue to drive the development of open models.

**Limitations of Prior Work**: However, AI models differ fundamentally from traditional software codebases. While users of traditional open-source software only require a CPU to compile and run code, AI models require massive resources to be "activated." This disparity manifests at multiple levels: pre-training requires thousands of GPUs for weeks to months (Llama 3 training costs are estimated to exceed $100 million); large-scale inference demands persistent GPU clusters and orchestration systems; steps that make models truly useful, such as fine-tuning, alignment, and tool integration, are typically closed; and licenses (e.g., Llama license) include restrictive terms and can be unilaterally revoked.

**Key Challenge**: Economically, open-source models are not pure public goods (non-rivalrous and non-excludable) but rather "impure public goods" or "club goods"—the model weights themselves are non-rivalrous, but the capability to perform inference, fine-tuning, and deployment to make them useful is excludable. The paper offers a brilliant analogy: imagine a library with an enormous collection and an open license, but as the catalog grows exponentially, ordinary users must hire private "guides" to find any books—the books remain public goods, but access to knowledge is mediated by fee-charging services for "search and retrieval."

**Goal**: (1) Diagnose the structural challenges facing open-source AI—why open weights $\neq$ democratized access? (2) Propose a "Public AI" framework as a complement—what institutional designs are required? (3) Systematically address five counter-arguments.

**Key Insight**: Starting from political economy and public goods theory, the authors analyze the "activation cost" of AI models, pointing out that the success of the open-source model in the software era relied on low-cost computing and interoperable standards—two conditions that no longer hold true in the era of large models.

**Core Idea**: Open-source AI must be embedded within a public AI ecosystem—supported by public funding, accountable through public governance, and ensuring public access—to achieve true AI democratization. Otherwise, open weights merely serve as a subsidy for the few with the capital to deploy them.

## Method

### Overall Architecture

As a position paper, the core architecture of this work is: diagnose the problem (why open-source AI is insufficient) $\to$ propose solutions (the four principles of public AI) $\to$ summarize global practices (four public AI modes) $\to$ address five counter-arguments $\to$ analyze technical and social impacts.

### Key Designs

1. **AI Models' "Activation Cost" Diagnosis**:

    - **Function**: Systematically analyze the gap between open-source AI weights and usable systems.
    - **Mechanism**: The paper identifies five fracture points between "usable weights" and "usable systems"—pre-training requires capital and scale (thousands of GPUs $\times$ months), inference is not free (continuous GPUs + orchestration systems), post-training involves private data and design choices, licenses are ambiguous or fragile (Llama can be unilaterally revoked), and transparency is partial and inconsistent. These characteristics make AI models "impure public goods" in economic terms.
    - **Design Motivation**: Only by accurately diagnosing the problem can targeted solutions be proposed. The paper emphasizes that this is not a failure of values, but a structural failure.

2. **Public AI Four-Principles Framework**:

    - **Function**: Define the core principles of public AI.
    - **Mechanism**: (1) **Public Support**—not limited to funding pre-training, but also covering public funding and infrastructure for inference, deployment, post-training, and data flywheels; (2) **Public Access**—making models buildable and usable for Global South researchers, civic technologists, and communities outside Big Tech; (3) **Public Governance**—provisioned and maintained by public institutions such as governments, national laboratories, universities, and non-profits; (4) **Private Commitment**—encouraging or requiring private actors to commit to openness and safety.
    - **Design Motivation**: The four principles cover the entire ecological chain from funding to access, governance, and the private sector, addressing the structural deficiencies of open-source AI at every level.

3. **Global Public AI Practice Classification**:

    - **Function**: Summarize existing global practices of public AI.
    - **Mechanism**: Four modes cover varying depths and methods of government involvement in AI, ranging from the lightest (outsourcing) to the heaviest (direct public provision), providing reference pathways for different national contexts.
    - **Design Motivation**: Demonstrate that public AI is not a theoretical fantasy, but one with abundant ongoing practical exploration globally.

### Systematic Responses to Five Alternative Viewpoints

| Counter-argument | Core Argument | Paper's Response | Key Data/Evidence |
|---------|---------|---------|-------------|
| "The market functions well" | Just let OpenAI and Meta dominate | Access $\neq$ governance; systems are non-transparent and unilaterally revocable | Sweden's GPT-SW3 was born because ChatGPT performed poorly on Nordic languages |
| "Open-source will eventually win" | Community innovation will surpass closed-source | The strongest open-source models are pre-trained by well-funded private firms | Hugging Face monthly downloads: Llama 3.1-8B (6M) vs OLMo 2-7B (29k) |
| "OSS + Hosting is sufficient" | The existing ecosystem is complete, no new governance is needed | Current usability $\neq$ long-term stability; commercial hosting can be revoked | LLaMA license terms contain revocability |
| "Regulation is better" | Shape AI through regulation rather than investment | Regulation is necessary but insufficient; it cannot guarantee access and equitable participation | Canada's SCALE AI funds both regulation and capability building |
| "Public AI will be inefficient" | Government projects are historically inefficient and prone to capture | GPS, the Internet, and Hubble are all achievements of public institutions | ERC, CERN, and W3C prove that public AI can resist capture |

## Key Experimental Results

### Main Data: Comparison of Open Source Model Adoption

| Model | Organization | Type | Monthly Hugging Face Downloads | Notes |
|------|------|------|---------------------|------|
| Llama 3.1-8B-Instruct | Meta (Commercial) | Restricted Open Source | ~6 Million | The strongest/most used open-source models are pre-trained by commercial companies |
| Pythia | EleutherAI (Non-profit) | Fully Open Source | ~900,000 | Usage of community models is far lower than commercial open-source |
| OLMo 2-7B | AI2 (Research Institute) | Fully Open Source | ~29,000 | Usage of academic models is even lower |

This data clearly demonstrates that although community and academic open-source AI projects are numerous, actual adoption is heavily dominated by the "restricted open-source" models of commercial companies.

### Analysis: Cost Structure of the AI Model Lifecycle

| Stage | Supported by Public Funding? | Description |
|------|------------------|------|
| Pre-training | Limited / Occasionally | Some academic projects receive state funding (e.g., BLOOM using French public supercomputing resources) |
| Inference | Extremely Rare | NDIF is a rare exception |
| Post-training (RLHF/Fine-tuning) | Almost None | Typically private and closed |
| Deployment/Hosting | Almost None | Relies on commercial hosting with variable terms |
| Data Flywheel | None | Decentralized deployment leads to RLHF/query data being scattered in isolated silos |

### Global Public AI Practice Modes

| Mode | Description | Representative Cases | Government Involvement Level |
|------|------|---------|----------|
| Outsourced Provision | Government contracts with private labs | IndiaAI, US NAIRR | Low |
| Networked Collaboration | Developed collaboratively by academic and civic actors | Empire AI, Canada's public computing investments | Medium |
| State-Backed Integration | High government control over private enterprises | Chinese AI projects, UAE AI Strategy | High |
| Public Option | Direct public provision of AI services | Sweden's GPT-SW3, Japan's ABCI | Highest |

### Key Findings

- **Open Weights $\neq$ Usable Systems**: As the complexity of AI systems grows—evolving from "token prediction on local GPUs" to "AI assistants that integrate multimodal reasoning, proprietary tools, and complex orchestration layers"—the gap between "usable weights" and "usable systems" continues to widen.
- **The Illusion of "Shared Infrastructure" in Open Source**: The valuation of community-contributed evaluation benchmarks, tool chains, datasets, and fine-tuning techniques is often captured by the frontier labs training closed-source models. While contributors believe they are building shared infrastructure, they may in fact be feeding the concentration of power.
- **Model Release Does Not Equal Commitment**: Companies like Meta can stop releasing models at any time or impose more restrictive terms on future licenses. This fragility is structural.

## Highlights & Insights

- **The brilliant analogy of "impure public goods"**: Comparing AI models to a library with an enormous collection but requiring a paid guide to access captures the structural contradiction of open-source AI beautifully, making it more persuasive than purely technical arguments.
- **Deep exposure of corporate "collusion" in open source**: Pointing out the strategic intent of corporations in promoting open-source AI—if public money subsidizes inference costs, the ultimate value will aggregate at the application layer (i.e., those same corporations). This insight is rare in AI policy discussions.
- **The diagnostic framework of "structure rather than values"**: Clearly distinguishing that "this is not a failure of values, but a structural failure" avoids moralizing and makes the arguments more palatable to policymakers.
- **Systematic classification of global practices**: Providing a structured reference framework of four public AI modes (outsourced/networked/state-backed/public option) alongside emerging proposals for "AI Airbus" and "AI CERN."

## Limitations & Future Work

- **Lack of quantitative analysis and empirical data**: As a position paper, it relies primarily on qualitative reasoning. While Hugging Face download counts are strong data points, the paper lacks a cost-quantification comparison across pre-training, inference, and deployment phases.
- **Vague implementation pathways for Public AI**: The four-principles framework describes "what should be" but lacks detailed policy recommendations on "how to do it"—e.g., funding scale, governance structures, and operational models for public inference infrastructure.
- **Overlooking the rapid progress of open-source AI**: Local deployment tools like `llama.cpp` and `Ollama` are rapidly lowering the barrier to inference, but the paper only mentions this in passing without fully discussing how this trend affects its arguments.
- **Under-discussion of Global South perspectives**: The paper mentions that public access should cover Global South researchers, but all case studies are concentrated in the US, Europe, and Japan.
- **Omission of the special role of open-source AI in safety research**: The transparency advantages of open-source models for safety research (such as auditability and reproducibility) are not sufficiently emphasized.

## Related Work & Insights

- **vs Bommasani et al. (2024) "Considerations for Governing Open Foundation Models"**: Bommasani et al. discuss the governance considerations of open foundation models, but primarily from a governance framework perspective. This paper goes further, arguing that governance frameworks themselves are insufficient and require public infrastructure to back them.
- **vs Mazzucato (2013) "The Entrepreneurial State"**: Mazzucato argues for the active role of the state in innovation (e.g., GPS, Internet). This paper directly applies this framework to the AI field, arguing that public AI is not synonymous with inefficiency.
- **vs Widder et al. (2024)**: Widder et al. point out how open-source contributions are captured by private entities. This paper builds on this by proposing structural solutions—ensuring that open-source contributions are not captured by private actors through Public AI.

## Rating

- Novelty: ⭐⭐⭐⭐ Systematically applying public goods theory to analyze the open-source AI ecosystem; the "impure public goods" framework is novel and highly explanatory.
- Experimental Thoroughness: ⭐⭐⭐ Position paper without traditional experiments, but with rigorous logic and rich case studies; the Hugging Face download comparison provides effective empirical support.
- Writing Quality: ⭐⭐⭐⭐⭐ Structurally clear arguments, detailed point-by-point responses to counter-arguments showing a high level of academic debate, and excellent analogies.
- Value: ⭐⭐⭐⭐ Highly valuable reference in AI governance and policy fields, offering inspiring reflections for the ML community to understand the limits of the open-source ecosystem.
---
title: >-
  [Paper Review] If Open Source Is to Win, It Must Go Public
description: >-
  [ICML 2025][Human Understanding] This is a position paper arguing that open-source AI alone cannot achieve the democratized access of AI. It must be embedded into a broader "Public AI" infrastructure—including public funding, public access, public governance, and private commitment—for open models to truly become public goods.
tags:
  - ICML 2025
  - Human Understanding
---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Democratic AI is Possible. The Democracy Levels Framework Shows How It Might Work](democratic_ai_is_possible_the_democracy_levels_framework_shows_how_it_might_work.md)
- [\[ACL 2025\] CiteEval: Principle-Driven Citation Evaluation for Source Attribution](../../ACL2025/others/citeeval_principle-driven_citation_evaluation_for_source_attribution.md)
- [\[ACL 2025\] Is Linguistically-Motivated Data Augmentation Worth It?](../../ACL2025/others/is_linguistically-motivated_data_augmentation_worth_it.md)
- [\[ACL 2025\] Using Source-Side Confidence Estimation for Reliable Translation into Unfamiliar Languages](../../ACL2025/others/using_source-side_confidence_estimation_for_reliable_translation_into_unfamiliar.md)
- [\[ACL 2025\] If Attention Serves as a Cognitive Model of Human Memory Retrieval, What is the Plausible Memory Representation?](../../ACL2025/others/if_attention_serves_as_a_cognitive_model_of_human_memory_retrieval_what_is_the_p.md)

</div>

<!-- RELATED:END -->
