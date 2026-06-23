---
title: >-
  [Paper Note] Preregistration for Experiments with AI Agents
description: >-
  [ICML 2026][LLM (Other)][Paper Note] This is a position paper advocating for the extension of preregistration practices—used in social sciences to combat the "reproducibility crisis"—to behavioral experiments where LLMs/AI agents serve as experimental subjects. It systematically catalogs the unique "researcher degrees of freedom" in AI agent experiments a
tags:
  - ICML 2026
  - LLM (Other)
date: 2026-05-08
content_hash: 2d286183646425b5
---
# Preregistration for Experiments with AI Agents

**Conference**: ICML2026  
**arXiv**: [2606.11217](https://arxiv.org/abs/2606.11217)  
**Code**: To be confirmed  
**Area**: NLP Understanding / Research Methodology  
**Keywords**: Preregistration, AI Agent Experiments, Researcher Degrees of Freedom, Reproducibility, Position Paper

## TL;DR
This is a position paper advocating for the extension of preregistration practices—used in social sciences to combat the "reproducibility crisis"—to behavioral experiments where LLMs/AI agents serve as experimental subjects. It systematically catalogs the unique "researcher degrees of freedom" in AI agent experiments and provides a tailored preregistration template for such studies.

## TL;DR Supplement
The single author (Michelle Vaccaro) presents a sharp core argument: the combination of "high flexibility + extremely low marginal cost" in AI agent experiments makes specification searches across $\text{prompt}\times\text{model}\times\text{temperature}\times\text{seed}\times\text{parsing}$ easy to perform yet difficult to detect. Consequently, preregistration is more necessary here than in human subject experiments to reintroduce "friction."

## Background & Motivation
**Background**: Increasingly, ML research treats LLMs as "participants" in behavioral experiments, involving them in economic games, cognitive tasks, moral dilemmas, and social scenarios to characterize their reasoning, bias, and alignment attributes (i.e., "in silico" behavioral experiments). Such experiments are more time- and cost-efficient than human studies and often yield response patterns similar to humans. As AI agents begin to negotiate, manage investments, and perform content moderation, understanding their behavior itself has become a research goal.

**Limitations of Prior Work**: Over the past two decades, social and behavioral sciences have been embroiled in a "reproducibility crisis." A large-scale replication project by the Open Science Collaboration in 2015 found that while 97% of original studies reported significant results, only 36% remained significant upon replication, with effect sizes approximately half of the original. The culprit is not blatant fraud, but "researcher degrees of freedom": flexible and often undisclosed choices regarding data collection, analysis, and reporting that can "manufacture" significance even when evidence is weak.

**Key Challenge**: AI agent experiments not only inherit these vulnerabilities but amplify them. Prompt phrasing, model selection, decoding parameters, retry strategies, and response parsing constitute a high-dimensional and high-consequence choice space. Researchers (intentionally or unintentionally) can navigate through these choices until the desired result appears. The decisive difference lies in the "cost-flexibility" trade-off: human experiments are flexible but face natural friction from recruitment, compensation, IRB, and data collection; traditional ML benchmarks are inexpensive but constrained by fixed test sets. Only AI agent experiments fall into the dangerous quadrant of "high flexibility + marginal costs of only seconds or cents," where specification searches can become both routine and nearly invisible.

**Goal**: Introduce preregistration into AI agent experiments to bake credibility into this new paradigm "before the crisis erupts," rather than as a post-hoc remedy.

**Key Insight**: Borrowing Gelman & Loken’s "garden of forking paths" metaphor—a research question can fork into a combinatorially explosive specification space. Preregistration ensures that confirmatory research follows only the path (or small cluster of paths) committed to in advance, whereas daily iterations often secretly traverse multiple paths while reporting only one.

**Core Idea**: Preregistration acts as a commitment device for "researcher degrees of freedom." It is not intended to eliminate flexibility but to make it visible, allowing readers to calibrate their confidence in the conclusions accordingly.

## Method
As a methodological position paper, the "Method" comprises the arguments and tools constructed by the author: first, a taxonomy to expose the degrees of freedom in AI agent experiments; second, a preregistration template with fields tailored for each threat; and finally, implementation recommendations for three classes of stakeholders.

### Overall Architecture
The argument proceeds along a clear chain: ① reviewing the reproducibility crisis in social sciences and why preregistration is effective; ② systematically cataloging researcher degrees of freedom in AI agent experiments (the taxonomy in Table 1) and mapping them to the p-hacking dynamics that drove preregistration initially; ③ providing a tailored preregistration template for AI agents, targeting specific threats section by section; ④ offering recommendations for researchers, conferences/journals, and funding agencies. Central to the text is the "cost-flexibility" chart: because AI agent experiments fall into the quadrant most susceptible to erosion by specification search due to low costs and high flexibility, they require preregistration the most to reintroduce friction.

### Key Designs

**1. Taxonomy of Researcher Degrees of Freedom: Mapping the "Garden of Forking Paths" in an AI Context**

The author's first step is to concretize vague "flexibility" into a classification table (Table 1) covering the entire experimental pipeline: model selection, prompt engineering, sampling parameters, experimental design, response processing, analysis, and reporting. The paper emphasizes three commonalities of these degrees of freedom: they are volatile, high-consequence for results, and lack principled defaults. Minor prompt perturbations can cause large differences in downstream output, effectively making "prompt phrasing" a high-dimensional treatment manipulation. Random controls such as temperature, top-$p$, and seeds can alter response content and refusal behavior, tempting researchers to "rerun/filter/stabilize" outputs until expectations are met. Inference budgets (tokens, rounds, tool calls) are also not neutral implementation details; they change agent strategies and serve as often-overlooked latent moderators. These factors interact multiplicatively—$\text{prompt}\times\text{model}\times\text{decoding}\times\text{retries}\times\text{parsing}\times\text{metric}$—easily generating thousands of plausible specifications, where papers typically report only one path.

**2. Core Fields of the Preregistration Template: Hedging Degrees of Freedom with "Advance Commitment"**

The template logic extends traditional preregistration (pre-committing to hypotheses, methods, and analysis) with structured fields for AI-specific freedoms, providing targeted remedies: ① Full specification of the computational environment—exact model identifiers and version checkpoints (e.g., `gpt-4-0125-preview` instead of a generic "GPT-4"), generation parameters (temperature, top-$p$, top-$k$, seed), and inference budgets (max tokens, timeouts, retry limits). API models must include API versions and access dates; open-source weights require checkpoint hashes and quantization schemes. ② Verbatim recording of the full prompt text, including system messages, user instructions, and few-shot examples, as subtle formatting or phrasing changes lead to behavioral drift. ③ Operationalizing the "confirmatory vs. exploratory" distinction—researchers declare which analyses are hypothesis testing versus hypothesis generating, pre-specifying primary outcome variables, statistical tests, and decision rules (significance thresholds, minimum effect sizes) for the former. For studies using LLMs as scorers, the evaluation prompt, evaluator model version, and handling of evaluator disagreement/refusal must be locked in (addressing systematic bias, position effects, and version sensitivity in LLM judges). ④ Pre-specifying exclusion criteria for malformed, refused, or filtered responses, as refusal handling (exclusion, imputation, retry, or separate categorization) is highly susceptible to post-hoc "optimization."

**3. Advance Commitment for Robustness and Multiverse + Staged Adaptive Design**

Addressing the issue of "robustness checks" often becoming selective post-hoc reporting, the template includes specific clauses. The core principle: the specification space intended for exploration must be defined beforehand, and results for the entire space must be reported. If testing replication across prompt variants, all variants must be listed in advance and reported (including null or contradictory results); if testing generalization across models, all models must be specified and reported individually, rather than just "the ones that worked." A strong contrast is highlighted: "pre-committing to test 3 models and reporting all 3" provides much stronger evidence than "testing 6 and reporting 3." Simultaneously, acknowledging that some research naturally requires sequential decision-making (e.g., using pilot data to calibrate difficulty), the author draws from "registered reports" to offer a "staged preregistration" option. This allows for initial designs with explicit decision rules, such as "if the pilot shows a floor effect (accuracy $<20\%$), increase context by 500 tokens; if a ceiling effect ($>90\%$), switch to a harder problem." The key constraint is that these rules must be fixed beforehand rather than invented post-hoc. For truly exploratory work, the template imposes no constraints other than requiring it to be clearly labeled as exploratory and separated from confirmatory claims—the goal is to "make flexibility visible" rather than eliminate it.

**4. Transparency Commitment + Statement of Non-commencement: A Gate Against "Post-hoc Registration"**

Finally, the template embeds transparency commitments for cumulative science: declarations on whether raw model outputs, processed data, and analysis code are shared and where. Archiving full input-output logs is encouraged for proprietary models/APIs (as behavior for the same version number can change over time). Fields are provided for links to code repositories, data archives, and supplementary materials to form a complete audit chain from preregistration to the final manuscript. A critical component is the "attestation" clause, requiring confirmation that data collection has not yet started. This directly addresses the pain point that the ease of rerunning AI experiments makes post-hoc registration tempting; while it cannot guarantee compliance, it raises the reputational cost of betrayal and signals the researcher's intent to conduct a genuine confirmatory test.

### An Example: Anchoring Effect Simulations Reveal an "Illusion of Robustness"
To illustrate "specification-driven variability," the author conducted a simulation examining the anchoring effect in LLMs across **2,430 experimental specifications**, varying model families, system prompts, anchor distances, delivery methods, question content, and outlier handling. The resulting "specification curve" showed the anchoring index distributed across the entire spectrum from strongly negative to strongly positive. Researchers could report "LLMs exhibit robust human-like anchoring," "no anchoring at all," or even "reverse anchoring" simply by choosing which path to report. More insidiously, one could write: "Across three different families and architectures, we consistently found that LLMs exhibit human-like anchoring bias"—a sentence that sounds like a robustness check but is actually cherry-picked from a larger space. This is the "illusion of robustness": a finding that appears to generalize across models and conditions, yet its surface universality reflects the researcher's navigation of the specification space rather than a stable attribute of the phenomenon itself.

## Key Experimental Results
As this is a methodological position paper, it lacks "SOTA-chasing" experiments. Its empirical support consists of the "cost-flexibility" framework, the taxonomy of freedoms, the anchoring simulation, and point-by-point rebuttals of alternative perspectives. The core arguments are summarized below.

### Comparison of "Cost-Flexibility" Across Three Research Paradigms (Core of Figure 2)

| Paradigm | Marginal Cost | Specification Flexibility | Specification Search Risk |
| :--- | :--- | :--- | :--- |
| Human Behavioral Experiments | High (Recruitment/Comp/IRB/Data) | High | Naturally inhibited by cost friction |
| Traditional ML Benchmarks | Low | Low (Fixed sets/Standard metrics) | Constrained by specification space |
| AI Agent Experiments | Extremely Low (Seconds/Cents per API call) | High (Prompt/Model/Decoding/Retry/Parsing) | Highest—High flexibility plus low friction |

### Taxonomy of Researcher Degrees of Freedom (Skeleton of Table 1)

| Pipeline Stage | Typical Degrees of Freedom | Why Dangerous |
| :--- | :--- | :--- |
| Model Selection | Family, version, checkpoint | Choice of model can become a result-dependent decision |
| Prompt Engineering | Phrasing, system prompt, few-shot | Tiny perturbations = high-dimensional treatment manipulation |
| Sampling Parameters | temperature, top-$p$, seed, retries | Can be rerun until output "aligns with expectations" |
| Inference Budget | Tokens, rounds, tool calls | Latent moderator; changes agent strategies |
| Response Processing | Parsing rules, refusal/exclusion criteria | Post-hoc "optimization" of exclusion rules |
| Analysis/Reporting | Metrics, statistical tests, path selection | Reporting only one path in a multiverse |

### Response to Six Alternative Perspectives (Section 6)

| Alternative View | Author's Rebuttal Points |
| :--- | :--- |
| 1. AI experiments are too cheap; preregistration is meaningless | Low marginal costs and invisible searches make preregistration's "friction" more necessary. |
| 2. Preregistration can be gamed (explore then register) | Disclosure of pilot history creates an auditable record, making deception harder to hide. |
| 3. Preregistration biases toward NHST | It can register estimators, loss functions, metrics, and thresholds—not just NHST. |
| 4. Mandatory preregistration stifles serendipity | It doesn't ban exploration; it just requires labeling it as exploratory. |
| 5. Any single specification is arbitrary | One can preregister the "multiverse" itself—a set of models/prompts + aggregation rules. |
| 6. Open source code is the correct trust mechanism | Open source solves reproducibility but doesn't expose the prior "forking paths"; preregistration solves credibility. |

### Key Findings
- **Reproducibility $\neq$ Credibility**: Open sourcing allows "the same pipeline to get the same result" (reproducibility), but it does not reveal the prompts, models, temperatures, or parsing logic explored before the final version. Preregistration fills the gap from reproducibility to credibility.
- **Preregistration Inverts the Detection Burden**: Open sourcing places the burden of detecting alternative specifications on reviewers and researchers who may not have the time or resources to investigate. Preregistration requires the original researcher to make specification choices transparent from the start.
- **The 2,430-specification anchoring curve** is the most powerful concrete evidence—showing that the same phenomenon can be "honestly" reported as positive, negative, or zero conclusions, confirming the dangers of specification search.

## Highlights & Insights
- **The "Cost-Flexibility" 2D chart is an excellent positioning tool**: By mapping marginal cost against defensible "forking paths," it clearly explains why AI agent experiments are more dangerous than human experiments or traditional benchmarks. This framework is transferable to any discussion of evaluation methodology.
- **Inverting "low cost" from a "reason against preregistration" to a "reason for it"**: Directly rebutting common skepticism (e.g., from Horton et al.) that preregistration is useless when experiments cost \$1 and 30 seconds.
- **The requirement to disclose pilot history is highly practical**: Requiring documentation of how many prompt variants were tried and which models were tested before arriving at the final hypothesis turns invisible search into an auditable record—a simple, adoptable change.
- **Staged preregistration balances rigor with reality**: Borrowing from registered reports to allow sequential designs with explicit rules avoids the common "preregistration = rigidity" objection, making it more feasible.

## Limitations & Future Work
- **No empirical test of preregistration's actual effect in AI experiments**: The paper relies on normative arguments and illustrative simulations, lacking evidence that false positives actually decrease after adoption.
- **Questions regarding execution cost and compliance**: The author acknowledges that verbatim recording and locking all specifications increases the upfront burden, and attestations cannot stop a determined liar—they only raise the reputational stakes.
- **Feasibility of multiverse preregistration**: Requiring the listing and reporting of all prompt/model variants in a space of thousands of specifications is nearly impossible; specific execution details for specification curves/hierarchical summaries are not deeply discussed.
- **Incentive structures remain unchanged**: Under "novelty-oriented" publication incentives, whether a template and advocacy can change community behavior depends on whether conferences, journals, and funders enforce these standards.

## Related Work & Insights
- **vs. Social Science Preregistration (Simmons et al., Nosek et al., OSF/AsPredicted)**: Directly imports their core ideas (advance commitment, distinguishing confirmatory/exploratory) but adds structured fields for AI-specific freedoms (prompts, models, decoding, parsing).
- **vs. ML Reproducibility Norms (Held-out sets, reproducibility checklists, Pineau et al.)**: Those target "overfitting evaluation data," but lack corresponding safeguards when the "experiment itself is the evaluation" and prompts are customized research tools; this paper fills that methodological void.
- **vs. Multiverse Analysis (Steegen et al.) / Specification Curves (Simonsohn et al.)**: Adopts the idea of reporting the full specification space rather than a single privileged path and argues that the multiverse itself can be preregistered.

## Rating
- Novelty: ⭐⭐⭐⭐ Adapts mature preregistration concepts precisely to the new freedoms of AI agent experiments; timely and clearly positioned.
- Experimental Thoroughness: ⭐⭐⭐ Uses only an anchoring simulation for illustration; sufficient for a position paper but not exhaustive validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Arguments advance logically; the "cost-flexibility" framework and rebuttals are particularly well-constructed.
- Value: ⭐⭐⭐⭐ High methodological value for the community, proposing to "build in credibility" at the onset of the AI agent behavioral experiment boom.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Speculative Actions: A Lossless Framework for Faster AI Agents](../../ICLR2026/llm_nlp/speculative_actions_faster_ai_agents.md)
- [\[AAAI 2026\] Whispering Agents: An Event-Driven Covert Communication Protocol for the Internet of Agents](../../AAAI2026/llm_nlp/whispering_agents_an_event-driven_covert_communication_protocol_for_the_internet.md)
- [\[ICML 2026\] Position: The ML Community Must Build an AI-Augmented Peer-Review Ecosystem](position_the_ml_community_must_build_an_ai-augmented_peer-review_ecosystem.md)
- [\[AAAI 2026\] Blue Teaming Function-Calling Agents](../../AAAI2026/llm_nlp/blue_teaming_function-calling_agents.md)
- [\[ACL 2026\] Big AI is Accelerating the Metacrisis: What Can We Do?](../../ACL2026/llm_nlp/big_ai_is_accelerating_the_metacrisis_what_can_we_do.md)

</div>

<!-- RELATED:END -->
