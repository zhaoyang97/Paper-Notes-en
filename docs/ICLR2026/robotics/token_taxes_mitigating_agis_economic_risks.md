---
title: >-
  [Paper Note] Token Taxes: Mitigating AGI's Economic Risks
description: >-
  [ICLR 2026][Robotics][AGI governance] This paper proposes the Token Tax — a surcharge levied on model inference token usage — as a first-line governance instrument for mitigating economic risks in the post-AGI era. It leverages cloud computing providers as intermediaries through a three-stage audit pipeline (black-box token verification → norm-based tax rates → white-box audit). Compared to conventional robot taxes, it offers two distinctive advantages: enforceability through existing compute governance infrastructure, and collection at the point of AI token consumption rather than model hosting location, thereby alleviating global inequality.
tags:
  - ICLR 2026
  - Robotics
  - AGI governance
  - token tax
  - robot tax
  - compute governance
  - economic risks
  - AI safety
date: 2026-05-08
content_hash: 9f92c430ba86b9f1
---

# Token Taxes: Mitigating AGI's Economic Risks

**Conference**: ICLR 2026
**arXiv**: [2603.04555](https://arxiv.org/abs/2603.04555)
**Code**: None
**Area**: AI Governance / Computational Economics
**Keywords**: AGI governance, token tax, robot tax, compute governance, economic risks, AI safety

## TL;DR

This paper proposes the Token Tax — a surcharge levied on model inference token usage — as a first-line governance instrument for mitigating economic risks in the post-AGI era. It leverages cloud computing providers as intermediaries through a three-stage audit pipeline (black-box token verification → norm-based tax rates → white-box audit). Compared to conventional robot taxes, it offers two distinctive advantages: enforceability through existing compute governance infrastructure, and collection at the point of AI token consumption rather than model hosting location, thereby alleviating global inequality.

## Background & Motivation

**Background**: AI safety research has long focused on capability risks (superintelligence, alignment), leaving economic risks from AGI comparatively understudied. Nevertheless, early evidence already shows a 16% increase in unemployment among early-career roles exposed to AI (Brynjolfsson et al., 2025). Historically, the "Engels Pause" (1790–1830) during the First Industrial Revolution saw real wages stagnate for 40 years — AGI may trigger even more severe economic disruption.

**Limitations of Prior Work**: AGI development threatens the economy along three dimensions: (1) **Fiscal crisis for governments** — AI substitution of labor erodes the largest tax base (labor taxes) while rising unemployment increases expenditures, weakening public finances on both sides; (2) **Gradual Disempowerment of citizens** — as AI becomes the engine of economic growth, governments lose the incentive to serve citizens, analogous to the "resource curse" in states like Venezuela, where revenues from oil rather than labor correlate with widespread citizen poverty; (3) **Worsening global inequality** — frontier AI chips and model companies are highly concentrated in the "Compute North" (the US, China, and a handful of developed nations), forcing the "Compute South" (developing countries) to rent compute resources, leaving AGI economic gains deeply unequal.

**Key Challenge**: AGI will substitute human labor across all economic sectors — unlike prior automation of narrow tasks, cross-sector AGI substitution implies a fundamental collapse of labor-based tax systems and social contracts. Existing robot tax proposals (corporate taxes, automation taxes, elimination of investment deductions) are firm-based, difficult to enforce across jurisdictions, and offer no global benefit.

**Goal**: To design an enforceable, usage-based taxation instrument for the AGI era that can also alleviate global inequality.

**Key Insight**: Existing LLM APIs are already billed per token — a Token Tax as a percentage surcharge on billing is naturally feasible both technically and commercially. Cloud providers already possess the capacity to collect compute metadata and oversee AI workloads (as seen in compute-threshold regulations under the EU AI Act and Biden Executive Order 14110), and can directly serve as tax intermediaries.

**Core Idea**: Levy a usage tax at the point of sale of model inference tokens — enforceable through existing compute infrastructure and collected at the point of consumption rather than hosting, enabling global equity.

## Method

### Overall Architecture

Token Tax is defined as a usage-based surcharge applied to model inference tokens, collected at the point of sale. Implementation: a percentage tax is added on top of provider-billed token costs (e.g., a 10% tax rate × \$1/token = \$0.10 token tax). Core architecture: cloud computing providers serve as intermediaries between AI model providers and governments, operating inference, collecting token billing data, determining tax amounts through a three-stage audit pipeline, and reporting to governments.

### Key Design 1: Three-Stage Audit Pipeline

- **Function**: Addresses the core enforcement problem of AI companies underreporting token counts to inflate profits.
- **Mechanism**: Three progressively escalating stages — **Stage 1 (Black-Box Token Audit)**: Cloud providers are required to collect token-level usage data and act as independent verifiers, cross-checking against token counts self-reported by AI companies; **Stage 2 (Norm-Based Tax Rate)**: Drawing on Norway's petroleum "Norm Tax" concept — if a company underreports, auditors may apply a standard tax rate based on average token usage per model category, requiring only black-box access; **Stage 3 (White-Box Audit)**: Legal requirements compel companies to share inference process information with third-party auditors, eliminating any remaining room for gaming.
- **Design Motivation**: Currently, only model providers have access to the full generation process; auditors can only observe outputs. This information asymmetry makes underreporting of token counts (e.g., hiding chain-of-thought tokens) profitable. The three-stage design progressively escalates from low-cost black-box verification to high-confidence white-box audit, balancing enforcement cost against accuracy.

### Key Design 2: Point-of-Use Collection to Alleviate Global Inequality

- **Function**: Ensures that AI-consuming nations (including Compute South developing countries) also benefit from AI tax revenues.
- **Mechanism**: Conventional corporate taxes are collected at the company's country of incorporation; FLOP taxes are collected at the location of compute facilities — both favor the Compute North. The Token Tax is collected at the point where tokens are consumed (point of sale), enabling any country using AI services to receive the corresponding tax revenue. This is analogous to the destination-based collection principle of value-added tax (VAT).
- **Design Motivation**: The AI supply chain (chips → training → inference → API) is highly concentrated in a small number of countries. If taxation occurs only on the supply side, the Compute South will be entirely excluded from the economic benefits of AGI.

### Key Design 3: Integration with Existing Compute Governance Infrastructure

- **Function**: Leverages existing regulatory frameworks and technical capabilities to achieve enforceability.
- **Mechanism**: The EU AI Act and Biden Executive Order 14110 have already introduced compute-based regulatory thresholds; hyperscale cloud providers (AWS/Azure/GCP) already collect metadata on compute consumption and workload types — extending this to token-level data collection is incremental rather than revolutionary.
- **Design Motivation**: A policy instrument's enforceability is proportional to the infrastructure it relies upon. The Token Tax can piggyback on existing compute governance infrastructure, avoiding the need to build new enforcement agencies from scratch.

### Comparison with Traditional Robot Taxes

Traditional robot tax proposals include raising corporate tax rates, levying "automation taxes," and eliminating tax deductions for corporate automation investment — all firm-based approaches. The Token Tax is usage-based, capturing value precisely at the terminal point of token consumption, independent of specific corporate form.

## Key Experimental Results

### Multidimensional Comparison: Token Tax vs. Alternative Approaches

| Dimension | Token Tax | FLOP Tax | Traditional Robot Tax (Corporate/Automation Tax) |
|-----------|-----------|----------|--------------------------------------------------|
| Tax base | Inference token usage | Floating-point operations | Corporate revenue / capital investment |
| Collection point | Point of AI consumption | Location of compute facility | Country of corporate registration |
| Global equity | ✅ Benefits Compute South | ❌ Benefits Compute North | ❌ Benefits country of incorporation |
| Enforcement mechanism | Cloud provider token logs + three-stage audit | Compute infrastructure monitoring | Corporate financial audit |
| Evasion potential | Requires three-stage audit to prevent underreporting | Relatively difficult to evade | Vulnerable to transfer pricing / base erosion |
| Compatibility with existing frameworks | ✅ API per-token billing already in place | ✅ EU AI Act compute thresholds already in place | ⚠️ Requires redefining "robot" |
| Complementarity | Can be combined with FLOP Tax | Can be combined with Token Tax | Standalone approach |

### Governance Response Mapping for Three Post-AGI Economic Risks

| Economic Risk | Risk Mechanism | Token Tax Response |
|---------------|---------------|--------------------|
| Government fiscal crisis | AI substitutes labor → labor tax base shrinks | Token tax supplements fiscal revenue, aligned with income tax rates to restore tax neutrality |
| Gradual disempowerment of citizens | Governments no longer depend on human labor → lose incentive to serve citizens | Ties government revenue to AI usage volume → maintains responsiveness to citizens |
| Worsening global inequality | Compute North monopolizes AI infrastructure | Collected at point of use → Compute South also receives AI tax revenues |

### Main Objections and Responses

| Objection | Paper's Response |
|-----------|-----------------|
| Token taxes will suppress innovation and prompt AI companies to relocate | Recommends Agent-Based Modeling (ABM) to predict market impacts; LLM-based ABM has already successfully simulated large-scale social interactions |
| FLOP tax is superior to Token tax | The two are not mutually exclusive — the optimal framework may be a hybrid approach, with FLOP tax capturing the training/compute side and Token tax capturing the consumption side |
| AI superpowers (US/China) can veto Token taxes | Drawing on GDPR/EU AI Act experience — a regional agreement among a "coalition of the willing" (e.g., the EU) is harder to veto than measures by a single country |

### Key Findings

- The distinctive value of the Token Tax lies in **simultaneously** satisfying enforceability (via existing compute governance infrastructure) and point-of-use collection (alleviating global inequality) — existing alternatives satisfy at most one of these criteria.
- Implementation experience with Digital Services Taxes (DSTs) in Europe demonstrates that regionally coordinated, consumption-based taxation schemes can be sustained even under US trade pressure.
- The emergence of hidden reasoning tokens (thinking tokens) complicates auditing (Sun et al., 2025 found that the existence of hidden tokens can be detected), making norm-based tax rates as a fallback particularly important.
- On-premise inference is the primary enforcement blind spot for the Token Tax — when enterprises run models on their own hardware, they bypass the cloud intermediary entirely.

## Highlights & Insights

- **Expanding AI safety from purely technical to economic risks**: Addresses the attention imbalance in AI governance research between "economic risks" and "capability risks" — societal instability may arrive before technical loss of control.
- **"Compute North vs. Compute South" framework**: Concisely and powerfully reveals the geo-economic landscape of the AGI era — advanced compute is highly concentrated in a small number of countries, and global inequality will be further exacerbated by AGI.
- **Inspirational power of the "rentier state" analogy**: Countries like Venezuela and Saudi Arabia that generate revenue from resources rather than labor tend to have citizens mired in poverty — once AGI substitutes labor, all nations face the risk of becoming "AI rentier states."
- **Engineering pragmatism of the three-stage audit pipeline**: Each stage has independent feasibility, progressing from low-cost black-box verification to white-box audit — no requirement for an all-or-nothing leap.
- **Complementary perspective of Token Tax + FLOP Tax**: Rather than dismissing competing proposals, the paper identifies how the two can be combined — training-side FLOP tax + inference-side Token tax = full lifecycle coverage.

## Limitations & Future Work

- **Lack of quantitative analysis**: This is a purely policy-oriented paper, providing no economic modeling, simulation experiments, or optimal tax rate estimates — the proposed ABM research remains at the suggestion stage.
- **On-premise deployment blind spot**: Enterprises running models on their own hardware entirely bypass the cloud intermediary — Token Tax enforcement depends on cloud middlemen and offers no solution for on-premise scenarios.
- **Neglect of open-source models**: Users of open-source models can run inference on any device without API call records — Token Tax coverage is essentially infeasible in this context.
- **Insufficient analysis of political feasibility**: Although GDPR precedent is cited, the economic interests and geopolitical dynamics involved in AI taxation are far more complex than data privacy (AI is directly tied to national competitiveness and military capability).
- **Ambiguity in the definition of inference**: The paper does not clearly define "token" — the concept of "token" varies greatly across model architectures (dense vs. MoE, autoregressive vs. diffusion).
- **Assumption that cloud compute centralization persists**: If edge inference or decentralized computing rises, the supervisory capacity of cloud intermediaries will be weakened.

## Related Work & Insights

- **vs. Robot Tax (Abbott & Bogenschneider, 2018)**: Traditional robot taxes are firm-based (collected at the country of corporate registration); the Token Tax is usage-based (collected at the point of consumption) — a fundamental distinction in the dimension of global equity.
- **vs. FLOP Tax**: FLOP taxes focus on the compute supply side (training and inference infrastructure); the Token Tax focuses on the consumption side (API calls) — the two are complementary rather than mutually exclusive.
- **vs. Gradual AI Disempowerment theory (Kulveit et al., 2025)**: Gradual Disempowerment describes the risk mechanism; the Token Tax provides an actionable policy response.
- **vs. Compute governance frameworks (Sastry et al., 2024)**: The Token Tax builds on existing compute governance infrastructure but focuses on the inference side rather than the training side — complementing that framework's gap on the consumption side.
- **Insight**: The "point of sale" collection approach of the Token Tax is worth generalizing — any output endpoint of AI-as-a-Service (not only tokens) could potentially serve as a taxation point.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic proposal of the Token Tax concept with a designed, enforceable three-stage audit scheme; extends AI governance from capability risks to economic risks.
- Experimental Thoroughness: ⭐⭐ A purely policy analysis paper with no quantitative experiments or simulation validation; the proposed ABM was not executed.
- Writing Quality: ⭐⭐⭐⭐ Clear argumentative structure with a complete logical chain: three risks → two advantages → three-stage audit → three objections.
- Value: ⭐⭐⭐⭐ Proposes a framework with genuine policy guidance for AI governance, though the absence of quantitative support limits its persuasiveness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Explaining and Mitigating Crosslingual Tokenizer Inequities](../../NeurIPS2025/robotics/explaining_and_mitigating_crosslingual_tokenizer_inequities.md)
- [\[AAAI 2026\] Unintended Misalignment from Agentic Fine-Tuning: Risks and Mitigation](../../AAAI2026/robotics/unintended_misalignment_from_agentic_fine-tuning_risks_and_m.md)
- [\[ACL 2026\] On Safety Risks in Experience-Driven Self-Evolving Agents](../../ACL2026/robotics/on_safety_risks_in_experience-driven_self-evolving_agents.md)
- [\[ICCV 2025\] Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning](../../ICCV2025/robotics/resolving_token-space_gradient_conflicts_token_space_manipulation_for_transforme.md)
- [\[NeurIPS 2025\] Toward Engineering AGI: Benchmarking the Engineering Design Capabilities of LLMs](../../NeurIPS2025/robotics/toward_engineering_agi_benchmarking_the_engineering_design_capabilities_of_llms.md)

</div>

<!-- RELATED:END -->
