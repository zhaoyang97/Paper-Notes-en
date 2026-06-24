---
title: >-
  [Paper Note] Position: Neglecting the Sustainability of AI is Fuelling a Global AI Arms Race
description: >-
  [ICML 2026][Recommender Systems][Sustainable AI] Utilizing Karl Marx's "base-superstructure" framework, this position paper argues that current "sustainable AI" discussions are dominated by environmental dimensions while neglecting economic and social ones. It calls for the simultaneous elevation of both **climate awareness** and **resource awareness** axes and proposes the CARAML five-layer action framework (Individual / Community / Industry / Government / Global) to curb th…
tags:
  - "ICML 2026"
  - "Recommender Systems"
  - "Sustainable AI"
  - "Climate Awareness"
  - "Resource Awareness"
  - "AI arms race"
  - "CARAML"
date: 2026-05-08
content_hash: e2c32024a0b50ee2
---

# Position: Neglecting the Sustainability of AI is Fuelling a Global AI Arms Race

**Conference**: ICML 2026  
**arXiv**: [2502.20016](https://arxiv.org/abs/2502.20016)  
**Code**: None  
**Area**: AI Safety / Sustainable AI / AI Governance  
**Keywords**: Sustainable AI, Climate Awareness, Resource Awareness, AI arms race, CARAML

## TL;DR
Utilizing Karl Marx's "base-superstructure" framework, this position paper argues that current "sustainable AI" discussions are dominated by environmental dimensions while neglecting economic and social ones. It calls for the simultaneous elevation of both **climate awareness** and **resource awareness** axes and proposes the CARAML five-layer action framework (Individual / Community / Industry / Government / Global) to curb the escalating "global AI arms race."

## Background & Motivation

**Background**: Sustainability has historically consisted of three pillars: environmental, economic, and social. However, discussions on "sustainable AI" almost exclusively focus on the environment (carbon emissions, water consumption, e-waste), leaving "democratization / accessibility" issues to a separate group of researchers, with minimal dialogue between the two.

**Limitations of Prior Work**: (i) While pushing for green datacenters, there are demands for Low- and Middle-Income Countries (LMICs) to use low-carbon energy despite their lack of basic electricity, exacerbating a "double penalty." (ii) Efficiency improvements like those from DeepSeek reduce GPU time by an order of magnitude, but according to the Jevons paradox, this triggers a rebound in usage (e.g., "reasoning models" exploding in token volume), leading to increased total energy consumption. (iii) Top500 supercomputers and AI investments are almost entirely concentrated in the US/CN/EU; in 2024, US private AI investment alone reached $109.1 billion—exceeding the GDP of 134 countries. The slogan of "AI Sovereignty" remains an empty promise for nations lacking a material foundation.

**Key Challenge**: Compressing "sustainable AI" solely into "environmental sustainability" creates structural tension with "social sustainability"—pursuing climate awareness alone excludes resource-poor nations, while pursuing resource awareness (increasing accessibility) drives compute expansion and rebound emissions.

**Goal**: (i) Redefine sustainable AI as being high on both "climate awareness" and "resource awareness" axes; (ii) use a historical materialism base-superstructure model to explain why the current "GPU-rich vs. GPU-poor" divide is reproduced by existing systems; (iii) provide the CARAML framework to distribute action obligations across five levels along the "agency × scope" axes.

**Key Insight**: The authors treat AI as **embodied infrastructure** in the sense of Crawford (2021)—it is not an abstract algorithm but a physical aggregation of minerals, labor, electricity, and capital. The material base determines the superstructure of policy and cultural narrative; without changing the base, the narrative cannot change.

**Core Idea**: Sustainable AI = the quadrant where both **climate-aware × resource-aware** axes are high; any solution that elevates only one axis (green AI / inclusive AI / SOTA AI / edge AI) collapses back into an arms race.

## Method

As a position paper, there are no models or training; the "method" consists of a three-layered progressive argument corresponding to three key designs: first, a 2D coordinate system formalizes "what counts as sustainable AI" and proves through counter-examples (low-carbon datacenters, efficiency rebounds) that pursuing only one axis leads to failure (Design 1); second, a base-superstructure model explains why this imbalance is self-reproduced by existing systems and cannot be resolved by individual effort (Design 2); finally, the CARAML five-layer action framework translates "elevating both axes" into specific levers from the individual to the global level (Design 3).

### Overall Architecture

Section 2 decomposes sustainable AI into environmental, economic, and social pillars, introduces the "climate awareness vs. resource awareness" axes, and uses two counter-examples (low-carbon datacenters, efficiency rebounds) to prove that pursuing only one axis undermines the whole—these steps form the first key design (2D four-quadrant). Section 3 uses a base-superstructure model to explain why the GPU-rich vs. GPU-poor gap is self-sustaining (Second Key Design). Section 4 presents the CARAML action framework (Third Key Design). Section 5 addresses opposing views such as "emissions will automatically plateau" and "distilled models have solved the problem" (see Key Findings and Related Work below).

### Key Designs

**1. Climate Awareness × Resource Awareness 2D Quadrant: Forcing every solution to declare its position**

The authors place the vague term "sustainable AI" into a categorizable, debatable coordinate system: the horizontal axis is resource awareness (decolonization, participability, opposition to resource concentration), and the vertical axis is climate awareness (carbon/water/e-waste consciousness). SOTA AI is low on both; Green AI is high on climate but low on resource; Inclusive AI is the reverse. Only the top-right quadrant, high on both, constitutes sustainable AI—while Edge AI is marked with a question mark due to difficult-to-define embodied carbon. This diagram explicates the latent friction between "democratization" and "green AI" factions: pursuing one axis inevitably leads to trade-offs, shifting the debate from "we all want sustainability" to "which quadrant do you intend to occupy."

**2. Base-Superstructure Model explaining AI system reproduction: Why individual efforts are consumed by structures**

Why do "individual efforts + technical efficiency" fail to automatically dissolve the AI arms race? Using historical materialism, the authors define the AI "base" as compute + data + capital + labor + knowledge commodities, while the "superstructure" consists of AI policies, regulations, research norms, and media narratives. The two form a positive feedback loop: material monopolies (NVIDIA + a few hyperscalers) shape superstructure discourses like AI sovereignty, export controls, and ethical guidelines, which in turn solidify the existing material order. The paper uses a hypothetical scenario: export controls justify compute restrictions via "national security" narratives; the restricted parties are forced to rebuild semiconductor supplies, merely creating a new form of dependency within existing relationships. This argument draws a direct analogy to colonial expansion (colonial capital and racial hierarchies as base and superstructure), suggesting contemporary AI is repeating this colonial structure in the GPU era.

**3. CARAML Action Framework (agency × scope): Translating the dual-axis goal into specific levers**

The objective is to pull both climate and resource axes simultaneously, but single-layer action is insufficient—individual agency is strong but scope is small; national and global levels are the opposite. CARAML uses an "agency × scope" diagram to split actions into five levels, providing at least one actionable item per level: individuals use perf-per-resource instead of just accuracy; the ML community implements large-scale experiment pre-registration to eliminate waste; industry imposes self-mandated carbon caps; governments integrate AI impact assessments into regulation (analogous to EIA); globally, "Right to Compute" is redefined as "Right to Sustainable Compute" via open-source/shared compute to combat the digital divide. CARAML also serves as a vehicle for empirical evidence—open-source models like LLaMA / BLOOM / Mistral are used to demonstrate which layers are active and which are absent.

### Loss & Training

Position paper, no model/loss. The evidence base includes TOP500 (2025) compute distribution, Maslej et al. (2025) investment data, ICML/ICLR/NeurIPS 2006–2025 author country distribution, IRENA and IEA electricity/renewable data, and a recalculable estimate for LLaMA-3.1 training: 30.84M GPU-hours $\Rightarrow$ 21.5 GWh / 8930 tCO2e (US grid).

## Key Experimental Results

### Main Results: Concentration of Global AI Compute / Investment / Academic Participation

| Dimension | Data | Source |
|------|------|------|
| Top500 Supercomputer Georgraphy | Primarily US/CN/EU; Africa almost absent | TOP500 (2025) |
| 2024 US Private AI Investment | $109.1B (≈12× China, ≈24× UK) | Maslej et al. (2025) |
| This Investment vs. Country GDP | Exceeds 2024 GDP of 134 countries worldwide | Ibid |
| LLM Leaderboard Institutional Makeup | Majority of entries from corporations, not academia | Moutawwakil & Pierrard (2023) |
| Compute Doubling Period | Since 2012, some frontier systems double every 3.4 months | Sevilla et al. (2022) |
| ICML/ICLR/NeurIPS Authors (2006–2025) | Significant underrepresentation of LMICs, widening gap | World Bank (2024) + Ours |

### Case Study: LLaMA-3.1 (405B) single training cost variation by geography

| Training Location | GPU Hours | Estimated Energy | Estimated Carbon |
|----------|----------|----------|----------|
| Any | 30.84M H100-h | 21.5 GWh | — |
| US Grid | Same | 21.5 GWh | 8,930 tCO2e |
| Sweden Grid | Same | 21.5 GWh | 750 tCO2e |
| India Grid | Same | 21.5 GWh | 14,737 tCO2e |

### Key Findings
- **DeepSeek Rebound**: DeepSeek-V3 used only 2.8M H800-h, an order of magnitude less than LLaMA-3-405B's 30.8M H100-h, but its popularity surged 2800% post-release and catalyzed "reasoning models"—where an 80-token response requires ~800 latent thinking tokens; efficiency gains are completely swallowed by usage growth.
- **Low-carbon datacenter paradox**: LMICs with the weakest clean power supply are the ones required to switch to renewables, forced to accept a "double penalty."
- **Carbon credits / carbon-neutral branding failure**: Probst et al. (2024) show that carbon crediting projects rarely deliver real reductions; "carbon-neutral training" claims are untenable.
- **Distilled models do not solve the problem**: Behind the low inference costs of QLoRA / Phi / small LLaMA series, the embodied carbon of the large models is amortized; manufacturing remains concentrated among a few actors, leaving the compute gap intact.
- **IEA Projection**: By 2030, AI could consume 945 TWh, roughly equal to Japan's annual electricity consumption; at current average carbon intensity, this is 447M tCO2e, nearly half of global commercial aviation emissions in 2023.

## Highlights & Insights
- The **2D Quadrant** brings the hidden conflict between "democratization" and "climate" factions to the surface, serving effectively as a review tool.
- **Integrating NVIDIA compute monopoly, export controls, and AI sovereignty narratives into a single base-superstructure explanation** provide the first clear theoretical grounding for why individual efforts are often neutralized by structures.
- The **CARAML "agency × scope" framework** provides a cross-domain transferable template—decomposing any governance initiative into five layers helps avoid empty calls for "global collaboration."
- Explicitly addressing the **rebound effect** as an opposing view acknowledges that efficiency gains do not equal net emission reductions, a level of honesty relatively rare in sustainability literature.

## Limitations & Future Work
- Lacks reproducible code/database; CARAML offers qualitative suggestions without identifying which actions have the highest emission reduction elasticity.
- The base-superstructure framework is a macro-narrative; it offers limited strategic guidance for how a single researcher can change the superstructure.
- Economic calculations are largely from third-party reports (Maslej, IEA, IRENA) without sensitivity analysis.
- Sharp critique of AI sovereignty, but the answer to "should LMICs build their own compute" is insufficiently concrete.

## Related Work & Insights
- **vs Van Wynsberghe (2021)**: Accepts the "environmental / economic / social" definition but operationalizes it into a debatable 2D quadrant.
- **vs Schwartz et al. (2020) Green AI**: Notes that Green AI falls in the top-left quadrant (high climate, low resource) and cannot be equated with sustainable AI.
- **vs Wright et al. (2025)**: Similarly emphasizes environmental $\neq$ sustainable; this paper advances that observation into a full dual-axis framework + CARAML scheme.
- **vs Patterson et al. (2022) (Counter-point)**: Argues AI emissions will naturally plateau; this paper refutes this using Google’s 2023 emissions (+13% YoY / +48% vs 2019 baseline).
- **vs Ahmed & Wahed (2020)**: Characterizes de-democratization via "compute divide"; this paper upgrades this to a base-superstructure explanation and CARAML intervention.

## Rating
- Novelty: ⭐⭐⭐⭐ The 2D quadrant + base-superstructure + CARAML triad is a rare integration in the AI governance field.
- Experimental Thoroughness: ⭐⭐⭐ Evidence is mostly third-party; no independent experiments, but cases (DeepSeek rebound, LLaMA carbon) are impactful.
- Writing Quality: ⭐⭐⭐⭐ Embedding a Marxian framework into an ICML paper while maintaining readability is impressive; broad literature coverage.
- Value: ⭐⭐⭐⭐⭐ Expands "sustainable AI" from a single environmental axis to a socio-environmental dual axis, providing discourse tools directly usable for funding and peer review.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Stop Preaching and Start Practising Data Frugality for Responsible Development of AI](position_stop_preaching_and_start_practising_data_frugality_for_responsible_deve.md)
- [\[ICML 2025\] Position: The Right to AI](../../ICML2025/recommender/the_right_to_ai.md)
- [\[NeurIPS 2025\] Position: Towards Bidirectional Human-AI Alignment](../../NeurIPS2025/recommender/position_towards_bidirectional_human-ai_alignment.md)
- [\[AAAI 2026\] Moral Change or Noise? On Problems of Aligning AI With Temporally Unstable Human Feedback](../../AAAI2026/recommender/moral_change_or_noise_on_problems_of_aligning_ai_with_temporally_unstable_human_.md)
- [\[ICLR 2026\] ProPerSim: Developing Proactive and Personalized AI Assistants through User-Assistant Simulation](../../ICLR2026/recommender/propersim_developing_proactive_and_personalized_ai_assistants_through_user-assis.md)

</div>

<!-- RELATED:END -->
