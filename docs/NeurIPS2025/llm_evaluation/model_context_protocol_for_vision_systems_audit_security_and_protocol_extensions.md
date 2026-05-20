---
title: >-
  [Paper Note] Model Context Protocol for Vision Systems: Audit, Security, and Protocol Extensions
description: >-
  [NeurIPS 2025][LLM Evaluation][Model Context Protocol] The first protocol-level audit of MCP deployment in vision systems, analyzing 91 public MCP servers and finding that 78% exhibit schema inconsistencies and 89% lack…
tags:
  - "NeurIPS 2025"
  - "LLM Evaluation"
  - "Model Context Protocol"
  - "vision system orchestration"
  - "protocol security"
  - "schema validation"
  - "multimodal agents"
date: 2026-05-08
content_hash: aa136a5b41b36ee2
---

# Model Context Protocol for Vision Systems: Audit, Security, and Protocol Extensions

**Conference**: NeurIPS 2025
**arXiv**: [2509.22814](https://arxiv.org/abs/2509.22814)  
**Code**: Coming soon (benchmark and validator suite)  
**Area**: AI Systems / Protocol Security / Computer Vision Workflows
**Keywords**: Model Context Protocol, vision system orchestration, protocol security, schema validation, multimodal agents

## TL;DR

The first protocol-level audit of MCP deployment in vision systems, analyzing 91 public MCP servers and finding that 78% exhibit schema inconsistencies and 89% lack runtime validation; the paper further proposes protocol extensions including semantic schemas, visual memory, and runtime validators.

## Background & Motivation

Model Context Protocol (MCP) is a schema-bound execution model for agent–tool interaction that enables modular computer vision workflows through typed schemas and dynamic context objects, without requiring model retraining. However, applying MCP to vision domains introduces unique challenges:

**High-dimensional tensor inputs**: Vision data involves large-scale image streams and multimodal semantic fusion, placing significant pressure on orchestration pipelines.

**Inconsistent spatial conventions**: Different tools adopt different coordinate systems (e.g., XYWH vs. X1Y1X2Y2).

**Lack of systematic auditing**: No prior work has conducted a protocol-level, deployment-scale audit of MCP in vision systems.

**Security risks**: Dynamic and multi-agent workflows expose attack surfaces for privilege escalation and untyped tool chaining.

Existing orchestration strategies primarily rely on end-to-end model training or prompt-tuned vision-language systems, which are brittle under tool specialization and yield opaque intermediate reasoning.

## Method

### Overall Architecture

This work presents a **systematic empirical audit** rather than a novel algorithmic method. The core contributions include:
- Filtering 91 vision-relevant MCP servers from the MCPServerCorpus (13,942 publicly registered deployments)
- Annotating servers along nine compositional fidelity dimensions
- Developing an executable benchmark and validator suite to detect and classify protocol violations

### Key Designs

1. **Taxonomy of Four Orchestration Patterns**:

    - **Static composition** (37%): Fixed tool sequences; highly auditable but low adaptability
    - **Retrieval-augmented selection** (29%): Embedding-based semantic matching; flexible but 87% have undeclared coordinate formats
    - **Dynamic orchestration** (21%): Runtime construction of execution graphs; 89% lack runtime schema checks
    - **Multi-agent coordination** (13%): Distributed control; 55% exhibit stale memory or cross-tool leakage

2. **Benchmark Validator Suite**:

    - **Schema format validator**: Detects inter-tool schema inconsistencies (detection rate: 78.0%)
    - **Coordinate convention validator**: Detects missing or inconsistent spatial references (detection rate: 24.6%)
    - **Mask–image consistency validator**: Detects dimension or channel mismatches (detection rate: 17.3%)
    - **Memory scope validator**: Detects undocumented visual state retention (avg. 33.8 warnings / 100 executions)
    - **Permission verification validator**: Detects privilege escalation or leakage via tool binding (detection rate: 41.0%)

3. **Security Threat Taxonomy**: Eight primary threat vectors are identified:

    - Prompt injection, schema bypass, remote code execution (RCE), privilege escalation
    - Stale memory access, untracked provenance, cross-tool leakage, type coercion injection

4. **Protocol Extension Proposals**:

    - **Semantic grounding schema**: Adds `semantic_role`, `modality`, and `coordinate_system` fields
    - **Protocol-native visual memory**: Encodes structured, versioned, semantically annotated intermediate states
    - **Runtime validators and compatibility contracts**: Validates spatial dimensions, tensor channel semantics, and coordinate alignment at runtime
    - **Composable benchmarking**: Evaluates orchestration fidelity, memory hygiene, and schema stability

### Analysis Methodology

Compatibility is formalized via a predicate function $comp: \mathcal{T} \times \mathcal{T} \rightarrow \{0,1\}$, where $\mathcal{T}$ is the set of tool schemas, determining whether the output of one tool is admissible as the input of another. All confidence intervals are reported at the 95% level.

## Key Experimental Results

### Main Results: Protocol Failure Mode Analysis (N=91)

| Failure Type | Rate | 95% CI |
|---------|--------|--------|
| Schema format divergence | 62% | — |
| No runtime schema validation | 89% | — |
| Undeclared coordinate conventions | 87% | — |
| Out-of-band bridging scripts | 41% | — |
| Undocumented memory retention logic | 55% | — |
| Declared compositional fallback strategy | Only 9% | — |
| Schema inconsistency (composite detection) | 78.0% | [68.45, 85.28] |
| Coordinate convention inconsistency | 24.6% | [16.90, 34.36] |
| Mask–image dimension mismatch | 17.3% | [10.90, 26.35] |

### Security Audit Results (N=47)

| Security Issue | Rate | 95% CI |
|---------|--------|--------|
| Untyped tool chaining | 89.0% | [76.80, 95.19] |
| Privilege escalation or data leakage risk | 41.0% | [28.02, 55.37] |
| Memory scope warnings | Avg. 33.8 / 100 executions | [28.4, 39.9] |

### Key Findings from Case Studies

| System | Audit Scale | Primary Issues Found |
|------|---------|---------------|
| ParaView-MCP | — | Binary textures embedded in nested JSON; latency spikes >2.3s |
| SUMO+YOLO-MCP | 134 call pairs | 27.6% exhibit projection conflicts or axis mismatches |
| ALITA | 143 tool chains | 18.4% produce malformed responses |
| FHIR-MCP | 108 outputs | 14.9% of caption outputs show scaling mismatches |
| Blender-RCP | 97 multi-step compositions | 22 produce orphaned references or cache conflicts |

### Key Findings

- Segmentation outputs across 91 servers span **5 incompatible formats**: URI-encoded masks, run-length encoding, base64 tensors, polygon contours, and per-pixel label maps
- Bounding box formats are inconsistent across absolute XYWH, corner-based X1Y1X2Y2, and center-normalized representations
- Only 8 of 91 servers implement post-call output inspection
- 41% of deployments rely on undocumented bridging scripts for format conversion

## Highlights & Insights

- **First protocol-level audit of vision systems**: Systematically exposes structural problems MCP faces in the vision domain, rather than defects in individual tools
- **Rigorous quantitative analysis**: All findings are accompanied by 95% confidence intervals; the sample of 91 servers is representative for this research area
- **Actionable security threat taxonomy**: The eight threat vectors and their mitigation strategies offer practical guidance for deployment
- **Orchestration pattern taxonomy** clarifies the trade-offs among different deployment strategies

## Limitations & Future Work

- Only publicly accessible servers are analyzed, excluding enterprise-grade and proprietary deployments; research prototypes are overrepresented
- Protocol extensions are implemented only as reference prototypes on controlled testbeds, without validation in heterogeneous production environments
- Security analysis covers only 47 servers, potentially failing to capture the full threat surface of larger-scale deployments
- No comparative analysis with alternative orchestration frameworks (e.g., LangChain, AutoGen)
- The MCP ecosystem is rapidly evolving, and the observed patterns may not reflect the current state of the art

## Related Work & Insights

- Unlike prompt-chaining systems such as LLaVA-Plus and MAGMA, MCP decouples reasoning from execution and supports dynamic tool loading at runtime
- The SPORT system demonstrates a lightweight, confidence-based tool prioritization strategy
- The AgentOrchestra case reveals the prevalence of out-of-band bridging scripts
- For building reliable multimodal agent systems, standardization at the protocol level—including semantic typing and memory scope management—is indispensable

## Rating

- **Novelty**: ⭐⭐⭐⭐ (First protocol-level audit of MCP for vision systems; pioneering contribution)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (91 servers, 5 case studies, quantified confidence intervals)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear structure; taxonomy and tables are well organized)
- **Value**: ⭐⭐⭐⭐ (Significant guidance for security and reliability in the MCP ecosystem)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Which LLM Multi-Agent Protocol to Choose?](../../ICLR2026/llm_evaluation/which_llm_multi-agent_protocol_to_choose.md)
- [\[NeurIPS 2025\] Model-Behavior Alignment under Flexible Evaluation: When the Best-Fitting Model Isn't the Right One](model-behavior_alignment_under_flexible_evaluation_when_the_best-fitting_model_i.md)
- [\[NeurIPS 2025\] Bayesian Evaluation of Large Language Model Behavior](bayesian_evaluation_of_large_language_model_behavior.md)
- [\[NeurIPS 2025\] MVSMamba: Multi-View Stereo with State Space Model](mvsmamba_multi-view_stereo_with_state_space_model.md)
- [\[NeurIPS 2025\] Reliably Detecting Model Failures in Deployment Without Labels](reliably_detecting_model_failures_in_deployment_without_labels.md)

</div>

<!-- RELATED:END -->
