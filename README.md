# OpenCog Central: Towards the CogPrime Vision for AGI

Welcome to `opencog-central`. This repository serves as a central point of information and a conceptual guide for understanding the **OpenCog** project and, specifically, the **CogPrime** architecture for Artificial General Intelligence (AGI).

## Introduction to OpenCog and CogPrime

**OpenCog** is an open-source software framework and a collaborative research project aimed at developing Artificial General Intelligence. It provides a diverse set of tools, libraries, and cognitive algorithms designed to support the creation of sophisticated AI systems. At its core, OpenCog features the **AtomSpace**, a weighted, labeled hypergraph database that serves as a flexible knowledge-representation store.

**CogPrime** is a comprehensive AGI architecture designed by Dr. Ben Goertzel. It outlines a theoretical and practical blueprint for achieving human-level (and potentially beyond) general intelligence. The fundamental idea is that

`OpenCogPrime = OpenCog + CogPrime`

This means CogPrime is a *design* intended to be implemented *within* the OpenCog framework, leveraging its existing components and extending them to realize a fully integrated cognitive system. The ultimate implementation of CogPrime within OpenCog is referred to as **OpenCogPrime**.

The CogPrime architecture is detailed in the paper: “[CogPrime: An Integrative Architecture for Embodied Artificial General Intelligence](docs/CogPrime_Integrative_Architecture_AGI.md)”.

## The Essence of CogPrime: Cognitive Synergy

A core hypothesis underlying the CogPrime approach is **cognitive synergy**. This refers to the idea that robust, human-level intelligence can emerge from the rich, dynamic, and cooperative interaction of multiple symbolic and subsymbolic learning and memory components, all integrated within an appropriate cognitive architecture and environment.

CogPrime emphasizes that simply having different AI algorithms is not enough; their careful integration and interoperation are crucial for achieving the emergent structures and dynamics characteristic of general intelligence.

## Key Architectural Components of CogPrime

CogPrime specifies several key memory types and associated cognitive processes:

1. **Knowledge Representation**
   * **AtomSpace** – weighted, labeled hypergraph of knowledge.
   * **Glocal Memory** – hybrid of explicit “local” keys and distributed “global” maps for each concept.

2. **Memory Types & Cognitive Processes**
   * **Declarative Memory** – **Probabilistic Logic Networks (PLN)** for uncertain reasoning.
   * **Procedural Memory** – **MOSES** evolutionary program learning.
   * **Episodic Memory** – internal **Simulation Engine** for replay and imagination.
   * **Attentional Memory** – **ECAN** economic attention allocation (STI/LTI).
   * **Intentional Memory (Goals)** – goal hierarchy, reasoned about by PLN and prioritized by ECAN.
   * **Sensory Memory** – envisioned integration with systems like **DeSTIN** (hierarchical spatio-temporal perception).

3. **Goal-Oriented Dynamics**

   “**Cognitive schematics**” of the form `Context → Procedure → Goal` drive behaviour.

4. **Embodiment & Development**

   Embodied agents (virtual or robotic) follow developmental learning trajectories analogous to human childhood.

## Project Structure (Conceptual)

```
.
├── docs/
│   └── CogPrime_Integrative_Architecture_AGI.md   # Core paper
│   └── (other design docs, whitepapers)
├── src/                  # Conceptual CogPrime implementation
│   ├── atomspace_integration/
│   ├── memory_systems/
│   │   ├── declarative/      # PLN
│   │   ├── procedural/       # MOSES
│   │   ├── episodic/         # Simulation engine
│   ├── cognitive_processes/  # PLN, MOSES, ECAN wrappers
│   ├── agent_core/           # Cognitive cycle
│   └── perception_action/
├── examples/
│   ├── virtual_agent_minecraft_like/
│   └── robot_preschool/
└── tests/
```

## Current Status & Future Vision

`opencog-central` is a documentation hub pointing to the broader OpenCog ecosystem.

* **Active & Stable**
  * [OpenCog AtomSpace](https://github.com/opencog/atomspace) – core hypergraph DB.
  * [Link Grammar](https://github.com/opencog/link-grammar) – linguistic parsing.

* **Research & Development**
  * [Learn](https://github.com/opencog/learn) – symbolic learning.
  * [Agents](https://github.com/opencog/agents) – interactive agent framework.

* **Historical Components (“Fossils”)**
  * Earlier implementations of PLN, MOSES, attention bank, etc., now archived but informative.

* **Hyperon**
  * Next-generation OpenCog, developed by SingularityNET, re-imagines CogPrime ideas with MeTTa and a distributed AtomSpace.

## Getting Started & Contributing

📚 **[Central Models Catalog](MODELS.md)** – Complete catalog of cognitive models with @models annotations and hypergraph structures.

🎯 **[Contributing Guidelines](CONTRIBUTING.md)** – Recursive documentation patterns and @mermaid-chart integration standards.

🔗 **Navigation**: [@models/EvaPhysicalSelfModel](MODELS.md#evaphysicalselfmodel) ↔️ [@mermaid-chart](docs/COGPRIME_ARCHITECTURE_DIAGRAM.md#eva-self-model-integration)

1. **Study the CogPrime paper** in `docs/`.
2. **Explore the OpenCog Wiki**: <https://wiki.opencog.org>.
3. **Review key repos**: `atomspace`, `learn`, `agents`, `link-grammar`.
4. **Follow Hyperon development** at <https://singularitynet.io>.
5. **Join the community** via mailing lists and chats.
6. **Contribute** – documentation, theory discussions, or code.

## Resources

* **OpenCog Wiki:** <https://wiki.opencog.org/w/The_Open_Cognition_Project>
* **CogPrime Architecture Paper:** `docs/CogPrime_Integrative_Architecture_AGI.md`
* **OpenCog GitHub Organization:** <https://github.com/opencog>
* **SingularityNET / Hyperon:** <https://singularitynet.io>
* **Original OpenCog Project Overview:** `profile/README.md`

## NanoCog: CogPrime-Aware AI Assistant

**NanoCog** (folder `NanoCog/`) is a lightweight GPT-based assistant fine-tuned on the *same* CogPrime/OpenCog corpus documented here.  It provides:

* **Conversational help** – ask about CogPrime theory, OpenCog internals, Hyperon, etc.  
* **Introspective diagnostics** – connect to a live AtomSpace and receive real-time analysis of attention allocation, goal dynamics, schematic success, and bottlenecks.  
* **Code generation & refactoring** – produce or improve Atomese/Scheme/MeTTa snippets for OpenCog / Hyperon.  

NanoCog includes:

* `prepare.py` – builds the training corpus from CogPrime docs & Scheme code.  
* `server.py` – FastAPI chatbot with `/chat` and `/diagnostics` endpoints.  
* `nctalk.py` – rich CLI interface with streaming completions and diagnostic mode.  
* `introspection/atomspace_client.py` – robust client for querying AtomSpace REST APIs.  

See **`NanoCog/README.md`** for full setup, training, and usage instructions.

---

This README aims to bridge the comprehensive vision of CogPrime with the existing OpenCog landscape, providing a clearer path for those interested in understanding and contributing to this ambitious AGI endeavor.
