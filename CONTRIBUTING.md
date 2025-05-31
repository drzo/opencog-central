# Contributing to OpenCog Central

Welcome to the OpenCog Central repository! This guide outlines our recursive documentation and visualization patterns designed to foster cognitive clarity and emergent synergy per the CogPrime vision.

## Overview

OpenCog Central implements a **transformative documentation enhancement** system using:
- `@models` docblocks for structural clarity
- `@mermaid-chart` system diagrams for visual architecture
- Cross-linking patterns for seamless navigation
- Recursive self-documentation for adaptive attention

## Documentation Patterns

### @models Docblock Standard

When creating or modifying cognitive model files (especially Scheme files), include an `@models` docblock in the header:

```scheme
; @models:
;   name: "ModelName"
;   type: "Cognitive Function Type"  
;   architecture: "CogPrime"
;   hypergraph_encoding: "AtomSpace"
;   cognitive_processes: ["Process1", "Process2"]
;   atom_types: ["ConceptNode", "LinkType"]
;   attention_integration: true/false
;   dependencies: ["required-file.scm"]
;   status: "active|experimental|deprecated"
;   diagram_ref: "path/to/diagram.md#section"
```

**Required Fields:**
- `name`: Unique identifier for the model
- `type`: Functional category (e.g., "Embodied Self-Awareness", "Reasoning System")
- `architecture`: Always "CogPrime" for this repository
- `hypergraph_encoding`: Storage mechanism, typically "AtomSpace"

**Recommended Fields:**
- `cognitive_processes`: List of cognitive functions implemented
- `atom_types`: AtomSpace node/link types used
- `attention_integration`: Whether model uses ECAN attention values
- `dependencies`: Required files or modules
- `status`: Current development state
- `diagram_ref`: Link to architectural diagrams

### @mermaid-chart Integration

Create or enhance Mermaid diagrams for:

1. **System Architecture**: High-level cognitive process flows
2. **Model Integration**: How specific models integrate with CogPrime
3. **Data Flows**: Information movement between components
4. **Attention Dynamics**: ECAN resource allocation patterns

**Diagram Guidelines:**
- Use consistent color coding (see existing diagrams)
- Include bidirectional arrows for AtomSpace interactions
- Show cognitive synergy with dotted lines
- Label data flows clearly
- Reference specific `@models` components

### Cross-Linking Patterns

Implement seamless navigation between code, documentation, and diagrams:

**In Documentation:**
```markdown
See [@models/ModelName](path/to/file.scm) ↔️ [@mermaid-chart](#diagram-section)
```

**In Code Comments:**
```scheme
;; See Mermaid chart: docs/DIAGRAM.md#specific-section
;; Related models: [@models/RelatedModel](path/to/related.scm)
```

**In README/MODELS.md:**
```markdown
- **Source**: [`file.scm`](path/to/file.scm)
- **Architecture**: [@mermaid-chart](docs/DIAGRAM.md#section)
```

## Recursive Documentation Workflow

### For New Models

1. **Create the model file** with full `@models` docblock
2. **Add entry to MODELS.md** with cross-references
3. **Update or create relevant Mermaid diagrams**
4. **Add cross-links in README.md** if architecturally significant
5. **Test navigation paths** between all related documents

### For Model Updates

1. **Update the `@models` docblock** to reflect changes
2. **Verify accuracy of MODELS.md entry**
3. **Update Mermaid diagrams** if architecture changed
4. **Check and update all cross-references**
5. **Maintain consistency** across documentation

### For Documentation-Only Changes

1. **Ensure diagram accuracy** matches current code state
2. **Verify all cross-links** work correctly
3. **Update timestamps** in `@models` metadata
4. **Maintain cognitive clarity** in descriptions

## Adaptive Attention Guidelines

### Prioritization Framework

Focus documentation efforts on:

1. **Central Cognitive Models**: Core AtomSpace, ECAN, PLN, MOSES
2. **Integration Patterns**: Where cognitive synergy occurs
3. **Embodiment Interfaces**: Perception/action boundaries
4. **Emergent Behaviors**: Unexpected cognitive phenomena

### Documentation Debt Management

Regularly review and update:
- Outdated `@models` metadata
- Broken cross-references
- Inconsistent Mermaid diagrams
- Missing integration patterns

## Cognitive Synergy Principles

### Documentation as Cognitive Architecture

Treat documentation as part of the cognitive system:
- **Self-referential**: Documentation describes its own patterns
- **Emergent**: Patterns arise from consistent application
- **Adaptive**: Evolves with cognitive architecture
- **Integrative**: Connects all system components

### Hypergraph-Aware Documentation

Structure documentation to mirror AtomSpace patterns:
- **Nodes**: Individual models, concepts, functions
- **Links**: Dependencies, relationships, data flows  
- **Truth Values**: Confidence in documentation accuracy
- **Attention Values**: Importance for cognitive understanding

## Quality Standards

### Code Documentation
- Complete `@models` docblocks for all cognitive models
- Clear hypergraph structure descriptions
- Cognitive synergy integration notes
- Practical usage examples

### Architectural Diagrams
- Accurate representation of current system state
- Consistent visual vocabulary
- Clear data flow indication
- Cognitive process interaction patterns

### Cross-References
- Bidirectional linking between related components
- Consistent link syntax and formatting
- Regular validation of link accuracy
- Seamless navigation experience

## Examples

### Model Integration Example

When adding a new PLN reasoning rule:

1. **Add @models docblock**:
```scheme
; @models:
;   name: "DeductionRule"
;   type: "Reasoning Rule"
;   architecture: "CogPrime"
;   hypergraph_encoding: "AtomSpace"
;   cognitive_processes: ["Deduction", "Inference"]
;   atom_types: ["ImplicationLink", "ConceptNode"]
;   attention_integration: true
;   dependencies: ["pln-config.scm"]
;   status: "active"
;   diagram_ref: "docs/COGPRIME_ARCHITECTURE_DIAGRAM.md#pln-reasoning"
```

2. **Update MODELS.md**:
```markdown
### DeductionRule
- **Type**: Reasoning Rule
- **Description**: Implements deductive inference in PLN
- **Source**: [`rules/deduction.scm`](path/to/rules/deduction.scm)
- **Architecture**: [@mermaid-chart](docs/COGPRIME_ARCHITECTURE_DIAGRAM.md#pln-reasoning)
```

3. **Enhance Mermaid diagram** to show rule integration with PLN

4. **Add cross-reference in README.md** if significant

## Getting Help

- Review existing `@models` examples in the Scheme/ directory
- Study Mermaid diagrams in docs/COGPRIME_ARCHITECTURE_DIAGRAM.md
- Check MODELS.md for integration patterns
- Follow cross-links to understand relationships

## Validation

Before submitting changes:
- [ ] All `@models` docblocks are complete and accurate
- [ ] Mermaid diagrams reflect current system state
- [ ] Cross-references work correctly
- [ ] MODELS.md is updated for new/changed models
- [ ] Documentation follows hypergraph-aware patterns
- [ ] Navigation paths are seamless and intuitive

---

*This contribution guide implements the recursive documentation pattern designed for cognitive synergy, distributed comprehension, and continual evolution of architectural transparency.*