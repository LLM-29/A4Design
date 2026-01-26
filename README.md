# A4: Generating and Correcting Design

Design diagrams (UML class diagrams, sequence diagrams, state machines) are essential for software engineering but often missing or outdated. LLMs can help generate diagrams from text or correct existing diagrams.

Research Questions

- How accurate are LLMs at generating formal design diagrams from natural
  language?
- Can multi-agent pipelines improve diagram consistency?
- How effective are LLMs at detecting and correcting structural errors in diagrams?

Minimum requirements

- Design Artifacts: Choose 3–5 software components (e.g., small systems, class hierarchies, workflows) from:
  - past course assignments
  - open-source documentation
  - your own designed examples
- System Implementation. Include:
  - One single-agent baseline→ An LLM generates or corrects diagrams directly from text.
  - One multi-agent workflow with ≥2 roles
- Evaluation. Use at least one of the following evaluation methods:
  - structural accuracy against a small human-created gold standard
  - consistency checking between diagram elements (e.g., missing methods, invalid relations)
  - error-detection and correction effectiveness

## Prerequisites

Before running the project, ensure you have the following components set up:

### 1. PlantUML Server

A local PlantUML server is required for syntax validation and diagram rendering.

- The easiest way is using Docker:
  ```bash
  docker run -d -p 8080:8080 plantuml/plantuml-server:jetty
  ```
- Alternatively, if you want to run the server locally, you can follow the official guide at `https://plantuml.com/starting`.

### 2. OpenRouter API Key

The project uses OpenRouter for accessing Large Language Models (LLMs). You need to obtain an API key from [OpenRouter](https://openrouter.ai/).

Create a `.env` file in the project root with the following content:

```
OPENROUTER_API_KEY=your_api_key_here
```

## Setup Instructions

1. **Clone the repository** and navigate to the project folder.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Alternatively, if using `pyproject.toml`:
   ```bash
   pip install -e .
   ```

## Project Structure

- [notebooks/](notebooks/): Jupyter notebooks for experimentation.
  - [multi-agent.ipynb](notebooks/multi-agent.ipynb): The main Jupyter notebook containing the multi-agent workflow.
- [scripts/](scripts/): Main scripts to run the project.
  - [main.py](scripts/main.py): Entry point script for running single-agent or multi-agent modes.
- [src/](src/): Source code of the project.
  - [agents/](src/agents/): Agent implementations.
    - [multi_agent/](src/agents/multi_agent/): Multi-agent workflow components (agents, config, main, memory, workflow).
    - [single_agent/](src/agents/single_agent/): Single-agent baseline implementation.
  - [core/](src/core/): Core modules shared across agents.
    - [few_shot_loader.py](src/core/few_shot_loader.py): Few-shot example loading utilities.
    - [logger.py](src/core/logger.py): Logging utilities.
    - [model_manager.py](src/core/model_manager.py): Dynamic model loading and management for different tasks.
    - [models.py](src/core/models.py): Data models and type definitions.
    - [plantuml.py](src/core/plantuml.py): PlantUML integration and validation.
    - [prompts.py](src/core/prompts.py): Prompt templates and system messages.
    - [utils.py](src/core/utils.py): Utility functions including safe LLM invocation and system initialization.
  - [evaluation/](src/evaluation/): Evaluation modules and threshold optimization.
    - [evaluation.py](src/evaluation/evaluation.py): Main evaluation utilities.
    - [joint_threshold_optimization.py](src/evaluation/joint_threshold_optimization.py): Joint threshold optimization.

- [data/](data/): Data files.
  - [index.faiss](data/index.faiss): FAISS index for retrieval.
  - [processed/](data/processed/): Processed data.
    - [diagrams.json](data/processed/diagrams.json): Processed diagrams used for validation.
    - [few_shot.json](data/processed/few_shot.json): Few-shot examples for prompt engineering.
    - [generated_diagrams_cv.json](data/processed/generated_diagrams_cv.json): Generated diagrams for cross-validation.
    - [labels.json](data/processed/labels.json): Labels for evaluation.
    - [test_exercises.json](data/processed/test_exercises.json): Test exercises.
  - [raw/](data/raw/): Raw data (if any).
- [output/](output/): Output results from runs.
  - [cache/](output/cache/): Cached results from threshold optimization.
    - [threshold_generation_cache.json](output/cache/threshold_generation_cache.json): Cached threshold optimization results.
  - [evaluation/](output/evaluation/): Evaluation results.
  - [multi_agent_critic/](output/multi_agent_critic/): Outputs from multi-agent workflow with critic evaluation.
  - [multi_agent_scorer/](output/multi_agent_scorer/): Outputs from multi-agent workflow with scorer evaluation.
  - [single_agent/](output/single_agent/): Outputs from single-agent baseline.
- [pyproject.toml](pyproject.toml): Project configuration for modern Python packaging.
- [requirements.txt](requirements.txt): Python dependencies.
- [README.md](README.md): This file.
- [**init**.py](__init__.py): Package initialization.

## System Architecture

The project implements two main approaches for generating and correcting UML diagrams from natural language requirements:

### Data Usage

The system uses a structured dataset for training, testing, and evaluation:

- **Few-shot Examples** (`few_shot.json`): 3 exercises used for in-context learning to improve generation quality.
- **Test Exercises** (`test_exercises.json`): 3 exercises used for evaluating the system's performance on unseen requirements.
- **Validation Diagrams** (`diagrams.json`): 14 processed diagrams used for cross-validation and structural accuracy assessment.

The first 3 exercises from the combined dataset are reserved for few-shot learning and are excluded from testing to ensure proper evaluation on unseen data.

### Shared Components

- **Model Manager** (`src/core/model_manager.py`): Dynamic loading and management of LLMs for different tasks, shared between single-agent and multi-agent systems.
- **Safe Invoke** (`src/core/utils.py`): Retry logic for reliable LLM calls with error handling.
- **Few-shot Learning**: Uses examples from `few_shot.json` to improve generation quality.
- **Retrieval-Augmented Generation**: Uses FAISS index for relevant example retrieval.

### Single-Agent Baseline

A straightforward LLM-based system that directly generates PlantUML code from text requirements using a single model call through the shared Model Manager.

**Workflow**:

1. **Single Agent Generation**: The LLM receives the natural language requirements and generates complete PlantUML class diagram code in one step, leveraging few-shot examples and retrieval-augmented generation for improved accuracy.

### Multi-Agent Workflow

A multi-agent system with specialized roles that iteratively refines the diagram generation process. The system supports two evaluation modes: critic-based and scorer-based.

**Shared Multi-Agent Components**:

- **Retriever**: Retrieves relevant few-shot examples from the knowledge base.
- **Class Extractor**: Identifies and extracts class definitions and attributes from requirements.
- **Relationship Extractor**: Determines relationships between classes (inheritance, association, aggregation, etc.).
- **Generator**: Produces initial PlantUML code based on extracted elements.
- **Syntax Checker**: Validates PlantUML syntax and reports errors.
- **Syntax Fixer**: Corrects PlantUML syntax errors.
- **Logical Fixer**: Addresses structural and logical inconsistencies in the diagram.

**Critic-Based Workflow**:

1. **Retrieve** → **Extract Classes** → **Extract Relationships** → **Generate** → **Syntax Check**
2. If syntax invalid: **Syntax Fixer** → **Syntax Check** (loop until valid)
3. **Critic** → If issues found: **Logical Fixer** → **Critic** (iterative refinement until valid or converged)
4. **END**

**Scorer-Based Workflow**:

1. **Retrieve** → **Extract Classes** → **Extract Relationships** → **Generate** → **Syntax Check**
2. If syntax invalid: **Syntax Fixer** → **Syntax Check** (loop until valid)
3. **Scorer** → Evaluates diagram quality on syntax, semantic, and pragmatic dimensions
4. If score below threshold: **Logical Fixer** → **Scorer** (iterative improvement with plateau detection)
5. **END** (when score meets threshold or convergence detected)

The multi-agent system uses the same shared Model Manager and Safe Invoke utilities as the single-agent baseline.

## How to Run

1. Ensure the PlantUML server is running and the `.env` file is set up with your OpenRouter API key.
2. Run the script from the project root:
   ```bash
   python scripts/main.py --mode single  # For single-agent baseline
   python scripts/main.py --mode multi --evaluation critic  # For multi-agent workflow with critic-based evaluation
   python scripts/main.py --mode multi --evaluation scorer  # For multi-agent workflow with scorer-based evaluation
   ```
   This will execute the respective workflows and save outputs to the [output/](output/) directory. Single-agent results are saved to `output/single_agent/`, while multi-agent results are saved to `output/multi_agent_critic/` or `output/multi_agent_scorer/` depending on the evaluation mode.

## Evaluation

The system evaluates generated diagrams against a human-created gold standard using:

- **Structural Accuracy**: Precision, Recall, and F1 score for Classes, Attributes, and Relationships.

### Threshold Optimization

To optimize evaluation accuracy, the project includes a joint threshold optimization script:

```bash
# Joint threshold optimization
python src/evaluation/joint_threshold_optimization.py
```

This script performs optimization and saves results to the `output/evaluation/` directory. Cached results are stored in `output/cache/` to avoid recomputation. The optimal thresholds are used in the main evaluation pipeline.
