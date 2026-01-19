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
    - [multi_agent/](src/agents/multi_agent/): Multi-agent workflow components (agents, config, main, memory, model_manager, workflow).
    - [single_agent/](src/agents/single_agent/): Single-agent baseline implementation.
  - [core/](src/core/): Core modules (few_shot_loader, logger, models, plantuml, prompts, utils).
  - [evaluation/](src/evaluation/): Evaluation modules.
- [data/](data/): Data files.
  - [index.faiss](data/index.faiss): FAISS index for retrieval.
  - [processed/](data/processed/): Processed data.
    - [diagrams.json](data/processed/diagrams.json): Processed diagrams.
    - [few_shot.json](data/processed/few_shot.json): Few-shot examples.
    - [test_exercises.json](data/processed/test_exercises.json): Test exercises.
    - [validation_exercises.json](data/processed/validation_exercises.json): Validation exercises.
  - [raw/](data/raw/): Raw data (if any).
- [output/](output/): Output results from runs.
  - [multi_agent/](output/multi_agent/): Outputs from multi-agent workflow.
  - [single_agent/](output/single_agent/): Outputs from single-agent baseline.
- [pyproject.toml](pyproject.toml): Project configuration for modern Python packaging.
- [setup.py](setup.py): Legacy setup script.
- [requirements.txt](requirements.txt): Python dependencies.
- [README.md](README.md): This file.
- [**init**.py](__init__.py): Package initialization.

## How to Run

1. Ensure the PlantUML server is running.
2. Run the script from the project root:
   ```bash
   python scripts/main.py --mode single  # For single-agent baseline
   python scripts/main.py --mode multi   # For multi-agent workflow
   ```
   This will execute the respective workflows and save outputs to the [output/](output/) directory.

## Evaluation

The system evaluates generated diagrams against a human-created gold standard using:

- **Structural Accuracy**: Precision, Recall, and F1 score for Classes, Attributes, and Relationships.
