# small-vlm

this is small-vlm used for experiments.

## Project Structure
```mermaid
flowchart TD
    %% Configuration Layer
    CONFIG["Configuration Module"]:::config

    %% Data Management Layer
    DATA["Data Module"]:::data

    %% Model Components Layer
    subgraph "Model Components"
        CONNECT["Connectors"]:::model
        LANG["Language Models"]:::model
        VISUAL["Visual Encoders"]:::model
    end

    %% Processing Layer
    subgraph "Processing"
        TRAIN["Training Engine"]:::train
        INF["Inference Pipeline"]:::train
    end

    %% Supporting Infrastructure
    subgraph "Supporting Infrastructure"
        UTIL["Utilities & Utils"]:::utils
        TEST["Testing Infrastructure"]:::testing
        CI["CI/CD Scripts"]:::cicd
        DEV["Dev Tools"]:::cicd
    end

    %% Main Flow: Configuration drives modules
    CONFIG -->|"drives"| DATA
    CONFIG -->|"drives"| CONNECT
    CONFIG -->|"drives"| LANG
    CONFIG -->|"drives"| VISUAL
    CONFIG -->|"drives"| TRAIN
    CONFIG -->|"drives"| INF

    %% Data flows into Processing
    DATA -->|"feeds"| TRAIN
    DATA -->|"feeds"| INF

    %% Model integration into Processing
    CONNECT -->|"used_by"| TRAIN
    LANG -->|"used_by"| TRAIN
    VISUAL -->|"used_by"| TRAIN

    CONNECT -->|"used_by"| INF
    LANG -->|"used_by"| INF
    VISUAL -->|"used_by"| INF

    %% Integration between model components
    VISUAL -->|"integrates"| CONNECT
    LANG -->|"integrates"| CONNECT

    %% Processing ancillary connections to Supporting Infrastructure
    TRAIN -.-> UTIL
    TRAIN -.-> TEST
    TRAIN -.-> CI
    TRAIN -.-> DEV

    INF -.-> UTIL
    INF -.-> TEST
    INF -.-> CI
    INF -.-> DEV

    %% Click Events
    click CONFIG "https://github.com/leo1oel/small-vlm/tree/main/src/vlm/config"
    click DATA "https://github.com/leo1oel/small-vlm/tree/main/src/vlm/data"
    click CONNECT "https://github.com/leo1oel/small-vlm/tree/main/src/vlm/models/connectors"
    click LANG "https://github.com/leo1oel/small-vlm/tree/main/src/vlm/models/language_models"
    click VISUAL "https://github.com/leo1oel/small-vlm/tree/main/src/vlm/models/visual_encoders"
    click TRAIN "https://github.com/leo1oel/small-vlm/tree/main/src/vlm/train"
    click INF "https://github.com/leo1oel/small-vlm/tree/main/src/vlm/inference"
    click UTIL "https://github.com/leo1oel/small-vlm/tree/main/src/vlm/utils"
    click TEST "https://github.com/leo1oel/small-vlm/tree/main/tests"
    click CI "https://github.com/leo1oel/small-vlm/tree/main/.github/workflows"
    click DEV "https://github.com/leo1oel/small-vlm/tree/main/devtools"

    %% Styles
    classDef config fill:#F9E79F,stroke:#B7950B,stroke-width:2px;
    classDef data fill:#AED6F1,stroke:#3498DB,stroke-width:2px;
    classDef model fill:#F5B7B1,stroke:#E74C3C,stroke-width:2px;
    classDef train fill:#A9DFBF,stroke:#27AE60,stroke-width:2px;
    classDef utils fill:#D2B4DE,stroke:#8E44AD,stroke-width:2px;
    classDef testing fill:#FAD7A0,stroke:#E67E22,stroke-width:2px;
    classDef cicd fill:#D5F5E3,stroke:#28B463,stroke-width:2px;
```
---

## Installing uv and Python

This project is set up to use [**uv**](https://docs.astral.sh/uv/), the new package
manager for Python. `uv` replaces traditional use of `pyenv`, `pipx`, `poetry`, `pip`,
etc. This is a quick cheat sheet on that:

On macOS or Linux, if you don't have `uv` installed, a quick way to install it:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For macOS, you prefer [brew](https://brew.sh/) you can install or upgrade uv with:

```shell
brew update
brew install uv
```

See [uv&#39;s docs](https://docs.astral.sh/uv/getting-started/installation/) for more
installation methods and platforms.

Now you can use uv to install a current Python environment:

```shell
uv python install 3.13 # Or pick another version.
```

## Development Workflows

For development workflows, see [development.md](development.md).

For instructions on publishing to PyPI, see [publishing.md](publishing.md).

---

*This project was built from
[simple-modern-uv](https://github.com/jlevy/simple-modern-uv).*
