import json
import sys
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.text import Text
from transformers import AutoConfig, AutoModel, AutoProcessor

console = Console()


# Import dynamically to avoid circular imports
def get_dynamic_vlm_class(path: str) -> tuple:
    """Import function dynamically to avoid circular imports."""
    try:
        from ..models import get_dynamic_vlm_class

        return get_dynamic_vlm_class(path)
    except Exception as e:
        console.print(f"[red]Error importing get_dynamic_vlm_class: {e}[/red]")
        raise ImportError(f"Failed to import dynamic VLM class: {e}") from e


def push_to_hub(pretrained: str, repo_name: str, force: bool = False) -> bool:
    """
    Push a VLM model to the Hugging Face Hub.

    Args:
        pretrained: Path to the pretrained model
        repo_name: Name of the repository on the Hub
        force: Whether to force push if the repository already exists

    Returns:
        True if the model was successfully pushed, False otherwise
    """
    console.print(
        Panel(
            Text(f"Pushing model to Hugging Face Hub: {repo_name}", style="bold green"),
            title="VLM Hub Push",
            border_style="green",
        )
    )

    # Validate inputs
    if not repo_name or "/" not in repo_name:
        console.print("[red]Invalid repository name. Format should be 'username/repo_name'[/red]")
        return False

    pretrained_path = Path(pretrained)
    templates_path = Path("templates")

    # Create progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[bold]{task.fields[state]}"),
        console=console,
    ) as progress:
        # Main task and subtasks
        main_task = progress.add_task("[cyan]Pushing to Hub...", total=4, state="initializing")

        # Step 1: Validate paths
        progress.update(main_task, description="Validating paths", state="checking")

        if not pretrained_path.exists():
            progress.update(main_task, state="failed")
            console.print(f"[red]Model path not found: {pretrained_path}[/red]")
            return False

        if not templates_path.exists():
            progress.update(main_task, state="failed")
            console.print(f"[red]Templates path not found: {templates_path}[/red]")
            return False

        progress.update(main_task, advance=1, state="paths validated")

        # Step 2: Get dynamic VLM class
        progress.update(main_task, description="Loading model classes", state="loading")
        try:
            parent_llm_class, parent_causal_llm_class, base_model_path = get_dynamic_vlm_class(
                str(pretrained_path)
            )
            progress.update(main_task, advance=1, state="classes loaded")
        except Exception as e:
            progress.update(main_task, state="failed")
            console.print(f"[red]Failed to get dynamic VLM class: {e}[/red]")
            return False

        # Step 3: Apply templates and update config
        progress.update(main_task, description="Applying templates", state="templating")
        try:
            _apply_templates(
                pretrained_path, parent_llm_class, parent_causal_llm_class, base_model_path
            )
            _update_config(pretrained_path)
            progress.update(main_task, advance=1, state="templates applied")
        except Exception as e:
            progress.update(main_task, state="failed")
            console.print(f"[red]Failed to apply templates: {e}[/red]")
            return False

        # Step 4: Push to hub
        progress.update(main_task, description="Pushing to Hub", state="pushing")
        try:
            model = AutoModel.from_pretrained(
                pretrained_path,
                trust_remote_code=True,
                torch_dtype="auto",
            )
            processor = AutoProcessor.from_pretrained(
                pretrained_path,
                trust_remote_code=True,
                torch_dtype="auto",
            )

            # Push to hub with appropriate options
            model.push_to_hub(repo_name, force=force)
            processor.push_to_hub(repo_name, force=force)

            progress.update(main_task, advance=1, state="complete")

        except Exception as e:
            progress.update(main_task, state="failed")
            console.print(f"[red]Failed to push to Hub: {e}[/red]")
            return False

    console.print(f"[green]Successfully pushed model to {repo_name}![/green]")
    return True


def _apply_templates(
    pretrained_path: Path, parent_llm_class: Any, parent_causal_llm_class: Any, base_model_path: str
) -> None:
    """
    Apply Jinja2 templates to generate model files.

    Args:
        pretrained_path: Path to the pretrained model
        parent_llm_class: Parent LLM class
        parent_causal_llm_class: Parent causal LLM class
        base_model_path: Path to the base model
    """
    try:
        env = Environment(loader=FileSystemLoader("templates"))

        console.print("[bold green]Rendering templates...[/bold green]")

        # Render modeling template
        _render_template(
            env,
            "modeling_vlm.py.j2",
            {
                "parent_class": parent_llm_class.__name__,
                "causal_parent_class": parent_causal_llm_class.__name__,
            },
            pretrained_path / "modeling_vlm.py",
        )
        console.print("[green]✓[/green] Generated modeling_vlm.py")

        # Render processing template
        _render_template(env, "processing_vlm.py.j2", {}, pretrained_path / "processing_vlm.py")
        console.print("[green]✓[/green] Generated processing_vlm.py")

        # Render connectors template
        _render_template(env, "connectors.py.j2", {}, pretrained_path / "connectors.py")
        console.print("[green]✓[/green] Generated connectors.py")

        # Get parent config class and render configuration template
        parent_config_class = AutoConfig.from_pretrained(
            base_model_path, trust_remote_code=True
        ).__class__

        _render_template(
            env,
            "configuration_vlm.py.j2",
            {
                "parent_class": parent_config_class.__name__,
            },
            pretrained_path / "configuration_vlm.py",
        )
        console.print("[green]✓[/green] Generated configuration_vlm.py")

    except Exception as e:
        console.print(f"[red]Failed to apply templates: {e}[/red]")
        raise


def _render_template(
    env: Environment, template_name: str, context: dict[str, Any], output_path: Path
) -> None:
    """
    Render a Jinja2 template and write it to a file.

    Args:
        env: Jinja2 environment
        template_name: Name of the template
        context: Template context variables
        output_path: Path to write the output file
    """
    try:
        template = env.get_template(template_name)
        output = template.render(**context)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(output)
    except Exception as e:
        console.print(f"[red]Failed to render template {template_name}: {e}[/red]")
        raise


def _update_config(pretrained_path: Path) -> None:
    """
    Update the model configuration files.

    Args:
        pretrained_path: Path to the pretrained model
    """
    config_path = pretrained_path / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        console.print("[bold green]Updating model config...[/bold green]")

        # Update model config
        with open(config_path) as f:
            config = json.load(f)

        config["auto_map"] = {
            "AutoConfig": "configuration_vlm.VLMConfig",
            "AutoModel": "modeling_vlm.VLMForCausalLM",
        }

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        console.print("[green]✓[/green] Updated config.json")

        # Create processor config
        processor_config_path = pretrained_path / "processor_config.json"
        processor_config = {
            "auto_map": {"AutoProcessor": "processing_vlm.VLMProcessor"},
            "processor_class": "VLMProcessor",
        }

        with open(processor_config_path, "w") as f:
            json.dump(processor_config, f, indent=2)

        console.print("[green]✓[/green] Created processor_config.json")

    except Exception as e:
        console.print(f"[red]Failed to update config: {e}[/red]")
        raise


def push_vlm_to_hub():
    """Main function for CLI execution with interactive prompts."""
    # Print banner
    console.print(
        Panel.fit(
            Text("VLM Hub Tool", style="bold cyan"),
            border_style="cyan",
        )
    )

    # Interactive input using Rich's Prompt
    console.print("[bold]Please provide the following information:[/bold]")

    # Get model path with input validation
    valid_path = False
    while not valid_path:
        pretrained = Prompt.ask("[cyan]Path to pretrained model[/cyan]", console=console)
        if Path(pretrained).exists():
            valid_path = True
        else:
            console.print(f"[red]Path does not exist: {pretrained}[/red]")

    # Get repository name with format validation
    valid_repo = False
    while not valid_repo:
        repo_name = Prompt.ask(
            "[cyan]Repository name on Hub[/cyan] [dim](format: username/repo_name)[/dim]",
            console=console,
        )
        if "/" in repo_name:
            valid_repo = True
        else:
            console.print(
                "[red]Invalid repository name. Format should be 'username/repo_name'[/red]"
            )

    # Force push option
    force = Confirm.ask(
        "[cyan]Force push if repository already exists?[/cyan]", default=False, console=console
    )

    # Display summary and confirm
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Model path: [green]{pretrained}[/green]")  # pyright: ignore
    console.print(f"  Repository: [green]{repo_name}[/green]")  # pyright: ignore
    console.print(
        f"  Force push: [{'green' if force else 'yellow'}]{force}[/{'green' if force else 'yellow'}]"
    )

    # Final confirmation
    if Confirm.ask(
        "\n[bold yellow]Proceed with these settings?[/bold yellow]", default=True, console=console
    ):
        success = push_to_hub(pretrained, repo_name, force)  # pyright: ignore
        sys.exit(0 if success else 1)
    else:
        console.print("[yellow]Operation cancelled by user[/yellow]")
        sys.exit(0)
