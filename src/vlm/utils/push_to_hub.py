import json
import os
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from jinja2 import Environment, FileSystemLoader
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.text import Text
from transformers import AutoConfig, AutoModel, AutoProcessor

from ..models import get_dynamic_vlm_class

console = Console()


def push_to_hub(pretrained: str, repo_id: str | None = None, force: bool = False) -> bool:
    """
    Process a VLM model and optionally push it to the Hugging Face Hub.

    Args:
        pretrained: Path to the pretrained model
        repo_id: Name of the repository on the Hub (e.g., 'username/repo_name').
                     If None, only local processing will be done.
        force: Whether to force push if the repository already exists (only if repo_id is provided).

    Returns:
        True if the model was successfully processed (and optionally pushed), False otherwise.
    """
    should_upload = repo_id is not None

    action_verb = (
        "Pushing model to Hugging Face Hub" if should_upload else "Processing model locally"
    )
    panel_title = "VLM Hub Push" if should_upload else "VLM Local Processing"
    progress_task_description = (
        f"{action_verb}: {repo_id}" if should_upload else "Processing VLM locally..."
    )

    console.print(
        Panel(
            Text(f"{action_verb}{f': {repo_id}' if should_upload else ''}", style="bold green"),
            title=panel_title,
            border_style="green",
        )
    )

    # Validate inputs specific to uploading
    if should_upload:
        if not repo_id or "/" not in repo_id:  # repo_id will not be None here
            console.print(
                "[red]Invalid repository name. Format should be 'username/repo_name'[/red]"
            )
            return False

    pretrained_path = Path(pretrained)
    templates_path = Path("templates")

    generated_files = []

    # Determine total steps for progress bar
    total_progress_steps = 5 if should_upload else 3

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[bold]{task.fields[state]}"),
        console=console,
    ) as progress:
        main_task = progress.add_task(
            progress_task_description, total=total_progress_steps, state="initializing"
        )

        # Step 1: Validate paths
        progress.update(main_task, description="Validating paths", state="checking")
        if not pretrained_path.exists():
            progress.update(main_task, state="failed")
            console.print(f"[red]Model path not found: {pretrained_path}[/red]")
            return False
        if not templates_path.exists():
            progress.update(main_task, state="failed")
            console.print(
                f"[red]Templates path not found: {templates_path} (current dir: {Path.cwd()})[/red]"
            )
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
        progress.update(
            main_task, description="Applying templates & updating config", state="templating"
        )
        try:
            # Ensure templates_path is used by _apply_templates if it's not hardcoded there
            file_list = _apply_templates(
                pretrained_path,
                parent_llm_class,
                parent_causal_llm_class,
                base_model_path,
                templates_path,
            )
            generated_files.extend(file_list)
            config_files = _update_config(pretrained_path)
            generated_files.extend(config_files)
            progress.update(main_task, advance=1, state="templates applied & config updated")
        except Exception as e:
            progress.update(main_task, state="failed")
            console.print(f"[red]Failed to apply templates or update config: {e}[/red]")
            return False

        if should_upload and repo_id:  # repo_id is checked for None again for type safety
            # Step 4: Push to hub (model and processor)
            progress.update(
                main_task, description="Pushing model and processor to Hub", state="pushing"
            )
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

                model.push_to_hub(
                    repo_id, create_pr=False, safe_serialization=True
                )  # Added safe_serialization and create_pr
                processor.push_to_hub(repo_id, create_pr=False)  # Added create_pr

                progress.update(main_task, advance=1, state="model pushed")

            except Exception as e:
                progress.update(main_task, state="failed")
                console.print(f"[red]Failed to push model/processor to Hub: {e}[/red]")
                return False

            # Step 5: Push custom files to hub
            progress.update(
                main_task, description="Pushing custom files to Hub", state="pushing files"
            )
            try:
                api = HfApi()
                token = os.environ.get("HF_TOKEN")  # Token already handled in CLI part

                for file_path in generated_files:
                    # Ensure file exists before uploading
                    if not file_path.exists():
                        console.print(
                            f"[yellow]Warning: File {file_path} not found, skipping upload.[/yellow]"
                        )
                        continue

                    relative_path = file_path.relative_to(pretrained_path)
                    console.print(f"[blue]Uploading {relative_path}...[/blue]")

                    api.upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=str(relative_path),
                        repo_id=repo_id,
                        token=token,  # Token will be None if not set, HfApi handles this
                        repo_type="model",
                        create_pr=False,
                    )
                    console.print(f"[green]✓[/green] Uploaded {relative_path}")

                progress.update(main_task, advance=1, state="custom files pushed")
            except Exception as e:
                progress.update(main_task, state="failed")
                console.print(f"[red]Failed to push custom files to Hub: {e}[/red]")
                return False

            progress.update(main_task, state="complete")  # Ensure it's marked complete
            console.print(
                f"[green]Successfully processed and pushed model and all custom files to {repo_id}![/green]"
            )
        else:  # Not uploading
            progress.update(main_task, state="complete")  # Mark as complete after step 3
            console.print(
                "[green]Successfully processed model locally. No upload was requested.[/green]"
            )
            console.print(f"Generated/updated files are in: {pretrained_path}")

    return True


def _apply_templates(
    pretrained_path: Path,
    parent_llm_class: Any,
    parent_causal_llm_class: Any,
    base_model_path: str,
    templates_dir: Path,
) -> list[Path]:
    """
    Apply Jinja2 templates to generate model files.
    ... (templates_dir added)
    """
    generated_files = []
    try:
        env = Environment(loader=FileSystemLoader(templates_dir))  # Use templates_dir

        console.print("[bold green]Rendering templates...[/bold green]")

        # Render modeling template
        modeling_file = pretrained_path / "modeling_vlm.py"
        _render_template(
            env,
            "modeling_vlm.py.j2",
            {
                "parent_class": parent_llm_class.__name__,
                "causal_parent_class": parent_causal_llm_class.__name__,
            },
            modeling_file,
        )
        generated_files.append(modeling_file)
        console.print(f"[green]✓[/green] Generated {modeling_file.name}")

        # Render processing template
        processing_file = pretrained_path / "processing_vlm.py"
        _render_template(env, "processing_vlm.py.j2", {}, processing_file)
        generated_files.append(processing_file)
        console.print(f"[green]✓[/green] Generated {processing_file.name}")

        # Render connectors template
        connectors_file = pretrained_path / "connectors.py"
        _render_template(env, "connectors.py.j2", {}, connectors_file)
        generated_files.append(connectors_file)
        console.print(f"[green]✓[/green] Generated {connectors_file.name}")

        # Get parent config class and render configuration template
        parent_config_class = AutoConfig.from_pretrained(
            base_model_path, trust_remote_code=True
        ).__class__

        config_file = pretrained_path / "configuration_vlm.py"
        _render_template(
            env,
            "configuration_vlm.py.j2",
            {
                "parent_class": parent_config_class.__name__,
            },
            config_file,
        )
        generated_files.append(config_file)
        console.print(f"[green]✓[/green] Generated {config_file.name}")

        return generated_files

    except Exception as e:
        console.print(f"[red]Failed to apply templates: {e}[/red]")
        raise


def _render_template(
    env: Environment, template_name: str, context: dict[str, Any], output_path: Path
) -> None:
    """
    Render a Jinja2 template and write it to a file.
    ... (no changes needed here) ...
    """
    try:
        template = env.get_template(template_name)
        output = template.render(**context)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:  # Added encoding
            f.write(output)
    except Exception as e:
        console.print(f"[red]Failed to render template {template_name}: {e}[/red]")
        raise


def _update_config(pretrained_path: Path) -> list[Path]:
    """
    Update the model configuration files.
    ... (no changes needed here) ...
    """
    updated_files = []
    config_path = pretrained_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        console.print("[bold green]Updating model config...[/bold green]")
        with open(config_path, encoding="utf-8") as f:  # Added encoding
            config = json.load(f)
        config["auto_map"] = {
            "AutoConfig": "configuration_vlm.VLMConfig",
            "AutoModel": "modeling_vlm.VLMForCausalLM",
        }
        # Ensure trust_remote_code is true if custom code is used
        config["trust_remote_code"] = True

        with open(config_path, "w", encoding="utf-8") as f:  # Added encoding
            json.dump(config, f, indent=2)
        updated_files.append(config_path)
        console.print(f"[green]✓[/green] Updated {config_path.name}")

        processor_config_path = pretrained_path / "processor_config.json"
        processor_config = {
            "auto_map": {"AutoProcessor": "processing_vlm.VLMProcessor"},
            "processor_class": "VLMProcessor",
        }
        with open(processor_config_path, "w", encoding="utf-8") as f:  # Added encoding
            json.dump(processor_config, f, indent=2)
        updated_files.append(processor_config_path)
        console.print(f"[green]✓[/green] Created {processor_config_path.name}")

        return updated_files
    except Exception as e:
        console.print(f"[red]Failed to update config: {e}[/red]")
        raise


def push_vlm_to_hub():
    """Main function for CLI execution with interactive prompts."""
    console.print(
        Panel.fit(
            Text("VLM Hub Tool", style="bold cyan"),
            border_style="cyan",
        )
    )
    console.print("[bold]Please provide the following information:[/bold]")

    valid_path = False
    pretrained_model_path_str = ""  # Initialize
    while not valid_path:
        pretrained_model_path_str = Prompt.ask(
            "[cyan]Path to pretrained model[/cyan]", console=console
        )
        if Path(pretrained_model_path_str).exists() and Path(pretrained_model_path_str).is_dir():
            valid_path = True
        else:
            console.print(
                f"[red]Path does not exist or is not a directory: {pretrained_model_path_str}[/red]"
            )

    # Ask if user wants to upload
    should_upload_to_hub = Confirm.ask(
        "[cyan]Do you want to upload the processed model to Hugging Face Hub?[/cyan]",
        default=True,
        console=console,
    )

    repo_id_for_upload: str | None = None
    force_push_flag = False
    token_is_set_or_provided = False  # For summary

    if should_upload_to_hub:
        valid_repo = False
        while not valid_repo:
            repo_id_for_upload = Prompt.ask(
                "[cyan]Repository ID on Hub[/cyan] [dim](format: username/repo_name or orgname/repo_name)[/dim]",
                console=console,
            )
            if repo_id_for_upload and "/" in repo_id_for_upload:
                valid_repo = True
            else:
                console.print(
                    "[red]Invalid repository ID. Format should be 'username/repo_name' or 'orgname/repo_name'[/red]"
                )

        force_push_flag = Confirm.ask(
            "[cyan]Force push if repository already exists?[/cyan]", default=False, console=console
        )

        # Token handling only if uploading
        token = os.environ.get("HF_TOKEN")
        if not token:
            console.print("[yellow]Warning: HF_TOKEN environment variable not set.[/yellow]")
            console.print(
                "[yellow]Will attempt to use cached credentials or prompt for login if needed by huggingface_hub library.[/yellow]"
            )
            if Confirm.ask(
                "[cyan]Would you like to set HF_TOKEN for this session (recommended for uploads)?[/cyan]",
                default=True,
            ):
                token = Prompt.ask(
                    "[cyan]Enter your Hugging Face token (will not be stored permanently)[/cyan]",
                    password=True,
                )
                if token:
                    os.environ["HF_TOKEN"] = token  # Set for current process
                    token_is_set_or_provided = True
                else:
                    console.print(
                        "[yellow]No token entered. Proceeding without explicitly set token for this session.[/yellow]"
                    )
            else:
                console.print(
                    "[yellow]Proceeding without explicitly set HF_TOKEN for this session.[/yellow]"
                )
        else:
            token_is_set_or_provided = True  # Token was already in env

    # Display summary and confirm
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Model path: [green]{pretrained_model_path_str}[/green]")

    if should_upload_to_hub:
        console.print("  Upload to Hub: [green]Yes[/green]")
        console.print(f"  Repository ID: [green]{repo_id_for_upload}[/green]")
        console.print(
            f"  Force push: [{'green' if force_push_flag else 'yellow'}]{force_push_flag}[/{'green' if force_push_flag else 'yellow'}]"
        )
        console.print(
            f"  HF Token: [{'green' if token_is_set_or_provided else 'yellow'}]"
            f"{'Set/Provided' if token_is_set_or_provided else 'Not explicitly set for session (will use cache/login if needed)'}"
            f"[/{'green' if token_is_set_or_provided else 'yellow'}]"
        )
    else:
        console.print("  Upload to Hub: [yellow]No (local processing only)[/yellow]")

    if Confirm.ask(
        "\n[bold yellow]Proceed with these settings?[/bold yellow]", default=True, console=console
    ):
        # Pass repo_id_for_upload (which will be None if not uploading)
        success = push_to_hub(
            pretrained_model_path_str, repo_id=repo_id_for_upload, force=force_push_flag
        )
        sys.exit(0 if success else 1)
    else:
        console.print("[yellow]Operation cancelled by user[/yellow]")
        sys.exit(0)
