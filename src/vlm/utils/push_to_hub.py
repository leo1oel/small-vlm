import json
import os
import sys
from pathlib import Path

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
        progress.update(main_task, advance=1, state="paths validated")

        # Step 2: Resolve base model path from local config
        progress.update(main_task, description="Resolving base model", state="loading")
        try:
            cfg_path = pretrained_path / "config.json"
            if not cfg_path.exists():
                raise FileNotFoundError(f"Missing config.json in {pretrained_path}")
            import json as _json

            with open(cfg_path, encoding="utf-8") as _f:
                _cfg = _json.load(_f)
            base_model_path = _cfg.get("hf_name") or _cfg.get("base_model_name_or_path")
            if not base_model_path:
                raise ValueError(
                    "config.json must contain 'hf_name' (or 'base_model_name_or_path') to locate the base LLM"
                )
            # Resolve parent classes for static generation
            parent_llm_class, parent_causal_llm_class, _ = get_dynamic_vlm_class(base_model_path)
            parent_config_class = AutoConfig.from_pretrained(
                base_model_path, trust_remote_code=True
            ).__class__
            parent_names = (
                parent_llm_class.__name__,
                parent_causal_llm_class.__name__,
                parent_config_class.__name__,
            )
            progress.update(main_task, advance=1, state="base resolved")
        except Exception as e:
            progress.update(main_task, state="failed")
            console.print(f"[red]Failed to resolve base model: {e}[/red]")
            return False

        # Step 3: Generate hub files from source models and update config
        progress.update(
            main_task, description="Generating hub files & updating config", state="generating"
        )
        try:
            # Copy processing/connectors from source
            file_list = _copy_from_models(pretrained_path)
            # Render modeling/config with static parent class names (offline-friendly)
            file_list += _render_template_files(pretrained_path, *parent_names)
            generated_files.extend(file_list)
            config_files = _update_config(pretrained_path, base_model_path)
            generated_files.extend(config_files)
            progress.update(main_task, advance=1, state="hub files generated & config updated")
        except Exception as e:
            progress.update(main_task, state="failed")
            console.print(f"[red]Failed to generate files or update config: {e}[/red]")
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
                    dtype="auto",
                )
                processor = AutoProcessor.from_pretrained(
                    pretrained_path,
                    trust_remote_code=True,
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


def _copy_from_models(pretrained_path: Path) -> list[Path]:
    """Copy processing/connectors from src/vlm/models to the model folder."""
    generated_files: list[Path] = []
    models_dir = Path(__file__).resolve().parents[1] / "models"
    sources = {
        "processing_vlm.py": models_dir / "processing_vlm.py",
        "connectors.py": models_dir / "connectors.py",
        # Encoder-free checkpoints reference RawImageProcessor from their
        # preprocessor_config.json; ship the module so remote-code loading
        # (and VLMProcessor's manual rebuild) can resolve it.
        "image_processing_raw.py": models_dir / "image_processing_raw.py",
        # Sibling modules the rendered modeling_vlm.py imports. The cross-modal
        # mask arms and the text->image generation stack (xmodal_mask,
        # gen_diffusion, gen_image, gen_rope) are imported at module load, so
        # they are always required; visual_distill and gen_perceptual are lazily
        # imported by the BREEN-distill / perceptual-loss paths but shipped too
        # so those checkpoints export self-contained.
        "xmodal_mask.py": models_dir / "xmodal_mask.py",
        "gen_diffusion.py": models_dir / "gen_diffusion.py",
        "gen_image.py": models_dir / "gen_image.py",
        "gen_rope.py": models_dir / "gen_rope.py",
        "visual_distill.py": models_dir / "visual_distill.py",
        "gen_perceptual.py": models_dir / "gen_perceptual.py",
    }

    console.print(
        "[bold green]Copying connectors and processing from src/vlm/models...[/bold green]"
    )
    for out_name, src_path in sources.items():
        dst_path = pretrained_path / out_name
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.write_text(src_path.read_text(encoding="utf-8"), encoding="utf-8")
        generated_files.append(dst_path)
        console.print(f"[green]✓[/green] Generated {dst_path.name}")
    return generated_files


def _render_template_files(
    pretrained_path: Path,
    parent_llm_class_name: str,
    parent_causal_llm_class_name: str,
    parent_config_class_name: str,
) -> list[Path]:
    """Render modeling_vlm.py and configuration_vlm.py using templates with resolved parent class names."""
    generated: list[Path] = []
    templates_dir = Path(__file__).resolve().parents[3] / "templates"
    env = Environment(loader=FileSystemLoader(templates_dir))

    # modeling_vlm.py
    modeling_file = pretrained_path / "modeling_vlm.py"
    tmpl = env.get_template("modeling_vlm.py.j2")
    output = tmpl.render(
        parent_class=parent_llm_class_name,
        causal_parent_class=parent_causal_llm_class_name,
    )
    modeling_file.write_text(output, encoding="utf-8")
    generated.append(modeling_file)
    console.print(f"[green]✓[/green] Generated {modeling_file.name}")

    # configuration_vlm.py
    config_file = pretrained_path / "configuration_vlm.py"
    tmpl = env.get_template("configuration_vlm.py.j2")
    output = tmpl.render(parent_class=parent_config_class_name)
    config_file.write_text(output, encoding="utf-8")
    generated.append(config_file)
    console.print(f"[green]✓[/green] Generated {config_file.name}")

    return generated


# templates used for modeling/config to ensure static parent classes


def _update_config(pretrained_path: Path, base_model_path: str | None = None) -> list[Path]:
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
            "AutoModelForCausalLM": "modeling_vlm.VLMForCausalLM",
        }
        # Ensure trust_remote_code is true if custom code is used
        config["trust_remote_code"] = True
        # Record base HF name/path for dynamic resolution if not present
        if base_model_path and "hf_name" not in config:
            config["hf_name"] = base_model_path

        with open(config_path, "w", encoding="utf-8") as f:  # Added encoding
            json.dump(config, f, indent=2)
        updated_files.append(config_path)
        console.print(f"[green]✓[/green] Updated {config_path.name}")

        # Encoder-free checkpoints: RawImageProcessor is not in transformers'
        # AutoImageProcessor registry, so point the preprocessor config at the
        # shipped module for remote-code loading.
        preprocessor_config_path = pretrained_path / "preprocessor_config.json"
        if preprocessor_config_path.exists():
            with open(preprocessor_config_path, encoding="utf-8") as f:
                preprocessor_config = json.load(f)
            if preprocessor_config.get("image_processor_type") == "RawImageProcessor":
                preprocessor_config["auto_map"] = {
                    "AutoImageProcessor": "image_processing_raw.RawImageProcessor",
                }
                with open(preprocessor_config_path, "w", encoding="utf-8") as f:
                    json.dump(preprocessor_config, f, indent=2)
                updated_files.append(preprocessor_config_path)
                console.print(f"[green]✓[/green] Updated {preprocessor_config_path.name}")

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
