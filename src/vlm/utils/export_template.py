"""Generate the static hub-export modeling template from the live model.

The push-to-hub artifact (`templates/modeling_vlm.py.j2`) is a *static* mirror
of the dynamically-built classes in ``src/vlm/models/modeling_vlm.py`` — the
base-LM parent classes baked in (no runtime ``AutoConfig`` resolution), so the
exported model loads offline via ``trust_remote_code``. Historically the mirror
was hand-maintained and drifted: methods/modules added to the live model
(BREEN learnable queries, visual-FFN experts, the visual-prefix stack,
text->image generation) were never copied across, so exporting such a
checkpoint produced a broken model.

This module removes the drift by *generating* the template from the live source
with a mechanical AST transform:

  * the two ``create_dynamic_*_class`` factories' inner methods are emitted as
    real class bodies (the ``type(name, bases, dict)`` assembly dict gives the
    exact method set + order);
  * every other module-level function (the helpers the methods call, e.g.
    ``install_visual_experts``, ``_routed_mlp_forward``) is copied verbatim;
  * ``super(self.__class__, self)`` -> ``super()`` and the ``pretrain_class``
    closure -> the static ``VLM`` name;
  * ``@override`` (a no-op hint that needs the import) is dropped; real
    decorators (``@torch.no_grad()``) are kept;
  * the base classes / parent import line carry Jinja ``{{ parent_class }}`` /
    ``{{ causal_parent_class }}`` placeholders, substituted by push_to_hub.

``tests/test_inference.py`` renders the committed template AND asserts it equals
a fresh generation, so any future change to ``modeling_vlm.py`` that isn't
re-exported fails CI instead of silently shipping a broken model.

Refresh the committed template after editing the live model::

    uv run python -m vlm.utils.export_template
"""

from __future__ import annotations

import ast
import builtins
import re
import symtable
import textwrap
from pathlib import Path

MODELING_PATH = Path(__file__).resolve().parents[1] / "models" / "modeling_vlm.py"
TEMPLATE_PATH = Path(__file__).resolve().parents[3] / "templates" / "modeling_vlm.py.j2"

# Module-level functions that build the dynamic classes — NOT part of the static
# export (the export bakes the parent classes in and emits the classes directly).
# Everything else at module scope is a helper the emitted methods rely on and is
# copied verbatim.
_FACTORY_FUNCS = frozenset(
    {
        "get_dynamic_vlm_class",
        "create_dynamic_vlm_class",
        "create_dynamic_causal_vlm_class",
        "get_dynamic_vlm",
    }
)

_HEADER = '''"""Static export of the small-vlm dynamic model (modeling_vlm.py in
src/vlm/models) with the base-LM parent classes baked in at push-to-hub time.

GENERATED FILE — do not edit by hand. Regenerate from the live model with
`uv run python -m vlm.utils.export_template` after changing modeling_vlm.py
(tests/test_inference.py pins this file to a fresh generation). It is a mirror
of the dynamically-constructed VLM / VLMForCausalLM classes, with
`super(self.__class__, self)` replaced by plain `super()` and the Auto*
registrations appended. Supports both model families: encoder-based
(CLIP/SigLIP/DINO vision tower) and encoder-free (raw patches +
image_position_ids, optional raw-waveform audio), plus the BREEN learnable-query,
visual-FFN expert, visual-prefix and text->image generation arms.
"""

import logging
from typing import Any

import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor, Tensor
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    PreTrainedModel,
    {{ causal_parent_class }},
    {{ parent_class }},
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from . import xmodal_mask as _xmodal_mask
from .configuration_vlm import VLMConfig
from .connectors import Connector, VisualPrefix, _RawPatchEmbedder, connector_map
from .gen_diffusion import (
    GenTimestepEmbedder,
    add_noise,
    euler_step,
    flow_matching_loss,
    sample_timesteps,
    to_velocity,
)
from .gen_image import assemble_generation_inputs, patches_to_pixels
from .gen_rope import Gen2DRotaryEmbedding, build_mrope_position_ids

log: logging.Logger = logging.getLogger(name=__name__)
'''

_FOOTER = """
AutoModel.register(VLMConfig, VLMForCausalLM, exist_ok=True)
AutoModelForCausalLM.register(VLMConfig, VLMForCausalLM, exist_ok=True)
"""


def _block_source(src_lines: list[str], node: ast.AST) -> str:
    """Dedented source of a def block, decorators excluded (node.lineno is the
    `def` line in 3.8+; decorators have their own nodes)."""
    block = "".join(src_lines[node.lineno - 1 : node.end_lineno])  # type: ignore[attr-defined]
    return textwrap.dedent(block)


def _kept_decorators(src: str, node: ast.FunctionDef) -> list[str]:
    """Decorator source lines to keep — everything except `override` (a no-op
    typing hint we drop so the export needn't import it)."""
    out: list[str] = []
    for dec in node.decorator_list:
        text = ast.get_source_segment(src, dec) or ""
        if text.split(".")[-1] == "override":
            continue
        out.append("@" + text)
    return out


def _method_text(src: str, src_lines: list[str], node: ast.FunctionDef, *, name: str) -> str:
    body = _block_source(src_lines, node)
    if node.name != name:
        body = body.replace(f"def {node.name}(", f"def {name}(", 1)
    body = body.replace("super(self.__class__, self)", "super()")
    text = "\n".join([*_kept_decorators(src, node), body.rstrip()])
    return textwrap.indent(text, "    ")


def _factory(tree: ast.Module, factory_name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == factory_name:
            return node
    raise ValueError(f"factory {factory_name!r} not found in modeling_vlm.py")


def _inner_funcs(factory: ast.FunctionDef) -> dict[str, ast.FunctionDef]:
    return {n.name: n for n in factory.body if isinstance(n, ast.FunctionDef)}


def _assembly_dict(factory: ast.FunctionDef) -> list[tuple[str, str]]:
    """Ordered (method_name, source_symbol) pairs from the factory's
    ``type(<name>, (<base>,), { ... })`` class-assembly call."""
    for node in ast.walk(factory):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "type"
            and len(node.args) == 3
            and isinstance(node.args[2], ast.Dict)
        ):
            pairs: list[tuple[str, str]] = []
            for key, value in zip(node.args[2].keys, node.args[2].values, strict=True):
                assert isinstance(key, ast.Constant), "assembly dict key must be a literal"
                if isinstance(value, ast.Name):
                    pairs.append((key.value, value.id))
                else:  # e.g. config_class: a closure arg, not a Name we emit
                    pairs.append((key.value, key.value))
            return pairs
    raise ValueError(f"no class-assembly type() call found in {factory.name!r}")


def _emit_class(
    *,
    class_name: str,
    base: str,
    factory: ast.FunctionDef,
    module_helpers: set[str],
    src: str,
    src_lines: list[str],
    init_replace: dict[str, str] | None = None,
) -> str:
    inner = _inner_funcs(factory)
    lines = [f"class {class_name}({base}):"]
    for method_name, symbol in _assembly_dict(factory):
        if method_name == "config_class":
            lines.append("    config_class = VLMConfig")
        elif symbol in module_helpers:
            # module-level function bound as a method (e.g. install_xmodal_masks)
            lines.append(f"    {method_name} = {symbol}")
        else:
            node = inner[symbol]
            text = _method_text(src, src_lines, node, name=method_name)
            if init_replace and method_name == "__init__":
                for old, new in init_replace.items():
                    text = text.replace(old, new)
            lines.append("")
            lines.append(text)
    return "\n".join(lines)


# Module globals Python always provides plus the implicit ``__class__`` cell a
# zero-arg ``super()`` introduces — never imported, never flagged as drift.
_IMPLICIT_GLOBALS = frozenset(
    {"__name__", "__file__", "__doc__", "__builtins__", "__spec__", "__loader__", "__package__", "__class__"}
)
_PLACEHOLDER_RE = re.compile(r"\{\{(.*?)\}\}", flags=re.DOTALL)


def _render_for_analysis(template_text: str) -> str:
    """Replace each Jinja ``{{ expr }}`` placeholder with a bare identifier so the
    template parses as real Python. The same ``expr`` maps to the same name, so a
    placeholder used as both an import binding and a base class stays consistent."""
    return _PLACEHOLDER_RE.sub(lambda m: re.sub(r"\W", "_", m.group(1).strip()), template_text)


def _module_bound_names(tree: ast.Module) -> set[str]:
    """Names bound at module scope of the emitted file: header imports, the
    emitted helpers/classes, and module-level assignments (e.g. ``log``)."""
    bound: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            bound.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            bound.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            bound.add(node.name)
        elif isinstance(node, ast.Assign):
            bound.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            bound.add(node.target.id)
    return bound


def _global_loads(table: symtable.SymbolTable) -> set[str]:
    """Every name resolved via a global (module-scope) lookup anywhere in the
    file. ``symtable`` does the real scope analysis, so method locals, args,
    comprehension targets, closure/free vars and attribute names are excluded —
    only genuinely free global ``Name`` loads remain."""
    names = {sym.get_name() for sym in table.get_symbols() if sym.is_global()}
    for child in table.get_children():
        names |= _global_loads(child)
    return names


def _assert_all_names_bound(template_text: str) -> None:
    """Fail generation if any global name the emitted bodies reference is not
    bound by the ``_HEADER`` imports, an emitted helper/class, a builtin, or an
    implicit module global. This pins the hand-maintained ``_HEADER`` import
    block to the names the live model actually uses: adding a top-level import to
    modeling_vlm.py that ``_HEADER`` lacks now fails here (and in the in-sync
    test) instead of shipping a NameError in the exported model."""
    source = _render_for_analysis(template_text)
    tree = ast.parse(source)
    bound = _module_bound_names(tree) | set(dir(builtins)) | _IMPLICIT_GLOBALS
    unbound = sorted(_global_loads(symtable.symtable(source, "<export_template>", "exec")) - bound)
    if unbound:
        raise AssertionError(
            "export template references unbound global name(s): "
            + ", ".join(unbound)
            + " — the exported model would raise NameError. Add the missing "
            "import(s) to export_template._HEADER to mirror modeling_vlm.py."
        )


def build_modeling_template(modeling_source: str | None = None) -> str:
    """Return the rendered-to-Jinja text of the static export modeling file."""
    src = modeling_source if modeling_source is not None else MODELING_PATH.read_text()
    src_lines = src.splitlines(keepends=True)
    tree = ast.parse(src)

    # Module-level helpers: every top-level function except the dynamic-class
    # factories. Emitted verbatim, in source order (dependency-respecting).
    helpers = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name not in _FACTORY_FUNCS
    ]
    module_helpers = {n.name for n in helpers}
    helper_text = "\n\n\n".join(_block_source(src_lines, n).rstrip() for n in helpers)

    vlm_class = _emit_class(
        class_name="VLM",
        base="{{ parent_class }}",
        factory=_factory(tree, "create_dynamic_vlm_class"),
        module_helpers=module_helpers,
        src=src,
        src_lines=src_lines,
    )
    causal_class = _emit_class(
        class_name="VLMForCausalLM",
        base="{{ causal_parent_class }}",
        factory=_factory(tree, "create_dynamic_causal_vlm_class"),
        module_helpers=module_helpers,
        src=src,
        src_lines=src_lines,
        # the causal __init__ closes over `pretrain_class` (the dynamic VLM
        # class); the static export names it VLM.
        init_replace={"pretrain_class(config)": "VLM(config)"},
    )

    parts = [_HEADER.rstrip(), "", "", helper_text, "", "", vlm_class, "", "", causal_class, _FOOTER]
    text = "\n".join(parts).rstrip() + "\n"
    _assert_all_names_bound(text)
    return text


def regenerate(write: bool = True) -> str:
    """Generate the template; write it to the committed path when ``write``."""
    text = build_modeling_template()
    if write:
        TEMPLATE_PATH.write_text(text)
    return text


if __name__ == "__main__":
    regenerate(write=True)
    print(f"wrote {TEMPLATE_PATH}")
