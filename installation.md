# Installing uv and Python

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

See [uv's docs](https://docs.astral.sh/uv/getting-started/installation/) for more
installation methods and platforms.

Now you can use uv to install a current Python environment:

```shell
uv python install 3.13 # Or pick another version.
```

## Known venv patches (re-apply after reinstalling these packages)

- `antlr4-python3-runtime==4.9.3` (pinned by omegaconf 2.3) is not Python 3.13
  compatible out of the box. After any reinstall, re-apply:

  ```bash
  grep -rl "typing.io" .venv/lib/python3.13/site-packages/antlr4/ \
    | xargs sed -i 's/from typing.io import/from typing import/'
  ```

- `lmms-eval==0.7.1` wheel ships without the extensionless
  `_default_template_yaml` task includes (upstream packaging bug); copy them
  from the source tarball of the same tag:

  ```bash
  curl -sL https://github.com/EvolvingLMMs-Lab/lmms-eval/archive/refs/tags/v0.7.1.tar.gz | tar xz -C /tmp
  cd /tmp/lmms-eval-0.7.1/lmms_eval/tasks && find . -type f -exec sh -c \
    'tgt="$VENV_TASKS/$1"; [ -f "$tgt" ] || { mkdir -p "$(dirname "$tgt")"; cp "$1" "$tgt"; }' _ {} \;
  # with VENV_TASKS=<repo>/.venv/lib/python3.13/site-packages/lmms_eval/tasks
  ```
