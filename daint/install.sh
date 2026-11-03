rm -rf .venv
rm uv.lock
uv lock
uv sync --all-packages