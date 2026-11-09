default:
    @just --list

test *args:
    uv run pytest {{ args }}

# The Sharded tests need several devices: four forced CPU devices, in one process.
sharded:
    XLA_FLAGS=--xla_force_host_platform_device_count=4 uv run pytest -n0 -k sharded

check:
    uv run ruff check src tests
    uv run vulture src --min-confidence 80

# What PyPI will get: the wheel and sdist, built into dist/.
build:
    uv build

# Release: bump the version (patch, minor or major), tag it and push the tag.
bump kind:
    uv version --bump {{kind}}
    jj tag set "v$(uv version --short)" -r @
    jj git push --remote origin --tag "v$(uv version --short)"
