# UV


## Ways to initalize:

`uv init [project name]`
- Parameters:
    - --app: Application
    - --library: Library
    - --package: package
    - --no-workspace: Here it will initalize without any sort of workspace even if the structure would be in a workspace

## Project Structure:

./.git: Git Management
./.gitignore: Files / Folders to Ignore with Git
./.python-version: Plain text file which contains the python version
./hello.py: Boiler Plate File
./pyproject.toml: Contains a sort of Configuration of the Project
./README.md: Empty Markdown File
./uv.lock: Log File

## Setup Virtual Environment

`uv run hello.py`: Simply setups up the .venv with the required requirements

## Add Libraries

`uv add [libary name]`: same as pip install [library name]

## Remove Libraries

`uv remove [library name]`

## Syncin with the Environment

`uv sync`

## Upgrade Package

```bash
╭ ranuga@ranuga ▸ ~/Jobs/Altrium/Altrium-PD/Learning/uv/my_project (main?)
╰ ⚡ uv lock --upgrade-package pandas
Resolved 16 packages in 576ms
```

## Dependancy Tree

```bash
╭ ranuga@ranuga ▸ ~/Jobs/Altrium/Altrium-PD/Learning/uv/my_project (main?)
╰ > uv tree
Resolved 16 packages in 0.72ms
my-project v0.1.0
├── fastapi v0.115.6
│   ├── pydantic v2.10.3
│   │   ├── annotated-types v0.7.0
│   │   ├── pydantic-core v2.27.1
│   │   │   └── typing-extensions v4.12.2
│   │   └── typing-extensions v4.12.2
│   ├── starlette v0.41.3
│   │   └── anyio v4.7.0
│   │       ├── idna v3.10
│   │       ├── sniffio v1.3.1
│   │       └── typing-extensions v4.12.2
│   └── typing-extensions v4.12.2
└── pandas v2.2.3
    ├── numpy v2.2.0
    ├── python-dateutil v2.9.0.post0
    │   └── six v1.17.0
    ├── pytz v2024.2
    └── tzdata v2024.2
```

## Workspaces

This is for much more bigger code bases where there are projects inside projects, so that specific project would be added as a member of a specific workspace which is the bigger project.
and whenever we add a package it would be added to that main workspaces log file.

## Tools

uv tool run [tool name]

uv tool run [tool name] [command of that tool]

uv tool install/uninstall [tool name]

uv tool upgrade [tool name]

## Python Versions

uv python list: to get all the variations

uv python install [specific python variation]: Download the specific python variation

uv venv --python 3.13.0: if we have already have this variation (can be found by uv python list) 

## Constraints

uv python install '>=3.9, <3.11'
