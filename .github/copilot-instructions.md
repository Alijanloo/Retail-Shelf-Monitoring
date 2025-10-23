## Instructions

* When implementing/refactoring modules, adhere to the Clean Architecture Instructions provided 
in `.github/clean_architecture_instructions.md`

* You can find the information and details of the current project state in `docs/project_tree.md`. 
Keep this file update with the last changes.

* Use the `uv` package manager:
```
uv pip list
uv pip install ...
uv run python <file.py>
```

## Notes

* For logging, get logger using the `multimodal_rag.frameworks.logging_config.get_logger` method.
* Initialize a single `ApplicationContainer` at application bootstrap and use it to resolve all runtime dependencies.
* No Inline comments at all, just the important doc strings
* Put application-wide constants in the `config.yaml` file.