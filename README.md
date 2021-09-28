# cider
## poverty prediction and location inference with mobile phone metadata

Documentation: https://docs.google.com/document/d/1CBIIIJ2wTy6UAdJ39JQYOOkc2JEPUwj_te6VC2iKEOY/edit?usp=sharing

### Deployment
To install, and manage dependencies and virtual environments this project uses Poetry. Follow the [instructions](https://python-poetry.org/docs/) to
install Poetry.

From the root directory `poetry update` followed by `poetry install`, this will establish a venv with all the needed dependencies.

Once your venv is made you can use `poetry run [command]` to run a single CLI command inside the venv.

You can use `poetry shell` to enter into the venv.

### Helper Functions
To support some helper functions that are portable across operating systems we use make. There are many implementations of this functionality for all
operating systems. Once you have downloaded one that suits you and setup poetry you can run:

* `make test [paths]` to run all pytests
* `make lint [paths]` to lint and apply changes automatically
* `make lint-dryrun [paths]` to lint the code but only print warnings
* `make clear-nb` to clear the results out of notebooks before committing them back to the repo. This helps avoid bloat from binary blobs, and keeps the changes to notebooks readable in diff tools.

### Contributing
Before contributing code please:

* Run `make clear-nb` if you have made any changes to Jupyter notebooks you would like to commit
* Run `make lint [paths]` and manually correct any errors in the files you are changing
* Run `poetry update` if you made any changes to the dependencies. This will regenerate the `poetry.lock` file.