# LanceDB

A Python library for [LanceDB](https://github.com/lancedb/lancedb).

## Installation

```bash
pip install lancedb
```

## Usage

### Basic Example

```python
import lancedb
db = lancedb.connect('<PATH_TO_LANCEDB_DATASET>')
table = db.open_table('my_table')
results = table.search([0.1, 0.3]).limit(20).to_list()
print(results)
```

## Development

LanceDb is based on the rust crate `lancedb` and is built with maturin.  In order to build with maturin
you will either need a conda environment or a virtual environment (venv).

```bash
python -m venv venv
. ./venv/bin/activate
```

Install the necessary packages:

```bash
python -m pip install .[tests,dev]
```

To build the python package you can use maturin:

```bash
# This will build the rust bindings and place them in the appropriate place
# in your venv or conda environment
matruin develop
```

To run the unit tests:

```bash
pytest
```

To run the doc tests:

```bash
pytest --doctest-modules python/lancedb
```

To run linter and automatically fix all errors:

```bash
ruff format python
ruff --fix python
```

If any packages are missing, install them with:

```bash
pip install <PACKAGE_NAME>
```

___
For **Windows** users, there may be errors when installing packages, so these commands may be helpful:

Activate the virtual environment:

```bash
. .\venv\Scripts\activate
```

You may need to run the installs separately:

```bash
pip install -e .[tests]
pip install -e .[dev]
```

`tantivy` requires `rust` to be installed, so install it with `conda`, as it doesn't support windows installation:

```bash
pip install wheel
pip install cargo
conda install rust
pip install tantivy
```
