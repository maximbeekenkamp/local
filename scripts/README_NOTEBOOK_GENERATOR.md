# Notebook Generator

This directory contains `generate_demo_notebook.py`, which generates the `notebooks/demo_training.ipynb` file programmatically.

## Why use a generator script?

Instead of editing the `.ipynb` file directly (which is JSON and hard to maintain), you can:
1. Edit the Python generator script (much easier to read and modify)
2. Run the script to regenerate the notebook
3. Avoid cell ordering issues and JSON formatting problems

## Usage

To regenerate the notebook after making changes:

```bash
python scripts/generate_demo_notebook.py
```

This will create/overwrite `notebooks/demo_training.ipynb` with the current cell definitions from the script.

## How to modify the notebook

1. **Open** `scripts/generate_demo_notebook.py` in your editor
2. **Find** the `CELLS` list (starts around line 34)
3. **Edit** the cell content you want to change:
   - For markdown cells: `markdown_cell("""Your markdown here""")`
   - For code cells: `code_cell("""Your Python code here""")`
4. **Run** the generator script: `python scripts/generate_demo_notebook.py`
5. **Verify** the changes in Jupyter

## Adding new cells

To add a new cell, simply add it to the `CELLS` list:

```python
CELLS = [
    # ... existing cells ...

    # Add a new markdown cell
    markdown_cell("""## New Section

This is my new section with **markdown** formatting."""),

    # Add a new code cell
    code_cell("""# Your Python code here
print("Hello from new cell!")"""),

    # ... more cells ...
]
```

The cells will appear in the notebook in the order they appear in the list.

## Removing cells

Simply delete or comment out the cell definition from the `CELLS` list, then regenerate.

## Benefits

- ✅ **Version control friendly**: Python code is much easier to diff than notebook JSON
- ✅ **No cell ordering issues**: Cells are always in the order defined in the script
- ✅ **Easier to edit**: Edit Python strings instead of JSON
- ✅ **Consistent formatting**: The generator ensures proper notebook structure
- ✅ **Easy to replicate**: Can use the same pattern for other notebooks

## Example workflow

```bash
# 1. Edit the generator script
vim scripts/generate_demo_notebook.py

# 2. Regenerate the notebook
python scripts/generate_demo_notebook.py

# 3. Test in Jupyter
jupyter notebook notebooks/demo_training.ipynb

# 4. Commit both files
git add scripts/generate_demo_notebook.py notebooks/demo_training.ipynb
git commit -m "Update notebook: Add new loss variant comparison"
```

## Technical details

- The script creates all 27 cells with proper notebook metadata
- Cells are numbered 0-26 in the order they appear
- Both markdown and code cells are supported
- The notebook uses Python 3 kernel by default
- Cell execution counts are set to `None` (unexecuted state)
