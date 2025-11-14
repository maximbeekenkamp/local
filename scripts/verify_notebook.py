#!/usr/bin/env python3
"""
Verify that demo_training.ipynb is correctly formatted.
Run this to check if your notebook is valid.
"""

import json
from pathlib import Path

def verify_notebook():
    notebook_path = Path('notebooks/demo_training.ipynb')
    
    if not notebook_path.exists():
        print("❌ Notebook not found!")
        print(f"   Expected: {notebook_path.absolute()}")
        return False
    
    try:
        with open(notebook_path, 'r') as f:
            nb = json.load(f)
        
        print("✅ Notebook is valid JSON")
        print(f"✅ nbformat version: {nb['nbformat']}.{nb['nbformat_minor']}")
        print(f"✅ Total cells: {len(nb['cells'])}")
        
        # Count cell types
        markdown_count = sum(1 for c in nb['cells'] if c['cell_type'] == 'markdown')
        code_count = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
        
        print(f"✅ Markdown cells: {markdown_count}")
        print(f"✅ Code cells: {code_count}")
        
        # Verify cell sources are lists
        for i, cell in enumerate(nb['cells']):
            if not isinstance(cell['source'], list):
                print(f"❌ Cell {i}: source is not a list!")
                return False
        
        print("✅ All cell sources are properly formatted")
        
        # Check first cell content
        first_cell = nb['cells'][0]
        if first_cell['cell_type'] == 'markdown' and first_cell['source']:
            first_line = first_cell['source'][0]
            if 'Neural Operator Training Demo' in first_line:
                print("✅ Content looks correct!")
            else:
                print(f"⚠️  First cell content unexpected: {first_line[:50]}")
        
        print("\n" + "="*70)
        print("✅ NOTEBOOK IS VALID - No generation errors!")
        print("="*70)
        print("\nIf you're seeing display issues:")
        print("1. Close the notebook in Jupyter")
        print("2. Restart Jupyter server (Ctrl+C in terminal, then restart)")
        print("3. Clear browser cache (Ctrl+Shift+Delete)")
        print("4. Reopen notebooks/demo_training.ipynb")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    verify_notebook()
