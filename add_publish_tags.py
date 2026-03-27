import nbformat

notebook_name = "ml-smm636-a02-private.ipynb"

nb = nbformat.read(notebook_name, as_version=4)
for cell in nb.cells:
    if cell.cell_type == "code":
        tags = cell.get("metadata", {}).setdefault("tags", [])
        if "publish" not in tags:
            tags.append("publish")
nbformat.write(nb, notebook_name)
