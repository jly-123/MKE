import argparse
import csv
import json
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Draw selected MKE nodes on the original molecule.")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--samples", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("figures/explanations"))
    parser.add_argument("--format", choices=("svg", "png"), default="svg")
    parser.add_argument("--width", type=int, default=700)
    parser.add_argument("--height", type=int, default=500)
    args = parser.parse_args(argv)
    if args.width < 1 or args.height < 1:
        parser.error("Image dimensions must be positive")
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError as exc:
        parser.exit(2, "RDKit is required for molecular rendering: {}\n".format(exc))
    source = args.run / "mke_new_explanations.csv"
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {}
    for row in rows:
        index = int(row["sample_index"])
        if index in indexed:
            raise ValueError("Duplicate sample_index {}".format(index))
        indexed[index] = row
    missing = sorted(set(args.samples) - set(indexed))
    if missing:
        raise ValueError("Requested sample indices not found: {}".format(missing))
    planned = []
    for index in args.samples:
        row = indexed[index]
        mol = Chem.MolFromSmiles(row["molecule_smiles"])
        if mol is None or not mol.GetNumAtoms():
            raise ValueError("Invalid molecule in sample {}".format(index))
        nodes = json.loads(row["explanation_nodes"])
        if not isinstance(nodes, list) or any(type(node) is not int for node in nodes):
            raise ValueError("Invalid explanation node list")
        if len(nodes) != len(set(nodes)) or any(node < 0 or node >= mol.GetNumAtoms() for node in nodes):
            raise ValueError("Duplicate or out-of-range explanation nodes")
        if "num_nodes" in row and int(row["num_nodes"]) != mol.GetNumAtoms():
            raise ValueError("SMILES atom count differs from recorded graph size")
        selected = set(nodes)
        bonds = [bond.GetIdx() for bond in mol.GetBonds()
                 if bond.GetBeginAtomIdx() in selected and bond.GetEndAtomIdx() in selected]
        planned.append((index, row, mol, nodes, bonds))
    args.output.mkdir(parents=True, exist_ok=True)
    for index, row, mol, nodes, bonds in planned:
        drawing = (rdMolDraw2D.MolDraw2DSVG(args.width, args.height)
                   if args.format == "svg" else
                   rdMolDraw2D.MolDraw2DCairo(args.width, args.height))
        drawing.drawOptions().addAtomIndices = True
        legend = "Sample {} | selected {}/{} | sparsity {:.4f}".format(
            index, len(nodes), mol.GetNumAtoms(), float(row["sparsity"]))
        color = (1.0, 0.65, 0.15)
        drawing.DrawMolecule(mol, legend=legend, highlightAtoms=nodes, highlightBonds=bonds,
                             highlightAtomColors={node: color for node in nodes},
                             highlightBondColors={bond: color for bond in bonds})
        drawing.FinishDrawing()
        target = args.output / ("sample_{}.{}".format(index, args.format))
        data = drawing.GetDrawingText()
        if isinstance(data, str):
            target.write_text(data, encoding="utf-8")
        else:
            target.write_bytes(data)
        print(str(target))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        sys.exit(1)
