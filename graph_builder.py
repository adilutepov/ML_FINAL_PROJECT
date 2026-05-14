from rdkit import Chem
import torch

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    # --- nodes (atoms) ---
    node_features = []
    for atom in mol.GetAtoms():
        node_features.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetHybridization(),
            atom.GetIsAromatic()
        ])

    x = torch.tensor(node_features, dtype=torch.float)

    # --- edges (bonds) ---
    edge_index = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        edge_index.append([i, j])
        edge_index.append([j, i])  # undirected graph

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return x, edge_index