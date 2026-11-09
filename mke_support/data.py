import os

import torch
import pandas as pd
import torch_geometric.data
from rdkit import Chem
import pickle
from torch_geometric.data import Data ,InMemoryDataset


x_map = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map = {
    'bond_type': [
        'misc',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'is_conjugated': [False, True],
}
class MoleculeDataset(InMemoryDataset):
    def build_mol_graph_data(self,dataset_smiles, train_index, test_index,valid_index,labels_name, group):
        pre_len_data = 0
        now_len_data = 0
        group_list = list(dataset_smiles['group'])
        labels_list = list(dataset_smiles[labels_name])
        smilesList = dataset_smiles['smiles']

        data_list = []
        tmpindex = -1
        for i, smiles in enumerate(smilesList):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print("第i个分子", i, "：smiles", smiles)
                continue


            xs = []
            for atom in mol.GetAtoms():
                x = []
                x.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
                x.append(x_map['chirality'].index(str(atom.GetChiralTag())))
                x.append(x_map['degree'].index(atom.GetTotalDegree()))
                x.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
                x.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
                x.append(x_map['num_radical_electrons'].index(
                    atom.GetNumRadicalElectrons()))
                x.append(x_map['hybridization'].index(
                    str(atom.GetHybridization())))
                x.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
                x.append(x_map['is_in_ring'].index(atom.IsInRing()))
                xs.append(x)
            x = torch.tensor(xs, dtype=torch.long).view(-1, 9)


            edge_indices, edge_attrs = [], []
            for bond in mol.GetBonds():
                a = bond.GetBeginAtomIdx()
                b = bond.GetEndAtomIdx()


                e = []
                e.append(e_map['bond_type'].index(str(bond.GetBondType())))
                e.append(e_map['stereo'].index(str(bond.GetStereo())))
                e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

                edge_indices += [[a, b], [b, a]]
                edge_attrs += [e, e]

            edge_index = torch.tensor(edge_indices)
            edge_index = edge_index.t().to(torch.long).view(2, -1)
            edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)


            if edge_index.numel() > 0:
                perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
                edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=labels_list[i],
                        smiles=smiles)

            data_list.append(data)
            now_len_data = len(data_list)
            if now_len_data > pre_len_data:
                pre_len_data = now_len_data
            else:
                print("第i个分子",i,"：smiles",smiles)

            if group_list[i] == group[0]:
                tmpindex = tmpindex + 1
                train_index.append(tmpindex)
            elif group_list[i] == group[1]:
                tmpindex = tmpindex + 1
                valid_index.append(tmpindex)
            else:
                tmpindex = tmpindex+1
                test_index.append(tmpindex)

        package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.path.join(package_root, 'data', 'processed')
        os.makedirs(cache_dir, exist_ok=True)
        processed_paths = os.path.join(cache_dir, 'process_data' + labels_name + '.pt')

        torch.save(self.collate(data_list), processed_paths)

        self.data, self.slices = torch.load(processed_paths)
