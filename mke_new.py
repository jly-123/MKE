import os
import sys
import json
import itertools
import math
import operator
import random
from datetime import datetime
import torch
import hydra
from concurrent.futures import ProcessPoolExecutor
from rdkit import Chem
import numpy as np
import time
import networkx as nx
import pandas as pd
from omegaconf import OmegaConf
from torch_geometric.utils import add_remaining_self_loops


PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PACKAGE_ROOT)

from mke_support.data import MoleculeDataset
from mke_support.release import file_sha256, validate_input_csv
from mke_support.utils import check_dir, configure_paths, resolve_input_csv
from mke_support.gnn import get_gnnNets
from mke_support.utils import my_get_dataloader
from mke_support.utils import compatible_state_dict


from mke_support.scoring import BatchedGNNScorer


GPU_CONFIG = {
    'batch_size': 1024,
    'sample_num': 100,
    'exact_threshold': 8,
    'num_workers': 4,
    'pin_memory': True,
    'explain_batch_size': 32,
    'substructure_workers': 8
}


def normalize_candidate_nodes(candidate, num_nodes):

    try:
        candidate = [operator.index(candidate)]
    except TypeError:
        pass
    nodes = tuple(sorted(set(operator.index(node) for node in candidate)))
    if any(node < 0 or node >= num_nodes for node in nodes):
        raise ValueError("Candidate node index is outside the input graph")
    return nodes


class ShapleyGroupGame:


    def __init__(self, candidate_subgraphs, value_batch, num_nodes,
                 exact_threshold=8, sample_num=100, seed=0, batch_size=1024):
        self.num_nodes = operator.index(num_nodes)
        self.exact_threshold = operator.index(exact_threshold)
        self.sample_num = operator.index(sample_num)
        self.batch_size = operator.index(batch_size)
        self.seed = operator.index(seed)
        if self.num_nodes < 0 or self.exact_threshold < 0:
            raise ValueError("num_nodes and exact_threshold must be nonnegative")
        if self.sample_num <= 0 or self.batch_size <= 0:
            raise ValueError("sample_num and batch_size must be positive")
        self.value_batch = value_batch
        candidates, original_indices, seen = [], [], {}
        for original_index, candidate in enumerate(candidate_subgraphs):
            nodes = normalize_candidate_nodes(candidate, self.num_nodes)
            if not nodes:
                continue
            if nodes in seen:
                original_indices[seen[nodes]].append(original_index)
            else:
                seen[nodes] = len(candidates)
                candidates.append(nodes)
                original_indices.append([original_index])
        self.candidate_nodes = tuple(candidates)
        self.original_candidate_indices = original_indices
        self.score_method = ("exact" if len(candidates) <= self.exact_threshold
                             else "monte_carlo")
        self.payoff_cache = {}
        self.score_cache = {(): 0.0}
        self.rng = random.Random(self.seed)
        self.stats = {"payoff_queries": 0, "payoff_batches": 0,
                      "group_scores": 0, "permutations": 0}

    def _group_key(self, candidate_indices):
        key = tuple(sorted(set(operator.index(i) for i in candidate_indices)))
        if any(i < 0 or i >= len(self.candidate_nodes) for i in key):
            raise ValueError("Unknown candidate index")
        return key

    def node_union(self, candidate_indices):
        key = self._group_key(candidate_indices)
        return tuple(sorted(set(itertools.chain.from_iterable(
            self.candidate_nodes[i] for i in key))))

    def _terms(self, group):
        group_nodes = set(self.node_union(group))
        outsiders = [i for i in range(len(self.candidate_nodes)) if i not in group]
        k = len(outsiders)
        backgrounds = []
        if self.score_method == "exact":
            denominator = math.factorial(k + 1)
            for size in range(k + 1):
                weight = (math.factorial(size) * math.factorial(k - size)
                          / denominator)
                backgrounds.extend((subset, weight) for subset in
                                   itertools.combinations(outsiders, size))
        else:

            for _ in range(self.sample_num):
                permutation = outsiders + [-1]
                self.rng.shuffle(permutation)
                prefix = permutation[:permutation.index(-1)]
                backgrounds.append((prefix, 1.0 / self.sample_num))
            self.stats["permutations"] += self.sample_num
        terms = []
        for subset, weight in backgrounds:
            before = self.node_union(subset)
            after = tuple(sorted(set(before).union(group_nodes)))


            terms.append((before, after, weight))
        return terms

    def scores(self, groups):
        keys = [self._group_key(group) for group in groups]
        pending, required = {}, {}
        for key in keys:
            if key in self.score_cache or key in pending:
                continue
            terms = self._terms(key)
            pending[key] = terms
            for before, after, _ in terms:
                for nodes in (before, after):
                    if nodes not in self.payoff_cache:
                        required[nodes] = None
        queries = list(required)
        for start in range(0, len(queries), self.batch_size):
            chunk = queries[start:start + self.batch_size]
            values = list(self.value_batch([list(nodes) for nodes in chunk]))
            if len(values) != len(chunk):
                raise ValueError("value_batch must return one payoff per node union")
            values = [float(value) for value in values]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("value_batch returned a non-finite payoff")
            self.payoff_cache.update(zip(chunk, values))
            self.stats["payoff_queries"] += len(chunk)
            self.stats["payoff_batches"] += 1
        for key, terms in pending.items():
            self.score_cache[key] = math.fsum(
                weight * (self.payoff_cache[after] - self.payoff_cache[before])
                for before, after, weight in terms)
            self.stats["group_scores"] += 1
        return [self.score_cache[key] for key in keys]

    def score(self, group):
        return self.scores([group])[0]


def greedy_group_select(game, max_nodes):

    max_nodes = operator.index(max_nodes)
    if max_nodes < 0:
        raise ValueError("max_nodes must be nonnegative")
    selected, current_score = (), 0.0
    history, proposals = [], []
    while True:
        feasible = []
        for candidate_id in range(len(game.candidate_nodes)):
            if candidate_id in selected:
                continue
            group = tuple(sorted(selected + (candidate_id,)))
            union = game.node_union(group)
            if len(union) <= max_nodes:
                feasible.append((candidate_id, group, union))
        if not feasible:
            stop_reason = "no_feasible_candidate"
            break
        scores = game.scores([item[1] for item in feasible])
        best_index, best_gain = None, 0.0
        round_records = []
        for i, ((candidate_id, group, union), score) in enumerate(zip(feasible, scores)):
            gain = score - current_score
            record = {"iteration": len(history) + 1,
                      "selected_before": list(selected),
                      "added_candidate_index": candidate_id,
                      "proposed_candidate_indices": list(group),
                      "proposed_nodes": list(union),
                      "node_count": len(union),
                      "previous_group_score": current_score,
                      "group_score": score, "marginal_gain": gain,
                      "accepted": False}
            round_records.append(record)
            if gain > best_gain:
                best_index, best_gain = i, gain
        proposals.extend(round_records)
        if best_index is None:
            stop_reason = "no_positive_gain"
            break
        round_records[best_index]["accepted"] = True
        history.append(dict(round_records[best_index]))
        selected = feasible[best_index][1]
        current_score = scores[best_index]
    return {"selected_candidate_indices": list(selected),
            "selected_nodes": list(game.node_union(selected)),
            "group_score": current_score, "history": history,
            "proposals": proposals, "stop_reason": stop_reason}


class GroupShapleyExplainer:


    def __init__(self, model, num_classes, device, subgraph_building_method,
                 batch_size=1024, exact_threshold=8, sample_num=100, seed=0):
        self.scorer = BatchedGNNScorer(model, device, subgraph_building_method,
                                     batch_size=batch_size)
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.exact_threshold = exact_threshold
        self.sample_num = sample_num
        self.seed = operator.index(seed)
        self.last_traces = []

    def explain_batch(self, inputs):
        results, self.last_traces = [], []
        for item in inputs:
            data, label = item["data"], int(item["label"])
            num_nodes = int(data.num_nodes)
            sparsity = float(item["node_size"])
            if not math.isfinite(sparsity) or not 0.0 <= sparsity <= 1.0:
                raise ValueError("node_size must be a sparsity in [0, 1]")
            if not 0 <= label < self.num_classes:
                raise ValueError("Invalid target class")
            max_nodes = math.floor((1.0 - sparsity) * num_nodes)
            sample_index = int(item["sample_index"])
            graph_seed = self.seed + 1000003 * sample_index


            base_prob = float(self.scorer.batch_score([[]], data, label)[0])
            if not math.isfinite(base_prob) or base_prob <= 0.0:
                raise ValueError("The clean target probability must be finite and positive")

            def value_batch(unions):
                nonempty = [nodes for nodes in unions if nodes]
                scores = self.scorer.batch_score(nonempty, data, label)
                if len(scores) != len(nonempty):
                    raise ValueError("The GNN scorer returned an unexpected number of probabilities")
                probs = iter(scores)

                return [1.0 - float(next(probs)) / base_prob if nodes else 0.0
                        for nodes in unions]

            game = ShapleyGroupGame(
                item["candidate_subgraphs"], value_batch, num_nodes,
                exact_threshold=self.exact_threshold, sample_num=self.sample_num,
                seed=graph_seed, batch_size=self.batch_size)
            selection = greedy_group_select(game, max_nodes)
            nodes = selection["selected_nodes"]


            if nodes:
                payoff = game.payoff_cache.get(tuple(nodes))
                if payoff is None:
                    explain_value = float(self.scorer.batch_score([nodes], data, label)[0])
                else:
                    explain_value = base_prob * (1.0 - payoff)
                explain_no_mask = float(self.scorer.batch_score_inverse([nodes], data, label)[0])
            else:
                explain_value = base_prob
                explain_no_mask = float(self.scorer.batch_score_inverse([[]], data, label)[0])
            if not all(math.isfinite(p) for p in (explain_value, explain_no_mask)):
                raise ValueError("Non-finite explanation probability")
            truelist = list(item["truelist"])
            input_base_prob = float(truelist[label])
            truelist[label] = base_prob
            trace = dict(selection)
            trace.update({"sample_index": sample_index, "target_class": label,
                          "num_nodes": num_nodes, "node_budget": max_nodes,
                          "target_sparsity": sparsity,
                          "actual_sparsity": 1.0 - len(nodes) / num_nodes if num_nodes else 1.0,
                          "score_method": game.score_method, "seed": graph_seed,
                          "sample_num": self.sample_num,
                          "exact_threshold": self.exact_threshold,
                          "candidate_nodes": [list(c) for c in game.candidate_nodes],
                          "original_candidate_indices": game.original_candidate_indices,
                          "stats": dict(game.stats),
                          "input_target_probability": input_base_prob,
                          "scoring_target_probability": base_prob,
                          "masked_probability": explain_value,
                          "retained_probability": explain_no_mask,
                          "fidelity_plus": base_prob - explain_value,
                          "fidelity_minus": base_prob - explain_no_mask,
                          "subgraph_building_method": self.scorer.subgraph_building_method,
                          "probability_precision": "shared_scorer_amp"})
            self.last_traces.append(trace)
            results.append((nodes, explain_value, explain_no_mask, truelist, data))
        return results


SUB_SMILES = [
    "C1(=O)CCC(=O)CC1", "C(=O)OC", "S(=O)(=O)", "C(=O)N",
    "C(=O)O", "COC", "C(=O)", "N(=O)O", "O", "N", "F", "S", "Cl"
]


def custom_node_match(n1, n2):

    return n1['atom'] == n2['atom']


def extract_atoms(smiles):

    molecule = Chem.MolFromSmiles(smiles)
    atoms = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    return atoms


def edge_index_to_adjacency_matrix(edge_index, num_nodes):

    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i]
        dst = edge_index[1, i]
        adjacency_matrix[src, dst] = 1
    return adjacency_matrix


def get_atom_list(smiles):

    molecule = Chem.MolFromSmiles(smiles)
    atom_list = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    return atom_list


def find_cycles(atom_list, adj_matrix, boolean_array):

    graph = nx.Graph()
    for i in range(len(atom_list)):
        graph.add_node(i, atom=atom_list[i])
    adj = np.asarray(adj_matrix)
    n = adj.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 1:
                graph.add_edge(i, j)

    cycles = nx.cycle_basis(graph)

    for sublist in cycles:
        for element in sublist:
            if not boolean_array[element]:
                boolean_array[element] = True

    orig_cycles = list(cycles)
    extra = []

    for i in range(len(orig_cycles)):
        cyc = orig_cycles[i]
        L = len(cyc)

        if L == 6:
            extra.append([cyc[0], cyc[1]])
            extra.append([cyc[2], cyc[3]])
            extra.append([cyc[4], cyc[5]])
            extra.append([cyc[0], cyc[5]])
            extra.append([cyc[1], cyc[2]])
            extra.append([cyc[3], cyc[4]])
        elif L == 5:
            extra.append([cyc[0], cyc[1]])
            extra.append([cyc[2], cyc[3], cyc[4]])
            extra.append([cyc[1], cyc[2]])
            extra.append([cyc[3], cyc[4], cyc[0]])
            extra.append([cyc[3], cyc[4]])
            extra.append([cyc[0], cyc[1], cyc[2]])

    return orig_cycles + extra


def dfs(adjacency_matrix, start_atom, visited):

    stack = [start_atom]
    subgraph = [start_atom]

    while stack:
        current_atom = stack.pop()
        visited.add(current_atom)
        neighbors = np.nonzero(adjacency_matrix[current_atom])[0]

        for neighbor in neighbors:
            if neighbor not in visited:
                subgraph.append(neighbor)
                stack.append(neighbor)

    return subgraph


def find_connected_subgraphs(adjacency_matrix, boolean_array):

    adjacency_matrix = np.array(adjacency_matrix)

    known_indices = []
    unknown_indices = []

    for index, value in enumerate(boolean_array):
        if value:
            known_indices.append(index)
        else:
            unknown_indices.append(index)

    for i in known_indices:
        adjacency_matrix[i, :] = 0
        adjacency_matrix[:, i] = 0

    connected_subgraphs = []
    visited = set()

    for atom_idx in unknown_indices:
        if atom_idx not in visited:
            subgraph = dfs(adjacency_matrix, atom_idx, visited)
            connected_subgraphs.append(subgraph)

    return connected_subgraphs


def smiles_to_graph(smiles0):

    atom_list0 = extract_atoms(smiles0)
    mol0 = Chem.MolFromSmiles(smiles0)
    adj_matrix0 = Chem.GetAdjacencyMatrix(mol0)

    original_graph = nx.Graph()
    for i in range(len(atom_list0)):
        original_graph.add_node(i, atom=atom_list0[i])
    for i in range(len(adj_matrix0)):
        for j in range(i + 1, len(adj_matrix0[i])):
            if adj_matrix0[i][j] == 1:
                original_graph.add_edge(i, j)
    return original_graph


def find_substructure(graph1, original_adj_matrix, atom_list, smiles_list, boolean_array):

    fs1_graph = graph1
    result = []

    res = find_cycles(atom_list=atom_list, adj_matrix=original_adj_matrix, boolean_array=boolean_array)
    result.append(res)

    for smiles in smiles_list:
        sub_atom_list = extract_atoms(smiles)
        mol = Chem.MolFromSmiles(smiles)
        adj_matrix = Chem.GetAdjacencyMatrix(mol)

        fs2_graph = nx.Graph()
        for i in range(len(sub_atom_list)):
            fs2_graph.add_node(i, atom=sub_atom_list[i])
        for i in range(len(adj_matrix)):
            for j in range(i + 1, len(adj_matrix[i])):
                if adj_matrix[i][j] == 1:
                    fs2_graph.add_edge(i, j)

        gm = nx.algorithms.isomorphism.GraphMatcher(fs1_graph, fs2_graph, node_match=custom_node_match)

        atom_indices = []
        for subgraph in gm.subgraph_isomorphisms_iter():
            if len(subgraph) == len(sub_atom_list):
                atom_indices.append(list(subgraph.keys()))

        result_list = []
        for sublist in atom_indices:
            flag = 0
            for element in sublist:
                if not boolean_array[element]:
                    flag = 1
                    boolean_array[element] = True
            if flag == 1:
                result_list.append(sublist)

        if len(sub_atom_list) == 1 and len(result_list) != 0:
            connections = []
            for element in result_list:
                for ato in element:
                    if sum(original_adj_matrix[ato]) >= 2:
                        connection = [i for i, value in enumerate(original_adj_matrix[ato]) if value == 1]
                        connection.append(ato)
                        connections.append(connection)
            result_list.extend(connections)
        result.append(result_list)

    remain_list = find_connected_subgraphs(original_adj_matrix, boolean_array)
    result.append(remain_list)

    result_tmp = [arr.copy() for arr in remain_list]
    all_numbers = []
    for arr in result_tmp:
        all_numbers.extend(arr)

    for num in all_numbers:

        result.append([[int(num)]])

    result_list = []
    for element in result:
        if len(element) != 0:
            result_list.extend(element)

    return result_list


def _extract_substructure_worker(smiles, original_adj_matrix):

    original_graph = smiles_to_graph(smiles)
    atom_list = extract_atoms(smiles)
    boolean_array = [False] * len(atom_list)
    return find_substructure(original_graph, original_adj_matrix, atom_list, SUB_SMILES, boolean_array)


def count_atoms(smiles):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("无法解析的 SMILES")
    return mol.GetNumAtoms()


def subgraph_nodes_to_smiles(smiles, node_indices):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        atom_indices = sorted({int(i) for i in node_indices if i is not None})
        if not atom_indices:
            return None
        return Chem.MolFragmentToSmiles(mol, atomsToUse=atom_indices, isomericSmiles=True)
    except Exception:
        return None


@hydra.main(config_path="config", config_name="config")
def pipeline(config):

    configure_paths(config, PACKAGE_ROOT)
    node_size = float(config.target_sparsity)
    if not math.isfinite(node_size) or not 0.0 <= node_size <= 1.0:
        raise ValueError("target_sparsity must be finite and in [0, 1]")
    for key, value in config.performance.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("performance.{} must be a positive integer".format(key))
        GPU_CONFIG[key] = value
    config.models.param = config.models.param[config.datasets.dataset_name]
    config.explainers.param = config.explainers.param[config.datasets.dataset_name]
    config.models.param.add_self_loop = False

    if not os.path.isdir(config.record_filename):
        os.makedirs(config.record_filename)
    config.record_filename = os.path.join(config.record_filename, f"{config.datasets.dataset_name}.json")
    print(OmegaConf.to_yaml(config))


    if torch.cuda.is_available():
        if not 0 <= int(config.device_id) < torch.cuda.device_count():
            raise ValueError("device_id is outside the visible CUDA device range")
        device = torch.device('cuda', index=config.device_id)

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"使用 GPU: {torch.cuda.get_device_name(device)}")
        print(f"显存: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        print("警告: 未检测到 GPU，使用 CPU")


    group = ["training", "valid", "test"]
    input_csv = resolve_input_csv(config, PACKAGE_ROOT)
    labels_name = str(config.datasets.label_column)
    input_report = validate_input_csv(input_csv, labels_name)
    data_origin = pd.read_csv(input_csv)

    train_index = []
    test_index = []
    valid_index = []
    dataset = MoleculeDataset()
    dataset.build_mol_graph_data(
        dataset_smiles=data_origin,
        train_index=train_index,
        test_index=test_index,
        valid_index=valid_index,
        labels_name=labels_name,
        group=group
    )

    dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long()
    index_list = [train_index, test_index, valid_index]

    if config.models.param.graph_classification:
        index_list_params = {
            'batch_size': config.models.param.batch_size,
            'index_list': index_list,
            'seed': config.datasets.seed
        }
        loader = my_get_dataloader(dataset, index_list_params)
        test_indices = loader['test'].dataset.indices


    model = get_gnnNets(
        input_dim=dataset.num_node_features,
        output_dim=dataset.num_classes,
        model_config=config.models
    )

    checkpoint_path = os.path.join(
        config.models.gnn_saving_dir,
        config.datasets.dataset_name,
        f"{config.models.gnn_name}_{len(config.models.param.gnn_latent_dim)}l_best.pth"
    )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError("Required checkpoint not found: {}".format(checkpoint_path))
    state_dict = compatible_state_dict(torch.load(checkpoint_path, map_location='cpu')['net'])

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()


    sampling_seed = int(config.random_seed)
    explainer = GroupShapleyExplainer(
        model=model,
        num_classes=dataset.num_classes,
        device=device,
        subgraph_building_method=config.explainers.param.subgraph_building_method,
        batch_size=GPU_CONFIG['batch_size'],
        exact_threshold=GPU_CONFIG['exact_threshold'],
        sample_num=GPU_CONFIG['sample_num'],
        seed=sampling_seed
    )

    method = "mke_new_group_shapley"
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    explanation_saving_dir = os.path.join(
        config.explainers.explanation_result_dir,
        config.datasets.dataset_name,
        config.models.gnn_name,
        method,
        config.explainers.param.reward_method,
        run_id
    )
    check_dir(explanation_saving_dir)
    trace_path = os.path.join(explanation_saving_dir, "group_selection_trace.jsonl")
    metadata = {
        'algorithm': method,
        'objective': 'merged-player Shapley group value; largest positive gain',
        'dataset': config.datasets.dataset_name,
        'input_csv': os.path.abspath(input_csv),
        'input_sha256': input_report['sha256'],
        'input_report': input_report,
        'label_column': labels_name,
        'num_valid_graphs': len(dataset),
        'num_skipped_rows': input_report['count'] - len(dataset),
        'checkpoint': os.path.abspath(checkpoint_path),
        'checkpoint_sha256': file_sha256(checkpoint_path),
        'implementation_sha256': file_sha256(__file__),
        'source_sha256': {
            name: file_sha256(os.path.join(PACKAGE_ROOT, name))
            for name in ['mke_new.py'] + [
                'mke_support/' + filename for filename in sorted(os.listdir(os.path.join(PACKAGE_ROOT, 'mke_support')))
                if filename.endswith('.py')
            ]
        },
        'software_versions': dict(
            {'python': sys.version},
            **{name: getattr(sys.modules.get(name), '__version__', 'unknown')
               for name in ('torch', 'torch_geometric', 'rdkit', 'numpy', 'pandas', 'networkx', 'hydra', 'omegaconf')}
        ),
        'test_count': len(test_indices),
        'target_sparsity': node_size,
        'node_budget_rule': 'floor((1 - target_sparsity) * num_nodes)',
        'candidate_normalization': 'unique nonempty node sets; preserve first occurrence',
        'exact_threshold': GPU_CONFIG['exact_threshold'],
        'threshold_applies_to': 'original deduplicated candidate count',
        'mc_permutations_per_group': GPU_CONFIG['sample_num'],
        'sampling_seed': sampling_seed,
        'per_graph_seed_rule': 'sampling_seed + 1000003 * dataset_index',
        'mc_cache': 'one estimate per candidate-ID group within each graph',
        'subgraph_building_method': config.explainers.param.subgraph_building_method,
        'probability_precision': 'shared AMP scorer for clean and perturbed graphs',
        'target_class': 'fixed clean predicted class',
        'metric_scope': 'all_test_graphs_including_empty_explanations',
        'gpu_config': GPU_CONFIG,
        'run_id': run_id,
    }
    with open(os.path.join(explanation_saving_dir, "run_metadata.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2, allow_nan=False)
    with open(os.path.join(explanation_saving_dir, "run_config.yaml"), "w", encoding="utf-8") as file:
        file.write(OmegaConf.to_yaml(config))


    score_list = []
    nomask_score_list = []
    sparsity_list = []
    explanation_rows = []

    start_time = time.time()
    print(f"开始解释 {len(test_indices)} 个分子...")
    print(f"GPU 配置: batch_size={GPU_CONFIG['batch_size']}, "
          f"exact_threshold={GPU_CONFIG['exact_threshold']}, "
          f"sample_num={GPU_CONFIG['sample_num']}, seed={sampling_seed}")
    print(f"选择目标: Shapley group value；结果目录: {explanation_saving_dir}")


    explain_batch_size = GPU_CONFIG.get('explain_batch_size', 4)
    substructure_workers = GPU_CONFIG.get('substructure_workers', 1)
    total_samples = len(test_indices)
    for batch_start in range(0, total_samples, explain_batch_size):
        batch_end = min(batch_start + explain_batch_size, total_samples)
        batch_indices = test_indices[batch_start:batch_end]
        batch_data = [dataset[idx] for idx in batch_indices]


        original_edge_index_list = [data.edge_index.clone() for data in batch_data]
        original_adj_matrices = [
            edge_index_to_adjacency_matrix(original_edge_index_list[i], len(batch_data[i].x))
            for i in range(len(batch_data))
        ]


        if substructure_workers > 1 and len(batch_data) > 1:
            max_workers = min(substructure_workers, len(batch_data))
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                substructure_results = list(executor.map(
                    _extract_substructure_worker,
                    [data.smiles for data in batch_data],
                    original_adj_matrices
                ))
        else:
            substructure_results = [
                _extract_substructure_worker(data.smiles, original_adj_matrices[i])
                for i, data in enumerate(batch_data)
            ]


        for data in batch_data:
            data.edge_index = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]

        from torch_geometric.data import Batch
        batch = Batch.from_data_list(batch_data).to(device)
        with torch.no_grad():

            batch_logits = model(batch.x, batch.edge_index, batch.batch)
            batch_probs = torch.softmax(batch_logits, dim=-1).cpu()
            batch_pred = batch_probs.argmax(-1).tolist()
            batch_truelist = batch_probs.tolist()

        explain_inputs = []
        for local_idx, data in enumerate(batch_data):
            data.to(device)
            explain_inputs.append({
                'x': data.x,
                'edge_index': data.edge_index,
                'label': batch_pred[local_idx],
                'candidate_subgraphs': substructure_results[local_idx],
                'data': data,
                'node_size': node_size,
                'truelist': batch_truelist[local_idx],
                'sample_index': int(batch_indices[local_idx])
            })


        batch_results = explainer.explain_batch(explain_inputs)

        with open(trace_path, "a", encoding="utf-8") as file:
            for trace in explainer.last_traces:
                file.write(json.dumps(trace, ensure_ascii=False, allow_nan=False) + "\n")

        for local_idx, data in enumerate(batch_data):
            global_idx = batch_start + local_idx
            data_smiles = data.smiles
            print(f"[{global_idx + 1}/{total_samples}] {data_smiles}")

            result_list, explain_value, explain_no_mask, truelist, ori_graph = batch_results[local_idx]

            prediction = batch_pred[local_idx]
            explain_nodes = [int(i) for i in result_list]
            explain_smiles = subgraph_nodes_to_smiles(data_smiles, explain_nodes)
            trace = explainer.last_traces[local_idx]
            explanation_rows.append({
                'sample_index': int(batch_indices[local_idx]),
                'molecule_smiles': data_smiles,
                'explanation_smiles': explain_smiles,
                'explanation_nodes': json.dumps(explain_nodes),
                'label': int(data.y.item()),
                'num_nodes': int(data.num_nodes),
                'prediction': int(prediction),
                'target_probability': float(truelist[prediction]),
                'explain_value': float(explain_value),
                'explain_no_mask': float(explain_no_mask),
                'fidelity_plus': float(truelist[prediction] - explain_value),
                'fidelity_minus': float(truelist[prediction] - explain_no_mask),
                'sparsity': trace['actual_sparsity'],
                'target_sparsity': node_size,
                'node_budget': trace['node_budget'],
                'candidate_count': len(trace['candidate_nodes']),
                'selected_candidate_indices': json.dumps(trace['selected_candidate_indices']),
                'group_score': trace['group_score'],
                'score_method': trace['score_method'],
                'sampling_seed': trace['seed'],
                'empty_explanation': not bool(explain_nodes),
                'stop_reason': trace['stop_reason']
            })

            score_list.append(truelist[prediction] - explain_value)
            nomask_score_list.append(truelist[prediction] - explain_no_mask)
            sparsity_list.append(trace['actual_sparsity'])


            processed = global_idx + 1
            if processed % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / processed
                remaining = avg_time * (total_samples - processed)

                if torch.cuda.is_available():
                    mem_used = torch.cuda.memory_allocated(device) / 1024**3
                    mem_cached = torch.cuda.memory_reserved(device) / 1024**3
                    print(f"  进度: {processed}/{total_samples}, "
                          f"平均耗时: {avg_time:.2f}s/分子, "
                          f"预计剩余: {remaining/60:.1f}min, "
                          f"GPU显存: {mem_used:.1f}GB/{mem_cached:.1f}GB")


    if explanation_rows:
        explanations_path = os.path.join(explanation_saving_dir, "mke_new_explanations.csv")
        pd.DataFrame(explanation_rows).to_csv(explanations_path, index=False)
        print(f"解释子图已保存到: {explanations_path}")
    else:
        raise ValueError("No test graphs were evaluated; cannot report dataset means")


    end_time = time.time()
    total_time = end_time - start_time

    score_list = np.array(score_list)
    nomask_score_list = np.array(nomask_score_list)
    sparsity_list = np.array(sparsity_list)

    fidelity = np.mean(score_list)
    sparsity = np.mean(sparsity_list)
    fidelity_inv = np.mean(nomask_score_list)

    print("\n" + "="*50)
    print(f"总耗时: {total_time:.2f}s ({total_time/60:.2f}min)")
    print(f"平均每分子: {total_time/len(test_indices):.2f}s")
    print(f"保真度 Fidelity+: {fidelity:.4f}")
    print(f"保真度 Fidelity-: {fidelity_inv:.4f}")
    print(f"稀疏度 Sparsity: {sparsity:.4f}")
    empty_count = sum(row['empty_explanation'] for row in explanation_rows)
    print(f"统计样本: {len(explanation_rows)}/{total_samples}，其中空解释: {empty_count}")
    print(f"逐轮 group-value 日志: {trace_path}")
    print("="*50)


    results_row = {
        'dataset_name': config.datasets.dataset_name,
        'method': method,
        'fidelity_plus': float(fidelity),
        'fidelity_minus': float(fidelity_inv),
        'sparsity': float(sparsity),
        'target_sparsity': node_size,
        'predictor_accuracy': sum(row['label'] == row['prediction'] for row in explanation_rows) / len(explanation_rows),
        'total_time_sec': float(total_time),
        'avg_time_per_mol': float(total_time/len(test_indices)),
        'num_evaluated': len(explanation_rows),
        'num_empty_explanations': empty_count,
        'metric_scope': metadata['metric_scope'],
        'sampling_seed': sampling_seed,
        'run_id': run_id,
        'result_directory': explanation_saving_dir,
        'gpu_config': str(GPU_CONFIG),
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    }
    with open(os.path.join(explanation_saving_dir, "summary.json"), "w", encoding="utf-8") as file:
        json.dump(results_row, file, ensure_ascii=False, indent=2, allow_nan=False)

    results_df = pd.DataFrame([results_row])
    excel_path = os.path.join(os.path.dirname(__file__), "mke_new_group_shapley_results.xlsx")

    if os.path.exists(excel_path):

        existing_df = pd.read_excel(excel_path)
        combined_df = pd.concat([existing_df, results_df], ignore_index=True)
    else:
        combined_df = results_df

    combined_df.to_excel(excel_path, index=False)
    print(f"结果已保存到: {excel_path}")


if __name__ == "__main__":
    pipeline()
