import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data, Batch
from typing import List, Tuple


def get_graph_build_func(build_method: str):

    if build_method == 'zero_filling':
        return graph_build_zero_filling
    elif build_method == 'split':
        return graph_build_split
    else:
        raise NotImplementedError(f"Unknown build method: {build_method}")


def graph_build_zero_filling(x: Tensor, edge_index: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:

    ret_x = x * mask.unsqueeze(1)
    return ret_x, edge_index


def graph_build_split(x: Tensor, edge_index: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:

    row, col = edge_index
    edge_mask = mask[row].bool() & mask[col].bool()
    ret_edge_index = edge_index[:, edge_mask]
    return x, ret_edge_index


class BatchedGNNScorer:


    def __init__(self, model, device, subgraph_building_method='zero_filling', batch_size=64):
        self.model = model
        self.model.eval()
        self.device = device
        self.subgraph_building_method = subgraph_building_method
        self.batch_size = batch_size
        self.subgraph_build_func = get_graph_build_func(subgraph_building_method)

    @torch.no_grad()
    def batch_score(self, coalitions: List[List[int]], data: Data, label: int) -> List[float]:

        if not coalitions:
            return []

        num_nodes = data.num_nodes
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)

        all_scores = []


        for batch_start in range(0, len(coalitions), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(coalitions))
            batch_coalitions = coalitions[batch_start:batch_end]


            data_list = []
            for coalition in batch_coalitions:

                mask = torch.ones(num_nodes, device=self.device)
                if coalition:
                    mask[coalition] = 0.0


                ret_x, ret_edge_index = self.subgraph_build_func(x, edge_index, mask)
                data_list.append(Data(x=ret_x, edge_index=ret_edge_index))


            batch_data = Batch.from_data_list(data_list)
            batch_data = batch_data.to(self.device)


            with torch.cuda.amp.autocast(enabled=True):

                logits = self.model(batch_data.x, batch_data.edge_index, batch_data.batch)
                probs = F.softmax(logits, dim=-1)


            batch_scores = probs[:, label].cpu().tolist()
            all_scores.extend(batch_scores)

        return all_scores

    @torch.no_grad()
    def batch_score_multi(self, items: List[Tuple[List[int], Data, int]]) -> List[float]:

        if not items:
            return []

        all_scores = []

        for batch_start in range(0, len(items), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(items))
            batch_items = items[batch_start:batch_end]

            data_list = []
            labels = []
            for coalition, data, label in batch_items:
                num_nodes = data.num_nodes
                x = data.x.to(self.device)
                edge_index = data.edge_index.to(self.device)


                if coalition:
                    coalition = [int(i) for i in coalition if 0 <= int(i) < num_nodes]


                mask = torch.ones(num_nodes, device=self.device)
                if coalition:
                    mask[coalition] = 0.0

                ret_x, ret_edge_index = self.subgraph_build_func(x, edge_index, mask)
                data_list.append(Data(x=ret_x, edge_index=ret_edge_index))
                labels.append(label)

            batch_data = Batch.from_data_list(data_list)
            batch_data = batch_data.to(self.device)

            with torch.cuda.amp.autocast(enabled=True):

                logits = self.model(batch_data.x, batch_data.edge_index, batch_data.batch)
                probs = F.softmax(logits, dim=-1)

            label_tensor = torch.tensor(labels, device=probs.device)

            label_tensor = label_tensor.clamp(0, probs.size(1) - 1)
            idx = torch.arange(len(labels), device=probs.device)
            batch_scores = probs[idx, label_tensor].cpu().tolist()
            all_scores.extend(batch_scores)

        return all_scores

    @torch.no_grad()
    def batch_score_inverse(self, coalitions: List[List[int]], data: Data, label: int) -> List[float]:

        if not coalitions:
            return []

        num_nodes = data.num_nodes
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)

        all_scores = []

        for batch_start in range(0, len(coalitions), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(coalitions))
            batch_coalitions = coalitions[batch_start:batch_end]

            data_list = []
            for coalition in batch_coalitions:

                mask = torch.zeros(num_nodes, device=self.device)
                if coalition:
                    mask[coalition] = 1.0

                ret_x, ret_edge_index = self.subgraph_build_func(x, edge_index, mask)
                data_list.append(Data(x=ret_x, edge_index=ret_edge_index))

            batch_data = Batch.from_data_list(data_list)
            batch_data = batch_data.to(self.device)

            with torch.cuda.amp.autocast(enabled=True):

                logits = self.model(batch_data.x, batch_data.edge_index, batch_data.batch)
                probs = F.softmax(logits, dim=-1)

            batch_scores = probs[:, label].cpu().tolist()
            all_scores.extend(batch_scores)

        return all_scores
