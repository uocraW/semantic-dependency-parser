# -*- coding: utf8 -*-
#
import torch
from torch import nn
import numpy

from layers.affine import Biaffine
from layers.mlp import MLP
from layers.transformer import TransformerEmbedding

from typing import Tuple, Dict, List, Set


class SemanticDependencyModel(nn.Module):
    def __init__(self, transformer: str, n_labels: int, n_tags: int=0,
                 n_edge_mlp=600, n_label_mlp=600, enable_tag=True):
        super(SemanticDependencyModel, self).__init__()

        # 控制是否把tag_emb加上
        self.enable_tag = enable_tag

        self.encoder = TransformerEmbedding(model=transformer,
                                            n_layers=4,
                                            pooling='mean',
                                            pad_index=0,
                                            dropout=0.33,
                                            requires_grad=True)

        if self.enable_tag:
            self.tag_embedding = nn.Embedding(num_embeddings=n_tags, embedding_dim=64)
            self.lstm = nn.LSTM(
                input_size=self.encoder.n_out + self.tag_embedding.embedding_dim,
                hidden_size=400,
                num_layers=3,
                dropout=0.33,
                batch_first=True,
                bidirectional=True
            )
        else:
                self.lstm = nn.LSTM(
                input_size=self.encoder.n_out,
                hidden_size=400,
                num_layers=3,
                dropout=0.33,
                batch_first=True,
                bidirectional=True
            )

        self.edge_mlp_d = MLP(n_in=800, n_out=n_edge_mlp, dropout=0.33)
        self.edge_mlp_h = MLP(n_in=800, n_out=n_edge_mlp, dropout=0.33)

        self.label_mlp_d = MLP(n_in=800, n_out=n_label_mlp, dropout=0.33)
        self.label_mlp_h = MLP(n_in=800, n_out=n_label_mlp, dropout=0.33)

        self.edge_attn = Biaffine(n_in=n_edge_mlp, n_out=2, bias_x=True, bias_y=True)
        self.label_attn = Biaffine(n_in=n_label_mlp, n_out=n_labels, bias_x=True, bias_y=True)
        self.criterion = nn.CrossEntropyLoss()

    def decode_mst(
        self, energy: numpy.ndarray, length: int, has_labels: bool = True
        ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Note: Counter to typical intuition, this function decodes the _maximum_
        spanning tree.

        Decode the optimal MST tree with the Chu-Liu-Edmonds algorithm for
        maximum spanning arborescences on graphs.

        # Parameters

        energy : `numpy.ndarray`, required.
            A tensor with shape (num_labels, timesteps, timesteps)
            containing the energy of each edge. If has_labels is `False`,
            the tensor should have shape (timesteps, timesteps) instead.
        length : `int`, required.
            The length of this sequence, as the energy may have come
            from a padded batch.
        has_labels : `bool`, optional, (default = `True`)
            Whether the graph has labels or not.
        """
        if has_labels and energy.ndim != 3:
            raise ValueError("The dimension of the energy array is not equal to 3.")
        elif not has_labels and energy.ndim != 2:
            raise ValueError("The dimension of the energy array is not equal to 2.")
        input_shape = energy.shape
        max_length = input_shape[-1]

        # Our energy matrix might have been batched -
        # here we clip it to contain only non padded tokens.
        if has_labels:
            energy = energy[:, :length, :length]
            # get best label for each edge.
            label_id_matrix = energy.argmax(axis=0)
            energy = energy.max(axis=0)
        else:
            energy = energy[:length, :length]
            label_id_matrix = None
        # get original score matrix
        original_score_matrix = energy
        # initialize score matrix to original score matrix
        score_matrix = numpy.array(original_score_matrix, copy=True)

        old_input = numpy.zeros([length, length], dtype=numpy.int32)
        old_output = numpy.zeros([length, length], dtype=numpy.int32)
        current_nodes = [True for _ in range(length)]
        representatives: List[Set[int]] = []

        for node1 in range(length):
            original_score_matrix[node1, node1] = 0.0
            score_matrix[node1, node1] = 0.0
            representatives.append({node1})

            for node2 in range(node1 + 1, length):
                old_input[node1, node2] = node1
                old_output[node1, node2] = node2

                old_input[node2, node1] = node2
                old_output[node2, node1] = node1

        final_edges: Dict[int, int] = {}

        # The main algorithm operates inplace.
        self.chu_liu_edmonds(
            length, score_matrix, current_nodes, final_edges, old_input, old_output, representatives
        )

        heads = numpy.zeros([max_length], numpy.int32)
        if has_labels:
            head_type = numpy.ones([max_length], numpy.int32)
        else:
            head_type = None

        for child, parent in final_edges.items():
            heads[child] = parent
            if has_labels:
                head_type[child] = label_id_matrix[parent, child]

        return heads, head_type

    def chu_liu_edmonds(
        self, length: int,
        score_matrix: numpy.ndarray,
        current_nodes: List[bool],
        final_edges: Dict[int, int],
        old_input: numpy.ndarray,
        old_output: numpy.ndarray,
        representatives: List[Set[int]],
):
        """
        Applies the chu-liu-edmonds algorithm recursively
        to a graph with edge weights defined by score_matrix.

        Note that this function operates in place, so variables
        will be modified.

        # Parameters

        length : `int`, required.
            The number of nodes.
        score_matrix : `numpy.ndarray`, required.
            The score matrix representing the scores for pairs
            of nodes.
        current_nodes : `List[bool]`, required.
            The nodes which are representatives in the graph.
            A representative at it's most basic represents a node,
            but as the algorithm progresses, individual nodes will
            represent collapsed cycles in the graph.
        final_edges : `Dict[int, int]`, required.
            An empty dictionary which will be populated with the
            nodes which are connected in the maximum spanning tree.
        old_input : `numpy.ndarray`, required.
        old_output : `numpy.ndarray`, required.
        representatives : `List[Set[int]]`, required.
            A list containing the nodes that a particular node
            is representing at this iteration in the graph.

        # Returns

        Nothing - all variables are modified in place.

        """
        # Set the initial graph to be the greedy best one.
        parents = [-1]
        for node1 in range(1, length):
            parents.append(0)
            if current_nodes[node1]:
                max_score = score_matrix[0, node1]
                for node2 in range(1, length):
                    if node2 == node1 or not current_nodes[node2]:
                        continue

                    new_score = score_matrix[node2, node1]
                    if new_score > max_score:
                        max_score = new_score
                        parents[node1] = node2

        # Check if this solution has a cycle.
        has_cycle, cycle = self._find_cycle(parents, length, current_nodes)
        # If there are no cycles, find all edges and return.
        if not has_cycle:
            final_edges[0] = -1
            for node in range(1, length):
                if not current_nodes[node]:
                    continue

                parent = old_input[parents[node], node]
                child = old_output[parents[node], node]
                final_edges[child] = parent
            return

        # Otherwise, we have a cycle so we need to remove an edge.
        # From here until the recursive call is the contraction stage of the algorithm.
        cycle_weight = 0.0
        # Find the weight of the cycle.
        index = 0
        for node in cycle:
            index += 1
            cycle_weight += score_matrix[parents[node], node]

        # For each node in the graph, find the maximum weight incoming
        # and outgoing edge into the cycle.
        cycle_representative = cycle[0]
        for node in range(length):
            if not current_nodes[node] or node in cycle:
                continue

            in_edge_weight = float("-inf")
            in_edge = -1
            out_edge_weight = float("-inf")
            out_edge = -1

            for node_in_cycle in cycle:
                if score_matrix[node_in_cycle, node] > in_edge_weight:
                    in_edge_weight = score_matrix[node_in_cycle, node]
                    in_edge = node_in_cycle

                # Add the new edge score to the cycle weight
                # and subtract the edge we're considering removing.
                score = (
                    cycle_weight
                    + score_matrix[node, node_in_cycle]
                    - score_matrix[parents[node_in_cycle], node_in_cycle]
                )

                if score > out_edge_weight:
                    out_edge_weight = score
                    out_edge = node_in_cycle

            score_matrix[cycle_representative, node] = in_edge_weight
            old_input[cycle_representative, node] = old_input[in_edge, node]
            old_output[cycle_representative, node] = old_output[in_edge, node]

            score_matrix[node, cycle_representative] = out_edge_weight
            old_output[node, cycle_representative] = old_output[node, out_edge]
            old_input[node, cycle_representative] = old_input[node, out_edge]

        # For the next recursive iteration, we want to consider the cycle as a
        # single node. Here we collapse the cycle into the first node in the
        # cycle (first node is arbitrary), set all the other nodes not be
        # considered in the next iteration. We also keep track of which
        # representatives we are considering this iteration because we need
        # them below to check if we're done.
        considered_representatives: List[Set[int]] = []
        for i, node_in_cycle in enumerate(cycle):
            considered_representatives.append(set())
            if i > 0:
                # We need to consider at least one
                # node in the cycle, arbitrarily choose
                # the first.
                current_nodes[node_in_cycle] = False

            for node in representatives[node_in_cycle]:
                considered_representatives[i].add(node)
                if i > 0:
                    representatives[cycle_representative].add(node)

        self.chu_liu_edmonds(
            length, score_matrix, current_nodes, final_edges, old_input, old_output, representatives
        )

        # Expansion stage.
        # check each node in cycle, if one of its representatives
        # is a key in the final_edges, it is the one we need.
        found = False
        key_node = -1
        for i, node in enumerate(cycle):
            for cycle_rep in considered_representatives[i]:
                if cycle_rep in final_edges:
                    key_node = node
                    found = True
                    break
            if found:
                break

        previous = parents[key_node]
        while previous != key_node:
            child = old_output[parents[previous], previous]
            parent = old_input[parents[previous], previous]
            final_edges[child] = parent
            previous = parents[previous]
    
    def _find_cycle(
        self, parents: List[int], length: int, current_nodes: List[bool]
) -> Tuple[bool, List[int]]:

        added = [False for _ in range(length)]
        added[0] = True
        cycle = set()
        has_cycle = False
        for i in range(1, length):
            if has_cycle:
                break
            # don't redo nodes we've already
            # visited or aren't considering.
            if added[i] or not current_nodes[i]:
                continue
            # Initialize a new possible cycle.
            this_cycle = set()
            this_cycle.add(i)
            added[i] = True
            has_cycle = True
            next_node = i
            while parents[next_node] not in this_cycle:
                next_node = parents[next_node]
                # If we see a node we've already processed,
                # we can stop, because the node we are
                # processing would have been in that cycle.
                if added[next_node]:
                    has_cycle = False
                    break
                added[next_node] = True
                this_cycle.add(next_node)

            if has_cycle:
                original = next_node
                cycle.add(original)
                next_node = parents[original]
                while next_node != original:
                    cycle.add(next_node)
                    next_node = parents[next_node]
                break
        return has_cycle, list(cycle)

    def forward(self, subwords, tags=None):
        if self.enable_tag:
            bert_out = self.encoder(subwords)
            tag_out = self.tag_embedding(tags)
            bert_out = torch.cat([bert_out, tag_out], dim=-1)
        else:
            bert_out = subwords

        lstm_out, _ = self.lstm(bert_out)

        edge_mlp_d = self.edge_mlp_d(lstm_out)
        edge_mlp_h = self.edge_mlp_h(lstm_out)

        label_mlp_d = self.label_mlp_d(lstm_out)
        label_mlp_h = self.label_mlp_h(lstm_out)

        s_edge = self.edge_attn(edge_mlp_d, edge_mlp_h).permute(0, 2, 3, 1)
        s_label = self.label_attn(label_mlp_d, label_mlp_h).permute(0, 2, 3, 1)
        return s_edge, s_label

    # @torch.no_grad()
    # def _loss2(self, s_edge, s_label, labels, mask, interpolation=0.1):
    #     criterion2 = nn.CrossEntropyLoss(reduction='none')
    #
    #     edge_mask = labels.ge(0) & mask
    #     edge_loss = criterion2(s_edge.reshape(-1, s_edge.size(-1)), edge_mask.reshape(-1).long())
    #     edge_loss = edge_loss[mask.reshape(-1)]
    #     edge_loss = edge_loss.sum() / edge_mask.size(0)
    #
    #     lab el_loss = criterion2(s_label.reshape(-1, s_label.size(-1)), labels.reshape(-1))
    #     label_loss = label_loss[edge_mask.reshape(-1)]
    #     label_loss = label_loss.sum() / label_loss.size(0)
    #
    #     print('loss2: ', interpolation * label_loss + (1 - interpolation) * edge_loss)

    # interpolation： importance of label
    def loss(self, s_edge, s_label, labels, mask, interpolation=0.1):
        # self._loss2(s_edge, s_label, labels, mask, interpolation)
        # interpolation -> importance of label
        
        L = s_edge.size(1)               # 真实节点数 (不含 ROOT)
        if labels.size(1) != L:
            labels = labels[:, :L, :L]   # 裁掉 ROOT 行列
        if mask.size(1) != L:
            mask = mask[:,:L, :L]
        
        edge_mask = labels.ge(0) & mask
        edge_loss = self.criterion(s_edge[mask], edge_mask[mask].long())
        label_loss = self.criterion(s_label[edge_mask], labels[edge_mask])
        return interpolation * label_loss + (1 - interpolation) * edge_loss

    def decode(self, s_edge, s_label):
        return s_label.argmax(-1).masked_fill_(s_edge.argmax(-1).lt(1), -1)

    def mst_decode(self, s_edge: torch.Tensor,
                   s_label: torch.Tensor) -> torch.Tensor:

        batch_size, seq_len, _, _ = s_edge.shape
        labels_pred = torch.full((batch_size, seq_len, seq_len), -1, dtype=torch.long, device=s_edge.device)   
        for batch_idx in range(batch_size):
            edge_scores = s_edge[batch_idx, :, :, 1] # (seq_len, seq_len)
            length = edge_scores.size(0)
            heads, _ = self.decode_mst(edge_scores.cpu().numpy(), length, has_labels=False)
            heads = torch.tensor(heads, dtype=torch.long, device=s_edge.device)
            label_scores = s_label[batch_idx]  # (seq_len, seq_len, n_labels)

            # 根据 MST 预测的边，取出最可能的标签
            for dep_idx, head_idx in enumerate(heads):
                if head_idx >= 0:  # 过滤掉根节点
                    labels_pred[batch_idx, dep_idx, head_idx] = label_scores[dep_idx, head_idx].argmax(-1)

        return labels_pred  # (batch_size, seq_len, seq_len)
