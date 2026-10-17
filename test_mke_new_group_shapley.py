import ast
import itertools
import math
import operator
from pathlib import Path
import random
import unittest


def load_group_core():
    source_path = Path(__file__).with_name("mke_new.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8-sig"),
                     filename=str(source_path))
    required = {"normalize_candidate_nodes", "ShapleyGroupGame",
                "greedy_group_select"}
    tree.body = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                 and node.name in required]
    found = {node.name for node in tree.body}
    if required != found:
        raise AssertionError("Missing standalone group core: {}".format(
            sorted(required - found)))
    namespace = {"itertools": itertools, "math": math, "operator": operator,
                 "random": random}
    exec(compile(tree, str(source_path), "exec"), namespace)
    return namespace["ShapleyGroupGame"], namespace["greedy_group_select"]


ShapleyGroupGame, greedy_group_select = load_group_core()


def load_group_explainer(scorer_type):
    source_path = Path(__file__).with_name("mke_new.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8-sig"),
                     filename=str(source_path))
    tree.body = [node for node in tree.body
                 if isinstance(node, ast.ClassDef)
                 and node.name == "GroupShapleyExplainer"]
    if len(tree.body) != 1:
        raise AssertionError("Missing GroupShapleyExplainer adapter")
    namespace = {"math": math, "operator": operator,
                 "BatchedGNNScorer": scorer_type,
                 "ShapleyGroupGame": ShapleyGroupGame,
                 "greedy_group_select": greedy_group_select}
    exec(compile(tree, str(source_path), "exec"), namespace)
    return namespace["GroupShapleyExplainer"]


def merged_permutation_oracle(candidate_nodes, group_ids, payoff):

    group = set(group_ids)
    if not group:
        return 0.0
    merged_player = object()
    players = [i for i in range(len(candidate_nodes)) if i not in group]
    players.append(merged_player)
    group_nodes = set()
    for index in group:
        group_nodes.update(candidate_nodes[index])
    contributions = []
    for order in itertools.permutations(players):
        predecessor_nodes = set()
        for player in order:
            if player is merged_player:
                before = payoff(tuple(sorted(predecessor_nodes)))
                after = payoff(tuple(sorted(predecessor_nodes | group_nodes)))
                contributions.append(after - before)
                break
            predecessor_nodes.update(candidate_nodes[player])
    return math.fsum(contributions) / len(contributions)


class RecordingPayoff:
    def __init__(self, payoff):
        self.payoff = payoff
        self.batches = []

    def __call__(self, node_unions):
        batch = [tuple(nodes) for nodes in node_unions]
        self.batches.append(batch)
        return [self.payoff(nodes) for nodes in batch]


def make_game(candidates, payoff, num_nodes, **kwargs):
    evaluator = RecordingPayoff(payoff)
    game = ShapleyGroupGame(candidates, evaluator, num_nodes, **kwargs)
    return game, evaluator


def unanimity_payoff(nodes):
    return 1.0 if set(nodes) == {0, 1, 2} else 0.0


class ExactGroupValueTests(unittest.TestCase):
    def test_all_groups_match_independent_merged_permutations(self):
        candidates = [(0, 1), (1, 2), (2, 3), (0, 3)]

        def payoff(nodes):
            present = set(nodes)
            linear = sum((0.7, -0.4, 1.2, -0.1)[i] for i in present)

            return (0.37 + linear + 0.9 * ({0, 2} <= present)
                    - 1.1 * ({1, 2, 3} <= present))

        game, _ = make_game(candidates, payoff, 4)
        groups = [group for size in range(5)
                  for group in itertools.combinations(range(4), size)]
        actual = game.scores(groups)
        self.assertEqual(game.score_method, "exact")
        self.assertEqual(len(actual), len(groups))
        for group, score in zip(groups, actual):
            with self.subTest(group=group):
                expected = merged_permutation_oracle(candidates, group, payoff)
                self.assertAlmostEqual(score, expected, places=12)
        self.assertEqual(game.score([]), 0.0)
        self.assertAlmostEqual(game.score(range(4)),
                               payoff((0, 1, 2, 3)) - payoff(()), places=12)

    def test_group_value_is_neither_singleton_sum_nor_direct_payoff(self):
        game, _ = make_game([(0,), (1,), (2,)], unanimity_payoff, 3)
        self.assertAlmostEqual(game.score([0]), 1.0 / 3.0, places=12)
        self.assertAlmostEqual(game.score([1]), 1.0 / 3.0, places=12)
        group_value = game.score([0, 1])
        self.assertAlmostEqual(group_value, 0.5, places=12)
        self.assertNotAlmostEqual(group_value, game.score([0]) + game.score([1]))
        self.assertNotAlmostEqual(group_value, unanimity_payoff((0, 1)))

    def test_same_node_union_can_have_different_group_values(self):
        game, evaluator = make_game([(0, 1), (0,), (1,)], len, 2)
        self.assertEqual(game.node_union([0]), (0, 1))
        self.assertEqual(game.node_union([0, 1]), (0, 1))
        self.assertAlmostEqual(game.score([0]), 1.0)
        evaluated_before = sum(map(len, evaluator.batches))
        self.assertAlmostEqual(game.score([0, 1]), 1.5)
        self.assertEqual(sum(map(len, evaluator.batches)), evaluated_before)
        self.assertAlmostEqual(game.score_cache[(0,)], 1.0)
        self.assertAlmostEqual(game.score_cache[(0, 1)], 1.5)
        self.assertEqual(set(game.payoff_cache), {(), (0,), (1,), (0, 1)})

    def test_canonical_candidates_and_group_cache_keys(self):
        game, evaluator = make_game([[1, 0, 1], [], [0, 1], [2], [2, 2]], len, 3)
        self.assertEqual(tuple(game.candidate_nodes), ((0, 1), (2,)))
        self.assertEqual(game.node_union([1, 0, 1]), (0, 1, 2))
        expected = game.score([0, 1])
        calls_before = len(evaluator.batches)
        self.assertEqual(game.score([1, 0, 1]), expected)
        self.assertEqual(len(evaluator.batches), calls_before)
        self.assertIn((0, 1), game.score_cache)

    def test_invalid_candidate_nodes_are_rejected(self):
        for candidates in [[(-1,)], [(2,)], [(0.5,)], [("0",)]]:
            with self.subTest(candidates=candidates):
                with self.assertRaises((ValueError, TypeError, IndexError)):
                    make_game(candidates, len, 2)

    def test_payoff_batches_are_bounded_and_each_union_evaluated_once(self):
        game, evaluator = make_game([(0,), (1,), (2,), (3,)],
                                    lambda nodes: len(nodes) ** 2, 4,
                                    batch_size=2)
        groups = [(0,), (1,), (0, 1), (2, 3), (0, 1, 2, 3)]
        game.scores(groups)
        all_evaluated = [nodes for batch in evaluator.batches for nodes in batch]
        self.assertTrue(evaluator.batches)
        self.assertTrue(all(1 <= len(batch) <= 2 for batch in evaluator.batches))
        self.assertEqual(len(all_evaluated), len(set(all_evaluated)))
        calls_before = len(evaluator.batches)
        game.scores(list(reversed(groups)))
        self.assertEqual(len(evaluator.batches), calls_before)

    def test_callback_length_mismatch_is_rejected(self):
        for extra in [-1, 1]:
            with self.subTest(extra=extra):
                def wrong_length(unions):
                    return [1.0] * (len(unions) + extra)

                game = ShapleyGroupGame([(0,)], wrong_length, 1)
                cache_before = dict(game.payoff_cache)
                with self.assertRaises(ValueError):
                    game.score([0])
                self.assertNotIn((0,), game.score_cache)
                self.assertEqual(game.payoff_cache, cache_before)

    def test_nonfinite_callback_payoffs_are_rejected(self):
        for invalid in [float("nan"), float("inf"), float("-inf")]:
            with self.subTest(invalid=invalid):
                game = ShapleyGroupGame([(0,)],
                                        lambda unions: [invalid] * len(unions), 1)
                cache_before = dict(game.payoff_cache)
                with self.assertRaises(ValueError):
                    game.score([0])
                self.assertNotIn((0,), game.score_cache)
                self.assertEqual(game.payoff_cache, cache_before)


class GreedyGroupTests(unittest.TestCase):
    def assert_valid_trace(self, game, result, max_nodes):
        selected = result["selected_candidate_indices"]
        self.assertEqual(len(selected), len(set(selected)))
        self.assertEqual(result["selected_nodes"], list(game.node_union(selected)))
        self.assertLessEqual(len(result["selected_nodes"]), max_nodes)
        self.assertAlmostEqual(result["group_score"], game.score(selected), places=12)
        self.assertIsInstance(result["stop_reason"], str)
        self.assertTrue(result["stop_reason"])
        for proposal in result["proposals"]:
            group = proposal["proposed_candidate_indices"]
            self.assertLessEqual(len(game.node_union(group)), max_nodes)
            self.assertAlmostEqual(proposal["group_score"], game.score(group), places=12)
            self.assertAlmostEqual(proposal["marginal_gain"],
                                   proposal["group_score"] - proposal["previous_group_score"],
                                   places=12)
            if proposal["accepted"]:
                self.assertGreater(proposal["marginal_gain"], 0.0)

    def test_greedy_accepts_synergy_that_direct_payoff_would_miss(self):
        game, _ = make_game([(0,), (1,), (2,)], unanimity_payoff, 3)
        result = greedy_group_select(game, max_nodes=2)
        self.assertEqual(result["selected_candidate_indices"], [0, 1])
        self.assertEqual(result["selected_nodes"], [0, 1])
        self.assertAlmostEqual(result["group_score"], 0.5, places=12)
        self.assertEqual(unanimity_payoff(result["selected_nodes"]), 0.0)
        self.assert_valid_trace(game, result, 2)

    def test_zero_new_node_candidate_can_improve_group_at_full_budget(self):
        game, _ = make_game([(0, 1), (0,), (1,)], len, 2)
        result = greedy_group_select(game, max_nodes=2)
        self.assertEqual(result["selected_candidate_indices"], [0, 1, 2])
        self.assertEqual(result["selected_nodes"], [0, 1])
        self.assertAlmostEqual(result["group_score"], 2.0)
        accepted = [proposal for proposal in result["proposals"]
                    if proposal["accepted"]]
        self.assertEqual([proposal["group_score"] for proposal in accepted],
                         [1.0, 1.5, 2.0])
        self.assert_valid_trace(game, result, 2)

    def test_nonpositive_marginals_are_not_accepted(self):
        for weights in [(0.0, 0.0, 0.0), (-1.0, -2.0, -3.0),
                        (3.0, -2.0, -1.0)]:
            with self.subTest(weights=weights):
                game, _ = make_game([(0,), (1,), (2,)],
                                    lambda nodes: sum(weights[i] for i in nodes), 3)
                result = greedy_group_select(game, max_nodes=3)
                expected = [0] if weights[0] > 0 else []
                self.assertEqual(result["selected_candidate_indices"], expected)
                self.assertAlmostEqual(result["group_score"], max(weights[0], 0.0))
                self.assert_valid_trace(game, result, 3)

    def test_budget_counts_unique_nodes_and_skips_oversized_candidate(self):
        game, _ = make_game([(0, 1, 2), (3,)], len, 4)
        result = greedy_group_select(game, max_nodes=1)
        self.assertEqual(result["selected_candidate_indices"], [1])
        self.assertEqual(result["selected_nodes"], [3])
        self.assert_valid_trace(game, result, 1)

    def test_zero_budget_and_empty_candidate_game(self):
        for candidates, num_nodes, budget in [([(0,)], 1, 0), ([], 0, 0), ([], 2, 2)]:
            with self.subTest(candidates=candidates, budget=budget):
                game, _ = make_game(candidates, len, num_nodes)
                result = greedy_group_select(game, max_nodes=budget)
                self.assertEqual(result["selected_candidate_indices"], [])
                self.assertEqual(result["selected_nodes"], [])
                self.assertEqual(result["group_score"], 0.0)
                self.assert_valid_trace(game, result, budget)


class MonteCarloGroupTests(unittest.TestCase):
    def test_default_threshold_is_exact_at_eight_and_monte_carlo_at_nine(self):
        for count, method in [(8, "exact"), (9, "monte_carlo")]:
            with self.subTest(count=count):
                game, _ = make_game([(node,) for node in range(count)], len, count)
                self.assertEqual(game.score_method, method)
                self.assertEqual(game.sample_num, 100)
                self.assertAlmostEqual(game.score([0, 1]), 2.0, places=12)

    def test_fixed_seed_is_reproducible_without_changing_global_rng(self):
        candidates = [(0,), (1,), (2,)]
        options = {"exact_threshold": 0, "sample_num": 500, "seed": 314159}
        before = random.getstate()
        first, _ = make_game(candidates, unanimity_payoff, 3, **options)
        groups = [(0,), (1,), (0, 1), (0, 1, 2)]
        scores = first.scores(groups)
        second, _ = make_game(candidates, unanimity_payoff, 3, **options)
        self.assertEqual(scores, second.scores(groups))
        self.assertEqual(scores, first.scores(groups))
        self.assertEqual(random.getstate(), before)
        self.assertEqual(first.score_method, "monte_carlo")

    def test_monte_carlo_approximates_independent_exact_values(self):
        candidates = [(0,), (1,), (2,)]
        game, _ = make_game(candidates, unanimity_payoff, 3,
                            exact_threshold=0, sample_num=20000, seed=2026)
        for group in [(0,), (1,), (0, 1), (0, 1, 2)]:
            with self.subTest(group=group):
                expected = merged_permutation_oracle(candidates, group, unanimity_payoff)
                self.assertAlmostEqual(game.score(group), expected, delta=0.025)

    def test_monte_carlo_preserves_overlapping_candidate_identities(self):
        candidates = [(0, 1), (0,), (1,)]
        game, _ = make_game(candidates, len, 2,
                            exact_threshold=0, sample_num=10000, seed=2026)
        for group in [(0,), (0, 1), (0, 1, 2)]:
            with self.subTest(group=group):
                expected = merged_permutation_oracle(candidates, group, len)
                self.assertAlmostEqual(game.score(group), expected, delta=0.035)


class FakeGraph:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes


class FakeProbabilityModel:


    def __init__(self, payoff, clean_probability=0.8):
        self.payoff = payoff
        self.clean_probability = clean_probability

    def __call__(self, *args, **kwargs):
        raise AssertionError("The adapter bypassed its shared probability scorer")

    def retained_probability(self, nodes):
        return 0.15 + 0.1 * len(nodes)


class FakeBatchedGNNScorer:
    def __init__(self, model, device, subgraph_building_method, batch_size=1024):
        self.model = model
        self.device = device
        self.subgraph_building_method = subgraph_building_method
        self.batch_size = batch_size
        self.mask_calls = []
        self.inverse_calls = []

    def batch_score(self, node_unions, data, label):
        unions = [tuple(nodes) for nodes in node_unions]
        self.mask_calls.append((unions, data, label))
        return [self.model.clean_probability * (1.0 - self.model.payoff(nodes))
                for nodes in unions]

    def batch_score_inverse(self, node_unions, data, label):
        unions = [tuple(nodes) for nodes in node_unions]
        self.inverse_calls.append((unions, data, label))
        return [self.model.retained_probability(nodes) for nodes in unions]


class ExplainerAdapterTests(unittest.TestCase):
    @staticmethod
    def table_payoff(nodes):
        table = {(): 0.0, (0,): 0.6, (1,): 0.2, (2,): 0.2,
                 (0, 1): 0.1, (0, 2): 0.1, (1, 2): 0.9,
                 (0, 1, 2): 1.0}
        return table[tuple(sorted(nodes))]

    def make_explainer(self, payoff=None, **options):
        model = FakeProbabilityModel(payoff or self.table_payoff)
        explainer_type = load_group_explainer(FakeBatchedGNNScorer)
        return explainer_type(model, num_classes=2, device="fake-cpu",
                              subgraph_building_method="zero_filling", **options)

    @staticmethod
    def make_input(sample_index=7, candidates=None, node_size=0.5):
        return {"data": FakeGraph(3), "label": 1,
                "truelist": [0.4, 0.6], "sample_index": sample_index,
                "node_size": node_size,
                "candidate_subgraphs": ([[0], [1], [2]] if candidates is None
                                         else candidates)}

    def assert_trace_matches_result(self, trace, result, item):
        nodes, masked, retained, probabilities, data = result
        self.assertEqual(nodes, trace["selected_nodes"])
        self.assertEqual(data, item["data"])
        self.assertEqual(trace["sample_index"], item["sample_index"])
        self.assertEqual(trace["target_class"], item["label"])
        self.assertEqual(trace["node_budget"],
                         math.floor((1.0 - item["node_size"]) * data.num_nodes))
        self.assertLessEqual(len(nodes), trace["node_budget"])
        self.assertAlmostEqual(trace["actual_sparsity"], 1.0 - len(nodes) / data.num_nodes)
        self.assertAlmostEqual(trace["masked_probability"], masked)
        self.assertAlmostEqual(trace["retained_probability"], retained)
        baseline = trace["scoring_target_probability"]
        self.assertAlmostEqual(probabilities[item["label"]], baseline)
        self.assertAlmostEqual(trace["fidelity_plus"], baseline - masked)
        self.assertAlmostEqual(trace["fidelity_minus"], baseline - retained)
        self.assertTrue(all(math.isfinite(value) for value in
                            (masked, retained, trace["group_score"],
                             trace["fidelity_plus"], trace["fidelity_minus"])))

    def test_selection_normalization_fidelity_and_frozen_target(self):
        explainer = self.make_explainer(batch_size=2)
        item = self.make_input()
        original_probabilities = list(item["truelist"])
        result = explainer.explain_batch([item])[0]
        trace = explainer.last_traces[0]
        self.assert_trace_matches_result(trace, result, item)


        self.assertEqual(result[0], [1])
        self.assertAlmostEqual(trace["group_score"], 0.4, places=12)
        self.assertEqual(trace["selected_candidate_indices"], [1])
        first_round = {proposal["added_candidate_index"]: proposal["group_score"]
                       for proposal in trace["proposals"] if proposal["iteration"] == 1}
        for candidate, expected in [(0, 0.2), (1, 0.4), (2, 0.4)]:
            self.assertAlmostEqual(first_round[candidate], expected, places=12)


        self.assertAlmostEqual(trace["input_target_probability"], 0.6)
        self.assertAlmostEqual(trace["scoring_target_probability"], 0.8)
        self.assertAlmostEqual(result[1], 0.64)
        self.assertAlmostEqual(result[2], 0.25)
        self.assertAlmostEqual(trace["fidelity_plus"], 0.16)
        self.assertAlmostEqual(trace["fidelity_minus"], 0.55)
        self.assertEqual(item["truelist"], original_probabilities)
        self.assertIsNot(result[3], item["truelist"])

        scorer = explainer.scorer
        all_calls = scorer.mask_calls + scorer.inverse_calls
        self.assertTrue(all(label == 1 for _, _, label in all_calls))
        self.assertTrue(all(data is item["data"] for _, data, _ in all_calls))
        evaluated = [nodes for batch, _, _ in scorer.mask_calls for nodes in batch]
        self.assertEqual(evaluated.count(()), 1)
        self.assertIn((0, 1, 2), evaluated)
        self.assertEqual(scorer.inverse_calls, [([(1,)], item["data"], 1)])
        self.assertEqual(scorer.batch_size, 2)
        self.assertTrue(all(len(batch) <= 2 for batch, _, _ in scorer.mask_calls))

    def test_pure_synergy_is_selected_despite_zero_direct_payoff(self):
        explainer = self.make_explainer(payoff=unanimity_payoff)
        item = self.make_input(node_size=0.25)
        result = explainer.explain_batch([item])[0]
        trace = explainer.last_traces[0]
        self.assertEqual(result[0], [0, 1])
        self.assertAlmostEqual(trace["group_score"], 0.5)
        self.assertEqual(unanimity_payoff(result[0]), 0.0)
        self.assertAlmostEqual(trace["fidelity_plus"], 0.0)
        self.assert_trace_matches_result(trace, result, item)

    def test_sample_index_seed_reproduces_results_across_batch_reordering(self):
        explainer = self.make_explainer(exact_threshold=0, sample_num=500, seed=123)
        inputs = [self.make_input(sample_index=7), self.make_input(sample_index=11)]
        first_results = explainer.explain_batch(inputs)
        first_traces = {trace["sample_index"]: trace for trace in explainer.last_traces}
        second_results = explainer.explain_batch(list(reversed(inputs)))
        second_traces = {trace["sample_index"]: trace for trace in explainer.last_traces}
        self.assertEqual(len(explainer.last_traces), 2)
        self.assertEqual(first_traces, second_traces)
        self.assertEqual(first_results, list(reversed(second_results)))
        self.assertNotEqual(first_traces[7]["seed"], first_traces[11]["seed"])
        for item, result in zip(inputs, first_results):
            trace = first_traces[item["sample_index"]]
            self.assertEqual(trace["score_method"], "monte_carlo")
            self.assertEqual(trace["sample_num"], 500)
            self.assert_trace_matches_result(trace, result, item)

    def test_empty_explanation_is_a_list_with_finite_fidelity(self):
        for candidates, node_size in [(None, 0.5), ([], 0.5), (None, 1.0)]:
            with self.subTest(candidates=candidates, node_size=node_size):
                explainer = self.make_explainer(payoff=lambda nodes: -0.05 * len(nodes))
                item = self.make_input(candidates=candidates, node_size=node_size)
                result = explainer.explain_batch([item])[0]
                trace = explainer.last_traces[0]
                self.assertEqual(result[0], [])
                self.assertIsInstance(result[0], list)
                self.assertEqual(trace["selected_candidate_indices"], [])
                self.assertEqual(trace["group_score"], 0.0)
                self.assertEqual(trace["actual_sparsity"], 1.0)
                self.assertAlmostEqual(result[1], 0.8)
                self.assertAlmostEqual(result[2], 0.15)
                self.assertAlmostEqual(trace["fidelity_plus"], 0.0)
                self.assertAlmostEqual(trace["fidelity_minus"], 0.65)
                self.assertEqual(explainer.scorer.inverse_calls,
                                 [([()], item["data"], 1)])
                self.assert_trace_matches_result(trace, result, item)


if __name__ == "__main__":
    unittest.main(verbosity=2)
