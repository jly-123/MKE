# MKE implementation

`mke_new.py` generates molecular candidate node sets, evaluates their Shapley group values against a fixed GCN, and greedily selects candidate combinations under a distinct-node budget. The model remains in evaluation mode during explanation.

## Candidate generation

Candidate generation uses the molecular graph before explicit self-loops are added. The generator applies the following rules in order:

1. Compute a NetworkX cycle basis and retain its cycles. Six-member cycles additionally yield adjacent two-node pieces in two alternating partitions. Five-member cycles yield three two-node/three-node partitions. Cycle nodes are marked covered.
2. Match the ordered `SUB_SMILES` templates using atom-element labels and graph connectivity. The matching graph contains binary adjacency and does not encode bond orders. A matched node set is retained when it contains at least one node not already covered; its nodes are then marked covered. Accepted single-atom matches with at least two neighbors also generate a closed-neighborhood candidate.
3. Find connected components among nodes remaining uncovered after those passes, then add those components and their singleton nodes.
4. Normalize every candidate to a sorted set of atom indices, discard empty candidates, and deduplicate identical node sets while preserving first occurrence. The trace retains the mapping to original candidate indices.

Candidates may overlap. Atom indices refer to the RDKit atom order for the original input SMILES. Candidate generation and tie order can depend on traversal order, so record the package versions used for each experiment.

## Perturbation and payoff

Let `p_y(G)` be the GCN probability of the clean predicted class `y`. The class is chosen once from the initial clean prediction and held fixed for all perturbations. The clean probability used in scoring is recomputed through the same scorer as the perturbed probabilities.

For a node set `U`, the default `split` perturbation removes all edges incident to nodes in `U`, including their explicit self-loops. It retains the original node-feature matrix and node count. Inverse scoring retains only edges whose endpoints are both in `U`. Consequently, nodes outside the retained set remain in the feature matrix and global pooling; this operation does not physically delete them.

Define the payoff by

$$
v(U)=1-\frac{p_y(G\setminus U)}{p_y(G)},\qquad v(\varnothing)=0.
$$

Here `G \setminus U` denotes the configured perturbation just described. The clean scoring probability must be finite and strictly positive. Negative payoffs are retained when a perturbation increases the target probability.

The alternative `zero_filling` mode zeros the masked nodes' input features while preserving edge indices. Selecting a different mode changes the experiment and is recorded in run metadata.

## Shapley group value

Let `N` be the set of candidate identities and `C_i` the nodes in candidate `i`. For a candidate group `A`, write `U_A` for the union of its nodes. Treat all identities in `A` as one merged player while retaining each remaining candidate as an individual player. With `k = |N \setminus A|`, the group value is

$$
\phi^g(A)=\sum_{T\subseteq N\setminus A}
\frac{|T|!\,(k-|T|)!}{(k+1)!}
\left[v(U_T\cup U_A)-v(U_T)\right].
$$

The empty group has score zero. Group value is evaluated for the proposed combination; the implementation does not substitute a sum of independently computed singleton values.

Candidate identities and node unions are distinct. Two groups with the same node union can have different group values because their remaining players differ. The implementation therefore caches group scores by candidate identities and caches GCN payoffs separately by node union. Shared atoms are perturbed and budgeted once.

When the original deduplicated candidate count is at most `performance.exact_threshold` (default `8`), all required outsider subsets are enumerated with exact Shapley weights. This threshold counts candidate substructures, not graph nodes. At eight candidates there are at most `2^8` candidate coalitions, and overlap may reduce the number of unique node-union payoffs.

For larger candidate families, each requested group value uses `performance.sample_num` (default `100`) uniform random permutations of the remaining candidate identities plus a marker representing the merged group. The candidates preceding that marker form the background coalition. The estimate averages the resulting marginal payoffs. Each group receives one cached estimate within a graph. This is a Monte Carlo approximation; its acceptance decisions are based on estimated gains. Background coalitions are not limited by the final explanation budget.

## Greedy selection and budget

For target sparsity `s` and a graph with `n` nodes, the node budget is

$$
B=\lfloor(1-s)n\rfloor.
$$

Starting with the empty group, the selector evaluates every unselected candidate whose addition yields a node union of size at most `B`. It accepts the candidate with the largest strictly positive increase in group value. Equal gains retain the first candidate encountered. It stops when no feasible candidate remains or no feasible candidate has positive gain.

The accepted candidate is kept intact. An overlapping candidate can be considered even at a full node budget if it adds no new nodes. The algorithm does not truncate fragments or fill unused budget with arbitrary atoms.

Actual sparsity is

$$
s_{\mathrm{actual}}=1-\frac{|U_A|}{n}.
$$

It can exceed the requested value because the budget is integral, candidate sets are indivisible, or no further candidate has positive gain. Results at different budgets are evaluated independently; neither Fidelity metric is artificially clipped or forced to be monotonic.

## Reported metrics

For each evaluated graph,

$$
\mathrm{Fidelity}^{+}=p_y(G)-p_y(G\setminus U_A),
$$

$$
\mathrm{Fidelity}^{-}=p_y(G)-p_y(G\mid U_A),
$$

where `G | U_A` is the inverse perturbation under the same configured masking convention. These are probability differences, not normalized payoffs or group values. Signed results, including negative values, are saved unchanged.

Dataset means include every valid test graph, including incorrect predictions and empty explanations. For an empty explanation, Fidelity+ is zero, actual sparsity is one, and Fidelity- is evaluated using the empty retained set. Predictor accuracy is reported separately.

## Precision and random seeds

The scorer uses CUDA automatic mixed precision for both clean scoring probabilities and perturbed probabilities. The initial class-selection forward is separate; the trace records its target probability as well as the recomputed scoring probability. CUDA execution also enables cuDNN benchmarking and TF32 where supported. On a CPU-only environment, CUDA autocast is disabled by PyTorch.

Each graph's local permutation generator is initialized with

```text
graph_seed = random_seed + 1000003 * sample_index
```

`sample_index` is the processed dataset index. A fixed seed reproduces the permutation sequence for unchanged candidates and scoring order; it does not by itself guarantee bitwise equality across hardware, CUDA/PyTorch versions, precision settings, or candidate-generation library versions. Training separately initializes Python, NumPy, and PyTorch seeds through `fix_random_seed`.

See [configuration](CONFIGURATION.md) for controls and [outputs](OUTPUTS.md) for the selection trace and full-precision result files.
