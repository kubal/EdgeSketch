## 1. Project structure

- `src/` - source modules (EdgeSketch, NodeSketch, GraphModels, etc.)
- `experiments/` - test and experiment functions
- `data/` - directory for datasets (download datasets here)
- `plots/` - output directory for generated figures 
- `LoadModules.jl` - script to load all modules


## 2. How to run

1. **Install Julia**:
   - The project was tested on Julia 1.10.10 (Long Term Support version as of Feb 2026). On newer Julia versions you may see deprecation or compatibility warnings.

2. **Go to the project directory**:
   - Open a terminal and navigate to the directory containing the project files.

3. **Load project modules** by running the following command:
   ```sh
   julia -t XXX -i LoadModules.jl
   ```
   where `XXX` is the number of threads to use.


## 3. Tests (`experiments/_Tests.jl`)

Unit tests for core functionality. In the Julia REPL, run:
```julia
run_tests()
```

## 4. Experiments (`experiments/_Experiments.jl`)

Experiments on synthetic graphs (Stochastic Block Model). Generate plots for the paper:

- `experiment_Louvain_estimator()` - modularity estimator accuracy (Figure 1b)
- `experiment_Louvain_varying_m()` - Louvain method on sketches with varying sketch sizes (Figures 1a, 1c)
- `experiment_Louvain_dynamic_graphs()` - Louvain method on streaming edge batches (Figure 1d)


## 5. Epinions (`experiments/_Epinions.jl`)

Experiments on real-world Epinions dataset.

- **Data source:** https://networkrepository.com/rec-epinions.php
- Download and extract to `data/rec-epinions/rec-epinions.edges`
- The dataset contains user-item ratings; it is projected to an item-item graph where edge weight = number of common users

Functions:

- `load_epinions_graph(; min_weight, max_users_per_item)` - loads and projects the graph
- `epinions_sketch_construction()` - tests sketch creation on Epinions
- `epinions_louvain_on_sketches()` - runs Louvain method on Epinions sketches
- `epinions_louvain_streaming()` - creates sketches directly from bipartite data and runs Louvain method (without storing adj_list)


## 6. ego-Gplus social graph (`experiments/_Gplus.jl`)

Experiments on the ego-Gplus (Google+) social graph (~107K nodes, avg degree ~224).

- **Data source:** https://snap.stanford.edu/data/ego-gplus.html
- Download and extract to `data/ego-Gplus/gplus_combined.txt`
  (`snap.stanford.edu/data/gplus_combined.txt.gz`, then gunzip)
- The dataset contains a directed Google+ social graph; it is loaded as an
  undirected graph

Functions:

- `load_gplus_graph(; filepath, weighted)` - loads the edge list into an
  undirected weighted adjacency list (symmetrizes arcs, drops self-loops,
  collapses parallel edges, remaps to consecutive ids)
- `gplus_louvain_on_sketches(; sketch_size)` - runs Louvain on the full graph and
  on the EdgeSketch and compares modularity / memory / time


## 7. Stock correlation graphs (`experiments/_Stocks.jl`)

A dense graph derived from stock prices: nodes = stocks, edge weight = how
similarly two stocks move (|Pearson correlation|^beta of daily log returns).

- **Data source:** Kaggle "Huge Stock Market Dataset" (Boris Marjanovic)
  https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs
- Unzip all per-ticker files (both the Stocks and ETFs folders of the archive)
  flat into `data/Kaggle/`

Functions:

- `load_stocks_graph(; dir, start_date, end_date, beta, corr_threshold, min_coverage, max_tickers)`
  - builds a weighted correlation graph: log returns -> Pearson correlations ->
    edge weight |corr|^beta; only edges with |corr| >= corr_threshold are kept;
    min_coverage drops tickers with too many missing days; max_tickers caps the node count

- `stocks_louvain_on_sketches(; sketch_size, max_tickers, ...)` - runs Louvain on
  the correlation graph and on the EdgeSketch and compares modularity / memory / time


## 8. Brain-1M single-cell graph (`experiments/_Brain1M.jl`)

A k-NN graph derived from single-cell RNA-seq data: nodes = cells, edges connect
each cell to its most similar cells in PCA space, weighted by the Jaccard overlap
of their neighbourhoods (the standard Scanpy/Seurat cell-clustering pipeline).

- **Data source:** 10x Genomics "1.3M Brain Cells" (E18 mouse, ~4 GB HDF5)
  https://cf.10xgenomics.com/samples/cell/1M_neurons/1M_neurons_filtered_gene_bc_matrices_h5.h5
- Place the downloaded `.h5` file at
  `data/brain1M/1M_neurons_filtered_gene_bc_matrices_h5.h5`

Functions:

- `load_brain1M_graph_streaming(; max_cells, k, n_hvg, n_pcs, ...)` - streams the
  HDF5, normalizes + selects highly variable genes + PCA, then builds the
  Jaccard-weighted k-NN graph (HNSW)
- `brain1M_streaming_louvain_on_sketches(; sketch_size, max_cells, k, ...)` - runs
  Louvain on the full graph and on the EdgeSketch and compares modularity /
  memory / time (set `max_cells` to subsample, e.g. 100_000)