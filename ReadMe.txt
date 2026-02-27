## 1. Project structure

- `src/` - source modules (EdgeSketch, NodeSketch, GraphModels, etc.)
- `experiments/` - test and experiment functions
- `data/` - directory for datasets (download Epinions dataset here)
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
- `experiment_graph_reconstruction()` - EdgeSketch vs NodeSketch precision (Figures 2a, 2b)

## 5. Epinions (`experiments/_Epinions.jl`)

Experiments on real-world Epinions dataset.

- **Data source:** https://networkrepository.com/rec-epinions.php
- Download and extract to `data/rec-epinions/rec-epinions.edges`
- The dataset contains user-item ratings. It is projected to an item-item graph where edge weight = number of common users.

Functions:

- `load_epinions_graph(; min_weight, max_users_per_item)` - loads and projects the graph
- `epinions_sketch_construction()` - tests sketch creation on Epinions
- `epinions_louvain_on_sketches()` - runs Louvain method on Epinions sketches
- `epinions_louvain_streaming()` - creates sketches directly from bipartite data and runs Louvain method (without storing adj_list)
- `epinions_graph_reconstruction()` - edge prediction on Epinions
