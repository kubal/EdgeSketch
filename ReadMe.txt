## Project structure

- `src/` - source modules (EdgeSketch, NodeSketch, GraphModels, etc.)
- `experiments/` - test and experiment functions
- `plots/` - output directory for generated figures 
- `LoadModules.jl` - script to load all modules

## Instruction to run tests

1. **Install Julia**:
   - The project was created in Julia 1.10.4. On newer Julia versions you may see deprecation or compatibility warnings.

2. **Go to the project directory**:
   - Open a terminal and navigate to the directory containing the project files.

3. **Load project modules** by running the following command:
   ```sh
   julia -t XXX -i LoadModules.jl
   ```
   where `XXX` is the number of threads to use.

4. **Run test functions**:
   - Once the modules are loaded, you can run the test functions from the `experiments/_Tests.jl`.
     For example, in the Julia REPL, execute:
     ```julia
     test_graph_reconstruction()
     ```

5. **Run experiment functions**:
   - Once the modules are loaded, you can run the experiment functions from the `experiments/_Experiments.jl`.
     For example, in the Julia REPL, execute:
     ```julia
     experiment_graph_reconstruction()
     ```


     
