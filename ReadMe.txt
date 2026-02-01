## Project structure

- `src/` - source modules (EdgeSketch, NodeSketch, GraphModels, etc.)
- `experiments/` - test and experiment functions (`_Tests.jl`)
- `plots/` - output directory for figures (created by experiment functions)
- `data/` - data directory
- `LoadModules.jl` - script to load all modules

## Instruction to run tests

1. **Install Julia**:
   - The project was created in Julia 1.10.4. On newer Julia versions you may see deprecation or compatibility warnings.

2. **Go to the project directory**:
   - Open a terminal and navigate to the directory containing the project files.

3. **Load project modules**:
   - **Standard Loading**:
     - Start Julia and load the project modules using the following command:
       ```sh
       julia -i LoadModules.jl
       ```
     - This command starts Julia interactively with the `LoadModules.jl` script, loading all necessary project modules.

   - **Multithreaded Loading**:
     - For improved performance, when working with large datasets, start Julia with a specified number of threads:
       ```sh
       julia -t XXX -i LoadModules.jl
       ```
     - Replace `XXX` with the number of threads you want to use.

4. **Run test functions**:
   - Once the modules are loaded, you can run the test functions from the `_Tests` module (located in `experiments/_Tests.jl`).
     For example, in the Julia REPL, execute:
     ```julia
     test_graph_reconstruction()
     ```

5. **Run experiment functions**:
   - Once the modules are loaded, you can run the functions that implement the experiments from the paper (in the `_Tests` module located in `experiments/_Tests.jl`).


     
