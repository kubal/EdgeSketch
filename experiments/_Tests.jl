__precompile__(false)

module _Tests

using Random
using StatsBase
using Plots
using Dates
using StatsPlots

using _FastExpSketch
using _GraphModels
using _EdgeSketch
using _NodeSketch
using _SketchOperations
using _GraphReconstruction
using _GraphModularity
using _SketchModularity
using _Logger

export run_tests



# --- Tests sketch construction, degree estimation accuracy, and memory usage
function test_sketch_construction(adj_list::Vector{Vector{Tuple{Int, Float64}}}, name::String = "CUSTOM GRAPH"; sketch_size::Int = 100)
    Random.seed!(123)

    # --- Graph statistics ---
    nr_nodes = length(adj_list)
    nr_edges = Int(sum(length(neighbors) for neighbors in adj_list) / 2)
    adj_list_size_mb = Base.summarysize(adj_list) / 1e6


    # --- EdgeSketch generation ---
    sketching_time = @timed begin
        graph_sketch = create_graph_sketch(adj_list, sketch_size)
    end
    sketch_memory_size_mb = Base.summarysize(graph_sketch) / 1e6


    # --- EdgeSketch estimate ---
    estimate_degrees = nodes_weights(graph_sketch)
    real_degrees = nodes_real_degrees(adj_list)
    mean_estimate = mean(estimate_degrees ./ real_degrees)
    variance_estimate = var(estimate_degrees ./ real_degrees)
    values = vcat(real_degrees, estimate_degrees)
    groups = vcat(fill("real degrees", length(real_degrees)), fill("estimate degrees", length(estimate_degrees)))
    

    # --- Print results ---
    emph("\nGRAPH")
    info(" - name            $(name)")
    info(" - nr_nodes        $(nr_nodes)")
    info(" - nr_edges        $(nr_edges)")
    info(" - adjList_memory  $(f4(adj_list_size_mb)) MB")

    emph("\nSKETCH")
    info(" - sketch_size     $(sketch_size)")
    info(" - sketching_time  $(f4(sketching_time.time)) sec")
    info(" - sketch_memory   $(f4(sketch_memory_size_mb)) MB")

    emph("\nRATIO OF ESTIMATED AND REAL DEGREES")
    info(" - mean            $(f4(mean_estimate))")
    info(" - variance        $(f4(variance_estimate))")

    emph("\nDEGREES DISTRIBIUTION (boxplot)")
    plt = plot()
    boxplot!(groups, values, alpha=1, legend=false)
    title!(plt, "Real and estimate degrees distribiutions (m=$(sketch_size))")
    xlabel!(plt, "Groups")
    ylabel!(plt, "Values")
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    plot_path = "plots/plot_$(timestamp)_$(nr_nodes)_degrees.pdf"
    savefig(plot_path)
    info(" - saved to        $(plot_path)")
end



# --- Tests graph reconstruction using EdgeSketch and NodeSketch similarity matrices
function test_graph_reconstruction(adj_list::Vector{Vector{Tuple{Int, Float64}}}, name::String = "CUSTOM GRAPH"; sketch_size::Int = 30)
    Random.seed!(123)

    # --- Graph statistics ---
    nr_nodes = length(adj_list)
    nr_edges = Int(sum(length(neighbors) for neighbors in adj_list) / 2)
    adj_list_size_mb = Base.summarysize(adj_list) / 1e6


    # --- EdgeSketch generation ---
    k = 4
    alpha = 0.2
    use_sample_edges = true
    time_es_sketch = @elapsed graph_sketch = create_graph_sketch(adj_list, sketch_size)
    es_sketch_memory_mb = Base.summarysize(graph_sketch) / 1e6
    time_es_simMatrix = @elapsed es_sim_matrix = create_similarity_matrix(graph_sketch, k, alpha, use_sample_edges)
    time_es_reconstruct = @elapsed es_predicted_edges = predict_top_edges(es_sim_matrix, nr_edges)


    # --- NodeSketch generation (sketch_size 3x larger when using sample_edges) ---
    ns_sketch_size = use_sample_edges ? sketch_size * 3 : sketch_size
    time_ns_sketch = @elapsed node_sketch = nodesketch(adj_list, k, ns_sketch_size, alpha)
    ns_sketch_memory_mb = Base.summarysize(node_sketch) / 1e6
    time_ns_simMatrix = @elapsed ns_sim_matrix = node_similarity_matrix(node_sketch)
    time_ns_reconstruct = @elapsed ns_predicted_edges = predict_top_edges(ns_sim_matrix, nr_edges)


    # --- Evaluation (precision, recall, and weighted recall for top-k predictions) ---
    top_k = Int(ceil(nr_edges * 0.3))  # predict 30% of total edges
    es_precision, es_recall, es_w_recall = evaluate_prediction(es_predicted_edges[1:min(end, top_k)], adj_list)
    ns_precision, ns_recall, ns_w_recall = evaluate_prediction(ns_predicted_edges[1:min(end, top_k)], adj_list)


    # --- Print results ---
    emph("\nGRAPH")
    info(" - name                  $(name)")
    info(" - nr_nodes              $(nr_nodes)")
    info(" - nr_edges              $(nr_edges)")
    info(" - adj_list_memory       $(f4(adj_list_size_mb)) MB")

    emph("\nEDGESKETCH")
    info(" - sketch_size           $(sketch_size)")
    info(" - neighborhood_k        $(k)")
    info(" - alpha                 $(f2(alpha))")
    info(" - use_sample_edges      $(use_sample_edges)")
    info(" - sketch_memory         $(f4(es_sketch_memory_mb)) MB")
    info(" - time_sketch           $(f4(time_es_sketch)) sec")
    info(" - time_simMatrix        $(f4(time_es_simMatrix)) sec")
    info(" - time_reconstruct      $(f4(time_es_reconstruct)) sec")

    emph("\nNODESKETCH")
    info(" - sketch_size           $(ns_sketch_size)")
    info(" - neighborhood_k        $(k)")
    info(" - alpha                 $(f2(alpha))")
    info(" - sketch_memory         $(f4(ns_sketch_memory_mb)) MB")
    info(" - time_sketch           $(f4(time_ns_sketch)) sec")
    info(" - time_simMatrix        $(f4(time_ns_simMatrix)) sec")
    info(" - time_reconstruct      $(f4(time_ns_reconstruct)) sec")

    emph("\n\nRESULTS (top $(top_k) edges)\n")
    info(" EdgeSketch:")
    info(" - precision             $(f4(es_precision))")
    info(" - recall                $(f4(es_recall))")
    info(" - w_recall              $(f4(es_w_recall))")
    info("")
    info(" NodeSketch:")
    info(" - precision             $(f4(ns_precision))")
    info(" - recall                $(f4(ns_recall))")
    info(" - w_recall              $(f4(ns_w_recall))")
end


# --- Tests Louvain community detection algorithm on sketches
function test_louvain_on_sketches(
        adj_list::Vector{Vector{Tuple{Int, Float64}}},
        name::String = "CUSTOM GRAPH";
        true_communities::Union{Vector{Int}, Nothing} = nothing,
        sketch_size::Int = 100,
        run_louvain_adj_list::Bool = false
    )
    Random.seed!(999)

    # --- Graph statistics ---
    nr_nodes = length(adj_list)
    nr_edges = Int(sum(length(neighbors) for neighbors in adj_list) / 2)


    # --- EdgeSketch generation ---
    t_s = @elapsed graph_sketch = create_graph_sketch(adj_list, sketch_size, 12)
    graph_sketch_verification = create_graph_sketch(adj_list, sketch_size, 445)
    memory_size_adj_list_mb = Base.summarysize(adj_list) / 1e6
    memory_size_sketched_graph_mb = Base.summarysize(graph_sketch) / 1e6


    # --- Louvain on sketches (best of two phases) ---
    t1 = @elapsed communities_1st_phase_sketch = louvain_method_sketch(graph_sketch)
    obtained_modularity_1 = calculate_modularity(adj_list, communities_1st_phase_sketch)
    estimated_modularity_1 = calculate_modularity_sketch(graph_sketch_verification, communities_1st_phase_sketch)
    t2 = @elapsed communities_2nd_phase_sketch = louvain_method_2nd_phase_sketch(graph_sketch, communities_1st_phase_sketch)
    obtained_modularity_2 = calculate_modularity(adj_list, communities_2nd_phase_sketch)
    estimated_modularity_2 = calculate_modularity_sketch(graph_sketch_verification, communities_2nd_phase_sketch)
    best_obtained, best_estimated, best_communities_sketch = estimated_modularity_1 > estimated_modularity_2 ?
        (obtained_modularity_1, estimated_modularity_1, communities_1st_phase_sketch) : 
        (obtained_modularity_2, estimated_modularity_2, communities_2nd_phase_sketch)


    # --- Louvain on adjacency list (reference) ---
    t_al_1, t_al_2, modularity_AL_2, communities_AL_2 = 0.0, 0.0, 0.0, Vector{Set{Int}}()
    if run_louvain_adj_list
        println("Running Louvain on adjacency list...")
        t_al_1 = @elapsed communities_AL_1 = louvain_method(adj_list)
        modularity_AL_1 = calculate_modularity(adj_list, communities_AL_1)
        println("Louvain AL phase 1: $(round(t_al_1, digits=2)) sec, modularity: $(round(modularity_AL_1, digits=4))")
        
        t_al_2 = @elapsed communities_AL_2 = louvain_method_2nd_phase(adj_list, communities_AL_1)
        modularity_AL_2 = calculate_modularity(adj_list, communities_AL_2)
        println("Louvain AL phase 2: $(round(t_al_2, digits=2)) sec, modularity: $(round(modularity_AL_2, digits=4))")
    end

    
    # --- Print results ---
    emph("\nGRAPH")
    info(" - name                 $(name)")
    info(" - nr_nodes             $(nr_nodes)")
    info(" - nr_edges             $(nr_edges)")
    emph("\nSKETCH")
    info(" - sketch_size          $(sketch_size)")
    info(" - sketching_time       $(f2(t_s)) sec")
    info(" - memory (adj_matrix)  $(nr_nodes^2 * 8 / 1e6)  MB")
    info(" - memory (adj_list)    $(greene(f4(memory_size_adj_list_mb)*" MB"))")
    info(" - memory (EdgeSketch)  $(greene(f4(memory_size_sketched_graph_mb)*" MB")) ")

    # knowledge based modularity (if known)
    emph("\nMODULARITY")
    if true_communities !== nothing
        communities = blocks_to_communities(true_communities)
        modularity = calculate_modularity(adj_list, communities)
        info(" - knowledge_based         $(f4(modularity))")
    end
    if run_louvain_adj_list
        info("")
        info(" - Louvain on adj_list         $(greene(f2(t_al_1+t_al_2)*" sec"))")
        info("   modularity                  $(f4(modularity_AL_2))")
        info("   clusters                    $(length(communities_AL_2))")
    end
    info("")
    info(" - Louvain on sketches         $(greene(f2(t1+t2)*" sec"))")
    info("   obtained modularity         $(f4(best_obtained))")
    info("   estimated modularity        $(f4(best_estimated))")
    info("   clusters                    $(length(best_communities_sketch))")
end


# --- Run all tests with default graphs ---
function run_tests(;
    run_sketch_construction::Bool = true,
    run_graph_reconstruction::Bool = true,
    run_louvain::Bool = true
)
    # --- Test 1: Sketch Construction (SBM graph, 10000 nodes) ---
    if run_sketch_construction
        emph("\n" * "="^60)
        emph("TEST: SKETCH CONSTRUCTION")
        emph("="^60)
        Random.seed!(123)
        adj_list_1 = generate_stochastic_block_graph(
            10000,  # nr_nodes
            10,     # nr_blocks
            0.5,    # p
            0.05,   # q
            true,   # use_weights
            true,   # remove_isolated
            false   # return_communities
        )
        test_sketch_construction(adj_list_1, "STOCHASTIC BLOCK MODEL (n=10000, k=10, p=0.5, q=0.05)"; sketch_size=20)
    end

    # --- Test 2: Graph Reconstruction (SBM graph, 2000 nodes) ---
    if run_graph_reconstruction
        emph("\n" * "="^60)
        emph("TEST: GRAPH RECONSTRUCTION")
        emph("="^60)
        Random.seed!(123)
        adj_list_2 = generate_stochastic_block_graph(
            2000,   # nr_nodes
            10,     # nr_blocks
            0.5,    # p
            0.05,   # q
            true,   # use_weights
            true,   # remove_isolated
            false   # return_communities
        )
        test_graph_reconstruction(adj_list_2, "STOCHASTIC BLOCK MODEL (n=2000, k=10, p=0.5, q=0.05)"; sketch_size=30)
    end

    # --- Test 3: Louvain on Sketches (SBM graph, 10000 nodes, with blocks) ---
    if run_louvain
        emph("\n" * "="^60)
        emph("TEST: LOUVAIN ON SKETCHES")
        emph("="^60)
        Random.seed!(1)
        adj_list_3, true_communities_4 = generate_stochastic_block_graph(
            10000,  # nr_nodes
            10,     # nr_blocks
            0.5,    # p
            0.05,   # q
            true,   # use_weights
            true,   # remove_isolated
            true    # return_communities
        )
        test_louvain_on_sketches(adj_list_3, "STOCHASTIC BLOCK MODEL (n=10000, k=10, p=0.5, q=0.05)"; true_communities=true_communities_4, sketch_size=100)
    end

    emph("\n" * "="^60)
    emph("ALL TESTS COMPLETED")
    emph("="^60)
end

end
