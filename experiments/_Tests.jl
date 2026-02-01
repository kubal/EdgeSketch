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
using _kEmbeddings
using _GraphReconstruction
using _GraphModularity
using _SketchModularity
using _Logger

export test_sketch_construction
export test_sketch_operations
export test_graph_reconstruction
export test_louvain_on_sketches



# --- Tests sketch construction, degree estimation accuracy, and memory usage
function test_sketch_construction(sketch_size::Int = 20)
    Random.seed!(123)


    # --- Graph generation ---
    name = "STOCHASTIC BLOCK MODEL"
    nr_nodes = 10000
    nr_blocks = 10
    p = 0.5
    q = 0.05
    use_weights = true
    remove_isolated = true
    return_communities = false
    adj_list = generate_stochastic_block_graph(nr_nodes, nr_blocks, p, q, use_weights, remove_isolated, return_communities)
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
    info(" - parameters      p = $(p), q = $(q)")
    info(" - use_weights     $(use_weights)")
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

    savefig("plots/plot_$(timestamp)_$(nr_nodes)_$(nr_blocks)_$(p)_$(q)_degrees.pdf")
end


# --- Tests sketch operations: union, section, and Jaccard similarity
function test_sketch_operations(sketch_size::Int = 100)

    # --- Graph generation ---
    name = "ERDOS RENYI MODEL"
    nr_nodes = 4
    p = 1.0
    use_weights = false
    remove_isolated = true
    adj_list = generate_erdos_renyi_graph(nr_nodes, p, use_weights, remove_isolated)
    nr_edges = Int(sum(length(neighbors) for neighbors in adj_list) / 2)


    # --- EdgeSketch generation ---
    graph_sketch = create_graph_sketch(adj_list, sketch_size)
    s1 = graph_sketch[1]
    s2 = graph_sketch[2]


    # --- Sketch operations ---
    union_weight = weight(sketch_union(s1, s2))
    jacc_weight = sketch_jacc(s1, s2)
    section_weight = sketch_section(s1, s2)


    # --- Print results ---
    emph("\nGRAPH")
    info(" - name            $(name)")
    info(" - p               $(p)")
    info(" - use_weights     $(use_weights)")
    info(" - nr_nodes        $(nr_nodes)")
    info(" - nr_edges        $(nr_edges)")

    emph("\nSKETCH OPERATIONS ON NODES 1 and 2")
    info(" - union           $(f4(union_weight))")
    info(" - section         $(f4(section_weight))")
    info(" - jaccard         $(f4(jacc_weight))")
end


# --- Tests graph reconstruction using EdgeSketch and NodeSketch similarity matrices
function test_graph_reconstruction(sketch_size::Int = 30)
    Random.seed!(123)


    # --- Graph generation ---
    name = "STOCHASTIC BLOCK MODEL"
    nr_nodes = 2000
    nr_blocks = 10
    p = 0.5
    q = 0.05
    parameters = "p = $(p), q = $(q)"
    use_weights = true
    remove_isolated = true
    return_communities = false
    adj_list = generate_stochastic_block_graph(nr_nodes, nr_blocks, p, q, use_weights, remove_isolated, return_communities)
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
    info(" - parameters            $(parameters)")
    info(" - use_weights           $(use_weights)")
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
function test_louvain_on_sketches(sketch_size::Int = 100)
    Random.seed!(4233)


    # --- Graph generation ---
    name = "STOCHASTIC BLOCK MODEL"
    nr_nodes = 10000
    nr_blocks = 10
    p = 0.5
    q = 0.05
    parameters = "p = $(p), q = $(q)"
    use_weights = true
    remove_isolated = true
    return_communities = true
    t_g = @elapsed adj_list, blocks = generate_stochastic_block_graph(nr_nodes, nr_blocks, p, q, use_weights, remove_isolated, return_communities)


    # --- EdgeSketch generation ---
    t_s = @elapsed graph_sketch = create_graph_sketch(adj_list, sketch_size, 12)
    graph_sketch_verification = create_graph_sketch(adj_list, sketch_size, 445)
    memory_size_adj_list_mb = Base.summarysize(adj_list) / 1e6
    memory_size_sketched_graph_mb = Base.summarysize(graph_sketch) / 1e6
    nr_edges = Int(sum(length(neighbors) for neighbors in adj_list) / 2)
    nr_nodes = length(adj_list)


    # --- Louvain on sketches ---
    t1 = @elapsed communities_1st_phase_sketch = louvain_method_sketch(graph_sketch)
    obtained_modularity_1 = calculate_modularity(adj_list, communities_1st_phase_sketch)
    estimated_modularity_1 = calculate_modularity_sketch(graph_sketch_verification, communities_1st_phase_sketch)
    t2 = @elapsed communities_2nd_phase_sketch = louvain_method_2nd_phase_sketch(graph_sketch, communities_1st_phase_sketch)
    obtained_modularity_2 = calculate_modularity(adj_list, communities_2nd_phase_sketch)
    estimated_modularity_2 = calculate_modularity_sketch(graph_sketch_verification, communities_2nd_phase_sketch)
    best_obtained, best_estimated = estimated_modularity_1 > estimated_modularity_2 ?
        (obtained_modularity_1, estimated_modularity_1) : (obtained_modularity_2, estimated_modularity_2)


    # --- Print results ---
    emph("\nGRAPH")
    info(" - name                 $(name)")
    info(" - parameters           $(parameters)")
    info(" - use_weights          $(use_weights)")
    info(" - nr_nodes             $(nr_nodes)")
    info(" - nr_edges             $(nr_edges)")
    if @isdefined(blocks)
        info(" - nr_blocks            $(nr_blocks)")
    end
    info(" - generating_time      $(f2(t_g)) sec")
    emph("\nSKETCH")
    info(" - sketch_size          $(sketch_size)")
    info(" - sketching_time       $(f2(t_s)) sec")
    info(" - memory (adj_matrix)  $(nr_nodes^2 * 8 / 1e6)  MB")
    info(" - memory (adj_list)    $(greene(f4(memory_size_adj_list_mb)*" MB"))")
    info(" - memory (sketch)      $(greene(f4(memory_size_sketched_graph_mb)*" MB")) ")

    # knowledge based modularity (if known)
    emph("\nMODULARITY")
    if name == "STOCHASTIC BLOCK MODEL"
        communities = blocks_to_communities(blocks)
        modularity = calculate_modularity(adj_list, communities)
        info(" - knowledge_based         $(f4(modularity))")
    end
    info("\n - Louvain on sketches         $(greene(f2(t1+t2)*" sec"))")
    info("   obtained modularity         $(f4(best_obtained))")
    info("   estimated modularity        $(f4(best_estimated))")
end

end
