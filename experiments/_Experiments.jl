__precompile__(false)

module _Experiments

using Random
using StatsBase
using Plots
using Dates
using StatsPlots
using LaTeXStrings

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

export experiment_Louvain_estimator
export experiment_Louvain_varying_m
export experiment_Louvain_dynamic_graphs
export experiment_graph_reconstruction


# --- Tests modularity estimator accuracy across varying sketch sizes 
# --- (Figure 1b in the paper)
function experiment_Louvain_estimator()
    Random.seed!(123)
    
    # --- Graph generation (SBM) ---
    nr_nodes = 10000
    nr_blocks = 10
    p = 0.5
    q = 0.05
    use_weights = true
    remove_isolated = true
    return_communities = true
    adj_list, blocks = generate_stochastic_block_graph(
        nr_nodes, nr_blocks, p, q, use_weights, remove_isolated, return_communities
    )

    # --- Simulation parameters ---
    # ms: sketch sizes to test
    # t: number of trials per sketch size
    ms = collect(20:20:300)
    t = 50


      # --- Louvain on full adjacency list (for reference) ---
    communities_step1 = louvain_method(adj_list)
    communities_AL = louvain_method_2nd_phase(adj_list, communities_step1)
    modularity_AL_ref = calculate_modularity(adj_list, communities_AL)


    # --- Modularity estimator over sketch sizes ---
    n_m = length(ms)
    modularity_est_arr = [Float64[] for _ in 1:n_m]
    means_est = zeros(n_m)
    stds_est = zeros(n_m)

    println("Total steps: $(length(ms)), max m: $(maximum(ms))")
    for (i, sketch_size) in enumerate(ms)
        println("Testing m = $(sketch_size)")
        for trial in 1:t
            seed = 1 + trial + sketch_size * 1000
            graph_sketch = create_graph_sketch(adj_list, sketch_size, seed)
            mod_est = calculate_modularity_sketch(graph_sketch, communities_AL)
            push!(modularity_est_arr[i], mod_est)
        end
        means_est[i] = mean(modularity_est_arr[i])
        stds_est[i] = std(modularity_est_arr[i])
    end


    # --- Plot setup ---
    pgfplotsx()
    default(
        fontfamily = "Computer Modern",
        framestyle = :box,
        guidefontsize = 22,
        tickfontsize = 22,
        legendfontsize = 22
    )


    # Axis ranges
    xlims = (minimum(ms), maximum(ms))
    y_vals_all = vcat(means_est .- stds_est, means_est .+ stds_est, [modularity_AL_ref])
    ymin, ymax = minimum(y_vals_all), maximum(y_vals_all)
    y_padding = 0.02 * (ymax - ymin)
    ylims = (ymin - y_padding, ymax + y_padding)

    # Bounds +/- 0.24/sqrt(m) around AL level
    bounds = 0.24 ./ sqrt.(ms)
    upper_bound = modularity_AL_ref .+ bounds
    lower_bound = modularity_AL_ref .- bounds

    plt = plot(
        size = (1000, 600),
        xlabel = "Sketch size (m)",
        ylabel = "Modularity",
        legend = :bottomright,
        margin = 5Plots.mm,
        grid = :y,
        gridalpha = 0.2,
        xlims = xlims,
        ylims = ylims,
    )

    # AL reference line
    hline!(plt, [modularity_AL_ref],
        label = "AL",
        color = :black,
        lw = 2,
        dash_pattern = "on 20pt off 10pt",
    )

    # +/- 0.24/sqrt(m) bounds
    plot!(plt, ms, upper_bound,
        label = "",
        color = :black,
        linestyle = :dash,
        lw = 2
    )
    plot!(plt, ms, lower_bound,
        label = L"$\pm 0.24/\sqrt{m}$",
        color = :black,
        linestyle = :dash,
        lw = 2
    )

    # ES estimated (mean and std)
    plot!(plt, ms, means_est,
        ribbon = stds_est,
        fillalpha = 0.3,
        label = "ES estimated (mean and std)",
        color = RGB(0.0, 0.4, 0.7),
        lw = 2,
        marker = :circle,
        markersize = 7,
        markerstrokecolor = :black,
        markerstrokewidth = 1,
    )

    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    filename = "Plots/EST_accuracy_$(timestamp)_nodes$(nr_nodes)_blocks$(nr_blocks)_p$(p)_q$(q)_trials$(t).pdf"
    savefig(plt, filename)

end


# --- Tests Louvain algorithm on sketches with varying sketch sizes
# --- (Figures 1a and 1c in the paper)
function experiment_Louvain_varying_m()
    Random.seed!(123)
    

    # --- Graph generation (SBM) ---
    nr_nodes = 10000
    nr_blocks = 10
    p = 0.5
    q = 0.05
    use_weights = true
    remove_isolated = true
    return_communities = true
    adj_list, blocks = generate_stochastic_block_graph(
        nr_nodes, nr_blocks, p, q, use_weights, remove_isolated, return_communities
    )


    # --- Simulation parameters ---
    # ms: sketch sizes to test
    # t: number of trials per sketch size
    ms = collect(20:20:300)
    t = 50

    
    # Calculate ground-truth modularity from known block structure
    communities = blocks_to_communities(blocks)
    modularity = calculate_modularity(adj_list, communities)
    println("Knowledge-based modularity: ", f4(modularity))

    # Reference modularity from Louvain on full adjacency list
    communities_AL = louvain_method(adj_list)
    communities_AL2 = louvain_method_2nd_phase(adj_list, communities_AL)
    modularity_AL_ref = calculate_modularity(adj_list, communities_AL2)
    println("Louvain on adjacency list: ", f4(modularity_AL_ref))


    # --- Run Louvain on sketches for each sketch size and trial ---
    println("Running $(t) trials for $(length(ms)) sketch sizes (max m=$(maximum(ms)))")
    n_m = length(ms)
    modularity_ES_arr = [Float64[] for _ in 1:n_m]
    modularity_est_arr = [Float64[] for _ in 1:n_m]
    means_ES = zeros(n_m)
    stds_ES = zeros(n_m)
    means_est = zeros(n_m)
    stds_est = zeros(n_m)

    for (i, sketch_size) in enumerate(ms)
        println("m=$(sketch_size) ($(i)/$(length(ms)))")
        for trial in 1:t
            seed1 = 42 + trial + sketch_size * 1000
            seed2 = 1_000_042 + trial + sketch_size * 2000
            graph_sketch1 = create_graph_sketch(adj_list, sketch_size, seed1)
            graph_sketch2 = create_graph_sketch(adj_list, sketch_size, seed2)
            communities_ES1 = louvain_method_sketch(graph_sketch1)
            communities_ES2 = louvain_method_2nd_phase_sketch(graph_sketch1, communities_ES1)
            obtained_modularity_1 = calculate_modularity(adj_list, communities_ES1)
            obtained_modularity_2 = calculate_modularity(adj_list, communities_ES2)
            estimated_modularity_1 = calculate_modularity_sketch(graph_sketch2, communities_ES1)
            estimated_modularity_2 = calculate_modularity_sketch(graph_sketch2, communities_ES2)
            obt, est = estimated_modularity_1 > estimated_modularity_2 ? 
                (obtained_modularity_1, estimated_modularity_1) : 
                (obtained_modularity_2, estimated_modularity_2)
            push!(modularity_ES_arr[i], obt)
            push!(modularity_est_arr[i], est)
        end
        means_ES[i] = mean(modularity_ES_arr[i])
        stds_ES[i] = std(modularity_ES_arr[i])
        means_est[i] = mean(modularity_est_arr[i])
        stds_est[i] = std(modularity_est_arr[i])
    end


    # --- Plot results: modularity vs sketch size ---
    pgfplotsx()
    default(
        fontfamily = "Computer Modern",
        framestyle = :box,
        guidefontsize = 22,
        tickfontsize = 22,
        legendfontsize = 22
    )

    xlims = (minimum(ms), maximum(ms))
    y_vals_all = vcat(means_ES .- stds_ES, means_ES .+ stds_ES, means_est .- stds_est, means_est .+ stds_est, [modularity_AL_ref])
    ymin = minimum(y_vals_all)
    ymax = maximum(y_vals_all)
    ylims = (ymin, ymax + 0.1)

    plt1 = plot(
        size = (1000, 600),
        xlabel = "Sketch size (m)",
        ylabel = "Modularity",
        legend = :bottomright,
        margin = 5Plots.mm,
        grid = :y,
        gridalpha = 0.2,
        xlims = xlims,
        ylims = ylims,
    )
    hline!(plt1, [modularity_AL_ref],
        label = "AL",
        color = :black,
        lw = 2,
        dash_pattern = "on 20pt off 10pt",
    )
    plot!(plt1, ms, means_ES,
        ribbon = stds_ES,
        fillalpha = 0.3,
        label = "ES (mean and std)",
        color = RGB(0.9, 0.4, 0.0),
        lw = 2,
        marker = :circle,
        markersize = 6,
        markerstrokecolor = :black,
        markerstrokewidth = 1,
    )

    plt2 = plot(
        size = (1000, 600),
        xlabel = "Sketch size (m)",
        ylabel = "Modularity",
        legend = :bottomright,
        margin = 5Plots.mm,
        grid = :y,
        gridalpha = 0.2,
        xlims = xlims,
        ylims = ylims,
    )
    hline!(plt2, [modularity_AL_ref],
        label = "AL",
        color = :black,
        lw = 2,
        dash_pattern = "on 20pt off 10pt",
    )
    plot!(plt2, ms, means_est,
        ribbon = stds_est,
        fillalpha = 0.3,
        label = "ES estimated (mean and std)",
        color = RGB(0.0, 0.4, 0.7),
        lw = 2,
        marker = :circle,
        markersize = 6,
        markerstrokecolor = :black,
        markerstrokewidth = 1,
    )

    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    savefig(plt1, "plots/ES_n$(nr_nodes)_k$(nr_blocks)_p$(p)_q$(q)_trials$(t)_$(timestamp).pdf")
    savefig(plt2, "plots/EST_n$(nr_nodes)_k$(nr_blocks)_p$(p)_q$(q)_trials$(t)_$(timestamp).pdf")
end


# --- Tests Louvain on dynamically growing graphs using streaming edge batches
# --- (Figure 1d in the paper) 
function experiment_Louvain_dynamic_graphs()
    Random.seed!(1234)
    

    # --- Graph generation (SBM) ---
    nr_nodes = 10000
    nr_blocks = 10
    p = 0.5
    q = 0.05
    use_weights = true
    remove_isolated = true
    return_communities = true

    adj_list, blocks = generate_stochastic_block_graph(
        nr_nodes, nr_blocks, p, q, use_weights, remove_isolated, return_communities
    )
    nr_nodes = length(adj_list)
    nr_edges = Int(sum(length(neighbors) for neighbors in adj_list) / 2)


    # --- Experiment parameters ---
    sketch_size = 100   # size of the edge sketch
    step_size = 60000   # number of edges processed in a single batch
    step_limit = 60     # maximum number of batches to run


    # Initialize variables
    graph_sketch1 = [EdgeSketch(sketch_size) for _ in 1:nr_nodes]
    graph_sketch2 = [EdgeSketch(sketch_size) for _ in 1:nr_nodes]

    left_edges = adj_list
    current_adj_list = [Vector{Tuple{Int64,Float64}}() for _ in 1:nr_nodes]
    seen_edges = Dict(i => Set{Int64}() for i in 1:nr_nodes)

    steps = min(ceil(Int, nr_edges / step_size), step_limit)
    step = 0

    modularity2 = Vector{Float64}(undef, steps)
    obtained_modularity2 = Vector{Float64}(undef, steps)
    estimated_modularity2 = Vector{Float64}(undef, steps)


    # --- Main loop: process edges in batches ---
    println("Number of batches: $steps")

    while any(!isempty, left_edges) && step < step_limit
        step += 1
        println("Processing batch: $step / $steps")

        # Update adjacency lists
        left_edges, step_edges = temporal_graph_step(left_edges, step_size)
        merge_adjacency_lists!(current_adj_list, step_edges, seen_edges)

        # Update sketches
        step_sketch1 = create_graph_sketch(step_edges, sketch_size, 1)
        step_sketch2 = create_graph_sketch(step_edges, sketch_size, 100000)
        graph_sketch_union!(graph_sketch1, step_sketch1)
        graph_sketch_union!(graph_sketch2, step_sketch2)

        # Louvain on adjacency lists
        communities_1st_phase = louvain_method(current_adj_list)
        communities_2nd_phase = louvain_method_2nd_phase(current_adj_list, communities_1st_phase)
        modularity2[step] = calculate_modularity(current_adj_list, communities_2nd_phase)

        # Louvain on sketches (1st phase)
        communities_1st_phase_sketch = louvain_method_sketch(graph_sketch1)
        obtained_modularity_1  = calculate_modularity(current_adj_list, communities_1st_phase_sketch)
        estimated_modularity_1 = calculate_modularity_sketch(graph_sketch2, communities_1st_phase_sketch)

        # Louvain on sketches 
        communities_2nd_phase_sketch = louvain_method_2nd_phase_sketch(graph_sketch1, communities_1st_phase_sketch)
        obtained_modularity_2  = calculate_modularity(current_adj_list, communities_2nd_phase_sketch)
        estimated_modularity_2 = calculate_modularity_sketch(graph_sketch2, communities_2nd_phase_sketch)
        obtained_modularity2[step], estimated_modularity2[step] =
            estimated_modularity_1 > estimated_modularity_2 ?
            (obtained_modularity_1, estimated_modularity_1) :
            (obtained_modularity_2, estimated_modularity_2)
       
    end

    # --- Plot setup ---
    pgfplotsx()
    default(
        fontfamily = "Computer Modern",
        framestyle = :box,
        guidefontsize = 22,
        tickfontsize = 22,
        legendfontsize = 22
    )

    # Axis ranges
    y_vals_all = vcat(
        obtained_modularity2[1:step],
        estimated_modularity2[1:step],
        modularity2[1:step]
    )
    ymin = minimum(y_vals_all)
    ymax = maximum(y_vals_all)
    y_padding = 0.1 * (ymax - ymin)
    legend_padding = 0.15 * (ymax - ymin)
    x_padding = 1
    xlims = (1 - x_padding, step)
    ylims = (ymin - y_padding, ymax + y_padding + legend_padding)

    plt = plot(
        size = (1000, 600),
        xlabel = "Batch number",
        ylabel = "Modularity",
        legend = :bottomright,
        margin = 5Plots.mm,
        grid = :y,
        gridalpha = 0.2,
        xlims = xlims,
        ylims = ylims
    )

    # AL (adjacency list reference)
    plot!(plt, modularity2,
        label = "AL",
        color = :black,
        lw = 3,
        linestyle = :solid
    )

    # ES (obtained from sketch)
    plot!(plt, obtained_modularity2,
        label = "",
        color = RGB(1.0, 0.25, 0.0),
        lw = 3,
        dash_pattern = "on 20pt off 10pt"
    )

    # Legend for ES (dummy point)
    plot!(plt, [0],
        label = "ES",
        color = RGB(1.0, 0.25, 0.0),
        lw = 3,
        dash_pattern = "on 6pt off 5pt"
    )

    # ES estimated
    plot!(plt, estimated_modularity2,
        label = "ES estimated",
        color = RGB(0.0, 0.4, 0.7),
        lw = 4,
        linestyle = :dot,
        dash_pattern = "on 5pt off 7pt"
    )

    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    savefig("Plots/dynamic_graphs_$(timestamp)_$(nr_nodes)_$(nr_blocks)_$(p)_$(q)_$(sketch_size)_$(use_weights).pdf")
end


# --- Compares EdgeSketch and NodeSketch precision for graph reconstruction
# --- (Figures 2a and 2b in the paper) 
function experiment_graph_reconstruction()
    Random.seed!(123)
    

    # --- Graph generation (SBM) ---
    nr_nodes = 10000
    nr_blocks = 10
    p = 0.5
    q = 0.05
    use_weights = true
    remove_isolated = true
    return_communities = false
    adj_list = generate_stochastic_block_graph(nr_nodes, nr_blocks, p, q, use_weights, remove_isolated, return_communities)
    nr_edges = Int(sum(length(neighbors) for neighbors in adj_list) / 2)


    # --- Evaluation parameters ---
    top_edges_percent = 1       # fraction of edges to analyze
    num_steps = 100             # number of evaluation steps along top edges
    k, alpha = 4, 0.2           # embedding/similarity parameters

    edge_sketch_with_sample_size = 100
    edge_sketch_no_sample_size  = edge_sketch_with_sample_size * 3
    node_sketch_size            = edge_sketch_with_sample_size * 3

    println("Number of edges: $(nr_edges)")
    println("Top edges to analyze: $(top_edges_percent * nr_edges)")

    
    # --- Create sketches and predictions ---
    edge_sketch_with_sample = create_graph_sketch(adj_list, edge_sketch_with_sample_size)
    edge_sketch_with_sample_sim = create_similarity_matrix(edge_sketch_with_sample, k, alpha, true)
    edge_sketch_with_sample_pred = predict_top_edges(edge_sketch_with_sample_sim, nr_edges)
    println("EdgeSketch WITH sample created (m=$(edge_sketch_with_sample_size))")

    edge_sketch_no_sample = create_graph_sketch(adj_list, edge_sketch_no_sample_size)
    edge_sketch_no_sample_sim = create_similarity_matrix(edge_sketch_no_sample, k, alpha, false)
    edge_sketch_no_sample_pred = predict_top_edges(edge_sketch_no_sample_sim, nr_edges)
    println("EdgeSketch WITHOUT sample created (m=$(edge_sketch_no_sample_size))")

    node_sketch = nodesketch(adj_list, k, node_sketch_size, alpha)
    node_sketch_sim_matrix = node_similarity_matrix(node_sketch)
    node_predicted_edges = predict_top_edges(node_sketch_sim_matrix, nr_edges)
    println("NodeSketch created (m=$(node_sketch_size))")


    # --- Memory usage ---
    edge_sketch_with_sample_mb = Base.summarysize(edge_sketch_with_sample) / (1024 * 1024)
    edge_sketch_no_sample_mb   = Base.summarysize(edge_sketch_no_sample) / (1024 * 1024)
    node_sketch_mb             = Base.summarysize(node_sketch) / (1024 * 1024)
    println("EdgeSketch WITH sample size: $(round(edge_sketch_with_sample_mb, digits=2)) MB")
    println("EdgeSketch WITHOUT sample size: $(round(edge_sketch_no_sample_mb, digits=2)) MB")
    println("NodeSketch size: $(round(node_sketch_mb, digits=2)) MB")


    # --- Prepare evaluation range and evaluate ---
    max_top_edges = round(Int, nr_edges * top_edges_percent)
    step_size = max(1, div(max_top_edges, num_steps))
    top_range = 1:step_size:max_top_edges
    steps = length(top_range)

    node_sketch_precision            = Vector{Float64}(undef, steps)
    edge_sketch_with_sample_precision = Vector{Float64}(undef, steps)
    edge_sketch_no_sample_precision   = Vector{Float64}(undef, steps)

    for (step, top_edges) in enumerate(top_range)
        println("Step $step/$steps, top_edges = $top_edges")

        node_sketch_precision[step], _, _            = evaluate_prediction(node_predicted_edges[1:min(end, top_edges)], adj_list)
        edge_sketch_with_sample_precision[step], _, _ = evaluate_prediction(edge_sketch_with_sample_pred[1:min(end, top_edges)], adj_list)
        edge_sketch_no_sample_precision[step], _, _   = evaluate_prediction(edge_sketch_no_sample_pred[1:min(end, top_edges)], adj_list)
    end


    # X-axis tick labels (percentage of edges)
    x_values  = collect(1:steps)
    pct       = 0:25:100
    positions = round.(Int, steps .* (pct ./ 100))
    positions[1] = 1
    labels    = [string(Int(p * top_edges_percent), "\\%") for p in pct]


    # --- Plot setup ---
    pgfplotsx()
    default(
        fontfamily     = "Computer Modern",
        framestyle     = :box,
        guidefontsize  = 22,
        tickfontsize   = 22,
        legendfontsize = 22
    )

    y_vals_all = vcat(node_sketch_precision, edge_sketch_with_sample_precision, edge_sketch_no_sample_precision)
    ymin = 0
    ymax = min(1.05, maximum(y_vals_all) + 0.1)

    plt = plot(
        size      = (1000, 600),
        xlabel    = "Top edges (\\% of total)",
        ylabel    = "Precision",
        xticks    = (positions, labels),
        xlims     = (1, steps),
        ylims     = (ymin, ymax),
        legend    = :topright,
        margin    = 5Plots.mm,
        grid      = :y,
        gridalpha = 0.2
    )

    # --- Plot series ---
    plot!(plt, x_values, edge_sketch_with_sample_precision,
        label        = "EdgeSketch\\ \\ (m=$(edge_sketch_with_sample_size))",
        color        = RGB(0.0, 0.4, 0.7),
        lw           = 3
    )

    plot!(plt, x_values, edge_sketch_no_sample_precision,
        label        = "EdgeSketch* (m=$(edge_sketch_no_sample_size))",
        color        = RGB(1.0, 0.5, 0.0),
        lw           = 3,
        dash_pattern = "on 5pt off 5pt"
    )

    plot!(plt, x_values, node_sketch_precision,
        label        = "NodeSketch\\ \\ (m=$(node_sketch_size))",
        color        = :black,
        lw           = 3,
        dash_pattern = "on 15pt off 15pt"
    )

    # --- Save ---
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    savefig(plt, "plots/precision_equal_memory_n$(nr_nodes)_b$(nr_blocks)_p$(p)_q$(q)_k$(k)_a$(alpha)_$(timestamp).pdf")
end

end
