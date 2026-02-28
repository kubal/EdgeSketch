__precompile__(false)

module _Epinions

using Random
using StatsBase
using Plots
using Dates
using StatsPlots
using LaTeXStrings
using SparseArrays
using Base.Threads

using _FastExpSketch
using _GraphModels
using _EdgeSketch
using _NodeSketch
using _SketchOperations
using _GraphReconstruction
using _GraphModularity
using _SketchModularity
using _Logger


export load_epinions_graph
export epinions_sketch_construction
export epinions_louvain_on_sketches
export epinions_louvain_streaming
export epinions_graph_reconstruction

# --- Loads Epinions bipartite graph and projects it to an item-item graph.
# The input file has format: user item rating (three columns).
# Source: https://networkrepository.com/rec-epinions.php
# The output is an item-item graph where edge weight = number of common users.
# min_weight: filters out edges with fewer than min_weight common users (default 1.0)
# max_users_per_item: skip items with more than this many users (default Inf, set to e.g. 200 to speed up)
function load_epinions_graph(; min_weight::Real = 30.0, max_users_per_item::Real = Inf)
    filepath = joinpath(@__DIR__, "..", "data", "rec-epinions", "rec-epinions.edges")
    
    # Step 1: Read bipartite edges (user, item, rating) and collect per-user item lists
    user_to_items = Dict{Int, Set{Int}}()
    all_items = Set{Int}()
    bipartite_edge_count = 0
    
    println("Loading Epinions bipartite graph...")
    open(filepath, "r") do file
        for line in eachline(file)
            parts = split(line)
            if length(parts) >= 2
                user = parse(Int, parts[1])
                item = parse(Int, parts[2])
                # rating in parts[3] is ignored — we only count co-occurrences
                push!(all_items, item)
                if !haskey(user_to_items, user)
                    user_to_items[user] = Set{Int}()
                end
                push!(user_to_items[user], item)
                bipartite_edge_count += 1
            end
        end
    end
    
    nr_users = length(user_to_items)
    nr_items_raw = length(all_items)
    println("Loaded $bipartite_edge_count bipartite edges, $nr_users unique users, $nr_items_raw unique items")
    
    # Step 2: Remap item and user IDs to consecutive indices
    item_list = sort(collect(all_items))
    item_to_idx = Dict(item => i for (i, item) in enumerate(item_list))
    n_items = length(item_list)
    
    user_list = sort(collect(keys(user_to_items)))
    user_to_uidx = Dict(u => i for (i, u) in enumerate(user_list))
    n_users = length(user_list)
    
    # Step 3: Build sparse bipartite matrices (CSC format)
    # B  (users × items): column j gives users who rated item j
    # Bt (items × users): column u gives items rated by user u
    I_vec = Int[]
    J_vec = Int[]
    for (user, items) in user_to_items
        uid = user_to_uidx[user]
        for item in items
            push!(I_vec, uid)
            push!(J_vec, item_to_idx[item])
        end
    end
    
    B = sparse(I_vec, J_vec, ones(Float64, length(I_vec)), n_users, n_items)
    Bt = sparse(B')
    
    # Free intermediate data before heavy computation
    I_vec = nothing; J_vec = nothing
    user_to_items = nothing; all_items = nothing
    GC.gc()
    
    # Step 4: Analyze item-user distribution and compute projection
    println("\nAnalyzing item-user distribution...")
    B_rv = rowvals(B)
    Bt_rv = rowvals(Bt)
    
    users_per_item = [length(nzrange(B, j)) for j in 1:n_items]
    
    # Filter items: must have >= min_weight users (otherwise can't create any edge)
    # and <= max_users_per_item (to avoid super-slow popular items)
    min_users_needed = Int(ceil(min_weight))
    items_to_process = [j for j in 1:n_items if users_per_item[j] >= min_users_needed && users_per_item[j] <= max_users_per_item]
    
    items_too_few = count(u -> u < min_users_needed, users_per_item)
    
    println("Items: $(n_items) total, $(length(items_to_process)) to process (skipped $(items_too_few) with <$(min_users_needed) users)")
    println("Users per item (all): min=$(minimum(users_per_item)), max=$(maximum(users_per_item)), median=$(Int(median(users_per_item))), mean=$(round(mean(users_per_item), digits=1))")
    
    # Compute item-item projection in parallel
    # For each item j: find users who rated it, collect all other items those users rated.
    nt = Threads.nthreads()
    println("\nComputing item-item projection (threaded, $nt threads, $(length(items_to_process)) items, min_weight=$min_weight)...")
    
    adj_list_raw = Vector{Vector{Tuple{Int,Float64}}}(undef, n_items)
    for j in 1:n_items
        adj_list_raw[j] = Tuple{Int,Float64}[]
    end
    
    accumulators = [zeros(Float64, n_items) for _ in 1:nt]
    
    progress = Threads.Atomic{Int}(0)
    edge_counter = Threads.Atomic{Int}(0)
    n_to_process = length(items_to_process)
    t_start = time()
    
    Threads.@threads for idx in 1:n_to_process
        j = items_to_process[idx]
        tid = Threads.threadid()
        acc = accumulators[tid]
        
        # Collect all items co-rated by users of item j
        touched = Int[]
        for idx_b in nzrange(B, j)
            u = B_rv[idx_b]
            for k in nzrange(Bt, u)
                item_k = Bt_rv[k]
                if item_k != j
                    if acc[item_k] == 0.0
                        push!(touched, item_k)
                    end
                    acc[item_k] += 1.0
                end
            end
        end
        
        # Extract neighbors above threshold, reset accumulator
        neighbors = Tuple{Int,Float64}[]
        for item_k in touched
            if acc[item_k] >= min_weight
                push!(neighbors, (item_k, acc[item_k]))
            end
            acc[item_k] = 0.0
        end
        adj_list_raw[j] = neighbors
        Threads.atomic_add!(edge_counter, length(neighbors))
        
        # Progress reporting (every 100K items processed)
        cnt = Threads.atomic_add!(progress, 1)
        if (cnt + 1) % 100000 == 0
            done = cnt + 1
            elapsed = time() - t_start
            pct = round(100.0 * done / n_to_process, digits=1)
            eta = round(elapsed * (n_to_process - done) / done, digits=0)
            rate = round(done / elapsed, digits=1)
            println("  $(done)/$(n_to_process) ($(pct)%) | $(round(elapsed, digits=1))s | ETA ~$(Int(eta))s | $(rate) items/s | ~$(edge_counter[] ÷ 2) edges")
        end
    end
    
    elapsed_total = round(time() - t_start, digits=1)
    total_directed = sum(length(adj_list_raw[j]) for j in 1:n_items)
    println("Projection done in $(elapsed_total)s, $(total_directed ÷ 2) unique edges")
    
    # Free accumulators and sparse matrices
    accumulators = nothing; B = nothing; Bt = nothing
    GC.gc()
    
    # Step 5: Remove isolated nodes and remap indices
    println("Removing isolated nodes and remapping...")
    non_isolated = [j for j in 1:n_items if !isempty(adj_list_raw[j])]
    old_to_new = Dict(old => new_idx for (new_idx, old) in enumerate(non_isolated))
    nr_nodes = length(non_isolated)
    
    adj_list = Vector{Vector{Tuple{Int,Float64}}}(undef, nr_nodes)
    new_item_list = Vector{Int}(undef, nr_nodes)
    
    for (new_idx, old_j) in enumerate(non_isolated)
        neighbors = Tuple{Int,Float64}[]
        for (k, w) in adj_list_raw[old_j]
            if haskey(old_to_new, k)
                push!(neighbors, (old_to_new[k], w))
            end
        end
        adj_list[new_idx] = neighbors
        new_item_list[new_idx] = item_list[old_j]
    end
    
    adj_list_raw = nothing
    GC.gc()
    
    nr_edges = sum(length(neighbors) for neighbors in adj_list) ÷ 2
    edge_to_node_ratio = round(nr_edges / nr_nodes, digits=2)
    
    println("Removed $(n_items - nr_nodes) isolated items")
    println("Epinions item-item graph: $nr_nodes nodes (|V|), $nr_edges edges (|E|), |E|/|V| = $edge_to_node_ratio")
    
    adj_list_mb = Base.summarysize(adj_list) / (1024 * 1024)
    println("Adjacency list size: $(round(adj_list_mb, digits=2)) MB")
    
    return adj_list, new_item_list
end


# --- Sketch construction on Epinions item-item graph ---
function epinions_sketch_construction(; sketch_size::Int = 100)
    Random.seed!(123)

    # --- Load Epinions item-item graph ---
    adj_list, _ = load_epinions_graph(min_weight=50)
    name = "EPINIONS (item-item)"
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
    plot_path = "plots/epinions_degrees_$(timestamp)_n$(nr_nodes)_m$(sketch_size).pdf"
    savefig(plot_path)
    info(" - saved to        $(plot_path)")
end


# --- Louvain community detection on Epinions item-item graph ---
function epinions_louvain_on_sketches(; 
        sketch_size::Int = 100,
        min_weight::Int = 30,
        max_users_per_item::Real = Inf)
    Random.seed!(1233)

    # --- Load Epinions item-item graph ---
    adj_list, _ = load_epinions_graph(min_weight=min_weight, max_users_per_item=max_users_per_item)
    name = "EPINIONS (item-item)"
    nr_nodes = length(adj_list)
    nr_edges = Int(sum(length(neighbors) for neighbors in adj_list) / 2)


    # --- EdgeSketch generation ---
    t_s = @elapsed graph_sketch = create_graph_sketch(adj_list, sketch_size, 12)
    graph_sketch_verification = create_graph_sketch(adj_list, sketch_size, 445)
    memory_size_adj_list_mb = Base.summarysize(adj_list) / 1e6
    memory_size_sketched_graph_mb = Base.summarysize(graph_sketch) / 1e6
  
    # --- Louvain on sketches ---
    println("Running Louvain on sketches...")
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
    println("Running Louvain on adjacency list...")
    t_al_1 = @elapsed communities_AL_1 = louvain_method(adj_list)
    modularity_AL_1 = calculate_modularity(adj_list, communities_AL_1)
    t_al_2 = @elapsed communities_AL_2 = louvain_method_2nd_phase(adj_list, communities_AL_1)
    modularity_AL_2 = calculate_modularity(adj_list, communities_AL_2)
    

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

    emph("\nMODULARITY")
    info(" - Louvain on adj_list         $(greene(f2(t_al_1+t_al_2)*" sec"))")
    info("   modularity                  $(f4(modularity_AL_2))")
    info("   clusters                    $(length(communities_AL_2))")
    info("")
    info(" - Louvain on sketches         $(greene(f2(t1+t2)*" sec"))")
    info("   obtained modularity         $(f4(best_obtained))")
    info("   estimated modularity        $(f4(best_estimated))")
    info("   clusters                    $(length(best_communities_sketch))")
end


# --- Streaming Louvain: builds EdgeSketch directly without storing adj_list ---
# This allows processing very large graphs (min_weight=1) that would otherwise
# exceed memory if we tried to store the full adjacency list.
function epinions_louvain_streaming(; 
        sketch_size::Int = 100, 
        min_weight::Real = 1.0, 
        max_users_per_item::Real = Inf,
        seed::Int = 123
    )
    Random.seed!(seed)
    
    filepath = joinpath(@__DIR__, "..", "data", "rec-epinions", "rec-epinions.edges")
    
    # Step 1: Read bipartite edges (user, item, rating) and collect per-user item lists
    user_to_items = Dict{Int, Set{Int}}()
    all_items = Set{Int}()
    bipartite_edge_count = 0
    
    println("=== STREAMING EDGESKETCH ===")
    println("Parameters: sketch_size=$sketch_size, min_weight=$min_weight, max_users_per_item=$max_users_per_item")
    println("\nLoading Epinions bipartite graph...")
    open(filepath, "r") do file
        for line in eachline(file)
            parts = split(line)
            if length(parts) >= 2
                user = parse(Int, parts[1])
                item = parse(Int, parts[2])
                push!(all_items, item)
                if !haskey(user_to_items, user)
                    user_to_items[user] = Set{Int}()
                end
                push!(user_to_items[user], item)
                bipartite_edge_count += 1
            end
        end
    end
    
    nr_users = length(user_to_items)
    nr_items_raw = length(all_items)
    println("Loaded $bipartite_edge_count bipartite edges, $nr_users unique users, $nr_items_raw unique items")
    
    # Step 2: Remap item and user IDs to consecutive indices
    item_list = sort(collect(all_items))
    item_to_idx = Dict(item => i for (i, item) in enumerate(item_list))
    n_items = length(item_list)
    
    user_list = sort(collect(keys(user_to_items)))
    user_to_uidx = Dict(u => i for (i, u) in enumerate(user_list))
    n_users = length(user_list)
    
    # Step 3: Build sparse bipartite matrices (CSC format)
    I_vec = Int[]
    J_vec = Int[]
    for (user, items) in user_to_items
        uid = user_to_uidx[user]
        for item in items
            push!(I_vec, uid)
            push!(J_vec, item_to_idx[item])
        end
    end
    
    B = sparse(I_vec, J_vec, ones(Float64, length(I_vec)), n_users, n_items)
    Bt = sparse(B')
   
   
    # Free intermediate data
    I_vec = nothing; J_vec = nothing
    user_to_items = nothing; all_items = nothing
    GC.gc()
    
    # Step 4: Filter items by user count (skip items with too few/many users)
    println("\nAnalyzing item-user distribution...")
    B_rv = rowvals(B)
    Bt_rv = rowvals(Bt)
    
    users_per_item = [length(nzrange(B, j)) for j in 1:n_items]
    
    min_users_needed = Int(ceil(min_weight))
    items_to_process = [j for j in 1:n_items if users_per_item[j] >= min_users_needed && users_per_item[j] <= max_users_per_item]
    items_to_process_set = Set(items_to_process)
    
    items_too_few = count(u -> u < min_users_needed, users_per_item)
    
    println("Items: $(n_items) total, $(length(items_to_process)) to process (skipped $(items_too_few) with <$(min_users_needed) users)")
    println("Users per item (all): min=$(minimum(users_per_item)), max=$(maximum(users_per_item)), median=$(Int(median(users_per_item))), mean=$(round(mean(users_per_item), digits=1))")
    
    # Step 5: Create EdgeSketch directly (streaming, no adj_list storage)
    nt = Threads.nthreads()
    println("\nCreating EdgeSketch directly (streaming, $nt threads, $(length(items_to_process)) items)...")
    
    h = x -> (hash((x, seed)) % Int64(1e9)) / 1e9
    
    # Create empty sketches for all items (items without edges will stay empty)
    graph_sketch = Vector{EdgeSketch}(undef, n_items)
    for j in 1:n_items
        graph_sketch[j] = EdgeSketch(sketch_size)
    end
    
    accumulators = [zeros(Float64, n_items) for _ in 1:nt]
    
    progress = Threads.Atomic{Int}(0)
    total_edges_directed = Threads.Atomic{Int}(0)
    n_to_process = length(items_to_process)
    t_start = time()
    
    Threads.@threads for idx in 1:n_to_process
        j = items_to_process[idx]
        tid = Threads.threadid()
        acc = accumulators[tid]
        
        # Collect all items co-rated by users of item j
        touched = Int[]
        for idx_b in nzrange(B, j)
            u = B_rv[idx_b]
            for k in nzrange(Bt, u)
                item_k = Bt_rv[k]
                if item_k != j
                    if acc[item_k] == 0.0
                        push!(touched, item_k)
                    end
                    acc[item_k] += 1.0
                end
            end
        end
        
        # Build StreamElement list for neighbors meeting threshold
        # Only include neighbors that are also in items_to_process (will have valid sketches)
        stream = StreamElement[]
        for item_k in touched
            w = acc[item_k]
            acc[item_k] = 0.0  # reset accumulator
            if w >= min_weight && item_k in items_to_process_set
                push!(stream, StreamElement(j, item_k, w))
            end
        end
        
        # Create sketch for this node using batch fast_expsketch
        if !isempty(stream)
            embedding, sample = fast_expsketch(stream, sketch_size, h)
            graph_sketch[j] = EdgeSketch(embedding, sample, sketch_size)
        end
        
        Threads.atomic_add!(total_edges_directed, length(stream))
        
        # Progress reporting (every 100K items processed)
        cnt = Threads.atomic_add!(progress, 1)
        if (cnt + 1) % 100000 == 0
            done = cnt + 1
            elapsed = time() - t_start
            pct = round(100.0 * done / n_to_process, digits=1)
            eta = round(elapsed * (n_to_process - done) / done, digits=0)
            rate = round(done / elapsed, digits=1)
            edges_so_far = total_edges_directed[] ÷ 2
            println("  $(done)/$(n_to_process) ($(pct)%) | $(round(elapsed, digits=1))s | ETA ~$(Int(eta))s | $(rate) items/s | ~$(edges_so_far) edges")
        end
    end
    
    elapsed_sketch = round(time() - t_start, digits=1)
    nr_edges_real = total_edges_directed[] ÷ 2
    println("Sketch creation done in $(elapsed_sketch)s, $(nr_edges_real) edges")
    
    # Free accumulators and sparse matrices
    accumulators = nothing; B = nothing; Bt = nothing
    GC.gc()
    
    # Step 6: Calculate graph statistics
    nr_nodes = count(j -> is_initialized(graph_sketch[j]), 1:n_items)
    edge_to_node_ratio = round(nr_edges_real / nr_nodes, digits=2)
    adj_list_mb = (nr_edges_real * 2 * 16 + nr_nodes * 40) / (1024 * 1024)
    sketch_mb = Base.summarysize(graph_sketch) / (1024 * 1024)
    
    # Step 7: Run Louvain on EdgeSketch
    println("\nRunning Louvain on EdgeSketch...")
    t1 = @elapsed communities_1 = louvain_method_sketch(graph_sketch)
    estimated_modularity_1 = calculate_modularity_sketch(graph_sketch, communities_1)
    t2 = @elapsed communities_2 = louvain_method_2nd_phase_sketch(graph_sketch, communities_1)
    estimated_modularity_2 = calculate_modularity_sketch(graph_sketch, communities_2)
    
    best_modularity, best_communities = estimated_modularity_1 > estimated_modularity_2 ?
        (estimated_modularity_1, communities_1) : (estimated_modularity_2, communities_2)
    total_louvain_time = t1 + t2
    
    # Final summary
    emph("\n=== FINAL RESULTS ===")
    info("Graph: $nr_nodes nodes, $nr_edges_real edges, |E|/|V| = $edge_to_node_ratio")
    info("Sketch: size=$sketch_size, memory=$(round(sketch_mb, digits=2)) MB, creation=$(elapsed_sketch)s")
    info("Adj_list size: $(round(adj_list_mb, digits=2)) MB ($(round(adj_list_mb / sketch_mb, digits=1))x larger than sketch)")
    info("Louvain: $(round(total_louvain_time, digits=2)) sec, estimated modularity: $(round(best_modularity, digits=4)), clusters: $(length(best_communities))")
    info("Total time: $(round(elapsed_sketch + total_louvain_time, digits=1))s")
    
    # return graph_sketch, best_communities, best_modularity
end


# --- Graph reconstruction on Epinions item-item graph ---
# Uses EdgeSketch to predict edges and evaluates precision at different percentages.
function epinions_graph_reconstruction(; 
        sketch_size::Int = 100, 
        top_percent::Float64 = 100.0,
        k::Int = 2,
        alpha::Float64 = 1.0,
        min_weight::Real = 30,
        max_users_per_item::Real = Inf
    )
    Random.seed!(123)

    # --- Load Epinions item-item graph ---
    adj_list, _ = load_epinions_graph(min_weight=min_weight, max_users_per_item=max_users_per_item)
    name = "EPINIONS (item-item, large-scale)"
    nr_nodes = length(adj_list)
    nr_edges = Int(sum(length(neighbors) for neighbors in adj_list) / 2)
    adj_list_size_mb = Base.summarysize(adj_list) / 1e6

    println("\n=== GRAPH RECONSTRUCTION ===")
    println("Parameters: sketch_size=$sketch_size, k=$k, alpha=$alpha, top_percent=$top_percent%")

    # --- EdgeSketch generation ---
    println("\nCreating EdgeSketch...")
    time_sketch = @elapsed graph_sketch = create_graph_sketch(adj_list, sketch_size)
    sketch_memory_mb = Base.summarysize(graph_sketch) / 1e6
    
    # Count unique edges in samples
    sample_edges_set = Set{Tuple{Int,Int}}()
    for i in 1:nr_nodes
        for (a, b) in graph_sketch[i].sample
            if a != b
                push!(sample_edges_set, (min(a, b), max(a, b)))
            end
        end
    end
    sample_edges_count = length(sample_edges_set)
    sample_edges_percent = 100.0 * sample_edges_count / nr_edges
    
    println("EdgeSketch: $(round(time_sketch, digits=2)) sec, $(round(sketch_memory_mb, digits=2)) MB")
    println("Sample edges: $sample_edges_count ($(round(sample_edges_percent, digits=2))% of graph)")

    # --- Memory-efficient edge prediction ---
    println("\nPredicting edges...")
    time_predict = @elapsed predicted_edges = predict_top_edges_fast(
        graph_sketch, nr_edges, k, alpha;
        top_percent=top_percent, use_sample_edges=true, verbose=true
    )

    # --- Evaluation for different percentages of graph ---
    println("\n" * "="^50)
    println("PRECISION AT DIFFERENT PERCENTAGES")
    println("="^50)
    
    total_predictions = length(predicted_edges)
    
    # Table header
    println("| % graph | # edges    | precision |")
    println("|---------|------------|-----------|")
    
    # Evaluate at 1%, 2%, ..., up to 100% of graph (or until we run out of predictions)
    for pct in 1:100
        edges_at_pct = Int(ceil(pct * nr_edges / 100))
        edges_to_eval = min(edges_at_pct, total_predictions)
        
        if edges_to_eval > 0
            precision, _, _ = evaluate_prediction(predicted_edges[1:edges_to_eval], adj_list)
            
            pct_str = lpad("$(pct)%", 7)
            edges_str = lpad(edges_to_eval, 10)
            precision_str = lpad("$(round(precision, digits=4))", 9)
            
            println("| $pct_str | $edges_str | $precision_str |")
        end
        
        # Stop if we've used all predictions
        if edges_at_pct >= total_predictions
            println("\n(Reached max predictions: $total_predictions)")
            break
        end
    end
    println()

    # --- Print summary ---
    emph("\nSUMMARY")
    info(" - graph               $nr_nodes nodes, $nr_edges edges")
    info(" - adj_list_memory     $(f4(adj_list_size_mb)) MB")
    info(" - sketch_memory       $(f4(sketch_memory_mb)) MB")
    info(" - predicted_edges     $(length(predicted_edges))")
    info(" - sample_edges        $sample_edges_count ($(round(sample_edges_percent, digits=1))%)")
    info(" - time_sketch         $(f2(time_sketch)) sec")
    info(" - time_predict        $(f2(time_predict)) sec")
    info(" - total_time          $(f2(time_sketch + time_predict)) sec")
end


end
