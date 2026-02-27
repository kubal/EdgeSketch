__precompile__(false)

module _GraphReconstruction

using Base.Threads
using Base.Iterators: flatten
using _EdgeSketch
using _SketchOperations

# Similarity computation
export create_similarity_matrix

# Edge prediction
export predict_top_edges
export predict_top_edges_fast

# Evaluation
export evaluate_prediction


# =============================================================================
# Helper functions for k-hop neighborhood computation
# =============================================================================

function extract_neighbours(
        graph_sketch::Vector{EdgeSketch}
    )::Vector{Set{Int}}  
    
    n = length(graph_sketch)
    neighbours = Vector{Set{Int}}(undef, n) 
    
    @threads for i in 1:n
        sample = graph_sketch[i].sample    
        neighbours[i] = Set(flatten(sample))  
    end
    
    return neighbours
end


function neighbourhood_expansion(
        neighbours::Vector{Set{Int}}, 
        edges::Vector{Set{Int}}  
    )::Vector{Set{Int}}  
    
    n = length(neighbours)
    neighbourhood_expanded = Vector{Set{Int}}(undef, n)  

    @threads for i in 1:n
        new_neighbours = Set{Int}()
        for neighbour in neighbours[i]
            push!(new_neighbours, neighbour)
            for edge in edges[neighbour]
                push!(new_neighbours, edge)
            end
        end
        neighbourhood_expanded[i] = new_neighbours 
    end

    return neighbourhood_expanded 
end


function get_edges_from_samples(
        graph_sketch::Vector{EdgeSketch}
    )::Matrix{Float64}
    
    n = length(graph_sketch)
    edges_from_samples = fill(0.0, n, n)

    for node in graph_sketch
        for (i, j) in node.sample
            i, j = min(i,j), max(i,j)
            if i != j
                edges_from_samples[i, j] = Inf
            end
        end
    end

    return edges_from_samples
end


# =============================================================================
# Similarity matrix computation (O(n²) approach)
# =============================================================================

function create_similarity_matrix(
        graph_sketch::Vector{EdgeSketch}, 
        k::Int64,
        alpha::Float64,
        use_sample_edges::Bool = false
    )::Matrix{Float64}
    
    n = length(graph_sketch)        
    sim_matrix = zeros(n,n)
    edges = extract_neighbours(graph_sketch)
    k_sketches = deepcopy(graph_sketch)
    neighbourhoods = Vector{Set{Int}}()
    
    for j in 0:k    
        if j == 0
            neighbourhoods = [Set([i]) for i in 1:n]
        else
            neighbourhoods = neighbourhood_expansion(neighbourhoods, edges)
        end            

        @threads for i in 1:n
            k_sketches[i] = create_subgraph_sketch(neighbourhoods[i], graph_sketch)
        end
        
        sim_matrix = sim_matrix +  alpha^j .* nodes_similarity(k_sketches) 
        
    end

       
    if use_sample_edges
        sim_matrix = sim_matrix + get_edges_from_samples(graph_sketch)
    end
    
    return sim_matrix
end


# =============================================================================
# Edge prediction
# =============================================================================

function predict_top_edges(
        similarity_matrix::Matrix{Float64}, 
        top::Int   
    )::Vector{Tuple{Int, Int}}

    n = size(similarity_matrix, 1)
    
    # Collect only upper triangle (i < j) - each unique edge once
    num_pairs = div(n * (n - 1), 2)
    scores = Vector{Float64}(undef, num_pairs)
    indices = Vector{Tuple{Int, Int}}(undef, num_pairs)
    
    idx = 1
    for j in 2:n
        for i in 1:(j-1)
            scores[idx] = similarity_matrix[i, j]
            indices[idx] = (i, j)
            idx += 1
        end
    end
    
    # Get top indices sorted by similarity (descending)
    top_count = min(top, num_pairs)
    top_perm = partialsortperm(scores, 1:top_count, rev=true)
    
    return [indices[p] for p in top_perm]
end


# Optimized edge prediction for large graphs: instead of computing similarities for all O(N^2) pairs, 
# only compares pairs of nodes that are within `candidate_hops` of each other via sample edges.
function predict_top_edges_fast(
        graph_sketch::Vector{EdgeSketch},
        nr_edges::Int,
        k_neighbors::Int,
        alpha::Float64;
        top_percent::Float64 = 100.0,
        use_sample_edges::Bool = true,
        candidate_hops::Int = 0,
        verbose::Bool = false
    )

    n = length(graph_sketch)
    nt = Threads.nthreads()

    # Default candidate_hops = k_neighbors (reasonable tradeoff between coverage and speed)
    if candidate_hops == 0
        candidate_hops = k_neighbors
    end

    target_edges = Int(ceil(top_percent * nr_edges / 100))
    # Max candidates per node: 3x average degree, min 10 for sparse graphs
    top_k_per_node = max(10, Int(ceil(3 * target_edges / n)))

    if verbose
        println("Parameters: n=$n, nr_edges=$nr_edges, top_percent=$top_percent%")
        println("Target edges: $target_edges, top_k_per_node: $top_k_per_node")
        println("Candidate hops: $candidate_hops (k_neighbors=$k_neighbors)")
        println("Threads: $nt")
    end

    # Copy helper
    copy_sketch(es::EdgeSketch) = EdgeSketch(copy(es.embedding), copy(es.sample), es.size)

    # =========================================================================
    # Phase 1: Build sample neighbor lists (multithreaded -- each node independent)
    # =========================================================================
    if verbose
        println("Phase 1: Building neighbor lists from samples...")
    end

    sample_neighbors = Vector{Vector{Int}}(undef, n)
    @threads for i in 1:n
        v = Int[]
        for (a, b) in graph_sketch[i].sample
            neighbor = (a == i) ? b : a
            if neighbor != i && neighbor >= 1 && neighbor <= n
                push!(v, neighbor)
            end
        end
        sort!(v)
        unique!(v)
        sample_neighbors[i] = v
    end

    # =========================================================================
    # Phase 2: Build candidate pairs via BFS on sample edges (multithreaded)
    # For each node i, find all nodes within candidate_hops, keep only j > i (upper triangle)
    # Per-thread BFS state to avoid race conditions.
    # =========================================================================
    if verbose
        println("Phase 2: Building candidate pairs (BFS depth=$candidate_hops)...")
    end

    candidates = Vector{Vector{Int}}(undef, n)
    total_candidates_atomic = Threads.Atomic{Int}(0)

    # Per-thread BFS buffers
    visited_pt2 = [falses(n) for _ in 1:nt]
    visited_nodes_pt2 = [Int[] for _ in 1:nt]

    @threads for src in 1:n
        tid = Threads.threadid()
        vis = visited_pt2[tid]
        vis_nodes = visited_nodes_pt2[tid]

        empty!(vis_nodes)
        vis[src] = true
        push!(vis_nodes, src)

        frontier = Int[src]
        for depth in 1:candidate_hops
            next_frontier = Int[]
            for v in frontier
                for u in sample_neighbors[v]
                    if !vis[u]
                        vis[u] = true
                        push!(vis_nodes, u)
                        push!(next_frontier, u)
                    end
                end
            end
            frontier = next_frontier
        end

        # Keep only j > src (upper triangle symmetry)
        cands = Int[]
        for u in vis_nodes
            if u > src
                push!(cands, u)
            end
        end
        sort!(cands)
        candidates[src] = cands
        Threads.atomic_add!(total_candidates_atomic, length(cands))

        # Reset visited flags
        for u in vis_nodes
            vis[u] = false
        end
    end

    total_candidates = total_candidates_atomic[]

    if verbose
        avg_candidates = total_candidates / n
        all_pairs = n * (n - 1) ÷ 2
        reduction_pct = all_pairs > 0 ? round(100.0 * total_candidates / all_pairs, digits=2) : 0.0
        println("Total candidate pairs: $total_candidates (avg $(round(avg_candidates, digits=1)) per node)")
        println("Reduction: $reduction_pct% of all $all_pairs pairs")
    end

    # =========================================================================
    # Phase 3: Compute k-hop sketches for all nodes via incremental BFS + sketch_union!
    # Multithreaded - each node's BFS is independent.
    # Per-thread BFS state, writes to hop_sketches[depth][src] are conflict-free.
    # =========================================================================
    if verbose
        println("Phase 3: Computing k-hop sketches for all nodes (k=$k_neighbors)...")
    end

    hop_sketches = [Vector{EdgeSketch}(undef, n) for _ in 0:k_neighbors]

    # Per-thread BFS buffers
    visited_pt3 = [falses(n) for _ in 1:nt]
    vis_nodes_pt3 = [Int[] for _ in 1:nt]
    frontier_pt3 = [Int[] for _ in 1:nt]
    next_frontier_pt3 = [Int[] for _ in 1:nt]

    @threads for src in 1:n
        tid = Threads.threadid()
        vis = visited_pt3[tid]
        vis_nodes = vis_nodes_pt3[tid]
        bfs_frontier = frontier_pt3[tid]
        bfs_next_frontier = next_frontier_pt3[tid]

        empty!(vis_nodes)
        empty!(bfs_frontier)
        empty!(bfs_next_frontier)

        vis[src] = true
        push!(vis_nodes, src)
        push!(bfs_frontier, src)

        running = copy_sketch(graph_sketch[src])
        hop_sketches[1][src] = copy_sketch(running)

        for depth in 1:k_neighbors
            empty!(bfs_next_frontier)
            for v in bfs_frontier
                for u in sample_neighbors[v]
                    if !vis[u]
                        vis[u] = true
                        push!(vis_nodes, u)
                        push!(bfs_next_frontier, u)
                        sketch_union!(running, graph_sketch[u])
                    end
                end
            end
            hop_sketches[depth + 1][src] = copy_sketch(running)
            bfs_frontier, bfs_next_frontier = bfs_next_frontier, bfs_frontier
        end

        # Store back swapped references for next reuse
        frontier_pt3[tid] = bfs_frontier
        next_frontier_pt3[tid] = bfs_next_frontier

        for u in vis_nodes
            vis[u] = false
        end
    end

    # =========================================================================
    # Phase 4: Score only candidate pairs (multithreaded)
    # Each node i produces unique edges (i, j) with j > i, no write conflicts.
    # Each thread collects edges separately, then we merge them at the end.
    # =========================================================================
    if verbose
        println("Phase 4: Scoring $total_candidates candidate pairs...")
    end

    edges_per_thread = [Vector{Tuple{Int,Int,Float64}}() for _ in 1:nt]

    @threads for i in 1:n
        tid = Threads.threadid()
        local_edges = edges_per_thread[tid]

        cands = candidates[i]
        if isempty(cands)
            continue
        end

        similarities = Vector{Tuple{Int, Float64}}()
        sizehint!(similarities, length(cands))

        for j in cands
            sim = 0.0
            for depth in 0:k_neighbors
                w = alpha^depth
                sim += w * sketch_jacc(hop_sketches[depth + 1][i], hop_sketches[depth + 1][j])
            end
            if sim > 0.0
                push!(similarities, (j, sim))
            end
        end

        # Keep top-k for this node
        if length(similarities) > top_k_per_node
            partialsort!(similarities, 1:top_k_per_node, by = x -> -x[2])
            resize!(similarities, top_k_per_node)
        end

        # Collect into thread-local vector
        for (j, sim) in similarities
            push!(local_edges, (i, j, sim))
        end
    end

    # Sequential merge of thread-local results into Dict
    all_edges = Dict{Tuple{Int,Int}, Float64}()
    for local_edges in edges_per_thread
        for (i, j, sim) in local_edges
            edge = (i, j)
            if !haskey(all_edges, edge) || all_edges[edge] < sim
                all_edges[edge] = sim
            end
        end
    end

    # =========================================================================
    # Phase 5: Add edges from samples (known edges get Inf score)
    # =========================================================================
    if use_sample_edges
        if verbose
            println("Phase 5: Adding edges from samples...")
        end
        for i in 1:n
            for (a, b) in graph_sketch[i].sample
                if a != b
                    edge = (min(a, b), max(a, b))
                    all_edges[edge] = Inf
                end
            end
        end
    end

    # =========================================================================
    # Phase 6: Sort and return top predicted edges
    # =========================================================================
    if verbose
        println("Phase 6: Sorting $(length(all_edges)) candidate edges...")
    end

    edges_list = [(e[1], e[2], s) for (e, s) in all_edges]
    sort!(edges_list, by = x -> -x[3])

    result_size = min(target_edges, length(edges_list))
    predicted_edges = [(e[1], e[2]) for e in edges_list[1:result_size]]

    if verbose
        println("Done. Returning $result_size predicted edges.")
    end

    return predicted_edges
end


# =============================================================================
# Evaluation
# =============================================================================

function evaluate_prediction(
        predicted_edges::Vector{Tuple{Int, Int}},
        adj_list::Vector{Vector{Tuple{Int, Float64}}}
    )::Tuple{Float64,Float64,Float64}

    true_edges_weights = sum((sum(weight for (_, weight) in neighbors; init=0.0) for neighbors in adj_list); init=0.0) / 2
    true_edges_count = sum(length(neighbors) for neighbors in adj_list; init=0) / 2

    predicted_true_edges_counter = 0
    predicted_true_edges_weights = 0
    
    for edge in predicted_edges
        i, j = edge
        for (neighbor, weight) in adj_list[i]
            if neighbor == j
                predicted_true_edges_counter += 1
                predicted_true_edges_weights += weight
                break
            end
        end
    end
    
    precision = round(predicted_true_edges_counter / length(predicted_edges), digits=5)
    recall = round(predicted_true_edges_counter / true_edges_count, digits=5)
    recall_weighted = round(predicted_true_edges_weights / true_edges_weights, digits=5)
    
    return precision, recall, recall_weighted
end


end
