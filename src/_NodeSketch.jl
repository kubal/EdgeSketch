__precompile__(false)

module _NodeSketch

using Random
using Distributions
using _GraphModels

# Sketch construction
export nodesketch

# Similarity computation
export node_similarity_matrix


# =============================================================================
# Helper functions (internal)
# =============================================================================

# Add self-loop (i, 1.0) to each node if not already present
function ensure_self_loops!(adj_list::Vector{Vector{Tuple{Int,Float64}}})
    for i in 1:length(adj_list)
        if !any(t -> t[1] == i, adj_list[i])
            push!(adj_list[i], (i, 1.0))
        end
    end
    return adj_list
end


# Get edge weight between nodes i and l (0.0 if no edge)
function get_weight(adj_list::Vector{Vector{Tuple{Int,Float64}}}, i::Int, l::Int)::Float64
    for (node, w) in adj_list[i]
        if node == l
            return w
        end
    end
    return 0.0
end


# Min-hash sample: node with minimum -log(hash)/weight
function generate_sample(row, hash_functions, j, neighbours)
    return argmin(i -> -log(hash_functions[j](i)) / row[i], neighbours)
end


# Recursive k-hop embedding computation using weighted min-hash
function embeddings(
        adj_list::Vector{Vector{Tuple{Int,Float64}}}, 
        order::Integer, 
        sketch_dimensions::Integer, 
        alpha::Number,
        hash_functions
    )::Matrix{Int}

    col_count = length(adj_list)
    emb = zeros(Int, sketch_dimensions, col_count)

    if order > 2
        emb = embeddings(adj_list, order - 1, sketch_dimensions, alpha, hash_functions)

        for i in 1:col_count
            neighbours = [n for (n, w) in adj_list[i] if w != 0]
            element_dict = Dict{Int,Int}(l => 0 for l in 1:col_count)

            for n in neighbours
                for j in 1:sketch_dimensions
                    val = emb[j, n]
                    if !haskey(element_dict, val)
                        element_dict[val] = 0
                    end
                    element_dict[val] += 1
                end
            end

            v = Vector{Float64}(undef, col_count)
            for l in 1:col_count
                s = element_dict[l]
                s *= alpha / sketch_dimensions
                w_li = get_weight(adj_list, i, l)
                v[l] = s + w_li
            end

            v_neighbours = [l for l in 1:col_count if v[l] != 0]
            emb[:, i] = [generate_sample(v, hash_functions, j, v_neighbours) for j in 1:sketch_dimensions]
        end

    elseif order == 2
        for i in 1:col_count
            row = fill(0.0, col_count)
            for (n, w) in adj_list[i]
                row[n] = w
            end
            neighbours = [n for (n, w) in adj_list[i] if w != 0]

            emb[:, i] = [generate_sample(row, hash_functions, j, neighbours) for j in 1:sketch_dimensions]
        end
    end

    return emb
end


# =============================================================================
# Sketch construction
# =============================================================================

# Create NodeSketch embeddings: k-hop neighborhood sketches via weighted min-hash
function nodesketch(
        adj_list::Vector{Vector{Tuple{Int, Float64}}}, 
        order::Integer, 
        sketch_dimensions::Integer, 
        alpha::Number
    )::Matrix{Int}

    adj_list = ensure_self_loops!(adj_list)
    hash_functions = [x -> (hash((x, seed)) % Int64(1e9)) / 1e9 for seed in rand(Int64, sketch_dimensions)]
    emb = embeddings(adj_list, order, sketch_dimensions, alpha, hash_functions)
    
    return emb
end


# =============================================================================
# Similarity computation
# =============================================================================

# Compute pairwise Jaccard similarity from embeddings (fraction of matching min-hash values)
function node_similarity_matrix(embeddings::Matrix{Int})::Matrix{Float64}
    sketch_dimensions = size(embeddings, 1) 
    col_count = size(embeddings, 2)   
    similarity_matrix = zeros(col_count, col_count)

    for i in 1:col_count
        for j in i:col_count
            matching_count = 0
            for k in 1:sketch_dimensions
                if embeddings[k, i] == embeddings[k, j]
                    matching_count += 1
                end
            end
            count = matching_count / sketch_dimensions

            similarity_matrix[i, j] = count
            if i != j
                similarity_matrix[j, i] = count
            end
        end        
    end
    
    return similarity_matrix
end


end
