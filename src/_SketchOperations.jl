__precompile__(false)

module _SketchOperations

using Base.Threads
using _EdgeSketch

# Weight and size
export weight, nodes_weights, sketch_size

# Sketch operations
export sketch_union, sketch_union!, sketch_jacc, sketch_section

# Graph sketch operations
export create_subgraph_sketch, graph_sketch_union!, nodes_similarity


# =============================================================================
# Weight and size - basic properties of sketches
# =============================================================================

# Estimate total edge weight from embedding 
function weight(embedding::Vector{Float64})::Float64
    m = length(embedding)  
    total = sum(embedding)  
    return (m - 1) / total
end

# Edge weight for a single sketch
function weight(sketch::EdgeSketch)::Float64
    return weight(sketch.embedding)
end

# Edge weights for all nodes in graph
function nodes_weights(graph_sketch::Vector{EdgeSketch})::Vector{Float64}
    return [weight(sketch.embedding) for sketch in graph_sketch]
end

# Size of a single sketch (embedding length)
function sketch_size(sketch::EdgeSketch)::Int64
    return length(sketch.embedding)
end

# Returns (sketch_size, num_nodes) for a graph sketch
function sketch_size(graph_sketch::Vector{EdgeSketch})::Tuple{Int64, Int64}
    return sketch_size(graph_sketch[1]), length(graph_sketch)
end


# =============================================================================
# Embedding operations - low-level vector operations
# =============================================================================

# Union of two embeddings (element-wise minimum)
function emb_union(emb1::Vector{Float64}, emb2::Vector{Float64})::Vector{Float64}
    return map(min, emb1, emb2)
end

# Jaccard similarity: fraction of identical elements
function emb_jacc(emb1::Vector{Float64}, emb2::Vector{Float64})::Float64
    return sum(emb1 .== emb2) / length(emb1)
end

# Intersection weight estimate
function emb_section(emb1::Vector{Float64}, emb2::Vector{Float64})::Float64
    return weight(emb_union(emb1, emb2)) * emb_jacc(emb1, emb2)
end


# =============================================================================
# Sketch operations - operations on EdgeSketch objects
# =============================================================================

# Union of two sketches (returns new sketch)
function sketch_union(sketch1::EdgeSketch, sketch2::EdgeSketch)::EdgeSketch
    embedding = Vector{Float64}(undef, sketch1.size)
    sample = Vector{Tuple{Int, Int}}(undef, sketch1.size)
    
    for i in 1:sketch1.size
        if sketch1.embedding[i] < sketch2.embedding[i]
            embedding[i] = sketch1.embedding[i]
            sample[i] = sketch1.sample[i]
        else
            embedding[i] = sketch2.embedding[i]
            sample[i] = sketch2.sample[i]
        end
    end
    
    return EdgeSketch(embedding, sample, sketch1.size)
end

# Union of two sketches (modifies sketch1 in-place)
function sketch_union!(sketch1::EdgeSketch, sketch2::EdgeSketch)
    for i in 1:sketch1.size
        if sketch1.embedding[i] > sketch2.embedding[i]
            sketch1.embedding[i] = sketch2.embedding[i]
            sketch1.sample[i] = sketch2.sample[i]
        end
    end
end

# Jaccard similarity between two sketches
function sketch_jacc(sketch1::EdgeSketch, sketch2::EdgeSketch)::Float64
    return emb_jacc(sketch1.embedding, sketch2.embedding)
end

# Intersection weight of two sketches
function sketch_section(sketch1::EdgeSketch, sketch2::EdgeSketch)::Float64
    return emb_section(sketch1.embedding, sketch2.embedding)
end


# =============================================================================
# Graph sketch operations - operations on whole graph sketches
# =============================================================================

# Create sketch for a subgraph (union of node sketches)
function create_subgraph_sketch(subgraph::Set{Int}, graph_sketch::Vector{EdgeSketch})::EdgeSketch
    m = sketch_size(graph_sketch[1])
    subgraph_sketch = EdgeSketch(m)
    
    for i in subgraph
        sketch_union!(subgraph_sketch, graph_sketch[i])
    end
    
    return subgraph_sketch
end

# Union of two graph sketches (modifies graph_sketch1 in-place)
function graph_sketch_union!(graph_sketch1::Vector{EdgeSketch}, graph_sketch2::Vector{EdgeSketch})
    for i in 1:length(graph_sketch1) 
        sketch_union!(graph_sketch1[i], graph_sketch2[i]) 
    end
end

# Pairwise Jaccard similarity matrix for all nodes (upper triangle only)
function nodes_similarity(graph_sketch::Vector{EdgeSketch})::Matrix{Float64}
    n = length(graph_sketch)
    sim_matrix = zeros(n, n)

    @threads for i in 1:n
        for j in i+1:n
            sim_matrix[i, j] = sketch_jacc(graph_sketch[i], graph_sketch[j])
        end
    end

    return sim_matrix 
end


end
