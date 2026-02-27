 __precompile__(false)

module _EdgeSketch

using _FastExpSketch
using Base.Threads

export create_graph_sketch
export EdgeSketch
export is_initialized


mutable struct EdgeSketch    
    embedding::Vector{Float64}
    sample::Vector{Tuple{Int, Int}}
    size::Int

    # constructor with default values
    function EdgeSketch(
            size::Int, 
            embedding::Vector{Float64} = fill(Inf, size),
            sample::Vector{Tuple{Int, Int}} = Vector{Tuple{Int, Int}}(undef, size)
        )
        new(embedding, sample, size)
    end

    # constructor without default values
    function EdgeSketch(
            embedding::Vector{Float64}, 
            sample::Vector{Tuple{Int, Int}}, 
            size::Int
        )
        new(embedding, sample, size)
    end
    
end


function create_graph_sketch(
        adj_list::Vector{Vector{Tuple{Int, Float64}}}, 
        sketch_size::Int,
        seed::Int=123
    )::Vector{EdgeSketch}

    if all(isempty, adj_list)
        error("The input graph has no edges.")
    end

    nodes_number = length(adj_list)
    graph_sketch = Vector{EdgeSketch}(undef, nodes_number)
    h = x -> (hash((x, seed)) % Int64(1e9)) / 1e9

    @threads for i in 1:nodes_number
        stream_elements = [StreamElement(i, j, weight) 
            for (j, weight) in adj_list[i]]
        
        embedding, sample = fast_expsketch(
            stream_elements, 
            sketch_size, 
            h)

        graph_sketch[i] = EdgeSketch(embedding, sample, sketch_size)        
    end

    return graph_sketch
end


function is_initialized(es::EdgeSketch)
    return es.embedding[1] != Inf
end

end

