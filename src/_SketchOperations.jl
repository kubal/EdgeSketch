 __precompile__(false)

module _SketchOperations

using Base.Threads
using _EdgeSketch

export weight, nodes_weights, sketch_size
export emb_union, emb_jacc, emb_section
export sketch_union, sketch_union!, sketch_jacc, sketch_section
export create_subgraph_sketch, nodes_similarity, graph_sketch_union!


function weight(
        embedding::Vector{Float64}
    )::Float64
    
    m = length(embedding)  
    total = sum(embedding)  
    return (m - 1) / total
end

function weight(
        sketch::EdgeSketch
    )::Float64
      
    return weight(sketch.embedding)
end


function nodes_weights(
        graph_sketch::Vector{EdgeSketch}
    )::Vector{Float64}
    
    return [weight(sketch.embedding) for sketch in graph_sketch]
end


function sketch_size(
        sketch::EdgeSketch
    )::Int64
      
    return length(sketch.embedding)
end


function sketch_size(
        graph_sketch::Vector{EdgeSketch}
    )::Tuple{Int64,Int64}
      
    return sketch_size(graph_sketch[1]), length(graph_sketch)
end


function emb_union(
        emb1::Vector{Float64},
        emb2::Vector{Float64}
    )::Vector{Float64}
    
    return map(min, emb1, emb2)
end


function emb_jacc(
        emb1::Vector{Float64},
        emb2::Vector{Float64}
    )::Float64
    
    return sum(emb1 .== emb2) / length(emb1)
end


function emb_section(
        emb1::Vector{Float64},
        emb2::Vector{Float64}
    )::Float64
    
    return weight(emb_union(emb1, emb2)) * emb_jacc(emb1, emb2)
end


function sketch_union(
        sketch1::EdgeSketch, 
        sketch2::EdgeSketch
    )::EdgeSketch

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


function sketch_union!(
        sketch1::EdgeSketch, 
        sketch2::EdgeSketch
    )
    
    for i in 1:sketch1.size
        if sketch1.embedding[i] > sketch2.embedding[i]
            sketch1.embedding[i] = sketch2.embedding[i]
            sketch1.sample[i] = sketch2.sample[i]
        end
    end
end


function sketch_jacc(
        sketch1::EdgeSketch,
        sketch2::EdgeSketch
    )::Float64
    
    return emb_jacc(sketch1.embedding, sketch2.embedding)
end


function sketch_section(
        sketch1::EdgeSketch,
        sketch2::EdgeSketch
    )::Float64
    
    return emb_section(sketch1.embedding, sketch2.embedding)
end


function create_subgraph_sketch(
        subgraph::Set{Int},
        graph_sketch::Vector{EdgeSketch}
    )::EdgeSketch
    
    m = sketch_size(graph_sketch[1])
    subgraph_sketch = EdgeSketch(m)
    
    for i in subgraph
        sketch_union!(subgraph_sketch, graph_sketch[i])
    end
    
    return subgraph_sketch
end


function graph_sketch_union!(
        graph_sketch1::Vector{EdgeSketch},
        graph_sketch2::Vector{EdgeSketch}
    )

    for i in 1:length(graph_sketch1) 
        sketch_union!(graph_sketch1[i],graph_sketch2[i]) 
    end

end

        

function nodes_similarity( 
        graph_sketch::Vector{EdgeSketch}
    )::Matrix{Float64}

    n = length(graph_sketch)
    sim_matrix = zeros(n,n)

    # for directed graphs go over all the postistions
    @threads for i in 1:n
        for j in i+1:n
            sim_matrix[i,j] = sketch_jacc(graph_sketch[i],graph_sketch[j])
        end
    end

    return sim_matrix 
end



end