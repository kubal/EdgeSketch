 __precompile__(false)

module _kEmbeddings

using Base.Threads
using Base.Iterators: flatten
using _EdgeSketch
using _SketchOperations

export extract_neighbours
export create_similarity_matrix
export create_similarity_matrix_efficient


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
    neighbourhoods = []
    
    for j in 0:k    

        if j==0
            neighbourhoods = [Set([i]) for i in 1:n]
        else
            neighbourhoods = neighbourhood_expansion(neighbourhoods, edges)
        end            

        @threads for i in 1:n
            k_sketches[i] = create_subgraph_sketch(neighbourhoods[i], graph_sketch)
        end
        
        sim_matrix = sim_matrix +  alpha^j .* nodes_similarity(k_sketches) 
        
    end

       
    if use_sample_edges # this may increase precision
        sim_matrix = sim_matrix + get_edges_from_samples(graph_sketch)
    end
    
    return sim_matrix
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

end