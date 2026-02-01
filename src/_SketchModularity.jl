__precompile__(false)

module _SketchModularity

using Random
using Base.Threads
using _EdgeSketch
using _SketchOperations
using _GraphModularity

export calculate_modularity_sketch
export louvain_method_sketch
export louvain_method_2nd_phase_sketch
export community_edges_weights


function community_edges_weights(
        graph_sketch::Vector{EdgeSketch},
        community::Set{Int}
    )        

    community_sketch = create_subgraph_sketch(community,graph_sketch)
    total_edges_weight = weight(community_sketch)    

    inner = 0
    m = length(community_sketch.sample)    
    
    for (i, j) in community_sketch.sample
        if i in  community && j in  community
            inner += 1
        end
    end
      
    inner_edges_weight = total_edges_weight * (inner/m)

    return total_edges_weight, inner_edges_weight
end


function calculate_modularity_sketch( 
        graph_sketch::Vector{EdgeSketch}, 
        communities::Vector{Set{Int}},
        degrees::Union{Vector{Float64}, Nothing} = nothing
    )::Float64

    if degrees === nothing
        degrees = nodes_weights(graph_sketch)
    end
    
    nr_edges = sum(degrees) / 2
    modularity = Threads.Atomic{Float64}(0.0) 
    
    @threads for community in communities  
        total_edges_weight, inner_edges_weight = community_edges_weights(graph_sketch, community)
        sigma_in = 2 * inner_edges_weight 
        sigma_tot = total_edges_weight + inner_edges_weight    
        
        Threads.atomic_add!(modularity, (sigma_in / (2 * nr_edges)) - ((sigma_tot / (2 * nr_edges))^2))
    end
    
    return modularity[]
end


function louvain_method_sketch(
        graph_sketch::Vector{EdgeSketch}
    )
    
    m, n = sketch_size(graph_sketch)
    degrees = nodes_weights(graph_sketch)
    edges_weight = sum(degrees)/2 
    nodes = findall(is_initialized, graph_sketch)
    communities = [i in nodes ? Set([i]) : Set{Int}() for i in 1:n]
    rounds = 0

    if edges_weight == 0
        return communities
    end
    
    new_modularity = calculate_modularity_sketch(graph_sketch, communities,degrees)
         
    improvement = true
    node_community = collect(1:n)
    neighbor_communities = Set{Int}()
    sample_neighbors = Vector{Vector{Int}}(undef, n)
    community_gains = Vector{Tuple{Float64, Int}}(undef, n)

    for v in nodes
     sample_neighbors[v] = get_sample_neighbors(v,graph_sketch)               
    end

  
    while improvement
       
        current_modularity = new_modularity
        for v in nodes
                      
            best_gain = -Inf 
            best_community = nothing
            best_community_index = nothing

            current_community_index = node_community[v]
            current_community = communities[current_community_index]
            delete!(current_community, v)

            empty!(neighbor_communities) 
            
            for neighbor in sample_neighbors[v]               
                neighbor_community_index = node_community[neighbor]
                push!(neighbor_communities, neighbor_community_index)
            end
  
            neighbor_communities_vec = collect(neighbor_communities)
                              
            @threads for i in 1:length(neighbor_communities_vec)
                community_index = neighbor_communities_vec[i]
                community = communities[community_index]
                community_gain = modularity_gain_fast_sketch(graph_sketch, m, v, community, edges_weight, degrees)
                community_gains[i] = (community_gain, community_index)
            end
            
            best_gain, best_community_index = community_gains[1]
            for i in 2:length(neighbor_communities_vec)
                if community_gains[i][1] > best_gain
                    best_gain, best_community_index = community_gains[i]
                end
            end 
                                 
            best_community = communities[best_community_index]
            push!(best_community, v)
            node_community[v] = best_community_index
        end
        
        new_modularity = calculate_modularity_sketch(graph_sketch, communities,degrees)
        if new_modularity - current_modularity < 1e-5
            improvement = false
        end
    end

    communities = filter(!isempty, communities)   
    return communities
end

function get_sample_neighbors(
        v::Int, 
        graph_sketch::Vector{EdgeSketch}
    )
    sample_edges = graph_sketch[v].sample
    sample_neighbors = Set{Int}()
    
    for (x, y) in sample_edges
        neighbor = (x == v) ? y : x
        push!(sample_neighbors, neighbor)
    end
    
    push!(sample_neighbors, v)
    return collect(sample_neighbors)
end

function modularity_gain_fast_sketch(  # < ---  this function is crucial !!!
        graph_sketch::Vector{EdgeSketch},
        sketch_size::Int,
        v::Int, 
        community::Set{Int},
        edges_weight::Float64, 
        degrees::Vector{Float64}
    )
    
    k_v = degrees[v]
    
    sum_tot = 0.0
    for node in community
        sum_tot += degrees[node]
    end

    inner = sum(i in community || j in community for (i, j) in graph_sketch[v].sample)
    sum_in_v = k_v * (inner/sketch_size)
        
    return 4 * edges_weight * sum_in_v - k_v * (2 * sum_tot + k_v)
end


function aggregate_communities_sketch(
        graph_sketch::Vector{EdgeSketch},
        communities::Vector{Set{Int}}
    )
    
    n = length(communities)
    new_graph_sketch = Vector{EdgeSketch}(undef, n)

    community_map = Dict{Int, Int}()    
    for (new_idx, community) in enumerate(communities)
        for node in community
            community_map[node] = new_idx
        end
    end

    for i in 1:n
      new_graph_sketch[i] = create_subgraph_sketch( communities[i] , graph_sketch )
      new_graph_sketch[i].sample =   [(community_map[u], community_map[v]) for (u, v) in new_graph_sketch[i].sample ]     end

    return new_graph_sketch, community_map
end


function louvain_method_2nd_phase_sketch(
        graph_sketch::Vector{EdgeSketch},
        communities_1st_phase::Vector{Set{Int}}
    )

        new_graph_sketch, community_map = aggregate_communities_sketch(graph_sketch, communities_1st_phase)
        super_communities = louvain_method_sketch(new_graph_sketch)    
        communities_2nd_phase = super_nodes_to_nodes(community_map, super_communities)

        return communities_2nd_phase    
end


end




