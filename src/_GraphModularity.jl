__precompile__(false)

module _GraphModularity

using Random
using Base.Threads

export calculate_modularity
export louvain_method
export louvain_method_2nd_phase
export blocks_to_communities
export super_nodes_to_nodes


function louvain_method(
        adj_list::Vector{Vector{Tuple{Int, Float64}}}
    )

    n = length(adj_list)
    communities = [Set([i]) for i in 1:n]
    edges_weight = isempty(adj_list) ? 0.0 : sum(sum(weight for (_, weight) in neighbors; init=0.0) for neighbors in adj_list) / 2.0

    if edges_weight == 0
        return communities
    end

    degrees = [isempty(neighbors) ? 0.0 : sum(weight for (_, weight) in neighbors) for neighbors in adj_list]
    new_modularity = calculate_modularity(adj_list, communities, degrees)    
    node_community = collect(1:n)
    neighbor_communities = Set{Int}()
    community_gains = Vector{Tuple{Float64, Int}}(undef, n)
    improvement = true

    while improvement
        
        current_modularity = new_modularity
        for v in 1:n
            best_gain = -Inf 
            best_community_index = nothing

            current_community_index = node_community[v]
            current_community = communities[current_community_index]
            delete!(current_community, v)

            empty!(neighbor_communities) 
            push!(neighbor_communities, current_community_index)
            for (neighbor, _) in adj_list[v]
                neighbor_community_index = node_community[neighbor]
                push!(neighbor_communities, neighbor_community_index)
            end
            
            neighbor_communities_vec = collect(neighbor_communities)
                    
            @threads for i in 1:length(neighbor_communities_vec)
                community_index = neighbor_communities_vec[i]
                community = communities[community_index]
                community_gain = modularity_gain_fast(adj_list, v, community, edges_weight, degrees)
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
        
        new_modularity = calculate_modularity(adj_list, communities, degrees)
        if new_modularity - current_modularity < 1e-5
            improvement = false
        end
    end

    communities = filter(!isempty, communities)
    return communities
end


function calculate_modularity(
        adj_list::Vector{Vector{Tuple{Int, Float64}}}, 
        communities::Vector{Set{Int}},
        degrees::Union{Vector{Float64}, Nothing} = nothing
    )::Float64

    if degrees === nothing
        degrees = [isempty(neighbors) ? 0.0 : sum(weight for (_, weight) in neighbors) for neighbors in adj_list]
    end

    nr_edges = sum(degrees) / 2
    modularity = Threads.Atomic{Float64}(0.0) 
   
    @threads for community in communities
        sigma_in = 0.0
        sigma_tot = 0.0
        
        for i in community
            sigma_tot += degrees[i]
            for (j, weight) in adj_list[i]
                if j in community
                    sigma_in += weight
                end
            end
        end

        Threads.atomic_add!(modularity, (sigma_in / (2 * nr_edges)) - ((sigma_tot / (2 * nr_edges))^2))        
    end
    
    return modularity[]
end


function modularity_gain(
        adj_list::Vector{Vector{Tuple{Int, Float64}}}, 
        v::Int,
        community::Set{Int}
    )::Float64


    old_modularity = calculate_modularity(adj_list, [community])
    new_community = union(community, Set([v]))
    new_modularity = calculate_modularity(adj_list, [new_community])

    return new_modularity - old_modularity
end


function modularity_gain_fast(
        adj_list::Vector{Vector{Tuple{Int, Float64}}}, 
        v::Int, 
        community::Set{Int},
        edges_weight::Float64, 
        degrees::Vector{Float64}
    )::Float64

    k_v = degrees[v]  
    
    sum_in_v = 0.0
    for (neighbor, weight) in adj_list[v]
        if neighbor in community
            sum_in_v += weight
        end
    end
    
    sum_tot = sum(degrees[node] for node in community; init=0.0)

    # return (4*edges_weight*sum_in_v - k_v*(2*sum_tot+k_v))/(4*edges_weight^2)
    return 4 * edges_weight * sum_in_v - k_v * (2 * sum_tot + k_v)
end


function aggregate_communities(
        adj_list::Vector{Vector{Tuple{Int, Float64}}}, 
        communities::Vector{Set{Int}}
    )
    
    n = length(communities)
    new_adj_list = [Dict{Int, Float64}() for _ in 1:n]

    community_map = Dict{Int, Int}()
    for (new_idx, community) in enumerate(communities)
        for node in community
            community_map[node] = new_idx
        end
    end

    for i in 1:length(adj_list)
        new_i = community_map[i]
        for (neighbor, weight) in adj_list[i]
            new_j = community_map[neighbor]
            # if new_i != new_j  comment this to take into account self-loops
                new_adj_list[new_i][new_j] = get(new_adj_list[new_i], new_j, 0.0) + weight
            # end
        end
    end

    final_adj_list = [Vector{Tuple{Int, Float64}}(collect(Tuple(kv) for kv in d)) for d in new_adj_list]
    
    return final_adj_list, community_map
end




function super_nodes_to_nodes(
        community_map::Dict{Int, Int},
        super_communities::Vector{Set{Int}}  
    )
    
    communities = Dict{Int, Set{Int}}()  

    reverse_map = Dict{Int, Set{Int}}()  
    for (original_vertex, super_vertex) in community_map
        if !haskey(reverse_map, super_vertex)
            reverse_map[super_vertex] = Set{Int}() 
        end
        push!(reverse_map[super_vertex], original_vertex)
    end

    for (super_idx, community) in enumerate(super_communities)
        new_community = Set{Int}()  
        for super_vertex in community
            union!(new_community, reverse_map[super_vertex])  
        end
        communities[super_idx] = new_community
    end

    return collect(values(communities))  
end


function louvain_method_2nd_phase(
        adj_list::Vector{Vector{Tuple{Int, Float64}}}, 
        communities_1st_phase::Vector{Set{Int}}
    )
    
    new_adj_list, community_map = aggregate_communities(adj_list, communities_1st_phase)
    
    super_communities = louvain_method(new_adj_list)
    
    communities_2nd_phase = super_nodes_to_nodes(community_map, super_communities)
    
    return communities_2nd_phase    
end


function blocks_to_communities(
        blocks::Vector{Int}
    )::Vector{Set{Int}}
    
    community_dict = Dict{Int, Set{Int}}()
    for (node, community) in enumerate(blocks)
        if !haskey(community_dict, community)
            community_dict[community] = Set{Int}() 
        end
        push!(community_dict[community], node) 
    end

    return collect(values(community_dict))  
end


end


