module _GraphModels

using Random
using Distributions
using Base.Threads

export generate_stochastic_block_graph
export generate_erdos_renyi_graph
export temporal_graph_step
export merge_adjacency_lists!
export nodes_real_degrees
export permute_adj_list!
export largest_component_size


function generate_stochastic_block_graph(
        n::Int, 
        nr_blocks::Int, 
        p::Float64, 
        q::Float64, 
        use_weights::Bool = false, 
        remove_isolated::Bool = true,
        return_blocks::Bool = false
    )
   
    adj_list = [Vector{Tuple{Int, Float64}}() for _ in 1:n]
    blocks = rand(1:nr_blocks, n)
    rand_var = Exponential(1)  #<-- use weights

    for y in 1:n
         for x in (y + 1):n

            if use_weights
                weight = rand(rand_var)  #<-- use weights   
            else
                weight = 1.0  
            end
            
            if blocks[x] == blocks[y]
                if rand() < p
                    push!(adj_list[x], (y, weight))
                    push!(adj_list[y], (x, weight))
                end
            elseif rand() < q
                push!(adj_list[x], (y, weight))
                push!(adj_list[y], (x, weight))
            end
        end
    end

    if remove_isolated 
        adj_list, blocks = remove_isolated_nodes(adj_list, blocks)
    end
        
    if return_blocks
        return adj_list, blocks
    else
        return adj_list
    end 
end


function generate_erdos_renyi_graph(
        n::Int, 
        p::Float64,
        weights::Bool = false,
        remove_isolated::Bool = true
    )
    
    adj_list = [Vector{Tuple{Int, Float64}}() for _ in 1:n]
    
    rand_var = Exponential(1)  #<-- use weights

    for y in 1:n
        for x in (y + 1):n
            if rand() < p
                if weights
                    weight = rand(rand_var)   #<-- use weights
                else
                    weight = 1.0  
                end         
                push!(adj_list[x], (y, weight))
                push!(adj_list[y], (x, weight))
            end
        end
    end

    if remove_isolated 
        adj_list = remove_isolated_nodes(adj_list)
    end

    return adj_list
end


function remove_isolated_nodes(
        adj_list::Vector{Vector{Tuple{Int, Float64}}}, 
        blocks::Union{Nothing, Vector{Int}} = nothing
    )

    non_isolated_indices = findall(x -> !isempty(adj_list[x]), 1:length(adj_list))
    new_adj_list = [adj_list[i] for i in non_isolated_indices]
    
    index_mapping = Dict(old_idx => new_idx for (new_idx, old_idx) in enumerate(non_isolated_indices))
    @threads for i in 1:length(new_adj_list)
        new_adj_list[i] = [(index_mapping[j], w) for (j, w) in new_adj_list[i] if haskey(index_mapping, j)]
    end
    
    if blocks !== nothing
        new_blocks = blocks[non_isolated_indices]
        return new_adj_list, new_blocks
    else
        return new_adj_list
    end
end


function temporal_graph_step(
    left_edges::Vector{Vector{Tuple{Int, Float64}}}, 
    step_size::Int
)
  
    total_edges = sum(length(neighbors) for neighbors in left_edges)/2   
    step_size = min(step_size, total_edges)

    step_edges = [Vector{Tuple{Int, Float64}}() for _ in 1:length(left_edges)]

    for _ in 1:step_size
        u = rand(findall(x -> !isempty(x), left_edges))
        edge = popfirst!(left_edges[u])
        push!(step_edges[u], edge)
        v = edge[1] 
        push!(step_edges[v], (u,edge[2])) 
        idx = findfirst(e -> e[1] == u, left_edges[v])
        if idx !== nothing
            deleteat!(left_edges[v], idx)
        end
    end

    return left_edges, step_edges 
end


function merge_adjacency_lists!(
        adj_list1::Vector{Vector{Tuple{Int64, Float64}}},
        adj_list2::Vector{Vector{Tuple{Int64, Float64}}},
        seen_edges::Dict{Int64, Set{Int64}}
    )
    
    n = length(adj_list1)
    
    for i in 1:n
        for (v, w) in adj_list2[i]
            if !in(v, seen_edges[i])
                push!(adj_list1[i], (v, w))
                push!(seen_edges[i], v) 
            end
        end
    end
    
end


function nodes_real_degrees(
        adj_list::Vector{Vector{Tuple{Int64, Float64}}}
    )
    
    real_degrees = [isempty(neighbors) ? 0.0 : sum(weight for (_, weight) in neighbors) for neighbors in adj_list]

    return real_degrees
end


function permute_adj_list!(
        adj_list::Vector{Vector{Tuple{Int64, Float64}}}
    )
    
    for i in 1:length(adj_list)
        shuffle!(adj_list[i])
    end
end


function largest_component_size(components::Vector{Set{Int}})
    return isempty(components) ? 0 : maximum(length(comp) for comp in components)
end

end