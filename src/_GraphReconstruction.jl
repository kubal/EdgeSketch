 __precompile__(false)

module _GraphReconstruction

using Base.Threads
using _EdgeSketch
using _kEmbeddings

export predict_top_edges
export evaluate_prediction
export evaluate_precision


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


function evaluate_prediction(
        predicted_edges::Vector{Tuple{Int, Int}},
        adj_list::Vector{Vector{Tuple{Int, Float64}}}
    )::Tuple{Float64,Float64,Float64}

    true_edges_weights = sum(sum(weight for (_, weight) in neighbors) for neighbors in adj_list) / 2
    true_edges_count = sum(length(neighbors) for neighbors in adj_list) / 2

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
    
    return precision , recall, recall_weighted
end



function evaluate_precision(
    predicted_edges::Vector{Tuple{Int, Int}},
    adj_list::Vector{Vector{Tuple{Int, Float64}}},
    true_edges_count::Int
)::Float64

# true_edges_count = sum(length(neighbors) for neighbors in adj_list) / 2
predicted_true_edges_counter = 0

for edge in predicted_edges
    i, j = edge
    for (neighbor, weight) in adj_list[i]
        if neighbor == j
            predicted_true_edges_counter += 1
            break
        end
    end
end

precision = round(predicted_true_edges_counter / length(predicted_edges), digits=5)
return precision 

end

end

