__precompile__(false)

module _FastExpSketch

using Random

export StreamElement
export fast_expsketch


struct StreamElement
    id1::Int
    id2::Int
    weight::Float64
end

function fast_expsketch(
        stream::Vector{<:StreamElement}, 
        m::Number, 
        h::Function
    )
    
    permInit = collect(1:m)
    M = fill(Inf, m)
    SAMPLE = fill((0, 0), m)
    maxValue = Inf

    for e in stream
        S = 0
        updateMax = false
        P = copy(permInit)
        id1, id2 = min(e.id1,e.id2), max(e.id1,e.id2)
        binI = bitstring(id1)*bitstring(id2)
        Random.seed!(hash(binI))

        for k in 1:m
            binK = bitstring(k)

            hashValue = h(binI * binK)
            sampleValue = -log(hashValue) / e.weight

            S += sampleValue / (m - k + 1)
            if S > maxValue
                break
            end

            r = rand(k:m)

            tmp = P[k];
            P[k] = P[r]
            P[r] = tmp;

            j = P[k]

            if M[j] == maxValue
                updateMax = true
            end

            if S < M[j]
                M[j] = S
                SAMPLE[j] = (id1, id2)
            end
        end

        if updateMax
            maxValue = maximum(M)
        end
    end
    
    return M, SAMPLE
end

end