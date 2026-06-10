__precompile__(false)

module _Gplus

using Random
using Base.Threads

using _FastExpSketch
using _GraphModels
using _EdgeSketch
using _GraphModularity
using _SketchModularity
using _Logger


export load_gplus_graph
export gplus_louvain_on_sketches


# ego-Gplus (SNAP, dense social graph; ~107K nodes).
# Download: snap.stanford.edu/data/gplus_combined.txt.gz
# Gunzip into data/ego-Gplus/gplus_combined.txt.
const GPLUS_DEFAULT_PATH = joinpath(@__DIR__, "..", "data", "ego-Gplus", "gplus_combined.txt")


# --- Loads the ego-Gplus plain-text edge list into an undirected weighted
# --- adjacency list.
#
# The graph is built as UNDIRECTED (symmetric): each parsed arc (u, v)
# adds both u->v and v->u. Self-loops are dropped and parallel edges are collapsed.
#
# weighted = false (default): every edge gets weight 1.0 (simple graph).
# weighted = true: edge weight = multiplicity of the (u, v) pair in the file.
#
# Returns (adj_list, node_list) where:
#   adj_list :: Vector{Vector{Tuple{Int, Float64}}} indexed by consecutive ids,
#   node_list :: Vector{String} mapping new consecutive id -> original node id.
function load_gplus_graph(; filepath::AbstractString = GPLUS_DEFAULT_PATH,
        weighted::Bool = false)
    if !isfile(filepath)
        error("ego-Gplus edge list not found: $(filepath)\n" *
              "Download snap.stanford.edu/data/gplus_combined.txt.gz and gunzip it " *
              "into data/ego-Gplus/gplus_combined.txt.")
    end

    println("Loading edge list from $(filepath) ...")

    # Map original (possibly non-numeric / huge) node ids to consecutive indices.
    id_map = Dict{String, Int}()
    node_list = String[]
    adj = Vector{Vector{Int}}()

    # Returns the consecutive index of a raw id token, creating it on demand.
    # `token` is typically a SubString from `split`; a String is materialised
    # only when a brand-new node is inserted.
    @inline function get_idx(token::AbstractString)
        idx = get(id_map, token, 0)
        if idx == 0
            s = String(token)
            push!(node_list, s)
            push!(adj, Int[])
            idx = length(node_list)
            id_map[s] = idx
        end
        return idx
    end

    arc_count = 0
    self_loops = 0
    t_start = time()

    open(filepath, "r") do file
        for line in eachline(file)
            isempty(line) && continue
            c = line[1]
            (c == '#' || c == '%') && continue

            parts = split(line)
            length(parts) < 2 && continue

            iu = get_idx(parts[1])
            iv = get_idx(parts[2])

            if iu == iv
                self_loops += 1
                continue
            end

            # Symmetrize: store both directions (undirected graph).
            push!(adj[iu], iv)
            push!(adj[iv], iu)
            arc_count += 1

            if arc_count % 20_000_000 == 0
                elapsed = round(time() - t_start, digits=1)
                println("  read $(arc_count) arcs, $(length(node_list)) nodes so far ($(elapsed)s)")
            end
        end
    end

    n = length(node_list)
    println("Parsed $(arc_count) arcs, $(n) nodes (skipped $(self_loops) self-loops)")

    # Collapse duplicate neighbors per node and assign weights.
    adj_list = Vector{Vector{Tuple{Int, Float64}}}(undef, n)
    for i in 1:n
        neighbors = adj[i]
        if isempty(neighbors)
            adj_list[i] = Tuple{Int, Float64}[]
            continue
        end
        sort!(neighbors)
        collapsed = Tuple{Int, Float64}[]
        prev = neighbors[1]
        count = 1
        for k in 2:length(neighbors)
            cur = neighbors[k]
            if cur == prev
                count += 1
            else
                push!(collapsed, (prev, weighted ? Float64(count) : 1.0))
                prev = cur
                count = 1
            end
        end
        push!(collapsed, (prev, weighted ? Float64(count) : 1.0))
        adj_list[i] = collapsed
        adj[i] = Int[]  # free intermediate storage
    end

    adj = nothing
    GC.gc()

    # Drop nodes that became isolated (e.g. their only edges were self-loops).
    non_isolated = [i for i in 1:n if !isempty(adj_list[i])]
    if length(non_isolated) != n
        old_to_new = Dict(old => new_idx for (new_idx, old) in enumerate(non_isolated))
        new_adj = Vector{Vector{Tuple{Int, Float64}}}(undef, length(non_isolated))
        new_node_list = Vector{String}(undef, length(non_isolated))
        for (new_idx, old) in enumerate(non_isolated)
            new_adj[new_idx] = [(old_to_new[j], w) for (j, w) in adj_list[old]]
            new_node_list[new_idx] = node_list[old]
        end
        adj_list = new_adj
        node_list = new_node_list
    end

    nr_nodes = length(adj_list)
    nr_edges = sum(length(neigh) for neigh in adj_list) ÷ 2
    edge_to_node_ratio = round(nr_edges / nr_nodes, digits=2)
    adj_list_mb = Base.summarysize(adj_list) / (1024 * 1024)

    println("Undirected graph: $(nr_nodes) nodes (|V|), $(nr_edges) edges (|E|), |E|/|V| = $(edge_to_node_ratio)")
    println("Adjacency list size: $(round(adj_list_mb, digits=2)) MB " *
            "(loaded in $(round(time() - t_start, digits=1))s)")

    return adj_list, node_list
end


# --- Runs Louvain on the full ego-Gplus graph and on the EdgeSketch, then
# --- compares modularity / memory / time. 
function gplus_louvain_on_sketches(;
        sketch_size::Int = 50,
        filepath::AbstractString = GPLUS_DEFAULT_PATH,
        weighted::Bool = false,
        seed::Int = 1233)
    Random.seed!(seed)

    name = "EGO-GPLUS (Google+ social graph)"

    # --- Load graph ---
    adj_list, _ = load_gplus_graph(; filepath=filepath, weighted=weighted)
    nr_nodes = length(adj_list)
    nr_edges = Int(sum(length(neighbors) for neighbors in adj_list) / 2)

    # --- EdgeSketch generation (two independent seeds: one for clustering,
    # --- one as an independent estimator for the modularity estimate) ---
    t_s = @elapsed graph_sketch = create_graph_sketch(adj_list, sketch_size, 12)
    graph_sketch_verification = create_graph_sketch(adj_list, sketch_size, 445)
    memory_size_adj_list_mb = Base.summarysize(adj_list) / 1e6
    memory_size_sketched_graph_mb = Base.summarysize(graph_sketch) / 1e6

    # --- Louvain on sketches ---
    println("Running Louvain on sketches...")
    t1 = @elapsed communities_1st_phase_sketch = louvain_method_sketch(graph_sketch)
    obtained_modularity_1 = calculate_modularity(adj_list, communities_1st_phase_sketch)
    estimated_modularity_1 = calculate_modularity_sketch(graph_sketch_verification, communities_1st_phase_sketch)

    t2 = @elapsed communities_2nd_phase_sketch = louvain_method_2nd_phase_sketch(graph_sketch, communities_1st_phase_sketch)
    obtained_modularity_2 = calculate_modularity(adj_list, communities_2nd_phase_sketch)
    estimated_modularity_2 = calculate_modularity_sketch(graph_sketch_verification, communities_2nd_phase_sketch)

    best_obtained, best_estimated, best_communities_sketch = estimated_modularity_1 > estimated_modularity_2 ?
        (obtained_modularity_1, estimated_modularity_1, communities_1st_phase_sketch) :
        (obtained_modularity_2, estimated_modularity_2, communities_2nd_phase_sketch)

    # --- Louvain on adjacency list (reference) ---
    println("Running Louvain on adjacency list...")
    t_al_1 = @elapsed communities_AL_1 = louvain_method(adj_list)
    modularity_AL_1 = calculate_modularity(adj_list, communities_AL_1)
    t_al_2 = @elapsed communities_AL_2 = louvain_method_2nd_phase(adj_list, communities_AL_1)
    modularity_AL_2 = calculate_modularity(adj_list, communities_AL_2)

    # --- Print results ---
    emph("\nGRAPH")
    info(" - name                 $(name)")
    info(" - nr_nodes             $(nr_nodes)")
    info(" - nr_edges             $(nr_edges)")
    emph("\nSKETCH")
    info(" - sketch_size          $(sketch_size)")
    info(" - sketching_time       $(f2(t_s)) sec")
    info(" - memory (adj_matrix)  $(nr_nodes^2 * 8 / 1e6)  MB")
    info(" - memory (adj_list)    $(greene(f4(memory_size_adj_list_mb)*" MB"))")
    info(" - memory (EdgeSketch)  $(greene(f4(memory_size_sketched_graph_mb)*" MB")) ")

    emph("\nMODULARITY")
    info(" - Louvain on adj_list         $(greene(f2(t_al_1+t_al_2)*" sec"))")
    info("   modularity                  $(f4(modularity_AL_2))")
    info("   clusters                    $(length(communities_AL_2))")
    info("")
    info(" - Louvain on sketches         $(greene(f2(t1+t2)*" sec"))")
    info("   obtained modularity         $(f4(best_obtained))")
    info("   estimated modularity        $(f4(best_estimated))")
    info("   clusters                    $(length(best_communities_sketch))")
end


end
