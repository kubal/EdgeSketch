__precompile__(false)

module _Brain1M

using Random
using Statistics
using LinearAlgebra
using SparseArrays
using Dates
using Base.Threads
using HDF5
using HNSW

using _FastExpSketch
using _GraphModels
using _EdgeSketch
using _NodeSketch
using _SketchOperations
using _GraphReconstruction
using _GraphModularity
using _SketchModularity
using _Logger


export preprocess_h5_streaming
export build_knn_adjlist
export load_brain1M_graph_streaming
export singlecell_louvain_on_sketches
export brain1M_streaming_louvain_on_sketches


# =============================================================================
# Single-cell RNA-seq derived k-NN graph (Brain-1M).
#
# A dense graph DERIVED from a single-cell gene-expression matrix: nodes = cells,
# edges connect each cell to its most similar cells in PCA space. Running
# Louvain/Leiden community detection on such a k-NN graph is the STANDARD way to
# discover cell types in scRNA-seq pipelines (Scanpy, Seurat), which makes it a
# real-world use case for EdgeSketch rather than a synthetic benchmark.
#
# Pipeline (mirrors a standard Scanpy/Seurat workflow), fully streamed so even
# the full 1.3M-cell dataset fits in memory:
#   1. read the 10x gene-cell count matrix (10x HDF5 .h5) in cell blocks;
#   2. library-size normalize per cell, then log1p;
#   3. pick the most highly variable genes;
#   4. reduce to a few principal components (randomized PCA)
#   5. connect cells with approximate k-NN (HNSW) and weight each edge by the
#      Jaccard overlap of the two cells' neighbourhoods (shared-nearest-neighbour);
#   6. build an EdgeSketch of the resulting weighted graph;
#   7. run Louvain on the sketch (and, as a reference, on the full adjacency
#      list) and compare modularity, memory and runtime.
#
# Steps 1-5 build the graph (load_brain1M_graph_streaming); steps 6-7 are the
# experiment (singlecell_louvain_on_sketches).
#
# Datasets:
#  - 1.3M Brain Cells (E18 mouse; ~4 GB HDF5):
#    https://cf.10xgenomics.com/samples/cell/1M_neurons/1M_neurons_filtered_gene_bc_matrices_h5.h5
#    Download this .h5 file and place it (unchanged) at exactly:
#      <project root>/data/brain1M/1M_neurons_filtered_gene_bc_matrices_h5.h5
# =============================================================================


const BRAIN1M_DEFAULT_PATH = joinpath(@__DIR__, "..", "data", "brain1M",
    "1M_neurons_filtered_gene_bc_matrices_h5.h5")


# --- Sorts the row indices within each column of a CSC matrix in place.
function _sort_csc_rows!(A::SparseMatrixCSC)
    colptr = A.colptr
    rv = rowvals(A)
    nv = nonzeros(A)
    @inbounds for c in 1:size(A, 2)
        lo = colptr[c]
        hi = colptr[c + 1] - 1
        hi <= lo && continue
        rng = lo:hi
        issorted(view(rv, rng)) && continue
        perm = sortperm(view(rv, rng))
        rv[rng] = rv[rng][perm]
        nv[rng] = nv[rng][perm]
    end
    return A
end


# --- Reads only cells c0..c1 from an open 10x HDF5 group as a sparse
# --- genes x (c1-c0+1) matrix (block-wise read keeps memory bounded).
function _read_h5_block(gobj, indptr::Vector{Int}, n_genes::Int, c0::Int, c1::Int)
    lo = indptr[c0] + 1
    hi = indptr[c1 + 1]
    nb = c1 - c0 + 1
    data = gobj["data"][lo:hi]
    indices = gobj["indices"][lo:hi]
    colptr = (indptr[c0:(c1 + 1)] .- indptr[c0]) .+ 1
    rowval = Vector{Int}(indices) .+ 1
    nzval = Float64.(data)
    A = SparseMatrixCSC(n_genes, nb, colptr, rowval, nzval)
    _sort_csc_rows!(A)
    return A
end


# --- In-place per-cell library-size normalization (to target_sum) + log1p. ---
function _normalize_block!(A::SparseMatrixCSC, target_sum::Float64)
    vals = nonzeros(A)
    @inbounds for c in 1:size(A, 2)
        libsize = 0.0
        for idx in nzrange(A, c)
            libsize += vals[idx]
        end
        scale = libsize > 0 ? target_sum / libsize : 0.0
        for idx in nzrange(A, c)
            vals[idx] = log1p(vals[idx] * scale)
        end
    end
    return A
end


# --- Accumulates per-gene sum and sum-of-squares (over cells) from a block. ---
function _accum_gene_stats!(gene_sum::Vector{Float64}, gene_sqsum::Vector{Float64},
        A::SparseMatrixCSC)
    rows = rowvals(A)
    vals = nonzeros(A)
    @inbounds for c in 1:size(A, 2)
        for idx in nzrange(A, c)
            gg = rows[idx]
            v = vals[idx]
            gene_sum[gg] += v
            gene_sqsum[gg] += v * v
        end
    end
end


# --- Streaming counts -> PCA embedding for a 10x HDF5 file ---
function preprocess_h5_streaming(path::AbstractString;
        genome::AbstractString = "",
        max_cells::Int = 0,
        n_hvg::Int = 1000,
        n_pcs::Int = 50,
        target_sum::Float64 = 1.0e4,
        block_cells::Int = 100_000,
        oversample::Int = 10,
        n_iter::Int = 3)
    isfile(path) || error("10x HDF5 file not found: $(path)")
    t0 = time()
    emb = h5open(path, "r") do f
        grp = isempty(genome) ? first(keys(f)) : genome
        g = f[grp]
        shape = read(g["shape"])
        n_genes = Int(shape[1])
        n_total = Int(shape[2])
        indptr = Vector{Int}(read(g["indptr"]))
        n_use = max_cells > 0 ? min(max_cells, n_total) : n_total
        println("Streaming preprocess: $(n_genes) genes x $(n_use) cells " *
                "(blocks of $(block_cells)) ..."); flush(stdout)

        # --- Pass 1: per-gene mean / variance over all cells ---
        gene_sum = zeros(Float64, n_genes)
        gene_sqsum = zeros(Float64, n_genes)
        c = 1
        while c <= n_use
            c1 = min(c + block_cells - 1, n_use)
            A = _read_h5_block(g, indptr, n_genes, c, c1)
            _normalize_block!(A, target_sum)
            _accum_gene_stats!(gene_sum, gene_sqsum, A)
            A = nothing
            println("  pass1 cells $(c)-$(c1) done ($(round(time() - t0, digits=1))s)")
            flush(stdout)
            c = c1 + 1
        end
        gene_mean = gene_sum ./ n_use
        gene_var = gene_sqsum ./ n_use .- gene_mean .^ 2
        n_keep = min(n_hvg, n_genes)
        hvg = sort(partialsortperm(gene_var, 1:n_keep; rev = true))
        mu = gene_mean[hvg]
        sd = sqrt.(max.(gene_var[hvg], 1e-12))

        # --- Pass 2: build HVG x cells matrix (Float32) block by block ---
        blocks = SparseMatrixCSC{Float32, Int}[]
        c = 1
        while c <= n_use
            c1 = min(c + block_cells - 1, n_use)
            A = _read_h5_block(g, indptr, n_genes, c, c1)
            _normalize_block!(A, target_sum)
            push!(blocks, SparseMatrixCSC{Float32, Int}(A[hvg, :]))
            A = nothing
            println("  pass2 cells $(c)-$(c1) done ($(round(time() - t0, digits=1))s)")
            flush(stdout)
            c = c1 + 1
        end
        At = reduce(hcat, blocks)          # HVG x cells, Float32
        empty!(blocks)
        println("  HVG matrix $(size(At)), nnz=$(nnz(At)); randomized PCA($(n_pcs)) ...")
        flush(stdout)
        npc = min(n_pcs, n_keep, n_use)
        return _randomized_pca(At, mu, sd, npc; oversample = oversample, n_iter = n_iter)
    end
    println("  embedding: $(size(emb, 1)) cells x $(size(emb, 2)) PCs " *
            "(in $(round(time() - t0, digits=1))s)"); flush(stdout)
    return emb
end


# --- Randomized truncated PCA ---
function _randomized_pca(At::SparseMatrixCSC{<:Real, <:Integer},
        mu::Vector{Float64}, sd::Vector{Float64}, npc::Int;
        oversample::Int = 10, n_iter::Int = 3, seed::Int = 0)
    g, n = size(At)
    l = min(npc + oversample, g, n)
    invsd = 1.0 ./ sd

    Zmul = function (X)
        Dx = X .* invsd
        AY = transpose(At) * Dx
        corr = transpose(mu) * Dx       # 1 x l
        return AY .- corr
    end
  
    Ztmul = function (Y)
        AtY = At * Y
        colsum = sum(Y; dims = 1)       # 1 x l
        return (AtY .- mu * colsum) .* invsd
    end

    thin_q = Y -> qr(Y).Q * Matrix{Float64}(I, size(Y, 1), size(Y, 2))

    rng = MersenneTwister(seed)
    Y = Zmul(randn(rng, g, l))
    Q = thin_q(Y)
    for _ in 1:n_iter
        Q = thin_q(Zmul(Ztmul(Q)))
    end
    B = Matrix(transpose(Ztmul(Q)))     # l x g  (= Q' Z)
    F = svd(B)
    U = Q * F.U
    scores = U[:, 1:npc] .* transpose(F.S[1:npc])
    return scores
end


# --- Jaccard overlap of two sorted integer vectors: |a cap b| / |a cup b|. ---
@inline function _jaccard_weight(a::Vector{Int}, b::Vector{Int})
    ia = 1
    ib = 1
    inter = 0
    la = length(a)
    lb = length(b)
    @inbounds while ia <= la && ib <= lb
        x = a[ia]
        y = b[ib]
        if x == y
            inter += 1
            ia += 1
            ib += 1
        elseif x < y
            ia += 1
        else
            ib += 1
        end
    end
    uni = la + lb - inter
    return uni == 0 ? 0.0 : inter / uni
end


# --- Symmetrizes directed neighbour lists into a Jaccard weighted adjacency list, 
# --- as in PhenoGraph (Levine et al., Cell 2015) and Seurat's SNN graph. 
function _symmetrize_jaccard_adjlist(dir::Vector{Vector{Int}}, n::Int)
    # Sorted neighbour sets including self (SNN convention).
    sets = Vector{Vector{Int}}(undef, n)
    @threads for i in 1:n
        di = dir[i]
        s = Vector{Int}(undef, length(di) + 1)
        @inbounds for t in eachindex(di)
            s[t] = di[t]
        end
        s[end] = i
        sort!(s)
        sets[i] = s
    end

    # Undirected union edges: keep, per node i, only neighbours j > i (dedup).
    upper = [Int[] for _ in 1:n]
    @inbounds for i in 1:n
        for j in dir[i]
            a, b = i < j ? (i, j) : (j, i)
            push!(upper[a], b)
        end
    end
    @threads for i in 1:n
        v = upper[i]
        isempty(v) && continue
        sort!(v)
        w = 1
        @inbounds for t in 2:length(v)
            if v[t] != v[w]
                w += 1
                v[w] = v[t]
            end
        end
        resize!(v, w)
    end

    # Jaccard weight per undirected edge (parallel; read-only on `sets`).
    wts = Vector{Vector{Float64}}(undef, n)
    @threads for i in 1:n
        ui = upper[i]
        si = sets[i]
        wi = Vector{Float64}(undef, length(ui))
        @inbounds for t in eachindex(ui)
            wi[t] = _jaccard_weight(si, sets[ui[t]])
        end
        wts[i] = wi
    end

    # Build the symmetric weighted adjacency list (single-threaded -> race-free).
    adj_list = [Tuple{Int, Float64}[] for _ in 1:n]
    @inbounds for i in 1:n
        ui = upper[i]
        wi = wts[i]
        for t in eachindex(ui)
            j = ui[t]
            w = wi[t]
            push!(adj_list[i], (j, w))
            push!(adj_list[j], (i, w))
        end
    end
    return adj_list
end


# --- Approximate k nearest neighbours via HNSW. ---
function _knn_dir_hnsw(embedding::AbstractMatrix{<:Real}, k::Int;
        M::Int = 16, ef_construction::Int = 100, ef::Int = 64)
    n = size(embedding, 1)
    data = [Vector{Float64}(@view embedding[i, :]) for i in 1:n]
    # ef must exceed k for the search to return k neighbours; ef_construction
    # should also scale with k, otherwise recall collapses for large k.
    efC = max(ef_construction, 2 * k)
    hnsw = HierarchicalNSW(data; efConstruction = efC, M = M,
        ef = max(ef, k + 1))
    add_to_graph!(hnsw)
    idxs, _ = knn_search(hnsw, data, k + 1)   # +1: the first hit is the point itself
    dir = [Int[] for _ in 1:n]
    @threads for i in 1:n
        di = Int[]
        for j in idxs[i]
            jj = Int(j)
            jj == i && continue
            push!(di, jj)
        end
        dir[i] = di
    end
    return dir
end


# --- Builds a symmetric, Jaccard-weighted k-NN adjacency list from a
# --- (cells x dims) embedding, using approximate-NN (HNSW) search.
function build_knn_adjlist(embedding::AbstractMatrix{<:Real};
        k::Int = 15,
        M::Int = 16,
        ef_construction::Int = 100,
        ef::Int = 64)
    n = size(embedding, 1)
    println("Building k-NN graph (k=$(k), n=$(n), backend=hnsw, weight=jaccard) ...")
    t_start = time()

    dir = _knn_dir_hnsw(embedding, k; M = M, ef_construction = ef_construction, ef = ef)
    adj_list = _symmetrize_jaccard_adjlist(dir, n)

    nr_edges = sum(length(neigh) for neigh in adj_list) ÷ 2
    ratio = round(nr_edges / n, digits = 2)
    println("  k-NN graph: $(n) nodes, $(nr_edges) edges, |E|/|V| = $(ratio) " *
            "(in $(round(time() - t_start, digits=1))s)")
    return adj_list
end


# --- Memory-bounded loader for very large cell counts (up to the full 1.3M):
# --- streaming HVG preprocessing, then a Jaccard-weighted k-NN (HNSW) graph.
function load_brain1M_graph_streaming(; path::AbstractString = BRAIN1M_DEFAULT_PATH,
        max_cells::Int = 0, k::Int = 15, n_hvg::Int = 1000, n_pcs::Int = 50,
        block_cells::Int = 100_000)
    embedding = preprocess_h5_streaming(path; max_cells = max_cells, n_hvg = n_hvg,
        n_pcs = n_pcs, block_cells = block_cells)
    return build_knn_adjlist(embedding; k = k)
end


# --- Runs Louvain on the full graph and on the EdgeSketch, then compares. ---
function singlecell_louvain_on_sketches(name::AbstractString,
        adj_list::Vector{Vector{Tuple{Int, Float64}}};
        sketch_size::Int = 50,
        seed::Int = 1233,
        run_reference::Bool = true)
    Random.seed!(seed)
    nr_nodes = length(adj_list)
    nr_edges = Int(sum(length(neighbors) for neighbors in adj_list) / 2)

    # --- EdgeSketch generation (two independent seeds) ---
    t_s = @elapsed graph_sketch = create_graph_sketch(adj_list, sketch_size, 12)
    graph_sketch_verification = create_graph_sketch(adj_list, sketch_size, 445)
    memory_size_adj_list_mb = Base.summarysize(adj_list) / 1e6
    memory_size_sketched_graph_mb = Base.summarysize(graph_sketch) / 1e6

    # --- Louvain on sketches ---
    println("Running Louvain on sketches..."); flush(stdout)
    t1 = @elapsed communities_1st_phase_sketch = louvain_method_sketch(graph_sketch)
    obtained_modularity_1 = calculate_modularity(adj_list, communities_1st_phase_sketch)
    estimated_modularity_1 = calculate_modularity_sketch(graph_sketch_verification, communities_1st_phase_sketch)

    t2 = @elapsed communities_2nd_phase_sketch = louvain_method_2nd_phase_sketch(graph_sketch, communities_1st_phase_sketch)
    obtained_modularity_2 = calculate_modularity(adj_list, communities_2nd_phase_sketch)
    estimated_modularity_2 = calculate_modularity_sketch(graph_sketch_verification, communities_2nd_phase_sketch)

    best_obtained, best_estimated, best_communities_sketch = estimated_modularity_1 > estimated_modularity_2 ?
        (obtained_modularity_1, estimated_modularity_1, communities_1st_phase_sketch) :
        (obtained_modularity_2, estimated_modularity_2, communities_2nd_phase_sketch)

    # --- Print graph + sketch results FIRST (and flush), so they are visible
    # --- before the (potentially many-hour) reference Louvain runs. ---
    emph("\nGRAPH")
    info(" - name                 $(name)")
    info(" - nr_nodes             $(nr_nodes)")
    info(" - nr_edges             $(nr_edges)")
    emph("\nSKETCH")
    info(" - sketch_size          $(sketch_size)")
    info(" - sketching_time       $(f2(t_s)) sec")
    info(" - memory (adj_list)    $(greene(f4(memory_size_adj_list_mb)*" MB"))")
    info(" - memory (EdgeSketch)  $(greene(f4(memory_size_sketched_graph_mb)*" MB")) ")
    emph("\nMODULARITY (sketch)")
    info(" - Louvain on sketches         $(greene(f2(t1+t2)*" sec"))")
    info("   obtained modularity         $(f4(best_obtained))")
    info("   estimated modularity        $(f4(best_estimated))")
    info("   clusters                    $(length(best_communities_sketch))")
    flush(stdout)

    if !run_reference
        info("\n(reference Louvain on adj_list skipped: run_reference=false)")
        flush(stdout)
        return
    end

    # --- Louvain on adjacency list (reference; can be very slow on huge graphs) ---
    println("\nRunning Louvain on adjacency list (reference)..."); flush(stdout)
    t_al_1 = @elapsed communities_AL_1 = louvain_method(adj_list)
    modularity_AL_1 = calculate_modularity(adj_list, communities_AL_1)
    t_al_2 = @elapsed communities_AL_2 = louvain_method_2nd_phase(adj_list, communities_AL_1)
    modularity_AL_2 = calculate_modularity(adj_list, communities_AL_2)

    emph("\nMODULARITY (reference)")
    info(" - Louvain on adj_list         $(greene(f2(t_al_1+t_al_2)*" sec"))")
    info("   modularity                  $(f4(modularity_AL_2))")
    info("   clusters                    $(length(communities_AL_2))")
    flush(stdout)
end


# --- Full-scale experiment wrapper: streams the 10x HDF5 (so even 1.3M cells
# --- fit in memory), builds the Jaccard k-NN graph, then runs the sketch
# --- Louvain (fast) and prints it BEFORE the reference Louvain (which may take
# --- many hours on a graph with hundreds of millions of edges).
function brain1M_streaming_louvain_on_sketches(; sketch_size::Int = 100,
        max_cells::Int = 0, k::Int = 500, n_hvg::Int = 1000, n_pcs::Int = 50,
        block_cells::Int = 100_000, run_reference::Bool = true,
        path::AbstractString = BRAIN1M_DEFAULT_PATH)
    adj_list = load_brain1M_graph_streaming(; path = path, max_cells = max_cells, k = k,
        n_hvg = n_hvg, n_pcs = n_pcs, block_cells = block_cells)
    ncells = length(adj_list)
    singlecell_louvain_on_sketches(
        "BRAIN-1M (scRNA-seq k-NN, k=$(k), jaccard, n=$(ncells))", adj_list;
        sketch_size = sketch_size, run_reference = run_reference)
end


end
