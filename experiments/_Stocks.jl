__precompile__(false)

module _Stocks

using Random
using Statistics
using LinearAlgebra
using Dates
using Base.Threads

using _FastExpSketch
using _GraphModels
using _EdgeSketch
using _NodeSketch
using _SketchOperations
using _GraphReconstruction
using _GraphModularity
using _SketchModularity
using _Logger


export load_stocks_graph
export stocks_louvain_on_sketches


# =============================================================================
# Data source (Kaggle "Huge Stock Market Dataset", Boris Marjanovic)
#
#   https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs
#
# Download (needs a free Kaggle account), then unzip so that all per-ticker
# files (both the Stocks and the ETFs folders of the archive) land flat in:
#   data/Kaggle/   (e.g. aapl.us.txt, msft.us.txt, spy.us.txt, ...)
#
# Each file is a CSV with a header line and split/dividend-adjusted prices:
#   Date,Open,High,Low,Close,Volume,OpenInt
#   2015-11-10,28.12,28.40,27.95,28.31,1234567,0
#   ...
# Some files are empty (0 bytes) or very short; they are skipped automatically.
# =============================================================================


# --- Reads a single Kaggle ticker file, returning (dates, closes) within the
# --- [start_date, end_date] window. Rows are kept in chronological order.
# dates  :: Vector{Date}    - trading days on which the ticker was quoted
# closes :: Vector{Float64} - closing price for each day
#      
# Returns empty vectors when the file is empty / unreadable / outside the window.
function _read_ticker_file(filepath::AbstractString, start_date::Date, end_date::Date)
    dates = Date[]
    closes = Float64[]

    filesize(filepath) == 0 && return dates, closes

    open(filepath, "r") do file
        for line in eachline(file)
            isempty(line) && continue
            # Skip the header line ("Date,Open,High,Low,Close,Volume,OpenInt").
            (line[1] == 'D' || line[1] == 'd') && continue

            parts = split(line, ',')
            length(parts) < 5 && continue

            local d::Date
            try
                d = Date(parts[1])  # ISO format yyyy-mm-dd
            catch
                continue
            end
            (d < start_date || d > end_date) && continue

            close = tryparse(Float64, parts[5])
            (close === nothing || close <= 0.0) && continue

            push!(dates, d)
            push!(closes, close)
        end
    end

    return dates, closes
end


# --- Loads a correlation graph from the Kaggle stock dataset.
#
# Pipeline (see the description in the paper):
#   1. read daily close prices for every ticker inside [start_date, end_date];
#   2. align all tickers to a common trading-day calendar (forward-filling small
#      gaps), keeping only liquid, well-covered tickers;
#   3. turn prices into daily log returns;
#   4. measure how similarly each pair of tickers moves (Pearson correlation);
#   5. assign edge weight w = |corr|^beta (soft thresholding: beta > 1 amplifies
#      strong correlations and suppresses noise near zero); discard edges with
#      |corr| < corr_threshold (set corr_threshold = 0.0 for a complete graph);
#   6. emit a symmetric weighted adjacency list (no self-loops), drop isolated
#      nodes and remap ids to 1..N.
#
# Keyword arguments:
#   dir            directory with per-ticker files (default data/Kaggle)
#   start_date,end_date   price window as "yyyy-mm-dd" strings
#   beta           soft-threshold exponent for |corr|^beta (default 4)
#   corr_threshold keep edges with |corr| >= this value (default 0.2; set 0.0
#                  for a complete graph)
#   min_coverage   min fraction of calendar days a ticker must actually trade
#   max_tickers    cap on number of nodes (most-covered tickers kept; controls
#                  graph size / density, default 2000)
#
# Returns (adj_list, tickers) with
#   adj_list :: Vector{Vector{Tuple{Int, Float64}}} indexed by consecutive ids,
#   tickers  :: Vector{String} mapping new id -> ticker symbol.
function load_stocks_graph(;
        dir::Union{AbstractString, AbstractVector} = joinpath(@__DIR__, "..", "data", "Kaggle"),
        start_date::AbstractString = "2015-11-10",
        end_date::AbstractString = "2017-11-10",
        beta::Real = 4,
        corr_threshold::Real = 0.2,
        min_coverage::Real = 0.98,
        max_tickers::Real = 2000)

    # Accept a single directory or several (e.g. Stocks + ETFs).
    dirs = dir isa AbstractString ? [dir] : collect(dir)
    for d in dirs
        if !isdir(d)
            error("Stock data directory not found: $(d)\n" *
                  "Download the Kaggle \"Huge Stock Market Dataset\" and unzip the " *
                  "per-ticker files into this folder (see the comment at the top of " *
                  "_Stocks.jl).")
        end
    end

    d_start = Date(start_date)
    d_end = Date(end_date)
    t_start = time()

    # --- Step 1: read every ticker file within the price window ---
    files = String[]
    for d in dirs
        append!(files, filter(f -> endswith(f, ".txt"), readdir(d; join=true)))
    end
    println("Reading $(length(files)) ticker files from $(join(dirs, ", ")) " *
            "($(start_date) .. $(end_date)) ...")

    ticker_dates = Vector{Vector{Date}}()
    ticker_closes = Vector{Vector{Float64}}()
    ticker_names = String[]

    for f in files
        dates, closes = _read_ticker_file(f, d_start, d_end)
        isempty(dates) && continue
        # Ticker symbol from "aapl.us.txt" -> "AAPL".
        name = uppercase(first(split(basename(f), '.')))
        push!(ticker_dates, dates)
        push!(ticker_closes, closes)
        push!(ticker_names, name)
    end

    nr_raw = length(ticker_names)
    println("  $(nr_raw) tickers have data in the window")
    nr_raw == 0 && error("No usable ticker data found in $(dir) for the given window.")

    # --- Step 2: build a common trading-day calendar ---
    # The most-observed ticker defines the master calendar (a liquid stock that
    # trades on every session). Other tickers are aligned to it.
    ref_idx = argmax(length.(ticker_dates))
    calendar = sort(unique(ticker_dates[ref_idx]))
    T = length(calendar)
    date_to_pos = Dict(d => i for (i, d) in enumerate(calendar))
    println("  master calendar: $(T) trading days " *
            "($(calendar[1]) .. $(calendar[end]))")

    # Align each ticker to the calendar (forward-fill small gaps). A ticker is
    # kept only if it trades on the first calendar day (no leading gap) and
    # covers at least `min_coverage` of the calendar with real observations.
    min_obs = ceil(Int, min_coverage * T)
    aligned_prices = Vector{Vector{Float64}}()
    kept_names = String[]
    kept_obs = Int[]

    for k in 1:nr_raw
        dates = ticker_dates[k]
        closes = ticker_closes[k]

        prices = fill(NaN, T)
        observed = 0
        for (d, c) in zip(dates, closes)
            pos = get(date_to_pos, d, 0)
            if pos != 0 && isnan(prices[pos])
                prices[pos] = c
                observed += 1
            end
        end

        observed < min_obs && continue
        isnan(prices[1]) && continue  # leading gap -> drop

        # Forward-fill internal gaps with the last known price.
        last_price = prices[1]
        for i in 2:T
            if isnan(prices[i])
                prices[i] = last_price
            else
                last_price = prices[i]
            end
        end

        push!(aligned_prices, prices)
        push!(kept_names, ticker_names[k])
        push!(kept_obs, observed)
    end

    println("  $(length(kept_names)) tickers kept after coverage filter " *
            "(min_coverage=$(min_coverage))")

    # Optionally cap the number of nodes, keeping the most-covered (most liquid)
    # tickers, to control graph size and memory.
    if length(kept_names) > max_tickers
        keep = sortperm(kept_obs; rev=true)[1:Int(max_tickers)]
        aligned_prices = aligned_prices[keep]
        kept_names = kept_names[keep]
        println("  capped to max_tickers=$(Int(max_tickers)) nodes (most-covered)")
    end

    # --- Step 3: prices -> daily log returns ---
    N = length(kept_names)
    N < 2 && error("Not enough tickers ($(N)) to build a correlation graph.")
    T_ret = T - 1
    # R[t, i] = log return of ticker i on day t.
    R = Matrix{Float64}(undef, T_ret, N)
    for i in 1:N
        p = aligned_prices[i]
        @inbounds for t in 1:T_ret
            R[t, i] = log(p[t + 1] / p[t])
        end
    end

    # --- Step 4: pairwise Pearson correlation ---
    valid = trues(N)
    for i in 1:N
        col = @view R[:, i]
        m = mean(col)
        col .-= m
        nrm = norm(col)
        if nrm == 0.0
            valid[i] = false   # constant series -> undefined correlation
        else
            col ./= nrm
        end
    end

    if !all(valid)
        keep = findall(valid)
        R = R[:, keep]
        kept_names = kept_names[keep]
        N = length(kept_names)
        println("  dropped $(count(!, valid)) constant-price tickers")
    end

    println("Computing $(N)x$(N) correlation matrix ...")
    C = transpose(R) * R   # N x N, entries in [-1, 1]

    # --- Step 5/6: correlation -> positive weights -> adjacency list ---
    beta_f = Float64(beta)
    thr = Float64(corr_threshold)
    println("Building adjacency list (beta=$(beta), corr_threshold=$(corr_threshold)) ...")

    adj_list = [Vector{Tuple{Int, Float64}}() for _ in 1:N]
    @threads for i in 1:N
        @inbounds for j in 1:N
            i == j && continue
            c = abs(C[i, j])
            c < thr && continue
            w = c^beta_f
            w <= 0.0 && continue
            push!(adj_list[i], (j, w))
        end
    end

    # --- Drop isolated nodes and remap ids to 1..N ---
    non_isolated = [i for i in 1:N if !isempty(adj_list[i])]
    if length(non_isolated) != N
        old_to_new = Dict(old => new_idx for (new_idx, old) in enumerate(non_isolated))
        new_adj = Vector{Vector{Tuple{Int, Float64}}}(undef, length(non_isolated))
        new_names = Vector{String}(undef, length(non_isolated))
        for (new_idx, old) in enumerate(non_isolated)
            new_adj[new_idx] = [(old_to_new[j], w) for (j, w) in adj_list[old] if haskey(old_to_new, j)]
            new_names[new_idx] = kept_names[old]
        end
        adj_list = new_adj
        kept_names = new_names
    end

    nr_nodes = length(adj_list)
    nr_edges = sum(length(neigh) for neigh in adj_list) ÷ 2
    edge_to_node_ratio = nr_nodes == 0 ? 0.0 : round(nr_edges / nr_nodes, digits=2)
    adj_list_mb = Base.summarysize(adj_list) / (1024 * 1024)

    println("Correlation graph: $(nr_nodes) nodes (|V|), $(nr_edges) edges (|E|), " *
            "|E|/|V| = $(edge_to_node_ratio)")
    println("Adjacency list size: $(round(adj_list_mb, digits=2)) MB " *
            "(built in $(round(time() - t_start, digits=1))s)")

    return adj_list, kept_names
end


# --- Runs Louvain on the full correlation graph and on the EdgeSketch, then
# --- compares modularity / memory / time. Mirrors gplus_louvain_on_sketches.
function stocks_louvain_on_sketches(;
        sketch_size::Int = 50,
        dir::Union{AbstractString, AbstractVector} = joinpath(@__DIR__, "..", "data", "Kaggle"),
        start_date::AbstractString = "2015-11-10",
        end_date::AbstractString = "2017-11-10",
        beta::Real = 4,
        corr_threshold::Real = 0.2,
        min_coverage::Real = 0.98,
        max_tickers::Real = 2000,
        seed::Int = 1233)
    Random.seed!(seed)

    # --- Load graph ---
    adj_list, _ = load_stocks_graph(; dir=dir, start_date=start_date, end_date=end_date,
        beta=beta, corr_threshold=corr_threshold, min_coverage=min_coverage, max_tickers=max_tickers)
    name = "STOCKS (correlation graph)"
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
