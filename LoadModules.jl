# Run this code in the catalog with the code with the command 
# ------------------------ 
# julia -i LoadModules.jl
# ------------------------
# 
# One can set the number of threads XXX to use
# ------------------------------- 
# julia -t XXX -i LoadModules.jl
# ------------------------------- 


using Pkg


# function to ensure that packeges are installed
function ensure_package_installed(pkg::String)

    io = IOBuffer()
    Pkg.status(; io=io)
    status_output = String(take!(io))

    installed_packages = [
        length(split_line) == 4 ? split_line[3] : split_line[2]
        for line in split(status_output, "\n") if occursin("[", line)
        for split_line in [split(line)]
    ]
    
    if pkg in installed_packages
        # println("$pkg is already installed.")
    else
        println("Installing $pkg...")
        Pkg.add(pkg)
    end
end


# ensure that packages are installed
packages = ["Random", "Revise", "Distributions", "StatsBase", "Printf", "Plots", "PGFPlotsX", "Dates", "StatsPlots", "LaTeXStrings", "SparseArrays"]

for pkg in packages
    ensure_package_installed(pkg)
end


# Revise is for development
using Revise

push!(LOAD_PATH, joinpath(pwd(), "src"))
push!(LOAD_PATH, joinpath(pwd(), "experiments"))

# this project modules
using _Logger
using _FastExpSketch
using _GraphModels
using _EdgeSketch
using _NodeSketch
using _SketchOperations
using _GraphReconstruction
using _GraphModularity
using _SketchModularity
using _Tests
using _Experiments
using _Epinions



