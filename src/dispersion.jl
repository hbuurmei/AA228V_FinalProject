using StaticArrays
using NearestNeighbors: KDTree, nn, knn
using ForwardDiff: gradient, hessian
using LinearAlgebra
using LogExpFunctions
using StatsBase: mean, median
using LazySets, Polyhedra

function softmin_objective_knn(
        ξ′::SVector{N, <:Real}, knn_points::Vector{<:SVector{N, <:Real}},
        tau::Float64
    ) where {N}
    distances_sq = [norm(ξ′ - x_i) for x_i in knn_points]
    neg_distances_scaled = -distances_sq ./ tau
    return -tau * logsumexp(neg_distances_scaled)
end

function optimize_point_knn(
        points::Vector{SVector{N, Float64}}, p::SVector{N, Float64};
        k_neighbors::Int = 6, tau::Float64 = 1.0, n_steps::Int = 3,
        tree = KDTree(hcat(points...); leafsize = 10), verbose = false
    ) where {N}
    # Start with a new random sample
    ξ = p
    Δ = zero(ξ)

    verbose && println("Initial Point (Random Sample): $(ξ)")
    initial_objective_val = softmin_objective_knn(ξ, points, tau) # Objective with respect to all points for initial value
    verbose && println("Initial Objective Value (all points): $(initial_objective_val)")


    for step in 1:n_steps
        idxs, dists = knn(tree, ξ + Δ, k_neighbors)
        knn_points = [points[i] for i in idxs]

        ∇f = gradient(Δ -> softmin_objective_knn(ξ + Δ, knn_points, tau), Δ)
        H = hessian(Δ -> softmin_objective_knn(ξ + Δ, knn_points, tau), Δ)

        Δ_step = -H \ ∇f
        Δ = Δ + SVector{N, Float64}(Δ_step)

        current_objective_val_knn = softmin_objective_knn(ξ + Δ, knn_points, tau)
        current_objective_val_all = softmin_objective_knn(ξ + Δ, points, tau)

        verbose && println("Step $(step): Objective Value (k-NN) = $(current_objective_val_knn), Objective Value (all points)=$(current_objective_val_all), Point = $(ξ + Δ)")
    end

    optimized_point = ξ + Δ
    verbose && println("Final Point: $(optimized_point)")
    final_objective_val_all = softmin_objective_knn(ξ + Δ, points, tau)
    verbose && println("Final Objective Value (all points): $(final_objective_val_all)")

    return optimized_point
end


function main_knn_random_start()
    N = 2 # Dimension
    num_points = 20
    points = [SVector(randn(N)...) for _ in 1:num_points]

    k_neighbors = 2 * N
    tau = 1.0
    n_steps = 3

    optimized_point = optimize_point_knn(points, SVector(randn(N)...); k_neighbors = k_neighbors, tau = tau, n_steps = n_steps)

    tree_optimized = KDTree(hcat(points...); leafsize = 10) # Rebuild tree with *all* original points

    idxs_optimized_knn, dists_optimized_knn = knn(tree_optimized, optimized_point, k_neighbors)
    knn_points_optimized = [points[i] for i in idxs_optimized_knn] # Use original 'points'

    min_dist_knn = minimum([norm(optimized_point - x) for x in knn_points_optimized])
    min_dist_all = minimum([norm(optimized_point - x) for x in points]) # Compare to all original 'points'

    println("\nKNN Optimization Done (Random Start).")
    println("Optimized point (KNN): $(optimized_point)")
    println("Minimum distance to k-NN points: $(min_dist_knn)")
    println("Minimum distance to all other points: $(min_dist_all)")

    # Now through brute-force
    points_proposed = [SVector(randn(N)...) for _ in 1:10_000]
    min_dist_brute_force = maximum(points_proposed) do p
        nn(tree_optimized, p)[2]
    end
    return println("Minimum distance through brute-force: $(min_dist_brute_force)")
end


# main_knn_random_start()

function maximum_dispersion(pts::AbstractVector{<:AbstractVector}, n_rep = 30)
    N = length(pts[1])
    P = VPolytope(convex_hull(pts))

    pts = SVector{N}.(pts)
    pts_mat = stack(pts; dims = 1)
    bounds = extrema.(eachcol(pts_mat))

    tree = KDTree(pts; leafsize = 10)

    candidates = [
        begin
                sample = first.(bounds) .+ rand(N) .* (last.(bounds) .- first.(bounds))
                while sample ∉ P
                    sample = first.(bounds) .+ rand(N) .* (last.(bounds) .- first.(bounds))
            end
                optimize_point_knn(pts, SVector{N}(sample...); k_neighbors = 2 * N, tau = 1.0, n_steps = 3, tree)
            end for _ in 1:n_rep
    ]

    valid_candidates = filter(candidates) do p
        p ∈ P
    end
    dist, idx = findmax(valid_candidates) do p
        nn(tree, p)[2]
    end
    point = valid_candidates[idx]
    return (; dist, point)
end

function average_dispersion(pts::AbstractVector{<:AbstractVector}, n_rep = 10_000)
    N = length(pts[1])
    tree = KDTree(SVector{N}.(pts))
    P = VPolytope(convex_hull(pts))
    #
    pts_mat = stack(pts; dims = 1)
    bounds = extrema.(eachcol(pts_mat))
    return [
        begin
                sample = first.(bounds) .+ rand(N) .* (last.(bounds) .- first.(bounds))
                while sample ∉ P
                    sample = first.(bounds) .+ rand(N) .* (last.(bounds) .- first.(bounds))
            end
                _idx, dist = nn(tree, sample)
                dist
            end for _ in 1:n_rep
    ] |> median
end
