using Optimization, OptimizationOptimJL
using ComponentArrays
import Distances: WeightedCityblock, WeightedEuclidean
using Base.Iterators

# dist = WeightedCityblock([0.0; 0.0; 1.0; 0.0; 0.0])
# dist = WeightedCityblock([1.0; 1.0; 1.0; 1.0; 0.0])
dist = WeightedEuclidean([0.0; 1.0; 0.5; 1.0; 0.0])

adjacent(xs) = zip(xs[1:(end - 1)], xs[2:end])


struct PiecedTrajectoryDistribution <: TrajectoryDistribution
    Ps # initial state distribution
    D  # disturbance distribution
    d  # depth
end
function PiecedTrajectoryDistribution(sys::System, d = get_depth(sys))
    D = DisturbanceDistribution{Any}(
        (o) -> Da(sys.agent, o),
        (s, a) -> Ds(sys.env, s, a),
        (s) -> Do(sys.sensor, s)
    )
    return PiecedTrajectoryDistribution(Ps(sys.env), D, d)
end
initial_state_distribution(p::PiecedTrajectoryDistribution) = Anything()
disturbance_distribution(p::PiecedTrajectoryDistribution, t) = p.D
depth(p::PiecedTrajectoryDistribution) = p.d


# n_partitions = 10
n_partitions = 2
τ₀ = rollout(sys, NominalTrajectoryDistribution(sys); d = 200)
noise_idx = (3:4)
x₀ = [s.x.xo[noise_idx] for s in τ₀]
τs = partition(τ₀, length(τ₀) ÷ n_partitions)
xs = partition(x₀, length(τ₀) ÷ n_partitions) |> collect
depths = length.(τs)
states = [first(τ).s for τ in τs]
# xs = mean.(xs)

# s_fail = [0.0; deg2rad(-13); 0; 0]
s_fail = [0.0; deg2rad(-5); 0; 0]

function fopt(us, ps)
    (; states, xs) = us
    xs = eachcol(xs)
    # @show sum(states)
    # @show sum(xs)
    (; sys, s_fail, dist, noise_idx) = ps
    # qτs = map(zip(xs, depths)) do (x, d)
    #     D = DisturbanceDistribution(
    #         (o) -> Da(sys.agent, o),
    #         (s, a) -> Ds(sys.env, s, a),
    #         (s) -> let x_ = zeros(length(s)-1)
    #             x_[noise_idx] .= x
    #             # x_[noise_idx] .-= 1.5
    #             Deterministic(x_)
    #         end
    #     )
    #     # PiecedTrajectoryDistribution(
    #     #     Ps(sys.env), D, d
    #     # )
    #     NominalTrajectoryDistribution(
    #         Ps(sys.env), D, d
    #     )
    # end

    depths_ = [0; cumsum(depths)]
    τs_new = map(zip(states, adjacent(depths_))) do (s1, (d_lhs, d_rhs))
        sys.env.pyenv.reset()
        # rollout(sys, s1, qτ; d)
        s = s1
        𝐱 = [
            let x_ = zeros(length(s) - 1)
                    @assert x isa AbstractVector
                    x_[noise_idx] .= x
                    (; xa = nothing, xs = nothing, xo = x_)
            end for x in xs[(d_lhs + 1):d_rhs]
        ]
        rollout(sys, s, 𝐱; d = (d_rhs - d_lhs))
    end
    # obj1 = dist(last(τs_new[end]).s, s_fail)
    # objective 1: violate the specification
    obj1 = let s_end = last(τs_new[end]).s
        # (s_end[1] + s_end[3])*1000
        s_end[3] * 1_000
    end

    # objective 2: reduce defect of adjacent trajectories
    obj2 = sum(adjacent(τs_new); init = 0.0) do (τ_lhs, τ_rhs)
        dist(last(τ_lhs).s, first(τ_rhs).s)
    end

    # objective 3: maximize likelihood of noise
    pτ = PiecedTrajectoryDistribution(sys)
    obj3 = -sum(τs_new) do τ
        # logpdf(pτ, τ)/1e6
        pdf(pτ, τ)
    end * 0
    # TODO: Probably fix this...
    # obj3 = 0

    # objective 4: maximize likelihood of initial state
    obj4 = -logpdf(
        initial_state_distribution(NominalTrajectoryDistribution(sys)),
        first(states)
    )

    return obj1 + obj2 + obj3 + obj4
end

# fopt((; states, xs=xmeans), (; sys, s_fail, dist))
fopt_ = OptimizationFunction(fopt, AutoFiniteDiff())
u₀ = ComponentVector((; states, xs = stack(x₀))) .|> x -> convert(Float32, x)
@show length(u₀)
prob = OptimizationProblem(fopt_, u₀, (; sys = sys, s_fail, dist, noise_idx))
prob′ = OptimizationProblem(fopt_, u₀, (; sys = sys′, s_fail, dist, noise_idx))

# sol = solve(prob′, ConjugateGradient(); show_trace=true)
# τ = rollout(sys, sol.u.states[1],
#         [let x_ = zeros(length(sol.u.states[1])-1)
#                 @assert x isa AbstractVector
#                 x_[noise_idx] .= x
#                 (; xa=nothing, xs=nothing, xo=x_)
#             end for x in eachcol(sol.u.xs)])
# isfailure(ψ, τ)

# prob = OptimizationProblem(fopt_, u₀, (; sys=sys, s_fail, dist, noise_idx))
# sol = solve(prob, ConjugateGradient(); show_trace=true, maxiters=100)
# sol = solve(prob, NelderMead(); show_trace=true, maxiters=100)
# sol = solve(prob, NelderMead(); maxiters=100)
"""
sol = solve(prob, ConjugateGradient(); show_trace=true, maxiters=100)
# solve(prob, LBFGS(); show_trace=true, maxiters=5)

expanded_noise = vcat([repeat([e], n) for (e, n) in zip(sol.u.xs, depths)]...)
D = DisturbanceDistribution(
    (o) -> Da(sys.agent, o),
    (s, a) -> Ds(sys.env, s, a),
    (s) -> let i = convert(Int, last(s))
        x = zeros(length(s)-1)
        x[noise_idx] .= expanded_noise[i]
        Deterministic(x)
    end
)
qτ = NominalTrajectoryDistribution(Ps(sys.env), D, get_depth(sys))
τ = rollout(sys, qτ; d=length(expanded_noise))
τ = rollout(sys, sol.u.states[1],
        [let x_ = zeros(length(sol.u.states[1])-1)
                @assert x isa AbstractVector
                x_[noise_idx] .= x
                (; xa=nothing, xs=nothing, xo=x_)
            end for x in eachcol(sol.u.xs)])
isfailure(ψ, τ)
# pdf()
"""
