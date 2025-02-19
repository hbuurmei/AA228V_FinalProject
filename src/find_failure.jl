using Optimization, OptimizationOptimJL
using ComponentArrays
import Distances: WeightedCityblock, WeightedEuclidean
using Base.Iterators

# dist = WeightedCityblock([0.0; 0.0; 1.0; 0.0; 0.0])
# dist = WeightedCityblock([1.0; 1.0; 1.0; 1.0; 0.0])
dist = WeightedEuclidean([1.0; 1.0; 1.0; 1.0; 0.0])

adjacent(xs) = zip(xs[1:end-1], xs[2:end])


struct PiecedTrajectoryDistribution <: TrajectoryDistribution
    Ps # initial state distribution
    D  # disturbance distribution
    d  # depth
end
function PiecedTrajectoryDistribution(sys::System, d=get_depth(sys))
    D = DisturbanceDistribution{Any}((o) -> Da(sys.agent, o),
                                (s, a) -> Ds(sys.env, s, a),
                                (s) -> Do(sys.sensor, s))
    return PiecedTrajectoryDistribution(Ps(sys.env), D, d)
end
initial_state_distribution(p::PiecedTrajectoryDistribution) = Anything()
disturbance_distribution(p::PiecedTrajectoryDistribution, t) = p.D
depth(p::PiecedTrajectoryDistribution) = p.d


# n_partitions = 10
n_partitions = 4
τ₀ = rollout(sys, NominalTrajectoryDistribution(sys); d=60)
noise_idx = (3:4)
x₀ = [s.x.xo[noise_idx] for s in τ₀]
τs = partition(τ₀, length(τ₀)÷n_partitions)
xs = partition(x₀, n_partitions)
depths = length.(τs)
states = [first(τ).s for τ in τs]
xs = mean.(xs)

# s_fail = [0.0; deg2rad(-13); 0; 0]
s_fail = [0.0; deg2rad(-5); 0; 0]

function fopt(us, ps)
    (; states, xs) = us
    (; sys, s_fail, dist, noise_idx) = ps
    qτs = map(zip(xs, depths)) do (x, d)
        D = DisturbanceDistribution(
            (o) -> Da(sys.agent, o),
            (s, a) -> Ds(sys.env, s, a),
            (s) -> let x_ = zeros(length(s)-1)
                x_[noise_idx] .= x
                # x_[noise_idx] .-= 1.5
                Deterministic(x_)
            end
        )
        # PiecedTrajectoryDistribution(
        #     Ps(sys.env), D, d
        # )
        NominalTrajectoryDistribution(
            Ps(sys.env), D, d
        )
    end

    τs_new = map(zip(states, qτs, depths)) do (s1, qτ, d)
        # rollout(sys, s1, qτ; d)
        rollout(sys, qτ; d)
    end
    # obj1 = dist(last(τs_new[end]).s, s_fail)
    # objective 1: violate the specification
    obj1 = let s_end = last(τs_new[end]).s
        s_end[1] + s_end[3]
    end

    # objective 2: reduce defect of adjacent trajectories
    obj2 = sum(adjacent(τs_new)) do (τ_lhs, τ_rhs)
        dist(last(τ_lhs).s, first(τ_rhs).s)
    end

    # objective 3: maximize likelihood of noise
    pτ = PiecedTrajectoryDistribution(sys)
    obj3 = -sum(τs_new) do τ
        logpdf(pτ, τ)
    end
    # TODO: Probably fix this...
    # obj3 = 0

    # objective 4: maximize likelihood of initial state
    obj4 = -logpdf(initial_state_distribution(NominalTrajectoryDistribution(sys)),
                   first(states))

    return obj1+obj2+obj3+obj4
end

# fopt((; states, xs=xmeans), (; sys, s_fail, dist))
fopt_ = OptimizationFunction(fopt, AutoFiniteDiff())
u₀ = ComponentVector((; states, xs)) .|> x->convert(Float32, x)
@show length(u₀)
prob = OptimizationProblem(fopt_, u₀, (; sys=sys′, s_fail, dist, noise_idx))
sol = solve(prob, NelderMead(); show_trace=true, maxiters=1_00)
# solve(prob, LBFGS(); show_trace=true)

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
isfailure(ψ, τ)
# pdf()
