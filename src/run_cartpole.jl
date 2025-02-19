@assert basename(pwd()) == "AA228V_FinalProject"
import Pkg; Pkg.activate(".")
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = ".venv/bin/python"
using PythonCall
using Revise
using Distributions, YAML
using SignalTemporalLogic
using IntervalSets

includet("system.jl")
includet("distributions.jl")
includet("cartpole.jl")
includet("specification.jl")

pyimport("sys").path.append("src")
gym = pyimport("gym"); gym.make("CartPole-v1")
np = pyimport("numpy")
RLAgent_py = pyimport("reinforcement_learning"=>"RLAgent")
ILAgent_py = pyimport("imitation_learning"=>"ILAgent")


# sensor = AdditiveNoiseSensor(Product([Deterministic(0.0),
#                                  Normal(0.0, 0.5),
#                                  Deterministic(0.0),
#                                  Deterministic(0.0)]))
sensor = AdditiveNoiseSensor(Product([Normal(0, 0.1),
                                 Normal(0.0, 0.1),
                                 Normal(0, 0.2),
                                 Normal(0, 0.2)]))
sys = System(
    RLAgent(expert_agent),
    CartPole(; render=true),
    sensor
)

@show rollout(sys; d=get_depth(sys))

ψ = LTLSpecification(@formula □(s->s[1] ∈ 0±2.4) ∧ □(s->rad2deg(s[3]) ∈ 0±12))


D = DisturbanceDistribution{Disturbance{Nothing, Nothing, Vector{Float64}}}(
        (o) -> Da(sys.agent, o),
        (s, a) -> Ds(sys.env, s, a),
        (s) -> Product([Deterministic(0), Deterministic(0),
                        Normal(0, deg2rad(5.0)), Normal(0, deg2rad(5.0))]))
qτ = NominalTrajectoryDistribution(Ps(sys.env), D, get_depth(sys))
rollout(sys, qτ; d=100)

sys′ = System(
    RLAgent(expert_agent),
    CartPole(),
    sensor
)
τs = map(1:100) do _
    rollout(sys′, qτ; d=100)
end
@show mean(τ->isfailure(ψ, τ), τs)
