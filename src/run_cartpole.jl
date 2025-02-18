@assert basename(pwd()) == "AA228V_FinalProject"
import Pkg; Pkg.activate(".")
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = ".venv/bin/python"
using PythonCall
using Revise
using Distributions, YAML

includet("system.jl")
includet("distributions.jl")
includet("cartpole.jl")

pyimport("sys").path.append("src")
gym = pyimport("gym"); gym.make("CartPole-v1")
np = pyimport("numpy")
RLAgent_py = pyimport("reinforcement_learning"=>"RLAgent")
ILAgent_py = pyimport("imitation_learning"=>"ILAgent")


sys = System(
    RLAgent(expert_agent),
    CartPole(),
    AdditiveNoiseSensor(Product([Deterministic(0.0), Normal(0.0, 0.5), Deterministic(0.0), Deterministic(0.0)]))
)

@show rollout(sys; d=get_depth(sys))
