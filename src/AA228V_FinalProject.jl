module AA228V_FinalProject
using PythonCall
using Distributions, YAML
using SignalTemporalLogic
using IntervalSets

export System, RLAgent, CartPole, AdditiveNoiseSensor
export NominalTrajectoryDistribution, DisturbanceDistribution
export rollout, step, isfailure
export expert_agent, imitation_agent

pyimport("sys").path.append(
    joinpath(split(ENV["JULIA_PYTHONCALL_EXE"], "/.venv")[1], "src"))
const gym = pyimport("gym"); gym.make("CartPole-v1")
const np = pyimport("numpy")
const RLAgent_py = pyimport("reinforcement_learning"=>"RLAgent")
const ILAgent_py = pyimport("imitation_learning"=>"ILAgent")

include("system.jl")
include("distributions.jl")
include("cartpole.jl")
include("specification.jl")

end
