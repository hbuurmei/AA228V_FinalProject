module AA228V_FinalProject
using PythonCall
using Distributions, YAML
using SignalTemporalLogic
using IntervalSets

export System, RLAgent, CartPole, AdditiveNoiseSensor
export NominalTrajectoryDistribution, DisturbanceDistribution
export rollout, step, isfailure
export expert_agent, imitation_agent
export estimate, ImportanceSamplingEstimator

const gym_ptr = Ref{Py}()
const np_ptr = Ref{Py}()
const RLAgent_py_ptr = Ref{Py}()
const ILAgent_py_ptr = Ref{Py}()

function __init__()
    src_path = joinpath(split(ENV["JULIA_PYTHONCALL_EXE"], ".venv")[1], "src")
    @show "Adding path $(src_path)"
    pyimport("sys").path.append(src_path)
    gym_ptr[] = pyimport("gym"); gym_ptr[].make("CartPole-v1")
    np_ptr[] = pyimport("numpy")
    RLAgent_py_ptr[] = pyimport("reinforcement_learning" => "RLAgent")
    return ILAgent_py_ptr[] = pyimport("imitation_learning" => "ILAgent")
end

include("system.jl")
include("distributions.jl")
include("cartpole.jl")
include("specification.jl")
include("failure_probability.jl")

end
