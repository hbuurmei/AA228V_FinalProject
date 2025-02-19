import YAML: load_file
# const State_t = SVector{4, Float64}
const State_t = Vector{Float64}
struct CartPole <: Environment
      pyenv
      s::State_t
end

CartPole(; render=false) = let pyenv = gym.envs.classic_control.CartPoleEnv(render_mode=(render ? "human" : ""))
    CartPole(pyenv, pyconvert(State_t, pyenv.reset()[0]))
end

function (env::CartPole)(s, a, xs=missing)
    i = last(s)
    env.pyenv.state = np.array(s[1:end-1])
    retval = env.pyenv.step(a)
    s_new = [pyconvert(State_t, retval[0]); i+1]
    terminated = pyconvert(Bool, retval[2])
    # terminated && env.pyenv.reset()
    s_new
end

Ps(env::CartPole) = Product([[
    Uniform(-0.05, 0.05)
    for _ in 1:4
]; Deterministic(1)])

struct AdditiveNoiseSensor <: Sensor
    Do
end

(sensor::AdditiveNoiseSensor)(s) = sensor(s, rand(Do(sensor, s)))
(sensor::AdditiveNoiseSensor)(s, x) = s + [x; 0.0]  # last one is iteration number

Do(sensor::AdditiveNoiseSensor, s) = sensor.Do

# Os(sensor::AdditiveNoiseSensor) = I

@enum AgentType begin
    expert_agent
    imitation_agent
end

struct RLAgent <: Agent
    pyagent
end
RLAgent(agent_t::AgentType) = RLAgent(Val(agent_t))
function RLAgent(::Val{expert_agent})
    cfg = YAML.load_file("config/train/rl_agent_cartpole.yaml")
    pyagent = RLAgent_py(cfg)
    pyagent.load_model("data/models/expert_policy.pt")
    pyagent.exploration_rate = 0
    RLAgent(pyagent)
end

function RLAgent(::Val{imitation_agent})
    cfg = YAML.load_file("config/train/il_agent_cartpole.yaml")
    pyagent = ILAgent_py(cfg)
    pyagent.load_model("data/models/BC_policy.pt")
    RLAgent(pyagent)
end

(agent::RLAgent)(s, x=nothing) = try
    # remove "iteration" state, clamp state
    s_ = clamp.(s[1:4], [0±4.8, 0±Inf, 0±0.41887903, 0±Inf])

    action = agent.pyagent.act(np.array(s_)) |> x->pyconvert(Int, x)
    @assert action ∈ [0, 1]
    return action
catch e
    @show s, x
    throw(e)
end

# const Project1MediumSystem::Type = System{ProportionalController, CartPole, AdditiveNoiseSensor}
# const Project2MediumSystem::Type = Project1MediumSystem
const CartPoleSystem = System{RLAgent, CartPole, AdditiveNoiseSensor}

get_depth(sys::CartPoleSystem) = 100
