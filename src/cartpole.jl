import YAML: load_file
# const State_t = SVector{4, Float64}
const State_t = Vector{Float64}
struct CartPole <: Environment
      pyenv
      s::State_t
end

CartPole() = let pyenv = gym.envs.classic_control.CartPoleEnv()
    CartPole(pyenv, pyconvert(State_t, pyenv.reset()[0]))
end

function (env::CartPole)(s, a, xs=missing)
    env.pyenv.state = np.array(s)
    s_new = env.pyenv.step(a)[0]
    pyconvert(State_t, s_new)
end

Ps(env::CartPole) = Product([
    Uniform(-0.05, 0.05)
    for _ in 1:4
])

struct AdditiveNoiseSensor <: Sensor
    Do
end

(sensor::AdditiveNoiseSensor)(s) = sensor(s, rand(Do(sensor, s)))
(sensor::AdditiveNoiseSensor)(s, x) = s + x

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

(agent::RLAgent)(s, a=missing) = agent.pyagent.act(np.array(s)) |> x->pyconvert(Int, x)

# const Project1MediumSystem::Type = System{ProportionalController, CartPole, AdditiveNoiseSensor}
# const Project2MediumSystem::Type = Project1MediumSystem
const CartPoleSystem = System{RLAgent, CartPole, AdditiveNoiseSensor}

get_depth(sys::CartPoleSystem) = 100
