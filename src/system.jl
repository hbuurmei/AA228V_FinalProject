abstract type Agent end
abstract type Environment end
abstract type Sensor end

struct System{A<:Agent, E<:Environment, S<:Sensor}
    agent::A
    env::E
    sensor::S
end

function Base.step(sys::System, s)
    o = sys.sensor(s)
    a = sys.agent(o)
    s′ = sys.env(s, a)
    return (; o, a, s′)
end

function rollout(sys::System; d=get_depth(sys))
    s = rand(Ps(sys.env))
    τ_t = @NamedTuple{s::State_t, o::State_t, a::Int}
    # τ = τ_t[]
    τ = []
    for t in 1:d
        o, a, s′ = step(sys, s)
        push!(τ, (; s, o, a))
        s = s′
    end
    return τ .|> identity
end

function rollout(sys::System, s; d)
    τ_t = @NamedTuple{s::State_t, o::State_t, a::Int}
	# τ = τ_t[]
	τ = []
	for t in 1:d
		o, a, s′ = step(sys, s)
		push!(τ, (; s, o, a))
		s = s′
	end
	return τ .|> identity
end

struct Disturbance{AT, ST, OT}
    xa::AT # agent disturbance
    xs::ST # environment disturbance
    xo::OT # sensor disturbance
end

struct DisturbanceDistribution{DT}
    Da # agent disturbance distribution
    Ds # environment disturbance distribution
    Do # sensor disturbance distribution
end
DisturbanceDistribution(Da, Ds, Do) = DisturbanceDistribution{Any}(Da, Ds, Do)
disturbance_type(::DisturbanceDistribution{DT}) where DT = DT

function Base.step(sys::System, s, D::DisturbanceDistribution)
    xo = rand(D.Do(s))
    o = sys.sensor(s, xo)
    xa = rand(D.Da(o))
    a = sys.agent(o, xa)
    xs = rand(D.Ds(s, a))
    s′ = sys.env(s, a, xs)
    x = Disturbance(xa, xs, xo)
    return (; o, a, s′, x)
end

function Distributions.fit(d::DisturbanceDistribution, samples, w)
    𝐱_agent = [s.x.x_agent for s in samples]
    𝐱_env = [s.x.x_env for s in samples]
    𝐱_sensor = [s.x.x_sensor for s in samples]
    px_agent = fit(d.px_agent, 𝐱_agent, w)
    px_env = fit(d.px_env, 𝐱_env, w)
    px_sensor = fit(d.px_sensor, 𝐱_sensor, w)
    return DisturbanceDistribution(px_agent, px_env, px_sensor)
end

Distributions.fit(𝐝::Vector, samples, w) = [fit(d, [s[t] for s in samples], w) for (t, d) in enumerate(𝐝)]

Distributions.fit(d::Sampleable, samples, w::Missing) = fit(typeof(d), samples)
Distributions.fit(d::Sampleable, samples, w) = fit_mle(typeof(d), samples, w)

abstract type TrajectoryDistribution end
function initial_state_distribution(p::TrajectoryDistribution) end
function disturbance_distribution(p::TrajectoryDistribution, t) end
function depth(p::TrajectoryDistribution) end

(p::TrajectoryDistribution)(τ) = pdf(p, τ)

struct NominalTrajectoryDistribution <: TrajectoryDistribution
    Ps # initial state distribution
    D  # disturbance distribution
    d  # depth
end

function NominalTrajectoryDistribution(sys::System, d=get_depth(sys))
    D = DisturbanceDistribution{Any}((o) -> Da(sys.agent, o),
                                (s, a) -> Ds(sys.env, s, a),
                                (s) -> Do(sys.sensor, s))
    return NominalTrajectoryDistribution(Ps(sys.env), D, d)
end

initial_state_distribution(p::NominalTrajectoryDistribution) = p.Ps
disturbance_distribution(p::NominalTrajectoryDistribution, t) = p.D
depth(p::NominalTrajectoryDistribution) = p.d

function Distributions.logpdf(D::DisturbanceDistribution, s, o, a, x)
    logp_xa = logpdf(D.Da(o), x.xa)
    logp_xs = logpdf(D.Ds(s, a), x.xs)
    logp_xo = logpdf(D.Do(s), x.xo)
    return logp_xa + logp_xs + logp_xo
end

function Distributions.logpdf(p::TrajectoryDistribution, τ)
    logprob = logpdf(initial_state_distribution(p), τ[1].s)
    for (t, step) in enumerate(τ)
        s, o, a, x = step
        logprob += logpdf(disturbance_distribution(p, t), s, o, a, x)
    end
    return logprob
end

Distributions.pdf(p::TrajectoryDistribution, τ) = exp(logpdf(p, τ))

function Base.step(sys::System, s, x)
    o = sys.sensor(s, x.xo)
    a = sys.agent(o, x.xa)
    s′ = sys.env(s, a, x.xs)
    return (; o, a, s′)
end

function rollout(sys::System, s, 𝐱::XT; d=length(𝐱)) where XT
    # τ_t = @NamedTuple{s::State_t, o::State_t, a::Int, x::XT}
    # τ = τ_t[]
    τ = []
    for t in 1:d
        x = 𝐱[t]
        o, a, s′ = step(sys, s, x)
        push!(τ, (; s, o, a, x))
        s = s′
    end
    return τ .|> identity
end

function rollout(sys::System, s, p::TrajectoryDistribution; d=depth(p))
    # X_t = disturbance_type(disturbance_distribution(p, 1))
    # τ_t = @NamedTuple{s::State_t, o::State_t, a::Int, x::X_t}
    # τ = τ_t[]
    τ = []
    for t = 1:d
        o, a, s′, x = step(sys, s, disturbance_distribution(p, t))
        push!(τ, (; s, o, a, x))
        s = s′
    end
    return τ
end

function rollout(sys::System, p::TrajectoryDistribution; d=depth(p))
    s = rand(initial_state_distribution(p))
    X_t = disturbance_type(disturbance_distribution(p, 1))
    τ_t = @NamedTuple{s::State_t, o::State_t, a::Int, x::X_t}
    # τ = τ_t[]
    τ = []
    for t = 1:d
        o, a, s′, x = step(sys, s, disturbance_distribution(p, t))
        push!(τ, (; s, o, a, x))
        s = s′
    end
    return τ .|> identity
end

function mean_step(sys::System, s, D::DisturbanceDistribution)
    xo = mean(D.Do(s))
    o = sys.sensor(s, xo)
    xa = mean(D.Da(o))
    a = sys.agent(o, xa)
    xs = mean(D.Ds(s, a))
    s′ = sys.env(s, a, xs)
    x = Disturbance(xa, xs, xo)
    return (; o, a, s′, x)
end

function mean_rollout(sys::System, p::TrajectoryDistribution; d=depth(p))
    s = mean(initial_state_distribution(p))
    X_t = disturbance_type(disturbance_distribution(p, 1))
    τ_t = @NamedTuple{s::State_t, o::State_t, a::Int, x::X_t}
    # τ = τ_t[]
    τ = []
    for t = 1:d
        o, a, s′, x = mean_step(sys, s, disturbance_distribution(p, t))
        push!(τ, (; s, o, a, x))
        s = s′
    end
    return τ .|> identity
end
