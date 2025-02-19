import Random: AbstractRNG
@kwdef struct Deterministic{T} <: ContinuousUnivariateDistribution
    val::T = nothing
end

function Distributions.rand(rng::AbstractRNG, d::Deterministic)
    rand(rng)
    return d.val
end
Distributions.logpdf(d::Deterministic, x::Number) = zero(x)
Distributions.logpdf(d::Deterministic, x::Nothing) = 0
Distributions.mean(d::Deterministic) = d.val

# Bijectors.bijector(d::Deterministic) = identity

Ds(env::Environment, s, a) = Deterministic()
Da(agent::Agent, o) = Deterministic()
Do(sensor::Sensor, s) = Deterministic()

function DisturbanceDistribution(sys::System)
    return DisturbanceDistribution((o) -> Da(sys.agent, o),
                                   (s, a) -> Ds(sys.env, s, a),
                                   (s) -> Do(sys.sensor, s))
end
