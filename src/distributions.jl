import Random: AbstractRNG
@kwdef struct Deterministic{T} <: ContinuousUnivariateDistribution
    val::T = nothing
end
struct Anything <: ContinuousUnivariateDistribution end
Distributions.logpdf(d::Anything, x::Number) = 0.0
Distributions.logpdf(d::Anything, x::AbstractVector{<:Number}) = 0.0

function Distributions.rand(rng::AbstractRNG, d::Deterministic)
    rand(rng)
    return d.val
end
Distributions.logpdf(d::Deterministic, x::Number) = 0.0
Distributions.logpdf(d::Deterministic, x::Nothing) = 0.0
Distributions.mean(d::Deterministic) = d.val

# Bijectors.bijector(d::Deterministic) = identity

Ds(env::Environment, s, a) = Deterministic()
Da(agent::Agent, o) = Deterministic()
Do(sensor::Sensor, s) = Deterministic()

function DisturbanceDistribution(sys::System)
    return DisturbanceDistribution(
        (o) -> Da(sys.agent, o),
        (s, a) -> Ds(sys.env, s, a),
        (s) -> Do(sys.sensor, s)
    )
end
