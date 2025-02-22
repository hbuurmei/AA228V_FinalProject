struct ImportanceSamplingEstimator
    p  # nominal distribution
    q  # proposal distribution
    m  # number of samples
end

function estimate(alg::ImportanceSamplingEstimator, sys, ψ)
    (; p, q, m) = alg
    τs = [rollout(sys, q) for i in 1:m]
    ps = [pdf(p, τ) for τ in τs]
    qs = [pdf(q, τ) for τ in τs]
    ws = ps ./ qs
    return mean(w * isfailure(ψ, τ) for (w, τ) in zip(ws, τs))
end
