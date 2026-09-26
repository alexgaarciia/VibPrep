from itertools import product

def generate_preprocessing_configs(
    average_options=(False, True),
    regions=("full", "fingerprint", "amide", "lipid", "nucleic"),
    baseline_options=("none", "polynomial", "als", "aspls"),
    scatter_options=("none", "snv"),
    smoothing_options=("none", "savitzky_golay", "moving_average"),
    derivative_options=("none", 1, 2, "1+2"),
    normalization_options=("none", "area", "l2", "max")):
    """
    Generate preprocessing configs (dicts) with rule-based filtering.

    Returns
    -------
    list[dict]
        Each dict has keys:
        average, region, baseline, scatter, smoothing, derivative, normalization
    """

    configs = []

    for avg, region, baseline, scatter, smoothing, deriv, norm in product(
        average_options,
        regions,
        baseline_options,
        scatter_options,
        smoothing_options,
        derivative_options,
        normalization_options,
    ):
        steps = []

        if baseline != "none":
            steps.append(("baseline", baseline))
        if scatter != "none":
            steps.append(("scatter", scatter))
        if smoothing != "none":
            steps.append(("smoothing", smoothing))
        if deriv != "none":
            steps.append(("derivative", deriv))
        if norm != "none":
            steps.append(("normalization", norm))

        configs.append({
            "average": avg,
            "region": region,
            "steps": steps,
            "baseline": baseline,
            "scatter": scatter,
            "smoothing": smoothing,
            "derivative": deriv,
            "normalization": norm,
        })
        
    return configs
