from sympy import symbols, Eq, solve


def estimate_logg_from_teff_MS(teff):
    if teff < 5000.0:
        return 4.6
    elif (5000 <= teff) and (teff < 6300):
        return 4.6 - 5.0e-4 * (teff - 5000)
    else:
        return 3.95


def estimate_teff_from_logg_rgb(logg):
    if logg < 3.65:
        return 5200.0 - 441 * (3.65 - logg)
    else:
        return 5900.0 - 1400 * (3.15 - logg)


def estimate_bprp(teff, logg, mh):
    x = symbols("x")
    eq1 = Eq(
        7981
        - 4138.3457 * x
        + 1264.9366 * x**2
        - 130.4388 * x**3
        + 285.8393 * logg
        - 324.2196 * logg * x
        + 106.8511 * logg * x**2
        - 4.9825 * logg * x**3
        - teff,
        0,
    )
    sol = solve(eq1)
    return sol[0]
