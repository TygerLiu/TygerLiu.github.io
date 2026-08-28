#!/usr/bin/env python3
"""Exact certificates for the finite computations in the paper.

Requires SymPy 1.14 or later.  All calculations take place over exact
polynomial or rational-function domains; no numerical approximation is used.
"""

from fractions import Fraction
import sympy as sp


def moments(P, kappa, ell, count):
    """Coefficients of P(x)g-kappa*x^3*(1-ell*x)*g^2=1-ell*x."""
    x = []
    for n in range(count):
        rhs = sp.Integer(1) if n == 0 else (-ell if n == 1 else sp.Integer(0))
        linear = sum(P[j] * x[n-j] for j in range(1, min(3, n) + 1))
        conv3 = sum(x[j] * x[n-3-j] for j in range(n-2)) if n >= 3 else 0
        conv4 = sum(x[j] * x[n-4-j] for j in range(n-3)) if n >= 4 else 0
        x.append(sp.expand(rhs - linear + kappa * conv3 - kappa * ell * conv4))
    return x


def hankels(mu, last):
    """h_n=det(mu[i+j])_(0<=i,j<=n), using fraction-free elimination."""
    ans = []
    for n in range(last + 1):
        matrix = sp.Matrix([[mu[i+j] for j in range(n+1)] for i in range(n+1)])
        ans.append(sp.factor(matrix.det(method="domain-ge")))
    return ans


def check_somos8(h, coeffs):
    alpha, beta, gamma, delta = coeffs
    for n in range(8, 12):
        residual = (
            h[n] * h[n-8]
            - alpha * h[n-1] * h[n-7]
            - beta * h[n-2] * h[n-6]
            - gamma * h[n-3] * h[n-5]
            - delta * h[n-4] ** 2
        )
        assert sp.cancel(residual) == 0


def coefficient_matrix(h):
    return sp.Matrix([
        [
            h[n-1] * h[n-7],
            h[n-2] * h[n-6],
            h[n-3] * h[n-5],
            h[n-4] ** 2,
        ]
        for n in range(8, 12)
    ])


def barry_conjectures_11_to_13():
    r = sp.symbols("r")

    d11 = r**4 - 2*r**3 + 8*r**2 + 2*r - 9
    c11 = (
        -(-r**8 + 8*r**7 - 21*r**6 + 40*r**5 - 35*r**4
          + 24*r**3 - 71*r**2 - 8*r) / d11,
        8*(r**9 - 6*r**8 + 17*r**7 - 30*r**6 + 15*r**5
           - 14*r**4 - r**3 - 14*r**2) / d11,
        8*(r**10 - 2*r**8 + 29*r**7 - 32*r**6 + 39*r**5
           + 18*r**4 + 11*r**3 - r**2 + r) / (r**3-r**2+7*r+9),
        -(-2*r**13 + 13*r**12 - 48*r**11 + 85*r**10 - 83*r**9
          + 11*r**8 - 124*r**7 + 454*r**6 - 364*r**5 + 263*r**4
          + 84*r**3 + 189*r**2 + 25*r + 9) / d11,
    )

    d12 = r**2 - 4*r + 3
    c12 = (
        -(-r**4 + 11*r**3 - 26*r**2 + 16*r + 5) / d12,
        -(-2*r**5 + 19*r**4 - 40*r**3 + 13*r**2 + 5*r) / d12,
        -(-3*r**6 + 12*r**5 - 15*r**4 - 25*r**3
          + 62*r**2 + 36*r + 5) / (r-3),
        -(-r**9 + 8*r**8 - 26*r**7 + 43*r**6 - 40*r**5 + 17*r**4
          + 23*r**3 - 27*r**2 - 19*r - 3) / d12,
    )

    d13 = 2*(r**3 - 3*r**2 - 5*r + 7)
    c13 = (
        -(r**7 - 8*r**6 + 25*r**5 - 20*r**4 - 37*r**3 + 75*r + 28) / d13,
        (r+1)*(r**8 - 11*r**7 + 47*r**6 - 83*r**5 + 17*r**4
               + 71*r**3 + 45*r**2 - 169*r + 210) / d13,
        (r**2-1)*(3*r**8 - 29*r**7 + 115*r**6 - 225*r**5 + 181*r**4
                  + 105*r**3 - 255*r**2 - 235*r + 84) / d13,
        -(r**10 - 17*r**9 + 96*r**8 - 212*r**7 + 54*r**6 + 594*r**5
          - 796*r**4 - 36*r**3 + 721*r**2 - 329*r - 588) / d13,
    )

    data = (
        ("Conjecture 11", [1, -(r+1), -1, r], c11, d11,
         -4757714578613175915968435866673495590560),
        ("Conjecture 12", [1, -(r+1), r-1, 0], c12, d12,
         9865855835586623025289744),
        ("Conjecture 13", [1, -(r+1), r-2, r], c13, d13,
         -1346418219599972300425255472),
    )

    for name, P, coeffs, denominator, rank_witness in data:
        h = hankels(moments(P, sp.Integer(1), r, 23), 11)
        check_somos8(h, coeffs)
        matrix = coefficient_matrix(h)
        rhs = sp.Matrix([h[n] * h[n-8] for n in range(8, 12)])
        det_matrix = sp.factor(matrix.det(method="domain-ge"))
        quotient_num, quotient_den = sp.fraction(sp.cancel(det_matrix / denominator))
        assert quotient_den == 1
        quotient = sp.Poly(quotient_num, r)
        assert sp.gcd(quotient, sp.Poly(denominator, r)).degree() == 0
        for column, coefficient in enumerate(coeffs):
            cramer = matrix.copy()
            cramer[:, column] = rhs
            cramer_det = cramer.det(method="domain-ge")
            assert sp.cancel(cramer_det - det_matrix * coefficient) == 0
        h_at_2 = [sp.Integer(z.subs(r, 2)) for z in h]
        assert coefficient_matrix(h_at_2).det(method="domain-ge") == rank_witness
        endpoint = sp.Poly(sp.cancel(denominator * coeffs[0]), r)
        assert sp.gcd(sp.Poly(denominator, r), endpoint).degree() == 0
        cramer_first = matrix.copy()
        cramer_first[:, 0] = rhs
        assert sp.factor(
            cramer_first.det(method="domain-ge")
            - quotient.as_expr() * endpoint.as_expr()
        ) == 0
        print(
            f"{name}: identities, Cramer quotients, generic uniqueness, "
            "and exceptional-fiber certificates verified"
        )


def barry_conjecture_7():
    r, s, t = sp.symbols("r s t")
    P = [1, -2, -(r+1), -s]
    h = hankels(moments(P, t, sp.Integer(1), 15), 7)
    alpha = t**2 * (r+2)**2
    gamma = t**3 * (
        r**3*t + r**2*(s+7*t)
        + 2*r*(s**2 + 2*(t+1)*s + t*(t+8))
        + s**3 + s**2*(3*t+4) + s*(t+2)*(3*t+2)
        + t*(t**2+4*t+12)
    )
    for n in (6, 7):
        assert sp.factor(
            h[n]*h[n-6] - alpha*h[n-1]*h[n-5] - gamma*h[n-3]**2
        ) == 0
    print("Conjecture 7: exact initial identities verified")


def examples_and_errata():
    # Example 15.
    h15 = hankels(moments([1, -3, 1, 1], 1, 2, 23), 11)
    assert h15 == [1, 1, -1, -4, -8, -13, 57, 241, 1093, 792, -30661, -246182]
    c15 = (sp.Rational(1, 2), sp.Rational(-5, 2),
           sp.Rational(11, 2), sp.Rational(17, 2))
    check_somos8(h15, c15)
    assert coefficient_matrix(h15).det(method="domain-ge") == 2489567175410

    # Corrected Example 8: (r,s,t)=(-3,-2,1), not (-2,-2,1).
    mu8 = moments([1, -2, 2, 2], 1, 1, 13)
    assert mu8[:13] == [1, 1, 0, -3, -7, -9, -5, 8, 32, 71, 129, 187, 153]

    # Corrected Example 10 is (-1)^n times Conjecture 7 at (0,1,2).
    mu10 = moments([1, -2, -1, -1], 2, 1, 13)
    expected = [1, -1, 3, -10, 26, -75, 224, -659, 1979,
                -6025, 18452, -57028, 177625]
    assert [(-1)**n * mu10[n] for n in range(13)] == expected
    print("Examples 8/10 corrections and Example 15 identity verified")


def chang_hu_initial_certificate():
    a, b, c = sp.symbols("a b c")
    m = [sp.Integer(1)]
    for n in range(1, 25):
        conv = sum(m[2*j] * m[n-2-2*j] for j in range((n-2)//2 + 1)) if n >= 2 else 0
        m.append(sp.expand(a*m[n-1] + (b*m[n-2] if n >= 2 else 0) + c*conv))
    h = [sp.Integer(1)]
    for n in range(1, 9):
        mat = sp.Matrix([[m[2*i+j] for j in range(n)] for i in range(n)])
        h.append(sp.factor(mat.det(method="domain-ge")))
    p = a*c**2*(a**2+b+c)**2
    q = -a**2*c**2*((b+c)**3 + a**2*c*(a**2+b+c))
    for n in range(4, 9):
        assert sp.factor(h[n]*h[n-4] - p*h[n-1]*h[n-3] - q*h[n-2]**2) == 0
    print("Chang--Hu: first five instances of the stronger Somos-4 identity verified")


if __name__ == "__main__":
    chang_hu_initial_certificate()
    barry_conjecture_7()
    barry_conjectures_11_to_13()
    examples_and_errata()
    print("All exact certificates passed.")
