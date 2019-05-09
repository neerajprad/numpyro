from __future__ import print_function, division

import jax.numpy as np
import jax
from jax import grad, hessian, jacfwd, jit


#@jax.partial(jit, static_argnums=(0,))
def dtheta_domega(loss, theta, omega):
    H = hessian(lambda _theta: np.sum(omega * loss(_theta)))(theta)
    Hinv = np.linalg.inv(H)
    J = np.transpose(jacfwd(loss)(theta))
    return -np.matmul(Hinv, J)

def dtheta(loss, theta, omega, domega):
    return np.matmul(dtheta_domega(loss, theta, omega), domega)

def dtheta2(loss, theta, omega, domega):
    return np.dot(jacfwd(lambda _omega: dtheta(loss,
                                               theta + np.matmul(dtheta_domega(loss, theta, omega), _omega),
                                               _omega, domega))(omega), domega) / 2.0

def dtheta3(loss, theta, omega, domega):
    return np.dot(jacfwd(lambda _omega: dtheta2(loss,
                                                theta + np.matmul(dtheta_domega(loss, theta, omega), _omega),
                                                _omega, domega))(omega), domega) / 3.0

def dtheta4(loss, theta, omega, domega):
    return np.dot(jacfwd(lambda _omega: dtheta3(loss,
                                                theta + np.matmul(dtheta_domega(loss, theta, omega), _omega),
                                                _omega, domega))(omega), domega) / 4.0

def perturb_theta(loss, theta, omega, domega, order=1):
    assert order in [1, 2, 3, 4]

    _theta = theta + dtheta(loss, theta, omega, domega)
    if order > 1:
        _theta += dtheta2(loss, theta, omega, domega)
    if order > 2:
        _theta += dtheta3(loss, theta, omega, domega)
    if order > 3:
        _theta += dtheta4(loss, theta, omega, domega)
    return _theta


"""
some tests
"""
if __name__ == '__main__':
    import numpy as vnp
    import warnings
    warnings.simplefilter("ignore")

    def do_test1(n_tests=3):

        X = np.array([1.0, 1.2], dtype=np.float64)

        def theta_omega_exact(omega):
            numerator = omega[0] + 2.0 * omega[1] * X[1] - np.sqrt(omega[0] * omega[0] + 4.0 * omega[0] * omega[1] * (X[1] - X[0]))
            return 0.5 * numerator / omega[1]

        def loss(theta):
            loss1 = np.power(X[0] - theta[0], 2.0) / 2.0
            loss2 = np.power(X[1] - theta[0], 3.0) / 3.0
            return np.array([loss1, loss2])

        def summed_loss(theta):
            return np.sum(omega * loss(theta))

        for trial in range(n_tests):
            print("\n[test 1: trial %d]  theta_error  approx_error" % trial)

            omega = np.array([1.0, 1.5]) + 0.1 * vnp.random.randn(2)

            theta = np.array([theta_omega_exact(omega)])
            assert grad(summed_loss)(theta) < 1.0e-5

            domega = 0.35 * vnp.random.randn(2)

            true_theta = theta_omega_exact(omega + domega)

            _dtheta = dtheta(loss, theta, omega, domega)
            _dtheta_exact = np.dot(grad(theta_omega_exact)(omega), domega)
            _dtheta_error = np.abs(_dtheta - _dtheta_exact)
            theta_1 = theta + _dtheta
            theta_1_error = np.abs(theta_1 - true_theta)
            print("1st order errors:  %.2e     %.2e" % (theta_1_error, _dtheta_error))
            assert theta_1_error < 1.0e-1
            assert _dtheta_error < 1.0e-7

            _dtheta2 = dtheta2(loss, theta, omega, domega)
            _dtheta2_exact = 0.5 * np.dot(grad(lambda om: np.dot(grad(theta_omega_exact)(om), domega))(omega), domega)
            _dtheta2_error = np.abs(_dtheta2 - _dtheta2_exact)
            theta_2 = theta + _dtheta + _dtheta2
            theta_2_error = np.abs(theta_2 - true_theta)
            print("2nd order errors:  %.2e     %.2e" % (theta_2_error, _dtheta2_error))
            assert theta_2_error < 1.0e-2
            assert _dtheta2_error < 1.0e-7

            _dtheta3 = dtheta3(loss, theta, omega, domega)
            _dtheta3_exact = np.dot(grad(lambda _om: np.dot(grad(lambda om: np.dot(grad(theta_omega_exact)(om), domega))(_om), domega))(omega), domega) / 6.0
            _dtheta3_error = np.abs(_dtheta3 - _dtheta3_exact)
            theta_3 = theta + _dtheta + _dtheta2 + _dtheta3
            theta_3_error = np.abs(theta_3 - true_theta)
            print("3rd order errors:  %.2e     %.2e" % (theta_3_error, _dtheta3_error))
            assert theta_3_error < 5.0e-3
            assert _dtheta3_error < 1.0e-7

            _dtheta4 = dtheta4(loss, theta, omega, domega)
            _dtheta4_exact = np.dot(grad(lambda __om: np.dot(grad(lambda _om: np.dot(grad(lambda om: np.dot(grad(theta_omega_exact)(om), domega))(_om), domega))(__om), domega))(omega), domega) / 24.0
            _dtheta4_error = np.abs(_dtheta4 - _dtheta4_exact)
            theta_4 = theta + _dtheta + _dtheta2 + _dtheta3 + _dtheta4
            theta_4_error = np.abs(theta_4 - true_theta)
            print("4th order errors:  %.2e     %.2e" % (theta_4_error, _dtheta4_error))
            assert theta_4_error < 2.0e-3
            assert _dtheta4_error < 1.0e-7

            assert theta_2_error < 0.8 * theta_1_error
            assert theta_3_error < 0.8 * theta_2_error
            assert theta_4_error < 0.8 * theta_3_error


    def do_test2(alphas=[0.5, 2.0, 3.0], D=3):

        X = vnp.array(vnp.random.rand(D), dtype=vnp.float64)

        for trial, alpha in enumerate(alphas):
            print("\n[test 2: trial %d]  theta_error  approx_error" % trial)

            def theta_omega_exact(omega):
                return np.power(np.sum(omega * X) / np.sum(omega), 1.0 / alpha)

            def loss(theta):
                return 0.5 * np.power(X - np.power(theta[0], alpha), 2.0)

            def summed_loss(theta):
                return np.sum(omega * loss(theta))

            omega = 1.0 + 0.2 * vnp.array(vnp.random.rand(D), dtype=vnp.float64)

            theta = np.array([theta_omega_exact(omega)])
            assert grad(summed_loss)(theta) < 1.0e-5

            domega = 0.35 * vnp.array(vnp.random.randn(D), dtype=vnp.float64)

            true_theta = theta_omega_exact(omega + domega)

            _dtheta = dtheta(loss, theta, omega, domega)
            _dtheta_exact = np.dot(grad(theta_omega_exact)(omega), domega)
            _dtheta_error = np.abs(_dtheta - _dtheta_exact)
            theta_1 = theta + _dtheta
            theta_1_error = np.abs(theta_1 - true_theta)
            print("1st order errors:  %.2e     %.2e" % (theta_1_error, _dtheta_error))
            assert theta_1_error < 1.0e-1
            assert _dtheta_error < 1.0e-7

            _dtheta2 = dtheta2(loss, theta, omega, domega)
            _dtheta2_exact = 0.5 * np.dot(grad(lambda om: np.dot(grad(theta_omega_exact)(om), domega))(omega), domega)
            _dtheta2_error = np.abs(_dtheta2 - _dtheta2_exact)
            theta_2 = theta + _dtheta + _dtheta2
            theta_2_error = np.abs(theta_2 - true_theta)
            print("2nd order errors:  %.2e     %.2e" % (theta_2_error, _dtheta2_error))
            assert theta_2_error < 1.0e-2
            assert _dtheta2_error < 1.0e-7

            _dtheta3 = dtheta3(loss, theta, omega, domega)
            _dtheta3_exact = np.dot(grad(lambda _om: np.dot(grad(lambda om: np.dot(grad(theta_omega_exact)(om), domega))(_om), domega))(omega), domega) / 6.0
            _dtheta3_error = np.abs(_dtheta3 - _dtheta3_exact)
            theta_3 = theta + _dtheta + _dtheta2 + _dtheta3
            theta_3_error = np.abs(theta_3 - true_theta)
            print("3rd order errors:  %.2e     %.2e" % (theta_3_error, _dtheta3_error))
            assert theta_3_error < 5.0e-3
            assert _dtheta3_error < 1.0e-7

            _dtheta4 = dtheta4(loss, theta, omega, domega)
            _dtheta4_exact = np.dot(grad(lambda __om: np.dot(grad(lambda _om: np.dot(grad(lambda om: np.dot(grad(theta_omega_exact)(om), domega))(_om), domega))(__om), domega))(omega), domega) / 24.0
            _dtheta4_error = np.abs(_dtheta4 - _dtheta4_exact)
            theta_4 = theta + _dtheta + _dtheta2 + _dtheta3 + _dtheta4
            theta_4_error = np.abs(theta_4 - true_theta)
            print("4th order errors:  %.2e     %.2e" % (theta_4_error, _dtheta4_error))
            assert theta_4_error < 2.0e-3
            assert _dtheta4_error < 1.0e-7

            assert theta_2_error < 0.8 * theta_1_error
            assert theta_3_error < 0.8 * theta_2_error
            assert theta_4_error < 0.8 * theta_3_error

    vnp.random.seed(0)
    do_test1()
    do_test2()

    print("\nYay! All tests passed")
