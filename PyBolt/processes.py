import numpy as np
from abc import ABC, abstractmethod
from scipy.interpolate import interp1d, CubicSpline
from scipy.special import kn  # Bessel function
from scipy.integrate import quad, fixed_quad
from .cosmology import Y_x_eq, Y_x_eq_massive, h_s, nmeq, npheq
from .constants import e_g
from tqdm import tqdm

# Default number of x nodes in Process.tabulate. Over x in [1e-3, 30] this matches
# adaptive quadrature to ~1e-7, the accuracy of the quadrature itself.
N_TABULATION_POINTS = 200

# Extra range in log x at each end of the table; LSODA steps slightly past the
# integration interval.
TABULATION_LOG_MARGIN = 0.02


# Classes of processes
class Process(ABC):
    """
    Base class for all the processes.
    """

    def __init__(self, m1: float, g_1: float, coupling: float, simplify: bool = False, **kwargs):
        self._m1 = m1
        self._g_1 = g_1
        self._coupling = coupling
        self.simplify = simplify
        self._kernel_table = None
        self._kernel_table_q = None

    @abstractmethod
    def rate(self, x: float, Y: float) -> float:
        """
        The rate that enters the number density Boltzmann equation [GeV**4]

        Parameters
        ----------
        x: float
            The inverse unit of temperature x = m/T
        Y: float
            The comoving abundance of particle X at the given value of x
        """
        
        pass
    
    @abstractmethod
    def collisionTerm(self, x: float, q: np.array, f: np.array, feq: np.array) -> np.array:
        """
        The collision term of particle X that enters the phase-space Boltzmann equation

        Parameters
        ----------
        x: float
            The inverse unit of temperature x = m/T [1]
        q: np.array
            The vector of momenta of particle X divided by T [1]
        f: np.array
            The distribution function of X particle (vector should be the same size as q) [1]
        feq: np.array
            The corresponding equilibrium distribution of X particle [1]
        """

        pass

    # -- the f-independent kernel ---------------------------------------------
    #
    # The ek integral runs over the energy of an equilibrium bath particle, so it
    # depends only on x and q, not on f or the coupling. It can therefore be
    # tabulated once and reused for every f_a.

    def _lower_limit(self, x: float, q: np.array) -> np.array:
        """Lower limit of the ek integral, per q. [1]"""

        raise NotImplementedError

    def _reduced_integrand(self, t: float, q_i: float, x2: float, a: float) -> float:
        """The ek integrand at ``ek = a + t``, with ``exp(-a)`` divided out.

        ``exp(-a)`` underflows at large x and small q; ``_kernel`` multiplies it
        back in analytically.
        """

        raise NotImplementedError

    def _kernel_prefactor(self, x: float, q: np.array) -> np.array:
        """Everything multiplying the ek integral in the collision term."""

        raise NotImplementedError

    def _reduced_integral(self, x: float, q: np.array) -> np.array:
        """The reduced ek integral evaluated by adaptive quadrature, per q."""

        x2 = x**2
        a = np.broadcast_to(self._lower_limit(x, q), np.shape(q))

        return np.array([
            quad(self._reduced_integrand, 0.0, np.inf, args=(q_i, x2, a_i))[0]
            for q_i, a_i in zip(q, a)
        ])

    def _reduced_integral_at(self, x: float, q: np.array) -> np.array:
        """The reduced integral, from the table when one has been built."""

        if self._kernel_table is None:
            return self._reduced_integral(x, q)

        if not np.array_equal(q, self._kernel_table_q):
            raise ValueError(
                "The kernel table was built for a different q grid. Call tabulate() "
                "again with the grid the solver will use, or drop the table."
            )

        return np.exp(self._kernel_table(np.log(x)))

    def _kernel(self, x: float, q: np.array) -> np.array:
        """``A(x, q)``: the collision term with the ``f``-dependent factor removed."""

        reduced = self._reduced_integral_at(x, q)
        damping = np.exp(-np.broadcast_to(self._lower_limit(x, q), np.shape(q)))

        return self._kernel_prefactor(x, q) * reduced * damping

    def tabulate(self, x: np.array, q: np.array,
                 n_points: int = N_TABULATION_POINTS) -> "Process":
        """Precompute the reduced integral on a log-spaced x grid and spline it.

        The spline is cubic in (log x, log integral).

        Parameters
        ----------
        x: np.array
            The x grid the solve will run over. Only its endpoints are used.
        q: np.array
            The momentum grid. The table is tied to it, and ``collisionTerm`` will
            refuse a different one.
        n_points: int
            Number of x nodes.

        Returns ``self``, so the call can be chained onto the constructor.
        """

        log_x = np.linspace(
            np.log(x[0]) - TABULATION_LOG_MARGIN,
            np.log(x[-1]) + TABULATION_LOG_MARGIN,
            n_points,
        )
        reduced = np.array([self._reduced_integral(xi, q) for xi in np.exp(log_x)])

        self._kernel_table = CubicSpline(log_x, np.log(reduced), axis=0)
        self._kernel_table_q = np.array(q, copy=True)

        return self

    def adopt_table(self, other: "Process") -> "Process":
        """Reuse another process's kernel table.

        Both processes must be the same class with the same ``simplify`` flag; the
        coupling may differ.

        Returns ``self``, so the call can be chained onto the constructor.
        """

        if type(self) is not type(other):
            raise TypeError(
                f"Cannot adopt a {type(other).__name__} table into a "
                f"{type(self).__name__}: the integrands differ."
            )
        if bool(self.simplify) != bool(other.simplify):
            raise ValueError(
                "Cannot adopt a table built with a different 'simplify' setting: "
                "it selects a different bath distribution under the integral."
            )
        if other._kernel_table is None:
            raise ValueError("The donor process has no table; call tabulate() first.")

        self._kernel_table = other._kernel_table
        self._kernel_table_q = other._kernel_table_q

        return self


class DecayToX(Process): # 1 -> 2 decay where X is massless
    """
    A class for the 2-body decay processes of a fermion particle into another massive fermion and a massless scalar DM
    collisionTerm is used for the fBE solution, while the rate is used for the nBE solution.
    """
    def __init__(self, m1, g_1, coupling, 
                 m2: float,
                 Msquared_stripped: float
                ):
        super().__init__(m1 = m1, g_1 = g_1, coupling = coupling)
        self._Msquared = coupling**2*Msquared_stripped
        self._mu = m2 / m1  # ratio of masses [1]
        self._Gamma = self._Msquared*(1 - self._mu**2)/self._m1/16/np.pi # GeV
        

    def rate(self, x, Y):
        
        return (
            8
            * self._Gamma
            * self._m1**3
            * (kn(1, x) / x)
            * (1 - Y / Y_x_eq(self._m1 / x))
            / (2 * np.pi) ** 2
        )

    def collisionTerm(self, x, q, f, feq):

        # The following structure comes from the integration limits
        Elim1 = np.max(
            [
                x * np.ones(q.shape),
                x * self._mu + q,
                x**2
                * (1 + 4 * q**2 / (x**2 * (1 - self._mu**2)) - self._mu**2)
                / 4
                / q,
            ],
            axis=0,
        )  # [1]
        Elim2 = np.max(
            [x - q, x * self._mu * np.ones(q.shape), x**2 / 4 / q], axis=0
        )  # [1]

        return (
            (2 * self._g_1 * self._Msquared * x / self._m1 / 8 / np.pi / q**3)
            * (np.log((1 + np.exp(-Elim1)) / (1 + np.exp(-Elim2))))
            * (f - feq)
        )


def lam_f(x: float, y: float, z: float) -> float:
        """ Auxiliary function for the calculation of the velocity-averaged cross section"""

        return (x - (y+z)**2)*(x - (y-z)**2) # [GeV**4]


class LeptonAnnihilationToAxionMB(Process): # l_i + l_j -> X + gamma_k 
    """
    Annihilation of two leptons into an axion and a photon. Axion (massless) is the particle of interest in this reaction. The leptons are Maxwell-Boltzmann. We also assume that the axion number of dof is 1.
    """

    def __init__(self, m1, g_1, coupling, simplify: bool = False):
        """
        Parameters
        ----------
        m1 : float
            The mass of the lepton
        g_1 : float
            The number of degrees of freedom of the annihilated particles.
        coupling : float
            The coupling constant for this process (C_l/f_a)
        simplify : bool
            Affects ``collisionTerm`` only.
            False (default): Bose-Einstein photon, 1/(1 - exp(-ek)), and the
            inverse process through (1 - f/f_eq).
            True: Maxwell-Boltzmann photon and no inverse process (f << f_eq).
        """
        super().__init__(m1 = m1, g_1 = g_1, coupling = coupling, simplify = simplify)

    def sigma_ann(self, s: float) -> float:
        """ Annihilation cross section """

        return (self._coupling*e_g*self._m1)**2*np.atanh(np.sqrt(1-4*self._m1**2/s))/(s-4*self._m1**2)/4/np.pi # [GeV**(-2)]

    def sigmaV_ann(self, x: float) -> float:
        """ Velocity-averaged annihilation cross section (for MB distributions)"""

        integral, _ = quad(lambda s: lam_f(s,self._m1,self._m1)*self.sigma_ann(s)*kn(1,np.sqrt(s)*x/self._m1)/np.sqrt(s),4*self._m1**2,np.inf)

        return x*integral/kn(2,x)**2/8/self._m1**5 # [GeV**-2]
        

    def rate(self, x, Y):

        return nmeq(x, self._g_1, self._m1)**2*self.sigmaV_ann(x)*(1 - Y/Y_x_eq(self._m1 / x)) # [GeV**4]
        

    def _lower_limit(self, x, q):
        return x**2 / q

    def _reduced_integrand(self, t, q_i, x2, a):
        ek = a + t
        w = np.sqrt(1 - x2/(ek*q_i))
        damped = ((2*ek*q_i - x2)*np.arctanh(w) - ek*q_i*w) * np.exp(-t)

        if self.simplify:
            return damped

        # Bose-Einstein factor; expm1 stays accurate as ek -> 0.
        return damped / -np.expm1(-ek)

    def _kernel_prefactor(self, x, q):
        prefactor = self._g_1**2 * (e_g * self._coupling)**2 * self._m1**3

        return prefactor * np.exp(-q) / (q * x * 2 * (2*np.pi)**3)

    def collisionTerm(self, x, q, f, feq):
        C_func = self._kernel(x, q)

        if self.simplify:
            return C_func
        else:
            return (1 - f/feq) * C_func


class PrimakoffScatteringMB(Process): # l_i + X -> l_j + gamma_k 
    """
    Primakoff scattering of axion on a lepton. Axion (massless) is the particle of interest in this reaction. The leptons are Maxwell-Boltzmann, except as described under ``simplify``. We also assume that the axion number of dof is 1.
    """

    def __init__(self, m1, g_1, coupling, simplify: bool = False):
        """
        Parameters
        ----------
        m1 : float
            The mass of the lepton
        g_1 : float
            The number of degrees of freedom of the lepton.
        coupling : float
            The coupling constant for this process (C_l/f_a)
        simplify : bool
            Affects ``collisionTerm`` only.
            False (default): Fermi-Dirac lepton in the ek integral,
            1/(exp(ek) + 1), and the inverse process through (1 - f/f_eq).
            True: Maxwell-Boltzmann lepton and no inverse process (f << f_eq).
        """
        super().__init__(m1 = m1, g_1 = g_1, coupling = coupling, simplify = simplify)

    def sigma_prim(self, s: float) -> float:
        """ Primakoff scattering cross section """
        
        return (
            (e_g*self._coupling*self._m1)**2
            * (2*s**2*np.log(s/self._m1**2) - 3*s**2 + 4*self._m1**2*s - self._m1**4)
            / s**2/(s-self._m1**2)/32/np.pi
        )# [GeV**-2]

    def sigmaV_prim(self, x: float) -> float:
        """ Velocity-averaged cross section of the Primakoff scattering (for MB distributions)"""

        integral, _ = quad(lambda s: lam_f(s,self._m1,0.0)*self.sigma_prim(s)*kn(1,np.sqrt(s)*x/self._m1)/np.sqrt(s),self._m1**2,np.inf)
        
        return x**3*integral/kn(2,x)/16/self._m1**5 # [GeV**-2]


    def rate(self, x, Y):

        return 2*nmeq(x, self._g_1, self._m1)*npheq(self._m1/x)*self.sigmaV_prim(x)*(1 - Y/Y_x_eq(self._m1 / x)) # [GeV**4]
        

    def _lower_limit(self, x, q):
        return x

    def _reduced_integrand(self, t, q_i, x2, a):
        ek = a + t

        def s_integral(s):
            return 2*s*np.log(s/x2) + 4*x2*np.log(s) - 5*s + x2**2/s

        root = 2*q_i*np.sqrt(ek**2 - x2)
        base = x2 + 2*ek*q_i
        damped = (s_integral(base + root) - s_integral(base - root)) * np.exp(-t)

        if self.simplify:
            return damped

        # Fermi-Dirac factor.
        return damped / (1.0 + np.exp(-ek))

    def _kernel_prefactor(self, x, q):
        prefactor = 2 * self._g_1**2 * (e_g * self._coupling)**2 * self._m1**3

        # The trailing /2: the collision term is half the kernel.
        return prefactor * np.exp(-q) / (q * x * 32 * (2*np.pi)**3) / 2

    def collisionTerm(self, x, q, f, feq):
        C_half = self._kernel(x, q)

        if self.simplify:
            return C_half
        else:
            return (1 - f/feq) * C_half
