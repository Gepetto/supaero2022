import unittest

import numpy as np


class TrajRef:
    def __init__(self, q0, omega, amplitude):
        self.q0 = q0.copy()
        self.omega = omega
        self.amplitude = amplitude
        self.q = self.q0.copy()
        self.vq = self.q0 * 0
        self.aq = self.q0 * 0

    def position(self, t):
        """Compute a reference position for time <t>."""
        self.q.flat[:] = self.q0
        self.q.flat[:] += self.amplitude * np.sin(self.omega * t)
        return self.q

    def velocity(self, t):
        """Compute and return the reference velocity at time <t>."""
        self.vq.flat[:] = self.omega * self.amplitude * np.cos(self.omega * t)
        return self.vq

    def acceleration(self, t):
        """Compute and return the reference acceleration at time <t>."""
        self.aq.flat[:] = -self.omega**2 * self.amplitude * np.sin(self.omega * t)
        return self.aq

    def __call__(self, t):
        return self.position(t)

    def copy(self):
        return self.q.copy()


### TEST ZONE ############################################################
### This last part is to automatically validate the versions of this example.
class TrajRefTest(unittest.TestCase):
    def test_logs(self):
        print(self.__class__.__name__)
        # %jupyter_snippet main
        qdes = TrajRef(
            np.array([0, 0, 0.0]), omega=np.array([1, 2, 3.0]), amplitude=1.5
        )
        t = 0.2
        print(qdes(t), qdes.velocity(t), qdes.acceleration(t))
        # %end_jupyter_snippet

        self.assertTrue(np.allclose(qdes(t), [0.298004, 0.58412751, 0.84696371]))
        self.assertTrue(
            np.allclose(qdes.velocity(t), [1.47009987, 2.76318298, 3.71401027])
        )
        self.assertTrue(
            np.allclose(qdes.acceleration(t), [-0.298004, -2.33651005, -7.62267339])
        )


if __name__ == "__main__":
    TrajRefTest().test_logs()
