from mace.calculators import mace
from ase.io import read
from ase.build import molecule
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.optimize import BFGS
from ase.io.trajectory import Trajectory
from ase import units
import numpy as np
import os

atoms = read("02_uio-66.cif")

co2 = molecule('CO2')

com = atoms.get_cell().sum(axis=0) / 2
offset = np.random.uniform(-0.5, 0.5, size=3)
co2.translate(com + offset)

system = atoms + co2
system.wrap()

model = "models/mace_agnesi_small.model"
default_dtype = "float32"
device = "cuda"
calc = mace.MACECalculator(model_paths=model, default_dtype=default_dtype ,device=device)

system.set_calculator(calc)

# Test energy/forces
e = system.get_potential_energy()
f = system.get_forces()
print("Initial energy (eV):", e)
print("Max force (eV/Å):", abs(f).max())

opt = BFGS(system, logfile=None)
opt.run(fmax=0.05)  # relax until max force <= 0.05 eV/Å

# 1) Initialize velocities and remove net momentum
MaxwellBoltzmannDistribution(system, temperature_K=300.0, rng=np.random.default_rng(42))
Stationary(system)

dt = 0.25 * units.fs
fric = 0.05
dyn = Langevin(system, dt, temperature_K=300.0, friction=fric)

report_interval = 100

steps = 100000

# 3) Clean observers, attach logger once
dyn.observers = []
os.makedirs("traj", exist_ok=True)
with Trajectory("traj/03_equilibration.traj", "w", system) as traj:
    dyn.attach(traj.write, interval=report_interval)

    def print_status():
        ekin = system.get_kinetic_energy()
        epot = system.get_potential_energy()
        N = len(system)
        T = 2.0 * ekin / (3.0 * N * units.kB)  # adjust ndof if you have constraints
        print(f"step={dyn.get_number_of_steps():4d}  Epot={epot:10.3f} eV  "
              f"Ekin={ekin:10.3f} eV  T={T:7.1f} K")
    dyn.attach(print_status, interval=report_interval)

    dyn.run(steps)