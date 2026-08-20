from mace.calculators import mace
from ase.io import read
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory
from ase import units
import os

system = read("traj/03_equilibration.traj", index = "-1")

model = "models/mace_agnesi_small.model"
default_dtype = "float32"
device = "cuda"
try:
    calc = mace.MACECalculator(model_paths=model, default_dtype=default_dtype ,device=device)
except:
    calc = mace.MACECalculator(model_paths=model, default_dtype=default_dtype ,device="cpu")

system.set_calculator(calc)

# Test energy/forces
e = system.get_potential_energy()
f = system.get_forces()
print("Initial energy (eV):", e)
print("Max force (eV/Å):", abs(f).max())

dt = 0.5 * units.fs
fric = 0.05
dyn = Langevin(system, dt, temperature_K=300.0, friction=fric)

report_interval = 100

steps = 100000

# 3) Clean observers, attach logger once
dyn.observers = []
os.makedirs("traj", exist_ok=True)
with Trajectory("traj/04_nvt.traj", "w", system) as traj:
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