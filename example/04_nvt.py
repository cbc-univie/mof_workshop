from mace.calculators import mace
from ase.io import read
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory
from ase import units
import os

atoms = read("traj/03_equilibration.traj", index = "-1")

model = "../../models/mace-mh-1.model"
default_dtype = "float32"
device = "cuda"
calc = mace.MACECalculator(model_paths=model, default_dtype=default_dtype ,device=device, head="omat_pbe")

atoms.calc = calc

dt = 0.5 * units.fs
fric = 0.05
dyn = Langevin(atoms, dt, temperature_K=300.0, friction=fric)

report_interval = 100

steps = 500000

dyn.observers = []
os.makedirs("traj", exist_ok=True)
with Trajectory("traj/04_nvt.traj", "w", atoms) as traj:
    dyn.attach(traj.write, interval=report_interval)

    def print_status():
        ekin = atoms.get_kinetic_energy()
        epot = atoms.get_potential_energy()
        N = len(atoms)
        T = 2.0 * ekin / (3.0 * N * units.kB)
        print(f"step={dyn.get_number_of_steps():4d}  Epot={epot:10.3f} eV  "
              f"Ekin={ekin:10.3f} eV  T={T:7.1f} K")
    dyn.attach(print_status, interval=report_interval)

    dyn.run(steps)