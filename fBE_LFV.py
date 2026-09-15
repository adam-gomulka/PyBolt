#!/usr/bin/env python3
import numpy as np
import argparse
import sys
import tqdm

from PyBolt import boltzmann_solver as bz
from PyBolt.processes import DecayToX

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Run Boltzmann solver for specific Reheating Ratio.')
    parser.add_argument('--lepton', type=str, required=True, 
                        help='Lepton type (e.g., muon, electron)')
    parser.add_argument('--ratio', type=float, required=True, 
                        help='T_reh_to_mass_ratio: Ratio of Reheating T to Lepton mass')
    parser.add_argument('--f_min', type=float, default=1e7,
                        help='Minimum axion decay constant f_a in GeV (default: 1e7)')
    parser.add_argument('--f_max', type=float, default=1e9,
                        help='Maximum axion decay constant f_a in GeV (default: 1e9)')
    args = parser.parse_args()

    # --- Constants and Parameters ---
    if args.lepton.lower() == 'taon' or args.lepton.lower() == 'tau':
        m_parent = 1.77686  # GeV
        m_daughter = 0.10565 # GeV
    elif args.lepton.lower() == 'muon' or args.lepton.lower() == 'mu':
        m_parent = 0.10565  # GeV
        m_daughter = 0.000511 # GeV
    else:
        raise ValueError("Unsupported lepton type. Please choose 'tau' or 'mu'.")

    mDM = 1.0e-10   # GeV, not really relevant
    g_x = 1.0 # we default to g_x = 1 for ALPs
    g_lepton = 2.0

    mu = m_daughter/m_parent

    f_vals = np.logspace(np.log10(args.f_min), np.log10(args.f_max), 100)
    coupling_vals = 1 / f_vals

    T_reh_to_mass_ratio = args.ratio
    T_reh = T_reh_to_mass_ratio * m_parent

    print(f"Starting run for T_reh/m_{args.lepton.lower} = {T_reh_to_mass_ratio} (T_reh = {T_reh:.4e} GeV)")

    if T_reh < 1e-3:
        print("Warning: reheating temperature below 1 MeV is unreliable in this setup.")

    # --- Setup Models ---
    AxionModel = {}
    N_x = 1000
    N_q = 400

    xstart = 1e-2
    xfin = 20.0

    qin = 0.03
    qend = 15.0

    x_lin = np.linspace(xstart, xfin, N_x)
    q_lin = np.linspace(qin, qend, N_q)

    # --- Solver Options ---
    solver_options = {
        'method': 'RK45',
        'atol': 1e-6,
        'rtol': 1e-3
    }

    # --- Output Setup ---
    filename = f"distributions/fa_{args.lepton.lower()}_dec_{T_reh_to_mass_ratio}_{args.f_min}_{args.f_max}.dat"

    print(f"Writing results incrementally to {filename}...")

    with open(filename, 'w') as file:
        file.write("# f_a, f(q)\n")
        file.flush()

        # --- Execution Loop ---
        for inv_f in tqdm.tqdm(coupling_vals, desc="Solving fBE"):
            AxionModel[inv_f] = bz.Model(m_parent, mDM, g_x, 'b')

            AxionModel[inv_f].changeGrid(x_lin, q_lin)
            Msquared = m_parent**4*(1-mu**2)**2/4 # GeV^2

            LeptonDecay = DecayToX(m_parent, g_lepton, inv_f, m_daughter, Msquared)

            AxionModel[inv_f].addCollisionTerm(LeptonDecay.collisionTerm)

            f0 = np.zeros(N_q)
            try:
                AxionModel[inv_f].solve_fBE(f0, solver_options)
                f_final = AxionModel[inv_f]._f[-1, :]

                f_a = 1/inv_f
                data_str = ",".join(map(lambda x: f"{x:.5e}", f_final))
                file.write(f"{f_a:.5e},{data_str}\n")
                file.flush()
                
            except Exception as e:
                print(f"Error solving for f={1/inv_f:.2e}: {e}", file=sys.stderr)
                sys.stderr.flush()

if __name__ == "__main__":
    main()
