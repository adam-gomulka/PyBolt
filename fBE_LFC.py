#!/usr/bin/env python3
import numpy as np
import argparse
import sys
import tqdm

from PyBolt import boltzmann_solver as bz
from PyBolt.processes import LeptonAnnihilationToAxionMB, PrimakoffScatteringMB

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Run Boltzmann solver for specific Reheating Ratio.')
    parser.add_argument('--lepton', type=str, required=True, 
                        help='Lepton type (e.g., muon, electron)')
    parser.add_argument('--ratio', type=float, required=True, 
                        help='T_reh_to_mass_ratio: Ratio of Reheating T to Lepton mass')
    parser.add_argument('--output', type=str, required=True, 
                        help='Name for the output file')
    parser.add_argument('--f_min', type=float, default=1e7,
                        help='Minimum axion decay constant f_a in GeV (default: 1e7)')
    parser.add_argument('--f_max', type=float, default=1e9,
                        help='Maximum axion decay constant f_a in GeV (default: 1e9)')
    parser.add_argument('--f_num', type=int, default=200, 
                        help='Length of axion decay const. grid (default: 200)')
    parser.add_argument('--simplify', action=argparse.BooleanOptionalAction, default=False, 
                        help='Simplify the collision terms by setting f/f_eq=1 and neglecting quantum corrections? (default=False)')

    args = parser.parse_args()

    # --- Constants and Parameters ---
    if args.lepton.lower() == 'taon' or args.lepton.lower() == 'tau':
        m_lepton = 1.77686  # GeV
        g_lepton = 2.0
    elif args.lepton.lower() == 'muon' or args.lepton.lower() == 'mu':
        m_lepton = 0.10565  # GeV
        g_lepton = 2.0
    elif args.lepton.lower() == 'electron' or args.lepton.lower() == 'e':
        m_lepton = 0.000511  # GeV
        g_lepton = 2.0
    else:
        raise ValueError("Unsupported lepton type. Please choose 'tau', 'muon' or 'electron'.")

    mDM = 1.0e-10   # GeV, not really relevant
    g_x = 1.0 # we default to g_x = 1 as for alps

    simplify = args.simplify	

    f_vals = np.logspace(np.log10(args.f_min), np.log10(args.f_max), args.f_num)
    yMDx_list = 1 / f_vals

    # Use the argument passed from the command line
    T_reh_to_mass_ratio = args.ratio
    T_reh = T_reh_to_mass_ratio * m_lepton
    
    print(f"Starting run for T_reh/m_{args.lepton.lower} = {T_reh_to_mass_ratio} (T_reh = {T_reh:.4e} GeV)")

    if T_reh < 1e-3:
        print("Warning: reheating temperature below 1 MeV is unreliable in this setup.")

    # --- Setup Models ---
    AxionModel = {}
    N_x = 500 
    N_q = 100 

    # Determine integration bounds based on T_reh
    xstart = (m_lepton / T_reh)
    xfin = 30.0 #if xstart < 1 else xstart * 20.0
    
    qin = 0.01
    qend = 15.0
    
    x_lin = np.linspace(xstart, xfin, N_x)
    q_lin = np.linspace(qin, qend, N_q) 
    
    # --- Output Setup ---
    filename = args.output
    
    print(f"Writing results incrementally to {filename}...")

    # Open the file ONCE before the loop starts
    with open(filename, 'w') as file:
        # Write header
        file.write("# f_a, f(q)\n")
        file.flush() # Ensure header is written immediately

        # --- Execution Loop ---
        for inv_f in tqdm.tqdm(yMDx_list, desc="Solving fBE"):

            solver_options = dict(
               method="LSODA",
               rtol=1e-6,
               atol=1e-24,
               lband=2,
               uband=2,
            )    


            AxionModel[inv_f] = bz.Model(m_lepton, mDM, g_x, 'b')
            
            AxionModel[inv_f].changeGrid(x_lin, q_lin)
            
            LeptonAnn = LeptonAnnihilationToAxionMB(m_lepton, g_lepton, inv_f, simplify)
            LeptonPrim = PrimakoffScatteringMB(m_lepton, g_lepton, inv_f, simplify)

            AxionModel[inv_f].addCollisionTerm(LeptonAnn.collisionTerm)
            AxionModel[inv_f].addCollisionTerm(LeptonPrim.collisionTerm)
            
            f0 = np.zeros(N_q) 
            try:
                # Solve the Boltzmann equation
                AxionModel[inv_f].solve_fBE(f0, solver_options)
                
                # Get the final distribution
                f_final = AxionModel[inv_f]._f[-1, :]
                
                # Prepare the string for writing
                f_a = 1/inv_f
                data_str = ",".join(map(lambda x: f"{x:.5e}", f_final))
                
                # Write to file immediately
                file.write(f"{f_a:.5e},{data_str}\n")
                file.flush()
                
            except Exception as e:
                print(f"Error solving for f={1/inv_f:.2e}: {e}", file=sys.stderr)
                sys.stderr.flush()

if __name__ == "__main__":
    main()
