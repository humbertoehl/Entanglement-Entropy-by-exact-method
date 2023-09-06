import matplotlib.pyplot as plt
import numpy as np

def plot_txt_data(file_path):
    # Read the data from the text file
    data = np.loadtxt(file_path, skiprows=1)  # Skip the header row

    # Extract the columns
    ratio = data[:, 0]
    n_variance = data[:, 1]
    n_variance2 = data[:, 2]
    n1 = data[:, 3]
    n2 = data[:, 4]
    f = data[:, 5]
    s_ent = data[:, 6]
    s_ent2 = data[:, 7]
    o_dw = data[:, 8]

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.semilogx(ratio, n_variance, label='N_Variance')
    plt.semilogx(ratio, n_variance2, label='N_Variance2')
    plt.semilogx(ratio, n1, label='N1')
    plt.semilogx(ratio, n2, label='N2')
    plt.semilogx(ratio, f, label='F')
    plt.semilogx(ratio, s_ent, label='S_Ent')
    plt.semilogx(ratio, s_ent2, label='S_Ent2')
    if CAVITY==True:
        plt.semilogx(ratio, o_dw, label='O_DW')

    # Set labels and title
    plt.xlabel('Ratio (log scale)')
    plt.ylabel('Value')
    plt.title('Data Plot')

    # Add legend
    plt.legend()

    # Show the plot
    plt.grid()
    plt.show()
    #plt.savefig("plt_from_txt")

while True:
    
    while True:
        N = input("Input number of particles N [4/6/8]: ")
        if N in ["4", "6", "8"]:
            N = int(N)
            break

    while True:
        M = input("Input number of lattice sites M [4/6/8]: ")
        if M in ["4", "6", "8"]:
            M = int(M)
            break

    while True:
        extended_model = input("Extended Bose-Hubbard (include cavity part)?  [y/n]: ").lower()
        if extended_model == "y":
            CAVITY = True
            break
        elif extended_model == "n":
            CAVITY = False
            break


    try:
        plot_txt_data("txt_results/results(N={},M={},Cav={}).txt".format(N,M,CAVITY))
        break
    except FileNotFoundError:
        print("File non-existent, try to create it using \"Perform_calculation.py\"")