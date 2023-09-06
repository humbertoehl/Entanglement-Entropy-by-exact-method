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
    plt.semilogx(ratio, o_dw, label='O_DW')

    # Set labels and title
    plt.xlabel('Ratio (log scale)')
    plt.ylabel('Value')
    plt.title('Data Plot')

    # Add legend
    plt.legend()

    # Show the plot
    plt.grid()
    #plt.show()
    plt.savefig("plt_from_txt")


plot_txt_data("results(N=4,M=4,Cav=False).txt")