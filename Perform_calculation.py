
import math as ma
from Exact_diag_functions import create_hamiltonian, calculate_expected_values, calculate_ground_state, create_ordered_basis, define_tags_and_ind, plot_results, calculate_condensate_factor, time_convert, create_diccionaries, calculate_entropy, create_index_mapping, calculate_entropy_V2, write_results_txt
import time

def main():
    print("Exact Diagonalization\n")
    start_time = time.time()

    # expected values arrays
    nvariance = []
    nvariance2 = []
    n1 = []
    n2 = []
    O_DW = []
    f = []
    S_ent = []
    S_ent2 = []

    #system configuration
    N = 8
    M = 8
    CAVITY = False

    # constants
    BASIS_DIM = int(ma.factorial(N + M - 1) / (ma.factorial(N) * ma.factorial(M - 1)))
    U = 1
    J = 1


    # begins diagonalization method
    ordered_basis = create_ordered_basis(N, M)
    sorted_tags, ind = define_tags_and_ind(BASIS_DIM, ordered_basis, M)

    diccionario, diccionario2 = create_diccionaries(M, ordered_basis, BASIS_DIM)
    index_mapping_1 = create_index_mapping(diccionario)
    index_mapping_2 = create_index_mapping(diccionario2)

    # cicle from MI to SF regimes
    ratio = []
    CORTE = 40
    step = -2.4
    for i_step in range(0, CORTE):

        step += 0.1
        print('\n current: ', i_step, '/', CORTE - 1)
        J = U * ma.pow(10, step)
        ratio.append(J / U)

        # Hamiltonian creation and solution
        H_total, D_sq = create_hamiltonian(M, sorted_tags, ind, BASIS_DIM, ordered_basis, U, J, CAVITY)
        GS, GS_transposed, E_g = calculate_ground_state(H_total)

        # expected values calculation
        nvariance, nvariance2, n1, n2, O_DW = calculate_expected_values(GS, BASIS_DIM, ordered_basis, M, D_sq, nvariance, nvariance2, n1, n2, O_DW)
        f = calculate_condensate_factor(M, BASIS_DIM, ordered_basis, sorted_tags, ind, GS, f, N)

        #S_ent,S_ent2 = calculate_entropy(GS,M,N,ordered_basis,BASIS_DIM,S_ent,S_ent2,diccionario, diccionario2)
        S_ent,S_ent2 = calculate_entropy_V2(GS, M, N, ordered_basis, BASIS_DIM, S_ent, S_ent2, diccionario, diccionario2, index_mapping_1, index_mapping_2)

    #plotting
    plot_results(ratio, O_DW, nvariance, nvariance2, n1, n2, N, f, CAVITY, S_ent, S_ent2)

    #store results in txt
    write_results_txt("txt_results/results(N={},M={},Cav={})".format(N,M,CAVITY), ratio, nvariance, nvariance2, n1, n2, f, S_ent, S_ent2, O_DW)

    end_time = time.time()
    time_lapsed = end_time - start_time
    time_convert(time_lapsed)






if __name__ == "__main__":
    main()