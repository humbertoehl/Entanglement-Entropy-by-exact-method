from itertools import combinations_with_replacement, permutations
from scipy.linalg import eigh, kron
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import dok_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
import time
from concurrent.futures import ProcessPoolExecutor



def generate_fock_basis(N, M):
    def fock_states(N, M, current_state=[]):
        if len(current_state) == M:
            if sum(current_state) == N:
                yield current_state
        elif len(current_state) < M:
            for i in range(N + 1):
                if sum(current_state) + i <= N:
                    yield from fock_states(N, M, current_state + [i])

    all_states = list(fock_states(N, M))
    all_states_sorted = sorted(all_states, reverse=True)
    
    return np.array(all_states_sorted)


def create_hamiltonian(basis, N, M, t, U, mu):
    dim = len(basis)
    H = dok_matrix((dim, dim), dtype=np.float64)

    #hopping term
    state_to_index = {tuple(state): idx for idx, state in enumerate(basis)}
    for v, state_v in enumerate(basis):
        for j in range(M):
            next_site = (j + 1) % M
            prev_site = (j - 1) % M

            if state_v[j] > 0:
                for k in [next_site, prev_site]:
                    new_state = state_v.copy()
                    new_state[j] -= 1
                    new_state[k] += 1

                    u = state_to_index.get(tuple(new_state))
                    if u is not None:
                        matrix_element = -t * np.sqrt(state_v[j] * (new_state[k])) / 2  # /2 to avoid double counting
                        H[v, u] += matrix_element
                        H[u, v] += matrix_element #Hermitian

        #local terms
        interaction_term = U * sum(n * (n - 1) / 2 for n in state_v)
        chemical_term = mu * sum(state_v)
        H[v, v] += interaction_term + chemical_term

    return csr_matrix(H)



def plot_sparsity_pattern(H):
    fig, ax = plt.subplots()
    ax.spy(H, markersize=1)
    ax.set_title('Sparsity Pattern of the Hamiltonian')
    ax.set_xlabel('Index')
    ax.set_ylabel('Index')
    plt.show()

def find_ground_state(H):
    eigenvalues, eigenvectors = eigsh(H, k=1, which='SA')
    ground_state_energy = eigenvalues[0]
    ground_state_vector = eigenvectors[:, 0]
    return ground_state_energy, ground_state_vector


def calculate_expected_values(basis, state, site_indices=[0], N=None):
    
    # Condensate factor calculation
    if N is None:
        N = np.sum(basis[0])
    expectations = []
    variances = []
    M = basis.shape[1] 

    OBDM = np.zeros((M, M), dtype=np.complex128)
    for i in range(M):
        for j in range(M):
            if i == j:
                OBDM[i, j] = np.sum(np.abs(state)**2 * basis[:, i])
            else:
                for v, psi_v in enumerate(state):
                    state_v = basis[v]
                    if state_v[i] > 0 and state_v[j] < N:
                        new_state = state_v.copy()
                        new_state[i] -= 1
                        new_state[j] += 1
                        u = np.where((basis == new_state).all(axis=1))[0]
                        if len(u) > 0:
                            u = u[0]
                            OBDM[i, j] += np.conj(state[u]) * state[v] * np.sqrt(state_v[i] * (state_v[j] + 1))

    eigenvalues = eigh(OBDM, eigvals_only=True)
    n_max = np.max(eigenvalues)
    condensate_fraction = n_max / N

    #occupation and variance calculation
    for site in site_indices:
        site_index = site

        n_i = basis[:, site_index]
        expectation_n_i = np.dot(n_i, np.abs(state)**2)

        n_i_squared = n_i**2
        expectation_n_i_squared = np.dot(n_i_squared, np.abs(state)**2)

        expectations.append(expectation_n_i)

        variance_n_i = np.sqrt(expectation_n_i_squared - expectation_n_i**2)
        variances.append(variance_n_i)

    return expectations, variances, condensate_fraction

def calculate_reduced_density_matrix(basis, ground_state, M, subsystem_A, subsystem_B):

    config_to_index_A = {}
    index = 0
    for state in basis:
        config_A = tuple(state[subsystem_A])
        if config_A not in config_to_index_A:
            config_to_index_A[config_A] = index
            index += 1

    dim_A = len(config_to_index_A)

    rho_AB = np.outer(ground_state, np.conj(ground_state))

    rho_A = np.zeros((dim_A, dim_A), dtype=np.complex128)

    for i in range(len(basis)):
        for j in range(len(basis)):
            if np.all(basis[i][subsystem_B] == basis[j][subsystem_B]):
                config_i_A = tuple(basis[i][subsystem_A])
                config_j_A = tuple(basis[j][subsystem_A])
                index_i_A = config_to_index_A[config_i_A]
                index_j_A = config_to_index_A[config_j_A]
                rho_A[index_i_A, index_j_A] += rho_AB[i, j]

    return rho_A


def calculate_partial_trace(basis, ground_state, M, subsystem_A, subsystem_B):

    config_to_index_A = {}
    config_to_index_B = {}
    state_list_A = []
    state_list_B = []

    for state in basis:
        config_A = tuple(state[subsystem_A])
        config_B = tuple(state[subsystem_B])
        if config_A not in config_to_index_A:
            config_to_index_A[config_A] = len(state_list_A)
            state_list_A.append(config_A)
        if config_B not in config_to_index_B:
            config_to_index_B[config_B] = len(state_list_B)
            state_list_B.append(config_B)

    dim_A = len(state_list_A)
    rho_A = np.zeros((dim_A, dim_A), dtype=np.complex128)

    for idx, (state_i, psi_i) in enumerate(zip(basis, ground_state)):
        config_i_A = tuple(state_i[subsystem_A])
        config_i_B = tuple(state_i[subsystem_B])
        index_i_A = config_to_index_A[config_i_A]
        index_i_B = config_to_index_B[config_i_B]

        for state_j, psi_j in zip(basis, ground_state):
            config_j_A = tuple(state_j[subsystem_A])
            config_j_B = tuple(state_j[subsystem_B])
            if config_i_B == config_j_B:
                index_j_A = config_to_index_A[config_j_A]
                rho_A[index_i_A, index_j_A] += psi_i * np.conj(psi_j)
    return rho_A

def calculate_entropy(rho_A):
    eigenvalues, V = eigh(rho_A)
    V_inv = np.linalg.inv(V)
    D_log = np.diag(np.log(eigenvalues.clip(min=1e-10)))

    intermediate_matrix = V @ D_log @ V_inv

    entropy = -np.trace(rho_A @ intermediate_matrix)

    return entropy

def time_convert(sec):
  mins = sec // 60
  sec = sec % 60
  hours = mins // 60
  mins = mins % 60
  print("Computation time = {0}:{1}:{2}".format(int(hours),int(mins),sec))

def plot_as_functions_of_t(N, M):
    U = 1.0
    mu = 0
    start_time = time.time()
    basis = generate_fock_basis(N, M)
    print('Fock basis generated')
    maxsteps = 1
    t_values = np.logspace(-2.5, 2.5, num=maxsteps)

    subsystem_Aeo = [i for i in range(M) if i % 2 == 0]
    subsystem_Beo = [i for i in range(M) if i % 2 != 0]
    subsystem_Ahh = range(M//2) 
    subsystem_Bhh = range(M//2, M)

    print('subsystem A (even-odd):',subsystem_Aeo)
    print('subsystem B (even-odd):',subsystem_Beo)
    print('subsystem A (half-half):',list(subsystem_Ahh))
    print('subsystem B (half-half):',list(subsystem_Bhh))

    # Storing parameters
    expectation_0 = []
    variance_0 = []
    condensate_fractions = []
    entropies_eo = []
    entropies_oe = []
    entropies_hh1 = []
    entropies_hh2 = []
    step = 0
    for t in t_values:
        step += 1
        print('step ',step,'/',maxsteps)
        H_BH = create_hamiltonian(basis, N, M, t, U, mu)
        ground_energy, ground_state = find_ground_state(H_BH)
        expectations, variances, condensate_fraction = calculate_expected_values(basis, ground_state, N=N)
        
        expectation_0.append(expectations[0])
        variance_0.append(variances[0])
        condensate_fractions.append(condensate_fraction)
        
        print('Calculating entropy even-odd...')
        # Entropies for different partitions 
        entropies_eo.append(calculate_entropy(calculate_partial_trace(basis, ground_state, M, subsystem_Aeo, subsystem_Beo)).real)
        #entropies_oe.append(calculate_entropy(calculate_reduced_density_matrix(basis, ground_state, M, subsystem_Beo, subsystem_Aeo)))

        print('Calculating entropy half-half...')
        entropies_hh1.append(calculate_entropy(calculate_partial_trace(basis, ground_state, M, subsystem_Ahh, subsystem_Bhh)).real)
        #entropies_hh2.append(calculate_entropy(calculate_reduced_density_matrix(basis, ground_state, M, subsystem_Bhh, subsystem_Ahh)))
    end_time = time.time()
    time_lapsed = end_time - start_time
    time_convert(time_lapsed)
        

    plt.figure(figsize=(12, 8))
    plt.plot(t_values, expectation_0, label=r'$<n_0>$')
    plt.plot(t_values, variance_0, label=r'$\Delta n_0$')
    plt.plot(t_values, condensate_fractions, label=r'$f_c$')
    plt.plot(t_values, entropies_eo, label=r'Entropy $S_{even-odd}$', marker='s')
    #plt.plot(t_values, entropies_oe, label='Entropy odd-even')
    plt.plot(t_values, entropies_hh1, label=r'Entropy $S_{half-half}$', marker='.')
    #plt.plot(t_values, entropies_hh2, label='Entropy half-half 2')
    plt.xscale('log')
    plt.xlabel('t/U')
    plt.ylabel('Parameters')
    plt.title(fr'Order parameters for Bose-Hubbard model with $M=${M}, $N=${N}, $\mu=${mu}')
    plt.legend()

    plt.tight_layout()
    plt.show()


def task_for_parallel_execution(t, basis, N, M, U, mu):
    H_BH = create_hamiltonian(basis, N, M, t, U, mu)
    ground_energy, ground_state = find_ground_state(H_BH)
    expectations, variances, condensate_fraction = calculate_expected_values(basis, ground_state, N=N)

    # Calculate entropies
    subsystem_Aeo = [i for i in range(M) if i % 2 == 0]
    subsystem_Beo = [i for i in range(M) if i % 2 != 0]
    subsystem_Ahh = list(range(M//2))
    subsystem_Bhh = list(range(M//2, M))

    entropy_eo = calculate_entropy(calculate_partial_trace(basis, ground_state, M, subsystem_Aeo, subsystem_Beo))
    entropy_hh1 = calculate_entropy(calculate_partial_trace(basis, ground_state, M, subsystem_Ahh, subsystem_Bhh))

    print(t)

    return expectations[0], variances[0], condensate_fraction, entropy_eo.real, entropy_hh1.real

def parallelized_plot_as_functions_of_t(N, M):
    U = 1.0
    mu = 0
    start_time = time.time()
    basis = generate_fock_basis(N, M)
    maxsteps = 2
    t_values = np.logspace(-2.5, 2.5, num=maxsteps)

    subsystem_Aeo = [i for i in range(M) if i % 2 == 0]
    subsystem_Beo = [i for i in range(M) if i % 2 != 0]
    subsystem_Ahh = range(M//2) 
    subsystem_Bhh = range(M//2, M)

    print('subsystem A (even-odd):',subsystem_Aeo)
    print('subsystem B (even-odd):',subsystem_Beo)
    print('subsystem A (half-half):',list(subsystem_Ahh))
    print('subsystem B (half-half):',list(subsystem_Bhh))

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(task_for_parallel_execution, t_values, [basis]*len(t_values), [N]*len(t_values), [M]*len(t_values), [U]*len(t_values), [mu]*len(t_values)))

    expectation_0, variance_0, condensate_fractions, entropies_eo, entropies_hh1 = zip(*results)

    end_time = time.time()
    time_lapsed = end_time - start_time
    time_convert(time_lapsed)

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.plot(t_values, expectation_0, label='Expectation <n_0>')
    plt.plot(t_values, variance_0, label='Variance of n_0')
    plt.plot(t_values, condensate_fractions, label='Condensate Fraction')
    plt.plot(t_values, entropies_eo, label='Entropy Even-Odd', marker='s')
    plt.plot(t_values, entropies_hh1, label='Entropy Half-Half', marker='o')
    plt.xscale('log')
    plt.xlabel('t/U')
    plt.ylabel('Parameters')
    plt.title('Order parameters for Bose-Hubbard model')
    plt.legend()
    plt.tight_layout()
    plt.show()


M = int(input('Number of sites: '))
N = int(input('Number of particles: '))
#plot_as_functions_of_t(M,N)

parallelized_plot_as_functions_of_t(N, M)