import numpy as np
import math as ma
from scipy.sparse import lil_matrix
import primme
import matplotlib.pyplot as plt
from scipy.sparse import bsr_matrix


def create_ordered_basis(particle_number, sites_number):
    BASIS_DIM = int(ma.factorial(particle_number + sites_number - 1) / (ma.factorial(particle_number) * ma.factorial(sites_number - 1)))
    ordered_basis = np.zeros(shape=(BASIS_DIM, sites_number), dtype=int)
    ordered_basis[0] = [0 * sites_number]
    ordered_basis[0][0] = particle_number
    for d in range(1, BASIS_DIM):
        for i in range(0, sites_number):
            j = sites_number - 2 - i
            if ordered_basis[d - 1][j] != 0:
                k = j
                break

        for i in range(0, k):
            ordered_basis[d][i] = ordered_basis[d - 1][i]

        ordered_basis[d][k] = ordered_basis[d - 1][k] - 1
        sum = 0
        for i in range(0, k + 1):
            sum += ordered_basis[d][i]

        ordered_basis[d][k + 1] = particle_number - sum
        for i in range(k + 2, sites_number):
            ordered_basis[d][i] = 0
    print("Ordered Fock basis created")

    return ordered_basis

def TagBase(d, ordered_basis, M):
    sum = 0
    for i in range(0, M):
        sum += ma.sqrt(100 * (i + 1) + 3) * ordered_basis[d][i]
    return sum

def TagVec(vec, sites_number):
    sum = 0
    for i in range(0, sites_number):
        sum += ma.sqrt(100 * (i + 1) + 3) * vec[i]
    return sum

def define_tags_and_ind(BASIS_DIM, ordered_basis, M):
    tags=[]
    for d in range(0, BASIS_DIM):
        tags.append(TagBase(d, ordered_basis, M))

    sorted_tags = np.sort(tags)
    ind = []
    for i in range(0, BASIS_DIM):
        ind.append(np.where(sorted_tags == tags[i])[0][0])

    return sorted_tags, ind

def create_hamiltonian(M, sorted_tags, ind, BASIS_DIM, ordered_basis, U, J, CAVITY = True):

    # Hamiltoniano parte interacción
    H_interaction = lil_matrix((BASIS_DIM, BASIS_DIM))
    for u in range(0, BASIS_DIM):
        sum = 0
        for i in range(0, M):
            sum += ordered_basis[u][i] * (ordered_basis[u][i] - 1)
        H_interaction[u, u] = (U / 2) * sum
    print('H_interaction created')

    # Matriz parte cinética

    H_kinetic = lil_matrix((BASIS_DIM, BASIS_DIM))
    for v in range(0, BASIS_DIM):
        us = []
        for i in range(0, M):
            factor = 0
            j = i + 1
            vec = ordered_basis.copy()[v]
            if j == M:
                j = 0
            vec[i] = vec[i] - 1
            vec[j] = vec[j] + 1
            ind_v = -1
            if TagVec(vec,M) in sorted_tags:
                ind_v = np.where(sorted_tags == TagVec(vec,M))[0][0]
                factor = -J * np.sqrt((ordered_basis[v][j] + 1) * ordered_basis[v][i])

                us.append(ind.index(ind_v))

                for u in us:
                    ind_u = ind[u]
                    if ind_u == ind_v:
                        H_kinetic[u, v] += factor
                        H_kinetic[v, u] = H_kinetic[u, v]
    #print(H_kinetic.todense())
    print('H_kinetic created')

    # Hamiltoniano parte cavidad
    H_cavity = lil_matrix((BASIS_DIM, BASIS_DIM))
    D_sq = lil_matrix((BASIS_DIM, BASIS_DIM))

    D_matrix = lil_matrix((BASIS_DIM, BASIS_DIM))
    for u in range(0, BASIS_DIM):
        sum = 0
        for i in range(0, M):
            sum += ma.pow(-1, i) * ordered_basis[u][i]
        D_sq[u, u] = ma.pow(sum, 2)
        H_cavity[u, u] = (-4 * U / M) * D_sq[u, u]
        D_matrix[u, u] = sum

    epsilon = ma.pow(10, -4)
    # epsilon=0

    if CAVITY:
        H_total = H_interaction + H_kinetic + H_cavity + epsilon * D_matrix
    else:
        H_total = H_interaction + H_kinetic
    return H_total, D_sq

def calculate_ground_state(H_total):
    eigshH_total = primme.eigsh(H_total, k=1, which='SA')
    E_g = eigshH_total[0][0]
    GS = eigshH_total[1]
    GS_transposed = GS.transpose()
    print('Ground state calculated')

    return GS, GS_transposed, E_g

def calculate_expected_values(GS, BASIS_DIM, ordered_basis, M, D_sq, nvariance, nvariance2, n1, n2, O_DW):

    GS_transposed = GS.transpose()
    i_n = 0
    i2_n = 1  #
    op_n = lil_matrix((BASIS_DIM, BASIS_DIM))
    op2_n = lil_matrix((BASIS_DIM, BASIS_DIM))  #

    for j in range(0, BASIS_DIM):
        op_n[j, j] = ordered_basis[j][i_n]
        op2_n[j, j] = ordered_basis[j][i2_n]  #

    # cálculo de valores esperados
    n_esperado = GS_transposed * op_n.todense() * GS
    n_esperado2 = GS_transposed * op2_n.todense() * GS  #

    op_n2 = op_n.todense() * op_n.todense()
    op2_n2 = op2_n.todense() * op2_n.todense()  #

    n2_esperado = GS_transposed * op_n2 * GS
    n2_esperado2 = GS_transposed * op2_n2 * GS  #

    # Dni=(n2_esperado[0,0]+n2_esperado2[0,0])/2-((n_esperado[0,0]+n_esperado2[0,0])/2)**2

    # nvariance.append(Dni)

    var_n = n2_esperado[0, 0] - n_esperado[0, 0] ** 2
    var2_n = n2_esperado2[0, 0] - n_esperado2[0, 0] ** 2  #

    nvariance.append(np.sqrt(var_n))
    nvariance2.append(np.sqrt(var2_n))  #

    n1.append(n_esperado[0, 0])
    n2.append(n_esperado2[0, 0])

    D_sq_esperado = GS_transposed * D_sq.todense() * GS
    o_dw = D_sq_esperado[0, 0] / ma.pow(M, 2)
    O_DW.append(o_dw)
    print('Expected values calculated')
    return nvariance, nvariance2, n1, n2, O_DW

def calculate_condensate_factor(M, BASIS_DIM, ordered_basis, sorted_tags, ind, GS, f, N ):
    GS_transposed = GS.transpose()
    # matriz de densidad reducida (para factor de condensado)
    RHO = lil_matrix((M, M))
    for i in range(0, M):
        for j in range(0, M):

            aiaj = lil_matrix((BASIS_DIM, BASIS_DIM))
            for v in range(0, BASIS_DIM):
                if i != j:
                    us = []
                    factor = 0
                    vec = ordered_basis.copy()[v]
                    vec[i] = vec[i] + 1
                    vec[j] = vec[j] - 1
                    ind_v = -1
                    if TagVec(vec,M) in sorted_tags:
                        ind_v = np.where(sorted_tags == TagVec(vec,M))[0][0]
                        factor = np.sqrt((ordered_basis[v][i] + 1) * ordered_basis[v][j])
                        us.append(ind.index(ind_v))
                        for u in us:
                            ind_u = ind[u]
                            if ind_u == ind_v:
                                aiaj[u, v] = factor
                if i == j:
                    aiaj[v, v] = ordered_basis.copy()[v][i]
            RHO[i, j] = (GS_transposed * aiaj.todense() * GS)[0, 0]
    print('Condensate fraction matrix created')
    f.append(primme.eigsh(RHO, k=1, which='LA')[0][0] / N)
    print('Condensate fraction calculated')
    return f

def time_convert(sec):
  mins = sec // 60
  sec = sec % 60
  hours = mins // 60
  mins = mins % 60
  print("Computation time = {0}:{1}:{2}".format(int(hours),int(mins),sec))

def search(groups, element, new_D):
    for i in range(0, new_D):
        for j in range(len(groups[i])):
            if groups[i][j] == element:
                return i

def NuevaBase(N, M):
    D_interna = int(ma.factorial(N + int(M / 2) - 1) / (ma.factorial(N) * ma.factorial(int(M / 2) - 1)))
    new_A = np.zeros(shape=(D_interna, int(M / 2)), dtype=int)
    new_A[0] = [0 * int(M / 2)]
    new_A[0][0] = N
    for d in range(1, D_interna):
        for i in range(0, int(M / 2)):
            j = int(M / 2) - 2 - i
            if new_A[d - 1][j] != 0:
                k = j
                break

        for i in range(0, k):
            new_A[d][i] = new_A[d - 1][i]

        new_A[d][k] = new_A[d - 1][k] - 1
        sum = 0
        for i in range(0, k + 1):
            sum += new_A[d][i]

        new_A[d][k + 1] = N - sum
        for i in range(k + 2, int(M / 2)):
            new_A[d][i] = 0
    
    return new_A

def create_diccionaries(M, A, D):
    if M == 4:
        diccionario = []  # sobre mitad
        for m_ in range(0, M + 1):
            for k in range(0, len(NuevaBase(m_,M))):
                a4, a5 = NuevaBase(m_,M)[k][0], NuevaBase(m_,M)[k][1]
                sublist = []
                for j in range(0, D):
                    if A[j][2] == a4 and A[j][3] == a5:
                        sublist.append(j)
                diccionario.append(sublist)

        #print(diccionario)

        diccionario2 = []  # sobre impares
        for m_ in range(0, M + 1):
            for k in range(0, len(NuevaBase(m_,M))):
                a4, a5 = NuevaBase(m_,M)[k][0], NuevaBase(m_,M)[k][1]
                sublist = []
                for j in range(0, D):
                    if A[j][1] == a4 and A[j][3] == a5:
                        sublist.append(j)
                diccionario2.append(sublist)


    if M == 6:
        diccionario = []  # sobre mitad
        for m_ in range(0, M + 1):
            for k in range(0, len(NuevaBase(m_,M))):
                a4, a5, a6 = NuevaBase(m_,M)[k][0], NuevaBase(m_,M)[k][1], NuevaBase(m_,M)[k][2]
                sublist = []
                for j in range(0, D):
                    if A[j][3] == a4 and A[j][4] == a5 and A[j][5] == a6:
                        sublist.append(j)
                diccionario.append(sublist)

        diccionario2 = []  # sobre impares
        for m_ in range(0, M + 1):
            for k in range(0, len(NuevaBase(m_,M))):
                a4, a5, a6 = NuevaBase(m_,M)[k][0], NuevaBase(m_,M)[k][1], NuevaBase(m_,M)[k][2]
                sublist = []
                for j in range(0, D):
                    if A[j][1] == a4 and A[j][3] == a5 and A[j][5] == a6:
                        sublist.append(j)
                diccionario2.append(sublist)

    if M == 8:
        diccionario = []  # sobre mitad
        for m_ in range(0, M + 1):
            for k in range(0, len(NuevaBase(m_,M))):
                a4, a5, a6, a7 = NuevaBase(m_,M)[k][0], NuevaBase(m_,M)[k][1], NuevaBase(m_,M)[k][2], NuevaBase(m_,M)[k][3]
                sublist = []
                for j in range(0, D):
                    if A[j][4] == a4 and A[j][5] == a5 and A[j][6] == a6 and A[j][7] == a7:
                        sublist.append(j)
                diccionario.append(sublist)

        diccionario2 = []  # sobre impares
        for m_ in range(0, M + 1):
            for k in range(0, len(NuevaBase(m_,M))):
                a4, a5, a6, a7 = NuevaBase(m_,M)[k][0], NuevaBase(m_,M)[k][1], NuevaBase(m_,M)[k][2], NuevaBase(m_,M)[k][3]
                sublist = []
                for j in range(0, D):
                    if A[j][1] == a4 and A[j][3] == a5 and A[j][5] == a6 and A[j][7] == a7:
                        sublist.append(j)
                diccionario2.append(sublist)
    return diccionario, diccionario2

def create_index_mapping(groups):
    index_mapping = {}
    for group_index, group in enumerate(groups):
        for element in group:
            index_mapping[element] = group_index
    return index_mapping

def search_index(index_mapping, element):
    return index_mapping[element]

#previous approach on calculating entropy (non-optimized)
def calculate_entropy(GS, M, N, A, D, S_ent, S_ent2, diccionario, diccionario2):


    new_D = 0
    for i in range(0, N + 1):
        new_D += int(ma.factorial(i + int(M / 2) - 1) / (ma.factorial(i) * ma.factorial(int(M / 2) - 1)))
    RHO_AB = GS * GS.transpose()
    RHO_trazada = lil_matrix((new_D, new_D))

    # Aquí hago la traza parcial sobre el sitios par
    for i in range(0, D):
        for j in range(0, D):
            if M==4:
                if A[i][0] == A[j][0] and A[i][2] == A[j][2]:
                    RHO_trazada[search(diccionario2, i, new_D), search(diccionario2, j, new_D)] += RHO_AB[i][j]
            elif M==6:
                if A[i][0] == A[j][0] and A[i][2] == A[j][2] and A[i][4]==A[j][4]:
                    RHO_trazada[search(diccionario2, i, new_D), search(diccionario2, j, new_D)] += RHO_AB[i][j]
            elif M==8:
                if A[i][0] == A[j][0] and A[i][2] == A[j][2] and A[i][4]==A[j][4] and A[i][6]==A[j][6]:
                    RHO_trazada[search(diccionario2, i, new_D), search(diccionario2, j, new_D)] += RHO_AB[i][j]

    RHO_trazada = bsr_matrix(RHO_trazada)
    RHO_eigsh = primme.eigsh(RHO_trazada, k=new_D, which='LM', tol=0)

    V = RHO_eigsh[1]
    V_inv = V.transpose()

    D_matrix = lil_matrix((new_D, new_D))
    for i in range(0, new_D):
        if RHO_eigsh[0][i] > ma.pow(10, -14):
            D_matrix[i, i] = RHO_eigsh[0][i]
        else:
            D_matrix[i, i] = 0

    Ln_D_matrix = lil_matrix((new_D, new_D))
    for i in range(0, new_D):
        if D_matrix[i, i] != 0:
            Ln_D_matrix[i, i] = np.log2(D_matrix[i, i])
        else:
            Ln_D_matrix[i, i] = 0

    # Smatrix=(RHO_trazada*(V*(Ln_D_matrix.todense()*V_inv)))
    Smatrix = np.matmul(RHO_trazada.todense(), np.matmul(V, np.matmul(Ln_D_matrix.todense(), V_inv)))
    S = -Smatrix.trace()[0, 0]
    S_ent.append(S)
    print('Traced over odd sites')

    RHO_AB = GS * GS.transpose()
    RHO_trazada = lil_matrix((new_D, new_D))

    # Aquí hago la traza parcial sobre primera mitad
    for i in range(0, D):
        for j in range(0, D):
            if M==4:
                if A[i][0] == A[j][0] and A[i][1] == A[j][1]:
                    RHO_trazada[search(diccionario, i, new_D), search(diccionario, j, new_D)] += RHO_AB[i][j]
            elif M==6:
                if A[i][0] == A[j][0] and A[i][1] == A[j][1] and A[i][2]==A[j][2]:
                    RHO_trazada[search(diccionario, i, new_D), search(diccionario, j, new_D)] += RHO_AB[i][j]
            elif M==8:
                if A[i][0] == A[j][0] and A[i][1] == A[j][1] and A[i][2]==A[j][2] and A[i][3]==A[j][3]:
                    RHO_trazada[search(diccionario, i, new_D), search(diccionario, j, new_D)] += RHO_AB[i][j]


    # RHO_trazada=bsr_matrix(RHO_trazada)
    RHO_eigsh = primme.eigsh(RHO_trazada, k=new_D, which='LM', tol=0)

    V = RHO_eigsh[1]
    V_inv = V.transpose()

    D_matrix = lil_matrix((new_D, new_D))
    for i in range(0, new_D):
        if RHO_eigsh[0][i] > ma.pow(10, -14):
            D_matrix[i, i] = RHO_eigsh[0][i]
        else:
            D_matrix[i, i] = 0

    Ln_D_matrix = lil_matrix((new_D, new_D))
    for i in range(0, new_D):
        if D_matrix[i, i] != 0:
            Ln_D_matrix[i, i] = np.log2(D_matrix[i, i])
        else:
            Ln_D_matrix[i, i] = 0

    # Smatrix=(RHO_trazada*(V*(Ln_D_matrix.todense()*V_inv)))
    Smatrix = np.matmul(RHO_trazada.todense(), np.matmul(V, np.matmul(Ln_D_matrix.todense(), V_inv)))
    S = -Smatrix.trace()[0, 0]
    S_ent2.append(S)
    print('Traced over first half sites')


    return S_ent,S_ent2

def plot_results(ratio, O_DW, nvariance, nvariance2, n1, n2, N, M, f, CAVITY, S_ent, S_ent2):
    plt.figure(figsize=(10, 10))
    if CAVITY:
        plt.plot(ratio, O_DW, 'tab:green', label='$O_{DW}$', marker='.', linestyle='-')
    plt.plot(ratio, nvariance, 'tab:red', label='$\Delta n_i$', marker='.', linestyle='-')
    plt.plot(ratio, n1, 'tab:blue', label='$<n_{i=1}>$', marker='.', linestyle='-')
    plt.plot(ratio, n2, 'tab:pink', label='$<n_{i=2}>$', marker='.', linestyle='-')
    plt.plot(ratio, f, 'tab:orange', label='$f_C$', marker='.', linestyle='-')
    plt.plot(ratio, S_ent, 'tab:blue', label='$S_{ent}$ trazado sobre impares', marker='o', linestyle='-')
    plt.plot(ratio, S_ent2, 'tab:red', label='$S_{ent}$ trazado sobre primera mitad', marker='o', linestyle='-')
    plt.xlabel('$J / U$', fontsize="22")
    plt.title('')
    plt.xscale('log')
    plt.ylim(bottom=0, top=4.5)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    axes = plt.gca()
    plt.legend(loc = "upper left",fontsize="18")
    #plt.draw()
    plt.savefig('Plots/RESULTADO(N={},M={},CAVITY={}).png'.format(N,M,CAVITY))

def write_results_txt(filename, ratio, nvariance, nvariance2, n1, n2, f, S_ent, S_ent2, O_DW):
    with open('{}.txt'.format(filename), 'w') as file:
        # Write the header row
        header = "{:<23} {:<23} {:<23} {:<23} {:<23} {:<23} {:<23} {:<23} {:<23}\n".format("Ratio", "N_Variance", "N_Variance2", "N1", "N2", "F", "S_Ent", "S_Ent2", "O_DW")
        file.write(header)

        # Write the data rows
        for i in range(len(ratio)):
            row = "{:<23.16f} {:<23.16f} {:<23.16f} {:<23} {:<23} {:<23.16f} {:<23.16f} {:<23.16f} {:<23.16f}\n".format(
                ratio[i], nvariance[i], nvariance2[i], n1[i], n2[i], f[i], S_ent[i], S_ent2[i], O_DW[i])
            file.write(row)

    print("\n Data has been written to {}.txt".format(filename))

#New approach on calculating entropy (optimized)
def calculate_entropy_V2(GS, M, N, A, D, S_ent, S_ent2, diccionario, diccionario2, index_mapping_1, index_mapping_2):


    new_D = 0
    for i in range(0, N + 1):
        new_D += int(ma.factorial(i + int(M / 2) - 1) / (ma.factorial(i) * ma.factorial(int(M / 2) - 1)))
    RHO_AB = GS * GS.transpose()
    RHO_trazada = lil_matrix((new_D, new_D))

    # Aquí hago la traza parcial sobre el sitios par
    for i in range(0, D):
        for j in range(0, D):
            if M==4:
                if A[i][0] == A[j][0] and A[i][2] == A[j][2]:
                    RHO_trazada[search_index(index_mapping_2, i), search_index(index_mapping_2, j)] += RHO_AB[i][j]
            elif M==6:
                if A[i][0] == A[j][0] and A[i][2] == A[j][2] and A[i][4]==A[j][4]:
                    RHO_trazada[search_index(index_mapping_2, i), search_index(index_mapping_2, j)] += RHO_AB[i][j]
            elif M==8:
                if A[i][0] == A[j][0] and A[i][2] == A[j][2] and A[i][4]==A[j][4] and A[i][6]==A[j][6]:
                    RHO_trazada[search_index(index_mapping_2, i), search_index(index_mapping_2, j)] += RHO_AB[i][j]
    
    RHO_trazada = bsr_matrix(RHO_trazada)
    RHO_eigsh = primme.eigsh(RHO_trazada, k=new_D, which='LM', tol=0)

    V = RHO_eigsh[1]
    V_inv = V.transpose()

    D_matrix = lil_matrix((new_D, new_D))
    for i in range(0, new_D):
        if RHO_eigsh[0][i] > ma.pow(10, -14):
            D_matrix[i, i] = RHO_eigsh[0][i]
        else:
            D_matrix[i, i] = 0

    Ln_D_matrix = lil_matrix((new_D, new_D))
    for i in range(0, new_D):
        if D_matrix[i, i] != 0:
            Ln_D_matrix[i, i] = np.log2(D_matrix[i, i])
        else:
            Ln_D_matrix[i, i] = 0

    # Smatrix=(RHO_trazada*(V*(Ln_D_matrix.todense()*V_inv)))
    Smatrix = np.matmul(RHO_trazada.todense(), np.matmul(V, np.matmul(Ln_D_matrix.todense(), V_inv)))
    S = -Smatrix.trace()[0, 0]
    S_ent.append(S)
    print('Traced over odd sites')

    RHO_AB = GS * GS.transpose()
    RHO_trazada = lil_matrix((new_D, new_D))

    # Aquí hago la traza parcial sobre primera mitad
    for i in range(0, D):
        for j in range(0, D):
            if M==4:
                if A[i][0] == A[j][0] and A[i][1] == A[j][1]:
                    RHO_trazada[search_index(index_mapping_1, i), search_index(index_mapping_1, j)] += RHO_AB[i][j]
            elif M==6:
                if A[i][0] == A[j][0] and A[i][1] == A[j][1] and A[i][2]==A[j][2]:
                    RHO_trazada[search_index(index_mapping_1, i), search_index(index_mapping_1, j)] += RHO_AB[i][j]
            elif M==8:
                if A[i][0] == A[j][0] and A[i][1] == A[j][1] and A[i][2]==A[j][2] and A[i][3]==A[j][3]:
                    RHO_trazada[search_index(index_mapping_1, i), search_index(index_mapping_1, j)] += RHO_AB[i][j]
            elif M==8:
                if A[i][0] == A[j][0] and A[i][1] == A[j][1] and A[i][2]==A[j][2] and A[i][3]==A[j][3] and A[i][4]==A[j][4]:
                    RHO_trazada[search_index(index_mapping_1, i), search_index(index_mapping_1, j)] += RHO_AB[i][j]


    # RHO_trazada=bsr_matrix(RHO_trazada)
    RHO_eigsh = primme.eigsh(RHO_trazada, k=new_D, which='LM', tol=0)

    V = RHO_eigsh[1]
    V_inv = V.transpose()

    D_matrix = lil_matrix((new_D, new_D))
    for i in range(0, new_D):
        if RHO_eigsh[0][i] > ma.pow(10, -14):
            D_matrix[i, i] = RHO_eigsh[0][i]
        else:
            D_matrix[i, i] = 0

    Ln_D_matrix = lil_matrix((new_D, new_D))
    for i in range(0, new_D):
        if D_matrix[i, i] != 0:
            Ln_D_matrix[i, i] = np.log2(D_matrix[i, i])
        else:
            Ln_D_matrix[i, i] = 0

    # Smatrix=(RHO_trazada*(V*(Ln_D_matrix.todense()*V_inv)))
    Smatrix = np.matmul(RHO_trazada.todense(), np.matmul(V, np.matmul(Ln_D_matrix.todense(), V_inv)))
    S = -Smatrix.trace()[0, 0]
    S_ent2.append(S)
    print('Traced over first half sites')


    return S_ent,S_ent2