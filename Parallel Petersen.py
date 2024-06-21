import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
#from multiprocessing import Pool
from multiprocessing import Pool, cpu_count

def generate_rows(k):
    """Генерация всех возможных строк длины k с суммой элементов, равной 3."""
    rows = []
    for comb in itertools.combinations_with_replacement(range(k), 3):
        row = [0] * k
        for index in comb:
            row[index] += 1
        rows.append(row)
    return rows

def is_valid_matrix(matrix):
    """Проверка матрицы на соответствие условиям."""
    k = len(matrix)
    for i in range(k):
        if matrix[i][i] == 3:  # Самопетли не допускаются
            return False
    for i in range(k):
        for j in range(k):
            if (matrix[i][j] == 0 or matrix[j][i] == 0) and matrix[i][j] != matrix[j][i]:  # Симметричность
                return False
    for i in range(k):
        for j in range(i + 1, k):
            if matrix[i] == matrix[j]:  # Различие строк
                return False
    for i in range(k):
        for j in range(i + 1, k):
            if all(matrix[row][i] == matrix[row][j] for row in range(k)):  # Уникальность столбцов
                return False
    return True

def matrices_are_equivalent(mat1, mat2):
    """Проверка эквивалентности двух матриц."""
    k = len(mat1)
    for row_perm in itertools.permutations(range(k)):
        permuted_mat1 = [mat1[i] for i in row_perm]
        for col_perm in itertools.permutations(range(k)):
            permuted_col_mat1 = [[permuted_mat1[i][j] for j in col_perm] for i in range(k)]
            if permuted_col_mat1 == mat2:
                return True
    return False

def filter_equivalent_matrices(matrices):
    """Отсев эквивалентных матриц и вывод допустимых матриц."""
    filtered = []
    for mat in matrices:
        if is_valid_matrix(mat):
            is_unique = True
            for existing_mat in filtered:
                if matrices_are_equivalent(mat, existing_mat):
                    is_unique = False
                    break
            if is_unique:
                filtered.append(mat)
    return filtered

def generate_matrix_chunk(rows, k, start, end):
    """Генерация части матриц параметров из строк."""
    matrices = []
    for i, perm in enumerate(itertools.product(rows, repeat=k)):
        if start <= i < end:
            matrix = [list(row) for row in perm]
            matrices.append(matrix)
    return matrices

def create_generalized_petersen_graph(n, k_param):
    """Создание обобщенного графа Петерсена П(n, k_param)."""
    V = list(range(2 * n))
    E = []
    for i in range(n):
        E.append((i, (i + 1) % n))  # Цикл внешнего многоугольника
        E.append((i, n + i))  # Соединения внешнего и внутреннего многоугольника
        E.append((n + i, n + (i + k_param) % n))  # Цикл внутреннего многоугольника с шагом k_param
    return V, E

def adjacency_matrix(graph, n):
    """Создание матрицы смежности для обобщенного графа Петерсена П(n, k)."""
    V, E = graph
    adj_matrix = np.zeros((2 * n, 2 * n), dtype=int)
    for u, v in E:
        adj_matrix[u][v] = 1
        adj_matrix[v][u] = 1
    return adj_matrix

def is_perfect_coloring(graph, coloring, matrix):
    """Проверка, является ли раскраска графа совершенной для данной матрицы параметров."""
    V, E = graph
    k = len(matrix)
    for i in range(k):
        for u in V:
            if coloring[u] == i:
                neighbor_colors = [coloring[v] for v in V if (u, v) in E or (v, u) in E]
                color_counts = [neighbor_colors.count(j) for j in range(k)]
                if color_counts != matrix[i]:
                    return False
    return True

def find_perfect_coloring(graph, k, matrix, max_time=15):
    """Поиск совершенной раскраски графа для данной матрицы параметров с ограничением времени."""
    V, E = graph
    start_time = time.time()
    for coloring in itertools.product(range(k), repeat=len(V)):
        if time.time() - start_time > max_time:
            return False, None
        if is_perfect_coloring(graph, coloring, matrix):
            return True, coloring
    return False, None

def parallel_find_perfect_coloring(args):
    graph, k, matrix, max_time = args
    return find_perfect_coloring(graph, k, matrix, max_time)

def draw_graph(graph, coloring=None):
    """Визуализация графа с раскраской."""
    G = nx.Graph()
    V, E = graph
    G.add_nodes_from(V)
    G.add_edges_from(E)
    pos = nx.spring_layout(G)
    
    plt.figure(figsize=(10, 8))
    if coloring:
        colors = [coloring[node] for node in G.nodes()]
        nx.draw(G, pos, node_color=colors, with_labels=True, node_size=500, cmap=plt.cm.rainbow, font_color='white')
    else:
        nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_color='black')
    plt.show()

"""def main():
    n = int(input("Введите число вершин в циклической части графа n: "))
    k_param = int(input("Введите параметр k для обобщенного графа Петерсена: "))
    num_colors = int(input("Введите число цветов k: "))
    begin_time = time.time()
    rows = generate_rows(num_colors)
    
    # Определение размера и количества частей
    k = num_colors
    total_matrices = len(rows) ** k
    num_chunks = 4  # Число процессов для распараллеливания
    chunk_size = total_matrices // num_chunks
    
    # Параллельное создание матриц
    with Pool(num_chunks) as pool:
        chunk_ranges = [(rows, k, i * chunk_size, (i + 1) * chunk_size) for i in range(num_chunks)]
        chunk_matrices = pool.starmap(generate_matrix_chunk, chunk_ranges)
    
    all_matrices = [matrix for chunk in chunk_matrices for matrix in chunk]
    
    #print("Все матрицы параметров:")
    #for matrix in all_matrices:
     #   print(np.array(matrix))
    
    print("\nДопустимые матрицы параметров:")
    filtered_matrices = filter_equivalent_matrices(all_matrices)
    i = 0
    for matrix in filtered_matrices:
        i += 1
        print(np.array(matrix))
    print(f"Количество допустимых матриц: {i}")
    
    graph = create_generalized_petersen_graph(n, k_param)
    adj_matrix = adjacency_matrix(graph, n)
    print(f"\nМатрица смежности графа Петерсена П({n}, {k_param}):\n{adj_matrix}")
    print(f"\nСовершенные раскраски для графа Петерсена П({n}, {k_param}):")
    max_time = 15  # Максимальное время выполнения поиска (в секундах)
    pool = Pool()
    perfect_colorings = []
    
    tasks = [(graph, num_colors, matrix, max_time) for matrix in filtered_matrices]
    results = pool.map(parallel_find_perfect_coloring, tasks)
    
    for (matrix, (is_perfect, coloring)) in zip(filtered_matrices, results):
        print(f"Матрица параметров:\n{np.array(matrix)}")
        if is_perfect:
            print("Эта матрица подходит для совершенной раскраски.\n")
            print(f"Раскраска:\n{coloring}")
            print(f"Матрица раскраски (построчно):")
            for vertex, color in enumerate(coloring):
                print(f"Вершина {vertex}: Цвет {color}")
            perfect_colorings.append(coloring)
        else:
            print("Эта матрица не подходит для совершенной раскраски.\n")
    
    print(time.time()-begin_time," - время параллельного выполнения(без учета времени раскраски) ")
    draw_graph(graph)
    for coloring in perfect_colorings:
        draw_graph(graph, coloring) 

if __name__ == "__main__":
    main()"""



def main():
    n = int(input("Введите число вершин в циклической части графа n: "))
    k_param = int(input("Введите параметр k для обобщенного графа Петерсена: "))
    num_colors = int(input("Введите число цветов k: "))
    begin_time = time.time()
    rows = generate_rows(num_colors)
    
    # Определение размера и количества частей
    k = num_colors
    total_matrices = len(rows) ** k
    num_chunks = min(cpu_count(), total_matrices)  # Число процессов для распараллеливания
    chunk_size = total_matrices // num_chunks
    
    # Параллельное создание матриц
    with Pool(num_chunks) as pool:
        chunk_ranges = [(rows, k, i * chunk_size, (i + 1) * chunk_size) for i in range(num_chunks)]
        chunk_matrices = pool.starmap(generate_matrix_chunk, chunk_ranges)
    
    all_matrices = [matrix for chunk in chunk_matrices for matrix in chunk]
    
    #print("Все матрицы параметров:")
    #for matrix in all_matrices:
     #   print(np.array(matrix))
    
    print("\nДопустимые матрицы параметров:")
    filtered_matrices = filter_equivalent_matrices(all_matrices)
    i = 0
    for matrix in filtered_matrices:
        i += 1
        print(np.array(matrix))
    print(f"Количество допустимых матриц: {i}")
    
    graph = create_generalized_petersen_graph(n, k_param)
    adj_matrix = adjacency_matrix(graph, n)
    print(f"\nМатрица смежности графа Петерсена П({n}, {k_param}):\n{adj_matrix}")
    print(f"\nСовершенные раскраски для графа Петерсена П({n}, {k_param}):")
    max_time = 15  # Максимальное время выполнения поиска (в секундах)
    
    perfect_colorings = []

    # Параллельный поиск совершенных раскрасок
   # begin_parallel_time = time.time()
    with Pool(num_chunks) as pool:
        tasks = [(graph, num_colors, matrix, max_time) for matrix in filtered_matrices]
        results = pool.map(parallel_find_perfect_coloring, tasks)
    #end_parallel_time = time.time()
    
    for (matrix, (is_perfect, coloring)) in zip(filtered_matrices, results):
        print(f"Матрица параметров:\n{np.array(matrix)}")
        if is_perfect:
            print("Эта матрица подходит для совершенной раскраски.\n")
            print(f"Раскраска:\n{coloring}")
            print(f"Матрица раскраски (построчно):")
            for vertex, color in enumerate(coloring):
                print(f"Вершина {vertex}: Цвет {color}")
            perfect_colorings.append(coloring)
        else:
            print("Эта матрица не подходит для совершенной раскраски.\n")
    
    print(time.time() - begin_time, " - время параллельного выполнения (без учета времени на отрисовку)")
    draw_graph(graph)
    for coloring in perfect_colorings:
        draw_graph(graph, coloring) 

if __name__ == "__main__":
    main()

