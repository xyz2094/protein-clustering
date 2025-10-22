import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- Configurações ---
NPY_FILE = 'best_clustering_labels.npy'
OUTPUT_IMAGE = 'grafico_distribuicao_clusters_TOP20.png'
TOP_N = 20 # Número de clusters principais que queremos visualizar
# ---------------------

def plotar_distribuicao_filtrada(npy_file, output_image, top_n):
    """
    Carrega o .npy e gera um gráfico de barras legível,
    mostrando o Ruído (-1), os Top N clusters, e agrupando os demais em "Outros".
    """
    try:
        # 1. Carregar os dados
        data = np.load(npy_file)
    except FileNotFoundError:
        print(f"Erro: Arquivo '{npy_file}' não encontrado.")
        print("Certifique-se de que o script 'protein_clustering.py' foi executado.")
        return
    except Exception as e:
        print(f"Ocorreu um erro ao carregar o arquivo: {e}")
        return

    # 2. Contar a frequência e colocar em um DataFrame para facilitar
    cluster_ids, counts = np.unique(data, return_counts=True)
    df = pd.DataFrame({
        'cluster_id': cluster_ids,
        'count': counts
    }).sort_values(by='count', ascending=False)

    print("--- Análise da Distribuição dos Clusters ---")

    # 3. Lidar com o cluster de "Ruído" (Noise) - (ID -1)
    #    (Comum em DBSCAN/OPTICS)
    noise_points = 0
    if -1 in df['cluster_id'].values:
        noise_points = df[df['cluster_id'] == -1]['count'].iloc[0]
        print(f"Pontos de Ruído (Cluster -1): {noise_points}")
        # Remover o ruído do dataframe para focar nos clusters reais
        df = df[df['cluster_id'] != -1]

    # 4. Separar os Top N clusters
    top_n_df = df.head(top_n)
    
    # 5. Agrupar todos os outros clusters em "Outros"
    outros_count = df.iloc[top_n:]['count'].sum()
    
    print(f"Total de clusters reais encontrados: {len(df)}")
    print(f"Soma dos Top {top_n} clusters: {top_n_df['count'].sum()}")
    print(f"Soma dos {len(df) - top_n} clusters restantes (Outros): {outros_count}")

    # 6. Preparar dados para o gráfico
    
    # Adicionar o cluster de "Outros" se houver
    if outros_count > 0:
        outros_df = pd.DataFrame([{'cluster_id': 'Outros', 'count': outros_count}])
        plot_df = pd.concat([top_n_df, outros_df])
    else:
        plot_df = top_n_df

    # Adicionar o cluster de "Ruído" se houver (coloca no início)
    if noise_points > 0:
        noise_df = pd.DataFrame([{'cluster_id': 'Ruído (-1)', 'count': noise_points}])
        plot_df = pd.concat([noise_df, plot_df])
        
    # Garantir que os IDs sejam strings para plotagem
    plot_labels = plot_df['cluster_id'].astype(str)
    plot_counts = plot_df['count']

    # 7. Criar o Gráfico de Barras
    plt.figure(figsize=(15, 8)) # Figura mais larga
    bars = plt.bar(plot_labels, plot_counts, color='skyblue', edgecolor='black')
    
    plt.title(f'Distribuição de Sequências (Top {top_n} Clusters + Ruído e Outros)', fontsize=16)
    plt.xlabel('ID do Cluster Previsto', fontsize=12)
    plt.ylabel('Número de Sequências (Frequência)', fontsize=12)
    
    try:
        plt.bar_label(bars, fmt='%d', padding=3)
    except AttributeError:
        # Fallback para versões mais antigas do matplotlib
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, yval, ha='center', va='bottom')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    try:
        plt.savefig(output_image)
        print(f"\nSucesso! Gráfico filtrado salvo como: '{output_image}'")
        # plt.show()
    except Exception as e:
        print(f"Erro ao salvar o gráfico: {e}")


# --- Ponto de Entrada Principal ---
if __name__ == "__main__":
    plotar_distribuicao_filtrada(NPY_FILE, OUTPUT_IMAGE, TOP_N)