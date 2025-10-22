import numpy as np
import pandas as pd
from scipy import sparse
from itertools import product
# 
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.metrics import (silhouette_score, calinski_harabasz_score, davies_bouldin_score, 
                             f1_score, adjusted_rand_score, normalized_mutual_info_score, 
                             fowlkes_mallows_score)
from sklearn.cluster import (KMeans, MiniBatchKMeans, AgglomerativeClustering, 
                         DBSCAN, OPTICS, SpectralClustering, MeanShift, Birch)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from inspect import signature

warnings.filterwarnings('ignore')
np.random.seed(42)

class ProteinSequenceAnalyzer:
    def __init__(self, fasta_file, n_components=300):
        self.fasta_file = fasta_file
        self.n_components = n_components
        self.amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
        # MODIFICAÇÃO: Seu kmer é 2X2 (4-mer), não 2x2 com skip de 1. 
        # A sua função 'extract_2x2_kmers' gera 4-mers (ex: AACC)
        # O espaço de features deve ser 20^4
        self.kmer_patterns = [''.join(p) for p in product(self.amino_acids, repeat=4)]
        self.pattern_to_idx = {pattern: idx for idx, pattern in enumerate(self.kmer_patterns)}
        self.internal_metric_to_optimize = 'silhouette' # Métrica interna para Req. 5
        
    def process_fasta(self):
        """Process FASTA file without using biopython"""
        sequences = []
        classes = []
        
        with open(self.fasta_file, 'r') as f:
            current_seq = []
            current_class = None
            
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq and current_class:
                        sequences.append(''.join(current_seq).upper())
                        classes.append(current_class)
                    current_seq = []
                    # Pega a classe (segundo elemento, ex: a.1.1.1)
                    current_class = line.split()[1] 
                else:
                    current_seq.append(line)
            
            if current_seq and current_class:
                sequences.append(''.join(current_seq).upper())
                classes.append(current_class)
        
        self.df = pd.DataFrame({
            'sequence': sequences,
            'class': classes
        })
        
        print(f"Total sequences: {len(self.df)}")
        print(f"Total classes: {self.df['class'].nunique()}")
        return self.df
    
    def extract_2x2_kmers(self, sequence, skip_size=1):
        """
        Extrai kmer 2X2 com skip.
        Ex: kmer = 'AA' + 'CC' de 'AAXCC'
        """
        kmers = set()
        # A janela móvel precisa de 4 caracteres + skip
        for i in range(len(sequence) - 3 - skip_size):
            # Pega 2 (i, i+1), pula 'skip_size', pega 2 (i+2+skip, i+3+skip)
            kmer = sequence[i:i+2] + sequence[i+2+skip_size:i+4+skip_size]
            if len(kmer) == 4 and all(aa in self.amino_acids for aa in kmer):
                kmers.add(kmer)
        return kmers
    
    def create_feature_matrix(self, skip_size=1):
        """Create binary feature matrix from sequences"""
        n_samples = len(self.df)
        n_features = len(self.kmer_patterns)
        rows, cols = [], []
        
        for idx, seq in enumerate(tqdm(self.df['sequence'], desc="Processing k-mers")):
            kmers = self.extract_2x2_kmers(seq, skip_size)
            for kmer in kmers:
                if kmer in self.pattern_to_idx:
                    rows.append(idx)
                    cols.append(self.pattern_to_idx[kmer])
        
        data = np.ones(len(rows))
        self.X = sparse.csr_matrix((data, (rows, cols)), 
                               shape=(n_samples, n_features),
                               dtype=np.float32)
        
        print(f"Feature matrix shape: {self.X.shape}")
        print(f"Matrix density: {self.X.nnz / (self.X.shape[0] * self.X.shape[1]):.4f}")
        return self.X
    
    def apply_pca(self):
        """Apply PCA (TruncatedSVD) with n_components"""
        # Garantir que não pedimos mais componentes do que o possível
        max_components = min(self.n_components, min(self.X.shape) - 1)
        if max_components < 1:
            print(f"Aviso: Número de componentes ({self.n_components}) é muito alto para os dados ({self.X.shape}).")
            print(f"Reduzindo n_components para {max_components}")
            if max_components < 1:
                raise ValueError(f"Muito poucas amostras/features para PCA (X.shape={self.X.shape})")
        else:
            max_components = self.n_components

        # Usar TruncatedSVD para matrizes esparsas
        self.svd = TruncatedSVD(n_components=max_components, random_state=42)
        
        # Normalizar dados ANTES do SVD é comum (TF-IDF L2, mas aqui L1)
        X_normalized = normalize(self.X, norm='l1', axis=1)
        
        self.Z = self.svd.fit_transform(X_normalized)
        
        print(f"PCA projection shape: {self.Z.shape}")
        
        # Calcular variância explicada
        self._explained_var_ratio = self.svd.explained_variance_ratio_
        explained_var = self._explained_var_ratio.sum()
        print(f"Cumulative explained variance: {explained_var:.4f}")

        return self.Z
    
    def evaluate_clustering(self, labels, true_labels, data=None):
        """
        Calcula métricas de clustering internas e externas.
        """
        try:
            if data is None:
                data = self.Z
            
            # Verificar se foi encontrado mais de 1 cluster (necessário para métricas)
            n_labels = len(set(labels))
            if n_labels <= 1:
                print("  (Apenas 1 cluster encontrado, métricas não aplicáveis)")
                return None
                
            # Métricas Internas (Req 4 e 5)
            internal_metrics = {
                'silhouette': silhouette_score(data, labels),
                'calinski_harabasz': calinski_harabasz_score(data, labels),
                'davies_bouldin': davies_bouldin_score(data, labels)
            }
            
            # Métricas Externas (Req 4)
            external_metrics = {
                'f1_macro': f1_score(true_labels, labels, average='macro'), # O F1-Score que você pediu
                'ari': adjusted_rand_score(true_labels, labels), # Padrão-ouro
                'nmi': normalized_mutual_info_score(true_labels, labels), # Padrão-ouro
                'fms': fowlkes_mallows_score(true_labels, labels) # "F1-like"
            }
            
            # Juntar dicionários
            metrics = {**internal_metrics, **external_metrics}
            return metrics
            
        except Exception as e:
            print(f"  (Erro ao calcular métricas: {e})")
            return None
    
    def run_clustering_algorithms(self):
        """
        Req 4: Roda múltiplos algoritmos e correlaciona métricas internas/externas.
        """
        le = LabelEncoder()
        true_labels = le.fit_transform(self.df['class'])
        n_clusters = len(np.unique(true_labels))
        print(f"Número real de classes (k_true): {n_clusters}")
        
        # Lista expandida de algoritmos do scikit-learn
        algorithms = {
            'KMeans': KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
            'MiniBatchKMeans': MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=10),
            'AgglomerativeClustering': AgglomerativeClustering(n_clusters=n_clusters),
            'Birch': Birch(n_clusters=n_clusters),
            'SpectralClustering': SpectralClustering(n_clusters=n_clusters, random_state=42, 
                                                     eigen_solver='dense', n_init=10, 
                                                     affinity='nearest_neighbors'),
            'MeanShift': MeanShift(), # Encontra k automaticamente
            'DBSCAN': DBSCAN(eps=0.5), # Encontra k automaticamente
            'OPTICS': OPTICS(min_samples=5) # Encontra k automaticamente
        }
        
        results = {}
        for name, algo in algorithms.items():
            print(f"\nRunning {name}...")
            try:
                labels = algo.fit_predict(self.Z)
                metrics = self.evaluate_clustering(labels, true_labels)
                if metrics:
                    results[name] = metrics
                    results[name]['n_clusters_found'] = len(set(labels))
            except Exception as e:
                print(f"Error in {name}: {str(e)}")
        
        self.clustering_results = pd.DataFrame(results).T.dropna()
        
        # --- CUMPRINDO REQUISITO 4 ---
        print("\n--- Req 4: Resultados da Comparação de Algoritmos ---")
        print(self.clustering_results.to_markdown(floatfmt=".3f"))
        
        print(f"\n--- Req 4: Correlação Métricas Internas vs. Externas ---")
        internal = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
        external = ['f1_macro', 'ari', 'nmi', 'fms']
        
        correlation_matrix = self.clustering_results[internal + external].corr()
        
        # Mostrar apenas a correlação entre os dois grupos
        correlation_internal_external = correlation_matrix.loc[internal, external]
        print(correlation_internal_external.to_markdown(floatfmt=".3f"))
        print("\n(Observação: Correlação ideal é alta [próxima de 1.0] ou baixa [próxima de -1.0])")
        
        return self.clustering_results
    
    def optimize_best_algorithm(self, subsample_size=1000):
        """
        Req 5: Varia parâmetros para obter a melhor MÉTRICA INTERNA.
        """
        
        # --- CUMPRINDO REQUISITO 5 ---
        print(f"\n--- Req 5: Otimizando com base na Métrica Interna ({self.internal_metric_to_optimize}) ---")

        le = LabelEncoder()
        true_labels = le.fit_transform(self.df['class'])
        n_clusters_true = len(np.unique(true_labels))

        # Filtrar por algoritmos que podemos otimizar (que aceitam n_clusters, etc.)
        optimizable_algos = ['KMeans', 'MiniBatchKMeans', 'AgglomerativeClustering', 'Birch']
        valid_results = self.clustering_results.loc[
            self.clustering_results.index.isin(optimizable_algos)
        ]
        
        if valid_results.empty:
            print("Nenhum algoritmo otimizável (KMeans, Birch, etc.) teve sucesso.")
            return None, None, None

        # Escolher melhor algoritmo com base na MÉTRICA INTERNA
        if self.internal_metric_to_optimize == 'silhouette' or self.internal_metric_to_optimize == 'calinski_harabasz':
            best_algo_name = valid_results[self.internal_metric_to_optimize].idxmax() # Maximizar
            best_score_internal = valid_results[self.internal_metric_to_optimize].max()
        else: # davies_bouldin
            best_algo_name = valid_results[self.internal_metric_to_optimize].idxmin() # Minimizar
            best_score_internal = valid_results[self.internal_metric_to_optimize].min()
            
        print(f"Melhor algoritmo (baseado em {self.internal_metric_to_optimize}): {best_algo_name} (Score: {best_score_internal:.3f})")
        
        # Subamostragem para otimização rápida
        if subsample_size and subsample_size < len(self.Z):
            print(f"Usando {subsample_size} amostras para otimização...")
            indices = np.random.choice(len(self.Z), subsample_size, replace=False)
            Z_opt = self.Z[indices]
            true_labels_opt = true_labels[indices]
        else:
            Z_opt = self.Z
            true_labels_opt = true_labels
            print("Usando dataset completo para otimização...")
        
        # Grades de parâmetros expandidas
        param_grid = {}
        base_model = None
        
        k_range = [n_clusters_true-1, n_clusters_true, n_clusters_true+1]
        k_range = [k for k in k_range if k > 1] # Garantir k > 1

        if best_algo_name == 'KMeans':
            param_grid = {'n_clusters': k_range, 'init': ['k-means++', 'random'], 'n_init': [5, 10]}
            base_model = KMeans
        elif best_algo_name == 'AgglomerativeClustering':
            param_grid = {'n_clusters': k_range, 'linkage': ['ward', 'average', 'complete']}
            base_model = AgglomerativeClustering
        elif best_algo_name == 'Birch':
            param_grid = {'n_clusters': k_range, 'threshold': [0.3, 0.5, 0.7], 'branching_factor': [30, 50]}
            base_model = Birch
        else:  # MiniBatchKMeans
            param_grid = {'n_clusters': k_range, 'init': ['k-means++', 'random'], 'batch_size': [256, 512, 1024]}
            base_model = MiniBatchKMeans
        
        # Configurar best_score inicial para otimização
        # Maximizar Silhouette/Calinski, Minimizar Davies-Bouldin
        best_score = -np.inf if self.internal_metric_to_optimize != 'davies_bouldin' else np.inf
        
        best_params = None
        best_labels_full = None
        
        print(f"Iniciando Grid Search para {best_algo_name} (Otimizando {self.internal_metric_to_optimize})...")
        params_list = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
        
        for current_params in tqdm(params_list):
            try:
                # Filtrar kwargs que o modelo aceita
                sig = signature(base_model)
                supported = set(sig.parameters.keys())
                kwargs = {k: v for k, v in current_params.items() if k in supported}
                if 'random_state' in supported:
                    kwargs['random_state'] = 42

                model = base_model(**kwargs)
                labels = model.fit_predict(Z_opt)
                metrics = self.evaluate_clustering(labels, true_labels_opt, Z_opt)
                
                if metrics:
                    current_internal_score = metrics[self.internal_metric_to_optimize]
                    
                    # Lógica de atualização baseada na métrica interna
                    is_better = False
                    if self.internal_metric_to_optimize == 'davies_bouldin':
                        if current_internal_score < best_score: is_better = True
                    else:
                        if current_internal_score > best_score: is_better = True

                    if is_better:
                        best_score = current_internal_score
                        best_params = current_params
                        
                        # Refit no dataset completo com os melhores parâmetros
                        print(f"\n  Novo melhor score ({self.internal_metric_to_optimize}): {best_score:.3f}")
                        print(f"  Refitting no dataset completo com params: {best_params}")
                        model_full = base_model(**kwargs) # Re-inicializa
                        best_labels_full = model_full.fit_predict(self.Z)

            except Exception as e:
                # print(f"Erro com parâmetros {current_params}: {str(e)}")
                continue
        
        print("\n--- Req 5: Sugestão de Melhor Configuração ---")
        print(f"Otimização concluída para o algoritmo: {best_algo_name}")
        print(f"Melhores parâmetros encontrados (baseado em {self.internal_metric_to_optimize}):")
        print(best_params)
        print(f"Melhor Score Interno ({self.internal_metric_to_optimize}): {best_score:.4f}")
        
        # Mostrar métricas externas da melhor configuração
        if best_labels_full is not None:
            print("\nMétricas Externas (Verdade) para esta configuração:")
            final_metrics = self.evaluate_clustering(best_labels_full, true_labels, self.Z)
            print(f"  F1-Macro: {final_metrics['f1_macro']:.3f}")
            print(f"  ARI:      {final_metrics['ari']:.3f}")
            print(f"  NMI:      {final_metrics['nmi']:.3f}")

        return best_params, best_score, best_labels_full
    
    def plot_pca_variance(self):
        """Plot cumulative explained variance"""
        if not hasattr(self, '_explained_var_ratio'):
            print("Variância do PCA não calculada. Rode 'apply_pca' primeiro.")
            return

        plt.figure(figsize=(10, 5))
        arr = getattr(self, '_explained_var_ratio', None)
        
        if arr is None or not np.isfinite(arr).any():
            plt.text(0.5, 0.5, 'Variância explicada não disponível', ha='center', va='center')
        else:
            cumulative_variance = np.cumsum(np.nan_to_num(arr))
            plt.plot(cumulative_variance)
            plt.axhline(y=0.9, color='r', linestyle='--', label='90% Variância')
            
            # Tentar encontrar onde 90% é atingido
            try:
                comps_90 = np.where(cumulative_variance >= 0.9)[0][0] + 1
                plt.axvline(x=comps_90, color='r', linestyle='--', label=f'{comps_90} Componentes')
                plt.legend()
            except IndexError:
                pass # Não atingiu 90%
                
        plt.xlabel('Número de Componentes')
        plt.ylabel('Variância Explicada Cumulativa')
        plt.title('Variância Explicada (TruncatedSVD)')
        plt.grid(True)
        plt.savefig('pca_variance.png')
        plt.close()
        print("Gráfico 'pca_variance.png' salvo.")

def main():
    # Inicializar analisador
    fasta_file = "astral-scopdom-seqres-gd-sel-95-2.08.fa"
    analyzer = ProteinSequenceAnalyzer(fasta_file, n_components=300)
    
    # 1. Processar dados
    print("1. Processando arquivo FASTA...")
    analyzer.process_fasta()
    
    # 2. Criar matriz de features
    print("\n2. Criando matriz de k-mer (skip=1)...")
    analyzer.create_feature_matrix(skip_size=1)
    
    # 3. Aplicar PCA
    print("\n3. Aplicando PCA (TruncatedSVD)...")
    analyzer.apply_pca()
    analyzer.plot_pca_variance()
    
    # 4. Rodar algoritmos e correlacionar
    print("\n4. Rodando algoritmos e correlacionando métricas...")
    results = analyzer.run_clustering_algorithms()
    
    # 5. Otimizar parâmetros (com base em métrica interna)
    print("\n5. Otimizando melhor algoritmo (baseado em métrica interna)...")
    best_params, best_score, best_labels = analyzer.optimize_best_algorithm(subsample_size=2000)
    
    # Salvar resultados
    results.to_csv('clustering_results.csv')
    if best_labels is not None:
        np.save('best_clustering_labels.npy', best_labels)
        print("\nRótulos do melhor cluster salvos em 'best_clustering_labels.npy'")

    
    print("\nAnálise completa! Resultados salvos em:")
    print("- clustering_results.csv")
    print("- pca_variance.png")

if __name__ == "__main__":
    main()