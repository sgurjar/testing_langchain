import arxiv

def download_papers(paper_ids: list):
    search = arxiv.Search(id_list=paper_ids)
    for result in search.results():
        # This saves each PDF to your local directory
        file_path = result.download_pdf(dirpath="./research_papers")
        print(f"Downloaded: {result.title} to {file_path}")

paper_ids = [
"2512.24880",
"2505.23735",
"2410.05258",
"2510.05364",
"2601.02360",
"2503.01124",
"2503.04112",
"2501.13353",
"2510.03989",
"2508.09834",
]

download_papers(paper_ids)

# Paper Title                                    | arXiv ID    | Key Contribution
# mHC: Manifold-Constrained Hyper-Connections    | 2512.24880  | Introduces "lane" expansion in residual streams to stabilize scaling and improve mixing.
# ATLAS: Learning to Optimally Memorize Context  | 2505.23735  | Proposes "DeepTransformers" that generalize the architecture to better memorize 10M+ context lengths.
# Differential Transformer (DIFF Transformer)    | 2410.05258  | Uses a subtraction-based attention map to cancel "attention noise" and improve retrieval.
# The End of Transformers? On Sub-Quadratic Rise | 2510.05364  | A systematic review of hybrids (SSMs, RNNs) that aim to replace pure attention.
# Heterogeneous Low-Bandwidth Pre-Training       | 2601.02360  | Focuses on architectural changes that allow training across decentralized, low-speed networks.
# Transformers without Normalization             | 2503.01124* | Introduces Dynamic Tanh (DyT) to replace LayerNorm/RMSNorm for faster training.
# The Forgetting Transformer (FoX)               | 2503.04112* | Integrates "forget gates" into softmax attention to handle irrelevant long-context data.
# Contrast: A Hybrid of Transformers and SSMs    | 2501.13353  | Combines Transformer global modeling with Mamba-like linear complexity for vision/language.
# A Mathematical Explanation of Transformers     | 2510.03989  | Interprets the architecture as a discretization of an integro-differential equation.
# Speed Always Wins: A Survey on Efficient LLMs  | 2508.09834  | Examines the latest optimizations in FFN layers and attention sparsification for 2025.
