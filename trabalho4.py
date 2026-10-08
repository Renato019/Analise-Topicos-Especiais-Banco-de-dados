import csv
import re
import time
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi


def separar_palavras(texto):
    # deixa tudo minúsculo e tira a pontuação, para "jogo?" virar "jogo"
    return re.findall(r"\w+", texto.lower())


with open("dataset_futebol_10k.csv", "r", encoding="utf-8-sig") as arquivo:
    frases = [linha["frase"] for linha in csv.DictReader(arquivo)]

print(f"Frases lidas: {len(frases)}")


perguntas = [
    "Quem se machucou durante o jogo?",
    "O juiz usou a tecnologia para rever a jogada?",
    "Como os fãs mostraram insatisfação com o treinador?",
    "Quais equipes compraram novos atletas?",
    "Garotos que subiram para a equipe adulta",
    "O que o investidor da SAF fez?",
    "Quem conquistou a Copinha?",
    "Quem garantiu vaga na Libertadores?",
    "O que aconteceu com o time sub-17?",
    "O que a imprensa falou sobre o esquema tático?",
]

modelos = ["paraphrase-multilingual-MiniLM-L12-v2"]
# modelos = ["paraphrase-multilingual-MiniLM-L12-v2", "BAAI/bge-m3"]

# busca lexical: o BM25 recebe cada frase separada em palavras
bm25 = BM25Okapi([separar_palavras(frase) for frase in frases])


cliente = chromadb.Client()

for modelo in modelos:
    print(f"\nProcessando o modelo {modelo}...")
    nome_curto = modelo.replace("/", "-")

    funcao_embedding = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=modelo,
        device="cpu",
    )
    funcao_embedding(["aquecimento"]) # carrega o modelo antes de medir o tempo

    colecao = cliente.create_collection(
        name=f"frases-{nome_curto.lower()}",
        embedding_function=funcao_embedding,
    )

    inicio = time.time()
    tamanho_lote = 1000
    for i in range(0, len(frases), tamanho_lote):
        lote = frases[i:i + tamanho_lote]
        ids = [str(n) for n in range(i, i + len(lote))]
        colecao.add(documents=lote, ids=ids)

    tempo_indexacao = time.time() - inicio
    print(f"Frases adicionadas em {tempo_indexacao:.1f} segundos")

    resultado = colecao.query(
        query_texts=perguntas,
        n_results=5,
    )

    nome_saida = f"trabalho4-output-{nome_curto}.txt"
    with open(nome_saida, "w", encoding="utf-8") as saida:
        saida.write(f"Modelo: {modelo}\n")
        saida.write(f"Tempo para adicionar as frases: {tempo_indexacao:.1f} segundos\n\n")

        for p, pergunta in enumerate(perguntas):
            saida.write(f"{p + 1}. {pergunta}\n")

            saida.write("   Busca vetorial:\n")
            frases_encontradas = resultado["documents"][p]
            distancias = resultado["distances"][p]
            for frase, distancia in zip(frases_encontradas, distancias):
                saida.write(f"      - {frase}  (distância: {distancia:.4f})\n")

            saida.write("   Busca lexical (BM25):\n")
            frases_lexical = bm25.get_top_n(separar_palavras(pergunta), frases, n=5)
            for frase in frases_lexical:
                saida.write(f"      - {frase}\n")

            saida.write("\n")
    print(f"Arquivo gerado: {nome_saida}")