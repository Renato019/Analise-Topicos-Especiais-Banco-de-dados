import csv
import re
import time
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi


def separar_palavras(texto):
    # deixa tudo minúsculo e tira a pontuação, para "jogo?" virar "jogo"
    return re.findall(r"\w+", texto.lower())


def rrf(lista_lexical, lista_vetorial, k):
    # cada frase ganha 1 / (k + posição) em cada lista em que aparece
    pontos = {}
    for lista in (lista_lexical, lista_vetorial):
        for posicao, frase in enumerate(lista, start=1):
            pontos[frase] = pontos.get(frase, 0) + 1 / (k + posicao)
    return sorted(pontos, key=pontos.get, reverse=True)


def posicao(frase, lista):
    # em que lugar a frase ficou na lista ("-" se não apareceu)
    if frase in lista:
        return f"{lista.index(frase) + 1}º"
    return "-"


with open("dataset_futebol_10k.csv", "r", encoding="utf-8-sig") as arquivo:
    linhas = list(csv.DictReader(arquivo))

frases = [linha["frase"] for linha in linhas]
assunto = {linha["frase"]: linha["assunto"] for linha in linhas}

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

candidatos = 100  # quantas frases cada busca traz antes da união
valores_k = [1, 10, 60, 100, 1000]

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
        n_results=candidatos,
    )

    nome_saida = f"trabalho4-output-{nome_curto}.txt"
    with open(nome_saida, "w", encoding="utf-8") as saida:
        saida.write(f"Modelo: {modelo}\n")
        saida.write(f"Tempo para adicionar as frases: {tempo_indexacao:.1f} segundos\n\n")

        for p, pergunta in enumerate(perguntas):
            saida.write(f"{p + 1}. {pergunta}\n")

            lista_vetorial = resultado["documents"][p]
            distancias = resultado["distances"][p]
            lista_lexical = bm25.get_top_n(separar_palavras(pergunta), frases, n=candidatos)

            saida.write("   Busca vetorial:\n")
            for frase, distancia in zip(lista_vetorial[:5], distancias[:5]):
                saida.write(f"      - [{assunto[frase]}] {frase}  (distância: {distancia:.4f})\n")

            saida.write("   Busca lexical (BM25):\n")
            for frase in lista_lexical[:5]:
                saida.write(f"      - [{assunto[frase]}] {frase}\n")

            for k in valores_k:
                saida.write(f"   RRF com k = {k}:\n")
                for frase in rrf(lista_lexical, lista_vetorial, k)[:5]:
                    saida.write(
                        f"      - [{assunto[frase]}] {frase}  "
                        f"(lexical: {posicao(frase, lista_lexical)} | "
                        f"vetorial: {posicao(frase, lista_vetorial)})\n"
                    )

            saida.write("\n")
    print(f"Arquivo gerado: {nome_saida}")