# Busca lexical, vetorial e híbrida

Trabalho 4 da disciplina **Tópicos Especiais em Banco de Dados (TEBD)**.

O trabalho compara três formas de buscar frases em uma base de 10 mil frases sobre futebol:

- **Busca lexical**: BM25, que procura as palavras da pergunta.
- **Busca vetorial**: ChromaDB com embeddings, que procura frases com o mesmo sentido.
- **Busca híbrida**: junta as duas listas por **RRF** (Reciprocal Rank Fusion) ou por **média ponderada**.

Também compara dois modelos de embedding: `paraphrase-multilingual-MiniLM-L12-v2` e `BAAI/bge-m3`.

## Arquivos do repositório

| Arquivo | O que é |
| --- | --- |
| [`trabalho4.py`](trabalho4.py) | Código final, com todas as buscas |
| [`dataset_futebol_10k.csv`](dataset_futebol_10k.csv) | 10.000 frases únicas (colunas `id`, `assunto`, `frase`) |
| [`etapa1-resultado.txt`](etapa1-resultado.txt) | Saída da etapa 1 |
| [`etapa2-resultado.txt`](etapa2-resultado.txt) | Saída da etapa 2 |
| [`etapa3-resultado.txt`](etapa3-resultado.txt) | Saída da etapa 3 |
| [`etapa4-resultado.txt`](etapa4-resultado.txt) | Saída da etapa 4 |
| [`etapa5-resultado-MiniLM-L12-v2.txt`](etapa5-resultado-MiniLM-L12-v2.txt) | Saída final com o modelo MiniLM |
| [`etapa5-resultado-BAAI-bge-m3.txt`](etapa5-resultado-BAAI-bge-m3.txt) | Saída final com o modelo BGE-M3 |
| [`analise-resultados.md`](analise-resultados.md) | Relatório com a contagem de acertos e as conclusões |

As frases do dataset estão divididas em 8 assuntos: Partida, Torcida, Arbitragem, Imprensa, Treinamento, Mercado da bola, Campeonato e Categorias de base.

## Etapas

O código foi construído em etapas, e cada uma tem o seu arquivo de resultado:

1. Busca vetorial no ChromaDB com o modelo MiniLM.
2. Busca lexical com BM25, ao lado da vetorial.
3. União das duas listas por RRF, testando vários valores de `k`. O assunto de cada frase passa a aparecer na saída.
4. União por média ponderada das notas, testando vários pesos.
5. Repetição de tudo com um segundo modelo, o `BAAI/bge-m3`.

## Como funciona

Para cada pergunta, cada busca traz as 100 frases mais próximas. As listas são unidas e as 5 primeiras frases vão para o arquivo de saída.

**Busca lexical (BM25).** Cada frase é colocada em minúsculas e separada em palavras, sem pontuação (`re.findall(r"\w+", texto.lower())`). O índice é feito com `BM25Okapi`, da biblioteca `rank_bm25`.

**Busca vetorial.** As frases são guardadas numa coleção do ChromaDB, que gera os embeddings com `SentenceTransformerEmbeddingFunction` (rodando na CPU). A busca devolve as frases e a distância de cada uma até a pergunta.

**RRF.** Cada frase ganha pontos pela posição em que aparece em cada lista:

```
pontos(frase) = soma de 1 / (k + posição), para cada lista em que a frase aparece
```

Valores testados: `k` = 1, 10, 60, 100 e 1000.

**Média ponderada.** As notas de cada busca são colocadas entre 0 e 1 (a maior vira 1 e a menor vira 0). Na vetorial, a nota é `1 − distância`. Depois:

```
pontos(frase) = peso × nota_vetorial + (1 − peso) × nota_lexical
```

Pesos testados para a busca vetorial: 0, 0,25, 0,5, 0,75 e 1.

## Como executar

Requisitos: Python 3.10 ou mais novo (testado com o 3.12).

```bash
pip install chromadb rank_bm25 sentence-transformers
python trabalho4.py
```

O script deve ser executado na mesma pasta do `dataset_futebol_10k.csv`. Ele gera um arquivo por modelo:

- `trabalho4-output-paraphrase-multilingual-MiniLM-L12-v2.txt`
- `trabalho4-output-BAAI-bge-m3.txt`

Os arquivos `etapa*-resultado*.txt` deste repositório são essas saídas, renomeadas.

> **Atenção:** na primeira execução os modelos são baixados da internet. Na CPU, o MiniLM levou cerca de 28 segundos para processar as 10 mil frases, e o BGE-M3 levou cerca de 19 minutos.

## Perguntas de teste

1. Quem se machucou durante o jogo?
2. O juiz usou a tecnologia para rever a jogada?
3. Como os fãs mostraram insatisfação com o treinador?
4. Quais equipes compraram novos atletas?
5. Garotos que subiram para a equipe adulta
6. O que o investidor da SAF fez?
7. Quem conquistou a Copinha?
8. Quem garantiu vaga na Libertadores?
9. O que aconteceu com o time sub-17?
10. O que a imprensa falou sobre o esquema tático?

Várias perguntas usam palavras que não estão nas frases (por exemplo, "machucou" em vez de "lesão"), para testar se a busca entende o sentido.

## Resultados

Para cada pergunta, contou-se quantas das 5 frases devolvidas respondem à pergunta. O máximo é 50 acertos (10 perguntas × 5 frases). O critério usado para cada pergunta está em [`analise-resultados.md`](analise-resultados.md).

| Configuração | MiniLM | BGE-M3 |
| --- | --- | --- |
| Só lexical (BM25) | 30 | 30 |
| Só vetorial | 32 | **45** |
| Melhor RRF | 32 (k = 10) | 37 (k = 100 e 1000) |
| Melhor média ponderada | **34** (peso 0,75) | **45** (peso 0,75 e 1,0) |

| Modelo | Tempo para processar as 10 mil frases |
| --- | --- |
| paraphrase-multilingual-MiniLM-L12-v2 | 27,6 s |
| BAAI/bge-m3 | 1.146,2 s (cerca de 19 min) |

## Conclusões

- **A busca lexical** acerta tudo quando a pergunta usa as mesmas palavras das frases e erra tudo quando usa outras palavras.
- **O BGE-M3** acertou bem mais que o MiniLM (45 contra 32), mas foi cerca de 40 vezes mais lento.
- **O RRF** ficou entre as duas buscas. Ele dá o mesmo valor às duas listas, então as frases erradas da lexical entram no resultado.
- **O peso** da média ponderada mudou muito mais o resultado do que o `k` do RRF. O melhor peso foi 0,75 para a busca vetorial.
- **A busca híbrida** ajudou com o MiniLM (34 acertos, mais que cada busca sozinha). Com o BGE-M3, no máximo empatou com a vetorial sozinha.
- **A pergunta 4** ("Quais equipes compraram novos atletas?") não foi respondida por nenhuma das buscas sozinhas.

A tabela completa, pergunta por pergunta, está em [`analise-resultados.md`](analise-resultados.md).

## Limitações

- São só 10 perguntas, então diferenças de 1 ou 2 acertos não significam muito.
- O dataset tem vocabulário pequeno e frases montadas por combinação, com muitas frases parecidas.
- O acerto foi contado por palavra-chave, o que é uma aproximação.

## Autor

Renato ([@Renato019](https://github.com/Renato019))
