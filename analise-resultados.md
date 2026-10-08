# Trabalho 4 (TEBD) — Resultados finais

Resultados da execução de 07/10/2026 feita por Renato com o `trabalho4.py` construído em etapas (versão final, 158 linhas) e o dataset `dataset_futebol_10k.csv` (10.000 frases únicas de futebol, 8 assuntos, 436 palavras diferentes). Apresentação em 09/10/2026.

## Como os acertos foram contados

Para cada pergunta, contou-se quantas das 5 frases devolvidas respondem à pergunta (máximo 5 por pergunta, 50 no total). O critério é por palavra-chave, definido pelo Claude; é aproximado e vale conferir lendo as frases.

| Pergunta | Conta como acerto se a frase tiver |
| --- | --- |
| 1. Quem se machucou durante o jogo? | "lesão" |
| 2. O juiz usou a tecnologia para rever a jogada? | "revisou", "monitor", "revisão" ou "árbitro de vídeo" |
| 3. Como os fãs mostraram insatisfação com o treinador? | assunto Torcida com "vaiou", "protesto" ou "saída do técnico" |
| 4. Quais equipes compraram novos atletas? | assunto Mercado da bola com "contratação", "reforço", "luvas", "transferência" ou "empréstimo" |
| 5. Garotos que subiram para a equipe adulta | "promovido ao elenco principal", "estreou no time profissional" ou "contrato profissional" |
| 6. O que o investidor da SAF fez? | "SAF" |
| 7. Quem conquistou a Copinha? | "Copinha" |
| 8. Quem garantiu vaga na Libertadores? | "vaga na Libertadores" |
| 9. O que aconteceu com o time sub-17? | "sub-17" |
| 10. O que a imprensa falou sobre o esquema tático? | assunto Imprensa com "esquema tático" |

## Tempo para processar as 10 mil frases

| Modelo | Tempo |
| --- | --- |
| paraphrase-multilingual-MiniLM-L12-v2 | 27,6 s |
| BAAI/bge-m3 | 1.146,2 s (cerca de 19 minutos, 40 vezes mais) |

## Buscas sozinhas: acertos por pergunta (de 5)

| Pergunta | Lexical (BM25) | Vetorial MiniLM | Vetorial BGE-M3 |
| --- | --- | --- | --- |
| 1 | 0 | 3 | 5 |
| 2 | 5 | 4 | 5 |
| 3 | 0 | 0 | 5 |
| 4 | 0 | 0 | 0 |
| 5 | 0 | 3 | 5 |
| 6 | 5 | 5 | 5 |
| 7 | 5 | 4 | 5 |
| 8 | 5 | 5 | 5 |
| 9 | 5 | 5 | 5 |
| 10 | 5 | 3 | 5 |
| **Total (de 50)** | **30** | **32** | **45** |

## Todas as configurações: total de acertos (de 50)

| Configuração | MiniLM | BGE-M3 |
| --- | --- | --- |
| Só lexical | 30 | 30 |
| Só vetorial | 32 | 45 |
| RRF k = 1 | 31 | 36 |
| RRF k = 10 | 32 | 36 |
| RRF k = 60 | 30 | 36 |
| RRF k = 100 | 30 | 37 |
| RRF k = 1000 | 30 | 37 |
| Média ponderada, peso vetorial 0,00 | 30 | 30 |
| Média ponderada, peso vetorial 0,25 | 30 | 30 |
| Média ponderada, peso vetorial 0,50 | 32 | 38 |
| Média ponderada, peso vetorial 0,75 | 34 | 45 |
| Média ponderada, peso vetorial 1,00 | 32 | 45 |

## Conclusões

1. **Lexical**: acerta tudo quando a pergunta usa palavras que estão nas frases (2, 6, 7, 8, 9, 10) e zera quando usa palavras diferentes (1, 3, 4, 5). Exemplo: para "Quem se machucou durante o jogo?" trouxe "O atual campeão empatou o jogo de ida durante a pré-temporada".
2. **Vetorial**: o BGE-M3 foi bem melhor que o MiniLM (45 contra 32), mas 40 vezes mais lento. Na pergunta 3, o BGE-M3 trouxe "o torcedor pediu a saída do técnico"; o MiniLM trouxe "o capitão do time deu um drible desconcertante".
3. **Pergunta 4**: praticamente ninguém acertou. A vetorial se prendeu a "novos" ("novo patrocinador", "novo uniforme", "novo talento").
4. **RRF**: ficou entre as duas buscas. Como dá o mesmo valor às duas listas, as frases erradas da lexical entram no resultado (BGE-M3 caiu de 45 para 36).
5. **Efeito do k**: pequeno (MiniLM de 30 a 32, BGE-M3 de 36 a 37). Nas perguntas em que as buscas discordam, nenhuma frase aparece nas duas listas, então o RRF só intercala uma de cada para qualquer k. Onde mudou: MiniLM, pergunta 2, de 4 acertos (k = 1 e 10) para 2 (k de 60 em diante), porque k grande favorece frases medianas nas duas listas ("marcou o impedimento").
6. **Efeito do peso**: grande (de 30 a 45 no BGE-M3). Até peso 0,25 o resultado é igual ao da lexical; o melhor foi 0,75.
7. **A união ajudou?** Com o MiniLM, sim: peso 0,75 deu 34, mais que cada busca sozinha (30 e 32). Exemplo: pergunta 7, a lexical tirou a frase errada "venceu por goleada". Com o BGE-M3, não: a união no máximo empatou com a vetorial sozinha (45).

## Explicação usada com o Renato (linguagem simples)

- Dois "jurados": o das palavras (lexical) e o do sentido (vetorial).
- RRF = campeonato por pontos pela colocação; k = o quanto vale ser o primeiro.
- Média ponderada = usa as notas, na mesma escala (normalização), e o peso diz quanto vale a opinião de cada jurado.

## Limitações

- São só 10 perguntas; diferenças de 1 ou 2 acertos não significam muito.
- Dataset com vocabulário pequeno e frases montadas por combinação; muitas frases parecidas, empates no BM25 e algumas frases sem sentido ("O olheiro conquistou a Copinha").
- O critério de acerto é por palavra-chave.
