# Lab 2 — Relatório individual
Todos os números abaixo foram produzidos pelo `lab02.ipynb` executado do zero (reiniciar kernel → executar tudo). Tabelas e figuras estão em `resultados/`.

## Reprodução
- **Python 3.13.16**; numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, matplotlib 3.11.2, torch 2.14.1 (build CUDA, executado só em CPU).
- **Sistema/dispositivo:** Linux x86-64, Intel Xeon @ 2,10 GHz, 2 threads do PyTorch, `DISPOSITIVO='cpu'`.
- **Semente:** `SEED = 42` (partições, embaralhamento, inicialização). A Parte C usa também as sementes declaradas `[0, 1, 2, 3, 4]`.
- **Execução:** `python -m venv .venv` → ativar → `pip install -r requirements.txt` → abrir `lab02.ipynb` → *Restart & Run All*. Tempo total ≈ 1 min em CPU.
- **Modificações em `apoio.py`:** nenhuma. O CSV é o original (SHA-256 conferido por `carregar_base`).
- Os resultados de modelo foram idênticos em duas execuções completas; só os tempos da A3 variam entre execuções (são medições de relógio).

## Base e definição do alvo
Red Wine Quality (Cortez et al., 2009), cópia da UCI incluída em `dados/` (CC BY 4.0; atribuição em `dados/FONTE-E-LICENCA.md`). Unidade de observação: uma amostra de vinho tinto. Entradas: **somente as 11 medidas físico-químicas** de `FEATURES`. Alvo: `alvo = (quality >= 7)` → classe 1 = 217 de 1599 linhas (13,57%). `quality`, `alvo`, `id_linha` e `grupo_perfil` nunca entram no modelo.

**Limitações:** 240 linhas repetem perfis inteiros (1359 perfis distintos); perfil idêntico não prova que é o mesmo vinho físico. Não há data, produtor ou região, então não é possível testar generalização temporal, para novos produtores ou outros domínios (vinho branco, outras regiões). A fronteira nota ≥ 7 é uma convenção do laboratório.

## B1 — Cinco cenários
1. **Avaliações de produtos (50 mil textos, 3 mil usuários).** Uso pressuposto: classificar avaliações de usuários que o sistema ainda não viu. Adotaria divisão **por usuário** (GroupKFold/GroupShuffleSplit com `usuario_id`). Numa divisão aleatória, o vocabulário, estilo e viés de nota de cada usuário (≈17 textos por pessoa) aparecem no treino e no teste, e o modelo pode acertar "quem escreveu" em vez de "o que foi dito". Se o uso for só para usuários já cadastrados, uma divisão temporal dentro de cada usuário seria defensável; declararia essa hipótese.
2. **Turbinas (20 turbinas, leituras por minuto, 2 anos, falha em 24 h).** Uso: prever falhas futuras. Adotaria divisão **temporal** (treino nos meses iniciais, validação e teste em períodos posteriores), com uma **lacuna ≥ 24 h** entre blocos porque o rótulo olha 24 h à frente; se o objetivo incluir turbinas novas, também reservaria turbinas inteiras. Numa divisão aleatória, minutos vizinhos quase idênticos e a própria janela da falha ficam dos dois lados, e o rótulo do treino "enxerga" o mesmo evento do teste.
3. **Spam (200 mil e-mails, 3% spam).** Sem informação de remetente ou data, uma divisão **aleatória estratificada** é defensável, porque a hipótese é que os e-mails são aproximadamente independentes. Hipóteses que mudariam isso: campanhas com e-mails quase duplicados (deduplicar ou agrupar por campanha/remetente) e mudança de padrão ao longo do tempo (spam evolui; uma divisão temporal estima melhor o uso futuro). A estratificação garante ≈ 3% de spam em cada conjunto (≈ 1.200 no teste).
4. **Radiografias (8 mil imagens de 900 pacientes).** Uso: diagnosticar pacientes novos. Divisão **por paciente** (todas as imagens de uma pessoa num único conjunto). Numa divisão aleatória, anatomia, implantes, marcadores do equipamento e o próprio achado crônico do mesmo paciente aparecem no treino e no teste, inflando a métrica. Se houver vários hospitais, um teste com hospital reservado mediria a generalização para outro equipamento.
5. **Cartões (2019–2026, 500 mil clientes, fraude).** **Tempo:** o modelo será usado no futuro, e os padrões de fraude mudam; adotaria treino até uma data, validação no período seguinte e teste no mais recente (ex.: 2026), sem usar variáveis calculadas com informação posterior à transação (chargebacks confirmados depois, agregados com janelas futuras). **Identidade:** se o objetivo é atender também **clientes novos**, reservaria clientes inteiros no teste (cruzando os dois critérios: clientes não vistos *e* período posterior); se o uso é só com a carteira conhecida, o histórico do mesmo cliente no treino é legítimo, desde que anterior no tempo. Uma divisão aleatória mistura futuro e passado do mesmo cliente e produz uma estimativa otimista para ambos os usos.

## B2 — Auditoria da separação
**Pergunta de uso:** dada uma amostra com as 11 medidas, prever se ela terá nota ≥ 7, para amostras cujo perfil de medidas ainda não foi visto. **Estratégia:** `GroupShuffleSplit` com grupos = `grupo_perfil`: 1º corte reserva 20% dos grupos para teste; 2º corte separa 25% dos grupos restantes para validação (0,25 × 0,80 = 0,20). Semente 42, escolhida antes e não ajustada pela métrica. Sem estratificação pelo alvo; as classes foram conferidas apenas como validação do desenho.

| Conjunto | Linhas | % das linhas | Positivos | Proporção positiva | Perfis | Repetições internas |
|---|---:|---:|---:|---:|---:|---:|
| Treino | 961 | 60,10% | 139 | 0,1446 | 815 | 146 |
| Validação | 318 | 19,89% | 33 | 0,1038 | 272 | 46 |
| Teste | 320 | 20,01% | 45 | 0,1406 | 272 | 48 |

| Par | Linhas em comum | Perfis em comum |
|---|---:|---:|
| treino × validação | 0 | 0 |
| treino × teste | 0 | 0 |
| validação × teste | 0 | 0 |

Cobertura: 961 + 318 + 320 = 1599 linhas; a união dos `id_linha` é exatamente o conjunto da base. Ambas as classes estão presentes nos três conjuntos. Nenhum perfil tem alvos conflitantes. As repetições internas (perfis repetidos dentro do mesmo conjunto) são permitidas: elas não atravessam a fronteira. A validação ficou com menos positivos (10,4%) que o treino (14,5%); é efeito da amostragem por grupos sem estratificação e deixa a validação com apenas 33 positivos, o que torna a escolha de limiar ruidosa (ver D5).

O que esta separação **não** testa: futuro temporal, novos produtores, outras regiões e outras variedades — a base não tem esses metadados.

## A1 — Dataset
`MeuDataset(df, features, media=None, desvio=None)`: no treino aprende média e desvio **populacional (ddof=0)**; na validação e no teste recebe os vetores do treino. Desvio zero/não finito vira 1 (nenhuma coluna precisou). X e y são convertidos uma vez para `float32`, na ordem das linhas.

- Comprimentos: treino 961, validação 318, teste 320.
- Um exemplo: X `[11]` `float32`, y `[]` (escalar) `float32`. Lote de 8: X `[8, 11]`, y `[8]`.
- 0 valores NaN/inf em qualquer partição.
- Evidência do uso das estatísticas do treino: `np.array_equal(ds_validacao.media, ds_treino.media)` e o mesmo para desvio/teste são verdadeiros; a média padronizada do treino é 0 em todas as colunas, enquanto validação e teste ficam ligeiramente deslocados (ex.: acidez volátil +0,134 na validação), como esperado ao reutilizar estatísticas de outro conjunto. Tabela completa em `resultados/a1_estatisticas.csv`.
- Alinhamento: para a linha 7 da validação, X desfeito da padronização reproduz as medidas originais e y coincide com `alvo`.

## A2 — DataLoader
| Conjunto | batch_size | shuffle | drop_last | num_workers | Justificativas |
|---|---|---|---|---|---|
| Treino | 64 | True (gerador com semente 42) | False | 0 | 64 dá 16 atualizações por época, com gradiente menos ruidoso que lotes minúsculos e custo baixo (A3). Embaralhar muda a composição dos lotes a cada época e evita ordem fixa; só reordena linhas *dentro* do treino. `drop_last=False` mantém as 961 linhas; o último lote tem 1 exemplo (961 = 15×64 + 1), o que é aceitável porque a rede principal não tem BatchNorm (ver A4). |
| Validação | 256 | False | False | 0 | Sem gradiente, o lote só afeta memória/velocidade; 256 cobre a validação em 2 passos. Sem embaralhar, a ordem dos escores fica igual à do Dataset; `drop_last=False` avalia todas as 318 linhas (conferido por asserção). |
| Teste | 256 | False | False | 0 | Mesmas razões da validação; as 320 linhas são avaliadas uma única vez, em ordem. |

`num_workers=0` foi **medido**: percorrer o loader de treino levou mediana de **3,8 ms com 0 workers** contra **71 ms com 2 workers** (o custo de criar processos e serializar lotes supera o trabalho de indexar um tensor já em memória). Em base pequena e pré-carregada, workers extras só atrasam.

## A3 — Tempo por época
Rede 11→16→1 sem BatchNorm, mesmo estado inicial copiado para todas as medições, Adam novo (lr 1e-3), mesmo treino (961 exemplos), `shuffle=True`, `drop_last=False`, CPU. Uma época de aquecimento por tamanho fora da medição, depois 3 repetições cronometradas com `time.perf_counter()` em torno de `treinar_epoca` (previsão, perda, retropropagação e atualização).

| Lote | Repetição 1 (s) | Repetição 2 (s) | Repetição 3 (s) | Mediana (s) | Passos | ms/passo |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0,6206 | 0,5545 | 0,5184 | 0,5545 | 961 | 0,58 |
| 8 | 0,1430 | 0,0938 | 0,1032 | 0,1032 | 121 | 0,85 |
| 64 | 0,0161 | 0,0180 | 0,0212 | 0,0180 | 16 | 1,12 |
| 512 | 0,0074 | 0,0071 | 0,0061 | 0,0071 | 2 | 3,53 |

Gráfico: `resultados/a3_tempo_por_lote.png` (eixos log, segundos × exemplos por passo, equipamento no rodapé).

**Análise:** o tempo cai de forma monotônica, mas o ganho diminui: ×5,4 de 1→8, ×5,7 de 8→64 e só ×2,5 de 64→512. A explicação está no custo por passo: com uma rede minúscula, cada passo é dominado por custo fixo (chamadas Python, montagem do lote, `zero_grad`, `backward`, `step` do Adam), ≈ 0,6 ms com 1 exemplo; esse custo cresce pouco até 64 (1,1 ms) e passa a crescer mais em 512 (3,5 ms), quando o trabalho numérico começa a pesar. Atenção: lotes maiores fazem **menos atualizações por época** (961 → 2), então "época mais rápida" não significa "convergência mais rápida"; com 512 o modelo recebe 2 atualizações por época. Os tempos variam entre execuções (outros processos na máquina; numa execução anterior completa as medianas diferiram entre 8% no lote 1 e ~40% no lote 512, mantendo a mesma ordem e o mesmo padrão de ganho decrescente), por isso uso a mediana e não interpreto diferenças pequenas.

## A4 — BatchNorm
- Subconjunto **somente do treino** com 65 observações (índices sorteados com semente 42); lote 64 → **resto 1** (conferido por `assert`). Lotes com `drop_last=False`: `[64, 1]`.
- Rede `criar_modelo(11, batchnorm=True)` em modo `train()`. Erro capturado especificamente (`except ValueError` com conferência do texto):
  `Expected more than 1 value per channel when training, got input size torch.Size([1, 16])`.
  Com um único exemplo, a variância do lote é indefinida e o BatchNorm não consegue normalizar.
- **Correção escolhida:** no carregador de treino, `drop_last=True` com `shuffle=True` (estado da rede restaurado antes, porque o primeiro lote já tinha atualizado os pesos). Três épocas rodaram com 1 passo cada, usando 64 de 65 exemplos. **Custo:** 1 observação fica de fora por época; com embaralhamento a excluída muda a cada época, então nenhuma linha é descartada permanentemente, mas cada época vê 98,5% dos dados.
- **Alternativa sem descarte** (demonstrada): lote 33 → lotes `[33, 32]`, sem resto 1; o custo é mudar o tamanho de lote (e o ruído do gradiente). A avaliação continua com `drop_last=False`: em `eval()` o BatchNorm usa médias móveis e aceita lote 1 (perda 0,63 sem erro). Não coloquei a rede em `eval()` durante o treino, que apenas esconderia o problema.

## C — Formas de vazamento
**Preservação do teste principal:** `c_desenvolvimento` = treino + validação da B2 (1279 linhas; asserção de interseção vazia com o teste). Partição interna por grupo de perfil, 75/25: `c_treino` = 961 linhas, `c_avaliacao` = 318 linhas, 0 perfis em comum. Tudo usa variáveis com prefixo `c_`; a rede da D é criada do zero depois.
**Desenho declarado antes de rodar:** rede 11→16→1 (12 entradas na forma 2 incorreta), Adam lr 1e-3, lote 64, **60 épocas fixas** (sem parada antecipada, para não usar `c_avaliacao` na seleção), métrica principal acurácia com corte 0,5 em `c_avaliacao`, secundária AUC-ROC; 5 sementes de inicialização fixadas antes. Reporto média e o intervalo das diferenças por semente — não escolhi sementes.

| Forma | Dados e partições internas | Métrica | Incorreto | Corrigido | Diferença/unidade |
|---|---|---|---|---|---|
| 2 — alvo entre as entradas | Red Wine real; `c_treino`/`c_avaliacao` por grupo | Acurácia (0,5), média de 5 sementes | 0,9937 (AUC 0,998) | 0,8717 (AUC 0,835) | **+12,20 p.p.** (11,01 a 13,52 por semente) |
| 1 — pré-processamento ajustado com dados reservados | Red Wine real; mesmas partições | Acurácia (0,5), média de 5 sementes | 0,8730 (AUC 0,834) | 0,8717 (AUC 0,835) | **+0,13 p.p.** (0,00 a 0,63) |
| 4 — duplicatas entre conjuntos (complementar) | Red Wine real; divisão por linha vs. por grupo | Acurácia (0,5), média de 5 sementes | 0,8781 (AUC 0,855) | 0,8717 (AUC 0,835) | +0,64 p.p. (−1,49 a +1,95) |

**Experimento 1 — Forma 2 (alvo nas entradas).** A informação protegida é `quality`, a nota sensorial que define o alvo e que só existe depois da avaliação que se quer antecipar. Ela atravessa a fronteira na montagem da matriz de entradas: na versão incorreta acrescentei a coluna original `quality` como 12ª característica (transformação declarada; padronizada com estatísticas de `c_treino`). A acurácia sobe de 0,872 para 0,994 e a AUC chega a ~1,0 em todas as sementes: a rede aprende quase só o corte "quality ≥ 7", que é a própria definição do rótulo. O efeito é grande, consistente entre sementes e explicado pelo mecanismo, então sustenta a interpretação. Não chega a 100% porque a rede pequena com 60 épocas não aprende um degrau perfeito em 7.

**Experimento 2 — Forma 1 (pré-processamento).** A informação protegida é a distribuição das medidas em `c_avaliacao`. Na versão incorreta, média e desvio foram calculados em `c_treino ∪ c_avaliacao`. O efeito medido é pequeno: +0,13 p.p. de acurácia em média (0 em 4 das 5 sementes, +0,63 em uma) e a AUC até caiu ligeiramente (0,834 vs 0,835). Isso é coerente com a magnitude da contaminação: as médias se deslocaram no máximo 0,033 desvio e os desvios mudaram entre −3,5% e +2,6%, porque as duas partes vêm da mesma distribuição e a padronização tem só 22 parâmetros. A diferença pequena **não** significa que o procedimento seja aceitável: o mecanismo existe (a avaliação deixou de ser independente do ajuste) e seu impacto cresce com pré-processamentos de maior capacidade (imputação por vizinhos, seleção de variáveis, PCA, reamostragem) ou conjuntos pequenos/deslocados. Também não ampliei o efeito artificialmente.

**Experimento complementar — Forma 4 (duplicatas).** Dividindo `c_desenvolvimento` por linha, 68 das 320 linhas de avaliação (21,2%) têm perfil idêntico no treino. A acurácia média sobe 0,64 p.p. e a AUC 0,02, mas o sinal varia entre sementes (−1,49 a +1,95 p.p.) e as duas versões avaliam linhas diferentes, então essa diferença mistura vazamento e variação de amostra. A evidência mais direta está dentro da divisão por linha: acurácia de **0,915 nos perfis já vistos** contra **0,868 nos perfis novos**. Uma rede de 16 neurônios memoriza pouco; um modelo com mais capacidade (ex.: k-NN ou árvores profundas) tenderia a ampliar esse efeito. Isso justifica a separação por grupos da B2.

## D7 — Treinamento e parada
Rede 11 → 16 (ReLU) → 1 (`criar_modelo`, sem BatchNorm), nova e independente da Parte C; Adam lr 1e-3; `BCEWithLogitsLoss` sem ponderação; lote 64; máximo **100 épocas**, paciência **10**, melhora mínima **0,0001** na perda de validação (valores definidos antes, os sugeridos no enunciado). O loader de teste não é argumento da função.

- **Melhor época: 49**, perda de validação **0,26523**. **Época final: 59.** Motivo: **paciência esgotada** (épocas 50–59 sem melhora maior que 0,0001); o máximo de 100 épocas não foi atingido.
- **Restauração:** a cópia foi feita com `copy.deepcopy(state_dict)`. Após restaurar, a perda de validação recalculada é 0,26523 (igual à melhor, conferido por `assert`), contra 0,26574 na última época; a soma das diferenças absolutas entre pesos restaurados e pesos da época 59 é 2,99, o que mostra que os pesos realmente voltaram.
- Curvas: `resultados/d7_curvas_perda.png`. A perda de validação fica abaixo da de treino em quase todo o treino; dois motivos: a de treino é a média durante a época (pesos ainda melhorando) e a validação tem menos positivos (10,4%), que são os casos mais difíceis. Depois da época ~30 a validação estabiliza em ~0,266–0,268 enquanto o treino continua caindo lentamente.

## D4 — Matriz de custos
**Ação:** prever positivo = **selecionar** a amostra para a análise especial de alta nota; prever negativo = não selecionar.

| | Previsto 0 (não seleciona) | Previsto 1 (seleciona) |
|---|---:|---:|
| Real 0 (nota < 7) | 0 | **C_FP = 1** |
| Real 1 (nota ≥ 7) | **C_FN = 3** | 0 |

Unidade: **unidades de custo hipotético (u.c.)**, em que 1 u.c. = uma análise especial gasta à toa. **Os custos são hipóteses do exercício**; o CSV não traz preços, perdas ou custos operacionais. Justificativa: um falso positivo custa uma análise desperdiçada; um falso negativo deixa de identificar uma amostra de alta nota (oportunidade de lote premium perdida), que supus valer o equivalente a três análises. A relação 3:1 também reflete que os positivos são raros (≈ 14%): perder um deles é pior que examinar uma amostra a mais. Acertos custam zero (simplificação).

## D5 e D6 — Escolha na validação
**Busca:** modelo restaurado; escores da validação via `obter_escores` (ordem conferida com `ds_validacao.y`). Candidatos = **todos os 272 escores únicos** + extremos 0 (seleciona tudo) e +∞ (não seleciona nada) = 274 limiares; busca exata, sem grade. Regra: positivo se escore ≥ limiar. **Desempate (fixado antes):** entre custos mínimos iguais, o maior limiar (menos seleções). Não houve empate.

| Regra (validação, n = 318) | Limiar | TP | FP | FN | TN | Selecionados | Custo total (u.c.) | Custo/obs. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Escolhido na validação** | **0,2766** | 24 | 33 | 9 | 252 | 57 | **60** | **0,189** |
| Padrão 0,5 | 0,5 | 8 | 12 | 25 | 273 | 20 | 87 | 0,274 |
| Fórmula C_FP/(C_FP+C_FN) | 0,25 | 24 | 35 | 9 | 250 | 59 | 62 | 0,195 |
| Seleciona tudo | 0 | 33 | 285 | 0 | 0 | 318 | 285 | 0,896 |
| Não seleciona nada | +∞ | 0 | 0 | 33 | 285 | 0 | 99 | 0,311 |

O limiar escolhido reduz o custo em 27 u.c. (−31%) frente a 0,5 na mesma validação, trocando 13 FP a mais por 16 FN a menos.

**Fórmula (D6):** C_FP/(C_FP+C_FN) = 1/4 = **0,25**, muito próximo do 0,2766 encontrado; na validação a diferença é de apenas 2 FP (62 vs 60 u.c.). A fórmula assume escores que sejam probabilidades bem calibradas para a população, custos constantes e acertos com custo zero. Investigação: Brier na validação = **0,079**, melhor que um escore constante igual à prevalência (0,093), mas o diagrama de calibração (8 bins por quantil, `resultados/d5_d6_custo_calibracao.png`) mostra **superestimação** nos escores altos (escore médio 0,49 → 37,5% de positivos observados) e a média dos escores (0,127) é maior que a prevalência da validação (0,104). Parte disso vem da diferença de prevalência treino (14,5%) × validação (10,4%), parte de amostra pequena (33 positivos; ~5 positivos por bin). Com tão poucos positivos, a curva de custo é serrilhada e a distância entre 0,25 e 0,277 está dentro do ruído — não é evidência de que um seja melhor. A sigmoide não garante calibração; não recalibrei o modelo.

**Uso da validação:** a mesma validação escolheu a época (D7) e o limiar (D5); por isso os custos acima são **otimistas** e não são a estimativa final.

**Configuração congelada antes do teste** (`resultados/decisao_congelada.json` + `resultados/modelo_final.pt`): 11 `FEATURES`; z-score com média/desvio do treino (ddof=0); pesos da época 49; C_FP = 1, C_FN = 3 (u.c. hipotéticas); limiar **0,276610**, positivo se escore ≥ limiar; desempate pelo maior limiar.

## D1 e D2 — Avaliação final (teste, n = 320)
Escores do teste gerados **uma única vez**, depois do congelamento. **Prevalência do teste: 45/320 = 0,1406.** Classe majoritária no **treino**: 0 (não selecionar). Convenções: precisão = 0 quando não há previsões positivas (`zero_division=0`); para o trivial as AUCs **não são reportadas**, pois um escore constante não gera curva (seria um único ponto). AUC-PR = área trapezoidal `auc(recall, precision)`; AP (`average_precision_score`) é reportada separadamente.

Matrizes de confusão (linhas = real, colunas = previsto, ordem [0, 1]):

| Trivial | prev. 0 | prev. 1 |  | Rede 0,5 | prev. 0 | prev. 1 |  | Rede 0,2766 | prev. 0 | prev. 1 |
|---|---:|---:|---|---|---:|---:|---|---|---:|---:|
| real 0 | 275 | 0 | | real 0 | 269 | 6 | | real 0 | 246 | 29 |
| real 1 | 45 | 0 | | real 1 | 33 | 12 | | real 1 | 17 | 28 |

| Regra | Limiar | Acurácia | Precisão | Revocação | F1 | AUC-ROC | AUC-PR (trap.) | Custo/obs. (u.c.) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Trivial do treino (classe 0) | não se aplica | 0,8594 | 0,0000 | 0,0000 | 0,0000 | não reportada | não reportada | 0,4219 |
| Rede: padrão | 0,5 | **0,8781** | **0,6667** | 0,2667 | 0,3810 | 0,8906 | 0,5641 | 0,3281 |
| Rede: escolhido na validação | 0,2766 | 0,8562 | 0,4912 | **0,6222** | **0,5490** | 0,8906 | 0,5641 | **0,2500** |

AP (average precision) da rede = 0,5736 — diferente da área trapezoidal (0,5641), como esperado. As AUCs são iguais nas duas linhas da rede porque os escores são os mesmos; o limiar só muda as classes.

**Leitura:** o trivial tem 85,9% de acurácia só por repetir a classe 0 (é a proporção de negativos do teste) e não encontra nenhum positivo — mostra por que acurácia sozinha não informa aqui. A rede com 0,5 ganha só 1,9 p.p. de acurácia sobre o trivial e encontra 12 de 45 positivos. Com o limiar fixado na validação, a acurácia **cai** para 85,6% (abaixo do trivial), mas a revocação sobe para 62% (28 de 45), F1 sobe de 0,38 para 0,55 e o custo hipotético por observação cai de 0,328 para **0,250** (−24% frente a 0,5; −41% frente ao trivial). Esse é o comportamento previsto pela matriz de custos: aceitar mais FP para reduzir FN. O custo no teste (0,250) ficou acima do da validação (0,189), coerente com o uso otimista da validação e com a prevalência maior no teste.

**Incerteza (bootstrap 2000× no teste, percentis 2,5–97,5%; só descreve, não escolhe nada):** AUC-ROC 0,848–0,930; revocação no limiar escolhido 0,48–0,77; precisão 0,37–0,62; custo/obs. 0,178–0,334 (escolhido) e 0,234–0,431 (0,5). Os intervalos dos dois custos se sobrepõem: com 45 positivos, a vantagem do limiar escolhido é provável, mas não está estabelecida com folga.

## D3 — Curvas
`resultados/d3_roc_pr_teste.png`: ROC (AUC 0,891) com a **diagonal** aleatória de (0,0) a (1,1); PR (AUC-PR trapezoidal 0,564; AP 0,574) com a linha horizontal na **prevalência do teste (0,141)**. Os pontos dos limiares 0,5 e 0,2766 estão marcados nas duas curvas.

## D8 — Recomendação
Para a triagem de amostras para a análise especial de alta nota, recomendo usar a rede com **limiar fixo de 0,2766** (seleciona a amostra se o escore ≥ 0,2766), escolhido na validação antes de abrir o teste e próximo do valor teórico 0,25 para custos de 1 (falso positivo) e 3 (falso negativo). No teste reservado de 320 amostras deste arquivo (45 de nota ≥ 7), essa regra selecionou 57 amostras e encontrou **28 dos 45 vinhos de alta nota (revocação 62%)**, com precisão de 49% — cerca de uma em cada duas análises especiais recai sobre um vinho realmente bom — e custo médio de **0,25 u.c. por amostra**, contra 0,33 com o corte 0,5 e 0,42 sem modelo. **Esses custos são hipotéticos**; se a razão real entre perder um vinho bom e fazer uma análise inútil não for próxima de 3:1, o limiar deve ser recalculado na validação. O erro que merece atenção são os **17 vinhos bons (38%) que continuam não selecionados**, e os 29 selecionados indevidamente que consomem análise. A estimativa vem de uma única amostra pequena (IC95% bootstrap da revocação 48–77%), os escores não estão bem calibrados (não devem ser lidos como probabilidade), e a base não tem data, produtor ou região: **não há evidência de desempenho para outras safras, produtores, regiões, vinho branco ou períodos futuros**, nem para amostras cujo perfil de medidas seja muito diferente do observado. Antes de uso real, recomendo validar com amostras novas, com custos reais e acompanhar a prevalência.

## Referências e limites
- Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Wine Quality* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56S3T (CC BY 4.0). Página no Kaggle: https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009
- PyTorch — Dataset/DataLoader: https://docs.pytorch.org/docs/stable/data.html ; BCEWithLogitsLoss: https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
- scikit-learn — GroupShuffleSplit: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html ; métricas: https://scikit-learn.org/stable/modules/model_evaluation.html ; limiar: https://scikit-learn.org/stable/modules/classification_threshold.html ; armadilhas de vazamento: https://scikit-learn.org/stable/common_pitfalls.html
- Enunciado e material do Lab 2 (Deep Learning, CESAR School).
- **Limites:** uma única partição (semente 42) e uma única inicialização na D; validação pequena (33 positivos) usada para parada e limiar; custos hipotéticos; perfis repetidos tratados como grupos sem prova de identidade física; sem metadados temporais/de produtor; tempos da A3 dependem da máquina.
- Uso de IA declarado em `USO-DE-IA.md`.
