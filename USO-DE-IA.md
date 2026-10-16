# Declaração de uso de IA — Lab 2
## Ferramenta

Claude (Anthropic).

## Finalidade

A ferramenta foi utilizada para auxiliar na interpretação dos requisitos do enunciado, esclarecer conceitos, discutir alternativas de implementação e apoiar a identificação de erros.

O uso incluiu a discussão de trechos desenvolvidos por mim, sugestões de código para partes do laboratório e a avaliação de como essas implementações funcionariam juntas. Também solicitei avaliações sobre a adequação das minhas soluções e possíveis formas de melhorá-las.

Além disso, a IA auxiliou na organização e revisão das explicações do `README.md`, na análise das saídas do notebook e na conferência da estrutura da entrega em relação ao enunciado.

## Trechos aproveitados

- O código das células de B2, A1–A4, C (formas 2, 1 e 4), D7, D4, D5/D6 e D1–D3 reuniu trechos desenvolvidos por mim e trechos sugeridos pela IA. Durante o desenvolvimento, apresentei minhas implementações à ferramenta para discutir se atendiam ao objetivo e se existiam formas mais adequadas de resolver cada etapa.
- A integração desses trechos passou por uma dupla revisão: eu conferi a lógica, a execução e a adequação ao enunciado, enquanto a IA auxiliou na análise de compatibilidade entre as partes e na identificação de possíveis problemas. O funcionamento conjunto foi verificado pela execução do notebook.
- O texto do `README.md` reuniu análises escritas por mim e contribuições da IA para redação, organização e revisão. Os números apresentados foram extraídos das saídas do notebook executado, e não estimados.
- O `apoio.py`, os dados (`dados/`), o `requirements.txt` e as células originais de preparação não foram alterados.

## Como verifiquei

- Reiniciei o kernel e executei o notebook completo, sem erros. A célula final retornou `Pendências estruturais: []`, indicando ausência de pendências estruturais nos itens verificados.
- Conferi as asserções de cobertura total e ausência de interseção entre as partições, tanto por linhas quanto por perfis, na B2.
- Verifiquei que a padronização da validação e do teste utiliza apenas as estatísticas calculadas no conjunto de treino.
- Conferi a condição de resto 1 na A4 e o tratamento do problema relacionado ao BatchNorm.
- Verifiquei a separação do teste principal na Parte C.
- Conferi a restauração dos melhores pesos na D7 e o alinhamento entre rótulos e escores.
- Comparei os números citados no `README.md` com as saídas do notebook e os arquivos de `resultados/`.
- Confirmei que o teste principal foi utilizado para gerar escores somente após o congelamento da decisão, registrado em `resultados/decisao_congelada.json`.

## Revisão pessoal

Li o código e revisei as análises para conferir sua correspondência com o enunciado, com atenção aos seguintes pontos:

- Partição por grupos de perfil na B2.
- Padronização com estatísticas exclusivas do treino na A1.
- Correção do problema de BatchNorm na A4.
- Comparação entre os procedimentos incorretos e corrigidos nas formas de vazamento da Parte C.
- Parada antecipada com restauração dos melhores pesos na D7.
- Escolha do limiar com base no conjunto de validação na D5.
- Coerência das respostas da B1, das análises da Parte C e da recomendação da D8.

Minha participação incluiu o desenvolvimento de trechos de código, a avaliação das sugestões recebidas, a conferência da integração entre as partes e a revisão das análises. As dúvidas e alternativas foram discutidas com a IA, e as soluções incorporadas foram conferidas por mim com base no enunciado e nos resultados da execução.

## Responsabilidade pela entrega

Assumo a responsabilidade pelo conteúdo entregue, incluindo as contribuições da IA incorporadas ao trabalho. Declaro que revisei o material e consigo explicar as implementações, os procedimentos de validação e as conclusões apresentadas.