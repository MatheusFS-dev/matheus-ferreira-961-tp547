# Recorte do estudo de caso usado no relatório

## Enunciado da fila principal

- Roteador IoT com capacidade limitada de armazenamento.
- Chegadas seguindo processo de Poisson com taxa `\lambda`.
- Atendimento exponencial com taxa `\mu`.
- Buffer com capacidade máxima de `K` pacotes em espera.
- Pacotes com duas classes:
  - alta prioridade com probabilidade `p`;
  - baixa prioridade com probabilidade `1-p`.
- Quando o buffer está cheio:
  - pacote de baixa prioridade é descartado imediatamente;
  - pacote de alta prioridade pode substituir o pacote de baixa prioridade mais antigo presente na fila;
  - se não houver pacote de baixa prioridade em espera, a nova chegada prioritária também é descartada.

## Itens obrigatórios do relatório

- Resumo incluindo o enunciado e os principais resultados.
- Introdução com cenários de aplicação.
- Modelagem com modelo do sistema e metodologia.
- Resultados com tabelas e gráficos comparativos.
- Conclusões.
- Referências bibliográficas.
- Anexo com visão geral e trechos representativos dos códigos.

## Itens do estudo de caso não usados no relatório

- Gravação de vídeo.
- Publicação no YouTube.
- Apresentação em sala.
