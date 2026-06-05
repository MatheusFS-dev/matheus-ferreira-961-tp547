# Simulação de Filas para Roteador IoT com Buffer Inteligente

## Objetivo

Implementar simulações discretas orientadas a eventos para comparar uma fila principal com descarte seletivo contra filas clássicas de referência usadas em redes e teoria de filas.

## Fila principal do exercício

A fila principal é a Fila M/M/1 com Buffer Inteligente e Descarte Seletivo.
As demais filas são referências para comparação.

## Funcionamento do buffer inteligente

A fila principal usa uma regra especial quando o buffer está cheio.

- Se chega um pacote de baixa prioridade, ele é descartado.
- Se chega um pacote de alta prioridade, o sistema procura o pacote de baixa prioridade mais antigo na fila.
- Se esse pacote existir, ele é substituído pelo pacote de alta prioridade.
- Se não existir pacote de baixa prioridade na fila, o pacote de alta prioridade também é descartado.

A prioridade não muda diretamente a ordem de atendimento do servidor. Ela atua na decisão de descarte e substituição dentro do buffer.

## Filas de comparação

- Fila M/M/1/J/J+1/∞/FCFS
- Fila M/M/1/∞/∞/∞/FCFS
- Fila M/G/1/∞/∞/∞/FCFS
- Fila M/M/m/∞/∞/∞/FCFS
- Filas com Prioridade e sem Preempção

## Estrutura do projeto

- `common.py`: utilitários compartilhados, geração de tempos, resumos estatísticos e escrita de arquivos.
- `sim_mm1_buffer_inteligente_descarte_seletivo.py`: fila principal do trabalho.
- `sim_mm1_buffer_finito_fcfs.py`: fila M/M/1 com buffer finito e bloqueio FCFS.
- `sim_mm1_buffer_infinito_fcfs.py`: fila M/M/1 com buffer infinito.
- `sim_mg1_buffer_infinito_fcfs.py`: fila M/G/1 com serviço exponencial, determinístico e uniforme.
- `sim_mmm_buffer_infinito_fcfs.py`: fila M/M/m com dois servidores paralelos.
- `sim_prioridade_sem_preempcao.py`: fila com duas classes e prioridade sem preempção.
- `make_plots.py`: geração dos gráficos comparativos.
- `results/`: saídas CSV, JSON e PNG geradas pelos scripts.

## Requisitos

- Python `3.10` ou superior.
- `matplotlib` para geração dos gráficos.
- Os demais módulos usados pelos scripts fazem parte da biblioteca padrão do Python:
  - `csv`
  - `json`
  - `math`
  - `random`
  - `heapq`
  - `pathlib`
  - `collections`
  - `dataclasses`

## Instalação

Para instalar a única dependência externa do trabalho:

```bash
pip install matplotlib
```

## Como executar

Para reproduzir os resultados a partir da pasta `trabalho_04`:

```bash
python sim_mm1_buffer_inteligente_descarte_seletivo.py
python sim_mm1_buffer_finito_fcfs.py
python sim_mm1_buffer_infinito_fcfs.py
python sim_mg1_buffer_infinito_fcfs.py
python sim_mmm_buffer_infinito_fcfs.py
python sim_prioridade_sem_preempcao.py
python make_plots.py
```

Os scripts de simulação geram arquivos CSV e JSON em `results/`. O script `make_plots.py` lê esses arquivos e atualiza os gráficos em `results/plots/`.

## Parâmetros principais

- `LAMBDA_BASE = 200.0`
- `MU_BASE = 250.0`
- `RHO_BASE = 0.8`
- `K_BASE = 5`
- `P_HIGH_BASE = 0.3`
- `SERVERS_BASE = 2`
- `SIM_TIME = 1000.0`
- `WARMUP_TIME = 100.0`
- `N_REPLICATIONS = 30`
- `BASE_SEED = 5472026`
- `TIME_SERIES_DT = 1.0`

## Cenários simulados

- Cenário base.
- Variações de utilização `ρ = 0.50`, `0.70`, `0.90` e `0.95`.
- Variações de buffer finito com `K = 2`, `10` e `20`.
- Variações de prioridade com `p_high = 0.10`, `0.50` e `0.80`.
- Variações de distribuição de serviço da M/G/1: exponencial, determinística e uniforme.

## Resultados gerados

Cada script grava em `results/<queue_slug>/`:

- `config.json`
- `replications.csv`
- `summary.csv`
- `class_summary.csv`
- `occupancy_states.csv`
- `time_series_sample.csv`

O script `make_plots.py` grava os PNGs em `results/plots/`.

## Métricas avaliadas

- Contagens de chegadas, aceitação, atendimento e descarte.
- Probabilidades de perda total e por prioridade.
- Probabilidade de substituição de pacotes de baixa prioridade.
- Vazão total e por prioridade.
- Utilização dos servidores.
- Ocupação média do sistema e da fila.
- Tempos médios de espera, de serviço e de permanência no sistema.
- Percentis `p50`, `p95` e `p99` do tempo no sistema.
- Erro absoluto e relativo da verificação de Little.

## Gráficos gerados

- `comparacao_perda_total_base.png`
- `comparacao_atraso_medio_base.png`
- `comparacao_ocupacao_media_base.png`
- `comparacao_utilizacao_base.png`
- `comparacao_vazao_base.png`
- `perda_por_prioridade_base.png`
- `atraso_por_prioridade_base.png`
- `efeito_do_buffer_K.png`
- `efeito_da_utilizacao_rho.png`
- `efeito_da_distribuicao_servico_mg1.png`
- `serie_temporal_ocupacao_fila_principal.png`

## Validação

- M/M/1 infinita: comparar `ρ`, ocupação média e tempos médios com a solução teórica.
- M/M/1 finita: comparar a probabilidade de bloqueio do cenário base com a fórmula teórica do material.
- Fila principal: verificar invariantes de fila, contagens por classe e substituição apenas no buffer.
- Todos os scripts: checar consistência aproximada com a Lei de Little.

## Escopo da simulação

A simulação usa hipóteses controladas para permitir comparação direta entre os modelos de fila.

Foram consideradas duas classes de prioridade, chegadas Poisson, atendimento exponencial no cenário principal e pacotes tratados de forma equivalente quanto ao tamanho.

Essas escolhas ajudam a isolar o efeito da política de descarte seletivo e concentram a análise no comportamento do buffer inteligente.

## Interpretação esperada

- A fila principal deve reduzir a perda de pacotes de alta prioridade em relação à fila finita FCFS.
- Pacotes de baixa prioridade devem pagar o custo dessa proteção por meio de maior perda e, em alguns cenários, maior atraso.
- O aumento de `K` deve reduzir perda total nas filas com buffer finito.
- O aumento de `ρ` deve elevar ocupação e atraso.
- Menor variância de serviço na M/G/1 deve reduzir atraso médio.
- Dois servidores devem reduzir espera em relação ao caso de um único servidor.
- Prioridade sem preempção deve beneficiar a classe alta sem interromper pacotes já em serviço.
