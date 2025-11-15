Aqui está o conteúdo formatado em Markdown, pronto para ser usado no seu `README.md`.

-----

# Multiplicação Distribuída de Matrizes

**Trabalho de Computação Paralela e Concorrente – AV3**
**Autor(es):** [Insira seu nome]

## Sobre o Projeto

Este projeto implementa uma simulação de computação distribuída aplicada à multiplicação de matrizes.
A matriz A é dividida em partes e enviada para diferentes servidores, onde cada servidor multiplica seu bloco por B.
O cliente recolhe os resultados parciais e reconstrói a matriz final C.

Este trabalho utiliza Python, sockets, threads e numpy.

-----

## Estrutura do Projeto

```
multiplicacao-distribuida/
│
├── server.py      # Servidor que processa submatrizes
├── client.py      # Cliente que divide a matriz e reúne o resultado
└── README.md
```

-----

## Tecnologias Utilizadas

  * Python 3.8 ou superior
  * Sockets (TCP)
  * Threads (`threading`)
  * Serialização com `pickle`
  * Estrutura de mensagens com `struct`
  * `numpy` para multiplicação de matrizes

-----

## Detalhamento dos Arquivos

### `server.py`

Representa cada servidor da rede distribuída.

#### Funcionalidades principais

  * `send_msg(conn, obj)`: Serializa um objeto Python com `pickle`, adiciona um cabeçalho contendo o tamanho da mensagem e envia pelo socket.
  * `recv_msg(conn)`: Recebe o cabeçalho de 4 bytes, depois lê a quantidade exata de bytes até reconstruir o objeto.
  * `handle_client(conn, addr)`:
      * Recebe uma submatriz de A (`A_sub`)
      * Recebe a matriz B
      * Executa a multiplicação `C_sub = A_sub @ B`
      * Envia a submatriz resultante
  * Cada cliente é processado em uma thread separada.
  * `start_server(host, port)`: Inicia o servidor, escuta conexões e cria threads para cada cliente recebido.

#### Objetivo

  * Simular uma máquina independente que executa parte do cálculo distribuído.

### `client.py`

Responsável por dividir, enviar e coordenar a computação distribuída.

#### Funcionalidades principais

  * `split_matrix_rows(A, parts)`: Divide a matriz A em blocos de linhas, um para cada servidor.
  * `worker_send_receive(server_addr, A_sub, B, queue, idx)`: Executado em uma thread. Conecta ao servidor, envia `A_sub` e `B`, recebe `C_sub` e armazena o resultado na fila.
  * `distributed_matmul(A, B, servers)`:
      * Divide A entre os servidores
      * Cria threads para enviar cada bloco
      * Recebe todos os resultados
      * Concatena as submatrizes na ordem correta
      * Retorna a matriz C completa

-----

## Como Executar

**1. Instalar dependências**

```bash
pip install numpy
```

**2. Iniciar os servidores**

Execute quantos servidores quiser. Cada servidor deve ser iniciado em um terminal diferente.

*Exemplo:*

*Terminal 1*

```bash
python server.py --port 5001
```

*Terminal 2*

```bash
python server.py --port 5002
```

É possível executar servidores em outras máquinas usando o argumento `--host`.

**3. Executar o cliente**

```bash
python client.py --servers 127.0.0.1:5001 127.0.0.1:5002
```

O cliente irá:

  * Gerar as matrizes A e B
  * Dividir A entre os servidores
  * Enviar cada parte
  * Receber os resultados parciais
  * Recompor a matriz final C
  * Comparar com a multiplicação local para verificar consistência

-----

## Testes

Para alterar o tamanho das matrizes, modifique no `client.py`:

```python
m, n, p = 6, 3, 4
```

Para testar matrizes grandes:

```python
m, n, p = 1000, 1000, 1000
```

-----

## Possíveis Extensões

Para melhorar o trabalho e enriquecer a apresentação:

  * Comparação de tempo serial vs distribuído
  * Cálculo de *speedup* e eficiência
  * Gráficos de desempenho
  * Balanceamento dinâmico de carga
  * Tolerância a falhas
  * Compressão dos blocos enviados