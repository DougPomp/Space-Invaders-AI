# Space Invaders com IA de Aprendizado por Reforço

![Imagem de uma captura de tela do jogo Space Invaders](https://placehold.co/800x400/000000/FFFFFF?text=Space+Invaders+AI)

Este projeto apresenta uma implementação do clássico jogo Space Invaders em Python usando a biblioteca Pygame, com um diferencial: um agente de Inteligência Artificial treinado com **Aprendizado por Reforço (Q-Learning)** que aprende a jogar sozinho.

---

## 🚀 Funcionalidades

* **Jogo Clássico:** Jogue o Space Invaders tradicional com controle manual, sons e música de fundo.
* **Ambiente de IA:** O jogo foi encapsulado em um ambiente compatível com bibliotecas de Aprendizado por Reforço, como o Gymnasium.
* **Agente Inteligente:** Um agente de Q-Learning que pode ser treinado do zero.
* **Treinamento e Visualização:** Scripts separados para treinar o agente (processo rápido, sem renderização) e para assistir ao agente treinado jogar em tempo real.
* **Persistência do Modelo:** A "inteligência" do agente (sua Q-Table) é salva em um arquivo (`q_table.pkl`) após o treinamento, permitindo que você assista ao agente sem precisar treiná-lo novamente.

---

## 🛠️ Tecnologias Utilizadas

* **Python 3.x**
* **Pygame:** Para a criação do jogo, renderização e gerenciamento de sons.
* **NumPy:** Para cálculos numéricos e manipulação eficiente dos vetores de estado.
* **Tqdm:** Para exibir uma barra de progresso amigável durante o treinamento.

---

## 📂 Estrutura do Projeto

O projeto é dividido em dois arquivos principais:

1.  `space_invaders_pygame_env.py`: Contém a classe `SpaceInvadersEnv`, que encapsula toda a lógica do jogo (regras, física, renderização) em um ambiente de treinamento. É o "universo" onde a IA vive.
2.  `q_learning_agent.py`: Contém a classe `QLearningAgent`, que é o "cérebro" da IA. Este arquivo implementa o algoritmo Q-Learning e inclui a lógica para treinar o agente e para assisti-lo jogar.

---

## ⚙️ Como Usar

Siga os passos abaixo para executar o projeto em sua máquina local.

### 1. Pré-requisitos

Certifique-se de ter o Python 3 instalado.

### 2. Clone o Repositório

```bash
git clone https://github.com/DougPomp/Space-Invaders-AI.git
cd Space-Invaders-AI
```

### 3. Instale as Dependências

É recomendado criar um ambiente virtual para isolar as dependências do projeto.

```bash
# Crie um ambiente virtual (opcional, mas recomendado)
python -m venv venv

# Ative o ambiente virtual
# No Windows:
venv\Scripts\activate
# No macOS/Linux:
source venv/bin/activate
```

Instale as bibliotecas necessárias:

```bash
pip install pygame numpy tqdm
```

### 4. Treine o Agente de IA

Para treinar a IA, execute o script do agente e escolha a opção 1. O treinamento ocorrerá em segundo plano para ser mais rápido.

```bash
python q_learning_agent.py
```

Siga as instruções no terminal e digite `1`. O processo pode levar alguns minutos. Ao final, um arquivo `q_table.pkl` será criado na pasta do projeto.

### 5. Assista à IA Jogar

Após o treinamento, execute o mesmo script novamente e escolha a opção 2 para assistir ao agente treinado.

```bash
python q_learning_agent.py
```

Digite `2` no terminal. Uma janela do Pygame se abrirá, e você verá a IA controlando a nave e jogando Space Invaders de forma autônoma.

---

## 🧠 Fundamentos da IA

O agente utiliza **Q-Learning**, um algoritmo de Aprendizado por Reforço. Ele aprende o valor (a "Qualidade") de cada ação em cada estado possível do jogo através de tentativa e erro. As recompensas (positivas por destruir inimigos, negativas por morrer) guiam o aprendizado, e o agente armazena esse conhecimento em uma **Q-Table**. Para tornar o aprendizado viável, o estado do jogo é **discretizado**, focando apenas na posição da nave, na posição do inimigo mais próximo e na direção da frota.

---

## 🎧 Mídia e Recursos Adicionais

Explore mais sobre o projeto através dos recursos abaixo. Para que funcionem, os ficheiros devem estar numa pasta `assets` na raiz do projeto.

* **Resumo em Áudio (MP3):** Ouça uma explicação em áudio sobre a jornada de criação deste projeto.
    <br>
    <audio controls>
      <source src="resumo_audio.mp3" type="audio/mpeg">
      Seu navegador não suporta o elemento de áudio.
    </audio>
* **Documento da Jornada (PDF):** Leia a explicação detalhada sobre os fundamentos e a implementação do jogo e da IA.
    * [📄 Visualizar o PDF da Jornada](documento_jornada.pdf)

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.
