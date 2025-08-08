import numpy as np
import pickle
import time
import pygame
from tqdm import tqdm  # Biblioteca para barras de progresso (instale com: pip install tqdm)

# IMPORTANTE: Este script espera que o arquivo 'space_invaders_pygame_env.py'
# esteja na mesma pasta para poder importar a classe do ambiente.
from space_invaders_pygame_env import SpaceInvadersEnv, LARGURA_TELA, ALTURA_TELA


class QLearningAgent:
    """
    Implementa um agente que aprende a jogar Space Invaders
    usando o algoritmo Q-Learning.
    """

    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon_start=1.0, epsilon_end=0.05,
                 epsilon_decay=0.9995):
        # Hiperparâmetros do aprendizado
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

        # A Q-Table será um dicionário para armazenar os valores Q para cada par (estado, ação)
        self.q_table = {}

        # Define o número de "caixas" para discretizar o estado contínuo
        self.num_bins_pos = 20  # Para posições X
        self.num_bins_dir = 2  # Para direção dos inimigos (esquerda/direita)

        self.action_space_size = 4  # 0: Esquerda, 1: Direita, 2: Atirar, 3: Parado

    def _get_discrete_state(self, state, enemy_direction):
        """
        Converte o vetor de estado contínuo do ambiente em um estado discreto (tupla),
        que pode ser usado como chave na Q-Table.
        """
        # Acha o inimigo mais próximo da nave
        nave_x = state[0] * LARGURA_TELA

        closest_enemy_dist = float('inf')
        closest_enemy_pos = (-1, -1)

        # O estado dos inimigos começa no índice 1 do vetor de estado
        enemy_states = state[1:1 + 50 * 2]
        for i in range(0, len(enemy_states), 2):
            enemy_x = enemy_states[i]
            if enemy_x != -1:  # Se o inimigo existe
                enemy_x_abs = enemy_x * LARGURA_TELA
                dist = abs(nave_x - enemy_x_abs)
                if dist < closest_enemy_dist:
                    closest_enemy_dist = dist
                    closest_enemy_pos = (enemy_x, enemy_states[i + 1])

        # Discretiza os valores
        pos_x_bin = int(np.floor(self.num_bins_pos * state[0]))
        closest_enemy_x_bin = int(np.floor(self.num_bins_pos * closest_enemy_pos[0]))

        # Direção do inimigo: 0 para esquerda (-1), 1 para direita (1)
        dir_bin = 0 if enemy_direction < 0 else 1

        return (pos_x_bin, closest_enemy_x_bin, dir_bin)

    def choose_action(self, state):
        """
        Decide a próxima ação usando a estratégia epsilon-greedy.
        Explora (ação aleatória) ou explora (melhor ação da Q-Table).
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_space_size)
        else:
            # Pega a melhor ação conhecida para o estado atual
            return np.argmax(self.q_table.get(state, np.zeros(self.action_space_size)))

    def update_q_table(self, state, action, reward, next_state):
        """
        Atualiza o valor Q para o par (estado, ação) usando a equação de Bellman.
        """
        # Garante que o estado exista na Q-Table
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space_size)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_space_size)

        # Fórmula do Q-Learning
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])

        new_value = old_value + self.lr * (reward + self.gamma * next_max - old_value)
        self.q_table[state][action] = new_value

    def decay_epsilon(self):
        """Reduz o epsilon para diminuir a exploração ao longo do tempo."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_q_table(self, filename="q_table.pkl"):
        """Salva a Q-Table em um arquivo."""
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-Table salva em {filename}")

    def load_q_table(self, filename="q_table.pkl"):
        """Carrega a Q-Table de um arquivo."""
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"Q-Table carregada de {filename}")


def train(agent, env, episodes):
    """Função para treinar o agente."""
    print("Iniciando o treinamento...")
    recompensas_por_episodio = []

    for episode in tqdm(range(episodes), desc="Progresso do Treinamento"):
        continuous_state = env.reset()
        current_discrete_state = agent._get_discrete_state(continuous_state, env.velocidade_inimigo_x)

        finalizado = False
        total_reward = 0

        while not finalizado:
            action = agent.choose_action(current_discrete_state)

            next_continuous_state, reward, finalizado, info = env.step(action)
            next_discrete_state = agent._get_discrete_state(next_continuous_state, env.velocidade_inimigo_x)

            agent.update_q_table(current_discrete_state, action, reward, next_discrete_state)

            current_discrete_state = next_discrete_state
            total_reward += reward

        recompensas_por_episodio.append(total_reward)
        agent.decay_epsilon()

    print("Treinamento concluído.")
    agent.save_q_table()


def watch_agent(agent, env):
    """Função para assistir o agente treinado jogar."""
    print("Assistindo ao agente treinado...")
    try:
        agent.load_q_table()
    except FileNotFoundError:
        print("Erro: Arquivo 'q_table.pkl' não encontrado. Treine o agente primeiro.")
        return

    agent.epsilon = 0  # Desliga a exploração para jogar de forma otimizada

    continuous_state = env.reset()
    finalizado = False
    rodando = True

    while rodando:
        # --- CORREÇÃO APLICADA AQUI ---
        # Este loop processa os eventos da janela, impedindo que ela trave.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                rodando = False

        discrete_state = agent._get_discrete_state(continuous_state, env.velocidade_inimigo_x)
        action = agent.choose_action(discrete_state)

        continuous_state, reward, finalizado, info = env.step(action)

        env.render()  # Renderiza cada quadro na tela

        if finalizado:
            print("Fim de jogo. Reiniciando em 2 segundos...")
            time.sleep(2)
            continuous_state = env.reset()
            finalizado = False


if __name__ == "__main__":
    # --- Menu Principal ---
    print("O que você gostaria de fazer?")
    print("1: Treinar um novo agente de IA")
    print("2: Assistir a um agente já treinado")
    print("3: Sair")

    escolha = input("Digite sua escolha (1, 2 ou 3): ")

    if escolha == '1':
        # Para treinar, não precisamos renderizar a tela, então usamos 'none'
        env_treino = SpaceInvadersEnv(render_mode='none')
        agent = QLearningAgent()
        # Aumente o número de episódios para um agente mais inteligente (ex: 10000)
        # 1000 episódios é um bom começo para um teste rápido.
        train(agent, env_treino, episodes=1000)
        env_treino.close()

    elif escolha == '2':
        # Para assistir, precisamos renderizar, então usamos 'human'
        env_assistir = SpaceInvadersEnv(render_mode='human')
        agent = QLearningAgent()
        watch_agent(agent, env_assistir)
        env_assistir.close()

    else:
        print("Saindo.")
