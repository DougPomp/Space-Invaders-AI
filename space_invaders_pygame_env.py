import pygame
import sys
import numpy as np

# --- Constantes ---
LARGURA_TELA = 800
ALTURA_TELA = 600
COR_PRETO = (0, 0, 0)
COR_BRANCO = (255, 255, 255)
COR_VERDE = (0, 255, 0)
COR_VERMELHO = (255, 0, 0)
VELOCIDADE_NAVE = 15
MAX_INIMIGOS = 50
MAX_BALAS = 10


class SpaceInvadersEnv:
    """
    Esta classe encapsula o jogo Space Invaders em um ambiente
    compatível com algoritmos de Aprendizado por Reforço.
    """

    def __init__(self, render_mode='human'):
        pygame.init()
        pygame.mixer.init()

        self.render_mode = render_mode
        if self.render_mode == 'human':
            self.tela = pygame.display.set_mode((LARGURA_TELA, ALTURA_TELA))
            pygame.display.set_caption("Ambiente Space Invaders para IA")
        else:  # 'none' ou outro modo para treinamento rápido sem renderização
            self.tela = pygame.Surface((LARGURA_TELA, ALTURA_TELA))

        self.relogio = pygame.time.Clock()
        self.fonte = pygame.font.Font(None, 36)
        self._carregar_sons()

        # Ações que a IA pode tomar: 0: Esquerda, 1: Direita, 2: Atirar, 3: Parado
        self.action_space = [0, 1, 2, 3]
        # O tamanho do vetor de estado que a IA receberá
        # [pos_nave_x] + [pos_inimigo_x, pos_inimigo_y] * MAX_INIMIGOS + [pos_bala_x, pos_bala_y] * MAX_BALAS
        self.observation_space_size = 1 + (MAX_INIMIGOS * 2) + (MAX_BALAS * 2)

    def _carregar_sons(self):
        """Carrega os arquivos de som."""
        try:
            self.som_tiro = pygame.mixer.Sound("assets/tiro.wav")
            self.som_explosao = pygame.mixer.Sound("assets/explosao.wav")
            pygame.mixer.music.load("assets/musica_fundo.wav")
            pygame.mixer.music.set_volume(0.3)
            self.sons_carregados = True
        except pygame.error:
            print("Aviso: Arquivos de som não encontrados. O jogo continuará sem som.")
            self.sons_carregados = False

    def reset(self):
        """
        Reinicia o ambiente para um novo episódio (partida).
        Retorna o estado inicial.
        """
        self.todas_as_sprites = pygame.sprite.Group()
        self.inimigos = pygame.sprite.Group()
        self.balas = pygame.sprite.Group()

        self.nave = self._Nave(self)
        self.todas_as_sprites.add(self.nave)

        self._criar_frota_inimiga()
        self.velocidade_inimigo_x = 2
        self.pontos = 0

        if self.sons_carregados:
            pygame.mixer.music.play(loops=-1)

        return self._get_state()

    def step(self, action):
        """
        Executa um passo no ambiente.
        A IA fornece uma 'action'.
        Retorna: (novo_estado, recompensa, finalizado, info)
        """
        recompensa = -0.01  # Pequena penalidade por existir, incentiva a terminar rápido
        finalizado = False

        # 1. Executar a ação da IA
        if action == 0:  # Mover para a esquerda
            self.nave.rect.x -= VELOCIDADE_NAVE
        elif action == 1:  # Mover para a direita
            self.nave.rect.x += VELOCIDADE_NAVE
        elif action == 2:  # Atirar
            # Impede a IA de спамить balas
            if len(self.balas) < MAX_BALAS:
                self.nave.atirar()

        # 2. Atualizar a lógica do jogo
        self.todas_as_sprites.update()
        self._mover_frota_inimiga()

        # 3. Verificar colisões e calcular recompensas
        colisoes = pygame.sprite.groupcollide(self.balas, self.inimigos, True, True)
        if colisoes:
            recompensa += 10 * len(colisoes)  # Recompensa por cada inimigo destruído
            self.pontos += 10 * len(colisoes)
            if self.sons_carregados:
                self.som_explosao.play()

        # 4. Verificar condições de fim de jogo
        if pygame.sprite.spritecollide(self.nave, self.inimigos, False):
            finalizado = True
            recompensa = -100  # Grande penalidade por morrer

        for inimigo in self.inimigos:
            if inimigo.rect.bottom >= self.nave.rect.top:
                finalizado = True
                recompensa = -100
                break

        if not self.inimigos:
            finalizado = True
            recompensa = 200  # Grande recompensa por vencer

        if finalizado and self.sons_carregados:
            pygame.mixer.music.stop()

        # 5. Obter o novo estado e retornar tudo
        novo_estado = self._get_state()
        info = {}  # Dicionário para informações de depuração (opcional)

        return novo_estado, recompensa, finalizado, info

    def _get_state(self):
        """
        Coleta o estado atual do jogo e o retorna como um vetor numpy.
        O vetor tem um tamanho fixo para ser compatível com a IA.
        """
        # Posição da nave (normalizada)
        estado_nave = [self.nave.rect.x / LARGURA_TELA]

        # Posições dos inimigos
        estado_inimigos = []
        for inimigo in self.inimigos:
            estado_inimigos.extend([inimigo.rect.x / LARGURA_TELA, inimigo.rect.y / ALTURA_TELA])
        # Preenche com valores padrão (-1) se houver menos inimigos que o máximo
        while len(estado_inimigos) < MAX_INIMIGOS * 2:
            estado_inimigos.extend([-1, -1])

        # Posições das balas
        estado_balas = []
        for bala in self.balas:
            estado_balas.extend([bala.rect.x / LARGURA_TELA, bala.rect.y / ALTURA_TELA])
        # Preenche com valores padrão (-1) se houver menos balas que o máximo
        while len(estado_balas) < MAX_BALAS * 2:
            estado_balas.extend([-1, -1])

        # Concatena tudo em um único vetor e o retorna
        estado = np.array(estado_nave + estado_inimigos[:MAX_INIMIGOS * 2] + estado_balas[:MAX_BALAS * 2])
        return estado

    def _criar_frota_inimiga(self):
        for linha in range(5):
            for coluna in range(10):
                inimigo = self._Inimigo(coluna * 60 + 50, linha * 40 + 50)
                self.todas_as_sprites.add(inimigo)
                self.inimigos.add(inimigo)

    def _mover_frota_inimiga(self):
        mudar_direcao = False
        for inimigo in self.inimigos:
            inimigo.rect.x += self.velocidade_inimigo_x
            if inimigo.rect.right > LARGURA_TELA or inimigo.rect.left < 0:
                mudar_direcao = True

        if mudar_direcao:
            self.velocidade_inimigo_x *= -1
            for inimigo in self.inimigos:
                inimigo.rect.y += 15

    def render(self):
        """Desenha o estado atual do jogo na tela."""
        if self.render_mode != 'human':
            return

        self.tela.fill(COR_PRETO)
        self.todas_as_sprites.draw(self.tela)
        self._mostrar_texto(f"Pontos: {self.pontos}", 24, LARGURA_TELA / 2, 10)
        self._mostrar_texto(f"FPS: {int(self.relogio.get_fps())}", 18, LARGURA_TELA - 50, 10)
        pygame.display.flip()
        self.relogio.tick(60)

    def _mostrar_texto(self, texto, tamanho, x, y):
        superficie_texto = self.fonte.render(texto, True, COR_BRANCO)
        rect_texto = superficie_texto.get_rect(center=(x, y))
        self.tela.blit(superficie_texto, rect_texto)

    def close(self):
        """Fecha o Pygame."""
        pygame.quit()
        sys.exit()

    # --- Classes Internas para Sprites ---
    # Aninhar as classes de sprite dentro do ambiente mantém tudo organizado.
    class _Nave(pygame.sprite.Sprite):
        def __init__(self, env):
            super().__init__()
            self.env = env
            self.image = pygame.Surface([60, 20])
            self.image.fill(COR_VERDE)
            self.rect = self.image.get_rect(centerx=LARGURA_TELA / 2, bottom=ALTURA_TELA - 20)

        def update(self):
            if self.rect.left < 0: self.rect.left = 0
            if self.rect.right > LARGURA_TELA: self.rect.right = LARGURA_TELA

        def atirar(self):
            if self.env.sons_carregados: self.env.som_tiro.play()
            bala = self.env._Bala(self.rect.centerx, self.rect.top)
            self.env.todas_as_sprites.add(bala)
            self.env.balas.add(bala)

    class _Inimigo(pygame.sprite.Sprite):
        def __init__(self, x, y):
            super().__init__()
            self.image = pygame.Surface([40, 30])
            self.image.fill(COR_VERMELHO)
            self.rect = self.image.get_rect(x=x, y=y)

    class _Bala(pygame.sprite.Sprite):
        def __init__(self, x, y):
            super().__init__()
            self.image = pygame.Surface([5, 15])
            self.image.fill(COR_BRANCO)
            self.rect = self.image.get_rect(centerx=x, bottom=y)
            self.velocidade_y = -10

        def update(self):
            self.rect.y += self.velocidade_y
            if self.rect.bottom < 0: self.kill()


if __name__ == '__main__':
    # Este bloco demonstra como usar o ambiente com um jogador humano.
    # O agente de IA usará uma lógica similar, mas sem o input do teclado.
    env = SpaceInvadersEnv(render_mode='human')
    estado = env.reset()
    finalizado = False
    pontuacao_total = 0

    while not finalizado:
        # Padrão: não fazer nada
        acao = 3

        # Coleta de input do jogador humano
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                finalizado = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            acao = 0
        elif keys[pygame.K_RIGHT]:
            acao = 1

        if keys[pygame.K_SPACE]:
            acao = 2

        # Executa a ação no ambiente
        novo_estado, recompensa, finalizado, info = env.step(acao)
        pontuacao_total += recompensa

        # Renderiza o jogo
        env.render()

        if finalizado:
            print(f"Fim de jogo! Pontuação da Recompensa: {pontuacao_total:.2f}")

    env.close()
