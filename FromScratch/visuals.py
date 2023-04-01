import pygame

WIDTH, HEIGHT = 450, 700
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Neural Network Visualization")
WINCOLOR = (211, 211, 211)  # light gray
FPS = 60


def drawWindow():
    WIN.fill(WINCOLOR)

    def line(C1, C2):
        pygame.draw.line(WIN, (0, 0, 0), C1, C2, 1)

    def dot(X, Y, Size=10):
        pygame.draw.circle(WIN, (255, 255, 255), (X, Y), Size)

    def IN(x, neurons):
        global INCoords
        INCoords = []
        y = (HEIGHT/neurons)-((HEIGHT/neurons) * 0.5)
        for i in range(1, neurons+1):
            dot(x, y, 2)
            INCoords.append((x, y))
            y += HEIGHT/neurons

    def HL1(x, neurons):
        global HL1Coords
        HL1Coords = []
        y = (HEIGHT/neurons)-((HEIGHT/neurons) * 0.5)
        for i in range(1, neurons+1):
            dot(x, y)
            HL1Coords.append((x, y))
            y += HEIGHT/neurons

    def HL2(x, neurons):
        global HL2Coords
        HL2Coords = []
        y = (HEIGHT/neurons)-((HEIGHT/neurons) * 0.5)
        for i in range(1, neurons+1):
            dot(x, y)
            HL2Coords.append((x, y))
            y += HEIGHT/neurons

    def OL(x, neurons):
        global OLCoords
        OLCoords = []
        y = (HEIGHT/neurons)-((HEIGHT/neurons) * 0.5)
        for i in range(1, neurons+1):
            dot(x, y)
            OLCoords.append((x, y))
            y += HEIGHT/neurons

    def Connections(fC, lC):
        global INCoords
        global HL1Coords
        global HL2Coords
        global OLCoords
        for ii in range(0, len(fC)):
            for i in range(0, len(lC)):
                line(fC[ii], lC[i])
    IN(50, 100)
    HL1(170, 5)
    HL2(290, 10)
    OL(410, 2)
    Connections(INCoords, HL1Coords)
    Connections(HL1Coords, HL2Coords)
    Connections(HL2Coords, OLCoords)
    IN(50, 100)
    HL1(170, 5)
    HL2(290, 10)
    OL(410, 2)
    pygame.display.update()


def main():
    clock = pygame.time.Clock()
    run = True
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        drawWindow()
    pygame.quit()



