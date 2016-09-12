import numpy

class POGridworld:
    def __init__(self, ncolors, grid, start_position,
                 color_reward_map={},
                 unpassable_colors=[0]):
        self.crm = color_reward_map
        self.ncolors = ncolors
        self.grid = grid
        self.start_position = start_position
        self.actions = {'up':0,'down':1,'left':2,'right':3}
        self.deltas = [[-1,0],[1,0],[0,-1],[0,1]]
        self.pos = start_position
        self.unpassable_colors = unpassable_colors
        self.getobs = self.getobs_square

    def act(self,a):
        d = self.deltas[a]
        r = 0
        if self.grid[self.pos[0]+d[0], self.pos[1]+d[1]] not in self.unpassable_colors:
            self.pos = [self.pos[0]+d[0], self.pos[1]+d[1]]
        colorat = self.grid[self.pos[0], self.pos[1]]
        if colorat in self.crm:
            r += self.crm[colorat]
        return r

    def getfullobs(self):
        n = numpy.prod(self.grid.shape)
        o = numpy.zeros((n,self.ncolors),'float32')
        o[n, self.grid.flatten()] = 1
        return o.flatten()

    def getobs_square(self,radius=1):
        n = radius*2+1
        o = numpy.zeros((n*n,self.ncolors),'float32')
        d = self.grid[self.pos[0]-radius:self.pos[0]+radius+1,
                      self.pos[1]-radius:self.pos[1]+radius+1]
        o[numpy.arange(n*n), d.flatten()] = 1
        return o.flatten()

    def matshow(self):
        import matplotlib.pyplot as pp
        pp.matshow(self.grid)
        pp.show()


def I_maze(corridor_length,I_width=3,p=0.5):
    assert (I_width%2) == 1
    n = corridor_length

    grid = numpy.zeros((n+4,I_width+2),'int32')
    center = (I_width+1) / 2
    for i in range(n):
        grid[i+2,center] = 1
    for i in range(I_width):
        grid[1,i+1] = 1
        grid[-2,i+1] = 1
    grid[-2,1] = 2
    grid[-2,-2] = 3
    if numpy.random.uniform(0,1) < p:
        grid[1,1] = 4
        crm = {2:1,3:-1}
    else:
        grid[1,1] = 5
        crm = {2:-1,3:1}
    start_pos = [1,center]
    pog = POGridworld(6, grid, start_pos, crm)
    return pog


if __name__ == "__main__":
    m = I_maze(4)
    print m.getobs().reshape((-1,6))
    print m.grid
