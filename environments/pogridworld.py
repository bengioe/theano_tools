import numpy


deltas = [[-1,0],[1,0],[0,-1],[0,1]]

class POGridworld:
    def __init__(self, ncolors, grid, start_position,
                 color_reward_map={},
                 unpassable_colors=[0],
                 stop_after='hit_any_reward'):
        self.crm = color_reward_map
        self.ncolors = ncolors
        self.grid = grid
        self.start_position = start_position
        self.actions = {'up':0,'down':1,'left':2,'right':3}
        self.deltas = deltas
        self.pos = start_position
        self.unpassable_colors = unpassable_colors
        self.getobs = self.getobs_square
        self.obsradius = 1
        self.remove_goals = True
        self.stop_after = stop_after
        self.isDone = False

    def act(self,a):
        d = self.deltas[a]
        r = 0
        if self.grid[self.pos[0]+d[0], self.pos[1]+d[1]] not in self.unpassable_colors:
            self.pos = [self.pos[0]+d[0], self.pos[1]+d[1]]
        colorat = self.grid[self.pos[0], self.pos[1]]
        if colorat in self.crm:
            r += self.crm[colorat]
            if self.remove_goals:
                self.grid[self.pos[0], self.pos[1]] = 1
            if self.stop_after == 'hit_any_reward':
                self.isDone = True
        return r

    def getfullobs(self,*args):
        n = numpy.prod(self.grid.shape)
        o = numpy.zeros((n,self.ncolors),'float32')
        o[numpy.arange(n), self.grid.flatten()] = 1
        return o.flatten()

    def getobs_square(self,radius=None):
        if radius is None: radius = self.obsradius
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

def random_blob_maze(size,density=0.33,indicator=True,prob=0.5,goal_count=1):
    grid=numpy.zeros((size+2,size+2),'int32')
    grid[1:-1,1:-1] = 1

    r = lambda: numpy.random.randint(1,size+1)
    x0,x1,x2,x3 = numpy.random.choice(range(1,size+1),4,replace=False)
    if goal_count == 1:
        grid[x0,r()] = 2
        grid[x1,r()] = 3
    else:
        for i in range(goal_count):
            grid[r(),r()] = 2
            grid[r(),r()] = 3
    if numpy.random.uniform(0,1) < prob:
        grid[x2,r()] = 4
        crm = {2:1,3:-1}
    else:
        grid[x2,r()] = 5
        crm = {2:-1,3:1}
    start_pos = x3,r()
    grid[start_pos[0],start_pos[1]]=1

    print start_pos
    n = [size*size*density] # number of 1s to remove
    def first_one(g):
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                if g[i,j] == 1:
                    return [i,j]

    def try_remove(x,y):
        if not (0<=x<size+2 and 0<=y<size+2): return False
        if grid[x,y] != 1: return False
        if x==start_pos[0] and y==start_pos[1]: return False
        # now we check if removing x,y disconnects the graph
        gc = grid.copy()
        gc[x,y] = 0
        mark = numpy.zeros((size+2,size+2),'int32')
        def dfs(i,j):
            if 0<=i<size+2 and 0<=j<size+2 and gc[i,j] > 0 and mark[i,j] == 0:
                mark[i,j] = 1
                for d in deltas:
                    dfs(i+d[0], j+d[1])
        dfs(*start_pos)
        if ((mark==1)!=(gc>0)).any():
            return False # the graph is disconnected
        # now we need to check if we can still reach the objectives
        # using only '1' nodes
        mark = numpy.zeros((size+2,size+2),'int32')
        def dfs(i,j):
            if  0<=i<size+2 and 0<=j<size+2 and gc[i,j] == 1 and mark[i,j] == 0:
                mark[i,j] = 1
                for d in deltas:
                    dfs(i+d[0], j+d[1])
                    mark[i+d[0],j+d[1]] = 1 # also mark neighbours of '1's as accessible
        dfs(*start_pos)
        if (mark[gc==2] == 0).any() or (mark[gc==3] == 0).any():
            return False # dfs didn't mark one of the objectives
        # all good, remove node
        grid[x,y] = 0
        n[0] -= 1
        return True

    free_cells = [(i,j) for i in range(1,size+1) for j in range(1,size+1)]
    numpy.random.shuffle(free_cells)
    while n[0] > 0 and len(free_cells):
        # try to remove random node
        x,y = free_cells.pop(0)
        if try_remove(x,y): # if we did remove
            # try a neighbour
            while numpy.random.random() < density and n[0]>0:
                d = deltas[numpy.random.randint(0,len(deltas))]
                x += d[0]; y += d[1]
                try_remove(x,y)

    free = list(numpy.int32((grid==1).nonzero()).T)

    start_pos = free.pop(numpy.random.randint(0,len(free)))
    pog = POGridworld(6,grid,start_pos, crm,
                      stop_after='hit_any_reward' if goal_count==1 else 'never')
    return pog

if __name__ == "__main__":
    # tiles explored
    # back and forth between flags

    #numpy.random.seed(5)
    import time
    t0 = time.time()
    for i in range(100):
        print i
        m = random_blob_maze(10)
    t1 = time.time()
    print t1-t0
    print m.getobs().reshape((-1,6))
    print m.grid, m.grid[m.grid==2].shape,m.grid[m.grid==3].shape,


    # t0 = time.time()
    # m = I_maze(5)
    # t1 = time.time()
    # print t1-t0
    # print m.getobs().reshape((-1,6))
    # print m.grid
