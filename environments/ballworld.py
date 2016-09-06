import numpy
import theano
import theano.tensor as T
import random
import pygame
import sys
import numpy as np

pi = np.pi

class Environment:
    def __init__(self, tiling):
        self.tiling = tiling
        self.actions = [-1,1]
        self.state = [0,0]

    def startEpisode(self,*args):
        self.state = [0,0]

    def toRepr(self, s=None):
        if s is None:
            s = self.state

        ntiles = self.tiling
        p = numpy.zeros((ntiles*ntiles), dtype='float32')
        p[int(self.state[0]+10)] = 1
        return numpy.float32(p)
        
    def isEpisodeOver(self):
        return self.state[0] > 10

    def takeAction(self, a):
        self.state[0] += a
        if self.state[0] >= 9: return 0.8
        if self.state[0] < -10: self.state[0] = -10
        #if a > 0: return 0.1
        return -1

    
            
class Ball2D:
    def __init__(self,x,y,t,radius=1):
        self.x = x
        self.y = y
        self.r = radius
        if t:
            color = [0,1,0,0]
            reward = 1
        else:
            color = [0,0,1,0]
            reward = -1
        self.color = color
        self.isSolid = False
        self.reward = reward
    
    def distanceToRay(self,ray):
        """returns a pair (d,t)
        d is the shortest distance from the ball to the ray

        t is the distance from the origin of the ray to the
        intersection of the ball and the ray"""
        x,y,u,v = ray

        """
        project (x-X,y-Y) on (v,-u)
        """
        d = ((x-self.x)*v - (y-self.y)*u) #/ np.sqrt(u*u+v*v)
        #print '---',[d*v,d*-u]
        #project center-rorigin (p) onto (u,v) and take the norm, norm of p dot (u,v) / |(u,v)|
        tm = ((self.x-x)*u + (self.y-y)*v)# / np.sqrt(u*u+v*v)
        if tm < 0: # the ball is behind the ray, we dont care
            return (numpy.nan,numpy.nan)
        #print ray,[self.x,self.y],d,tm
        #print "distances",d**2,tm**2,d**2+tm**2,(self.x-x)**2+(self.y-y)**2
        tz = np.sqrt(self.r*self.r - d*d)
        t = abs(tm) - tz
        #print t**2
        #t = abs(a*self.x+b*self.y+c) / np.sqrt(x*x+y*y)
        return (d,t)
        
    def intersectsRay(self, ray):
        return abs(self.distanceToRay(ray)[0]) <= self.r
    
    def intersectsBall(self, b):
        return np.sqrt((b.x-self.x)**2+(b.y-self.y)**2) <= self.r + b.r
            
        
    def __str__(self):
        return "<Ball %.2f:%.2f>"%(self.x,self.y)

class Wall2D:
    def __init__(self,x0,y0,x1,y1):
        self.p = [x0,y0,x1,y1]
        self.length = np.sqrt((x1-x0)**2 + (y1-y0)**2)
        self.ray = [x0,y0, (x1-x0)/self.length, (y1-y0)/self.length]#[y1-y0, x1-x0, (x0-x1)*y0 + (y1-y0)*x0,   x0, y0]
        self.color = [1,0,0,0]
        self.isSolid = True

    def distanceToRay(self, ray):
        x,y,u,v = ray
        h,j,k,l = self.ray
        # l u-k v!=0 and 
        #t = (h l-j k+k y-l x)/(l u-k v)
        t = (h*l - j*k + k*y - l*x) / (l*u - k*v)
        #d = (h v-j u+u y-v x)/(l u-k v)
        d = (h*v - j*u + u*y - v*x) / (l*u - k*v)
        return (d, t)

    def distanceToRay_(self, ray):
        a,b,c,u,v = ray
        j,k,l,r,s = self.ray
        y = (a*l-c*j) / (b*j-a*k)
        x = (c*k-b*l) / (b*j-a*k)
        t = np.sqrt((u-x)**2+(v-y)**2)
        return (0, t)

    def intersectsRay(self, ray):
        return 0 <= self.distanceToRay(ray)[0] <= self.length

    def intersectsBall(self, b):
        d,t = b.distanceToRay(self.ray)
        return 0 <= t <= self.length and d <= b.r

    def __str__(self):
        return "<Wall %s>"%(str(self.p))
        
        #a,b,c = ray
  
class BallAgent:
    def __init__(self,x,y,world):
        self.world = world
        self.b = Ball2D(x,y,1)
        self.orientation = 0 # radians
        

    def moveForward(self, q):
        ray = self.getRay()
        newpos = [ray[0] + ray[2] * q, ray[1] + ray[3] * q]
        b = Ball2D(newpos[0], newpos[1], 1)
        r = -0.1
        balls = []
        for o in self.world.objects:
            if o == self: continue
            if o.intersectsBall(b):
                if o.isSolid:
                    return [r,[]]
                else: # we touched a reward ball!
                    r += o.reward
                    #print b,"intersect",o
                    balls.append(o)
        self.b = b
        return [r,balls]

    def getRay(self, theta=0):
        # return [x,y, u,v]
        st = np.sin(self.orientation+theta)
        ct = np.cos(self.orientation+theta)
        return [self.b.x,self.b.y, ct, st]

    def getRay_(self):
        # return ax+by+c=0 as [a,b,c]
        # and the origin of the ray [x,y]
        b = self.b
        st = np.sin(self.orientation)
        ct = np.cos(self.orientation)
        return [-st,-ct,ct*(b.y+st)+st*(b.x+ct),
                self.b.x,self.b.y]
      
class Void:
    def __init__(self):
        self.color = [0,0,0,1]

class BallWorld:
    def __init__(self):
        self.screen = None
        self.objects = []
        self.agent = BallAgent(4,5,self)
        self.void = Void()
        self.actions = ['l','r','f']
        #self.actions = ['l','r']
        self.angles = [i*pi/24 for i in range(-5,6)]
        #self.angles = [i*pi/24 for i in range(0,1)]
        print self.angles
        self.nfeatures = 5*len(self.angles)
        self.lastMinray = 10
        self.r = lambda: np.random.uniform(1,19)
    def startEpisode(self,randomStart=False):
        self.objects = [Wall2D(0,0,20,0),Wall2D(0,0,0,20),
                        Wall2D(0,20,20,20),Wall2D(20,0,20,20)]
        #self.objects += [Ball2D(self.r(),self.r(), True) for i in range(5)]
        self.objects += [Ball2D(i*2,i*2, True) for i in range(5)]
        #self.objects += [Ball2D(self.r(),self.r(), False) for i in range(25)]

        
        self.agent = BallAgent(self.r(),self.r(),self)
        self.agent.orientation = np.random.uniform(0,2*3.14)

    def toRepr(self, maxDist = 20.):
        minDists = []
        data = []
        for theta in self.angles:
            minDist = maxDist # is maximum sight distance
            minObj = self.void
            ray = self.agent.getRay(theta)
            for o in self.objects:
                d,t = o.distanceToRay(ray)
                if not o.intersectsRay(ray):
                    continue
                #print o,d,t
                if 0 <= t <= minDist:
                    minDist = t
                    minObj = o
            data += [minDist/maxDist]+minObj.color
            minDists.append(minDist)
        self.lastMinray = minDists
        return np.float32(data)
        
    def isEpisodeOver(self):
        return False

    def takeAction(self, a):
        if a == 'l':
            self.agent.orientation += np.pi / 16
        elif a == 'r':
            self.agent.orientation -= np.pi / 16
        #if 1:
        elif a == 'f':
            r,b = self.agent.moveForward(0.8)
            for i in b:
                i.x = self.r()
                i.y = self.r()
            return r
        if a != 'f':
            r,b = self.agent.moveForward(0.2)
            for i in b:
                i.x = self.r()
                i.y = self.r()
            return r
            return 0#-0.05 # we don't want the agent to stay in place too long


    def setupVisual(self):
        import pygame
        self.screen = pygame.display.set_mode((202,202))
        
    def draw(self):
        if self.screen is None: return
        self.screen.fill((0,0,0))
        I = numpy.int32
        for i in self.objects:
            if isinstance(i, Wall2D):
                pygame.draw.line(self.screen, [255,255,255],
                                 I([i.p[0]*10,i.p[1]*10]),
                                 I([i.p[2]*10,i.p[3]*10]))
            elif isinstance(i, Ball2D) and i.reward > 0:
                pygame.draw.circle(self.screen, [128,255,128],
                                   I([i.x*10,i.y*10]), 10)
            elif isinstance(i, Ball2D) and i.reward <= 0:
                pygame.draw.circle(self.screen, [255,128,128],
                                   I([i.x*10,i.y*10]), 10)
        for theta,Q in zip(self.angles,self.lastMinray):
            ray = self.agent.getRay(theta)
            Q = Q * 10
            pygame.draw.line(self.screen, [128,128,255],
                             [ray[0]*10,ray[1]*10],
                             [ray[0]*10+ray[2]*Q,ray[1]*10+ray[3]*Q])
        pygame.draw.circle(self.screen, [255,255,255],
                           [int(self.agent.b.x*10),
                            int(self.agent.b.y*10)], 10)
        pygame.display.flip()
        

class QLearning:
    def __init__(self):
        memory = []
        #self.screen = pygame.display.set_mode((800,600))
        # cool avec epsilon .95, lr 0.0001, gamma 0.8, batch 100
        batch_size = 32
        epsilon = .5
        gamma = 0.8
        lr = 0.001
        env = BallWorld()
        
        #env.setupVisual()
        
        actions = env.actions
        nactions = len(actions)
        nfeatures = env.nfeatures

        net = LayeredNetwork(T.matrix(), #env.stateActionPairSize
                             [nfeatures, 100, 100, 100, nactions],
                             [T.tanh,T.tanh,T.tanh,lambda x:x])
        net.build()

        
        until_restart = 10
        until_restart_max = until_restart
        max_episode_len = 1000

        for episode in range(1000):
            env.startEpisode(episode % 10 != 0) # do a non random start every 10 episodes
            S = [list(env.toRepr())]
            Phi = [env.toRepr()]
            cnt = 0
            cnts = [0,0,0]
            totalr = 0
            while not env.isEpisodeOver():
                if episode < 20:
                    if cnt % 100==0:env.draw()
                else:
                    env.draw()
                #env.agent.b.x = numpy.random.randint(1,9)
                #env.agent.b.y = numpy.random.randint(1,9)
                #env.agent.orientation = numpy.random.uniform(0,2*3.1415)

                cnt+=1 
                Q_sa = net.evaluate([env.toRepr()])[0]
                #print net.evaluate([env.toRepr([0,0])]),Q_sa
                if numpy.random.rand() > epsilon:
                    best_a = Q_sa.argmax()
                    best_Q = Q_sa.max()
                else:
                    best_a = numpy.random.randint(0,len(env.actions))
                    best_Q = Q_sa[best_a]
                cnts[best_a] += 1
                r = env.takeAction(actions[best_a])
                totalr += r
                Phi.append(env.toRepr())
                memory.append([Phi[-2], best_a, r, Phi[-1]])
                if cnt % 5 == 0 or 1:
                    batch = [random.choice(memory) for i in range(batch_size)]
                    ys = numpy.float32([r + gamma * net.evaluate([phip])[0].max() for _,_,r,phip in batch])
                    As = numpy.uint8([a for _,a,_,_ in batch])
                    phis = numpy.float32([phi for phi,_,_,_ in batch])
                    
                    net.learn(phis, ys, As, lr)
                    

                #print cnt,"           \r",
                sys.stdout.flush()
                if cnt >= max_episode_len:
                    #epsilon += (0.5-epsilon) * 0.1 * epsilon
                    #until_restart -= 1
                    break
            print "Episode",episode,"over", cnt, len(memory), epsilon,cnts,totalr," "*10
            #self.show_Qfunc(net,env)
            epsilon *= 0.96
            memory = memory[-20000:]

    def show_Qfunc(self,net,env):
        z = numpy.zeros((800,600,3))
        for i in range(100):
            for j in range(100):
                c = [0,0,0]
                p = env.toRepr([i/100.*1.4-1.2,
                                j/100.*2*0.07-0.07])
                Q_sa = net.evaluate([p])[0]
                best_Q = Q_sa.max()
                best_a = Q_sa.argmax()
                z[i,j,best_a] = best_Q
                z[i+100,j,0] = Q_sa[0]
                z[i+100,j+100,1] =Q_sa[1]
                z[i,j+100,2] = Q_sa[2]
        z -= z.min()
        z = z / z.max()
        pygame.surfarray.blit_array(self.screen, numpy.uint8(255*z))
        pygame.display.flip()



a = BallAgent(5,0,None)
a.orientation = 0#-pi*3/4.
r = a.getRay()
b = Ball2D(0,0,False)
print b.distanceToRay(r)
print b.intersectsRay(r)


if __name__ == "__main__":
    m = QLearning()



