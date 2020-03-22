# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 09:03:06 2020

@author: vyas
"""
import numpy as np
import random as random
import matplotlib.pyplot as plt

class Weight:
    def __init__(self, name, init_val):
        self.name = name
        self.w = [init_val]
    
    def GetCurrentWeight(self):
        return self.w[(len(self.w) - 1)]
    
    def UpdateWeight(self, w):
        self.w.append(w)
        return
    
    def ReturnData(self):
        return self.w

class Grid:
    def __init__(self, name, init_row, init_col):
        self.valid_action = np.array([
                                     ["RD" ,"A",  "LRD",  "B",    "LD"], 
                                     ["URD","LRUD", "LRUD", "LRUD",  "LUD",],
                                     ["URD","LRUD", "LRUD", "LRUD",  "LUD",],
                                     ["URD","LRUD", "LRUD", "LRUD",  "LUD",],
                                     ["UR", "LRU",  "LRU",    "LRU",  "LU"]])
        self.row = init_row
        self.col = init_col
        
    def GetNextStateReward(self, action):
        valid_actions = self.valid_action[self.row][self.col]
        result = valid_actions.find(action)
        ## Special cases
        reward = -1
        if(valid_actions == "A"):
            reward = 10
            self.row = 4
            self.col = 1
        elif (valid_actions == "B"):
            self.row = 2
            self.col = 3
            reward = 5
        ## See whether it was a valid action
        elif (result == -1):
            reward = -1
        else:
            reward = 0
            if(action == "U"):
                self.row -= 1
            elif (action == "D"):
                self.row +=1
            elif (action == "R"):
                self.col += 1
            elif (action == "L"):
                self.col -= 1
        
        return reward, self.row, self.col
    

    def GetNextRandomAction(self):
        actions = np.array(["U", "D", "R", "L"])
        return actions[int(np.floor(random.random()*4))]
    
    def GetCurrentState(self):
        return self.row*5 + self.col
    

class Compute:
    def __init__(self, name,  c, grid, total_steps):
        self.c = c
        self.t = 0
        self.alphat = 0.5 # init value
        self.w = []
        for i in np.arange(25):
            self.w.append(Weight(i,0))
        
        self.rhattplus1 = 0
        self.rhatt = 0
        self.grid = grid
        self.total_steps = total_steps
        self.rhatarray = [0]
    
    def Update(self, t, reward, current_state, next_state):
        if (t%10 == 0):
            self.alphat = 1.0/float(np.ceil((self.t+1)/10))
        self.t = t
        self.beta = self.alphat*self.c
        self.rhattplus1 = self.rhatt + self.beta*(reward - self.rhatt)
        self.rhatt = self.rhattplus1
        w_next_state = self.w[next_state]
        w_current_state = self.w[current_state]
        self.deltat = reward - self.rhattplus1 + w_next_state.GetCurrentWeight() - w_current_state.GetCurrentWeight()
        self.rhatarray.append(self.rhatt)
        for i in np.arange(25):
            if (i == current_state):
                w_current_state.UpdateWeight(w_current_state.GetCurrentWeight() + self.alphat*self.deltat)
            else:
                x = self.w[i]
                x.UpdateWeight(x.GetCurrentWeight())
    
    def GetRhat(self):
        return self.rhatarray
    
    def Print(self):
        print("c = ", self.c)
        
    def PrintW(self):
        print("[")
        for i in np.arange(5):
            print(" [",end =" ")
            for j in np.arange(5):
                state_num = i*5+j
                state = self.w[state_num]
                a = round(state.GetCurrentWeight(), ndigits=2)
                print(" %+0.2f" % a, end =" ")
            print(" ],")
        print("]")
                
    def GetData(self, state_num):
        return self.w[state_num].ReturnData()
    

class PlotMyGraph:
    def __init__(self):
        plt.figure(figsize=(8,5))
        return
    
    def Plot(self, w, state_num, legend):
        
        plt.plot(w, label=legend)
        #plt.gcf().autofmt_xdate()
    
    def Show(self):
        plt.xlabel("time")
        plt.ylabel("rhat")
        plt.grid()
        plt.legend()
        plt.show()
        return              
            

def UT(steps):
    g = Grid("assignment", 4, 4) # row = 0, col = 0
    c = Compute("test", 1, g, steps)
    c.Print()
    t = 0
    while (t < steps):
        current_state = g.GetCurrentState()
        action = g.GetNextRandomAction()
        reward, r,col = g.GetNextStateReward(action)
        next_state = g.GetCurrentState()
        #c.Print()
        c.Update(t, reward, current_state, next_state)
        #print("current_state: ", current_state, "Action: ", action, "Reward: ", reward, "NextState: ", next_state)
        t +=1
    
    c.PrintW()
    p = PlotMyGraph()
    #p.Plot(c.GetData(0), 0,  "State 1")
    #p.Plot(c.GetData(12), 11, "State 13")
    #p.Plot(c.GetData(24), 24, "State 25")
    p.Plot(c.GetRhat(), 0, "rhat")
    p.Show()
    
    
    return        

if ("__main__" == __name__):
  
    UT(50000)
