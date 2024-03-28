# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 13:34:48 2024

@author: ncksh
"""

import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

def normalize(x):
    return x / np.linalg.norm(x)

def clamp(x, minval, maxval):
    return min(maxval, max(x,minval))

class SurvivalEnv(gym.Env):
    # Defines a survival world in which agents exist as a disk
    # in a 2D flatland. Agents must survive by finding food and avoiding
    # predators.
    
    metadata = {"render_modes":["human","rgb_array"], "render_fps":30}
    
    def __init__(self, render_mode = None, view_size=256, world_size=1024):
        
        # The observation space is the same as the rendered game, 
        # which is a size x size image
        self.observation_space = spaces.Box(0,255,shape=(3,view_size,view_size),dtype=np.uint8)
        
        
        # The actions pace is:
        # [ thrust in [-1,1], angular velocity in [-1,1]]
        #self.action_space = spaces.Box(-1.0,1.0,shape=(2,1), dtype=np.float32)
        # for discrete version, support 9 actions: thrust in (-1,0,1) x angular vel in (-1,0,1)
        self.action_space = spaces.Discrete(9)
        
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        # Rendering vars
        self.window = None
        self.window_size = view_size
        self.clock = None
        
        # Simulation vars
        self.world_size = world_size
        # these are all in units of pixels (per sec, per sec^2, etc)
        self.food_size_px = 5
        self.food_energy = 75
        self.max_vel = 150
        self.max_ang_vel = (np.pi *3.0)
        self.max_thrust = 200
        self.drag = 0.02
        # spawn a new layer of food every X steps
        self.food_rate: 300
        self.dt = 1.0/30.0
        self.max_energy = 150        
        self.view_size = view_size
        self.max_steps = 300
        self.steps = 0
        self.total_reward = 0
        self.food_collected = 0
        self.thrust_energy_cost = 1
        # when a new layer is spawned, 
        # food will be sampled at an average density of 
        # 1 food per N pixels
        self.food_density = 150
        
        # don't fill the map with food
        self.max_food = int((self.world_size * self.world_size) / (self.food_density * self.food_density))
        
        self.grid_size_px = 100
        
        self._init_state()
        
        self.val_map = {
            "x":0, "y":1, "radius":2,"velocity":3,"bearing":4,"angular_velocity":5,"thrust":6,"energy":7,
            "pos":[0,1]
        }
    
    def _init_state(self):
        self.steps = 0
        self.food_collected = 0
        self.total_reward = 0
        #internal state
        agent_init_pos = np.array([self.world_size/2.0, self.world_size/2.0])
        # accelerate food collision with spatial data structure
        # simple radix binning
        self.bins = 10
        self.bin_size = self.world_size/self.bins
        food = [[] for x in range (self.bins*self.bins)]
        food_spots = np.random.rand(self.max_food,2) * self.world_size
        for f in food_spots:
            bin_x = int(f[0] / self.bin_size)
            bin_y = int(f[1] / self.bin_size)
            food[bin_x * self.bins + bin_y].append(f)
        
        self._state = {
            # agent state: [x,y,radius,vx,vy,bearing,angular_v,thrust, energy]
            "agent":np.array([agent_init_pos[0],agent_init_pos[1],15.0,0.,0.,0.,0.,self.max_energy]),
            
            # Food: list of [x,y]
            # In the future, this should be a spatial data structure
            # for now randomly sample a single food near the agent
            # "food":[normalize(np.random.rand(2) - np.array([0.5,0.5] )) * (self.view_size/2.0) * 0.75 + agent_init_pos]
            "food":food
        }
        #print(f"food: {self._state['food'][0]}")
    
    def get_agent_val(self, val):
        return self._state["agent"][self.val_map[val]]
    def set_agent_val(self,key,val):
        self._state["agent"][self.val_map[key]] = val
    
    def init_pygame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

    def render(self, output_frame = False):
        return self._render(output_frame)
    
    def _render(self, output_frame = False):
        #print("Rendering")
        if self.render_mode == "human":
            self.init_pygame()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
       
        # Render window is centered on agent
        agent_to_world = self._state["agent"][0:2]
        world_to_agent = -1 * agent_to_world
        agent_to_view = np.array([self.window_size /2.0, self.window_size/2.0])
        world_to_view = world_to_agent + agent_to_view
        
        #print(f"world_to_agent:{world_to_agent}")
        
        # First, the grid
        for x in range(int(self.world_size / self.grid_size_px)):
            # horizontal
            pygame.draw.line(
                canvas,
                (200,200,200),
                tuple(np.array([0, self.grid_size_px * x]) + world_to_view),
                tuple(np.array([self.world_size, self.grid_size_px* x]) + world_to_view),
                width=1,
            )
            # vertical
            pygame.draw.line(
                canvas,
                (200,200,200),
                tuple(np.array([self.grid_size_px * x, 0]) + world_to_view),
                tuple(np.array([self.grid_size_px * x, self.world_size]) + world_to_view),
                width=1,
            )
        
        # Then food
        # need to overlap view window with food bins
        bin_x_start = clamp(int((-self.view_size/2.0 + agent_to_world[0]) / self.bin_size),0,self.bins-1)
        bin_x_end = clamp(int((self.view_size/2.0 + agent_to_world[0]) / self.bin_size),0,self.bins-1)
        bin_y_start = clamp(int((-self.view_size/2.0 + agent_to_world[1]) / self.bin_size),0,self.bins-1)
        bin_y_end = clamp(int((self.view_size/2.0 + agent_to_world[1]) / self.bin_size),0,self.bins-1)
        for bin_x in range(bin_x_start,bin_x_end+1):
            for bin_y in range(bin_y_start, bin_y_end+1):
                food_bin = self._state["food"][bin_x * self.bins + bin_y]
                for f in food_bin:
                    pygame.draw.circle(
                        canvas,
                        (0, 255, 0),
                        tuple(f + world_to_view),
                        self.food_size_px,
                    )
                     
        # Draw the agent on top of everything else
        bearing = self.get_agent_val("bearing")
        radius = self.get_agent_val("radius")
        # -- Body
        pygame.draw.circle(
            canvas,
            (0, 255, 255),
            tuple(agent_to_view),
            radius,
        )
        # -- Bearing indicator
        pygame.draw.line(
            canvas,
            0,
            tuple(agent_to_view),
            tuple(agent_to_view + radius * np.array([np.cos(bearing), np.sin(bearing)])),
            width=2,
        )
        
        # Draw the state indicators for the agent to use
        # agent state: [x,y,radius,v,bearing,angular_v,thrust, energy]
        # draw black bar 30 px high
        # (left,top,width,height)
        # top left of screen is 0,0
        pygame.draw.rect(canvas,0,(0,0,self.window_size,30))
        
        # for each stat, draw a 20x20 box with the color-encoded value
        # evenly spaced across the window
        stats = np.array([
            # values will range from 0-1
            self.get_agent_val("velocity")/self.max_vel,
            (self.get_agent_val("thrust")+1.0)/2.0,
            self.get_agent_val("energy")/self.max_energy]
        )
        for x in range(stats.shape[0]):
            pygame.draw.rect(canvas,int(stats[x]*255),(10 + x * 30, 5, 20,20))
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        if output_frame:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(2,0,1)
            )
                
    def step(self, action):
        reward = 0
        energy = self.get_agent_val("energy")
        truncated = self.steps >= self.max_steps
        terminated = self.get_agent_val("energy") <= 0
        
        if not truncated and not terminated:
            #print("Stepping...")
            action = self.action2inputs(action)
            self.set_agent_val("thrust",clamp(action["thrust"],-1,1))
            self.set_agent_val("angular_velocity",clamp(action["angular_velocity"],-1,1))
            #print(f"thrust:{action[0]}, angular_vel:{action[1]}")
            # update agent state
            pos = self.get_agent_val("pos")
            #TODO: we don't need velx and vely. We only need them to render the individual components on the screen.
            # velocity should be a single scalar to represent magnitude and use the bearing to turn into a 2d vector.
            vel = self.get_agent_val("velocity")
            bearing = self.get_agent_val("bearing")
            angular_vel = self.get_agent_val("angular_velocity")
            next_pos = pos + normalize(np.array([np.cos(bearing), np.sin(bearing)])) * vel * self.dt
            next_vel = vel + self.get_agent_val("thrust") * self.max_thrust * self.dt - self.drag * vel
            next_bearing = (bearing + ( angular_vel * self.max_ang_vel * self.dt)) % (2.0*np.pi)
            
            self.set_agent_val("pos",next_pos)
            self.set_agent_val("velocity",next_vel)
            self.set_agent_val("bearing",next_bearing)
            
            
            food_eaten = 0
            food_to_remove = []
            bin_x = clamp(int(pos[0]/self.bin_size),0,self.bins-1)
            bin_y = clamp(int(pos[1]/self.bin_size),0,self.bins-1)
            
            food = self._state["food"][bin_x*self.bins + bin_y]
            # resolve food 
            for f in range(len(food)):
                if np.linalg.norm(pos-food[f]) <= self.get_agent_val("radius"):
                    food_eaten = food_eaten +1
                    food_to_remove.append(f)
            # remove eaten food
            # since our removal indices are guaranteed to be in order,
            # construt a new list in a single pass by skipping appending
            # the items to be removed in order
            fix = 0
            new_food = []
            for e in food_to_remove:
                while fix < e and fix < len(food):
                    new_food.append(food[fix])
                    fix = fix + 1
                # skip the matching ele
                fix = fix + 1
            while fix < len(food):
                new_food.append(food[fix])
                fix = fix + 1
            self._state["food"][bin_x*self.bins+bin_y] = new_food
            self.food_collected = self.food_collected + food_eaten
            
            # update energy
            energy_prev = energy
            thrust_cost = abs(action["thrust"]) * self.thrust_energy_cost
            energy = clamp(energy + food_eaten * self.food_energy - thrust_cost -1, -1, self.max_energy)
            self.set_agent_val("energy", energy)
            #print(f"vel:{vel}, bearing:{bearing}, next_pos:{next_pos}, next_vel:{next_vel}, next_bearing:{next_bearing}, energy:{energy}")
            
            # terminate if energy gone
            if energy <= 0:
                terminated = True
            
            # reward is energy gained this frame, plus bonus for finding food, plus 1  for living
            reward = energy - energy_prev + food_eaten * 50 + 1
            self.total_reward = self.total_reward + reward
            self.steps = self.steps + 1
            
        if terminated or truncated:
            print(f"End of Episode! total steps: {self.steps}, food collected: {self.food_collected}, final energy: {energy}, total_reward: {self.total_reward}")
        observation = self.render(True)
        info = self._get_info()
        # give outputs
        return observation, reward, terminated, truncated, info
    
    def play(self):
        self.init_pygame()
        running = True
        while running:
            # poll for events
            # pygame.QUIT event means the user clicked X to close your window
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break 
            self.render(False)
            
            thrust = 0
            angular_vel = 0
            
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                thrust = 1
            if keys[pygame.K_s]:
                thrust = -1
            if keys[pygame.K_a]:
                angular_vel = -1
            if keys[pygame.K_d]:
                angular_vel = 1
            
            observation, reward, terminated, truncated, info = self.step(self.inputs2action({"thrust":thrust, "angular_velocity":angular_vel}))
            running = not terminated and not truncated
            
            self.clock.tick(self.metadata["render_fps"])
        pygame.quit()
    
    def action2inputs(self, action):
        ret = {}
        ret["thrust"] = int(action/3)-1
        ret["angular_velocity"] = int(action % 3) - 1
        return ret
    
    def inputs2action(self, inputs):
        return int((inputs["thrust"] + 1) * 3) + int((inputs["angular_velocity"] + 1) % 3)
    
    
    def _get_obs(self):
        return self._render(True)
    
    def _get_info(self):
        return {"agent_pos":self.get_agent_val("pos"), "energy":self.get_agent_val("energy")}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_state();
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), self._get_info()
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

        
        