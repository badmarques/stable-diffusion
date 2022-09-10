from concurrent.futures import ThreadPoolExecutor
from operator import truediv
from tracemalloc import start
from optimizedSD.renderer import NeuralRender
import pygame
from pygame import surfarray
from pygame.display import set_mode, flip
from pygame.constants import SCALED, OPENGL
from pygame.event import get
from pygame import Surface, Rect
from pygame import Color

from pygame.time import Clock
import numpy as np
from timeit import default_timer as timer
import asyncio
import time
from math import inf

from Scene import Scene, Transition, build_plot
import Storytelling


class Game():
    def __init__(self) -> None:
        pygame.init()
        self.font = pygame.font.SysFont("arial", 24)
        self.screen_surface = set_mode(size=(512, 712), flags=SCALED, vsync=1)
        
        self.neural_renderer = NeuralRender()
        self.neural_surface = Surface((512,512))
        self.neural_thread = None # executor thread
        self.neural_queue = []
        
        
        self.text_surface = None
        self.text_rendering_time = 0.0
        self.text_rendering_done = False
        #self.ui_surface = pygame.image.load("./UI.png")
        self.ui_surface = Surface((512,200))

        self.clock = Clock()

        self.states = {
            "live_prompt" : "" ,    #live prompt text
            "prompt_done" : False,  #done writing prompt
            "scene" : None,         #current scene
            "scene_done" : False    #ready to transition
        }



    def _render_text(self, text, elapsed_time):
        
        if(self.text_rendering_time >= 1.0):
            self.text_surface = None
            self.text_rendering_done = True
            self.text_rendering_time = 0.0
            return

        if(not self.text_surface):
            self.text_surface = self.font.render(text, True, Color(255, 255, 255))
            self.text_rendering_time=0.0

        size = self.font.size(text)
        crop_rect = pygame.Rect(0,0, self.text_rendering_time *size[0], size[1])
        self.screen_surface.blit(self.text_surface, (16, 544), crop_rect)

        self.text_rendering_time+= (elapsed_time/1000.0)
        ##TODO break lines


    def update(self):
        with ThreadPoolExecutor(max_workers=1) as executor:

            while True:
                elapsed = self.clock.tick()

                if(not self.neural_thread):
                    if ( len(self.neural_queue) > 0):
                        self.neural_thread = executor.submit(self.neural_renderer.sample, self.neural_queue.pop(0))
                else:
                    if (self.neural_thread.done()): #done rendering neural image
                        img = self.neural_thread.result()
                        surfarray.blit_array(self.neural_surface, img.transpose(1,0,2))
                        self.screen_surface.blit(self.neural_surface, (0,0))
                        self.neural_thread = None

                if(self.states["prompt_done"]): #done typing prompt
                    
                    self.neural_queue.append(self.states["live_prompt"])
                    #self.neural_thread = executor.submit(self.neural_renderer.sample, self.states["live_prompt"])
                    self.states["live_prompt"] = ""
                    self.states["prompt_done"] = False

                if self.states["scene"].text and not self.text_rendering_done:
                    self._render_text(self.states["scene"].text, elapsed)

                if self.states["scene_done"]: #check transition
                    transition = self.storytelling_plot[self.states["scene"]]
                    if transition.condition:
                        self.load_scene(transition.scene_to)

                events = list(get())
                self.event_handler(events)
                flip()


    def event_handler(self, events):
        for event in events:
            if (event.type == pygame.KEYUP):

                if (event.key == pygame.K_RETURN ):
                    self.states["prompt_done"] = True
                    self.states["scene_done"] = True
                    
                elif( self.states["scene"].prompt_enabled and self.states["scene"].prompt_content == "free"):
                    self.states["live_prompt"] += event.unicode
                    self.states["prompt_done"] = False
            
        
    def load_scene(self, scene: Scene):
        self.states["scene"] = scene
        self.states["scene_done"] = False

        self.screen_surface.blit(self.ui_surface, (0,512)) 
        self.text_rendering_done = False

        if scene.neural_image:
            self.neural_queue.append(scene.neural_image)
        #scene.prompt_enabled
   



    def start(self):

        storytelling_members = {attr: getattr(Storytelling, attr) for attr in dir(Storytelling) if not callable(getattr(Storytelling, attr)) and not attr.startswith("__")}
        self.storytelling_plot = build_plot(storytelling_members, storytelling_members["s1"])
        
        pygame.draw.rect(self.ui_surface, color=(64, 64, 64, 255), rect= Rect(0, 0, 512, 200), width=2, border_radius=4) 
        self.load_scene(self.storytelling_plot["intro"])

        
        self.update()

if __name__ == "__main__":  


    game = Game()
    game.start()







