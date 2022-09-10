import enum
import json
from random import random
from collections import namedtuple

class Scene:
    def __init__(self, name, text, neural_image, prompt_enabled=False, prompt_content="free") -> None:
        self.name = name
        self.text = text
        self.neural_image = neural_image
        self.prompt_enabled = prompt_enabled
        self.prompt_content = prompt_content
      
    #def add_transition(Transition):
    #    self.transitions

class Transition:
    _condition_type = namedtuple("Condition", ["always", "probability"])
    ConditionTypes = _condition_type(always=True, probability=lambda p : random() <= p)
    
    #ActionTypes = namedtuple("Action", ["imediate", "timed"])
 

    def __init__(self, scene_from :Scene, scene_to : Scene, condition_function: ConditionTypes) -> None:
        self.scene_from = scene_from
        self.scene_to = scene_to
        self.condition = condition_function
        #self.action_mode = action_mode

       



# build plot dictionary with from_scene : transition map
def build_plot(member_list, entry_scene):
    plot = {"intro" : entry_scene}
    for member in member_list.values():
        if isinstance(member, Transition):
            plot[member.scene_from] = member
    return plot



