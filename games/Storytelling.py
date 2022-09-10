from Scene import Scene, Transition


s1 = Scene(name="Intro Scene",
           text="You wake up and look at the mirror, what do you see ?", 
           neural_image = "big mirror in an empty room", 
           prompt_enabled=True)

s2 = Scene(name="Character Setup", 
           text="Do you like what you see ?", 
           neural_image=None, 
           prompt_enabled=True,
           prompt_content=[["Yes"],["No"]])

t = Transition(s1, s2, Transition.ConditionTypes.always)

