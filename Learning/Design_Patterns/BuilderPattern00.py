"""
# Builder Pattern

Seperate the consutrction of an object from its representation

class House:
    stories: 1 or 2
    door type: single or double
    roof type: pointly or flat

When we initalize the House there are different variations of them, such as:
    House Obj:
        Stories: 1
        Door Type: single
        Roof Type: pointly

    House obj:
        Stories: 2
        Door Type: double
        Roof Type: flat

These objects are the representation's of the House class it self.

1 story house = new House(stories: 1, door type: double, roof type: pointly)

In the Builder Pattern:

class House Builder:
    stories
    door type
    roof type

    setStories
    setDoorType
    setRoofType

    build: return a new House(self)

House() updated consutrctor: House(HouseBuilder)
"""

class House:
    def __init__(self, builder):
        self.stories = builder.stories
        self.door_type = builder.door_type
        self.roof_type = builder.roof_type

class HouseBuilder:
    def __init__(self, builder):
        self.stories = None
        self.door_type = None
        self.roof_type = None

    def set_stories(self, stories):
        self.stories = stories
        return self

    def set_door_type(self, door_type):
        self.door_type = door_type
        return self

    def set_roof_type(self, roof_type):
        self.roof_type = roof_type
        return self

    def build(self):
        return House(self)

one_story_house = HouseBuilder().house_builder.set_stories(1).set_door_type("single").set_roof_type("pointly").build()

class Director:
    def __init__(self, builder):
        self.builder = builder

    def build_1_story_house(self):
        return self.builder.set_stories(1).set_door_type("single").set_roof_type("pointly").build()

    def build_2_story_house(self):
        return self.builder.set_stories(2).set_door_type("double").set_roof_type("flat").build()

director = Director()
director.build_1_story_house()
