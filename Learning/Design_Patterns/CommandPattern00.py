from abc import ABC, abstractmethod

# Command Interface
class Command(ABC):
    @abstractmethod
    def execute(self):
        pass

# Receiver Classes (Devices)
class Light:
    def turn_on(self):
        print("The light is ON")

    def turn_off(self):
        print("The light is OFF")

class Fan:
    def turn_on(self):
        print("The fan is ON")

    def turn_off(self):
        print("The fan is OFF")

class AirConditioner:
    def turn_on(self):
        print("The Air Conditioner is ON")

    def turn_off(self):
        print("The Air Conditioner is OFF")

# Concrete Command Classes
class LightOnCommand(Command):
    def __init__(self, light):
        self.light = light

    def execute(self):
        self.light.turn_on()

class LightOffCommand(Command):
    def __init__(self, light):
        self.light = light

    def execute(self):
        self.light.turn_off()

class FanOnCommand(Command):
    def __init__(self, fan):
        self.fan = fan

    def execute(self):
        self.fan.turn_on()

class FanOffCommand(Command):
    def __init__(self, fan):
        self.fan = fan

    def execute(self):
        self.fan.turn_off()

class AirConditionerOnCommand(Command):
    def __init__(self, air_conditioner):
        self.air_conditioner = air_conditioner

    def execute(self):
        self.air_conditioner.turn_on()

class AirConditionerOffCommand(Command):
    def __init__(self, air_conditioner):
        self.air_conditioner = air_conditioner

    def execute(self):
        self.air_conditioner.turn_off()

# Invoker Class
class RemoteControl:
    def __init__(self):
        self.commands = {}

    def set_command(self, button, command):
        self.commands[button] = command

    def press_button(self, button):
        if button in self.commands:
            self.commands[button].execute()
        else:
            print("No command assigned to this button.")

# Client Code (Setting up the system)
if __name__ == "__main__":
    # Receivers
    light = Light()
    fan = Fan()
    air_conditioner = AirConditioner()

    # Commands
    light_on = LightOnCommand(light)
    light_off = LightOffCommand(light)
    fan_on = FanOnCommand(fan)
    fan_off = FanOffCommand(fan)
    ac_on = AirConditionerOnCommand(air_conditioner)
    ac_off = AirConditionerOffCommand(air_conditioner)

    # Invoker
    remote = RemoteControl()

    # Setting commands for different buttons
    remote.set_command("Button1", light_on)
    remote.set_command("Button2", light_off)
    remote.set_command("Button3", fan_on)
    remote.set_command("Button4", fan_off)
    remote.set_command("Button5", ac_on)
    remote.set_command("Button6", ac_off)

    # Pressing buttons (client requests actions)
    print("Pressing Button1 (Turn Light ON):")
    remote.press_button("Button1")
    
    print("\nPressing Button2 (Turn Light OFF):")
    remote.press_button("Button2")
    
    print("\nPressing Button3 (Turn Fan ON):")
    remote.press_button("Button3")
    
    print("\nPressing Button5 (Turn AC ON):")
    remote.press_button("Button5")
    
    print("\nPressing Button6 (Turn AC OFF):")
    remote.press_button("Button6")
    
    print("\nPressing Button4 (Turn Fan OFF):")
    remote.press_button("Button4")

