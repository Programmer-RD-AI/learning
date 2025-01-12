from abc import ABC, abstractmethod

# Step 1: Component Interface
class Coffee(ABC):
    @abstractmethod
    def cost(self) -> float:
        pass

    @abstractmethod
    def description(self) -> str:
        pass

# Step 2: Concrete Component
class BasicCoffee(Coffee):
    def cost(self) -> float:
        return 2.0

    def description(self) -> str:
        return "Basic Coffee"

# Step 3: Base Decorator
class CoffeeDecorator(Coffee):
    def __init__(self, coffee: Coffee):
        self._coffee = coffee

    def cost(self) -> float:
        return self._coffee.cost()

    def description(self) -> str:
        return self._coffee.description()

# Step 4: Concrete Decorators
class MilkDecorator(CoffeeDecorator):
    def cost(self) -> float:
        return self._coffee.cost() + 0.5

    def description(self) -> str:
        return self._coffee.description() + ", Milk"

class SugarDecorator(CoffeeDecorator):
    def cost(self) -> float:
        return self._coffee.cost() + 0.2

    def description(self) -> str:
        return self._coffee.description() + ", Sugar"

class CaramelDecorator(CoffeeDecorator):
    def cost(self) -> float:
        return self._coffee.cost() + 0.7

    def description(self) -> str:
        return self._coffee.description() + ", Caramel"

# Example Usage:
if __name__ == "__main__":
    # Start with basic coffee
    my_coffee = BasicCoffee()
    print(f"{my_coffee.description()} - ${my_coffee.cost()}")

    # Add milk
    my_coffee = MilkDecorator(my_coffee)
    print(f"{my_coffee.description()} - ${my_coffee.cost()}")

    # Add sugar
    my_coffee = SugarDecorator(my_coffee)
    print(f"{my_coffee.description()} - ${my_coffee.cost()}")

    # Add caramel
    my_coffee = CaramelDecorator(my_coffee)
    print(f"{my_coffee.description()} - ${my_coffee.cost()}")



