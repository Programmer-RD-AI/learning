import java.util.Scanner;

public class Main {
    public static void main(String[] args){
        // Polymorphism = many shapes/forms
        // dynamic = after compialation (during runtime)
        // ex: A corvette is a: corvette and a car and a vehicle yk?

        Scanner scanner = new Scanner(System.in);
        Animal animal;

        System.out.println("What animal do you want?");
        System.out.print("(1 = Dog, 2 = Cat): ");
        int choice = scanner.nextInt();

        if (choice == 1){
            animal = new Dog();
        }
        else if (choice == 2){
            animal = new Cat();
        }
        else {
            animal = new Animal();
            System.out.println("invalid choice");
        }
        animal.speak();
    }
}
