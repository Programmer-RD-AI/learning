import java.util.Scanner;

public class UserInput {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.println("What is your name?");
        String name = scanner.nextLine();
        System.out.println("How old are you?");
        int age = scanner.nextInt();
        scanner.nextLine();
        System.out.println("What is your favoriate food?");
        String food = scanner.nextLine();
        System.out.println("Hello, " + name + "! How old are you?");
        System.out.println("You are " + age + " years old.");
        System.out.println("What is your favoriate food?");
        System.out.println("You like " + food + "!");
    }
}
