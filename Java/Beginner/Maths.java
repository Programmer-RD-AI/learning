import java.util.Scanner;

public class Maths{
    public static void main(String[] args){
        Scanner scanner = new Scanner(System.in);
        System.out.println("Give the a: ");
        double a = Double.parseDouble(scanner.nextLine());
        System.out.println("Give the b :");
        double b = Double.parseDouble(scanner.nextLine());
        double c = Math.sqrt(Math.pow(a, 2) + Math.pow(b, 2));
        System.out.println("The Hypotonias is: " + c);
        scanner.close();
    }

    public static void Intro(){
        double x = 3.14;
        double y = -10;
        double z = Math.max(x, y); // Math.min(x, y), Math.abs(y), Math.sqrt(x), Math.round(x), Math.ceil(x), Math.floor(x)
        System.out.println(z);
        return;
    }
}