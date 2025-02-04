import java.util.Scanner;

public class Main {
    public static void main(String[] args){
        Scanner scan = new Scanner(System.in);
        System.out.println("Enter your age: ");
        int age = scan.nextInt();
        try{
            checkAge(age);
        }
        catch (Exception e){
            System.out.println(e.getMessage());
        }
        return;
    }

    static void checkAge(int age) throws AgeException{
        if (age<18){
            throw new AgeException("under 18");
        }
        else {
            System.out.println("signed up!");
        }
    }
}
