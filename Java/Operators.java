import java.util.Scanner;

public class Operators{
    public static void main(String[] args){
        int temp = 25;
        Scanner scanner = new Scanner(System.in);
        System.out.println("You are playing a game! Press q or Q to quit");
        String response = scanner.next();
        if (response.equals("q") || response.equals("Q")){
            System.out.println("You quit the game");
        }
        else{
            System.out.println("You are still playing the game");
        }
//        if (temp > 30){
//            System.out.println("It is hot outside");
//        } else if (temp >= 20 && temp <= 30) {
//            System.out.println("It is warm outside");
//        }
//        else {
//            System.out.println("It is cold outisde");
//        }

        return;
    }
}

