import java.util.Random;

public class DiceRoller {
    Random random;
    int number;
    DiceRoller(){
        random = new Random();
        number = 0;
//        roll(random, number);
        roll();
    }
//    void roll(Random random, int number){
    void roll(){
        number = random.nextInt(6) + 1;
        System.out.println(number);
    }
}

