public class Main {
    public static void Main(String[] args){
        Greeting greeting = new Greeting(){
            @Override
            public void Welcome(){
                System.out.println("Yo");
            }
        };
        Greeting greeting2 = new Greeting();
        greeting2.Welcome();
        greeting.Welcome();
        return;
    }
}
