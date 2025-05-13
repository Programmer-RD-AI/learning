public class Friend {
    String name;
    static int numberOfFriends;

    Friend(String name){
        this.name = name;
        numberOfFriends++;
    }

    static void displayFriends(){
        System.out.println(numberOfFriends);
    }
}
//Friend friend1 = new Friend("Spongebomb");
//Friend friend2 = new Friend("Patrick");
//        System.out.println(Friend.numberOfFriends);
