import java.io.*;

public class Main {
    public static void main(String[] args) {
        //  Serialization = saving an file with the object's information
        //  Deserialization = loading the saved file

        User user = new User();
        user.name = "broe";
        user.password = "pifg";
        user.sayHello();

        FileOutputStream fileOut = new FileOutputStream("UserInfo.ser");
        ObjectOutputStream out = new ObjectOutputStream(fileOut);
        out.write(user);
        out.close();
        fileOut.close();

        System.out.println("object info saved");
    }
}
