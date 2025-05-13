package com.phone;

public class OperatingSystemFactory {
    public OS getInstance(String str){
        if (str.equals("Open")){
            return new Android();
        }
        else if (str.equals("Closed")){
            return new iOS();
        }
        return new Windows();
    }
}
