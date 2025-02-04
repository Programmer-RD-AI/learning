public class Main {
    public static void main(String[] args) throws InterruptedException {
        // Daemon thread is a low priority thread that runs in background to perform tasks such as garbage collection
        //				JVM terminates itself when all user threads (non-daemon threads) finish their execution

//        System.out.println(Thread.activeCount());
//        Thread.currentThread().setName("testing");
//        System.out.println(Thread.currentThread().getName());
//        Thread.currentThread().setPriority(10);
//        System.out.println(Thread.currentThread().getPriority());
//        System.out.println(Thread.currentThread().isAlive());
//
//        for (int i = 3; i>0; i--){
//            System.out.println(i);
//            Thread.sleep(1000);
//        }
//
//        System.out.println("done");
        MyThread thread2 = new MyThread();
        thread2.setDaemon(true);
        System.out.println(thread2.isDaemon());
        thread2.start();
        thread2.setPriority(10);
        System.out.println(thread2.isAlive());
        System.out.println(thread2.getPriority());
        System.out.println(Thread.activeCount());
    }
}
