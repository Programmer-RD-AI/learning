const int ledPin = 13;
void setup()
{
    Serial.begin(9600);      // Initial the serial output
    pinMode(ledPin, OUTPUT); // Set the pin as OUTPUT
}
void loop()
{
    static int delayPeriod = 100; // In ms
    static int countDir = 1;
    digitalWrite(ledPin, HIGH); // Turn on the LED with HIGH output V
    delay(delayPeriod);         // Delay the amount of delayPeriod
    digitalWrite(ledPin, LOW);  // Turn off the LED with LOW output V
    delay(delayPeriod);         // Delay the amount of delayPeriod
    countDir = checkDirChange(delayPeriod, countDir);
}
int checkDirChange(int delayPeriod, int countDir)
{
    if ((delayPeriod == 1000) || (delayPeriod == 0))
    {
        countDir *= -1;
        if (countDir < 0)
        {
            Serial.println("Going Down");
        }
        else
        {
            Serial.println("Going Up");
        }
    }
}
