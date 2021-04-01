#include <Wire.h>
#include <Adafruit_MLX90614.h>
Adafruit_MLX90614 mlx = Adafruit_MLX90614();
const int buzzer=8;
const int threshold=35;

void setup() {
  Serial.begin(9600);  
//  Serial.println("Temp-*C");
  pinMode(8, OUTPUT);
  mlx.begin();  
}
void loop() {
//  Serial.print("DATA,TIME,");
  Serial.print(mlx.readObjectTempC()+5); 

  Serial.println();
    if (mlx.readObjectTempC()>threshold){
    digitalWrite(buzzer, HIGH);
  } else {
    digitalWrite(buzzer, LOW);
  }
  delay(1000);
}
