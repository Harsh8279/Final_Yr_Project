#include <Wire.h>
#include <Adafruit_MLX90614.h>
Adafruit_MLX90614 mlx = Adafruit_MLX90614();
const int buzzer=8;
const int threshold=35;

void setup() {
  Serial.begin(9600);  
  pinMode(8, OUTPUT);
  mlx.begin();  
}
void loop() {
  Serial.print(mlx.readObjectTempC()+5); 
  if (mlx.readObjectTempC()>threshold){
    digitalWrite(buzzer, HIGH);
    Serial.print(",HIGH");
  } else {
    digitalWrite(buzzer, LOW);
  }
  delay(1000);
  Serial.println();
}
