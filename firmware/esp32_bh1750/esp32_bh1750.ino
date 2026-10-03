/**
 * ESP32 + BH1750 Ambient Light Sensor Interface
 * 
 * CEP Project: AI-Powered Adaptive Vision System
 * Hardware: ESP32 DevKit V1 + GY-30 / BH1750 Sensor
 * 
 * Pinout (I2C):
 *  - VCC  -> 3.3V / 5V
 *  - GND  -> GND
 *  - SCL  -> GPIO 22 (Default ESP32 I2C SCL)
 *  - SDA  -> GPIO 21 (Default ESP32 I2C SDA)
 *  - ADDR -> GND (Address 0x23) or 3.3V (Address 0x5C)
 * 
 * Outputs formatted JSON to Serial at 115200 baud for sensor.py
 */

#include <Wire.h>

#define BH1750_I2C_ADDRESS_LOW   0x23
#define BH1750_I2C_ADDRESS_HIGH  0x5C

// Selected I2C Address (default is 0x23)
uint8_t bh1750_address = BH1750_I2C_ADDRESS_LOW;

// Opcodes for BH1750
#define BH1750_POWER_ON           0x01
#define BH1750_RESET              0x07
#define BH1750_CONTINUOUS_HIGH_RES 0x10 // 1 lux resolution, 120ms measurement time

void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 2000) {
        // Wait for serial monitor
    }

    Serial.println("\n--- BH1750 Ambient Light Sensor Initializing ---");

    // Initialize I2C on standard ESP32 pins (SDA=21, SCL=22)
    Wire.begin(21, 22);

    // Scan for sensor address
    Wire.beginTransmission(BH1750_I2C_ADDRESS_LOW);
    if (Wire.endTransmission() == 0) {
        bh1750_address = BH1750_I2C_ADDRESS_LOW;
        Serial.println("[INFO] BH1750 detected at address 0x23");
    } else {
        Wire.beginTransmission(BH1750_I2C_ADDRESS_HIGH);
        if (Wire.endTransmission() == 0) {
            bh1750_address = BH1750_I2C_ADDRESS_HIGH;
            Serial.println("[INFO] BH1750 detected at address 0x5C");
        } else {
            Serial.println("[WARN] BH1750 not found on I2C bus! Check wiring (SDA=21, SCL=22).");
        }
    }

    // Power on and set to continuous high resolution mode
    Wire.beginTransmission(bh1750_address);
    Wire.write(BH1750_POWER_ON);
    Wire.endTransmission();
    delay(10);

    Wire.beginTransmission(bh1750_address);
    Wire.write(BH1750_CONTINUOUS_HIGH_RES);
    Wire.endTransmission();
    delay(180);

    Serial.println("[READY] BH1750 streaming active.");
}

float readBH1750Lux() {
    Wire.requestFrom((int)bh1750_address, 2);
    if (Wire.available() == 2) {
        uint16_t raw_val = Wire.read();
        raw_val = (raw_val << 8) | Wire.read();
        // Standard formula: Lux = raw_value / 1.2
        return (float)raw_val / 1.2f;
    }
    return -1.0f;
}

void loop() {
    float lux = readBH1750Lux();

    if (lux >= 0) {
        // Output JSON structured stream for Python sensor.py
        Serial.printf("{\"lux\": %.2f, \"timestamp\": %lu, \"status\": \"ok\"}\n", lux, millis());
    } else {
        Serial.printf("{\"lux\": 0.00, \"timestamp\": %lu, \"status\": \"error\"}\n", millis());
    }

    // 5 Hz sampling rate (every 200 ms)
    delay(200);
}
