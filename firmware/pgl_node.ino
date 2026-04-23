// PGL Sensor Node Firmware (Tinkercad/ESP32 Compatible)
// Core Patent: Persistence Gating (P1-P4 Tiers)
// Pins: A0=Piezo(RPOT1), A2=Sigma(RPOT2), 13=LED, 8=Pulse

#define PIEZO_PIN A0
#define SALINITY_PIN A2
#define ALERT_LED 13
#define PULSE_PIN 8

int window_samples = 512;

void setup() {
  pinMode(ALERT_LED, OUTPUT);
  pinMode(PULSE_PIN, OUTPUT);
  Serial.begin(9600);
  Serial.println("=== PGL Heart Active (Patent Core) ===");
  Serial.println("RPOT1 Twist: Noise(0%) → P2(30%+) → P1(60%+)");
}

void loop() {
  // (a) PULSE-AT Acoustic Burst
  tone(PULSE_PIN, 100, 200);  // Pipe excitation
  delay(250);

  // (b) Inputs & c(σ,T)
  float sigma = analogRead(SALINITY_PIN) * 35.0 / 1023.0;
  float c = 1449 + 4.6*20 + 1.34*(sigma - 35);  // Real physics
  float noise = analogRead(A1) / 1023.0;  // Tidal proxy
  float thresh = (300 + noise*500) / c * 100;  // Adaptive

  // (c-f) Persistence Gating (Core IP)
  int persistent = 0;
  for(int i = 0; i < window_samples; i++) {
    float sample = analogRead(PIEZO_PIN);
    if(sample > thresh) persistent++;
    delayMicroseconds(125);  // 8kHz sim
  }
  float metric = persistent / (float)window_samples;

  // Output
  Serial.print("Sigma:"); Serial.print(sigma,1);
  Serial.print(" | Thresh:"); Serial.print(thresh,0);
  Serial.print(" | Metric:"); Serial.print(metric*100,1); Serial.println("%");

  digitalWrite(ALERT_LED, LOW);
  String p_level;
  if(metric > 0.6) { p_level = "P1 CRITICAL"; digitalWrite(ALERT_LED, HIGH); }
  else if(metric > 0.3) { p_level = "P2 PRIORITY"; digitalWrite(ALERT_LED, HIGH); }
  else p_level = "GATED Noise";
  Serial.println(p_level);

  delay(2000);
}