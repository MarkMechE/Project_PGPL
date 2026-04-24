// PGL Sensor Node Firmware (Tinkercad/ESP32 Compatible)
// Core Patent: Persistence Gating (P1-P4 Tiers)
// Pins: A0=Piezo(RPOT1), A2=Sigma(RPOT2), 13=LED, 8=Pulse

#define PIEZO A0
#define SIGMA A2
#define NOISE  A1
#define LED    13
#define BUZZ    8

const int   N      = 64;
const float ZONE_W = 0.5;

void setup() {
  pinMode(LED,  OUTPUT);
  pinMode(BUZZ, OUTPUT);
  Serial.begin(9600);
  Serial.println("PGPL GATING CORE — 6 Events / P1-P4");
  Serial.println("RPOT1=A0(signal) RPOT2=A2(salinity) PHOTO=A1(noise)");
}

void loop() {
  tone(BUZZ, 100, 50); delay(50);

  // --- Sensor reads ---
  float s    = analogRead(SIGMA) * 10.0 / 1023.0;  // 0-10 psu
  float nois = analogRead(NOISE) / 1023.0;          // 0-1

  // --- Adaptive threshold ---
  float thresh = 100.0 + s * 12.0 + nois * 60.0;

  // --- Sample window ---
  float sumSq  = 0;
  float mx     = 0;
  float mn     = 1023;
  float sumOdd = 0;
  float sumAll = 0;
  int   pers   = 0;

  for (int i = 0; i < N; i++) {
    float val = analogRead(PIEZO);
    sumSq += val * val;
    if (val > mx) mx = val;
    if (val < mn) mn = val;
    if (val > thresh) pers++;
    if (i % 2 == 0) sumOdd += val;
    sumAll += val;
  }

  float rms   = sqrt(sumSq / N);
  float rmsn  = rms / 1023.0;
  // Use peak-to-peak instead of mx/rms for crest
  // — potentiometers give near-DC so mx/rms ≈ 1 always
  float pp    = (mx - mn) / 1023.0;               // peak-to-peak normalised
  float midR  = sumAll > 0 ? sumOdd / sumAll : 0; // narrowband proxy
  float persm = pers / (float)N;
  float conf  = rmsn;

  // --- 6-class event detection (pot-friendly thresholds) ---
  String ev    = "Tidal";
  float  typeW = 0.0;

  if      (rmsn > 0.60 && persm > 0.55)           { ev = "Burst";        typeW = 1.0; }
  else if (rmsn > 0.30 && midR > 0.55 && pp < 0.3){ ev = "Crack";        typeW = 0.8; }
  else if (rmsn > 0.20 && persm > 0.25)            { ev = "Micro";        typeW = 0.6; }
  else if (s > 7.0 && rmsn < 0.15)                 { ev = "PressureDrop"; typeW = 0.4; }
  else if (nois > 0.50)                             { ev = "Pump";         typeW = 0.1; }

  // --- Severity: 0.4*conf + 0.3*typeW + 0.3*zoneW ---
  float sev = 0.4 * conf + 0.3 * typeW + 0.3 * ZONE_W;

  // --- Serial output ---
  Serial.print("Sal:");    Serial.print(s, 1);
  Serial.print(" RMSn:");  Serial.print(rmsn, 2);
  Serial.print(" PP:");    Serial.print(pp, 2);
  Serial.print(" Pers:");  Serial.print(persm * 100, 0); Serial.print("%");
  Serial.print(" Ev:");    Serial.print(ev);
  Serial.print(" Sev:");   Serial.println(sev, 2);

  // --- P1-P4 ---
  digitalWrite(LED, LOW);

  if (sev >= 0.80) {
    Serial.print(">>> P1 — "); Serial.print(ev); Serial.println(" DISPATCH NOW <<<");
    digitalWrite(LED, HIGH);
  }
  else if (sev >= 0.60) {
    Serial.print("P2 — "); Serial.print(ev); Serial.println(" Urgent");
    digitalWrite(LED, HIGH);
  }
  else if (sev >= 0.40) {
    Serial.print("P3 — "); Serial.print(ev); Serial.println(" Schedule");
    digitalWrite(LED, HIGH);
  }
  else {
    Serial.print("P4 — "); Serial.print(ev); Serial.println(" Monitor");
  }

  delay(1500);
}