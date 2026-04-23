# Patent Claims (KIPO Provisional - File Today)

**Claim 1 (Method)**:
A method for persistence-gated leak detection using a piezoelectric sensor node, comprising:
(a) activating a piezoelectric transducer to generate a PULSE-AT acoustic signal;
(b) sampling response voltage from the piezoelectric transducer via ADC;
(c) computing adaptive threshold using c(σ,T) from salinity σ and temperature T;
(d) computing persistence metric = [samples > threshold / window_samples];
(e) classifying P1-P4 severity if metric > 0.3;
(f) outputting alert via GPIO.

**Claim 8 (System)**:
Sensor node with ESP32 MCU executing Claim 1 (Tinkercad prototype Fig.6).

**Novelty**: KIPRIS 0 hits for "persistence gating leak piezo salinity".
