# PGL: Persistence-Gated Piezo Leak Detection (Busan Eco-Delta City)

[![Demo Video](https://img.youtube.com/vi/demo.png)](your-demo-video)  
**Live Tinkercad Demo**: [Click Here](YOUR-TINKERCAD-PUBLIC-LINK) – Twist RPOT1 → P2 LED Alert!

## 🎯 Quick Demo (2 Minutes)
1. Open Tinkercad → **RPOT1 Low**: "Metric:5% | GATED Noise" (LED OFF).
2. **Rapid Twist RPOT1**: "Metric:45% | P2 PRIORITY!" (LED ON).
3. **RPOT2 High (Salty)**: Harder threshold test.
4. **Serial Proof**: ![Serial Demo](docs/serial_demo.png)

## 🏗️ Architecture
![Full Diagram](docs/architecture.png)

## 📊 Results
| Test | Metric | P-Level | RMSE |
|------|--------|---------|------|
| Noise | 5% | Gated | N/A |
| P2 Leak | 45% | P2 | 6.2m |
| P1 Critical | 100% | P1 | 5.53m |

**78% Accuracy on Real KOHO Busan Data.**

## 🚀 Setup & Run
### Firmware (Tinkercad/ESP32)