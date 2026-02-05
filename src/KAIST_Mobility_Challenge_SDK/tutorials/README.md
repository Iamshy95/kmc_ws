# SDK 예제 튜토리얼 로드맵

---

## 0. SDK 구성

### `KMC_HARDWARE::Driver`  
- 내부 스레드가 UART I/O를 전담합니다.
- 주기 제어(0xA5)와 주기 요청(0xB3/0xAF)을 수행합니다.
- `setCommand()`로 원하는 값을 업데이트합니다.
- 메시지 큐를 통해 수신 데이터를 처리할 수 있습니다.

### `KMC_HARDWARE::UartClient`
- 타이밍/요청을 직접 구현합니다.
- 디버깅이나 특정 커스텀 실험에 좋습니다.

---

## 1. 사용 방식별 튜토리얼 목록

### Driver (Basic + Intermediate)
- [01. driver_demo](./Driver_Basic/01_driver_demo.md): `Driver`로 주행 명령(0xA5) 송신 기본
- [02. driver_observe](./Driver_Intermediate/01_driver_observe.md): `Driver` 수신 큐 처리
- [03. read_allstate](./Driver_Intermediate/02_read_allstate.md): `Driver`로 AllState(0xAF) 읽기
- [04. high_rate_control](./Driver_Intermediate/03_high_rate_control.md): `Driver` 기반 1 kHz 제어 스트림

### Driver ROS2
- [01. driver_demo](./Driver_ROS2/01_driver_ros2.md): ROS2로 driver_demo 구성
- [02. driver_observe](./Driver_ROS2/02_driver_observe_ros2.md): ROS2로 driver_observe 구성
- [03. driver_read_allstate](./Driver_ROS2/03_driver_read_allstate_ros2.md): ROS2로 read_allstate 구성
- [04. high_rate_control](./Driver_ROS2/04_high_rate_control_ros2.md): ROS2로 high_rate_control 구성

### UartClient (Advanced)
- [01. read_speed](./UartClient_Advanced/01_read_speed.md): `UartClient`로 속도(0xB3) 요청/출력
- [02. read_battery](./UartClient_Advanced/02_read_battery.md): `UartClient`로 배터리 전압(0xAF) 읽기
- [03. read_allstate](./UartClient_Advanced/03_read_allstate.md): `UartClient`로 AllState(0xAF) 읽기
- [04. teleop_and_speed](./UartClient_Advanced/04_teleop_and_speed.md): `UartClient` 직접 스케줄링 (0xA5 송신 + 0xB3 수신)

---

## 2. 알고리즘 구현 시 필요한 사항
- 알고리즘 출력값을 `setCommand(v, omega)`로 매핑
- 안정성을 위해 주기적으로 업데이트

수신 데이터 처리 흐름은 `driver_observe`에서 다룹니다.

---

## 3. 공통 가이드
- 빌드 방법: 프로젝트 루트의 `README.md` 참고
- 하드웨어 설정 (RTS/CTS): `README_QUICKSTART.md` 참고
- API 상세 레퍼런스: `README_SDK.md` 참고
