# KMC Driving Project

## 4. 환경 준비 (Host 측 설정)

### 4-1) 도커 설치 및 권한 부여 (최초 1회)
차량 컴퓨터에 도커가 설치되어 있지 않거나, sudo 없이 명령어가 안 먹을 때 실행합니다.
```bash
# 1. 도커 설치
sudo apt update && sudo apt install docker.io -y

# 2. 현재 사용자를 도커 그룹에 추가 (명령어 입력 후 로그아웃/로그인 필요)
sudo usermod -aG docker $USER

# 3. 설치 및 작동 확인 (버전 정보가 뜨면 정상)
docker version
```

### 4-2) 하드웨어 연결 및 권한 설정
차량 USB를 연결한 뒤 해당 포트의 읽기/쓰기 권한을 개방합니다.
```bash
# 포트 권한 개방 (ttyUSB0 기준)
sudo chmod 666 /dev/ttyUSB0
```

## 5. 실행 방법 (실제 차량용)

### 5-1) 하드웨어 연결 및 권한 설정 (Host)
차량 모터 제어기(USB)가 정상적으로 인식되었는지 확인하고 권한을 부여합니다.
```bash
# 1. 포트 연결 확인 (보통 ttyUSB0 또는 ttyACM0)
ls -l /dev/ttyUSB*

# 2. 포트 권한 부여 (ttyUSB0 기준, 장치명에 따라 수정)
sudo chmod 666 /dev/ttyUSB0
```

### 5-2) Docker 이미지 빌드 및 실행
프로젝트 루트 폴더(`~/kmc_ws`)에서 실행합니다.
```bash
# 1. 이미지 빌드
docker build -t kmc_image:v1 .

# 2. 컨테이너 실행 (장치 매핑: Host의 ttyUSB0를 Docker 내부 ttyKMC로 연결)
docker run -it --rm \
  --net=host \
  --ipc=host \
  --device=/dev/ttyUSB0:/dev/ttyKMC \
  -v $(pwd)/src:/home/user/kmc_ws/src \
  --name kmc_main \
  kmc_image:v1
```

### 5-3) 노드 실행 (Container 내부)
컨테이너 실행 시 자동으로 빌드(`colcon build`)가 수행됩니다. 이후 아래 명령어로 주행을 시작합니다.
```bash
# [Terminal 1] C++ 드라이버 노드 실행
ros2 run kmc_hardware driver_node

# [Terminal 2] 새 터미널에서 실행 중인 컨테이너 접속 후 주행 알고리즘 실행
docker exec -it kmc_main bash
ros2 run controller drive_basic
```