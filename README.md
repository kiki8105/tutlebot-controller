# TurtleBot3 PyQt Controller

ROS 2 Humble 기반 TurtleBot3 조종기입니다. 한 화면에서 카메라, ROS 파라미터,
지도·웨이포인트, 수동 주행을 다룹니다.

창을 작게 줄이면 대시보드에 스크롤바가 나타납니다. 주행 버튼과 입력 필드는
최소 크기를 유지하므로 글자와 버튼이 눌리거나 겹치지 않습니다.

## 필요 환경

- Ubuntu 22.04
- ROS 2 Humble 및 Nav2
- Python 3, PyQt5, NumPy, Pillow, PyYAML
- 빌드된 TurtleBot3 워크스페이스. 기본 경로는 `~/turtlebot3`입니다.

필요한 Ubuntu 패키지는 다음과 같이 설치할 수 있습니다.

```bash
sudo apt install -y python3-pyqt5 python3-numpy python3-pil python3-yaml
```

## 실행

```bash
cd ~/turtlebot3/tutlebot-controller
./control_64
```

실제 TurtleBot3가 Domain 226을 사용하면 다음 스크립트를 실행합니다.

```bash
./control_226
```

워크스페이스 위치가 다르면 실행 전에 지정할 수 있습니다.

```bash
TURTLEBOT3_WS=/다른/워크스페이스 ./control_226
```

## 테스트

```bash
source /opt/ros/humble/setup.bash
QT_QPA_PLATFORM=offscreen python3 -m unittest -v test_cmd_vel_controller.py
```

축소 레이아웃 테스트는 700×500 창에서 스크롤바가 활성화되고 모든 주행
버튼이 46px 이상의 높이를 유지하는지 검사합니다.
