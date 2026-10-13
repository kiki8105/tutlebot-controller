# TurtleBot3 PyQt Controller

ROS 2 Humble 기반 TurtleBot3 조종기입니다. 한 화면에서 카메라, ROS 파라미터,
지도·웨이포인트, 수동 주행을 다룹니다.

지도에서 단일 `/goal_pose`를 전송하거나 여러 경로점을 순서대로 보낼 수 있으며,
초기 위치를 HOME으로 저장해 복귀 목표로 다시 전송할 수 있습니다.
단일 Goal/HOME 기능을 쓰려면 주행 노드가 `/goal_pose`를 구독해야 합니다.
빨간 정지 버튼과 수동 주행 입력은 조종기가 발견한 Pure Pursuit 노드의
`drive_enabled`를 먼저 끄므로 `/cmd_vel` 발행자가 서로 충돌하지 않습니다.

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
QT_QPA_PLATFORM=offscreen python3 -m unittest -v test_cmd_vel_controller.py && \
QT_QPA_PLATFORM=offscreen python3 -m unittest -v test_navigation_panels.py && \
QT_QPA_PLATFORM=offscreen python3 -m unittest -v test_parameter_browser.py
```

축소 레이아웃 테스트는 700×500 창에서 스크롤바가 활성화되고 모든 주행
버튼이 46px 이상의 높이를 유지하는지 검사합니다.
