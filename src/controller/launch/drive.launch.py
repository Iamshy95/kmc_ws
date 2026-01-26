from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # 1. 인자 설정 (ns는 차량 번호, port는 USB 포트)
    # 기본값을 CAV_03으로 설정하면 사용자님 환경에 더 잘 맞을 겁니다.
    ns_arg = DeclareLaunchArgument('ns', default_value='CAV_22')
    port_arg = DeclareLaunchArgument('port', default_value='/dev/ttyUSB0')

    ns = LaunchConfiguration('ns')

    return LaunchDescription([
        ns_arg,
        port_arg,

        # 2. 통합 드라이버 노드 실행
        Node(
            package='kmc_hardware',
            executable='kmc_driver_node',
            namespace=ns,  # 모든 토픽이 /ns/... 아래로 들어감
            parameters=[{'port': LaunchConfiguration('port')}],
            output='screen'
        ),

        # 3. 주행 코드(Python) 실행
        Node(
            package='controller',
            executable='drive_basic', 
            namespace=ns,
            output='screen',
            # [중요] 리매핑 설정
            # 주행 코드의 'pose' 구독 토픽을 실제 MoCap 토픽인 /CAV_03 등으로 연결
            remappings=[
                ('pose', ['/', ns]) # 상대 경로 pose를 절대 경로 /CAV_03 등으로 변경
            ]
        )
    ])