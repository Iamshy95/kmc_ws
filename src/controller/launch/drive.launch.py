from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchSubstitution, LaunchConfiguration

def generate_launch_description():
    # 1. 터미널에서 입력받을 인자(Argument) 설정
    ns_arg = DeclareLaunchArgument('ns', default_value='CAV_01')
    port_arg = DeclareLaunchArgument('port', default_value='/dev/ttyUSB0')

    return LaunchDescription([
        ns_arg,
        port_arg,

        # 2. 통합 드라이버 노드 실행
        Node(
            package='kmc_hardware',
            executable='kmc_driver_node',
            namespace=LaunchConfiguration('ns'), # 방 번호 자동 지정
            parameters=[{'port': LaunchConfiguration('port')}],
            output='screen'
        ),

        # 3. 주행 코드(Python) 실행
        Node(
            package='controller',
            executable='drive_basic', # setup.py에 등록된 이름 확인!
            namespace=LaunchConfiguration('ns'), # 드라이버와 같은 방에 넣기
            output='screen',
            # 여기서 토픽 이름을 맞춰줍니다 (Remapping)
            remappings=[
                ('pose', 'pose') # (내코드이름, 실제토픽이름) - 나중에 여기서 수정!
            ]
        )
    ])